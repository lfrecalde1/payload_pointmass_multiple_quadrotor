#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation as R
import time
from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosSim
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
from typing import Dict, List
from payload_pointmass_multiple_quadrotor.lim_min_multiple_simple import plan_three_quad_point_mass

class PayloadControlMujocoMultiplePointMass(Node):
    def __init__(self):
        super().__init__('MultiplePointMass')

        # Runtime parameters (mirrors dq_nmpc style parameterization).
        self.declare_parameter('planner.ts', 0.05)
        self.declare_parameter('planner.horizon_time', 2.0)
        self.declare_parameter('nmpc.jerk_limit', [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.declare_parameter('model.norm_regularization_eps', 1e-8)
        self.declare_parameter('model.unit_vector_stabilization_gain', 5.0)
        self.declare_parameter('model.angular_orthogonality_gain', 5.0)

        # Time Definition
        self.t_N = 1.5
        self.N_prediction = int(31)
        self.ts = self.t_N / self.N_prediction

        self.norm_regularization_eps = float(self.get_parameter('model.norm_regularization_eps').value)
        self.unit_vector_stabilization_gain = float(self.get_parameter('model.unit_vector_stabilization_gain').value)
        self.angular_orthogonality_gain = float(self.get_parameter('model.angular_orthogonality_gain').value)
        self.reference_start_time = None

        # Prediction Node of the NMPC formulation
        print(self.N_prediction)

        # Internal parameters defintion
        self.robot_num = 3
        self.mass = 0.2
        self.gravity = 9.81


        # Quadrotor paramaters
        self.mass_quad = 1.24

        # Cable length
        self.length = 0.75
        self.e3 = ca.DM([0, 0, 1])

        ## Gains Controller 
        self.kp_min = 10.0
        self.kv_min = 5.0
        self.weight_cable_direction = 10.0
        self.weight_quadrotor_position = 1.0
        self.weight_r = 0.1
        self.weight_acceleration = 0.1
        self.norm_constraint_slack_weight = 10.0
        self.unit_vector_norm_tol = 1e-3


        ## Compute the initial tension based on the the Wrench
        # Position of the system payload
        pos_0 = np.array([0.34049933598831467, -0.0007520805616463245, 0.8936953489145677], dtype=np.double)
        # Linear velocity of the payload
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        
        ## Quadrotors section --------------------------------------
        ## Create the initial states for quadrotor
        pos_quad_1 = np.array([-0.0029774502873919804, -0.30020808379855246, 1.4896822896707809], dtype=np.double)
        ## Linear velocity of the sytem respect to the inertial frame
        vel_quad_1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ## Angular velocity respect to the Body frame
        omega_quad_1 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ## Initial Orientation expressed as quaternionn
        quat_quad_1 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq_1 = np.hstack((pos_quad_1, vel_quad_1, quat_quad_1, omega_quad_1))

        pos_quad_2 = np.array([-0.003912773485618989, 0.29969367235921324, 1.4896964934919243], dtype=np.double)
        ## Linear velocity of the sytem respect to the inertial frame
        vel_quad_2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ## Angular velocity respect to the Body frame
        omega_quad_2 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ## Initial Orientation expressed as quaternionn
        quat_quad_2 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq_2 = np.hstack((pos_quad_2, vel_quad_2, quat_quad_2, omega_quad_2))

        pos_quad_3 = np.array([0.8063607699386893, -0.00040573095469479846, 1.4825609159860986], dtype=np.double)        ## Linear velocity of the sytem respect to the inertial frame
        ## Linear velocity of the sytem respect to the inertial frame
        vel_quad_3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ## Angular velocity respect to the Body frame
        omega_quad_3 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        ## Initial Orientation expressed as quaternionn
        quat_quad_3 = np.array([1.0, 0.0, 0.0, 0.0])

        self.xq_3 = np.hstack((pos_quad_3, vel_quad_3, quat_quad_3, omega_quad_3))
        
        # Init Tension of the cables so we can get initial cable direction
        self.init = np.hstack((pos_0, vel_0))

        # Verification direction vectors

        # create functions
        ##  ----------------------------------------------------------------- Funtion Casadi ---------------------------------
        self.payload_to_quadrotor_unit = self.quadrotor_payload_unit_vector_c()
        self.cable_angular_velocity = self.cable_angular_velocity_c()
        self.quadrotors_position = self.quadrotor_position_c()
        self.quadrotors_velocity = self.quadrotor_velocity_c()
        self.tensions = self.cable_tension_c()
        unit_vectors_init = self.payload_to_quadrotor_unit(pos_0, np.hstack((pos_quad_1, pos_quad_2, pos_quad_3)))
        ##  ----------------------------------------------------------------- Funtion Casadi ---------------------------------

        # This is just the cable direction
        q1_eq = self.normalize(pos_0-pos_quad_1)
        q2_eq = self.normalize(pos_0-pos_quad_2)
        q3_eq = self.normalize(pos_0-pos_quad_3)

        q_eq_list = [q1_eq, q2_eq, q3_eq]

        ## Compute the cable direction initial condition
        ## This is how to copute the cable directions based on the wrench
        self.n_init = np.array(unit_vectors_init).reshape((self.robot_num*3, ))

        lambdas_eq, tensions_eq = self.build_hover_equilibrium_lambdas(
            payload_mass=self.mass,
            gravity=self.gravity,
            q_eq_list=q_eq_list,
        )

        self.tension_min = 0.2*np.array(tensions_eq)
        self.tension_max = 5*np.array(tensions_eq)

        print("Tensions")
        print(tensions_eq)
        print(self.tension_min)
        print(self.tension_max)
        print("Cable direciton")
        print(self.n_init)
        
        ## Compute the cable initial angular velocity
        self.r_init = np.array(
            self.cable_angular_velocity(
                np.hstack((pos_0, vel_0)),
                np.hstack((pos_quad_1, pos_quad_2, pos_quad_3)),
                np.hstack((vel_quad_1, vel_quad_2, vel_quad_3)),
            ),
            dtype=np.double,
        ).reshape((self.robot_num * 3,))

        ## Init states for the optimizer
        self.aq1_init = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.aq2_init = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.aq3_init = np.array([0.0, 0.0, 0.0], dtype=np.double)

        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init))


        ## Acceleration input and acceleration-state initialization.
        self.u_equilibrium = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)

        ## Bounds for jerk input [m/s^3].
        self.acceleration_limit = np.array(self.get_parameter('nmpc.jerk_limit').value, dtype=np.double).reshape((3*self.robot_num,))
        self.u_min = -self.acceleration_limit.copy()
        self.u_max = self.acceleration_limit.copy()

        ## Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]
        
        print("Verify payload states and control actions also dimensions")
        print(self.x_0)
        print(self.u_equilibrium)
        print(self.n_x)
        print(self.n_u)

        ## Define odometry subscriber
        self.subscriber_payload_ = self.create_subscription(Odometry, "/quadrotor1/payload/odom", self.callback_get_odometry_payload, 10)
        self.publisher_desired_payload = self.create_publisher(Path, "/quadrotor1/payload/desired_path", 10)

        self.publisher_cable_angular_velocity = self.create_publisher(
            Float64MultiArray,
            "/payload/cable_angular_velocity",
            10,
        )
        self.publisher_cable_direction = self.create_publisher(
            Float64MultiArray,
            "/payload/cable_direction",
            10,
        )

        ## Subcriber of each drone
        self.subscriber_drone_1_ = self.create_subscription(Odometry, "/quadrotor1/odom", self.callback_get_odometry_drone_1, 10)
        self.subscriber_drone_2_ = self.create_subscription(Odometry, "/quadrotor2/odom", self.callback_get_odometry_drone_2, 10)
        self.subscriber_drone_3_ = self.create_subscription(Odometry, "/quadrotor3/odom", self.callback_get_odometry_drone_3, 10)

        ## TF We can verify cable direction if they make sense or not
        self.tf_broadcaster = TransformBroadcaster(self)

        ## Publisher desired states for quadrotor
        self.publisher_ref_quadrotor_1 = self.create_publisher(PositionCommand, "/quadrotor1/payload_planner_quadrotor_cmd", 10)
        self.publisher_prediction_quadrotor_1 = self.create_publisher(Path, "/quadrotor1/predicted_path", 10)

        ## Publisher desired states for quadrotor
        self.publisher_ref_quadrotor_2 = self.create_publisher(PositionCommand, "/quadrotor2/payload_planner_quadrotor_cmd", 10)
        self.publisher_prediction_quadrotor_2 = self.create_publisher(Path, "/quadrotor2/predicted_path", 10)

        ## Publisher desired states for quadrotor
        self.publisher_ref_quadrotor_3 = self.create_publisher(PositionCommand, "/quadrotor3/payload_planner_quadrotor_cmd", 10)
        self.publisher_prediction_quadrotor_3 = self.create_publisher(Path, "/quadrotor3/predicted_path", 10)

        self.publisher_prediction_payload = self.create_publisher(Path, "payload/predicted_path", 10)

        ## Casadi Model multiple quadrotor and paylaod
        self.flag = 0
        self.code_export_directory ="c_generated_code"
        self.json_file = "acados_ocp_planner_payload_pointmass_multiple.json"

        ## Define desired Values 
        self.xd = np.zeros((self.n_x, ), dtype=np.double)
        self.ud = np.zeros((self.n_u, ), dtype=np.double)
        planner_goal = np.array([0.5, 0.0, 1.0], dtype=np.double)

        self.reference_plan = plan_three_quad_point_mass(
            p0=pos_0,
            pf=planner_goal,
            T_total=self.t_N,
            n_samples=max(self.N_prediction + 1, 201),
            payload_mass=self.mass,
            gravity=self.gravity,
            cable_lengths=np.full((self.robot_num,), self.length, dtype=np.double),
        )
        self.update_reference_from_plan(0.0)
        self.timer = self.create_timer(self.ts, self.run)


    def build_hover_equilibrium_lambdas(self, payload_mass: float, gravity: float, q_eq_list: List[np.ndarray]):
        N = np.column_stack(q_eq_list)  # 3x3
        rhs = -payload_mass * gravity * np.array(self.e3, dtype=np.double).reshape((3,))
        tensions = np.linalg.solve(N, rhs)

        lambdas_eq = [tensions[i] * q_eq_list[i] for i in range(3)]
        tension_eq = [tensions[i] for i in range(3)]
        return lambdas_eq, tension_eq

    def normalize(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-9:
            raise ValueError("Cannot normalize a near-zero vector.")
        return v / n
    
    def quadrotor_payload_unit_vector_c(self):
        x = ca.MX.sym('x', 3, 1)
        x_p   = x[0:3]

        xq = ca.MX.sym('xq', 3*self.robot_num, 1)
        xq_p = ca.reshape(xq, 3, self.robot_num)

        # Vectorized expression:
        cols = []
        for k in range(self.robot_num):
            term = x_p - xq_p[:, k]
            norm_term = ca.sqrt(ca.dot(term, term) + self.norm_regularization_eps)
            n_k = term / norm_term
            cols.append(n_k)
        quad_payload_mat = ca.hcat(cols)             # 3 x m
        quad_payload_vec = ca.reshape(quad_payload_mat, 3*self.robot_num, 1)  # (3m) x 1
        quadrotor_payload_vector_f = ca.Function('quadrotor_payload_vector_f', [x, xq], [quad_payload_vec])
        return quadrotor_payload_vector_f

    def cable_angular_velocity_c(self):
        # state & input
        x = ca.MX.sym('x', 6, 1)

        xQ_p = ca.MX.sym('xQ_p', 3*self.robot_num, 1)  # general: 3 thrust comps + 3m 'r' comps
        xQ_p_matrix = ca.reshape(xQ_p, 3, self.robot_num)

        xQ_v = ca.MX.sym('xQ_v', 3*self.robot_num, 1)  # general: 3 thrust comps + 3m 'r' comps
        xQ_v_matrix = ca.reshape(xQ_v, 3, self.robot_num)

        # unpack state
        x_p   = x[0:3]      # 3x1
        v_p   = x[3:6]      # 3x1

        # Vectorized expression:
        cols = []
        for k in range(self.robot_num):
            term = x_p - xQ_p_matrix[:, k]
            # Cable Direction
            norm_term = ca.sqrt(ca.dot(term, term) + self.norm_regularization_eps)
            n_k = term / norm_term

            x_Q = xQ_p_matrix[:, k]
            v_Q = xQ_v_matrix[:, k]

            a = x_p - x_Q
            norm_a = ca.sqrt(ca.dot(a, a) + self.norm_regularization_eps)
            dot_a = ca.dot(a, a) + self.norm_regularization_eps
            I = ca.MX.eye(3)
            a_dot = v_p - v_Q

            n_dot_k = (1/norm_a)*(I - (a@a.T)/dot_a)@a_dot
            r_k = ca.cross(n_k, n_dot_k)
            cols.append(r_k)

        quad_payload_angular_velocity_mat = ca.hcat(cols)             # 3 x m
        quad_payload_angular_velocity_vec = ca.reshape(quad_payload_angular_velocity_mat, 3*self.robot_num, 1)  # (3m) x 1
        r_velocity_f = ca.Function('r_velocity_f', [x, xQ_p, xQ_v], [quad_payload_angular_velocity_vec])
        return r_velocity_f


    def callback_get_odometry_payload(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((6, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        x[3] = msg.twist.twist.linear.x
        x[4] = msg.twist.twist.linear.y
        x[5] = msg.twist.twist.linear.z

        # Get quadrotor  positions
        xquadrotor1 = self.xq_1[0:3]
        # Get quadrotor  velocities
        vquadrotor1 = self.xq_1[3:6]

        xquadrotor2 = self.xq_2[0:3]
        # Get quadrotor  velocities
        vquadrotor2 = self.xq_2[3:6]

        xquadrotor3 = self.xq_3[0:3]
        # Get quadrotor  velocities
        vquadrotor3 = self.xq_3[3:6]

        # Get Full vector quadrotor
        x_quadrotors = np.hstack((xquadrotor1, xquadrotor2, xquadrotor3))
        v_quadrotors = np.hstack((vquadrotor1, vquadrotor2, vquadrotor3))

        # Compute unit Vector
        unit = np.array(self.payload_to_quadrotor_unit(x[0:3], x_quadrotors)).reshape((self.robot_num*3, ))

        # Extended Vector of the Payload
        payload_states = np.hstack((x, unit))

        ## Compute cable angular velocity
        r = np.array(
            self.cable_angular_velocity(x, x_quadrotors, v_quadrotors),
            dtype=np.double,
        ).reshape((self.robot_num * 3,))

        self.x_0 = np.hstack((x, unit, r))
        self.publish_cable_direction(unit)
        self.publish_cable_angular_velocity(r)
        #self.try_initialize_reference()
        return None

    def publish_cable_angular_velocity(self, r: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(r, dtype=np.double).reshape((self.robot_num * 3,)).tolist()
        self.publisher_cable_angular_velocity.publish(msg)
        return None

    def publish_cable_direction(self, unit: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(unit, dtype=np.double).reshape((self.robot_num * 3,)).tolist()
        self.publisher_cable_direction.publish(msg)
        return None

    def update_reference_from_plan(self, t_query: float):
        times = self.reference_plan["t"]
        idx = int(np.clip(np.searchsorted(times, t_query, side="left"), 0, len(times) - 1))

        self.xd[0:3] = self.reference_plan["payload_p"][idx]
        self.xd[3:6] = self.reference_plan["payload_v"][idx]

        q_ref = self.reference_plan["q"][:, idx, :]
        qdot_ref = self.reference_plan["qdot"][:, idx, :]
        r_ref = np.cross(q_ref, qdot_ref)

        self.xd[6:15] = q_ref.reshape((self.robot_num * 3,))
        self.xd[15:24] = r_ref.reshape((self.robot_num * 3,))
        self.ud[:] = self.reference_plan["quad_a"][:, idx, :].reshape((self.robot_num * 3,))
        return None

    def callback_get_odometry_drone_1(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        x[3] = msg.twist.twist.linear.x
        x[4] = msg.twist.twist.linear.y
        x[5] = msg.twist.twist.linear.z

        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w
    
        # Put values in the vector
        self.xq_1 = x
        return None

    def callback_get_odometry_drone_2(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        x[3] = msg.twist.twist.linear.x
        x[4] = msg.twist.twist.linear.y
        x[5] = msg.twist.twist.linear.z

        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w
    
        # Put values in the vector
        self.xq_2 = x

    def callback_get_odometry_drone_3(self, msg):
        # Empty Vector for classical formulation
        x = np.zeros((13, ))

        # Get positions of the system
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        # Get linear velocities Inertial frame
        x[3] = msg.twist.twist.linear.x
        x[4] = msg.twist.twist.linear.y
        x[5] = msg.twist.twist.linear.z

        # Get angular velocity body frame
        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z
        
        # Get quaternions
        x[7] = msg.pose.pose.orientation.x
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w
    
        # Put values in the vector
        self.xq_3 = x

    def publish_transforms(self):
        # Payload
        tf_world_load = TransformStamped()
        tf_world_load.header.stamp = self.get_clock().now().to_msg()
        tf_world_load.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load.child_frame_id = 'payload'          # <-- imu_link is rotated

        tf_world_load.transform.translation.x = self.x_0[0]
        tf_world_load.transform.translation.y = self.x_0[1]
        tf_world_load.transform.translation.z = self.x_0[2]

        tf_world_load.transform.rotation.w = 1.0
        tf_world_load.transform.rotation.x = 0.0
        tf_world_load.transform.rotation.y = 0.0
        tf_world_load.transform.rotation.z = 0.0

        # Payload Verification with unit vector
        tf_world_load_verification = TransformStamped()
        tf_world_load_verification.header.stamp = self.get_clock().now().to_msg()
        tf_world_load_verification.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load_verification.child_frame_id = 'payload_verification_q_1'          # <-- imu_link is rotated

        tf_world_load_verification.transform.translation.x = self.xq_1[0] + self.x_0[6]*self.length
        tf_world_load_verification.transform.translation.y = self.xq_1[1] + self.x_0[7]*self.length
        tf_world_load_verification.transform.translation.z = self.xq_1[2] + self.x_0[8]*self.length

        tf_world_load_verification.transform.rotation.w = 1.0
        tf_world_load_verification.transform.rotation.x = 0.0
        tf_world_load_verification.transform.rotation.y = 0.0
        tf_world_load_verification.transform.rotation.z = 0.0

        tf_world_load_verification_2 = TransformStamped()
        tf_world_load_verification_2.header.stamp = self.get_clock().now().to_msg()
        tf_world_load_verification_2.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load_verification_2.child_frame_id = 'payload_verification_q_2'          # <-- imu_link is rotated

        tf_world_load_verification_2.transform.translation.x = self.xq_2[0] + self.x_0[9]*self.length
        tf_world_load_verification_2.transform.translation.y = self.xq_2[1] + self.x_0[10]*self.length
        tf_world_load_verification_2.transform.translation.z = self.xq_2[2] + self.x_0[11]*self.length

        tf_world_load_verification_2.transform.rotation.w = 1.0
        tf_world_load_verification_2.transform.rotation.x = 0.0
        tf_world_load_verification_2.transform.rotation.y = 0.0
        tf_world_load_verification_2.transform.rotation.z = 0.0

        tf_world_load_verification_3 = TransformStamped()
        tf_world_load_verification_3.header.stamp = self.get_clock().now().to_msg()
        tf_world_load_verification_3.header.frame_id = 'world'            # <-- world is the parent
        tf_world_load_verification_3.child_frame_id = 'payload_verification_q_3'          # <-- imu_link is rotated

        tf_world_load_verification_3.transform.translation.x = self.xq_3[0] + self.x_0[12]*self.length
        tf_world_load_verification_3.transform.translation.y = self.xq_3[1] + self.x_0[13]*self.length
        tf_world_load_verification_3.transform.translation.z = self.xq_3[2] + self.x_0[14]*self.length

        tf_world_load_verification_3.transform.rotation.w = 1.0
        tf_world_load_verification_3.transform.rotation.x = 0.0
        tf_world_load_verification_3.transform.rotation.y = 0.0
        tf_world_load_verification_3.transform.rotation.z = 0.0

        # Quadrotor
        tf_world_quad1 = TransformStamped()
        tf_world_quad1.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad1.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad1.child_frame_id = 'quadrotor1'          # <-- imu_link is rotated

        tf_world_quad1.transform.translation.x = self.xq_1[0]
        tf_world_quad1.transform.translation.y = self.xq_1[1]
        tf_world_quad1.transform.translation.z = self.xq_1[2]

        tf_world_quad1.transform.rotation.x = self.xq_1[7]
        tf_world_quad1.transform.rotation.y = self.xq_1[8]
        tf_world_quad1.transform.rotation.z = self.xq_1[9]
        tf_world_quad1.transform.rotation.w = self.xq_1[6]

        tf_world_quad2 = TransformStamped()
        tf_world_quad2.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad2.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad2.child_frame_id = 'quadrotor2'          # <-- imu_link is rotated

        tf_world_quad2.transform.translation.x = self.xq_2[0]
        tf_world_quad2.transform.translation.y = self.xq_2[1]
        tf_world_quad2.transform.translation.z = self.xq_2[2]

        tf_world_quad2.transform.rotation.x = self.xq_2[7]
        tf_world_quad2.transform.rotation.y = self.xq_2[8]
        tf_world_quad2.transform.rotation.z = self.xq_2[9]
        tf_world_quad2.transform.rotation.w = self.xq_2[6]

        tf_world_quad3 = TransformStamped()
        tf_world_quad3.header.stamp = self.get_clock().now().to_msg()
        tf_world_quad3.header.frame_id = 'world'            # <-- world is the parent
        tf_world_quad3.child_frame_id = 'quadrotor3'          # <-- imu_link is rotated

        tf_world_quad3.transform.translation.x = self.xq_3[0]
        tf_world_quad3.transform.translation.y = self.xq_3[1]
        tf_world_quad3.transform.translation.z = self.xq_3[2]

        tf_world_quad3.transform.rotation.x = self.xq_3[7]
        tf_world_quad3.transform.rotation.y = self.xq_3[8]
        tf_world_quad3.transform.rotation.z = self.xq_3[9]
        tf_world_quad3.transform.rotation.w = self.xq_3[6]

        self.tf_broadcaster.sendTransform([tf_world_load, tf_world_quad1, tf_world_load_verification, tf_world_load_verification_2, tf_world_load_verification_3, tf_world_quad2, tf_world_quad3])
        return None

    def payloadModel(self) -> AcadosModel:
        model_name = "planner_payload_pointmass_multiple"
        p_x = ca.MX.sym("p_x")
        p_y = ca.MX.sym("p_y")
        p_z = ca.MX.sym("p_z")
        x_p = ca.vertcat(p_x, p_y, p_z)

        vx_p = ca.MX.sym("vx_p")
        vy_p = ca.MX.sym("vy_p")
        vz_p = ca.MX.sym("vz_p")
        v_p = ca.vertcat(vx_p, vy_p, vz_p)

        # Cable kinematics
        nx_1 = ca.MX.sym('nx_1')
        ny_1 = ca.MX.sym('ny_1')
        nz_1 = ca.MX.sym('nz_1')
        n1 = ca.vertcat(nx_1, ny_1, nz_1)

        nx_2 = ca.MX.sym('nx_2')
        ny_2 = ca.MX.sym('ny_2')
        nz_2 = ca.MX.sym('nz_2')
        n2 = ca.vertcat(nx_2, ny_2, nz_2)

        nx_3 = ca.MX.sym('nx_3')
        ny_3 = ca.MX.sym('ny_3')
        nz_3 = ca.MX.sym('nz_3')
        n3 = ca.vertcat(nx_3, ny_3, nz_3)

        # Cable kinematics
        rx_1 = ca.MX.sym('rx_1')
        ry_1 = ca.MX.sym('ry_1')
        rz_1 = ca.MX.sym('rz_1')
        r1 = ca.vertcat(rx_1, ry_1, rz_1)

        rx_2 = ca.MX.sym('rx_2')
        ry_2 = ca.MX.sym('ry_2')
        rz_2 = ca.MX.sym('rz_2')
        r2 = ca.vertcat(rx_2, ry_2, rz_2)

        rx_3 = ca.MX.sym('rx_3')
        ry_3 = ca.MX.sym('ry_3')
        rz_3 = ca.MX.sym('rz_3')
        r3 = ca.vertcat(rx_3, ry_3, rz_3)
        
        # Full states of the system
        x = ca.vertcat(x_p, v_p, n1, n2, n3, r1, r2, r3)
        
        # Control actions acceleration of each quadrotor
        ax_q1 = ca.MX.sym("ax_q1")
        ay_q1 = ca.MX.sym("ay_q1")
        az_q1 = ca.MX.sym("az_q1")
        a_1 = ca.vertcat(ax_q1, ay_q1, az_q1)

        ax_q2 = ca.MX.sym("ax_q2")
        ay_q2 = ca.MX.sym("ay_q2")
        az_q2 = ca.MX.sym("az_q2")
        a_2 = ca.vertcat(ax_q2, ay_q2, az_q2)

        ax_q3 = ca.MX.sym("ax_q3")
        ay_q3 = ca.MX.sym("ay_q3")
        az_q3 = ca.MX.sym("az_q3")
        a_3 = ca.vertcat(ax_q3, ay_q3, az_q3)
        u = ca.vertcat(a_1, a_2, a_3)
        
        # Matrix of cable directions
        N = ca.hcat([n1, n2, n3])

        # Matrix of cable angular velocities
        W = ca.hcat([r1, r2, r3])

        # Matrix of control actions
        U = ca.hcat([a_1, a_2, a_3])

        d_1 = ca.dot(N[:, 0], U[:, 0]) - self.length * ca.dot(W[:, 0], W[:, 0])
        d_2 = ca.dot(N[:, 1], U[:, 1]) - self.length * ca.dot(W[:, 1], W[:, 1])
        d_3 = ca.dot(N[:, 2], U[:, 2]) - self.length * ca.dot(W[:, 2], W[:, 2])

        
        m = self.mass
        I3 = ca.MX.eye(3)
        z = ca.MX.zeros(1, 1)
        M = ca.vertcat(
            ca.hcat([m * I3, n1, n2, n3]),
            ca.hcat([n1.T, z, z, z]),
            ca.hcat([n2.T, z, z, z]),
            ca.hcat([n3.T, z, z, z]),
        )



        linear_velocity = v_p
        gravity_vec = self.gravity * self.e3
        gravity_vec_mass = -m*self.gravity * self.e3

        b = ca.vertcat(gravity_vec_mass, d_1, d_2, d_3)

        acceleration_tension = ca.solve(M, b)

        linear_acceleration = acceleration_tension[0:3]
        tensions_expresion = acceleration_tension[3:6]

        k_n = self.unit_vector_stabilization_gain
        k_r = self.angular_orthogonality_gain
        a_p = acceleration_tension[0:3]

        n1_dot = ca.cross(r1, n1)
        r1_dot = (1.0 / self.length) * ca.cross(n1, (a_p - U[:, 0]))

        n2_dot = ca.cross(r2, n2)
        r2_dot = (1.0 / self.length) * ca.cross(n2, (a_p - U[:, 1]))

        n3_dot = ca.cross(r3, n3)
        r3_dot = (1.0 / self.length) * ca.cross(n3, (a_p - U[:, 2]))

        f_expl = ca.vertcat(linear_velocity, linear_acceleration, n1_dot, n2_dot, n3_dot, r1_dot, r2_dot, r3_dot)

        nx = x.shape[0]
        x_dot = ca.MX.sym("x_dot", nx, 1)
        f_impl_expr = x_dot - f_expl

        ref_params = ca.MX.sym("ref_params", nx + u.shape[0], 1)
        cost_params = ca.MX.sym("cost_params", nx + nx + u.shape[0], 1)

        model = AcadosModel()
        model.x = x
        model.xdot = x_dot
        model.x_dot = x_dot
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl_expr
        model.u = u
        #model.p = ca.vertcat(ref_params, cost_params)
        model.p = ref_params
        model.name = model_name
        return model, tensions_expresion

    def solver(self, x0):
        model, tensions_expresion = self.payloadModel()

        ocp = AcadosOcp()
        ocp.model = model
        ocp.code_gen_opts.code_export_directory = str(self.code_export_directory)

        nx = model.x.size()[0]
        nu = model.u.size()[0]

        ocp.dims.N = self.N_prediction
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        x = ocp.model.x
        u = ocp.model.u
        p = ocp.model.p
        
        print("OCP DIMENSIONS")
        print(x.shape)
        print(u.shape)
        print(p.shape)
        
        ## Split Values from the states and desired states
        x_p = x[0:3]
        v_p = x[3:6]

        n1 = x[6:9]
        n2 = x[9:12]
        n3 = x[12:15]

        r1 = x[15:18]
        r2 = x[18:21]
        r3 = x[21:24]
        
        # Split control actions
        a_q1 = u[0:3]
        a_q2 = u[3:6]
        a_q3 = u[6:9]
        
        ## Split desired values
        x_p_d = p[0:3]
        v_p_d = p[3:6]

        n1_d = p[6:9]
        n2_d = p[9:12]
        n3_d = p[12:15]

        r1_d = p[15:18]
        r2_d = p[18:21]
        r3_d = p[21:24]

        error_position = x_p - x_p_d
        error_velocity = v_p - v_p_d

        error_n1 = ca.cross(n1_d, n1)
        error_n2 = ca.cross(n2_d, n2)
        error_n3 = ca.cross(n3_d, n3)

        tangent_projector_1 = ca.MX.eye(3) - n1 @ n1.T
        tangent_projector_2 = ca.MX.eye(3) - n2 @ n2.T
        tangent_projector_3 = ca.MX.eye(3) - n3 @ n3.T

        r1_error = r1 - tangent_projector_1 @ r1_d
        r2_error = r2 - tangent_projector_2 @ r2_d
        r3_error = r3 - tangent_projector_3 @ r3_d

        xq1 = x_p - self.length * n1
        xq2 = x_p - self.length * n2
        xq3 = x_p - self.length * n3

        xq1_d = x_p_d - self.length * n1_d
        xq2_d = x_p_d - self.length * n2_d
        xq3_d = x_p_d - self.length * n3_d

        xq1_error = xq1 - xq1_d
        xq2_error = xq2 - xq2_d
        xq3_error = xq3 - xq3_d


        #orthogonality_error = ca.dot(n1, r1)
        #tension_expr = self.mass * (
        #    self.length * ca.dot(r1, r1)
        #    - ca.dot(n1, (a_q + self.gravity * self.e3))
        #)

        lyapunov_position = (
            100.0 * self.kp_min * (error_position.T @ error_position)
            + 0.5 * self.kv_min * self.mass * (error_velocity.T @ error_velocity)
        )

        ocp.model.cost_expr_ext_cost = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_cable_direction * (error_n2.T @ error_n2)
            + self.weight_cable_direction * (error_n3.T @ error_n3)
            + self.weight_quadrotor_position * (xq1_error.T @ xq1_error)
            + self.weight_quadrotor_position * (xq2_error.T @ xq2_error)
            + self.weight_quadrotor_position * (xq3_error.T @ xq3_error)
            + self.weight_r * (r1_error.T @ r1_error)
            + self.weight_r * (r2_error.T @ r2_error)
            + self.weight_r * (r3_error.T @ r3_error)
            + self.weight_acceleration * (a_q1.T @ a_q1)
            + self.weight_acceleration * (a_q2.T @ a_q2)
            + self.weight_acceleration * (a_q3.T @ a_q3))
        #    + self.weight_orthogonality * (orthogonality_error ** 2)
        #)
        ocp.model.cost_expr_ext_cost_e = (
            lyapunov_position
            + self.weight_cable_direction * (error_n1.T @ error_n1)
            + self.weight_cable_direction * (error_n2.T @ error_n2)
            + self.weight_cable_direction * (error_n3.T @ error_n3)
            + self.weight_quadrotor_position * (xq1_error.T @ xq1_error)
            + self.weight_quadrotor_position * (xq2_error.T @ xq2_error)
            + self.weight_quadrotor_position * (xq3_error.T @ xq3_error)
            + self.weight_r * (r1_error.T @ r1_error)
            + self.weight_r * (r2_error.T @ r2_error)
            + self.weight_r * (r3_error.T @ r3_error))

        ref_params = np.hstack((self.x_0, self.u_equilibrium))
        cost_params = np.zeros((nx + nx + nu,), dtype=np.double)
        #ocp.parameter_values = np.concatenate([ref_params, cost_params])
        ocp.parameter_values = ref_params

        ocp.constraints.constr_type = "BGH"
        ocp.constraints.lbu = self.u_min
        ocp.constraints.ubu = self.u_max
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        ocp.constraints.x0 = x0

        ocp.model.con_h_expr = ca.vertcat(
            ca.dot(n1, n1),
            ca.dot(n2, n2),
            ca.dot(n3, n3),
            tensions_expresion,
        )
        nh = 6
        nsh = nh
        ocp.cost.zl = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.cost.Zl = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.cost.zu = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.cost.Zu = self.norm_constraint_slack_weight * np.ones((nsh,))
        ocp.constraints.lh = np.concatenate(
            (
                np.full((3,), 1.0 - self.unit_vector_norm_tol, dtype=np.double),
                np.asarray(self.tension_min, dtype=np.double).reshape((3,)),
            )
        )
        ocp.constraints.uh = np.concatenate(
            (
                np.full((3,), 1.0 + self.unit_vector_norm_tol, dtype=np.double),
                np.asarray(self.tension_max, dtype=np.double).reshape((3,)),
            )
        )
        ocp.constraints.lsh = np.zeros((nsh,))
        ocp.constraints.ush = np.zeros((nsh,))
        ocp.constraints.idxsh = np.array(range(nsh), dtype=np.int32)

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = self.N_prediction
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 2
        ocp.solver_options.sim_method_newton_iter = 20
        ocp.solver_options.sim_method_newton_tol = 1e-10
        ocp.solver_options.levenberg_marquardt = 1.0
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.nlp_solver_max_iter = 2
        ocp.solver_options.Tsim = self.ts
        ocp.solver_options.tf = self.t_N
        ocp.solver_options.N_horizon = self.N_prediction
        ocp.solver_options.regularize_method = "CONVEXIFY"
        return ocp

    def quadrotor_position_c(self):
        x = ca.MX.sym('x', 3, 1)
        n = ca.MX.sym('n', 3*self.robot_num, 1)  # general: 3 thrust comps + 3m 'r' comps
        n_matrix = ca.reshape(n, 3, self.robot_num)

        # unpack state
        x_p   = x[0:3]      # 3x1

        # Vectorized expression:
        cols = []
        for k in range(self.robot_num):
            quadrotor = x_p - (self.length * n_matrix[:, k])  # 3 x m
            cols.append(quadrotor)

        quadrotors_location = ca.hcat(cols)             # 3 x m
        quadrotors_location_vec = ca.reshape(quadrotors_location, 3*self.robot_num, 1)  # (3m) x 1
        quadrotors_location_funtion = ca.Function('quadrotors_location', [x, n], [quadrotors_location_vec])
        return quadrotors_location_funtion

    def quadrotor_velocity_c(self):

        x = ca.MX.sym('x', 3, 1)

        n = ca.MX.sym('n', 3*self.robot_num, 1)  # general: 3 thrust comps + 3m 'r' comps
        n_matrix = ca.reshape(n, 3, self.robot_num)

        w = ca.MX.sym('w', 3*self.robot_num, 1)  # general: 3 thrust comps + 3m 'r' comps
        w_matrix = ca.reshape(w, 3, self.robot_num)

        # unpack state
        v_p = x[0:3]

        cols = []
        for k in range(self.robot_num):
            r_p = w_matrix[:, k]
            n_p = n_matrix[:, k]
            term_n   = self.length * ca.cross(r_p, n_p)
            v_k      = v_p - term_n     
            cols.append(v_k)

        quadrotors_velocity = ca.hcat(cols)             # 3 x m
        quadrotors_velocity_vec = ca.reshape(quadrotors_velocity, 3*self.robot_num, 1)  # (3m) x 1
        quadrotors_velocity_funtion = ca.Function('quadrotors_velocity', [x, n, w], [quadrotors_velocity_vec])
        return quadrotors_velocity_funtion

    def publish_prediction(self):
        # Create one Path message per drone
        path_msgs = []
        payload_msgs = []

        # Quadrotors
        for i in range(self.robot_num):
            msg = Path()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "world"
            path_msgs.append(msg)
        
        # Payload
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        payload_msgs.append(msg)
        
        # Fill poses for each drone
        for k in range(self.N_prediction):
            x_k = self.acados_ocp_solver.get(k, "x")
            xq = np.array(self.quadrotors_position(x_k[0:3], x_k[6:15])).reshape((self.robot_num * 3,))

            # Quadrotor positions
            for i in range(self.robot_num):
                pose = PoseStamped()
                pose.header = path_msgs[i].header
                pose.pose.position.x = xq[3*i + 0]
                pose.pose.position.y = xq[3*i + 1]
                pose.pose.position.z = xq[3*i + 2]
                path_msgs[i].poses.append(pose)

            # Payload positions
            pose = PoseStamped()
            pose.header = payload_msgs[0].header
            pose.pose.position.x = x_k[0]
            pose.pose.position.y = x_k[1]
            pose.pose.position.z = x_k[2]
            payload_msgs[0].poses.append(pose)

        # Publish drone and payload desired path
        self.publisher_prediction_quadrotor_1.publish(path_msgs[0])
        self.publisher_prediction_quadrotor_2.publish(path_msgs[1])
        self.publisher_prediction_quadrotor_3.publish(path_msgs[2])
        self.publisher_prediction_payload.publish(payload_msgs[0])
    
    def send_position_cmd(self, publisher, x, v, a, tension, direction):
        position_cmd_msg = PositionCommand()
        position_cmd_msg.position.x = x[0]
        position_cmd_msg.position.y = x[1]
        position_cmd_msg.position.z = x[2]

        position_cmd_msg.velocity.x = v[0]
        position_cmd_msg.velocity.y = v[1]
        position_cmd_msg.velocity.z = v[2]
        
        position_cmd_msg.acceleration.x = a[0]
        position_cmd_msg.acceleration.y = a[1]
        position_cmd_msg.acceleration.z = a[2]

        cable_force = tension*direction

        position_cmd_msg.cable_force.x = cable_force[0]
        position_cmd_msg.cable_force.y = cable_force[1]
        position_cmd_msg.cable_force.z = cable_force[2]

        publisher.publish(position_cmd_msg)
        return None

    def cable_tension_c(self):
        x = ca.MX.sym("x", 24, 1)
        u = ca.MX.sym("u", 9, 1)

        x_p = x[0:3]
        v_p = x[3:6]

        n1 = x[6:9]
        n2 = x[9:12]
        n3 = x[12:15]

        r1 = x[15:18]
        r2 = x[18:21]
        r3 = x[21:24]

        a_1 = u[0:3]
        a_2 = u[3:6]
        a_3 = u[6:9]

        N = ca.hcat([n1, n2, n3])
        U = ca.hcat([a_1, a_2, a_3])
        W = ca.hcat([r1, r2, r3])

        d_1 = ca.dot(N[:, 0], U[:, 0]) - self.length * ca.dot(W[:, 0], W[:, 0])
        d_2 = ca.dot(N[:, 1], U[:, 1]) - self.length * ca.dot(W[:, 1], W[:, 1])
        d_3 = ca.dot(N[:, 2], U[:, 2]) - self.length * ca.dot(W[:, 2], W[:, 2])

        m = self.mass
        I3 = ca.MX.eye(3)
        z = ca.MX.zeros(1, 1)

        M = ca.vertcat(
            ca.hcat([m * I3, n1, n2, n3]),
            ca.hcat([n1.T, z, z, z]),
            ca.hcat([n2.T, z, z, z]),
            ca.hcat([n3.T, z, z, z]),
        )

        b = ca.vertcat(
            -m * self.gravity * self.e3,
            d_1,
            d_2,
            d_3,
        )

        solution = ca.solve(M, b)
        tensions = solution[3:6]
        return ca.Function("cable_tensions", [x, u], [tensions])

    def prepare(self):
        if self.flag == 0:
            self.flag = 1
            self.reference_start_time = time.monotonic()
            # Init Optimization Problem
            for k in range(5000):
                arr_str = np.array2string(self.x_0, precision=3, separator=", ", suppress_small=True)
                #self.get_logger().info(f"state[] = {arr_str}")
    
            self.ocp = self.solver(self.x_0)
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=str(self.json_file), build=True, generate=True)
            ### Reset Solver
            self.acados_ocp_solver.reset()
    
            ### Initial Conditions optimization problem
            for stage in range(self.N_prediction + 1):
                self.acados_ocp_solver.set(stage, "x", self.x_0)
            for stage in range(self.N_prediction):
                self.acados_ocp_solver.set(stage, "u", self.ud)
        return None

    def run(self):
        self.prepare()

        if not np.all(np.isfinite(self.x_0)):
            self.get_logger().error("Skipping MPC solve because x_0 contains non-finite values.")
            return None

        if self.reference_start_time is None:
            self.reference_start_time = time.monotonic()
        elapsed = min(time.monotonic() - self.reference_start_time, float(self.reference_plan["t"][-1]))
        self.update_reference_from_plan(elapsed)

        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        # Keep the SQP_RTI iterate close to the current measured state.
        for stage in range(self.N_prediction + 1):
            self.acados_ocp_solver.set(stage, "x", self.x_0)
        for stage in range(self.N_prediction):
            self.acados_ocp_solver.set(stage, "u", self.ud)

        # Desired Trajectory of the system
        for j in range(self.N_prediction):
            yref = self.xd
            uref = self.ud
            aux_ref = np.hstack((yref, uref))
            self.acados_ocp_solver.set(j, "p", aux_ref)
        # Desired Trayectory at the last Horizon
        yref_N = self.xd
        uref_N = self.ud
        aux_ref_N = np.hstack((yref_N, uref_N))
        self.acados_ocp_solver.set(self.N_prediction, "p", aux_ref_N)
        # Check Solution since there can be possible errors 
        status = self.acados_ocp_solver.solve()
        if status != 0:
            self.get_logger().error(f"acados solver failed with status {status}")
            return None

        # get Control Actions and predictions
        u = self.acados_ocp_solver.get(0, "u")
        x_k = self.acados_ocp_solver.get(1, "x")

        self.publish_prediction()
        
        # This compute the position velocity and acceleration of each quadrotor
        xQ = np.array(self.quadrotors_position(x_k[0:3], x_k[6:15])).reshape((self.robot_num*3, ))
        xQ_dot = np.array(self.quadrotors_velocity(x_k[3:6], x_k[6:15], x_k[15:24])).reshape((self.robot_num*3, ))
        xQ_dot_dot = u
        tensions = self.tensions(x_k, u)

        self.send_position_cmd(
            self.publisher_ref_quadrotor_1,
            xQ[0:3],
            xQ_dot[0:3],
            xQ_dot_dot[0:3],
            float(tensions[0]),
            x_k[6:9],
        )

        self.send_position_cmd(
            self.publisher_ref_quadrotor_2,
            xQ[3:6],
            xQ_dot[3:6],
            xQ_dot_dot[3:6],
            float(tensions[1]),
            x_k[9:12],
        )

        self.send_position_cmd(
            self.publisher_ref_quadrotor_3,
            xQ[6:9],
            xQ_dot[6:9],
            xQ_dot_dot[6:9],
            float(tensions[2]),
            x_k[12:15],
        )
        self.get_logger().info("Solving the MPC problem")
        self.publish_transforms()


def main(arg = None):
    rclpy.init(args=arg)
    payload_node = PayloadControlMujocoMultiplePointMass()
    try:
        rclpy.spin(payload_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        payload_node.get_logger().info('Simulation stopped manually.')
    finally:
        payload_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()
