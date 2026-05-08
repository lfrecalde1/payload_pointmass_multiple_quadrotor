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

class PayloadControlMujocoMultiplePointMass(Node):
    def __init__(self):
        super().__init__('MultiplePointMass')

        # Runtime parameters (mirrors dq_nmpc style parameterization).
        self.declare_parameter('planner.ts', 0.05)
        self.declare_parameter('planner.horizon_time', 2.0)
        self.declare_parameter('nmpc.jerk_limit', [10.0, 10.0, 10.0])

        # Time Definition
        self.ts = float(self.get_parameter('planner.ts').value)
        self.final = 30

        # Prediction Node of the NMPC formulation
        self.t_N = float(self.get_parameter('planner.horizon_time').value)
        self.N = np.arange(0, self.t_N + self.ts, self.ts)
        self.N_prediction = self.N.shape[0]
        print(self.N_prediction)

        # Internal parameters defintion
        self.robot_num = 3
        self.mass = 0.2
        self.gravity = 9.81


        # Quadrotor paramaters
        self.mass_quad = 1.24

        # Cable length
        self.length = 0.76
        self.e3 = ca.DM([0, 0, 1])

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
        print("Tensions")
        print(tensions_eq)
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
        self.aq_init = np.array([0.0, 0.0, 0.0], dtype=np.double)
        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init))
        print(self.x_0)


        ## Acceleration input and acceleration-state initialization.
        self.u_equilibrium = np.array([0.0, 0.0, 0.0], dtype=np.double)

        ## Bounds for jerk input [m/s^3].
        self.acceleration_limit = np.array(self.get_parameter('nmpc.jerk_limit').value, dtype=np.double).reshape((3,))
        self.u_min = -self.acceleration_limit.copy()
        self.u_max = self.acceleration_limit.copy()

        ## Define state dimension and control action
        self.n_x = self.x_0.shape[0]
        self.n_u = self.u_equilibrium.shape[0]

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

        ## Subcriber of each drone
        self.subscriber_drone_1_ = self.create_subscription(Odometry, "/quadrotor1/odom", self.callback_get_odometry_drone_1, 10)
        self.subscriber_drone_2_ = self.create_subscription(Odometry, "/quadrotor2/odom", self.callback_get_odometry_drone_2, 10)
        self.subscriber_drone_3_ = self.create_subscription(Odometry, "/quadrotor3/odom", self.callback_get_odometry_drone_3, 10)

        ## TF We can verify cable direction if they make sense or not
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(self.ts, self.run)


    def build_hover_equilibrium_lambdas(self, payload_mass: float, gravity: float, q_eq_list: List[np.ndarray]):
        N = np.column_stack(q_eq_list)  # 3x3
        print(N)
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
            n_k      = (term/ca.norm_2(term))
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
            n_k      = (term/ca.norm_2(term))

            x_Q = xQ_p_matrix[:, k]
            v_Q = xQ_v_matrix[:, k]

            a = x_p - x_Q
            norm_a = ca.norm_2(a)
            dot_a = a.T@a
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
        self.publish_cable_angular_velocity(r)
        #self.try_initialize_reference()

        arr_str = np.array2string(self.x_0, precision=3, separator=', ', suppress_small=True)
        self.get_logger().info(f"x_0 = {arr_str}")
        return None

    def publish_cable_angular_velocity(self, r: np.ndarray):
        msg = Float64MultiArray()
        msg.data = np.asarray(r, dtype=np.double).reshape((self.robot_num * 3,)).tolist()
        self.publisher_cable_angular_velocity.publish(msg)
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
    
    def run(self):
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
