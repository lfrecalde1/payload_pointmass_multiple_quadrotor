#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import casadi as ca
from casadi import Function
from pathlib import Path as FilePath
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
from std_msgs.msg import Float64, Float64MultiArray
from typing import Dict, List
from payload_pointmass_multiple_quadrotor.lim_min_multiple_simple import (
    ACCELS_PLOT_PATH as POINT_TO_POINT_ACCELS_PLOT_PATH,
    HAS_MPL,
    SIGNALS_PLOT_PATH as POINT_TO_POINT_SIGNALS_PLOT_PATH,
    plan_three_quad_point_mass,
    plot_signal_diagnostics,
    verify_smoothness,
)
from payload_pointmass_multiple_quadrotor.lissajous_multiple_simple import (
    ACCELS_PLOT_PATH as LISSAJOUS_ACCELS_PLOT_PATH,
    SIGNALS_PLOT_PATH as LISSAJOUS_SIGNALS_PLOT_PATH,
    plan_three_quad_lissajous_payload,
)
if HAS_MPL:
    import matplotlib.pyplot as plt

class PayloadControlMujocoMultiplePointMass(Node):
    def __init__(self):
        super().__init__('MultiplePointMass')

        # Runtime parameters (mirrors dq_nmpc style parameterization).

        # Time Definition
        self.t_N = 1.5
        self.N_prediction = int(31)
        self.ts = self.t_N / self.N_prediction
        self.planner_duration = 5.0
        self.planner_type = "lissajous"

        self.reference_start_time = None
        self.results_saved = False
        self.tracking_log = []
        self.tracking_npz_path = FilePath(__file__).with_name("controller_tracking_results.npz")
        self.tracking_plot_path = FilePath(__file__).with_name("controller_tracking_comparison.png")
        self.tracking_metrics_plot_path = FilePath(__file__).with_name("controller_tracking_metrics.png")
        self.reference_plan_signals_path = POINT_TO_POINT_SIGNALS_PLOT_PATH
        self.reference_plan_accels_path = POINT_TO_POINT_ACCELS_PLOT_PATH

        # Prediction Node of the NMPC formulation
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


        ##  ----------------------------------------------------------------- Funtion Casadi ---------------------------------
        self.payload_to_quadrotor_unit = self.quadrotor_payload_unit_vector_c()
        self.cable_angular_velocity = self.cable_angular_velocity_c()
        self.quadrotors_position = self.quadrotor_position_c()
        self.quadrotors_velocity = self.quadrotor_velocity_c()
        self.tensions = self.cable_tension_c()
        ##  ----------------------------------------------------------------- Funtion Casadi ---------------------------------

        unit_vectors_init = self.payload_to_quadrotor_unit(pos_0, np.hstack((pos_quad_1, pos_quad_2, pos_quad_3)))
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

        self.tension_min = 0.5*np.array(tensions_eq)
        self.tension_max = 10*np.array(tensions_eq)
        
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
        self.x_0 = np.hstack((pos_0, vel_0, self.n_init, self.r_init))

        ## Acceleration input equilibrium of each quadrotor
        self.u_equilibrium = np.array([0.0, 0.0, 0.0]*self.robot_num, dtype=np.double)

        ## Bounds for jerk input [m/s^3].
        self.acceleration_limit = np.array([50.0, 50.0, 50.0]*self.robot_num, dtype=np.double)
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
        self.planner_goal = np.array([5.0, 1.2, 0.3], dtype=np.double)
        self.lissajous_offset = pos_0.copy()
        self.lissajous_x_amp = 3.0
        self.lissajous_y_amp = 0.5
        self.lissajous_z_amp = 0.5
        self.lissajous_period = 4.0
        self.lissajous_num_cycles = 4.0
        self.lissajous_ramp_time = 6.0
        self.lissajous_x_num_periods = 1.0
        self.lissajous_y_num_periods = 2.0
        self.lissajous_z_num_periods = 1.0

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
            norm_term = ca.sqrt(ca.dot(term, term))
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
            x_Q = xQ_p_matrix[:, k]
            v_Q = xQ_v_matrix[:, k]

            term = x_p - x_Q
            # Cable Direction
            norm_term = ca.sqrt(ca.dot(term, term))
            n_k = term / norm_term
            
            # This is for the cable angular velocity
            a = x_p - x_Q
            norm_a = ca.sqrt(ca.dot(a, a))
            dot_a = ca.dot(a, a)
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

        ## Compute cable angular velocity
        r = np.array(
            self.cable_angular_velocity(x, x_quadrotors, v_quadrotors),
            dtype=np.double,
        ).reshape((self.robot_num * 3,))

        self.x_0 = np.hstack((x, unit, r))
        self.publish_cable_direction(unit)
        self.publish_cable_angular_velocity(r)
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

    def reference_from_plan(self, t_query: float):
        times = self.reference_plan["t"]
        idx = int(np.clip(np.searchsorted(times, t_query, side="left"), 0, len(times) - 1))

        xd = np.zeros((self.n_x,), dtype=np.double)
        ud = np.zeros((self.n_u,), dtype=np.double)

        xd[0:3] = self.reference_plan["payload_p"][idx]
        xd[3:6] = self.reference_plan["payload_v"][idx]

        q_ref = self.reference_plan["q"][:, idx, :]
        qdot_ref = self.reference_plan["qdot"][:, idx, :]
        r_ref = np.cross(q_ref, qdot_ref)

        xd[6:15] = q_ref.reshape((self.robot_num * 3,))
        xd[15:24] = r_ref.reshape((self.robot_num * 3,))
        ud[:] = self.reference_plan["quad_a"][:, idx, :].reshape((self.robot_num * 3,))
        return xd, ud

    def update_reference_from_plan(self, t_query: float):
        self.xd, self.ud = self.reference_from_plan(t_query)
        return None

    def build_reference_plan(self):
        common_kwargs = {
            "n_samples": max(self.N_prediction + 1, 201),
            "payload_mass": self.mass,
            "gravity": self.gravity,
            "cable_lengths": np.full((self.robot_num,), self.length, dtype=np.double),
        }

        if self.planner_type == "lissajous":
            self.reference_plan_signals_path = LISSAJOUS_SIGNALS_PLOT_PATH
            self.reference_plan_accels_path = LISSAJOUS_ACCELS_PLOT_PATH
            return plan_three_quad_lissajous_payload(
                offset=np.asarray(self.lissajous_offset, dtype=np.double),
                x_amp=self.lissajous_x_amp,
                y_amp=self.lissajous_y_amp,
                z_amp=self.lissajous_z_amp,
                period=self.lissajous_period,
                num_cycles=self.lissajous_num_cycles,
                ramp_time=self.lissajous_ramp_time,
                x_num_periods=self.lissajous_x_num_periods,
                y_num_periods=self.lissajous_y_num_periods,
                z_num_periods=self.lissajous_z_num_periods,
                **common_kwargs,
            )

        self.reference_plan_signals_path = POINT_TO_POINT_SIGNALS_PLOT_PATH
        self.reference_plan_accels_path = POINT_TO_POINT_ACCELS_PLOT_PATH
        return plan_three_quad_point_mass(
            p0=self.x_0[0:3],
            pf=self.planner_goal,
            T_total=self.planner_duration,
            **common_kwargs,
        )

    def save_reference_plan_signals(self):
        smoothness = verify_smoothness(self.reference_plan)
        self.get_logger().info(
            "planner smoothness endpoint_ok="
            + ("true" if bool(smoothness["endpoint_ok"]) else "false")
        )
        if HAS_MPL:
            plot_signal_diagnostics(self.reference_plan)
            if self.planner_type == "lissajous":
                if POINT_TO_POINT_SIGNALS_PLOT_PATH.exists():
                    FilePath(self.reference_plan_signals_path).write_bytes(
                        POINT_TO_POINT_SIGNALS_PLOT_PATH.read_bytes()
                    )
                if POINT_TO_POINT_ACCELS_PLOT_PATH.exists():
                    FilePath(self.reference_plan_accels_path).write_bytes(
                        POINT_TO_POINT_ACCELS_PLOT_PATH.read_bytes()
                    )
            self.get_logger().info(f"saved planner signal diagnostics: {self.reference_plan_signals_path}")
            self.get_logger().info(f"saved planner acceleration diagnostics: {self.reference_plan_accels_path}")
        else:
            self.get_logger().warning("matplotlib not available; skipping planner signal plots.")
        return None

    def log_tracking_sample(self, t_now: float, control_u: np.ndarray):
        quad_vel = np.hstack((self.xq_1[3:6], self.xq_2[3:6], self.xq_3[3:6]))
        quad_vel_des = np.array(
            self.quadrotors_velocity(self.xd[3:6], self.xd[6:15], self.xd[15:24]),
            dtype=np.double,
        ).reshape((self.robot_num * 3,))
        self.tracking_log.append({
            "t": float(t_now),
            "payload_pos": self.x_0[0:3].copy(),
            "payload_vel": self.x_0[3:6].copy(),
            "quad_vel": quad_vel.copy(),
            "cable_dir": self.x_0[6:15].copy(),
            "cable_ang_vel": self.x_0[15:24].copy(),
            "payload_pos_des": self.xd[0:3].copy(),
            "payload_vel_des": self.xd[3:6].copy(),
            "quad_vel_des": quad_vel_des.copy(),
            "cable_dir_des": self.xd[6:15].copy(),
            "cable_ang_vel_des": self.xd[15:24].copy(),
            "control_u": np.asarray(control_u, dtype=np.double).reshape((self.n_u,)).copy(),
        })
        return None

    def save_tracking_results(self):
        if self.results_saved or not self.tracking_log:
            return None

        payload_pos = np.vstack([sample["payload_pos"] for sample in self.tracking_log])
        payload_vel = np.vstack([sample["payload_vel"] for sample in self.tracking_log])
        quad_vel = np.vstack([sample["quad_vel"] for sample in self.tracking_log])
        cable_dir = np.vstack([sample["cable_dir"] for sample in self.tracking_log])
        cable_ang_vel = np.vstack([sample["cable_ang_vel"] for sample in self.tracking_log])
        payload_pos_des = np.vstack([sample["payload_pos_des"] for sample in self.tracking_log])
        payload_vel_des = np.vstack([sample["payload_vel_des"] for sample in self.tracking_log])
        quad_vel_des = np.vstack([sample["quad_vel_des"] for sample in self.tracking_log])
        cable_dir_des = np.vstack([sample["cable_dir_des"] for sample in self.tracking_log])
        cable_ang_vel_des = np.vstack([sample["cable_ang_vel_des"] for sample in self.tracking_log])
        control_u = np.vstack([sample["control_u"] for sample in self.tracking_log])

        payload_position_error = payload_pos - payload_pos_des
        payload_velocity_error = payload_vel - payload_vel_des
        payload_position_error_norm = np.linalg.norm(payload_position_error, axis=1)
        payload_velocity_error_norm = np.linalg.norm(payload_velocity_error, axis=1)
        payload_position_rmse = float(np.sqrt(np.mean(np.sum(payload_position_error**2, axis=1))))
        payload_velocity_rmse = float(np.sqrt(np.mean(np.sum(payload_velocity_error**2, axis=1))))

        data = {
            "t": np.array([sample["t"] for sample in self.tracking_log], dtype=np.double),
            "payload_pos": payload_pos,
            "payload_vel": payload_vel,
            "quad_vel": quad_vel,
            "cable_dir": cable_dir,
            "cable_ang_vel": cable_ang_vel,
            "payload_pos_des": payload_pos_des,
            "payload_vel_des": payload_vel_des,
            "quad_vel_des": quad_vel_des,
            "cable_dir_des": cable_dir_des,
            "cable_ang_vel_des": cable_ang_vel_des,
            "control_u": control_u,
            "payload_position_error_norm": payload_position_error_norm,
            "payload_velocity_error_norm": payload_velocity_error_norm,
            "payload_position_rmse": np.array(payload_position_rmse, dtype=np.double),
            "payload_velocity_rmse": np.array(payload_velocity_rmse, dtype=np.double),
        }
        np.savez(self.tracking_npz_path, **data)

        if HAS_MPL:
            fig, axes = plt.subplots(15, 3, figsize=(16, 44), sharex=True)
            labels = ("x", "y", "z")

            for axis in range(3):
                axes[0, axis].plot(data["t"], data["payload_pos"][:, axis], label="actual")
                axes[0, axis].plot(data["t"], data["payload_pos_des"][:, axis], "--", label="desired")
                axes[0, axis].set_title(f"payload position {labels[axis]}")
                axes[0, axis].grid(True, alpha=0.3)
                if axis == 0:
                    axes[0, axis].legend()

                axes[1, axis].plot(data["t"], data["payload_vel"][:, axis], label="actual")
                axes[1, axis].plot(data["t"], data["payload_vel_des"][:, axis], "--", label="desired")
                axes[1, axis].set_title(f"payload velocity {labels[axis]}")
                axes[1, axis].grid(True, alpha=0.3)

            cable_labels = [("q1", slice(0, 3)), ("q2", slice(3, 6)), ("q3", slice(6, 9))]
            for cable_idx, (name, slc) in enumerate(cable_labels):
                dir_actual = data["cable_dir"][:, slc]
                dir_des = data["cable_dir_des"][:, slc]
                ang_actual = data["cable_ang_vel"][:, slc]
                ang_des = data["cable_ang_vel_des"][:, slc]

                row_dir = 2 + 2 * cable_idx
                row_ang = row_dir + 1

                for axis in range(3):
                    axes[row_dir, axis].plot(data["t"], dir_actual[:, axis], label="actual")
                    axes[row_dir, axis].plot(data["t"], dir_des[:, axis], "--", label="desired")
                    axes[row_dir, axis].set_title(f"{name} direction {labels[axis]}")
                    axes[row_dir, axis].grid(True, alpha=0.3)
                    if axis == 0:
                        axes[row_dir, axis].legend()

                    axes[row_ang, axis].plot(data["t"], ang_actual[:, axis], label="actual")
                    axes[row_ang, axis].plot(data["t"], ang_des[:, axis], "--", label="desired")
                    axes[row_ang, axis].set_title(f"{name} angular velocity {labels[axis]}")
                    axes[row_ang, axis].grid(True, alpha=0.3)
                    if axis == 0:
                        axes[row_ang, axis].legend()

            for axis, (name, slc) in enumerate(cable_labels):
                axes[8, axis].plot(
                    data["t"],
                    np.linalg.norm(data["cable_dir"][:, slc] - data["cable_dir_des"][:, slc], axis=1),
                )
                axes[8, axis].set_title(f"{name} direction error norm")
                axes[8, axis].grid(True, alpha=0.3)

                axes[9, axis].plot(
                    data["t"],
                    np.linalg.norm(data["cable_ang_vel"][:, slc] - data["cable_ang_vel_des"][:, slc], axis=1),
                )
                axes[9, axis].set_title(f"{name} angular-velocity error norm")
                axes[9, axis].grid(True, alpha=0.3)

            velocity_norm_entries = [
                ("payload", data["payload_vel"], data["payload_vel_des"]),
                ("quad 1", data["quad_vel"][:, 0:3], data["quad_vel_des"][:, 0:3]),
                ("quad 2", data["quad_vel"][:, 3:6], data["quad_vel_des"][:, 3:6]),
                ("quad 3", data["quad_vel"][:, 6:9], data["quad_vel_des"][:, 6:9]),
            ]
            velocity_norm_slots = [(10, 0), (10, 1), (10, 2), (11, 0)]
            for (name, actual_vel, desired_vel), (row, col) in zip(velocity_norm_entries, velocity_norm_slots):
                axes[row, col].plot(data["t"], np.linalg.norm(actual_vel, axis=1), label="actual")
                axes[row, col].plot(data["t"], np.linalg.norm(desired_vel, axis=1), "--", label="desired")
                axes[row, col].set_title(f"{name} velocity norm")
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend()
            axes[11, 1].axis("off")
            axes[11, 2].axis("off")

            control_labels = [("u1", slice(0, 3)), ("u2", slice(3, 6)), ("u3", slice(6, 9))]
            for control_idx, (name, slc) in enumerate(control_labels):
                control_values = data["control_u"][:, slc]
                row = 12 + control_idx
                for axis in range(3):
                    axes[row, axis].plot(data["t"], control_values[:, axis])
                    axes[row, axis].set_title(f"{name} acceleration cmd {labels[axis]}")
                    axes[row, axis].grid(True, alpha=0.3)
                    axes[row, axis].set_xlabel("time [s]")

            fig.tight_layout()
            fig.savefig(self.tracking_plot_path, dpi=200)
            plt.close(fig)

            metrics_fig, metrics_axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            metrics_axes[0].plot(data["t"], payload_position_error_norm, label=f"RMSE={payload_position_rmse:.4f} m")
            metrics_axes[0].set_title("Payload Position Error Norm")
            metrics_axes[0].set_ylabel("position error [m]")
            metrics_axes[0].grid(True, alpha=0.3)
            metrics_axes[0].legend()

            metrics_axes[1].plot(data["t"], payload_velocity_error_norm, label=f"RMSE={payload_velocity_rmse:.4f} m/s")
            metrics_axes[1].set_title("Payload Velocity Error Norm")
            metrics_axes[1].set_xlabel("time [s]")
            metrics_axes[1].set_ylabel("velocity error [m/s]")
            metrics_axes[1].grid(True, alpha=0.3)
            metrics_axes[1].legend()

            metrics_fig.tight_layout()
            metrics_fig.savefig(self.tracking_metrics_plot_path, dpi=200)
            plt.close(metrics_fig)

        self.results_saved = True
        self.get_logger().info(f"saved tracking results: {self.tracking_npz_path}")
        if HAS_MPL:
            self.get_logger().info(f"saved tracking plot: {self.tracking_plot_path}")
            self.get_logger().info(f"saved tracking metrics plot: {self.tracking_metrics_plot_path}")
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

        ## Gains Controller 
        self.norm_constraint_slack_weight = 10.0
        self.unit_vector_norm_tol = 1e-3
        
        ## gains for payload
        self.Kp = ca.MX.zeros(3, 3)
        self.Kp[0, 0] = 250.0
        self.Kp[1, 1] = 250.0
        self.Kp[2, 2] = 250.0

        self.Kv = ca.MX.zeros(3, 3)
        self.Kv[0, 0] = 1.0
        self.Kv[1, 1] = 1.0
        self.Kv[2, 2] = 1.0
        
        ## Gains for cable direcitions
        self.Kp_n1 = ca.MX.zeros(3, 3)
        self.Kp_n1[0, 0] = 30
        self.Kp_n1[1, 1] = 30
        self.Kp_n1[2, 2] = 30

        self.Kp_n2 = ca.MX.zeros(3, 3)
        self.Kp_n2[0, 0] = 30
        self.Kp_n2[1, 1] = 30
        self.Kp_n2[2, 2] = 30

        self.Kp_n3 = ca.MX.zeros(3, 3)
        self.Kp_n3[0, 0] = 30
        self.Kp_n3[1, 1] = 30
        self.Kp_n3[2, 2] = 30

        # Gains for cable angular velocity
        self.Kp_r1 = ca.MX.zeros(3, 3)
        self.Kp_r1[0, 0] = 1
        self.Kp_r1[1, 1] = 1
        self.Kp_r1[2, 2] = 1

        self.Kp_r2 = ca.MX.zeros(3, 3)
        self.Kp_r2[0, 0] = 1
        self.Kp_r2[1, 1] = 1
        self.Kp_r2[2, 2] = 1

        self.Kp_r3 = ca.MX.zeros(3, 3)
        self.Kp_r3[0, 0] = 1
        self.Kp_r3[1, 1] = 1
        self.Kp_r3[2, 2] = 1

        self.R_q1 = ca.MX.zeros(3, 3)
        self.R_q1[0, 0] = 0.1
        self.R_q1[1, 1] = 0.1
        self.R_q1[2, 2] = 0.1

        self.R_q2 = ca.MX.zeros(3, 3)
        self.R_q2[0, 0] = 0.1
        self.R_q2[1, 1] = 0.1
        self.R_q2[2, 2] = 0.1

        self.R_q3 = ca.MX.zeros(3, 3)
        self.R_q3[0, 0] = 0.1
        self.R_q3[1, 1] = 0.1
        self.R_q3[2, 2] = 0.1


        lyapunov_position = ((error_position.T @ self.Kp @ error_position) + self.mass * (error_velocity.T @ self.Kv @ error_velocity))

        ocp.model.cost_expr_ext_cost = (
            lyapunov_position
            + (error_n1.T @ self.Kp_n1 @error_n1)
            + (error_n2.T @ self.Kp_n2 @error_n2)
            + (error_n3.T @ self.Kp_n3 @error_n3)
            + (r1_error.T @ self.Kp_r1 @r1_error)
            + (r2_error.T @ self.Kp_r2 @r2_error)
            + (r3_error.T @ self.Kp_r3 @r3_error)
            + (a_q1.T @ self.R_q1 @a_q1)
            + (a_q2.T @ self.R_q2 @a_q2)
            + (a_q3.T @ self.R_q3 @a_q3))
        ocp.model.cost_expr_ext_cost_e = (
            lyapunov_position
            + (error_n1.T @ self.Kp_n1 @error_n1)
            + (error_n2.T @ self.Kp_n2 @error_n2)
            + (error_n3.T @ self.Kp_n3 @error_n3)
            + (r1_error.T @ self.Kp_r1 @r1_error)
            + (r2_error.T @ self.Kp_r2 @r2_error)
            + (r3_error.T @ self.Kp_r3 @r3_error))

        ref_params = np.hstack((self.x_0, self.u_equilibrium))
        cost_params = np.zeros((nx + nx + nu,), dtype=np.double)
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
        ocp.solver_options.levenberg_marquardt = 10.0
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

        position_cmd_msg.tension = tension

        #cable_force = tension*direction

        #position_cmd_msg.cable_force.x = cable_force[0]
        #position_cmd_msg.cable_force.y = cable_force[1]
        #position_cmd_msg.cable_force.z = cable_force[2]

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
            # Init Optimization Problem
            for k in range(5000):
                arr_str = np.array2string(self.x_0, precision=3, separator=", ", suppress_small=True)
                self.get_logger().info(f"state[] = {arr_str}")
    
            self.reference_plan = self.build_reference_plan()
            self.ocp = self.solver(self.x_0)
            self.acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file=str(self.json_file), build=True, generate=True)
            ### Reset Solver
            self.acados_ocp_solver.reset()
            self.update_reference_from_plan(0.0)
            self.save_reference_plan_signals()
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
        current_time = time.monotonic()
        plan_end_time = float(self.reference_plan["t"][-1])
        stop_time = self.reference_start_time + plan_end_time + self.ts
        elapsed_raw = current_time - self.reference_start_time
        elapsed = min(elapsed_raw, plan_end_time)
        self.update_reference_from_plan(elapsed)

        self.acados_ocp_solver.set(0, "lbx", self.x_0)
        self.acados_ocp_solver.set(0, "ubx", self.x_0)

        # Keep the SQP_RTI iterate close to the current measured state.
        #for stage in range(self.N_prediction + 1):
        #    self.acados_ocp_solver.set(stage, "x", self.x_0)
        #for stage in range(self.N_prediction):
        #    self.acados_ocp_solver.set(stage, "u", self.ud)

        # Desired trajectory over the prediction horizon.
        for j in range(self.N_prediction):
            t_stage = min(elapsed + j * self.ts, float(self.reference_plan["t"][-1]))
            yref, uref = self.reference_from_plan(t_stage)
            if j == 0:
                self.xd = yref.copy()
                self.ud = uref.copy()
            aux_ref = np.hstack((yref, uref))
            self.acados_ocp_solver.set(j, "p", aux_ref)
        # Desired trajectory at the terminal stage.
        t_terminal = min(elapsed + self.N_prediction * self.ts, float(self.reference_plan["t"][-1]))
        yref_N, uref_N = self.reference_from_plan(t_terminal)
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
        self.log_tracking_sample(elapsed, u)

        if current_time >= stop_time:
            self.save_tracking_results()
            self.timer.cancel()
            self.get_logger().info("Controller finished planned trajectory; timer cancelled.")
            return None

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
        try:
            payload_node.save_tracking_results()
        except Exception as exc:
            payload_node.get_logger().error(f"Failed to save tracking results on shutdown: {exc}")
        payload_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()
