#!/usr/bin/env python3
"""
Simple verification planner for a 3-quadrotor + point-mass payload following a
ramped Lissajous payload trajectory.

This mirrors the main ideas used in the ROS2 Lissajous payload generator:
- a scalar path-progress variable with acceleration / constant-speed /
  deceleration phases,
- payload position as a Lissajous curve of that scalar progress,
- cable-force sharing reconstructed into cable directions and quad states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from payload_pointmass_multiple_quadrotor.lim_min_multiple_simple import (
    E3,
    build_hover_equilibrium_lambdas,
    normalize,
    plot_signal_diagnostics,
    reconstruct_cable,
    verify_smoothness,
)


PLOT_PATH = Path(__file__).with_name("trajectories_lissajous_payload.png")
SIGNALS_PLOT_PATH = Path(__file__).with_name("trajectory_signals_lissajous_payload.png")
ACCELS_PLOT_PATH = Path(__file__).with_name("trajectory_accelerations_lissajous_payload.png")


def _progress_profile(
    t: float,
    period: float,
    num_cycles: float,
    ramp_time: float,
) -> tuple[float, float, float, float, float, float]:
    """Return (s, sdot, sddot, sdddot, sddddot, total_time)."""
    total_s = period * num_cycles
    tr = float(ramp_time)
    if period <= 0.0 or num_cycles <= 0.0 or tr <= 0.0:
        raise ValueError("period, num_cycles, and ramp_time must be positive.")

    a4 = 35.0 / tr**4
    a5 = -84.0 / tr**5
    a6 = 70.0 / tr**6
    a7 = -20.0 / tr**7

    ramp_s = a7 * tr**8 / 8.0 + a6 * tr**7 / 7.0 + a5 * tr**6 / 6.0 + a4 * tr**5 / 5.0
    const_time = total_s - 2.0 * ramp_s
    total_time = 2.0 * tr + const_time
    if const_time < 0.0:
        raise ValueError("ramp_time is too large for the chosen period*num_cycles.")

    tc = float(np.clip(t, 0.0, total_time))
    if tc < tr:
        s = a7 * tc**8 / 8.0 + a6 * tc**7 / 7.0 + a5 * tc**6 / 6.0 + a4 * tc**5 / 5.0
        sdot = a7 * tc**7 + a6 * tc**6 + a5 * tc**5 + a4 * tc**4
        sddot = 7.0 * a7 * tc**6 + 6.0 * a6 * tc**5 + 5.0 * a5 * tc**4 + 4.0 * a4 * tc**3
        sdddot = 42.0 * a7 * tc**5 + 30.0 * a6 * tc**4 + 20.0 * a5 * tc**3 + 12.0 * a4 * tc**2
        sddddot = 210.0 * a7 * tc**4 + 120.0 * a6 * tc**3 + 60.0 * a5 * tc**2 + 24.0 * a4 * tc
    elif tc < total_time - tr:
        s = ramp_s + tc - tr
        sdot = 1.0
        sddot = 0.0
        sdddot = 0.0
        sddddot = 0.0
    else:
        te = total_time - tc
        s = 2.0 * ramp_s + const_time - a7 * te**8 / 8.0 - a6 * te**7 / 7.0 - a5 * te**6 / 6.0 - a4 * te**5 / 5.0
        sdot = a7 * te**7 + a6 * te**6 + a5 * te**5 + a4 * te**4
        sddot = -7.0 * a7 * te**6 - 6.0 * a6 * te**5 - 5.0 * a5 * te**4 - 4.0 * a4 * te**3
        sdddot = 42.0 * a7 * te**5 + 30.0 * a6 * te**4 + 20.0 * a5 * te**3 + 12.0 * a4 * te**2
        sddddot = -210.0 * a7 * te**4 - 120.0 * a6 * te**3 - 60.0 * a5 * te**2 - 24.0 * a4 * te

    return s, sdot, sddot, sdddot, sddddot, total_time


def _lissajous_payload_state(
    s: float,
    sdot: float,
    sddot: float,
    sdddot: float,
    sddddot: float,
    offset: np.ndarray,
    x_amp: float,
    y_amp: float,
    z_amp: float,
    x_num_periods: float,
    y_num_periods: float,
    z_num_periods: float,
    period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return payload (p, v, a, j, snap) following the C++ generator formulas."""
    x_coeff = 2.0 * np.pi * x_num_periods / period
    y_coeff = 2.0 * np.pi * y_num_periods / period
    z_coeff = 2.0 * np.pi * z_num_periods / period

    x_coeff2 = x_coeff * x_coeff
    x_coeff3 = x_coeff2 * x_coeff
    x_coeff4 = x_coeff3 * x_coeff
    y_coeff2 = y_coeff * y_coeff
    y_coeff3 = y_coeff2 * y_coeff
    y_coeff4 = y_coeff3 * y_coeff
    z_coeff2 = z_coeff * z_coeff
    z_coeff3 = z_coeff2 * z_coeff
    z_coeff4 = z_coeff3 * z_coeff

    p = np.zeros(3, dtype=float)
    v = np.zeros(3, dtype=float)
    a = np.zeros(3, dtype=float)
    j = np.zeros(3, dtype=float)
    snap = np.zeros(3, dtype=float)

    p[0] = x_amp * (1.0 - np.cos(x_coeff * s))
    p[1] = y_amp * np.sin(y_coeff * s)
    p[2] = z_amp * np.sin(z_coeff * s)

    v[0] = x_amp * x_coeff * np.sin(x_coeff * s) * sdot
    v[1] = y_amp * y_coeff * np.cos(y_coeff * s) * sdot
    v[2] = z_amp * z_coeff * np.cos(z_coeff * s) * sdot

    a[0] = x_amp * (x_coeff2 * np.cos(x_coeff * s) * sdot * sdot + x_coeff * np.sin(x_coeff * s) * sddot)
    a[1] = y_amp * (-y_coeff2 * np.sin(y_coeff * s) * sdot * sdot + y_coeff * np.cos(y_coeff * s) * sddot)
    a[2] = z_amp * (-z_coeff2 * np.sin(z_coeff * s) * sdot * sdot + z_coeff * np.cos(z_coeff * s) * sddot)

    j[0] = x_amp * (-x_coeff3 * np.sin(x_coeff * s) * sdot**3 + 3.0 * x_coeff2 * np.cos(x_coeff * s) * sdot * sddot + x_coeff * np.sin(x_coeff * s) * sdddot)
    j[1] = y_amp * (-y_coeff3 * np.cos(y_coeff * s) * sdot**3 - 3.0 * y_coeff2 * np.sin(y_coeff * s) * sdot * sddot + y_coeff * np.cos(y_coeff * s) * sdddot)
    j[2] = z_amp * (-z_coeff3 * np.cos(z_coeff * s) * sdot**3 - 3.0 * z_coeff2 * np.sin(z_coeff * s) * sdot * sddot + z_coeff * np.cos(z_coeff * s) * sdddot)

    snap[0] = x_amp * (
        -x_coeff4 * np.cos(x_coeff * s) * sdot**4
        + 3.0 * x_coeff2 * np.cos(x_coeff * s) * sddot**2
        + x_coeff * np.sin(x_coeff * s) * sddddot
        - 6.0 * x_coeff3 * np.sin(x_coeff * s) * sddot * sdot**2
        + 4.0 * x_coeff2 * np.cos(x_coeff * s) * sdddot * sdot
    )
    snap[1] = y_amp * (
        y_coeff4 * np.sin(y_coeff * s) * sdot**4
        - 3.0 * y_coeff2 * np.sin(y_coeff * s) * sddot**2
        + y_coeff * np.cos(y_coeff * s) * sddddot
        - 6.0 * y_coeff3 * np.cos(y_coeff * s) * sddot * sdot**2
        - 4.0 * y_coeff2 * np.sin(y_coeff * s) * sdddot * sdot
    )
    snap[2] = z_amp * (
        z_coeff4 * np.sin(z_coeff * s) * sdot**4
        - 3.0 * z_coeff2 * np.sin(z_coeff * s) * sddot**2
        + z_coeff * np.cos(z_coeff * s) * sddddot
        - 6.0 * z_coeff3 * np.cos(z_coeff * s) * sddot * sdot**2
        - 4.0 * z_coeff2 * np.sin(z_coeff * s) * sdddot * sdot
    )

    p += offset
    return p, v, a, j, snap


def plan_three_quad_lissajous_payload(
    offset: np.ndarray,
    x_amp: float,
    y_amp: float,
    z_amp: float,
    period: float,
    num_cycles: float,
    ramp_time: float,
    x_num_periods: float = 1.0,
    y_num_periods: float = 1.0,
    z_num_periods: float = 1.0,
    n_samples: int = 401,
    payload_mass: float = 1.0,
    gravity: float = 9.81,
    cable_lengths: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Build a ramped Lissajous payload trajectory and reconstruct cable/quad states.

    The output structure mirrors `plan_three_quad_point_mass()` so it can be used
    by the same downstream Python tooling.
    """
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,):
        raise ValueError("offset must be length-3.")

    if cable_lengths is None:
        cable_lengths = np.array([0.76, 0.76, 0.76], dtype=float)
    else:
        cable_lengths = np.asarray(cable_lengths, dtype=float)

    if alpha is None:
        alpha = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        alpha = np.asarray(alpha, dtype=float)

    if alpha.shape != (3,) or not np.isclose(alpha.sum(), 1.0):
        raise ValueError("alpha must be length-3 and sum to 1.")

    quad_1_init = np.array([-0.0029774502873919804, -0.30020808379855246, 1.4896822896707809])
    quad_2_init = np.array([-0.003912773485618989, 0.29969367235921324, 1.4896964934919243])
    quad_3_init = np.array([0.8063607699386893, -0.00040573095469479846, 1.4825609159860986])
    payload_init = np.array([0.34049933598831467, -0.0007520805616463245, 0.8936953489145677])

    q1_eq = normalize(payload_init - quad_1_init)
    q2_eq = normalize(payload_init - quad_2_init)
    q3_eq = normalize(payload_init - quad_3_init)
    q_eq_list = [q1_eq, q2_eq, q3_eq]
    lambdas_eq = build_hover_equilibrium_lambdas(
        payload_mass=payload_mass,
        gravity=gravity,
        q_eq_list=q_eq_list,
    )

    _, _, _, _, _, total_time = _progress_profile(0.0, period, num_cycles, ramp_time)
    times = np.linspace(0.0, total_time, n_samples)

    payload_p = np.zeros((n_samples, 3))
    payload_v = np.zeros((n_samples, 3))
    payload_a = np.zeros((n_samples, 3))
    payload_j = np.zeros((n_samples, 3))
    payload_s = np.zeros((n_samples, 3))
    payload_c = np.zeros((n_samples, 3))

    lambdas = np.zeros((3, n_samples, 3))
    lambdas_dot = np.zeros((3, n_samples, 3))
    lambdas_ddot = np.zeros((3, n_samples, 3))
    tensions = np.zeros((3, n_samples))
    q = np.zeros((3, n_samples, 3))
    qdot = np.zeros((3, n_samples, 3))
    qddot = np.zeros((3, n_samples, 3))
    quad_p = np.zeros((3, n_samples, 3))
    quad_v = np.zeros((3, n_samples, 3))
    quad_a = np.zeros((3, n_samples, 3))

    for k, t in enumerate(times):
        s, sdot, sddot, sdddot, sddddot, _ = _progress_profile(t, period, num_cycles, ramp_time)
        p, v, a, j, snap = _lissajous_payload_state(
            s,
            sdot,
            sddot,
            sdddot,
            sddddot,
            offset,
            x_amp,
            y_amp,
            z_amp,
            x_num_periods,
            y_num_periods,
            z_num_periods,
            period,
        )

        payload_p[k] = p
        payload_v[k] = v
        payload_a[k] = a
        payload_j[k] = j
        payload_s[k] = snap

        for i in range(3):
            lambdas[i, k] = lambdas_eq[i] - alpha[i] * payload_mass * a
            lambdas_dot[i, k] = -alpha[i] * payload_mass * j
            lambdas_ddot[i, k] = -alpha[i] * payload_mass * snap

            cable = reconstruct_cable(lambdas[i, k], lambdas_dot[i, k], lambdas_ddot[i, k])
            tensions[i, k] = cable.T
            q[i, k] = cable.q
            qdot[i, k] = cable.qdot
            qddot[i, k] = cable.qddot

            Li = cable_lengths[i]
            quad_p[i, k] = p - Li * cable.q
            quad_v[i, k] = v - Li * cable.qdot
            quad_a[i, k] = a - Li * cable.qddot

    if n_samples > 1:
        dt = float(times[1] - times[0])
        payload_c[1:-1] = np.gradient(payload_s, dt, axis=0)[1:-1]
        payload_c[0] = payload_c[1]
        payload_c[-1] = payload_c[-2]

    return {
        "t": times,
        "total_time": total_time,
        "payload_mass": payload_mass,
        "gravity": gravity,
        "payload_p": payload_p,
        "payload_v": payload_v,
        "payload_a": payload_a,
        "payload_j": payload_j,
        "payload_s": payload_s,
        "payload_c": payload_c,
        "lambda": lambdas,
        "lambda_dot": lambdas_dot,
        "lambda_ddot": lambdas_ddot,
        "tension": tensions,
        "q": q,
        "qdot": qdot,
        "qddot": qddot,
        "quad_p": quad_p,
        "quad_v": quad_v,
        "quad_a": quad_a,
        "q_eq": np.array(q_eq_list),
        "lambda_eq": np.array(lambdas_eq),
        "offset": offset,
        "period": float(period),
        "num_cycles": float(num_cycles),
        "ramp_time": float(ramp_time),
        "x_amp": float(x_amp),
        "y_amp": float(y_amp),
        "z_amp": float(z_amp),
        "x_num_periods": float(x_num_periods),
        "y_num_periods": float(y_num_periods),
        "z_num_periods": float(z_num_periods),
    }


def main() -> None:
    data = plan_three_quad_lissajous_payload(
        offset=np.array([0.34049933598831467, -0.0007520805616463245, 0.8936953489145677]),
        x_amp=0.5,
        y_amp=0.5,
        z_amp=0.2,
        period=4.0,
        num_cycles=1.0,
        ramp_time=1.0,
        x_num_periods=1.0,
        y_num_periods=2.0,
        z_num_periods=1.0,
        n_samples=401,
        payload_mass=0.2,
        gravity=9.81,
        cable_lengths=np.array([0.76, 0.76, 0.76], dtype=float),
    )

    smoothness = verify_smoothness(data)
    print("\n=== Lissajous payload verification summary ===")
    print(f"payload start: {data['payload_p'][0]}")
    print(f"payload end  : {data['payload_p'][-1]}")
    print(f"total time   : {data['total_time']:.3f} s")
    print(f"smoothness endpoint_ok: {bool(smoothness['endpoint_ok'])}")

    if HAS_MPL:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(data["payload_p"][:, 0], data["payload_p"][:, 1], data["payload_p"][:, 2], label="payload", linewidth=2)
        for i in range(3):
            ax.plot(data["quad_p"][i, :, 0], data["quad_p"][i, :, 1], data["quad_p"][i, :, 2], label=f"quad {i+1}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("3 quads + payload (ramped Lissajous verification)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=200)
        plot_signal_diagnostics(data)
        signal_target = Path(SIGNALS_PLOT_PATH)
        accel_target = Path(ACCELS_PLOT_PATH)
        # Re-save using lissajous-specific names if the generic helper created files.
        default_signal = Path(__file__).with_name("trajectory_signals_poly9.png")
        default_accel = Path(__file__).with_name("trajectory_accelerations_poly9.png")
        if default_signal.exists():
            signal_target.write_bytes(default_signal.read_bytes())
        if default_accel.exists():
            accel_target.write_bytes(default_accel.read_bytes())
        print(f"saved trajectory plot            : {PLOT_PATH}")
        print(f"saved signal diagnostics plot    : {SIGNALS_PLOT_PATH}")
        print(f"saved acceleration diagnostics   : {ACCELS_PLOT_PATH}")


if __name__ == "__main__":
    main()
