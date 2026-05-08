#!/usr/bin/env python3
"""
Simple verification script for a 3-quadrotor + point-mass payload planner.

What this script does
---------------------
1. Defines only the initial and final PAYLOAD positions.
2. Builds a 9th-order payload trajectory x_L(t) with zero velocity,
   acceleration, jerk, and snap at both endpoints.
3. Chooses a simple, consistent cable-force sharing law
       lambda_i(t) = lambda_i_eq - alpha_i * m_L * a_L(t)
   where the equilibrium cable forces lambda_i_eq sum to -m_L g e3.
4. Reconstructs, for each cable i:
       T_i, q_i, qdot_i, qddot_i
   from lambda_i, lambda_dot_i, lambda_ddot_i.
5. Reconstructs, for each quadrotor i:
       x_i, v_i, a_i
   using
       x_i = x_L - L_i q_i
       v_i = v_L - L_i qdot_i
       a_i = a_L - L_i qddot_i

This script is intentionally simple and meant only to verify that the
reconstruction chain works and that the quadrotor endpoint velocities and
accelerations are zero when the payload trajectory satisfies zero v/a/j/s at
its endpoints.

Important note
--------------
With only payload endpoints prescribed, the internal force split among the
three cables is NOT unique. This script chooses one simple admissible split.
If you want the full differential-flatness planner later, we can replace this
with explicit flat outputs (x_L, lambda_2, lambda_3, yaw_1, yaw_2, yaw_3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


E3 = np.array([0.0, 0.0, 1.0], dtype=float)
PLOT_PATH = Path(__file__).with_name("trajectories_poly9.png")
SIGNALS_PLOT_PATH = Path(__file__).with_name("trajectory_signals_poly9.png")
ACCELS_PLOT_PATH = Path(__file__).with_name("trajectory_accelerations_poly9.png")


def normalize(v: np.ndarray) -> np.ndarray:
    """Return v / ||v||, raising if v is too small."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def derivative_row(degree: int, deriv: int, tau: float) -> np.ndarray:
    """
    Row vector so that row @ coeffs gives the deriv-th derivative
    with respect to normalized time tau.
    """
    row = np.zeros(degree + 1, dtype=float)
    for k in range(deriv, degree + 1):
        row[k] = factorial(k) / factorial(k - deriv) * tau ** (k - deriv)
    return row


def poly9_coeffs(
    p0: np.ndarray,
    pf: np.ndarray,
    T: float,
    v0: np.ndarray | None = None,
    vf: np.ndarray | None = None,
    a0: np.ndarray | None = None,
    af: np.ndarray | None = None,
    j0: np.ndarray | None = None,
    jf: np.ndarray | None = None,
    s0: np.ndarray | None = None,
    sf: np.ndarray | None = None,
) -> np.ndarray:
    """
    9th-order polynomial in normalized time tau = t/T:
        p(tau) = sum_{k=0}^9 c_k tau^k

    Boundary constraints at tau=0 and tau=1 on:
        p, p_dot, p_ddot, p_dddot, p_ddddot

    Physical-time derivatives are scaled by T^r in the right-hand side.
    """
    p0 = np.asarray(p0, dtype=float)
    pf = np.asarray(pf, dtype=float)

    v0 = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float)
    vf = np.zeros(3) if vf is None else np.asarray(vf, dtype=float)
    a0 = np.zeros(3) if a0 is None else np.asarray(a0, dtype=float)
    af = np.zeros(3) if af is None else np.asarray(af, dtype=float)
    j0 = np.zeros(3) if j0 is None else np.asarray(j0, dtype=float)
    jf = np.zeros(3) if jf is None else np.asarray(jf, dtype=float)
    s0 = np.zeros(3) if s0 is None else np.asarray(s0, dtype=float)
    sf = np.zeros(3) if sf is None else np.asarray(sf, dtype=float)

    degree = 9
    A = np.zeros((10, 10), dtype=float)

    # Start constraints at tau = 0
    for r in range(5):
        A[r, :] = derivative_row(degree, r, 0.0)

    # End constraints at tau = 1
    for r in range(5):
        A[5 + r, :] = derivative_row(degree, r, 1.0)

    b = np.vstack([
        p0,
        T * v0,
        T**2 * a0,
        T**3 * j0,
        T**4 * s0,
        pf,
        T * vf,
        T**2 * af,
        T**3 * jf,
        T**4 * sf,
    ])

    coeffs = np.linalg.solve(A, b)
    return coeffs


@dataclass
class Poly9Eval:
    p: np.ndarray
    v: np.ndarray
    a: np.ndarray
    j: np.ndarray
    s: np.ndarray
    c: np.ndarray  # 5th derivative (crackle)


def eval_poly9(coeffs: np.ndarray, t: float, T: float) -> Poly9Eval:
    """Evaluate 9th-order polynomial and physical-time derivatives up to 5th order."""
    degree = 9
    tau = np.clip(t / T, 0.0, 1.0)

    vals_tau = []
    for r in range(6):
        row = derivative_row(degree, r, tau)
        vals_tau.append(row @ coeffs)

    return Poly9Eval(
        p=vals_tau[0],
        v=vals_tau[1] / T,
        a=vals_tau[2] / T**2,
        j=vals_tau[3] / T**3,
        s=vals_tau[4] / T**4,
        c=vals_tau[5] / T**5,
    )


@dataclass
class CableState:
    T: float
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray


def reconstruct_cable(
    lam: np.ndarray,
    lam_dot: np.ndarray,
    lam_ddot: np.ndarray,
    eps: float = 1e-6,
) -> CableState:
    """
    Reconstruct T, q, qdot, qddot from
        lambda = T q
    with q a unit vector.
    """
    T = np.linalg.norm(lam)
    if T < eps:
        raise ValueError("Cable force magnitude is too small; q is not well-defined.")

    q = lam / T

    # Tdot = q^T lambda_dot
    Tdot = float(q @ lam_dot)

    # qdot = (lambda_dot - Tdot q) / T
    qdot = (lam_dot - Tdot * q) / T

    # Tddot = q^T lambda_ddot + qdot^T lambda_dot
    Tddot = float(q @ lam_ddot + qdot @ lam_dot)

    # qddot = (lambda_ddot - Tddot q - 2 Tdot qdot) / T
    qddot = (lam_ddot - Tddot * q - 2.0 * Tdot * qdot) / T

    return CableState(T=T, q=q, qdot=qdot, qddot=qddot)


def build_hover_equilibrium_lambdas(
    payload_mass: float,
    gravity: float,
    q_eq_list: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Given 3 desired equilibrium cable directions q_i, solve for positive tensions
    T_i such that
        sum_i T_i q_i = -m g e3.

    Returns lambda_i_eq = T_i q_i.
    """
    if len(q_eq_list) != 3:
        raise ValueError("This simple script expects exactly 3 cables / 3 quadrotors.")

    Q = np.column_stack(q_eq_list)  # 3x3
    rhs = -payload_mass * gravity * E3
    tensions = np.linalg.solve(Q, rhs)

    if np.any(tensions <= 0.0):
        raise ValueError(
            f"Equilibrium tensions are not all positive: {tensions}. "
            "Choose different equilibrium cable directions."
        )

    lambdas_eq = [tensions[i] * q_eq_list[i] for i in range(3)]
    return lambdas_eq


def plan_three_quad_point_mass(
    p0: np.ndarray,
    pf: np.ndarray,
    T_total: float = 2.0,
    n_samples: int = 201,
    payload_mass: float = 1.0,
    gravity: float = 9.81,
    cable_lengths: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Minimal point-to-point planner for a point-mass payload carried by 3 quads.

    Inputs
    ------
    p0, pf:
        Initial and final payload positions.
    T_total:
        Trajectory duration.
    n_samples:
        Number of sampled points for verification / plotting.
    payload_mass:
        Payload mass.
    gravity:
        Gravity constant.
    cable_lengths:
        Length-3 array of cable lengths.
    alpha:
        Length-3 load-sharing weights, summing to 1. These weights determine how
        the payload acceleration term is distributed into the 3 cable-force vectors:
            lambda_i(t) = lambda_i_eq - alpha_i * m_L * a_L(t)

    Returns
    -------
    A dictionary containing payload and quad trajectories sampled over time.
    """
    p0 = np.asarray(p0, dtype=float)
    pf = np.asarray(pf, dtype=float)

    if cable_lengths is None:
        cable_lengths = np.array([1.0, 1.0, 1.0], dtype=float)
    else:
        cable_lengths = np.asarray(cable_lengths, dtype=float)

    if alpha is None:
        alpha = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        alpha = np.asarray(alpha, dtype=float)

    if alpha.shape != (3,) or not np.isclose(alpha.sum(), 1.0):
        raise ValueError("alpha must be length-3 and sum to 1.")

    # ------------------------------------------------------------------
    # 1) Choose a non-degenerate equilibrium cable geometry.
    #    These q_i point from each quadrotor to the payload.
    #    Since the quads are above the payload, q_i have negative z components.
    # ------------------------------------------------------------------
    quad_1_init = np.array([-0.0029774502873919804, -0.30020808379855246, 1.4896822896707809])
    quad_2_init = np.array([-0.003912773485618989, 0.29969367235921324, 1.4896964934919243])
    quad_3_init = np.array([0.8063607699386893, -0.00040573095469479846, 1.4825609159860986])
    payload_init = np.array([0.34049933598831467, -0.0007520805616463245, 0.8936953489145677])

    q1_eq = normalize(payload_init-quad_1_init)
    q2_eq = normalize(payload_init-quad_2_init)
    q3_eq = normalize(payload_init-quad_3_init)
    q_eq_list = [q1_eq, q2_eq, q3_eq]

    # Build nonzero equilibrium lambda_i that balance gravity at hover.
    lambdas_eq = build_hover_equilibrium_lambdas(
        payload_mass=payload_mass,
        gravity=gravity,
        q_eq_list=q_eq_list,
    )

    # ------------------------------------------------------------------
    # 2) Payload 9th-order trajectory: x_L(t)
    #    Zero v, a, j, s at both endpoints.
    # ------------------------------------------------------------------
    coeffs_payload = poly9_coeffs(
        p0,
        pf,
        T_total,
        v0=np.zeros(3),
        vf=np.zeros(3),
        a0=np.zeros(3),
        af=np.zeros(3),
        j0=np.zeros(3),
        jf=np.zeros(3),
        s0=np.zeros(3),
        sf=np.zeros(3),
    )

    times = np.linspace(0.0, T_total, n_samples)

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

    # ------------------------------------------------------------------
    # 3) Sample the trajectory and reconstruct cable / quad states.
    # ------------------------------------------------------------------
    for k, t in enumerate(times):
        out = eval_poly9(coeffs_payload, t, T_total)
        payload_p[k] = out.p
        payload_v[k] = out.v
        payload_a[k] = out.a
        payload_j[k] = out.j
        payload_s[k] = out.s
        payload_c[k] = out.c

        # Simple admissible load-sharing assumption:
        #   lambda_i(t) = lambda_i_eq - alpha_i * m_L * a_L(t)
        # so that sum_i lambda_i = -m_L(g e3 + a_L).
        for i in range(3):
            lambdas[i, k] = lambdas_eq[i] - alpha[i] * payload_mass * out.a
            lambdas_dot[i, k] = -alpha[i] * payload_mass * out.j
            lambdas_ddot[i, k] = -alpha[i] * payload_mass * out.s

            cable = reconstruct_cable(
                lambdas[i, k],
                lambdas_dot[i, k],
                lambdas_ddot[i, k],
            )
            tensions[i, k] = cable.T
            q[i, k] = cable.q
            qdot[i, k] = cable.qdot
            qddot[i, k] = cable.qddot

            # Quadrotor kinematics from payload + cable geometry.
            Li = cable_lengths[i]
            quad_p[i, k] = out.p - Li * cable.q
            quad_v[i, k] = out.v - Li * cable.qdot
            quad_a[i, k] = out.a - Li * cable.qddot

    return {
        "t": times,
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
    }


def _finite_difference_norm(signal: np.ndarray, dt: float) -> np.ndarray:
    """Return the sample-wise norm of the first finite difference."""
    diffs = np.diff(signal, axis=0) / dt
    return np.linalg.norm(diffs, axis=-1)


def verify_smoothness(data: Dict[str, np.ndarray], tol: float = 1e-6) -> Dict[str, float]:
    """
    Compute lightweight smoothness diagnostics for the sampled planner outputs.

    These checks do not prove global optimality; they verify that the generated
    signals satisfy expected endpoint conditions and evolve without discrete
    jumps in the sampled reconstruction.
    """
    times = data["t"]
    dt = float(times[1] - times[0])

    payload_v0 = float(np.linalg.norm(data["payload_v"][0]))
    payload_vf = float(np.linalg.norm(data["payload_v"][-1]))
    payload_a0 = float(np.linalg.norm(data["payload_a"][0]))
    payload_af = float(np.linalg.norm(data["payload_a"][-1]))
    payload_j0 = float(np.linalg.norm(data["payload_j"][0]))
    payload_jf = float(np.linalg.norm(data["payload_j"][-1]))
    payload_s0 = float(np.linalg.norm(data["payload_s"][0]))
    payload_sf = float(np.linalg.norm(data["payload_s"][-1]))

    q_norm_error = float(np.max(np.abs(np.linalg.norm(data["q"], axis=2) - 1.0)))
    q_orth_error = float(np.max(np.abs(np.sum(data["q"] * data["qdot"], axis=2))))

    quad_v0 = [float(np.linalg.norm(data["quad_v"][i, 0])) for i in range(3)]
    quad_vf = [float(np.linalg.norm(data["quad_v"][i, -1])) for i in range(3)]
    quad_a0 = [float(np.linalg.norm(data["quad_a"][i, 0])) for i in range(3)]
    quad_af = [float(np.linalg.norm(data["quad_a"][i, -1])) for i in range(3)]

    payload_jerk = np.linalg.norm(data["payload_j"], axis=1)
    payload_snap = np.linalg.norm(data["payload_s"], axis=1)

    max_tension_rate = 0.0
    max_q_rate = 0.0
    max_qacc_rate = 0.0
    max_quad_acc_rate = 0.0
    max_quad_jerk = 0.0

    for i in range(3):
        max_tension_rate = max(
            max_tension_rate,
            float(np.max(np.abs(np.diff(data["tension"][i]) / dt))),
        )
        max_q_rate = max(
            max_q_rate,
            float(np.max(_finite_difference_norm(data["q"][i], dt))),
        )
        max_qacc_rate = max(
            max_qacc_rate,
            float(np.max(_finite_difference_norm(data["qddot"][i], dt))),
        )
        quad_acc_rate = _finite_difference_norm(data["quad_a"][i], dt)
        max_quad_acc_rate = max(max_quad_acc_rate, float(np.max(quad_acc_rate)))
        quad_jerk = np.gradient(data["quad_a"][i], dt, axis=0)
        max_quad_jerk = max(
            max_quad_jerk,
            float(np.max(np.linalg.norm(quad_jerk, axis=1))),
        )

    endpoint_ok = (
        payload_v0 < tol and payload_vf < tol and
        payload_a0 < tol and payload_af < tol and
        payload_j0 < tol and payload_jf < tol and
        payload_s0 < tol and payload_sf < tol and
        max(quad_v0) < 1e-4 and max(quad_vf) < 1e-4 and
        max(quad_a0) < 1e-4 and max(quad_af) < 1e-4
    )

    return {
        "payload_v0_norm": payload_v0,
        "payload_vf_norm": payload_vf,
        "payload_a0_norm": payload_a0,
        "payload_af_norm": payload_af,
        "payload_j0_norm": payload_j0,
        "payload_jf_norm": payload_jf,
        "payload_s0_norm": payload_s0,
        "payload_sf_norm": payload_sf,
        "quad_v0_max_norm": float(max(quad_v0)),
        "quad_vf_max_norm": float(max(quad_vf)),
        "quad_a0_max_norm": float(max(quad_a0)),
        "quad_af_max_norm": float(max(quad_af)),
        "endpoint_ok": float(endpoint_ok),
        "max_q_norm_error": q_norm_error,
        "max_q_qdot_orthogonality_error": q_orth_error,
        "max_payload_jerk_norm": float(np.max(payload_jerk)),
        "max_payload_snap_norm": float(np.max(payload_snap)),
        "max_tension_rate": max_tension_rate,
        "max_q_rate": max_q_rate,
        "max_qddot_rate": max_qacc_rate,
        "max_quad_acc_rate": max_quad_acc_rate,
        "max_quad_jerk_norm": max_quad_jerk,
    }


def plot_signal_diagnostics(data: Dict[str, np.ndarray]) -> None:
    """Save time-series plots for payload, cable, and quadrotor signals."""
    if not HAS_MPL:
        return

    t = data["t"]
    fig, axes = plt.subplots(7, 3, figsize=(15, 23), sharex=True)

    payload_series = [
        ("payload position [m]", data["payload_p"]),
        ("payload velocity [m/s]", data["payload_v"]),
    ]
    for row, (title, series) in enumerate(payload_series):
        for axis in range(3):
            ax = axes[row, axis]
            ax.plot(t, series[:, axis], linewidth=2)
            ax.set_title(f"{title} {'xyz'[axis]}")
            ax.grid(True, alpha=0.3)

    for axis in range(3):
        ax = axes[2, axis]
        for i in range(3):
            ax.plot(t, data["q"][i, :, axis], label=f"q{i+1}", linewidth=1.8)
        ax.set_title(f"cable direction {'xyz'[axis]}")
        ax.grid(True, alpha=0.3)
        if axis == 0:
            ax.legend()

    for axis in range(3):
        ax = axes[3, axis]
        for i in range(3):
            ax.plot(t, data["qdot"][i, :, axis], label=f"qdot{i+1}", linewidth=1.8)
        ax.set_title(f"cable direction rate {'xyz'[axis]}")
        ax.grid(True, alpha=0.3)
        if axis == 0:
            ax.legend()

    quad_series = [
        ("quad position [m]", data["quad_p"]),
        ("quad velocity [m/s]", data["quad_v"]),
    ]
    for row_offset, (title, series) in enumerate(quad_series, start=4):
        for axis in range(3):
            ax = axes[row_offset, axis]
            for i in range(3):
                ax.plot(t, series[i, :, axis], label=f"quad{i+1}", linewidth=1.8)
            ax.set_title(f"{title} {'xyz'[axis]}")
            ax.grid(True, alpha=0.3)
            if axis == 0:
                ax.legend()

    tension_ax = axes[6, 0]
    for i in range(3):
        tension_ax.plot(t, data["tension"][i], label=f"T{i+1}", linewidth=1.8)
    tension_ax.set_title("cable tension [N]")
    tension_ax.grid(True, alpha=0.3)
    tension_ax.legend()
    tension_ax.set_xlabel("time [s]")

    axes[6, 1].axis("off")
    axes[6, 2].axis("off")

    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel("time [s]")

    fig.suptitle("Planner Signal Diagnostics (poly9 payload)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(SIGNALS_PLOT_PATH, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    signal_groups = [
        ("payload acceleration [m/s^2]", data["payload_a"][None, :, :], "payload"),
        ("quad acceleration [m/s^2]", data["quad_a"], "quad"),
        ("cable acceleration qddot [1/s^2]", data["qddot"], "qddot"),
    ]
    for row, (title, series, prefix) in enumerate(signal_groups):
        for axis in range(3):
            ax = axes[row, axis]
            for i in range(series.shape[0]):
                label = prefix if series.shape[0] == 1 else f"{prefix}{i+1}"
                ax.plot(t, series[i, :, axis], label=label, linewidth=1.8)
            ax.set_title(f"{title} {'xyz'[axis]}")
            ax.grid(True, alpha=0.3)
            if axis == 0:
                ax.legend()
            if row == 2:
                ax.set_xlabel("time [s]")

    fig.suptitle("Planner Acceleration Diagnostics (poly9 payload)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(ACCELS_PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    # ------------------------------------------------------------------
    # Define only the payload endpoints here.
    # ------------------------------------------------------------------
    p0 = np.array([0.34049933598831467, -0.0007520805616463245, 0.8936953489145677])
    pf = np.array([2.0, -0.0007520805616463245, 0.8936953489145677], dtype=float)

    payload_mass = 0.2
    gravity = 9.81

    data = plan_three_quad_point_mass(
        p0=p0,
        pf=pf,
        T_total=5.0,
        n_samples=201,
        payload_mass=payload_mass,
        gravity=gravity,
        cable_lengths=np.array([0.75, 0.75, 0.75], dtype=float),
        alpha=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float),
    )

    print("\n=== Simple verification summary (poly9 payload) ===")
    print(f"payload start: {data['payload_p'][0]}")
    print(f"payload goal : {data['payload_p'][-1]}")
    print()

    for i in range(3):
        print(f"quad {i+1} start position: {data['quad_p'][i, 0]}")
        print(f"quad {i+1} final position: {data['quad_p'][i, -1]}")
        print(f"quad {i+1} start velocity norm: {np.linalg.norm(data['quad_v'][i, 0]):.6e}")
        print(f"quad {i+1} final velocity norm: {np.linalg.norm(data['quad_v'][i, -1]):.6e}")
        print(f"quad {i+1} start accel norm   : {np.linalg.norm(data['quad_a'][i, 0]):.6e}")
        print(f"quad {i+1} final accel norm   : {np.linalg.norm(data['quad_a'][i, -1]):.6e}")
        print(f"quad {i+1} min tension        : {data['tension'][i].min():.6f}")
        print(f"quad {i+1} max tension        : {data['tension'][i].max():.6f}")
        print()

    # Quick consistency check:
    # lambda1 + lambda2 + lambda3 should equal -m (a_L + g e3)
    lhs = data["lambda"][0] + data["lambda"][1] + data["lambda"][2]
    rhs = -payload_mass * (data["payload_a"] + gravity * E3[None, :])
    err = np.max(np.linalg.norm(lhs - rhs, axis=1))
    print(f"max force-balance reconstruction error: {err:.3e}")

    smoothness = verify_smoothness(data)
    print("\n=== Smoothness diagnostics ===")
    print(f"payload start velocity norm      : {smoothness['payload_v0_norm']:.3e}")
    print(f"payload final velocity norm      : {smoothness['payload_vf_norm']:.3e}")
    print(f"payload start acceleration norm  : {smoothness['payload_a0_norm']:.3e}")
    print(f"payload final acceleration norm  : {smoothness['payload_af_norm']:.3e}")
    print(f"payload start jerk norm          : {smoothness['payload_j0_norm']:.3e}")
    print(f"payload final jerk norm          : {smoothness['payload_jf_norm']:.3e}")
    print(f"payload start snap norm          : {smoothness['payload_s0_norm']:.3e}")
    print(f"payload final snap norm          : {smoothness['payload_sf_norm']:.3e}")
    print(f"max quad start velocity norm     : {smoothness['quad_v0_max_norm']:.3e}")
    print(f"max quad final velocity norm     : {smoothness['quad_vf_max_norm']:.3e}")
    print(f"max quad start accel norm        : {smoothness['quad_a0_max_norm']:.3e}")
    print(f"max quad final accel norm        : {smoothness['quad_af_max_norm']:.3e}")
    print(f"max cable unit-norm error        : {smoothness['max_q_norm_error']:.3e}")
    print(f"max q·qdot orthogonality error   : {smoothness['max_q_qdot_orthogonality_error']:.3e}")
    print(f"max payload jerk norm            : {smoothness['max_payload_jerk_norm']:.3e}")
    print(f"max payload snap norm            : {smoothness['max_payload_snap_norm']:.3e}")
    print(f"max tension rate                 : {smoothness['max_tension_rate']:.3e}")
    print(f"max cable direction rate         : {smoothness['max_q_rate']:.3e}")
    print(f"max cable angular-accel rate     : {smoothness['max_qddot_rate']:.3e}")
    print(f"max quad acceleration rate       : {smoothness['max_quad_acc_rate']:.3e}")
    print(f"max quad jerk norm               : {smoothness['max_quad_jerk_norm']:.3e}")
    print(
        "smoothness verdict               : "
        + ("PASS" if smoothness["endpoint_ok"] else "CHECK")
    )

    if HAS_MPL:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(data["payload_p"][:, 0], data["payload_p"][:, 1], data["payload_p"][:, 2],
                label="payload", linewidth=2)
        for i in range(3):
            ax.plot(data["quad_p"][i, :, 0], data["quad_p"][i, :, 1], data["quad_p"][i, :, 2],
                    label=f"quad {i+1}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("3 quads + point-mass payload (poly9 verification)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(PLOT_PATH, dpi=200)
        print(f"saved trajectory plot            : {PLOT_PATH}")
        plot_signal_diagnostics(data)
        print(f"saved signal diagnostics plot    : {SIGNALS_PLOT_PATH}")
        print(f"saved acceleration diagnostics   : {ACCELS_PLOT_PATH}")
        plt.show()
    else:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
