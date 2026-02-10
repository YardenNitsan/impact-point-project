"""
Classical 4th-order Runge–Kutta (RK4) time integrator
for planar 3DOF rigid-body dynamics.

This module implements a single explicit RK4 integration step
for systems represented by State3DOF objects.

State vector
------------

The system state is represented as:

    S = [x, z, vx, vz, θ, q]^T

The derivative function defines:

    dS/dt = f(S, t)

Numerical method
----------------

The RK4 update is:

    Sₙ₊₁ = Sₙ + (dt/6)(k₁ + 2k₂ + 2k₃ + k₄)

where:

    k₁ = f(Sₙ)
    k₂ = f(Sₙ + dt/2 · k₁)
    k₃ = f(Sₙ + dt/2 · k₂)
    k₄ = f(Sₙ + dt · k₃)

This integrator is:

• explicit
• fixed-step
• 4th-order accurate
• suitable for smooth rigid-body dynamics
"""

from __future__ import annotations

from typing import Callable

from modules.state.state import State3DOF, StateDerivatives3DOF


# ============================================================
# RK4 integration coefficients
# ============================================================

RK4_STAGE_TWO_FACTOR: float = 0.5
RK4_STAGE_THREE_FACTOR: float = 0.5
RK4_STAGE_FOUR_FACTOR: float = 1.0

RK4_WEIGHT_K1: float = 1.0
RK4_WEIGHT_K2: float = 2.0
RK4_WEIGHT_K3: float = 2.0
RK4_WEIGHT_K4: float = 1.0

RK4_NORMALIZATION: float = 1.0 / 6.0


# ============================================================
# RK4 integration step
# ============================================================

def rk4_step(
    derivative_function: Callable[..., StateDerivatives3DOF],
    current_state: State3DOF,
    time_step_seconds: float,
    **kwargs,
) -> State3DOF:
    """
    Perform a single classical RK4 integration step.

    Parameters
    ----------
    derivative_function : Callable
        Function implementing the system dynamics:

            dS/dt = f(S, t)

        Must return a StateDerivatives3DOF object.

    current_state : State3DOF
        Current state vector Sₙ

    time_step_seconds : float
        Integration step size dt

    **kwargs
        Additional parameters forwarded to derivative_function

    Returns
    -------
    State3DOF
        Updated state Sₙ₊₁
    """

    dt = float(time_step_seconds)

    # --------------------------------------------------------
    # Stage k1
    # --------------------------------------------------------

    k1 = derivative_function(current_state, **kwargs)

    # --------------------------------------------------------
    # Stage k2
    # --------------------------------------------------------

    state_k2 = State3DOF(
        x=current_state.x + dt * RK4_STAGE_TWO_FACTOR * k1.x,
        z=current_state.z + dt * RK4_STAGE_TWO_FACTOR * k1.z,
        vx=current_state.vx + dt * RK4_STAGE_TWO_FACTOR * k1.vx,
        vz=current_state.vz + dt * RK4_STAGE_TWO_FACTOR * k1.vz,
        theta=current_state.theta + dt * RK4_STAGE_TWO_FACTOR * k1.theta,
        q=current_state.q + dt * RK4_STAGE_TWO_FACTOR * k1.q,
    )

    k2 = derivative_function(state_k2, **kwargs)

    # --------------------------------------------------------
    # Stage k3
    # --------------------------------------------------------

    state_k3 = State3DOF(
        x=current_state.x + dt * RK4_STAGE_THREE_FACTOR * k2.x,
        z=current_state.z + dt * RK4_STAGE_THREE_FACTOR * k2.z,
        vx=current_state.vx + dt * RK4_STAGE_THREE_FACTOR * k2.vx,
        vz=current_state.vz + dt * RK4_STAGE_THREE_FACTOR * k2.vz,
        theta=current_state.theta + dt * RK4_STAGE_THREE_FACTOR * k2.theta,
        q=current_state.q + dt * RK4_STAGE_THREE_FACTOR * k2.q,
    )

    k3 = derivative_function(state_k3, **kwargs)

    # --------------------------------------------------------
    # Stage k4
    # --------------------------------------------------------

    state_k4 = State3DOF(
        x=current_state.x + dt * RK4_STAGE_FOUR_FACTOR * k3.x,
        z=current_state.z + dt * RK4_STAGE_FOUR_FACTOR * k3.z,
        vx=current_state.vx + dt * RK4_STAGE_FOUR_FACTOR * k3.vx,
        vz=current_state.vz + dt * RK4_STAGE_FOUR_FACTOR * k3.vz,
        theta=current_state.theta + dt * RK4_STAGE_FOUR_FACTOR * k3.theta,
        q=current_state.q + dt * RK4_STAGE_FOUR_FACTOR * k3.q,
    )

    k4 = derivative_function(state_k4, **kwargs)

    # --------------------------------------------------------
    # Weighted RK4 combination
    # --------------------------------------------------------

    weighted_dt = dt * RK4_NORMALIZATION

    return State3DOF(
        x=current_state.x + weighted_dt * (
            RK4_WEIGHT_K1 * k1.x
            + RK4_WEIGHT_K2 * k2.x
            + RK4_WEIGHT_K3 * k3.x
            + RK4_WEIGHT_K4 * k4.x
        ),
        z=current_state.z + weighted_dt * (
            RK4_WEIGHT_K1 * k1.z
            + RK4_WEIGHT_K2 * k2.z
            + RK4_WEIGHT_K3 * k3.z
            + RK4_WEIGHT_K4 * k4.z
        ),
        vx=current_state.vx + weighted_dt * (
            RK4_WEIGHT_K1 * k1.vx
            + RK4_WEIGHT_K2 * k2.vx
            + RK4_WEIGHT_K3 * k3.vx
            + RK4_WEIGHT_K4 * k4.vx
        ),
        vz=current_state.vz + weighted_dt * (
            RK4_WEIGHT_K1 * k1.vz
            + RK4_WEIGHT_K2 * k2.vz
            + RK4_WEIGHT_K3 * k3.vz
            + RK4_WEIGHT_K4 * k4.vz
        ),
        theta=current_state.theta + weighted_dt * (
            RK4_WEIGHT_K1 * k1.theta
            + RK4_WEIGHT_K2 * k2.theta
            + RK4_WEIGHT_K3 * k3.theta
            + RK4_WEIGHT_K4 * k4.theta
        ),
        q=current_state.q + weighted_dt * (
            RK4_WEIGHT_K1 * k1.q
            + RK4_WEIGHT_K2 * k2.q
            + RK4_WEIGHT_K3 * k3.q
            + RK4_WEIGHT_K4 * k4.q
        ),
    )
