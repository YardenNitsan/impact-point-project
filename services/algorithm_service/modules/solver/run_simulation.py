from __future__ import annotations

"""
3DOF simulation time-marching loop (RK4) with wind coupling and ground impact detection.

This module performs the numerical integration for the planar 3DOF model
defined in `modules.solver.solver_3dof.derivatives`.

Responsibilities
----------------
1) Advance the 3DOF state using a fixed-step Runge–Kutta 4 (RK4) integrator.
2) Couple a wind model to the dynamics (deterministic shear / turbulence / none).
3) Detect ground impact (z <= 0) and stop the integration.
4) Return a trajectory that ends exactly at z = 0 by linear interpolation
   between the last above-ground state and the first below-ground state.
5) Apply loose numerical sanity checks to catch divergence early.

State conventions
-----------------
The state is expressed in a planar inertial frame:

    x     : downrange distance along the launch direction [m]
    z     : altitude above ground (positive upward) [m]
    vx    : x-axis inertial velocity [m/s]
    vz    : z-axis inertial velocity [m/s]
    theta : pitch angle (model convention: radians) [rad]
    q     : pitch rate [rad/s]

Ground impact condition
-----------------------
Ground is defined at:

    z_ground = 0.0 [m]

Impact is detected when the trajectory crosses z_ground. The returned last
sample is interpolated to satisfy z == z_ground exactly.

Interpolation model
-------------------
Linear interpolation is applied in state-space between two consecutive states
(s_before, s_after) that straddle the ground plane:

    z(t) = z_before + alpha * (z_after - z_before)
    Solve for alpha such that z(t) = z_ground.

Assumptions & limitations
-------------------------
• Fixed time step integration (dt constant).
• Ground is a flat plane at z=0 (no terrain).
• Linear interpolation for impact time/state (sufficient for small dt).
• Numerical sanity checks are deliberately loose to avoid false positives.
"""

import math
from typing import Any, List, Optional, Protocol, Tuple, TypeAlias

from modules.dynamics.integrators import rk4_step
from modules.solver.solver_3dof import derivatives
from modules.state.state import State3DOF


# ============================================================
# Numeric & physical constants
# ============================================================

GROUND_ALTITUDE_M: float = 0.0
"""Ground altitude reference used by the integrator stop condition."""

INTERPOLATION_EPSILON: float = 1e-12
"""Small epsilon to guard division by near-zero altitude differences."""

# Loose safety limits (detect divergence / NaNs early; tune if needed)
MAX_ABSOLUTE_VELOCITY_MPS: float = 2.0e4
MAX_ABSOLUTE_PITCH_RATE_RADPS: float = 1.0e3
MAX_ABSOLUTE_PITCH_ANGLE_RAD: float = 1.0e3


# ============================================================
# Wind model typing (best-effort, adapter-based)
# ============================================================

WindXZ: TypeAlias = Tuple[float, float]


class DeterministicWindModel(Protocol):
    """
    Deterministic wind profile interface.

    Implementations should provide wind components in the solver frame:
        wind_x : along-downrange wind component [m/s]
        wind_z : vertical wind component [m/s]
    """

    def wind_at_height(self, z_m: float) -> WindXZ:  # pragma: no cover
        ...


class StepWindModel(Protocol):
    """
    Stochastic / step-based wind interface (e.g., Dryden turbulence).

    The model advances internal state each call and returns wind_x, wind_z.
    Some implementations expose turbulence components (ut, wt) that represent
    gust velocity in the body/inertial frame; we support that convention.
    """

    # Optional public attributes in some turbulence models
    ut: float
    wt: float

    def step(self, z_m: float, airspeed_mps: float, dt_s: float) -> WindXZ:  # pragma: no cover
        ...


WindModel: TypeAlias = Optional[Any]  # kept permissive for project flexibility


# ============================================================
# Sanity checks
# ============================================================

def _is_finite_state(state: State3DOF) -> bool:
    return all(
        math.isfinite(v)
        for v in (state.x, state.z, state.vx, state.vz, state.theta, state.q)
    )


def is_state_numerically_sane(state: State3DOF) -> bool:
    """
    Loose sanity guard for integrator stability.

    Returns False if:
    - any value is NaN/Inf
    - velocity exceeds a very high ceiling
    - pitch rate / pitch angle exceed very high ceilings
    """
    if not _is_finite_state(state):
        return False

    if abs(state.vx) > MAX_ABSOLUTE_VELOCITY_MPS or abs(state.vz) > MAX_ABSOLUTE_VELOCITY_MPS:
        return False

    if abs(state.q) > MAX_ABSOLUTE_PITCH_RATE_RADPS:
        return False

    if abs(state.theta) > MAX_ABSOLUTE_PITCH_ANGLE_RAD:
        return False

    return True


# ============================================================
# Ground crossing interpolation
# ============================================================

def _lerp(a: float, b: float, alpha: float) -> float:
    """Linear interpolation between scalars."""
    return a + (b - a) * alpha


def interpolate_state_to_ground(
    state_before: State3DOF,
    state_after: State3DOF,
    *,
    ground_altitude_m: float = GROUND_ALTITUDE_M,
) -> State3DOF:
    """
    Interpolate between two states to the exact ground crossing (z = ground_altitude_m).

    Preconditions:
        state_before.z > ground_altitude_m
        state_after.z  <= ground_altitude_m

    If the altitude delta is extremely small, the function falls back to alpha=0.
    """

    z0 = float(state_before.z)
    z1 = float(state_after.z)

    dz = z1 - z0  # expected negative when crossing downward
    if abs(dz) < INTERPOLATION_EPSILON:
        alpha = 0.0
    else:
        alpha = (ground_altitude_m - z0) / dz

    # Clamp alpha to protect against numerical noise (should already be in [0, 1])
    alpha = max(0.0, min(1.0, alpha))

    return State3DOF(
        x=_lerp(state_before.x, state_after.x, alpha),
        z=ground_altitude_m,
        vx=_lerp(state_before.vx, state_after.vx, alpha),
        vz=_lerp(state_before.vz, state_after.vz, alpha),
        theta=_lerp(state_before.theta, state_after.theta, alpha),
        q=_lerp(state_before.q, state_after.q, alpha),
    )


# ============================================================
# Wind adapter
# ============================================================

def compute_wind_components(
    *,
    wind_model: WindModel,
    current_state: State3DOF,
    dt_s: float,
) -> WindXZ:
    """
    Compute wind components (wind_x, wind_z) in solver axes.

    Supported conventions:
    1) Deterministic profile:
        wind_model.wind_at_height(z) -> (wind_x, wind_z)

    2) Step-based / turbulence:
        wind_model.step(z, Va, dt) -> (wind_x, wind_z)

        Where Va is estimated from current inertial velocity minus the model's
        gust components (ut, wt) if they exist.

    3) None or unknown model: returns (0, 0).

    Notes:
    - This function is intentionally defensive and avoids strict isinstance checks.
    - Returning floats guarantees consistent downstream usage.
    """

    if wind_model is None:
        return 0.0, 0.0

    # Deterministic wind profile
    wind_at_height = getattr(wind_model, "wind_at_height", None)
    if callable(wind_at_height):
        wx, wz = wind_at_height(float(current_state.z))
        return float(wx), float(wz)

    # Step-based wind model (e.g., turbulence)
    step = getattr(wind_model, "step", None)
    if callable(step):
        ut = float(getattr(wind_model, "ut", 0.0))
        wt = float(getattr(wind_model, "wt", 0.0))

        # Best-effort estimate of airspeed magnitude relative to gust components
        va = math.hypot(current_state.vx - ut, current_state.vz - wt)

        wx, wz = step(float(current_state.z), float(va), float(dt_s))
        return float(wx), float(wz)

    return 0.0, 0.0


# ============================================================
# Public API
# ============================================================

def run_simulation(
    state0: State3DOF,
    dt: float,
    max_time: float,
    *,
    wind_model: WindModel = None,
    ground_altitude_m: float = GROUND_ALTITUDE_M,
    **params: Any,
) -> List[State3DOF]:
    """
    Run the fixed-step 3DOF simulation and return the trajectory states.

    Parameters
    ----------
    state0:
        Initial state at t=0.
    dt:
        Fixed integration time step [s].
    max_time:
        Maximum simulated time horizon [s].
    wind_model:
        Optional wind model (deterministic profile or turbulence stepper).
    ground_altitude_m:
        Ground plane altitude used for termination (default: 0.0).
    **params:
        Forwarded to `derivatives(...)` (mass, Iyy, g, T0, P0, aero_ref, aero_table, lcg, ...).

    Returns
    -------
    List[State3DOF]
        Trajectory states. The last sample is guaranteed to satisfy:
            state.z == ground_altitude_m
        if a ground crossing occurs within max_time.

    Raises
    ------
    RuntimeError
        If initial state is not numerically sane, or if a numerical blow-up is detected.
    """

    dt_s = float(dt)
    max_time_s = float(max_time)

    if dt_s <= 0.0:
        raise ValueError("dt must be positive")
    if max_time_s <= 0.0:
        raise ValueError("max_time must be positive")

    if not is_state_numerically_sane(state0):
        raise RuntimeError("Initial state is not numerically sane")

    t_s = 0.0
    state = state0
    trajectory: List[State3DOF] = []

    while t_s < max_time_s:

        trajectory.append(state)

        # If we are already at/below ground, clamp and stop.
        if state.z <= ground_altitude_m:
            trajectory[-1] = State3DOF(
                x=state.x,
                z=ground_altitude_m,
                vx=state.vx,
                vz=state.vz,
                theta=state.theta,
                q=state.q,
            )
            break

        wind_x, wind_z = compute_wind_components(
            wind_model=wind_model,
            current_state=state,
            dt_s=dt_s,
        )

        next_state = rk4_step(
            derivatives,
            state,
            dt_s,
            wind_x_mps=wind_x,
            wind_z_mps=wind_z,
            **params,
        )

        if not is_state_numerically_sane(next_state):
            raise RuntimeError(f"Numerical blow-up detected at t={t_s:.3f}s")

        # Ground crossing detection (downward crossing)
        if next_state.z <= ground_altitude_m:
            impact_state = interpolate_state_to_ground(
                state,
                next_state,
                ground_altitude_m=ground_altitude_m,
            )
            trajectory.append(impact_state)
            break

        state = next_state
        t_s += dt_s

    return trajectory
