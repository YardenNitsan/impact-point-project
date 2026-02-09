# modules/solver/run_simulation.py

import math
from typing import Optional, List

from modules.dynamics.integrators import rk4_step
from modules.solver.solver_3dof import derivatives
from modules.state.state import State3DOF
from modules.atmosphere.wind import DrydenWindModel


MAX_ABS_VELOCITY = 2e4
MAX_ABS_Q = 1e3
MAX_ABS_THETA = 1e3


def _is_sane(s: State3DOF) -> bool:
    vals = [s.x, s.z, s.vx, s.vz, s.theta, s.q]
    if not all(math.isfinite(v) for v in vals):
        return False
    if abs(s.vx) > MAX_ABS_VELOCITY or abs(s.vz) > MAX_ABS_VELOCITY:
        return False
    if abs(s.q) > MAX_ABS_Q:
        return False
    if abs(s.theta) > MAX_ABS_THETA:
        return False
    return True


def _interpolate_to_ground(s0: State3DOF, s1: State3DOF) -> State3DOF:
    """Linear interpolation between two states to find z=0 crossing."""
    z0, z1 = s0.z, s1.z
    denom = (z0 - z1)
    if abs(denom) < 1e-12:
        t = 0.0
    else:
        t = z0 / denom  # z(t)=0
    t = max(0.0, min(1.0, t))

    def lerp(a: float, b: float) -> float:
        return a + (b - a) * t

    s = State3DOF(
        x=lerp(s0.x, s1.x),
        z=0.0,
        vx=lerp(s0.vx, s1.vx),
        vz=lerp(s0.vz, s1.vz),
        theta=lerp(s0.theta, s1.theta),
        q=lerp(s0.q, s1.q),
    )
    return s


def run_simulation(
    state0: State3DOF,
    dt: float,
    max_time: float,
    *,
    wind_model: Optional[DrydenWindModel] = None,
    **params,
):
    """Run 3DOF simulation and return trajectory list.

    Matches project requirement: stop exactly at ground impact by interpolating
    last step to z=0 (not just 'first state with z<=0').
    """

    t = 0.0
    state = state0
    trajectory: List[State3DOF] = []

    if not _is_sane(state):
        raise RuntimeError("Initial state not sane")

    while t < max_time:
        trajectory.append(state)

        # Already at ground
        if state.z <= 0.0:
            trajectory[-1] = State3DOF(
                x=state.x, z=0.0, vx=state.vx, vz=state.vz, theta=state.theta, q=state.q
            )
            break

        # Wind (turbulence) update
        if wind_model is None:
            wind_x, wind_z = 0.0, 0.0
        else:
            Va = math.hypot(state.vx - wind_model.ut, state.vz - wind_model.wt)
            wind_x, wind_z = wind_model.step(state.z, Va, dt)

        # Integrate one step
        state_next = rk4_step(
            derivatives,
            state,
            dt,
            wind_x=wind_x,
            wind_z=wind_z,
            **params,
        )

        if not _is_sane(state_next):
            raise RuntimeError(f"Numerical blow-up detected at t={t:.3f}s")

        # Ground crossing: add interpolated impact state and stop
        if state_next.z <= 0.0:
            impact_state = _interpolate_to_ground(state, state_next)
            trajectory.append(impact_state)
            break

        state = state_next
        t += dt

    return trajectory
