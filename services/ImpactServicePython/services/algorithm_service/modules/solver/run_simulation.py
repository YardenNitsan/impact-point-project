# services/algorithm_service/modules/solver/run_simulation.py
import math
from services.algorithm_service.modules.dynamics.integrators import rk4_step
from services.algorithm_service.modules.solver.solver_3dof import derivatives
from services.algorithm_service.modules.state.state import State3DOF


def _is_sane(s: State3DOF) -> bool:
    vals = [s.x, s.z, s.vx, s.vz, s.theta, s.q]
    if not all(math.isfinite(v) for v in vals):
        return False
    if abs(s.vx) > 2e4 or abs(s.vz) > 2e4:
        return False
    if abs(s.q) > 1e3:
        return False
    if abs(s.theta) > 1e3:
        return False
    return True


def run_simulation(state0, dt, max_time, **params):
    t = 0.0
    state = state0
    trajectory = []

    if not _is_sane(state):
        raise RuntimeError("Initial state not sane")

    while t < max_time:
        trajectory.append(state)

        # already hit
        if state.z <= 0.0:
            break

        next_state = rk4_step(state, dt, derivatives, **params)

        # sanity stop
        if not _is_sane(next_state):
            raise RuntimeError(f"Numerical blow-up detected at t={t:.3f}s: {next_state}")

        # impact interpolation (avoid going below ground)
        if next_state.z <= 0.0:
            frac = state.z / (state.z - next_state.z)  # in (0,1]

            impact = State3DOF(
                x=state.x + frac * (next_state.x - state.x),
                z=0.0,
                vx=state.vx + frac * (next_state.vx - state.vx),
                vz=state.vz + frac * (next_state.vz - state.vz),
                theta=state.theta + frac * (next_state.theta - state.theta),
                q=state.q + frac * (next_state.q - state.q),
            )

            trajectory.append(impact)
            break

        state = next_state
        t += dt

    return trajectory
