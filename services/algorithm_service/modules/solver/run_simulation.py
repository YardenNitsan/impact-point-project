from __future__ import annotations

"""
3DOF simulation time-marching loop (RK4) with optional cached weather coupling.
"""

import math
from typing import Any, List, Optional, Protocol, Tuple, TypeAlias

from modules.dynamics.integrators import rk4_step
from modules.solver.solver_3dof import derivatives
from modules.state.state import State3DOF
from modules.atmosphere.weather_runtime import TrajectoryWeatherRuntime

GROUND_ALTITUDE_M: float = 0.0
INTERPOLATION_EPSILON: float = 1e-12
MAX_ABSOLUTE_VELOCITY_MPS: float = 2.0e4
MAX_ABSOLUTE_PITCH_RATE_RADPS: float = 1.0e3
MAX_ABSOLUTE_PITCH_ANGLE_RAD: float = 1.0e3

WindXZ: TypeAlias = Tuple[float, float]


class DeterministicWindModel(Protocol):
    def wind_at_height(self, z_m: float) -> WindXZ:  # pragma: no cover
        ...


class StepWindModel(Protocol):
    ut: float
    wt: float

    def step(self, z_m: float, airspeed_mps: float, dt_s: float) -> WindXZ:  # pragma: no cover
        ...


WindModel: TypeAlias = Optional[Any]


def _is_finite_state(state: State3DOF) -> bool:
    return all(
        math.isfinite(v)
        for v in (state.x, state.z, state.vx, state.vz, state.theta, state.q)
    )


def is_state_numerically_sane(state: State3DOF) -> bool:
    if not _is_finite_state(state):
        return False

    if abs(state.vx) > MAX_ABSOLUTE_VELOCITY_MPS or abs(state.vz) > MAX_ABSOLUTE_VELOCITY_MPS:
        return False

    if abs(state.q) > MAX_ABSOLUTE_PITCH_RATE_RADPS:
        return False

    if abs(state.theta) > MAX_ABSOLUTE_PITCH_ANGLE_RAD:
        return False

    return True


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def interpolate_state_to_ground(
    state_before: State3DOF,
    state_after: State3DOF,
    *,
    ground_altitude_m: float = GROUND_ALTITUDE_M,
) -> State3DOF:
    z0 = float(state_before.z)
    z1 = float(state_after.z)

    dz = z1 - z0
    if abs(dz) < INTERPOLATION_EPSILON:
        alpha = 0.0
    else:
        alpha = (ground_altitude_m - z0) / dz

    alpha = max(0.0, min(1.0, alpha))

    return State3DOF(
        x=_lerp(state_before.x, state_after.x, alpha),
        z=ground_altitude_m,
        vx=_lerp(state_before.vx, state_after.vx, alpha),
        vz=_lerp(state_before.vz, state_after.vz, alpha),
        theta=_lerp(state_before.theta, state_after.theta, alpha),
        q=_lerp(state_before.q, state_after.q, alpha),
    )


def compute_wind_components(
    *,
    wind_model: WindModel,
    current_state: State3DOF,
    dt_s: float,
) -> WindXZ:
    if wind_model is None:
        return 0.0, 0.0

    wind_at_height = getattr(wind_model, "wind_at_height", None)
    if callable(wind_at_height):
        wx, wz = wind_at_height(float(current_state.z))
        return float(wx), float(wz)

    step = getattr(wind_model, "step", None)
    if callable(step):
        ut = float(getattr(wind_model, "ut", 0.0))
        wt = float(getattr(wind_model, "wt", 0.0))
        va = math.hypot(current_state.vx - ut, current_state.vz - wt)
        wx, wz = step(float(current_state.z), float(va), float(dt_s))
        return float(wx), float(wz)

    return 0.0, 0.0


def _resolve_step_environment(
    *,
    state: State3DOF,
    elapsed_time_s: float,
    wind_model: WindModel,
    weather_runtime: TrajectoryWeatherRuntime | None,
    params: dict[str, Any],
    dt_s: float,
) -> tuple[float, float, float, float]:
    if weather_runtime is not None:
        sample = weather_runtime.get_sample(
            x_m=float(state.x),
            altitude_m=float(state.z),
            elapsed_time_s=float(elapsed_time_s),
        )
        return (
            float(sample.temperature_K),
            float(sample.pressure_Pa),
            float(sample.wind_x_mps),
            float(sample.wind_z_mps),
        )

    if "temperature_K" not in params or "pressure_Pa" not in params:
        raise ValueError("temperature_K and pressure_Pa must be provided when weather_runtime is not used")

    wind_x, wind_z = compute_wind_components(
        wind_model=wind_model,
        current_state=state,
        dt_s=float(dt_s),
    )

    return (
        float(params["temperature_K"]),
        float(params["pressure_Pa"]),
        float(wind_x),
        float(wind_z),
    )


def run_simulation(
    state0: State3DOF,
    dt: float,
    max_time: float,
    *,
    wind_model: WindModel = None,
    weather_runtime: TrajectoryWeatherRuntime | None = None,
    ground_altitude_m: float = GROUND_ALTITUDE_M,
    **params: Any,
) -> List[State3DOF]:
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

        temperature_K, pressure_Pa, wind_x, wind_z = _resolve_step_environment(
            state=state,
            elapsed_time_s=t_s,
            wind_model=wind_model,
            weather_runtime=weather_runtime,
            params=params,
            dt_s=dt_s,
        )

        next_state = rk4_step(
            derivatives,
            state,
            dt_s,
            wind_x_mps=wind_x,
            wind_z_mps=wind_z,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            **params,
        )

        if not is_state_numerically_sane(next_state):
            raise RuntimeError(f"Numerical blow-up detected at t={t_s:.3f}s")

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


def run_simulation_impact_only(
    state0: State3DOF,
    dt: float,
    max_time: float,
    *,
    wind_model: WindModel = None,
    weather_runtime: TrajectoryWeatherRuntime | None = None,
    ground_altitude_m: float = GROUND_ALTITUDE_M,
    **params: Any,
) -> tuple[State3DOF, int]:
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
    raw_points_count = 1

    while t_s < max_time_s:
        if state.z <= ground_altitude_m:
            return (
                State3DOF(
                    x=state.x,
                    z=ground_altitude_m,
                    vx=state.vx,
                    vz=state.vz,
                    theta=state.theta,
                    q=state.q,
                ),
                raw_points_count,
            )

        temperature_K, pressure_Pa, wind_x, wind_z = _resolve_step_environment(
            state=state,
            elapsed_time_s=t_s,
            wind_model=wind_model,
            weather_runtime=weather_runtime,
            params=params,
            dt_s=dt_s,
        )

        next_state = rk4_step(
            derivatives,
            state,
            dt_s,
            wind_x_mps=wind_x,
            wind_z_mps=wind_z,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            **params,
        )
        raw_points_count += 1

        if not is_state_numerically_sane(next_state):
            raise RuntimeError(f"Numerical blow-up detected at t={t_s:.3f}s")

        if next_state.z <= ground_altitude_m:
            impact_state = interpolate_state_to_ground(
                state,
                next_state,
                ground_altitude_m=ground_altitude_m,
            )
            return impact_state, raw_points_count

        state = next_state
        t_s += dt_s

    raise RuntimeError("No ground impact detected within max_time")


def run_simulation_sampled(
    state0: State3DOF,
    dt: float,
    max_time: float,
    *,
    dx_sample_m: float,
    wind_model: WindModel = None,
    weather_runtime: TrajectoryWeatherRuntime | None = None,
    ground_altitude_m: float = GROUND_ALTITUDE_M,
    **params: Any,
) -> tuple[State3DOF, List[State3DOF], int]:
    dt_s = float(dt)
    max_time_s = float(max_time)

    if dt_s <= 0.0:
        raise ValueError("dt must be positive")
    if max_time_s <= 0.0:
        raise ValueError("max_time must be positive")
    if dx_sample_m <= 0.0:
        raise ValueError("dx_sample_m must be positive")

    if not is_state_numerically_sane(state0):
        raise RuntimeError("Initial state is not numerically sane")

    t_s = 0.0
    state = state0
    raw_points_count = 1

    sampled_states: List[State3DOF] = [state0]
    last_sample_x = float(state0.x)

    while t_s < max_time_s:
        if state.z <= ground_altitude_m:
            clamped = State3DOF(
                x=state.x,
                z=ground_altitude_m,
                vx=state.vx,
                vz=state.vz,
                theta=state.theta,
                q=state.q,
            )

            if sampled_states[-1].z != ground_altitude_m:
                sampled_states.append(clamped)

            return clamped, sampled_states, raw_points_count

        temperature_K, pressure_Pa, wind_x, wind_z = _resolve_step_environment(
            state=state,
            elapsed_time_s=t_s,
            wind_model=wind_model,
            weather_runtime=weather_runtime,
            params=params,
            dt_s=dt_s,
        )

        next_state = rk4_step(
            derivatives,
            state,
            dt_s,
            wind_x_mps=wind_x,
            wind_z_mps=wind_z,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            **params,
        )
        raw_points_count += 1

        if not is_state_numerically_sane(next_state):
            raise RuntimeError(f"Numerical blow-up detected at t={t_s:.3f}s")

        if next_state.z <= ground_altitude_m:
            impact_state = interpolate_state_to_ground(
                state,
                next_state,
                ground_altitude_m=ground_altitude_m,
            )

            if (impact_state.x - last_sample_x) >= dx_sample_m:
                sampled_states.append(impact_state)
            elif sampled_states[-1].z != ground_altitude_m:
                sampled_states.append(impact_state)

            return impact_state, sampled_states, raw_points_count

        if (next_state.x - last_sample_x) >= dx_sample_m:
            sampled_states.append(next_state)
            last_sample_x = float(next_state.x)

        state = next_state
        t_s += dt_s

    raise RuntimeError("No ground impact detected within max_time")