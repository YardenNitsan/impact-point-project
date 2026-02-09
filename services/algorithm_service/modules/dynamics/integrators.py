from __future__ import annotations

from modules.state.state import State3DOF


def rk4_step(derivs, state: State3DOF, dt: float, **kwargs) -> State3DOF:
    """One RK4 step for State3DOF.

    Signature intentionally: rk4_step(derivs, state, dt, **kwargs)
    so callers can write rk4_step(derivatives, state, dt, ...).
    """

    k1 = derivs(state, **kwargs)

    k2 = derivs(
        State3DOF(
            x=state.x + 0.5 * dt * k1.x,
            z=state.z + 0.5 * dt * k1.z,
            vx=state.vx + 0.5 * dt * k1.vx,
            vz=state.vz + 0.5 * dt * k1.vz,
            theta=state.theta + 0.5 * dt * k1.theta,
            q=state.q + 0.5 * dt * k1.q,
        ),
        **kwargs,
    )

    k3 = derivs(
        State3DOF(
            x=state.x + 0.5 * dt * k2.x,
            z=state.z + 0.5 * dt * k2.z,
            vx=state.vx + 0.5 * dt * k2.vx,
            vz=state.vz + 0.5 * dt * k2.vz,
            theta=state.theta + 0.5 * dt * k2.theta,
            q=state.q + 0.5 * dt * k2.q,
        ),
        **kwargs,
    )

    k4 = derivs(
        State3DOF(
            x=state.x + dt * k3.x,
            z=state.z + dt * k3.z,
            vx=state.vx + dt * k3.vx,
            vz=state.vz + dt * k3.vz,
            theta=state.theta + dt * k3.theta,
            q=state.q + dt * k3.q,
        ),
        **kwargs,
    )

    return State3DOF(
        x=state.x + (dt / 6.0) * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
        z=state.z + (dt / 6.0) * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
        vx=state.vx + (dt / 6.0) * (k1.vx + 2.0 * k2.vx + 2.0 * k3.vx + k4.vx),
        vz=state.vz + (dt / 6.0) * (k1.vz + 2.0 * k2.vz + 2.0 * k3.vz + k4.vz),
        theta=state.theta + (dt / 6.0) * (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta),
        q=state.q + (dt / 6.0) * (k1.q + 2.0 * k2.q + 2.0 * k3.q + k4.q),
    )
