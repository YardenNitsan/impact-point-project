from modules.state.state import State3DOF


def rk4_step(state: State3DOF, dt: float, derivs, **kwargs):
    k1 = derivs(state, **kwargs)

    k2 = derivs(
        State3DOF(
            state.x + 0.5 * dt * k1.x,
            state.z + 0.5 * dt * k1.z,
            state.vx + 0.5 * dt * k1.vx,
            state.vz + 0.5 * dt * k1.vz,
            state.theta + 0.5 * dt * k1.theta,
            state.q + 0.5 * dt * k1.q,
        ),
        **kwargs
    )

    k3 = derivs(
        State3DOF(
            state.x + 0.5 * dt * k2.x,
            state.z + 0.5 * dt * k2.z,
            state.vx + 0.5 * dt * k2.vx,
            state.vz + 0.5 * dt * k2.vz,
            state.theta + 0.5 * dt * k2.theta,
            state.q + 0.5 * dt * k2.q,
        ),
        **kwargs
    )

    k4 = derivs(
        State3DOF(
            state.x + dt * k3.x,
            state.z + dt * k3.z,
            state.vx + dt * k3.vx,
            state.vz + dt * k3.vz,
            state.theta + dt * k3.theta,
            state.q + dt * k3.q,
        ),
        **kwargs
    )

    return State3DOF(
        x=state.x + dt / 6 * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
        z=state.z + dt / 6 * (k1.z + 2 * k2.z + 2 * k3.z + k4.z),
        vx=state.vx + dt / 6 * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx),
        vz=state.vz + dt / 6 * (k1.vz + 2 * k2.vz + 2 * k3.vz + k4.vz),
        theta=state.theta + dt / 6 * (k1.theta + 2 * k2.theta + 2 * k3.theta + k4.theta),
        q=state.q + dt / 6 * (k1.q + 2 * k2.q + 2 * k3.q + k4.q),
    )
