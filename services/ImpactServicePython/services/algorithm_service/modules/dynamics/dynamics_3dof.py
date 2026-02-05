from dataclasses import dataclass
from typing import Tuple

# =====================================================
# 3DOF inertial equations
#
# Convention used in THIS project:
# x: horizontal forward [m]
# z: altitude UP (positive upward) [m]
#
# vx, vz: velocities
# theta: pitch angle
# q: pitch rate
#
# x_ddot = Xv / m
# z_ddot = Zv / m - g
# theta_ddot = My / Iyy
# =====================================================

@dataclass(frozen=True)
class State3DOF:
    x: float
    z: float
    vx: float
    vz: float
    theta: float
    q: float


@dataclass(frozen=True)
class StateDerivatives3DOF:
    x: float
    z: float
    vx: float
    vz: float
    theta: float
    q: float


def accelerations_3dof_inertial(
    Xv: float,
    Zv: float,
    My: float,
    m: float,
    Iyy: float,
    g: float
) -> Tuple[float, float, float]:

    x_ddot = Xv / m
    z_ddot = (Zv / m) - g
    theta_ddot = My / Iyy

    return x_ddot, z_ddot, theta_ddot


def state_derivatives(
    state: State3DOF,
    Xv: float,
    Zv: float,
    My: float,
    m: float,
    Iyy: float,
    g: float
) -> StateDerivatives3DOF:

    x_ddot, z_ddot, theta_ddot = accelerations_3dof_inertial(
        Xv, Zv, My, m, Iyy, g
    )

    return StateDerivatives3DOF(
        x=state.vx,
        z=state.vz,
        vx=x_ddot,
        vz=z_ddot,
        theta=state.q,
        q=theta_ddot,
    )
