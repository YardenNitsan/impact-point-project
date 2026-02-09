# modules/solver/solver_3dof.py

import math

from modules.state.state import State3DOF
from modules.atmosphere.isa import isa_atmosphere, speed_of_sound
from modules.aerodynamics.aerodynamics import compute_Xv_Zv_My_from_table, AeroRef
from modules.aerodynamics.aero_tables import AeroTable2D, wrap_to_pi


def derivatives(
    state: State3DOF,
    *,
    mass: float,
    Iyy: float,
    g: float,
    T0: float,
    P0: float,
    aero_ref: AeroRef,
    aero_table: AeroTable2D,
    wind_x: float,
    wind_z: float,
    lcg: float = 0.0,
):

    z = state.z
    vx = state.vx
    vz = state.vz
    theta = state.theta
    q = state.q

    T, P, _rho = isa_atmosphere(z, T0=T0, P0=P0)
    a = speed_of_sound(T)

    vx_rel = vx - wind_x
    vz_rel = vz - wind_z

    V_rel = math.hypot(vx_rel, vz_rel)
    gamma_rel = math.atan2(vz_rel, vx_rel)

    # 🔧 FIX: wrap alpha
    alpha = wrap_to_pi(theta - gamma_rel)

    mach = (V_rel / a) if a > 0.0 else 0.0

    coeffs = aero_table.lookup(alpha, mach)

    Xv, Zv, My, _Mach_used, _CM_eff = compute_Xv_Zv_My_from_table(
        P=P,
        T=T,
        vx=vx,
        vz=vz,
        wind_x=wind_x,
        wind_z=wind_z,
        alpha=alpha,
        mach=mach,
        coeffs=coeffs,
        ref=aero_ref,
        q=q,
        lcg=lcg,
    )

    dx = vx
    dz = vz

    dvx = Xv / mass
    dvz = Zv / mass - g

    dtheta = q
    dq = My / Iyy

    return State3DOF(dx, dz, dvx, dvz, dtheta, dq)
