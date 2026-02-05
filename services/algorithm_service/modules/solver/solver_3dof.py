import math
from modules.state.state import State3DOF

from modules.atmosphere.isa import isa_atmosphere
from modules.dynamics.dynamics_3dof import accelerations_3dof_inertial
from modules.aerodynamics.aerodynamics import compute_Xv_Zv_My_from_table, AeroRef
from modules.aerodynamics.aero_tables import AeroTable2D


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


ALPHA_MAX = math.radians(20.0)  # for table validity/safety
Q_MAX = 50.0                    # q clamp used ONLY inside aero lookup
QDOT_MAX = 200.0                # safety clamp to avoid numerical runaway


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
    lcg: float = 0.0,
):
    # altitude (z positive UP)
    h = max(0.0, state.z)

    # Atmosphere
    T, P, rho = isa_atmosphere(h, T0, P0)

    # Speed / flight path angle (gamma)
    V = math.hypot(state.vx, state.vz)
    gamma = math.atan2(state.vz, state.vx)

    # Wrap theta to avoid drifting
    theta_use = _wrap_pi(state.theta)

    # AoA = theta - gamma
    alpha = _clamp(theta_use - gamma, -ALPHA_MAX, ALPHA_MAX)

    # Mach for table lookup
    a = math.sqrt(1.4 * 287.05 * T)
    Mach = V / a if a > 1e-9 else 0.0

    coeffs = aero_table.lookup(alpha, Mach)

    # IMPORTANT:
    # Clamp q ONLY for aerodynamic model stability,
    # but DO NOT use q_use as theta_dot.
    q_use = _clamp(state.q, -Q_MAX, Q_MAX)

    # Aero forces & moment in Vehicle-Carried/Inertial (XV, ZV, My)
    Xv, Zv, My, Mach_used, CM_eff = compute_Xv_Zv_My_from_table(
        P=P,
        T=T,
        vx=state.vx,
        vz=state.vz,
        q=q_use,
        alpha=alpha,
        CD=coeffs.CD,
        CL=coeffs.CL,
        CM0=coeffs.CM0,
        Cm_alpha=coeffs.Cm_alpha,
        Cmq=coeffs.Cmq,
        ref=aero_ref,
        lcg=lcg
    )

    # Accelerations per the thesis equations:
    # x_ddot = Xv/m
    # z_ddot = Zv/m - g
    # q_dot  = My/Iyy  (and theta_dot = q)
    ax, az, q_dot = accelerations_3dof_inertial(Xv, Zv, My, mass, Iyy, g)

    # safety clamp (optional)
    q_dot = _clamp(q_dot, -QDOT_MAX, QDOT_MAX)

    # Return derivatives vector:
    return State3DOF(
        x=state.vx,
        z=state.vz,
        vx=ax,
        vz=az,
        theta=state.q,
        q=q_dot
    )
