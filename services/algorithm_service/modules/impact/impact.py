from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from modules.state.state import State3DOF


@dataclass(frozen=True)
class ImpactResult:
    """Interpolated impact state at z=0 plus some metadata."""
    state_at_impact: State3DOF
    index_before: int        # index of last state with z > 0
    index_after: int         # index of first state with z <= 0
    frac: float              # interpolation fraction in [0,1] from before->after


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_state(s0: State3DOF, s1: State3DOF, t: float) -> State3DOF:
    """Linear interpolation between two State3DOF."""
    return State3DOF(
        x=_lerp(s0.x, s1.x, t),
        z=_lerp(s0.z, s1.z, t),
        vx=_lerp(s0.vx, s1.vx, t),
        vz=_lerp(s0.vz, s1.vz, t),
        theta=_lerp(s0.theta, s1.theta, t),
        q=_lerp(s0.q, s1.q, t),
    )


def compute_impact_from_trajectory(traj: List[State3DOF]) -> Optional[ImpactResult]:
    """
    Finds the first crossing of z from >0 to <=0 and interpolates the state at z=0.

    Returns None if:
    - trajectory is too short
    - no crossing was found (never hit ground)
    - starts at/under ground (first state z <= 0)
    """
    if len(traj) < 2:
        return None

    if traj[0].z <= 0.0:
        return None

    for i in range(1, len(traj)):
        z_prev = traj[i - 1].z
        z_curr = traj[i].z

        if z_prev > 0.0 and z_curr <= 0.0:
            denom = (z_prev - z_curr)
            if abs(denom) < 1e-12:
                t = 0.0
            else:
                # Solve z(t)=0 for t in [0,1]:
                # z(t) = z_prev + (z_curr - z_prev)*t => t = z_prev/(z_prev - z_curr)
                t = z_prev / denom

            # clamp safety
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            s_imp = interpolate_state(traj[i - 1], traj[i], t)
            # force exact ground
            s_imp = State3DOF(
                x=s_imp.x,
                z=0.0,
                vx=s_imp.vx,
                vz=s_imp.vz,
                theta=s_imp.theta,
                q=s_imp.q,
            )

            return ImpactResult(
                state_at_impact=s_imp,
                index_before=i - 1,
                index_after=i,
                frac=t,
            )

    return None
