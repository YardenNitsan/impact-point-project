from dataclasses import dataclass

@dataclass(frozen=True)
class State3DOF:
    x: float      # [m]
    z: float      # [m]
    vx: float     # [m/s]
    vz: float     # [m/s]
    theta: float  # [rad]
    q: float      # [rad/s]
