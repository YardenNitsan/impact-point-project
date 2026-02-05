import math
from typing import Tuple

# ISA constants (troposphere model up to 11 km)
G = 9.81            # [m/s^2]
R = 287.05          # [J/(kg*K)]
L = 0.0065          # [K/m]
MAX_ALTITUDE = 11000.0  # [m]

def isa_atmosphere(h: float, T0: float, P0: float) -> Tuple[float, float, float]:
    """
    ISA troposphere (clamped to [0, 11km]).
    Inputs:
      h  : altitude [m]
      T0 : sea-level temperature [K]
      P0 : sea-level pressure [Pa]
    Returns:
      T   [K], P [Pa], rho [kg/m^3]
    """
    h = max(0.0, min(float(h), MAX_ALTITUDE))

    T = T0 - L * h
    exponent = G / (R * L)
    P = P0 * (T / T0) ** exponent
    rho = P / (R * T)
    return T, P, rho

def speed_of_sound(T: float, gamma: float = 1.4, R_gas: float = R) -> float:
    """a = sqrt(gamma * R * T)"""
    return math.sqrt(gamma * R_gas * T)
