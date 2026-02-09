# modules/atmosphere/wind.py

import math
import random
from typing import Optional, Tuple

# ============================================================
# Dryden model constants (as used in your reference)
# ============================================================

DRYDEN_A = 0.177
DRYDEN_B = 0.0027
DRYDEN_EXP_L = 1.2
DRYDEN_EXP_SIGMA = 0.4
DRYDEN_VERTICAL_SCALE = 0.1

MIN_ALTITUDE = 1.0   # [m]
MIN_AIRSPEED = 1.0   # [m/s]

GAUSS_MEAN = 0.0
GAUSS_STD = 1.0
TWO = 2.0


class DrydenWindModel:
    """Dryden wind turbulence model (discrete-time).

    Produces turbulence components (ut, vt, wt) in [m/s].

    Important: The stochastic (noise) term scales with sqrt(dt),
    not dt. The continuous-time form:
        d u = -(Va/Lu) u dt + sigma_u * sqrt(2 Va/Lu) dW
    Discretized with Euler-Maruyama:
        u_{k+1} = u_k + (-(Va/Lu) u_k) dt + sigma_u * sqrt(2 Va/Lu * dt) * N(0,1)
    """

    def __init__(self, Vw_ground: float, seed: Optional[int] = None):
        self.Vw_ground = float(Vw_ground)
        self._rng = random.Random(seed)

        self.ut = 0.0
        self.vt = 0.0
        self.wt = 0.0

    def step(self, z: float, Va: float, dt: float) -> Tuple[float, float]:
        h = max(float(z), MIN_ALTITUDE)
        base = DRYDEN_A + DRYDEN_B * h

        Lu = h / (base ** DRYDEN_EXP_L)
        Lv = Lu
        Lw = h

        sigma_u = self.Vw_ground / (base ** DRYDEN_EXP_SIGMA)
        sigma_v = sigma_u
        sigma_w = DRYDEN_VERTICAL_SCALE * self.Vw_ground

        eta_u = self._rng.gauss(GAUSS_MEAN, GAUSS_STD)
        eta_v = self._rng.gauss(GAUSS_MEAN, GAUSS_STD)
        eta_w = self._rng.gauss(GAUSS_MEAN, GAUSS_STD)

        Va_eff = max(float(Va), MIN_AIRSPEED)
        dt_eff = max(float(dt), 1e-6)

        # Euler-Maruyama update (correct sqrt(dt) scaling)
        self.ut += (-Va_eff * self.ut / Lu) * dt_eff + sigma_u * math.sqrt(TWO * Va_eff / Lu * dt_eff) * eta_u
        self.vt += (-Va_eff * self.vt / Lv) * dt_eff + sigma_v * math.sqrt(TWO * Va_eff / Lv * dt_eff) * eta_v
        self.wt += (-Va_eff * self.wt / Lw) * dt_eff + sigma_w * math.sqrt(TWO * Va_eff / Lw * dt_eff) * eta_w

        return self.ut, self.wt
