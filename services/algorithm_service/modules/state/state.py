from __future__ import annotations

"""
Core 3DOF state representation.

This module defines the immutable state vector used throughout the
3DOF simulation engine.

The design goal is to provide:

• A clear physical contract between solver modules
• Immutable state objects (functional integration style)
• Explicit unit documentation
• Structural symmetry between state and derivatives

State vector definition
-----------------------

The 3DOF state is defined in a planar inertial frame:

    x     : downrange position [m]
    z     : altitude above ground [m]
    vx    : inertial velocity along x [m/s]
    vz    : inertial velocity along z [m/s]
    theta : pitch angle [rad]
    q     : pitch rate [rad/s]

All modules in the simulation must treat this structure as the
canonical state representation.

Immutability
------------

State objects are frozen dataclasses to guarantee that integration
steps always produce new states rather than mutating existing ones.
This improves numerical traceability and debugging.
"""

from dataclasses import dataclass
from typing import Tuple


# ============================================================
# 3DOF state definition
# ============================================================

@dataclass(frozen=True, slots=True)
class State3DOF:
    """
    Physical state of the planar 3DOF system.

    Attributes
    ----------
    x : float
        Downrange position [m]
    z : float
        Altitude above ground [m]
    vx : float
        Inertial velocity along x-axis [m/s]
    vz : float
        Inertial velocity along z-axis [m/s]
    theta : float
        Pitch angle [rad]
    q : float
        Pitch rate [rad/s]
    """

    x: float
    z: float
    vx: float
    vz: float
    theta: float
    q: float

    # --------------------------------------------------------
    # Utility helpers (numerical convenience)
    # --------------------------------------------------------

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return the state as a flat tuple.

        Useful for debugging, logging, or numerical adapters.
        """
        return self.x, self.z, self.vx, self.vz, self.theta, self.q


# ============================================================
# State derivatives
# ============================================================

@dataclass(frozen=True, slots=True)
class StateDerivatives3DOF:
    """
    Time-derivatives of the 3DOF state.

    Structure mirrors State3DOF exactly:

        dx/dt     = vx
        dz/dt     = vz
        dvx/dt    = ax
        dvz/dt    = az
        dtheta/dt = q
        dq/dt     = pitch acceleration

    This symmetry enables clean integrator implementations.
    """

    x: float
    z: float
    vx: float
    vz: float
    theta: float
    q: float

    # --------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """Return derivatives as a flat tuple."""
        return self.x, self.z, self.vx, self.vz, self.theta, self.q
