import math
import matplotlib.pyplot as plt
import numpy as np

from services.algorithm_service.modules.state.state import State3DOF
from services.algorithm_service.modules.solver.run_simulation import run_simulation
from services.algorithm_service.modules.atmosphere.environment import get_sea_level_environment
from services.algorithm_service.modules.aerodynamics.aero_tables import default_demo_table
from services.algorithm_service.modules.aerodynamics.aerodynamics import AeroRef


def main():
    # ===== same as your simulate_impact() =====
    alt = 1000.0
    elevation_deg = 45.0
    lat0 = 32.0
    lon0 = 34.8
    mass = 10.0
    V0 = 1500.0

    elevation = math.radians(elevation_deg)
    vx0 = V0 * math.cos(elevation)
    vz0 = V0 * math.sin(elevation)

    state0 = State3DOF(
        x=0.0,
        z=alt,
        vx=vx0,
        vz=vz0,
        theta=elevation,
        q=0.0
    )

    T0, P0 = get_sea_level_environment(lat0, lon0)

    dt = 0.01
    traj = run_simulation(
        state0=state0,
        dt=dt,
        max_time=300.0,
        mass=mass,
        Iyy=2.0,
        g=9.81,
        T0=T0,
        P0=P0,
        aero_ref=AeroRef(Sref=0.05, lref=0.30),
        aero_table=default_demo_table(),
        lcg=0.02
    )

    # ===== RAW time series (NO downsample) =====
    t = [i * dt for i in range(len(traj))]
    z = [s.z for s in traj]

    plt.figure()
    plt.plot(t, z)
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude z (m)")
    plt.title("RAW Altitude vs Time (no downsample)")
    plt.grid(True)
    plt.show()

    print("Final altitude:", traj[-1].z)
    print("Final vz:", traj[-1].vz)
    print("Total time:", t[-1])
    print("Points:", len(traj))


if __name__ == "__main__":
    main()
