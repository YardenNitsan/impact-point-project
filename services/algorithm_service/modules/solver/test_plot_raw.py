"""
Professional test harness for the 3DOF impact simulation.

This script runs a deterministic simulation scenario and produces
diagnostic plots for trajectory analysis.

Outputs
-------
1) Ground range vs altitude
2) Speed vs time
3) Pitch angle vs time

This script is intended for:

• Model validation
• Numerical sanity checking
• Visual inspection of dynamics
• Regression testing

Coordinate model
----------------
Range is computed using a local tangent-plane approximation
(equirectangular projection) relative to the launch point.
"""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt

from modules.impact.simulated_impact import simulate_impact


# ============================================================
# Geographic helpers
# ============================================================

EARTH_RADIUS_M = 6_371_000.0


def horizontal_distance_m(
    lat0: float,
    lon0: float,
    lat1: float,
    lon1: float,
) -> float:
    """
    Approximate horizontal distance between two nearby geographic points.

    Uses an equirectangular approximation suitable for short ranges.
    """

    lat0_rad = math.radians(lat0)
    lat1_rad = math.radians(lat1)

    dlat = lat1_rad - lat0_rad
    dlon = math.radians(lon1 - lon0)

    lat_mean = 0.5 * (lat0_rad + lat1_rad)

    x = EARTH_RADIUS_M * dlon * math.cos(lat_mean)
    y = EARTH_RADIUS_M * dlat

    return math.hypot(x, y)


# ============================================================
# Scenario definition
# ============================================================

def build_test_scenario() -> dict:
    """
    Return a deterministic test scenario.

    Launch point: Tel Aviv region
    """

    return {
        "alt": 100.0,
        "azimuth": 210.0,
        "elevation": 30.0,
        "lat": 32.0853,
        "lon": 34.7818,
        "mass": 0.6,
        "initialSpeed": 220.0,
    }


# ============================================================
# Data extraction
# ============================================================

def extract_time_series(result: dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Convert simulation output into plotting arrays.

    Returns:
        range_m, altitude_m, speed_mps, theta_rad
    """

    trajectory = result["trajectory"]
    lat0 = trajectory[0]["lat"]
    lon0 = trajectory[0]["lon"]

    range_m: List[float] = []
    altitude_m: List[float] = []
    speed_mps: List[float] = []
    theta_rad: List[float] = []

    for point in trajectory:
        r = horizontal_distance_m(lat0, lon0, point["lat"], point["lon"])
        v = math.hypot(point["vx"], point["vz"])

        range_m.append(r)
        altitude_m.append(point["alt"])
        speed_mps.append(v)
        theta_rad.append(point["theta"])

    return range_m, altitude_m, speed_mps, theta_rad


def build_time_axis(result: dict, sample_count: int) -> List[float]:
    """
    Construct a time vector based on physical simulation time.
    """

    total_time = float(result["physical_time"])
    dt = total_time / max(sample_count - 1, 1)

    return [i * dt for i in range(sample_count)]


# ============================================================
# Plotting
# ============================================================

def plot_results(
    range_m: List[float],
    altitude_m: List[float],
    time_s: List[float],
    speed_mps: List[float],
    theta_rad: List[float],
) -> None:
    """
    Generate diagnostic plots.
    """

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(range_m, altitude_m)
    plt.title("Trajectory (Range vs Altitude)")
    plt.xlabel("Range [m]")
    plt.ylabel("Altitude [m]")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time_s, speed_mps)
    plt.title("Speed vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time_s, theta_rad)
    plt.title("Pitch Angle vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [rad]")
    plt.grid()

    plt.tight_layout()
    plt.show()


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    """
    Run the test scenario and visualize results.
    """

    scenario = build_test_scenario()

    result = simulate_impact(scenario)

    range_m, altitude_m, speed_mps, theta_rad = extract_time_series(result)

    time_s = build_time_axis(result, len(range_m))

    plot_results(range_m, altitude_m, time_s, speed_mps, theta_rad)


if __name__ == "__main__":
    main()
