import math
from typing import Dict, List

from modules.state.state import State3DOF
from modules.solver.run_simulation import run_simulation
from modules.impact.impact import compute_impact_from_trajectory

from modules.atmosphere.environment import get_sea_level_environment
from modules.aerodynamics.aero_tables import default_demo_table
from modules.aerodynamics.aerodynamics import AeroRef


EARTH_RADIUS = 6371000.0  # meters


def meters_to_latlon(dx: float, dy: float, lat0: float, lon0: float):
    """
    Convert local ENU displacement (meters) to lat/lon.
    dx = east, dy = north
    """
    dlat = (dy / EARTH_RADIUS) * (180.0 / math.pi)
    dlon = (dx / (EARTH_RADIUS * math.cos(math.radians(lat0)))) * (180.0 / math.pi)
    return lat0 + dlat, lon0 + dlon

def downsample_distance(path: List[Dict], min_dist: float = 5.0) -> List[Dict]:
    """
    Reduce trajectory points by keeping only points that are
    at least min_dist meters apart.
    """

    if len(path) < 2:
        return path

    filtered = [path[0]]

    for p in path[1:]:
        last = filtered[-1]

        dx = (p["lon"] - last["lon"]) * 111000
        dy = (p["lat"] - last["lat"]) * 111000

        if math.hypot(dx, dy) >= min_dist:
            filtered.append(p)

    # always keep final impact point
    if filtered[-1] != path[-1]:
        filtered.append(path[-1])

    return filtered

def adaptive_downsample(path: List[Dict], target_points: int = 500) -> List[Dict]:
    """
    Automatically choose a distance threshold to keep
    roughly target_points in the trajectory.
    """

    if len(path) <= target_points:
        return path

    start = path[0]
    end = path[-1]

    dx = (end["lon"] - start["lon"]) * 111000
    dy = (end["lat"] - start["lat"]) * 111000

    total_range = math.hypot(dx, dy)

    min_dist = total_range / target_points

    # safety clamp
    min_dist = max(5.0, min_dist)

    return downsample_distance(path, min_dist)


def simulate_impact(initial_data: Dict) -> Dict:
    # ----------------------------
    # 1. unpack input
    # ----------------------------
    alt = float(initial_data["alt"])
    azimuth = math.radians(initial_data["azimuth"])
    elevation = math.radians(initial_data["elevation"])
    lat0 = float(initial_data["lat"])
    lon0 = float(initial_data["lon"])
    mass = float(initial_data["mass"])
    V0 = float(initial_data["initialSpeed"])

    # ----------------------------
    # 2. initial velocity
    # ----------------------------
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

    # ----------------------------
    # 3. environment
    # ----------------------------
    T0, P0 = get_sea_level_environment(lat0, lon0)

    # ----------------------------
    # 4. run simulation
    # ----------------------------
    dt = 0.01  # [sec] physical time step

    trajectory = run_simulation(
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

    # ----------------------------
    # 5. impact point
    # ----------------------------
    impact = compute_impact_from_trajectory(trajectory)
    if impact is None:
        raise RuntimeError("No ground impact detected")

    s_imp = impact.state_at_impact

    # ----------------------------
    # 6. convert trajectory to lat/lon
    # ----------------------------
    path: List[Dict] = []

    for s in trajectory:
        dx = s.x * math.sin(azimuth)
        dy = s.x * math.cos(azimuth)

        lat, lon = meters_to_latlon(dx, dy, lat0, lon0)

        path.append({
            "lat": lat,
            "lon": lon,
            "alt": s.z,
            "vx": s.vx,
            "vz": s.vz,
            "theta": s.theta
        })

    # ----------------------------
    # Replace last point with exact impact
    # ----------------------------
    dx_imp = s_imp.x * math.sin(azimuth)
    dy_imp = s_imp.x * math.cos(azimuth)
    lat_imp, lon_imp = meters_to_latlon(dx_imp, dy_imp, lat0, lon0)

    path.pop()

    path.append({
        "lat": lat_imp,
        "lon": lon_imp,
        "alt": 0.0,
        "vx": s_imp.vx,
        "vz": s_imp.vz,
        "theta": s_imp.theta
    })

    # downsample trajectory before returning
    path = adaptive_downsample(path, target_points=500)

    # ----------------------------
    # 8. physical simulation time
    # ----------------------------
    physical_time = dt * (len(trajectory) - 1)  # keep this
    raw_points = len(trajectory)  # add this

    return {
        "impact": {
            "lat": lat_imp,
            "lon": lon_imp,
            "alt": 0.0,
            "vx": s_imp.vx,
            "vz": s_imp.vz,
            "theta": s_imp.theta
        },
        "trajectory": path,
        "physical_time": physical_time,
        "points_count": len(path),
        "raw_points_count": raw_points
    }

if __name__ == "__main__":
    initial_data = {
        "alt": 1000.0,
        "azimuth": 90.0,        #east
        "elevation": 10.0,      #degrees
        "lat": 32.0,
        "lon": 34.8,
        "mass": 10.0,
        "initialSpeed": 300.0
    }

    result = simulate_impact(initial_data)

    print("\nIMPACT POINT")
    print(result["impact"])

    print("\nFIRST 5 TRAJECTORY POINTS")
    for p in result["trajectory"][:5]:
        print(p)

    print("\nLAST 5 TRAJECTORY POINTS")
    for p in result["trajectory"][-5:]:
        print(p)

    print("\nSIMULATION SUMMARY")
    print(f"Physical flight time: {result['physical_time']:.2f} seconds")
    print(f"Trajectory points: {result['points_count']}")





