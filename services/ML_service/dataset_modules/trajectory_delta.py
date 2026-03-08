def trajectory_to_deltas(trajectory):
    """
    Convert trajectory absolute points to delta representation.
    """

    if not trajectory:
        return None, []

    start = trajectory[0]

    deltas = []

    prev = start

    for p in trajectory[1:]:

        deltas.append({
            "dlat": p["lat"] - prev["lat"],
            "dlon": p["lon"] - prev["lon"],
            "dalt": p["alt"] - prev["alt"]
        })

        prev = p

    return start, deltas

def build_dataset_row(sample, simulation_result):

    trajectory = simulation_result["trajectory"]

    start, deltas = trajectory_to_deltas(trajectory)

    return {

        # input parameters
        "latitude": sample["latitude"],
        "longitude": sample["longitude"],
        "altitude": sample["altitude"],
        "speed": sample["speed"],

        "sin_az": sample["sin_az"],
        "cos_az": sample["cos_az"],
        "sin_el": sample["sin_el"],
        "cos_el": sample["cos_el"],

        "mass": sample["mass"],
        "T0": sample["T0"],
        "P0": sample["P0"],
        "wind_x": sample["wind_x"],
        "wind_z": sample["wind_z"],

        # trajectory representation
        "start": start,
        "deltas": deltas,

        # optional
        "impact": simulation_result["impact"],
        "flight_time": simulation_result["physical_time"]
    }