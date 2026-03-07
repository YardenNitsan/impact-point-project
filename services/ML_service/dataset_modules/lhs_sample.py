import numpy as np
import math


# ============================================================
# Parameter ranges (project assumptions)
# ============================================================

RANGES = {
    "latitude": (-90, 90),          # degrees
    "longitude": (-180, 180),       # degrees

    "altitude": (0, 20000),       # meters
    "speed": (1, 1200),         # m/s
    "azimuth": (0, 360),         # degrees
    "elevation": (-35, 85),      # degrees
    "mass": (1, 5000),            # kg

    "T0": (200, 320),            # Kelvin
    "P0": (5000, 110000),       # Pascal

    "wind_x": (-100, 100),         # m/s
    "wind_z": (-20, 20)          # m/s
}


# ============================================================
# Latin Hypercube Sampling
# ============================================================

def latin_hypercube_sampling(n_samples: int, dimensions: int):

    result = np.zeros((n_samples, dimensions))

    for i in range(dimensions):

        perm = np.random.permutation(n_samples)

        result[:, i] = (perm + np.random.rand(n_samples)) / n_samples

    return result


# ============================================================
# Build samples
# ============================================================

def generate_samples(n_samples: int):

    keys = list(RANGES.keys())
    dims = len(keys)

    lhs = latin_hypercube_sampling(n_samples, dims)

    samples = []

    for row in lhs:

        sample = {}

        for i, key in enumerate(keys):

            low, high = RANGES[key]

            value = low + row[i] * (high - low)

            sample[key] = float(value)

        # angles in radians
        az_rad = math.radians(sample["azimuth"])
        el_rad = math.radians(sample["elevation"])

        sample["sin_az"] = math.sin(az_rad)
        sample["cos_az"] = math.cos(az_rad)

        sample["sin_el"] = math.sin(el_rad)
        sample["cos_el"] = math.cos(el_rad)

        samples.append(sample)

    return samples

# ============================================================
# test run
# ============================================================

if __name__ == "__main__":

    samples = generate_samples(20)

    for i, s in enumerate(samples):
        print(f"\nSample {i+1}")
        for k, v in s.items():
            print(f"{k:12s}: {v:.4f}")