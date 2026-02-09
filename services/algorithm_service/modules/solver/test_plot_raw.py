import math
import matplotlib.pyplot as plt

from modules.impact.simulated_impact import simulate_impact

# -----------------------------
# Test scenario
# -----------------------------

initial_data = {
    "alt": 1000.0,
    "azimuth": 0.0,
    "elevation": 45.0,
    "lat": 32.0,
    "lon": 34.0,
    "mass": 10.0,
    "initialSpeed": 1000.0
}

result = simulate_impact(initial_data)

traj = result["trajectory"]

# -----------------------------
# Extract data
# -----------------------------

x = []
z = []
v = []
theta = []

for i, p in enumerate(traj):
    # approximate horizontal distance from start
    dx = (p["lon"] - traj[0]["lon"]) * 111000
    dy = (p["lat"] - traj[0]["lat"]) * 111000
    dist = math.hypot(dx, dy)

    x.append(dist)
    z.append(p["alt"])

    speed = math.hypot(p["vx"], p["vz"])
    v.append(speed)

    theta.append(p["theta"])

t = [i * 0.01 for i in range(len(traj))]

# -----------------------------
# Plot graphs
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x, z)
plt.title("Trajectory (Range vs Altitude)")
plt.xlabel("Range [m]")
plt.ylabel("Altitude [m]")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, v)
plt.title("Speed vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, theta)
plt.title("Pitch Angle vs Time")
plt.xlabel("Time [s]")
plt.ylabel("Theta [rad]")
plt.grid()

plt.tight_layout()
plt.show()
