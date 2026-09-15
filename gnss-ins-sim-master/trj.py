import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# -----------------------------
# 1️⃣ LOAD CSV TRAJECTORY
# -----------------------------
csv_file = "trajectory.csv"

with open(csv_file, 'r') as f:
    lines = f.readlines()

# Initial state
init_vals = list(map(float, lines[1].strip().split(',')))
yaw_deg = init_vals[6]

cmd_df = pd.read_csv(csv_file, skiprows=3)

yaw = np.deg2rad(yaw_deg)

R = np.array([
    [np.cos(yaw), -np.sin(yaw)],
    [np.sin(yaw),  np.cos(yaw)]
])

x_csv, y_csv = [0], [0]

for _, row in cmd_df.iterrows():
    v_body = np.array([
        row['vx_body (m/s)'],
        row['vy_body (m/s)']
    ])
    dt = row['command duration (s)']

    v_global = R @ v_body

    x_csv.append(x_csv[-1] + v_global[0] * dt)
    y_csv.append(y_csv[-1] + v_global[1] * dt)


# -----------------------------
# 2️⃣ LOAD JSON TRAJECTORY
# -----------------------------
json_file = "test_obs2.json"

with open(json_file) as f:
    data = json.load(f)

traj = data["trajectory"]

lat0 = traj["initPosition"]["latitude"]
lon0 = traj["initPosition"]["longitude"]

speed = traj["initVelocity"]["speed"]
course = np.deg2rad(traj["initVelocity"]["course"])

vx = speed * np.cos(course)
vy = speed * np.sin(course)

x, y = [0], [0]

dt_sim = 1.0  # 1 sec resolution

for segment in traj["trajectoryList"]:

    seg_type = segment["type"]

    # -----------------
    if seg_type == "Const":
        T = segment["time"]
        steps = int(T / dt_sim)

        for _ in range(steps):
            x.append(x[-1] + vx * dt_sim)
            y.append(y[-1] + vy * dt_sim)

    # -----------------
    elif seg_type == "ConstAcc":
        T = segment.get("time", 0)
        a = segment.get("acceleration", 0)

        steps = int(T / dt_sim)

        for _ in range(steps):
            v = np.sqrt(vx**2 + vy**2)
            v += a * dt_sim

            direction = np.arctan2(vy, vx)
            vx = v * np.cos(direction)
            vy = v * np.sin(direction)

            x.append(x[-1] + vx * dt_sim)
            y.append(y[-1] + vy * dt_sim)

    # -----------------
    elif seg_type == "HorizontalTurn":
        angle = np.deg2rad(segment["angle"])

        if "time" in segment:
            T = segment["time"]
            omega = angle / T
        else:
            radius = segment["radius"]
            v = np.sqrt(vx**2 + vy**2)
            omega = v / radius * np.sign(angle)

        steps = int(abs(angle) / abs(omega * dt_sim))

        for _ in range(steps):
            theta = np.arctan2(vy, vx)
            theta += omega * dt_sim

            v = np.sqrt(vx**2 + vy**2)
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)

            x.append(x[-1] + vx * dt_sim)
            y.append(y[-1] + vy * dt_sim)

    # -----------------
    else:
        # Ignore vertical/jerk for XY plot
        continue


# -----------------------------
# 3️⃣ PLOT BOTH
# -----------------------------
plt.figure()

plt.plot(x_csv, y_csv, marker='o', label="CSV Trajectory")
plt.plot(x, y, label="JSON Trajectory")

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory Comparison")
plt.legend()
plt.grid()
plt.axis('equal')

plt.show()