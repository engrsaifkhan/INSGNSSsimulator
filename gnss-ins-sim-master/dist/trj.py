import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ============================================
# FIND ACTUAL TRAJECTORY FILES
# ============================================
print("="*60)
print("SEARCHING FOR TRAJECTORY OUTPUT FILES")
print("="*60)

base_dir = Path("C:/Users/sabeen.murtaza/Desktop/gnss-ins-sim-master/dist")

# List all files in the directory
print(f"\nFiles in {base_dir}:")
all_files = list(base_dir.glob("*"))
for file in sorted(all_files):
    size = file.stat().st_size
    print(f"  {file.name:40s} {size:>10,d} bytes")

# Look for common output patterns
print("\n" + "="*60)
print("IDENTIFYING TRAJECTORY FILES")
print("="*60)

# GNSS-INS-SIM typically outputs:
# 1. Reference trajectory files (truth data)
# 2. Simulation output files
trajectory_files = {
    'ref_traj': None,
    'sim_output': None,
    'csv_trajectory': None
}

for file in all_files:
    name_lower = file.name.lower()
    if 'ref' in name_lower and file.suffix in ['.csv', '.txt']:
        trajectory_files['ref_traj'] = file
        print(f"Reference trajectory: {file.name}")
    elif 'allan' not in name_lower and file.suffix in ['.csv', '.txt'] and 'trajectory' in name_lower:
        trajectory_files['csv_trajectory'] = file
        print(f"CSV trajectory: {file.name}")
    elif file.suffix == '.json' and 'test_obs' not in name_lower:
        print(f"Other JSON: {file.name}")

# ============================================
# READ THE ACTUAL TRAJECTORY CSV (if it exists)
# ============================================
trajectory_csv = base_dir / "trajectory.csv"

if trajectory_csv.exists():
    print(f"\nReading {trajectory_csv}...")
    
    # Read the whole file first to understand structure
    with open(trajectory_csv, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    print(f"First line (header): {lines[0].strip()}")
    if len(lines) > 1:
        print(f"Second line: {lines[1].strip()}")
    
    # The file seems to have mixed data types
    # Let's try to extract just the trajectory points if they exist
    
    # Check if there's a separate section for trajectory points
    print("\nSearching for coordinate data in the file...")
    
    # Try reading with different approaches
    try:
        # Try to read numeric data only
        df = pd.read_csv(trajectory_csv, header=None, skiprows=1, 
                        on_bad_lines='skip', engine='python')
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with all NaN
        numeric_df = numeric_df.dropna(how='all')
        
        if len(numeric_df.columns) >= 3:
            print(f"Found {len(numeric_df)} rows with numeric data")
            print(f"Columns: {numeric_df.columns.tolist()}")
            
            # Assume first 3 columns are lat, lon, alt
            lla_points = numeric_df.iloc[:, :3].values
            print(f"First 5 points (lat, lon, alt):")
            print(lla_points[:5])
        else:
            print("Not enough numeric columns for trajectory data")
    except Exception as e:
        print(f"Error reading numeric data: {e}")

# ============================================
# CHECK IF THERE ARE OTHER OUTPUT FILES
# ============================================
print("\n" + "="*60)
print("LOOKING FOR SIMULATION OUTPUT IN OTHER LOCATIONS")
print("="*60)

# Check parent directories
parent_dir = base_dir.parent
print(f"\nFiles in {parent_dir}:")
parent_files = list(parent_dir.glob("*"))
for file in sorted(parent_files):
    if file.is_file():
        size = file.stat().st_size
        print(f"  {file.name:50s} {size:>10,d} bytes")

# Check for output subdirectories
print("\nChecking for output directories...")
for item in parent_dir.iterdir():
    if item.is_dir() and item.name != 'dist':
        print(f"\n  Contents of {item.name}/:")
        sub_files = list(item.glob("*"))
        for sf in sorted(sub_files)[:20]:  # Show first 20 files
            print(f"    {sf.name}")

# ============================================
# VISUALIZE THE JSON TRAJECTORY DEFINITION
# ============================================
print("\n" + "="*60)
print("VISUALIZING TRAJECTORY DEFINITION FROM JSON")
print("="*60)

with open("C:/Users/sabeen.murtaza/Desktop/gnss-ins-sim-master/dist/test_obs2.json") as f:
    config = json.load(f)

# Extract trajectory segments
segments = config['trajectory']['trajectoryList']
init_pos = config['trajectory']['initPosition']
init_vel = config['trajectory']['initVelocity']

# Create a timeline visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Trajectory timeline
ax1 = axes[0]
current_time = 0
colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))

for i, seg in enumerate(segments):
    seg_type = seg.get('type', 'Unknown')
    seg_time = seg.get('time', 0)
    
    if seg_time > 0:
        ax1.barh(0, seg_time, left=current_time, height=0.5, 
                color=colors[i], alpha=0.8, edgecolor='black')
        ax1.text(current_time + seg_time/2, 0, f"{seg_type}\n{seg_time}s", 
                ha='center', va='center', fontsize=8)
    
    current_time += seg_time

ax1.set_xlabel('Time (seconds)')
ax1.set_yticks([])
ax1.set_title('Trajectory Timeline')
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Acceleration/Jerk profile
ax2 = axes[1]
current_time = 0
accel_values = []
time_points = []

for seg in segments:
    seg_type = seg.get('type', 'Unknown')
    seg_time = seg.get('time', 0)
    
    accel = 0
    if 'acceleration' in seg:
        accel = seg['acceleration']
    elif seg_type == 'Jerk':
        accel = seg.get('acceleration', 0)
    
    accel_values.append(accel)
    time_points.append(current_time)
    current_time += seg_time

time_points.append(current_time)
ax2.step(time_points, accel_values + [accel_values[-1]], 
        where='post', linewidth=2, color='red')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Acceleration (m/s²)')
ax2.set_title('Acceleration Profile')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Plot 3: Initial conditions
ax3 = axes[2]
ax3.axis('off')
info_text = f"""
INITIAL CONDITIONS:
------------------
Position (LLA):
  Latitude:  {init_pos.get('latitude', 'N/A')}°
  Longitude: {init_pos.get('longitude', 'N/A')}°
  Altitude:  {init_pos.get('altitude', 'N/A')} m

Velocity (Body):
  Vx: {init_vel.get('vx', 'N/A')} m/s
  Vy: {init_vel.get('vy', 'N/A')} m/s
  Vz: {init_vel.get('vz', 'N/A')} m/s

TRAJECTORY SUMMARY:
------------------
Total segments: {len(segments)}
Total duration: {current_time:.1f} seconds ({current_time/60:.1f} minutes)
"""

# Add command breakdown
info_text += "\nCOMMAND BREAKDOWN:\n"
cmd_counts = {}
for seg in segments:
    cmd_type = seg.get('type', 'Unknown')
    cmd_counts[cmd_type] = cmd_counts.get(cmd_type, 0) + 1

for cmd_type, count in cmd_counts.items():
    total_cmd_time = sum(s.get('time', 0) for s in segments if s.get('type') == cmd_type)
    info_text += f"  {cmd_type}: {count} segments, {total_cmd_time:.1f}s total\n"

ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('GNSS-INS-SIM Trajectory Configuration', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# SUGGESTIONS FOR GETTING ACTUAL TRAJECTORY
# ============================================
print("\n" + "="*60)
print("HOW TO GET ACTUAL TRAJECTORY DATA")
print("="*60)
print("""
The JSON file defines HOW to generate the trajectory, but doesn't contain
the actual computed points. The CSV file 'trajectory.csv' in the dist folder
appears to contain just initial conditions and commands.

To get the actual trajectory points, you need to:

1. Check if GNSS-INS-SIM generated output files:
   - Look for files like 'ref.bin', 'ref.csv', or 'truth.csv'
   - These contain the actual simulated positions

2. Run the simulation to generate trajectory points:
   python gnss-ins-sim.py --config=test_obs2.json

3. Check for output in directories like:
   - /output/
   - /data/
   - /results/

The simulation should generate files with actual position data
at each time step that can be plotted and compared.
""")

# Check for reference data in the demo datasets
demo_dir = parent_dir / "demo" / "imu_data"
if demo_dir.exists():
    print(f"\nFound demo data directory: {demo_dir}")
    demo_files = list(demo_dir.glob("*"))
    for df in demo_files:
        print(f"  {df.name}")