# -*- coding: utf-8 -*-
# Filename: imu_simulator.py

"""
IMU Data Generator Executable
Generates simulated IMU data from GNSS Simulator JSON configuration files.
Uses signalsim_to_gnss_ins_sim.py for JSON-to-CSV conversion.
Outputs time in UTC seconds format in time.csv and gps_time.csv only.
"""

import os
import sys
import math
import argparse
import json
import csv
import tempfile
import glob
from datetime import datetime, timedelta
from pathlib import Path

# Handle PyInstaller resource path
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# Import the external converter
from signalsim_to_gnss_ins_sim import convert_signalsim_to_gnss_motion

D2R = math.pi/180

def extract_utc_start_time(json_data):
    """
    Extract UTC start time from SignalSim JSON configuration.
    """
    try:
        time_config = json_data.get('time', {})
        if time_config.get('type') == 'UTC':
            year = int(time_config['year'])
            month = int(time_config['month'])
            day = int(time_config['day'])
            hour = int(time_config['hour'])
            minute = int(time_config['minute'])
            second = int(time_config['second'])
            return datetime(year, month, day, hour, minute, second)
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not parse UTC time from JSON: {e}")
    return None

def modify_time_to_utc_seconds(directory, utc_start_time):
    """
    Modify ONLY time.csv and gps_time.csv to use UTC time in seconds format.
    Converts UTC start time to seconds since Unix epoch and adds sample intervals.
    Does NOT modify other CSV files.
    """
    import numpy as np
    
    # Find time.csv to determine the correct subdirectory
    time_files = glob.glob(os.path.join(directory, '**', 'time.csv'), recursive=True)
    
    if not time_files:
        print("Warning: Could not find time.csv to modify")
        return
    
    for time_file in time_files:
        actual_dir = os.path.dirname(time_file)
        print(f"\nModifying time files in: {actual_dir}")
        
        # Convert UTC start time to seconds since Unix epoch
        utc_epoch = datetime(1970, 1, 1)
        start_time_seconds = (utc_start_time - utc_epoch).total_seconds()
        
        # ============ MODIFY time.csv ============
        try:
            # Read original time values
            time_data = np.loadtxt(time_file, delimiter=',', skiprows=1, dtype=str)
            if time_data.ndim == 0:
                time_data = np.array([time_data])
            relative_times = time_data.astype(float)
            
            # Convert relative times to absolute UTC in seconds
            # UTC_time_in_seconds = start_time_seconds + relative_time
            utc_seconds = start_time_seconds + relative_times
            
            # Overwrite time.csv with UTC in seconds
            np.savetxt(time_file, utc_seconds, delimiter=',', 
                      header='time(UTC_seconds)', comments='', fmt='%.6f')
            print(f"   time.csv -> UTC seconds")
            print(f"   Start: {utc_seconds[0]:.3f} ({utc_start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
            print(f"   End:   {utc_seconds[-1]:.3f}")
            print(f"   Step:  {relative_times[1] - relative_times[0]:.4f} seconds")
            
        except Exception as e:
            print(f"  Error modifying time.csv: {e}")
        
        # ============ MODIFY gps_time.csv (if exists) ============
        gps_time_file = os.path.join(actual_dir, 'gps_time.csv')
        if os.path.exists(gps_time_file):
            try:
                gps_data = np.loadtxt(gps_time_file, delimiter=',', skiprows=1, dtype=str)
                if gps_data.ndim == 0:
                    gps_data = np.array([gps_data])
                gps_relative = gps_data.astype(float)
                
                # Convert GPS relative times to absolute UTC in seconds
                gps_utc_seconds = start_time_seconds + gps_relative
                
                # Overwrite gps_time.csv with UTC in seconds
                np.savetxt(gps_time_file, gps_utc_seconds, delimiter=',',
                          header='gps_time(UTC_seconds)', comments='', fmt='%.6f')
                print(f"   gps_time.csv -> UTC seconds")
                print(f"   Start: {gps_utc_seconds[0]:.3f}")
                print(f"   Step:  {gps_relative[1] - gps_relative[0]:.4f} seconds")
                
            except Exception as e:
                print(f"  Error modifying gps_time.csv: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='GNSS-INS-SIM: Generate simulated IMU data from GNSS Simulator JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -t gnss_config.json
  %(prog)s -t gnss_config.json -o ./imu_output
  %(prog)s -t gnss_config.json --fs 400 --no-plot
        """
    )
    
    # Required arguments
    parser.add_argument('-t', '--trajectory', 
                       required=True,
                       help='Path to GNSS Simulator JSON configuration file')
    
    # Optional arguments
    parser.add_argument('-o', '--output', 
                       default='./imu_data/',
                       help='Output directory for generated data (default: ./imu_data/)')
    
    parser.add_argument('-a', '--imu-accuracy', 
                       default='jz105-rlg',
                       choices=['low-accuracy', 'mid-accuracy', 'high-accuracy', 'jz105-rlg'],
                       help='IMU accuracy model (default: jz105-rlg)')
    
    parser.add_argument('--fs', 
                       type=float, 
                       default=200.0,
                       help='IMU sample frequency in Hz (default: 200.0)')
    
    parser.add_argument('--fs-gps', 
                       type=float, 
                       default=10.0,
                       help='GPS sample frequency in Hz (default: 10.0)')
    
    parser.add_argument('--no-plot', 
                       action='store_true',
                       help='Disable plotting')
    
    parser.add_argument('--ref-frame', 
                       type=int,
                       default=0,
                       choices=[0, 1],
                       help='Reference frame: 0=NED, 1=Virtual inertial frame (default: 0)')
    
    parser.add_argument('--axis', 
                       type=int,
                       default=6,
                       choices=[6, 9],
                       help='IMU axis: 6=gyro+accel, 9=gyro+accel+mag (default: 6)')
    
    parser.add_argument('--gps', 
                       action='store_true',
                       help='Generate GPS data')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("                    GNSS-INS-SIM: IMU Data Generator")
    print("=" * 70)
    print(f"Config file:      {args.trajectory}")
    print(f"Output directory: {args.output}")
    print(f"IMU accuracy:     {args.imu_accuracy}")
    print(f"IMU sample rate:  {args.fs} Hz")
    print(f"GPS sample rate:  {args.fs_gps} Hz")
    print(f"Reference frame:  {'NED' if args.ref_frame == 0 else 'Virtual Inertial'}")
    print("=" * 70)
    
    # Validate config file exists
    if not os.path.exists(args.trajectory):
        print(f"\nERROR: Config file not found: {args.trajectory}")
        sys.exit(1)
    
    # Read and parse JSON file
    print("\nReading JSON configuration...")
    try:
        with open(args.trajectory, 'r') as f:
            json_data = json.load(f)
        print("JSON file loaded successfully")
        
        # Extract UTC start time
        utc_start_time = extract_utc_start_time(json_data)
        if utc_start_time:
            print(f"UTC start time: {utc_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("ℹ No UTC time in JSON, using relative timestamps")
        
        # Convert JSON to CSV using external converter
        print("Converting JSON to motion definition format...")
        csv_rows, audit_info = convert_signalsim_to_gnss_motion(
            json_data,
            trajectory_name=None,
            gps_visible=1
        )
        print("Conversion complete")
        
        # Write CSV to temporary file
        temp_csv_path = None
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_file:
            writer = csv.writer(temp_file)
            writer.writerows(csv_rows)
            temp_csv_path = temp_file.name
        
        # Save trajectory CSV to output directory
        os.makedirs(args.output, exist_ok=True)
        trajectory_csv_path = os.path.join(args.output, 'trajectory.csv')
        with open(trajectory_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Trajectory CSV saved: {trajectory_csv_path}")
        
        try:
            # Setup simulation
            fs_mag = None if args.axis == 6 else args.fs
            
            print("\nInitializing IMU model...")
            imu = imu_model.IMU(
                accuracy=args.imu_accuracy, 
                axis=args.axis, 
                gps=args.gps
            )
            
            print("Creating simulation...")
            sim = ins_sim.Sim(
                [args.fs, args.fs_gps, fs_mag],
                temp_csv_path,
                ref_frame=args.ref_frame,
                imu=imu,
                mode=None,
                env=None,
                algorithm=None
            )
            
            print("Running simulation...")
            sim.run(1)
            
            # Stats
            duration = sim.time[-1] if hasattr(sim, 'time') and len(sim.time) > 0 else 0
            num_samples = len(sim.time) if hasattr(sim, 'time') else 0
            
            print(f"\nSimulation complete!")
           
            
            # Save results
            print(f"\nSaving results to {args.output}...")
            sim.results(args.output)
            
            # Modify ONLY time.csv and gps_time.csv to use UTC in seconds
            if utc_start_time:
                modify_time_to_utc_seconds(args.output, utc_start_time)
            
            # Plot if requested
            if not args.no_plot:
                print("\nGenerating plots...")
                try:
                    import matplotlib.pyplot as plt
                    sim.plot(['ref_pos', 'gyro', 'accel'], opt={'ref_pos': '3d'})
                    plt.show(block=True)
                except Exception as e:
                    print(f"Warning: Could not generate plots: {e}")
            
            print("\n" + "=" * 70)
            print("Data generation complete!")
            
            # List generated files
            # print("\nGenerated files:")
            # actual_dirs = glob.glob(os.path.join(args.output, '*/'))
            # for actual_dir in actual_dirs:
            #     print(f"\nIn {actual_dir}:")
            #     for file in sorted(os.listdir(actual_dir)):
            #         file_path = os.path.join(actual_dir, file)
            #         if os.path.isfile(file_path):
            #             size = os.path.getsize(file_path) / 1024
            #             print(f"  - {file} ({size:.1f} KB)")
                    
        finally:
            try:
                if temp_csv_path:
                    os.unlink(temp_csv_path)
            except:
                pass
                
    except json.JSONDecodeError as e:
        print(f"\nERROR: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()