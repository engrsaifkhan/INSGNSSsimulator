# -*- coding: utf-8 -*-
# Filename: imu_simulator_direct_teensy.py

"""
Generate simulated IMU data entirely in memory and send it directly to Teensy.

No trajectory.csv, accel.csv, gyro.csv, time.csv, or result directory is made.
The simulation still completes first because gnss-ins-sim generates complete
NumPy arrays inside Sim.run(). The arrays are then streamed directly from RAM.
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim
from signalsim_to_gnss_ins_sim import convert_signalsim_to_gnss_motion

try:
    from demo_algorithms import free_integration
except ModuleNotFoundError:
    package_root = os.path.dirname(os.path.dirname(os.path.dirname(ins_sim.__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from demo_algorithms import free_integration

D2R = math.pi / 180.0


def extract_utc_start_time(json_data):
    """Extract the UTC start time from a SignalSim JSON configuration."""
    try:
        time_config = json_data.get("time", {})
        if time_config.get("type") == "UTC":
            return datetime(
                int(time_config["year"]),
                int(time_config["month"]),
                int(time_config["day"]),
                int(time_config["hour"]),
                int(time_config["minute"]),
                int(time_config["second"]),
            )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Warning: Could not parse UTC start time: {exc}")

    return None


def csv_rows_to_memory_string(csv_rows):
    """Convert converter output to a CSV-formatted string held only in RAM."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(csv_rows)
    return buffer.getvalue()


def initial_state_from_motion_rows(csv_rows):
    """Return the first motion-definition row in estimator units."""
    initial_state = np.asarray(csv_rows[1], dtype=np.float64)
    initial_state[[0, 1, 6, 7, 8]] *= D2R
    return initial_state


def plot_reference_and_estimated_trajectory(sim, plt, tca_csv_path=""):
    """Plot PocketSDR GNSS and TCA positions on shared ENU axes."""
    gnss_positions = []
    tca_positions = []
    if tca_csv_path and os.path.isfile(tca_csv_path):
        with open(tca_csv_path, newline="", encoding="utf-8") as tca_file:
            for row in csv.DictReader(tca_file):
                try:
                    gnss_positions.append([
                        float(row["gnss_lat_deg"]) * D2R,
                        float(row["gnss_lon_deg"]) * D2R,
                        float(row["gnss_h_m"]),
                    ])
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    tca_positions.append([
                        float(row["tca_lat_deg"]) * D2R,
                        float(row["tca_lon_deg"]) * D2R,
                        float(row["tca_h_m"]),
                    ])
                except (KeyError, TypeError, ValueError):
                    continue
    if not gnss_positions:
        raise RuntimeError("PocketSDR GNSS PVT trajectory is empty")

    reference = np.asarray(gnss_positions, dtype=np.float64)
    origin = reference[0]
    earth_radius_m = 6378137.0
    cos_latitude = np.cos(origin[0])

    def to_local_enu(position):
        north = (position[:, 0] - origin[0]) * earth_radius_m
        east = (position[:, 1] - origin[1]) * earth_radius_m * cos_latitude
        up = position[:, 2] - origin[2]
        return east, north, up

    figure = plt.figure("Reference vs IMU-estimated trajectory")
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*to_local_enu(reference), label="PocketSDR GNSS", linewidth=2.0)
    if tca_positions:
        axis.plot(*to_local_enu(np.asarray(tca_positions)), label="TCA", linewidth=1.4)
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_zlabel("Up (m)")
    axis.set_title("PocketSDR GNSS vs TCA trajectory")
    axis.legend()
    axis.grid(True)
    return figure


def select_simulation_run(data, run_index=0):
    """
    Return one NumPy array from gnss-ins-sim data.

    Data such as accel and gyro are normally dictionaries keyed by simulation
    run number, while time is normally a direct NumPy array.
    """
    if isinstance(data, dict):
        if run_index in data:
            data = data[run_index]
        elif str(run_index) in data:
            data = data[str(run_index)]
        elif data:
            first_key = sorted(data.keys(), key=str)[0]
            print(
                f"Warning: simulation run {run_index} was not found; "
                f"using run {first_key}."
            )
            data = data[first_key]
        else:
            raise RuntimeError("The simulation returned an empty data dictionary")

    return np.asarray(data, dtype=np.float64)


def get_imu_arrays_from_simulation(sim, utc_start_time=None, run_index=0):
    """Read time, accelerometer, and gyroscope data directly from Sim memory."""
    result = sim.get_data(["time", "accel", "gyro"])
    if result is None or len(result) != 3:
        available = sim.get_names_of_available_data()
        raise RuntimeError(
            "Could not obtain time/accel/gyro from the simulation. "
            f"Available data: {available}"
        )

    timestamps = select_simulation_run(result[0], run_index).reshape(-1)
    accel = select_simulation_run(result[1], run_index)
    gyro = select_simulation_run(result[2], run_index)

    if accel.ndim == 1:
        if accel.size % 3 != 0:
            raise RuntimeError(f"Unexpected accelerometer shape: {accel.shape}")
        accel = accel.reshape((-1, 3))

    if gyro.ndim == 1:
        if gyro.size % 3 != 0:
            raise RuntimeError(f"Unexpected gyroscope shape: {gyro.shape}")
        gyro = gyro.reshape((-1, 3))

    if accel.ndim != 2 or accel.shape[1] < 3:
        raise RuntimeError(f"Unexpected accelerometer shape: {accel.shape}")
    if gyro.ndim != 2 or gyro.shape[1] < 3:
        raise RuntimeError(f"Unexpected gyroscope shape: {gyro.shape}")

    sample_count = min(len(timestamps), len(accel), len(gyro))
    if sample_count <= 0:
        raise RuntimeError("The simulation produced no synchronized IMU samples")

    timestamps = timestamps[:sample_count]
    accel = accel[:sample_count, :3]
    gyro = gyro[:sample_count, :3]

    if utc_start_time is not None:
        unix_epoch = datetime(1970, 1, 1)
        start_seconds = (utc_start_time - unix_epoch).total_seconds()
        timestamps = start_seconds + timestamps

    return timestamps, accel, gyro


def wait_for_exact_line(serial_connection, expected, timeout_seconds):
    """Wait for an exact newline-terminated Teensy response."""
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        response = serial_connection.readline().decode(
            "ascii", errors="replace"
        ).strip()

        if response:
            print(f"Teensy: {response}")

        if response == expected:
            return

    raise TimeoutError(f"Teensy did not return {expected!r}")


def stream_imu_arrays_to_teensy(
    timestamps,
    accel,
    gyro,
    serial_port,
    baud_rate,
    sample_rate_hz,
    wait_for_ready=True,
    ready_timeout_seconds=5.0,
    realtime=True,
):
    """
    Stream in-memory IMU arrays to Teensy.

    Packet format:
        IMU,index,time,ax,ay,az,gx,gy,gz\n
    Control messages:
        IMU_STREAM_BEGIN,count,sample_rate\n
        IMU_STREAM_END,count\n
    """
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "PySerial is required. Install it using: python -m pip install pyserial"
        ) from exc

    sample_count = min(len(timestamps), len(accel), len(gyro))
    if sample_count <= 0:
        raise RuntimeError("There are no IMU samples to transmit")

    print("\n" + "=" * 70)
    print("Direct IMU transmission from RAM to Teensy")
    print(f"Serial port: {serial_port}")
    print(f"Baud rate:   {baud_rate}")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Samples:     {sample_count}")
    print("=" * 70)

    serial_connection = serial.Serial(
        port=serial_port,
        baudrate=baud_rate,
        timeout=0.1,
        write_timeout=2.0,
    )

    try:
        # USB serial can take a short time to enumerate after opening.
        time.sleep(1.5)
        serial_connection.reset_input_buffer()
        serial_connection.reset_output_buffer()

        begin_packet = (
            f"IMU_STREAM_BEGIN,{sample_count},{sample_rate_hz:.6f}\n"
        )
        serial_connection.write(begin_packet.encode("ascii"))
        serial_connection.flush()

        if wait_for_ready:
            wait_for_exact_line(
                serial_connection,
                expected="READY",
                timeout_seconds=ready_timeout_seconds,
            )

        period_seconds = (
            1.0 / sample_rate_hz
            if realtime and sample_rate_hz > 0.0
            else 0.0
        )
        next_deadline = time.perf_counter()

        for index in range(sample_count):
            ax, ay, az = accel[index]
            gx, gy, gz = gyro[index]

            packet = (
                f"IMU,{index},{timestamps[index]:.6f},"
                f"{ax:.9f},{ay:.9f},{az:.9f},"
                f"{gx:.9f},{gy:.9f},{gz:.9f}\n"
            )
            serial_connection.write(packet.encode("ascii"))

            if realtime:
                next_deadline += period_seconds
                remaining = next_deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

            if (index + 1) % 1000 == 0 or index + 1 == sample_count:
                print(f"Sent {index + 1}/{sample_count} samples")

        serial_connection.write(
            f"IMU_STREAM_END,{sample_count}\n".encode("ascii")
        )
        serial_connection.flush()

        # Give Teensy a brief opportunity to return DONE,count.
        done_deadline = time.monotonic() + 2.0
        while time.monotonic() < done_deadline:
            response = serial_connection.readline().decode(
                "ascii", errors="replace"
            ).strip()
            if response:
                print(f"Teensy: {response}")
                if response.startswith("DONE,"):
                    break

        print("Direct IMU transmission completed")

    finally:
        serial_connection.close()



def stream_imu_arrays_to_console(
    timestamps,
    accel,
    gyro,
    sample_rate_hz,
    realtime=True,
):
    """Print the exact outgoing serial packets without opening a COM port."""
    sample_count = min(len(timestamps), len(accel), len(gyro))
    if sample_count <= 0:
        raise RuntimeError("There are no IMU samples to display")

    print("\n" + "=" * 70)
    print("CONSOLE-ONLY TRANSMISSION TEST (no Teensy connected)")
    print(f"Sample rate: {sample_rate_hz} Hz")
    print(f"Samples:     {sample_count}")
    print("=" * 70)

    begin_packet = f"IMU_STREAM_BEGIN,{sample_count},{sample_rate_hz:.6f}"
    print(f"TX > {begin_packet}", flush=True)

    period_seconds = (
        1.0 / sample_rate_hz
        if realtime and sample_rate_hz > 0.0
        else 0.0
    )
    next_deadline = time.perf_counter()

    for index in range(sample_count):
        ax, ay, az = accel[index]
        gx, gy, gz = gyro[index]

        packet = (
            f"IMU,{index},{timestamps[index]:.6f},"
            f"{ax:.9f},{ay:.9f},{az:.9f},"
            f"{gx:.9f},{gy:.9f},{gz:.9f}"
        )
        print(f"TX > {packet}", flush=True)

        if realtime:
            next_deadline += period_seconds
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)

    print(f"TX > IMU_STREAM_END,{sample_count}", flush=True)
    print("Console-only transmission test completed")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated IMU data in RAM and send it directly to Teensy"
        )
    )

    parser.add_argument(
        "-t",
        "--trajectory",
        required=True,
        help="Path to the SignalSim JSON configuration file",
    )
    parser.add_argument(
        "--serial-port",
        help="Teensy port, for example COM5 or /dev/ttyACM0",
    )
    parser.add_argument(
        "--tca-trajectory",
        default="",
        help="CSV file containing PocketSDR TCA/GNSS PVT samples for plotting",
    )
    parser.add_argument(
        "--console-only",
        action="store_true",
        help=(
            "Print the outgoing IMU packets in the terminal without opening "
            "a serial port"
        ),
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=1_000_000,
        help="Serial baud rate; default: 1000000",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=200.0,
        help="IMU simulation and transmission frequency in Hz; default: 200",
    )
    parser.add_argument(
        "--fs-gps",
        type=float,
        default=10.0,
        help="GPS sample frequency in Hz; default: 10",
    )
    parser.add_argument(
        "-a",
        "--imu-accuracy",
        default="jz105-rlg",
        choices=[
            "low-accuracy",
            "mid-accuracy",
            "high-accuracy",
            "jz105-rlg",
        ],
        help="IMU accuracy model; default: jz105-rlg",
    )
    parser.add_argument(
        "--ref-frame",
        type=int,
        default=0,
        choices=[0, 1],
        help="0=NED, 1=virtual inertial frame; default: 0",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=6,
        choices=[6, 9],
        help="6=gyro+accel, 9=gyro+accel+mag; default: 6",
    )
    parser.add_argument(
        "--gps",
        action="store_true",
        help="Generate GPS data internally",
    )
    parser.add_argument(
        "--no-wait-ready",
        action="store_true",
        help="Do not wait for the Teensy READY response",
    )
    parser.add_argument(
        "--fast-stream",
        action="store_true",
        help="Transmit as fast as possible instead of pacing at --fs",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display simulation plots",
    )

    args = parser.parse_args()

    if not args.console_only and not args.serial_port:
        parser.error(
            "--serial-port is required unless --console-only is used"
        )
    if args.fs <= 0.0:
        parser.error("--fs must be greater than zero")
    if args.fs_gps <= 0.0:
        parser.error("--fs-gps must be greater than zero")
    if not os.path.isfile(args.trajectory):
        parser.error(f"Configuration file not found: {args.trajectory}")

    print("=" * 70)
    print("GNSS-INS-SIM: Direct IMU-to-Teensy Stream")
    print("=" * 70)
    print(f"Configuration: {args.trajectory}")
    print(f"IMU model:     {args.imu_accuracy}")
    print(f"IMU rate:      {args.fs} Hz")
    if args.console_only:
        print("Output mode:    terminal preview")
        print("Teensy port:   not required")
    else:
        print("Output mode:    serial transmission")
        print(f"Teensy port:   {args.serial_port}")
        print(f"Serial baud:   {args.baud}")
    print("CSV output:    disabled")
    print("=" * 70)

    try:
        with open(args.trajectory, "r", encoding="utf-8") as config_file:
            json_data = json.load(config_file)

        utc_start_time = extract_utc_start_time(json_data)

        print("Converting SignalSim JSON to an in-memory motion definition...")
        csv_rows, _audit_info = convert_signalsim_to_gnss_motion(
            json_data,
            trajectory_name=None,
            gps_visible=1,
        )
        motion_definition = csv_rows_to_memory_string(csv_rows)

        estimator = free_integration.FreeIntegration(
            initial_state_from_motion_rows(csv_rows)
        )

        fs_mag = None if args.axis == 6 else args.fs
        imu = imu_model.IMU(
            accuracy=args.imu_accuracy,
            axis=args.axis,
            gps=args.gps,
        )

        sim = ins_sim.Sim(
            [args.fs, args.fs_gps, fs_mag],
            motion_definition,
            ref_frame=args.ref_frame,
            imu=imu,
            mode=None,
            env=None,
            algorithm=estimator,
        )

        print("Running simulation in memory...")
        sim.run(1)

        print("Reading accel, gyro, and time directly from simulation memory...")
        timestamps, accel, gyro = get_imu_arrays_from_simulation(
            sim,
            utc_start_time=utc_start_time,
            run_index=0,
        )

        print(
            f"Simulation generated {len(timestamps)} synchronized IMU samples"
        )

        if args.console_only:
            stream_imu_arrays_to_console(
                timestamps=timestamps,
                accel=accel,
                gyro=gyro,
                sample_rate_hz=args.fs,
                realtime=not args.fast_stream,
            )
        else:
            stream_imu_arrays_to_teensy(
                timestamps=timestamps,
                accel=accel,
                gyro=gyro,
                serial_port=args.serial_port,
                baud_rate=args.baud,
                sample_rate_hz=args.fs,
                wait_for_ready=not args.no_wait_ready,
                realtime=not args.fast_stream,
            )

        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt

                plot_reference_and_estimated_trajectory(
                    sim, plt, args.tca_trajectory
                )
                sim.plot(["gyro", "accel"])
                plt.show(block=True)
            except Exception as exc:
                print(f"Warning: Could not generate plots: {exc}")

    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON file: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTransmission interrupted by user")
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
