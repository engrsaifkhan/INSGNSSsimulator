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
import gc
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta

import numpy as np

from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim
from signalsim_to_gnss_ins_sim import convert_signalsim_to_gnss_motion
from demo_algorithms import free_integration

D2R = math.pi / 180.0


def extract_utc_start_time(json_data):
    """Extract the UTC scenario start time from a SignalSim JSON file."""
    try:
        time_config = json_data.get("time", {})
        if time_config.get("type") == "UTC":
            sec = float(time_config["second"])
            whole_sec = int(math.floor(sec))
            frac_sec = sec - whole_sec
            base = datetime(
                int(time_config["year"]),
                int(time_config["month"]),
                int(time_config["day"]),
                int(time_config["hour"]),
                int(time_config["minute"]),
                whole_sec,
            )
            return base + timedelta(seconds=frac_sec)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Warning: Could not parse UTC start time: {exc}")

    return None


# UTC dates at which GPS-UTC increased by one second. GPS time itself has no
# leap seconds, so the offset equals the number of entries effective at UTC.
_GPS_UTC_LEAP_EFFECTIVE_DATES = (
    datetime(1981, 7, 1), datetime(1982, 7, 1),
    datetime(1983, 7, 1), datetime(1985, 7, 1),
    datetime(1988, 1, 1), datetime(1990, 1, 1),
    datetime(1991, 1, 1), datetime(1992, 7, 1),
    datetime(1993, 7, 1), datetime(1994, 7, 1),
    datetime(1996, 1, 1), datetime(1997, 7, 1),
    datetime(1999, 1, 1), datetime(2006, 1, 1),
    datetime(2009, 1, 1), datetime(2012, 7, 1),
    datetime(2015, 7, 1), datetime(2017, 1, 1),
)


def utc_to_gps_tow(utc_time):
    """Convert naive UTC datetime to GPS time-of-week seconds."""
    if utc_time is None:
        raise RuntimeError(
            "First-sample synchronization requires JSON time.type == 'UTC'"
        )
    gps_epoch = datetime(1980, 1, 6)
    leap_seconds = sum(
        1 for effective in _GPS_UTC_LEAP_EFFECTIVE_DATES
        if utc_time >= effective
    )
    gps_seconds = (utc_time - gps_epoch).total_seconds() + leap_seconds
    return gps_seconds % 604800.0


def wrapped_gps_tow_difference(later_tow, earlier_tow):
    """Signed GPS-TOW difference robust to a GPS week rollover."""
    return (later_tow - earlier_tow + 302400.0) % 604800.0 - 302400.0


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


def plot_reference_and_estimated_trajectory(sim, plt, tca_csv_path="", json_trajectory_path=""):
    """Plot PocketSDR GNSS, TCA, and the actual converted SignalSim reference trajectory."""
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

    default_xlim = None
    default_ylim = None
    default_zlim = None
    pan_active = False
    pan_last_x = None
    pan_last_y = None

    def reset_view():
        if default_xlim is not None:
            axis.set_xlim(default_xlim)
        if default_ylim is not None:
            axis.set_ylim(default_ylim)
        if default_zlim is not None:
            axis.set_zlim(default_zlim)
        figure.canvas.draw_idle()

    def wheel_zoom_3d(event):
        if event.inaxes is not axis:
            return
        scale = 1.1 if event.step > 0 else 1.0 / 1.1
        current_x = axis.get_xlim()
        current_y = axis.get_ylim()
        current_z = axis.get_zlim()
        cx = event.xdata if event.xdata is not None else 0.0
        cy = event.ydata if event.ydata is not None else 0.0

        x_span = current_x[1] - current_x[0]
        y_span = current_y[1] - current_y[0]
        z_span = current_z[1] - current_z[0]

        x_center = (current_x[0] + current_x[1]) / 2.0
        y_center = (current_y[0] + current_y[1]) / 2.0
        z_center = (current_z[0] + current_z[1]) / 2.0

        if cx is not None:
            x_center = cx
        if cy is not None:
            y_center = cy

        new_x_span = x_span * scale
        new_y_span = y_span * scale
        new_z_span = z_span * scale

        axis.set_xlim([x_center - new_x_span / 2.0, x_center + new_x_span / 2.0])
        axis.set_ylim([y_center - new_y_span / 2.0, y_center + new_y_span / 2.0])
        axis.set_zlim([z_center - new_z_span / 2.0, z_center + new_z_span / 2.0])
        figure.canvas.draw_idle()

    def button_press(event):
        nonlocal pan_active, pan_last_x, pan_last_y
        if event.inaxes is not axis or event.button != 2:
            return
        pan_active = True
        pan_last_x = event.x
        pan_last_y = event.y
        figure.canvas.draw_idle()

    def button_release(event):
        nonlocal pan_active, pan_last_x, pan_last_y
        if event.inaxes is not axis or event.button != 2:
            return
        pan_active = False
        pan_last_x = None
        pan_last_y = None
        figure.canvas.draw_idle()

    def motion_pan(event):
        nonlocal pan_active, pan_last_x, pan_last_y
        if not pan_active or event.inaxes is not axis:
            return
        if pan_last_x is None or pan_last_y is None:
            return

        dx = event.x - pan_last_x
        dy = event.y - pan_last_y

        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        z_limits = axis.get_zlim()
        x_span = x_limits[1] - x_limits[0]
        y_span = y_limits[1] - y_limits[0]
        z_span = z_limits[1] - z_limits[0]

        x_per_pixel = x_span / max(1, figure.canvas.get_width_height()[0])
        y_per_pixel = y_span / max(1, figure.canvas.get_width_height()[1])
        z_per_pixel = z_span / max(1, figure.canvas.get_width_height()[1])

        x_shift = -dx * x_per_pixel
        y_shift = dy * y_per_pixel
        z_shift = -dy * z_per_pixel

        axis.set_xlim([x_limits[0] + x_shift, x_limits[1] + x_shift])
        axis.set_ylim([y_limits[0] + y_shift, y_limits[1] + y_shift])
        axis.set_zlim([z_limits[0] + z_shift, z_limits[1] + z_shift])

        pan_last_x = event.x
        pan_last_y = event.y
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("scroll_event", wheel_zoom_3d)
    figure.canvas.mpl_connect("button_press_event", button_press)
    figure.canvas.mpl_connect("button_release_event", button_release)
    figure.canvas.mpl_connect("motion_notify_event", motion_pan)

    axis.plot(*to_local_enu(reference), label="PocketSDR GNSS", linewidth=2.0)
    if tca_positions:
        axis.plot(
            *to_local_enu(np.asarray(tca_positions)),
            label="TCA",
            linewidth=1.4,
        )

    sim_reference = []
    available = sim.get_names_of_available_data() if sim is not None else []
    if "ref_pos" in available:
        sim_reference = np.asarray(sim.get_data(["ref_pos"])[0], dtype=np.float64)
        if sim_reference.ndim == 1:
            sim_reference = sim_reference.reshape((-1, 3))
        if sim_reference.shape[0] > 0:
            axis.plot(
                *to_local_enu(sim_reference),
                label="SignalSim converted",
                linewidth=1.2,
                linestyle="--",
            )

    default_xlim = axis.get_xlim()
    default_ylim = axis.get_ylim()
    default_zlim = axis.get_zlim()

    button_ax = figure.add_axes([0.78, 0.92, 0.16, 0.05])
    reset_button = plt.Button(button_ax, "Reset view")
    reset_button.on_clicked(lambda _event: reset_view())

    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_zlabel("Up (m)")
    axis.set_title("PocketSDR GNSS vs TCA and SignalSim converted trajectory")
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



def _write_sync_file(path, text):
    """Atomically create/update a synchronization marker file."""
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as sync_file:
        sync_file.write(text)
        sync_file.flush()
        try:
            os.fsync(sync_file.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _wait_for_file(path, timeout_seconds, description):
    path = os.path.abspath(os.path.expanduser(path))
    deadline = (
        time.monotonic() + timeout_seconds
        if timeout_seconds is not None and timeout_seconds > 0.0
        else None
    )
    while not os.path.isfile(path):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {description}: {path}")
        time.sleep(0.001)
    return path


def _read_first_gnss_obssec(epoch_file, timeout_seconds):
    path = _wait_for_file(epoch_file, timeout_seconds, "first GNSS obssec")
    deadline = (
        time.monotonic() + timeout_seconds
        if timeout_seconds is not None and timeout_seconds > 0.0
        else None
    )
    while True:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
            if text:
                return float(text.split()[0])
        except (OSError, ValueError):
            pass
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Could not read valid GNSS obssec from {path}")
        time.sleep(0.001)


def find_aligned_imu_start_index(
    timestamps, first_gnss_obssec, trajectory_start_tow, sample_rate_hz
):
    """Select the first IMU sample at or after the first GNSS epoch."""
    relative_times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(relative_times) <= 0:
        raise RuntimeError("No IMU timestamps are available for alignment")
    relative_times = relative_times - relative_times[0]

    offset_sec = wrapped_gps_tow_difference(
        first_gnss_obssec, trajectory_start_tow
    )
    if offset_sec < -0.5 / sample_rate_hz:
        raise RuntimeError(
            "First GNSS obssec precedes JSON trajectory start: "
            f"first_obssec={first_gnss_obssec:.9f}, "
            f"start_tow={trajectory_start_tow:.9f}, "
            f"offset={offset_sec:.9f} s"
        )
    offset_sec = max(0.0, offset_sec)

    if offset_sec > relative_times[-1] + 0.5 / sample_rate_hz:
        raise RuntimeError(
            "First GNSS obssec lies beyond generated IMU trajectory: "
            f"offset={offset_sec:.6f} s, "
            f"IMU duration={relative_times[-1]:.6f} s"
        )

    # Never send an IMU sample from before the GNSS epoch.  A nearest-neighbor
    # choice could select an earlier sample when the GNSS epoch falls between
    # two IMU samples, causing the two streams to start at different times.
    start_index = int(np.searchsorted(relative_times, offset_sec, side="left"))
    if (
        start_index > 0
        and math.isclose(
            float(relative_times[start_index - 1]),
            offset_sec,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        start_index -= 1
    if start_index >= len(relative_times):
        raise RuntimeError(
            "No IMU sample exists at or after the first GNSS obssec: "
            f"offset={offset_sec:.9f} s, "
            f"IMU duration={relative_times[-1]:.9f} s"
        )
    selected_rel_time = float(relative_times[start_index])
    alignment_error = selected_rel_time - offset_sec
    previous_gap_sec = (
        selected_rel_time - float(relative_times[start_index - 1])
        if start_index > 0
        else 1.0 / sample_rate_hz
    )
    if alignment_error > previous_gap_sec + 1e-9:
        raise RuntimeError(
            "Forward IMU/GNSS alignment exceeds one IMU sample period: "
            f"alignment_error={alignment_error:.9f} s, "
            f"sample_gap={previous_gap_sec:.9f} s"
        )
    return start_index, offset_sec, selected_rel_time, alignment_error


def plot_final_imu_gnss_interval(
    timestamps, first_gnss_obssec, trajectory_start_tow
):
    """Plot the final IMU sample relative to the first GNSS epoch."""
    import matplotlib.pyplot as plt

    final_imu_tow = (
        trajectory_start_tow + float(timestamps[-1] - timestamps[0])
    ) % 604800.0
    interval_sec = wrapped_gps_tow_difference(
        final_imu_tow, first_gnss_obssec
    )

    figure, axis = plt.subplots(figsize=(9, 2.8))
    axis.axvline(0.0, color="tab:green", linewidth=2, label="GNSS epoch")
    axis.axvline(
        interval_sec,
        color="tab:blue",
        linewidth=2,
        label="Last IMU sample",
    )
    axis.plot([0.0, interval_sec], [0.0, 0.0], color="0.35", linewidth=3)
    axis.scatter([0.0, interval_sec], [0.0, 0.0], s=70, zorder=3)
    axis.annotate(
        f"GNSS epoch\n{first_gnss_obssec:.6f} TOW",
        (0.0, 0.0),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        color="tab:green",
    )
    axis.annotate(
        f"Last IMU sample\n{final_imu_tow:.6f} TOW",
        (interval_sec, 0.0),
        xytext=(0, -42),
        textcoords="offset points",
        ha="center",
        color="tab:blue",
    )
    axis.set_title(
        f"Final IMU sample relative to GNSS epoch: {interval_sec * 1000.0:+.3f} ms"
    )
    axis.set_xlabel("Time from GNSS epoch (s)")
    axis.set_yticks([])
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()

    print(
        "Final IMU/GNSS interval: "
        f"last IMU TOW={final_imu_tow:.9f}, "
        f"GNSS TOW={first_gnss_obssec:.9f}, "
        f"interval={interval_sec * 1000.0:+.3f} ms"
    )
    return figure


def stream_imu_arrays_to_teensy(
    timestamps,
    accel,
    gyro,
    serial_port,
    baud_rate,
    sample_rate_hz,
    realtime=True,
    sync_ready_file=None,
    sync_epoch_file=None,
    sync_aligned_file=None,
    sync_go_file=None,
    sync_start_file=None,
    sync_timeout=120.0,
    trajectory_start_tow=None,
    timing_csv_path="imu_pc_deep_timing_events.csv",
):
    """
    Stream in-memory IMU arrays to Teensy and report transmission diagnostics.

    Packet format:
        IMU,index,time,ax,ay,az,gx,gy,gz\n

    Control messages:
        IMU_STREAM_BEGIN,count,sample_rate\n
        IMU_STREAM_END,count\n

    The PC-side diagnostics distinguish three things:

      1. Scheduling lateness:
         How late Python reaches the intended serial.write() start time.

      2. Write-start interval:
         Time between consecutive serial.write() calls. At 200 Hz the target
         interval is 5.000 ms.

      3. serial.write() call duration:
         How long the host-side PySerial write call takes to return.

    Important:
        serial.write() duration is a PC/driver-side measurement. It does not
        prove the exact physical arrival time of the packet at Teensy.
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

    # ------------------------------------------------------------
    # Transmission diagnostic counters
    # ------------------------------------------------------------
    tx_attempted_packets = 0
    tx_successful_packets = 0
    tx_short_writes = 0
    tx_errors = 0

    tx_bytes_requested = 0
    tx_bytes_written = 0

    stream_start_perf = None
    stream_end_perf = None

    # These are initialized here so the final summary can always access them.
    period_ns = 0
    late_threshold_ns = 0
    interval_high_threshold_ns = 0
    interval_low_threshold_ns = 0
    serial_slow_threshold_ns = 1_000_000  # 1.000 ms

    pc_late_packets = 0
    total_schedule_lateness_ns = 0
    max_schedule_lateness_ns = 0

    previous_write_start_ns = None
    first_write_start_ns = None
    last_write_start_ns = None

    interval_count = 0
    total_interval_ns = 0
    min_interval_ns = None
    max_interval_ns = 0
    interval_over_high = 0
    interval_under_low = 0

    total_write_duration_ns = 0
    max_write_duration_ns = 0
    slow_writes = 0

    last_schedule_lateness_ns = 0
    last_interval_ns = 0
    last_write_duration_ns = 0

    # ------------------------------------------------------------
    # Deep PC timing diagnostics.
    #
    # These measurements are intentionally kept in RAM during the stream.
    # CSV is written only after streaming so disk I/O cannot disturb 200 Hz.
    # ------------------------------------------------------------
    deep_event_rows = []
    deep_event_limit = 20_000
    deep_events_dropped = 0
    deep_cause_counts = {}

    total_sleep_calls = 0
    max_sleep_overshoot_ns = 0
    max_prep_wall_ns = 0
    max_prep_cpu_ns = 0
    max_prep_desched_ns = 0
    max_spin_desched_ns = 0
    max_progress_print_duration_ns = 0

    previous_write_duration_ns = 0
    previous_serial_write_end_ns = None
    last_progress_print_duration_ns = 0

    timing_csv_actual_path = None

    # Track cyclic-GC pauses without printing from the callback.
    gc_diag = {
        "active_start_ns": 0,
        "count": 0,
        "total_duration_ns": 0,
        "max_duration_ns": 0,
        "last_end_ns": 0,
        "last_duration_ns": 0,
    }
    gc_callback = None
    gc_callback_installed = False

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

        # --------------------------------------------------------
        # First-sample trajectory synchronization.
        # READY is created only after all IMU arrays are generated and the
        # Teensy serial port is open/settled. PocketSDR then starts its receiver,
        # holds the first valid TCA packet, and publishes that packet's obssec.
        # The IMU selects the matching array index before either stream is freed.
        # --------------------------------------------------------
        start_index = 0
        first_gnss_obssec = None
        alignment_error = 0.0

        aligned_mode = bool(
            sync_ready_file and sync_epoch_file and
            sync_aligned_file and sync_go_file
        )

        if aligned_mode:
            if trajectory_start_tow is None:
                raise RuntimeError(
                    "trajectory_start_tow is required for aligned sync mode"
                )

            _write_sync_file(
                sync_ready_file,
                (
                    f"IMU_READY pid={os.getpid()} "
                    f"wall_time={time.time():.9f} "
                    f"samples={sample_count} "
                    f"trajectory_start_tow={trajectory_start_tow:.9f}\n"
                ),
            )
            print(f"GNSS/IMU sync: IMU READY -> {sync_ready_file}")
            print(f"GNSS/IMU sync: waiting for first GNSS obssec -> {sync_epoch_file}")

            first_gnss_obssec = _read_first_gnss_obssec(
                sync_epoch_file, sync_timeout
            )
            (
                start_index, offset_sec, selected_rel_time, alignment_error
            ) = find_aligned_imu_start_index(
                timestamps=timestamps,
                first_gnss_obssec=first_gnss_obssec,
                trajectory_start_tow=trajectory_start_tow,
                sample_rate_hz=sample_rate_hz,
            )

            print("\n" + "=" * 70)
            print("FIRST-SAMPLE GNSS/IMU TRAJECTORY ALIGNMENT")
            print("=" * 70)
            print(f"JSON trajectory start GPS TOW : {trajectory_start_tow:.9f}")
            print(f"First held TCA obssec          : {first_gnss_obssec:.9f}")
            print(f"Required trajectory offset    : {offset_sec:.9f} s")
            print(f"Selected IMU source index     : {start_index}")
            print(f"Selected IMU relative time    : {selected_rel_time:.9f} s")
            print(f"Alignment error               : {alignment_error * 1000.0:.3f} ms")
            print("=" * 70)

            _write_sync_file(
                sync_aligned_file,
                (
                    f"IMU_ALIGNED first_obssec={first_gnss_obssec:.9f} "
                    f"trajectory_start_tow={trajectory_start_tow:.9f} "
                    f"offset_sec={offset_sec:.9f} "
                    f"start_index={start_index} "
                    f"selected_rel_time={selected_rel_time:.9f} "
                    f"alignment_error_sec={alignment_error:.9f}\n"
                ),
            )
            print(f"GNSS/IMU sync: IMU ALIGNED -> {sync_aligned_file}")
            print(f"GNSS/IMU sync: waiting for common GO -> {sync_go_file}")
            _wait_for_file(sync_go_file, sync_timeout, "common GO")
            print(
                "GNSS/IMU sync: GO detected; starting aligned IMU stream "
                f"from source index {start_index}"
            )

        elif sync_ready_file and sync_start_file:
            # Backward-compatible legacy barrier. This synchronizes wall-clock
            # start only and does NOT align trajectory samples.
            _write_sync_file(
                sync_ready_file,
                (
                    f"IMU_READY_LEGACY pid={os.getpid()} "
                    f"wall_time={time.time():.9f} samples={sample_count}\n"
                ),
            )
            print(f"GNSS/IMU sync: legacy READY -> {sync_ready_file}")
            _wait_for_file(sync_start_file, sync_timeout, "legacy START")
            print("GNSS/IMU sync: legacy START detected")

        # --------------------------------------------------------
        # Start stream.
        # --------------------------------------------------------
        stream_count = sample_count - start_index
        if stream_count <= 0:
            raise RuntimeError(
                f"Aligned IMU start index {start_index} leaves no samples"
            )

        begin_packet = (
            f"IMU_STREAM_BEGIN,{stream_count},{sample_rate_hz:.6f}\n"
        )
        begin_data = begin_packet.encode("ascii")

        begin_write_start_ns = time.perf_counter_ns()
        begin_written = serial_connection.write(begin_data)
        begin_write_end_ns = time.perf_counter_ns()
        begin_write_duration_ns = begin_write_end_ns - begin_write_start_ns

        if begin_written != len(begin_data):
            raise IOError(
                "Short write while sending IMU_STREAM_BEGIN: "
                f"requested={len(begin_data)}, written={begin_written}"
            )

        # --------------------------------------------------------
        # PC-side pacing and timing diagnostics.
        #
        # Absolute targets are used:
        #   target(n) = stream_start + n * period
        #
        # Therefore an occasional late sample does not intentionally shift all
        # later sample targets. At 200 Hz, period_ns = 5,000,000 ns.
        # --------------------------------------------------------
        if realtime:
            period_ns = int(round(1_000_000_000.0 / sample_rate_hz))

            # A packet is counted as a meaningful PC scheduling-late event only
            # when it starts >5% of one sample period late, with a floor of
            # 0.100 ms. At 200 Hz this threshold is 0.250 ms.
            late_threshold_ns = max(100_000, int(round(period_ns * 0.05)))

            # Interval anomaly limits: +/-10% around the requested period.
            # At 200 Hz these are 4.500 ms and 5.500 ms.
            interval_high_threshold_ns = int(round(period_ns * 1.10))
            interval_low_threshold_ns = int(round(period_ns * 0.90))

            # Minimal cyclic-GC timing callback. It does not print or write files.
            def _gc_timing_callback(phase, info):
                if phase == "start":
                    gc_diag["active_start_ns"] = time.perf_counter_ns()
                elif phase == "stop":
                    end_ns = time.perf_counter_ns()
                    start_ns = gc_diag["active_start_ns"]
                    duration_ns = max(0, end_ns - start_ns) if start_ns else 0
                    gc_diag["count"] += 1
                    gc_diag["total_duration_ns"] += duration_ns
                    gc_diag["last_end_ns"] = end_ns
                    gc_diag["last_duration_ns"] = duration_ns
                    if duration_ns > gc_diag["max_duration_ns"]:
                        gc_diag["max_duration_ns"] = duration_ns
                    gc_diag["active_start_ns"] = 0

            gc_callback = _gc_timing_callback
            gc.callbacks.append(gc_callback)
            gc_callback_installed = True

            stream_start_ns = time.perf_counter_ns()
            stream_start_perf = time.perf_counter()

            print("\nPC timing diagnostics enabled")
            print(f"Target interval             : {period_ns / 1e6:.6f} ms")
            print(
                f"Scheduling-late threshold   : "
                f"{late_threshold_ns / 1e6:.6f} ms"
            )
            print(
                f"Interval acceptable window  : "
                f"{interval_low_threshold_ns / 1e6:.6f} .. "
                f"{interval_high_threshold_ns / 1e6:.6f} ms"
            )
            print(
                f"BEGIN serial.write duration : "
                f"{begin_write_duration_ns / 1e6:.6f} ms"
            )
        else:
            stream_start_perf = time.perf_counter()
            stream_start_ns = None
            print("\nPC real-time pacing disabled (--fast-stream)")

        # --------------------------------------------------------
        # Transmit all IMU samples
        # --------------------------------------------------------
        for source_index in range(start_index, sample_count):
            stream_index = source_index - start_index

            # Keep the absolute target independent of work done in this iteration.
            if realtime:
                target_ns = stream_start_ns + stream_index * period_ns
            else:
                target_ns = 0

            iteration_start_ns = time.perf_counter_ns()
            iteration_start_cpu_ns = time.thread_time_ns()

            previous_progress_print_duration_ns = last_progress_print_duration_ns
            previous_tail_ns = (
                iteration_start_ns - previous_serial_write_end_ns
                if previous_serial_write_end_ns is not None
                else 0
            )

            gc_count_at_iteration_start = gc_diag["count"]
            gc_total_at_iteration_start_ns = gc_diag["total_duration_ns"]

            # ----------------------------------------------------
            # Measure packet preparation separately.
            # ----------------------------------------------------
            prep_start_ns = time.perf_counter_ns()
            prep_start_cpu_ns = time.thread_time_ns()

            ax, ay, az = accel[source_index]
            gx, gy, gz = gyro[source_index]

            packet = (
                f"IMU,{stream_index},{timestamps[source_index]:.6f},"
                f"{ax:.9f},{ay:.9f},{az:.9f},"
                f"{gx:.9f},{gy:.9f},{gz:.9f}\n"
            )
            data = packet.encode("ascii")

            prep_end_cpu_ns = time.thread_time_ns()
            prep_end_ns = time.perf_counter_ns()

            prep_wall_ns = prep_end_ns - prep_start_ns
            prep_cpu_ns = prep_end_cpu_ns - prep_start_cpu_ns
            prep_desched_ns = max(0, prep_wall_ns - prep_cpu_ns)

            if prep_wall_ns > max_prep_wall_ns:
                max_prep_wall_ns = prep_wall_ns
            if prep_cpu_ns > max_prep_cpu_ns:
                max_prep_cpu_ns = prep_cpu_ns
            if prep_desched_ns > max_prep_desched_ns:
                max_prep_desched_ns = prep_desched_ns

            tx_attempted_packets += 1
            tx_bytes_requested += len(data)

            # Per-iteration deep wait measurements.
            sleep_calls_this_packet = 0
            sleep_requested_ns = 0
            sleep_actual_ns = 0
            sleep_cpu_ns = 0
            sleep_overshoot_ns = 0

            spin_start_ns = None
            spin_start_cpu_ns = None
            spin_wall_ns = 0
            spin_cpu_ns = 0
            spin_desched_ns = 0

            pre_wait_lateness_ns = 0

            # ----------------------------------------------------
            # Wait for THIS packet's absolute PC-side target time.
            # ----------------------------------------------------
            if realtime:
                wait_entry_ns = time.perf_counter_ns()
                pre_wait_lateness_ns = max(0, wait_entry_ns - target_ns)

                while True:
                    now_ns = time.perf_counter_ns()
                    remaining_ns = target_ns - now_ns

                    if remaining_ns <= 0:
                        break

                   

                wait_done_ns = time.perf_counter_ns()
                wait_done_cpu_ns = time.thread_time_ns()

                if spin_start_ns is not None:
                    spin_wall_ns = wait_done_ns - spin_start_ns
                    spin_cpu_ns = wait_done_cpu_ns - spin_start_cpu_ns
                    spin_desched_ns = max(0, spin_wall_ns - spin_cpu_ns)
                    if spin_desched_ns > max_spin_desched_ns:
                        max_spin_desched_ns = spin_desched_ns

                # Timestamp immediately before the host serial.write() call.
                write_start_ns = time.perf_counter_ns()
                schedule_lateness_ns = max(0, write_start_ns - target_ns)
                last_schedule_lateness_ns = schedule_lateness_ns

                total_schedule_lateness_ns += schedule_lateness_ns
                if schedule_lateness_ns > max_schedule_lateness_ns:
                    max_schedule_lateness_ns = schedule_lateness_ns

                if schedule_lateness_ns > late_threshold_ns:
                    pc_late_packets += 1

            else:
                write_start_ns = time.perf_counter_ns()
                schedule_lateness_ns = 0
                last_schedule_lateness_ns = 0

            # ----------------------------------------------------
            # Measure actual interval between host write starts.
            # ----------------------------------------------------
            if first_write_start_ns is None:
                first_write_start_ns = write_start_ns

            if previous_write_start_ns is not None:
                interval_ns = write_start_ns - previous_write_start_ns
                last_interval_ns = interval_ns

                interval_count += 1
                total_interval_ns += interval_ns

                if min_interval_ns is None or interval_ns < min_interval_ns:
                    min_interval_ns = interval_ns
                if interval_ns > max_interval_ns:
                    max_interval_ns = interval_ns

                if realtime:
                    if interval_ns > interval_high_threshold_ns:
                        interval_over_high += 1
                    if interval_ns < interval_low_threshold_ns:
                        interval_under_low += 1
            else:
                interval_ns = 0
                last_interval_ns = 0

            previous_write_start_ns = write_start_ns
            last_write_start_ns = write_start_ns

            # ----------------------------------------------------
            # Deep classification of meaningful scheduling-late events.
            # Current serial.write() cannot cause current write-start lateness;
            # it is measured below and can affect the NEXT packet.
            # ----------------------------------------------------
            late_event_record = None

            if realtime and schedule_lateness_ns > late_threshold_ns:
                gc_count_this_iteration = (
                    gc_diag["count"] - gc_count_at_iteration_start
                )
                gc_time_this_iteration_ns = max(
                    0,
                    gc_diag["total_duration_ns"] - gc_total_at_iteration_start_ns,
                )

                if (
                    previous_progress_print_duration_ns > late_threshold_ns
                    and pre_wait_lateness_ns > late_threshold_ns
                ):
                    cause = "PREVIOUS_PROGRESS_PRINT"
                elif (
                    gc_count_this_iteration > 0
                    and gc_time_this_iteration_ns > late_threshold_ns
                ):
                    cause = "PYTHON_GC"
                elif prep_desched_ns > late_threshold_ns:
                    cause = "PREP_PREEMPTION"
                elif (
                    previous_write_duration_ns > late_threshold_ns
                    and pre_wait_lateness_ns > late_threshold_ns
                ):
                    cause = "PREVIOUS_SERIAL_WRITE_OR_TAIL"
                elif pre_wait_lateness_ns > late_threshold_ns:
                    cause = "ALREADY_LATE_BEFORE_WAIT"
                elif sleep_overshoot_ns > late_threshold_ns:
                    cause = "SLEEP_WAKEUP_OVERSHOOT"
                elif spin_desched_ns > late_threshold_ns:
                    cause = "BUSY_WAIT_PREEMPTION"
                else:
                    cause = "OTHER_SCHEDULER_KERNEL"

                deep_cause_counts[cause] = deep_cause_counts.get(cause, 0) + 1

                late_event_record = {
                    "stream_index": stream_index,
                    "source_index": source_index,
                    "target_ns": target_ns,
                    "write_start_ns": write_start_ns,
                    "lateness_ms": schedule_lateness_ns / 1e6,
                    "interval_ms": interval_ns / 1e6 if interval_ns else 0.0,
                    "prep_wall_ms": prep_wall_ns / 1e6,
                    "prep_cpu_ms": prep_cpu_ns / 1e6,
                    "prep_desched_ms": prep_desched_ns / 1e6,
                    "pre_wait_late_ms": pre_wait_lateness_ns / 1e6,
                    "sleep_calls": sleep_calls_this_packet,
                    "sleep_requested_ms": sleep_requested_ns / 1e6,
                    "sleep_actual_ms": sleep_actual_ns / 1e6,
                    "sleep_overshoot_ms": sleep_overshoot_ns / 1e6,
                    "sleep_cpu_ms": sleep_cpu_ns / 1e6,
                    "spin_wall_ms": spin_wall_ns / 1e6,
                    "spin_cpu_ms": spin_cpu_ns / 1e6,
                    "spin_desched_ms": spin_desched_ns / 1e6,
                    "previous_write_ms": previous_write_duration_ns / 1e6,
                    "previous_tail_ms": previous_tail_ns / 1e6,
                    "previous_progress_print_ms": (
                        previous_progress_print_duration_ns / 1e6
                    ),
                    "gc_count": gc_count_this_iteration,
                    "gc_time_ms": gc_time_this_iteration_ns / 1e6,
                    "current_serial_write_ms": 0.0,
                    "cause": cause,
                }

            # ----------------------------------------------------
            # Measure how long serial.write() itself takes to return.
            # ----------------------------------------------------
            try:
                serial_write_start_ns = time.perf_counter_ns()
                written = serial_connection.write(data)
                serial_write_end_ns = time.perf_counter_ns()
            except Exception as exc:
                tx_errors += 1
                print(
                    f"TX ERROR: source_index={source_index}, "
                    f"exception={type(exc).__name__}: {exc}"
                )
                raise

            write_duration_ns = serial_write_end_ns - serial_write_start_ns
            last_write_duration_ns = write_duration_ns
            total_write_duration_ns += write_duration_ns

            if write_duration_ns > max_write_duration_ns:
                max_write_duration_ns = write_duration_ns
            if write_duration_ns > serial_slow_threshold_ns:
                slow_writes += 1

            tx_bytes_written += written

            if written != len(data):
                tx_short_writes += 1
                print(
                    f"TX SHORT WRITE: source_index={source_index}, "
                    f"requested={len(data)}, written={written}"
                )
                raise IOError(
                    f"Short serial write at IMU source index {source_index}: "
                    f"requested={len(data)}, written={written}"
                )

            tx_successful_packets += 1

            # Finish the late-event record after current serial.write() is known.
            if late_event_record is not None:
                late_event_record["current_serial_write_ms"] = (
                    write_duration_ns / 1e6
                )
                if len(deep_event_rows) < deep_event_limit:
                    deep_event_rows.append(late_event_record)
                else:
                    deep_events_dropped += 1

            previous_write_duration_ns = write_duration_ns
            previous_serial_write_end_ns = serial_write_end_ns

            # ----------------------------------------------------
            # Progress display once per 1000 transmitted samples.
            # Measure print duration because a slow terminal write can make
            # the NEXT IMU sample late.
            # ----------------------------------------------------
            sent_count = stream_index + 1
            current_progress_print_duration_ns = 0

            if sent_count % 1000 == 0 or sent_count == stream_count:
                progress_print_start_ns = time.perf_counter_ns()

                if realtime:
                    current_interval_ms = (
                        last_interval_ns / 1e6 if last_interval_ns else 0.0
                    )
                    print(
                        f"Sent {sent_count}/{stream_count} samples "
                        f"(source index {source_index}) | "
                        f"PC_late>{late_threshold_ns / 1e6:.3f}ms="
                        f"{pc_late_packets} | "
                        f"now_late={last_schedule_lateness_ns / 1e6:.3f} ms | "
                        f"max_late={max_schedule_lateness_ns / 1e6:.3f} ms | "
                        f"interval={current_interval_ms:.3f} ms | "
                        f"write={last_write_duration_ns / 1e6:.3f} ms | "
                        f"max_write={max_write_duration_ns / 1e6:.3f} ms"
                    )
                else:
                    print(
                        f"Sent {sent_count}/{stream_count} samples "
                        f"(source index {source_index}) | "
                        f"write={last_write_duration_ns / 1e6:.3f} ms | "
                        f"max_write={max_write_duration_ns / 1e6:.3f} ms"
                    )

                progress_print_end_ns = time.perf_counter_ns()
                current_progress_print_duration_ns = (
                    progress_print_end_ns - progress_print_start_ns
                )
                if (
                    current_progress_print_duration_ns
                    > max_progress_print_duration_ns
                ):
                    max_progress_print_duration_ns = (
                        current_progress_print_duration_ns
                    )

            last_progress_print_duration_ns = current_progress_print_duration_ns

        stream_end_perf = time.perf_counter()

        # --------------------------------------------------------
        # End packet
        # --------------------------------------------------------
        end_packet = f"IMU_STREAM_END,{stream_count}\n"
        end_data = end_packet.encode("ascii")
        end_written = serial_connection.write(end_data)

        if end_written != len(end_data):
            raise IOError(
                "Short write while sending IMU_STREAM_END: "
                f"requested={len(end_data)}, written={end_written}"
            )

        # Ensure all queued bytes are handed to the OS/serial driver
        # before closing the COM port.
        serial_connection.flush()

        # --------------------------------------------------------
        # Write deep timing events AFTER streaming so disk I/O cannot disturb
        # the 200 Hz transmission loop.
        # --------------------------------------------------------
        if realtime and timing_csv_path:
            timing_path = Path(timing_csv_path).expanduser()
            if not timing_path.is_absolute():
                timing_path = Path.cwd() / timing_path
            timing_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = [
                "stream_index",
                "source_index",
                "target_ns",
                "write_start_ns",
                "lateness_ms",
                "interval_ms",
                "prep_wall_ms",
                "prep_cpu_ms",
                "prep_desched_ms",
                "pre_wait_late_ms",
                "sleep_calls",
                "sleep_requested_ms",
                "sleep_actual_ms",
                "sleep_overshoot_ms",
                "sleep_cpu_ms",
                "spin_wall_ms",
                "spin_cpu_ms",
                "spin_desched_ms",
                "previous_write_ms",
                "previous_tail_ms",
                "previous_progress_print_ms",
                "gc_count",
                "gc_time_ms",
                "current_serial_write_ms",
                "cause",
            ]

            with open(timing_path, "w", newline="", encoding="utf-8") as timing_file:
                writer = csv.DictWriter(timing_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(deep_event_rows)

            timing_csv_actual_path = str(timing_path.resolve())

        # --------------------------------------------------------
        # Final diagnostic summary
        # --------------------------------------------------------
        print("\n" + "=" * 70)
        print("PYTHON / TEENSY IMU TRANSMISSION DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"Generated IMU samples       : {sample_count}")
        print(f"Aligned source start index  : {start_index}")
        print(f"Transmitted IMU samples     : {stream_count}")
        print(f"TX attempted packets        : {tx_attempted_packets}")
        print(f"TX successful packets       : {tx_successful_packets}")
        print(f"TX short writes             : {tx_short_writes}")
        print(f"TX write exceptions         : {tx_errors}")
        print(f"TX bytes requested          : {tx_bytes_requested}")
        print(f"TX bytes reported written   : {tx_bytes_written}")

        print("\nPC-SIDE TIMING DIAGNOSTICS")
        print("-" * 70)

        avg_write_duration_ns = (
            total_write_duration_ns / tx_successful_packets
            if tx_successful_packets > 0 else 0.0
        )
        print(
            f"Average serial.write time   : "
            f"{avg_write_duration_ns / 1e6:.6f} ms"
        )
        print(
            f"Maximum serial.write time   : "
            f"{max_write_duration_ns / 1e6:.6f} ms"
        )
        print(
            f"serial.write > 1.000 ms     : "
            f"{slow_writes}"
        )

        if realtime:
            average_schedule_lateness_ns = (
                total_schedule_lateness_ns / tx_successful_packets
                if tx_successful_packets > 0 else 0.0
            )
            average_interval_ns = (
                total_interval_ns / interval_count
                if interval_count > 0 else 0.0
            )

            print(f"Target write-start interval : {period_ns / 1e6:.6f} ms")
            print(
                f"Late-event threshold        : "
                f"{late_threshold_ns / 1e6:.6f} ms"
            )
            print(
                f"PC scheduling late events   : "
                f"{pc_late_packets}"
            )
            print(
                f"Average scheduling lateness : "
                f"{average_schedule_lateness_ns / 1e6:.6f} ms"
            )
            print(
                f"Maximum scheduling lateness : "
                f"{max_schedule_lateness_ns / 1e6:.6f} ms"
            )

            if min_interval_ns is not None:
                print(
                    f"Minimum write-start interval: "
                    f"{min_interval_ns / 1e6:.6f} ms"
                )
                print(
                    f"Average write-start interval: "
                    f"{average_interval_ns / 1e6:.6f} ms"
                )
                print(
                    f"Maximum write-start interval: "
                    f"{max_interval_ns / 1e6:.6f} ms"
                )

            print(
                f"Intervals > {interval_high_threshold_ns / 1e6:.3f} ms"
                f"      : {interval_over_high}"
            )
            print(
                f"Intervals < {interval_low_threshold_ns / 1e6:.3f} ms"
                f"      : {interval_under_low}"
            )

            # This is the most useful cumulative-drift check on the PC.
            if (
                first_write_start_ns is not None
                and last_write_start_ns is not None
                and tx_successful_packets > 1
            ):
                actual_first_to_last_ns = (
                    last_write_start_ns - first_write_start_ns
                )
                expected_first_to_last_ns = (
                    (tx_successful_packets - 1) * period_ns
                )
                cumulative_drift_ns = (
                    actual_first_to_last_ns - expected_first_to_last_ns
                )
                achieved_start_rate_hz = (
                    (tx_successful_packets - 1)
                    / (actual_first_to_last_ns / 1_000_000_000.0)
                    if actual_first_to_last_ns > 0 else 0.0
                )

                print(
                    f"Expected first->last time    : "
                    f"{expected_first_to_last_ns / 1e9:.6f} s"
                )
                print(
                    f"Actual first->last time      : "
                    f"{actual_first_to_last_ns / 1e9:.6f} s"
                )
                print(
                    f"Cumulative PC timing drift   : "
                    f"{cumulative_drift_ns / 1e6:+.6f} ms"
                )
                print(
                    f"Write-start achieved rate   : "
                    f"{achieved_start_rate_hz:.6f} samples/s"
                )

            print("\nDEEP LATE-EVENT ROOT-CAUSE DIAGNOSTICS")
            print("-" * 70)
            print(
                f"Maximum packet-prep wall time: "
                f"{max_prep_wall_ns / 1e6:.6f} ms"
            )
            print(
                f"Maximum packet-prep CPU time : "
                f"{max_prep_cpu_ns / 1e6:.6f} ms"
            )
            print(
                f"Maximum prep deschedule time : "
                f"{max_prep_desched_ns / 1e6:.6f} ms"
            )
            print(
                f"Total time.sleep calls       : "
                f"{total_sleep_calls}"
            )
            print(
                f"Maximum sleep overshoot      : "
                f"{max_sleep_overshoot_ns / 1e6:.6f} ms"
            )
            print(
                f"Maximum busy-spin deschedule : "
                f"{max_spin_desched_ns / 1e6:.6f} ms"
            )
            print(
                f"Maximum progress print time  : "
                f"{max_progress_print_duration_ns / 1e6:.6f} ms"
            )
            print(
                f"Python cyclic-GC collections : "
                f"{gc_diag['count']}"
            )
            print(
                f"Maximum Python GC pause      : "
                f"{gc_diag['max_duration_ns'] / 1e6:.6f} ms"
            )
            print(
                f"Recorded detailed late events: "
                f"{len(deep_event_rows)}"
            )
            if deep_events_dropped:
                print(
                    f"Detailed events dropped      : "
                    f"{deep_events_dropped}"
                )

            if deep_cause_counts:
                print("Late-event classifications:")
                for cause, count in sorted(
                    deep_cause_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                ):
                    print(f"  {cause:30s}: {count}")

            if deep_event_rows:
                print("\nWorst 10 late events:")
                worst_events = sorted(
                    deep_event_rows,
                    key=lambda row: row["lateness_ms"],
                    reverse=True,
                )[:10]
                for row in worst_events:
                    print(
                        f"  stream={row['stream_index']:6d} "
                        f"late={row['lateness_ms']:9.3f} ms "
                        f"interval={row['interval_ms']:9.3f} ms "
                        f"prep_desched={row['prep_desched_ms']:8.3f} ms "
                        f"sleep_ov={row['sleep_overshoot_ms']:8.3f} ms "
                        f"spin_desched={row['spin_desched_ms']:8.3f} ms "
                        f"prev_tail={row['previous_tail_ms']:8.3f} ms "
                        f"gc={row['gc_time_ms']:8.3f} ms "
                        f"cause={row['cause']}"
                    )

            if timing_csv_actual_path:
                print(
                    f"Detailed timing CSV          : "
                    f"{timing_csv_actual_path}"
                )

            print(
                "\nInterpretation:"
                "\n  - SLEEP_WAKEUP_OVERSHOOT -> sleep()/scheduler wake-up is late."
                "\n  - PREP_PREEMPTION -> process was descheduled while formatting packet."
                "\n  - PYTHON_GC -> cyclic garbage collection overlapped the late event."
                "\n  - PREVIOUS_PROGRESS_PRINT -> terminal output delayed the next packet."
                "\n  - PREVIOUS_SERIAL_WRITE_OR_TAIL -> prior write/tail work consumed budget."
                "\n  - BUSY_WAIT_PREEMPTION -> process was preempted during final spin."
                "\n  - OTHER_SCHEDULER_KERNEL -> likely scheduler/kernel noise not explained above."
                "\n  - cumulative drift near 0 ms -> no progressive PC-side timing delay."
            )
        else:
            print("Real-time pacing             : DISABLED (--fast-stream)")

        if (
            stream_start_perf is not None
            and stream_end_perf is not None
            and stream_end_perf > stream_start_perf
        ):
            elapsed = stream_end_perf - stream_start_perf
            actual_rate = tx_successful_packets / elapsed

            print(f"\nTransmission elapsed time   : {elapsed:.3f} s")
            print(f"Average achieved TX rate    : {actual_rate:.3f} samples/s")

        print("Teensy READY required        : NO")
        print("Teensy DONE required         : NO")
        print(
            "RESULT                       : "
            "Python transmission completed independently"
        )
        print("=" * 70)
        print("Direct IMU transmission completed")

        return first_gnss_obssec

    finally:
        if gc_callback_installed and gc_callback is not None:
            try:
                gc.callbacks.remove(gc_callback)
            except ValueError:
                pass
        serial_connection.close()

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
        required=True,
        help="Teensy port, for example COM5 or /dev/ttyACM0",
    )
    parser.add_argument(
        "--tca-trajectory",
        default="",
        help="CSV file containing PocketSDR TCA LLH samples for plotting",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=921_600,
        help="Serial baud rate; default: 921600 (must match Teensy Serial2)",
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
        "--fast-stream",
        action="store_true",
        help="Transmit as fast as possible instead of pacing at --fs",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display simulation plots",
    )
    parser.add_argument(
        "--sync-ready-file",
        default="",
        help="Marker created after IMU preprocessing and serial setup complete",
    )
    parser.add_argument(
        "--sync-epoch-file",
        default="",
        help="File in which PocketSDR publishes the first held TCA obssec",
    )
    parser.add_argument(
        "--sync-aligned-file",
        default="",
        help="Marker created after IMU arrays are aligned to first GNSS obssec",
    )
    parser.add_argument(
        "--sync-go-file",
        default="",
        help="Common GO marker created by PocketSDR after IMU alignment",
    )
    parser.add_argument(
        "--sync-start-file",
        default="",
        help="Deprecated legacy wall-clock START marker",
    )
    parser.add_argument(
        "--sync-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for each synchronization stage; 0 waits indefinitely",
    )
    parser.add_argument(
        "--timing-csv",
        default="imu_pc_deep_timing_events.csv",
        help=(
            "CSV file for detailed PC late-event diagnostics. "
            "Written only after streaming; default: imu_pc_deep_timing_events.csv"
        ),
    )
    parser.add_argument(
        "--trajectory-start-obssec",
        type=float,
        default=None,
        help=(
            "Optional override for the JSON trajectory start GPS obssec/TOW. "
            "Normally derived automatically from JSON UTC time."
        ),
    )

    args = parser.parse_args()

    if args.fs <= 0.0:
        parser.error("--fs must be greater than zero")
    if args.fs_gps <= 0.0:
        parser.error("--fs-gps must be greater than zero")
    aligned_sync_flags = [
        bool(args.sync_ready_file),
        bool(args.sync_epoch_file),
        bool(args.sync_aligned_file),
        bool(args.sync_go_file),
    ]
    if any(aligned_sync_flags) and not all(aligned_sync_flags):
        parser.error(
            "Aligned sync requires --sync-ready-file, --sync-epoch-file, "
            "--sync-aligned-file, and --sync-go-file together"
        )
    if args.sync_start_file and not args.sync_ready_file:
        parser.error(
            "Legacy --sync-start-file requires --sync-ready-file"
        )
    if args.sync_timeout < 0.0:
        parser.error("--sync-timeout must be >= 0")
    if not os.path.isfile(args.trajectory):
        parser.error(f"Configuration file not found: {args.trajectory}")

    print("=" * 70)
    print("GNSS-INS-SIM: Independent Direct IMU-to-Teensy Stream")
    print("=" * 70)
    print(f"Configuration: {args.trajectory}")
    print(f"IMU model:     {args.imu_accuracy}")
    print(f"IMU rate:      {args.fs} Hz")
    print(f"Teensy port:   {args.serial_port}")
    print(f"Serial baud:   {args.baud}")
    print("CSV output:    disabled")
    print("=" * 70)

    try:
        with open(args.trajectory, "r", encoding="utf-8") as config_file:
            json_data = json.load(config_file)

        utc_start_time = extract_utc_start_time(json_data)
        trajectory_start_tow = (
            utc_to_gps_tow(utc_start_time)
            if utc_start_time is not None else None
        )
        if args.trajectory_start_obssec is not None:
            trajectory_start_tow = float(args.trajectory_start_obssec) % 604800.0
            print(
                "Using user override for trajectory start obssec/TOW: "
                f"{trajectory_start_tow:.9f} s"
            )
        if trajectory_start_tow is not None:
            print(
                f"JSON trajectory start GPS TOW: "
                f"{trajectory_start_tow:.9f} s"
            )

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

        first_gnss_obssec = stream_imu_arrays_to_teensy(
            timestamps=timestamps,
            accel=accel,
            gyro=gyro,
            serial_port=args.serial_port,
            baud_rate=args.baud,
            sample_rate_hz=args.fs,
            realtime=not args.fast_stream,
            sync_ready_file=args.sync_ready_file or None,
            sync_epoch_file=args.sync_epoch_file or None,
            sync_aligned_file=args.sync_aligned_file or None,
            sync_go_file=args.sync_go_file or None,
            sync_start_file=args.sync_start_file or None,
            sync_timeout=args.sync_timeout,
            trajectory_start_tow=trajectory_start_tow,
            timing_csv_path=args.timing_csv,
        )

        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt

                plot_reference_and_estimated_trajectory(
                    sim,
                    plt,
                    args.tca_trajectory,
                    args.trajectory,
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
