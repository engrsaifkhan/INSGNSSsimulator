#!/usr/bin/env python3
"""
Launch PocketSDR GUI and the Teensy IMU simulator with first-sample trajectory
synchronization.

Handshake:
  1. IMU generates its complete arrays, opens/settles the Teensy serial port,
     then creates IMU_READY.
  2. PocketSDR auto-starts only after IMU_READY.
  3. PocketSDR holds its first valid TCA packet and publishes its obssec.
  4. IMU converts JSON UTC start -> GPS TOW, selects the precomputed IMU sample
     representing that obssec, and creates IMU_ALIGNED.
  5. PocketSDR creates GO and transmits the held first TCA packet.
  6. IMU detects GO and transmits from the aligned sample as stream index 0.

The GNSS JSON remains configured independently in the GNSS simulator/PocketSDR
setup. --trajectory is passed only to the IMU simulator.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def terminate_process(proc, name):
    if proc is None or proc.poll() is not None:
        return
    print(f"Stopping {name} (pid={proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Launch PocketSDR GUI + 200 Hz IMU with first-sample "
            "trajectory synchronization"
        )
    )
    parser.add_argument("--pocket", required=True,
                        help="Path to pocket_sdr_first_sample_sync.py")
    parser.add_argument("--imu", required=True,
                        help="Path to imu_simulator_direct_teensy_first_sample_sync.py")
    parser.add_argument("-t", "--trajectory", required=True,
                        help="SignalSim JSON used ONLY by the IMU simulator")
    parser.add_argument("--serial-port", required=True,
                        help="Teensy IMU serial port, e.g. /dev/ttyUSB1")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--fs", type=float, default=200.0)
    parser.add_argument("--fs-gps", type=float, default=10.0)
    parser.add_argument(
        "--imu-accuracy",
        default="jz105-rlg",
        choices=["low-accuracy", "mid-accuracy", "high-accuracy", "jz105-rlg"],
    )
    parser.add_argument(
        "--pocket-python", default=sys.executable,
        help="PocketSDR Python interpreter/venv; default: current Python",
    )
    parser.add_argument(
        "--imu-python", default=sys.executable,
        help="IMU Python interpreter/venv; default: current Python",
    )
    parser.add_argument(
        "--sync-timeout", type=float, default=0,
        help="Timeout for each synchronization stage; 0 = indefinitely",
    )
    parser.add_argument(
        "--trajectory-start-obssec", type=float, default=None,
        help=(
            "Optional override for trajectory start GPS obssec/TOW. "
            "Normally derived from the IMU JSON UTC start time."
        ),
    )
    parser.add_argument(
        "--sync-dir", default="",
        help="Optional marker directory; a temporary directory is used by default",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot", dest="plot", action="store_true",
        help="Show reference and IMU-estimated trajectory plots",
    )
    plot_group.add_argument(
        "--no-plot", dest="plot", action="store_false",
        help="Do not show IMU plots after streaming",
    )
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    if args.fs <= 0.0:
        parser.error("--fs must be > 0")
    if args.sync_timeout < 0.0:
        parser.error("--sync-timeout must be >= 0")

    pocket_script = Path(args.pocket).expanduser().resolve()
    imu_script = Path(args.imu).expanduser().resolve()
    trajectory = Path(args.trajectory).expanduser().resolve()

    for label, path in (
        ("PocketSDR script", pocket_script),
        ("IMU script", imu_script),
        ("trajectory JSON", trajectory),
    ):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")

    made_temp_dir = not bool(args.sync_dir)
    sync_dir = (
        Path(tempfile.mkdtemp(prefix="pocket_imu_first_sample_sync_"))
        if made_temp_dir
        else Path(args.sync_dir).expanduser().resolve()
    )
    sync_dir.mkdir(parents=True, exist_ok=True)

    ready_file = sync_dir / "imu_ready.flag"
    epoch_file = sync_dir / "first_gnss_obssec.txt"
    aligned_file = sync_dir / "imu_aligned.flag"
    go_file = sync_dir / "go.flag"
    tca_trajectory_file = sync_dir / "tca_trajectory.csv"
    markers = (ready_file, epoch_file, aligned_file, go_file, tca_trajectory_file)

    for marker in markers:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass

    # Preserve caller environment, including the user's TCA settings such as:
    # POCKETSDR_TCA_PORT, POCKETSDR_TCA_BAUDRATE and POCKETSDR_TCA_ENABLE.
    pocket_env = os.environ.copy()
    pocket_env["POCKETSDR_AUTOSTART"] = "1"
    pocket_env["POCKETSDR_SIM_READY_FILE"] = str(ready_file)
    pocket_env["POCKETSDR_SIM_EPOCH_FILE"] = str(epoch_file)
    pocket_env["POCKETSDR_SIM_ALIGNED_FILE"] = str(aligned_file)
    pocket_env["POCKETSDR_SIM_GO_FILE"] = str(go_file)
    pocket_env.setdefault("POCKETSDR_SIM_SYNC_POLL_MS", "2")
    pocket_env.setdefault("POCKETSDR_TCA_SERIAL_CYCLE_MS", "5")
    pocket_env["POCKETSDR_TCA_PVT_CSV_ENABLE"] = "1"
    pocket_env["POCKETSDR_TCA_PVT_CSV_PATH"] = str(tca_trajectory_file)

    imu_cmd = [
        "taskset", "-c", "10",
        args.imu_python, str(imu_script),
        "-t", str(trajectory),
        "--serial-port", args.serial_port,
        "--baud", str(args.baud),
        "--fs", str(args.fs),
        "--fs-gps", str(args.fs_gps),
        "--imu-accuracy", args.imu_accuracy,
        "--sync-ready-file", str(ready_file),
        "--sync-epoch-file", str(epoch_file),
        "--sync-aligned-file", str(aligned_file),
        "--sync-go-file", str(go_file),
        "--sync-timeout", str(args.sync_timeout),
        "--tca-trajectory", str(tca_trajectory_file),
    ]
    if args.trajectory_start_obssec is not None:
        imu_cmd += [
            "--trajectory-start-obssec", str(args.trajectory_start_obssec)
        ]
    if not args.plot:
        imu_cmd.append("--no-plot")

    pocket_cmd = [ "taskset", "-c", "4-9", args.pocket_python, str(pocket_script)]

    print("=" * 76)
    print("POCKETSDR + IMU FIRST-SAMPLE TRAJECTORY SYNCHRONIZATION")
    print("=" * 76)
    print(f"PocketSDR : {pocket_script}")
    print(f"Pocket cwd: {pocket_script.parent}")
    print(f"IMU       : {imu_script}")
    print(f"IMU cwd   : {imu_script.parent}")
    print(f"Trajectory: {trajectory}  (IMU only)")
    print(f"IMU port  : {args.serial_port} @ {args.baud}")
    print(f"IMU rate  : {args.fs} Hz")
    print(f"READY     : {ready_file}")
    print(f"EPOCH     : {epoch_file}")
    print(f"ALIGNED   : {aligned_file}")
    print(f"GO        : {go_file}")
    print("=" * 76)

    imu_proc = None
    pocket_proc = None
    imu_done_reported = False

    try:
        imu_proc = subprocess.Popen(
            imu_cmd,
            cwd=str(imu_script.parent),
            env=os.environ.copy(),
        )
        pocket_proc = subprocess.Popen(
            pocket_cmd,
            cwd=str(pocket_script.parent),
            env=pocket_env,
        )

        print(
            f"Processes launched: IMU pid={imu_proc.pid}, "
            f"PocketSDR pid={pocket_proc.pid}"
        )

        while True:
            imu_rc = imu_proc.poll()
            pocket_rc = pocket_proc.poll()

            if pocket_rc is not None:
                print(f"PocketSDR exited with code {pocket_rc}")
                terminate_process(imu_proc, "IMU simulator")
                return pocket_rc

            if imu_rc is not None:
                if imu_rc != 0:
                    print(f"IMU simulator exited with error code {imu_rc}")
                    terminate_process(pocket_proc, "PocketSDR")
                    return imu_rc
                if not imu_done_reported:
                    print(
                        "Aligned IMU transmission completed successfully. "
                        "PocketSDR remains open until you close it normally."
                    )
                    imu_done_reported = True

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        terminate_process(imu_proc, "IMU simulator")
        terminate_process(pocket_proc, "PocketSDR")
        return 130
    finally:
        for marker in markers:
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
        if made_temp_dir:
            try:
                shutil.rmtree(sync_dir)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
