#!/usr/bin/env python3
"""
Convert a SignalSim JSON configuration file into a gnss-ins-sim motion profile CSV.

Usage:
    python signalsim_to_gnss_ins_sim.py test_obs2.json motion_def.csv --audit audit.csv

Notes:
- Converts trajectory kinematics only: initial LLA, initial velocity, and trajectoryList.
- SignalSim satellite settings, ephemeris, RINEX output, signal masks, and power settings
  have no direct gnss-ins-sim motion-profile equivalent and are ignored.
- SignalSim has no roll/pitch/yaw attitude. This converter assumes:
      roll = 0 deg, pitch = 0 deg, yaw = SignalSim course
  and aligns the vehicle body x-axis with the horizontal track direction.
- SignalSim vertical velocity is up-positive. gnss-ins-sim/NED convention is down-positive,
  so vz_body = -up.
- SignalSim Jerk segments are approximated by an equivalent constant acceleration that
  preserves the same delta-V over the segment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

EPS = 1e-12


def speed_to_mps(value: float, unit: str | None = "mps") -> float:
    """Convert SignalSim speed units to m/s."""
    unit = (unit or "mps").lower()
    value = float(value)
    factors = {
        "mps": 1.0,
        "m/s": 1.0,
        "kph": 1000.0 / 3600.0,
        "km/h": 1000.0 / 3600.0,
        "knot": 0.5144444444444445,
        "knots": 0.5144444444444445,
        "mph": 0.44704,
    }
    if unit not in factors:
        raise ValueError(f"Unsupported speedUnit: {unit}")
    return value * factors[unit]


def angle_to_deg(value: float, unit: str | None = "degree") -> float:
    """Convert SignalSim angle units to degrees."""
    unit = (unit or "degree").lower()
    value = float(value)
    if unit in {"degree", "deg", "degrees"}:
        return value
    if unit in {"rad", "radian", "radians"}:
        return math.degrees(value)
    raise ValueError(f"Unsupported angleUnit: {unit}")


def parse_lla_angle(value: float, fmt: str | None = "d") -> float:
    """Parse SignalSim LLA angle formats: d, dm, dms, rad."""
    fmt = (fmt or "d").lower()
    v = float(value)
    sign = -1.0 if v < 0 else 1.0
    a = abs(v)

    if fmt == "d":
        return v
    if fmt == "rad":
        return math.degrees(v)
    if fmt == "dm":
        deg = math.floor(a / 100.0)
        minutes = a - deg * 100.0
        return sign * (deg + minutes / 60.0)
    if fmt == "dms":
        deg = math.floor(a / 10000.0)
        rem = a - deg * 10000.0
        minutes = math.floor(rem / 100.0)
        seconds = rem - minutes * 100.0
        return sign * (deg + minutes / 60.0 + seconds / 3600.0)

    raise ValueError(f"Unsupported LLA format: {fmt}")


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """WGS84 ECEF [m] to LLA [deg, deg, m]."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)

    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - e2))

    for _ in range(10):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = p / max(math.cos(lat), EPS) - N
        lat_new = math.atan2(z, p * (1.0 - e2 * N / (N + alt)))
        if abs(lat_new - lat) < 1e-14:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), EPS) - N
    return math.degrees(lat), math.degrees(lon), alt


def ecef_velocity_to_enu(
    vx: float, vy: float, vz: float, lat_deg: float, lon_deg: float
) -> Tuple[float, float, float]:
    """Convert ECEF velocity [m/s] to local ENU velocity [m/s]."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    east = -sin_lon * vx + cos_lon * vy
    north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz
    up = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz
    return east, north, up


def read_position(traj: Dict[str, Any]) -> Tuple[float, float, float]:
    pos = traj["initPosition"]
    ptype = pos.get("type", "LLA").upper()

    if ptype == "LLA":
        fmt = pos.get("format", "d")
        lat = parse_lla_angle(pos["latitude"], fmt)
        lon = parse_lla_angle(pos["longitude"], fmt)
        alt = float(pos.get("altitude", 0.0))
        return lat, lon, alt

    if ptype == "ECEF":
        return ecef_to_lla(float(pos["x"]), float(pos["y"]), float(pos["z"]))

    raise ValueError(f"Unsupported initPosition type: {ptype}")


def read_initial_velocity(
    traj: Dict[str, Any], lat_deg: float, lon_deg: float
) -> Tuple[float, float, float]:
    """
    Return SignalSim initial velocity as:
        horizontal_speed_mps, course_deg_clockwise_from_north, up_speed_mps
    """
    vel = traj.get("initVelocity", {})
    vtype = vel.get("type", "SCU").upper()
    speed_unit = vel.get("speedUnit", "mps")
    angle_unit = vel.get("angleUnit", "degree")

    if vtype == "SCU":
        h_speed = speed_to_mps(vel.get("speed", 0.0), speed_unit)
        course = angle_to_deg(vel.get("course", 0.0), angle_unit) % 360.0
        up = speed_to_mps(vel.get("up", 0.0), speed_unit)
        return h_speed, course, up

    if vtype == "ENU":
        east = speed_to_mps(vel.get("east", 0.0), speed_unit)
        north = speed_to_mps(vel.get("north", 0.0), speed_unit)
        up = speed_to_mps(vel.get("up", 0.0), speed_unit)
        h_speed = math.hypot(east, north)
        course = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
        return h_speed, course, up

    if vtype == "ECEF":
        vx = speed_to_mps(vel.get("x", 0.0), speed_unit)
        vy = speed_to_mps(vel.get("y", 0.0), speed_unit)
        vz = speed_to_mps(vel.get("z", 0.0), speed_unit)
        east, north, up = ecef_velocity_to_enu(vx, vy, vz, lat_deg, lon_deg)
        h_speed = math.hypot(east, north)
        course = (math.degrees(math.atan2(east, north)) + 360.0) % 360.0
        return h_speed, course, up

    raise ValueError(f"Unsupported initVelocity type: {vtype}")


def fmt(x: float) -> str:
    """Compact numeric formatting for CSV."""
    if abs(x) < 5e-12:
        x = 0.0
    return f"{x:.12g}"


def add_row(
    rows: List[List[str]],
    command_type: int,
    yaw: float,
    pitch: float,
    roll: float,
    vx: float,
    vy: float,
    vz: float,
    duration: float,
    gps_visible: int = 1,
) -> None:
    if duration is None or duration <= EPS:
        return
    rows.append(
        [
            str(command_type),
            fmt(yaw),
            fmt(pitch),
            fmt(roll),
            fmt(vx),
            fmt(vy),
            fmt(vz),
            fmt(duration),
            str(gps_visible),
        ]
    )


def solve_const_acc_segment(current_speed: float, seg: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Solve SignalSim ConstAcc/VerticalAcc into duration, signed acceleration, target speed.
    SignalSim accepts two of: time, acceleration, speed.
    """
    has_t = "time" in seg
    has_a = "acceleration" in seg
    has_s = "speed" in seg

    if has_t and has_a:
        duration = float(seg["time"])
        accel = float(seg["acceleration"])
        target_speed = current_speed + accel * duration
        return duration, accel, target_speed

    if has_t and has_s:
        duration = float(seg["time"])
        target_speed = float(seg["speed"])
        accel = (target_speed - current_speed) / max(duration, EPS)
        return duration, accel, target_speed

    if has_a and has_s:
        target_speed = float(seg["speed"])
        accel_mag = abs(float(seg["acceleration"]))
        delta = target_speed - current_speed
        if abs(delta) <= EPS:
            return 0.0, 0.0, target_speed
        if accel_mag <= EPS:
            raise ValueError(f"Cannot reach speed {target_speed} with zero acceleration: {seg}")
        accel = math.copysign(accel_mag, delta)
        duration = abs(delta) / accel_mag
        return duration, accel, target_speed

    raise ValueError(f"Invalid acceleration segment; need two of time/acceleration/speed: {seg}")


def solve_jerk_segment(
    current_accel: float, seg: Dict[str, Any]
) -> Tuple[float, float, float]:
    """
    Approximate SignalSim Jerk with equivalent constant acceleration.

    Returns:
        duration, equivalent_acceleration, target_acceleration
    """
    has_t = "time" in seg
    has_r = "rate" in seg
    has_a = "acceleration" in seg

    if has_t and has_a:
        duration = float(seg["time"])
        target_accel = float(seg["acceleration"])
    elif has_t and has_r:
        duration = float(seg["time"])
        target_accel = current_accel + float(seg["rate"]) * duration
    elif has_r and has_a:
        target_accel = float(seg["acceleration"])
        rate = abs(float(seg["rate"]))
        if rate <= EPS:
            raise ValueError(f"Cannot solve jerk duration with zero rate: {seg}")
        duration = abs(target_accel - current_accel) / rate
    else:
        raise ValueError(f"Invalid jerk segment; need two of time/rate/acceleration: {seg}")

    equivalent_accel = 0.5 * (current_accel + target_accel)
    return duration, equivalent_accel, target_accel


def solve_horizontal_turn(
    h_speed: float, seg: Dict[str, Any], angle_unit: str = "degree"
) -> Tuple[float, float]:
    """
    Solve SignalSim HorizontalTurn into yaw angle [deg] and duration [s].
    Positive yaw angle means clockwise/right turn from north, matching SignalSim course.
    """
    has_t = "time" in seg
    has_angle = "angle" in seg
    has_acc = "acceleration" in seg
    has_rate = "rate" in seg
    has_radius = "radius" in seg

    if has_angle:
        angle_deg = angle_to_deg(seg["angle"], angle_unit)

        if has_t:
            return angle_deg, float(seg["time"])

        angle_rad_abs = abs(math.radians(angle_deg))

        if has_rate:
            rate_deg_s = abs(angle_to_deg(seg["rate"], angle_unit))
            if rate_deg_s <= EPS:
                raise ValueError(f"HorizontalTurn has zero rate: {seg}")
            return angle_deg, abs(angle_deg) / rate_deg_s

        if has_acc:
            acc = abs(float(seg["acceleration"]))
            if h_speed <= EPS or acc <= EPS:
                raise ValueError(f"Need nonzero speed and acceleration for turn duration: {seg}")
            # centripetal acceleration a = v * omega, so time = angle / omega = angle*v/a
            return angle_deg, angle_rad_abs * h_speed / acc

        if has_radius:
            radius = abs(float(seg["radius"]))
            if h_speed <= EPS or radius <= EPS:
                raise ValueError(f"Need nonzero speed and radius for turn duration: {seg}")
            return angle_deg, angle_rad_abs * radius / h_speed

        raise ValueError(f"HorizontalTurn with angle also needs time/rate/acceleration/radius: {seg}")

    # No angle given: derive angle from time plus rate/acceleration/radius.
    if not has_t:
        raise ValueError(f"HorizontalTurn without angle needs time plus rate/acceleration/radius: {seg}")

    duration = float(seg["time"])

    if has_rate:
        rate_deg_s = angle_to_deg(seg["rate"], angle_unit)
        return rate_deg_s * duration, duration

    if has_acc:
        acc = float(seg["acceleration"])
        if h_speed <= EPS:
            raise ValueError(f"Need nonzero speed for acceleration-based turn: {seg}")
        angle_rad = acc / h_speed * duration
        return math.degrees(angle_rad), duration

    if has_radius:
        radius = float(seg["radius"])
        if h_speed <= EPS or abs(radius) <= EPS:
            raise ValueError(f"Need nonzero speed and radius for radius-based turn: {seg}")
        # sign of radius controls turn direction when angle is absent.
        angle_rad = h_speed / radius * duration
        return math.degrees(angle_rad), duration

    raise ValueError(f"Invalid HorizontalTurn: {seg}")


def pick_trajectory(config: Dict[str, Any], trajectory_name: str | None = None) -> Dict[str, Any]:
    traj = config.get("trajectory")
    if traj is None:
        raise ValueError("SignalSim config has no 'trajectory' object.")

    if isinstance(traj, dict):
        return traj

    if isinstance(traj, list):
        if trajectory_name:
            for t in traj:
                if t.get("name") == trajectory_name:
                    return t
            raise ValueError(f"No trajectory named {trajectory_name!r} found.")
        return traj[0]

    raise ValueError("'trajectory' must be an object or array.")


def convert_signalsim_to_gnss_motion(
    config: Dict[str, Any], trajectory_name: str | None = None, gps_visible: int = 1
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    traj = pick_trajectory(config, trajectory_name)
    lat, lon, alt = read_position(traj)
    h_speed, course, up_speed = read_initial_velocity(traj, lat, lon)

    rows: List[List[str]] = []
    audit: List[Dict[str, Any]] = []

    rows.append(
        [
            "Ini lat (deg)",
            "ini lon (deg)",
            "ini alt (m)",
            "ini vx_body (m/s)",
            "ini vy_body (m/s)",
            "ini vz_body (m/s)",
            "ini yaw (deg)",
            "ini pitch (deg)",
            "ini roll (deg)",
        ]
    )
    rows.append([fmt(lat), fmt(lon), fmt(alt), fmt(h_speed), "0", fmt(-up_speed), fmt(course), "0", "0"])
    rows.append(
        [
            "command type",
            "yaw (deg)",
            "pitch (deg)",
            "roll (deg)",
            "vx_body (m/s)",
            "vy_body (m/s)",
            "vz_body (m/s)",
            "command duration (s)",
            "GPS visibility",
        ]
    )

    # SignalSim has angleUnit under initVelocity; use it for trajectory angles if present.
    angle_unit = traj.get("initVelocity", {}).get("angleUnit", "degree")

    h_accel = 0.0  # SignalSim acceleration along horizontal velocity direction, m/s^2
    v_accel = 0.0  # SignalSim vertical acceleration, up-positive, m/s^2

    for i, seg in enumerate(traj.get("trajectoryList", []), start=1):
        stype = seg.get("type")
        if not stype:
            raise ValueError(f"trajectoryList item {i} has no type: {seg}")
        stype_norm = stype.lower()
        before = {"h_speed": h_speed, "course": course, "up_speed": up_speed, "h_accel": h_accel, "v_accel": v_accel}

        if stype_norm == "const":
            duration = float(seg["time"])
            add_row(rows, 1, 0, 0, 0, 0, 0, 0, duration, gps_visible)
            h_accel = 0.0
            v_accel = 0.0
            note = "constant velocity"

        elif stype_norm == "constacc":
            duration, accel, target_speed = solve_const_acc_segment(h_speed, seg)
            add_row(rows, 1, 0, 0, 0, accel, 0, 0, duration, gps_visible)
            h_speed = target_speed
            h_accel = accel
            v_accel = 0.0
            note = "horizontal acceleration mapped to vx_body rate"

        elif stype_norm == "verticalacc":
            duration, up_accel, target_up = solve_const_acc_segment(up_speed, seg)
            add_row(rows, 1, 0, 0, 0, 0, 0, -up_accel, duration, gps_visible)
            up_speed = target_up
            v_accel = up_accel
            h_accel = 0.0
            note = "up-positive SignalSim vertical accel mapped to negative vz_body rate"

        elif stype_norm == "jerk":
            duration, equiv_accel, target_accel = solve_jerk_segment(h_accel, seg)
            add_row(rows, 1, 0, 0, 0, equiv_accel, 0, 0, duration, gps_visible)
            h_speed += equiv_accel * duration
            h_accel = target_accel
            v_accel = 0.0
            note = "jerk approximated by equivalent constant acceleration preserving delta-V"

        elif stype_norm == "horizontalturn":
            angle_deg, duration = solve_horizontal_turn(h_speed, seg, angle_unit)
            yaw_rate = angle_deg / max(duration, EPS)
            add_row(rows, 1, yaw_rate, 0, 0, 0, 0, 0, duration, gps_visible)
            course = (course + angle_deg) % 360.0
            h_accel = 0.0
            v_accel = 0.0
            note = "turn mapped to yaw-rate command; speed magnitude unchanged"

        else:
            raise ValueError(f"Unsupported SignalSim trajectory segment type {stype!r} at item {i}")

        audit.append(
            {
                "index": i,
                "signalsim_type": stype,
                "signalsim_segment": json.dumps(seg, separators=(",", ":")),
                "note": note,
                "before_h_speed_mps": before["h_speed"],
                "before_course_deg": before["course"],
                "before_up_speed_mps": before["up_speed"],
                "after_h_speed_mps": h_speed,
                "after_course_deg": course,
                "after_up_speed_mps": up_speed,
            }
        )

    return rows, audit


def write_csv(path: Path, rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_audit(path: Path, audit: List[Dict[str, Any]]) -> None:
    if not audit:
        return
    fields = list(audit[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SignalSim JSON trajectory into a gnss-ins-sim motion_def CSV."
    )
    parser.add_argument("signalsim_json", help="Input SignalSim .json config file")
    parser.add_argument("output_csv", help="Output gnss-ins-sim motion profile .csv")
    parser.add_argument("--trajectory-name", help="Trajectory name if the config contains multiple trajectories")
    parser.add_argument("--gps-visible", type=int, default=1, choices=[0, 1], help="GPS visibility value written to every command row")
    parser.add_argument("--audit", help="Optional audit CSV explaining each converted segment")
    args = parser.parse_args()

    input_path = Path(args.signalsim_json)
    output_path = Path(args.output_csv)

    with input_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    rows, audit = convert_signalsim_to_gnss_motion(
        config, trajectory_name=args.trajectory_name, gps_visible=args.gps_visible
    )

    write_csv(output_path, rows)
    if args.audit:
        write_audit(Path(args.audit), audit)

    print(f"Wrote gnss-ins-sim motion profile: {output_path}")
    if args.audit:
        print(f"Wrote conversion audit: {args.audit}")


if __name__ == "__main__":
    main()
