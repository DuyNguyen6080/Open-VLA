#!/usr/bin/env python3
"""Compute deployment/trial metrics from UR3e RTDE CSV logs.

This script is intended for report support, not robot control. It reads trial CSV
files that contain actual_TCP_pose_0..2 and optional target x/y/z columns, then
prints duration, path length, displacement, and distance-to-target statistics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


POSE_COLUMNS = ("actual_TCP_pose_0", "actual_TCP_pose_1", "actual_TCP_pose_2")
TARGET_COLUMNS = ("target x", "target y", "target z")


def parse_float(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if value == "":
        return None
    return float(value)


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def read_xyz(row: dict[str, str], columns: tuple[str, str, str]) -> tuple[float, float, float] | None:
    values = [parse_float(row, column) for column in columns]
    if any(value is None for value in values):
        return None
    return values[0], values[1], values[2]


def analyze_csv(path: Path, fill_forward_target: bool) -> dict[str, object]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"{path} has no rows")

    points: list[tuple[float, float, float]] = []
    targets: list[tuple[float, float, float] | None] = []
    last_target: tuple[float, float, float] | None = None
    missing_targets = 0

    for row in rows:
        point = read_xyz(row, POSE_COLUMNS)
        if point is None:
            raise ValueError(f"{path} has a row with missing TCP pose columns")
        points.append(point)

        target = read_xyz(row, TARGET_COLUMNS)
        if target is None:
            missing_targets += 1
            targets.append(last_target if fill_forward_target else None)
        else:
            last_target = target
            targets.append(target)

    duration_s = float(rows[-1]["timestamp"]) - float(rows[0]["timestamp"])
    path_length_m = sum(distance(points[i], points[i - 1]) for i in range(1, len(points)))
    net_displacement_m = distance(points[-1], points[0])

    distances = [
        distance(point, target)
        for point, target in zip(points, targets)
        if target is not None
    ]

    return {
        "file": path.name,
        "rows": len(rows),
        "valid_target_rows": len(rows) - missing_targets,
        "missing_target_rows": missing_targets,
        "duration_s": duration_s,
        "path_length_m": path_length_m,
        "net_displacement_m": net_displacement_m,
        "initial_dist_m": distances[0] if distances else None,
        "min_dist_m": min(distances) if distances else None,
        "last_available_dist_m": distances[-1] if distances else None,
    }


def format_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze UR3e OpenVLA trial CSV files.")
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument(
        "--fill-forward-target",
        action="store_true",
        help="Use the previous non-empty target when later target cells are blank.",
    )
    args = parser.parse_args()

    columns = [
        "file",
        "rows",
        "valid_target_rows",
        "missing_target_rows",
        "duration_s",
        "path_length_m",
        "net_displacement_m",
        "initial_dist_m",
        "min_dist_m",
        "last_available_dist_m",
    ]
    print(",".join(columns))
    for csv_path in args.csv_files:
        result = analyze_csv(csv_path, args.fill_forward_target)
        print(",".join(format_value(result[column]) for column in columns))


if __name__ == "__main__":
    main()
