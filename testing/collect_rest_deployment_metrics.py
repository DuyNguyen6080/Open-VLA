#!/usr/bin/env python3
"""Collect REST deployment metrics for the OpenVLA robot pipeline.

Run this from the robot-side/client machine while an OpenVLA REST server is
running. It measures client-side capture, resize, JSON serialization, request
round-trip time, response decode time, payload size, and optional GPU samples on
the local machine.

For true inference latency and server GPU memory/utilization, run this against
testing/instrumented_openvla_server.py, which returns server-side metrics in the
response body.
"""

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

cv2 = None
requests = None


def load_runtime_dependencies() -> None:
    """Import deployment dependencies only after argparse handles --help."""
    global cv2, requests

    import cv2 as cv2_module
    import json_numpy
    import requests as requests_module

    json_numpy.patch()
    cv2 = cv2_module
    requests = requests_module


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def sample_gpu() -> Dict[str, Optional[float]]:
    """Sample local GPU utilization/memory with nvidia-smi if available."""
    if shutil.which("nvidia-smi") is None:
        return {
            "local_gpu_util_percent": None,
            "local_gpu_memory_used_mib": None,
            "local_gpu_memory_total_mib": None,
        }

    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, universal_newlines=True, timeout=2.0)
        first_line = output.strip().splitlines()[0]
        util, mem_used, mem_total = [float(part.strip()) for part in first_line.split(",")]
        return {
            "local_gpu_util_percent": util,
            "local_gpu_memory_used_mib": mem_used,
            "local_gpu_memory_total_mib": mem_total,
        }
    except Exception:
        return {
            "local_gpu_util_percent": None,
            "local_gpu_memory_used_mib": None,
            "local_gpu_memory_total_mib": None,
        }


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def build_frame(camera: Any, cap: Any, image_path: Optional[Path]) -> Any:
    if image_path is not None:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image file: {image_path}")
        return frame

    if cap is None:
        raise RuntimeError("Camera capture was not initialized")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Failed to read frame from camera {camera}")
    return frame


def run_once(
    server_url: str,
    frame: Any,
    instruction: str,
    unnorm_key: str,
    timeout_s: float,
) -> Dict[str, Any]:
    if cv2 is None or requests is None:
        raise RuntimeError("Runtime dependencies were not loaded")

    resize_start = now_ms()
    frame = cv2.resize(frame, (256, 256))
    resize_ms = now_ms() - resize_start

    payload = {
        "image": frame,
        "instruction": instruction,
        "unnorm_key": unnorm_key,
    }

    encode_start = now_ms()
    payload_bytes = json.dumps(payload).encode("utf-8")
    json_encode_ms = now_ms() - encode_start

    gpu_before = sample_gpu()
    request_start = now_ms()
    response = requests.post(server_url, json=payload, timeout=timeout_s)
    roundtrip_ms = now_ms() - request_start
    gpu_after = sample_gpu()

    decode_start = now_ms()
    response_json = response.json()
    response_decode_ms = now_ms() - decode_start

    response_bytes = len(response.content)
    action = response_json.get("action", response_json)
    server_metrics = response_json.get("metrics", {})

    result = {
        "status_code": response.status_code,
        "resize_ms": resize_ms,
        "json_encode_ms": json_encode_ms,
        "request_payload_bytes": len(payload_bytes),
        "response_bytes": response_bytes,
        "roundtrip_ms": roundtrip_ms,
        "response_decode_ms": response_decode_ms,
        "action_length": len(action) if isinstance(action, list) else None,
        "local_gpu_util_before_percent": gpu_before["local_gpu_util_percent"],
        "local_gpu_util_after_percent": gpu_after["local_gpu_util_percent"],
        "local_gpu_memory_before_mib": gpu_before["local_gpu_memory_used_mib"],
        "local_gpu_memory_after_mib": gpu_after["local_gpu_memory_used_mib"],
        **{f"server_{key}": value for key, value in server_metrics.items()},
    }
    if result.get("server_total_ms") is not None:
        result["estimated_communication_overhead_ms"] = (
            result["roundtrip_ms"] - float(result["server_total_ms"])
        )
    return result


def write_summary(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    numeric_keys = [
        "resize_ms",
        "json_encode_ms",
        "roundtrip_ms",
        "estimated_communication_overhead_ms",
        "response_decode_ms",
        "server_preprocess_ms",
        "server_inference_ms",
        "server_total_ms",
        "server_gpu_util_percent",
        "server_gpu_memory_used_mib",
    ]

    print(f"\nWrote {len(rows)} metric rows to {output_csv}")
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and row[key] not in (None, "")
        ]
        stats = summarize(values)
        if stats["mean"] is None:
            continue
        print(
            f"{key}: "
            f"mean={stats['mean']:.2f}, "
            f"median={stats['median']:.2f}, "
            f"min={stats['min']:.2f}, "
            f"max={stats['max']:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect OpenVLA REST deployment metrics.")
    parser.add_argument("--server-url", required=True, help="Full URL, for example http://host:9999/act")
    parser.add_argument("--camera", default="0", help="Camera index or video path")
    parser.add_argument("--image-file", type=Path, help="Use one static image instead of a camera")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--instruction",
        default="pick up the orange object in the middle of the table",
    )
    parser.add_argument("--unnorm-key", default="bridge_orig")
    args = parser.parse_args()
    load_runtime_dependencies()

    try:
        camera = int(args.camera)
    except ValueError:
        camera = args.camera

    cap = None
    if args.image_file is None:
        if cv2 is None:
            raise RuntimeError("cv2 was not loaded")
        cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera/video source: {camera}")

    rows = []  # type: List[Dict[str, Any]]
    try:
        total_iterations = args.warmup + args.iterations
        for index in range(total_iterations):
            capture_start = now_ms()
            frame = build_frame(camera, cap, args.image_file)
            capture_ms = now_ms() - capture_start

            row = run_once(
                args.server_url,
                frame,
                args.instruction,
                args.unnorm_key,
                args.timeout_s,
            )
            row["iteration"] = index - args.warmup + 1
            row["warmup"] = index < args.warmup
            row["capture_ms"] = capture_ms

            if not row["warmup"]:
                rows.append(row)

            print(
                f"iteration={index + 1}/{total_iterations} "
                f"warmup={row['warmup']} "
                f"status={row['status_code']} "
                f"roundtrip_ms={row['roundtrip_ms']:.1f}"
            )
    finally:
        if cap is not None:
            cap.release()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with args.output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
