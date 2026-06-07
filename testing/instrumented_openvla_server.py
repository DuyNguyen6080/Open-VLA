#!/usr/bin/env python3
"""Instrumented OpenVLA REST server for deployment metric collection.

This test server mirrors Final-OpenVLA/server/main.py but returns timing and GPU
metrics with each /act response. Use it for report measurements, then keep the
production server simple.
"""

import argparse
import importlib.util
import json
import os
import os.path
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

json_numpy = None
torch = None
uvicorn = None
FastAPI = None
JSONResponse = None
Image = None
AutoModelForVision2Seq = None
AutoProcessor = None


def load_runtime_dependencies() -> None:
    """Import server dependencies only after argparse handles --help."""
    global json_numpy, torch, uvicorn, FastAPI, JSONResponse
    global Image, AutoModelForVision2Seq, AutoProcessor

    import json_numpy as json_numpy_module
    import torch as torch_module
    import uvicorn as uvicorn_module
    from fastapi import FastAPI as FastAPI_class
    from fastapi.responses import JSONResponse as JSONResponse_class
    from PIL import Image as Image_class
    from transformers import AutoModelForVision2Seq as AutoModelForVision2Seq_class
    from transformers import AutoProcessor as AutoProcessor_class

    json_numpy_module.patch()
    json_numpy = json_numpy_module
    torch = torch_module
    uvicorn = uvicorn_module
    FastAPI = FastAPI_class
    JSONResponse = JSONResponse_class
    Image = Image_class
    AutoModelForVision2Seq = AutoModelForVision2Seq_class
    AutoProcessor = AutoProcessor_class

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def get_openvla_prompt(instruction: str, openvla_path: Any) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def sample_gpu() -> Dict[str, Optional[float]]:
    if shutil.which("nvidia-smi") is None:
        return {
            "gpu_util_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
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
            "gpu_util_percent": util,
            "gpu_memory_used_mib": mem_used,
            "gpu_memory_total_mib": mem_total,
        }
    except Exception:
        return {
            "gpu_util_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
        }


class InstrumentedOpenVLAServer:
    def __init__(self, openvla_path: Any, attn_implementation: str = "auto", require_cuda: bool = False):
        if torch is None or AutoProcessor is None or AutoModelForVision2Seq is None:
            raise RuntimeError("Runtime dependencies were not loaded")

        self.openvla_path = openvla_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.gpu_name = None
        self.gpu_major = 0
        self.gpu_minor = 0
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_major, self.gpu_minor = torch.cuda.get_device_capability(0)
            self.dtype = torch.bfloat16 if self.gpu_major >= 8 else torch.float16
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
        else:
            self.dtype = torch.float32

        if require_cuda and not torch.cuda.is_available():
            print("CUDA was required but is not available to PyTorch.", flush=True)
            print("Diagnostics:", flush=True)
            print("  torch_version={}".format(getattr(torch, "__version__", None)), flush=True)
            print("  torch_cuda_version={}".format(getattr(torch.version, "cuda", None)), flush=True)
            print("  CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
            print("  nvidia_smi_path={}".format(shutil.which("nvidia-smi")), flush=True)
            if shutil.which("nvidia-smi") is not None:
                try:
                    print(
                        subprocess.check_output(["nvidia-smi"], universal_newlines=True, timeout=5.0),
                        flush=True,
                    )
                except Exception as exc:
                    print("  nvidia-smi failed: {}".format(exc), flush=True)
            raise RuntimeError(
                "CUDA is not available inside this Python/Docker environment. "
                "Check Docker GPU passthrough and install a CUDA-enabled PyTorch build."
            )

        selected_attn = None
        if attn_implementation == "auto":
            if (
                torch.cuda.is_available()
                and self.gpu_major >= 8
                and importlib.util.find_spec("flash_attn") is not None
            ):
                selected_attn = "flash_attention_2"
            elif torch.cuda.is_available():
                selected_attn = "sdpa"
        elif attn_implementation not in ("none", "None", ""):
            selected_attn = attn_implementation
        self.attn_implementation = selected_attn

        print("OpenVLA metric server configuration:", flush=True)
        print("  openvla_path={}".format(self.openvla_path), flush=True)
        print("  device={}".format(self.device), flush=True)
        print("  cuda_available={}".format(torch.cuda.is_available()), flush=True)
        print("  torch_version={}".format(getattr(torch, "__version__", None)), flush=True)
        print("  torch_cuda_version={}".format(getattr(torch.version, "cuda", None)), flush=True)
        print("  CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
        print("  nvidia_smi_path={}".format(shutil.which("nvidia-smi")), flush=True)
        if torch.cuda.is_available():
            print("  gpu_name={}".format(self.gpu_name), flush=True)
            print("  gpu_capability={}.{}".format(self.gpu_major, self.gpu_minor), flush=True)
            print("  dtype={}".format(self.dtype), flush=True)
            print("  attn_implementation={}".format(self.attn_implementation), flush=True)
            print(
                "  cuda_memory_allocated_mib={:.1f}".format(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                ),
                flush=True,
            )
            print(
                "  cuda_memory_reserved_mib={:.1f}".format(
                    torch.cuda.memory_reserved() / (1024 * 1024)
                ),
                flush=True,
            )
        else:
            print("  gpu_name=None", flush=True)
            print("  dtype={}".format(self.dtype), flush=True)
            print("  attn_implementation={}".format(self.attn_implementation), flush=True)

        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            **model_kwargs
        ).to(self.device)
        self.vla.eval()

        print("OpenVLA model loaded:", flush=True)
        print("  device={}".format(self.device), flush=True)
        print("  dtype={}".format(self.dtype), flush=True)
        if torch.cuda.is_available():
            print(
                "  cuda_memory_allocated_mib={:.1f}".format(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                ),
                flush=True,
            )
            print(
                "  cuda_memory_reserved_mib={:.1f}".format(
                    torch.cuda.memory_reserved() / (1024 * 1024)
                ),
                flush=True,
            )

        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as stats_file:
                self.vla.norm_stats = json.load(stats_file)

    def predict_action(self, payload: Dict[str, Any]):
        if torch is None or Image is None or JSONResponse is None:
            raise RuntimeError("Runtime dependencies were not loaded")

        total_start = now_ms()
        try:
            double_encode = "encoded" in payload
            if double_encode:
                payload = json.loads(payload["encoded"])

            parse_done = now_ms()
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(
                prompt,
                Image.fromarray(image).convert("RGB"),
            ).to(self.device, dtype=self.dtype)
            preprocess_done = now_ms()

            gpu_before = sample_gpu()
            with torch.no_grad():
                action = self.vla.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_done = now_ms()
            gpu_after = sample_gpu()

            metrics = {
                "server_parse_ms": parse_done - total_start,
                "preprocess_ms": preprocess_done - parse_done,
                "inference_ms": inference_done - preprocess_done,
                "total_ms": inference_done - total_start,
                "gpu_util_percent": gpu_after["gpu_util_percent"],
                "gpu_memory_used_mib": gpu_after["gpu_memory_used_mib"],
                "gpu_memory_total_mib": gpu_after["gpu_memory_total_mib"],
                "gpu_util_before_percent": gpu_before["gpu_util_percent"],
                "gpu_memory_before_mib": gpu_before["gpu_memory_used_mib"],
                "device": str(self.device),
                "gpu_name": self.gpu_name,
                "gpu_capability": (
                    "{}.{}".format(self.gpu_major, self.gpu_minor)
                    if torch.cuda.is_available()
                    else None
                ),
                "dtype": str(self.dtype),
                "attn_implementation": self.attn_implementation,
                "torch_cuda_memory_allocated_mib": (
                    torch.cuda.memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available()
                    else None
                ),
                "torch_cuda_memory_reserved_mib": (
                    torch.cuda.memory_reserved() / (1024 * 1024)
                    if torch.cuda.is_available()
                    else None
                ),
            }

            if double_encode:
                print(
                    "Request metrics: device={} dtype={} preprocess_ms={:.1f} "
                    "inference_ms={:.1f} total_ms={:.1f} gpu_util={} gpu_mem_mib={}".format(
                        self.device,
                        self.dtype,
                        metrics["preprocess_ms"],
                        metrics["inference_ms"],
                        metrics["total_ms"],
                        metrics["gpu_util_percent"],
                        metrics["gpu_memory_used_mib"],
                    ),
                    flush=True,
                )
                return JSONResponse(json_numpy.dumps({"action": action, "metrics": metrics}))
            print(
                "Request metrics: device={} dtype={} preprocess_ms={:.1f} "
                "inference_ms={:.1f} total_ms={:.1f} gpu_util={} gpu_mem_mib={}".format(
                    self.device,
                    self.dtype,
                    metrics["preprocess_ms"],
                    metrics["inference_ms"],
                    metrics["total_ms"],
                    metrics["gpu_util_percent"],
                    metrics["gpu_memory_used_mib"],
                ),
                flush=True,
            )
            return JSONResponse({"action": action, "metrics": metrics})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    def run(self, host: str, port: int) -> None:
        if FastAPI is None or uvicorn is None:
            raise RuntimeError("Runtime dependencies were not loaded")

        app = FastAPI()
        app.post("/act")(self.predict_action)
        uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an instrumented OpenVLA REST server.")
    parser.add_argument("--openvla-path", default="openvla/openvla-7b")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--attn-implementation", default="auto")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail at startup instead of falling back to CPU when CUDA is unavailable.",
    )
    args = parser.parse_args()
    load_runtime_dependencies()

    server = InstrumentedOpenVLAServer(
        args.openvla_path,
        attn_implementation=args.attn_implementation,
        require_cuda=args.require_cuda,
    )
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
