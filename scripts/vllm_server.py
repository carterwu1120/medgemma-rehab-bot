import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start local vLLM server using values from .env by default."
    )
    parser.add_argument("--model", default=None, help="Model ID. Defaults to VLLM_MODEL in .env")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    args = parse_args()
    model = args.model or os.getenv("VLLM_MODEL")
    if not model:
        print("VLLM_MODEL is not set. Add it to .env or pass --model.", file=sys.stderr)
        sys.exit(1)

    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        print("Cannot find `vllm` in PATH. Activate your .venv-vllm first.", file=sys.stderr)
        sys.exit(1)

    max_model_len = args.max_model_len or int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    gpu_memory_utilization = args.gpu_memory_utilization or float(
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")
    )
    tensor_parallel_size = args.tensor_parallel_size or int(os.getenv("VLLM_TP_SIZE", "1"))

    env = os.environ.copy()
    if env.get("HF_TOKEN") and not env.get("HUGGING_FACE_HUB_TOKEN"):
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]

    cmd = [
        vllm_bin,
        "serve",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
    ]

    print("Starting vLLM server with:")
    print(" ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
