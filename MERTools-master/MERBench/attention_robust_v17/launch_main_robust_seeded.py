#!/usr/bin/env python3
import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Best effort reproducibility for CUDA matmul.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_torch_threads_from_env() -> None:
    # PyTorch recommends setting thread counts explicitly for CPU stability.
    # We prioritize TORCH_* vars, then fall back to OMP settings.
    num_threads = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    interop_threads = os.environ.get("TORCH_NUM_INTEROP_THREADS")

    if num_threads:
        try:
            torch.set_num_threads(max(1, int(num_threads)))
        except Exception:
            pass
    if interop_threads:
        try:
            torch.set_num_interop_threads(max(1, int(interop_threads)))
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seeded launcher for main-robust.py without modifying core training code."
    )
    parser.add_argument("--seed", type=int, required=True, help="global random seed")
    parser.add_argument(
        "--main_robust",
        type=str,
        default="main-robust.py",
        help="path to main-robust.py",
    )
    args, unknown = parser.parse_known_args()

    set_seed(args.seed)
    configure_torch_threads_from_env()
    main_path = Path(args.main_robust).resolve()
    main_dir = str(main_path.parent)
    if main_dir not in sys.path:
        sys.path.insert(0, main_dir)

    sys.argv = [str(main_path)] + unknown
    runpy.run_path(str(main_path), run_name="__main__")


if __name__ == "__main__":
    main()
