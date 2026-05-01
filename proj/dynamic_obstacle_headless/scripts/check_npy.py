#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def summarize_file(path: Path) -> None:
    arr = np.load(path)
    finite_mask = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if finite_mask.any():
        finite_vals = arr[finite_mask]
        min_val = float(finite_vals.min())
        max_val = float(finite_vals.max())
        mean_val = float(finite_vals.mean())
    else:
        min_val = math.nan
        max_val = math.nan
        mean_val = math.nan

    print(f"file: {path}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  min: {min_val}")
    print(f"  max: {max_val}")
    print(f"  mean: {mean_val}")
    print(f"  NaNs: {nan_count}")
    print(f"  Infs: {inf_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OFM headless .npy outputs")
    parser.add_argument("path", type=Path, help="Path to a .npy file or directory containing .npy files")
    args = parser.parse_args()

    path = args.path
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.npy"))

    if not files:
        raise RuntimeError(f"No .npy files found in: {path}")

    for file in files:
        summarize_file(file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
