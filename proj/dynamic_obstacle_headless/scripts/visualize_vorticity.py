#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_vtk(arr: np.ndarray, path: Path):
    try:
        import pyvista as pv

        grid = pv.UniformGrid()
        grid.dimensions = np.array(arr.shape) + 1
        grid.cell_data["vorticity"] = arr.flatten(order="F")
        grid.save(path)

    except Exception as e:
        print(f"[WARN] VTK export failed: {e}")


def save_isosurface(arr: np.ndarray, path: Path, level: float):
    try:
        from skimage.measure import marching_cubes
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        verts, faces, normals, values = marching_cubes(arr, level=level)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        mesh = Poly3DCollection(verts[faces], alpha=0.7)
        mesh.set_facecolor("orange")
        ax.add_collection3d(mesh)

        ax.set_xlim(0, arr.shape[0])
        ax.set_ylim(0, arr.shape[1])
        ax.set_zlim(0, arr.shape[2])

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] Isosurface render failed: {e}")


def parse_index(index_arg: str, axis_size: int) -> int:
    if index_arg == "mid":
        return axis_size // 2
    idx = int(index_arg)
    if idx < 0:
        idx += axis_size
    if idx < 0 or idx >= axis_size:
        raise ValueError(f"Slice index {index_arg} out of range for axis size {axis_size}")
    return idx


def slice_frame(frame: np.ndarray, axis: str, index: int) -> np.ndarray:
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[axis]
    slicer = [slice(None), slice(None), slice(None)]
    slicer[axis_idx] = index
    return frame[tuple(slicer)]


def maybe_write_gif(fig_paths: list[Path], gif_path: Path, fps: int) -> Optional[str]:
    if not fig_paths:
        return None
    try:
        import imageio.v2 as imageio

        frames = [imageio.imread(path) for path in fig_paths]
        imageio.mimsave(gif_path, frames, duration=1.0 / max(fps, 1))
        return f"Saved animated GIF: {gif_path}"
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize OFM headless vorticity .npy outputs")
    parser.add_argument("output_dir", type=Path, help="Directory containing vorticity_*.npy files")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="Slice axis")
    parser.add_argument("--index", default="mid", help="Slice index or 'mid'")
    parser.add_argument("--log", action="store_true", help="Apply log10(vorticity + eps)")
    parser.add_argument("--eps", type=float, default=1e-12, help="epsilon for log scaling")
    parser.add_argument("--gif", action="store_true", help="Also write animated GIF")
    parser.add_argument("--fps", type=int, default=5, help="GIF FPS")
    parser.add_argument("--vtk", action="store_true", help="Export VTK files for ParaView")
    parser.add_argument("--iso", action="store_true", help="Render isosurface images")
    parser.add_argument("--level", type=float, default=None, help="Isosurface level (default: auto)")
    args = parser.parse_args()
    if args.vtk:
        print("[INFO] VTK export requires pyvista")
    if args.iso:
        print("[INFO] Isosurface requires scikit-image")

    npy_files = sorted(args.output_dir.glob("vorticity_*.npy"))
    if not npy_files:
        raise RuntimeError(f"No vorticity_*.npy files found in {args.output_dir}")

    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[Path] = []
    for frame_id, npy_path in enumerate(npy_files):
        arr = np.load(npy_path)
        if args.vtk:
            vtk_path = args.output_dir / f"{npy_path.stem}.vtk"
            save_vtk(arr, vtk_path)

        if args.iso:
            level = args.level
            if level is None:
                level = float(np.percentile(arr[np.isfinite(arr)], 95))

            iso_path = figures_dir / f"{npy_path.stem}_iso.png"
            save_isosurface(arr, iso_path, level)

        finite = np.isfinite(arr)
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())

        if finite.any():
            finite_vals = arr[finite]
            min_val = float(finite_vals.min())
            max_val = float(finite_vals.max())
            mean_val = float(finite_vals.mean())
        else:
            min_val = float("nan")
            max_val = float("nan")
            mean_val = float("nan")

        print(f"frame {frame_id:06d}: {npy_path.name}")
        print(f"  shape={arr.shape}, dtype={arr.dtype}, min={min_val:.6e}, max={max_val:.6e}, mean={mean_val:.6e}, NaNs={nan_count}, Infs={inf_count}")

        axis_map = {"x": 0, "y": 1, "z": 2}
        slice_idx = parse_index(args.index, arr.shape[axis_map[args.axis]])
        slice_2d = slice_frame(arr, args.axis, slice_idx)

        display_data = np.log10(np.maximum(slice_2d, 0.0) + args.eps) if args.log else slice_2d

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(display_data.T, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(f"{npy_path.name} ({args.axis}={slice_idx})")
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        cbar_label = "log10(vorticity + eps)" if args.log else "vorticity norm"
        fig.colorbar(im, ax=ax, label=cbar_label)

        output_png = figures_dir / f"vorticity_{frame_id:06d}_{args.axis}{slice_idx}.png"
        fig.tight_layout()
        fig.savefig(output_png, dpi=150)
        plt.close(fig)
        saved_images.append(output_png)

    print(f"Saved {len(saved_images)} slice images to {figures_dir}")

    if args.gif:
        gif_msg = maybe_write_gif(saved_images, figures_dir / f"vorticity_{args.axis}mid.gif", args.fps)
        if gif_msg:
            print(gif_msg)
        else:
            print("GIF export skipped (imageio unavailable or GIF writing failed).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
