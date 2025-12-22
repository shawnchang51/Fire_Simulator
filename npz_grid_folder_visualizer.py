from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def display_array(arr: Any, title: str | None = None, cmap: str = "viridis") -> plt.Figure:
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)

    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(np.nan)

    ndim = getattr(arr, "ndim", 0)

    if ndim == 0:
        ax.text(0.5, 0.5, str(arr), ha="center", va="center")
        ax.axis("off")
    elif ndim == 1:
        ax.plot(arr)
        ax.set_ylabel("value")
        ax.set_xlabel("index")
    elif ndim == 2:
        im = ax.imshow(arr, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax)
    elif ndim == 3:
        if arr.shape[2] in (3, 4):
            ax.imshow(arr)
            ax.axis("off")
        else:
            idx = arr.shape[2] // 2
            im = ax.imshow(arr[:, :, idx], cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"slice {idx} (axis=2)")
    else:
        ax.text(
            0.5, 0.5,
            f"array ndim={arr.ndim} not directly plottable",
            ha="center"
        )
        ax.axis("off")

    return fig


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Visualize `grid` arrays from a folder of .npz files")
    p.add_argument("folder", help="Folder containing .npz files")
    p.add_argument("--key", default="grid", help="Array key to visualize (default: grid)")
    p.add_argument("--cmap", default="viridis", help="Colormap")
    p.add_argument("--save-dir", help="If set, save figures to this directory instead of showing")
    args = p.parse_args(argv)

    if not os.path.isdir(args.folder):
        print(f"Not a folder: {args.folder}")
        return 2

    npz_files = sorted(
        f for f in os.listdir(args.folder)
        if f.endswith(".npz")
    )

    if not npz_files:
        print("No .npz files found in folder.")
        return 0

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for fname in npz_files:
        path = os.path.join(args.folder, fname)
        print(f"\nLoading {fname}")

        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        if args.key not in data:
            print(f"  Key `{args.key}` not found, available keys: {data.files}")
            continue

        arr = data[args.key]
        print(f"  grid shape={arr.shape}, dtype={arr.dtype}")

        fig = display_array(arr, title=f"{fname} :: {args.key}", cmap=args.cmap)

        if args.save_dir:
            out = os.path.join(
                args.save_dir,
                fname.replace(".npz", f"_{args.key}.png")
            )
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved -> {out}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
