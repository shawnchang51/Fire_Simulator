"""NPZ Visualizer

Usage: python npz_visualizer.py path/to/file.npz [--key KEY] [--list] [--save out.png]

Simple tool to list arrays inside a .npz and display one using matplotlib.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def list_contents(npz: np.lib.npyio.NpzFile) -> None:
    print("Contents:")
    for k in npz.files:
        arr = npz[k]
        print(f" - {k}: shape={arr.shape}, dtype={arr.dtype}")


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
        # If third dimension looks like channels (3 or 4), show as RGB(A)
        if arr.shape[2] in (3, 4):
            ax.imshow(arr)
            ax.axis("off")
        else:
            # show middle slice along last axis
            idx = arr.shape[2] // 2
            im = ax.imshow(arr[:, :, idx], cmap=cmap)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"slice {idx} (axis=2)")
    else:
        # For higher dims, show middle slice of the first two dims
        slicer = tuple(s//2 for s in arr.shape[2:])
        try:
            index = (Ellipsis,) + slicer
            slice_arr = arr[index]
        except Exception:
            slice_arr = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr
        if slice_arr.ndim == 2:
            im = ax.imshow(slice_arr, cmap=cmap)
            fig.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, f"array ndim={arr.ndim} not directly plottable", ha="center")
            ax.axis("off")

    return fig


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(description="Inspect and visualize .npz files")
    p.add_argument("npz", help="Path to .npz file")
    p.add_argument("--key", help="Specific array key to display")
    p.add_argument("--list", action="store_true", help="List arrays and exit")
    p.add_argument("--cmap", default="viridis", help="Colormap for 2D/3D slices")
    p.add_argument("--save", help="Save the displayed figure to a file instead of showing")
    args = p.parse_args(argv)

    if not os.path.exists(args.npz):
        print(f"File not found: {args.npz}")
        return 2

    try:
        data = np.load(args.npz, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load NPZ: {e}")
        return 3

    if args.list:
        list_contents(data)
        return 0

    keys = data.files
    if not keys:
        print("No arrays found in the NPZ.")
        return 0

    key = args.key
    if key is None:
        if len(keys) == 1:
            key = keys[0]
        else:
            print("Multiple arrays found:")
            for i, k in enumerate(keys):
                arr = data[k]
                print(f"[{i}] {k}: shape={arr.shape}, dtype={arr.dtype}")
            sel = input("Choose an index or key to display: ").strip()
            if sel.isdigit():
                idx = int(sel)
                try:
                    key = keys[idx]
                except Exception:
                    print("Invalid index")
                    return 4
            else:
                key = sel

    if key not in keys:
        print(f"Key not found: {key}")
        list_contents(data)
        return 5

    arr = data[key]
    print(f"Displaying `{key}`: shape={arr.shape}, dtype={arr.dtype}")

    fig = display_array(arr, title=key, cmap=args.cmap)

    if args.save:
        fig.savefig(args.save, bbox_inches="tight")
        print(f"Saved figure to {args.save}")
        return 0

    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
