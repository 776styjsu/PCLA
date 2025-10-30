#!/usr/bin/env python3
"""
Compute image similarity/distance decay from a CARLA measurements.jsonl.

Supports SSIM (default), FSIM, and LPIPS:
  - SSIM / FSIM: higher = more similar
  - LPIPS: lower = more similar

Example:
  python compute_sim__decay.py \
      /path/to/measurements.jsonl \
      --img-root /path/to/run_root  \
      --metric ssim --max-k 10 \
      --resize 180 320 \
      --out-plot ssim_decay.png \
      --out-csv ssim_decay.csv

  python compute_sim__decay.py \
      /path/to/measurements.jsonl \
      --metric fsim --out-plot fsim_decay.png --out-csv fsim_decay.csv

  python compute_sim__decay.py \
      /path/to/measurements.jsonl \
      --metric lpips --lpips-backbone vgg \
      --out-plot lpips_decay.png --out-csv lpips_decay.csv
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import piq
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Stepwise similarity vs step distance (k).")
    p.add_argument("jsonl", type=str, help="Path to measurements.jsonl")
    p.add_argument("--img-root", type=str, default=None,
                   help="Root directory to resolve image_path (default: jsonl's parent)")
    p.add_argument("--metric", type=str, choices=["ssim", "fsim", "lpips"], default="ssim",
                   help="Similarity/distance metric to use.")
    p.add_argument("--lpips-backbone", type=str, choices=["alex", "vgg", "squeeze"], default="vgg",
                   help="LPIPS backbone (only used if --metric lpips).")
    p.add_argument("--max-k", type=int, default=10, help="Max step distance K (inclusive).")
    p.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=None,
                   help="Resize images to (H W). If omitted, original size is used (must be >= 11x11 for SSIM).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to run on: 'cuda' or 'cpu'.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional: use only first N frames after sorting (for quick tests).")
    p.add_argument("--ssim-win-size", type=int, default=11,
                   help="SSIM window size (default 11). Requires H,W >= this (only for SSIM).")
    p.add_argument("--out-plot", type=str, default="decay.png",
                   help="Output plot path.")
    p.add_argument("--out-csv", type=str, default=None,
                   help="Optional: write k,mean,std,count to CSV.")
    return p.parse_args()


def read_jsonl(jsonl_path: Path) -> List[Dict]:
    data = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    data = [d for d in data if "image_path" in d and "frame" in d]
    data.sort(key=lambda d: d["frame"])
    return data


def load_image(path: Path, resize_hw: Tuple[int, int] = None) -> torch.Tensor:
    """Load an RGB image -> torch.FloatTensor of shape (1,3,H,W) in [0,1]."""
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        h, w = resize_hw
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def ensure_min_size(images: List[torch.Tensor], min_hw: int):
    """Ensure all images have H,W >= min_hw (needed for SSIM window)."""
    for t in images:
        _, _, h, w = t.shape
        if h < min_hw or w < min_hw:
            raise ValueError(
                f"Image too small for SSIM window {min_hw}: got {(h, w)}. "
                f"Use --resize H W with H,W >= {min_hw}."
            )


def build_ordered_image_list(entries: List[Dict], root: Path, limit: int) -> List[Path]:
    paths = []
    for d in entries:
        img_rel = d["image_path"]
        img_path = (root / img_rel).resolve()
        paths.append(img_path)
    if limit is not None:
        paths = paths[:limit]
    return paths


@torch.no_grad()
def compute_over_ks(
    img_paths: List[Path],
    metric: str = "ssim",
    device: str = "cpu",
    max_k: int = 10,
    resize_hw: Tuple[int, int] = None,
    ssim_win_size: int = 11
) -> Dict[int, List[float]]:
    """
    Preload all images as tensors on device, then compute metric for all k.
    Returns {k: [values]} where values are per-pair scores.
      - SSIM/FSIM: higher = more similar (range roughly [0..1])
      - LPIPS: lower = more similar (range ~[0..1+], model-dependent)
    """
    # Preload images
    tensors: List[torch.Tensor] = []
    for p in img_paths:
        if not p.exists():
            print(f"[WARN] Missing image: {p}")
            continue
        t = load_image(p, resize_hw=resize_hw)
        tensors.append(t)
    if len(tensors) < 2:
        raise RuntimeError("Not enough images to compare. Check paths / input JSONL.")

    # Move to device
    tensors = [t.to(device) for t in tensors]
    if metric == "ssim":
        ensure_min_size(tensors, ssim_win_size)

    # Prepare metric op
    lpips_model = None
    if metric == "lpips":
        lpips_model = piq.LPIPS(reduction='none').to(device)
        lpips_model.eval()

    results: Dict[int, List[float]] = {k: [] for k in range(1, max_k + 1)}
    n = len(tensors)

    for k in range(1, max_k + 1):
        if n - k <= 0:
            break
        batch_x = torch.cat(tensors[:-k], dim=0)  # (N-k, 3, H, W)
        batch_y = torch.cat(tensors[k:], dim=0)   # (N-k, 3, H, W)

        if metric == "ssim":
            vals = piq.ssim(batch_x, batch_y, data_range=1.0,
                            kernel_size=ssim_win_size, reduction='none')  # (N-k,)
        elif metric == "fsim":
            # FSIM: full-reference similarity; higher is better
            vals = piq.fsim(batch_x, batch_y, data_range=1.0, reduction='none')  # (N-k,)
        else:  # lpips
            # LPIPS: learned perceptual distance; lower is better
            vals = lpips_model(batch_x, batch_y)  # (N-k,)

        results[k] = vals.detach().float().cpu().tolist()

    return results


def summarize(results: Dict[int, List[float]]) -> List[Tuple[int, float, float, int]]:
    table = []
    for k in sorted(results.keys()):
        vals = np.array(results[k], dtype=np.float32)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size >= 2 else 0.0
        table.append((k, mean, std, int(vals.size)))
    return table


def save_csv(table: List[Tuple[int, float, float, int]], out_csv: Path, metric: str):
    with out_csv.open("w") as f:
        f.write(f"k,mean_{metric},std_{metric},count\n")
        for k, mean, std, count in table:
            f.write(f"{k},{mean:.6f},{std:.6f},{count}\n")


def plot_decay(table: List[Tuple[int, float, float, int]], out_plot: Path, metric: str):
    ks = [row[0] for row in table]
    means = [row[1] for row in table]
    stds = [row[2] for row in table]

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(ks, means, yerr=stds, fmt='o-', capsize=3)
    plt.xlabel("Step distance k")
    ylabel = {
        "ssim": "SSIM (mean ± std) — higher = more similar",
        "fsim": "FSIM (mean ± std) — higher = more similar",
        "lpips": "LPIPS (mean ± std) — lower = more similar",
    }[metric]
    plt.ylabel(ylabel)
    plt.title(f"Stepwise {metric.upper()} vs. Step Distance")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)
    print(f"[OK] Plot saved to {out_plot}")


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    img_root = Path(args.img_root).resolve() if args.img_root else jsonl_path.parent.resolve()

    print(f"[INFO] Loading {jsonl_path}")
    entries = read_jsonl(jsonl_path)
    if len(entries) == 0:
        raise RuntimeError("No valid JSON lines with 'frame' and 'image_path' found.")

    img_paths = build_ordered_image_list(entries, img_root, args.limit)
    print(f"[INFO] Found {len(img_paths)} frames (after optional limit).")
    print(f"[INFO] Metric: {args.metric} | Device: {args.device} | Max-k: {args.max_k} | "
          f"Resize: {args.resize} | SSIM win: {args.ssim_win_size if args.metric=='ssim' else 'N/A'}")

    results = compute_over_ks(
        img_paths,
        metric=args.metric,
        device=args.device,
        max_k=args.max_k,
        resize_hw=tuple(args.resize) if args.resize else None,
        ssim_win_size=args.ssim_win_size
    )

    table = summarize(results)
    for k, mean, std, count in table:
        print(f"k={k:>3} | mean {args.metric}={mean:.4f} | std={std:.4f} | pairs={count}")

    if args.out_csv:
        save_csv(table, Path(args.out_csv), args.metric)
        print(f"[OK] CSV saved to {args.out_csv}")

    plot_decay(table, Path(args.out_plot), args.metric)


if __name__ == "__main__":
    main()
