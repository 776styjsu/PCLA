#!/usr/bin/env python3
"""
scatter_saliency_vs_image.py  (pandas-free, with zoom + reporting)

Usage:
    python scatter_saliency_vs_image.py \
        --csv saliency_ssim.csv \
        --out plot.png \
        --topk 0.10 \
        --img-min 0.9 --sal-max 0.55 \
        --dump-csv interesting_pairs.csv

What this does:
1. Plots saliency_ssim (y) vs image_ssim (x), and highlights the top-K% steer_sim.
2. Lets you optionally "zoom"/filter by threshold ranges on:
      image_ssim, saliency_ssim, steer_sim
   The thresholds do TWO things:
      - Set axis limits (for image_ssim -> x, saliency_ssim -> y)
      - Select a subset of points to REPORT.
3. Prints (and optionally writes to CSV) the frame pairs in that zoomed subset
   that are ALSO in the top-K steer_sim group (the orange group).

Why top-K?: those are the "similar steering output" pairs, which is what you
usually care about. If you want everything in the zoom region, not just top-K,
use --include-non-topk.

CSV input format (header required):
    town,prev_frame,curr_frame,prev_path,curr_path,ssim,image_ssim,steer_sim,note
    Town01,1294,1295,out_shap/Town01/Town01_001294/saliency.png,...

Output CSV (if --dump-csv is given):
    town,prev_frame,curr_frame,prev_path,curr_path,ssim,image_ssim,steer_sim,is_topk

Examples
--------
Example: "bottom-right orange corner" idea:
- high image_ssim (>=0.9)
- high steer_sim (we'll let topk do that)
- low saliency_ssim (<=0.55)

    python scatter_saliency_vs_image.py \
        --csv saliency_ssim.csv \
        --out zoom.png \
        --topk 0.10 \
        --img-min 0.9 \
        --sal-max 0.55 \
        --dump-csv suspect_pairs.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def safe_float(x):
    """Convert string to float, return None if blank or bad."""
    try:
        x = x.strip()
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def load_rows(csv_path):
    """
    Read the CSV manually and return a list of dicts with numeric fields parsed.
    Drops rows missing any of ssim, image_ssim, steer_sim.
    """
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ssim = safe_float(r.get("ssim", ""))
            img_ssim = safe_float(r.get("image_ssim", ""))
            steer_sim = safe_float(r.get("steer_sim", ""))

            # skip rows with missing core metrics
            if ssim is None or img_ssim is None or steer_sim is None:
                continue

            rows.append({
                "town": r.get("town", ""),
                "prev_frame": r.get("prev_frame", ""),
                "curr_frame": r.get("curr_frame", ""),
                "prev_path": r.get("prev_path", ""),
                "curr_path": r.get("curr_path", ""),
                "prev_img_path": r.get("prev_img_path", ""),
                "curr_img_path": r.get("curr_img_path", ""),
                "ssim": ssim,                # saliency similarity
                "image_ssim": img_ssim,      # image similarity
                "steer_sim": steer_sim,      # steering similarity
            })
    return rows


def build_zoom_mask(image_ssim, saliency_ssim, steer_sim, args):
    """
    Build a boolean mask selecting points that lie within the user-specified
    zoom ranges. Any bound not provided is ignored.
    """
    m = np.ones_like(image_ssim, dtype=bool)

    if args.img_min is not None:
        m &= image_ssim >= args.img_min
    if args.img_max is not None:
        m &= image_ssim <= args.img_max

    if args.sal_min is not None:
        m &= saliency_ssim >= args.sal_min
    if args.sal_max is not None:
        m &= saliency_ssim <= args.sal_max

    if args.steer_min is not None:
        m &= steer_sim >= args.steer_min
    if args.steer_max is not None:
        m &= steer_sim <= args.steer_max

    return m


def maybe_set_axis_limits(args, image_ssim, saliency_ssim):
    """
    If user gave zoom ranges, clamp x/y axes accordingly.
    x-axis comes from image_ssim (img_*), y-axis from saliency_ssim (sal_*).
    steer_* doesn't affect axes.
    """
    # X limits (image similarity)
    if args.img_min is not None or args.img_max is not None:
        xmin = args.img_min if args.img_min is not None else float(np.min(image_ssim))
        xmax = args.img_max if args.img_max is not None else float(np.max(image_ssim))
        plt.xlim(xmin, xmax)

    # Y limits (saliency similarity)
    if args.sal_min is not None or args.sal_max is not None:
        ymin = args.sal_min if args.sal_min is not None else float(np.min(saliency_ssim))
        ymax = args.sal_max if args.sal_max is not None else float(np.max(saliency_ssim))
        plt.ylim(ymin, ymax)


def dump_report(rows, mask_zoom, mask_topk, out_csv_path=None, include_non_topk=False):
    """
    Print (and optionally write CSV of) the rows that satisfy the zoom mask.
    By default we only include rows that are ALSO in top-k steer_sim (mask_topk),
    unless include_non_topk=True.

    Output columns:
      town, prev_frame, curr_frame,
      prev_path, curr_path,
      ssim, image_ssim, steer_sim, is_topk
    """
    final_mask = mask_zoom & mask_topk
    if include_non_topk:
        final_mask = mask_zoom  # ignore top-k restriction

    idxs = np.where(final_mask)[0].tolist()
    if not idxs:
        print("[report] No rows matched the zoom criteria.")
        return

    print(f"[report] {len(idxs)} row(s) matched zoom criteria:")
    for i in idxs:
        r = rows[i]
        print(
            f"- {r['town']} {r['prev_frame']}->{r['curr_frame']}  "
            f"img_ssim={r['image_ssim']:.3f}  "
            f"sal_ssim={r['ssim']:.3f}  "
            f"steer_sim={r['steer_sim']:.3f}  "
            f"topk={bool(mask_topk[i])}\n"
            f"    prev_png={r['prev_path']}\n"
            f"    curr_png={r['curr_path']}"
        )

    if out_csv_path:
        outp = Path(out_csv_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "town", "prev_frame", "curr_frame",
                "prev_path", "curr_path", "prev_img_path", "curr_img_path",
                "ssim", "image_ssim", "steer_sim", "is_topk"
            ])
            for i in idxs:
                r = rows[i]
                writer.writerow([
                    r["town"],
                    r["prev_frame"],
                    r["curr_frame"],
                    r["prev_path"],
                    r["curr_path"],
                    r["prev_img_path"],
                    r["curr_img_path"],
                    f"{r['ssim']:.6f}",
                    f"{r['image_ssim']:.6f}",
                    f"{r['steer_sim']:.6f}",
                    str(bool(mask_topk[i])),
                ])
        print(f"[report] Wrote {len(idxs)} rows to {outp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file with pair stats.")
    ap.add_argument("--out", default="out.png", help="Where to save the plot.")
    ap.add_argument("--title", default="Saliency vs Image Similarity",
                    help="Plot title.")
    ap.add_argument("--topk", type=float, default=0.10,
                    help="Top K fraction for steer_sim highlight "
                         "(default 0.10 = top 10%).")

    # Zoom / filter thresholds
    ap.add_argument("--img-min", type=float, default=None,
                    help="Only consider/report pairs with image_ssim >= this.")
    ap.add_argument("--img-max", type=float, default=None,
                    help="Only consider/report pairs with image_ssim <= this.")
    ap.add_argument("--sal-min", type=float, default=None,
                    help="Only consider/report pairs with saliency ssim >= this.")
    ap.add_argument("--sal-max", type=float, default=None,
                    help="Only consider/report pairs with saliency ssim <= this.")
    ap.add_argument("--steer-min", type=float, default=None,
                    help="Only consider/report pairs with steer_sim >= this.")
    ap.add_argument("--steer-max", type=float, default=None,
                    help="Only consider/report pairs with steer_sim <= this.")

    # Reporting controls
    ap.add_argument("--dump-csv", default=None,
                    help="If set, write matching zoomed rows to this CSV.")
    ap.add_argument("--include-non-topk", action="store_true",
                    help="If set, report ALL zoom-matching rows, not just top-K steer_sim.")

    args = ap.parse_args()

    # Load data
    rows = load_rows(args.csv)
    if not rows:
        raise RuntimeError("No valid rows after parsing. Check your CSV columns?")

    # Grab vectors
    image_ssim = np.array([d["image_ssim"] for d in rows], dtype=float)
    saliency_ssim = np.array([d["ssim"] for d in rows], dtype=float)
    steer_sim = np.array([d["steer_sim"] for d in rows], dtype=float)

    # Compute cutoff for "top K%" steer similarity
    frac = max(0.0, min(args.topk, 1.0))
    cutoff = np.quantile(steer_sim, 1.0 - frac)
    high_mask = steer_sim >= cutoff
    low_mask = ~high_mask

    # --- Plot ---
    plt.figure(figsize=(7, 5), dpi=150)

    # normal points
    plt.scatter(
        image_ssim[low_mask],
        saliency_ssim[low_mask],
        s=20,
        alpha=0.5,
        label="normal steer_sim",
        edgecolors="none",
        color="#1f77b4",
    )

    # high-steer-sim points (top-K)
    plt.scatter(
        image_ssim[high_mask],
        saliency_ssim[high_mask],
        s=40,
        alpha=0.9,
        label=f"top {int(frac*100)}% steer_sim (≥ {cutoff:.3f})",
        edgecolors="black",
        linewidths=0.5,
        color="#ff7f0e",
    )

    plt.xlabel("Image similarity (image_ssim)")
    plt.ylabel("Saliency similarity (ssim)")
    plt.title(args.title)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(loc="best", frameon=True)

    # If zoom thresholds were provided, also zoom the plot axes.
    maybe_set_axis_limits(args, image_ssim, saliency_ssim)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"[plot] Saved plot to {out_path} (top-k cutoff steer_sim >= {cutoff:.4f})")

    # --- Reporting of interesting cases ---
    zoom_mask = build_zoom_mask(image_ssim, saliency_ssim, steer_sim, args)
    dump_report(
        rows,
        mask_zoom=zoom_mask,
        mask_topk=high_mask,
        out_csv_path=args.dump_csv,
        include_non_topk=args.include_non_topk,
    )


if __name__ == "__main__":
    main()
