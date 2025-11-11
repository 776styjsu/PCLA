#!/usr/bin/env python3
"""
scatter_saliency_vs_image.py  (pandas-free, with zoom + reporting + corner tracking)

Usage:
    python scatter_saliency_vs_image.py \
        --csv saliency_ssim.csv \
        --out plot.png \
        --topk 0.10 \
        --img-min 0.9 --sal-max 0.55 \
        --dump-csv interesting_pairs.csv \
        --dump-corner-csv corners.csv \
        --corner-frac 0.20

What this does:
1. Plots saliency_ssim (y) vs image_ssim (x), and highlights the top-K% steer_sim.
2. Lets you optionally "zoom"/filter by threshold ranges on:
      image_ssim, saliency_ssim, steer_sim
   The thresholds do TWO things:
      - Set axis limits (for image_ssim -> x, saliency_ssim -> y)
      - Select a subset of points to REPORT.
3. Prints (and optionally writes to CSV) the frame pairs in that zoomed subset
   that are ALSO in the top-K steer_sim group (the orange group). Use
   --include-non-topk to report all zoom-matching rows instead.
4. Corner tracking within TOP-K:
   - top-left (TL):  x <= Qx(frac)  AND  y >= Qy(1-frac)
   - bottom-right (BR): x >= Qx(1-frac) AND y <= Qy(frac)
   where quantiles are computed on the TOP-K subset only, and frac defaults to 0.20.
   These TL/BR points are overlaid with distinct markers and can be dumped via --dump-corner-csv.

Default axes are fixed to [0,1]. Supplying any --img-*/--sal-* bounds zooms that axis.
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
    Default: fixed unit square [0,1]×[0,1].
    If any min/max is provided, we 'zoom' that axis to the given bound(s),
    still clamped within [0,1].
    """
    # Start with fixed unit axes
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0

    # Apply optional zooms (clamped to [0,1])
    if args.img_min is not None:
        xmin = max(0.0, min(1.0, float(args.img_min)))
    if args.img_max is not None:
        xmax = max(0.0, min(1.0, float(args.img_max)))

    if args.sal_min is not None:
        ymin = max(0.0, min(1.0, float(args.sal_min)))
    if args.sal_max is not None:
        ymax = max(0.0, min(1.0, float(args.sal_max)))

    # Guard against inverted ranges by swapping if needed
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def compute_corner_masks(image_ssim: np.ndarray,
                         saliency_ssim: np.ndarray,
                         topk_mask: np.ndarray,
                         frac: float):
    """
    Compute TL/BR corner masks within the TOP-K subset using quantiles.
    Returns (mask_tl, mask_br, thresholds_dict).
    TL: low x (<= x_lo) and high y (>= y_hi)
    BR: high x (>= x_hi) and low y (<= y_lo)
    """
    frac = float(np.clip(frac, 1e-6, 0.49))  # keep sane and avoid degenerate tails
    x_top = image_ssim[topk_mask]
    y_top = saliency_ssim[topk_mask]
    if x_top.size < 2:
        empty = np.zeros_like(topk_mask, dtype=bool)
        return empty, empty, {"x_lo": np.nan, "x_hi": np.nan, "y_lo": np.nan, "y_hi": np.nan}

    x_lo = float(np.quantile(x_top, frac))
    x_hi = float(np.quantile(x_top, 1.0 - frac))
    y_lo = float(np.quantile(y_top, frac))
    y_hi = float(np.quantile(y_top, 1.0 - frac))

    tl = topk_mask & (image_ssim <= x_lo) & (saliency_ssim >= y_hi)
    br = topk_mask & (image_ssim >= x_hi) & (saliency_ssim <= y_lo)

    return tl, br, {"x_lo": x_lo, "x_hi": x_hi, "y_lo": y_lo, "y_hi": y_hi}


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


def dump_corner_csv(rows, tl_mask: np.ndarray, br_mask: np.ndarray, out_csv_path: str):
    """Write a CSV of TL/BR rows with a 'corner' column."""
    outp = Path(out_csv_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "corner", "town", "prev_frame", "curr_frame",
            "prev_path", "curr_path", "prev_img_path", "curr_img_path",
            "ssim", "image_ssim", "steer_sim"
        ])
        for corner_name, m in (("top_left", tl_mask), ("bottom_right", br_mask)):
            for i in np.where(m)[0].tolist():
                r = rows[i]
                w.writerow([
                    corner_name,
                    r["town"], r["prev_frame"], r["curr_frame"],
                    r["prev_path"], r["curr_path"], r["prev_img_path"], r["curr_img_path"],
                    f"{r['ssim']:.6f}", f"{r['image_ssim']:.6f}", f"{r['steer_sim']:.6f}",
                ])
    print(f"[report] Wrote TL/BR corner rows to {outp}")


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

    # Corner tracking options
    ap.add_argument("--corner-frac", type=float, default=0.20,
                    help="Quantile fraction within TOP-K used to define corners "
                         "(TL uses x<=Qx(frac) & y>=Qy(1-frac); BR uses x>=Qx(1-frac) & y<=Qy(frac)).")
    ap.add_argument("--dump-corner-csv", default=None,
                    help="If set, write TL/BR top-K corner rows to this CSV (adds 'corner' column).")

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
    topk_frac = max(0.0, min(args.topk, 1.0))
    cutoff = np.quantile(steer_sim, 1.0 - topk_frac)
    high_mask = steer_sim >= cutoff
    low_mask = ~high_mask

    # Corners within TOP-K
    tl_mask, br_mask, thr = compute_corner_masks(image_ssim, saliency_ssim, high_mask, args.corner_frac)

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
        label=f"top {int(topk_frac*100)}% steer_sim (≥ {cutoff:.3f})",
        edgecolors="black",
        linewidths=0.5,
        color="#ff7f0e",
    )

    # Overlays: TL (triangle), BR (square)
    if tl_mask.any():
        plt.scatter(
            image_ssim[tl_mask], saliency_ssim[tl_mask],
            s=70, alpha=1.0, marker="^",
            edgecolors="black", linewidths=0.6, color="#d62728",
            label=f"top-K TL (x≤{thr['x_lo']:.2f}, y≥{thr['y_hi']:.2f})"
        )
    if br_mask.any():
        plt.scatter(
            image_ssim[br_mask], saliency_ssim[br_mask],
            s=70, alpha=1.0, marker="s",
            edgecolors="black", linewidths=0.6, color="#2ca02c",
            label=f"top-K BR (x≥{thr['x_hi']:.2f}, y≤{thr['y_lo']:.2f})"
        )

    plt.xlabel("Image similarity (image_ssim)")
    plt.ylabel("Saliency similarity (ssim)")
    plt.title(args.title)

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(loc="best", frameon=True)

    # Fixed [0,1] axes by default; zoom if bounds provided
    maybe_set_axis_limits(args, image_ssim, saliency_ssim)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"[plot] Saved plot to {out_path} (top-k cutoff steer_sim >= {cutoff:.4f})")

    # Corner summary
    n_tl, n_br = int(tl_mask.sum()), int(br_mask.sum())
    print(f"[corners] frac={args.corner_frac:.2f}  "
          f"x_lo={thr['x_lo']:.3f}, x_hi={thr['x_hi']:.3f}, "
          f"y_lo={thr['y_lo']:.3f}, y_hi={thr['y_hi']:.3f}")
    print(f"[corners] top-left (TL) count: {n_tl}")
    print(f"[corners] bottom-right (BR) count: {n_br}")

    if args.dump_corner_csv:
        dump_corner_csv(rows, tl_mask, br_mask, args.dump_corner_csv)

    # --- Reporting of interesting cases (zoom region) ---
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
