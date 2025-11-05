#!/usr/bin/env python3
"""
analyze_saliency_ssim.py

Compute SSIM between consecutive frames' saliency maps under:
  <out_shap>/<TownXX>/<TownXX_XXXXXX>/saliency.png
Skip the next K frames after any respawn event. Optionally compute SSIM over the
original image pairs and plot trend/relationship figures.

Output CSV columns:
    town, prev_frame, curr_frame, prev_path, curr_path, prev_img_path, curr_img_path,
    ssim, image_ssim, steer_sim, note
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="Path to measurements.jsonl (may include event lines).")
    p.add_argument("--out-shap", type=Path, required=True, help="Root dir of saliency outputs.")
    p.add_argument("--out-csv", type=Path, default=Path("saliency_ssim.csv"), help="Where to write CSV.")
    p.add_argument("--skip-after-respawn", type=int, default=4, help="Skip next K frames after respawn.")
    p.add_argument("--pad", type=int, default=6, help="Zero-pad width for frame directories (default 6).")
    p.add_argument("--fallback-name", type=str, default="shap_overlay.png",
                   help="Fallback filename if saliency.png missing.")
    p.add_argument("--with-image-ssim", action="store_true",
                   help="Also compute SSIM on original image pairs (grayscale).")
    p.add_argument("--img-root", type=Path, default=None,
                   help="Root for resolving relative image_path; default = jsonl parent.")
    p.add_argument("--plot-trend", action="store_true",
                   help="Generate time-series and scatter plots for similarity trends/relationship.")
    p.add_argument("--plot-dir", type=Path, default=None,
                   help="Directory to save plots (default: <out_csv>.parent / 'plots').")
    p.add_argument("--show-plots", action="store_true", help="Show plots interactively.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar output.")
    return p.parse_args()


def load_jsonl_records(jsonl_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Return (measurements, respawns) preserving insertion order."""
    measurements: List[Dict] = []
    respawns: List[Dict] = []
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "event" in obj and obj.get("event") == "respawn_teleport":
                if "frame" in obj and "town" in obj:
                    respawns.append(obj)
                continue
            # Measurement lines should have at least frame/town/image_path
            if "frame" in obj and "town" in obj and "image_path" in obj:
                measurements.append(obj)
    return measurements, respawns


def extract_frame_from_image_path(image_path: str) -> Optional[int]:
    """Extract frame (int) from '.../Town01_001313.png'."""
    m = re.search(r"_([0-9]+)\.(png|jpg|jpeg)$", image_path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def saliency_path(out_shap_root: Path, town: str, frame: int, pad: int,
                  prefer: str = "saliency.png", fallback: Optional[str] = None) -> Optional[Path]:
    """Build path <out_shap>/<TownXX>/<TownXX_XXXXXX>/<prefer> (fallback optional)."""
    frame_dir = f"{town}__{frame:0{pad}d}"
    base = out_shap_root / town / frame_dir
    prefer_path = base / prefer
    if prefer_path.exists():
        return prefer_path
    if fallback:
        fb = base / fallback
        if fb.exists():
            return fb
    return prefer_path if prefer_path.exists() else None


def resolve_image_path(jsonl_dir: Path, img_root: Optional[Path], image_path: str) -> Path:
    """Resolve relative image_path against --img-root or the JSONL's directory."""
    p = Path(image_path)
    if p.is_absolute():
        return p
    base = img_root if img_root is not None else jsonl_dir
    return (base / p).resolve()


def to_gray01(path: Path, target_size_wh: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load image -> grayscale float32 in [0,1]. Optional resize to (W,H)."""
    img = Image.open(path).convert("L")
    if target_size_wh is not None and img.size != target_size_wh:
        img = img.resize(target_size_wh, resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def steer_similarity(s_prev: float, s_curr: float) -> float:
    """
    Similarity in [0,1] assuming steer ∈ [-1,1]: 1 - |Δ|/2.
    Clips inputs to [-1,1] and output to [0,1].
    """
    sp = max(-1.0, min(1.0, float(s_prev)))
    sc = max(-1.0, min(1.0, float(s_curr)))
    sim = 1.0 - abs(sc - sp) / 2.0
    return float(np.clip(sim, 0.0, 1.0))


def make_plots(xs: List[int],
               sal_vals: List[Optional[float]],
               img_vals: List[Optional[float]],
               steer_vals: List[Optional[float]],
               out_dir: Path,
               base_name: str,
               show: bool = False) -> None:
    """Generate time-series and scatter plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to arrays with NaNs for missing
    x = np.asarray(xs, dtype=np.int64)
    sal = np.array([np.nan if v is None else float(v) for v in sal_vals], dtype=np.float32)
    img = np.array([np.nan if v is None else float(v) for v in img_vals], dtype=np.float32)
    st  = np.array([np.nan if v is None else float(v) for v in steer_vals], dtype=np.float32)

    # 1) Time series
    plt.figure(figsize=(10, 4.5), dpi=140)
    plt.plot(x, sal, label="Saliency SSIM")  # NaNs break the line automatically
    if not np.all(np.isnan(img)):
        plt.plot(x, img, label="Image SSIM")
    if not np.all(np.isnan(st)):
        plt.plot(x, st, label="Steer similarity")
    plt.xlabel("Pair index (consecutive frames within-towns only)")
    plt.ylabel("Similarity (0..1)")
    plt.title("Similarities over time")
    plt.legend()
    ts_path = out_dir / f"{base_name}_timeseries.png"
    plt.tight_layout()
    plt.savefig(ts_path)
    if show:
        plt.show()
    plt.close()

    # 2) Relationship scatter: Saliency SSIM vs Image SSIM (only points with both)
    mask_img = ~np.isnan(sal) & ~np.isnan(img)
    if np.count_nonzero(mask_img) >= 2:
        s = sal[mask_img]; g = img[mask_img]
        r = float(np.corrcoef(s, g)[0, 1])
        m, b = np.polyfit(s, g, 1)
        xfit = np.linspace(np.nanmin(s), np.nanmax(s), 100, dtype=np.float32)
        yfit = m * xfit + b

        plt.figure(figsize=(6.5, 6.0), dpi=140)
        plt.scatter(s, g, alpha=0.6, s=10)
        plt.plot(xfit, yfit, linewidth=2, label=f"fit: y={m:.3f}x+{b:.3f}")
        plt.xlabel("Saliency SSIM")
        plt.ylabel("Image SSIM")
        plt.title(f"Relationship (Pearson r = {r:.3f})")
        plt.legend()
        sc_path = out_dir / f"{base_name}_scatter.png"
        plt.tight_layout()
        plt.savefig(sc_path)
        if show:
            plt.show()
        plt.close()

    # 3) Relationship scatter: Saliency SSIM vs Steer similarity
    mask_st = ~np.isnan(sal) & ~np.isnan(st)
    if np.count_nonzero(mask_st) >= 2:
        s = sal[mask_st]; g = st[mask_st]
        r = float(np.corrcoef(s, g)[0, 1])
        m, b = np.polyfit(s, g, 1)
        xfit = np.linspace(np.nanmin(s), np.nanmax(s), 100, dtype=np.float32)
        yfit = m * xfit + b

        plt.figure(figsize=(6.5, 6.0), dpi=140)
        plt.scatter(s, g, alpha=0.6, s=10)
        plt.plot(xfit, yfit, linewidth=2, label=f"fit: y={m:.3f}x+{b:.3f}")
        plt.xlabel("Saliency SSIM")
        plt.ylabel("Steer similarity")
        plt.title(f"Relationship (Pearson r = {r:.3f})")
        plt.legend()
        sc_path = out_dir / f"{base_name}_scatter_steer.png"
        plt.tight_layout()
        plt.savefig(sc_path)
        if show:
            plt.show()
        plt.close()


def main():
    args = parse_args()
    jsonl_dir = args.jsonl.parent
    plot_dir = args.plot_dir if args.plot_dir is not None else (args.out_csv.parent / "plots")
    base_plot_name = args.out_csv.stem
    use_progress = not args.no_progress

    measurements, respawns = load_jsonl_records(args.jsonl)
    if not measurements:
        print("No measurement lines found; nothing to do.")
        return

    # Sort measurements by (town, frame). Use frame from record if present; else derive from image_path.
    # items: (town, frame, image_path, steer)
    items: List[Tuple[str, int, str, Optional[float]]] = []
    for m in measurements:
        town = m["town"]
        frame = m.get("frame", None)
        if frame is None:
            frame = extract_frame_from_image_path(m.get("image_path", ""))
        if frame is None:
            continue

        steer_val: Optional[float] = None
        ctrl = m.get("control", None)
        if isinstance(ctrl, dict) and "steer" in ctrl:
            try:
                steer_val = float(ctrl["steer"])
            except (TypeError, ValueError):
                steer_val = None

        items.append((town, int(frame), m["image_path"], steer_val))
    items.sort(key=lambda x: (x[0], x[1]))

    # Build a set of frames to skip per town after respawn.
    skip_after: Dict[str, Set[int]] = {}
    for e in respawns:
        town = e["town"]
        f = int(e["frame"])
        s = skip_after.setdefault(town, set())
        for k in range(1, args.skip_after_respawn + 1):
            s.add(f + k)

    rows: List[Dict] = []
    total_pairs = 0
    computed = 0
    skipped_respawn = 0
    skipped_missing = 0
    skipped_singleton = 0

    # For plotting
    pair_indices: List[int] = []
    sal_series: List[Optional[float]] = []
    img_series: List[Optional[float]] = []
    steer_series: List[Optional[float]] = []

    # Pre-group by town (so we can count candidate pairs for the progress bar)
    from itertools import groupby
    town2frames: Dict[str, List[Tuple[str, int, str, Optional[float]]]] = {}
    for town, group in groupby(items, key=lambda t: t[0]):
        town2frames[town] = list(group)

    candidate_pairs = sum(max(0, len(frames) - 1) for frames in town2frames.values())

    # Progress bar
    class _NoBar:
        def update(self, n): pass
        def close(self): pass

    pbar = tqdm(total=candidate_pairs, desc="Comparing frames", unit="pair",
                ncols=0, mininterval=0.3, disable=not use_progress) if use_progress else _NoBar()

    global_pair_idx = 0
    with_image_ssim = args.with_image_ssim

    for town, frames in town2frames.items():
        if len(frames) < 2:
            skipped_singleton += 1
            continue

        # Preload paths/metadata for this town
        sal_paths: Dict[int, Optional[Path]] = {}
        img_paths: Dict[int, Path] = {}
        steer_map: Dict[int, Optional[float]] = {}

        for _, fr, img_rel, steer_val in frames:
            sal_paths[fr] = saliency_path(args.out_shap, town, fr, args.pad,
                                          prefer="saliency.png", fallback=args.fallback_name)
            img_paths[fr] = resolve_image_path(jsonl_dir, args.img_root, img_rel)
            steer_map[fr] = steer_val

        # Compare consecutive frames
        for i in range(1, len(frames)):
            _, f_prev, _img_prev, _steer_prev = frames[i - 1]
            _, f_curr, _img_curr, _steer_curr = frames[i]
            total_pairs += 1

            # Skip window after respawn
            if f_curr in skip_after.get(town, set()):
                rows.append({
                    "town": town,
                    "prev_frame": f_prev,
                    "curr_frame": f_curr,
                    "prev_path": str(sal_paths.get(f_prev) or ""),
                    "curr_path": str(sal_paths.get(f_curr) or ""),
                    "prev_img_path": str(img_paths.get(f_prev, "")),
                    "curr_img_path": str(img_paths.get(f_curr, "")),
                    "ssim": "",
                    "image_ssim": "",
                    "steer_sim": "",
                    "note": "skipped: respawn_window"
                })
                pair_indices.append(global_pair_idx)
                sal_series.append(None)
                img_series.append(None)
                steer_series.append(None)
                global_pair_idx += 1
                skipped_respawn += 1
                pbar.update(1)
                continue

            p_prev = sal_paths.get(f_prev)
            p_curr = sal_paths.get(f_curr)
            note_parts: List[str] = []

            # SALIENCY SSIM
            sal_val: Optional[float] = None
            if p_prev is None or p_curr is None or not (p_prev.exists() and p_curr.exists()):
                note_parts.append("missing_saliency")
            else:
                g_prev = to_gray01(p_prev)
                g_curr = to_gray01(p_curr, target_size_wh=g_prev.shape[::-1])
                sal_val = float(ssim(g_prev, g_curr, data_range=1.0, gaussian_weights=True, sigma=1.5))

            # IMAGE SSIM (optional)
            img_val: Optional[float] = None
            if with_image_ssim:
                ip_prev = img_paths.get(f_prev)
                ip_curr = img_paths.get(f_curr)
                if ip_prev is None or ip_curr is None or not (ip_prev.exists() and ip_curr.exists()):
                    note_parts.append("missing_image")
                else:
                    gi_prev = to_gray01(ip_prev)
                    gi_curr = to_gray01(ip_curr, target_size_wh=gi_prev.shape[::-1])
                    img_val = float(ssim(gi_prev, gi_curr, data_range=1.0, gaussian_weights=True, sigma=1.5))

            # STEER similarity
            steer_val: Optional[float] = None
            s_prev = steer_map.get(f_prev, None)
            s_curr = steer_map.get(f_curr, None)
            if s_prev is None or s_curr is None:
                note_parts.append("missing_steer")
            else:
                try:
                    steer_val = steer_similarity(s_prev, s_curr)
                except Exception:
                    note_parts.append("steer_compute_error")

            if sal_val is None and (with_image_ssim and img_val is None) and steer_val is None:
                skipped_missing += 1
            else:
                computed += 1

            rows.append({
                "town": town,
                "prev_frame": f_prev,
                "curr_frame": f_curr,
                "prev_path": str(p_prev or ""),
                "curr_path": str(p_curr or ""),
                "prev_img_path": str(img_paths.get(f_prev, "")),
                "curr_img_path": str(img_paths.get(f_curr, "")),
                "ssim": "" if sal_val is None else f"{sal_val:.6f}",
                "image_ssim": "" if img_val is None else f"{img_val:.6f}",
                "steer_sim": "" if steer_val is None else f"{steer_val:.6f}",
                "note": ";".join(note_parts)
            })

            # Collect for plotting
            pair_indices.append(global_pair_idx)
            sal_series.append(sal_val)
            img_series.append(img_val)
            steer_series.append(steer_val)
            global_pair_idx += 1

            pbar.update(1)

    pbar.close()

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "town", "prev_frame", "curr_frame", "prev_path", "curr_path",
            "prev_img_path", "curr_img_path", "ssim", "image_ssim", "steer_sim", "note"
        ])
        w.writeheader()
        w.writerows(rows)

    # Plots
    if args.plot_trend:
        make_plots(pair_indices, sal_series, img_series, steer_series,
                   out_dir=plot_dir, base_name=base_plot_name, show=args.show_plots)

    # Summary
    print(f"Wrote: {args.out_csv}")
    if args.plot_trend:
        print(f"Plots in: {plot_dir}")
    print(f"Total candidate pairs: {candidate_pairs}")
    print(f"Total iterated pairs:  {total_pairs}")
    print(f"Computed (at least one metric): {computed}")
    print(f"Skipped (respawn):     {skipped_respawn}")
    print(f"Skipped (missing):     {skipped_missing}")
    print(f"Skipped (singleton towns): {skipped_singleton}")


if __name__ == "__main__":
    main()
