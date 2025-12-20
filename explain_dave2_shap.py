#!/usr/bin/env python3
"""
explain_dave2_shap.py

Compute SHAP attributions for DAVE2 steering predictions.

Modes
-----
1) Single-image (legacy):
   python explain_dave2_shap.py --ckpt CKPT --image IMG --out OUTDIR [--img-root ROOT] ...

2) Batch JSONL:
   python explain_dave2_shap.py --ckpt CKPT --jsonl all_measurements.jsonl \
       --img-root /path/to/run_root --out-root out_shap \
       --input-h 180 --input-w 320 --device auto --bg 8 --skip-existing \
       --bg-mode fixed_dataset_kmeans

Outputs (per image)
-------------------
- pred.json                : steer prediction + metadata
- shap_overlay.png         : SHAP red/blue overlay on the MODEL INPUT (resized HxW)
- saliency.png             : grayscale saliency (|SHAP| over channels), no overlay
- shap_values.npy          : raw SHAP values, shape (C,H,W)
- shap_sum_abs.npy         : |SHAP| summed across channels, shape (H,W)
- preprocessed_input.npy   : exact model input (1,C,H,W)
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any, List

import numpy as np
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans  # <-- NEW: For K-means clustering
from tqdm import tqdm                # <-- NEW: For progress bar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap  # pip install shap

# Import your model class without dragging in CARLA dependency
from agents.dave2.models import DAVE2v1


# ------------------------- model & preprocessing -------------------------

def load_dave2(ckpt_path: str, input_shape=(180, 320), device: str = "cuda") -> Tuple[nn.Module, str]:
    dev = device
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = DAVE2v1(input_shape=input_shape).to(dev)
    ckpt = torch.load(ckpt_path, map_location=dev)

    # Handle various checkpoint formats
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    for k in ("speed_mean", "speed_std"):
        if k in sd:
            print(f"[WARN] dropping non-param key from state_dict: {k}", flush=True)
            sd.pop(k)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print("[WARN] Unexpected keys ignored:", unexpected, flush=True)
    if missing:
        print("[WARN] Missing keys (not in checkpoint):", missing, flush=True)

    model.eval()  # gradients still flow under eval
    return model, dev


def build_preprocess(input_shape):
    H, W = input_shape
    return Compose(
        [
            Resize((H, W)),
            ToTensor(),  # -> [0,1], CxHxW
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1,1]
        ]
    )


def pil_to_model_input(img_pil: Image.Image, preprocess, device: str):
    x = preprocess(img_pil).unsqueeze(0).to(device)  # 1,C,H,W
    return x


def build_fixed_background(n: int, C: int, H: int, W: int, device: str):
    """
    Fixed mid-gray baseline in normalized space (zeros after Normalize(mean=0.5, std=0.5)).
    Reuse this for ALL frames so cross-frame comparisons are apples-to-apples.
    """
    n = max(1, int(n))
    return torch.zeros((n, C, H, W), device=device)  # float32


def build_blur_background(img_pil: Image.Image, preprocess, device: str, n: int = 8, seed: int = 42):
    """
    Per-frame blurred/jittered background (legacy behavior).
    """
    rng = np.random.RandomState(seed)
    imgs = []
    for _ in range(max(1, int(n))):
        radius = 1.0 + 2.0 * rng.rand()
        bg = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        # brightness jitter
        arr = np.asarray(bg).astype(np.float32)
        gain = 0.9 + 0.2 * rng.rand()
        arr = np.clip(arr * gain, 0, 255).astype(np.uint8)
        bg = Image.fromarray(arr)
        imgs.append(preprocess(bg))
    bg_batch = torch.stack(imgs, dim=0).to(device)  # n,C,H,W
    return bg_batch


# ---------------- NEW: k-means and dataset-sampled backgrounds ----------------

def pick_evenly_spaced_indices(n_total: int, k: int) -> List[int]:
    if k <= 0 or n_total <= 0:
        return []
    # inner K points (exclude endpoints) to avoid biasing toward first/last frames
    idx = np.linspace(0, n_total - 1, k + 2, dtype=int)[1:-1]
    # dedupe and sort just in case
    return sorted(set(int(i) for i in idx.tolist()))

def collect_existing_image_paths_from_lines(lines: List[str], img_root: Path) -> List[Path]:
    paths: List[Path] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            entry = json.loads(s)
        except json.JSONDecodeError:
            continue
        img_rel = entry.get("image_path")
        if not img_rel:
            continue
        p = (img_root / img_rel)
        p = p if p.is_absolute() else p.resolve()
        if p.exists():
            paths.append(p)
    return paths

def build_fixed_dataset_blur_background_from_lines(lines: List[str], img_root: Path,
                                                   preprocess, device: str, k: int,
                                                   seed: int = 42) -> torch.Tensor:
    """
    Pick K evenly spaced existing frames from the dataset, blur+jitter each,
    preprocess, and stack into a single (K,C,H,W) tensor for a shared explainer.
    """
    all_paths = collect_existing_image_paths_from_lines(lines, img_root)
    if not all_paths:
        raise RuntimeError("No valid image paths found to build background set.")
    k = min(max(1, int(k)), len(all_paths))
    idxs = pick_evenly_spaced_indices(len(all_paths), k)

    rng = np.random.RandomState(seed)
    bgs = []
    for i in idxs:
        img = Image.open(all_paths[i]).convert("RGB")
        radius = 1.0 + 2.0 * rng.rand()
        img = img.filter(ImageFilter.GaussianBlur(radius))
        # brightness jitter
        arr = np.asarray(img).astype(np.float32)
        gain = 0.9 + 0.2 * rng.rand()
        arr = np.clip(arr * gain, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        bgs.append(preprocess(img))
    return torch.stack(bgs, dim=0).to(device)  # (K,C,H,W)


def build_fixed_dataset_kmeans_background_from_lines(
    lines: List[str],
    img_root: Path,
    preprocess,
    device: str,
    k: int,
    model: DAVE2v1,
    seed: int = 42,
    max_samples_for_kmeans: int = 5000
) -> torch.Tensor:
    """
    Builds a shared background set by running K-means on model features
    from a large sample of the dataset, and picking the K exemplars
    (closest images to centers).
    """
    print(f"[INFO] Collecting image paths for K-means background...", flush=True)
    all_paths = collect_existing_image_paths_from_lines(lines, img_root)
    if not all_paths:
        raise RuntimeError("No valid image paths found to build K-means background set.")

    k = min(max(1, int(k)), len(all_paths))
    
    # --- 1. Sub-sample paths if dataset is too large ---
    rng = np.random.RandomState(seed)
    if len(all_paths) > max_samples_for_kmeans:
        print(f"[INFO] Sampling {max_samples_for_kmeans} images (from {len(all_paths)}) for K-means fitting...", flush=True)
        sample_indices = rng.choice(len(all_paths), max_samples_for_kmeans, replace=False)
        paths_to_sample = [all_paths[i] for i in sample_indices]
    else:
        print(f"[INFO] Using all {len(all_paths)} images for K-means fitting...", flush=True)
        paths_to_sample = all_paths
        sample_indices = np.arange(len(all_paths))

    # --- 2. Extract features using the model ---
    features_list = []
    model.eval() # Ensure model is in eval mode
    print(f"[INFO] Extracting features from {len(paths_to_sample)} images...", flush=True)
    with torch.no_grad():
        for img_path in tqdm(paths_to_sample, desc="Extracting features"):
            try:
                img = Image.open(img_path).convert("RGB")
                x = pil_to_model_input(img, preprocess, device) # (1,C,H,W)
                # Use the new method to get features
                feats = model.get_flattened_features(x) # (1, D)
                features_list.append(feats.cpu().numpy())
            except Exception as e:
                print(f"[WARN] Skipping {img_path} during feature extraction: {e}", flush=True)
                continue
    
    if not features_list:
        raise RuntimeError("No features could be extracted for K-means.")
        
    features = np.vstack(features_list) # (N, D)

    # --- 3. Run K-means ---
    print(f"[INFO] Running K-means to find {k} clusters...", flush=True)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(features)

    # --- 4. Find exemplars (images closest to centers) ---
    # Get distances from each sample to each cluster center
    distances = kmeans.transform(features) # (N, k)
    exemplar_indices_in_sample = np.argmin(distances, axis=0) # (k,)
    
    # Map these sample-relative indices back to the *original* all_paths indices
    exemplar_paths = [paths_to_sample[i] for i in exemplar_indices_in_sample]
    
    # De-duplicate in case one sample is closest to multiple centers
    exemplar_paths = sorted(list(set(exemplar_paths)))
    print(f"[INFO] Found {len(exemplar_paths)} unique K-means exemplars.", flush=True)

    # --- 5. Load, Preprocess (NO BLUR), and Stack ---
    bgs = []
    print("[INFO] Loading and preprocessing K-means exemplars...", flush=True)
    for img_path in exemplar_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            # CRITICAL: Just preprocess, do not blur
            bgs.append(preprocess(img))
        except Exception as e:
            print(f"[WARN] Failed to load exemplar {img_path}: {e}", flush=True)

    if not bgs:
        raise RuntimeError("Failed to load any K-means exemplars.")

    return torch.stack(bgs, dim=0).to(device)  # (K,C,H,W)


# ------------------------------ saving utils -----------------------------

def save_overlay(image_hwc_uint8: np.ndarray,
                 shap_chw: np.ndarray,
                 out_path: str,
                 title: str = "SHAP attribution"):
    """
    Use shap.image_plot to render a red/blue overlay and save to disk.
    image_hwc_uint8: (H,W,3) uint8  -- MUST match SHAP spatial dims
    shap_chw:        (C,H,W) float
    """
    Hs, Ws = shap_chw.shape[1], shap_chw.shape[2]
    Hi, Wi = image_hwc_uint8.shape[0], image_hwc_uint8.shape[1]
    if (Hi, Wi) != (Hs, Ws):
        raise ValueError(f"Overlay size mismatch: image ({Hi},{Wi}) vs SHAP ({Hs},{Ws})")

    # Convert to shapes shap.image_plot expects
    x_vis = image_hwc_uint8.astype(np.float32) / 255.0         # (H, W, 3)
    x_vis = x_vis[None, ...]                                   # (1, H, W, 3)
    shap_hwc = shap_chw.transpose(1, 2, 0)[None, ...]          # (1, H, W, C)

    plt.figure(figsize=(8, 4.5), dpi=160)
    shap.image_plot([shap_hwc], x_vis, show=False)
    plt.suptitle(title, y=0.95)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_saliency_grayscale(saliency_hw: np.ndarray, out_path: str):
    """
    Save |SHAP| (H,W) as 8-bit grayscale PNG with robust normalization.
    """
    v = saliency_hw.astype(np.float32)
    vmax = float(np.percentile(v, 99.0)) if np.any(v > 0) else float(v.max())
    if vmax <= 1e-12:
        vmax = 1.0
    v = np.clip(v / vmax, 0.0, 1.0)
    img = (v * 255.0).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def tensor_to_vis_uint8(x_1chw: torch.Tensor) -> np.ndarray:
    """
    Inverse-normalize model input (1,C,H,W) from [-1,1] back to uint8 HxWx3 for visualization.
    """
    v = x_1chw[0].detach().cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
    # Inverse Normalize(mean=0.5, std=0.5): x = (x*std)+mean
    v = (v * 0.5) + 0.5
    v = np.clip(v, 0.0, 1.0)
    return (v * 255.0).astype(np.uint8)


# ---------------------------- SHAP per image -----------------------------

def explain_one_image(model: nn.Module,
                      dev: str,
                      img_path: Path,
                      out_dir: Path,
                      preprocess,
                      bg_mode: str,
                      shared_explainer: Optional[shap.GradientExplainer],
                      bg_count: int,
                      seed: int = 42) -> bool:
    """
    Compute prediction + SHAP for a single image file and write artifacts to out_dir.
    Returns True on success, False on (recoverable) failure.
    """
    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] cannot open image: {img_path} ({e})", flush=True)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    # Input tensor (1,C,H,W) at model input size
    x = pil_to_model_input(img_pil, preprocess, dev)  # 1,C,H,W
    np.save(out_dir / "preprocessed_input.npy", x.detach().cpu().numpy())

    with torch.no_grad():
        pred = model(x).view(-1).item()

    # Choose explainer per bg_mode
    # --- MODIFIED: Added fixed_dataset_kmeans to shared explainer logic ---
    if bg_mode in ("fixed_zero", "fixed_dataset_blur", "fixed_dataset_kmeans"):
        explainer = shared_explainer
        if explainer is None:
            raise RuntimeError(f"shared_explainer is None for {bg_mode} mode")
        
        # Reset seeds for determinism across images
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        shap_vals_list = explainer.shap_values(x)
    elif bg_mode == "per_frame_blur":
        bg = build_blur_background(img_pil, preprocess, dev, n=bg_count)
        explainer_local = shap.GradientExplainer(model, bg)
        shap_vals_list = explainer_local.shap_values(x)
    else:
        raise ValueError(f"Unknown bg_mode: {bg_mode}")

    shap_vals = shap_vals_list[0] if isinstance(shap_vals_list, list) else shap_vals_list
    if torch.is_tensor(shap_vals):
        shap_vals = shap_vals.detach().cpu().numpy()  # (1,C,H,W)
    shap_chw = shap_vals[0]  # (C,H,W)

    # Saliency = sum |SHAP| across channels
    shap_sum_abs = np.abs(shap_chw).sum(axis=0)  # (H,W)
    
    # Debug checksum
    # print(f"[DEBUG] {img_path.name} SHAP checksum: {np.sum(shap_chw):.6f}", flush=True)

    np.save(out_dir / "shap_values.npy", shap_chw)
    np.save(out_dir / "shap_sum_abs.npy", shap_sum_abs)

    # Visualization at the SAME spatial size as SHAP (model input size)
    img_for_overlay = tensor_to_vis_uint8(x)  # (H,W,3) uint8 matching SHAP dims

    save_overlay(
        img_for_overlay,
        shap_chw,
        str(out_dir / "shap_overlay.png"),
        title=f"SHAP (Gradient) — steer={pred:.3f}",
    )
    save_saliency_grayscale(shap_sum_abs, str(out_dir / "saliency.png"))

    # Report
    in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
    report = {
        "image": str(img_path.resolve()),
        "input_shape": [in_h, in_w],
        "device": dev,
        "predicted_steer": float(pred),
        "bg_mode": bg_mode,
        "bg_samples": int(bg_count),
        "artifacts": {
            "overlay_png": str((out_dir / "shap_overlay.png").resolve()),
            "saliency_png": str((out_dir / "saliency.png").resolve()),
            "shap_values_npy": str((out_dir / "shap_values.npy").resolve()),
            "shap_sum_abs_npy": str((out_dir / "shap_sum_abs.npy").resolve()),
            "preprocessed_input_npy": str((out_dir / "preprocessed_input.npy").resolve()),
        },
    }
    with open(out_dir / "pred.json", "w") as f:
        json.dump(report, f, indent=2)

    return True


# ----------------------------- JSONL helpers -----------------------------

_TOWN_RE = re.compile(r"(Town\d+HD|Town\d+)", re.IGNORECASE)

def infer_town(path_like: Union[str, Path]) -> Optional[str]:
    m = _TOWN_RE.search(str(path_like))
    return m.group(1) if m else None


def out_dir_for(entry: Dict[str, Any], img_rel: Union[str, Path], out_root: Path) -> Path:
    """
    Build out_shap/TownXX/<filename_stem>/ given entry and relative image path.
    """
    town = entry.get("town") or infer_town(str(img_rel)) or "UnknownTown"
    stem = Path(str(img_rel)).stem  # e.g., Town13_124742
    return out_root / town / stem


# ---------------------------------- CLI ----------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to DAVE2 checkpoint")
    # Single-image mode:
    ap.add_argument("--image", help="Path to a single RGB image (png/jpg)")
    ap.add_argument("--out", help="Output directory (single-image mode)")
    # Batch mode:
    ap.add_argument("--jsonl", help="Path to all_measurements.jsonl for batch processing")
    ap.add_argument("--img-root", type=str, default="", help="Root dir to prepend to image_path in JSONL")
    ap.add_argument("--out-root", type=str, default="out_shap", help="Root directory for batch outputs")
    # Common:
    ap.add_argument("--input-h", type=int, default=180)
    ap.add_argument("--input-w", type=int, default=320)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--bg", type=int, default=16, help="# background samples for SHAP (K for kmeans/blur modes)")
    ap.add_argument("--bg-seed", type=int, default=42, help="Seed for building blurred/kmeans backgrounds")
    ap.add_argument("--bg-mode", default="fixed_dataset_kmeans",
                    choices=["fixed_zero", "per_frame_blur", "fixed_dataset_blur", "fixed_dataset_kmeans"],
                    help="Baseline strategy: "
                         "'fixed_zero' uses zeros in normalized space (mid-gray), shared explainer; "
                         "'per_frame_blur' builds blurred baselines per frame (legacy, not cross-comparable); "
                         "'fixed_dataset_blur' samples K evenly-spaced frames, blurs them once, and reuses shared explainer; "
                         "'fixed_dataset_kmeans' runs K-means on features from a large sample, finds K real (un-blurred) "
                         "exemplar images, and reuses shared explainer (RECOMMENDED for cross-frame comparisons).")
    
    # Batch controls:
    ap.add_argument("--max", type=int, default=0, help="Max samples to process (0=all)")
    ap.add_argument("--start", type=int, default=0, help="Start index (0-based) in JSONL")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip if saliency.png already exists for that sample")
    ap.add_argument("--kmeans-sample-size", type=int, default=5000, 
                    help="How many images to sample from JSONL for K-means fitting")
    return ap.parse_args()


def main():
    # Force deterministic algorithms for CuBLAS (needed for torch.use_deterministic_algorithms)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = parse_args()

    # Ensure determinism
    seed = args.bg_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # Older torch versions might not have this or warn_only

    # --- MODIFIED: Cast model to DAVE2v1 type hint ---
    model: DAVE2v1
    model, dev = load_dave2(args.ckpt, input_shape=(args.input_h, args.input_w), device=args.device)
    preprocess = build_preprocess((args.input_h, args.input_w))

    C = 3
    H, W = args.input_h, args.input_w
    shared_explainer = None

    # ---- Single-image mode ----
    if args.image and not args.jsonl:
        # Build shared explainer if needed for single image
        if args.bg_mode == "fixed_zero":
            bg_fixed = build_fixed_background(args.bg, C, H, W, device=dev)
            shared_explainer = shap.GradientExplainer(model, bg_fixed)
        elif args.bg_mode in ("fixed_dataset_blur", "fixed_dataset_kmeans"):
            raise SystemExit(f"--bg-mode {args.bg_mode} requires --jsonl to sample backgrounds.")
        elif args.bg_mode == "per_frame_blur":
            pass  # built per-frame inside explain_one_image

        if not args.out:
            raise SystemExit("--out is required for single-image mode")
        img_path = Path(args.image)
        out_dir = Path(args.out)
        ok = explain_one_image(model, dev, img_path, out_dir, preprocess,
                               args.bg_mode, shared_explainer, args.bg,
                               seed=args.bg_seed)
        if ok:
            print(f"[OK] Wrote artifacts to: {out_dir.resolve()}", flush=True)
        else:
            print(f"[ERR] Failed for: {img_path}", flush=True)
        return

    # ---- Batch JSONL mode ----
    if not args.jsonl:
        raise SystemExit("Provide --jsonl for batch mode or --image/--out for single-image mode.")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL not found: {jsonl_path}")

    img_root = Path(args.img_root) if args.img_root else Path(".")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Read all lines once
    with open(jsonl_path, "r") as f:
        all_lines = f.readlines()

    # Build shared explainer for fixed modes
    if args.bg_mode == "fixed_zero":
        bg_fixed = build_fixed_background(args.bg, C, H, W, device=dev)
        shared_explainer = shap.GradientExplainer(model, bg_fixed)
    elif args.bg_mode == "fixed_dataset_blur":
        bg_fixed = build_fixed_dataset_blur_background_from_lines(
            all_lines, img_root, preprocess, dev, k=args.bg, seed=args.bg_seed
        )
        print(f"[INFO] Built fixed_dataset_blur background set with {bg_fixed.shape[0]} samples.", flush=True)
        shared_explainer = shap.GradientExplainer(model, bg_fixed)
    # --- NEW: K-means background builder ---
    elif args.bg_mode == "fixed_dataset_kmeans":
        bg_fixed = build_fixed_dataset_kmeans_background_from_lines(
            all_lines, img_root, preprocess, dev, k=args.bg,
            model=model, # Pass the model in
            seed=args.bg_seed,
            max_samples_for_kmeans=args.kmeans_sample_size
        )
        print(f"[INFO] Built fixed_dataset_kmeans background set with {bg_fixed.shape[0]} samples.", flush=True)
        print(f"[INFO] Background checksum: {bg_fixed.sum().item():.4f}", flush=True)
        shared_explainer = shap.GradientExplainer(model, bg_fixed)


    total = 0
    processed = 0
    skipped_missing = 0
    skipped_existing = 0
    errors = 0

    # Slicing controls
    lines = all_lines[args.start:]
    if args.max > 0:
        lines = lines[:args.max]

    for i, line in enumerate(lines, start=args.start):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON parse error at line {i}: {e}", flush=True)
            errors += 1
            continue

        img_rel = entry.get("image_path")
        if not img_rel:
            print(f"[WARN] no 'image_path' at line {i}", flush=True)
            errors += 1
            continue

        # Compose absolute image path (absolute img_rel will override img_root automatically)
        img_abs = (img_root / img_rel)
        img_abs = img_abs if Path(img_abs).is_absolute() else Path(img_abs).resolve()

        if not Path(img_abs).exists():
            print(f"[WARN] missing image, skipping: {img_abs}", flush=True)
            skipped_missing += 1
            total += 1
            continue

        # Determine output dir like out_shap/TownXX/TownXX_xxxxxx/
        out_dir = out_dir_for(entry, img_rel, out_root)
        if args.skip_existing and (out_dir / "saliency.png").exists():
            skipped_existing += 1
            total += 1
            continue

        ok = explain_one_image(model, dev, Path(img_abs), out_dir, preprocess,
                               args.bg_mode, shared_explainer, args.bg,
                               seed=args.bg_seed)
        if ok:
            processed += 1
        else:
            errors += 1

        total += 1
        if total % 25 == 0:
            print(f"[PROGRESS] total={total} processed={processed} "
                  f"missing={skipped_missing} skipped_existing={skipped_existing} errors={errors}",
                  flush=True)

        # (Optional) free some GPU memory between iterations
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"[DONE] total={total} processed={processed} "
          f"missing={skipped_missing} skipped_existing={skipped_existing} errors={errors}",
          flush=True)


if __name__ == "__main__":
    # --- MODIFIED: This is the CORRECT implementation ---
    def get_flattened_features(self: DAVE2v1, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method to extract features for K-means.
        This MUST match the model's forward() pass.
        We'll extract the 100-dim output of the first FC layer.
        """
        # Run the conv part of the forward pass
        x = self.bn1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Flatten the conv features
        x = x.flatten(1) 
        
        # Get the 100-dim feature vector from the first FC layer
        x = F.relu(self.fc1(x))  
        return x

    # Monkey-patch the new method onto the DAVE2v1 class
    DAVE2v1.get_flattened_features = get_flattened_features
    # --- End modification ---
    
    main()