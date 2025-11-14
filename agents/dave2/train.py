# train/train.py
import glob
import json
import math
import numpy as np
import os
import time
from pathlib import Path
from typing import List, Dict
import hashlib
import random

import torch
import torch.nn as nn
import re
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ColorJitter
from PIL import Image, ImageFile, ImageOps

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from models.dave2 import DAVE2v1


def _p(msg: str) -> None:
    """Print with flush so logs show up promptly."""
    print(msg, flush=True)


ImageFile.LOAD_TRUNCATED_IMAGES = True

class CarlaSteerDataset(Dataset):
    """
    Expects a list of JSONL files where each line has (at least):
      { "image_path": "...", "steer": float, or "control": {"steer": ...} }
    If your logs store image file names relative to a root, set img_root.
    """

    def __init__(self,
                 jsonl_files: List[Path],
                 img_root: Path = None,
                 resize=(180, 320),
                 prefilter_missing: bool = True,
                 augment: bool = False,
                 hflip_prob: float = 0.5,
                 jitter: float = 0.2,
                 noise_std: float = 0.01):
        self.samples: List[Dict] = []
        self.img_root = img_root

        # Augmentation config
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.noise_std = noise_std

        # Build transforms in pieces so we can mix PIL- and tensor-based steps
        self.resize = Resize(resize)
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
        self.color_jitter = (
            ColorJitter(brightness=jitter,
                        contrast=jitter,
                        saturation=jitter,
                        hue=0.05)
            if augment and jitter > 0.0 else None
        )

        n_total = 0
        n_kept = 0
        for f in jsonl_files:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n_total += 1
                    rec = json.loads(line)

                    # Resolve path and (optionally) prefilter if it doesn't exist
                    p = self._resolve_path(rec.get("image_path", ""))
                    if prefilter_missing:
                        if p is None or not p.exists():
                            continue
                    self.samples.append(rec)
                    n_kept += 1

        if prefilter_missing:
            _p(f"[CarlaSteerDataset] loaded records: {n_kept}/{n_total} kept "
               f"(dropped {n_total - n_kept} missing paths)")

    def _resolve_path(self, path_str: str) -> Path:
        if not path_str:
            return None
        p = Path(path_str)
        if self.img_root is not None and not p.is_absolute():
            p = self.img_root / p
        return p

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return None on any load error; collate_fn will drop it.
        r = self.samples[idx]
        try:
            # image path
            p = self._resolve_path(r["image_path"])
            if p is None or not p.exists():
                return None  # missing file => skip

            with Image.open(p) as im:
                img = im.convert("RGB")

            # steering: control.steer (fallback to top-level "steer")
            if "control" in r and "steer" in r["control"]:
                steer = float(r["control"]["steer"])
            else:
                steer = float(r["steer"])

            # ---- TRAIN-TIME AUGMENTATION (if enabled) ----
            if self.augment:
                # 1) Horizontal flip + sign flip on steering
                if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
                    img = ImageOps.mirror(img)
                    steer = -steer

                # 2) Geometric resize
                img = self.resize(img)

                # 3) Color jitter on PIL image
                if self.color_jitter is not None:
                    img = self.color_jitter(img)

                # 4) To tensor
                x = self.to_tensor(img)

                # 5) Add small Gaussian noise on tensor (simulate sensor noise)
                if self.noise_std > 0.0:
                    noise = torch.randn_like(x) * self.noise_std
                    x = x + noise
                    # Keep in valid [0,1] range before Normalize
                    x = x.clamp(0.0, 1.0)

                # 6) Normalize
                x = self.normalize(x)

            else:
                # No augmentation: same behavior as before
                img = self.resize(img)
                x = self.normalize(self.to_tensor(img))

            # Clamp steering to [-1, 1] after any sign flip
            steer = max(-1.0, min(1.0, steer))
            y = torch.tensor([steer], dtype=torch.float32)
            return x, y

        except Exception:
            # Any problem (file missing, unreadable, bad JSON fields, etc.) => skip
            return None


def collate_skip_missing(batch):
    """Filter out None samples produced by the Dataset on load errors."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # training loop will skip this batch
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)

def _infer_group(rec: Dict, group_key: str, from_path: bool) -> str:
    # Try explicit key first
    if group_key and (group_key in rec) and rec[group_key] not in (None, ""):
        return str(rec[group_key])
    if from_path:
        s = rec.get("image_path", "") or ""
        m = re.search(r"(Town\d+HD|Town\d+)", s)
        if m:
            return m.group(1)
    return "unknown"

def _stable_group_seed(group: str, global_seed: int) -> int:
    """
    Turn (group name, global seed) into a stable 32-bit integer seed.
    This is independent of Python's hash() randomization.
    """
    data = f"{group}|{global_seed}".encode("utf-8")
    h = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(h, "little") & 0xFFFFFFFF

def _build_equal_subset_indices(samples: List[Dict],
                                group_key: str,
                                from_path: bool,
                                subset_total: int,
                                subset_per_group: int,
                                subset_percent: float,
                                seed: int):
    """
    Deterministically select equal counts per group (town).
    Precedence: per_group > total > percent.
    Using a stable per-group RNG so larger targets are supersets of smaller ones.
    """
    # group -> list of sample indices
    groups = {}
    for i, r in enumerate(samples):
        g = _infer_group(r, group_key, from_path)
        groups.setdefault(g, []).append(i)

    # decide target per group
    G = len(groups)
    if G == 0:
        return None, {}
    if subset_per_group and subset_per_group > 0:
        target = subset_per_group
    elif subset_total and subset_total > 0:
        target = max(1, subset_total // G)
    elif subset_percent and subset_percent > 0:
        min_size = min(len(v) for v in groups.values())
        target = max(1, int(min_size * float(subset_percent)))
    else:
        return None, {g: len(v) for g, v in groups.items()}  # no subsetting

    # deterministic per-group shuffle, take first k
    chosen = []
    out_counts = {}
    for g, idxs in groups.items():
        # Stable seed derived from group name + global seed.
        # This makes the permutation for each (group, seed) pair fixed,
        # so larger targets are supersets of smaller ones.
        group_seed = _stable_group_seed(str(g), seed)
        rng = np.random.default_rng(group_seed)

        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)

        k = min(target, len(idxs))
        out_counts[g] = int(k)
        if k > 0:
            chosen.append(idxs[:k])

    if not chosen:
        return None, {g: 0 for g in groups.keys()}
    chosen = np.concatenate(chosen)
    # keep global order deterministic but irrelevant (DataLoader shuffles anyway)
    chosen.sort(kind="mergesort")
    return chosen.tolist(), out_counts

def expand_jsonl_args(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        # explicit glob patterns
        if any(ch in p for ch in "*?[]"):
            for q in glob.glob(p, recursive=True):
                Q = Path(q)
                if Q.suffix.lower() == ".jsonl" and Q.exists():
                    rp = Q.resolve()
                    if rp not in seen:
                        out.append(Q); seen.add(rp)
            continue
        P = Path(p)
        if P.is_dir():
            for Q in sorted(P.rglob("*.jsonl")):
                rp = Q.resolve()
                if rp not in seen:
                    out.append(Q); seen.add(rp)
        else:
            rp = P.resolve()
            if rp not in seen:
                out.append(P); seen.add(rp)
    if not out:
        raise FileNotFoundError(f"No JSONL files found in {paths}")
    return out


def train_one_epoch(model, loader, opt, loss_fn, device="cuda", amp=False, scaler=None, epoch=0):
    model.train()
    total = 0.0
    n = 0
    start = time.time()

    num_batches = len(loader)
    interval = max(1, min(100, num_batches // 10))

    for i, batch in enumerate(loader):
        if batch is None:          # all items in this batch were bad => skip
            continue
        x, y = batch
        if x.numel() == 0:         # paranoia guard
            continue

        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                yhat = model(x)
                loss = loss_fn(yhat, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()

        bs = x.size(0)
        total += float(loss) * bs
        n += bs

        if (i + 1) % interval == 0 or (i + 1) == num_batches:
            elapsed = time.time() - start
            fps = n / elapsed if elapsed > 0 else float("inf")
            _p(f"[epoch {epoch:02d}] batch {i+1:>5d}/{num_batches:<5d} "
               f"curr_loss={loss.item():.6f} avg_loss={total/max(1,n):.6f} "
               f"samples={n} ~{fps:.1f} samples/s")

    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device="cuda", epoch=0):
    if loader is None:
        return float("nan")
    model.eval()
    total = 0.0
    n = 0
    start = time.time()
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        if x.numel() == 0:
            continue
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss = loss_fn(yhat, y)
        total += float(loss) * x.size(0)
        n += x.size(0)
    elapsed = time.time() - start
    _p(f"[epoch {epoch:02d}] validation done in {elapsed:.2f}s")
    return total / max(1, n)

def _set_seeds(seed: int, device: str):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _save_checkpoint(model, opt, scaler, out_path: Path, meta: Dict):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "meta": meta,
    }, out_path)

def _load_checkpoint(path: Path, model, opt=None, scaler=None, map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if not strict and (missing or unexpected):
        _p(f"[resume] non-strict load: missing={missing} unexpected={unexpected}")

    if opt is not None and ckpt.get("optimizer") is not None:
        try:
            opt.load_state_dict(ckpt["optimizer"])
            # ensure optimizer tensors are on the right device
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(next(model.parameters()).device, non_blocking=True)
        except Exception as e:
            _p(f"[resume] optimizer state not loaded: {e}")

    if scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            _p(f"[resume] scaler state not loaded: {e}")

    meta = ckpt.get("meta", {}) or {}
    return meta

def _maybe_resume(model, opt, scaler, args, device):
    start_epoch = 1
    best = math.inf
    if args.resume:
        meta = _load_checkpoint(Path(args.resume), model, opt, scaler,
                                map_location="cpu", strict=not args.resume_relaxed)
        start_epoch = int(meta.get("epoch", 0)) + 1
        best = float(meta.get("val_loss", math.inf))
        _p(f"[resume] from {args.resume} -> start_epoch={start_epoch} best_val={best}")
    return start_epoch, best



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", nargs="+", help="Paths to JSONL log files for training")
    parser.add_argument("--val-jsonl", nargs="*", default=[], help="Paths to JSONL log files for validation")
    parser.add_argument("--img-root", type=str, default=None, help="Root to prepend to relative image_path entries")
    parser.add_argument(
        "--val-img-root",
        type=str,
        default=None,
        help="Root directory for validation images; defaults to --img-root if not set."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="checkpoints/dave2v1.pt")
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a checkpoint every K epochs (0 disables periodic saves)")

    # dataloader & reproducibility
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # training niceties
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")

    # --- data augmentation (train only) ---
    parser.add_argument("--augment", action="store_true",
                        help="Enable train-time data augmentation (flip, jitter, noise).")
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5,
                        help="Probability of horizontal flip for train images.")
    parser.add_argument("--aug-jitter", type=float, default=0.2,
                        help="Color jitter strength (0 disables if 0).")
    parser.add_argument("--aug-noise-std", type=float, default=0.01,
                        help="Std dev of Gaussian noise added after ToTensor for train images.")

    # --- subset options (balanced by town) ---
    parser.add_argument("--subset-per-group", type=int, default=0,
                        help="Equal count per group (e.g., per town).")
    parser.add_argument("--subset-total", type=int, default=0,
                        help="Total samples; split evenly across groups.")
    parser.add_argument("--subset-percent", type=float, default=0.0,
                        help="Keep this percent of the smallest group's size for all groups (0.2 => 20%).")
    parser.add_argument("--group-key", type=str, default="town",
                        help="JSON field name for grouping (default: 'town').")
    parser.add_argument("--group-from-path", action="store_true",
                        help="Infer group from image_path using regex '(Town\\d+HD|Town\\d+)'.")

    parser.add_argument("--resume", type=str, default="",
        help="Path to a checkpoint .pt to resume from.")
    parser.add_argument("--resume-relaxed", action="store_true",
        help="Load model with strict=False (allows missing/unexpected keys).")

    args = parser.parse_args()

    # If val-img-root is not set, default it to img-root
    if args.val_img_root is None:
        args.val_img_root = args.img_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _set_seeds(args.seed, device)

    # Datasets / loaders
    train_paths = expand_jsonl_args(args.train_jsonl)
    val_paths = expand_jsonl_args(args.val_jsonl) if args.val_jsonl else []

    _p("=== Run config ===")
    _p(f"device={device}  amp={args.amp}  seed={args.seed}")
    _p(f"image_size=({args.height},{args.width})  batch_size={args.bs}  lr={args.lr}")
    _p(f"train_files={len(train_paths)}  val_files={len(val_paths)}")
    if args.img_root:
        _p(f"img_root={args.img_root}")
    if args.val_img_root and args.val_img_root != args.img_root:
        _p(f"val_img_root={args.val_img_root}")
    _p(f"workers(train/val)=({args.workers}/{args.val_workers})")
    _p(f"checkpoint_out={args.out}")
    if args.save_every:
        _p(f"periodic_saves=every {args.save_every} epoch(s)")
    _p(f"augment={args.augment}  hflip_prob={args.aug_hflip_prob}  "
       f"jitter={args.aug_jitter}  noise_std={args.aug_noise_std}")
    _p("===================")

    train_ds = CarlaSteerDataset(
        train_paths,
        img_root=Path(args.img_root) if args.img_root else None,
        resize=(args.height, args.width),
        prefilter_missing=True,
        augment=args.augment,
        hflip_prob=args.aug_hflip_prob,
        jitter=args.aug_jitter,
        noise_std=args.aug_noise_std,
    )

    val_ds = (
        CarlaSteerDataset(
            val_paths,
            img_root=Path(args.val_img_root) if args.val_img_root else None,
            resize=(args.height, args.width),
            prefilter_missing=True,
            augment=False,  # <-- IMPORTANT: keep validation clean
        )
        if val_paths else None
    )


    _p(f"train_samples={len(train_ds)}  val_samples={0 if val_ds is None else len(val_ds)}")

    # === Balanced subsetting by group (e.g., by town) ===
    subset_indices, subset_counts = _build_equal_subset_indices(
        samples=getattr(train_ds, "samples", []),
        group_key=args.group_key,
        from_path=args.group_from_path,
        subset_total=args.subset_total,
        subset_per_group=args.subset_per_group,
        subset_percent=args.subset_percent,
        seed=args.seed
    )

    if subset_indices is not None:
        # Wrap with Subset for deterministic, town-balanced training
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, subset_indices)
        kept = sum(subset_counts.values())
        _p(f"[subset] enabled: kept={kept} samples "
           f"(equal per group). Breakdown: {subset_counts}")
    else:
        _p("[subset] disabled (using full training set)")

    pw_train = args.workers > 0
    pw_val = args.val_workers > 0
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, pin_memory=(device=="cuda"), persistent_workers=pw_train, collate_fn=collate_skip_missing)
    val_ld = (DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                         num_workers=args.val_workers, pin_memory=(device=="cuda"), persistent_workers=pw_val, collate_fn=collate_skip_missing)
              if val_ds else None)

    # Model (no speed)
    model = DAVE2v1(input_shape=(args.height, args.width)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device == "cuda") else None

    # Make sure output dir exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Logging paths & history container
    log_csv = out_path.parent / f"{out_path.stem}_log.csv"
    plot_png = out_path.parent / f"{out_path.stem}_loss.png"
    history = []

    # Train
    best = math.inf

    start_epoch = 1
    start_epoch, best = _maybe_resume(model, opt, scaler, args, device)

    _p("Starting training...")
    run_start = time.time()
    for ep in range(start_epoch, start_epoch + args.epochs):
        ep_start = time.time()

        tr = train_one_epoch(model, train_ld, opt, loss_fn, device,
                             amp=args.amp, scaler=scaler, epoch=ep)
        va = evaluate(model, val_ld, loss_fn, device, epoch=ep) if val_ld else float("nan")

        # LR reporting (handles schedulers if added later)
        lr_now = None
        for pg in opt.param_groups:
            lr_now = pg.get("lr", None)
            break

        ep_time = time.time() - ep_start
        _p(f"epoch {ep:02d}  train {tr:.6f}  val {va:.6f}  "
           f"lr={lr_now if lr_now is not None else 'n/a'}  time={ep_time:.2f}s")

        # Record & persist per-epoch metrics
        row = {
            "epoch": ep,
            "train_loss": float(tr),
            "val_loss": float(va),
            "lr": float(lr_now) if lr_now is not None else float("nan"),
            "time_sec": float(ep_time),
            "steps_per_epoch": len(train_ld),
            "batch_size": args.bs,
        }
        history.append(row)
        new_file = not log_csv.exists()
        try:
            with open(log_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new_file:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            _p(f"[warn] failed to write log CSV: {e}")

        # Periodic checkpoint every K epochs (if enabled)
        if args.save_every and (ep % args.save_every == 0):
            save_dir = out_path.parent
            base = out_path.stem
            suffix = out_path.suffix or ".pt"
            per_ep_path = save_dir / f"{base}_ep{ep:03d}{suffix}"
            _save_checkpoint(
                model, opt, scaler, per_ep_path,
                meta={
                    "input_shape": (args.height, args.width),
                    "epoch": ep,
                    "device": device,
                    "train_loss": float(tr),
                    "val_loss": float(va),
                }
            )
            _p(f"[checkpoint] periodic save @ epoch {ep} -> {per_ep_path}")

        # Save best if we have a validation set
        if val_ld and va < best:
            best = va
            _save_checkpoint(
                model, opt, scaler, out_path,
                meta={
                    "input_shape": (args.height, args.width),
                    "epoch": ep,
                    "device": device,
                    "train_loss": float(tr),
                    "val_loss": float(va),
                }
            )
            _p(f"[checkpoint] new best val={va:.6f} saved to {out_path}")

    total_time = time.time() - run_start

    # Fallback save if no val set
    if not val_ld:
        _save_checkpoint(
            model, opt, scaler, out_path,
            meta={
                "input_shape": (args.height, args.width),
                "epoch": args.epochs,
                "device": device,
            }
        )
        _p(f"[checkpoint] (no val) final model saved to {out_path}")

    # Loss curve plot
    try:
        if len(history) > 0:
            epochs = [int(r["epoch"]) for r in history]
            tr_vals = [float(r["train_loss"]) for r in history]
            has_val = any(not math.isnan(r["val_loss"]) for r in history)
            if has_val:
                va_vals = [float(r["val_loss"]) for r in history]

            plt.figure(figsize=(7, 4.2))
            plt.plot(epochs, tr_vals, label="train loss")
            if has_val:
                plt.plot(epochs, va_vals, label="val loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.title("Training/Validation Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

            ax = plt.gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            xmin = min(epochs)
            xmax = max(epochs)
            plt.xlim(int(xmin), int(xmax))  # lock to integer endpoints

            ymin = 0.0  # MSE >= 0
            ymax_candidates = tr_vals + (va_vals if has_val else [])
            ymax = max(ymax_candidates) if len(ymax_candidates) > 0 else 1.0
            if not math.isfinite(ymax) or ymax <= ymin:
                ymax = ymin + 1.0
            plt.ylim(ymin, ymax)  # fixed y-limits (no autorange)

            plt.savefig(plot_png, dpi=150)
            _p(f"[plot] loss curve saved to {plot_png}")
        else:
            _p("[plot] matplotlib unavailable or no history; skipping loss curve")
    except Exception as e:
        _p(f"[plot] failed to save loss curve: {e}")

    _p(f"Training complete in {total_time:.2f}s")
