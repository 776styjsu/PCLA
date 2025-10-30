# train/train.py
import glob
import json
import math
import numpy as np
import os
import time
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from PIL import Image, ImageFile

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

    def __init__(self, jsonl_files: List[Path], img_root: Path = None, resize=(150, 200),
                 prefilter_missing: bool = True):
        self.samples: List[Dict] = []
        self.img_root = img_root
        self.tf = Compose([
            Resize(resize),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

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
            x = self.tf(img)  # [3,H,W]

            # steering: control.steer (fallback to top-level "steer")
            if "control" in r and "steer" in r["control"]:
                steer = float(r["control"]["steer"])
            else:
                steer = float(r["steer"])

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", nargs="+", help="Paths to JSONL log files for training")
    parser.add_argument("--val_jsonl", nargs="*", default=[], help="Paths to JSONL log files for validation")
    parser.add_argument("--img_root", type=str, default=None, help="Root to prepend to relative image_path entries")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="checkpoints/dave2v1.pt")
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)

    # dataloader & reproducibility
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # training niceties
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only)")

    args = parser.parse_args()

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
    _p(f"workers(train/val)=({args.workers}/{args.val_workers})")
    _p(f"checkpoint_out={args.out}")
    _p("===================")

    train_ds = CarlaSteerDataset(train_paths,
                                 img_root=Path(args.img_root) if args.img_root else None,
                                 resize=(args.height, args.width))

    val_ds = (CarlaSteerDataset(val_paths,
                                img_root=Path(args.img_root) if args.img_root else None,
                                resize=(args.height, args.width))
              if val_paths else None)

    _p(f"train_samples={len(train_ds)}  val_samples={0 if val_ds is None else len(val_ds)}")

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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Train
    best = math.inf
    _p("Starting training...")
    run_start = time.time()
    for ep in range(1, args.epochs + 1):
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

        # Save best if we have a validation set
        if val_ld and va < best:
            best = va
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "meta": {
                    "input_shape": (args.height, args.width),
                    "epoch": ep,
                    "device": device,
                }
            }, args.out)
            _p(f"[checkpoint] new best val={va:.6f} saved to {args.out}")

    total_time = time.time() - run_start

    # Fallback save if no val set
    if not val_ld:
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "meta": {
                "input_shape": (args.height, args.width),
                "epoch": args.epochs,
                "device": device,
            }
        }, args.out)
        _p(f"[checkpoint] (no val) final model saved to {args.out}")

    _p(f"Training complete in {total_time:.2f}s")
