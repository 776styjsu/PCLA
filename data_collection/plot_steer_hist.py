#!/usr/bin/env python3
"""
Plot histograms of leftward and rightward steering from JSONL logs.

Each line in the input file(s) is a JSON object with "control.steer".
You may pass files, directories (recursively scanned for *.jsonl), or glob patterns.

Examples:
  python plot_steer_hist.py logs/session.jsonl
  python plot_steer_hist.py logs_dir/                 # recurse for *.jsonl
  python plot_steer_hist.py "logs/**/run*.jsonl"      # glob (quote in zsh)
  python plot_steer_hist.py logs_dir/ other.jsonl --bins 60 --deadzone 0.03

Notes:
- "deadzone" ignores tiny |steer| values (treated as neutral).
- "domain" sets the histogram range half-span; for CARLA, steer ∈ [-1, 1], so 1.0 is a good default.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def iter_jsonl_files(inputs: Iterable[str]) -> List[Path]:
    """Resolve input arguments into a sorted list of JSONL files."""
    files: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.is_file():
            if p.suffix == ".jsonl":
                files.append(p)
        else:
            # Treat as glob pattern
            matches = [m for m in Path().glob(raw) if m.is_file() and m.suffix == ".jsonl"]
            files.extend(sorted(matches))
    # De-duplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def load_steers(jsonl_path: Path, args: argparse.Namespace) -> List[float]:
    steers = []
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if args.recording_only and obj.get("recording", False):
                    continue
                s = obj.get("control", {}).get("steer")
                if s is None:
                    continue
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(s):
                    steers.append(s)
    except OSError as e:
        print(f"Warning: could not read {jsonl_path}: {e}")
    return steers


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to JSONL file(s), directories (recurse), or glob patterns",
    )
    p.add_argument("--bins", type=int, default=40, help="Histogram bins (default: 40)")
    p.add_argument(
        "--deadzone",
        type=float,
        default=0.02,
        help="Treat |steer| <= deadzone as neutral and ignore (default: 0.02)",
    )
    p.add_argument(
        "--domain",
        type=float,
        default=1.0,
        help="Histogram half-range; x in [-domain, domain] (default: 1.0)",
    )
    p.add_argument(
        "--recording-only",
        action="store_true",
        help="Only graph data that has field `recording` set to true"
    )
    p.add_argument(
        "--out",
        default="steer_hist.png",
        help="Save figure to this path (default: steer_hist.png). Use '' to show instead.",
    )
    args = p.parse_args()

    files = iter_jsonl_files(args.inputs)
    if not files:
        print("No JSONL files found from the given inputs.")
        return
    print(f"Found {len(files)} JSONL file(s).")

    # Aggregate all steering samples
    all_steers: List[float] = []
    for f in files:
        steers = load_steers(f, args)
        all_steers.extend(steers)

    dz = float(args.deadzone)
    domain = max(0.0, float(args.domain))

    left = [s for s in all_steers if s < -dz]
    right = [s for s in all_steers if s > dz]
    neutral = [s for s in all_steers if -dz <= s <= dz]

    print(f"Loaded {len(all_steers)} steering samples from {len(files)} file(s).")
    print(f"Left (< -{dz:.3f}):     {len(left)}")
    print(f"Neutral (|s|<= {dz:.3f}): {len(neutral)}")
    print(f"Right (> {dz:.3f}):    {len(right)}")

    # Two subplots sharing y-axis to compare shapes side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    axes[0].hist(left, bins=args.bins, range=(-domain, 0), edgecolor="black")
    axes[0].set_title(f"Leftward steer (s < -{dz:.2f})")
    axes[0].set_xlabel("steer")
    axes[0].set_ylabel("count")
    axes[0].set_xlim(-domain, 0)

    axes[1].hist(right, bins=args.bins, range=(0, domain), edgecolor="black")
    axes[1].set_title(f"Rightward steer (s > {dz:.2f})")
    axes[1].set_xlabel("steer")
    axes[1].set_xlim(0, domain)

    fig.suptitle("Steering Histograms (Left vs Right)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
