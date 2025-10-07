#!/usr/bin/env python3
"""
Plot histograms of vehicle speed from JSONL logs.

Each line in the input file(s) is a JSON object with "speed_kmh".
You may pass files, directories (recursively scanned for *.jsonl), or glob patterns.

Examples:
  python plot_speed_hist.py logs/session.jsonl
  python plot_speed_hist.py logs_dir/                 # recurse for *.jsonl
  python plot_speed_hist.py "logs/**/run*.jsonl"      # glob (quote in zsh)
  python plot_speed_hist.py logs_dir/ other.jsonl --bins 60

Notes:
- "domain" sets the speed histogram range in [0, x].
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


def load_speed(jsonl_path: Path, args: argparse.Namespace) -> List[float]:
    speed = []
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

                # Keep only entries with recording==True if --recording-only is set
                if args.recording_only and not bool(obj.get("recording", False)):
                    continue

                s = obj.get("speed_kmh")
                if s is None:
                    continue

                if s < args.min_speed:
                    continue

                try:
                    s = float(s)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(s):
                    speed.append(s)
    except OSError as e:
        print(f"Warning: could not read {jsonl_path}: {e}")
    return speed


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to JSONL file(s), directories (recurse), or glob patterns",
    )
    p.add_argument("--bins", type=int, default=40, help="Histogram bins (default: 40)")
    p.add_argument(
        "--domain",
        type=float,
        default=100.0,
        help="Histogram range; x in [0, domain] (default: 100.0)",
    )
    p.add_argument(
        "--min-speed",
        type=float,
        default=0.0,
        help="Ignore all data points with speed_kmh < min_speed when plotting")
    p.add_argument(
        "--recording-only",
        action="store_true",
        help="Only graph data with `recording` == true",
    )
    p.add_argument(
        "--out",
        default="speed_hist.png",
        help="Save figure to this path (default: speed_hist.png). Use '' to show instead.",
    )
    args = p.parse_args()

    files = iter_jsonl_files(args.inputs)
    if not files:
        print("No JSONL files found from the given inputs.")
        return
    print(f"Found {len(files)} JSONL file(s).")

    # Aggregate all speed samples
    all_speed: List[float] = []
    for f in files:
        all_speed.extend(load_speed(f, args))

    if not all_speed:
        print("No speed samples found.")
        return

    # ---- PLOT ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(all_speed, bins=args.bins, range=(0.0, args.domain))
    ax.set_xlim(0.0, args.domain)
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Frame count")
    ax.set_title(f"Vehicle speed histogram (n={len(all_speed)})")
    ax.grid(True, alpha=0.3)

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
