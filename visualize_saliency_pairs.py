#!/usr/bin/env python3
"""Render saliency comparison panels from a CSV of consecutive frame metrics."""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize image and saliency pairs with metrics")
    parser.add_argument("csv", type=Path, help="CSV with saliency SSIM summary rows")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the figure")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to save multi-part figures when data exceeds max rows")
    parser.add_argument("--max-rows", type=int, default=100, help="Maximum rows rendered per figure")
    parser.add_argument("--row-limit", type=int, default=None,
                        help="Optional cap on total rows loaded from the CSV")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display")
    args = parser.parse_args()

    if args.output and args.output_dir:
        parser.error("--output and --output-dir are mutually exclusive")
    if args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if args.row_limit is not None and args.row_limit <= 0:
        parser.error("--row-limit must be positive")

    return args


def load_rows(csv_path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def to_label(value: str, name: str) -> str:
    if not value:
        return f"{name}: N/A"
    try:
        return f"{name}: {float(value):.4f}"
    except ValueError:
        return f"{name}: {value}"


def _create_figure(rows: List[Dict[str, str]], args: argparse.Namespace,
                   part_idx: Optional[int] = None, total_parts: Optional[int] = None) -> plt.Figure:
    width_ratios = [1.1, 1.1, 0.9]
    fig_height = max(3.4 * len(rows), 3.5)
    fig = plt.figure(figsize=(sum(width_ratios) * 2.3, fig_height))
    gs = fig.add_gridspec(nrows=len(rows), ncols=len(width_ratios), width_ratios=width_ratios,
                          hspace=0.45, wspace=0.2)

    for idx, row in enumerate(rows):
        prev_spec = gs[idx, 0].subgridspec(2, 1, hspace=0.05)
        curr_spec = gs[idx, 1].subgridspec(2, 1, hspace=0.05)

        axes_info = [
            (prev_spec[0, 0], "RGB prev", row.get("prev_img_path", ""), True),
            (prev_spec[1, 0], "Saliency prev", row.get("prev_path", ""), False),
            (curr_spec[0, 0], "RGB curr", row.get("curr_img_path", ""), True),
            (curr_spec[1, 0], "Saliency curr", row.get("curr_path", ""), False),
        ]

        for spec, title, key_path, show_title in axes_info:
            ax = fig.add_subplot(spec)
            if key_path and Path(key_path).exists():
                ax.imshow(plt.imread(key_path))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            if show_title:
                ax.set_title(title, fontsize=8)
            else:
                ax.set_title("")
                ax.text(0.5, -0.08, title, transform=ax.transAxes,
                        ha="center", va="top", fontsize=8)
            ax.axis("off")

        ax_text = fig.add_subplot(gs[idx, 2])
        text_lines = [
            f"Town: {row.get('town', 'N/A')}",
            f"Frames: {row.get('prev_frame', '?')} → {row.get('curr_frame', '?')}",
            to_label(row.get("ssim", ""), "Saliency SSIM"),
            to_label(row.get("image_ssim", ""), "Image SSIM"),
            to_label(row.get("steer_sim", ""), "Steer sim"),
        ]
        note = row.get("note", "")
        if note:
            text_lines.append(f"Note: {note}")
        ax_text.text(0.02, 0.95, "\n".join(text_lines), ha="left", va="top", fontsize=8)
        ax_text.axis("off")

    fig.tight_layout()

    return fig


def render(rows: List[Dict[str, str]], args: argparse.Namespace) -> None:
    if not rows:
        print("No rows to visualize.")
        return

    chunk_size = args.max_rows

    if args.output_dir:
        save_dir = args.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        total_parts = math.ceil(len(rows) / chunk_size)
        for part_idx in range(total_parts):
            start = part_idx * chunk_size
            end = start + chunk_size
            chunk = rows[start:end]
            fig = _create_figure(chunk, args, part_idx=part_idx, total_parts=total_parts)
            output_name = f"{args.csv.stem}_part{part_idx + 1:03d}.png"
            output_path = save_dir / output_name
            fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved figure to {output_path}")
            plt.close(fig)
        return

    fig = _create_figure(rows, args)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {args.output}")

    if not args.no_show and not plt.isinteractive():
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    limit = args.row_limit
    if limit is None and args.output_dir is None:
        limit = args.max_rows
    rows = load_rows(args.csv, limit)
    render(rows, args)


if __name__ == "__main__":
    main()
