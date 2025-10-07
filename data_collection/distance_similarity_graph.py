"""
Plot similarity vs. timestep difference.

X-axis: timestep gap (Δ = 1, 2, 3, ... up to a user-specified max or list)
Y-axis: similarity between frames t and t+Δ, computed from either:
  - vector embeddings (e.g., cosine similarity), or
  - a provided pairwise *distance* source (we'll map distance -> similarity)

We *do not* implement image/vector extraction here; we assume inputs already
contain embeddings or distances. This script just aggregates and plots.

Inputs
------
1) measurements JSONL (required): one JSON object per line describing each frame.
   Common fields seen in CARLA logs include "frame", "image_path", etc.
   We only need an ordering and/or a frame id.

   Example line:
     {"frame": 153601, "image_path": "images/153601.png", ...}

2) Embeddings *or* pairwise distances (depends on --mode):
   --mode vector:
       --embeddings: one of
         a) .npy / .npz
              - If .npz: expects key "embeddings" (N x D). Alternative keys can
                be provided via --npz-key.
              - If .npy: expects array (N x D).
            By default we assume row i aligns with the i-th JSONL line.
            If your rows correspond to specific frame ids, pass --embedding-ids
            (N integers) to map rows to frame ids.
         b) .jsonl : each line: {"frame": <int>, "embedding": [float, ...]}
         c) .csv   : header with "frame" and embedding columns e0,e1,... (or
                     no header; use --csv-has-header flag accordingly).

       Similarity is derived from embeddings using --sim-metric
       (cosine|dot|euclidean). For euclidean we map distance->similarity using
       --dist-to-sim (one_over_one_plus_d|exp|one_minus_minmax).

   --mode distance:
       Provide --pairwise file with distances between frames (not similarities).
       Formats supported:
         a) .npy/.npz: an (N x N) square matrix D where D[i,j] is distance.
            If .npz, pick key with --npz-key. If your matrix is indexed by
            frame ids instead of row order, also pass --embedding-ids to give
            the frame id for each row/col.
         b) .jsonl: each line {"frame_i": int, "frame_j": int, "distance": float}
            (symmetric not required; we read both directions as given)
       Distance -> similarity mapping is controlled by --dist-to-sim.

Outputs
-------
- A PNG plot (path via --plot-out)
- Optional CSV of aggregated stats per gap (via --csv-out). Columns:
    gap,count,mean,median,std,p10,p90

Usage
-----
  python plot_similarity_vs_timestep.py \
      --measurements measurements.jsonl \
      --mode vector \
      --embeddings embs.npy \
      --sim-metric cosine \
      --max-gap 30 \
      --plot-out sim_vs_gap.png \
      --csv-out sim_vs_gap.csv \
      --scatter

  # Using pairwise distance matrix:
  python plot_similarity_vs_timestep.py \
      --measurements measurements.jsonl \
      --mode distance \
      --pairwise dists.npy \
      --dist-to-sim one_over_one_plus_d \
      --max-gap 50 \
      --plot-out sim_vs_gap.png

Notes
-----
- If your measurements JSONL lacks a "frame" field, we fall back to 0..N-1.
- "gap" is measured in line index order (i.e., temporal order of JSONL).
- If your data skips frames, this still works as long as ordering is temporal.
- For --gaps "1,2,5,10", we compute only those gaps (overrides --max-gap).
"""

import argparse
import csv
import gzip
import io
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def _open_maybe_gzip(path: str, mode: str = "rt", encoding: str = "utf-8"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)  # type: ignore[arg-type]
    return open(path, mode, encoding=encoding)


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with _open_maybe_gzip(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _maybe_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


# ----------------------------
# Measurements
# ----------------------------

def load_measurements(path: str, frame_key: str = "frame") -> Tuple[List[int], List[dict]]:
    """
    Returns:
      frame_ids: list of frame ids (if missing, we use 0..N-1)
      rows     : the raw json objects in temporal order
    """
    rows = _read_jsonl(path)
    frame_ids = []
    missing = False
    for i, r in enumerate(rows):
        if frame_key in r:
            frame_ids.append(int(r[frame_key]))
        else:
            missing = True
            frame_ids.append(i)
    if missing:
        print(f"[warn] '{frame_key}' missing in some/all rows; using 0..N-1 as ids", file=sys.stderr)
    return frame_ids, rows


# ----------------------------
# Embeddings loaders
# ----------------------------

def load_embeddings(
    path: str,
    npz_key: str = "embeddings",
    csv_has_header: bool = True,
    embedding_ids_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      E: (N, D) embeddings
      ids: (N,) frame ids for each row, or None if row order = measurement order
    """
    ext = os.path.splitext(path)[1].lower()
    ids = None

    if ext in [".npy"]:
        E = np.load(path)
    elif ext in [".npz"]:
        data = np.load(path)
        if npz_key not in data:
            raise KeyError(f"Key '{npz_key}' not found in {path}. Available: {list(data.keys())}")
        E = data[npz_key]
    elif ext in [".jsonl", ".json"]:
        rows = _read_jsonl(path) if ext == ".jsonl" else json.load(open(path, "r", encoding="utf-8"))
        vecs = []
        ids_list = []
        for r in rows:
            if "embedding" not in r:
                raise ValueError("JSON line missing 'embedding' field")
            vecs.append(np.asarray(r["embedding"], dtype=np.float32))
            fid = int(r.get("frame", len(ids_list)))
            ids_list.append(fid)
        E = np.stack(vecs, axis=0)
        ids = np.asarray(ids_list, dtype=np.int64)
    elif ext in [".csv"]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = None
            if csv_has_header:
                header = next(reader, None)
            rows = list(reader)
        # If header contains "frame", use it, else assume first col is frame
        start_col = 0
        frame_col = None
        if csv_has_header and header is not None:
            for j, name in enumerate(header):
                if name.strip().lower() == "frame":
                    frame_col = j
                    break
            start_col = 0
        # Build arrays
        ids_list = []
        vecs = []
        for row in rows:
            if frame_col is not None:
                fid = int(row[frame_col])
                emb = [float(x) for j, x in enumerate(row) if j != frame_col]
            else:
                fid = int(row[0])
                emb = [float(x) for x in row[1:]]
            ids_list.append(fid)
            vecs.append(emb)
        E = np.asarray(vecs, dtype=np.float32)
        ids = np.asarray(ids_list, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported embeddings file extension: {ext}")

    if embedding_ids_path:
        # Optional sidecar ids file aligns with rows of E
        ext2 = os.path.splitext(embedding_ids_path)[1].lower()
        if ext2 in [".npy"]:
            ids = np.load(embedding_ids_path).astype(np.int64)
        elif ext2 in [".npz"]:
            data2 = np.load(embedding_ids_path)
            key = "ids" if "ids" in data2 else list(data2.keys())[0]
            ids = data2[key].astype(np.int64)
        elif ext2 in [".txt", ".csv"]:
            with open(embedding_ids_path, "r", encoding="utf-8") as f:
                ids_list = [int(x.strip().split(",")[0]) for x in f if x.strip()]
            ids = np.asarray(ids_list, dtype=np.int64)
        else:
            raise ValueError(f"Unsupported embedding-ids file extension: {ext2}")

    return E, ids


# ----------------------------
# Pairwise distance loaders
# ----------------------------

class DistanceLookup:
    def __init__(self):
        self.matrix: Optional[np.ndarray] = None
        self.index_ids: Optional[np.ndarray] = None  # frame id for each row/col
        self.map_pair: Dict[Tuple[int, int], float] = {}

    def load(self, path: str, npz_key: str = "D", embedding_ids_path: Optional[str] = None):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".npy"]:
            self.matrix = np.load(path).astype(np.float32)
        elif ext in [".npz"]:
            data = np.load(path)
            if npz_key not in data:
                # fallback: the first key
                npz_key = list(data.keys())[0]
            self.matrix = data[npz_key].astype(np.float32)
        elif ext in [".jsonl", ".json"]:
            rows = _read_jsonl(path) if ext == ".jsonl" else json.load(open(path, "r", encoding="utf-8"))
            for r in rows:
                i = int(r["frame_i"])
                j = int(r["frame_j"])
                d = float(r["distance"])
                self.map_pair[(i, j)] = d
        else:
            raise ValueError(f"Unsupported pairwise distance file: {ext}")

        if self.matrix is not None:
            if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
                raise ValueError("Distance matrix must be square (N x N).")
            if embedding_ids_path:
                ext2 = os.path.splitext(embedding_ids_path)[1].lower()
                if ext2 in [".npy"]:
                    self.index_ids = np.load(embedding_ids_path).astype(np.int64)
                elif ext2 in [".npz"]:
                    data2 = np.load(embedding_ids_path)
                    key = "ids" if "ids" in data2 else list(data2.keys())[0]
                    self.index_ids = data2[key].astype(np.int64)
                elif ext2 in [".txt", ".csv"]:
                    with open(embedding_ids_path, "r", encoding="utf-8") as f:
                        ids_list = [int(x.strip().split(",")[0]) for x in f if x.strip()]
                    self.index_ids = np.asarray(ids_list, dtype=np.int64)
                else:
                    raise ValueError(f"Unsupported embedding-ids file extension: {ext2}")

    def get(self, frame_i: int, frame_j: int) -> Optional[float]:
        # JSONL map path
        if self.map_pair:
            return self.map_pair.get((frame_i, frame_j)) or self.map_pair.get((frame_j, frame_i))
        # Matrix path
        if self.matrix is None:
            return None
        if self.index_ids is None:
            # assume frame ids == row/col indices
            n = self.matrix.shape[0]
            if 0 <= frame_i < n and 0 <= frame_j < n:
                return float(self.matrix[frame_i, frame_j])
            return None
        else:
            # map frame ids to row/col via search
            # build reversed index lazily
            if not hasattr(self, "_rev"):
                self._rev = {int(fid): idx for idx, fid in enumerate(self.index_ids.tolist())}
            ii = self._rev.get(int(frame_i), None)
            jj = self._rev.get(int(frame_j), None)
            if ii is None or jj is None:
                return None
            return float(self.matrix[ii, jj])


# ----------------------------
# Similarity functions
# ----------------------------

def similarity_from_embeddings(u: np.ndarray, v: np.ndarray, metric: str, normalize: bool) -> float:
    if normalize:
        u = _maybe_normalize(u)
        v = _maybe_normalize(v)
    if metric == "cosine":
        # cosine similarity for 1D vectors
        uu = _maybe_normalize(u.reshape(1, -1))[0]
        vv = _maybe_normalize(v.reshape(1, -1))[0]
        return float((uu * vv).sum())
    elif metric == "dot":
        return float((u * v).sum())
    elif metric == "euclidean":
        d = float(np.linalg.norm(u - v))
        # Caller will map distance -> similarity
        return -d  # sentinel: negative meaning "distance (negated)" to be mapped later
    else:
        raise ValueError(f"Unknown sim metric: {metric}")


def map_distance_to_similarity(d: float, mode: str, sigma: float, min_d: Optional[float] = None, max_d: Optional[float] = None) -> float:
    """
    Map a distance (>=0) to a similarity in (0,1] using a chosen transform.
      - one_over_one_plus_d:  1 / (1 + d)
      - exp:                  exp(-d / sigma)  (sigma > 0)
      - one_minus_minmax:     1 - (d - min_d) / (max_d - min_d + eps)
    """
    d = float(d)
    if d < 0:
        d = -d  # in case caller passed negative as sentinel

    if mode == "one_over_one_plus_d":
        return 1.0 / (1.0 + d)
    elif mode == "exp":
        sigma = max(sigma, 1e-12)
        return math.exp(-d / sigma)
    elif mode == "one_minus_minmax":
        if min_d is None or max_d is None:
            raise ValueError("min_d and max_d required for one_minus_minmax")
        denom = (max_d - min_d) if (max_d - min_d) > 1e-12 else 1e-12
        return 1.0 - (d - min_d) / denom
    else:
        raise ValueError(f"Unknown distance->similarity mapping: {mode}")


# ----------------------------
# Main aggregation
# ----------------------------

def collect_pairs_for_gap(
    order_frame_ids: List[int],
    gap: int,
    mode: str,
    E: Optional[np.ndarray] = None,
    E_ids: Optional[np.ndarray] = None,
    pairwise: Optional[DistanceLookup] = None,
    sim_metric: str = "cosine",
    normalize: bool = True,
    dist_to_sim: str = "one_over_one_plus_d",
    sigma: float = 1.0,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
) -> List[float]:
    """Return list of similarities for all pairs (t, t+gap)."""
    sims: List[float] = []

    # Build frame->embedding row index if needed
    frame_to_row = None
    if mode == "vector":
        if E is None:
            raise ValueError("Embeddings E required for mode=vector")
        if E_ids is not None:
            frame_to_row = {int(fid): idx for idx, fid in enumerate(E_ids.tolist())}

    for i in range(0, len(order_frame_ids) - gap):
        f0 = int(order_frame_ids[i])
        f1 = int(order_frame_ids[i + gap])

        if mode == "vector":
            if E_ids is None:
                # assume row order matches measurement order
                if f0 >= E.shape[0] or f1 >= E.shape[0]:
                    continue
                u = E[f0]
                v = E[f1]
            else:
                i0 = frame_to_row.get(f0) if frame_to_row else None
                i1 = frame_to_row.get(f1) if frame_to_row else None
                if i0 is None or i1 is None:
                    continue
                u = E[i0]
                v = E[i1]

            s = similarity_from_embeddings(u, v, sim_metric, normalize)
            if sim_metric == "euclidean":
                # negative s encodes -distance, so flip and map to similarity
                d = -s
                s = map_distance_to_similarity(d, dist_to_sim, sigma, min_d, max_d)
            sims.append(float(s))

        elif mode == "distance":
            if pairwise is None:
                raise ValueError("pairwise DistanceLookup required for mode=distance")
            d = pairwise.get(f0, f1)
            if d is None:
                continue
            s = map_distance_to_similarity(float(d), dist_to_sim, sigma, min_d, max_d)
            sims.append(float(s))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return sims


def aggregate(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "p10": float("nan"), "p90": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": float(np.nanstd(arr)),
        "p10": float(np.nanpercentile(arr, 10)),
        "p90": float(np.nanpercentile(arr, 90)),
    }


def parse_gaps(gaps_str: Optional[str], max_gap: Optional[int], n: int) -> List[int]:
    if gaps_str:
        gaps = [int(x) for x in gaps_str.split(",") if x.strip()]
    else:
        if max_gap is None:
            max_gap = min(50, max(1, n - 1))
        gaps = list(range(1, max_gap + 1))
    gaps = [g for g in gaps if 1 <= g < n]
    return gaps


def main():
    ap = argparse.ArgumentParser(description="Plot similarity vs timestep gap")
    ap.add_argument("--measurements", required=True, help="Path to measurements.jsonl (optionally .gz)")
    ap.add_argument("--frame-key", default="frame", help="JSON key used as frame id (default: frame)")

    ap.add_argument("--mode", choices=["vector", "distance"], required=True,
                    help="Use embeddings to compute similarity, or read pairwise distances.")
    # Embedding inputs
    ap.add_argument("--embeddings", help="Path to embeddings (.npy/.npz/.jsonl/.csv)")
    ap.add_argument("--embedding-ids", help="Optional sidecar ids file (row->frame id) .npy/.npz/.txt/.csv")
    ap.add_argument("--npz-key", default="embeddings", help="Key in .npz for embeddings or distances (default: embeddings)")
    ap.add_argument("--csv-has-header", action="store_true", help="Set if embeddings CSV has a header row")

    # Pairwise distance inputs
    ap.add_argument("--pairwise", help="Path to pairwise *distance* file (.npy/.npz/.jsonl)")
    # Similarity/distance controls
    ap.add_argument("--sim-metric", default="cosine", choices=["cosine", "dot", "euclidean"],
                    help="Similarity metric when mode=vector (euclidean maps to similarity via --dist-to-sim)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize vectors before similarity/dot")
    ap.add_argument("--dist-to-sim", default="one_over_one_plus_d",
                    choices=["one_over_one_plus_d", "exp", "one_minus_minmax"],
                    help="Map distance -> similarity (used for mode=distance or sim-metric=euclidean)")
    ap.add_argument("--sigma", type=float, default=1.0, help="Sigma for dist-to-sim=exp")
    ap.add_argument("--precompute-minmax", action="store_true",
                    help="Scan distances to compute (min,max) for one_minus_minmax mapping")

    # Domain (gaps)
    ap.add_argument("--max-gap", type=int, help="Maximum timestep gap to evaluate (1..max_gap)")
    ap.add_argument("--gaps", type=str, help='Explicit comma-separated gaps, e.g. "1,2,5,10" (overrides --max-gap)')

    # Outputs
    ap.add_argument("--plot-out", required=True, help="Output .png path for the plot")
    ap.add_argument("--csv-out", help="Optional CSV path for aggregated stats")
    ap.add_argument("--title", default="Similarity vs. timestep gap", help="Plot title")
    ap.add_argument("--scatter", action="store_true", help="Overlay scatter of all pair (gap, sim)")
    ap.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    ap.add_argument("--width", type=float, default=7.0, help="Figure width inches")
    ap.add_argument("--height", type=float, default=4.2, help="Figure height inches")

    args = ap.parse_args()

    # Load measurements
    frame_ids, rows = load_measurements(args.measurements, frame_key=args.frame_key)
    n = len(frame_ids)
    if n < 2:
        raise ValueError("Need at least 2 measurements")

    # Prepare inputs depending on mode
    E = None
    E_ids = None
    pairwise = None

    if args.mode == "vector":
        if not args.embeddings:
            raise ValueError("--embeddings is required for mode=vector")
        E, E_ids = load_embeddings(
            args.embeddings,
            npz_key=args.npz_key,
            csv_has_header=args.csv_has_header,
            embedding_ids_path=args.embedding_ids,
        )
        if E.ndim != 2:
            raise ValueError("Embeddings must be 2D (N x D)")

    elif args.mode == "distance":
        if not args.pairwise:
            raise ValueError("--pairwise is required for mode=distance")
        pairwise = DistanceLookup()
        pairwise.load(args.pairwise, npz_key=args.npz_key, embedding_ids_path=args.embedding_ids)

    # Determine gaps
    gaps = parse_gaps(args.gaps, args.max_gap, n)

    # Optional global (min,max) for one_minus_minmax
    global_min_d, global_max_d = None, None
    if args.dist_to_sim == "one_minus_minmax" and args.precompute_minmax:
        # Collect a sample across a subset of gaps for efficiency (or all if small)
        d_vals = []
        sample_gaps = gaps if len(gaps) <= 50 else np.linspace(0, len(gaps)-1, 50, dtype=int).tolist()
        for g in [gaps[i] for i in sample_gaps]:
            sims_or_d = []
            for i in range(0, n - g):
                f0 = int(frame_ids[i]); f1 = int(frame_ids[i+g])
                if args.mode == "vector":
                    # use euclidean only for minmax
                    if E_ids is None:
                        if f0 >= E.shape[0] or f1 >= E.shape[0]:
                            continue
                        u = E[f0]; v = E[f1]
                    else:
                        ft = {int(fid): idx for idx, fid in enumerate(E_ids.tolist())}
                        i0 = ft.get(f0); i1 = ft.get(f1)
                        if i0 is None or i1 is None:
                            continue
                        u = E[i0]; v = E[i1]
                    d = float(np.linalg.norm(u - v))
                    sims_or_d.append(d)
                else:
                    d = pairwise.get(f0, f1) if pairwise else None
                    if d is not None:
                        sims_or_d.append(float(d))
            if sims_or_d:
                d_vals.append(np.asarray(sims_or_d))
        if d_vals:
            all_d = np.concatenate(d_vals, axis=0)
            global_min_d = float(np.nanmin(all_d))
            global_max_d = float(np.nanmax(all_d))
            print(f"[info] Precomputed global (min_d, max_d)=({global_min_d:.6f}, {global_max_d:.6f})", file=sys.stderr)

    # Compute per-gap similarities
    gap_values: Dict[int, List[float]] = {}
    stats_rows = []
    for g in gaps:
        vals = collect_pairs_for_gap(
            order_frame_ids=frame_ids,
            gap=g,
            mode=args.mode,
            E=E,
            E_ids=E_ids,
            pairwise=pairwise,
            sim_metric=args.sim_metric,
            normalize=args.normalize,
            dist_to_sim=args.dist_to_sim,
            sigma=args.sigma,
            min_d=global_min_d,
            max_d=global_max_d,
        )
        gap_values[g] = vals
        st = aggregate(vals)
        stats_rows.append((g, st["count"], st["mean"], st["median"], st["std"], st["p10"], st["p90"]))

    # Save CSV if requested
    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gap", "count", "mean", "median", "std", "p10", "p90"])
            for row in stats_rows:
                w.writerow(row)

    # Plot
    gaps_sorted = sorted(gap_values.keys())
    means = [aggregate(gap_values[g])["mean"] for g in gaps_sorted]
    stds  = [aggregate(gap_values[g])["std"] for g in gaps_sorted]

    plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    plt.plot(gaps_sorted, means, marker="o", linewidth=2, label="mean similarity")
    # Error band (±1 std)
    upper = [m + s if not (np.isnan(m) or np.isnan(s)) else np.nan for m, s in zip(means, stds)]
    lower = [m - s if not (np.isnan(m) or np.isnan(s)) else np.nan for m, s in zip(means, stds)]
    plt.fill_between(gaps_sorted, lower, upper, alpha=0.2, label="±1 std")

    if args.scatter:
        xs = []
        ys = []
        for g in gaps_sorted:
            sims = gap_values[g]
            xs.extend([g] * len(sims))
            ys.extend(sims)
        if xs and ys:
            plt.scatter(xs, ys, s=8, alpha=0.25, label="pairs")

    plt.xlabel("Timestep gap (Δ)")
    plt.ylabel("Similarity")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot_out)
    print(f"[ok] Wrote plot: {args.plot_out}")
    if args.csv_out:
        print(f"[ok] Wrote stats CSV: {args.csv_out}")


if __name__ == "__main__":
    main()