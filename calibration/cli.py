"""Single-model calibration analysis CLI.

Usage:
    python -m calibration.cli \
        --model-dir /raid_encrypted/.../immuvis_609_loo \
        --model-id 609 \
        --output-dir ./outputs/609 \
        [--num-workers 8] [--auroc-budget 2000000] [--seed 42] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config as cfg
from .auroc_subsample import cap_pool, concat_pools
from .io_utils import chunked, list_npz_files, parse_metadata
from .metrics_aggregate import (
    build_binary_mask_analysis,
    build_coverage_curves,
    build_global_metrics,
    build_paper_correlation,
)
from .metrics_pixel import process_npz_chunk
from .plotting import make_all_figures


def _worker(args):
    paths, seed = args
    return process_npz_chunk(paths, seed=seed, per_pc_cap=cfg.DEFAULT_PER_PC_CAP)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ImmuVis variance-head calibration analysis (single model).")
    p.add_argument("--model-dir", required=True, type=str, help="Directory of recn-*.npz files for one model.")
    p.add_argument("--model-id", required=True, type=str, help="Short identifier for the model (e.g. '609').")
    p.add_argument("--output-dir", required=True, type=str, help="Where to write csv/ and figures/.")
    p.add_argument("--num-workers", type=int, default=cfg.DEFAULT_NUM_WORKERS)
    p.add_argument("--auroc-budget", type=int, default=cfg.DEFAULT_AUROC_BUDGET)
    p.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N npz files (smoke test).")
    p.add_argument("--no-figures", action="store_true", help="Skip plotting.")
    p.add_argument(
        "--restrict-dataset",
        default=None,
        help="If set (e.g. 'hn'), only process npz files whose metadata.dataset_name "
             "matches. Lets you produce a clean per-dataset output dir from a "
             "multi-dataset LOO dump without recomputing the whole thing.",
    )
    p.add_argument(
        "--filter-workers",
        type=int,
        default=8,
        help="Threads to peek metadata when --restrict-dataset is set. "
             "I/O-bound; 8 is plenty.",
    )
    return p.parse_args()


def _filter_by_dataset(
    npz_files: list[str],
    dataset_name: str,
    n_workers: int = 8,
) -> list[str]:
    """Keep only the files whose metadata.dataset_name matches `dataset_name`.

    Each npz is opened only to read its scalar `metadata` field — fast.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _peek(path: str) -> tuple[str, str]:
        with np.load(path, allow_pickle=True) as d:
            try:
                ds = str(parse_metadata(d["metadata"]).get("dataset_name", ""))
            except Exception:
                ds = ""
        return path, ds

    keep: list[str] = []
    seen_other: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        for path, ds in tqdm(
            ex.map(_peek, npz_files),
            total=len(npz_files),
            desc=f"filtering by dataset='{dataset_name}'",
        ):
            if ds == dataset_name:
                keep.append(path)
            else:
                seen_other[ds] = seen_other.get(ds, 0) + 1

    if seen_other:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(seen_other.items()))
        print(f"[calibration] skipped: {summary}")
    return sorted(keep)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list_npz_files(args.model_dir)
    n_total_files = len(npz_files)
    if args.restrict_dataset:
        npz_files = _filter_by_dataset(
            npz_files, args.restrict_dataset, n_workers=args.filter_workers
        )
        if not npz_files:
            raise RuntimeError(
                f"--restrict-dataset {args.restrict_dataset!r} matched 0 files in "
                f"{args.model_dir} (out of {n_total_files})"
            )
        print(
            f"[calibration] restrict_dataset={args.restrict_dataset!r}: "
            f"kept {len(npz_files)}/{n_total_files} files"
        )
    if args.limit is not None:
        npz_files = npz_files[: args.limit]
    print(f"[calibration] model_id={args.model_id}  files={len(npz_files)}  workers={args.num_workers}")

    t0 = time.time()
    all_rows: list[dict] = []
    pool_chunks: list[pd.DataFrame] = []

    chunks = list(chunked(npz_files, n_chunks=max(1, args.num_workers * 4)))
    seeds = (np.random.SeedSequence(args.seed).spawn(len(chunks)))
    seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
    work = list(zip(chunks, seed_ints))

    if args.num_workers <= 1:
        iterator = (process_npz_chunk(c, seed=s, per_pc_cap=cfg.DEFAULT_PER_PC_CAP) for c, s in work)
        for rows, pool in tqdm(iterator, total=len(work), desc="processing"):
            all_rows.extend(rows)
            if len(pool):
                pool_chunks.append(pool)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.num_workers) as pool_mp:
            for rows, pool in tqdm(
                pool_mp.imap_unordered(_worker, work), total=len(work), desc="processing"
            ):
                all_rows.extend(rows)
                if len(pool):
                    pool_chunks.append(pool)

    print(f"[calibration] processed {len(all_rows)} (patch, channel) rows in {time.time() - t0:.1f}s")

    if not all_rows:
        raise RuntimeError("no rows produced — check LOO assertion and input files")

    per_pc = pd.DataFrame.from_records(all_rows)

    # Sanity checks (loud, fail fast).
    _sanity_checks(per_pc)

    # Build the AUROC subsample pool (cap globally to budget, uniform).
    rng = np.random.default_rng(args.seed)
    pool = concat_pools(pool_chunks)
    pool = cap_pool(pool, args.auroc_budget, rng)
    pool_path = csv_dir / "auroc_pool.parquet"
    try:
        pool.to_parquet(pool_path, index=False)
    except Exception as e:  # parquet engine missing
        print(f"[calibration] parquet write failed ({e}); falling back to CSV")
        pool_path = csv_dir / "auroc_pool.csv.gz"
        pool.to_csv(pool_path, index=False, compression="gzip")
    print(f"[calibration] AUROC pool: {len(pool)} pixels -> {pool_path}")

    # Write CSVs.
    per_pc_path = csv_dir / "per_patch_channel.csv"
    per_pc.to_csv(per_pc_path, index=False)
    print(f"[calibration] wrote {per_pc_path}  ({len(per_pc)} rows, {per_pc.shape[1]} cols)")

    coverage_curves = build_coverage_curves(per_pc, args.model_id)
    coverage_curves.to_csv(csv_dir / "coverage_curves.csv", index=False)

    global_metrics = build_global_metrics(per_pc, pool, args.model_id)
    global_metrics.to_csv(csv_dir / "global_metrics.csv", index=False)

    paper_corr = build_paper_correlation(per_pc, args.model_id)
    paper_corr.to_csv(csv_dir / "paper_correlation.csv", index=False)

    binary = build_binary_mask_analysis(per_pc, pool, args.model_id)
    binary.to_csv(csv_dir / "binary_mask_analysis.csv", index=False)

    run_meta = {
        "model_id": args.model_id,
        "model_dir": str(args.model_dir),
        "restrict_dataset": args.restrict_dataset,
        "n_npz_files_processed": len(npz_files),
        "n_patch_channel_rows": int(len(per_pc)),
        "auroc_budget": int(args.auroc_budget),
        "auroc_pool_size": int(len(pool)),
        "alpha_grid": cfg.ALPHA_GRID.tolist(),
        "binary_mask_quantiles": list(cfg.BINARY_MASK_QUANTILES),
        "var_clip": list(cfg.VAR_CLIP),
        "logvar_clip": list(cfg.LOGVAR_CLIP),
        "per_pc_cap": cfg.DEFAULT_PER_PC_CAP,
        "min_pixels_per_marker_auroc": cfg.MIN_PIXELS_PER_MARKER_AUROC,
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "limit": args.limit,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(csv_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print("[calibration] CSV summary:")
    g = global_metrics[global_metrics["group_type"] == "global"].iloc[0]
    print(
        f"    global ECE_reg={g['ece_reg']:.4f}  Pearson r(log_var, log_MAE)={g['pearson_logvar_logmae']:.4f}  "
        f"AUROC@p90={g['auroc_pixel_top10pct']}  pool_n={g['pool_n_pixels']}"
    )

    if not args.no_figures:
        make_all_figures(out_dir, args.model_id)
        print(f"[calibration] figures in {fig_dir}")


def _sanity_checks(per_pc: pd.DataFrame) -> None:
    # Coverage monotonicity (counts must be non-decreasing in α).
    sample = per_pc.sample(min(200, len(per_pc)), random_state=0)
    cov_cols = [cfg.alpha_col(a) for a in cfg.ALPHA_GRID]
    arr = sample[cov_cols].to_numpy()
    diffs = np.diff(arr, axis=1)
    if (diffs < 0).any():
        bad = int((diffs < 0).any(axis=1).sum())
        raise AssertionError(
            f"coverage counts not monotonic in α for {bad}/{len(sample)} sampled rows"
        )

    # No NaN/Inf in headline aggregate-input columns.
    for col in ("mse", "mae", "mean_var", "mean_sigma", "mean_logvar", "mean_nll"):
        v = per_pc[col].to_numpy()
        if not np.isfinite(v).all():
            n_bad = int((~np.isfinite(v)).sum())
            raise AssertionError(f"{n_bad} non-finite values in column {col}")

    n_pix = int(per_pc["n_pixels"].sum())
    print(f"[calibration] total masked pixels: {n_pix:,}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
