"""Multi-model calibration comparison CLI.

Usage:
    python -m calibration_compare.cli \
        --model outputs/603 \
        --model "ID=outputs/603" \
        --model "Zero-shot=outputs/612" \
        --output-dir ./outputs/compare \
        [--reference 0] [--no-figures] [--restrict-dataset hn]

Each --model is either '<dir>' or '<label>=<dir>'. The --reference index
selects which model deltas are computed against (default: first).

If --restrict-dataset is set, every model's global / per-marker rows are
re-aggregated from per_patch_channel.csv filtered to that dataset_name
(and its pool rows where AUROC/AURC/F1 need them). This lets a multi-
dataset model (e.g. 622) be compared fairly against single-dataset runs
(e.g. 612, 603) on a shared slice such as `hn`.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from .loader import ModelOutputs, assert_marker_overlap, load_model, parse_model_arg
from .metrics import (
    build_ause_long,
    build_ause_summary,
    build_compare_per_marker,
    build_compare_summary,
    build_f1_vs_q_long,
    build_paper_correlation_long,
    build_reliability_long,
)
from .plotting import make_all_figures, set_palette_overrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare calibration metrics across N ImmuVis models.")
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="One per model: '<output_dir>' or '<label>=<output_dir>'. Repeat for each model.",
    )
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument(
        "--reference",
        type=int,
        default=0,
        help="Index of the reference model (deltas computed as other - reference). Default 0.",
    )
    p.add_argument("--no-figures", action="store_true")
    p.add_argument(
        "--no-pool",
        action="store_true",
        help="Don't load any auroc_pool.{parquet,csv.gz}. Pool-dependent plots "
             "(risk-coverage, AUROC/AURC per-marker) are skipped, and pool-derived "
             "metrics (per-marker AUROC/AURC under --restrict-dataset, F1 with HN-global "
             "thresholds) fall back to per-pc TP/FP sums or are dropped.",
    )
    p.add_argument(
        "--color",
        action="append",
        default=None,
        help="Per-model colour override: '<label>=<color>'. Repeat per model. "
             "<color> is anything matplotlib accepts (named, '#hex', or rgb tuple). "
             "Labels not assigned fall back to the default palette.",
    )
    p.add_argument(
        "--restrict-dataset",
        default=None,
        help="If set (e.g. 'hn'), re-aggregate every model's global/per-marker "
             "metrics from per_patch_channel.csv filtered to that dataset_name. "
             "Use this to fairly compare a multi-dataset model against single-"
             "dataset runs on a shared slice.",
    )
    return p.parse_args()


def _dedupe_labels(models: list[ModelOutputs]) -> list[ModelOutputs]:
    seen: dict[str, int] = {}
    out: list[ModelOutputs] = []
    for m in models:
        if m.label in seen:
            seen[m.label] += 1
            new_label = f"{m.label}#{seen[m.label]}"
            print(f"[compare] WARN: duplicate label '{m.label}' -> '{new_label}'")
            out.append(
                ModelOutputs(
                    label=new_label,
                    model_id=m.model_id,
                    output_dir=m.output_dir,
                    global_metrics=m.global_metrics,
                    coverage_curves=m.coverage_curves,
                    paper_correlation=m.paper_correlation,
                    binary_mask=m.binary_mask,
                    run_metadata=m.run_metadata,
                )
            )
        else:
            seen[m.label] = 1
            out.append(m)
    return out


def run(args: argparse.Namespace) -> None:
    if len(args.model) < 2:
        raise ValueError("need at least two --model entries to compare")

    models: list[ModelOutputs] = []
    for spec in args.model:
        path, label = parse_model_arg(spec)
        models.append(
            load_model(
                path,
                label=label,
                restrict_dataset=args.restrict_dataset,
                skip_pool=args.no_pool,
            )
        )
    models = _dedupe_labels(models)
    if args.restrict_dataset:
        print(f"[compare] restrict_dataset='{args.restrict_dataset}' — per-marker / "
              "global rows re-aggregated from per_patch_channel.csv on that slice.")

    if not (0 <= args.reference < len(models)):
        raise ValueError(f"--reference {args.reference} out of range [0, {len(models)})")

    print(f"[compare] {len(models)} models:")
    for i, m in enumerate(models):
        marker = " (reference)" if i == args.reference else ""
        print(f"    [{i}] {m.label}{marker}  id={m.model_id}  dir={m.output_dir}")

    markers = assert_marker_overlap(models)
    print(f"[compare] {len(markers)} markers in intersection")

    out_dir = Path(args.output_dir)
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = build_compare_summary(models, reference_idx=args.reference)
    summary.to_csv(csv_dir / "compare_summary.csv", index=False)
    print(f"[compare] wrote {csv_dir / 'compare_summary.csv'}")

    per_marker = build_compare_per_marker(models, markers, reference_idx=args.reference)
    per_marker_alpha = per_marker.sort_values("marker")
    per_marker_alpha.to_csv(csv_dir / "compare_per_marker.csv", index=False)

    # Sorted-by-delta-ECE variant if there's at least one non-reference model.
    other_idx = next((i for i in range(len(models)) if i != args.reference), None)
    if other_idx is not None:
        other = models[other_idx]
        ref = models[args.reference]
        delta_col = f"delta_ece_reg__{other.label}__minus__{ref.label}"
        if delta_col in per_marker.columns:
            per_marker_sorted = per_marker.sort_values(delta_col, ascending=False)
            per_marker_sorted.to_csv(
                csv_dir / "compare_per_marker_sorted_by_delta_ece.csv", index=False
            )

    rel_global_long = build_reliability_long(models, "global")
    rel_global_long.to_csv(csv_dir / "reliability_global_compare.csv", index=False)

    rel_per_marker_long = build_reliability_long(models, "per_marker")
    rel_per_marker_long.to_csv(csv_dir / "reliability_per_marker_compare.csv", index=False)

    f1_long = build_f1_vs_q_long(models)
    f1_long.to_csv(csv_dir / "f1_global_vs_q.csv", index=False)

    paper_long = build_paper_correlation_long(models)
    paper_long.to_csv(csv_dir / "loo_scatter_compare.csv", index=False)

    ause_long = build_ause_long(models)
    if not ause_long.empty:
        ause_long.to_csv(csv_dir / "ause_curves.csv", index=False)
    ause_summary_long = build_ause_summary(models)
    if not ause_summary_long.empty:
        ause_summary_long.to_csv(csv_dir / "ause_summary.csv", index=False)

    run_meta = {
        "models": [
            {
                "label": m.label,
                "model_id": m.model_id,
                "output_dir": str(m.output_dir),
                "n_npz_files_processed": m.run_metadata.get("n_npz_files_processed"),
                "n_patch_channel_rows": m.run_metadata.get("n_patch_channel_rows"),
                "auroc_pool_size": m.run_metadata.get("auroc_pool_size"),
                "seed": m.run_metadata.get("seed"),
            }
            for m in models
        ],
        "reference_index": args.reference,
        "marker_intersection_size": len(markers),
        "marker_intersection": markers,
        "restrict_dataset": args.restrict_dataset,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(csv_dir / "run_metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print("[compare] CSV summary (global):")
    for _, row in summary.iterrows():
        cols = [f"{m.label}={row[m.label]:.4f}" for m in models if m.label in row and row[m.label] == row[m.label]]
        print(f"    {row['metric']:30s}  " + "  ".join(cols))

    palette_overrides: dict[str, str] = {}
    for spec in (args.color or []):
        if "=" not in spec:
            raise ValueError(
                f"--color must be '<label>=<color>', got {spec!r}"
            )
        label, color = spec.split("=", 1)
        palette_overrides[label] = color
    known_labels = {m.label for m in models}
    unmatched = [lab for lab in palette_overrides if lab not in known_labels]
    if unmatched:
        print(
            f"[compare] WARN: --color label(s) not in --model set: {unmatched}\n"
            f"          known labels: {sorted(known_labels)}"
        )
    set_palette_overrides(palette_overrides)
    if palette_overrides:
        print("[compare] palette overrides:")
        for k, v in palette_overrides.items():
            print(f"    {k!r:40s} -> {v}")

    if not args.no_figures:
        make_all_figures(
            models=models,
            markers=markers,
            compare_per_marker=per_marker,
            f1_long=f1_long,
            paper_long=paper_long,
            out_dir=fig_dir,
            reference_idx=args.reference,
            ause_long=ause_long,
            ause_summary_long=ause_summary_long,
            skip_pool=args.no_pool,
        )
        print(f"[compare] figures in {fig_dir}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
