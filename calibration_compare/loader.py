"""Load per-model calibration CSVs produced by `python -m calibration.cli`."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ModelOutputs:
    label: str
    model_id: str
    output_dir: Path
    global_metrics: pd.DataFrame
    coverage_curves: pd.DataFrame
    paper_correlation: pd.DataFrame
    binary_mask: pd.DataFrame
    run_metadata: dict

    @property
    def csv_dir(self) -> Path:
        return self.output_dir / "csv"


def load_model(output_dir: Path, label: str | None = None) -> ModelOutputs:
    csv_dir = output_dir / "csv"
    if not csv_dir.exists():
        raise FileNotFoundError(f"missing csv/ subdir under {output_dir}")

    required = {
        "global_metrics": csv_dir / "global_metrics.csv",
        "coverage_curves": csv_dir / "coverage_curves.csv",
        "paper_correlation": csv_dir / "paper_correlation.csv",
        "binary_mask": csv_dir / "binary_mask_analysis.csv",
        "run_metadata": csv_dir / "run_metadata.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing files for {output_dir}: {missing}")

    with open(required["run_metadata"]) as f:
        meta = json.load(f)
    model_id = str(meta.get("model_id", output_dir.name))
    return ModelOutputs(
        label=label or model_id,
        model_id=model_id,
        output_dir=output_dir,
        global_metrics=pd.read_csv(required["global_metrics"]),
        coverage_curves=pd.read_csv(required["coverage_curves"]),
        paper_correlation=pd.read_csv(required["paper_correlation"]),
        binary_mask=pd.read_csv(required["binary_mask"]),
        run_metadata=meta,
    )


def parse_model_arg(spec: str) -> tuple[Path, str | None]:
    """Accept 'path' or 'label:path' (label may not contain ':' or use 'label=path')."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return Path(path), label
    if ":" in spec and not spec.startswith("/") and Path(spec.split(":", 1)[1]).exists():
        label, path = spec.split(":", 1)
        return Path(path), label
    return Path(spec), None


def load_pool(model: ModelOutputs) -> pd.DataFrame | None:
    """Load a model's AUROC pixel pool (parquet preferred, gzipped CSV fallback)."""
    parquet = model.csv_dir / "auroc_pool.parquet"
    csvgz = model.csv_dir / "auroc_pool.csv.gz"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csvgz.exists():
        return pd.read_csv(csvgz)
    return None


def assert_marker_overlap(models: list[ModelOutputs]) -> list[str]:
    """Return the intersection of per-marker group_values across all models; warn on diffs."""
    marker_sets = []
    for m in models:
        pm = m.global_metrics[m.global_metrics["group_type"] == "per_marker"]
        marker_sets.append(set(pm["group_value"].astype(str).tolist()))
    intersection = set.intersection(*marker_sets) if marker_sets else set()
    union = set.union(*marker_sets) if marker_sets else set()
    diff = union - intersection
    if diff:
        for m, s in zip(models, marker_sets):
            only = sorted(s - intersection)
            if only:
                print(f"[compare] WARN: markers only in '{m.label}': {only}")
    return sorted(intersection)
