"""npz loading + metadata parsing utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator


def list_npz_files(model_dir: str | os.PathLike) -> list[str]:
    p = Path(model_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"model dir not found: {p}")
    files = sorted(str(f) for f in p.glob("recn-*.npz"))
    if not files:
        raise FileNotFoundError(f"no recn-*.npz files under {p}")
    return files


def parse_metadata(meta_array) -> dict:
    """Decode the scalar object-dtype `metadata` field of an npz."""
    raw = meta_array.item() if hasattr(meta_array, "item") else meta_array
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"unexpected metadata type: {type(raw)}")


def chunked(seq: list, n_chunks: int) -> Iterator[list]:
    """Split `seq` into `n_chunks` near-equal contiguous slices."""
    if n_chunks <= 0:
        n_chunks = 1
    step = (len(seq) + n_chunks - 1) // n_chunks
    for i in range(0, len(seq), step):
        yield seq[i : i + step]
