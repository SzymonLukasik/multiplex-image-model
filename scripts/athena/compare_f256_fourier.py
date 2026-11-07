#!/usr/bin/env python
"""LOO table for equiv-v3-f256-fourier-zs-hn next to V3-256 (szary) and V3-512 (Athena),
plus the pre-registered decision at e79.

    python compare_f256_fourier.py [--v3-256-full <path to V3-256 e79 *_full.csv from szary>]

Per-image Pearson = mean of the per (image, marker) Pearson over the 149 hn images x
40 markers in *_full.csv; MSE = mean of the per-row MSE (same definition as the
V3-256 numbers quoted from szary and as V3-512 here: e79 .6311 / .00586).
The paired per-image z needs the V3-256 per-image file, which lives on szary.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

BASE = "/net/tscratch/people/plgslukasik/immu-vis/multiplex-image-model"
EPOCHS = [9, 19, 29, 49, 79, 89, 99, 109, 119, 129, 139, 149, 159, 169, 179, 189, 199]
V3_256 = {9: (.5546, .00848), 29: (.6059, .00746), 49: (.6128, .00693), 79: (.6242, .00648)}   # szary, quoted
THRESH = 0.004

ap = argparse.ArgumentParser()
ap.add_argument("--v3-256-full", default=None)
args = ap.parse_args()


def load(tag, e):
    g = glob.glob(f"{BASE}/validation_results_zs_hn_{tag}_e{e}_r0.0/*_full.csv")
    return pd.read_csv(g[0]) if g else None


rows = []
for e in EPOCHS:
    f, v5 = load("f256_fourier", e), load("equiv", e)
    if f is None:
        continue
    assert len(f) == 5960, (e, len(f))
    r = {"epoch": e, "fourier_r": f.pearson.mean(), "fourier_mse": f.mse.mean()}
    if e in V3_256:
        r["v3_256_r"], r["v3_256_mse"] = V3_256[e]
        r["d_r_vs_256"] = r["fourier_r"] - r["v3_256_r"]
        r["mse_rel_vs_256"] = r["fourier_mse"] / r["v3_256_mse"] - 1
    if v5 is not None:
        r["v3_512_r"], r["v3_512_mse"] = v5.pearson.mean(), v5.mse.mean()
    rows.append(r)

if not rows:
    print("no Fourier LOO results yet")
    raise SystemExit
t = pd.DataFrame(rows).set_index("epoch")
pd.set_option("display.width", 160)
print(t.round(4).to_string())

if 79 in t.index:
    d = t.loc[79, "d_r_vs_256"]
    verdict = "GAIN" if d >= THRESH else ("LOSS" if d <= -THRESH else "NULL")
    print(f"\nPRE-REGISTERED PRIMARY (e79 per-image LOO Pearson, Fourier - V3-256): "
          f"{d:+.4f} -> {verdict}  (threshold +/-{THRESH})")
    print(f"LOO MSE change vs V3-256 at e79: {100 * t.loc[79, 'mse_rel_vs_256']:+.1f}% "
          f"(downstream on szary if GAIN or MSE improves by >= 3%)")
    if args.v3_256_full:
        a = load("f256_fourier", 79)
        b = pd.read_csv(args.v3_256_full)
        key = ["image_path", "marker"]
        a["img"] = a.image_path.map(os.path.basename); b["img"] = b.image_path.map(os.path.basename)
        pa = a.groupby("img").pearson.mean(); pb = b.groupby("img").pearson.mean()
        dd = (pa - pb).dropna()
        z = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
        print(f"paired per-image: n={len(dd)}, mean diff {dd.mean():+.4f}, z = {z:.2f}")
