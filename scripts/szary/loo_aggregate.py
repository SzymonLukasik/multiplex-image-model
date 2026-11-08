"""Per-marker aggregate of one LOO run (same definitions as aggregated_per_marker_f256_e*.csv) and a
one-line summary appended to logs/loo_f256_series.txt.   usage: python loo_aggregate.py <epoch> [tag]
(tag defaults to f256; the noact ablation uses f256_noact)"""
import sys, glob, numpy as np, pandas as pd
E = int(sys.argv[1]); T = sys.argv[2] if len(sys.argv) > 2 else "f256"
R = "/home/szlukasik/immu-vis/multiplex-image-model/mim-f256"
f = glob.glob(f"{R}/validation_results/loo_hn_{T}_e{E}/*_full.csv")[0]
d = pd.read_csv(f); g = d.groupby("marker")
s = g[["sum_pred", "sum_gt", "sum_pred2", "sum_gt2", "sum_predgt", "n_pix"]].sum(); n = s.n_pix
cov = s.sum_predgt / n - (s.sum_pred / n) * (s.sum_gt / n)
vp = s.sum_pred2 / n - (s.sum_pred / n) ** 2; vg = s.sum_gt2 / n - (s.sum_gt / n) ** 2
a = pd.DataFrame({"mse_mean": g.mse.mean(), "pearson_mean": g.pearson.mean(), "pooled_pearson": cov / np.sqrt(vp * vg),
                  "n_images": g.size()}).reset_index()
a.to_csv(f"{R}/validation_results/loo_reference_per_marker/aggregated_per_marker_{T}_e{E}.csv", index=False)
line = (f"e{E}: Pearson {a.pearson_mean.mean():.4f}  pooled {a.pooled_pearson.mean():.4f}  MSE {a.mse_mean.mean():.5f}  "
        f"median MSE {a.mse_mean.median():.5f}  corr(logsigma,logMSE) {np.corrcoef(d.logsigma, np.log(d.mse + 1e-8))[0, 1]:.3f}")
open(f"{R}/logs/loo_{T}_series.txt", "a").write(line + "\n"); print(line)
