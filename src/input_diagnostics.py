"""
input_diagnostics.py -- direction check for the four rule inputs (Section 7.3).

For each unit-level rate (v, u, p, e) and for the per-turn share of unchecked
writing requests, reports the Spearman correlation with the continuous weak
label y, a 95% student-cluster bootstrap CI for it, and the AUC for the
top-quartile label. No parameter is fitted, so there is no fold to leak from.
The result is merged into the statistics file as "E6_inputs".

    python src/input_diagnostics.py --units results/units.pkl \
                                    --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def cluster_boot_rho(x, y, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        out.append(spearmanr(x[take], y[take])[0])
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()
    U = pickle.load(open(a.units, "rb"))
    y = np.array([u["y"] for u in U])
    yb = (y >= np.quantile(y, 0.75)).astype(int)
    g = np.array([u["group_key"] for u in U])
    cols = {k: np.array([u[k] for u in U]) for k in ("v", "u", "p", "e")}
    cols["verbatim_all_turns"] = np.array([u["num"][:, 1].mean() for u in U])
    res = {}
    for k, x in cols.items():
        rho, p = spearmanr(x, y)
        res[k] = dict(mean=float(x.mean()), spearman_y=float(rho), spearman_p=float(p),
                      ci95=cluster_boot_rho(x, y, g), auc=float(roc_auc_score(yb, x)))
    st = json.load(open(a.stats))
    st["E6_inputs"] = res
    json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
