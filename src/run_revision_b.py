"""
run_revision_b.py -- addendum R1b (PREREG_R1b.md): within-group label checks.

    python src/run_revision_b.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle, warnings
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import run_transfer as T
from run_revision import Split, auc_ci, WCP
from reproduce_benchmark import folds
from fuzzy_rules import FuzzyRuleBase
from oc_tsk import OCTSK, INPUTS as OC_INPUTS

warnings.filterwarnings("ignore")


def group_only_pooled(units):
    y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
    yb = (y >= np.quantile(y, 0.75)).astype(int)
    G = OneHotEncoder(sparse_output=False).fit_transform(np.array([u["group"] for u in units])[:, None])
    a = [T.auc(yb[te], LogisticRegression(max_iter=1000).fit(G[tr], yb[tr]).predict_proba(G[te])[:, 1])
         for tr, te in folds(len(yb), yb, g, 42)]
    return float(np.mean(a))


def within_ranks(x, cell):
    r = np.zeros(len(x))
    for c in np.unique(cell):
        m = cell == c; r[m] = (rankdata(x[m]) - 1) / max(m.sum() - 1, 1)
    return r


def direction_check(units):
    y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
    cell = np.array([u["group_key"].rsplit("_", 1)[1] + u["group"] for u in units])
    ry = within_ranks(y, cell)
    C = np.log1p(T.counts(units, WCP))
    X = {"v": np.array([u["v"] for u in units]), "u": np.array([u["u"] for u in units]),
         "p": np.array([u["p"] for u in units]), "e": np.array([u["e"] for u in units]),
         "log_W": C[:, 0], "log_C": C[:, 1], "log_P": C[:, 2]}
    rng = np.random.default_rng(0); ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    boots = [np.concatenate([idx[k] for k in rng.choice(ug, len(ug))]) for _ in range(2000)]
    out = {}
    for k, x in X.items():
        rx = within_ranks(x, cell)
        bs = [spearmanr(rx[t], ry[t])[0] for t in boots]
        out[k] = {"rho": float(spearmanr(rx, ry)[0]), "ci": np.nanpercentile(bs, [2.5, 97.5]).tolist()}
    return out


def two_by_two(units, d):
    sp = Split(units, *d, label="group")
    rates = lambda idx: {k: np.array([units[i][k] for i in idx]) for k in "vupe"}
    s_fr = FuzzyRuleBase().fit(rates(sp.tr)).score(rates(sp.te))[0]
    oc_in = lambda idx: {k: np.array([units[i][k] for i in idx]) for k in OC_INPUTS}
    s_oc = OCTSK().fit(oc_in(sp.tr), sp.yb_tr).score(oc_in(sp.te))[0]
    X = np.log1p(sp.counts(WCP))
    ec = T.CFTSK(); ec.a = np.quantile(X[sp.tr], 0.25, 0); ec.b = np.quantile(X[sp.tr], 0.75, 0)
    ec.b = np.where(ec.b - ec.a < 1e-9, ec.a + 1e-3, ec.b)
    ec.z = np.array([((t[0] == "HIGH") + (t[1] == "LOW") + (t[2] == "HIGH")) / 3 for t in T.RULES])
    cf = T.CFTSK().fit(X[sp.tr], sp.yb_tr)
    out = {"author_rules_shares": auc_ci(sp, s_fr), "learned_shares_octsk": auc_ci(sp, s_oc),
           "author_dirs_volumes_ectsk": auc_ci(sp, ec.score(X[sp.te])[0]),
           "learned_volumes_cftsk": auc_ci(sp, cf.score(X[sp.te])[0]),
           "cf_consequents": {" ".join(f"{s}={t[0]}" for s, t in zip("WCP", terms)): float(z) for terms, z in zip(T.RULES, cf.z)},
           "train_base_rate": float(sp.yb_tr.mean()), "test_prevalence": float(sp.yb_te.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/revision_b_stats.json")
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb"))
    res = {"group_only_pooled_cv": group_only_pooled(units), "direction_within_group": direction_check(units),
           "two_by_two": {k: two_by_two(units, d) for k, d in (("primary", ("f24", "s25")), ("reverse", ("s25", "f24")))}}
    json.dump(res, open(a.out, "w"), indent=1)
    st = json.load(open(a.stats)); st["R1b"] = res; json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
