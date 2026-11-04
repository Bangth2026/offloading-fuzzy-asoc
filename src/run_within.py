"""
run_within.py -- addendum R1c (PREREG_R1c.md): within-group signal of held-out predictions.

    python src/run_within.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle, warnings
import numpy as np, torch
from scipy.stats import spearmanr
import run_fuzzy_experiments as R
import run_transfer as T
from run_revision import Split, lr, WCP
from run_followup import BROAD
from run_revision_b import within_ranks
from run_constraint_ci import lstm_pred
from fuzzy_rules import FuzzyRuleBase

warnings.filterwarnings("ignore")


def within_rho(sp, units, s):
    y = np.array([u["y"] for u in units])[sp.te]; cell = sp.grp[sp.te]
    rs, ry = within_ranks(s, cell), within_ranks(y, cell)
    g = sp.gt; ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        t = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        bs.append(spearmanr(rs[t], ry[t])[0])
    return {"rho": float(spearmanr(rs, ry)[0]), "ci": np.nanpercentile(bs, [2.5, 97.5]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/within_stats.json"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); n_da = R.set_n_da(units); st = json.load(open(a.stats))
    out = {}
    for key, d in (("primary", ("f24", "s25")), ("reverse", ("s25", "f24"))):
        sp = Split(units, *d)
        X = np.log1p(sp.counts(WCP)); cf10 = T.CFTSK().fit(X[sp.tr], sp.yb_tr)
        sel = st["E11_nested"][key]["selected"]; Xn = np.log1p(sp.counts(sel)); cf11 = T.CFTSK().fit(Xn[sp.tr], sp.yb_tr)
        C8 = sp.counts(BROAD)
        rates = lambda idx: {k: np.array([units[i][k] for i in idx]) for k in "vupe"}
        S = {"cf_tsk_e10": cf10.score(X[sp.te])[0], "cf_tsk_e11": cf11.score(Xn[sp.te])[0],
             "lr8": lr().fit(C8[sp.tr], sp.yb_tr).predict_proba(C8[sp.te])[:, 1],
             "lr3_e11": lr().fit(Xn[sp.tr], sp.yb_tr).predict_proba(Xn[sp.te])[:, 1],
             "author_rules": FuzzyRuleBase().fit(rates(sp.tr)).score(rates(sp.te))[0]}
        W10 = cf10.score(X[sp.tr])[1]; tf = []
        for s in range(a.seeds):
            m = T.train_tf(units, sp.tr, sp.yb_tr, s, "none", W10, cf10.z, a.epochs)
            with torch.no_grad():
                tf.append(m(*R.pad(units, sp.te)).numpy())
        S["transformer_seedavg"] = np.mean(tf, 0)
        Wn = cf11.score(Xn[sp.tr])[1]
        S["lstm_seedavg"] = np.mean([lstm_pred(units, sp.tr, sp.te, sp.yb_tr, s, Wn, cf11.z, n_da, False, a.epochs)
                                     for s in range(a.seeds)], 0)
        out[key] = {k: within_rho(sp, units, v) for k, v in S.items()}
        print(key, {k: (round(v["rho"], 3), [round(x, 3) for x in v["ci"]]) for k, v in out[key].items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    st = json.load(open(a.stats)); st["R1c"] = out; json.dump(st, open(a.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
