"""
run_nested.py -- E11, input selection nested inside the training semester (PREREG_E11.md).

    python src/run_nested.py --units results/units.pkl \
        --stats results/paper_stats_fuzzy.json --out results/nested_stats.json
"""
import argparse, json, pickle
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import run_transfer as T
from run_followup import BROAD

POOLED = ["writing_request", "conceptual_questions", "provide_context"]
CONF = 97.5


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def cv_auc(X, yb, g, seed=42):
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, yb, g):
        if len(set(yb[tr])) < 2 or len(set(yb[te])) < 2:
            continue
        aucs.append(roc_auc_score(yb[te], lr().fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def direction(units, tr_sem, te_sem):
    sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
    tr, te = np.where(sem == tr_sem)[0], np.where(sem == te_sem)[0]
    y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
    thr = float(np.quantile(y[tr], 0.75))
    yb_tr, yb_te = (y[tr] >= thr).astype(int), (y[te] >= thr).astype(int)
    C8 = T.counts(units)
    full = cv_auc(C8[tr], yb_tr, g[tr])
    drops = {b: full - cv_auc(np.delete(C8[tr], k, 1), yb_tr, g[tr]) for k, b in enumerate(BROAD)}
    order = sorted(BROAD, key=lambda b: (-drops[b], BROAD.index(b)))
    sel = order[:3]
    X3 = np.log1p(T.counts(units, sel))
    cf = T.CFTSK().fit(X3[tr], yb_tr)
    s_cf = cf.score(X3[te])[0]
    s_lr8 = lr().fit(C8[tr], yb_tr).predict_proba(C8[te])[:, 1]
    gt = g[te]
    T.CONF = CONF
    out = {"train": tr_sem, "test": te_sem, "threshold_y": thr, "train_cv_auc_lr8": full,
           "drops": drops, "ranking": order, "selected": sel, "same_as_pooled": sorted(sel) == sorted(POOLED),
           "auc_cf_nested": float(T.auc(yb_te, s_cf)), "auc_lr8": float(T.auc(yb_te, s_lr8)),
           "H1_diff": [float(T.auc(yb_te, s_cf) - T.auc(yb_te, s_lr8)),
                       T.boot(yb_te, gt, lambda t: T.auc(yb_te, s_cf, t) - T.auc(yb_te, s_lr8, t))],
           "H2_ci": T.boot(yb_te, gt, lambda t: T.auc(yb_te, s_cf, t)),
           "consequents": [{"rule": " and ".join(f"{c} {term}" for c, term in zip(sel, terms)), "z": float(z)}
                           for terms, z in zip(T.RULES, cf.z)],
           "frozen": {"inputs_log1p_counts_of": sel, "breakpoint_low": cf.a.tolist(), "breakpoint_high": cf.b.tolist(),
                      "rules_terms_in_input_order": [list(r) for r in T.RULES], "consequents": cf.z.tolist(),
                      "background_weight": T.W0, "background_output": T.Z0, "label_threshold_y": thr,
                      "trained_on": tr_sem}}
    out["H1_noninferior"] = out["H1_diff"][1][0] > -0.05
    out["H2_transfer"] = out["H2_ci"][0] > 0.5
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/nested_stats.json")
    ap.add_argument("--frozen", default="results/frozen_cf_tsk_f24.json")
    ap.add_argument("--sems", default="f24,s25")
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb"))
    s1, s2 = a.sems.split(",")
    res = {"primary": direction(units, s1, s2), "reverse": direction(units, s2, s1), "ci_level": CONF}
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(res["primary"]["frozen"], open(a.frozen, "w"), indent=1)
    st = json.load(open(a.stats)); st["E11_nested"] = res; json.dump(st, open(a.stats, "w"), indent=1)
    for k in ("primary", "reverse"):
        o = res[k]
        print(k, o["selected"], "same as pooled:", o["same_as_pooled"], "| cf", round(o["auc_cf_nested"], 3),
              "lr8", round(o["auc_lr8"], 3), "| H1", o["H1_diff"], o["H1_noninferior"], "| H2", o["H2_ci"], o["H2_transfer"])
        print("   drops", {b: round(v, 3) for b, v in o["drops"].items()})


if __name__ == "__main__":
    main()
