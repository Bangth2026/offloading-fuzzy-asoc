"""
run_followup.py -- experiments E7-E9 as specified in PREREG_E7_E9.md.

    python src/run_followup.py --units results/units.pkl \
        --stats results/paper_stats_fuzzy.json --out results/followup_stats.json

Reads the E1 per-seed AUCs of the expert implication model from the statistics
file (comparison 3) and writes all follow-up statistics to a separate file,
which is also merged into the statistics file under "E7_E9_followup".
"""
import argparse, json, pickle
import numpy as np, torch, torch.nn as nn
from scipy.stats import wilcoxon, ttest_rel, rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import run_fuzzy_experiments as R
from oc_tsk import OCTSK, RULES, INPUTS, interval_implication_loss
from fuzzy_rules import FuzzyRuleBase

BROAD = ["conceptual_questions", "writing_request", "provide_context", "contextual_questions",
         "verification", "editing_request", "off_topic", "misc"]


def inputs(units, idx):
    return {k: np.array([units[i][k] for i in idx]) for k in INPUTS}


def counts(units):
    return np.array([[u["broad"].count(b) for b in BROAD] for u in units], float)


def pct_rank(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def cv_pass(units, ybin, groups, seed, kind, epochs=15, lam=1.0):
    """kind: 'none' or 'oc_imp'. Same seeding and split as cv_run in run_fuzzy_experiments."""
    torch.manual_seed(seed); np.random.seed(seed)
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    oof = np.full(len(units), np.nan); folds = []; zs = []
    s_oc = np.full(len(units), np.nan); s_ex = np.full(len(units), np.nan)
    lr = np.full(len(units), np.nan); C = counts(units)
    for tr, te in skf.split(np.zeros(len(units)), ybin, groups):
        if len(set(ybin[tr])) < 2:
            continue
        oc = OCTSK().fit(inputs(units, tr), ybin[tr]); zs.append(oc.z.tolist())
        _, W_tr = oc.score(inputs(units, tr))
        model = R.make_model(); opt = torch.optim.Adam(model.parameters(), 1e-3)
        yt = torch.tensor(ybin[tr], dtype=torch.float)
        for _ in range(epochs):
            model.train()
            for b in np.array_split(np.random.permutation(len(tr)), max(1, len(tr) // 32)):
                r = model(*R.pad(units, tr[b]))
                loss = nn.functional.binary_cross_entropy(r, yt[b])
                if kind == "oc_imp":
                    loss = loss + lam * interval_implication_loss(
                        r, torch.tensor(W_tr[b], dtype=torch.float), oc.z)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            oof[te] = model(*R.pad(units, te)).numpy()
        s_oc[te] = oc.score(inputs(units, te))[0]
        frb = FuzzyRuleBase().fit({k: np.array([units[i][k] for i in tr]) for k in "vupe"})
        s_ex[te] = frb.score({k: np.array([units[i][k] for i in te]) for k in "vupe"})[0]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(C[tr], ybin[tr])
        lr[te] = m.predict_proba(C[te])[:, 1]
        folds.append(te)
    return dict(oof=oof, s_oc=s_oc, s_ex=s_ex, lr=lr, folds=folds, z=zs)


def fold_auc(ybin, score, folds):
    return float(np.mean([roc_auc_score(ybin[te], score[te]) for te in folds]))


def holm(ps):
    return R.holm(ps)


def paired(a, b):
    a, b = np.array(a), np.array(b); d = a - b
    w = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return {"mean_a": float(a.mean()), "sd_a": float(a.std(ddof=1)), "mean_b": float(b.mean()),
            "mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()), "wilcoxon_p": float(w),
            "ttest_p": float(ttest_rel(a, b).pvalue), "cohen_d": float(d.mean() / (d.std(ddof=1) + 1e-12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/followup_stats.json")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); R.set_n_da(units)
    st = json.load(open(a.stats))
    y = np.array([u["y"] for u in units]); groups = np.array([u["group_key"] for u in units])
    ybin = (y >= np.quantile(y, 0.75)).astype(int)
    per = {k: [] for k in ("none", "oc_imp", "fus_tf_oc", "lr", "fus_lr_oc", "oc_rule", "ex_rule")}
    keep = None
    for sd in range(a.seeds):
        n = cv_pass(units, ybin, groups, sd, "none", a.epochs)
        o = cv_pass(units, ybin, groups, sd, "oc_imp", a.epochs)
        f = n["folds"]
        per["none"].append(fold_auc(ybin, n["oof"], f))
        per["oc_imp"].append(fold_auc(ybin, o["oof"], f))
        per["fus_tf_oc"].append(fold_auc(ybin, (pct_rank(n["oof"]) + pct_rank(n["s_oc"])) / 2, f))
        per["lr"].append(fold_auc(ybin, n["lr"], f))
        per["fus_lr_oc"].append(fold_auc(ybin, (pct_rank(n["lr"]) + pct_rank(n["s_oc"])) / 2, f))
        per["oc_rule"].append(fold_auc(ybin, n["s_oc"], f))
        per["ex_rule"].append(fold_auc(ybin, n["s_ex"], f))
        if sd == 0:
            keep = n
        print(sd, {k: round(v[-1], 3) for k, v in per.items()}, flush=True)
    ex_imp = st["E1_accuracy"]["implication"]["per_seed"][:a.seeds]
    fam = [("oc_imp_vs_none", per["oc_imp"], per["none"]),
           ("fus_tf_oc_vs_none", per["fus_tf_oc"], per["none"]),
           ("oc_imp_vs_expert_imp", per["oc_imp"], ex_imp),
           ("fus_lr_oc_vs_lr", per["fus_lr_oc"], per["lr"])]
    out = {"seeds": a.seeds, "per_seed": per, "expert_imp_per_seed": ex_imp,
           "none_matches_E1": float(np.max(np.abs(np.array(per["none"]) -
                                                  np.array(st["E1_accuracy"]["none"]["per_seed"][:a.seeds]))))}
    res = {name: paired(x, z) for name, x, z in fam}
    for (name, _, _), ph in zip(fam, holm([res[n]["wilcoxon_p"] for n, _, _ in fam])):
        res[name]["wilcoxon_p_holm"] = float(ph)
    out["primary"] = res
    out["means"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in per.items()}
    ok = ~np.isnan(keep["s_oc"])
    out["oc_vs_expert_rule_auc_seed0"] = R.cluster_boot_auc_diff(ybin[ok], keep["s_oc"][ok], keep["s_ex"][ok], groups[ok])
    out["oc_rule_auc_seed0"] = float(roc_auc_score(ybin[ok], keep["s_oc"][ok]))
    out["ex_rule_auc_seed0"] = float(roc_auc_score(ybin[ok], keep["s_ex"][ok]))
    out["learned_z_seed0"] = {RULES[j][0]: [z[j] for z in keep["z"]] for j in range(len(RULES))}
    out["expert_z"] = {r[0]: r[2] for r in RULES}
    json.dump(out, open(a.out, "w"), indent=1)
    st["E7_E9_followup"] = out
    json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps(out["primary"], indent=1)); print(json.dumps(out["means"], indent=1))
    print(json.dumps(out["learned_z_seed0"], indent=1)); print("none matches E1 (max abs diff):", out["none_matches_E1"])


if __name__ == "__main__":
    main()
