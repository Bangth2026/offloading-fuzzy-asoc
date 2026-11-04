"""
run_transfer.py -- E10 cross-semester confirmation, as specified in PREREG_E10.md.

    python src/run_transfer.py --units results/units.pkl \
        --stats results/paper_stats_fuzzy.json --out results/transfer_stats.json
"""
import argparse, json, pickle, itertools
import numpy as np, torch, torch.nn as nn
from scipy.optimize import lsq_linear
from scipy.stats import wilcoxon, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import run_fuzzy_experiments as R
from run_followup import BROAD
from oc_tsk import interval_implication_loss
from fuzzy_rules import FuzzyRuleBase

CF_INPUTS = ("writing_request", "conceptual_questions", "provide_context")
SHORT = {"writing_request": "W", "conceptual_questions": "C", "provide_context": "P"}
RULES = list(itertools.product(("LOW", "HIGH"), repeat=3))   # (W, C, P) terms
W0, Z0, ALPHA, MARGIN, CONF = 0.05, 0.5, 1.0, 0.05, 98.75


def counts(units, cats=BROAD):
    return np.array([[u["broad"].count(b) for b in cats] for u in units], float)


class CFTSK:
    def fit(self, X, ybin):
        self.a = np.quantile(X, 0.25, 0); self.b = np.quantile(X, 0.75, 0)
        self.b = np.where(self.b - self.a < 1e-9, self.a + 1e-3, self.b)
        W = self.firing(X); den = W.sum(1) + W0
        A = W / den[:, None]; bv = W0 * Z0 / den
        prior = np.full(len(RULES), ybin.mean())
        M = np.vstack([A, np.sqrt(ALPHA) * np.eye(len(RULES))])
        t = np.concatenate([ybin - bv, np.sqrt(ALPHA) * prior])
        self.z = lsq_linear(M, t, bounds=(0, 1)).x
        return self

    def firing(self, X):
        hi = np.clip((X - self.a) / (self.b - self.a), 0, 1)
        W = np.ones((len(X), len(RULES)))
        for j, terms in enumerate(RULES):
            for i, term in enumerate(terms):
                W[:, j] *= hi[:, i] if term == "HIGH" else 1 - hi[:, i]
        return W

    def score(self, X):
        W = self.firing(X)
        return (W @ self.z + W0 * Z0) / (W.sum(1) + W0), W


def boot(yb, groups, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}; out = []
    for _ in range(B):
        t = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(set(yb[t])) < 2:
            continue
        out.append(fn(t))
    lo, hi = np.percentile(out, [(100 - CONF) / 2, 100 - (100 - CONF) / 2])
    return [float(lo), float(hi)]


def auc(yb, s, t=None):
    t = np.arange(len(yb)) if t is None else t
    return roc_auc_score(yb[t], s[t])


def pct(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def train_tf(units, tr, ybin, seed, kind, W_tr=None, z=None, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    model = R.make_model(); opt = torch.optim.Adam(model.parameters(), 1e-3)
    yt = torch.tensor(ybin, dtype=torch.float)
    for _ in range(epochs):
        model.train()
        for b in np.array_split(np.random.permutation(len(tr)), max(1, len(tr) // 32)):
            r = model(*R.pad(units, tr[b]))
            loss = nn.functional.binary_cross_entropy(r, yt[b])
            if kind == "cf_imp":
                loss = loss + interval_implication_loss(r, torch.tensor(W_tr[b], dtype=torch.float), z)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def one_direction(units, tr_sem, te_sem, seeds, epochs, own_threshold=False, with_tf=True):
    sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
    tr, te = np.where(sem == tr_sem)[0], np.where(sem == te_sem)[0]
    y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
    thr = np.quantile(y[tr], 0.75)
    yb_tr = (y[tr] >= thr).astype(int)
    yb_te = (y[te] >= (np.quantile(y[te], 0.75) if own_threshold else thr)).astype(int)
    C8 = counts(units); X3 = np.log1p(counts(units, CF_INPUTS))
    cf = CFTSK().fit(X3[tr], yb_tr)
    s_cf = cf.score(X3[te])[0]
    lr8 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(C8[tr], yb_tr)
    s_lr8 = lr8.predict_proba(C8[te])[:, 1]
    lr3 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X3[tr], yb_tr)
    s_lr3 = lr3.predict_proba(X3[te])[:, 1]
    rin = lambda idx: {k: np.array([units[i][k] for i in idx]) for k in "vupe"}
    s_ex = FuzzyRuleBase().fit(rin(tr)).score(rin(te))[0]
    s_fus = (pct(s_lr8) + pct(s_cf)) / 2
    gt = g[te]
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "test_prevalence": float(yb_te.mean()),
           "auc": {k: float(auc(yb_te, s)) for k, s in
                   [("cf_tsk", s_cf), ("lr8", s_lr8), ("lr3", s_lr3), ("expert_rules", s_ex), ("fusion_lr8_cf", s_fus)]},
           "H1_cf_minus_lr8": [float(auc(yb_te, s_cf) - auc(yb_te, s_lr8)),
                               boot(yb_te, gt, lambda t: auc(yb_te, s_cf, t) - auc(yb_te, s_lr8, t))],
           "H2_cf_ci": boot(yb_te, gt, lambda t: auc(yb_te, s_cf, t)),
           "H4_fusion_minus_lr8": [float(auc(yb_te, s_fus) - auc(yb_te, s_lr8)),
                                   boot(yb_te, gt, lambda t: auc(yb_te, s_fus, t) - auc(yb_te, s_lr8, t))],
           "cf_minus_lr3": [float(auc(yb_te, s_cf) - auc(yb_te, s_lr3)),
                            boot(yb_te, gt, lambda t: auc(yb_te, s_cf, t) - auc(yb_te, s_lr3, t))],
           "consequents": {"".join(f"{SHORT[c]}={t[0]}" + (" " if i < 2 else "") for i, (c, t) in
                                   enumerate(zip(CF_INPUTS, terms))): float(z)
                           for terms, z in zip(RULES, cf.z)},
           "train_base_rate": float(yb_tr.mean()),
           "breakpoints_log1p": {"low": cf.a.tolist(), "high": cf.b.tolist()}}
    if with_tf:
        _, W_tr = cf.score(X3[tr]); tf0, tf1 = [], []
        for sd in range(seeds):
            for kind, store in (("none", tf0), ("cf_imp", tf1)):
                m = train_tf(units, tr, yb_tr, sd, kind, W_tr, cf.z, epochs)
                with torch.no_grad():
                    p = m(*R.pad(units, te)).numpy()
                store.append(float(auc(yb_te, p)))
            print(tr_sem, "->", te_sem, "seed", sd, round(tf0[-1], 3), round(tf1[-1], 3), flush=True)
        d = np.array(tf1) - np.array(tf0)
        out["H3"] = {"none": tf0, "cf_imp": tf1, "mean_none": float(np.mean(tf0)), "sd_none": float(np.std(tf0, ddof=1)),
                     "mean_cf_imp": float(np.mean(tf1)), "sd_cf_imp": float(np.std(tf1, ddof=1)),
                     "mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()),
                     "wilcoxon_p": float(wilcoxon(tf1, tf0).pvalue) if np.any(d != 0) else 1.0}
    return out


def decide(o):
    return {"H1_noninferior": o["H1_cf_minus_lr8"][1][0] > -MARGIN,
            "H2_transfer": o["H2_cf_ci"][0] > 0.5,
            "H3_constraint_helps": ("H3" in o and o["H3"]["wilcoxon_p"] < 0.0125 and o["H3"]["mean_diff"] > 0),
            "H4_complementary": o["H4_fusion_minus_lr8"][1][0] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/transfer_stats.json")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--sems", default="f24,s25")
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); R.set_n_da(units)
    s1, s2 = a.sems.split(",")
    res = {"primary": one_direction(units, s1, s2, a.seeds, a.epochs)}
    res["decisions"] = decide(res["primary"])
    res["reverse"] = one_direction(units, s2, s1, a.seeds, a.epochs)
    res["reverse_decisions"] = decide(res["reverse"])
    res["own_threshold"] = one_direction(units, s1, s2, a.seeds, a.epochs, own_threshold=True, with_tf=False)
    res["settings"] = {"margin": MARGIN, "ci_level": CONF, "alpha_bonferroni": 0.0125, "seeds": a.seeds}
    json.dump(res, open(a.out, "w"), indent=1)
    st = json.load(open(a.stats)); st["E10_transfer"] = res; json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("decisions", "reverse_decisions")}, indent=1))
    print(json.dumps({k: v for k, v in res["primary"].items() if k != "H3"}, indent=1))


if __name__ == "__main__":
    main()
