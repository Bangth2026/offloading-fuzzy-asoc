"""
run_revision.py -- revision analyses R1-A, B, C, D, F, G, H (PREREG_R1.md).

    python src/run_revision.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle, warnings
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import run_transfer as T
from run_nested import cv_auc
from run_followup import BROAD

warnings.filterwarnings("ignore")
WCP = ["writing_request", "conceptual_questions", "provide_context"]


def lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def ci(yb, g, fn, conf=95.0):
    T.CONF = conf
    return T.boot(yb, g, fn)


class Split:
    """Training/held-out split with every threshold taken from the training semester."""
    def __init__(self, units, tr_sem, te_sem, label="pooled"):
        self.units = units
        sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
        self.tr, self.te = np.where(sem == tr_sem)[0], np.where(sem == te_sem)[0]
        y = np.array([u["y"] for u in units]); grp = np.array([u["group"] for u in units])
        self.g = np.array([u["group_key"] for u in units]); self.grp = grp
        if label == "pooled":
            thr = np.quantile(y[self.tr], 0.75); yb = (y >= thr).astype(int)
        else:   # within (semester, group): group thresholds from the training semester
            yb = np.zeros(len(y), int)
            for k in np.unique(grp):
                t = np.quantile(y[self.tr][grp[self.tr] == k], 0.75)
                yb[grp == k] = (y[grp == k] >= t).astype(int)
        self.yb_tr, self.yb_te, self.gt = yb[self.tr], yb[self.te], self.g[self.te]

    def counts(self, cats, per_assign=False):
        C = T.counts(self.units, cats)
        if per_assign:
            C = C / np.array([u["n_assign"] for u in self.units])[:, None]
        return C


def select(sp, per_assign=False):
    C8 = sp.counts(BROAD, per_assign)
    full = cv_auc(C8[sp.tr], sp.yb_tr, sp.g[sp.tr])
    drops = {b: full - cv_auc(np.delete(C8[sp.tr], k, 1), sp.yb_tr, sp.g[sp.tr]) for k, b in enumerate(BROAD)}
    return sorted(BROAD, key=lambda b: (-drops[b], BROAD.index(b)))[:3], drops


def cf_scores(sp, cats, per_assign=False, alpha=None):
    X = np.log1p(sp.counts(cats, per_assign))
    if alpha is not None:
        T.ALPHA = alpha
    m = T.CFTSK().fit(X[sp.tr], sp.yb_tr)
    T.ALPHA = 1.0
    return m, m.score(X[sp.te])[0], X


def auc_ci(sp, s, conf=95.0):
    return [float(T.auc(sp.yb_te, s)), ci(sp.yb_te, sp.gt, lambda t: T.auc(sp.yb_te, s, t), conf)]


def diff_ci(sp, a, b, conf=95.0):
    return [float(T.auc(sp.yb_te, a) - T.auc(sp.yb_te, b)),
            ci(sp.yb_te, sp.gt, lambda t: T.auc(sp.yb_te, a, t) - T.auc(sp.yb_te, b, t), conf)]


# ---------------------------------------------------------------- R1-A
def r1a(units, d):
    sp = Split(units, *d)
    X = np.log1p(sp.counts(WCP))
    ec = T.CFTSK(); ec.a = np.quantile(X[sp.tr], 0.25, 0); ec.b = np.quantile(X[sp.tr], 0.75, 0)
    ec.b = np.where(ec.b - ec.a < 1e-9, ec.a + 1e-3, ec.b)
    ec.z = np.array([((t[0] == "HIGH") + (t[1] == "LOW") + (t[2] == "HIGH")) / 3 for t in T.RULES])
    s_ec = ec.score(X[sp.te])[0]
    _, s_cf, _ = cf_scores(sp, WCP)
    return {"ec_tsk": auc_ci(sp, s_ec), "cf_tsk_same_inputs": auc_ci(sp, s_cf),
            "cf_minus_ec": diff_ci(sp, s_cf, s_ec), "ec_consequents": ec.z.tolist()}


# ---------------------------------------------------------------- R1-B
def r1b(units, d):
    out = {}
    sp = Split(units, *d)
    oh = OneHotEncoder(sparse_output=False).fit(sp.grp[:, None])
    G = oh.transform(sp.grp[:, None])
    s = LogisticRegression(max_iter=1000).fit(G[sp.tr], sp.yb_tr).predict_proba(G[sp.te])[:, 1]
    out["group_only"] = auc_ci(sp, s)
    for name, label, pa in (("within_group_label", "group", False), ("per_assignment_counts", "pooled", True),
                            ("both", "group", True)):
        spv = Split(units, *d, label=label)
        sel, _ = select(spv, pa)
        _, s_cf, _ = cf_scores(spv, sel, pa)
        C8 = spv.counts(BROAD, pa)
        s_lr8 = lr().fit(C8[spv.tr], spv.yb_tr).predict_proba(C8[spv.te])[:, 1]
        out[name] = {"selected": sel, "cf_tsk": auc_ci(spv, s_cf), "lr8": auc_ci(spv, s_lr8),
                     "cf_minus_lr8": diff_ci(spv, s_cf, s_lr8), "test_prevalence": float(spv.yb_te.mean())}
    return out


def prevalence(units):
    y = np.array([u["y"] for u in units]); sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
    grp = np.array([u["group"] for u in units]); out = {}
    for name, thr in (("pooled_threshold", np.quantile(y, 0.75)), ("f24_threshold", np.quantile(y[sem == "f24"], 0.75)),
                      ("s25_threshold", np.quantile(y[sem == "s25"], 0.75))):
        out[name] = {"threshold": float(thr)}
        for s_ in ("f24", "s25"):
            for g_ in ("g1", "g2", "g3"):
                m = (sem == s_) & (grp == g_)
                out[name][f"{s_}_{g_}"] = {"n": int(m.sum()), "rate": float((y[m] >= thr).mean())}
            out[name][f"{s_}_all"] = float((y[sem == s_] >= thr).mean())
    return out


# ---------------------------------------------------------------- R1-C
def r1c(units, d):
    sp = Split(units, *d)
    sel, _ = select(sp)
    _, _, X = cf_scores(sp, sel)
    m = T.CFTSK().fit(X[sp.tr], sp.yb_tr)
    s = m.score(X)[0]
    pe = np.array([u["prior_exam"] for u in units])
    te = sp.te[~np.isnan(pe[sp.te])]
    yb_all = np.zeros(len(units), int); yb_all[sp.te] = sp.yb_te
    def coef(idx):
        Z = np.column_stack([s[idx], pe[idx]]); Z = (Z - Z.mean(0)) / Z.std(0)
        return LogisticRegression(C=1e6, max_iter=2000).fit(Z, yb_all[idx]).coef_[0]
    c = coef(te)
    rng = np.random.default_rng(0); gte = sp.g[te]; ug = np.unique(gte)
    idx = {k: te[gte == k] for k in ug}; bs = []
    for _ in range(2000):
        t = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        if len(set(yb_all[t])) == 2:
            bs.append(coef(t))
    bs = np.array(bs)
    return {"n_units": int(len(te)), "coef_cf": float(c[0]), "coef_cf_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
            "coef_prior": float(c[1]), "coef_prior_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
            "auc_cf_subset": float(T.auc(yb_all, s, te)), "auc_prior_subset": float(T.auc(yb_all, -pe, te))}


def prior_corr(units):
    pe = np.array([u["prior_exam"] for u in units]); ok = ~np.isnan(pe)
    C = np.log1p(T.counts(units, WCP)); n = np.log1p([len(u["da"]) for u in units])
    out = {}
    for k, x in (("log_W", C[:, 0]), ("log_C", C[:, 1]), ("log_P", C[:, 2]), ("log_turns", np.array(n))):
        r, p = spearmanr(x[ok], pe[ok]); out[k] = {"rho": float(r), "p": float(p)}
    out["n"] = int(ok.sum())
    return out


# ---------------------------------------------------------------- R1-D
class ANFIS(torch.nn.Module):
    def __init__(self, a, b, base):
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor((a + b) / 2, dtype=torch.float))
        self.k = torch.nn.Parameter(torch.tensor(4.0 / np.maximum(b - a, 1e-3), dtype=torch.float))
        self.th = torch.nn.Parameter(torch.full((len(T.RULES),), float(np.log(base / (1 - base)))))

    def forward(self, X):
        hi = torch.sigmoid(self.k * (X - self.c)); cols = []
        for terms in T.RULES:
            w = torch.ones(len(X))
            for i, term in enumerate(terms):
                w = w * (hi[:, i] if term == "HIGH" else 1 - hi[:, i])
            cols.append(w)
        W = torch.stack(cols, 1); z = torch.sigmoid(self.th)
        return ((W * z).sum(1) + T.W0 * T.Z0) / (W.sum(1) + T.W0)


def anfis_scores(Xtr, ytr, Xte, seeds=10):
    a, b = np.quantile(Xtr, 0.25, 0), np.quantile(Xtr, 0.75, 0); b = np.where(b - a < 1e-9, a + 1e-3, b)
    out = []
    for s in range(seeds):
        torch.manual_seed(s)
        m = ANFIS(a, b, max(min(ytr.mean(), 0.99), 0.01)); opt = torch.optim.Adam(m.parameters(), 0.01)
        xt, yt = torch.tensor(Xtr, dtype=torch.float), torch.tensor(ytr, dtype=torch.float)
        for _ in range(300):
            loss = torch.nn.functional.binary_cross_entropy(m(xt).clamp(1e-6, 1 - 1e-6), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out.append(m(torch.tensor(Xte, dtype=torch.float)).numpy())
    return np.mean(out, 0)


def wang_mendel(Xtr, ytr, Xte):
    cf = T.CFTSK(); cf.a = np.quantile(Xtr, 0.25, 0); cf.b = np.quantile(Xtr, 0.75, 0)
    cf.b = np.where(cf.b - cf.a < 1e-9, cf.a + 1e-3, cf.b)
    W = cf.firing(Xtr); win = W.argmax(1); z = np.full(len(T.RULES), ytr.mean()); used = []
    for j in range(len(T.RULES)):
        m = win == j
        if m.any():
            z[j] = np.average(ytr[m], weights=W[m, j]); used.append(j)
    cf.z = z
    Wte = cf.firing(Xte); mask = np.zeros(len(T.RULES)); mask[used] = 1
    return (Wte * mask) @ z / ((Wte * mask).sum(1) + 1e-9), len(used)


def r1d(units, d):
    sp = Split(units, *d); sel, _ = select(sp)
    _, s_cf, X = cf_scores(sp, sel)
    Xtr, Xte, ytr = X[sp.tr], X[sp.te], sp.yb_tr
    S = {"cf_tsk": s_cf,
         "lr3": lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1],
         "lr3_interactions": make_pipeline(PolynomialFeatures(2, interaction_only=True, include_bias=False), StandardScaler(),
                                           LogisticRegression(max_iter=2000)).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
         "tree_d2": DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
         "tree_d3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
         "anfis": anfis_scores(Xtr, ytr, Xte)}
    S["wang_mendel"], n_wm = wang_mendel(Xtr, ytr, Xte)
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        S["ebm"] = ExplainableBoostingClassifier(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    except Exception as e:
        print("EBM not run:", e)
    out = {"selected": sel, "wm_rules_used": n_wm}
    for k, s in S.items():
        out[k] = {"auc": auc_ci(sp, s)}
        if k != "cf_tsk":
            out[k]["cf_minus"] = diff_ci(sp, s_cf, s)
    return out


# ---------------------------------------------------------------- R1-F, R1-G
def r1f(units, d):
    sp = Split(units, *d); sel, _ = select(sp); out = {}
    for a in (0.1, 1.0, 10.0):
        m, s, _ = cf_scores(sp, sel, alpha=a)
        out[str(a)] = {"auc": auc_ci(sp, s), "z": m.z.tolist()}
    return out


def r1g(units, d, reps=20):
    sp = Split(units, *d); sel, _ = select(sp)
    m, _, X = cf_scores(sp, sel)
    rng = np.random.default_rng(0); out = {}
    for q in (0.1, 0.2, 0.3, 0.4):
        a = []
        for _ in range(reps):
            Cn = []
            for i in sp.te:
                lab = list(units[i]["broad"])
                for k in range(len(lab)):
                    if rng.random() < q:
                        lab[k] = rng.choice([b for b in BROAD if b != lab[k]])
                Cn.append([lab.count(c) for c in sel])
            a.append(T.auc(sp.yb_te, m.score(np.log1p(np.array(Cn, float)))[0]))
        out[str(q)] = {"mean": float(np.mean(a)), "min": float(np.min(a)), "max": float(np.max(a))}
    out["0.0"] = float(T.auc(sp.yb_te, m.score(X[sp.te])[0]))
    return out


# ---------------------------------------------------------------- R1-H
def r1h(units, st):
    conf = 100 * (1 - 0.05 / 7)
    out = {"conf": conf}
    sp = Split(units, "f24", "s25")
    X = np.log1p(sp.counts(WCP)); cf = T.CFTSK().fit(X[sp.tr], sp.yb_tr); s_cf = cf.score(X[sp.te])[0]
    C8 = sp.counts(BROAD); s_lr8 = lr().fit(C8[sp.tr], sp.yb_tr).predict_proba(C8[sp.te])[:, 1]
    from scipy.stats import rankdata
    pr = lambda x: (rankdata(x) - 1) / (len(x) - 1)
    s_fus = (pr(s_lr8) + pr(s_cf)) / 2
    out["E10_H1"] = diff_ci(sp, s_cf, s_lr8, conf); out["E10_H2"] = auc_ci(sp, s_cf, conf)
    out["E10_H4"] = diff_ci(sp, s_fus, s_lr8, conf)
    sel, _ = select(sp); _, s_n, _ = cf_scores(sp, sel)
    out["E11_H1"] = diff_ci(sp, s_n, s_lr8, conf); out["E11_H2"] = auc_ci(sp, s_n, conf)
    out["E10_H3_p"] = st["E10_transfer"]["primary"]["H3"]["wilcoxon_p"]
    out["E12_H5_p"] = st["E12_lstm"]["primary"]["wilcoxon_p"]
    out["decisions"] = {"E10_H1": out["E10_H1"][1][0] > -0.05, "E10_H2": out["E10_H2"][1][0] > 0.5,
                        "E10_H3": out["E10_H3_p"] < 0.05 / 7, "E10_H4": out["E10_H4"][1][0] > 0,
                        "E11_H1": out["E11_H1"][1][0] > -0.05, "E11_H2": out["E11_H2"][1][0] > 0.5,
                        "E12_H5": out["E12_H5_p"] < 0.05 / 7}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/revision_stats.json"); ap.add_argument("--noise-reps", type=int, default=20)
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); st = json.load(open(a.stats))
    D = {"primary": ("f24", "s25"), "reverse": ("s25", "f24")}
    res = {"prevalence": prevalence(units), "prior_corr": prior_corr(units)}
    for name, fn in (("R1A", r1a), ("R1B", r1b), ("R1C", r1c), ("R1D", r1d), ("R1F", r1f)):
        res[name] = {k: fn(units, d) for k, d in D.items()}
        print(name, "done", flush=True)
    res["R1G"] = {k: r1g(units, d, a.noise_reps) for k, d in D.items()}
    res["R1H"] = r1h(units, st)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    st["R1"] = json.loads(json.dumps(res, default=float)); json.dump(st, open(a.stats, "w"), indent=1)
    print(json.dumps(res, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
