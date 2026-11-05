"""
run_r4.py -- revision R4 analyses (PREREG_R4.md). Exploratory.

    python study2/run_r4.py --tutortrace PATH --recipe CSV --out results/r4_stats.json     # R4-A..D (real data)
    python study2/run_r4.py --synthetic --out results/r4_synthetic.json                    # R4-E (simulated data only)
"""
import argparse, itertools, json, os, sys, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "study3"), os.path.join(HERE, "..", "src")]
import run_s2 as S2
import run_s3 as S3
import run_transfer as T
import run_r2 as R
import run_r3 as Q
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ helpers
def within_point(x, y, g):
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    df = df[df.groupby("g").g.transform("size") >= 2].copy()
    if len(df) < 4:
        return 0.0
    for c in ("x", "y"):
        df["r" + c] = df.groupby("g")[c].transform(lambda v: (rankdata(v) - 1) / max(len(v) - 1, 1))
    r = spearmanr(df.rx, df.ry)[0]
    return 0.0 if np.isnan(r) else float(r)


def scw_selection(tr, cols, k=3, seed=0):
    """SCw-TSK: eligibility by the sign of the within-student rank correlation in both halves (R4-A)."""
    X, y, g = S2.X_of(tr, cols), tr.y.values, tr.student_id.astype(str).values
    rng = np.random.default_rng(seed); stu = tr.student_id.unique(); rng.shuffle(stu)
    half = tr.student_id.isin(stu[: len(stu) // 2]).values
    rows = []
    for j, c in enumerate(cols):
        if np.unique(X[:, j]).size < 2:
            continue
        a = roc_auc_score(y, X[:, j]); d = np.sign(a - 0.5)
        r1 = within_point(X[half, j], y[half], g[half]); r2 = within_point(X[~half, j], y[~half], g[~half])
        ok = bool(d != 0 and np.sign(r1) == np.sign(r2) == d)
        rows.append(dict(feature=c, auc=float(a), w1=r1, w2=r2, eligible=ok))
    rows.sort(key=lambda r: -abs(r["auc"] - 0.5))
    sel = [r["feature"] for r in rows if r["eligible"]][:k]
    pad = [r["feature"] for r in rows if r["feature"] not in sel][: k - len(sel)] if len(sel) < k else []
    sign = {r["feature"]: (1 if r["auc"] > 0.5 else -1) for r in rows}
    s = sel + pad
    return s, [sign[c] for c in s], rows, pad


class TSK3:
    """Zero-order TSK with three terms per input (27 rules), consequents by Eq. (2)."""
    def fit(self, X, y):
        self.q = np.quantile(X, [0.25, 0.5, 0.75], axis=0)
        self.rules = list(itertools.product(range(3), repeat=X.shape[1]))
        W = self.firing(X); den = W.sum(1) + T.W0
        A = W / den[:, None]; bv = T.W0 * T.Z0 / den
        prior = np.full(len(self.rules), y.mean())
        M = np.vstack([A, np.sqrt(T.ALPHA) * np.eye(len(self.rules))])
        t = np.concatenate([y - bv, np.sqrt(T.ALPHA) * prior])
        self.z = lsq_linear(M, t, bounds=(0, 1)).x
        return self

    def mem(self, X):
        q1, q2, q3 = self.q
        lo = np.maximum(q2 - q1, 1e-3); hi = np.maximum(q3 - q2, 1e-3)
        L = np.clip((q2 - X) / lo, 0, 1); H = np.clip((X - q2) / hi, 0, 1); Mm = 1 - L - H
        return np.stack([L, Mm, H], axis=2)

    def firing(self, X):
        m = self.mem(X); W = np.ones((len(X), len(self.rules)))
        for j, r in enumerate(self.rules):
            for i, t in enumerate(r):
                W[:, j] *= m[:, i, t]
        return W

    def score(self, X):
        W = self.firing(X)
        return (W @ self.z + T.W0 * T.Z0) / (W.sum(1) + T.W0)


def gaps(tr, te, s_tr, s_te, struct, markov_col, prop_col):
    y, g = te.y.values, te.student_id.astype(str).values
    S_tr, S_te = tr[struct].astype(float).values, te[struct].astype(float).values
    st = R.lr().fit(S_tr, tr.y.values).predict_proba(S_te)[:, 1]
    cb = R.lr().fit(np.column_stack([S_tr, s_tr]), tr.y.values).predict_proba(np.column_stack([S_te, s_te]))[:, 1]
    return {"auc": R.auc_ci(y, s_te, g), "structure": R.diff_ci(y, cb, st, g),
            "persistence": R.diff_ci(y, s_te, te[markov_col].values, g),
            "propensity": R.diff_ci(y, s_te, te[prop_col].values, g)}


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": k, "n": int((b == k).sum()), "mean_p": float(p[b == k].mean()), "obs": float(y[b == k].mean())}
            for k in range(bins) if (b == k).any()]


def block(tr, te, struct, markov_col, prop_col, study, cols):
    y = te.y.values
    sel, signs, _, _ = Q.select3(tr, cols)
    Xtr, Xte = S2.X_of(tr, sel), S2.X_of(te, sel)
    cf = T.CFTSK().fit(Xtr, tr.y.values)
    zm = R.fit_monotone(cf, Xtr, tr.y.values, signs); cm = T.CFTSK(); cm.a, cm.b, cm.z = cf.a, cf.b, zm
    out = {"selected": sel, "signs": signs}
    # R4-A
    sw, swsig, rows, pad = scw_selection(tr, cols)
    c2 = T.CFTSK().fit(S2.X_of(tr, sw), tr.y.values)
    out["scw"] = {"selected": sw, "padded": pad, "eligible": [r["feature"] for r in rows if r["eligible"]],
                  **gaps(tr, te, c2.score(S2.X_of(tr, sw))[0], c2.score(S2.X_of(te, sw))[0], struct, markov_col, prop_col)}
    # R4-B
    t3 = TSK3().fit(Xtr, tr.y.values)
    out["tsk3"] = gaps(tr, te, t3.score(Xtr), t3.score(Xte), struct, markov_col, prop_col)
    hg = HistGradientBoostingClassifier(monotonic_cst=signs, random_state=0).fit(Xtr, tr.y.values)
    out["mono_gbm"] = gaps(tr, te, hg.predict_proba(Xtr)[:, 1], hg.predict_proba(Xte)[:, 1], struct, markov_col, prop_col)
    out["sc_gaps"] = gaps(tr, te, cf.score(Xtr)[0], cf.score(Xte)[0], struct, markov_col, prop_col)
    out["sm_gaps"] = gaps(tr, te, cm.score(Xtr)[0], cm.score(Xte)[0], struct, markov_col, prop_col)
    # R4-D
    out["calibration"] = {"base_rate_test": float(y.mean()), "base_rate_train": float(tr.y.mean())}
    for nm, m in (("sc_tsk", cf), ("sm_tsk", cm)):
        p = np.clip(m.score(Xte)[0], 0, 1)
        out["calibration"][nm] = {"brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "reliability": reliability(y, p),
                                  "brier_base": float(brier_score_loss(y, np.full(len(y), tr.y.mean())))}
    out["sm_z"] = zm.tolist(); out["sc_z"] = cf.z.tolist()
    return out


def fast_auc(y, s, t=None):
    """Mann-Whitney AUC with average ranks for ties; identical to roc_auc_score."""
    if t is not None:
        y, s = y[t], s[t]
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_within_b(x, y, g, B, rng):
    """As run_r3.within_b, with spearmanr replaced by the Pearson correlation of ranks (identical value)."""
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    df = df[df.groupby("g").g.transform("size") >= 2].copy()
    for c in ("x", "y"):
        df["r" + c] = df.groupby("g")[c].transform(lambda v: (rankdata(v) - 1) / max(len(v) - 1, 1))
    gs = df.g.values; ug = np.unique(gs); idx = {k: np.where(gs == k)[0] for k in ug}
    rx, ry = df.rx.values, df.ry.values
    bs = []
    for _ in range(B):
        t = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        a, b = rankdata(rx[t]), rankdata(ry[t])
        bs.append(np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan)
    return None, np.nanpercentile(bs, [2.5, 97.5])


# ------------------------------------------------------------------ R4-E
def simulate(conf, beta, nr, rng):
    rows = []
    for s in range(100):
        cell = rng.integers(3); off = (-1.0, 0.0, 1.0)[cell]; a = rng.normal(); n = rng.integers(nr[0], nr[1] + 1)
        prev = None; ys = []
        for t in range(n):
            e = rng.normal()
            if conf == "A":
                c = off; cx = off
            elif conf == "B":
                c = a; cx = a
            else:
                c = 0.0 if prev is None else (1.1 if prev == 1 else -1.1); cx = float(np.mean(ys)) if ys else 0.0
            p = 1 / (1 + np.exp(-(c + beta * e)))
            y = int(rng.random() < p)
            rows.append(dict(student_id=f"s{s}", cell=cell, x1=cx + e, x2=rng.normal(), x3=rng.normal(), y=y,
                             markov=0.5 if prev is None else float(prev), prop=float(np.mean(ys)) if ys else 0.5,
                             c0=float(cell == 0), c1=float(cell == 1), c2=float(cell == 2)))
            ys.append(y); prev = y
    return pd.DataFrame(rows)


def one_condition(args):
    conf, beta, nr, reps, B = args
    T.auc = fast_auc
    rec = {k: 0 for k in ("x1_selected", "noise_eligible_split", "noise_eligible_within", "naive", "check1",
                          "check2_structure", "check2_persistence", "check2_propensity", "all_checks")}
    used = 0
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        tr, te = simulate(conf, beta, nr, rng), simulate(conf, beta, nr, rng)
        if tr.y.nunique() < 2 or te.y.nunique() < 2:
            continue
        used += 1
        cols = ["x1", "x2", "x3"]
        _, ranking = S2.sign_checked_selection(tr, cols)
        rec["noise_eligible_split"] += np.mean([r["eligible"] for r in ranking if r["feature"] in ("x2", "x3")])
        _, _, rows, _ = scw_selection(tr, cols)
        rec["noise_eligible_within"] += np.mean([r["eligible"] for r in rows if r["feature"] in ("x2", "x3")])
        sel, signs, _, _ = Q.select3(tr, cols)
        cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values)
        s_tr, s = cf.score(S2.X_of(tr, sel))[0], cf.score(S2.X_of(te, sel))[0]
        y, g = te.y.values, te.student_id.values
        T.CONF = 95.0
        nv = T.boot(y, g, lambda t: T.auc(y, s, t), B=B)[0] > 0.5
        rec["naive"] += int(nv)
        if "x1" not in sel:
            continue
        rec["x1_selected"] += 1
        d = signs[sel.index("x1")]; x = S2.X_of(tr, ["x1"])[:, 0]
        c1 = True
        for grp in (tr.cell.astype(str).values, tr.student_id.values):
            _, ci = fast_within_b(x, tr.y.values, grp, B, rng)
            c1 &= bool((ci[0] > 0 and d > 0) or (ci[1] < 0 and d < 0))
        rec["check1"] += int(c1)
        S_tr, S_te = tr[["c0", "c1", "c2"]].values, te[["c0", "c1", "c2"]].values
        st = LogisticRegression(max_iter=1000).fit(S_tr, tr.y).predict_proba(S_te)[:, 1]
        cb = LogisticRegression(max_iter=1000).fit(np.column_stack([S_tr, s_tr]), tr.y).predict_proba(np.column_stack([S_te, s]))[:, 1]
        c2 = {}
        for nm, a_, b_ in (("structure", cb, st), ("persistence", s, te.markov.values), ("propensity", s, te.prop.values)):
            c2[nm] = T.boot(y, g, lambda t: T.auc(y, a_, t) - T.auc(y, b_, t), B=B)[0] > 0
            rec["check2_" + nm] += int(c2[nm])
        rec["all_checks"] += int(c1 and nv and all(c2.values()))
    out = {k: v / used for k, v in rec.items()}
    out.update(confound=conf, beta=beta, requests=f"{nr[0]}-{nr[1]}", reps=used)
    print(out, flush=True)
    return out


def r4e(reps=100, B=500):
    grid = [(c, b, nr, reps, B) for c in ("A", "B", "C") for b in (0.0, 0.25, 0.5, 1.0) for nr in ((3, 5), (6, 10))]
    with Pool(2) as pool:
        return pool.map(one_condition, grid)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutortrace"); ap.add_argument("--recipe"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--shuffle-smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=100); ap.add_argument("--B", type=int, default=500)
    a = ap.parse_args()
    if a.synthetic:
        json.dump(r4e(a.reps, a.B), open(a.out, "w"), indent=1); return
    res = {"S2": {}, "S3": {}}
    Q.set_study("S2")
    D = R.tutortrace(a.tutortrace)
    if a.shuffle_smoke:
        for k in D:
            D[k]["y"] = np.random.default_rng(k).permutation(D[k].y.values)
    for name, tr, te in (("D1_to_D2", D[1], D[2]), ("D2_to_D1", D[2], D[1]),
                         ("D12_to_D3", pd.concat([D[1], D[2]], ignore_index=True), D[3])):
        res["S2"][name] = block(tr, te, Q.S2_STRUCT, "prev_dependent", "prop_earlier", "S2", S2.behav_cols(tr))
        print("S2", name, "done", flush=True)
    Q.set_study("S3")
    F = R.recipe(a.recipe)
    sw, oth = F[F.course == "SW"].reset_index(drop=True), F[F.course != "SW"].reset_index(drop=True)
    for name, tr, te in (("SW_to_AW_IRW", sw, oth), ("AW_IRW_to_SW", oth, sw)):
        cols = S2.behav_cols(tr)
        o = block(tr, te, S3.STRUCT, "prev_request_production", "prop_earlier_sessions", "S3", cols)
        # R4-C (i) candidates without n_prev_turns
        c_ = [c for c in cols if c != "n_prev_turns"]
        sel, signs, _, pad = Q.select3(tr, c_)
        cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values)
        o["no_turns"] = {"selected": sel, "padded": pad,
                         **gaps(tr, te, cf.score(S2.X_of(tr, sel))[0], cf.score(S2.X_of(te, sel))[0], S3.STRUCT,
                                "prev_request_production", "prop_earlier_sessions")}
        # R4-C (ii) bagged
        btr, bte = np.zeros(len(tr)), np.zeros(len(te))
        for seed in range(1, 201):
            s3, _, _, _ = Q.select3(tr, cols, seed=seed)
            m = T.CFTSK().fit(S2.X_of(tr, s3), tr.y.values)
            btr += m.score(S2.X_of(tr, s3))[0] / 200; bte += m.score(S2.X_of(te, s3))[0] / 200
        o["bagged"] = gaps(tr, te, btr, bte, S3.STRUCT, "prev_request_production", "prop_earlier_sessions")
        res["S3"][name] = o; print("S3", name, "done", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
