"""
run_r5.py -- revision R5 analyses (PREREG_R5.md). Exploratory.

    python study2/run_r5.py --tutortrace PATH --recipe CSV --units results/units.pkl --weka DIR --out results/r5_stats.json
    python study2/run_r5.py --synthetic --out results/r5_synthetic.json
"""
import argparse, json, os, pickle, subprocess, sys, tempfile, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import lsq_linear, minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "study3"), os.path.join(HERE, "..", "src")]
import run_s2 as S2
import run_s3 as S3
import run_transfer as T
import run_r2 as R
import run_r3 as Q
import run_r4 as F4
warnings.filterwarnings("ignore")
SESOI = 0.01


# ------------------------------------------------------------------ verdict rule (R5-A)
def ci_at(y, s, g, conf=95.0, B=2000):
    T.CONF = conf
    return [float(roc_auc_score(y, s)), T.boot(y, g, lambda t: T.auc(y, s, t), B=B)]


def gain_at(y, a, b, g, conf=95.0, B=2000):
    T.CONF = conf
    return [float(roc_auc_score(y, a) - roc_auc_score(y, b)), T.boot(y, g, lambda t: T.auc(y, a, t) - T.auc(y, b, t), B=B)]


def check1(tr, sel, signs, within_fn=None):
    k = len(sel); conf = 100 * (1 - 0.05 / k); res = {}
    g = tr.student_id.astype(str).values
    for s, d in zip(sel, signs):
        x = S2.X_of(tr, [s])[:, 0]
        w = R.within_rho(x, tr.y.values, g, conf) if within_fn is None else within_fn(x, tr.y.values, g, conf)
        lo, hi = w["ci"]
        res[s] = {"rho": w["rho"], "ci": [lo, hi], "state": "supported" if (lo > 0 and d > 0) or (hi < 0 and d < 0)
                  else ("contradicted" if (lo > 0 and d < 0) or (hi < 0 and d > 0) else "inconclusive")}
    st = [v["state"] for v in res.values()]
    verdict = "contradicted" if "contradicted" in st else ("supported" if all(x == "supported" for x in st) else "inconclusive")
    return {"inputs": res, "verdict": verdict, "level": conf}


def incremental(tr, te, f_tr, f_te, cols_b, B=2000):
    y, g = te.y.values, te.student_id.astype(str).values
    Btr, Bte = tr[cols_b].astype(float).values, te[cols_b].astype(float).values
    b = R.lr().fit(Btr, tr.y.values)
    bf = R.lr().fit(np.column_stack([Btr, f_tr]), tr.y.values)
    sb, sbf = b.predict_proba(Bte)[:, 1], bf.predict_proba(np.column_stack([Bte, f_te]))[:, 1]
    return gain_at(y, sbf, sb, g, B=B), sb, b.predict_proba(Btr)[:, 1]


def state2(gain):
    lo, hi = gain[1]
    return "pass" if lo > 0 else ("fail" if hi < SESOI else "inconclusive")


def state3(auc):
    lo, hi = auc[1]
    return "pass" if lo > 0.5 else ("fail" if hi <= 0.5 else "inconclusive")


def overall(c1, c2, c3):
    if c1 == "contradicted" or "fail" in (c2, c3):
        return "not behavioural"
    if c1 == "supported" and c2 == "pass" and c3 == "pass":
        return "behavioural"
    return "inconclusive"


def audit(tr, te, f_tr, f_te, struct, mcol, pcol, c1):
    y, g = te.y.values, te.student_id.astype(str).values
    allb = struct + [mcol, pcol]
    gain, _, _ = incremental(tr, te, f_tr, f_te, allb)
    per = {nm: incremental(tr, te, f_tr, f_te, cols)[0] for nm, cols in (("structure", struct), ("persistence", [mcol]), ("propensity", [pcol]))}
    auc = ci_at(y, f_te, g)
    c2, c3 = state2(gain), state3(auc)
    return {"auc": auc, "gain_all": gain, "gain_each": per, "check1": c1, "check2": c2, "check3": c3,
            "overall": overall(c1, c2, c3)}


# ------------------------------------------------------------------ RF-TSK (R5-D)
def deviations(df, cols, keys):
    X = S2.X_of(df, cols); out = np.zeros_like(X)
    order = df.reset_index(drop=True)
    for _, idx in order.groupby(keys, sort=False).indices.items():
        idx = np.sort(idx)
        cs = np.cumsum(X[idx], axis=0)
        prev_mean = np.vstack([np.zeros((1, X.shape[1])), cs[:-1] / np.arange(1, len(idx))[:, None]])
        out[idx] = X[idx] - prev_mean
        out[idx[0]] = 0.0
    return out


class ResTSK:
    def fit(self, X, r):
        self.a = np.quantile(X, 0.25, 0); self.b = np.quantile(X, 0.75, 0)
        self.b = np.where(self.b - self.a < 1e-9, self.a + 1e-3, self.b)
        W = self.firing(X); A = W / (W.sum(1) + T.W0)[:, None]
        M = np.vstack([A, np.sqrt(T.ALPHA) * np.eye(len(T.RULES))]); t = np.concatenate([r, np.zeros(len(T.RULES))])
        self.z = lsq_linear(M, t, bounds=(-1, 1)).x
        return self

    firing = T.CFTSK.firing

    def score(self, X):
        W = self.firing(X)
        return (W @ self.z) / (W.sum(1) + T.W0)


def rf_tsk(tr, te, sel, struct, mcol, pcol, keys):
    allb = struct + [mcol, pcol]
    _, p0_te, p0_tr = incremental(tr, te, np.zeros(len(tr)), np.zeros(len(te)), allb, B=10)
    Dtr, Dte = deviations(tr, sel, keys), deviations(te, sel, keys)
    m = ResTSK().fit(Dtr, tr.y.values - p0_tr)
    s_tr, s_te = p0_tr + m.score(Dtr), p0_te + m.score(Dte)
    y, g = te.y.values, te.student_id.astype(str).values
    gain = gain_at(y, s_te, p0_te, g)
    auc = ci_at(y, s_te, g)
    return {"z": m.z.tolist(), "auc": auc, "gain_all": gain, "check2": state2(gain), "check3": state3(auc)}


# ------------------------------------------------------------------ FURIA
def furia(tr, te, cols, weka):
    cp = ":".join(os.path.join(weka, j) for j in ("weka.jar", "furia.jar", "bounce.jar", "compress.jar"))
    with tempfile.TemporaryDirectory() as d:
        for nm, df in (("tr", tr), ("te", te)):
            X = S2.X_of(df, cols)
            with open(os.path.join(d, nm + ".arff"), "w") as f:
                f.write("@relation r\n" + "".join(f"@attribute a{i} numeric\n" for i in range(len(cols))) + "@attribute y {0,1}\n@data\n")
                for row, t in zip(X, df.y.values):
                    f.write(",".join(f"{v:.6g}" for v in row) + f",{int(t)}\n")
        out = subprocess.run(["java", "-cp", cp, "weka.classifiers.rules.FURIA", "-t", os.path.join(d, "tr.arff"), "-T",
                              os.path.join(d, "te.arff"), "-classifications", "weka.classifiers.evaluation.output.prediction.CSV -distribution"],
                             capture_output=True, text=True).stdout
        mdl = subprocess.run(["java", "-cp", cp, "weka.classifiers.rules.FURIA", "-t", os.path.join(d, "tr.arff"), "-no-cv"],
                             capture_output=True, text=True).stdout
    p = []
    lines = out.splitlines(); i = [k for k, l in enumerate(lines) if l.startswith("inst#")][0]
    for l in lines[i + 1:]:
        parts = l.split(",")
        if len(parts) < 6 or not parts[0].strip().isdigit():
            continue
        p.append(float(parts[-1].replace("*", "")))
    nrules = next((int(l.split(":")[1]) for l in mdl.splitlines() if l.startswith("Number of Rules")), -1)
    return np.array(p), nrules


# ------------------------------------------------------------------ real data
def block(tr, te, struct, mcol, pcol, keys, weka, study):
    cols = S2.behav_cols(tr)
    sel, signs, _, _ = Q.select3(tr, cols)
    Xtr, Xte = S2.X_of(tr, sel), S2.X_of(te, sel)
    cf = T.CFTSK().fit(Xtr, tr.y.values)
    zm = R.fit_monotone(cf, Xtr, tr.y.values, signs); cm = T.CFTSK(); cm.a, cm.b, cm.z = cf.a, cf.b, zm
    c1 = check1(tr, sel, signs)
    out = {"selected": sel, "signs": signs, "check1": c1}
    out["sc"] = audit(tr, te, cf.score(Xtr)[0], cf.score(Xte)[0], struct, mcol, pcol, c1["verdict"])
    out["sm"] = audit(tr, te, cm.score(Xtr)[0], cm.score(Xte)[0], struct, mcol, pcol, c1["verdict"])
    out["rf"] = rf_tsk(tr, te, sel, struct, mcol, pcol, keys)
    out["rf"]["overall"] = overall(c1["verdict"], out["rf"]["check2"], out["rf"]["check3"])
    for nm, cl in (("furia3", sel), ("furia_all", cols)):
        ptr, nr = furia(tr, tr, cl, weka); pte, _ = furia(tr, te, cl, weka)
        o = audit(tr, te, ptr, pte, struct, mcol, pcol, c1["verdict"] if nm == "furia3" else "not computed")
        o["rules"] = nr; out[nm] = o
    # support thresholds and consequent intervals
    W = cf.firing(Xtr); sup = np.array([(W[:, j] >= 0.5).sum() for j in range(len(T.RULES))])
    out["support"] = sup.tolist(); out["below"] = {str(t): int((sup < t).sum()) for t in (5, 10, 20, 30)}
    out["distinct_sm"] = int(len(np.unique(np.round(zm, 3))))
    if keys is not None and study in ("S2_D1_to_D2", "S3_SW_to_AW_IRW"):
        rng = np.random.default_rng(0); stu = tr.student_id.unique(); Z = []
        for _ in range(500):
            pick = rng.choice(stu, len(stu)); bt = pd.concat([tr[tr.student_id == s] for s in pick], ignore_index=True)
            Xb = S2.X_of(bt, sel); m = T.CFTSK(); m.a, m.b = cf.a, cf.b
            Wb = m.firing(Xb); den = Wb.sum(1) + T.W0; A = Wb / den[:, None]; bv = T.W0 * T.Z0 / den
            Mx = np.vstack([A, np.sqrt(T.ALPHA) * np.eye(8)]); tt = np.concatenate([bt.y.values - bv, np.sqrt(T.ALPHA) * np.full(8, bt.y.mean())])
            m.z = lsq_linear(Mx, tt, bounds=(0, 1)).x
            Z.append(R.fit_monotone(m, Xb, bt.y.values, signs))
        Z = np.array(Z); lo, hi = np.percentile(Z, [2.5, 97.5], axis=0)
        out["sm_ci"] = {"lo": lo.tolist(), "hi": hi.tolist(), "unstable": int(((hi - lo) > 0.5).sum())}
    # SLSQP starts
    rng = np.random.default_rng(1); dev = 0.0
    Wt = cf.firing(Xtr); den = Wt.sum(1) + T.W0; A = Wt / den[:, None]; bv = T.W0 * T.Z0 / den; prior = np.full(8, tr.y.mean())
    obj = lambda z: np.sum((A @ z + bv - tr.y.values) ** 2) + T.ALPHA * np.sum((z - prior) ** 2)
    jac = lambda z: 2 * A.T @ (A @ z + bv - tr.y.values) + 2 * T.ALPHA * (z - prior)
    cons = [{"type": "ineq", "fun": (lambda z, j=j, k=k, s=s: s * (z[k] - z[j]))} for j, k, s in R.pairs_for(signs)]
    for _ in range(10):
        r = minimize(obj, rng.random(8), jac=jac, bounds=[(0, 1)] * 8, constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
        dev = max(dev, float(np.max(np.abs(r.x - zm))))
    out["slsqp_max_dev"] = dev
    return out


def study1(units_path, smoke=False):
    U = pickle.load(open(units_path, "rb"))
    if smoke:
        ys = np.random.default_rng(0).permutation([u["y"] for u in U])
        U = [dict(u, y=float(v)) for u, v in zip(U, ys)]
    sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in U]); grp = np.array([u["group"] for u in U])
    y = np.array([u["y"] for u in U]); g = np.array([u["group_key"] for u in U])
    cell = np.char.add(sem, grp); yr = y.copy()
    for c in np.unique(cell):
        yr[cell == c] = y[cell == c] - y[cell == c].mean()
    C8 = T.counts(U); X3 = np.log1p(T.counts(U, T.CF_INPUTS))
    G = np.column_stack([(grp == k).astype(float) for k in ("g1", "g2", "g3")])
    res = {}
    sems = sorted(set(sem))
    for a, b in ((sems[0], sems[1]), (sems[1], sems[0])):
        tr, te = np.where(sem == a)[0], np.where(sem == b)[0]
        thr = np.quantile(yr[tr], 0.75); ytr, yte = (yr[tr] >= thr).astype(int), (yr[te] >= thr).astype(int)
        cf = T.CFTSK().fit(X3[tr], ytr)
        lr8 = R.lr().fit(C8[tr], ytr); grpm = R.lr().fit(G[tr], ytr)
        res[f"{a}_to_{b}"] = {"cf_tsk": ci_at(yte, cf.score(X3[te])[0], g[te]), "lr8": ci_at(yte, lr8.predict_proba(C8[te])[:, 1], g[te]),
                             "group_only": ci_at(yte, grpm.predict_proba(G[te])[:, 1], g[te]), "base_rate_test": float(yte.mean())}
    return res


# ------------------------------------------------------------------ simulation (R5-E)
def simulate(conf, beta, nr, rng, viol=None):
    rows = []
    nsess = 3 if viol == "sessions" else 1
    for s in range(100):
        cell = rng.integers(3); off = (-1.0, 0.0, 1.0)[cell]; a = rng.normal()
        for ses in range(nsess):
            n = rng.integers(nr[0], nr[1] + 1); prev = None; ys = []
            for t in range(n):
                e = rng.normal()
                if conf == "A":
                    c = off; cx = off
                elif conf == "B":
                    c = a; cx = a
                elif conf == "AB":
                    c = off + a; cx = off + a
                else:
                    c = 0.0 if prev is None else (1.1 if prev == 1 else -1.1); cx = float(np.mean(ys)) if ys else 0.0
                eff = beta * e
                if viol == "nonmonotone":
                    eff = beta * (e ** 2 - 1)
                elif viol == "threshold":
                    eff = 2 * beta * max(e, 0.0)
                elif viol == "timevarying":
                    eff = beta * e * 2 * (t + 1) / n
                y = int(rng.random() < 1 / (1 + np.exp(-(c + eff))))
                if viol == "labelnoise" and rng.random() < 0.15:
                    y = 1 - y
                x1 = cx + e + (rng.normal() if viol == "measurement" else 0.0)
                x2 = 0.7 * a + rng.normal(0, 0.7) if viol == "correlated" else rng.normal()
                rows.append(dict(student_id=f"s{s}", session=ses, cell=cell, x1=x1, x2=x2, x3=rng.normal(), y=y,
                                 markov=0.5 if prev is None else float(prev), prop=float(np.mean(ys)) if ys else 0.5,
                                 c0=float(cell == 0), c1=float(cell == 1), c2=float(cell == 2)))
                ys.append(y); prev = y
    df = pd.DataFrame(rows)
    if viol == "missing":
        df = df[rng.random(len(df)) >= 0.2].reset_index(drop=True)
    return df


def fast_within(x, y, g, conf):
    lo_hi = F4.fast_within_b(x, y, g, 500, np.random.default_rng(0))[1] if conf == 95.0 else None
    if lo_hi is None:
        df = pd.DataFrame({"x": x, "y": y, "g": g})
        df = df[df.groupby("g").g.transform("size") >= 2].copy()
        for c in ("x", "y"):
            df["r" + c] = df.groupby("g")[c].transform(lambda v: (pd.Series(v).rank().values - 1) / max(len(v) - 1, 1))
        gs = df.g.values; ug = np.unique(gs); idx = {k: np.where(gs == k)[0] for k in ug}
        rng = np.random.default_rng(0); bs = []
        from scipy.stats import rankdata
        for _ in range(500):
            t = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
            a_, b_ = rankdata(df.rx.values[t]), rankdata(df.ry.values[t])
            bs.append(np.corrcoef(a_, b_)[0, 1] if a_.std() > 0 and b_.std() > 0 else np.nan)
        lo_hi = np.nanpercentile(bs, [(100 - conf) / 2, 100 - (100 - conf) / 2])
    return {"rho": 0.0, "ci": [float(lo_hi[0]), float(lo_hi[1])]}


def one(args):
    conf, beta, nr, viol, reps = args
    T.auc = F4.fast_auc
    warnings.simplefilter("ignore")
    from statsmodels.discrete.conditional_models import ConditionalLogit
    rec = {k: 0 for k in ("behavioural", "not_behavioural", "inconclusive", "c1_supported", "c2_pass", "c2_fail", "fe_behavioural")}
    aucs = {"sc": [], "sm": [], "sm_reversed": []}; used = 0
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        tr, te = simulate(conf, beta, nr, rng, viol), simulate(conf, beta, nr, rng, viol)
        if tr.y.nunique() < 2 or te.y.nunique() < 2:
            continue
        used += 1
        sel, signs, _, _ = Q.select3(tr, ["x1", "x2", "x3"])
        Xtr, Xte = S2.X_of(tr, sel), S2.X_of(te, sel)
        cf = T.CFTSK().fit(Xtr, tr.y.values)
        zm = R.fit_monotone(cf, Xtr, tr.y.values, signs); cm = T.CFTSK(); cm.a, cm.b, cm.z = cf.a, cf.b, zm
        rs = list(signs); i1 = sel.index("x1") if "x1" in sel else 0; rs[i1] = -rs[i1]
        zr = R.fit_monotone(cf, Xtr, tr.y.values, rs); cr = T.CFTSK(); cr.a, cr.b, cr.z = cf.a, cf.b, zr
        for nm, m in (("sc", cf), ("sm", cm), ("sm_reversed", cr)):
            aucs[nm].append(F4.fast_auc(te.y.values, m.score(Xte)[0]))
        # Check 1 on x1 only (the input whose status is known), within cells and within students
        if "x1" in sel:
            d = signs[sel.index("x1")]; x = S2.X_of(tr, ["x1"])[:, 0]; st = []
            for grp in (tr.cell.astype(str).values, tr.student_id.values):
                ci = fast_within(x, tr.y.values, grp, 95.0)["ci"]
                st.append("supported" if (ci[0] > 0 and d > 0) or (ci[1] < 0 and d < 0) else
                          ("contradicted" if (ci[0] > 0 and d < 0) or (ci[1] < 0 and d > 0) else "inconclusive"))
            c1 = "contradicted" if "contradicted" in st else ("supported" if all(s_ == "supported" for s_ in st) else "inconclusive")
        else:
            c1 = "inconclusive"
        y, g = te.y.values, te.student_id.values
        cb = ["c0", "c1", "c2", "markov", "prop"]
        b = LogisticRegression(max_iter=1000).fit(tr[cb], tr.y); s_tr = cf.score(Xtr)[0]; s_te = cf.score(Xte)[0]
        bf = LogisticRegression(max_iter=1000).fit(np.column_stack([tr[cb].values, s_tr]), tr.y)
        sb, sbf = b.predict_proba(te[cb])[:, 1], bf.predict_proba(np.column_stack([te[cb].values, s_te]))[:, 1]
        T.CONF = 95.0
        gain = [0, T.boot(y, g, lambda t: T.auc(y, sbf, t) - T.auc(y, sb, t), B=500)]
        auc = [0, T.boot(y, g, lambda t: T.auc(y, s_te, t), B=500)]
        c2, c3 = state2(gain), state3(auc); v = overall(c1, c2, c3)
        rec[{"behavioural": "behavioural", "not behavioural": "not_behavioural", "inconclusive": "inconclusive"}[v]] += 1
        rec["c1_supported"] += int(c1 == "supported"); rec["c2_pass"] += int(c2 == "pass"); rec["c2_fail"] += int(c2 == "fail")
        try:
            xx = (tr.x1 - tr.x1.mean()) / tr.x1.std()
            grp_fe = tr.student_id + "_" + tr.session.astype(str)
            r = ConditionalLogit(tr.y.values, xx.values[:, None], groups=grp_fe.values).fit(disp=0)
            lo, hi = r.conf_int()[0]; d = signs[sel.index("x1")] if "x1" in sel else 1
            rec["fe_behavioural"] += int((lo > 0 and d > 0) or (hi < 0 and d < 0))
        except Exception:
            pass
    out = {k: v / used for k, v in rec.items()}
    out.update({f"auc_{k}": float(np.mean(v)) for k, v in aucs.items()})
    out.update(confound=conf, beta=beta, requests=f"{nr[0]}-{nr[1]}", violation=viol or "none", reps=used)
    print(out, flush=True)
    return out


def r5e(reps=100):
    grid = [(c, b, nr, None, reps) for c in ("A", "B", "C") for b in (0.0, 0.5, 1.0) for nr in ((3, 5), (6, 10))]
    grid += [("AB", b, (6, 10), v, reps) for v in ("nonmonotone", "threshold", "labelnoise", "measurement", "sessions",
                                                    "correlated", "missing", "timevarying") for b in (0.0, 0.5)]
    with Pool(2) as pool:
        return pool.map(one, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutortrace"); ap.add_argument("--recipe"); ap.add_argument("--units"); ap.add_argument("--weka")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-smoke", action="store_true"); ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    if a.synthetic:
        json.dump(r5e(a.reps), open(a.out, "w"), indent=1); return
    res = {"S2": {}, "S3": {}}
    Q.set_study("S2")
    D = R.tutortrace(a.tutortrace)
    if a.shuffle_smoke:
        for k in D:
            D[k]["y"] = np.random.default_rng(k).permutation(D[k].y.values)
    for name, tr, te in (("D1_to_D2", D[1], D[2]), ("D2_to_D1", D[2], D[1]),
                         ("D12_to_D3", pd.concat([D[1], D[2]], ignore_index=True), D[3])):
        o = block(tr, te, Q.S2_STRUCT, "prev_dependent", "prop_earlier", ["student_id"], a.weka, f"S2_{name}")
        # (ii) baselines on non-first queries
        nf = te.groupby("student_id").cumcount().values > 0
        sel = o["selected"]; cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values); s = cf.score(S2.X_of(te, sel))[0]
        y, g = te.y.values[nf], te.student_id.astype(str).values[nf]
        o["non_first"] = {"n": int(nf.sum()), "sc": ci_at(y, s[nf], g), "persistence": ci_at(y, te.prev_dependent.values[nf], g),
                          "propensity": ci_at(y, te.prop_earlier.values[nf], g),
                          "sc_minus_persistence": gain_at(y, s[nf], te.prev_dependent.values[nf], g),
                          "sc_minus_propensity": gain_at(y, s[nf], te.prop_earlier.values[nf], g)}
        res["S2"][name] = o; print("S2", name, "done", flush=True)
    Q.set_study("S3")
    F = R.recipe(a.recipe)
    F = F.sort_values(["student_id", "course", "week", "session", "idx"]).reset_index(drop=True)
    sw, oth = F[F.course == "SW"].reset_index(drop=True), F[F.course != "SW"].reset_index(drop=True)
    for name, tr, te in (("SW_to_AW_IRW", sw, oth), ("AW_IRW_to_SW", oth, sw)):
        res["S3"][name] = block(tr, te, S3.STRUCT, "prev_request_production", "prop_earlier_sessions",
                                ["student_id", "course", "week", "session"], a.weka, f"S3_{name}")
        print("S3", name, "done", flush=True)
    if a.units:
        res["S1_residualised"] = study1(a.units, a.shuffle_smoke); print("S1 done", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
