"""
run_r3.py -- revision R3 analyses (PREREG_R3.md). Exploratory.

    python study2/run_r3.py --tutortrace PATH --recipe CSV --out results/r3_stats.json        # R3-A..F (real data)
    python study2/run_r3.py --synthetic --out results/r3_synthetic.json                       # R3-G (simulated data only)
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "study3"), os.path.join(HERE, "..", "src")]
import run_s2 as S2
import run_s3 as S3
import run_transfer as T
import run_r2 as R
warnings.filterwarnings("ignore")

S2_STRUCT = ["is_first_query", "query_index", "time_since_session_start_s", "test_progress"]
S2_STRUCT_A1 = ["is_first_query", "query_index", "time_since_session_start_s"]
S2_BEHAV = S2.behav_cols


def set_study(k):
    if k == "S2":
        S2.STRUCT = S2_STRUCT; S2.behav_cols = S2_BEHAV
    else:
        S2.STRUCT = S3.STRUCT
        S2.behav_cols = lambda df, with_len=False: S3.BEHAV + (["query_length_chars"] if with_len else [])


def ci_conf(y, s, g, conf):
    T.CONF = conf
    return [float(roc_auc_score(y, s)), T.boot(y, g, lambda t: T.auc(y, s, t))]


def diff_conf(y, a, b, g, conf):
    T.CONF = conf
    return [float(roc_auc_score(y, a) - roc_auc_score(y, b)), T.boot(y, g, lambda t: T.auc(y, a, t) - T.auc(y, b, t))]


def select3(tr, cols, seed=0):
    sel, ranking = S2.sign_checked_selection(tr, cols, seed=seed)
    pad = []
    if len(sel) < 3:
        pad = [r["feature"] for r in ranking if r["feature"] not in sel][: 3 - len(sel)]
    sign = {r["feature"]: (1 if r["auc"] > 0.5 else -1) for r in ranking}
    s = sel + pad
    return s, [sign[c] for c in s], ranking, pad


# ------------------------------------------------------------------ R3-A
def counts_within(x, y, g):
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    df = df[df.groupby("g").g.transform("size") >= 2]
    both = df.groupby("g").y.transform(lambda v: v.nunique() == 2)
    return {"students_ge2": int(df.g.nunique()), "students_both_classes": int(df[both].g.nunique()),
            "requests_both_classes": int(both.sum())}


def cond_logit(x, y, g):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    x = (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    inf = df.groupby("g").y.transform(lambda v: v.nunique() == 2)
    df = df[inf]
    if df.g.nunique() < 5 or df.x.std() == 0:
        return {"coef": None, "ci": None, "students": int(df.g.nunique())}
    r = ConditionalLogit(df.y.values, df[["x"]].values, groups=df.g.values).fit(disp=0)
    lo, hi = r.conf_int()[0]
    return {"coef": float(r.params[0]), "ci": [float(lo), float(hi)], "p": float(r.pvalues[0]),
            "students": int(df.g.nunique()), "requests": int(len(df))}


def r3a(tr, te, sel):
    out = {}
    for s in sel:
        o = {}
        for nm, d in (("train", tr), ("test", te)):
            x, y, g = S2.X_of(d, [s])[:, 0], d.y.values, d.student_id.astype(str).values
            o[nm] = {"counts": counts_within(x, y, g), "cond_logit": cond_logit(x, y, g)}
        out[s] = o
    return out


# ------------------------------------------------------------------ R3-B
def r3b(tr, te, cols):
    y = te.y.values; g = te.student_id.astype(str).values
    elig = {c: 0 for c in cols}; sets = {}; aucs = []
    for seed in range(1, 201):
        sel, ranking = S2.sign_checked_selection(tr, cols, seed=seed)
        for r in ranking:
            elig[r["feature"]] += int(r["eligible"])
        s3, _, _, _ = select3(tr, cols, seed=seed)
        key = "|".join(sorted(s3)); sets[key] = sets.get(key, 0) + 1
        cf = T.CFTSK().fit(S2.X_of(tr, s3), tr.y.values); aucs.append(float(roc_auc_score(y, cf.score(S2.X_of(te, s3))[0])))
    share = {c: v / 200 for c, v in elig.items()}
    _, ranking = S2.sign_checked_selection(tr, cols, seed=0)
    order = [r["feature"] for r in ranking]
    stable = [c for c in order if share.get(c, 0) >= 0.8][:3]
    pad = [c for c in order if c not in stable][: 3 - len(stable)] if len(stable) < 3 else []
    ss = stable + pad
    cf = T.CFTSK().fit(S2.X_of(tr, ss), tr.y.values)
    return {"eligible_share": {c: v for c, v in sorted(share.items(), key=lambda kv: -kv[1]) if v > 0},
            "selected_sets": dict(sorted(sets.items(), key=lambda kv: -kv[1])),
            "auc_median": float(np.median(aucs)), "auc_p2_5": float(np.percentile(aucs, 2.5)),
            "auc_p97_5": float(np.percentile(aucs, 97.5)),
            "stability_selected": ss, "stability_padded": pad,
            "stability_auc": ci_conf(y, cf.score(S2.X_of(te, ss))[0], g, 95.0)}


# ------------------------------------------------------------------ R3-E, F
def r3ef(tr, te, sel, signs, struct, markov_col, prop_col, markov_default, prop_default, study):
    y, g = te.y.values, te.student_id.astype(str).values
    Xtr, Xte = S2.X_of(tr, sel), S2.X_of(te, sel)
    cf = T.CFTSK().fit(Xtr, tr.y.values)
    W = cf.firing(Xtr); sup = [int((W[:, j] >= 0.5).sum()) for j in range(len(T.RULES))]
    out = {}
    T.ALPHA = len(tr) / 10.0
    cN = T.CFTSK().fit(Xtr, tr.y.values)
    T.ALPHA = 1.0
    out["alpha_N10"] = {"alpha": len(tr) / 10.0, "auc": ci_conf(y, cN.score(Xte)[0], g, 95.0),
                        "low_support_cells": [{"rule": list(T.RULES[j]), "support": sup[j], "z_alpha1": float(cf.z[j]),
                                               "z_alphaN10": float(cN.z[j])} for j in range(len(T.RULES)) if sup[j] < 10],
                        "base_rate": float(tr.y.mean())}
    zm = R.fit_monotone(cf, Xtr, tr.y.values, signs)
    cm = T.CFTSK(); cm.a, cm.b, cm.z = cf.a, cf.b, zm
    raw = S2.X_of(tr, sel)
    mono = {}
    for nm, model in (("sc_tsk", cf), ("sm_tsk", cm)):
        v = []
        for i in range(len(sel)):
            iqr = np.quantile(raw[:, i], 0.75) - np.quantile(raw[:, i], 0.25)
            delta = 0.1 * iqr if iqr > 1e-9 else 1e-3
            Xp = Xte.copy(); Xp[:, i] += delta
            d = signs[i] * (model.score(Xp)[0] - model.score(Xte)[0])
            v.append(float(np.mean(d < -1e-9)))
        mono[nm] = {"per_input": dict(zip(sel, v)), "overall": float(np.mean(v))}
    out["output_monotonicity"] = mono
    out["defaults"] = {"persistence": int(markov_default.sum()), "propensity": int(prop_default.sum()), "n_test": int(len(te))}
    if study == "S2":
        S_tr, S_te = tr[struct].astype(float).values, te[struct].astype(float).values
        st = R.lr().fit(S_tr, tr.y.values).predict_proba(S_te)[:, 1]
        cols = S2.behav_cols(tr)
        la = R.lr().fit(S2.X_of(tr, cols), tr.y.values).predict_proba(S2.X_of(te, cols))[:, 1]
        out["within_student_scores"] = {"structure_only": R.within_rho(st, y, g), "lr_all_behaviour": R.within_rho(la, y, g),
                                        "sc_tsk": R.within_rho(cf.score(Xte)[0], y, g)}
    return out


# ------------------------------------------------------------------ R3-D
def noisy_prev(raw, e, seed):
    """Recompute lagged-label inputs of Study 3 from labels flipped with probability e (current labels unchanged)."""
    rng = np.random.default_rng(seed)
    d = raw.copy()
    isreq = d.intent_final.isin(S3.PRODUCTION | S3.GUIDANCE).values
    prod = d.intent_final.isin(S3.PRODUCTION).values.astype(float)
    flip = (rng.random(len(d)) < e) & isreq
    d["noisy_prod"] = np.where(flip, 1 - prod, prod)
    d["isR"] = d.intent_final.str.startswith("R ")
    rows = {}
    for _, gq in d.sort_values("idx").groupby(["student_id", "course", "week", "session"], sort=False):
        gq = gq.sort_values("idx")
        npv, isr = gq.noisy_prod.values, gq.isR.values
        for i, ix in enumerate(gq.idx.values):
            m = isr[:i]
            share = float(npv[:i][m].mean()) if m.any() else 0.0
            last = float(npv[:i][m][-1]) if m.any() else 0.5
            rows[(gq.student_id.iloc[i], gq.course.iloc[i], gq.week.iloc[i], gq.session.iloc[i], ix)] = (share, last)
    return rows


def r3d(csv, F, seeds=20):
    raw = pd.read_csv(csv); raw["intent_final"] = raw["intent_final"].astype(str).str.strip()
    out = {}
    for e in (0.1, 0.2, 0.3):
        res = {"SW_to_AW_IRW": {"sc": [], "markov": []}, "AW_IRW_to_SW": {"sc": [], "markov": []}}
        for sd in range(seeds):
            rows = noisy_prev(raw, e, sd)
            G = F.copy()
            key = list(zip(G.student_id, G.course, G.week, G.session, G.idx))
            G["prev_production_share"] = [rows[k][0] for k in key]
            G["prev_request_production"] = [rows[k][1] for k in key]
            sw, oth = G[G.course == "SW"].reset_index(drop=True), G[G.course != "SW"].reset_index(drop=True)
            for nm, tr, te in (("SW_to_AW_IRW", sw, oth), ("AW_IRW_to_SW", oth, sw)):
                sel, _, _, _ = select3(tr, S2.behav_cols(tr))
                cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values)
                res[nm]["sc"].append(float(roc_auc_score(te.y, cf.score(S2.X_of(te, sel))[0])))
                res[nm]["markov"].append(float(roc_auc_score(te.y, te.prev_request_production)))
        out[str(e)] = {k: {m: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for m, v in d.items()}
                       for k, d in res.items()}
    return out


def r3d_conditional(tr, te, sel):
    sub = te[te.prev_production_share == 0].reset_index(drop=True)
    y, g = sub.y.values, sub.student_id.astype(str).values
    cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values)
    o = {"n": int(len(sub)), "pos": int(sub.y.sum()), "sc_tsk": ci_conf(y, cf.score(S2.X_of(sub, sel))[0], g, 95.0)}
    for s in sel:
        if s != "prev_production_share" and sub[s].nunique() > 1:
            o[s] = ci_conf(y, S2.X_of(sub, [s])[:, 0], g, 95.0)
    return o


# ------------------------------------------------------------------ R3-G
def within_b(x, y, g, B=200, rng=None):
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    df = df[df.groupby("g").g.transform("size") >= 2].copy()
    for c in ("x", "y"):
        df["r" + c] = df.groupby("g")[c].transform(lambda v: (rankdata(v) - 1) / max(len(v) - 1, 1))
    gs = df.g.values; ug = np.unique(gs); idx = {k: np.where(gs == k)[0] for k in ug}
    bs = []
    for _ in range(B):
        t = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        bs.append(spearmanr(df.rx.values[t], df.ry.values[t])[0])
    return float(spearmanr(df.rx, df.ry)[0]), np.nanpercentile(bs, [2.5, 97.5])


def simulate(scn, rng):
    rows = []
    for s in range(100):
        cell = rng.integers(3); off = (-1.0, 0.0, 1.0)[cell]; a = rng.normal(); n = rng.integers(3, 9)
        prev = None; ys = []
        for t in range(n):
            e = rng.normal()
            if scn == "S0":
                p = 1 / (1 + np.exp(-off)); x1 = off + rng.normal()
            elif scn == "S1":
                p = 1 / (1 + np.exp(-a)); x1 = a + rng.normal()
            elif scn == "S2":
                p = 1 / (1 + np.exp(-(off + a + e))); x1 = off + a + e
            else:
                p = 0.4 if prev is None else (0.75 if prev == 1 else 0.25)
                x1 = (np.mean(ys) if ys else 0.0) + rng.normal(0, 0.1)
            y = int(rng.random() < p)
            prs = 0.5 if prev is None else float(prev)
            prp = float(np.mean(ys)) if ys else 0.5
            rows.append(dict(student_id=f"s{s}", cell=cell, x1=x1, x2=rng.normal(), x3=rng.normal(), y=y,
                             markov=prs, prop=prp, c0=float(cell == 0), c1=float(cell == 1), c2=float(cell == 2)))
            ys.append(y); prev = y
    return pd.DataFrame(rows)


def r3g(reps=200):
    out = {}
    for scn in ("S0", "S1", "S2", "S3"):
        naive = audit = sel_x1 = 0
        for rep in range(reps):
            rng = np.random.default_rng(rep)
            tr, te = simulate(scn, rng), simulate(scn, rng)
            if tr.y.nunique() < 2 or te.y.nunique() < 2:
                continue
            sel, signs, _, _ = select3(tr, ["x1", "x2", "x3"])
            cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values)
            s = cf.score(S2.X_of(te, sel))[0]
            y, g = te.y.values, te.student_id.values
            T.CONF = 95.0
            lo = T.boot(y, g, lambda t: T.auc(y, s, t), B=200)[0]
            has = "x1" in sel; sel_x1 += has
            naive += int(has and lo > 0.5)
            if not has:
                continue
            d = signs[sel.index("x1")]
            x = S2.X_of(tr, ["x1"])[:, 0]
            ok = True
            for grp in (tr.cell.astype(str).values, tr.student_id.values):
                r, ci = within_b(x, tr.y.values, grp, 200, rng)
                ok &= bool((ci[0] > 0 and d > 0) or (ci[1] < 0 and d < 0))
            if ok:
                st = LogisticRegression(max_iter=1000).fit(tr[["c0", "c1", "c2"]], tr.y).predict_proba(te[["c0", "c1", "c2"]])[:, 1]
                for b in (st, te.markov.values, te.prop.values):
                    lo_d = T.boot(y, g, lambda t: T.auc(y, s, t) - T.auc(y, b, t), B=200)[0]
                    ok &= bool(lo_d > 0)
            audit += int(ok)
        out[scn] = {"reps": reps, "x1_selected": sel_x1 / reps, "naive_behavioural": naive / reps, "audit_behavioural": audit / reps}
        print(scn, out[scn], flush=True)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutortrace"); ap.add_argument("--recipe"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shuffle-smoke", action="store_true", help="permute labels (code test only)")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    if a.synthetic:
        json.dump(r3g(a.reps), open(a.out, "w"), indent=1); return
    res = {"S2": {}, "S3": {}}
    set_study("S2")
    D = R.tutortrace(a.tutortrace)
    if a.shuffle_smoke:
        for k in D:
            D[k]["y"] = np.random.default_rng(k).permutation(D[k].y.values)
    for k in D:
        n = D[k].groupby("student_id").cumcount()
        D[k]["_markov_def"] = (n == 0).values; D[k]["_prop_def"] = (n == 0).values
    for name, tr, te in (("D1_to_D2", D[1], D[2]), ("D2_to_D1", D[2], D[1]),
                         ("D12_to_D3", pd.concat([D[1], D[2]], ignore_index=True), D[3])):
        cols = S2.behav_cols(tr); sel, signs, _, pad = select3(tr, cols)
        y, g = te.y.values, te.student_id.astype(str).values
        o = {"selected": sel, "A": r3a(tr, te, sel), "B": r3b(tr, te, cols)}
        cf = T.CFTSK().fit(S2.X_of(tr, sel), tr.y.values); s_sc = cf.score(S2.X_of(te, sel))[0]
        A1_tr, A1_te = tr[S2_STRUCT_A1].astype(float).values, te[S2_STRUCT_A1].astype(float).values
        st = R.lr().fit(A1_tr, tr.y.values).predict_proba(A1_te)[:, 1]
        cb = R.lr().fit(np.column_stack([A1_tr, cf.score(S2.X_of(tr, sel))[0]]), tr.y.values).predict_proba(np.column_stack([A1_te, s_sc]))[:, 1]
        o["C"] = {"structure_A1": ci_conf(y, st, g, 95.0), "H2_gain_A1": diff_conf(y, cb, st, g, 100 * (1 - 0.05 / 3)),
                  "sc_minus_structure_A1": diff_conf(y, s_sc, st, g, 95.0)}
        o["EF"] = r3ef(tr, te, sel, signs, S2_STRUCT, "prev_dependent", "prop_earlier", te._markov_def.values, te._prop_def.values, "S2")
        res["S2"][name] = o; print("S2", name, "done", flush=True)
    set_study("S3")
    F = R.recipe(a.recipe)
    F["_markov_def"] = F.prev_request_production == 0.5
    so = F.week * 100 + F.session
    F["_prop_def"] = [not ((F.student_id == r.student_id) & (F.course == r.course) & (so < so[i])).any() for i, r in F.iterrows()]
    sw, oth = F[F.course == "SW"].reset_index(drop=True), F[F.course != "SW"].reset_index(drop=True)
    for name, tr, te in (("SW_to_AW_IRW", sw, oth), ("AW_IRW_to_SW", oth, sw)):
        cols = S2.behav_cols(tr); sel, signs, _, pad = select3(tr, cols)
        o = {"selected": sel, "A": r3a(tr, te, sel), "B": r3b(tr, te, cols), "D_conditional": r3d_conditional(tr, te, sel)}
        o["EF"] = r3ef(tr, te, sel, signs, S3.STRUCT, "prev_request_production", "prop_earlier_sessions",
                       te._markov_def.values, np.asarray(te._prop_def.values, bool), "S3")
        res["S3"][name] = o; print("S3", name, "done", flush=True)
    res["S3"]["D_noise"] = r3d(a.recipe, F)
    print("S3 noise done", flush=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
