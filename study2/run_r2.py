"""
run_r2.py -- revision R2 analyses (PREREG_R2.md) for Studies 1-3.

    python study2/run_r2.py --tutortrace PATH --recipe CSV --units results/units.pkl --out results/r2_stats.json
"""
import argparse, json, os, sys, pickle, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "study3"), os.path.join(HERE, "..", "src")]
import run_s2 as S2
import run_s3 as S3
import run_transfer as T
warnings.filterwarnings("ignore")
rng0 = np.random.default_rng(0)
PADDED = []


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def auc_ci(y, s, g, conf=95.0):
    T.CONF = conf
    return [float(roc_auc_score(y, s)), T.boot(y, g, lambda t: T.auc(y, s, t))]


def diff_ci(y, a, b, g):
    T.CONF = 95.0
    return [float(roc_auc_score(y, a) - roc_auc_score(y, b)), T.boot(y, g, lambda t: T.auc(y, a, t) - T.auc(y, b, t))]


def within_rho(x, y, g, conf=95.0):
    """Spearman of within-student ranks (students with >= 2 rows)."""
    df = pd.DataFrame({"x": x, "y": y, "g": g})
    df = df[df.groupby("g").g.transform("size") >= 2].copy()
    for c in ("x", "y"):
        df["r" + c] = df.groupby("g")[c].transform(lambda v: (rankdata(v) - 1) / max(len(v) - 1, 1))
    gs = df.g.values; ug = np.unique(gs); idx = {k: np.where(gs == k)[0] for k in ug}
    bs = []
    for _ in range(2000):
        t = np.concatenate([idx[k] for k in rng0.choice(ug, len(ug))])
        bs.append(spearmanr(df.rx.values[t], df.ry.values[t])[0])
    lo, hi = np.nanpercentile(bs, [(100 - conf) / 2, 100 - (100 - conf) / 2])
    return {"rho": float(spearmanr(df.rx, df.ry)[0]), "ci": [float(lo), float(hi)], "n": int(len(df)), "students": int(len(ug))}


# ---------------------------------------------------------------- monotone TSK
def pairs_for(signs):
    out = []
    for i, s in enumerate(signs):
        for j, tj in enumerate(T.RULES):
            if tj[i] == "LOW":
                k = T.RULES.index(tuple("HIGH" if q == i else tj[q] for q in range(3)))
                out.append((j, k, s))          # s * (z_k - z_j) >= 0
    return out


def fit_monotone(cf, Xtr, ytr, signs):
    W = cf.firing(Xtr); den = W.sum(1) + T.W0; A = W / den[:, None]; bv = T.W0 * T.Z0 / den
    prior = np.full(len(T.RULES), ytr.mean())
    obj = lambda z: np.sum((A @ z + bv - ytr) ** 2) + T.ALPHA * np.sum((z - prior) ** 2)
    jac = lambda z: 2 * A.T @ (A @ z + bv - ytr) + 2 * T.ALPHA * (z - prior)
    cons = [{"type": "ineq", "fun": (lambda z, j=j, k=k, s=s: s * (z[k] - z[j]))} for j, k, s in pairs_for(signs)]
    r = minimize(obj, np.clip(cf.z, 0, 1), jac=jac, bounds=[(0, 1)] * len(T.RULES), constraints=cons, method="SLSQP",
                 options={"maxiter": 500, "ftol": 1e-10})
    return r.x


def violations(z, signs, tol=1e-6):
    return int(sum(s * (z[k] - z[j]) < -tol for j, k, s in pairs_for(signs)))


def sc_parts(tr, te, cols_override=None, with_len=False):
    cols = cols_override or S2.behav_cols(tr, with_len)
    sel, ranking = S2.sign_checked_selection(tr, cols)
    if len(sel) < 3:      # fewer than three eligible inputs: pad with the best-ranked ineligible ones (reported)
        PADDED.append([r["feature"] for r in ranking if r["feature"] not in sel][: 3 - len(sel)])
        sel = sel + PADDED[-1]
    sign = {r["feature"]: (1 if r["auc"] > 0.5 else -1) for r in ranking}
    Xtr, Xte = S2.X_of(tr, sel), S2.X_of(te, sel)
    cf = T.CFTSK().fit(Xtr, tr.y.values)
    return sel, [sign[s] for s in sel], cf, Xtr, Xte, ranking


def study_block(tr, te, struct, markov_col, prop_col, extra=None):
    y, g = te.y.values, te.student_id.astype(str).values
    sel, signs, cf, Xtr, Xte, ranking = sc_parts(tr, te)
    s_sc = cf.score(Xte)[0]
    out = {"selected": sel, "signs": signs, "n_train": int(len(tr)), "pos_train": int(tr.y.sum()),
           "n_test": int(len(te)), "pos_test": int(te.y.sum())}
    # M6 support, monotone
    W = cf.firing(Xtr)
    out["cell_support"] = [int((W[:, j] >= 0.5).sum()) for j in range(len(T.RULES))]
    out["z_sc"] = cf.z.tolist(); out["viol_sc"] = violations(cf.z, signs)
    zm = fit_monotone(cf, Xtr, tr.y.values, signs); out["z_sm"] = zm.tolist(); out["viol_sm"] = violations(zm, signs)
    cm = T.CFTSK(); cm.a, cm.b, cm.z = cf.a, cf.b, zm
    s_sm = cm.score(Xte)[0]
    out["auc_sc"] = auc_ci(y, s_sc, g); out["auc_sm"] = auc_ci(y, s_sm, g); out["sm_minus_sc"] = diff_ci(y, s_sm, s_sc, g)
    out["degenerate"] = {s: bool(cf.b[i] - cf.a[i] <= 1.0001e-3) for i, s in enumerate(sel)}
    # M3 persistence
    out["auc_markov"] = auc_ci(y, te[markov_col].values, g); out["sc_minus_markov"] = diff_ci(y, s_sc, te[markov_col].values, g)
    # M2 propensity and within-student
    out["auc_propensity"] = auc_ci(y, te[prop_col].values, g); out["sc_minus_propensity"] = diff_ci(y, s_sc, te[prop_col].values, g)
    out["within_student"] = {s: {"train": within_rho(S2.X_of(tr, [s])[:, 0], tr.y.values, tr.student_id.astype(str).values),
                                 "test": within_rho(S2.X_of(te, [s])[:, 0], y, g)} for s in sel}
    out["within_student_sc_score"] = within_rho(s_sc, y, g)
    # M4 overlap with structure
    out["overlap"] = {s: {v: float(spearmanr(tr[s], tr[v])[0]) for v in struct if tr[v].nunique() > 1} for s in sel}
    dup = [s for s in sel if any(abs(r) >= 0.7 for r in out["overlap"][s].values())]
    out["duplicates"] = dup
    if dup:
        cols = [c for c in S2.behav_cols(tr) if c not in dup]
        sel2, _, cf2, Xtr2, Xte2, _ = sc_parts(tr, te, cols_override=cols)
        S_tr, S_te = tr[struct].astype(float).values, te[struct].astype(float).values
        st = lr().fit(S_tr, tr.y.values).predict_proba(S_te)[:, 1]
        cb = lr().fit(np.column_stack([S_tr, cf2.score(Xtr2)[0]]), tr.y.values).predict_proba(np.column_stack([S_te, cf2.score(Xte2)[0]]))[:, 1]
        out["H2_without_duplicates"] = {"selected": sel2, "gain": diff_ci(y, cb, st, g), "auc_sc": auc_ci(y, cf2.score(Xte2)[0], g)}
    # minor 9 sensitivity
    sens = {}
    for w0 in (0.01, 0.05, 0.2):
        for al in (0.1, 1.0, 10.0):
            T.W0, T.ALPHA = w0, al
            c2 = T.CFTSK().fit(Xtr, tr.y.values); sens[f"w0={w0},alpha={al}"] = float(roc_auc_score(y, c2.score(Xte)[0]))
    T.W0, T.ALPHA = 0.05, 1.0
    out["sensitivity"] = sens
    if extra:
        out.update(extra(tr, te, y, g, s_sc))
    return out


# ---------------------------------------------------------------- data
def tutortrace(root):
    D = {d: S2.load(root, d) for d in (1, 2, 3)}
    for d, df in D.items():
        df.sort_values(["student_id", "query_index"], inplace=True)
        df["prev_dependent"] = df.groupby("student_id").y.shift(1).fillna(0.5)
        cs = df.groupby("student_id").y.cumsum() - df.y; n = df.groupby("student_id").cumcount()
        df["prop_earlier"] = np.where(n > 0, cs / n.replace(0, 1), 0.5)
        D[d] = df.reset_index(drop=True)
    return D


def recipe(csv):
    f = S3.build(csv)
    f["sess_order"] = f.week * 100 + f.session
    prop = []
    for _, r in f.iterrows():
        m = (f.student_id == r.student_id) & (f.course == r.course) & (f.sess_order < r.sess_order)
        prop.append(float(f.y[m].mean()) if m.any() else 0.5)
    f["prop_earlier_sessions"] = prop
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutortrace", required=True); ap.add_argument("--recipe", required=True)
    ap.add_argument("--units", required=True); ap.add_argument("--out", default="results/r2_stats.json")
    a = ap.parse_args()
    res = {}
    # ---------- Study 2
    D = tutortrace(a.tutortrace)
    S2.STRUCT = ["is_first_query", "query_index", "time_since_session_start_s", "test_progress"]
    res["S2"] = {"usable_structure": S2.STRUCT}
    for name, tr, te in (("D1_to_D2", D[1], D[2]), ("D2_to_D1", D[2], D[1]),
                         ("D12_to_D3", pd.concat([D[1], D[2]], ignore_index=True), D[3])):
        res["S2"][name] = study_block(tr, te, S2.STRUCT, "prev_dependent", "prop_earlier")
        print("S2", name, "done", flush=True)
    # ---------- Study 3
    F = recipe(a.recipe)
    S2.STRUCT = S3.STRUCT
    S2.behav_cols = lambda df, with_len=False: S3.BEHAV + (["query_length_chars"] if with_len else [])
    sw, oth = F[F.course == "SW"].reset_index(drop=True), F[F.course != "SW"].reset_index(drop=True)
    RESTR = ["idx", "essay_len_start"]

    def extra3(tr, te, y, g, s_sc):
        o = {}
        st = lr().fit(tr[RESTR].values, tr.y.values).predict_proba(te[RESTR].values)[:, 1]
        o["auc_structure_restricted"] = auc_ci(y, st, g)
        sel, _, cf, Xtr, Xte, _ = sc_parts(tr, te)
        cb = lr().fit(np.column_stack([tr[RESTR].values, cf.score(Xtr)[0]]), tr.y.values).predict_proba(
            np.column_stack([te[RESTR].values, cf.score(Xte)[0]]))[:, 1]
        o["gain_over_restricted"] = diff_ci(y, cb, st, g)
        cols = [c for c in S3.BEHAV if c != "prev_production_share"]
        sel2, _, cf2, _, Xte2, _ = sc_parts(tr, te, cols_override=cols)
        o["without_lagged_label"] = {"selected": sel2, "auc": auc_ci(y, cf2.score(Xte2)[0], g)}
        return o

    res["S3"] = {"usable_structure_cross_course": RESTR, "full_structure": S3.STRUCT}
    for name, tr, te in (("SW_to_AW_IRW", sw, oth), ("AW_IRW_to_SW", oth, sw)):
        res["S3"][name] = study_block(tr, te, S3.STRUCT, "prev_request_production", "prop_earlier_sessions", extra3)
        print("S3", name, "done", flush=True)
    # within-course CV
    wc = {}
    for course, df in list(F.groupby("course")) + [("ALL", F)]:
        df = df.reset_index(drop=True)
        if df.y.sum() < 20:
            continue
        oof = {k: np.full(len(df), np.nan) for k in ("structure", "sc_tsk", "structure_plus_sc")}
        stc = [c for c in S3.STRUCT if df[c].nunique() > 1]
        for trn, tst in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df, df.y, df.student_id):
            tr, te = df.iloc[trn].reset_index(drop=True), df.iloc[tst].reset_index(drop=True)
            sel, _, cf, Xtr, Xte, _ = sc_parts(tr, te)
            m = lr().fit(tr[stc].values, tr.y.values); oof["structure"][tst] = m.predict_proba(te[stc].values)[:, 1]
            oof["sc_tsk"][tst] = cf.score(Xte)[0]
            m2 = lr().fit(np.column_stack([tr[stc].values, cf.score(Xtr)[0]]), tr.y.values)
            oof["structure_plus_sc"][tst] = m2.predict_proba(np.column_stack([te[stc].values, cf.score(Xte)[0]]))[:, 1]
        y, g = df.y.values, df.student_id.astype(str).values
        wc[course] = {"n": int(len(df)), "pos": int(y.sum()), "structure_vars": stc,
                      **{k: auc_ci(y, v, g) for k, v in oof.items()},
                      "gain": diff_ci(y, oof["structure_plus_sc"], oof["structure"], g)}
        print("S3 within-course", course, "done", flush=True)
    res["S3"]["within_course_cv"] = wc
    # ---------- Study 1 equivalence and counts
    U = pickle.load(open(a.units, "rb"))
    import run_revision_b as RB
    y = np.array([u["y"] for u in U]); g = np.array([u["group_key"] for u in U])
    cell = np.array([u["group_key"].rsplit("_", 1)[1] + u["group"] for u in U]); ry = RB.within_ranks(y, cell)
    C = np.log1p(T.counts(U, ["writing_request", "conceptual_questions", "provide_context"]))
    X = {"v": np.array([u["v"] for u in U]), "u": np.array([u["u"] for u in U]), "p": np.array([u["p"] for u in U]),
         "e": np.array([u["e"] for u in U]), "log_W": C[:, 0], "log_C": C[:, 1], "log_P": C[:, 2]}
    ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
    boots = [np.concatenate([idx[k] for k in rng0.choice(ug, len(ug))]) for _ in range(2000)]
    eq = {}
    for k, x in X.items():
        rx = RB.within_ranks(x, cell); bs = [spearmanr(rx[t], ry[t])[0] for t in boots]
        lo, hi = np.nanpercentile(bs, [5, 95])
        eq[k] = {"rho": float(spearmanr(rx, ry)[0]), "ci90": [float(lo), float(hi)], "equivalent": bool(-0.1 < lo and hi < 0.1)}
    thr = np.quantile(y[np.char.endswith(g.astype(str), "f24")], 0.75)
    counts = {}
    for c in np.unique(cell):
        m = cell == c; counts[c] = {"n": int(m.sum()), "pos_f24_threshold": int((y[m] >= thr).sum()), "zeros": int((y[m] == 0).sum())}
    res["S1"] = {"equivalence_sesoi_0.1": eq, "cell_counts": counts, "f24_threshold": float(thr),
                 "pooled_threshold": float(np.quantile(y, 0.75)), "share_zero_overall": float((y == 0).mean())}
    res["padded_selections"] = PADDED
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("written", a.out)


if __name__ == "__main__":
    main()
