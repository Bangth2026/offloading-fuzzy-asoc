"""
run_s2.py -- Study 2 on TutorTrace, as pre-registered in study2/PREREG_S2_tutortrace.md.

    python study2/run_s2.py --data /path/to/TutorTrace_dataset_and_benchmark --out study2/results_s2.json
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import run_transfer as T
from run_revision import anfis_scores, wang_mendel

warnings.filterwarnings("ignore")
STRUCT = ["is_first_query", "query_index", "time_since_session_start_s", "test_progress"]
BEHAV_EXTRA = ["thinking_time_s", "implementing_time_s", "debugging_time_s", "testing_time_s", "seeking_help_time_s",
               "thinking_task_s", "thinking_llm_s", "thinking_error_s", "thinking_code_s", "time_since_last_query_s"]


def load(root, d):
    q = pd.read_csv(os.path.join(root, "dataset/observable_metrics/query_level", f"deployment_{d}_queries.csv"))
    lab = pd.read_csv(os.path.join(root, "dataset/query_labels", f"deployment_{d}_labels.csv"))
    lab["query_type"] = lab["query_type"].astype(str).str.strip().str.lower()
    df = q.merge(lab[lab.query_type.isin(["guided", "dependent"])], on=["student_id", "query_index"], how="inner")
    df["y"] = (df.query_type == "dependent").astype(int)
    df["test_progress"] = (df.test_passed_at_query / df.test_total_at_query.replace(0, np.nan)).fillna(0)
    df["is_first_query"] = df.is_first_query.astype(float)
    df["deployment"] = d
    return df


def behav_cols(df, with_len=False):
    cols = [c for c in df.columns if c.startswith("pre_")] + [c for c in BEHAV_EXTRA if c in df.columns]
    return cols + (["query_length_chars"] if with_len else [])


def X_of(df, cols):
    X = df[cols].astype(float).fillna(0.0).values
    return np.where(X >= 0, np.log1p(np.clip(X, 0, None)), -np.log1p(-X))   # signed log; most are non-negative


def sign_checked_selection(tr, cols, k=3, seed=0):
    X, y = X_of(tr, cols), tr.y.values
    rng = np.random.default_rng(seed); stu = tr.student_id.unique(); rng.shuffle(stu)
    half = tr.student_id.isin(stu[: len(stu) // 2]).values
    rows = []
    for j, c in enumerate(cols):
        if np.unique(X[:, j]).size < 2:
            continue
        a = roc_auc_score(y, X[:, j])
        a1 = roc_auc_score(y[half], X[half, j]) if len(set(y[half])) == 2 else 0.5
        a2 = roc_auc_score(y[~half], X[~half, j]) if len(set(y[~half])) == 2 else 0.5
        ok = np.sign(a1 - 0.5) == np.sign(a2 - 0.5) and a1 != 0.5
        rows.append(dict(feature=c, auc=float(a), auc_half1=float(a1), auc_half2=float(a2), eligible=bool(ok)))
    rows.sort(key=lambda r: -abs(r["auc"] - 0.5))
    sel = [r["feature"] for r in rows if r["eligible"]][:k]
    return sel, rows


def boot(te, fn, conf):
    T.CONF = conf
    return T.boot(te.y.values, te.student_id.astype(str).values, fn)


def evaluate(tr, te, with_len=False, conf_h=100 * (1 - 0.05 / 3)):
    cols = behav_cols(tr, with_len)
    sel, ranking = sign_checked_selection(tr, cols)
    Xtr, Xte = X_of(tr, sel), X_of(te, sel)
    ytr, yte = tr.y.values, te.y.values
    cf = T.CFTSK().fit(Xtr, ytr)
    s = {"sc_tsk": cf.score(Xte)[0]}
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    S_tr, S_te = tr[STRUCT].astype(float).values, te[STRUCT].astype(float).values
    st = lr().fit(S_tr, ytr); s["structure_only"] = st.predict_proba(S_te)[:, 1]
    comb = lr().fit(np.column_stack([S_tr, cf.score(Xtr)[0]]), ytr)
    s["structure_plus_sc_tsk"] = comb.predict_proba(np.column_stack([S_te, s["sc_tsk"]]))[:, 1]
    Ba, Bb = X_of(tr, cols), X_of(te, cols)
    s["lr_all_behaviour"] = lr().fit(Ba, ytr).predict_proba(Bb)[:, 1]
    s["lr3"] = lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    s["tree_d2"] = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    s["wang_mendel"] = wang_mendel(Xtr, ytr, Xte)[0]
    s["anfis"] = anfis_scores(Xtr, ytr, Xte)
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        s["ebm"] = ExplainableBoostingClassifier(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    except Exception as e:
        print("EBM not run:", e)
    A = lambda v, t=None: T.auc(yte, v, t)
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "train_dependent_rate": float(ytr.mean()),
           "test_dependent_rate": float(yte.mean()), "test_students": int(te.student_id.nunique()),
           "selected": sel, "ranking_top10": ranking[:10],
           "consequents": [{"rule": " and ".join(f"{c} {t}" for c, t in zip(sel, terms)), "z": float(z)}
                           for terms, z in zip(T.RULES, cf.z)],
           "breakpoints": {"low": cf.a.tolist(), "high": cf.b.tolist()},
           "auc": {k: [float(A(v)), boot(te, lambda t, v=v: A(v, t), 95.0)] for k, v in s.items()}}
    d2 = lambda a, b, conf: [float(A(s[a]) - A(s[b])), boot(te, lambda t: A(s[a], t) - A(s[b], t), conf)]
    out["H1"] = [float(A(s["sc_tsk"])), boot(te, lambda t: A(s["sc_tsk"], t), conf_h)]
    out["H2"] = d2("structure_plus_sc_tsk", "structure_only", conf_h)
    out["H3"] = d2("sc_tsk", "lr_all_behaviour", conf_h)
    out["decisions"] = {"H1": out["H1"][1][0] > 0.5, "H2": out["H2"][1][0] > 0, "H3": out["H3"][1][0] > -0.05}
    out["cf_minus"] = {k: d2("sc_tsk", k, 95.0) for k in s if k != "sc_tsk"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="study2/results_s2.json")
    a = ap.parse_args()
    D = {d: load(a.data, d) for d in (1, 2, 3)}
    res = {"counts": {d: {"queries": int(len(v)), "students": int(v.student_id.nunique()), "dependent_rate": float(v.y.mean())}
                      for d, v in D.items()}}
    res["primary_D1_to_D2"] = evaluate(D[1], D[2])
    res["secondary_D2_to_D1"] = evaluate(D[2], D[1])
    res["secondary_D12_to_D3"] = evaluate(pd.concat([D[1], D[2]], ignore_index=True), D[3])
    res["with_query_length_D1_to_D2"] = evaluate(D[1], D[2], with_len=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for k in ("primary_D1_to_D2", "secondary_D2_to_D1", "secondary_D12_to_D3", "with_query_length_D1_to_D2"):
        o = res[k]
        print(k, "| sel", o["selected"], "| H", o["decisions"], "| H1", o["H1"], "| H2", o["H2"], "| H3", o["H3"])
        print("   ", {m: f"{v[0]:.3f} [{v[1][0]:.3f},{v[1][1]:.3f}]" for m, v in o["auc"].items()})


if __name__ == "__main__":
    main()
