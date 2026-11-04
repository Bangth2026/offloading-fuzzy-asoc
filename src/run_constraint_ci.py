"""
run_constraint_ci.py -- R1-E (PREREG_R1.md): re-run E10 (Transformer) and E12 (LSTM) with the same
seeds, keep held-out predictions, check that per-seed AUCs reproduce the reported ones, and give a
student-cluster bootstrap interval for the AUC difference of seed-averaged predictions.

    python src/run_constraint_ci.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle
import numpy as np, torch, torch.nn as nn
import run_fuzzy_experiments as R
import run_transfer as T
from reproduce_benchmark import SeqLSTM
from oc_tsk import interval_implication_loss


def lstm_pred(units, tr, te, yb_tr, seed, W_tr, z, n_da, constrained, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SeqLSTM(n_da, True); opt = torch.optim.Adam(m.parameters(), 1e-3)
    yt = torch.tensor(yb_tr, dtype=torch.float)
    for _ in range(epochs):
        m.train()
        for b in np.array_split(np.random.permutation(len(tr)), max(1, len(tr) // 32)):
            r = m(*R.pad(units, tr[b]))
            loss = nn.functional.binary_cross_entropy(r, yt[b])
            if constrained:
                loss = loss + interval_implication_loss(r, torch.tensor(W_tr[b], dtype=torch.float), z)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(*R.pad(units, te)).numpy()


def split(units, tr_sem, te_sem, thr):
    sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
    tr, te = np.where(sem == tr_sem)[0], np.where(sem == te_sem)[0]
    y = np.array([u["y"] for u in units])
    return tr, te, (y[tr] >= thr).astype(int), (y[te] >= thr).astype(int)


def summarise(p0, p1, yb_te, gt):
    a0, a1 = np.mean(p0, 0), np.mean(p1, 0)
    T.CONF = 95.0
    d = float(T.auc(yb_te, a1) - T.auc(yb_te, a0))
    lo, hi = T.boot(yb_te, gt, lambda t: T.auc(yb_te, a1, t) - T.auc(yb_te, a0, t))
    return {"auc_avg_none": float(T.auc(yb_te, a0)), "auc_avg_constrained": float(T.auc(yb_te, a1)),
            "diff": d, "ci": [lo, hi], "negligible": bool(-0.01 < lo and hi < 0.01)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/constraint_ci.json"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); n_da = R.set_n_da(units); st = json.load(open(a.stats))
    y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
    out = {}
    for key, (s1, s2) in (("primary", ("f24", "s25")), ("reverse", ("s25", "f24"))):
        sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
        thr = float(np.quantile(y[sem == s1], 0.75))
        tr, te, yb_tr, yb_te = split(units, s1, s2, thr)
        # E10 Transformer with the pooled-input CF-TSK
        X3 = np.log1p(T.counts(units, T.CF_INPUTS)); cf = T.CFTSK().fit(X3[tr], yb_tr); W_tr = cf.score(X3[tr])[1]
        p0, p1, a0, a1 = [], [], [], []
        for sd in range(a.seeds):
            for kind, P, A in (("none", p0, a0), ("cf_imp", p1, a1)):
                m = T.train_tf(units, tr, yb_tr, sd, kind, W_tr, cf.z, a.epochs)
                with torch.no_grad():
                    p = m(*R.pad(units, te)).numpy()
                P.append(p); A.append(float(T.auc(yb_te, p)))
        rep = st["E10_transfer"][key]["H3"]
        tf = summarise(p0, p1, yb_te, g[te])
        tf["reproduces"] = bool(np.allclose(a0, rep["none"], atol=1e-6) and np.allclose(a1, rep["cf_imp"], atol=1e-6))
        # E12 LSTM with the E11 rule base
        N = st["E11_nested"][key]; Xn = np.log1p(T.counts(units, N["selected"]))
        cfn = T.CFTSK().fit(Xn[tr], yb_tr); Wn = cfn.score(Xn[tr])[1]
        q0 = [lstm_pred(units, tr, te, yb_tr, s, Wn, cfn.z, n_da, False, a.epochs) for s in range(a.seeds)]
        q1 = [lstm_pred(units, tr, te, yb_tr, s, Wn, cfn.z, n_da, True, a.epochs) for s in range(a.seeds)]
        rep2 = st["E12_lstm"][key]
        ls = summarise(q0, q1, yb_te, g[te])
        ls["reproduces"] = bool(np.allclose([T.auc(yb_te, p) for p in q0], rep2["lstm"], atol=1e-6)
                                and np.allclose([T.auc(yb_te, p) for p in q1], rep2["lstm_cf"], atol=1e-6))
        out[key] = {"transformer": tf, "lstm": ls}
        print(key, json.dumps(out[key]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    st = json.load(open(a.stats)); st["R1E"] = out; json.dump(st, open(a.stats, "w"), indent=1)


if __name__ == "__main__":
    main()
