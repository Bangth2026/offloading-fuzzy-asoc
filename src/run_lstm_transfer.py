"""
run_lstm_transfer.py -- E12, LSTM encoder with and without the CF-TSK constraint,
cross-semester (PREREG_E12.md). Uses the E11 rule base of each training semester.

    python src/run_lstm_transfer.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle
import numpy as np, torch, torch.nn as nn
from scipy.stats import wilcoxon
import run_fuzzy_experiments as R
import run_transfer as T
from reproduce_benchmark import SeqLSTM
from oc_tsk import interval_implication_loss


def train_eval(units, tr, te, yb_tr, yb_te, seed, W_tr, z, n_da, constrained, epochs):
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
        p = m(*R.pad(units, te)).numpy()
    return float(T.auc(yb_te, p))


def direction(units, nested, seeds, epochs, n_da):
    tr_sem, te_sem, sel = nested["train"], nested["test"], nested["selected"]
    sem = np.array([u["group_key"].rsplit("_", 1)[1] for u in units])
    tr, te = np.where(sem == tr_sem)[0], np.where(sem == te_sem)[0]
    y = np.array([u["y"] for u in units]); thr = nested["threshold_y"]
    yb_tr, yb_te = (y[tr] >= thr).astype(int), (y[te] >= thr).astype(int)
    X3 = np.log1p(T.counts(units, sel))
    cf = T.CFTSK().fit(X3[tr], yb_tr)
    assert np.allclose(cf.z, nested["frozen"]["consequents"]), "rule base differs from E11"
    W_tr = cf.score(X3[tr])[1]
    a0, a1 = [], []
    for s in range(seeds):
        a0.append(train_eval(units, tr, te, yb_tr, yb_te, s, W_tr, cf.z, n_da, False, epochs))
        a1.append(train_eval(units, tr, te, yb_tr, yb_te, s, W_tr, cf.z, n_da, True, epochs))
        print(tr_sem, "->", te_sem, s, round(a0[-1], 3), round(a1[-1], 3), flush=True)
    d = np.array(a1) - np.array(a0)
    p = float(wilcoxon(a1, a0).pvalue) if np.any(d != 0) else 1.0
    return {"lstm": a0, "lstm_cf": a1, "mean_lstm": float(np.mean(a0)), "sd_lstm": float(np.std(a0, ddof=1)),
            "mean_lstm_cf": float(np.mean(a1)), "sd_lstm_cf": float(np.std(a1, ddof=1)),
            "mean_diff": float(d.mean()), "n_pos": int((d > 0).sum()), "wilcoxon_p": p,
            "H5": bool(p < 0.05 and d.mean() > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/lstm_transfer_stats.json")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); n_da = R.set_n_da(units)
    st = json.load(open(a.stats)); N = st["E11_nested"]
    res = {k: direction(units, N[k], a.seeds, a.epochs, n_da) for k in ("primary", "reverse")}
    json.dump(res, open(a.out, "w"), indent=1)
    st["E12_lstm"] = res; json.dump(st, open(a.stats, "w"), indent=1)
    for k, v in res.items():
        print(k, {x: v[x] for x in ("mean_lstm", "sd_lstm", "mean_lstm_cf", "sd_lstm_cf", "mean_diff", "n_pos", "wilcoxon_p", "H5")})


if __name__ == "__main__":
    main()
