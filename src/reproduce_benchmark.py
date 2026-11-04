"""
reproduce_benchmark.py -- E0: benchmark (Table 3), LR seed stability, permutation test,
dialogue-act ablation (Section 7.5) and model complementarity (Section 7.7), all on the
pooled 5-fold group-aware protocol of Section 6.1. Writes results/benchmark_stats.json,
results/fig3_benchmark.png and results/fig4_ablation_permutation.png, and merges the
statistics into the main statistics file under "E0_benchmark".

This script re-implements, inside the code archive, the analyses that the earlier
version of the paper ran with separate scripts, so that every number in Sections 7.2,
7.5 and 7.7 can be regenerated from the archive.

    python src/reproduce_benchmark.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, os, pickle
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import run_fuzzy_experiments as R
from run_followup import BROAD

SEED = 42


class SeqLSTM(nn.Module):
    def __init__(self, n_da, use_da=True, d=32):
        super().__init__()
        self.use_da = use_da
        self.emb = nn.Embedding(n_da + 1, d - 8, padding_idx=n_da) if use_da else None
        self.num = nn.Linear(4, 8 if use_da else d)
        self.rnn = nn.LSTM(d, d, batch_first=True)
        self.head = nn.Linear(d, 1)

    def forward(self, da, num, mask):
        x = torch.cat([self.emb(da), self.num(num)], -1) if self.use_da else self.num(num)
        h, _ = self.rnn(x)
        last = (~mask).sum(1).clamp(min=1) - 1
        return torch.sigmoid(self.head(h[torch.arange(len(h)), last])).squeeze(-1)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def folds(n, yb, g, seed):
    return [(tr, te) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), yb, g)
            if len(set(yb[tr])) == 2 and len(set(yb[te])) == 2]


def cv_tabular(X, yb, g, seed=SEED):
    oof = np.full(len(yb), np.nan); aucs = []
    for tr, te in folds(len(yb), yb, g, seed):
        p = lr().fit(X[tr], yb[tr]).predict_proba(X[te])[:, 1]
        oof[te] = p; aucs.append(roc_auc_score(yb[te], p))
    return float(np.mean(aucs)), oof


def cv_seq(units, yb, g, make, seed=SEED, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    oof = np.full(len(yb), np.nan); aucs = []
    for tr, te in folds(len(yb), yb, g, seed):
        m = make(); opt = torch.optim.Adam(m.parameters(), 1e-3)
        yt = torch.tensor(yb[tr], dtype=torch.float)
        for _ in range(epochs):
            m.train()
            for b in np.array_split(np.random.permutation(len(tr)), max(1, len(tr) // 32)):
                loss = nn.functional.binary_cross_entropy(m(*R.pad(units, tr[b])), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            p = m(*R.pad(units, te)).numpy()
        oof[te] = p; aucs.append(roc_auc_score(yb[te], p))
    return float(np.mean(aucs)), oof


def pct(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 1) / (len(x) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default="results/benchmark_stats.json"); ap.add_argument("--figdir", default="results")
    ap.add_argument("--perms", type=int, default=500); ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    units = pickle.load(open(a.units, "rb")); n_da = R.set_n_da(units)
    y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
    yb = (y >= np.quantile(y, 0.75)).astype(int)
    C8 = np.array([[u["broad"].count(b) for b in BROAD] for u in units], float)
    CS = np.zeros((len(units), n_da))
    for i, u in enumerate(units):
        CS[i] = np.bincount(u["da"], minlength=n_da)
    n_spec_used = int((CS.sum(0) > 0).sum())
    res = {"n_specific_labels": n_spec_used, "seed": SEED}
    rows = {}
    rows["lr_broad"] = cv_tabular(C8, yb, g)
    rows["lr_specific"] = cv_tabular(CS, yb, g)
    rows["lstm"] = cv_seq(units, yb, g, lambda: SeqLSTM(n_da, True), epochs=a.epochs)
    rows["dkt"] = cv_seq(units, yb, g, lambda: SeqLSTM(n_da, False), epochs=a.epochs)
    rows["transformer"] = cv_seq(units, yb, g, R.make_model, epochs=a.epochs)
    res["table3"] = {k: {"auc": v[0], "r": float(pearsonr(v[1][~np.isnan(v[1])], y[~np.isnan(v[1])])[0])}
                     for k, v in rows.items()}
    print(json.dumps(res["table3"], indent=1), flush=True)

    s = [cv_tabular(C8, yb, g, seed)[0] for seed in range(15)]
    res["lr_seeds"] = {"mean": float(np.mean(s)), "sd": float(np.std(s, ddof=1)), "min": float(min(s)), "max": float(max(s))}

    obs = rows["lr_broad"][0]; rng = np.random.default_rng(0); null = []
    for _ in range(a.perms):
        null.append(cv_tabular(C8, rng.permutation(yb), g)[0])
    null = np.array(null)
    res["permutation"] = {"observed": obs, "n": a.perms, "null_mean": float(null.mean()), "null_max": float(null.max()),
                          "p": float((1 + (null >= obs).sum()) / (1 + a.perms)), "null": null.tolist()}

    res["ablation"] = {b: {"auc": cv_tabular(np.delete(C8, k, 1), yb, g)[0]} for k, b in enumerate(BROAD)}
    for b in BROAD:
        res["ablation"][b]["drop"] = obs - res["ablation"][b]["auc"]

    lo, to = rows["lr_broad"][1], rows["transformer"][1]
    ok = ~np.isnan(lo) & ~np.isnan(to)
    tq = lambda x: x >= np.quantile(x[ok], 0.75)
    fl, ft = tq(lo) & ok, tq(to) & ok
    t_only = ft & ~fl & (yb == 1); l_only = fl & ~ft & (yb == 1)
    nturn = np.array([len(u["da"]) for u in units]); va = np.array([u["num"][:, 1].mean() for u in units])
    ur = np.array([u["u"] for u in units])
    fus = np.full(len(yb), np.nan); fus[ok] = (pct(lo[ok]) + pct(to[ok])) / 2
    rec = {}
    for k in (10, 15, 20, 25, 30):
        def recall(sc):
            thr = np.quantile(sc[ok], 1 - k / 100); return float(((sc >= thr) & ok & (yb == 1)).sum() / (yb[ok] == 1).sum())
        rec[str(k)] = {"lr": recall(lo), "fusion": recall(fus), "gain": recall(fus) - recall(lo)}
    res["complementarity"] = {"spearman": float(spearmanr(lo[ok], to[ok])[0]), "pearson": float(pearsonr(lo[ok], to[ok])[0]),
                              "transformer_only_tp": int(t_only.sum()), "lr_only_tp": int(l_only.sum()),
                              "t_only_turns": float(nturn[t_only].mean()), "all_turns": float(nturn[ok].mean()),
                              "t_only_unchecked_share": float(va[t_only].mean()), "all_unchecked_share": float(va[ok].mean()),
                              "t_only_surface": float(ur[t_only].mean()), "all_surface": float(ur[ok].mean()),
                              "recall_at_budget": rec}
    json.dump(res, open(a.out, "w"), indent=1)
    st = json.load(open(a.stats)); st["E0_benchmark"] = {k: v for k, v in res.items()}
    st["E0_benchmark"]["permutation"] = {k: v for k, v in res["permutation"].items() if k != "null"}
    json.dump(st, open(a.stats, "w"), indent=1)
    figures(res, a.figdir)
    print(json.dumps({k: v for k, v in res.items() if k not in ("table3", "permutation")}, indent=1))
    print({k: v for k, v in res["permutation"].items() if k != "null"})


def figures(res, figdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#5b636e", "#d9dde3"
    plt.rcParams.update({"font.size": 9})
    names = [("lr_broad", "LogReg\n(broad DA)"), ("lr_specific", "LogReg\n(specific DA)"), ("lstm", "LSTM"),
             ("dkt", "DKT-style"), ("transformer", "Transformer\n(λ = 0)")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, key, lab in ((ax[0], "auc", "AUC (top-quartile)"), (ax[1], "r", "Pearson r with y")):
        v = [res["table3"][k][key] for k, _ in names]
        a.bar(range(len(v)), v, color=BLUE, width=0.6)
        for i, x in enumerate(v):
            a.text(i, x + 0.005, f"{x:.3f}", ha="center", fontsize=8, color=INK)
        a.set_xticks(range(len(v)), [n for _, n in names]); a.set_ylabel(lab)
        a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].axhline(0.5, color=MUTED, ls="--", lw=1); ax[0].set_ylim(0.4, max(0.72, max(res["table3"][k]["auc"] for k, _ in names) + 0.03))
    ax[0].set_title("(a) Classification performance", loc="left"); ax[1].set_title("(b) Correlation with continuous target", loc="left")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "fig3_benchmark.png"), dpi=300); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ab = sorted(res["ablation"].items(), key=lambda kv: kv[1]["drop"])
    ax[0].barh(range(len(ab)), [d["drop"] for _, d in ab], color=[ORANGE if d["drop"] > 0 else MUTED for _, d in ab], height=0.6)
    ax[0].set_yticks(range(len(ab)), [k for k, _ in ab]); ax[0].axvline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel("AUC drop when the category is removed"); ax[0].set_title("(a) Ablation: broad dialogue-act features", loc="left")
    p = res["permutation"]
    ax[1].hist(p["null"], bins=30, color=BLUE, alpha=0.85)
    ax[1].axvline(p["observed"], color=ORANGE, lw=2, label=f"observed {p['observed']:.3f}")
    ax[1].set_xlabel("AUC under shuffled labels"); ax[1].set_ylabel("count")
    ax[1].set_title(f"(b) Permutation test ({p['n']} shuffles, p = {p['p']:.3f})", loc="left"); ax[1].legend(frameon=False)
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "fig4_ablation_permutation.png"), dpi=300); plt.close()


if __name__ == "__main__":
    main()
