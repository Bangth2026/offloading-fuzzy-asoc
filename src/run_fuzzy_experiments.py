"""
run_fuzzy_experiments.py  --  experiments E1-E5 for the ASOC revision.

Every number that the revised manuscript marks [TBD:...] is written to
paper_stats_fuzzy.json by this script; the figure files fig4_fuzzy.png and
fig5_fuzzy.png replace Figures 4 and 5 of the EAAI version.

USAGE
  python run_fuzzy_experiments.py --demo            # synthetic smoke test
  python run_fuzzy_experiments.py --units units.pkl # real run

units.pkl = list of dicts, one per (student, semester, assessment-group) unit,
produced from run_real_analysis_v2.py (Algorithms 1-2). Required keys:
  'group_key' : str   student + '_' + semester  (CV grouping, Algorithm 3)
  'da'        : int array (n_turns,)   specific-DA index 0..30
  'num'       : float array (n_turns, 4) latency(norm), verbatim, self_attempt, surface_dep
  'y'         : float  continuous weak target (Section 3)
  'v','u','p','e' : float  unit-level rates (verbatim, surface-dependency,
                           provide-context, verification)
Optional: 'lr_oof' (logistic-regression out-of-fold risk) for E5.

If you prefer to reuse TransformerRisk from run_real_analysis.py so that the
lambda=0 condition reproduces Table 2 exactly, replace make_model() below.
"""
from __future__ import annotations
import argparse, json, pickle, itertools
import numpy as np
import torch, torch.nn as nn
from scipy.stats import wilcoxon, ttest_rel, spearmanr, pearsonr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from fuzzy_rules import FuzzyRuleBase, linear_signal, rule_loss, RULES

N_DA, MAXLEN = 31, 200


# ------------------------------------------------------------------ model
class TinyTransformer(nn.Module):
    """2 layers, 2 heads, d=32, [ASSIGN] pooling token (Table 2 configuration)."""
    def __init__(self, d=32, heads=2, layers=2, n_num=4):
        super().__init__()
        self.emb = nn.Embedding(N_DA + 1, d - 8, padding_idx=N_DA)
        self.num = nn.Linear(n_num, 8)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Embedding(MAXLEN + 1, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, da, num, mask):
        x = torch.cat([self.emb(da), self.num(num)], -1)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        x = x + self.pos(torch.arange(x.size(1), device=x.device))
        m = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool), mask], 1)
        return torch.sigmoid(self.head(self.enc(x, src_key_padding_mask=m)[:, 0])).squeeze(-1)


def make_model():
    return TinyTransformer()


def pad(units, idx):
    L = min(MAXLEN, max(len(units[i]["da"]) for i in idx))
    da = torch.full((len(idx), L), N_DA, dtype=torch.long)
    num = torch.zeros(len(idx), L, 4)
    mask = torch.ones(len(idx), L, dtype=torch.bool)
    for b, i in enumerate(idx):
        d = units[i]["da"][-L:]; n = units[i]["num"][-L:]
        da[b, :len(d)] = torch.as_tensor(d); num[b, :len(d)] = torch.as_tensor(n, dtype=torch.float)
        mask[b, :len(d)] = False
    return da, num, mask


def rule_inputs(units, idx):
    return {k: np.array([units[i][k] for i in idx]) for k in ("v", "u", "p", "e")}


# ------------------------------------------------------------------ one CV run
def cv_run(units, ybin, y, groups, seed, kind="none", lam=0.0, epochs=15,
           q=(0.25, 0.75)):
    torch.manual_seed(seed); np.random.seed(seed)
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    oof = np.full(len(units), np.nan)
    s_oof = {"linear": np.full(len(units), np.nan), "fuzzy": np.full(len(units), np.nan),
             "crisp": np.full(len(units), np.nan)}
    W_oof = np.full((len(units), len(RULES)), np.nan)
    viol = []
    aucs = []
    for tr, te in skf.split(np.zeros(len(units)), ybin, groups):
        if len(set(ybin[tr])) < 2:
            continue
        frb = FuzzyRuleBase(*q).fit(rule_inputs(units, tr))          # train fold only
        crb = FuzzyRuleBase(*q, mode="crisp").fit(rule_inputs(units, tr))
        s_tr, W_tr = frb.score(rule_inputs(units, tr))
        lin_tr = linear_signal(rule_inputs(units, tr))
        model = make_model(); opt = torch.optim.Adam(model.parameters(), 1e-3)
        yt = torch.tensor(ybin[tr], dtype=torch.float)
        for _ in range(epochs):
            model.train()
            for b in np.array_split(np.random.permutation(len(tr)), max(1, len(tr) // 32)):
                r = model(*pad(units, tr[b]))
                loss = nn.functional.binary_cross_entropy(r, yt[b])
                if lam > 0:
                    tgt = lin_tr[b] if kind == "linear_mse" else s_tr[b]
                    loss = loss + lam * rule_loss(r, kind, torch.tensor(tgt, dtype=torch.float),
                                                  torch.tensor(W_tr[b], dtype=torch.float))
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = model(*pad(units, te)).numpy()
        oof[te] = p
        aucs.append(roc_auc_score(ybin[te], p))
        s_te, W_te = frb.score(rule_inputs(units, te))
        s_oof["fuzzy"][te], W_oof[te] = s_te, W_te
        s_oof["crisp"][te] = crb.score(rule_inputs(units, te))[0]
        s_oof["linear"][te] = linear_signal(rule_inputs(units, te))
        with torch.no_grad():
            viol.append(float(rule_loss(torch.tensor(p), "implication",
                                        W=torch.tensor(W_te, dtype=torch.float))))
    ok = ~np.isnan(oof)
    return dict(auc=float(np.mean(aucs)), r=float(pearsonr(oof[ok], y[ok])[0]),
                oof=oof, s=s_oof, W=W_oof,
                cons_rho=float(spearmanr(oof[ok], s_oof["fuzzy"][ok])[0]),
                viol=float(np.mean(viol)))


# ------------------------------------------------------------------ experiments
def cluster_boot_auc_diff(ybin, a, b, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed); g = np.unique(groups)
    idx_by = {k: np.where(groups == k)[0] for k in g}
    d = []
    for _ in range(B):
        idx = np.concatenate([idx_by[k] for k in rng.choice(g, len(g))])
        if len(set(ybin[idx])) < 2:
            continue
        d.append(roc_auc_score(ybin[idx], a[idx]) - roc_auc_score(ybin[idx], b[idx]))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); m = len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i])); out[i] = run
    return out


def main(units, seeds=10, sweep_seeds=3, epochs=15, out_json="paper_stats_fuzzy.json", fig_dir="."):
    y = np.array([u["y"] for u in units]); groups = np.array([u["group_key"] for u in units])
    ybin = (y >= np.quantile(y, 0.75)).astype(int)
    stats = {"N_units": len(units), "N_students": int(len(set(groups))),
             "n_highrisk": int(ybin.sum()), "seeds": seeds, "epochs": epochs}

    # E1 accuracy, paired over seeds
    kinds = ["none", "linear_mse", "tsk_mse", "implication"]
    res = {k: [] for k in kinds}; res_r = {k: [] for k in kinds}; keep = {}
    for sd in range(seeds):
        for k in kinds:
            out = cv_run(units, ybin, y, groups, sd, k, 0.0 if k == "none" else 1.0, epochs)
            res[k].append(out["auc"]); res_r[k].append(out["r"])
            if sd == 0:
                keep[k] = out
    E1 = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v,
              "r_mean": float(np.mean(res_r[k]))}
          for k, v in res.items()}
    comps = [("implication", "none"), ("tsk_mse", "none"), ("linear_mse", "none"),
             ("implication", "linear_mse")]
    ps = []
    for a, b in comps:
        da = np.array(res[a]) - np.array(res[b])
        w = wilcoxon(res[a], res[b]).pvalue if np.any(da != 0) else 1.0
        t = ttest_rel(res[a], res[b]).pvalue
        E1[f"{a}_vs_{b}"] = {"mean_diff": float(da.mean()), "wilcoxon_p": float(w),
                            "ttest_p": float(t), "n_pos": int((da > 0).sum()),
                            "cohen_d": float(da.mean() / (da.std(ddof=1) + 1e-12))}
        ps.append(w)
    for (a, b), ph in zip(comps, holm(ps)):
        E1[f"{a}_vs_{b}"]["wilcoxon_p_holm"] = float(ph)
    stats["E1_accuracy"] = E1

    # E2 consistency sweep
    E2 = {}
    for k in ["linear_mse", "tsk_mse", "implication"]:
        for lam in [0.0, 0.3, 0.6, 1.0]:
            outs = [cv_run(units, ybin, y, groups, sd, k, lam, epochs) for sd in range(sweep_seeds)]
            E2[f"{k}@{lam}"] = {"spearman_pred_vs_fuzzy_s": float(np.mean([o["cons_rho"] for o in outs])),
                                "implication_violation": float(np.mean([o["viol"] for o in outs])),
                                "auc": float(np.mean([o["auc"] for o in outs]))}
    stats["E2_consistency"] = E2

    # E3 rule-signal quality on its own (out-of-fold, breakpoints from train folds)
    s = keep["none"]["s"]; ok = ~np.isnan(s["fuzzy"])
    E3 = {}
    for name in ["linear", "crisp", "fuzzy"]:
        E3[name] = {"auc": float(roc_auc_score(ybin[ok], s[name][ok])),
                    "spearman_y": float(spearmanr(s[name][ok], y[ok])[0])}
    E3["fuzzy_minus_linear_auc"] = cluster_boot_auc_diff(ybin[ok], s["fuzzy"][ok], s["linear"][ok], groups[ok])
    E3["fuzzy_minus_crisp_auc"] = cluster_boot_auc_diff(ybin[ok], s["fuzzy"][ok], s["crisp"][ok], groups[ok])
    stats["E3_signal"] = E3

    # E4 membership-breakpoint sensitivity (signal AUC, no retraining needed)
    E4 = {"fuzzy": {}, "crisp": {}}
    for ql, qh in itertools.product([0.1, 0.2, 0.3], [0.6, 0.7, 0.8, 0.9]):
        for mode in ["fuzzy", "crisp"]:
            sc = np.full(len(units), np.nan)
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
                    np.zeros(len(units)), ybin, groups):
                rb = FuzzyRuleBase(ql, qh, mode=mode).fit(rule_inputs(units, tr))
                sc[te] = rb.score(rule_inputs(units, te))[0]
            E4[mode][f"{ql}-{qh}"] = float(roc_auc_score(ybin, sc))
    for mode in ["fuzzy", "crisp"]:
        v = np.array(list(E4[mode].values()))
        E4[mode + "_summary"] = {"min": float(v.min()), "max": float(v.max()),
                                 "mean": float(v.mean()), "sd": float(v.std(ddof=1))}
    stats["E4_sensitivity"] = E4

    # E5 explanation profile of the implication model's top-quartile flags
    imp = keep["implication"]; ok = ~np.isnan(imp["oof"])
    flag = imp["oof"] >= np.nanquantile(imp["oof"], 0.75)
    W = imp["W"]
    E5 = {"mean_firing_flagged": {RULES[j][0]: float(np.nanmean(W[flag & ok, j])) for j in range(len(RULES))},
          "mean_firing_rest": {RULES[j][0]: float(np.nanmean(W[~flag & ok, j])) for j in range(len(RULES))},
          "share_flagged_with_rule_ge_0.5": float(np.mean(np.nanmax(W[flag & ok][:, [0, 1, 2]], 1) >= 0.5))}
    if "lr_oof" in units[0]:
        lr = np.array([u["lr_oof"] for u in units])
        tq = lambda x: x >= np.nanquantile(x, 0.75)
        only_tf = tq(imp["oof"]) & ~tq(lr) & (ybin == 1)
        E5["transformer_only_true_highrisk_n"] = int(only_tf.sum())
        E5["transformer_only_mean_firing"] = {RULES[j][0]: float(np.nanmean(W[only_tf, j]))
                                              for j in range(len(RULES))}
    stats["E5_explanation"] = E5

    json.dump(stats, open(out_json, "w"), indent=2)
    make_figures(stats, fig_dir)
    print(json.dumps({k: stats[k] for k in ["E1_accuracy", "E3_signal"]}, indent=1)[:3000])


def make_figures(st, fig_dir="."):
    import os
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    E1 = st["E1_accuracy"]; kinds = ["none", "linear_mse", "tsk_mse", "implication"]
    lab = ["λ=0", "linear\nMSE", "fuzzy TSK\nMSE", "fuzzy\nimplication"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i in range(st["seeds"]):
        ax[0].plot(range(4), [E1[k]["per_seed"][i] for k in kinds], color="0.75", lw=0.8)
    ax[0].plot(range(4), [E1[k]["mean"] for k in kinds], "o-", color="navy", lw=2)
    ax[0].set_xticks(range(4), lab); ax[0].set_ylabel("5-fold AUC (per seed)")
    ax[0].set_title("(a) Discrimination by rule-injection variant")
    for k, c in zip(["linear_mse", "tsk_mse", "implication"], ["gray", "darkorange", "purple"]):
        lams = [0.0, 0.3, 0.6, 1.0]
        ax[1].plot(lams, [st["E2_consistency"][f"{k}@{l}"]["spearman_pred_vs_fuzzy_s"] for l in lams], "o-", c=c, label=k)
        ax[2].plot(lams, [st["E2_consistency"][f"{k}@{l}"]["implication_violation"] for l in lams], "o-", c=c, label=k)
    ax[1].set_xlabel("λ"); ax[1].set_ylabel("Spearman(pred, fuzzy rule score)"); ax[1].set_title("(b) Rule consistency")
    ax[2].set_xlabel("λ"); ax[2].set_ylabel("mean implication violation"); ax[2].set_title("(c) Rule violation on held-out units")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "fig4_fuzzy.png"), dpi=300); plt.close()

    E4 = st["E4_sensitivity"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for a, mode in zip(ax, ["crisp", "fuzzy"]):
        M = np.array([[E4[mode][f"{ql}-{qh}"] for qh in [0.6, 0.7, 0.8, 0.9]] for ql in [0.1, 0.2, 0.3]])
        im = a.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=max(0.7, M.max()))
        for i in range(3):
            for j in range(4):
                a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)
        a.set_xticks(range(4), [0.6, 0.7, 0.8, 0.9]); a.set_yticks(range(3), [0.1, 0.2, 0.3])
        a.set_xlabel("upper breakpoint quantile"); a.set_title(f"{mode} rule score AUC")
    ax[0].set_ylabel("lower breakpoint quantile"); fig.colorbar(im, ax=ax, shrink=0.8)
    plt.savefig(os.path.join(fig_dir, "fig5_fuzzy.png"), dpi=300, bbox_inches="tight"); plt.close()


def synthetic_units(n_students=120, seed=0):
    rng = np.random.default_rng(seed); units = []
    for s in range(n_students):
        lat = rng.beta(2, 5)
        for g in range(3):
            n = rng.integers(5, 60); da = rng.integers(0, N_DA, n)
            v, u, p, e = rng.beta(2, 4), rng.beta(1.5, 6), rng.beta(2, 5), rng.beta(1, 8)
            y = float(np.clip(0.4 * v + 0.2 * u + 0.2 * max(p - e, 0) + 0.3 * lat + rng.normal(0, .15), 0, 1))
            units.append(dict(group_key=f"s{s}_f24", da=da, num=rng.random((n, 4)), y=y,
                              v=v, u=u, p=p, e=e))
    return units


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="results/paper_stats_fuzzy.json"); ap.add_argument("--figdir", default="results")
    a = ap.parse_args()
    import os; os.makedirs(a.figdir, exist_ok=True); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.demo:
        main(synthetic_units(), seeds=2, sweep_seeds=1, epochs=2, out_json=a.out, fig_dir=a.figdir)
    else:
        main(pickle.load(open(a.units, "rb")), seeds=a.seeds, epochs=a.epochs, out_json=a.out, fig_dir=a.figdir)
