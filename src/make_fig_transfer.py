"""Figure 7: cross-semester transfer (E10). Reads results/transfer_stats.json."""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/transfer_stats.json"
out = sys.argv[2] if len(sys.argv) > 2 else "results/fig7_transfer.png"
T = json.load(open(src))
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#5b636e", "#d9dde3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK})

models = [("expert_rules", "Expert rules FR1–FR5"), ("lr8", "LR, 8 broad counts"),
          ("fusion_lr8_cf", "Fusion LR8 + CF-TSK"), ("lr3", "LR, 3 log counts"),
          ("cf_tsk", "CF-TSK (8 learned rules)"), ("tf", "Transformer λ = 0 (10 seeds)"),
          ("tf_cf", "Transformer + CF-TSK (10 seeds)")]
dirs = [("primary", "Fall 2024 → Spring 2025", BLUE, "o", -0.14),
        ("reverse", "Spring 2025 → Fall 2024", ORANGE, "s", 0.14)]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]
for key, lab, col, mk, off in dirs:
    o = T[key]
    for i, (m, _) in enumerate(models):
        if m == "tf":
            v, e = o["H3"]["mean_none"], o["H3"]["sd_none"]
        elif m == "tf_cf":
            v, e = o["H3"]["mean_cf_imp"], o["H3"]["sd_cf_imp"]
        else:
            v, e = o["auc"][m], None
        yy = i + off
        if m == "cf_tsk":
            lo, hi = o["H2_cf_ci"]
            a.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle="round")
        if e is not None:
            a.plot([v - e, v + e], [yy, yy], color=col, lw=2, solid_capstyle="round")
        a.plot(v, yy, mk, color=col, ms=7, mec="white", mew=1.5, label=lab if i == 0 else None)
a.axvline(0.5, color=MUTED, lw=1, ls="--")
a.text(0.502, len(models) - 0.45, "chance", color=MUTED, fontsize=8)
a.set_yticks(range(len(models)), [m[1] for m in models])
a.set_xlim(0.4, 0.8); a.set_ylim(-0.6, len(models) - 0.3)
a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True)
for s in ("top", "right"):
    a.spines[s].set_visible(False)
a.set_xlabel("AUC on the held-out semester")
a.set_title("(a) Transfer to the other semester", loc="left", color=INK)
a.legend(frameon=False, loc="lower right", fontsize=8)

b = ax[1]
z = T["primary"]["consequents"]
M = np.zeros((2, 4)); txt = [["", "", "", ""], ["", "", "", ""]]
cols = [("L", "L"), ("L", "H"), ("H", "L"), ("H", "H")]   # (C, P)
for k, v in z.items():
    w, c, p = (s.split("=")[1] for s in k.split())
    r = 0 if w == "L" else 1; j = cols.index((c, p)); M[r, j] = v
im = b.imshow(M, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
for r in range(2):
    for j in range(4):
        b.text(j, r, f"{M[r, j]:.2f}", ha="center", va="center",
               color="white" if M[r, j] > 0.38 else INK, fontsize=9)
b.set_xticks(range(4), ["C LOW\nP LOW", "C LOW\nP HIGH", "C HIGH\nP LOW", "C HIGH\nP HIGH"])
b.set_yticks(range(2), ["W LOW", "W HIGH"])
b.set_title(f"(b) Learned consequents, trained on Fall 2024\n(base rate {T['primary']['train_base_rate']:.2f})",
            loc="left", color=INK)
cb = fig.colorbar(im, ax=b, shrink=0.8); cb.set_label("rule consequent z (risk)")
b.text(1.5, 2.05, "W: writing requests · C: conceptual questions · P: provide context (log counts)",
       ha="center", va="top", fontsize=7.5, color=MUTED, transform=b.transData)
plt.tight_layout()
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
