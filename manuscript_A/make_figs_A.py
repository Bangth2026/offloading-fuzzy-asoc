"""Figures for manuscript A, drawn only from the statistics files in the repository."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
st = json.load(open(os.path.join(ROOT, "results/paper_stats_fuzzy.json")))
s2 = json.load(open(os.path.join(ROOT, "study2/results_s2.json")))
r2 = json.load(open(os.path.join(ROOT, "results/r2_stats.json")))
r4 = json.load(open(os.path.join(ROOT, "results/r4_stats.json")))
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f2328", "#5b636e", "#d9dde3"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})
OUT = os.path.join(HERE, "figs")


def clean(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)


# ---------------------------------------------------------------- Figure 1: audit protocol
fig, ax = plt.subplots(figsize=(10, 1.9)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0.85, 2.5)
boxes = [(0.1, "Rule model\n(author-specified or\nlearned TSK)", "#eef3fb"),
         (2.55, "Check 1\nSign of each antecedent\nwithin cells and students", "#fff4ec"),
         (5.0, "Check 2\nTrivial baselines: structure,\npersistence, propensity", "#fff4ec"),
         (7.45, "Check 3\nTransfer across cohorts\nunder a committed plan", "#fff4ec")]
for x, t, c in boxes:
    ax.add_patch(FancyBboxPatch((x, 1.0), 2.25, 1.35, boxstyle="round,pad=0.04", fc=c, ec=MUTED, lw=1))
    ax.text(x + 1.125, 1.675, t, ha="center", va="center", fontsize=10.5, color=INK)
for x in (2.35, 4.8, 7.25):
    ax.annotate("", xy=(x + 0.2, 1.675), xytext=(x, 1.675), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_audit.png"), dpi=300, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 2: StudyChat audit
P = st["R1"]["prevalence"]["f24_threshold"]; R1 = st["R1"]; R1b = st["R1b"]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.35, 1.15]})
a = ax[0]; x = np.arange(3)
for k, (sem, col, lab) in enumerate((("f24", BLUE, "Fall 2024"), ("s25", ORANGE, "Spring 2025"))):
    v = [P[f"{sem}_g{i}"]["rate"] for i in (1, 2, 3)]
    a.bar(x + (k - 0.5) * 0.36, v, 0.34, color=col, label=lab)
    for i, val in enumerate(v):
        a.text(i + (k - 0.5) * 0.36, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5, color=INK)
a.set_xticks(x, ["Group 1\n(Exam 1)", "Group 2\n(Exam 2)", "Group 3\n(Final)"])
a.set_ylabel("share labelled high-risk"); a.legend(frameon=False, fontsize=8); clean(a)
a.set_title("(a) Label prevalence by assessment group", loc="left", fontsize=9.5)
a = ax[1]
rows = [("Group only", R1["R1B"]["primary"]["group_only"], R1["R1B"]["reverse"]["group_only"]),
        ("CF-TSK (learned consequents)", R1["R1A"]["primary"]["cf_tsk_same_inputs"], R1["R1A"]["reverse"]["cf_tsk_same_inputs"]),
        ("TSK, author-set consequents", R1["R1A"]["primary"]["ec_tsk"], R1["R1A"]["reverse"]["ec_tsk"]),
        ("LR, 3 counts", R1["R1D"]["primary"]["lr3"]["auc"], R1["R1D"]["reverse"]["lr3"]["auc"]),
        ("ANFIS-style TSK, 3 counts", R1["R1D"]["primary"]["anfis"]["auc"], R1["R1D"]["reverse"]["anfis"]["auc"])]
for i, (lab, p, r) in enumerate(rows[::-1]):
    for v, off, col, mk in ((p, -0.13, BLUE, "o"), (r, 0.13, ORANGE, "s")):
        a.plot(v[1], [i + off] * 2, color=col, lw=2, solid_capstyle="round")
        a.plot(v[0], i + off, mk, color=col, ms=6, mec="white", mew=1.2)
a.axvline(0.5, color=MUTED, ls="--", lw=1); a.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]])
a.set_xlabel("AUC on the held-out semester (95% CI)"); a.set_xlim(0.3, 0.86); clean(a)
a.plot([], [], "o", color=BLUE, label="Fall → Spring"); a.plot([], [], "s", color=ORANGE, label="Spring → Fall")
a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2); a.set_title("(b) Cross-semester AUC, pooled label", loc="left", fontsize=9.5)
a = ax[2]; D = R1b["direction_within_group"]
keys = ["v", "u", "p", "e", "log_W", "log_C", "log_P"]
labs = ["v  unchecked acceptance", "u  surface dependency", "p  provide-context share", "e  verification share",
        "log W  writing requests", "log C  conceptual qs", "log P  provide context"]
for i, k in enumerate(keys[::-1]):
    lo, hi = D[k]["ci"]; a.plot([lo, hi], [i, i], color=MUTED, lw=2); a.plot(D[k]["rho"], i, "o", color=INK, ms=5)
a.axvline(0, color=MUTED, ls="--", lw=1); a.set_yticks(range(len(keys)), labs[::-1]); clean(a)
a.set_xlabel("within-cell Spearman ρ with y (95% CI)"); a.set_title("(c) Behaviour vs label within groups", loc="left", fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_studychat_audit.png"), dpi=300, bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 3: TutorTrace
fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
models = [("sc_tsk", "SC-TSK (proposed, 8 rules)"), ("structure_only", "Structure only"),
          ("structure_plus_sc_tsk", "Structure + SC-TSK"), ("lr_all_behaviour", "LR, all behavioural features"),
          ("lr3", "LR, same 3 inputs"), ("wang_mendel", "Wang–Mendel"), ("anfis", "ANFIS-style TSK"),
          ("tree_d2", "Tree, depth 2"), ("ebm", "EBM")]
splits = [("primary_D1_to_D2", "D1 → D2", BLUE, "o", -0.2), ("secondary_D2_to_D1", "D2 → D1", ORANGE, "s", 0.0),
          ("secondary_D12_to_D3", "D1+D2 → D3 (new task)", "#1baf7a", "^", 0.2)]
for key, lab, col, mk, off in splits:
    for i, (m, _) in enumerate(models[::-1]):
        v = s2[key]["auc"][m]
        a.plot(v[1], [i + off] * 2, color=col, lw=1.6); a.plot(v[0], i + off, mk, color=col, ms=5.5, mec="white", mew=1)
    a.plot([], [], mk, color=col, label=lab)
a.axvline(0.5, color=MUTED, ls="--", lw=1); a.set_yticks(range(len(models)), [m[1] for m in models[::-1]])
a.set_xlabel("AUC, dependent vs guided query (95% CI)"); a.set_xlim(0.45, 0.85); clean(a)
a.legend(frameon=False, fontsize=8, loc="lower right"); a.set_title("(a) Held-out deployments", loc="left", fontsize=9.5)
b = ax[1]; o = s2["primary_D1_to_D2"]; sel = o["selected"]; z = r4["S2"]["D1_to_D2"]["sm_z"]
SUP2 = r2["S2"]["D1_to_D2"]["cell_support"]
M = np.array(z).reshape(2, 4)    # rows: first input LOW/HIGH; cols: (second, third) LL, LH, HL, HH
im = b.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for r in range(2):
    for c in range(4):
        lowc = SUP2[r * 4 + c] < 10
        if lowc:
            b.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///", ec="#9a9a9a", lw=0))
        b.text(c, r, f"{M[r, c]:.2f}\n(n={SUP2[r * 4 + c]})" + ("\nn < 10" if lowc else ""), ha="center", va="center",
               color=("white" if M[r, c] > 0.6 else INK) if not lowc else "#555555", fontsize=8.5, style="italic" if lowc else "normal",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8) if lowc else None)
short = {"pre_net_code_growth": "code growth", "pre_chars_inserted": "chars inserted", "pre_duration_s": "pre-query duration"}
n1, n2, n3 = (short.get(s, s) for s in sel)
b.set_yticks([0, 1], [f"{n1} LOW", f"{n1} HIGH"])
b.set_xticks(range(4), ["ins L\ndur L", "ins L\ndur H", "ins H\ndur L", "ins H\ndur H"], fontsize=8)
b.set_xlabel(f"ins = {n2}, dur = {n3}", fontsize=8, color=MUTED)
fig.colorbar(im, ax=b, shrink=0.8).set_label("consequent (relative score)")
b.set_title(f"(b) SM-TSK rule table, trained on D1 (base rate {o['train_dependent_rate']:.2f})", loc="left", fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_tutortrace.png"), dpi=300, bbox_inches="tight"); plt.close()
print("figures written")

# ---------------------------------------------------------------- Figure 4: RECIPE4U
s3 = json.load(open(os.path.join(ROOT, "study3/results_s3.json")))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
splits3 = [("primary_SW_to_AW_IRW", "SW → AW + IRW", BLUE, "o", -0.13), ("secondary_AW_IRW_to_SW", "AW + IRW → SW", ORANGE, "s", 0.13)]
for key, lab, col, mk, off in splits3:
    for i, (m, _) in enumerate(models[::-1]):
        v = s3[key]["auc"][m]
        a.plot(v[1], [i + off] * 2, color=col, lw=1.6); a.plot(v[0], i + off, mk, color=col, ms=5.5, mec="white", mew=1)
    a.plot([], [], mk, color=col, label=lab)
a.axvline(0.5, color=MUTED, ls="--", lw=1); a.set_yticks(range(len(models)), [m[1] for m in models[::-1]])
a.set_xlabel("AUC, production vs guidance request (95% CI)"); a.set_xlim(0.3, 0.85); clean(a)
a.legend(frameon=False, fontsize=8, loc="lower left"); a.set_title("(a) Held-out courses", loc="left", fontsize=9.5)
b = ax[1]; o = s3["primary_SW_to_AW_IRW"]; z = r4["S3"]["SW_to_AW_IRW"]["sm_z"]; SUP3 = r2["S3"]["SW_to_AW_IRW"]["cell_support"]
M = np.array(z).reshape(2, 4)
im = b.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for r in range(2):
    for c in range(4):
        lowc = SUP3[r * 4 + c] < 10
        if lowc:
            b.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///", ec="#9a9a9a", lw=0))
        b.text(c, r, f"{M[r, c]:.2f}\n(n={SUP3[r * 4 + c]})" + ("\nn < 10" if lowc else ""), ha="center", va="center",
               color=("white" if M[r, c] > 0.6 else INK) if not lowc else "#555555", fontsize=8.5, style="italic" if lowc else "normal",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8) if lowc else None)
b.set_yticks([0, 1], ["earlier production\nrequests: none", "earlier production\nrequests: some"])
b.set_xticks(range(4), ["reply L\nturns L", "reply L\nturns H", "reply H\nturns L", "reply H\nturns H"], fontsize=8)
b.set_xlabel("reply = mean length of earlier ChatGPT replies, turns = earlier turns in session", fontsize=8, color=MUTED)
fig.colorbar(im, ax=b, shrink=0.8).set_label("consequent (relative score)")
b.set_title(f"(b) SM-TSK rule table, trained on SW (base rate {o['train_dependent_rate']:.2f})", loc="left", fontsize=9.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_recipe4u.png"), dpi=300, bbox_inches="tight"); plt.close()
print("figure 4 written")

# ---------------------------------------------------------------- Figure 5: simulation (R5-E)
syn = [o for o in json.load(open(os.path.join(ROOT, "results/r5_synthetic.json"))) if o["violation"] == "none"]
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
names = {"A": "(a) Structural confound", "B": "(b) Propensity confound", "C": "(c) Persistence confound"}
betas = [0.0, 0.5, 1.0]
for a, c in zip(ax, ("A", "B", "C")):
    for req, col, mk in (("3-5", BLUE, "o"), ("6-10", ORANGE, "s")):
        rows = {o["beta"]: o for o in syn if o["confound"] == c and o["requests"] == req}
        a.plot(betas, [100 * rows[b]["behavioural"] for b in betas], marker=mk, color=col, lw=2, label=f"audit, {req} requests")
        a.plot(betas, [100 * rows[b]["fe_behavioural"] for b in betas], ls="--", color=col, lw=1.2, alpha=0.8, label=f"fixed effects, {req}")
    a.set_title(names[c], loc="left", fontsize=9.5); a.set_xticks(betas); a.set_xlabel("within-student effect β (0 = no effect)")
    a.set_ylim(-3, 103); clean(a)
ax[0].set_ylabel("% of replications called behavioural")
ax[0].legend(frameon=False, fontsize=7.5, loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_simulation.png"), dpi=300, bbox_inches="tight"); plt.close()
print("figure 5 written")

# ---------------------------------------------------------------- Figure S1: reliability diagrams (R4-D)
fig, ax = plt.subplots(1, 5, figsize=(13, 2.9), sharey=True)
splits = [("S2", "D1_to_D2", "D1 → D2"), ("S2", "D2_to_D1", "D2 → D1"), ("S2", "D12_to_D3", "D1+D2 → D3"),
          ("S3", "SW_to_AW_IRW", "SW → AW+IRW"), ("S3", "AW_IRW_to_SW", "AW+IRW → SW")]
for a, (st_, k, lab) in zip(ax, splits):
    c = r4[st_][k]["calibration"]
    for m, col, mk in (("sc_tsk", BLUE, "o"), ("sm_tsk", ORANGE, "s")):
        rel = c[m]["reliability"]
        a.plot([r["mean_p"] for r in rel], [r["obs"] for r in rel], marker=mk, color=col, lw=1.5, ms=4, label=m.replace("_", "-").upper())
    a.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1); a.axhline(c["base_rate_test"], color=GRID, lw=1)
    a.set_title(lab, fontsize=9); a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("mean score in bin"); clean(a)
ax[0].set_ylabel("observed share positive"); ax[0].legend(frameon=False, fontsize=7.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figS1_reliability.png"), dpi=300, bbox_inches="tight"); plt.close()
print("figure S1 written")
