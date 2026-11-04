"""
fill_manuscript.py -- render the ASOC manuscript and cover letter from real results.

    python manuscript/fill_manuscript.py \
        --stats results/paper_stats_fuzzy.json \
        --repo-url https://github.com/<owner>/<repo> --doi 10.5281/zenodo.<id>

Every number in Sections 7.3-7.7, the abstract, discussion and conclusion is
read from the statistics file written by src/run_fuzzy_experiments.py, and every
interpretive sentence is chosen by the decision rules fixed in Section 6.3:
  * accuracy difference "significant"  <=> Holm-adjusted Wilcoxon p < 0.05
  * rule-score difference "reliable"   <=> 95% bootstrap CI excludes 0
  * membership scheme "more stable"    <=> lower AUC SD across the grid
Rendering uses Jinja2 StrictUndefined: a missing statistic stops the build
instead of leaving a gap, so no number can be filled by hand or by guess.
"""
import argparse, json, os, shutil, subprocess, sys, datetime
from jinja2 import Environment, StrictUndefined

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def build_context(st, repo_url, doi):
    E1, E2, E3, E4, E5 = (st[k] for k in ("E1_accuracy", "E2_consistency", "E3_signal",
                                          "E4_sensitivity", "E5_explanation"))
    S = {"seeds": st["seeds"], "repo_url": repo_url, "doi": doi}
    for short, k in [("none", "none"), ("lin", "linear_mse"), ("tsk", "tsk_mse"), ("imp", "implication")]:
        S[f"{short}_m"], S[f"{short}_sd"] = f3(E1[k]["mean"]), f3(E1[k]["sd"])
        S[f"{short}_r"] = f3(E1[k]["r_mean"])
    for key, comp in [("c_imp_none", "implication_vs_none"), ("c_tsk_none", "tsk_mse_vs_none"),
                      ("c_lin_none", "linear_mse_vs_none"), ("c_imp_lin", "implication_vs_linear_mse")]:
        c = E1[comp]
        S[key] = {"d": f3(c["mean_diff"]), "npos": c["n_pos"], "p": fp(c["wilcoxon_p_holm"]),
                  "tp": fp(c["ttest_p"]), "cd": f3(c["cohen_d"])}
    ci = E1["implication_vs_none"]
    S["imp_d"], S["imp_d_abs"], S["imp_p"] = f3(ci["mean_diff"]), f3(abs(ci["mean_diff"])), fp(ci["wilcoxon_p_holm"])
    S["imp_sig"] = ci["wilcoxon_p_holm"] < ALPHA
    S["imp_pos"] = ci["mean_diff"] > 0
    S["imp_vs_lin_sig"] = E1["implication_vs_linear_mse"]["wilcoxon_p_holm"] < ALPHA

    v0 = E2["implication@0.0"]["implication_violation"]; v1 = E2["implication@1.0"]["implication_violation"]
    S["v_imp0"], S["v_imp1"] = f3(v0), f3(v1)
    S["v_imp1_num"] = v1
    S["v_lin1_num"] = E2["linear_mse@1.0"]["implication_violation"]
    S["imp_fewer_viol_than_lin"] = S["v_lin1_num"] > 0 and (S["v_lin1_num"] - v1) / S["v_lin1_num"] >= 0.10
    S["v_lin1"] = f3(S["v_lin1_num"]); S["v_tsk1"] = f3(E2["tsk_mse@1.0"]["implication_violation"])
    S["rho_imp0"] = f3(E2["implication@0.0"]["spearman_pred_vs_fuzzy_s"])
    S["rho_imp1"] = f3(E2["implication@1.0"]["spearman_pred_vs_fuzzy_s"])
    # "reduces violation": at least a 10% relative reduction (pre-specified, Section 6.3 spirit)
    S["viol_drop"] = v0 > 0 and (v0 - v1) / v0 >= 0.10
    S["viol_drop_pct"] = f"{100 * (v0 - v1) / v0:.0f}" if v0 > 0 else "0"
    # smallest lambda > 0 whose sweep AUC is within one seed-SD of lambda = 0
    sd0, a0 = E1["none"]["sd"], E2["implication@0.0"]["auc"]
    S["lam_ok"], S["lam_ok_viol_drop"] = None, False
    for lam in ("0.3", "0.6", "1.0"):
        e = E2[f"implication@{lam}"]
        if e["auc"] >= a0 - sd0:
            S["lam_ok"] = lam
            S["lam_ok_viol_drop"] = e["implication_violation"] < v0
            break

    S["f_auc_num"] = E3["fuzzy"]["auc"]
    S["f_auc"], S["c_auc"], S["l_auc"] = f3(E3["fuzzy"]["auc"]), f3(E3["crisp"]["auc"]), f3(E3["linear"]["auc"])
    S["f_rho"] = f3(E3["fuzzy"]["spearman_y"])
    for tag, k in [("fc", "fuzzy_minus_crisp_auc"), ("fl", "fuzzy_minus_linear_auc")]:
        m, (lo, hi) = E3[k]
        S[f"{tag}_d"], S[f"{tag}_lo"], S[f"{tag}_hi"] = f3(m), f3(lo), f3(hi)
        S[f"{tag}_sig"] = lo > 0 or hi < 0
        S[f"{tag}_pos"] = m > 0

    for tag, k in [("e4f", "fuzzy_summary"), ("e4c", "crisp_summary")]:
        for s in ("min", "max", "sd"):
            S[f"{tag}_{s}"] = f3(E4[k][s])
    S["e4_stable"] = E4["fuzzy_summary"]["sd"] < E4["crisp_summary"]["sd"]

    S["e5_share"] = f"{100 * E5['share_flagged_with_rule_ge_0.5']:.0f}"
    for j in ("FR1", "FR2", "FR3"):
        S[f"{j.lower()}_f"] = f3(E5["mean_firing_flagged"][j])
        S[f"{j.lower()}_r"] = f3(E5["mean_firing_rest"][j])
    S["fr1_r_num"] = E5["mean_firing_rest"]["FR1"]
    S["to_n"] = E5.get("transformer_only_true_highrisk_n", 0)
    if S["to_n"]:
        S["to_fr1_num"] = E5["transformer_only_mean_firing"]["FR1"]
        S["to_fr1"] = f3(S["to_fr1_num"])
    S["fr1_lower_flagged"] = E5["mean_firing_flagged"]["FR1"] < E5["mean_firing_rest"]["FR1"]
    # standalone rule score "close to chance": AUC below 0.55 (descriptive, not a test)
    S["f_near_chance"] = E3["fuzzy"]["auc"] < 0.55
    E6 = st["E6_inputs"]
    for k, tag in [("v", "v"), ("u", "u"), ("p", "p"), ("e", "e"), ("verbatim_all_turns", "va")]:
        S[f"e6_{tag}_rho"] = f3(E6[k]["spearman_y"])
        S[f"e6_{tag}_lo"], S[f"e6_{tag}_hi"] = f3(E6[k]["ci95"][0]), f3(E6[k]["ci95"][1])
        S[f"e6_{tag}_p"] = fp(E6[k]["spearman_p"])
        S[f"e6_{tag}_auc"] = f3(E6[k]["auc"])
        S[f"e6_{tag}_mean"] = f"{E6[k]['mean']:.2f}"
        S[f"e6_{tag}_neg"] = E6[k]["ci95"][1] < 0
        S[f"e6_{tag}_null"] = E6[k]["ci95"][0] < 0 < E6[k]["ci95"][1]
    S["today"] = datetime.date.today().strftime("%d %B %Y")
    return S


def render(src, S, dst):
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    txt = env.from_string(open(src, encoding="utf8").read()).render(S=S)
    for bad in ("[TBD", "{{", "{%", "[Author", "!NOTE", "!PLACEHOLDER"):
        if bad in txt:
            sys.exit(f"refusing to build: '{bad}' left in {dst}")
    open(dst, "w", encoding="utf8").write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--figdir", default=None, help="folder with fig4_fuzzy.png and fig5_fuzzy.png")
    ap.add_argument("--repo-url", default="")
    ap.add_argument("--doi", default="", help="Zenodo DOI, e.g. 10.5281/zenodo.1234567")
    ap.add_argument("--outdir", default=os.path.join(HERE, "build"))
    ap.add_argument("--allow-fewer-seeds", action="store_true", help="tests only")
    a = ap.parse_args()
    st = json.load(open(a.stats))
    if st["seeds"] != 10 and not a.allow_fewer_seeds:
        sys.exit(f"the manuscript reports ten paired seeds; this statistics file has {st['seeds']}")
    S = build_context(st, a.repo_url, a.doi)
    os.makedirs(a.outdir, exist_ok=True)
    figdir = a.figdir or os.path.dirname(os.path.abspath(a.stats))
    for f in ("fig4_fuzzy.png", "fig5_fuzzy.png"):
        shutil.copy(os.path.join(figdir, f), os.path.join(HERE, "figs", f))
    render(os.path.join(HERE, "manuscript_template.txt"), S, os.path.join(a.outdir, "manuscript.txt"))
    render(os.path.join(HERE, "cover_letter.txt"), S, os.path.join(a.outdir, "cover_letter.txt"))
    json.dump({k: v for k, v in S.items()}, open(os.path.join(a.outdir, "filled_values.json"), "w"),
              indent=1, default=str)
    for name in ("manuscript", "cover_letter"):
        subprocess.run(["node", os.path.join(HERE, "build_docx.js"),
                        os.path.join(a.outdir, f"{name}.txt"),
                        os.path.join(a.outdir, f"ASOC_{name}.docx")], cwd=HERE, check=True)
    print("built:", a.outdir)


if __name__ == "__main__":
    main()
