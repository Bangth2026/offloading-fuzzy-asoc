"""
fill_A.py -- render manuscript A from the statistics files (no number is typed by hand).

    python manuscript_A/fill_A.py [--repo-url URL] [--doi DOI]
"""
import argparse, json, os, subprocess, sys, datetime
from jinja2 import Environment, StrictUndefined

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def ci(v):        # [est, [lo, hi]] -> dict
    return {"e": f3(v[0]), "lo": f3(v[1][0]), "hi": f3(v[1][1])}


def context(repo_url, doi):
    st = json.load(open(os.path.join(ROOT, "results/paper_stats_fuzzy.json")))
    s2 = json.load(open(os.path.join(ROOT, "study2/results_s2.json")))
    cells = json.load(open(os.path.join(ROOT, "results/audit_cells.json")))
    S = {"repo_url": repo_url, "doi": doi, "today": datetime.date.today().strftime("%d %B %Y")}
    # --- Study 1 headline numbers
    S["n_units"], S["n_students"] = st["N_units"], st["N_students"]
    S["f_auc"] = f3(st["E3_signal"]["fuzzy"]["auc"]); S["c_auc"] = f3(st["E3_signal"]["crisp"]["auc"])
    c = st["E1_accuracy"]["implication_vs_none"]
    S["imp_d"], S["imp_p"] = f3(c["mean_diff"]), f"{c['wilcoxon_p_holm']:.3f}"
    v = st["E6_inputs"]["v"]; S["v_rho"], S["v_lo"], S["v_hi"] = f3(v["spearman_y"]), f3(v["ci95"][0]), f3(v["ci95"][1])
    S["oc_auc"] = f3(st["E7_E9_followup"]["means"]["oc_rule"]["mean"])
    B = st["E0_benchmark"]; Q = st["E0_seq_seeds"]
    S["lr_m"], S["lr_sd"] = f3(B["lr_seeds"]["mean"]), f3(B["lr_seeds"]["sd"])
    S["lstm_m"], S["lstm_sd"] = f3(Q["lstm"]["mean"]), f3(Q["lstm"]["sd"])
    S["tf_m"], S["tf_sd"] = f3(st["E1_accuracy"]["none"]["mean"]), f3(st["E1_accuracy"]["none"]["sd"])
    S["perm_p"] = f"{B['permutation']['p']:.3f}"
    T = st["E10_transfer"]
    for tag, k in (("p", "primary"), ("r", "reverse")):
        o = T[k]
        S[f"cf_{tag}"], S[f"lr8_{tag}"], S[f"lr3_{tag}"] = f3(o["auc"]["cf_tsk"]), f3(o["auc"]["lr8"]), f3(o["auc"]["lr3"])
        S[f"ex_{tag}"] = f3(o["auc"]["expert_rules"])
        S[f"cfci_{tag}"] = {"lo": f3(o["H2_cf_ci"][0]), "hi": f3(o["H2_cf_ci"][1])}
        N = st["E11_nested"][k]; S[f"nest_{tag}"] = f3(N["auc_cf_nested"]); S[f"nestsel_{tag}"] = ", ".join(N["selected"])
        L = st["E12_lstm"][k]; S[f"lstmx_{tag}"] = f3(L["mean_lstm"])
    R1 = st["R1"]
    P = R1["prevalence"]["f24_threshold"]
    for s_ in ("f24", "s25"):
        for g in ("g1", "g2", "g3"):
            S[f"prev_{s_}_{g}"] = f2(P[f"{s_}_{g}"]["rate"])
    S["zero_f24_g2"] = f"{100 * cells['f24_g2']['zero_share']:.0f}"
    S["grp_p"], S["grp_r"] = ci(R1["R1B"]["primary"]["group_only"]), ci(R1["R1B"]["reverse"]["group_only"])
    S["grp_cv"] = f3(st["R1b"]["group_only_pooled_cv"])
    S["ec_p"], S["ec_r"] = ci(R1["R1A"]["primary"]["ec_tsk"]), ci(R1["R1A"]["reverse"]["ec_tsk"])
    S["cfec_p"] = ci(R1["R1A"]["primary"]["cf_minus_ec"])
    D = st["R1b"]["direction_within_group"]
    S["dir"] = {k: {"r": f3(x["rho"]), "lo": f3(x["ci"][0]), "hi": f3(x["ci"][1])} for k, x in D.items()}
    S["dir_max"] = f3(max(abs(x["rho"]) for x in D.values()))
    S["dir_all_null"] = all(x["ci"][0] < 0 < x["ci"][1] for x in D.values())
    W = st["R1c"]
    S["win"] = {k: {m: {"r": f3(x["rho"]), "lo": f3(x["ci"][0]), "hi": f3(x["ci"][1])} for m, x in W[k].items()} for k in W}
    S["win_range"] = (f2(min(x["rho"] for k in W for x in W[k].values())), f2(max(x["rho"] for k in W for x in W[k].values())))
    D1 = R1["R1D"]
    S["cmp"] = {k: {m: f3(D1[k][m]["auc"][0]) for m in D1[k] if isinstance(D1[k][m], dict)} for k in D1}
    E = st["R1E"]
    S["con"] = {k: {m: {"d": f3(E[k][m]["diff"]), "lo": f3(E[k][m]["ci"][0]), "hi": f3(E[k][m]["ci"][1])} for m in E[k]} for k in E}
    C = R1["R1C"]["primary"]
    S["prior_p"] = {"b": f2(C["coef_cf"]), "lo": f2(C["coef_cf_ci"][0]), "hi": f2(C["coef_cf_ci"][1])}
    C = R1["R1C"]["reverse"]
    S["prior_r"] = {"b": f2(C["coef_cf"]), "lo": f2(C["coef_cf_ci"][0]), "hi": f2(C["coef_cf_ci"][1])}
    G = R1["R1G"]["primary"]; S["noise_lo"], S["noise_hi"] = f3(min(G[q]["mean"] for q in ("0.1", "0.2", "0.3", "0.4"))), f3(max(G[q]["mean"] for q in ("0.1", "0.2", "0.3", "0.4")))
    # --- Study 2
    S["tt_counts"] = {str(k): v for k, v in s2["counts"].items()}
    for tag, k in (("a", "primary_D1_to_D2"), ("b", "secondary_D2_to_D1"), ("c", "secondary_D12_to_D3"), ("q", "with_query_length_D1_to_D2")):
        o = s2[k]
        S[f"t{tag}"] = {m: ci(v) for m, v in o["auc"].items()}
        S[f"t{tag}"]["H1"], S[f"t{tag}"]["H2"], S[f"t{tag}"]["H3"] = ci(o["H1"]), ci(o["H2"]), ci(o["H3"])
        S[f"t{tag}"]["dec"] = o["decisions"]; S[f"t{tag}"]["sel"] = o["selected"]
        S[f"t{tag}"]["base"] = f2(o["train_dependent_rate"]); S[f"t{tag}"]["ntest"] = o["n_test"]
        S[f"t{tag}"]["z"] = [f2(c["z"]) for c in o["consequents"]]
    s3 = json.load(open(os.path.join(ROOT, "study3/results_s3.json")))
    S["rc"] = s3["counts"]
    S["rc_fmt"] = {c: {"n": v["requests"], "s": v["students"], "p": f2(v["production_rate"])} for c, v in s3["counts"].items()}
    for tag, k in (("r1", "primary_SW_to_AW_IRW"), ("r2", "secondary_AW_IRW_to_SW"), ("rq", "with_query_length_SW_to_AW_IRW")):
        o = s3[k]
        S[tag] = {m: ci(v) for m, v in o["auc"].items()}
        S[tag]["H1"], S[tag]["H2"], S[tag]["H3"] = ci(o["H1"]), ci(o["H2"]), ci(o["H3"])
        S[tag]["dec"] = o["decisions"]; S[tag]["sel"] = o["selected"]; S[tag]["base"] = f2(o["train_dependent_rate"])
        S[tag]["z"] = [f2(c["z"]) for c in o["consequents"]]
        S[tag]["excluded"] = [r["feature"] for r in o["ranking_top10"] if not r["eligible"]][:2]
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/Bangth2026/offloading-fuzzy-asoc")
    ap.add_argument("--doi", default="10.5281/zenodo.23940473")
    a = ap.parse_args()
    S = context(a.repo_url, a.doi)
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    out = os.path.join(HERE, "build"); os.makedirs(out, exist_ok=True)
    for name in [n for n in ("manuscript_A", "cover_letter_A") if os.path.exists(os.path.join(HERE, f"{n}.txt"))]:
        txt = env.from_string(open(os.path.join(HERE, f"{name}.txt"), encoding="utf8").read()).render(S=S)
        for bad in ("[TBD", "{{", "{%", "[CẦN"):
            if bad in txt:
                sys.exit(f"refusing to build: '{bad}' left in {name}")
        open(os.path.join(out, f"{name}.txt"), "w", encoding="utf8").write(txt)
        subprocess.run(["node", os.path.join(ROOT, "manuscript", "build_docx.js"), os.path.join(out, f"{name}.txt"),
                        os.path.join(out, f"{name}.docx")], cwd=HERE, check=True)
    json.dump(S, open(os.path.join(out, "filled_values.json"), "w"), indent=1, default=str)
    print("built", out)


if __name__ == "__main__":
    main()
