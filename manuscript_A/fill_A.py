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
    GM = json.load(open(os.path.join(ROOT, "results/group_means.json")))
    S["gm"] = {g: {k: f3(v) if k != "writing" else f"{v:.1f}" for k, v in GM[g].items() if k != "n"} for g in GM}
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
    r2 = json.load(open(os.path.join(ROOT, "results/r2_stats.json")))
    Q = {}
    for st_, keys in (("S2", ("D1_to_D2", "D2_to_D1", "D12_to_D3")), ("S3", ("SW_to_AW_IRW", "AW_IRW_to_SW"))):
        for k in keys:
            o = r2[st_][k]; q = {}
            for m in ("auc_sc", "auc_sm", "sm_minus_sc", "auc_markov", "sc_minus_markov", "auc_propensity", "sc_minus_propensity"):
                q[m] = ci(o[m])
            q["viol_sc"], q["viol_sm"] = o["viol_sc"], o["viol_sm"]
            q["lowsup"] = sum(1 for n in o["cell_support"] if n < 10)
            q["pos_train"], q["n_train"], q["pos_test"], q["n_test"] = o["pos_train"], o["n_train"], o["pos_test"], o["n_test"]
            q["ws"] = {s: {"tr": f3(v["train"]["rho"]), "trlo": f3(v["train"]["ci"][0]), "trhi": f3(v["train"]["ci"][1]),
                           "te": f3(v["test"]["rho"]), "telo": f3(v["test"]["ci"][0]), "tehi": f3(v["test"]["ci"][1])}
                       for s, v in o["within_student"].items()}
            q["ws_score"] = {"r": f3(o["within_student_sc_score"]["rho"]), "lo": f3(o["within_student_sc_score"]["ci"][0]),
                             "hi": f3(o["within_student_sc_score"]["ci"][1])}
            q["maxov"] = f2(max(abs(v) for s in o["overlap"].values() for v in s.values()))
            q["dup"] = o["duplicates"]
            if "H2_without_duplicates" in o:
                q["h2nd"] = ci(o["H2_without_duplicates"]["gain"]); q["h2nd_sel"] = o["H2_without_duplicates"]["selected"]
            sv = list(o["sensitivity"].values()); q["sens_lo"], q["sens_hi"] = f3(min(sv)), f3(max(sv))
            if st_ == "S3":
                q["restr"] = ci(o["auc_structure_restricted"]); q["restr_gain"] = ci(o["gain_over_restricted"])
                q["nolag"] = ci(o["without_lagged_label"]["auc"]); q["nolag_sel"] = o["without_lagged_label"]["selected"]
            Q[f"{st_}_{k}"] = q
    S["q"] = Q
    S["wc"] = {c: {"n": v["n"], "pos": v["pos"], "st": ci(v["structure"]), "sc": ci(v["sc_tsk"]), "both": ci(v["structure_plus_sc"]),
                   "gain": ci(v["gain"])} for c, v in r2["S3"]["within_course_cv"].items()}
    EQ = r2["S1"]["equivalence_sesoi_0.1"]
    S["eq"] = {k: {"r": f3(v["rho"]), "lo": f3(v["ci90"][0]), "hi": f3(v["ci90"][1])} for k, v in EQ.items()}
    S["eq_maxbound"] = f2(max(max(abs(v["ci90"][0]), abs(v["ci90"][1])) for v in EQ.values()))
    S["eq_none"] = not any(v["equivalent"] for v in EQ.values())
    S["s1cells"] = r2["S1"]["cell_counts"]; S["s1_zero_share"] = f"{100 * r2['S1']['share_zero_overall']:.0f}"
    S["s1_thr"] = f3(r2["S1"]["pooled_threshold"]); S["s1_thr_f24"] = f3(r2["S1"]["f24_threshold"]); S["padded"] = r2["padded_selections"]

    # --- R3 (PREREG_R3.md)
    r3 = json.load(open(os.path.join(ROOT, "results/r3_stats.json")))
    R3 = {}
    for st_, keys in (("S2", ("D1_to_D2", "D2_to_D1", "D12_to_D3")), ("S3", ("SW_to_AW_IRW", "AW_IRW_to_SW"))):
        for k in keys:
            o = r3[st_][k]; q = {"sel": o["selected"]}
            q["fe"] = {}
            for inp, a in o["A"].items():
                d = {}
                for part in ("train", "test"):
                    cl = a[part]["cond_logit"]
                    d[part] = {"b": f2(cl["coef"]), "lo": f2(cl["ci"][0]), "hi": f2(cl["ci"][1]), "n": cl["students"], "req": cl["requests"]}
                d["counts"] = a["train"]["counts"]; q["fe"][inp] = d
            B = o["B"]; one = "|".join(sorted(o["selected"]))
            q["stab"] = {"same": B["selected_sets"].get(one, 0), "nsets": len(B["selected_sets"]), "med": f3(B["auc_median"]),
                         "lo": f3(B["auc_p2_5"]), "hi": f3(B["auc_p97_5"]), "top": max(B["selected_sets"].items(), key=lambda kv: kv[1]),
                         "ssel": B["stability_selected"], "spad": B["stability_padded"], "sauc": ci(B["stability_auc"]),
                         "elig": B["eligible_share"]}
            if "C" in o:
                q["C"] = {"st": ci(o["C"]["structure_A1"]), "h2": ci(o["C"]["H2_gain_A1"]), "d": ci(o["C"]["sc_minus_structure_A1"])}
            E = o["EF"]
            q["aN"] = {"auc": ci(E["alpha_N10"]["auc"]), "alpha": f"{E['alpha_N10']['alpha']:.0f}",
                       "low": [{"rule": c["rule"], "n": c["support"], "z1": f2(c["z_alpha1"]), "zN": f2(c["z_alphaN10"])} for c in E["alpha_N10"]["low_support_cells"]],
                       "base": f2(E["alpha_N10"]["base_rate"])}
            q["mono"] = {"sc": f"{100 * E['output_monotonicity']['sc_tsk']['overall']:.1f}", "sm": f"{100 * E['output_monotonicity']['sm_tsk']['overall']:.1f}",
                         "per": {i: f"{100 * v:.1f}" for i, v in E["output_monotonicity"]["sc_tsk"]["per_input"].items()}}
            q["dflt"] = E["defaults"]
            if "within_student_scores" in E:
                q["wss"] = {m: {"r": f3(v["rho"]), "lo": f3(v["ci"][0]), "hi": f3(v["ci"][1])} for m, v in E["within_student_scores"].items()}
            if "D_conditional" in o:
                c = o["D_conditional"]
                q["cond"] = {"n": c["n"], "pos": c["pos"], "sc": ci(c["sc_tsk"]),
                             "other": {i: ci(v) for i, v in c.items() if i not in ("n", "pos", "sc_tsk")}}
            R3[f"{st_}_{k}"] = q
    R3["noise"] = {e: {sp: {m: f3(v["mean"]) for m, v in d.items()} | {"sc_min": f3(d["sc"]["min"]), "sc_max": f3(d["sc"]["max"])}
                       for sp, d in dd.items()} for e, dd in r3["S3"]["D_noise"].items()}
    syn = json.load(open(os.path.join(ROOT, "results/r3_synthetic.json")))
    R3["syn"] = {k: {m: f"{100 * v:.1f}" for m, v in o.items() if m != "reps"} | {"reps": o["reps"]} for k, o in syn.items()}
    S["r3"] = R3

    # --- R4 (PREREG_R4.md)
    r4 = json.load(open(os.path.join(ROOT, "results/r4_stats.json")))
    R4 = {}
    def gp(g):
        o = {"auc": ci(g["auc"])}
        for b in ("structure", "persistence", "propensity"):
            o[b] = ci(g[b]); o[b]["pass"] = g[b][1][0] > 0
        o["all2"] = all(g[b][1][0] > 0 for b in ("structure", "persistence", "propensity"))
        return o
    for st_, keys in (("S2", ("D1_to_D2", "D2_to_D1", "D12_to_D3")), ("S3", ("SW_to_AW_IRW", "AW_IRW_to_SW"))):
        for k in keys:
            o = r4[st_][k]; q = {"sel": o["selected"]}
            for m in ("sc_gaps", "sm_gaps", "tsk3", "mono_gbm"):
                q[m] = gp(o[m])
            q["scw"] = gp(o["scw"]); q["scw"]["sel"] = o["scw"]["selected"]; q["scw"]["same"] = o["scw"]["selected"] == o["selected"]
            q["scw"]["nelig"] = len(o["scw"]["eligible"])
            c = o["calibration"]
            q["cal"] = {"bt": f2(c["base_rate_train"]), "bte": f2(c["base_rate_test"])}
            for m in ("sc_tsk", "sm_tsk"):
                q["cal"][m] = {"brier": f3(c[m]["brier"]), "ece": f3(c[m]["ece"]), "brier_base": f3(c[m]["brier_base"])}
            q["smz"] = [f2(z) for z in o["sm_z"]]
            for m in ("no_turns", "bagged"):
                if m in o:
                    q[m] = gp(o[m]); q[m]["sel"] = o[m].get("selected"); q[m]["pad"] = o[m].get("padded")
            R4[f"{st_}_{k}"] = q
    S["r4"] = R4

    # --- R5 (PREREG_R5.md)
    r5 = json.load(open(os.path.join(ROOT, "results/r5_stats.json")))
    R5 = {}
    def au(a):
        o = {"auc": ci(a["auc"]), "gain": ci(a["gain_all"]), "c2": a["check2"], "c3": a["check3"], "v": a.get("overall")}
        if "gain_each" in a:
            o["each"] = {b: ci(g) for b, g in a["gain_each"].items()}
        if "rules" in a:
            o["rules"] = a["rules"]
        return o
    for st_, keys in (("S2", ("D1_to_D2", "D2_to_D1", "D12_to_D3")), ("S3", ("SW_to_AW_IRW", "AW_IRW_to_SW"))):
        for k in keys:
            o = r5[st_][k]
            q = {"sel": o["selected"], "c1": o["check1"]["verdict"], "c1lvl": f"{o['check1']['level']:.2f}",
                 "c1in": {i: {"r": f3(v["rho"]), "lo": f3(v["ci"][0]), "hi": f3(v["ci"][1]), "st": v["state"]} for i, v in o["check1"]["inputs"].items()}}
            for m in ("sc", "sm", "furia3", "furia_all", "rf"):
                q[m] = au(o[m])
            q["below"] = o["below"]; q["distinct"] = o["distinct_sm"]; q["slsqp"] = f"{o['slsqp_max_dev']:.0e}"
            if "sm_ci" in o:
                q["smci"] = {"lo": [f2(x) for x in o["sm_ci"]["lo"]], "hi": [f2(x) for x in o["sm_ci"]["hi"]], "unstable": o["sm_ci"]["unstable"]}
            if "non_first" in o:
                n = o["non_first"]; q["nf"] = {"n": n["n"], "sc": ci(n["sc"]), "pers": ci(n["persistence"]), "prop": ci(n["propensity"]),
                                               "dpers": ci(n["sc_minus_persistence"]), "dprop": ci(n["sc_minus_propensity"])}
            R5[f"{st_}_{k}"] = q
    R5["s1"] = {k: {m: ci(v) for m, v in o.items() if m != "base_rate_test"} for k, o in r5["S1_residualised"].items()}
    S["r5"] = R5
    S["s3struct4"] = (f"{json.load(open(os.path.join(ROOT, 'study3/results_s3.json')))['primary_SW_to_AW_IRW']['auc']['structure_only'][0]:.4f}", f"{r2['S3']['SW_to_AW_IRW']['auc_structure_restricted'][0]:.4f}")
    S["slsqp_max"] = f"{max(r5[st_][k]['slsqp_max_dev'] for st_ in ('S2', 'S3') for k in r5[st_]):.0e}"
    syn5 = json.load(open(os.path.join(ROOT, "results/r5_synthetic.json")))
    S["syn5_b1max"] = f"{100 * max(o['behavioural'] for o in syn5 if o['beta'] == 1.0 and o['violation'] == 'none'):.0f}"
    S["syn5"] = {f"{o['confound']}_{o['beta']}_{o['requests']}_{o['violation']}": {k: (f"{100 * v:.0f}" if isinstance(v, float) and not k.startswith("auc") and k != "beta" else (f3(v) if k.startswith("auc") else v)) for k, v in o.items()} for o in syn5}
    S["sup2"] = r2["S2"]["D1_to_D2"]["cell_support"]; S["sup3"] = r2["S3"]["SW_to_AW_IRW"]["cell_support"]
    syn4 = json.load(open(os.path.join(ROOT, "results/r4_synthetic.json")))
    S["syn4"] = {f"{o['confound']}_{o['beta']}_{o['requests']}": {k: (f"{100 * v:.0f}" if isinstance(v, float) and k not in ("beta",) else v) for k, v in o.items()} for o in syn4}
    S["syn4_noise_split"] = f"{100 * min(o['noise_eligible_split'] for o in syn4):.0f}–{100 * max(o['noise_eligible_split'] for o in syn4):.0f}"
    S["syn4_noise_within"] = f"{100 * min(o['noise_eligible_within'] for o in syn4):.0f}–{100 * max(o['noise_eligible_within'] for o in syn4):.0f}"
    nab = [o["naive"] for o in syn4 if o["beta"] == 0 and o["confound"] in ("A", "B")]
    S["syn4_naive_ab"] = f"{100 * min(nab):.0f}–{100 * max(nab):.0f}"
    S["syn4_fp_max"] = f"{100 * max(o['all_checks'] for o in syn4 if o['beta'] == 0):.0f}"
    sens_src = {f"S2_{k}": r2["S2"][k]["sensitivity"] for k in ("D1_to_D2", "D2_to_D1", "D12_to_D3")}
    sens_src.update({f"S3_{k}": r2["S3"][k]["sensitivity"] for k in ("SW_to_AW_IRW", "AW_IRW_to_SW")})
    S["sens_keys"] = list(sens_src["S2_D1_to_D2"].keys())
    S["sens"] = {sp: {k: f3(v) for k, v in d.items()} for sp, d in sens_src.items()}
    S["sens_all"] = (f3(min(min(d.values()) for d in sens_src.values())), f3(max(max(d.values()) for d in sens_src.values())))
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
    ap.add_argument("--vdoi", default="10.5281/zenodo.23969402")  # version DOI of the release cited
    ap.add_argument("--release", default="v1.2.0")
    a = ap.parse_args()
    S = context(a.repo_url, a.doi)
    S["vdoi"] = a.vdoi; S["release"] = a.release
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
    out = os.path.join(HERE, "build"); os.makedirs(out, exist_ok=True)
    for name in [n for n in ("manuscript_A", "cover_letter_A", "response_A", "supplementary_A") if os.path.exists(os.path.join(HERE, f"{n}.txt"))]:
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
