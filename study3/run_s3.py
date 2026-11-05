"""
run_s3.py -- Study 3 on RECIPE4U, as pre-registered in study3/PREREG_S3_recipe4u.md.
Re-uses the Study 2 protocol (study2/run_s2.py) with RECIPE4U features.

    python study3/run_s3.py --csv /path/to/reciep4u_dataset.csv --out study3/results_s3.json
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "study2"))
import run_s2 as S2

PRODUCTION = {"R Generation", "R Revision", "R Translation"}
GUIDANCE = {"R Information", "R Evaluation", "R Language Use", "R Confirmation"}
BEHAV = ["n_prev_turns", "prev_production_share", "prev_edit_count", "prev_mean_rating",
         "prev_mean_user_len", "prev_mean_reply_len", "essay_len_prev", "essay_growth_prev"]
STRUCT = ["course_SW", "course_AW", "course_IRW", "week", "session", "idx"]


def build(csv):
    d = pd.read_csv(csv)
    d["intent_final"] = d["intent_final"].astype(str).str.strip()
    d["is_essay_edited"] = d["is_essay_edited"].astype(str).str.lower().eq("true")
    for c in ("user", "chatgpt_after", "essay"):
        d[c + "_len"] = d[c].fillna("").astype(str).str.len()
    rows = []
    for _, g in d.sort_values("idx").groupby(["student_id", "course", "week", "session"], sort=False):
        g = g.sort_values("idx").reset_index(drop=True)
        for i in range(len(g)):
            prev = g.iloc[:i]; cur = g.iloc[i]
            req_prev = prev[prev.intent_final.str.startswith("R ")]
            r = dict(student_id=cur.student_id, course=cur.course, week=cur.week, session=cur.session, idx=cur.idx,
                     intent=cur.intent_final, query_length_chars=cur.user_len,
                     n_prev_turns=len(prev),
                     prev_production_share=float(req_prev.intent_final.isin(PRODUCTION).mean()) if len(req_prev) else 0.0,
                     prev_edit_count=float(prev.is_essay_edited.sum()),
                     prev_mean_rating=float(prev.rating.mean()) if len(prev) else 0.0,
                     prev_mean_user_len=float(prev.user_len.mean()) if len(prev) else 0.0,
                     prev_mean_reply_len=float(prev.chatgpt_after_len.mean()) if len(prev) else 0.0,
                     essay_len_prev=float(prev.essay_len.iloc[-1]) if len(prev) else 0.0,
                     essay_growth_prev=float(prev.essay_len.iloc[-1] - prev.essay_len.iloc[0]) if len(prev) else 0.0,
                     # added in R2 (PREREG_R2.md); not used by the Study 3 models
                     essay_len_start=float(g.essay_len.iloc[0]),
                     prev_request_production=(float(req_prev.intent_final.iloc[-1] in PRODUCTION) if len(req_prev) else 0.5))
            rows.append(r)
    f = pd.DataFrame(rows)
    f = f[f.intent.isin(PRODUCTION | GUIDANCE)].copy()
    f["y"] = f.intent.isin(PRODUCTION).astype(int)
    for c in ("SW", "AW", "IRW"):
        f[f"course_{c}"] = (f.course == c).astype(float)
    return f.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", default="study3/results_s3.json")
    a = ap.parse_args()
    f = build(a.csv)
    S2.STRUCT = STRUCT
    S2.behav_cols = lambda df, with_len=False: BEHAV + (["query_length_chars"] if with_len else [])
    sw, other = f[f.course == "SW"].reset_index(drop=True), f[f.course != "SW"].reset_index(drop=True)
    res = {"counts": {c: {"requests": int((f.course == c).sum()), "students": int(f[f.course == c].student_id.nunique()),
                          "production_rate": float(f[f.course == c].y.mean())} for c in ("SW", "AW", "IRW")}}
    res["primary_SW_to_AW_IRW"] = S2.evaluate(sw, other)
    res["secondary_AW_IRW_to_SW"] = S2.evaluate(other, sw)
    res["with_query_length_SW_to_AW_IRW"] = S2.evaluate(sw, other, with_len=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(res["counts"])
    for k in ("primary_SW_to_AW_IRW", "secondary_AW_IRW_to_SW", "with_query_length_SW_to_AW_IRW"):
        o = res[k]
        print(k, "| sel", o["selected"], "| H", o["decisions"], "| H1", o["H1"], "| H2", o["H2"], "| H3", o["H3"])
        print("   ", {m: f"{v[0]:.3f} [{v[1][0]:.3f},{v[1][1]:.3f}]" for m, v in o["auc"].items()})


if __name__ == "__main__":
    main()
