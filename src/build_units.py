"""
build_units.py -- reference implementation of Algorithms 1 and 2 (Sections 3, 5.1).

Reads the gated StudyChat release (download it yourself after accepting the
dataset terms on Hugging Face) and writes units.pkl for run_fuzzy_experiments.py.

    python src/build_units.py --data /path/to/StudyChat --config config/studychat.json \
                              --out results/units.pkl

The script checks the result against the published sample (175 students,
464 units, 116 high-risk units) and stops if they differ, unless --no-check.
If you still have the original run_real_analysis_v2.py, export from it instead
and use this file only as a cross-check.
"""
from __future__ import annotations
import argparse, json, os, pickle, re, sys
from collections import defaultdict
import numpy as np
import pandas as pd

WRITING, PROVIDE, CONTEXTUAL, CONCEPTUAL = ("writing_request", "provide_context",
                                            "contextual_questions", "conceptual_questions")
EDIT_VERIFY = {"editing_request", "verification"}
EXCLUDE = {"off_topic"}
GROUPS = [("g1", ["a1", "a2"], "e1"), ("g2", ["a3", "a4"], "e2"), ("g3", ["a5", "a6", "a7"], "e3")]
EXPECTED = {"students": 175, "units": 464, "highrisk": 116}


# ------------------------------------------------------------------ Algorithm 1
def weak_labels(scores: pd.DataFrame, max_points: dict, cols: dict) -> pd.DataFrame:
    rows = []
    for _, s in scores.iterrows():
        sem = s["semester"]
        for name, assigns, exam in GROUPS:
            vals = [s[cols[a]] for a in assigns]
            if any(pd.isna(v) for v in vals) or pd.isna(s[cols[exam]]):
                continue
            w = [max_points[sem][a] for a in assigns]
            llm = float(np.dot(vals, w) / np.sum(w))
            no_llm = float(s[cols[exam]])
            y = float(np.clip((llm - no_llm) / llm, 0, 1)) if llm > 0 else 0.0
            rows.append(dict(userId=str(s[cols["userId"]]), semester=sem, group=name, y=y))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Algorithm 2
def turn_features(turns: list[dict], specific_vocab: dict):
    turns = sorted(turns, key=lambda d: (d["chatId"], d["interactionCount"]))
    feats, prev_ts, prev_chat, run = [], None, None, 0
    by_chat = defaultdict(list)
    for k, d in enumerate(turns):
        by_chat[d["chatId"]].append(k)
    broad = [d["llm_label"].split(">")[0].strip() for d in turns]
    for k, d in enumerate(turns):
        b = broad[k]
        spec = d["llm_label"].strip()
        ts = pd.Timestamp(d["timestamp"]).timestamp()
        lat = ts - prev_ts if (prev_chat == d["chatId"] and prev_ts is not None) else 0.0
        later = [j for j in by_chat[d["chatId"]] if j > k]
        earlier = [j for j in by_chat[d["chatId"]] if j < k]
        verbatim = int(b == WRITING and not any(broad[j] in EDIT_VERIFY for j in later))
        self_att = int(b == WRITING and any(broad[j] == PROVIDE for j in earlier))
        if prev_chat != d["chatId"]:
            run = 0                      # runs do not cross conversations
        if b == CONTEXTUAL:
            run += 1
        elif b == CONCEPTUAL:
            run = 0                      # reset by a conceptual question (Algorithm 2, line 7)
        surf = int(run >= 3)
        feats.append(dict(da=specific_vocab.setdefault(spec, len(specific_vocab)), broad=b,
                          lat=np.log1p(max(lat, 0.0)), verbatim=verbatim, self_att=self_att, surf=surf))
        prev_ts, prev_chat = ts, d["chatId"]
    return feats


def unit_rates(feats):
    w = [f for f in feats if f["broad"] == WRITING]
    n = max(len(feats), 1)
    return dict(v=float(np.mean([f["verbatim"] for f in w])) if w else 0.0,
                u=float(np.mean([f["surf"] for f in feats])) if feats else 0.0,
                p=sum(f["broad"] == PROVIDE for f in feats) / n,
                e=sum(f["broad"] == "verification" for f in feats) / n)


def assignment_of(rec, pattern):
    m = re.search(pattern, str(rec.get("topic", "")), flags=re.I)
    return f"a{m.group(1)}" if m else None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/units.pkl"); ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if any(v is None for sem in cfg["max_points"].values() for v in sem.values()):
        sys.exit("config: fill every max_points entry from scores/README.md first")

    scores = []
    for sem, fn in cfg["score_files"].items():
        df = pd.read_csv(os.path.join(a.data, fn)); df["semester"] = sem; scores.append(df)
    labels = weak_labels(pd.concat(scores), cfg["max_points"], cfg["score_columns"])

    turns = defaultdict(list); unknown = 0
    for sem, fn in cfg["dialogue_files"].items():
        with open(os.path.join(a.data, fn), encoding="utf8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["llm_label"].split(">")[0].strip() in EXCLUDE:
                    continue
                asg = assignment_of(r, cfg["assignment_pattern"])
                if asg is None:
                    unknown += 1; continue
                g = next(n for n, asgs, _ in GROUPS if asg in asgs)
                turns[(str(r["userId"]), sem, g)].append(r)
    print(f"turns without an assignment tag: {unknown}")

    vocab, units = {}, []
    for _, lab in labels.iterrows():
        key = (lab.userId, lab.semester, lab.group)
        if key not in turns:
            continue
        feats = turn_features(turns[key], vocab)
        num = np.array([[f["lat"], f["verbatim"], f["self_att"], f["surf"]] for f in feats], dtype=float)
        units.append(dict(group_key=f"{lab.userId}_{lab.semester}", group=lab.group,
                          da=np.array([f["da"] for f in feats]), num=num, y=float(lab.y),
                          **unit_rates(feats)))
    lat_max = max(u["num"][:, 0].max() for u in units) or 1.0
    for u in units:
        u["num"][:, 0] /= lat_max
    y = np.array([u["y"] for u in units])
    got = {"students": len({u["group_key"] for u in units}), "units": len(units),
           "highrisk": int((y >= np.quantile(y, 0.75)).sum())}
    print("sample:", got, "expected:", EXPECTED, "| specific dialogue acts:", len(vocab))
    if got != EXPECTED and not a.no_check:
        sys.exit("sample differs from the published one; check config before running experiments")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(units, open(a.out, "wb"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
