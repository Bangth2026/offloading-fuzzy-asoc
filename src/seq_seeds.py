"""Seed stability of the LSTM and DKT-style baselines (ten seeds, same protocol as Table 3).

    python src/seq_seeds.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
"""
import argparse, json, pickle
import numpy as np
import run_fuzzy_experiments as R
from reproduce_benchmark import cv_seq, SeqLSTM

ap = argparse.ArgumentParser()
ap.add_argument("--units", required=True); ap.add_argument("--stats", required=True)
ap.add_argument("--out", default="results/seq_seeds.json"); ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--epochs", type=int, default=15)
a = ap.parse_args()
units = pickle.load(open(a.units, "rb")); n_da = R.set_n_da(units)
y = np.array([u["y"] for u in units]); g = np.array([u["group_key"] for u in units])
yb = (y >= np.quantile(y, 0.75)).astype(int)
out = {}
for name, use_da in (("lstm", True), ("dkt", False)):
    v = [cv_seq(units, yb, g, lambda: SeqLSTM(n_da, use_da), seed=s, epochs=a.epochs)[0] for s in range(a.seeds)]
    out[name] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(min(v)), "max": float(max(v))}
    print(name, out[name], flush=True)
st = json.load(open(a.stats)); st["E0_seq_seeds"] = out; json.dump(st, open(a.stats, "w"), indent=1)
json.dump(out, open(a.out, "w"), indent=1)
