"""Descriptive means per assessment group (Study 1), used in the text of Section 4. Not an analysis."""
import json, pickle, numpy as np
U = pickle.load(open("results/units.pkl", "rb"))
out = {}
for g in ("g1", "g2", "g3"):
    s = [u for u in U if u["group"] == g]
    out[g] = dict(n=len(s), v=float(np.mean([u["v"] for u in s])), y=float(np.mean([u["y"] for u in s])),
                  writing=float(np.mean([u["broad"].count("writing_request") for u in s])))
json.dump(out, open("results/group_means.json", "w"), indent=1)
print(out)
