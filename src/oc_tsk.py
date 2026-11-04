"""
oc_tsk.py -- outcome-checked TSK rule base and interval implication loss (E7-E9).

Specified in PREREG_E7_E9.md before being run. Seven rules over six unit-level
rates; antecedents and memberships as in fuzzy_rules.py, consequents learned on
the training fold by bounded, ridge-to-expert least squares.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import lsq_linear

try:
    import torch
except ImportError:
    torch = None

INPUTS = ("v", "u", "p", "e", "w", "c")
RULES = [
    ("FR1", [("v", "HIGH")], 1.00, "generated output is usually accepted without a following edit or check"),
    ("FR2", [("u", "HIGH")], 0.75, "long runs of assignment-specific questions without a conceptual question"),
    ("FR3", [("p", "HIGH"), ("e", "LOW")], 1.00, "code or errors are pasted for the model to fix, with little verification"),
    ("FR4", [("v", "LOW"), ("e", "HIGH")], 0.00, "output is checked or edited rather than accepted as-is"),
    ("FR5", [("v", "LOW"), ("u", "LOW"), ("p", "LOW")], 0.00, "no delegation pattern is present"),
    ("FR6", [("w", "HIGH"), ("c", "LOW")], 1.00, "many writing requests and few conceptual questions"),
    ("FR7", [("c", "HIGH")], 0.00, "many conceptual questions"),
]
Z_EXPERT = np.array([r[2] for r in RULES])
W0, Z0 = 0.05, 0.5
ALPHA = 1.0


class OCTSK:
    def __init__(self, q_low=0.25, q_high=0.75, alpha=ALPHA, learn=True):
        self.q_low, self.q_high, self.alpha, self.learn = q_low, q_high, alpha, learn
        self.z = Z_EXPERT.copy()

    def _X(self, inp):
        return np.column_stack([np.asarray(inp[k], float) for k in INPUTS])

    def firing(self, inp):
        X = self._X(inp)
        hi = np.clip((X - self.a) / (self.b - self.a), 0, 1)
        mu = {k: {"HIGH": hi[:, i], "LOW": 1 - hi[:, i]} for i, k in enumerate(INPUTS)}
        W = np.ones((len(X), len(RULES)))
        for j, (_, ante, _, _) in enumerate(RULES):
            for k, term in ante:
                W[:, j] *= mu[k][term]
        return W

    def fit(self, inp, ybin):
        X = self._X(inp)
        self.a = np.quantile(X, self.q_low, 0)
        self.b = np.quantile(X, self.q_high, 0)
        self.b = np.where(self.b - self.a < 1e-9, self.a + 1e-3, self.b)
        if self.learn:
            W = self.firing(inp); den = W.sum(1) + W0
            A = W / den[:, None]; bvec = W0 * Z0 / den
            M = np.vstack([A, np.sqrt(self.alpha) * np.eye(len(RULES))])
            t = np.concatenate([np.asarray(ybin, float) - bvec, np.sqrt(self.alpha) * Z_EXPERT])
            self.z = lsq_linear(M, t, bounds=(0, 1)).x
        return self

    def score(self, inp):
        W = self.firing(inp)
        return (W @ self.z + W0 * Z0) / (W.sum(1) + W0), W


def interval_implication_loss(r, W, z):
    """w z <= r <= 1 - w (1 - z) for every rule (Lukasiewicz reading of 'antecedent -> r ~ z')."""
    z = torch.as_tensor(z, dtype=r.dtype)
    lo = torch.relu(W * z - r[:, None])
    hi = torch.relu(r[:, None] - 1 + W * (1 - z))
    return (lo + hi).mean()
