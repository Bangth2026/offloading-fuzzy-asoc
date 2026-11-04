"""
fuzzy_rules.py
Fuzzy behavioural rule base for cognitive-offloading risk (ASOC revision).

Replaces the crisp rules R1-R3 of the EAAI/KBS version with
  (i)  linguistic membership functions whose breakpoints are fitted on the
       TRAINING fold only (no test-fold leakage),
  (ii) a zero-order Takagi-Sugeno-Kang (TSK) rule base that produces a graded
       rule score s in [0,1] and per-rule firing strengths (the explanation),
  (iii) two ways of injecting the rules into training:
       - "tsk_mse":  lambda * MSE(r, s_tsk)            (fuzzy target)
       - "implication": lambda * mean Lukasiewicz implication violation,
          i.e. for a risk-raising rule j:  max(0, w_j * z_j - r)
               for a risk-lowering rule j: max(0, r - (1 - w_j))
         which penalises the network only where a rule actually fires.
  The previous linear signal (mean of the three rates) is kept as "linear_mse"
  so the old configuration is reproduced as a baseline.

Inputs per unit (all rates in [0,1], computed by the existing Algorithm 2):
  v : verbatim-acceptance rate          (R1)
  u : surface-dependency rate           (R2)
  p : provide-context rate              (R3, delegation side)
  e : verification rate                 (protective factor, Table 1)

The rule base below is fixed a priori from R1-R3 and Table 1. Do NOT add rules
derived from the Section 7.4 ablation (writing_request / conceptual_questions):
that ablation was run on the full data, so rules built from it would leak.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
except ImportError:  # module still usable for numpy-only analyses
    torch = None

INPUTS = ("v", "u", "p", "e")

# (name, antecedent as list of (input, 'HIGH'|'LOW'), consequent z, plain-language text)
RULES = [
    ("FR1", [("v", "HIGH")],               1.00,
     "generated output is usually accepted without a following edit or check"),
    ("FR2", [("u", "HIGH")],               0.75,
     "long runs of assignment-specific questions without a conceptual question"),
    ("FR3", [("p", "HIGH"), ("e", "LOW")], 1.00,
     "code or errors are pasted for the model to fix, with little verification"),
    ("FR4", [("v", "LOW"), ("e", "HIGH")], 0.00,
     "output is checked or edited rather than accepted as-is"),
    ("FR5", [("v", "LOW"), ("u", "LOW"), ("p", "LOW")], 0.00,
     "no delegation pattern is present"),
]
RAISING = [j for j, r in enumerate(RULES) if r[2] >= 0.5]
LOWERING = [j for j, r in enumerate(RULES) if r[2] < 0.5]
BACKGROUND_W, BACKGROUND_Z = 0.05, 0.5   # keeps s defined when no rule fires


def _as_matrix(inputs: dict) -> np.ndarray:
    return np.column_stack([np.asarray(inputs[k], dtype=float) for k in INPUTS])


class FuzzyRuleBase:
    """Zero-order TSK rule base with ramp memberships.

    HIGH(x) = clip((x - a) / (b - a), 0, 1), LOW(x) = 1 - HIGH(x),
    with a = quantile(q_low), b = quantile(q_high) of the training fold.
    mode='crisp' replaces the ramp by a step at the midpoint (a+b)/2, which
    reproduces threshold-style rules and serves as the crisp control.
    """

    def __init__(self, q_low=0.25, q_high=0.75, tnorm="product", mode="fuzzy"):
        assert mode in ("fuzzy", "crisp")
        assert tnorm in ("product", "min")
        self.q_low, self.q_high, self.tnorm, self.mode = q_low, q_high, tnorm, mode
        self.a = self.b = None

    def fit(self, train_inputs: dict):
        X = _as_matrix(train_inputs)
        self.a = np.quantile(X, self.q_low, axis=0)
        self.b = np.quantile(X, self.q_high, axis=0)
        same = self.b - self.a < 1e-9                  # degenerate input
        self.b = np.where(same, self.a + 1e-3, self.b)
        return self

    def memberships(self, inputs: dict) -> dict:
        X = _as_matrix(inputs)
        if self.mode == "fuzzy":
            hi = np.clip((X - self.a) / (self.b - self.a), 0.0, 1.0)
        else:
            hi = (X > (self.a + self.b) / 2.0).astype(float)
        return {k: {"HIGH": hi[:, i], "LOW": 1.0 - hi[:, i]}
                for i, k in enumerate(INPUTS)}

    def firing(self, inputs: dict) -> np.ndarray:
        """Return W with shape (n_units, n_rules)."""
        mu = self.memberships(inputs)
        n = len(next(iter(mu.values()))["HIGH"])
        W = np.ones((n, len(RULES)))
        for j, (_, ante, _, _) in enumerate(RULES):
            for k, term in ante:
                W[:, j] = (W[:, j] * mu[k][term] if self.tnorm == "product"
                           else np.minimum(W[:, j], mu[k][term]))
        return W

    def score(self, inputs: dict):
        """TSK output s in [0,1] and firing strengths W."""
        W = self.firing(inputs)
        z = np.array([r[2] for r in RULES])
        s = (W @ z + BACKGROUND_W * BACKGROUND_Z) / (W.sum(1) + BACKGROUND_W)
        return s, W

    def explain(self, inputs: dict, idx: int, top=2, min_w=0.2) -> str:
        _, W = self.score(inputs)
        order = np.argsort(-W[idx])[:top]
        parts = [f"{RULES[j][0]} ({W[idx, j]:.2f}): {RULES[j][3]}"
                 for j in order if W[idx, j] >= min_w]
        return "; ".join(parts) if parts else "no rule fires above %.1f" % min_w


def linear_signal(inputs: dict) -> np.ndarray:
    """Signal used in the EAAI version (Section 5.4): mean of R1, R2, R3 proxy."""
    v, u = np.asarray(inputs["v"]), np.asarray(inputs["u"])
    r3 = np.clip(np.asarray(inputs["p"]) - np.asarray(inputs["e"]), 0, 1)
    s = (v + u + r3) / 3.0
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s


# ---------------------------------------------------------------- losses (torch)
def rule_loss(r, kind, s=None, W=None):
    """r: predicted risk in (0,1), shape (B,). s: rule target (B,). W: firing (B,J)."""
    if kind == "none":
        return r.new_zeros(())
    if kind in ("linear_mse", "tsk_mse"):
        return torch.mean((r - s) ** 2)
    if kind == "implication":
        z = torch.tensor([RULES[j][2] for j in range(len(RULES))],
                         dtype=r.dtype, device=r.device)
        viol = []
        for j in RAISING:     # truth(w -> r>=z) = min(1, 1 - w*z + r)
            viol.append(torch.relu(W[:, j] * z[j] - r))
        for j in LOWERING:    # truth(w -> r<=0) = min(1, 1 - w + (1 - r))
            viol.append(torch.relu(r - (1.0 - W[:, j])))
        return torch.stack(viol, 1).mean()
    raise ValueError(kind)


if __name__ == "__main__":   # tiny sanity check
    rng = np.random.default_rng(0)
    inp = {k: rng.beta(2, 5, 200) for k in INPUTS}
    frb = FuzzyRuleBase().fit(inp)
    s, W = frb.score(inp)
    assert s.min() >= 0 and s.max() <= 1 and W.shape == (200, len(RULES))
    crisp = FuzzyRuleBase(mode="crisp").fit(inp)
    print("fuzzy s range", s.min().round(3), s.max().round(3),
          "| crisp unique s", np.unique(crisp.score(inp)[0].round(3)).size)
    print("example explanation:", frb.explain(inp, int(np.argmax(s))))
