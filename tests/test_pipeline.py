"""Smoke tests on synthetic data: no StudyChat data is needed or shipped."""
import json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fuzzy_rules import FuzzyRuleBase, rule_loss, RULES, INPUTS
from build_units import turn_features, unit_rates


def test_rule_base_bounds_and_crisp_limit():
    rng = np.random.default_rng(1); x = {k: rng.random(300) for k in INPUTS}
    s, W = FuzzyRuleBase().fit(x).score(x)
    assert 0 <= s.min() and s.max() <= 1 and W.shape == (300, len(RULES))
    sc, Wc = FuzzyRuleBase(mode="crisp").fit(x).score(x)
    assert set(np.unique(Wc)) <= {0.0, 1.0}


def test_implication_loss_zero_when_satisfied():
    import torch
    W = torch.tensor([[1.0, 0, 0, 0, 0]]); r = torch.tensor([1.0])
    assert float(rule_loss(r, "implication", W=W)) == 0.0
    assert float(rule_loss(torch.tensor([0.0]), "implication", W=W)) > 0


def test_turn_features_rules():
    t = lambda c, i, lab: dict(chatId=c, interactionCount=i, timestamp=str(1729128916365 + 60000 * i), llm_label={"label": lab, "justification": ""})
    turns = [t("c", 0, "provide_context>x"), t("c", 1, "writing_request>y"),
             t("c", 2, "contextual_questions>z"), t("c", 3, "contextual_questions>z"),
             t("c", 4, "contextual_questions>z")]
    f = turn_features(turns, {})
    assert f[1]["verbatim"] == 1 and f[1]["self_att"] == 1 and f[4]["surf"] == 1 and f[3]["surf"] == 0
    r = unit_rates(f); assert r["v"] == 1.0 and abs(r["p"] - 0.2) < 1e-9


def test_end_to_end_fill(tmp_path=None):
    tmp = tempfile.mkdtemp()
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "run_fuzzy_experiments.py"), "--demo",
                    "--out", os.path.join(tmp, "stats.json"), "--figdir", tmp], check=True)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import pickle
    from run_fuzzy_experiments import synthetic_units
    pickle.dump(synthetic_units(), open(os.path.join(tmp, "units.pkl"), "wb"))
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "input_diagnostics.py"),
                    "--units", os.path.join(tmp, "units.pkl"), "--stats", os.path.join(tmp, "stats.json")],
                   check=True, stdout=subprocess.DEVNULL)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "manuscript", "fill_manuscript.py"),
                          "--stats", os.path.join(tmp, "stats.json"), "--repo-url", "https://example.org/demo",
                          "--doi", "10.0000/demo", "--allow-fewer-seeds", "--outdir", os.path.join(tmp, "build")], check=True)
    txt = open(os.path.join(tmp, "build", "manuscript.txt"), encoding="utf8").read()
    assert "[TBD" not in txt and "{{" not in txt
