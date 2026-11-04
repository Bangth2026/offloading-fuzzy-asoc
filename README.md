# Fuzzy rule-constrained detection of cognitive offloading in student–LLM dialogues

Code accompanying the manuscript *Fuzzy Rule-Constrained Sequential Learning for Detecting Cognitive Offloading in Student–Large Language Model Tutoring Dialogues* (H. B. Truong, submitted to *Applied Soft Computing*).

A Transformer encoder scores cognitive-offloading risk from turn-level student–ChatGPT dialogue. A five-rule zero-order Takagi–Sugeno–Kang (TSK) fuzzy rule base, with membership breakpoints fitted on each training fold, constrains training through a Łukasiewicz implication loss and supplies the explanation attached to each score.

## Layout

| Path | Purpose |
|---|---|
| `src/build_units.py` | Algorithms 1–2: weak labels and turn features from the raw StudyChat release → `units.pkl` |
| `src/fuzzy_rules.py` | Membership functions, rule base FR1–FR5, TSK inference, explanations, rule losses |
| `src/run_fuzzy_experiments.py` | Experiments E1–E5 (Section 6.3) → `results/paper_stats_fuzzy.json`, Figures 5–6 |
| `manuscript/fill_manuscript.py` | Renders manuscript and cover letter from the statistics file |
| `manuscript/manuscript_template.txt` | Manuscript source with statistics slots and pre-specified interpretation rules |
| `config/studychat.json` | File names, score columns and maximum points (to be completed from the dataset README) |
| `tests/` | Smoke tests on synthetic data |

## Data

StudyChat (McNichols, Ikram & Lan, LAK 2026) is CC BY 4.0 but access-gated at <https://huggingface.co/datasets/wmcnicho/StudyChat>. Request access with your own account and download the `dialogues/` and `scores/` folders. No StudyChat data are included in this repository.

## Reproducing the results

```bash
pip install -r requirements.txt && npm install   # npm only for building the .docx
pytest -q                                         # synthetic smoke tests, no data needed

# 1. complete config/studychat.json (max_points from scores/README.md; check column names)
python src/build_units.py --data /path/to/StudyChat --config config/studychat.json --out results/units.pkl
#    stops unless the sample matches the paper: 175 students, 464 units, 116 high-risk units

# 2. experiments E1–E5 (10 paired seeds; about 90 minutes on 2 CPU cores)
python src/run_fuzzy_experiments.py --units results/units.pkl --out results/paper_stats_fuzzy.json --figdir results
#    direction check of the rule inputs (E6, Section 7.3), merged into the same file
python src/input_diagnostics.py --units results/units.pkl --stats results/paper_stats_fuzzy.json

# 3. manuscript and cover letter (.docx) from the statistics file
python manuscript/fill_manuscript.py --stats results/paper_stats_fuzzy.json \
       --repo-url https://github.com/<owner>/<repo> --doi 10.5281/zenodo.<id>
#    both flags are optional; without them the availability statement says the
#    archive is supplied with the submission and the DOI follows before publication
```

The statistics file and figures behind the submitted manuscript are included in `results/`
(`paper_stats_fuzzy.json`, `fig4_fuzzy.png`, `fig5_fuzzy.png`, `run_log.txt`).

## How numbers reach the manuscript

Every new statistic in the manuscript comes from `paper_stats_fuzzy.json`; none is typed by hand. The renderer uses Jinja2 with `StrictUndefined`, so a missing statistic stops the build. Interpretive sentences are selected by rules fixed in Section 6.3 before the experiments were run:

- an accuracy difference is *significant* only if its Holm-adjusted Wilcoxon p < 0.05 (four comparisons, ten paired seeds);
- a rule-score difference is *reliable* only if its 95% student-level cluster-bootstrap interval excludes zero;
- a membership scheme is *more stable* only if its AUC standard deviation across the 3 × 4 breakpoint grid is lower;
- a rule term *reduces violation* only if held-out implication violation falls by at least 10% relative to λ = 0;
- if a rule term costs accuracy, the recommended weight is the smallest λ whose AUC stays within one seed SD of λ = 0.

`manuscript/build/filled_values.json` records every value and decision used for a given build.

## Leakage safeguards

Cross-validation uses `StratifiedGroupKFold` grouped by student and semester. Membership breakpoints are refitted inside each training fold. The rule base was fixed from the crisp rules R1–R3 and the dialogue-act categories before any fuzzy experiment, and was not tuned on the ablation, which used the full data.

## Citation

See `CITATION.cff`. Licence: MIT (code). The StudyChat data remain under their own CC BY 4.0 licence and access terms.
