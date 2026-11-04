# Expert versus data-learned fuzzy rules for cognitive-offloading risk in student–LLM dialogues

Code accompanying the manuscript *Expert versus Data-Learned Fuzzy Rules for Cognitive-Offloading Risk in Student–LLM Tutoring Dialogues: A Cross-Semester Evaluation* (H. B. Truong, submitted to *Applied Soft Computing*).

The study compares an expert five-rule Takagi–Sugeno–Kang (TSK) rule base, used as a standalone score and as a Łukasiewicz implication constraint on sequence encoders, with CF-TSK, an eight-rule grid over dialogue-act volumes whose consequents are learned on training data. CF-TSK is tested across the two StudyChat semesters under analysis plans committed before each run (`PREREG_*.md`; the commit order is in `git-history.bundle` or the repository history). `results/frozen_cf_tsk_f24.json` holds the Fall 2024 model for scoring new cohorts without refitting.

## Layout

| Path | Purpose |
|---|---|
| `src/build_units.py` | Algorithms 1–2: weak labels and turn features from the raw StudyChat release → `units.pkl` |
| `src/fuzzy_rules.py` | Membership functions, rule base FR1–FR5, TSK inference, explanations, rule losses |
| `src/input_diagnostics.py` | E6: direction of each rule input against the label |
| `src/oc_tsk.py`, `src/run_followup.py` | E7–E9: outcome-checked rules, interval implication loss |
| `src/run_transfer.py`, `src/make_fig_transfer.py` | E10: CF-TSK cross-semester confirmation, Figure 7 |
| `src/run_nested.py` | E11: input selection inside the training semester; writes the frozen model |
| `src/run_lstm_transfer.py` | E12: LSTM encoder with and without the CF-TSK constraint |
| `src/reproduce_benchmark.py`, `src/seq_seeds.py` | Table 3, permutation test, ablation, complementarity, seed checks |
| `PREREG_*.md` | Analysis plans, each committed before its run |
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
#    follow-up E7-E9 (Section 7.8; plan in PREREG_E7_E9.md, committed before the run; ~25 min)
python src/run_followup.py --units results/units.pkl --stats results/paper_stats_fuzzy.json --out results/followup_stats.json
#    cross-semester confirmation E10 (Section 7.9; plan in PREREG_E10.md, committed before the run; ~5 min)
python src/run_transfer.py --units results/units.pkl --stats results/paper_stats_fuzzy.json --out results/transfer_stats.json
python src/make_fig_transfer.py results/transfer_stats.json results/fig7_transfer.png
#    E11 nested input selection + frozen model (plan in PREREG_E11.md)
python src/run_nested.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
#    benchmark, ablation, permutation, complementarity (Table 3, Sections 7.2, 7.5, 7.7)
python src/reproduce_benchmark.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
python src/seq_seeds.py --units results/units.pkl --stats results/paper_stats_fuzzy.json
#    E12 LSTM encoder with the E11 rule base, cross-semester (plan in PREREG_E12.md)
python src/run_lstm_transfer.py --units results/units.pkl --stats results/paper_stats_fuzzy.json

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
