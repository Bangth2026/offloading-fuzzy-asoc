# Check the premises before imposing the rules: an audit of fuzzy rule models on three student–AI interaction datasets

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23940473.svg)](https://doi.org/10.5281/zenodo.23940473)

Code, time-stamped analysis plans and statistics files for the manuscript *Check the Premises Before Imposing the Rules: An Audit of Fuzzy Rule Models on Three Student–AI Interaction Datasets* (H. B. Truong, submitted to *Applied Soft Computing*). The manuscript is rendered from the statistics files by `manuscript_A/fill_A.py`; no number in it is typed by hand. No third-party data are redistributed.

## Versions

- v1.0.0 (`10.5281/zenodo.23940474`): earlier draft on StudyChat only. Its behavioural conclusions are withdrawn.
- v1.1.0 (`10.5281/zenodo.23941973`): audit of Study 1 (R1–R1c) and Study 2 on TutorTrace.
- v1.2.0 (`10.5281/zenodo.23969402`): Study 3 on RECIPE4U and revision R2 after an internal review.
- v1.3.0: revision R3 after a second internal review (`PREREG_R3.md`, `study2/run_r3.py` → `results/r3_stats.json`, `results/r3_synthetic.json`): conditional logistic within-student checks, sign-check stability over 200 splits, alternative structure partition, noisy lagged labels, support-scaled prior, output monotonicity, and a simulation with a known generating process; revision R4 after a third internal review (`PREREG_R4.md`, `study2/run_r4.py` → `results/r4_stats.json`, `results/r4_synthetic.json`): check-by-check verdicts, SCw-TSK, three-term TSK, monotone boosting, calibration and mixed-confound simulation. Manuscript, supplementary material and response letter in `manuscript_A/`. All versions: `10.5281/zenodo.23940473`.

## What the audit found

The audit has three checks: antecedent signs within structural cells and within students; trivial baselines (structure, label persistence, student propensity); and cross-cohort transfer under a written plan.

- **Study 1, StudyChat** (`src/`). A model that knows only the assessment group (AUC 0.699 / 0.780) outranks every behavioural model, including the count-based TSK rule base CF-TSK (0.663 / 0.642). No within-group association larger than about 0.16 can be detected (equivalence test, SESOI 0.1).
- **Study 2, TutorTrace** (`study2/`). A sign-checked eight-rule TSK classifier for dependent queries transfers across deployments (0.642 / 0.644 / 0.684). Its inputs keep their sign within students. On the new task it beats label persistence by 0.202 AUC, but it trails regression on all features.
- **Study 3, RECIPE4U** (`study3/`). The same classifier for production requests beats course structure, including within-course cross-validation. The one-line rule "the previous request asked for production" comes within 0.032, so the rule base mainly describes persistence.
- **Monotone consequents (SM-TSK).** These remove all sign violations at a cost of 0.01 AUC or less in four of five splits.

## Analysis plans

Each `PREREG_*.md` file (and `study2/PREREG_S2_tutortrace.md`, `study3/PREREG_S3_recipe4u.md`) was committed before its run. Section 3.2 and Table 1 of the manuscript state for which plans GitHub's push log shows the plan on the public host before the run: Study 3 and R2 (`PREREG_R2.md`, pushed 03:21 UTC on 5 November 2026). For the others the order rests on local commit timestamps. These are time-stamped analysis plans, not registrations with a third party.

R2 code: `study2/run_r2.py` → `results/r2_stats.json`, `results/r2_log.txt`. Descriptive group means: `src/group_means.py` → `results/group_means.json`.

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
       --repo-url https://github.com/Bangth2026/offloading-fuzzy-asoc --doi 10.5281/zenodo.<id>
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
