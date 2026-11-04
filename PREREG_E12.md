# Analysis plan for E12: the LSTM as sequential encoder (written before it was run)

Date: 4 November 2026. Committed before any E12 code touched the real units.

## Why

The in-archive benchmark (Table 3) found the LSTM to be the most accurate pooled model
(0.673 ± 0.017 over ten seeds, against 0.646 ± 0.007 for the regression and 0.620 ± 0.022
for the Transformer). The author decided to adopt the LSTM as the sequential encoder. E12
tests the rule constraint and cross-semester transfer with it. The decision to run E12 was
taken after seeing Table 3; the plan below was fixed before E12 itself was run.

## Design

* Encoder: the LSTM of reproduce_benchmark.py (per-turn DA embedding and four turn features,
  one LSTM layer, d = 32), 15 epochs, batch 32, Adam 1e-3, seeds 0–9.
* Split and label: as in E11 (train on one semester, threshold = training 75th percentile,
  score the other semester once). Primary direction Fall 2024 → Spring 2025; reverse secondary.
* Rule base: the E11 CF-TSK of the training semester (inputs selected inside the training
  semester, consequents learned there), with the interval implication loss of PREREG_E7_E9.md
  at λ = 1.
* Conditions: LSTM λ = 0 and LSTM + CF-TSK, paired by seed.

## Hypothesis (single confirmatory test, primary direction)

* H5: LSTM + CF-TSK beats LSTM λ = 0 on the held-out semester; Wilcoxon signed-rank over the ten
  paired seeds, p < 0.05 and mean difference > 0.

## Reported descriptively

* Held-out AUC of both LSTM conditions (mean ± SD), next to CF-TSK (E11) and LR8.
* The same comparison in the reverse direction.

No other encoder variant, λ, or epoch budget will be run on the held-out semesters.
