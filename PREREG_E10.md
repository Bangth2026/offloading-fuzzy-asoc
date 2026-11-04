# Analysis plan for E10: cross-semester confirmation (written before it was run)

Date: 4 November 2026. Committed before any E10 code touched the real units.

## Why and what is already known

E1–E9 used pooled five-fold CV over both semesters. They showed that the expert rule
base and its outcome-checked variant, which read unit-level *shares*, rank units near
chance, while a logistic regression on raw broad dialogue-act *counts* reaches AUC ≈ 0.64.
Section 7.8 therefore proposed that the signal lies in the volume of each kind of turn.
That hypothesis, and the choice of inputs below, came from analyses that used both
semesters. E10 is a confirmation on a split that no model in this study has been fitted
or evaluated on as a held-out set, not a test on unseen data; the manuscript says so.

## Split

* Training (discovery): Fall 2024 units (183 units, 65 students).
* Confirmation: Spring 2025 units (281 units, 110 students). Students are disjoint.
* Primary label: ybin = 1 if y ≥ the 75th percentile of y in Fall 2024; the same
  threshold is applied to Spring 2025.
* Every fitted quantity (breakpoints, consequents, scalers, weights) uses Fall 2024 only.
  Spring 2025 is scored once.

## Models

* **CF-TSK (proposed): count-based fuzzy rule base.** Inputs x_W, x_C, x_P = log(1 + count)
  of writing_request, conceptual_questions and provide_context turns in the unit.
  Each input has LOW/HIGH ramp memberships between the training 25th and 75th percentiles.
  The rule base is the full 2 × 2 × 2 grid (eight rules, product t-norm), so no rule
  direction is assumed. Zero-order TSK output with background weight 0.05 at 0.5. Consequents
  are fitted by bounded least squares (0 ≤ z ≤ 1) against ybin with a ridge pull (α = 1)
  toward the training base rate.
  Disclosure: the three categories are the three largest drops in the pooled ablation
  (Section 7.5); writing and conceptual turns are also the categories singled out by the
  source study [2].
* **LR8 (reference):** StandardScaler + LogisticRegression(C = 1, max_iter = 1000) on the eight
  broad counts.
* **LR3 (secondary):** the same on the three CF-TSK inputs (log1p counts), to separate the
  effect of the fuzzy structure from the effect of the input choice.
* **Transformer λ = 0** and **Transformer + CF-TSK interval implication** (λ = 1; loss as in
  PREREG_E7_E9.md): same architecture, 15 epochs, batch 32, Adam 1e-3, trained on all Fall 2024
  units, seeds 0–9.
* **Fusion:** mean of percentile ranks (ranks taken within Spring 2025) of LR8 and CF-TSK.

## Hypotheses and decision rules (family of four, Bonferroni α = 0.0125)

Confidence intervals: 2,000 student-cluster bootstrap resamples of Spring 2025, percentile
intervals at 98.75%.

* H1 non-inferiority: AUC(CF-TSK) − AUC(LR8) has a lower bound above −0.05.
  The margin of 0.05 AUC is fixed here, before analysis.
* H2 transfer: AUC(CF-TSK) has a lower bound above 0.5.
* H3 neuro-fuzzy constraint: Transformer + CF-TSK implication vs Transformer λ = 0, AUC on
  Spring 2025 over ten paired seeds, Wilcoxon signed-rank p < 0.0125 and mean difference > 0.
* H4 complementarity: AUC(Fusion) − AUC(LR8) has a lower bound above 0.

## Secondary (reported, no decisions attached)

* The same four quantities with Spring 2025 → Fall 2024 (reverse direction).
* Spring 2025 labels thresholded at its own 75th percentile.
* LR3 vs CF-TSK, and the eight learned consequents (the explanation table).
* Expert FR1–FR5 rule score transferred Fall 2024 → Spring 2025, for reference.

All results are reported whatever their direction. No other variant will be run on
Spring 2025 for this manuscript.
