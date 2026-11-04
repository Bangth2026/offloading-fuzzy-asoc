# Analysis plan for E11: input selection nested inside the training semester (written before it was run)

Date: 4 November 2026. Committed before any E11 code touched the real units.

## Purpose

In E10 the three CF-TSK inputs (writing_request, conceptual_questions, provide_context)
were taken from the pooled ablation of Section 7.5, which used both semesters. E11 removes
that leak: the input selection is repeated using the training semester only, so the
held-out semester informs neither the inputs, the breakpoints, the consequents nor the label
threshold. No third cohort is available (StudyChat covers Fall 2024 and Spring 2025 only).

## Procedure, for each direction (primary: Fall 2024 → Spring 2025; secondary: reverse)

1. Label threshold: the training semester's 75th percentile of y, applied to both semesters.
2. Selection, training semester only: logistic regression (StandardScaler + LogisticRegression,
   C = 1, max_iter = 1000) on the eight broad counts, StratifiedGroupKFold(5, shuffle, seed 42)
   grouped by student; AUC = mean over folds. Refit it eight times, each time without one
   category. Rank categories by AUC drop (largest first; ties broken by the fixed category order
   used in the code). Keep the top three.
3. Build CF-TSK exactly as in PREREG_E10.md on log(1 + count) of the three selected categories,
   fit on the whole training semester, score the held-out semester once.
4. Reference: LR8 fitted on the whole training semester.

## Hypotheses (family of two, Bonferroni α = 0.025, 97.5% student-cluster bootstrap intervals, 2,000 resamples)

* H1-nested, non-inferiority: lower bound of AUC(CF-TSK nested) − AUC(LR8) above −0.05.
* H2-nested, transfer: lower bound of AUC(CF-TSK nested) above 0.5.

Decisions are confirmatory for the primary direction and secondary for the reverse one.

## Also reported

* The selected categories and the full drop ranking in each direction, and whether the
  selection equals the pooled one.
* The learned consequents of the nested model.
* A frozen copy of the primary-direction model (selected inputs, breakpoints, consequents,
  label threshold) written to results/frozen_cf_tsk_f24.json, so that any later cohort can be
  scored without refitting.

No other variant will be run on the held-out semesters for this manuscript.
