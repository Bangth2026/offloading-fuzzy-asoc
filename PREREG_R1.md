# Analysis plan for the first revision (R1), written before the analyses were run

Date: 4 November 2026. Committed before any R1 code touched the real units. The analyses answer
an internal pre-submission review (items M1–M11, m1–m14). E1–E12 are unchanged; nothing below
replaces a confirmatory result reported earlier. Unless stated otherwise: inputs, breakpoints,
consequents, scalers and label thresholds are fitted on the training semester only; intervals
are 95% student-cluster bootstrap intervals (2,000 resamples) on the held-out semester; both
directions (Fall 2024 → Spring 2025 primary, reverse secondary) are reported.

## R1-A. Missing cell of the 2 × 2 design (M1)

EC-TSK: the CF-TSK grid on the E10 inputs (log counts of writing_request W, conceptual_questions
C, provide_context P) with consequents fixed by the author-specified directions of FR1/FR3/FR6/FR7:
z_j = (1[W HIGH] + 1[C LOW] + 1[P HIGH]) / 3. Nothing is learned except the breakpoints.
Reported: held-out AUC with interval, both directions, next to learned CF-TSK on the same inputs.
Reading: the author-specified directions transfer if the lower bound exceeds 0.5.

## R1-B. Label structure (M7)

1. Positive rate of the label by semester × assessment group (pooled threshold and the
   training-semester threshold used in E10–E12).
2. Group-only baseline: logistic regression on one-hot assessment group, cross-semester AUC.
3. Sensitivity of E11 (nested selection, CF-TSK, LR8 reference) to
   (i) a label thresholded at the 75th percentile within each (semester, group) cell,
   (ii) counts divided by the number of assignments in the group (2, 2, 3),
   (iii) both. Reading: the transfer result is robust if the CF-TSK lower bound exceeds 0.5 in
   the primary direction under each variant.

## R1-C. Prior ability (M8)

Units of groups 2 and 3, whose preceding unaided exam (Exam 1 for group 2, Exam 2 for group 3)
is available. (1) Spearman correlation of log W, log C, log P and total turns with the prior exam.
(2) On the held-out semester, logistic regression of the label on the standardised frozen
CF-TSK score (E11 model) and the standardised prior exam; reported: CF-TSK coefficient with
interval. Reading: CF-TSK carries signal beyond prior ability if that interval excludes 0.

## R1-D. Interpretable comparators (M3, M4, m3)

On the E11 nested inputs of each training semester: LR3; LR3 with all pairwise interactions;
decision trees of depth 2 and 3 (min_samples_leaf = 10); a Wang–Mendel rule base on the same
LOW/HIGH memberships (one rule per occupied antecedent cell, consequent = membership-weighted
label mean); an ANFIS-style TSK (eight rules, sigmoid memberships initialised at the quartile
ramps, memberships and consequents learned by gradient descent on BCE, Adam 0.01, 300 full-batch
epochs, seeds 0–9, mean prediction); an Explainable Boosting Machine if the `interpret` package
installs (default settings, seed 0). FURIA and genetic FRBCS have no maintained Python
implementation and are not run. Reported: AUC with interval and the difference from CF-TSK with
interval. Secondary: non-inferiority of CF-TSK to LR3 at the 0.05 margin.

## R1-E. Size of the constraint effect (M9)

E10 (Transformer) and E12 (LSTM) are re-run with identical seeds, saving held-out predictions;
the per-seed AUCs must reproduce the reported ones. For each encoder and direction, the
difference in AUC between the seed-averaged constrained and unconstrained predictions gets a
cluster bootstrap interval. A difference smaller than 0.01 AUC is defined as practically
negligible; the effect is called negligible if the whole interval lies within (−0.01, 0.01).

## R1-F. Ridge strength (m5)

CF-TSK of E11 with α ∈ {0.1, 1, 10}; AUC with interval and consequents.

## R1-G. Dialogue-act label noise (m11)

On the held-out semester, each turn's broad label is replaced with probability q ∈ {0.1, 0.2,
0.3, 0.4} by a uniformly drawn other category (20 repetitions each); the frozen E11 model scores
the noisy counts. Reported: mean and range of AUC by q.

## R1-H. Multiplicity across E10–E12 (M5)

Spring 2025 served as the held-out semester in seven confirmatory tests (E10 H1–H4, E11 H1–H2,
E12 H5). Each is re-evaluated at Bonferroni α = 0.05/7: intervals at 99.29% for the
interval-based hypotheses and p < 0.0071 for the Wilcoxon-based ones.

No further variant will be run for this revision beyond those listed.
