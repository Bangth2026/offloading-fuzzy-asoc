# Analysis plan for revision R2 (written before the analyses were run)

Date: 5 November 2026. Committed and pushed to GitHub before any R2 code touched the data.
Responds to an internal review of manuscript A (items M1–M10, minor 1–17). Studies 1–3 are not
re-run; R2 adds checks around them. Unless stated: 95% student-cluster bootstrap intervals (2,000
resamples), primary and secondary splits as in Studies 2 and 3.

## R2-A Structure baselines that are meaningful across courses (M1)

Study 3: (i) cross-course structure baseline restricted to position in session (`idx`) and essay
length at the first exchange of the session; (ii) within-course 5-fold StratifiedGroupKFold by
student (seed 42) in each course with ≥ 20 positives and in all courses pooled (with course
indicators): AUC of the full structure model, SC-TSK (selection inside each training fold),
and structure + SC-TSK; reported as pooled out-of-fold AUC with interval. The variables usable in
each split are listed.

## R2-B Student-level checks (M2)

For each SC-TSK input of Studies 2 and 3: within-student association, i.e. Spearman correlation of
within-student ranks of the input and of y, with interval, on the training and test sets (students
with ≥ 2 requests and both classes not required). Student-propensity baseline: Study 3, a student's
share of production requests in earlier sessions of the same course (0.5 if none); Study 2, the
student's share of dependent queries among earlier queries of the session (0.5 if none).

## R2-C Persistence baselines (M3)

Markov baseline: the previous request of the same session was a production request (Study 3) /
the previous query was dependent (Study 2); 0.5 if none. Reported: AUC with interval, and AUC(SC-TSK)
− AUC(Markov) with interval. Study 3 also: SC-TSK re-selected with the lagged-label input
(`prev_production_share`) removed from the candidates.

## R2-D Overlap with structure (M4)

Spearman correlation of each selected input with each structural variable on the training set.
Inputs with |ρ| ≥ 0.7 to any structural variable are removed and H2 (structure + SC-TSK vs
structure) is repeated with the next eligible input.

## R2-E Monotone consequents (M6)

Support of every rule cell: number of training requests with firing degree ≥ 0.5. SM-TSK: the
bounded ridge least squares of SC-TSK with linear inequality constraints that make the consequents
monotone in each input in the direction selected by the sign check (for every pair of rules that
differ only in that input). Solved by SLSQP from the SC-TSK solution. Reported: AUC with interval in
every split, and the number of monotonicity violations of SC-TSK and SM-TSK.

## R2-F Equivalence of within-group associations (M7)

Study 1: 90% cluster bootstrap intervals of the within-cell Spearman correlations (R1b). Smallest
effect of interest |ρ| = 0.1, chosen now, after the review and after R1b; an input is called
equivalent to no association if its 90% interval lies inside (−0.1, 0.1).

## R2-G Sensitivity of SC-TSK to w0 and α (minor 9)

Primary splits of Studies 2 and 3: w0 ∈ {0.01, 0.05, 0.2} × α ∈ {0.1, 1, 10}; AUC.

## R2-H Counts and membership degeneracy (minors 8, 12)

Positives and negatives per split; Study 1 counts per cell; for every selected input in every split,
whether its training quartiles coincide (step membership).

No other analysis will be run for this revision.
