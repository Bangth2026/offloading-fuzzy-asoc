# Analysis plan for revision R4 (written before the analyses were run)

Date: 5 November 2026. Committed and pushed to GitHub before any R4 code touches the real data.
Responds to a third internal (pre-submission) review of manuscript A (M1–M9, minor 1–10). All R4
analyses are exploratory. Unless stated: 95% student-cluster bootstrap intervals (2,000 resamples),
the five splits of Studies 2 and 3, SC-TSK as published (seed 0).

## Verdict rule (M1), fixed now
The audit verdict is reported per check, not as one binary label:
- Check 1 passes for an input if its within-student (and, in Study 1, within-cell) rank correlation
  on the training set has a 95% interval excluding 0 with the selected sign.
- Check 2 passes against baseline b if the lower 95% bound of AUC(model) − AUC(b) on the test set
  exceeds 0; reported separately for structure (structure + model vs structure), persistence and
  propensity.
- Check 3 passes if the held-out AUC has a lower bound above 0.5 under a plan pushed before the run.
A rule base is called behavioural only if all three checks pass.

## R4-A Within-student sign check inside construction (M2)
SCw-TSK: as Algorithm 2, but a candidate is eligible only if its within-student rank correlation
(R2 estimator, point estimate) has the same non-zero sign in both halves of the training students and
that sign agrees with the pooled direction. Real data, five splits: selected inputs, held-out AUC, and
the Check 2 gaps against structure, persistence and propensity.

## R4-B Richer and constrained comparators (M4)
On the three SC-TSK inputs of every split: (i) TSK3: three terms per input (LOW shoulder q25→q50,
MEDIUM triangle q25–q50–q75, HIGH shoulder q50→q75; 27 rules, product t-norm, consequents by Eq. (2)
with α = 1, w0 = 0.05); (ii) monotone gradient boosting: scikit-learn HistGradientBoostingClassifier
with monotonic_cst equal to the selected directions, default settings, random_state = 0. Held-out AUC
with interval and the Check 2 gaps. FURIA is not run (no maintained Python implementation).

## R4-C Study 3 inputs (M6)
(i) Candidates exclude `n_prev_turns`, which has the same ranks as position in session and is
structural by Table 1. (ii) Bagged SC-TSK: mean score of the 200 rule bases of R3-B (seeds 1–200).
For both: held-out AUC, Check 2 gaps (structure + model vs full structure, persistence, propensity).

## R4-D Calibration (M9)
Brier score and expected calibration error (10 equal-width bins) of SC-TSK and SM-TSK scores on every
held-out set, with the test base rate; reliability diagrams in the supplement.

## R4-E Simulation with mixed processes (M5)
Same design as R3-G, scenarios renamed: A structure, B propensity, C persistence, D true effect.
Mixed design: confound ∈ {structure, propensity, persistence} × within-student effect β ∈ {0, 0.25,
0.5, 1} × requests per student ∈ {3–5, 6–10}. For confound term c (cell offset; student intercept
a_s ~ N(0,1); persistence term +1.1 if the previous label is 1, −1.1 if 0, 0 for the first request):
logit p = c + β e and x1 = c* + e, with e ~ N(0,1) per request and c* = c for structure and propensity
and the share of earlier positives in the session for persistence. x2, x3 are N(0,1) noise.
100 replications per condition (seeds 0–99), bootstrap with 500 resamples. Recorded per condition:
share with x1 selected; share with a noise input selected by the split-half check and by the
within-student check (R4-A); naive pass (held-out AUC lower bound > 0.5); Check 1 pass for x1;
Check 2 pass against each baseline; all checks pass. Truth: x1 behavioural iff β > 0.

No other analysis will be run for this revision.
