# Analysis plan for revision R5 (written before the analyses were run)

Date: 5 November 2026. Committed and pushed to GitHub before any R5 code touches the real data.
Responds to two further internal (pre-submission) reviews of manuscript A. All R5 analyses are
exploratory. Student-cluster bootstrap, 2,000 resamples on real data unless stated.

## R5-A Verdict rule (fixed now; replaces the R4 rule)
One family-wise level, 95%, for every audit decision. Provenance of the plan is reported separately
and is not part of any check.
- Check 1 (per rule base): within-student (Study 1: within-cell and within-student) rank correlation of
  each of the k selected inputs, Bonferroni intervals at 1 − 0.05/k. Supported: every interval excludes
  0 with the selected sign. Contradicted: some interval excludes 0 with the opposite sign. Otherwise
  inconclusive.
- Check 2 (incremental): b_all = logistic regression on structural variables + persistence score +
  propensity score, fitted on the training cohort; gain = AUC(b_all + f) − AUC(b_all) on the test
  cohort, f entered as one extra column. Pass: lower 95% bound > 0. Fail: upper 95% bound < 0.01
  (smallest gain of interest 0.01 AUC, chosen now). Otherwise inconclusive. Incremental gains over
  each baseline alone (b + f vs b) are reported as secondary.
- Check 3 (statistical transfer): pass if the lower 95% bound of the held-out AUC > 0.5, fail if the
  upper bound ≤ 0.5, otherwise inconclusive.
- Overall: behavioural if all three pass; not behavioural if any check fails or is contradicted;
  inconclusive otherwise.

## R5-B Real data, Studies 2 and 3 (five splits)
(i) The verdict of R5-A for SC-TSK, SM-TSK, RF-TSK (R5-D), FURIA on the three SC-TSK inputs and
FURIA on all candidates (Weka 3.8.6 with fuzzyUnorderedRuleInduction 1.0.2 from Maven Central, default
options, class-1 distribution as score). (ii) Study 2: the persistence and propensity comparisons
restricted to queries that are not a student's first. (iii) Rule support: number of rules with
support below 5, 10, 20 and 30; 95% student-cluster bootstrap intervals (500 resamples) of every
SM-TSK consequent for D1 → D2 and SW → AW + IRW; a rule is unstable if its interval is wider than 0.5.
(iv) SLSQP from 10 random starts in [0, 1]^8: maximum absolute difference of the SM-TSK consequents
from the reported solution.

## R5-C Study 1 with a residualised label
y_res = y − mean of y in its semester × assessment-group cell; high risk = top quartile of y_res in the
training semester. Cross-semester AUC with interval for CF-TSK (refitted), LR8 and group-only.

## R5-D RF-TSK, a construction aligned with the audit
(1) p0 = b_all of R5-A on the training cohort. (2) Inputs: the three SC-TSK inputs (transformed as in
Algorithm 2) minus the mean of the same student's earlier requests in the session or deployment (0 for
the first). (3) Two-term quartile ramps on these deviations, eight rules, consequents fitted by bounded
least squares on the residual y − p0, bounds [−1, 1], prior 0, α = 1. (4) Score p0 + s_res.
Check 2 for RF-TSK compares its score with p0.

## R5-E Simulation
Same generator as R4-E, 100 replications per condition (seeds 0–99), 500 bootstrap resamples.
(i) Grid: confound {structure, propensity, persistence} × β {0, 0.5, 1} × requests {3–5, 6–10}.
(ii) Violations, with a combined structure + propensity confound, β ∈ {0, 0.5}, 6–10 requests:
non-monotone effect (logit term β(e² − 1)); threshold effect (2β·max(e, 0)); label noise (15% of
labels flipped); measurement error (observed x1 = x1 + N(0, 1)); three sessions per student;
correlated non-causal candidate (x2 = 0.7 a_s + N(0, 0.7)); 20% of requests missing at random;
time-varying effect (β · 2t/n at position t of n).
Recorded per condition: the R5-A verdict (three outcomes) for SC-TSK; a student fixed-effects
comparator (conditional logistic regression of y on x1, behavioural if the 95% interval excludes 0
with the selected sign); held-out AUC of SC-TSK, SM-TSK and SM-TSK with the direction of x1 reversed.

No other analysis will be run for this revision.
