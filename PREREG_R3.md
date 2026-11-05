# Analysis plan for revision R3 (written before the analyses were run)

Date: 5 November 2026. Committed and pushed to GitHub before any R3 code touches the real data.
Responds to a second internal (pre-submission) review of manuscript A (items M1–M10, minor 1–16).
Studies 1–3 and R2 are not re-run. All R3 analyses are exploratory. Unless stated: 95% student-cluster
bootstrap intervals (2,000 resamples), the five splits of Studies 2 and 3 (D1→D2, D2→D1, D1+D2→D3,
SW→AW+IRW, AW+IRW→SW), selected inputs as in R2.

## R3-A Within-student check, estimator and model-based alternative (M2)

(i) Counts behind the R2 within-student rank correlation, for every selected input and split: students
with ≥ 2 requests, of these the students with both classes, and the requests they contribute.
(ii) Conditional (fixed-effects) logistic regression by student, y ~ x for each selected input
separately, x standardised on the set used: coefficient, 95% Wald interval, number of informative
students; on training and test sets. Signs are compared with the R2 rank correlations.

## R3-B Stability of the sign check over random splits (M3)

For each split, the split-half sign check of Algorithm 2 is repeated with seeds 1–200. Reported: share
of seeds in which each candidate is eligible; frequency of each selected three-input set; median and
2.5th–97.5th percentiles of the held-out AUC of SC-TSK. Stability-selection variant (chosen now): an
input is eligible if it is eligible in ≥ 80% of the 200 seeds; the top three such inputs by
|AUC − 0.5| form the grid; held-out AUC with interval.

## R3-C Structure/behaviour boundary (M4)

Criterion (stated now): a variable is structural if its value is fixed by the course design, the
dataset's record keeping or the clock independently of how the student works (course, week, session,
position in session, elapsed time); it is behavioural if it results from the student's own actions.
Under this criterion `test_progress` (Study 2) is behavioural. Alternative partition A1: Study 2
structure = {is_first_query, query_index, time_since_session_start_s}. Reported for the three Study 2
splits: AUC of structure-only, H2 gain of structure + SC-TSK over structure, with 98.33% interval
(as in Study 2), and SC-TSK minus structure-only with 95% interval.

## R3-D Study 3 without true lagged labels and conditional on persistence (M6)

(i) Label noise: every earlier request's production/guidance label is flipped independently with
probability e ∈ {0.1, 0.2, 0.3} before computing `prev_production_share` and
`prev_request_production`, in training and test sets alike (current labels unchanged); SC-TSK is
re-selected and re-fitted; persistence is recomputed from the noisy labels. 20 repetitions per e
(seeds 0–19); mean and range of held-out AUC for SC-TSK and persistence, both Study 3 splits.
(ii) Conditional comparison: test requests with no earlier production request in the session
(`prev_production_share` = 0). AUC of the SC-TSK score and of each of its other two inputs within
this subset, with intervals; numbers of requests and positives.

## R3-E Consequents of weakly supported rules (M7)

SC-TSK refitted with α = N/10 (N = training requests; a prior worth 10% of the sample per rule) in all
five splits: held-out AUC with interval and the consequents of cells with support < 10. Rule tables
in the figures will mark cells with support < 10.

## R3-F Output monotonicity and small checks (minors 4, 5, 6, 11)

(i) Output monotonicity on test data: for each test request and each selected input i, the transformed
input is increased by δ = 0.1 × (training 75th − 25th percentile, or 10⁻³ if zero); a violation is
d_i · (s(x + δ e_i) − s(x)) < −10⁻⁹. Share of violations for SC-TSK and SM-TSK, every split.
(ii) Number of test requests receiving the default 0.5 for persistence and for propensity, every split.
(iii) Within-student rank correlation (R2 estimator) of the structure-only and all-feature regression
scores with y on the test sets, every Study 2 split.
(iv) The w0 × α grid of R2-G is reported for all five splits from r2_stats.json (no new run).

## R3-G Synthetic validation of the audit (M10)

Data are simulated with a known generating process; no real data are used. Each cohort: 100 students,
each with one session of 3–8 requests (uniform), students assigned at random to three structural cells
with logit offsets (−1, 0, 1). Candidate inputs x1, x2, x3 (x2, x3 pure noise). Scenarios:
- S0 structure only: logit p = cell offset; x1 = cell offset + N(0, 1). No behavioural effect.
- S1 student propensity: logit p = a_s, a_s ~ N(0, 1); x1 = a_s + N(0, 1) per request. No within-student effect.
- S2 behavioural effect: logit p = cell offset + a_s + 1.0 · e, x1 = cell offset + a_s + e, e ~ N(0, 1)
  per request. x1 has a true within-student effect.
- S3 persistence: y_1 ~ Bernoulli(0.4); P(y_t = 1 | y_{t−1}) = 0.75 if y_{t−1} = 1, else 0.25;
  x1 = share of earlier positives in the session (0 if none) + N(0, 0.1). No effect beyond persistence.
Per replication a training and a test cohort are generated, SC-TSK (Algorithm 2) is built on the
training cohort from {x1, x2, x3}, and two verdicts about x1 are recorded:
- naive: x1 is among the selected inputs and the lower 95% bound of the held-out AUC of SC-TSK > 0.5;
- audit (Algorithm 1): x1 is among the selected inputs, its within-cell and within-student rank
  correlations on the training cohort have 95% intervals that exclude 0 with its selected sign, and
  the lower 95% bounds of AUC(SC-TSK) − AUC(b) on the test cohort exceed 0 for b = structure
  (logistic regression on cell indicators), persistence and propensity.
Correct verdict: x1 behavioural in S2, not behavioural in S0, S1, S3. 200 replications per scenario
(seeds 0–199), bootstrap with 200 resamples inside the simulation for speed. Reported: share of
replications with each verdict per scenario.

No other analysis will be run for this revision.
