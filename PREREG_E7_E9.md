# Analysis plan for experiments E7–E9 (written before they were run)

Date: 4 November 2026. Committed to the repository before any E7–E9 code was executed on the
real units. E1–E6 had already been run; this plan responds to E6, which showed that the
antecedent of the main expert rule (FR1, verbatim acceptance) has the opposite sign to the one
assumed. The design is therefore informed by results on the same data and is reported in the
manuscript as a follow-up analysis, not as part of the original pre-specified test.

## Revised rule base: outcome-checked TSK (OC-TSK)

* Inputs: v, u, p, e as before, plus
  * w — share of writing_request turns in the unit,
  * c — share of conceptual_questions turns in the unit.
  Both are motivated by the source study [2]: the writing-delegation cluster scored lower on
  exams, and conceptual questions were associated with better exam outcomes. Disclosure: the
  ablation in Section 7.5 (full data) also points to these two categories.
* Rules: FR1–FR5 unchanged in antecedent, plus
  * FR6: w HIGH and c LOW → risk (expert consequent 1.0)
  * FR7: c HIGH → no risk (expert consequent 0.0)
* Memberships: ramps between the training-fold 25th and 75th percentiles (unchanged).
* Consequents learned on the training fold only: z = argmin ||A z + b − ybin_tr||² + α||z − z_expert||²,
  subject to 0 ≤ z ≤ 1, with A = W / (ΣW + w0), b = w0 z0 / (ΣW + w0), α = 1.0 fixed in advance.
  The test fold is never used to fit z or the breakpoints.

## Rule injection: interval implication loss

For every rule j with firing w_j and consequent z_j, the Łukasiewicz reading of
"if the antecedent holds then risk ≈ z_j" gives the bounds w_j z_j ≤ r ≤ 1 − w_j (1 − z_j).
Loss = mean over rules and units of relu(w_j z_j − r) + relu(r − 1 + w_j (1 − z_j)).
For z = 1 and z = 0 this reduces to the raising and lowering terms used in E1. λ = 1, 15 epochs,
same Transformer, same folds and seeds (0–9) as E1.

## Aggregate baseline and fusion

* LR: logistic regression (StandardScaler + LogisticRegression, C = 1, max_iter 1000) on the
  eight broad dialogue-act counts, fitted on each training fold of the same split.
* Fusion score: mean of the out-of-fold percentile ranks of the two scores being fused.

## Comparisons (ten paired seeds; AUC = mean over the five test folds)

Primary family, Holm-adjusted over four Wilcoxon signed-rank tests:

1. Transformer + interval implication (OC-TSK) vs Transformer λ = 0
2. Fusion(Transformer λ = 0, OC-TSK score) vs Transformer λ = 0
3. Transformer + interval implication (OC-TSK) vs Transformer + implication (expert rules, E1)
4. Fusion(LR, OC-TSK score) vs LR

Secondary (descriptive, with 95% student-cluster bootstrap CI, seed-0 folds):

* standalone AUC of OC-TSK vs the expert fuzzy rule score
* learned consequents per fold (reported in full, so sign reversals are visible)

Decision rules are those of Section 6.3: a difference is significant only if the Holm-adjusted
p < 0.05; a rule-score difference is reliable only if its 95% CI excludes 0. Whatever the
outcome, all four comparisons are reported. No other variant (rule set, α, λ, epochs,
membership quantiles) will be tried on these data for this manuscript.
