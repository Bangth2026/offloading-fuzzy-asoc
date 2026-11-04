# Addendum R1b to the revision plan, written after R1 and before the analyses below were run

Date: 4 November 2026. Committed before any R1b code touched the real units.

## Why

R1-B showed that the pooled weak label is dominated by the assessment group: with the Fall 2024
threshold, the high-risk rate is 0.47 / 0.00 / 0.31 for groups 1–3 in Fall 2024 and 0.45 / 0.13 /
0.16 in Spring 2025, and a model that knows only the group reaches AUC 0.699 (Fall → Spring) and
0.780 (Spring → Fall), above every behavioural model. Every behavioural conclusion drawn with the
pooled label may therefore partly reflect group structure. R1b re-checks the main conclusions
with the label thresholded within each (semester, group) cell (training-semester thresholds per
group, as in R1-B), which removes between-group differences by construction.

## Analyses (95% student-cluster bootstrap intervals, 2,000 resamples)

1. Group-only baseline under the pooled protocol of Table 3 (pooled label, StratifiedGroupKFold(5),
   seed 42, logistic regression on one-hot group): AUC, for context only.
2. Direction check within groups: for v, u, p, e (shares) and log W, log C, log P (volumes), the
   Spearman correlation between within-cell ranks of the input and within-cell ranks of y
   (cells = semester × group), with a cluster bootstrap interval. Reading: a direction is
   established if the interval excludes 0.
3. The 2 × 2 design under the within-group label, cross-semester, both directions:
   author rules on shares (FR1–FR5, Section 5.3); learned consequents on shares (OC-TSK, Section
   7.8); author-direction consequents on volumes (EC-TSK, R1-A); learned consequents on volumes
   (CF-TSK on W, C, P). AUC with interval. Reading: a cell transfers if its lower bound exceeds 0.5.
4. The eight learned CF-TSK consequents (W, C, P) under the within-group label, both directions.

These are the only additional analyses for this revision.
