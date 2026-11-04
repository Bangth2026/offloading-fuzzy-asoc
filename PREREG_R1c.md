# Addendum R1c, written after R1b and before the analysis below was run

Date: 4 November 2026. Committed before any R1c code touched the real units.

## Why

The within-group binary label pre-specified in R1-B and R1b is degenerate in one cell: in
Fall 2024 group 2, 84% of units have y = 0, so the cell's 75th percentile is 0 and the rule
"y ≥ threshold" marks every unit of that group positive in both semesters. The R1-B and R1b
results that use this label (within_group_label, both, two_by_two) therefore still carry a
group effect and are reported as such, not as group-free evidence. The direction check of R1b
(within-cell rank correlations with continuous y) is not affected.

## Analysis

For each cross-semester model already fitted under the pooled label — CF-TSK of E10, CF-TSK of
E11, LR8, LR3 on the E11 inputs, the author rule base FR1–FR5 — and for the seed-averaged LSTM and
Transformer predictions of R1-E, compute on the held-out semester the Spearman correlation between
within-cell ranks of the score and within-cell ranks of y (cells = assessment group), with a
95% student-cluster bootstrap interval (2,000 resamples). Both directions.

Reading: a model carries within-group signal if its interval excludes 0. This is the last
analysis of this revision.
