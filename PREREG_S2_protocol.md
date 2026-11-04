# Study 2 protocol (pre-registration draft): behavioural signal of cognitive offloading within assessment groups

Status: draft protocol, 4 November 2026. Written after the audit R1–R1c and before any Study 2
analysis. To be frozen (tagged release + Zenodo DOI) before any confirmatory data are scored.
Target venue: Applied Soft Computing.

## 1. What the audit established (Study 1, E1–E12 and R1–R1c)

1. The weak label of Study 1 (relative assisted–unaided gap, clipped at 0, top quartile pooled over
   assessment groups) is dominated by assessment structure. With the Fall 2024 threshold the
   high-risk rate is 0.47 / 0.00 / 0.31 in groups 1–3 (Fall 2024) and 0.45 / 0.13 / 0.16
   (Spring 2025). A model that knows only the group reaches AUC 0.699 (Fall → Spring), 0.780
   (Spring → Fall) and 0.730 in pooled CV, above every behavioural model.
2. Within groups, no behavioural input is associated with the gap: within-cell Spearman
   correlations of shares (v, u, p, e) and volumes (log W, log C, log P) all have 95% intervals
   containing 0 (largest |ρ| = 0.078). The "wrong-sign" expert rule FR1 was a between-group effect.
3. Held-out predictions of all cross-semester models carry at most weak within-group signal
   (ρ = 0.02–0.14; only one interval, Transformer Fall → Spring, excludes 0).
4. Clipping at 0 makes the label degenerate where the exam is easier than the assignments
   (84% zeros in Fall 2024 group 2).

Study 2 therefore changes the outcome, the estimand and the analysis, and treats StudyChat as
exploratory only.

## 2. Research question

Do turn-level dialogue behaviours, summarised by an interpretable fuzzy rule base, predict a
student's assisted–unaided gap **within** an assessment group, beyond assessment structure, prior
unaided performance and overall activity?

## 3. Outcome

* d = llm − no_llm (assignment average minus unaided exam, both on the release's normalised scale),
  **not clipped**.
* Primary outcome: z = (d − mean_cell) / sd_cell, cell = (semester or cohort, assessment group),
  with cell mean and SD taken from the training cohort when a model is transferred.
* No dichotomisation in the primary analysis. Secondary: top quartile of z within cell, using
  rank-based quartiles so that ties cannot make a cell degenerate.

## 4. Predictors

* Volumes: log(1 + count per assignment) of the eight broad dialogue acts.
* Covariates (always included, never counted as signal): cell indicators; prior unaided exam where
  it exists (groups 2 and 3); log total turns per assignment.

## 5. Models

* **Proposed — sign-checked TSK regression (SC-TSK).** Zero-order TSK on three inputs selected
  inside the training cohort by within-cell rank correlation with z; 2 × 2 × 2 grid with
  quartile ramps; consequents fitted to the residual of z after the covariates, by bounded least
  squares with a ridge pull toward 0 (α fixed in advance); an input enters only if the sign of
  its within-cell association is the same in two halves of the training cohort split by student
  (the "sign check" that the audit shows is needed before rules are trusted).
* Comparators: covariates only; covariates + linear volumes (8 and 3 inputs); ANFIS-style TSK;
  Wang–Mendel; shallow regression tree; EBM; LSTM on turn sequences with covariates appended.

## 6. Confirmatory hypotheses (to be tested once on a new cohort)

* H1 (signal): on the confirmation cohort, the within-cell Spearman correlation between the
  SC-TSK prediction and z is positive, with the lower bound of the 95% student-cluster bootstrap
  interval above 0.
* H2 (incremental): SC-TSK + covariates explains more within-cell variance than covariates alone;
  95% interval of the ΔR² above 0.
* H3 (interpretability at no cost): SC-TSK is non-inferior to the best non-fuzzy comparator on
  within-cell ρ with a margin of 0.05.
* Multiplicity: Bonferroni over H1–H3 (α = 0.0167).

## 7. Sample size

Minimum detectable within-cell correlation at 80% power, two-sided α = 0.05, with a design
effect for ~2.6 units per student:

| Test set | Units | Students | MDE ρ (ICC 0) | MDE ρ (ICC 0.3) |
|---|---|---|---|---|
| StudyChat Spring 2025 | 281 | 110 | 0.17 | 0.20 |
| StudyChat Fall 2024 | 183 | 65 | 0.21 | 0.26 |

The audit's largest within-cell |ρ| for a single input was 0.078. Detecting ρ = 0.08 needs about
1,220 effective units; ρ = 0.10 needs about 780. **StudyChat alone cannot confirm an effect of
the size the audit suggests.** A confirmatory test needs a new cohort of at least ~300 students
(~800 units) with the same assignment/exam structure.

## 8. Data plan

* Exploratory (development) data: StudyChat Fall 2024 and Spring 2025. All model choices, α,
  input selection rule and code are fixed on these and frozen with a tagged release.
* Confirmation data, in order of preference:
  1. a later StudyChat release from the same course, if published;
  2. a prospective cohort in an SIU programming course with consented collection of
     student–LLM dialogues, assignments and unaided exams (requires ethics approval, since this
     is primary data collection, unlike Study 1);
  3. pooled data from two or more courses, with course added to the cell definition.
* The confirmation data are scored once with the frozen release.

## 9. What Study 2 can contribute to ASOC

A sign-checked, residualised TSK regression is a concrete soft-computing contribution motivated
by a documented failure mode (rules that encode between-group structure rather than behaviour).
The paper would report: the audit as motivation; the method; development results on StudyChat as
exploratory; the confirmatory test on new data. Without new data, the honest publishable unit is
the audit itself.
