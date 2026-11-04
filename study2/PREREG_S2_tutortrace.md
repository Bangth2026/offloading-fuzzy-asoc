# Study 2 pre-registration: fuzzy rules for anticipating dependent AI help requests (TutorTrace)

Date: 4 November 2026. Committed before any relation between features and labels in TutorTrace was
examined. Only file schemas (column names, row counts) have been read.

## Data

TutorTrace (UIST '26; github.com/divadbaroon/TutorTrace_dataset_and_benchmark, branch `uist`):
introductory Python, four in-class deployments with an integrated AI tutor. Used here:
`dataset/observable_metrics/query_level/deployment_{1,2,3}_queries.csv` joined to
`dataset/query_labels/deployment_{1,2,3}_labels.csv` on (student_id, query_index).
Chat text and code payloads are not used. Licence: `croissant.json` states CC BY 4.0 while
`datasheet.md` marks the licence "TO CONFIRM"; the authors will be asked to confirm before
publication. Nothing from TutorTrace is redistributed; only code and derived statistics are.

## Outcome

y = 1 if the query is labelled *dependent* (asks for the answer or code), 0 if *guided*.
Labels are GPT-4o labels validated against humans for Deployments 1–2 (κ = .709/.690); Deployment 3
labels are unvalidated. Dependent help-seeking is the operational form of cognitive offloading at
the moment of asking.

## Features (all observable before or at the query)

Allowed: all `pre_*` columns, `thinking_time_s`, `implementing_time_s`, `debugging_time_s`,
`testing_time_s`, `seeking_help_time_s`, `thinking_task_s`, `thinking_llm_s`, `thinking_error_s`,
`thinking_code_s`, `time_since_session_start_s`, `time_since_last_query_s`, `is_first_query`,
`query_index`, and test progress at the query (`test_passed_at_query / test_total_at_query`).
Excluded as future or outcome information: `total_queries`, `task_completed`, `final_*`,
`session_duration_s`, `ai_response_length_chars`, all `post_*`, `label_query_no_effort`.
Excluded as properties of the query text itself: `query_length_chars` (secondary analysis only).
Missing values (event-conditioned metrics that do not apply) are set to 0 with an indicator only
where the comparator needs it; non-negative features enter the rule base as log(1 + x).

**Structure features** (the lesson of the StudyChat audit, where a structure-only model beat every
behavioural one): `is_first_query`, `query_index`, `time_since_session_start_s`, test progress.

## Models (fitted on the training deployment only)

* **SC-TSK (proposed): sign-checked TSK classifier.** Candidate behavioural features (allowed
  features minus structure features) are ranked by |AUC − 0.5| on the training deployment. A
  feature is eligible only if the sign of its association (AUC above or below 0.5) is the same in
  two halves of the training deployment split by student (seeded split, seed 0). The top three
  eligible features form a 2 × 2 × 2 grid of LOW/HIGH ramp memberships (training quartiles) with
  product t-norm; consequents fitted by bounded least squares with a ridge pull (α = 1) toward the
  training base rate (the CF-TSK estimator of Study 1).
* **Structure-only:** logistic regression on the structure features.
* **Structure + SC-TSK:** logistic regression on the structure features and the SC-TSK score.
* **Comparators on the same three inputs:** LR3, decision tree depth 2 (min leaf 10), Wang–Mendel,
  ANFIS-style TSK (as in PREREG_R1.md, R1-D), EBM.
* **Comparator on all allowed behavioural features:** L2 logistic regression (C = 1, standardised).
* Reference only: the original paper's query-type AUROC (.717, D1 → D2, different features).

## Splits

Primary: train D1, test D2 (same task, validated labels). Secondary: train D2, test D1; train D1+D2,
test D3 (different task, unvalidated labels). Each test deployment is scored once.

## Hypotheses (primary split; Bonferroni over three, α = 0.0167; 98.33% student-cluster bootstrap
intervals, 2,000 resamples)

* H1 transfer: lower bound of AUC(SC-TSK) above 0.5.
* H2 behaviour beyond structure: lower bound of AUC(structure + SC-TSK) − AUC(structure-only)
  above 0.
* H3 interpretability at no cost: lower bound of AUC(SC-TSK) − AUC(LR on all behavioural features)
  above −0.05.

## Also reported

Selected features and their sign-check outcome; the eight consequents (rule table); secondary splits;
comparators with 95% intervals; the same analyses with `query_length_chars` added to the candidates.

No other variant will be run on D2 or D3 for this study.
