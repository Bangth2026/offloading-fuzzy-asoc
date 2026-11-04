# Study 3 pre-registration: sign-checked fuzzy rules for production requests in EFL writing (RECIPE4U)

Date: 4 November 2026. Committed before any relation between features and the target was examined.
Read so far: column names, row counts, students per course, and the marginal distributions of
`intent_final`, `is_quiz`, `is_essay_edited` and `rating`.

## Data and terms

RECIPE4U (Han et al., LREC-COLING 2024): 1,913 student–ChatGPT exchanges from 130 students in three
EFL writing courses (SW 74 students, AW 30, IRW 26), obtained through the authors' consent form for
research use only. Nothing from the dataset is redistributed; only code and derived statistics are.
Essay and utterance texts are used only through their lengths, never their content.

## Target

Among student utterances whose `intent_final` is a request (label starting with "R "):
y = 1 for production requests that ask ChatGPT to write, rewrite or translate the student's text
(R Generation, R Revision, R Translation); y = 0 for guidance requests (R Information, R Evaluation,
R Language Use, R Confirmation). Non-request utterances are excluded. `is_quiz` (26 positives) is
too rare to serve as a target. The labels are the dataset's own intent annotations.

## Features (computed from earlier exchanges of the same session only)

Structure: course (one-hot), week, session, position of the utterance in the session (`idx`).
Behaviour before the utterance, from rows of the same (student, course, week, session) with smaller
`idx`: number of earlier student turns; share of earlier requests that were production requests;
number of earlier exchanges followed by an essay edit (`is_essay_edited`); mean earlier
satisfaction rating; mean length of earlier student utterances; mean length of earlier ChatGPT
replies; essay length at the previous exchange; essay growth from the first to the previous
exchange of the session. First utterances get zeros with the counts at zero.
Excluded: anything from the current or later exchanges (current `user`, `chatgpt_after`, `rating`,
`is_essay_edited`, `intent_final`), except the length of the current utterance, which enters only a
secondary analysis.

## Models and protocol

Identical to Study 2 (study2/PREREG_S2_tutortrace.md and study2/run_s2.py): SC-TSK with the
student-split sign check and top three eligible features; structure-only logistic regression;
structure + SC-TSK; logistic regression on all behavioural features; LR3, tree depth 2,
Wang–Mendel, ANFIS-style TSK, EBM on the same three inputs.

Splits (cross-course transfer): primary, train SW, test AW + IRW; secondary, train AW + IRW,
test SW. Each test set is scored once.

## Hypotheses (primary split; Bonferroni over three, α = 0.0167; 98.33% student-cluster intervals)

* H1 transfer: lower bound of AUC(SC-TSK) above 0.5.
* H2 behaviour beyond structure: lower bound of AUC(structure + SC-TSK) − AUC(structure-only) above 0.
* H3 non-inferiority: lower bound of AUC(SC-TSK) − AUC(LR on all behavioural features) above −0.05.

No other variant will be run on the test courses for this study.
