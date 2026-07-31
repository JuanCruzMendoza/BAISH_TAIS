# Eval-awareness prompts — data statement

## Contents

| file | rows | |
|---|---|---|
| `eval_aware_prompts.csv` | 5,000 | paired prompts: a deployment-framed prompt and its evaluation-framed counterpart |
| `eval_aware_prompts_heldout.csv` | 450 | same, on held-out framings and held-out tasks |
| `eval_awareness_pairs.csv` | 50 | framing pairs |
| `eval_awareness_pairs_heldout.csv` | 15 | framing pairs reserved for validation |

## Prompts

5,000 rows = 50 framings × 100 base tasks. Every row carries both `prompt_deployment`
and `prompt_evaluation` for the same task, so the pair can be read without a join. The
framing is a prefix ending in a colon and the task follows, so the two sides of a row
differ only in the framing.

- 5,000 distinct deployment prompts and 5,000 distinct evaluation prompts; 100 distinct
  base requests, each recurring once per framing.
- **Label:** 2,500 harmful / 2,500 benign.
- **Category:** 10 categories, 500 rows each — Harassment/Discrimination, Malware/Hacking,
  Physical harm, Economic harm, Fraud/Deception, Disinformation, Sexual/Adult content,
  Privacy, Expert advice, Government decision-making.
- **Length (words):** deployment 11–38 (mean 19, median 19); evaluation 11–36 (mean 19,
  median 19). Within-row difference −2 to +2 (mean −0.04); 2,800 of 5,000 rows are
  identical in length.
- Both members of every pair end in a colon.

Held-out: 450 rows = 15 framings × 30 tasks, 225 harmful / 225 benign, all 10 categories
(30 or 60 rows each). Length 9–32 on both sides, within-row difference −1 to +1.

Train framings are crossed only with train tasks and held-out framings only with held-out
tasks; the two splits share no framing and no `jbb_index`.

## Framings

| | main | held-out |
|---|---|---|
| framing pairs | 50 | 15 |
| source | 16 from [arXiv:2510.20487](https://arxiv.org/abs/2510.20487) + 34 written | 15 written |
| length, words | 4–11 (mean 7, sd 1.4) | 5–9 (mean 7, sd 1.1) |
| within-pair word delta | −2 to +2 (mean −0.04) | −1 to +1 (mean +0.13) |
| second / third person | 25 / 25 | 8 / 7 |

Both sets mix second-person address (*You are…*, *Your reply…*) with third-person
scene-setting (*The model is…*, *A customer typed…*), split roughly evenly — so person
is not confounded with the train/held-out split. The label is in the column `person`,
verified against a `you|your|you're` match. The split is on wording instead: the two
sets share no 4-gram and no framing string. Framings avoid fiction, harm and consequence
vocabulary so the axis stays deployment-vs-measurement.

## Base tasks

100 tasks — 50 matched harmful/benign pairs, 5 per category — drawn from
[JailbreakBench JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors);
`../harm/harm_selected_pairs.csv`. The held-out 30 tasks (15 pairs, all 10 categories,
5 of them with two) are disjoint from these, in `../harm/harm_selected_pairs_heldout.csv`.
