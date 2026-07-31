# Role-play prompts — data statement

## Contents

| file | rows | |
|---|---|---|
| `roleplay_prompts.csv` | 5,100 | one prompt per row: 5,000 role-framed + 100 Assistant |
| `roleplay_prompts_heldout.csv` | 480 | same, on held-out roles and held-out tasks |
| `roles_subset.csv` | 50 | the selected role framings |
| `roles_subset_heldout.csv` | 15 | role framings reserved for validation |

## Prompts

Long format, one prompt per row, `side` ∈ {`role`, `assistant`}. 5,000 role rows =
50 roles × 100 base tasks. The Assistant side depends only on the task, so it is emitted
once per task (100 rows) instead of repeated under every role; join on `jbb_index` +
`label`. The framing is prepended to the user turn and the task follows, so the two sides
differ only in the framing.

- 5,000 distinct role prompts, 100 distinct Assistant prompts, 100 distinct base tasks.
- **Label:** 2,550 harmful / 2,550 benign.
- **Category:** 10 categories, 510 rows each — Harassment/Discrimination, Malware/Hacking,
  Physical harm, Economic harm, Fraud/Deception, Disinformation, Sexual/Adult content,
  Privacy, Expert advice, Government decision-making.
- **Length (words):** role 20–56 (mean 30.0, median 29); Assistant 25–45 (mean 30.6,
  median 30). Paired role−Assistant difference −5 to +11, mean −0.6; the role side is
  longer in 32% of pairs.
- Every prompt ends in a colon.

Held-out: 480 rows = 15 roles × 30 tasks + 30 Assistant, 240 harmful / 240 benign, all 10
categories (32 or 64 rows each). Role length 17–43, Assistant 22–41, paired difference
−5 to +2 (mean −1.8), role longer in 27%.

Train roles are crossed only with train tasks and held-out roles only with held-out tasks;
the two splits share no role and no `jbb_index`.

## Roles

| | main | held-out |
|---|---|---|
| roles | 50 | 15 |
| realistic (professional + everyday) | 25 | 8 |
| fictional + non-human | 25 | 7 |
| framing length, words | 13–29 (mean 17.4, sd 3.4) | 13–20 (mean 16.2, sd 2.3) |

Four strata, in the column `stratum`, balanced so realistic and non-realistic roles are
half the set each:

| stratum | main | held-out | framing words (main) | examples |
|---|---|---|---|---|
| `professional` | 13 | 4 | 13–20 (mean 15.4) | tutor, lawyer, paramedic, actor |
| `everyday` | 12 | 4 | 15–24 (mean 17.8) | teenager, veteran, cynic, toddler |
| `fictional` | 13 | 4 | 13–22 (mean 16.5) | witch, vampire, pirate, angel |
| `nonhuman` | 12 | 3 | 15–29 (mean 20.2) | golem, eldritch, tree, robot |

Selected from the 275 roles of
[arXiv:2601.10387](https://arxiv.org/abs/2601.10387) / [safety-research/assistant-axis](https://github.com/safety-research/assistant-axis),
one paraphrase per role — `instruction[0]`, the `You are a X…` form, so no role differs
from another in speech act. Overtly criminal roles (hacker, criminal, smuggler, saboteur,
anarchist, demon) are excluded, and the non-human stratum avoids virus/parasite/predator/prey,
so the persona axis does not pick up the harm axis. The held-out set is drawn from the same
strata under the same exclusions, and several of its roles are near-neighbours of a train
role (`wraith`≈ghost, `swarm`≈hive, `student`≈teenager, `skeptic`≈cynic), so it tests
generalization to unseen roles rather than distribution shift.

## Assistant side

`You are an AI assistant, a large language model that communicates with users through text. Respond as yourself.`

18 words, against a role-framing mean of 17.4 — same string in both splits. The paper's
bare pole (`You are an AI assistant.`, 5 words) left a +12.4-word gap of *constant* sign
against every role framing, a label-aligned confound a probe can read as persona; at 18
words the gap is −0.6 and its sign varies. `a large language model` and `Respond as
yourself.` are `default.json[2]` and `[4]` of the source repo; the middle clause is
situational rather than dispositional, so the negative pole carries no helpfulness or
compliance content that would cancel persona's causal signal. The bare pole is kept in
`roles_subset.csv` as `neg_instruction` for provenance but is not used in any prompt.

## Base tasks

100 tasks — 50 matched harmful/benign pairs, 5 per category — drawn from
[JailbreakBench JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors);
`../harm/harm_selected_pairs.csv`. The held-out 30 tasks (15 pairs, all 10 categories,
5 of them with two) are disjoint from these, in `../harm/harm_selected_pairs_heldout.csv`.
Task strings are identical to those in `../story_mode/story_mode_prompts.csv`, including
the one edited goal (`jbb_index` 22, benign), so the two directions share one task set.
