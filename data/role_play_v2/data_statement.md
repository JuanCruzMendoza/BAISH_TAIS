# Role-play pairs v2 — data statement

## Objective

Extract the persona/role-play direction from 1,000 pairs instead of v1's 50, by widening the
**role** axis (200 of the 275 assistant-axis roles, each with all 5 of its paraphrases) rather
than crossing 50 roles with 100 requests. v1's 5,000-row crossed table bought nothing the
50-pair construction did not already give (`docs/extraction/insights.md`), so the extra rows here go
into framing variety, which is what the vector averages over.

It also closes v1's two pole defects: the negative pole was **one** string (or 5 at 10× reuse),
perfectly label-aligned in whatever is idiosyncratic to it, and the role side was 50 framings of
one speech act. Here both poles carry a pool.

`../role_play/` is not deleted — it is the transfer check: extract here, read out there.

## Contents

| file | rows | |
|---|---|---|
| `pairs.jsonl` | 800 | train |
| `pairs_heldout.jsonl` | 200 | held-out |
| `roles.jsonl` / `roles_heldout.jsonl` | 160 / 40 | curated roles, 5 framings each |
| `roles.csv` / `roles_heldout.csv` | 800 / 200 | one row per (role, paraphrase) |
| `roleplay_v2_pairs.csv` | 1,000 | wide: one row per pair, both prompts rendered |
| `roleplay_v2_prompts.csv` | 2,000 | long: one row per prompt, `pole` 1 = role |

`select_roles.py` curates the roles, `build_pairs.py` builds and renders the pairs,
`verify_pairs.py` audits them (0 failures). One pair =

```
<framing>\n\n<request>
```

on both arms with the **same** request, so the contrast is framing-only while the read position
sits after a request (spec 0.2(a)). The separator must stay byte-identical to
`experiments/common/prompts.py::with_task`. Nothing is pre-rendered in the JSONL.

## Roles

200 of 275, all 5 paraphrases each — 1,000 pairs is what 200 × 5 lands on exactly. Two removal
lists, both explicit in `select_roles.py`:

| removed | n | |
|---|---|---|
| harm-adjacent | 21 | v1's list (hacker, criminal, smuggler, saboteur, anarchist, demon, virus, parasite, predator, prey) widened with addict, destroyer, fixer, gossip, provocateur, rogue, soldier, spy, vigilante, warrior, zealot |
| `assistant` | 1 | it is the negative pole |
| near-synonyms | 53 | one kept per cluster — 7 educators → tutor/teacher, 10 assessors → auditor/critic/judge, 6 coordinators → organizer, … |

Harm-adjacent roles are excluded because role-descriptor criminality would sit **only** on the
positive arm and be perfectly label-aligned, so the persona vector would carry a +criminality
component. The near-synonym cut stops one semantic cluster from dominating the pole mean.

| stratum | train | held-out |
|---|---|---|
| `professional` | 61 | 15 |
| `everyday` | 62 | 15 |
| `fictional` | 22 | 6 |
| `nonhuman` | 15 | 4 |

Framing words: train 11–29 (mean 16.6), held-out 10–24 (mean 16.1). Held-out is **all 15 of v1's
held-out roles** plus 25 more, and contains **no v1 train role**, so a v1 vector has never been
fitted on a v2 held-out role. All 50 v1 train roles survive curation and are in v2 train.
Several held-out roles are near-neighbours of a train role (curator≈archivist, mentor≈tutor,
wraith≈ghost, swarm≈hive, student≈teenager), so the split tests unseen roles rather than
distribution shift.

## Assistant pole

Every source string, pooled: `assistant.json[0–4]`, `default.json[1,2,4]` (`[0]` is empty, `[3]`
is `You are {model_name}.`), and v1's 5 `NEG_POOL` recombinations — 13 variants, read from
`../role_play/roles.jsonl` where possible so the wording cannot drift. The held-out pool is 12
string-disjoint variants: v1's 3 held-out paraphrases plus 9 authored on the same clause
inventory, in the same composition (~29% dispositional, ~8% bare, ~63% situational) so held-out
AUROC measures unseen wording rather than a shifted pole.

The pool means 13.0 words against a role mean of 16.6, so **usage is deliberately not uniform**:
`quotas` is the flattest weighting whose length mean matches the role side, and `rank_match`
then pairs the poles in length order. Uniform usage would leave the negative arm 3.6 words
shorter in nearly every pair — a constant-sign length confound, the one v1 spent its padded
pole fixing.

| | train | held-out |
|---|---|---|
| variants | 13 | 12 |
| uses per variant | 10–181 (max 23% of the pole) | 8–20 (max 10%) |
| effective n (1/Σp²) | 7.8 | 11.4 |
| distinct negative prompts | 690 / 800 | 200 / 200 |

v1's pole had effective n = 5.0 at 20% each. The 110 repeated negative prompts in train are
exactly the forced minimum (a variant used more than 100 times must reuse one of the 100
goals); no cell repeats otherwise, and the largest multiplicity is 3. A LOPO fit therefore
retains ≤ 2 copies of the held-out pair's negative (0.25% weight) against v1's 18%. **Group
LOPO by `neg_variant`** to remove it entirely.

## Requests

100 train goals (50 harmful + 50 benign, `../harm/harm_selected_pairs.csv`, each used 8×) and
30 held-out goals (15 + 15, each 6–7×). Same `GOAL_FIXES` as `../story_mode/`, so the task set
is shared with the story direction. Goals are byte-identical across a pair and carry no trailing
punctuation.

| | train | held-out |
|---|---|---|
| harmful / benign | 400 / 400 | 100 / 100 |
| harmful share by paraphrase index | 50% (all 5) | 50% (all 5) |
| harmful share by stratum | 49–50% | 49–51% |
| harmful share by variant | 49–53% (A7 40%, 10 uses) | 39–62% (8–20 uses) |

Labels come from `(role_idx + para_idx)` parity, which forces the first two rows exactly and
leaves the third within ±1%. The fourth is then repaired by swapping variants **within a length
tie**, which changes no paired gap — unrepaired it ran 27–63%, since variant follows framing
length and label follows paraphrase index, and the two correlate. No goal repeats under one role.

## Achieved controls

| | train | held-out |
|---|---|---|
| paired gap, framing words (role − assistant) | −3 to +9, mean **+0.01** | −2 to +9, mean **+0.03** |
| \|gap\| ≤ 1 | 85% of pairs | 54% |
| role longer in | 13% of pairs | 26% |
| full prompt | 16–50 words (mean 29.0) | same file |

- **Request identical within pair**, so request content and the harm label cancel in the
  diff-in-means rather than only in expectation, and both arms end on the same token.
- **Mean framing length matched to two decimal places**, with the sign of the per-pair gap
  varying (v1's criterion: the confound was the sign alignment, not the size).
- **No harm content in any framing**, on either arm.
- **Both poles are pools**, so neither is one string's idiosyncrasy.

## Not controlled

- **Speech act.** The role side is 38% declarative / 62% imperative (`You are …` vs `Please be
  …` / `Act as …` / `Embody …`); the assistant side is 80/20 (train), 76/24 (held-out), matched
  within pair in only 42–44%. The four imperative source strings average 11.25 words against the
  role side's 16.6, so form matching and length matching are in direct conflict and length won.
  `role_form` / `neg_form` are columns: report the form-matched subset's AUROC separately before
  trusting the pooled vector.
- **Realistic vs non-realistic is 77/23**, not v1's 50/50. Only 47 of the kept roles are
  fictional or non-human, so a balanced split would cap the set at ~90 roles. The ablation
  handle survives as 185 non-realistic pairs in train (110 `fictional` + 75 `nonhuman`),
  reportable per stratum, but it is no longer a balanced crosscut.
- **Dispositional content on the negative pole.** `assistant.json`'s five paraphrases carry
  helpfulness content (`prioritizes being helpful, informative, and supportive`), 28% of the
  train pole. v1 kept this off the pole deliberately, on the grounds that a −helpfulness
  component cancels persona's causal signal. Kept here because the brief was to pool all source
  strings; `neg_source` isolates the 225 pairs if the pooled vector looks blunted.
- **Length blindness.** v1's persona vector read the length foil at 0.97 with a +0.2-word pole
  gap, so matching the framing means is necessary but demonstrably not sufficient.
- **Second person and imperative mood are on both arms**, as in v1 — the axis is persona with
  address held, not role-play versus no framing.

## Pipeline

No loader yet. `views.py::_persona` still reads v1's `roles_subset_paraphrased.csv` and attaches
tasks at load time via `--append-task`; a `persona_v2` entry would instead read `pairs.jsonl` and
render `with_task` on both arms, with `append_task` off because the request is already in the
pair. Grouped LOPO by `role` (200 folds of 5) and by `neg_variant` are both wanted here — pair-wise
LOPO is not out-of-sample, since the held-out pair's role appears in 4 others.
