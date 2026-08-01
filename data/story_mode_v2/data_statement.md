# Narrativity pairs v2 — data statement

## Objective

Extract the narrativity/fictionality direction from a **request-free** contrast: a short fictional
narration against a length- and topic-matched non-narrative text. No base task, no `{request}`
slot, no harm content on either arm.

This replaces `../story_mode/` for extraction. That set embedded the 100 JBB requests inside 65
narrative wrappers and 5 matched non-narrative frames; its audit (`../story_mode/audit/`) found
request-fit strain, period conflict against modern-technology requests, rival-direction leakage
(persona, consequence, eval), and a structural asymmetry at the read position — the narrative arm
ended on a transitive writing verb with a human agent, all negative arms on a meta-statement about
the text. Dropping the request removes all four.

**`../story_mode/` is not deleted.** It is the transfer check: extract here, read out there. High
AUROC → the axis is request-invariant. Collapse → the request slot carried the signal.

## Contents

| file | rows | |
|---|---|---|
| `pairs.jsonl` | 50 | train |
| `pairs_heldout.jsonl` | 15 | held-out |
| `narrativity_pairs.csv` | 65 | wide: one row per pair, both prompts rendered |
| `narrativity_texts.csv` | 130 | long: one row per text, `label` 1 = narrative |

`build_dataset.py` renders the CSVs; `verify_pairs.py` audits the JSONL. Preamble
`Continue the text below.` + `\n\n` is prepended to both arms at render time and is not stored in
the JSONL. **Held-out context families are disjoint from train**, so held-out AUROC measures topic
generalisation as well as style generalisation.

## The contrast

Fictionality is **inside** the positive arm, by design — all 65 narratives are invented. The axis is
therefore fiction-narrative against fact-non-narrative, not narrativity with fictionality factored
out. That entanglement is deliberate: the target construct is "this is made up", which is what a
narrative jailbreak asserts.

The operative distinction is **predicate type**, not participants or tense. Genericising a process
description does not make it non-narrative — "a balloon rose sideways and the cord came taut" is
still a temporally-ordered change of state with the participant deleted. The non-narrative arm is
built from copular, relational, classificatory, deontic and quantificational predication; the
narrative arm from individuated change-of-state events in temporal order.

Consequences for the style list: **no procedural style** (a procedure is ordered events in time
with a generic participant — a generic narrative, unfixable while remaining a procedure), and the
hard rung is **criterial**, not "generic account of the same events" — the categories and thresholds
used to classify outcomes in a domain, rather than the outcomes unfolding.

## Narrative arm

| mode | train | held-out | closes |
|---|---|---|---|
| `scenic_past` | 20 | 6 | anchor |
| `present` | 14 | 4 | tense (else past ≡ narrative) |
| `oral` — anecdotal, informal register | 10 | 3 | formality |
| `folktale` — legend/fable register | 6 | 2 | overt fiction marking |

Crosscut, not a mode: `realism` 24 realistic / 26 non-realistic (train), 8 / 7 (held-out). Both
halves are fiction; the split is the ablation handle carried over from v1's 25/25.

## Non-narrative arm

| style | train | held-out | releases → controls |
|---|---|---|---|
| `criterial` | 12 | 4 | **the hard rung** — shared lexicon, only individuation differs |
| `technical` — structure and relations of a device | 8 | 2 | concreteness |
| `taxonomy` — A differs from B | 6 | 2 | maximally stative |
| `topographic` — stative spatial description | 6 | 2 | **setting without events** |
| `statistical` — survey/report prose | 6 | 2 | **animacy** — people, no events |
| `encyclopedic` — definitional | 5 | 1 | far-negative anchor |
| `spec` — requirements, deontic | 4 | 1 | formal register |
| `essay` — argumentative/analytic | 3 | 1 | opinionated register |

`technical`, `topographic` and `statistical` close **concreteness** and **animacy**, both of which
v1's data statement lists as *not controlled* / *deferred*. That is the main gain from dropping the
request slot.

`negation` (v1's diagnostic style, asserting the absence of narrative in narrative vocabulary) is
dropped. `criterial` is the better lexical-detector check: it shares the vocabulary rather than
denying it.

## Achieved controls

| | train | held-out |
|---|---|---|
| words, narrative | 63–75 (mean 69.1) | 61–72 (mean 67.8) |
| words, non-narrative | 64–72 (mean 68.4) | 62–72 (mean 67.9) |
| max \|Δ\| within pair | 3 | 3 |

- **Topic matched within pair**, so topic cancels in the diff-in-means rather than only in
  expectation.
- **Tense polarity crossed.** Narrative 36 past / 14 present; non-narrative 25 past / 25 present
  (train). Per pair: `default` 23 (past narr / present non-narr), `both_past` 13, `reversed` 12
  (present narr / past non-narr), `both_present` 2. Tense cannot be a monotone confound.
- **Both arms end on a full stop**, mid-text, continuable. The final token is identical across all
  130 prompts — the direct fix for the v1 read-position asymmetry.
- **Third person throughout, both arms.** No second-person address, no first person (hard-checked).
  First-person fiction would mark invention more strongly but collides with the persona axis, which
  is the rival H1 most has to exclude. Cost: further from jailbreak surface form, where second-person
  imperatives are the norm.
- **No harm content on either arm**, so the vector cannot carry a −harm component by construction
  and the plan §6 benign-task control is satisfied trivially.
- Non-narrative arm kept **non-assertive about contested empirical facts** — definitions, specs,
  taxonomies, structural descriptions — so the negative pole does not become a truth axis
  (plan §11, "realness = truth direction").

## Criterial overlap

Shared content-word types per pair, narrative ∩ non-narrative:

| | n | count | fraction of narrative types |
|---|---|---|---|
| `criterial` | 16 | 8–14 (mean 9.0) | 0.28–0.42 (mean 0.34) |
| other styles | 49 | 1–13 (mean 4.1) | 0.03–0.42 (mean 0.16)

An earlier target of ≥60% was wrong and is not achievable: removing eventive predicates necessarily
removes the narrative's event verbs, which are a large share of its content words. Overlap is on
**nouns**, and the floor enforced is ≥8 shared types on `criterial` pairs.

**Report the 16 criterial pairs' AUROC as its own number.** Length, register, concreteness and
lexicon are all matched there, so separation must be narrativity. High on the other 49 and chance on
these 16 means the probe is reading surface form — averaging that in would hide it.

## Residual soft findings

`verify_pairs.py` reports 15, all reviewed and kept:

- 11 functional or metaphorical predicates in non-narrative arms (`carries` of a vibration,
  `a stopped train`, `reaches` of a standard, `were left alone`). Generic-functional, not
  event-sequential.
- 2 discourse rather than temporal uses of `then` / `next`.
- 2 month names mid-sentence (`July`, `August`, `October`) inside seasonal statements — not
  individuating.

## Not controlled

- **Second person and imperative mood.** Absent from both arms; real jailbreaks are second-person
  imperatives inside fiction. Whether the axis transfers is H2 (spec §3), not settled here.
- **Fictionality, separately from narrativity.** Entangled on purpose. Separating them would need a
  third arm per item (factual narration), which is the design this one was chosen over.
- **The negative arm is not neutral.** With matched framing on both sides the axis is narrativity
  *with register held*, not story-versus-no-framing.
- **`statistical` imports measurement vocabulary** (`sample`, `median`, `percent`) — the same lexical
  field as the eval-awareness axis. 8 pairs, so a small +eval component on the negative pole is
  possible; check `cos(story, eval)` against the per-style breakdown before trusting the pooled
  vector.
