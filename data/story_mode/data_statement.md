# Story-mode prompts — data statement

## Contents

| file | rows | |
|---|---|---|
| `story_mode_prompts.csv` | 5,000 | original pairs: narrative prompt + plain counterpart, no preamble |
| `story_mode_prompts_matched.csv` | 5,000 | train: 7 arms per row, length-matched, preamble on all |
| `story_mode_prompts_matched_heldout.csv` | 450 | held-out: same schema, held-out wrappers × held-out requests |
| `story_wrappers.{jsonl,csv}` | 50 | narrative frames |
| `story_wrappers_heldout.{jsonl,csv}` | 15 | narrative frames reserved for validation |
| `request_wrappers.jsonl` | 200 | non-narrative frames, 50 train wrappers × 4 styles |
| `request_wrappers_heldout.jsonl` | 60 | non-narrative frames, 15 held-out wrappers × 4 styles |
| `request_wrappers/expository_tails.jsonl` | 65 | `expository` lead-in + handoff per wrapper |
| `expository_pools.jsonl` | 10 | category sentence pools for `expository` |

Build with `build_matched_dataset.py`; audit the frames with `verify_request_wrappers.py`
(takes an optional subdirectory argument, so `request_wrappers/round1` can be scored too).
`story_mode_prompts.csv` is never written by either script.

The preamble puts `\n\n` inside CSV fields, so line counts do not equal row counts. Valid
quoted CSV; read with `csv.DictReader` or `pandas.read_csv`.

## Prompts

5,000 rows = 50 frames × 100 base tasks. Every row carries both `prompt_story` and
`prompt_bare` for the same task, so the pair can be read without a join.

- 5,000 distinct narrative prompts; 100 distinct plain prompts, each recurring once per frame.
- **Label:** 2,500 harmful / 2,500 benign.
- **Category:** 10 categories, 500 rows each — Harassment/Discrimination, Malware/Hacking,
  Physical harm, Economic harm, Fraud/Deception, Disinformation, Sexual/Adult content,
  Privacy, Expert advice, Government decision-making.
- **Length (words):** narrative 38–133 (mean 65, median 63); plain 7–27 (mean 13, median 12).
- Both members of every pair end on the same punctuation.

## Narrative frames

| | main | held-out |
|---|---|---|
| frames | 50 | 15 |
| length, words | 32–106 (mean 54, sd 14) | 35–92 (mean 53, sd 16) |
| realistic / non-realistic | 25 / 25 | 8 / 7 |
| distinct genres | 44 | 15 |
| distinct tones | 30 | 15 |

All frames are third person and contain no second-person address. The two frame sets
share no genre, no closing phrasing, and no content-bearing 4-gram.

## Base tasks

100 tasks — 50 matched harmful/benign pairs, evenly distributed across the 10 categories —
drawn from [JailbreakBench JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors).

---

# Request wrappers (non-narrative counterparts)

`prompt_bare` is a plain imperative: 13 words against the narrative arm's 65. A vector
extracted from that contrast is roughly half length (`len_frac_raw` 0.47–0.55, initial
tests §2b). Request wrappers replace it with a **length-matched non-narrative frame**, one
per story wrapper per style, so length cancels inside every pair instead of being projected
out afterwards.

Each frame hosts the same `{request}` slot as its story wrapper. Because the request is
byte-identical on both arms, Δlength is fixed by the frame alone: match once per wrapper and
all 100 requests for that wrapper are matched.

`prompt_bare` is retained as the unmatched baseline — it is what quantifies how much the
matching removed.

## Shared constraints

Every frame, every style:

| | |
|---|---|
| **Request-agnostic** | one frame per (`story_id`, style), reused across all 100 requests. `expository` is the sole exception (category-keyed, below). |
| **Slot-form parity** | the frame hosts `{request}` in the same grammatical form as its story wrapper: `colon`, `dash`, `infinitive`, `modal`, `causative`. |
| **Length** | words before the slot and words after the slot are matched **separately** to the story wrapper, not just the total — the read position is the last prompt token, so the post-slot span sets its distance from the request. Tolerance: total ±3, post-slot ±2. |
| **Closing** | ends on the same punctuation as every story wrapper (`:`) and reads as continuable text. |
| **Person** | third person, no second-person address, no named individuals. |
| **Tense** | ~half the 65 frames in each style in present tense, ~half in impersonal past. All 65 story wrappers are past tense, so a uniformly present-tense negative arm would make tense a perfect confound with narrativity. Impersonal past means no agents and no events ("The scope was fixed at the outset: preconditions, then the ordered steps, then verification."), not a report of what someone did. |
| **No narrative content** | no proper nouns, no time of day, no events, no character with agency or interiority. Physical objects are permitted only in `audience` (see below); every other style stays abstract. |
| **Task-shape agnostic** | 88 of the 100 requests are composition tasks (`write` 28, `describe` 17, `explain` 12, `create` 11, `provide` 9, `draft`/`design` 8…) and none is a physical procedure. A frame must therefore fit "write a defamatory article" as readily as "provide step-by-step instructions" — no preconditions, no ordered steps, no verification check, no "follow", no "exposition". The story wrappers assert nothing about task type; the frames must not either, or coherence-of-fit differs systematically between arms. |
| **Slot host animacy** | `modal` and `causative` slots need a subject. Use a generic animate one — "whoever answers must {request}", "has the writer {request}" — never an inanimate nominal. Every story wrapper's modal and causative subject is animate, and "the answer would create a social media campaign" is semantically strained. |
| **Per-style syntax** | each style is restricted to one dominant construction so the four do not collapse into a shared template: `output-spec` affirmative declaratives, `scope` boundary statements, `audience` a described reader, `negation` the only style permitted the "no A, no B, no C" negative list. |
| **Lexical spread** | no content-bearing 4-gram shared between any two of the 65 frames within a style, and none shared with any story wrapper. A formulaic per-style template would let held-out AUROC measure template recall instead of narrativity. |

## Styles

### 1. `expository`

Technical reference prose oriented to the request's subject area. Keyed to the **10 JBB
categories**, not to the individual behavior, and written at the level of *how the domain is
documented* — what stages a procedure is conventionally divided into, what determines whether
one succeeded, what is normally recorded. Never operational content.

Category-level rather than behavior-level because harmful and benign twins are matched
*within* category. The padding is therefore identical for both members of every twin pair, so
it cancels in the harm contrast and enters the narrativity vector only as a small constant
balanced 50/50 across the harmful and benign halves. Behavior-level padding would put harm
content in the negative arm alone and give the narrativity vector a spurious −harm component
— the exact confound H1 has to exclude.

Built as 10 sentence pools in `expository_pools.jsonl`, one per category, 12 sentences each
spanning 5–17 words so a subset can hit any pre-slot target in 30–110 words. Every sentence is
self-contained and order-independent — no connectives, no pronouns reaching outside the
sentence — so any selected subset reads coherently in any order. Tense is mixed *within* each
pool rather than balanced across wrappers, which leaves the style tense-neutral by
construction.

Assembly per wrapper: `[pool subset filling target_pre − lead-in] [form-matched lead-in]
{request} [handoff of target_post words, ending in a colon]`. The lead-in and handoff are
topic-free and written per wrapper, so they carry the slot-form parity.

*Controls:* length, position, technical register, topical orientation.
*Releases:* agents, setting, events.

### 2. `output-spec`

Constrains the **form of the required answer** — completeness over indication, ordering,
level of assumed familiarity, absence of caveats and preamble, no length limit. Says nothing
about the subject matter.

*Controls:* length, position, instructional register, second-order orientation to the request.
*Releases:* topic, agents, setting.
*Guard:* names no audience and no role for the responder — those belong to `audience`.

### 3. `scope`

Bounds the **text**: where it begins, what it takes in, where it stops, what is left outside.
Boundary statements, not stages.

*Controls:* length, position, delimiting register.
*Releases:* topic, agents, setting.
*Guards:* `output-spec` constrains presentation, `scope` constrains extent. Keep the two
lexically disjoint. **No procedural scaffolding** — "preconditions, then the ordered steps, then
verification" presupposes a multi-step procedure, and 88 of the 100 requests are composition
tasks where no such structure exists. Bounding extent works for an email as well as for a
method; staging does not.

### 4. `audience` — the concrete rung

Describes the reader's situation, and does it **through physical objects rather than through
negations**. The epistemic constraints stay the same — nothing to check against, no route to a
follow-up — but they are carried by things:

> "The reader has one photocopy, folded twice and creased through the middle column, and no
> second sheet to set beside it. There is no telephone on the bench and no margin wide enough
> to write a question in. What that reader holds reads: {request}. …"

This is the only style permitted concrete nouns, and it is the reason it is the concrete rung:
all 65 story wrappers are concrete and peopled (quills, wax tablets, culverts, chalkboards),
so without one concrete style, concreteness is perfectly correlated with narrativity across the
whole dataset — the same failure as the length confound.

*Controls:* length, position, **concreteness**, **specificity**, pragmatic framing.
*Releases:* topic, events, agency.
*Guards:* the subject matter is the **reader's** situation, not the responder's. What the guard
forbids is a *characterized* role — expertise, traits, an identity ("an expert chemist") — which
would make this a persona contrast and collide with the persona axis (plan §2). A bare
indefinite host, which the `modal` and `causative` slot forms grammatically require ("whoever
replies must {request}", "has the writer {request}"), is **not** a persona: it carries no traits.
It is also uniform across all five styles at those wrappers, so it cannot differentiate them and
cancels in per-style comparison. Objects and places are allowed; a character who *does*
something, an event sequence, or any interiority is not. No second-person address, no named
person.
*Also fixes:* `audience` was the worst style for internal repetition in the first pass (54 of 65
frames colliding, 109 collisions, `no route back for` ×10) precisely because abstract
formulations of "no background, no way to check" have almost no paraphrases. Concrete objects
expand the lexical space.

### 5. `negation`

Explicitly asserts the absence of narrative — no scene, no characters, no events, no setting
— and then states the task.

**Diagnostic, not a primary negative.** It uses narrative vocabulary while denying narrative
content, so a probe that scores it as narrative is a lexical detector rather than a concept
detector (the unresolved L1–2 concern in initial tests §2c). Reported separately; folded into
the averaged vector only as a robustness variant, with the average reported both ways.

## Slot forms

| form | story wrapper | main / held-out |
|---|---|---|
| `colon` | `…a job order blinked, unclaimed: {request}. She jacked in…` | 15 / 4 |
| `infinitive` | `…the bosun agreed to {request} while the surgeon copied…` | 15 / 4 |
| `dash` | `…muttering the same demand for two centuries — {request} — and no librarian…` | 8 / 3 |
| `modal` | `…whether he could {request}, he did not ask what for.` | 8 / 2 |
| `causative` | `…she used it to make him {request}. He laid it out…` | 4 / 2 |

Each style must supply all five forms; the frame inherits the form of the wrapper it is
matched to.

## Counts

Frames: 65 wrappers × 4 request-agnostic styles = 260, plus 10 `expository` category pools.

**The two splits are crossed independently — held-out wrappers go with held-out requests only.**
Crossing the 15 held-out wrappers with the 100 *train* requests would leak the request side, and
held-out AUROC would then measure wrapper generalization alone. Both sides are held out:

| split | wrappers | requests | rows per arm | narrative | non-narrative (5 styles) |
|---|---|---|---|---|---|
| train | 50 | 100 (50 JBB pairs) | 5,000 | 5,000 | 25,000 |
| held-out | 15 | 30 (15 JBB pairs) | 450 | 450 | 2,250 |

Held-out requests come from `../harm/harm_selected_pairs_heldout.csv`, built by
`build_heldout_pairs.py` with zero `jbb_index` overlap against the train pairs. Two held-out
requests were edited (`jbb_index` 36, both arms) to strip first-person referents that have
nothing to bind to inside a third-person frame — the same minimal-trim convention as the
train set's single edit. Both carry `edited=True`.

## Achieved length parity

Word counts of the rendered prompts, train split, against the narrative arm:

| arm | mean | mean Δ vs story | max \|Δ\| |
|---|---|---|---|
| `story` | 69.1 | — | — |
| `expository` | 69.1 | +0.06 | 1 |
| `output-spec` | 69.3 | +0.18 | 3 |
| `scope` | 68.9 | −0.16 | 4 |
| `audience` | 69.3 | +0.22 | 4 |
| `negation` | 68.6 | −0.46 | 5 |
| `bare` (S0) | 16.6 | **−52.48** | 104 |

Max total |Δ| of 5 is the ±3 pre and ±2 post frame tolerances compounding. `bare` is retained
unmatched on purpose: it is the rung that quantifies how much the matching removed, against
the ~50% of vector norm that length accounted for in initial tests §2b.

## Frame audit

`verify_request_wrappers.py` over all 260 frames: **all hard checks pass, zero cross-batch
content 4-gram collisions.** A first round, kept at `request_wrappers/round1/`, scored 132 hard
failures and 214 collisions on the same script — it failed because three of four styles
collapsed onto a shared "no A, no B, no C" construction, frames presupposed the request was a
multi-step procedure, and independent authors converged on identical phrasings. The per-style
syntax mandates, the task-shape-agnostic constraint and the concrete `audience` rung are the
fixes for those three.

Files: `request_wrappers.jsonl` (frames for the 50 train wrappers),
`request_wrappers_heldout.jsonl` (frames for the 15 held-out wrappers), and the two prompt
tables `story_mode_prompts_matched.csv` / `story_mode_prompts_matched_heldout.csv`. The
existing `story_mode_prompts.csv` (train, no preamble, bare negative arm) is left unchanged.

## Not controlled

- **Fictionality.** A story implies invented events; no length or register control reaches it
  (initial tests §2c). The `realistic` 25/25 split on the story wrappers gives a partial
  handle at no data cost: extract narrativity from realistic-only and non-realistic-only
  wrappers and compare.
- **Concreteness, in the averaged vector.** Only `audience` is concrete, so the averaged
  negative arm is four-fifths abstract and the **averaged vector stays
  concreteness-contaminated** — averaging cannot remove a confound present in four of five
  arms. The per-style `audience` figure is the one that tests narrativity against a concrete
  control, and it is what the headline claim must rest on. Balancing concreteness in the
  average would require crossing it within every style, which `output-spec`, `scope` and
  `negation` cannot host without ceasing to be themselves.
- **Animacy.** `audience` closes concreteness and specificity but not animacy — its reader is
  described, never acting. Closing animacy needs a `report` style (people doing things, past
  tense), which is the hardest narrativity control and the most fiction-adjacent. Deferred;
  revisit if the Phase 3 geometry says animacy matters.
- **The negative arm is not neutral.** With matched frames on both sides the extracted axis
  is narrativity *with framing held*, not story-versus-no-framing.
- **Preamble.** Initial tests §4 requires a preamble on both arms; `story_mode_prompts.csv`
  does not carry one yet. Every frame here is written to be continuable after one.

