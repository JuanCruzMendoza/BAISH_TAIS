# Harm v2 — data statement

Harmful-vs-benign request pairs, crossed with four framing families so the harm axis is
read under framings rather than only bare. The v1 `harm` direction is the `bare` cell of
this table.

## Contents

| file | rows | |
|---|---|---|
| `pairs.jsonl` | 800 | train pairs |
| `pairs_heldout.jsonl` | 200 | held-out pairs |
| `harm_v2_prompts.csv` | 1,000 | both splits, one row per pair |
| `build_dataset.py` / `verify_pairs.py` | | build and audit |

One pair = (framing + harmful request) vs (framing + benign request). The framing is
byte-identical across the pair; the two arms differ only in the request slot, which
`verify_pairs.py` checks by masking the slot and comparing the remainder.

## Cells

| family | arm | train | heldout | framing source |
|---|---|---|---|---|
| bare | bare | 50 | 14 | — |
| roleplay | role | 125 | 31 | `../role_play_v2/roles{,_heldout}.jsonl` |
| roleplay | assistant | 125 | 31 | `../role_play_v2/build_pairs.py` pools (13 / 12) |
| story | story | 125 | 31 | `../story_mode/story_wrappers{,_heldout}.jsonl` |
| story | nonstory | 125 | 31 | `../story_mode/request_wrappers{,_heldout}.jsonl` |
| eval | evaluation | 125 | 31 | `../eval_aware/eval_awareness_pairs{,_heldout}.csv` |
| eval | deployment | 125 | 31 | same file, other column |

Framing is the nuisance axis, so each framed family splits evenly between its two
directions and no direction can correlate with the harm label. `bare` is the null
framing: it has one direction and is capped at one pair per request, since a second
would be a byte-identical duplicate — which is why the held-out bare cell is 14, not 15
(200 minus an odd bare count cannot be split into three even families).

The assistant pool is imported from `role_play_v2/build_pairs.py` rather than copied, so
the pole cannot drift. Role framings are drawn one paraphrase per role, so 125 role
pairs use 125 distinct roles. Non-story frames are drawn round-robin over the four
styles (audience, negation, output-spec, scope).

## Requests

All 50 pairs of `../harm/harm_selected_pairs.csv` (JBB-Behaviors, 5 per category × 10
categories) in train, each used exactly 16 times; the 15 pairs of
`harm_selected_pairs_heldout.csv` in held-out, 13–14 times each. Categories are 80 pairs
each in train; held-out inherits JBB's uneven coverage (13–28).

Train and held-out share no framing and no `jbb_index`, so held-out tests framing *and*
request generalisation at once.

## Render conventions

Each family keeps its source table's convention byte for byte, so a harm_v2 cell stays
comparable with the `persona` / `eval` / `story_v1` directions:

| family | prompt |
|---|---|
| bare | `<request>` |
| roleplay | `<framing>\n\n<request>` (`prompts.with_task`) |
| eval | `<framing> <request>:` |
| story | `Continue the text below.\n\n<frame with {request} filled>` |

The convention is constant within a pair, so it cannot enter the harm contrast; across
families it is deliberate surface variance. Mean prompt length: bare 14w, eval 20w,
roleplay 28w, story 70w. 62% of prompts end in a colon (story + eval).

The story family drops the request into a mid-sentence slot, so it is lower-cased and
takes `../story_mode`'s two grammar edits — `GOAL_FIXES` for jbb 22 benign, and
`HELDOUT_EDITS` for jbb 36, whose first-person referents have nothing to bind to inside
a third-person frame. The other three families read the request as its own sentence and
use the JBB goal unedited, which keeps the bare cell identical to the v1 `harm`
direction.

## Known confounds

- **Length.** The harmful goal is on average 1.84 words longer than its benign twin
  (−9 to +12, longer in 57% of pairs). This is inherited from JBB and is present in the
  v1 `harm` direction as well; the framings dilute it in relative terms (2 words in a
  70-word story prompt) but do not remove it.
- **Read position.** In the bare and roleplay families the last token is the last word
  of the request itself, and it differs across the pair in half of all pairs. A
  saturated AUROC there should be checked against `token_info` in the view.
- **Lexical overlap.** 83% of pairs open on the same verb, by construction of the JBB
  benign twins — the contrast is the object of the request, not its speech act.
- `jbb 83` harmful carries an upstream typo ("treating and  someone with
  schizophrenia"), kept so the request text matches the v1 tables.
