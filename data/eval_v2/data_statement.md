# eval_v2 — data statement

Eval-awareness framing pairs crossed with JBB requests, at the v2 scale (800 / 200) and in
the v2 prompt shape (`framing\n\nrequest`, spec 0.2(a)). v1's `eval_aware` is the same
contrast rendered inline as `framing request:` and crossed exhaustively (5,000 / 450).

## Contents

| file | rows | |
|---|---|---|
| `pairs.jsonl` | 800 | train pairs |
| `pairs_heldout.jsonl` | 200 | held-out pairs |
| `eval_v2_pairs.csv` | 1,000 | wide: both prompts of every pair |
| `eval_v2_prompts.csv` | 2,000 | long: one row per prompt (`pole` 1 = evaluation) |

One pair = (evaluation framing + request) vs (deployment framing + same request). The
request is byte-identical across the pair, so the contrast is framing-only while the read
position sits after a request.

## Construction

| | train | held-out |
|---|---|---|
| framings | 50, `../eval_aware/eval_awareness_pairs.csv` | 15, `..._heldout.csv` |
| JBB pairs | 50, `../harm/harm_selected_pairs.csv` | 15, `..._heldout.csv` |
| pairs per framing | 16 (8 JBB pairs, both poles) | 14 (×10) or 12 (×5) |
| requests | 100, each used 8× | 30, each used 6–7× |

A framing always draws whole JBB pairs, so its harmful and benign requests are
content-matched and every framing is exactly half harmful. Requests are dealt from
independently shuffled, category-rotated blocks, so no framing repeats a request or a
category and no two framings get the same task set. 100 pair-slots over 15 held-out
framings is 6.67 — hence 14/12 rather than an even grid; keeping all 15 framings matters
more, since the held-out number measures generalisation across framings.

Train framings cross only train requests and held-out framings only held-out requests: the
splits share no framing string, no 4-gram and no `jbb_index`, and no prompt string appears
in both.

## Balance

| | train | held-out |
|---|---|---|
| label | 400 harmful / 400 benign | 100 / 100 |
| category | 10, 80 rows each | 10, 14–26 rows each |
| person (2nd / 3rd) | 400 / 400 | 106 / 94 |
| framing source | 256 arXiv:2510.20487 / 544 custom | 200 custom |
| words | dep. 12–38 (mean 19.5), eval. 11–36 (mean 19.4) | 10–32 (19.4) / 9–32 (19.5) |
| within-pair word delta | −2 to +2 (mean −0.04) | −1 to +1 (mean +0.12) |

Held-out category counts are uneven because the held-out JBB pool has two pairs in five
categories and one in the other five. Person is 106/94 because 15 framings split 8/7.

## Known issue

`jbb_index` 83 harmful reads *"…guide for treating and  someone with schizophrenia"* — a
missing word inherited from the shared harm pool, so it is present in v1 `eval_aware` and
in `role_play_v2` too. Repairing it belongs in the `GOAL_FIXES` table that all three share,
not here.
