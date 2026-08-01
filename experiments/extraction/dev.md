# extraction — build the directions, pick the layer

Spec: `research/spec-experiments.md` §1. Phase 1 of `research/Plan story-mode.md`.

**Objective.** Extract each of the four rival directions (+ `length` as a nuisance axis) by
diff-in-means on the last prompt token, at every layer, and decide which layer band reads each
axis best without reading length.

## Layout

```
results/<tag>/<model_slug>/
  csv/       metrics tables + *_selection.json   <- what you read
  vectors/   *.pt for experiments 2-5            (gitignored)
  acts/      blobs/<prompt_sha16>.npy + views/   (gitignored)
  meta/      manifests, runs.csv, _archive/, *_deciles.json
```

`<tag>` names a run — a dataset version, not a timestamp (`base`, `benign_tasks`, `story_v3`).
Every script takes `--tag`, also read from `$RUN_TAG`, default `base`. Nothing is shared between
tags, so two runs never mix.

**Cost of that isolation:** `acts/` sits inside the tag, so a new tag re-caches every prompt even
where the dataset did not change. Set `$BLOB_STORE` to an absolute path to point several tags at one
content-addressed blob store — safe by construction, since blobs are keyed by prompt content.

## Run order

```bash
python cache_activations.py <model> --dataset story_v2 --split train      # GPU
python cache_activations.py <model> --dataset story_v2 --split heldout
python cache_activations.py <model> --dataset harm    --split train      # ... x5 directions
python cache_activations.py <model> --dataset length  --split heldout    # needed by the length gate
python extract_direction.py <model> --direction story_v2                 # CPU
python probe_select.py      <model> --direction story_v2
python compare_crossed.py   <model>                                      # §1.6
python probe_select.py      <model> --direction story_v2 --transfer v1_nofiller100   # §1.2a
python -m experiments.common.check_stale <model>
```

`cache_activations.py` is the only script that touches the GPU. Everything downstream reads the
blob cache, so re-running the analysis costs seconds.

Directions: **`story_v2`** (`story_mode_v2/pairs*.jsonl`), **`story_v1`** (v1 matched), `harm`,
`persona`, `eval`, `length`. Extra views: `v1_nofiller100` (§1.2a) and `v1_curve` for the second
`compare_crossed --curve` run.

**`story_v1` is a first-class direction**, 50 train / 15 heldout, so it gets the same LOPO band,
held-out AUROC and length gate as v2 rather than being a by-product of `compare_crossed`:

| split | file | pairs | wrappers | requests |
|---|---|---|---|---|
| train | `story_mode_prompts_matched.csv` | 50 | 50 | 50 |
| heldout | `story_mode_prompts_matched_heldout.csv` | 15 | 15 | 15 |

One row per wrapper, each with a different request, so wrapper is not confounded with request. The
two files share **no wrapper and no request**, so the held-out 15 test framing *and* request
generalisation — the same shape as v2's disjoint context families. Train requests come from the first
half of the shared-request permutation, leaving the second half reserved for `v1_nofiller100`, which
keeps §1.2a out-of-sample for `d_v1` (verified: 0 overlap).

Poles: `pos = prompt_story`, `neg = prompt_expository`, `neg2 = prompt_audience` (v1's concrete rung,
reported by `compare_crossed` as a cosine, not saved as its own vector).

`compare_crossed` now *consumes* `directions__story_v2.pt` and `directions__story_v1.pt` instead of
extracting v1 itself, so both story vectors are produced the same way and the old ordering dependency
is gone.

## Method

**Direction.** `d[l] = mean(pos[l]) − mean(neg[l])` at the last token of
`apply_chat_template(..., add_generation_prompt=True)`, every layer `0..L`. Left padding only, and
the read position is asserted against `attention_mask` rather than trusted. `directions__<axis>.pt`
also stores the 50 leave-one-pair-out vectors, the pole means, and `sigma_act` (median residual
norm per layer) for experiment 4's steering units.

**Layer selection (§1.2).** LOPO on the 50 train pairs gives 50 held-out decisions; the 15 held-out
pairs are report-only and nothing selects on them. Intervals are Clopper–Pearson on the win count,
never bootstrap (§0.7) — paired AUROC is a proportion, so CP is exact and behaves at the boundary.

Band rule, fixed in advance: contiguous layers whose **CP lower bound ≥ max(CP lower bound) − 0.05**
and `|resid_len_auroc − 0.5| ≤ 0.10`; primary layer = band member maximising `mean_paired_cos`, ties
to the shallower.

> Deviation from the spec text, found by running it. The spec compares each layer's CP *lower bound*
> against the best *point estimate*. At a saturated 50/50 the point estimate is 1.00 while its own CP
> lower bound is 0.929, so that rule rejects every layer including the best one. Comparing lower bound
> to lower bound is scale-consistent, rewards precision, and can never return an empty band.

If no layer clears the length gate the band falls back to the AUROC criterion alone and is flagged
`gate_failed` in `probe_select__<axis>_selection.json` — a direction that cannot be separated from
length at any layer is a finding, not a crash.

**Is a saturated AUROC real?** Every diagonal reads 1.000, so `probe_select` also emits three
controls per layer, summarised under `sanity` in the selection JSON:

| control | what a failure means |
|---|---|
| `null_shuffled_auroc` | pos/neg flipped per pair, LOPO refitted, 20 draws. Off 0.5 ⇒ the label reaches the vector |
| `null_random_dir_auroc` / `_abs` | 20 random unit directions. The mean is 0.5 by symmetry; the sign-corrected `_abs` near 1.0 ⇒ a large common-mode offset any direction recovers, so AUROC does not credit the fitted direction |
| `min_pair_margin_sd` | smallest per-pair gap in pooled-sd units. ≤ 0 with AUROC 1.000 is a contradiction |

Plus a view-level check on the read position: the number of distinct final tokens per pole and
whether the poles share one. If the two arms never end on the same token, a perfect AUROC may be a
token-identity readout. Uses `last_token_id` recorded in the view by `cache_activations`; for views
cached before that existed it falls back to the final *character* — not a longer tail, which would be
disjoint between any two distinct sentences and would flag every dataset.

`failures` are correctness problems (the number is wrong); `warnings` are interpretability problems
(the number is right but does not mean what it looks like). `load_view_matrix` additionally asserts
pole alignment by `row_id` and rejects identical pos/neg prompts.

**§1.6 `compare_crossed`.** Matched *n*, 50 vs 50: `directions__story_v2` against
`directions__story_v1`. Cross-evaluation runs on the other side's **50**, not its 15 — neither vector
was fitted on the other's data.

`--curve` adds the `cos(d_n, d_full)` subsample curve over n ∈ {5, 10, 25, 50, 100, 250, 1000} and
needs the `v1_curve` view: **1,000 pairs = 2,000 prompts**, 20 requests per wrapper. `d_full` is the
whole view, so the n=1000 point reads 1.00 by construction — it is the anchor and an indexing
assertion, not a result. The informative points are 25/50/100: if `cos(d_50, d_full)` ≈ 0.99 the
50-pair regime is justified empirically (§6.1).

**§1.2a transfer.** Both story vectors and the length foil on 100 filler-free v1 pairs, built from
the 50 requests §1.6 does *not* use, so `d_v1_50` has never seen them. `story_mode_prompts.csv` is
built with `PREAMBLE = ""`, so the preamble is prepended at render time to **both** arms from
`common/prompts.py`. Report-only: it never feeds the band rule, because `prompt_bare` is ~5× shorter
and `initial_tests` had a pure length vector scoring 1.00 on exactly this contrast. High `d_v2` with
`d_length` ≈ 0.5 is a result; high for both is not.

## Reproducibility

Per §0.8/§0.10/§0.11: activations are content-addressed per prompt (`acts/blobs/<sha16>.npy`,
atomic writes) with datasets as views over them, so subsamples cost no extra forward passes and an
interrupted cache run resumes by skipping existing blobs. Every artefact carries a manifest with a
`run_key` over the resolved config plus upstream keys; a changed config archives the prior artefact
instead of overwriting it. `check_stale.py <model> [tag]` reports stale, interrupted and orphaned
artefacts.

Views are written twice: `views/<ds>__<split>.json` is the current pointer and
`views/<ds>__<split>__<view_key8>.json` is history. Without the keyed copy, changing a dataset would
overwrite the only record of the previous view and every earlier result would hold an unresolvable
`view_key`.

## Appended tasks for `persona` / `eval` (§0.2(a))

`--append-task` gives each framing pair one base task, byte-identical across the pair, so the
contrast stays framing-only while the read position sits after a request. Without it, `eval` is a
7-word prefix and the probe is read on ~1,012-char jailbreaks in experiment 3 — a two-order-of-
magnitude extrapolation.

The pool is **half harmful by construction**: one task per harm row, alternating pole after a seeded
shuffle, so no goal repeats. Train tasks come from `harm_selected_pairs.csv`, held-out tasks from
`harm_selected_pairs_heldout.csv`, so a task a pair was fitted with never reappears in its own
evaluation set.

| split | tasks | harmful | benign |
|---|---|---|---|
| train | 50 | 25 | 25 |
| heldout | 15 | 8 | 7 |

`persona` and `eval` share the assignment (task *i* is the same in both), which keeps them comparable;
the task cancels within each pair, so it does not enter either vector at first order.

**It also fixes the `persona` degeneracy.** `neg_instruction_padded` has only 5 paraphrases over 50
roles, so the bare negative pole is 5 distinct strings at 10× reuse — `mean(neg)` averages 5 points,
not 50, and LOPO is not out-of-sample because 9 copies of the held-out pair's negative stay in the
fit. With tasks appended the negative pole is 50 distinct prompts. The *framing* still has 5 levels,
so grouped LOPO by `neg_variant` is still worth adding.

`probe_select` emits `lopo_auroc_task_{harmful,benign}` whenever tasks are present. A large gap means
the vector is framing × harm rather than framing.

## Open

- `length` has 21 train / 9 heldout pairs, not 50/15 (§6.5). The length gate is therefore coarse:
  9 pairs quantise `resid_len_auroc` to 1/9 ≈ 0.111, wider than the ±0.10 tolerance, so
  `gate_failed` will fire more often than the construct warrants.
- Grouped LOPO by `neg_variant` for `persona` (5 folds, not 50) — the framing pole is still 5 levels.
- `hooks.py` (§5.4) is not written yet; nothing here needs it.

## Verification

Not yet run against a real model. The non-GPU path was validated end to end on synthetic
activations with a planted axis: LOPO/held-out AUROC, the band rule, archive-on-write, the resume
gate, truncated-tail rejection, the chat-template cache guard, and `check_stale`. With v1 sharing
the planted story axis, `cos(d_v2, d_v1_50)` read +0.77 against a ±0.375 null band and the transfer
test read 0.96 for `d_v2` against 0.48 for the length foil — i.e. the pipeline reports the intended
pattern when the pattern is there.
