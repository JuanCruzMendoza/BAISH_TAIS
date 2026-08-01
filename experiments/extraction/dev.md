# extraction — build the directions, pick the layer

Spec: `research/spec-experiments.md` §1. Phase 1 of `research/Plan story-mode.md`.

**Objective.** Extract each of the four rival directions (+ `length` as a nuisance axis) by
diff-in-means on the last prompt token, at every layer, and decide which layer band reads each
axis best without reading length.

## Run order

```bash
python cache_activations.py <model> --dataset story   --split train      # GPU
python cache_activations.py <model> --dataset story   --split heldout
python cache_activations.py <model> --dataset harm    --split train      # ... x5 directions
python cache_activations.py <model> --dataset length  --split heldout    # needed by the length gate
python extract_direction.py <model> --direction story                    # CPU
python probe_select.py      <model> --direction story
python compare_crossed.py   <model>                                      # §1.6
python probe_select.py      <model> --direction story --transfer v1_nofiller100   # §1.2a
python -m experiments.common.check_stale <model>
```

`cache_activations.py` is the only script that touches the GPU. Everything downstream reads the
blob cache, so re-running the analysis costs seconds.

Datasets: `story` (`story_mode_v2/pairs*.jsonl`), `harm`, `persona`, `eval`, `length`, plus the two
v1 story views `v1_fair50` (§1.6) and `v1_nofiller100` (§1.2a), and `v1_curve` for the second
`compare_crossed --curve` run.

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

**§1.6 `compare_crossed`.** Matched *n*, 50 vs 50: v2 against a 50-row v1 subsample (all 50
wrappers of `story_wrappers.csv`, each with a different request). Cross-evaluation runs on the other
side's **50**, not its 15 — neither vector was fitted on the other's data.

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
instead of overwriting it. `check_stale.py` reports stale, interrupted and orphaned artefacts.

## Open

- **`--append-task` defaults to off.** §0.2(a) recommends pairing each `persona`/`eval` framing 1:1
  with a rotated base task, since `eval` is a 7-word prefix and jailbreaks are ~1,012 chars. It is
  implemented (`--append-task`, rotating the 50 harm goals) but not the default, pending a decision.
- `length` has 21 train / 9 heldout pairs, not 50/15 (§6.5). The length gate is therefore coarse:
  9 pairs quantise `resid_len_auroc` to 1/9 ≈ 0.111, which is wider than the ±0.10 tolerance.
- `hooks.py` (§5.4) is not written yet; nothing here needs it.

## Verification

Not yet run against a real model. The non-GPU path was validated end to end on synthetic
activations with a planted axis: LOPO/held-out AUROC, the band rule, archive-on-write, the resume
gate, truncated-tail rejection, the chat-template cache guard, and `check_stale`. With v1 sharing
the planted story axis, `cos(d_v2, d_v1_50)` read +0.77 against a ±0.375 null band and the transfer
test read 0.96 for `d_v2` against 0.48 for the length foil — i.e. the pipeline reports the intended
pattern when the pattern is there.
