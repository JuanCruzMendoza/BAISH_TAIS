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

## Method

**Direction.** `d[l] = mean(pos[l]) − mean(neg[l])` at the last token of
`apply_chat_template(..., add_generation_prompt=True)`, every layer `0..L`. Left padding only, and
the read position is asserted against `attention_mask` rather than trusted. `directions__<axis>.pt`
also stores the 50 leave-one-pair-out vectors, the pole means, and `sigma_act` (median residual
norm per layer) for experiment 4's steering units.

**Layer selection (§1.2).** LOPO on the 50 train pairs gives 50 held-out decisions; the 15 held-out
pairs are report-only and nothing selects on them. 

**Metrics per layer.** `lopo_auroc` + `lopo_ci_lo` (the selector and the band rule's input),
`heldout_auroc` + `heldout_ci_lo`, `mean_paired_cos`, `cohens_dz_train` and `cohens_dz_heldout`
(AUROC saturates, so the effect sizes are what rank layers),
`min_/median_pair_margin_sd`, `norm_over_sigma` (exp 4 steering units), `resid_len_auroc` and
`len_frac` (the length foil), `acc_at_train_thr` (calibration transfer, the one metric that fails
where AUROC saturates), and two nulls on the saturation: `null_shuffled_auroc` (pos/neg flipped per
pair, LOPO refitted — off 0.5 ⇒ the label reaches the vector) and `null_random_dir_abs` (random unit
directions, sign-corrected — near 1.0 ⇒ a common-mode offset any direction recovers, so AUROC does
not credit the fitted direction). With appended tasks, `lopo_auroc_task_{harmful,benign}` (§0.2(a)).

Upper CP bounds, win/tie counts, `lopo_sign_p` (the same binomial as `lopo_ci_lo`),
`lopo_cos_stability` (0.997–0.9998 everywhere — it cannot fail at LOPO), the raw
`null_random_dir_auroc` (0.5 by symmetry) and `norm` are not emitted. Constant *n* columns live in
`*_selection.json` and the manifests.

**§1.6 `compare_crossed`.** Matched *n*, 50 vs 50: `directions__story_v2` against
`directions__story_v1`. Cross-evaluation runs on the other side's **50**, not its 15 — neither vector
was fitted on the other's data.

`--curve` adds the `cos(d_n, d_full)` subsample curve over n ∈ {5, 10, 25, 50, 100, 250, 1000} 

**§1.2a transfer.** Both story vectors and the length foil on 100 filler-free v1 pairs, built from
the 50 requests §1.6 does *not* use, so `d_v1_50` has never seen them. `story_mode_prompts.csv` is
built with `PREAMBLE = ""`, so the preamble is prepended at render time to **both** arms from
`common/prompts.py`. Report-only


**Appended tasks for `persona` / `eval` (§0.2(a))**

`--append-task` gives each framing pair one base task, byte-identical across the pair, so the
contrast stays framing-only while the read position sits after a request. Without it, `eval` is a
7-word prefix and the probe is read on ~1,012-char jailbreaks in experiment 3 — a two-order-of-
magnitude extrapolation.

