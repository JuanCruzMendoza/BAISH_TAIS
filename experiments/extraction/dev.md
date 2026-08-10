# extraction — build the directions, pick the layer

Spec: `research/spec-experiments.md` §1. Phase 1 of `research/Plan story-mode.md`.

**Objective.** Extract each of the four rival directions (+ `length` as a nuisance axis) by
diff-in-means on the last prompt token, at every layer, and decide which layer band reads each
axis best without reading length.

## Layout

```
results/<tag>/<model_slug>/
  csv/       metrics tables + *_summary.json     <- what you read
  figures/   *.png plotted from csv/
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

## 50_per_direction

### Run order

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

### Method

**Direction.** `d[l] = mean(pos[l]) − mean(neg[l])` at the last token of
`apply_chat_template(..., add_generation_prompt=True)`, every layer `0..L`. Left padding only, and
the read position is asserted against `attention_mask` rather than trusted. `directions__<axis>.pt`
also stores the pole means and `sigma_act` (median residual norm per layer) for experiment 4's
steering units. The leave-one-pair-out vectors are **not** stored — they are a closed-form update of
the pole sums, so `probe_select` and `cross_auroc` recompute them per layer.

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

## 1K_per_direction

**Objective.** Re-extract the axes at 20× the sample size, and measure how many pairs each direction
actually needs before its vector stops moving. At n=50 every AUROC saturated and `mean_paired_cos`
was the only metric that ranked anything.

### Changes vs `50_per_direction`

|               | 50_per_direction                                            | 1K_per_direction                                  |
| ------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| directions    | `story_v2`, `story_v1`, `harm`, `persona`, `eval`, `length` | `story_v2_1k`, `persona_v2`, `eval_v2`, `harm_v2` |
| datasets      | 50 train / 15 held-out                                      | **800 train / 200 held-out** each                 |
| appended task | `--append-task` at render time                              | already in the v2 pairs — flag refused            |
| length foil   | gates layer selection                                       | **dropped**                                       |
| layer choice  | automatic: band rule + length gate                          | **manual**, read off `cohens_dz_train`            |
| §1.6 / §1.2a  | `compare_crossed`, v1 transfer                              | **not run** (both are v1 arms)                    |

`story_v2_1k` reads `story_mode_v2/pairs_1k.jsonl`; `persona_v2` (`role_play_v2`), `eval_v2` and
`harm_v2` read their own `pairs.jsonl` / `pairs_heldout.jsonl`. **`story_v2_1k` is a separate
direction name, not a bigger `story_v2`** — repointing `story_v2` would invalidate every
`50_per_direction` artefact. The 50-pair `story_mode_v2/pairs.jsonl` is a subset of the 1k file, so
the two tags are not independent: this is a resample of the same construction, not a replication.

`persona_v2` and `eval_v2` render as `framing + task` from the pairs file, byte-identical to the
builders' own prompt tables, so `--append-task` is refused rather than ignored.

**Layer selection is manual.** No band rule and no primary layer — `probe_select` emits the per-layer
table and stops there. `*_selection.json` is replaced by `*_summary.json`, which carries only the
reporting band, the constant *n*'s and the saturation sanity block. We choose the layers by
**`cohens_dz_train`**, confirming against `cohens_dz_heldout` (200 pairs, so it is precise enough to
confirm with), `mean_paired_cos` and `lopo_ci_lo`. **The chosen layers live in `insights.md` and
experiments 2–5 read them from there**, not from any JSON.

No length foil is cached at this tag, so `resid_len_auroc` and `len_frac` are simply absent from the
table (the columns remain available at tags that do cache `length`, report-only). Length is therefore
an unmeasured confound here.

**LOPO at n=800.** 800 held-out decisions per layer per direction, closed-form (rank-1 update of the
pole means), so cost is unchanged. Everything else in *Method* above still holds.

### New: the saturation curve

Two measurements of how much data the direction needs, both band-mean over L11–25 and per layer:

- **`cos(d_n, d_800)`** for n ∈ {50, 100, 150, …, 750}, subsampled from the 800 train pairs, **5
  seeds per n**, reported as mean ± sd. The knee is the pair count that direction actually needs;
  compare it against `n ≥ 4/c²` predicted from `mean_paired_cos` (§0.7), which at n=50 said 9–10
  pairs for story and 98 for `eval`.
- **`cos(d_800_train, d_200_heldout)`** — one number per direction, two vectors fitted on disjoint
  pairs. This is the honest test: unlike the curve, neither vector contains the other, so it cannot
  be inflated by shared rows. Read against the ±3/√d null band.

Both live in `extract_direction.py --curve`, which writes
`csv/directions__<axis>_curve.json` for **every** layer, not just the band. (`compare_crossed --curve`
is untouched — it is a v1 arm and does not run at this tag.)

### Figures

Two per direction, **8 in total**, written to `figures/` by `plot_figures.py` from `csv/` alone:

| figure | x | y |
|---|---|---|
| `plot__<axis>_cos_curve.png` | pairs in the subsample | `cos(d_n, d_full)`, one curve per layer — 10 uniform over 0..L by default, `--layers` to override |
| `plot__<axis>_cohens_dz_train.png` | layer | `cohens_dz_train` (`--with-heldout` adds `cohens_dz_heldout`) |

### Run order

```bash
M=<model>; export RUN_TAG=1K_per_direction
python cache_activations.py $M --dataset story_v2_1k,persona_v2,eval_v2,harm_v2 \
                               --split train,heldout                 # GPU, one model load
python extract_direction.py $M --direction story_v2_1k --curve       # CPU; vectors + curve.json
python probe_select.py      $M --direction story_v2_1k
python plot_figures.py      $M                                       # all 8 figures
python -m experiments.common.check_stale $M
```

~6,400 train + 1,600 held-out prompts cached in total. `--dataset` / `--split` take comma lists and
run the cross product in one process — each cell keeps its own view, manifest and run_key, so resume
and `check_stale` are unchanged, but the model loads once instead of eight times, which is most of
the wall clock. Set `$BLOB_STORE` to share blobs with `50_per_direction` — the 50 story pairs are
byte-identical there and will hit the cache.

