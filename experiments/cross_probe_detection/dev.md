# cross_probe_detection — is it one axis or four? (H1)

Spec §2. **Objective:** score every probe from `extraction/` against every axis (paired AUROC, §2.2)
and measure the geometry between them (cosines, residual fractions, §2.3).

## Run order

No GPU. Both scripts read `extraction/results/<tag>/<model>/` for the `.pt` vectors and the blob
cache; they need `extract_direction.py` plus cached `train` **and** `heldout` views per axis.
`probe_select.py` is not consumed.

```bash
python cross_auroc.py Qwen/Qwen2.5-7B-Instruct --tag 50_per_direction
python geometry.py    Qwen/Qwen2.5-7B-Instruct --tag 50_per_direction
```

## Design

**Axes and probes** are all of `views.DIRECTIONS`, 6×6 plus a `random_<seed>` null row. `story_v1`
is kept beside `story_v2` as a **positive control for the off-diagonal machinery**: the two share an
axis (§1.6, cos +0.76), so their off-diagonal cells must read high. Without a cell that is *supposed*
to be non-null, a matrix of nulls is indistinguishable from cross-dataset pairing destroying all
signal.

**Cells (§2.2).** Off-diagonal = paired AUROC on the target axis's pooled train + held-out (65 pairs;
30 for `length`) — the probe was never fitted on any of it, so its 50 train pairs are as
out-of-sample as its 15. Diagonal = LOPO on train (n=50) and the held-out 15 as two rows; the pooled
in-sample number is never reported.

**Own-best layer** is decided at run time as the `mean_paired_cos` peak inside the band, recomputed
from cached activations on **train pairs only** (§0.7 keeps the held-out 15 out of every selection).
`probe_select`'s `primary` is deliberately unused: it maximises the same quantity but only inside the
length-gated band, and at 9 length pairs that gate admits 1–2 layers, so `primary` lands up to 9
layers off the peak (`story_v1` L7 vs L16).

**Cosine floor (§2.3).** Each axis's train pairs split into two disjoint halves; the cosine between
the half-vectors is the noise floor. A cross-axis cosine of 0.2 means nothing until you know the axis
agrees with itself at 0.97.

## Metrics

**`cross_auroc_tensor.csv`** — every probe × axis × layer. `_matched.csv` (§2.2a, all probes at
depth 0.65) and `_ownbest.csv` (§2.2b) are the same cells at one layer, plus band columns.

| column | meaning |
|---|---|
| `cell_type` | `diag_lopo` / `diag_heldout` / `offdiag_pooled` / `null_pooled` |
| `auroc` | paired AUROC of the probe on that axis's contrast |
| `wins`, `ties`, `n` | raw counts; Clopper–Pearson is a pure function of these, so an interval is reconstructable without carrying one |
| `auroc_folded` | `max(a, 1−a)` — sign-free leakage magnitude. A probe reading a rival at 0.17 reads it as strongly as at 0.83 |
| `null_folded` | what 20 random unit directions score (folded) on this axis and layer. **The reference for a cell, not 0.5** |
| `excess_over_null` | `auroc_folded − null_folded`. ≤0 means the probe reads the rival axis no better than an arbitrary direction |
| `cos_probe_axis` | cosine between the probe and the axis's own direction, placed next to the AUROC on purpose — it shows the AUROC saturating while the cosine still moves |
| `delta_excluded` | tightest δ ∈ {.60,.65,.70,.75} whose region [1−δ, δ] contains the whole 95% interval. The H1 absence verdict; blank = clears none |
| `axis_mean_paired_cos`, `auroc_sd`, `n_draws` | null rows only: the axis's contrast consistency, and the spread over draws |
| `band_mean_auroc`, `band_max_folded`, `band_mean_excess` | matrix files only: the same quantities over L11–25, guarding against single-layer wiggles |

**`geometry_selfsplit.csv`** — the calibration table everything else is read against.

| column | meaning |
|---|---|
| `split_cos` | cosine between two disjoint half-sample estimates of the same axis. The noise floor |
| `reliability` | Spearman–Brown of `split_cos`: reliability of the *full* vector, since a split-half value is a half-sized one and would over-correct |
| `n_half` | pairs per half (10 for `length`, 25 elsewhere) |

**`geometry_cos.csv`** — `cos` per axis pair per layer, with `reliability_a`/`_b` and
`cos_disattenuated` = `cos / sqrt(r_a · r_b)`, the estimated cosine between the *true* axes once each
estimate's own noise is removed. Null band ±3/√d = ±0.050, recorded in the manifest.

**`geometry_residual.csv`** — `resid_frac` = ‖v − P_span(basis) v‖ / ‖v‖, i.e. the share of an axis
lying outside the others. 1.0 = orthogonal. `basis` is `others` (every basis 4-dimensional: a story
variant's sibling is dropped and `story_v1` never enters another axis's basis, so rows are
comparable) or `story_v2` (the reverse residual).

## Deviations from the spec

- **No principal angles / fold subspaces.** §2.3 asks for 5-dim subspaces; they ranked pairs
  identically to the cosines and added a second calibration to explain.
- **Equivalence is two-sided** — a cell passes δ when its interval lies inside [1−δ, δ], not when a
  one-sided upper bound is below δ. The story axes read length *inverted* (0.20), so absence has to
  exclude leakage of either sign.
- **Trimmed columns:** `cohens_dz` (monotone in `cos_probe_axis`), `ci_hi_1s` (superseded by
  `delta_excluded`), `ci_lo`/`ci_hi` (recoverable from `wins`/`ties`/`n`), `abs_cos`, the constant
  `null_band`, and the `norm` columns duplicated from `probe_select`.
- **One stem, `cross_auroc`, three artefact suffixes** instead of §0.1's separate
  `__matched`/`__ownbest` stems: they share a `run_key`, and a stem that prefixes another would make
  `Run._own_files()` archive the other's files.

## Open

- `length` has 21 train pairs, so its split-half floor rests on halves of 10 (`n_half` records it).
- `story_v1`'s `neg2` (audience) arm is not run here; `compare_crossed` reports
  cos(d_v2, d_v1_audience) = +0.45.
- `eval`'s split-half cosine is 0.612, so every `eval` comparison is noise-dominated. Needs ~100
  pairs (§0.7); nothing here fixes it.
