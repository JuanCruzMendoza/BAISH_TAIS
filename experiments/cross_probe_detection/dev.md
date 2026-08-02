# cross_probe_detection — is it one axis or four? (H1)

Spec: `research/spec-experiments.md` §2. Phase 2 of `research/Plan story-mode.md`.

**Objective.** Test whether the rival directions from `extraction/` are one axis or several:
score every probe against every axis (paired AUROC, §2.2) and measure the geometry between them
(cosines, residual fractions, §2.3).

## Layout and run order

No GPU. Both scripts read `extraction/results/<tag>/<model>/` for the `.pt` vectors and the blob
cache, and write to their own `results/<tag>/<model>/{csv,meta}/`.

```bash
python cross_auroc.py Qwen/Qwen2.5-7B-Instruct --tag 50_per_direction
python geometry.py    Qwen/Qwen2.5-7B-Instruct --tag 50_per_direction
```

Prerequisite: `extract_direction.py` for every axis, plus cached `train` **and** `heldout` views
(the off-diagonal pools both). `probe_select.py` is not consumed — see below.

| file | contents |
|---|---|
| `cross_auroc_tensor.csv` | every probe × axis × layer cell |
| `cross_auroc_matched.csv` | §2.2(a) all probes at one depth (default 0.65) |
| `cross_auroc_ownbest.csv` | §2.2(b) each probe at its own `mean_paired_cos` peak |
| `geometry_selfsplit.csv` | per axis: split-half cosine, reliability, ‖d‖/σ_act |
| `geometry_cos.csv`, `_residual.csv` | pairwise cosines and residual fractions, per layer |

## Method

**Axes and probes.** All six of `views.DIRECTIONS`, so 6×6 plus a `random_<seed>` null row.

`story_v1` is kept alongside `story_v2` as a **positive control for the off-diagonal machinery**.
The two are known to share an axis (§1.6, cos +0.76), so their off-diagonal cells must read high.
Without a cell that is *supposed* to be non-null, a matrix of nulls cannot be distinguished from
cross-dataset pairing having destroyed all signal.

**Cells (§2.2).** Off-diagonal = paired AUROC on the target axis's pooled train + held-out (65
pairs; 30 for `length`) — the probe was never fitted on any of it, so its 50 train pairs are as
out-of-sample as its 15. Diagonal = LOPO on train (n=50) and the held-out 15, as two rows; the
pooled in-sample number is never reported. Every cell carries a Clopper–Pearson interval.

**Own-best layer is decided at run time**: the peak of `mean_paired_cos` inside the reporting band,
recomputed here from the cached activations on **train pairs only** (§0.7 keeps the held-out 15 out
of every selection). `probe_select`'s `primary` is deliberately not used — it maximises the same
quantity but only inside the length-gated band, and at 9 length pairs that gate admits 1–2 layers,
so `primary` lands up to 9 layers off the peak (`story_v1` L7 vs L16).

**Cosine floor (§2.3).** Each axis's train pairs are split into two disjoint halves; the cosine
between the two half-vectors is the noise floor. A cross-axis cosine of 0.2 means nothing until you
know the axis agrees with itself at 0.97. Cross-axis cosines are also reported
attenuation-corrected, using Spearman–Brown on the split-half value.

## Deviations from the spec, and why

- **No principal angles / fold subspaces.** §2.3 asks for 5-dim subspaces and cross-axis angles.
  They ranked the pairs identically to the cosines and added a second calibration to explain, so
  cosine against the split-half floor is the whole geometry story.
- **Equivalence is two-sided**: a cell passes δ when its 95% interval lies inside [1−δ, δ], not
  when the one-sided upper bound is below δ. The story axes read length *inverted* (AUROC 0.20),
  so absence has to exclude leakage of either sign.
- **`null_folded` / `excess_over_null` / `cos_probe_axis` columns.** The reference for an
  off-diagonal cell is not 0.5 — it is what a random direction already earns on that axis, which
  here is 0.60–0.76. Putting each cell's cosine next to its AUROC makes the matrix self-diagnosing.
- **Spearman–Brown, not the raw split-half cosine**, for disattenuation: a split-half cosine is the
  reliability of a *half*-sized vector and would over-correct.
- **Trimmed metrics.** `cohens_dz` and the one-sided `ci_hi_1s` are not emitted: `d_z` is monotone
  in `cos_probe_axis`, which is already per cell, and the one-sided bound is superseded by the
  two-sided `delta_excluded`. `abs_cos`, the constant `null_band` (now in the manifest), and the
  `norm` columns duplicated from `probe_select` are gone from the geometry tables.
- **One stem, `cross_auroc`, with three artefact suffixes** instead of §0.1's
  `cross_auroc__matched` / `__ownbest`. All three come from one config, so they share a `run_key`;
  and a stem that is a prefix of another stem would make `Run._own_files()` archive the other's
  files.

## Open

- `length` has 21 train pairs, so its split-half floor rests on halves of 10. `n_half` is recorded
  per row.
- `story_v1`'s `neg2` (audience) arm is not run here. §3.1 wants all negative arms; §2 keeps the
  matrix square. `compare_crossed` already reports cos(d_v2, d_v1_audience) = +0.45.
- `eval`'s split-half cosine is 0.612, so every `eval` comparison here is noise-dominated. Nothing
  in this experiment fixes that; it needs ~100 pairs (§0.7).
