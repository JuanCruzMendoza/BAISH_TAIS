
# 50_per_direction

Qwen2.5-7B-Instruct (L=28, d=3584), 50 train / 15 held-out pairs per direction, last token.
`persona` and `eval` carry an appended base task (25 harmful / 25 benign). Band = L11–25.

We compare the story_v1 direction with the story_v2 direction, also against the story_v1 unmatched in order to know whether they can actually read "narrativity" 

These direction seem to be almost the same, although all the AUROC metrcis are saturated

## Do the axes exist?

Band-mean over L11–25; held-out is 15 pairs, CP interval.

| direction | lopo AUROC | held-out | mean_paired_cos | ‖d‖/σ_act | verdict |
|---|---|---|---|---|---|
| `story_v2` | 1.000 | 1.000 | 0.716 | 0.34 | strong |
| `story_v1` | 1.000 | 1.000 | 0.698 | 0.13 | strong |
| `persona` | 1.000 | 1.000 | 0.573 | 0.13 | strong |
| `length` | 1.000 | 1.000 | 0.689 | 0.14 | strong |
| `harm` | 0.987 | 0.960 | 0.582 | 0.49 | strong, deep only |
| `eval` | 0.949 | 0.978 | 0.295 | 0.05 | **weak** |

Sample size is settled by `c = mean_paired_cos` (§0.7), not by AUROC:

| direction | c | cos(d̂, d) at n=50 | n for cos ≥ 0.95 |
|---|---|---|---|
| `story_v2` / `story_v1` | 0.70–0.72 | 0.99 | 9–10 |
| `harm` / `persona` | 0.57–0.58 | 0.98 | 19 |
| `eval` | 0.295 | 0.91 | **98** |

## Is the saturated AUROC real?

Three controls added to `probe_select`, band-mean. `shuffled` refits LOPO with the pos/neg
assignment flipped per pair (20 draws); `rand±` is the sign-corrected mean over 20 random unit
directions; `margin` is the smallest per-pair gap in pooled-sd units.

| direction | shuffled (want 0.5) | rand± | margin min | final token | verdict |
|---|---|---|---|---|---|
| `story_v2` | 0.496 | 0.834 | +1.14 | identical, all 50 pairs | real, wide |
| `story_v1` | 0.521 | 0.697 | +0.92 | identical, all 50 pairs | real, wide |
| `harm` | 0.494 | 0.762 | +0.18 | **differs within pair** | real, thin |
| `persona` | 0.509 | 0.723 | +0.29 | identical within pair | real, thin |
| `eval` | 0.496 | 0.611 | **−0.21** | identical within pair | not saturated (0.98) |
| `length` | 0.505 | 0.776 | +0.33 | identical within pair | real, thin |

- **No label leakage.** The shuffled null sits at 0.494–0.521 everywhere, so LOPO closes every path
  from label to vector and the 1.000s are not an artifact of the pipeline.
- **Not a token-identity readout for 5 of 6.** `story_v2`/`story_v1` end every prompt on the same
  character; `persona`/`eval`/`length` end identically *within* each pair (the appended task is
  byte-identical). Only `harm` has poles that end differently, so it is the one direction where
  final-token identity remains a live alternative explanation.
- **`rand±` 0.61–0.83 is the real caveat.** A *random* direction, sign-corrected, already separates
  the poles at 0.83 for `story_v2`. So AUROC 1.000 is credit for the contrast being consistent, not
  evidence that the fitted direction is special — geometry (§2), not AUROC, has to carry H1.
- **Margins split the directions.** `story_v2`/`story_v1` clear the boundary by ~1 sd; `harm` and
  `persona` are perfect but within 0.2–0.3 sd of touching, so their 1.000 would not survive a few
  more pairs. `eval`'s classes already overlap, consistent with its 0.98.

Length gap at the read position (chars, pos − neg): `eval` +0.4, `persona` +9.5, `harm` +15.0,
`story_v2` −22.1, `story_v1` −39.6, `length` +200.5. Both story arms are *shorter* on the narrative
side, which is the opposite sign to the naive story-vs-bare contrast.

## The length foil

`resid_len_auroc`: each direction applied to filler-long vs filler-short. 0.5 = length-blind.

| direction | L11–25 mean | shape across depth |
|---|---|---|
| `story_v2` | 0.170 | inverted at every layer (long scores *lower*) |
| `story_v1` | 0.163 | same |
| `harm` | 0.770 | 1.000 shallow → 0.11 by L27 |
| `persona` | 0.970 | 1.000 to L23, 0.67–0.89 after |
| `eval` | 1.000 | 1.000 at every layer |

**No direction is length-blind anywhere in the band.** story reads length backwards (consistent with
`initial_tests` §2c); `persona` and `eval` read it almost perfectly forwards.

## Layer selection is being decided by the gate, not the signal

The foil has **9 held-out pairs**, so `resid_len_auroc` is quantised to steps of 1/9 and the
`|x − 0.5| ≤ 0.10` gate only admits 0.444 / 0.5 / 0.556.

| direction | layers passing the gate | band chosen | primary (depth) | where `mean_paired_cos` actually peaks |
|---|---|---|---|---|
| `story_v2` | 2 of 29 (L0, L20) | L20–20 | 20 (0.71) | L16 (0.776) |
| `story_v1` | 5 of 29 (L0,1,6,7,20) | L6–7 | 7 (0.25) | L16 (0.767) |
| `harm` | 6 of 29 | L22–24 | 22 (0.79) | L21 (0.708) |
| `persona` | 1 of 29 (L0 only) | gate failed → L1–28 | 7 (0.25) | L12 (0.707) |
| `eval` | 2 of 29 (L0, L1) | gate failed → L1–28 | 14 (0.50) | L14 (0.332) |
| `length` | — (self) | gate failed → L1–28 | 2 (0.07) | L8 |

`story_v1`'s shallow primary (L7) vs `story_v2`'s (L20) is a gate artifact, not a real difference:
both peak at L15–18 on every non-saturating metric.

## v2 vs v1 (§1.6, matched n = 50 vs 50)

| quantity | L11–25 mean | note |
|---|---|---|
| cos(d_v2, d_v1) | **+0.759** | null band ±0.050; peaks +0.799 at L14–20 |
| cos(d_v2, d_v1_audience) | +0.454 | v1's negative arm changes the vector a lot |
| AUROC d_v2 → v1's 50 pairs | 1.000 | |
| AUROC d_v1 → v2's 50 pairs | 1.000 | |

Same axis, not the same vector (0.76 ≈ 40°). The 5,000-row crossed table bought nothing the 50-pair request-free construction does not already give.

## §1.2a transfer — uninformative, as pre-registered

All three probes on the 100 filler-free v1 pairs (`prompt_story` vs `prompt_bare`):

| probe | band AUROC | Cohen's d_z |
|---|---|---|
| `d_length` | 1.000 | **5.26** |
| `d_v1_50` | 1.000 | 4.71 |
| `d_v2` | 1.000 | 4.33 |

The length foil scores the *highest*. Per §1.2a the test is only readable when `d_length` ≈ 0.5, so it
says nothing about request-invariance here.

## Appended tasks

| direction | AUROC on harmful tasks | on benign | gap |
|---|---|---|---|
| `persona` | 1.000 | 1.000 | +0.000 |
| `eval` | 0.949 | 0.949 | +0.000 |

## Findings

- **Five of six axes are linearly readable at n=50; `eval` is the exception** — c=0.295 means its
  vector needs ~98 pairs, and its norm is 5–10× smaller than the others'.
- **Nothing is length-blind.** story reads length inverted; `persona` and `eval` read it at ~1.0.
- **The 9-pair length gate, not the data, is choosing layers.** It admits ~2 of 29 layers by
  quantisation accident and overrides `mean_paired_cos`, which peaks at L15–18 for both story axes.
- **`story_v2` and `story_v1` are the same axis** (cos +0.76 vs a ±0.05 null, cross-AUROC 1.000 both ways), so the crossed table is not needed — but v1's negative arm matters (audience 0.45 vs expository 0.76).
- **§1.2a is uninformative**: `d_length` separates the filler-free contrast better than either story
  vector, exactly the outcome the arm was built to detect.
- **Appended tasks worked**: zero AUROC gap between harmful and benign requests, so `persona` and `eval` are framing axes rather than framing × harm.
- AUROC saturates everywhere and cannot rank layers — every selection decision rests on
  `mean_paired_cos` and the foil.
- **The saturation is real but weakly diagnostic**: no label leakage (shuffled null at chance) and no
  token-identity artifact except possibly `harm`, but sign-corrected random directions already reach
  0.61–0.83, so 1.000 credits contrast consistency rather than the fitted direction.

## Actions

1. `len_frac` is blank for 5 of 6 directions — `probe_select` ran before `directions__length.pt`
   existed. Re-run `probe_select` for all directions (CPU, seconds).
2. Rebuild the `length` contrast at 50/15 pairs, or drop the foil from the selection rule and report
   it only. At 9 pairs it cannot support a ±0.10 tolerance.
3. Decide `eval`: more pairs, or demote it from the four-rival set.
