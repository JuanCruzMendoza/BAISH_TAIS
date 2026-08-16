# cross_probe_detection — insights

## 50_per_direction

Qwen2.5-7B-Instruct, band L11–25, 65 pooled pairs per axis (30 for `length`). 

Own-best layer = peak `mean_paired_cos` on train, in band: 
- `story_v2` L16, 
- `story_v1` L16, 
- `harm` L21, 
- `persona` L12,
- `eval` L14, 
- `length` L13. 

All numbers below are band means unless stated.

### The AUROC matrix is saturated and cannot answer H1

Rows = probe, `*` = diagonal (LOPO, n=50):

| probe | story_v2 | story_v1 | harm | persona | eval | length |
|---|---|---|---|---|---|---|
| story_v2 | 1.000\* | 1.000 | 0.676 | 0.643 | 0.412 | 0.247 |
| story_v1 | 1.000 | 1.000\* | 0.289 | 0.742 | 0.382 | 0.269 |
| harm | 0.800 | 0.516 | 0.987\* | 0.550 | 0.529 | 0.751 |
| persona | 0.832 | 0.880 | 0.472 | 1.000\* | 0.768 | 0.958 |
| eval | 0.242 | 0.149 | 0.583 | 0.933 | 0.949\* | 1.000 |
| length | 0.216 | 0.217 | 0.609 | 0.966 | 0.845 | 1.000\* |
| random | 0.527 | 0.518 | 0.517 | 0.479 | 0.488 | 0.480 |


### The geometry can, and says the axes are distinct


| pair                | cos         |
| ------------------- | ----------- |
| story_v2 – story_v1 | **+0.759**  |
| eval – length       | +0.259      |
| persona – length    | +0.245      |
| persona – eval      | +0.227      |
| story_v1 – persona  | +0.127      |
| harm – length       | +0.101      |
| story_v2 – persona  | +0.093      |
| story_v2 – harm     | **+0.055**  |
| the other 7 pairs   | −0.08…+0.03 |


### Findings

- **H1 holds geometrically.** `story_v2` keeps 98% of its norm outside span{harm, persona, eval,
  length} and its cosine to each is ≤ 0.093 — while the same pairs read 0.64–0.80 in the matrix.
- **`story_v2` – `harm` is the cleanest orthogonality in the study** (cos +0.055) and still reads
  0.80 AUROC. That single cell is the whole argument against the AUROC matrix.
- **`story_v2` ≈ `story_v1`, not =**: cos +0.76, 65% of v1 outside v2. 

---

## 1K_per_direction

Qwen2.5-7B-Instruct, 1,000 pooled pairs per axis, chosen layers: `story_v2_1k` **L23 and L15**,
`persona_v2` L15, `harm_v2` L21, `eval_v2` L9. Each probe is read at **its own** layer, so the
matrices are not symmetric. `story_v2_1k` gets a second row at L15 — its detection-best layer, and
`persona_v2`'s — to test whether story's leakage is a property of the vector or of the read position.
No LOPO — the diagonal is the deployed 800-pair vector on its 200 held-out pairs.

### AUROC and Cohen's d_z at the chosen layers

Rows = probe at L_row, `*` = self-cell (held-out 200). `null` is what 20 random unit directions
already score (folded) on that axis and layer — it, not 0.5, is the reference.

| probe (L) | story_v2_1k | persona_v2 | harm_v2 | eval_v2 |
|---|---|---|---|---|
| story_v2_1k (23) | **1.000\*** / +3.74 | 0.794 / +0.79 | 0.684 / +0.44 | 0.459 / −0.11 |
| story_v2_1k (15) | **1.000\*** / +2.88 | 0.725 / +0.66 | 0.620 / +0.29 | 0.357 / −0.28 |
| persona_v2 (15) | 0.939 / +1.56 | **1.000\*** / +2.45 | 0.193 / −0.70 | 0.825 / +0.88 |
| harm_v2 (21) | 0.855 / +1.06 | 0.452 / −0.06 | **0.985\*** / +1.47 | 0.514 / +0.03 |
| eval_v2 (9) | 0.340 / −0.40 | 0.734 / +0.65 | 0.422 / −0.10 | **0.950\*** / +1.55 |
| *null (folded)* | *0.72* | *0.67* | *0.66* | *0.59* |

Only 3 of 12 off-diagonal cells clear δ=0.60, and 4 read their rival **worse** than a random
direction does. Net of the null the largest leakage is `persona_v2` → `story_v2_1k` (+0.231) — the
AUROC matrix is as uninformative at n=1000 as at n=65.

**Story@L15 leaks no more than story@L23**, and slightly less: its whole off-diagonal row is at or
below L23's (persona 0.725 vs 0.794, harm 0.620 vs 0.684), at a self-cell d_z of 2.88 vs 3.74. Moving
story to the layer where it detects jailbreaks best does not buy it any rival axis.

**Cohen's d_z is what the extra pairs bought.** The diagonal spans 1.47–3.74 where AUROC spans
0.950–1.000, so d_z ranks the axes (`story_v2_1k` ≫ `persona_v2` > `eval_v2` ≈ `harm_v2`) where
AUROC cannot. Off-diagonal it also separates sign from magnitude: `persona_v2` → `harm_v2` is
AUROC 0.193 / d_z −0.70, i.e. a strong *inverted* read, not a null.

**Dropping LOPO cost nothing**: the deployed vector on its 200 unseen pairs reproduces the LOPO
diagonal to ≤0.015 AUROC and ≤0.05 d_z, and most of that gap is the 200-vs-800 sample rather than
the refit — as `extraction`'s ~0.005 d_z estimate predicted.

### Net of the null, the diagonal barely stands out

`excess_over_null` = folded AUROC − what 20 random unit directions get on that axis and layer:

| probe (L)        | story_v2_1k | persona_v2 | harm_v2    | eval_v2    |
| ---------------- | ----------- | ---------- | ---------- | ---------- |
| story_v2_1k (23) | **+0.289**  | +0.146     | +0.017     | −0.047     |
| story_v2_1k (15) | **+0.292**  | +0.098     | −0.015     | +0.048     |
| persona_v2 (15)  | +0.231      | **+0.373** | +0.172     | +0.230     |
| harm_v2 (21)     | +0.094      | −0.121     | **+0.341** | −0.078     |
| eval_v2 (9)      | −0.014      | +0.031     | +0.011     | **+0.356** |

![excess over null](results/1K_per_direction/Qwen_Qwen2.5-7B-Instruct/figures/plot_matrices_excess_over_null.png)

- **The diagonal tops out at +0.29…+0.37**, so a fitted probe beats an arbitrary direction on its own
  axis by less than 0.4 AUROC. This is the same point as `rand±` in `extraction`: paired AUROC
  credits contrast consistency, not the vector.
- **`persona_v2` is the one leaky probe** — its whole row is positive (+0.17…+0.23), and its best
  off-diagonal (+0.231) sits only 0.06 below `story_v2_1k`'s own diagonal (+0.289). Its cosines to
  those axes are +0.170 / −0.051 / +0.081, so the leakage is not geometric: it reads `harm_v2` at
  +0.172 excess from a direction that is orthogonal to it.
- **`eval_v2`'s row is flat at the null** (−0.014…+0.031) and `harm_v2`'s is at or below it except a
  small +0.094 on story — those two read nothing but themselves.
- **Story's two rows are interchangeable net of the null** (+0.292 at L15 vs +0.289 at L23 on its own
  axis, ≤0.05 apart on every rival), so the +0.86 d_z the L23 row has over the L15 one is the axis
  being more separable at L23, not the L15 vector being a worse story probe.

### Geometry — the axes are distinct

Split-half floor at 800 pairs: `story_v2_1k` +0.997, `harm_v2` +0.992, `persona_v2` +0.991,
`eval_v2` +0.968 (reliab. ≥0.98). Null band ±0.050. Cosines with the row vector re-read at the
column's layer:

| | story_v2_1k L23 | story_v2_1k L15 | persona_v2 L15 | harm_v2 L21 | eval_v2 L9 |
|---|---|---|---|---|---|
| story_v2_1k | 1.000 | 1.000 | +0.137 | +0.074 | −0.052 |
| persona_v2 | +0.170 | +0.137 | 1.000 | −0.051 | +0.081 |
| harm_v2 | +0.040 | +0.053 | **−0.240** | 1.000 | −0.016 |
| eval_v2 | −0.012 | −0.045 | **+0.296** | +0.078 | 1.000 |

Residual of each axis outside span{the other three}, band mean: `story_v2_1k` 0.982, `harm_v2`
0.975, `eval_v2` 0.943, `persona_v2` 0.917. Against `story_v2_1k` alone: 0.989–0.999.

**The story vector at its two layers is nearly orthogonal to itself**: cos(d_story[23], d_story[15])
= **+0.206**, against a split-half floor of +0.997. So L15 and L23 are not two readings of one
direction — they are two directions, and every result that turns on which one is deployed has to say
so.

### Findings

- **H1 holds at n=800, more cleanly than at n=50.** Every off-diagonal cosine is ≤0.30 against a
  floor of 0.97–1.00, and `story_v2_1k` keeps 98% of its norm outside the other three — while the
  same pairs read 0.68–0.94 AUROC. Geometry and the matrix disagree exactly as at 50 pairs.
- **`persona_v2` – `eval_v2` is the one real off-diagonal** (cos +0.296, mutual AUROC 0.83 / 0.73).
  Both are framing axes over the same task pool, so this is the pair to watch downstream.
- **`harm_v2` – `persona_v2` is −0.240**, i.e. *anti*-aligned, despite 8% verbatim prompt overlap —
  the overlap cannot be what produces it.
- **`story_v2_1k` is the most isolated axis**: max |cos| to any rival 0.170, residual 0.982.
- **Story@L15 is not persona.** At L15, where `persona_v2` is chosen, cos(story, persona) = **+0.137**
  — below the +0.170 the same pair reads at L23, and 7× below the axes' own +0.99 floor. Whatever
  story@L15 shares with persona in a *steering* result, it is not the direction.
- **The two cosine conventions differ by ~5×** (story–persona +0.032 own-layer vs +0.137/+0.170
  matched). Most of that is the vectors sitting 8 layers apart, not extra orthogonality, so
  `cos_own` must not be quoted as the stronger H1 result — read `cos_matched`.
- **The matrix is asymmetric where the layers are far apart**: `persona_v2` reads story at 0.939 but
  story reads persona at 0.794, purely because L15 and L23 are different read positions.

### Caveats

- **No positive control at this tag.** `story_v1` is absent, so no cell is *supposed* to read high;
  a matrix of nulls is not self-validating here.
- `eval_v2`'s L9 is outside the reporting band L11–25, so its band columns and its null are not
  measured where its cells are.

## 1K_per_direction - Gemma 9B

`google/gemma-2-9b-it`, 1,000 pooled pairs per axis, chosen layers `story_v2_1k` **L28 and L15**,
`persona_v2` L15, `harm_v2` L19, `eval_v2` L8. Each probe is read at **its own** layer, so the
matrices are not symmetric. `--diag heldout`: the deployed 800-pair vector on its 200 held-out pairs.

### AUROC and Cohen's d_z at the chosen layers

Rows = probe at L_row, `*` = self-cell. `null` is what 20 random unit directions score (folded) on
that axis — it, not 0.5, is the reference.

| probe (L) | story_v2_1k | persona_v2 | harm_v2 | eval_v2 |
|---|---|---|---|---|
| story_v2_1k (28) | **1.000\*** / +3.81 | 0.888 / +0.93 | 0.651 / +0.29 | 0.306 / −0.48 |
| story_v2_1k (15) | **1.000\*** / +3.22 | 0.806 / +0.85 | 0.797 / +0.78 | 0.560 / +0.07 |
| persona_v2 (15) | 0.887 / +1.21 | **1.000\*** / +2.54 | 0.463 / +0.09 | 0.520 / +0.11 |
| harm_v2 (19) | 0.959 / +1.70 | 0.486 / −0.02 | **0.990\*** / +1.19 | 0.450 / −0.09 |
| eval_v2 (8) | 0.104 / −1.32 | 0.522 / +0.14 | 0.441 / −0.04 | **1.000\*** / +2.27 |
| *null (folded)* | *0.68* | *0.64* | *0.63* | *0.61* |

7 of 15 off-diagonal cells sit at or below the random null, and 4/12 exclude δ=0.65. **As on Qwen,
the matrix cannot settle H1** — the folded null tracks how separable each axis is by *any* direction,
so a high off-diagonal cell is not evidence of shared structure.

**Unlike Qwen, story@L15 leaks slightly more than story@L28, not less** — harm 0.797 vs 0.651, eval
0.560 vs 0.306 — while its self-cell d_z is lower (3.22 vs 3.81). On Qwen the L15 row was uniformly
at or below L23's. The direction of that difference is small and sits near the null either way.

**`eval_v2` → `story_v2_1k` is an inverted read, not a null**: AUROC 0.104 / d_z −1.32, the single
largest off-diagonal magnitude in the matrix. Qwen's equivalent (eval → story 0.340 / −0.40) was
weaker but had the same sign.

### Net of the null

| probe (L) | story_v2_1k | persona_v2 | harm_v2 | eval_v2 |
|---|---|---|---|---|
| story_v2_1k (28) | **+0.316** | +0.153 | −0.046 | +0.083 |
| story_v2_1k (15) | **+0.316** | +0.071 | +0.100 | −0.041 |
| persona_v2 (15) | +0.203 | **+0.359** | −0.111 | −0.040 |
| harm_v2 (19) | +0.226 | −0.127 | **+0.344** | −0.040 |
| eval_v2 (8) | +0.226 | −0.110 | −0.014 | **+0.354** |

![excess over null](results/1K_per_direction/google_gemma-2-9b-it/figures/plot_matrices_excess_over_null.png)

The diagonal tops out at **+0.32…+0.36**, the same ceiling as Qwen's +0.29…+0.37: a fitted probe
beats an arbitrary direction on its own axis by under 0.4 AUROC. **Story's two rows are again
interchangeable net of the null** (+0.316 both), so the 0.59 d_z advantage L28 has is the axis being
more separable there, not L15 being a worse story probe — exactly the Qwen result.

### Geometry — the axes are distinct

Split-half floor, band mean: `story_v2_1k` +0.996, `persona_v2` +0.989, `harm_v2` +0.987, `eval_v2`
+0.969. Null band ±0.050. Cosines with the row vector re-read at the column's layer:

| | story L28 | story L15 | persona L15 | harm L19 | eval L8 |
|---|---|---|---|---|---|
| story_v2_1k | 1.000 | 1.000 | +0.080 | +0.094 | **−0.171** |
| persona_v2 | +0.136 | +0.080 | 1.000 | −0.006 | +0.020 |
| harm_v2 | +0.024 | +0.098 | +0.017 | 1.000 | −0.008 |
| eval_v2 | −0.073 | +0.014 | +0.037 | −0.040 | 1.000 |

Residual outside span{the other three}, band mean: `story_v2_1k` 0.984, `eval_v2` 0.979, `harm_v2`
0.948, `persona_v2` 0.932. Against `story_v2_1k` alone: 0.992–0.999.

**cos(d_story[28], d_story[15]) = +0.219**, against a split-half floor of +0.996 — so story's two
layers are two *directions*, not two readings of one. Qwen measured +0.206 for the same quantity at
L23/L15. That replication is what justifies carrying both layers downstream.

### Findings

- **H1 holds, and more cleanly than on Qwen.** Every off-diagonal cosine is ≤0.17 (Qwen: ≤0.30)
  against a floor of 0.97–1.00, and the largest pair is `story`–`eval` at −0.171.
- **The one strong Qwen off-diagonal does not replicate.** `persona_v2` – `eval_v2` was +0.296 there
  and is **+0.020** here — the pair to watch downstream on Qwen is a null on gemma.
- **`harm_v2` – `persona_v2` also does not replicate**: −0.240 on Qwen, −0.006 here, despite the same
  8% verbatim prompt overlap. Whatever produced the Qwen anti-alignment was model-specific.
- **`story_v2_1k` is again the most isolated axis** (max |cos| 0.171, residual 0.984), and
  `persona_v2` again the least (0.932).
- **Story@L15 is not persona here either**: cos(story@15, persona@15) = **+0.080**, below Qwen's
  +0.137 and 12× under the axes' own floor. As at 4_run, whatever story@L15 shares with persona in a
  *steering* result is not the direction.
- **Geometry and the AUROC matrix disagree the same way on both models** — 0.79–0.96 AUROC between
  pairs whose cosine is under 0.10.

### Caveats

- No positive control at this tag (`story_v1` absent), so a matrix of nulls is not self-validating.
- `eval_v2` L8 and `persona_v2`/`story_v2_1k` L15 are outside the band L17–38, so their band columns
  and nulls are not measured where their cells are.
