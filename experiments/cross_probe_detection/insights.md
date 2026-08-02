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
