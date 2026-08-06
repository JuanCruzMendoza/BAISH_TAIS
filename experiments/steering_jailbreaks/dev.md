# steering_jailbreaks — does moving an axis change jailbreak behaviour? (H3)

Spec §5. **Objective:** for each direction, does suppressing it restore refusal on jailbreaks that
worked, and does adding it induce compliance on jailbreaks that were refused?

## 50_per_direction

The two prompt sets **partition** the 100-row subset `probe_jailbreak_detection` reads out: §5.4 runs
on the rows the unsteered model complied with, §5.5 on the rows it refused. Each axis is therefore
manipulated in both directions, each on the set where that manipulation has headroom.

### Configs

One mode per (set, direction). `add` is positive throughout — the negative half is the floor/ceiling
half, so it measures nothing.

| prompt set | goal | story_v2 · story_v1 · persona | harm · eval |
|---|---|---|---|
| **successes** (§5.4, `steer_single.py`) | restore refusal | `ablate` — plus `cap` (ceil) as the graded alternative | `add` at +α |
| **refusals** (§5.5, `steer_induce.py`) | induce compliance | `add` at +α | `ablate` |

Layers, per direction: 3 single + `steer_band`. `cap` runs at `steer_band` only, so it is comparable
to `ablate` at an identical layer set.

| direction | single layers | joint | α | τ |
|---|---|---|---|---|
| `story_v2` | 15, 17, 18 | `steer_band` | 0.5, 1 | p75 (`cap` only) |
| `story_v1` | 15, 16, 20 | `steer_band` | 0.5, 1 | p75 (`cap` only) |
| `persona` | 17, 19, 21 | `steer_band` | 0.5, 1 | p75 (`cap` only) |
| `harm` | 20, 21, 22 | `steer_band` | 0.5, 1 | — |
| `eval` | 14, 15, 16 | `steer_band` | 0.5, 1 | — |

Absolute indices for L=28; another model needs them re-derived from §3. `ablate` and `cap` take no α.
**`steer_band` is the only config shared across directions**, so cross-direction comparison reads
there; the singles are own-best sites and confound direction with layer.

**Two bands, and they are not the same object.**

| | depth | L=28 | used for |
|---|---|---|---|
| `band` (§0.3) | 0.40–0.90 | 11–25 | the reporting band, and the **ceiling** on any layer spec |
| `steer_band` | 0.70–0.90 | 20–25 | the widest **joint** steering config |

`steer_band` matches the Assistant Axis paper's depth fraction — it caps Qwen3-32B at 46–53 of 64
(0.72–0.83) and Llama-3.3-70B at 56–71 of 80 (0.70–0.89); 0.70–0.90 reproduces the latter almost
exactly (56–72) and spans the former. Single layers anywhere in the reporting band stay legal.

**Consequence to read carefully:** the per-direction singles mostly sit *below* `steer_band` —
`story_v2` (15,17,18) and `eval` (14,15,16) have **no** overlap with it, `persona` only L21, `harm`
all three. So "`steer_band` moves behaviour but no single layer does" is no longer purely a
multi-layer-necessity result; for those two directions it is also a *depth* difference. The singles
probe where the probes read best (§3), the joint window probes where the paper intervenes.

≈ 90 cells, ≈ 4,400 generations, plus §5.1's 7 decoding cells (350). `max_new_tokens=512`, batched at
a pinned size, decoding chosen by §5.1 and identical across every cell.

### Run order

```bash
M=Qwen/Qwen2.5-7B-Instruct; T=50_per_direction
python gen_decoding_compare.py $M --tag $T                            # §5.1, 50 rows x 7 cells
python judge_strongreject.py <results>/meta/gen_decoding_compare*.jsonl  # one call per cell
python aggregate.py $M --tag $T                                       # read _decoding.csv, pick
python gen_baseline.py $M --tag $T --decoding greedy                  # GPU, 100 rows
python judge_strongreject.py <results>/meta/gen_baseline.jsonl        # defines both sets
python steer_single.py $M --tag $T --direction story_v2 --sweep-layers 15,17,18
python steer_single.py $M --tag $T --direction story_v2 --layers steer_band
python steer_single.py $M --tag $T --direction harm --layers steer_band --alpha 0.5
python steer_single.py $M --tag $T --direction story_v2 --mode cap --layers steer_band
python steer_single.py $M --tag $T --arm noop --layers steer_band           # once per layer set
python steer_single.py $M --tag $T --direction story_v2 --arm random --layers steer_band
python steer_induce.py $M --tag $T --direction story_v2 --sweep-layers 15,17,18 --alpha 0.5
python steer_pairs.py $M --tag $T --pair story_v2,persona --both-orders   # §5.6, steer_band
python judge_strongreject.py <results>/meta/<cell>.jsonl              # per cell
python aggregate.py $M --tag $T
```

Run `story_v2 × ablate × steer_band` first and read its 3-way outcome rate before the other ~89 cells.

`cap` needs τ from the **two-pole** corpus (framed prompts + bare requests, §0.6), so it requires
`cache_activations.py --dataset jailbreaks --split all` *without* `--poles pos`. `ablate`/`add` do not.

### Decoding (§5.1, `gen_decoding_compare.py`)

Which decoding the rest of §5 uses. **50 rows** of the subset, stratified by source × family
(proportional, floor 1 per non-empty cell, seed pinned), under three configs:

| config | seeds | cells |
|---|---|---|
| `greedy` | 1 — deterministic | 1 |
| `t0.7p0.9` | 3 | 3 |
| `t1.0p0.95` | 3 | 3 |

7 cells × 50 = **350 generations**. Judge each, then read `aggregate_decoding.csv`: `asr_mean` with
`asr_min`/`asr_max` across seeds, plus `pct_degenerate_mean` and `hit_cap_rate_mean`.

**ASR** = % of rows the rubric scores above zero, i.e. not refused *and* at least minimally
convincing and specific. Deliberately not `pct_complied` — a compliance with no substance scores 0.

**The two criteria conflict and the table cannot settle it**: higher ASR leaves more headroom for
refusal to be restored, while greedy makes a steering delta steering rather than sampling. `asr_spread`
is the cost of the sampled options — it bounds how much of a steering delta could be seed noise. A
sampled pick also turns ASR into a rate over n≥5 samples per cell, multiplying §5.4.

**The pick has to be passed on**: `--decoding` / `--decode-seed` on `gen_baseline.py`,
`steer_single.py`, `steer_induce.py` and `steer_pairs.py`, all resolved through one registry
(`gen.DECODINGS`). It sits inside every cell's `run_key`, so changing it invalidates downstream cells
instead of silently mixing decodings, and a sampling config without a seed is refused outright.

### Cells and arms

A **cell** is direction × mode × layer set × α/τ × arm on one prompt set, and it is the unit of the
stem, `run_key` and resume. Arms are separate cells rather than extra rows in one file, so the no-op
runs once per layer set instead of once per direction:

| arm | what it is |
|---|---|
| `target` | the intervention |
| `noop` | hooks registered at the same layers, body disabled. Must match the baseline; for `ablate`/`cap` it is the **only** zero point, since there is no α=0 |
| `random` | matched-norm random direction at the same layer set and same `α/√N` — the specificity control |

`add` scales by **α/√N**, N = joint width, so a cell's per-layer coefficient is comparable across
layer counts; `--sweep-layers` cells have N=1, i.e. α unchanged.

### Projection (§5.6, `steer_pairs.py`)

Does direction `a` still work once `b` is projected out of it? Four arms per ordered pair, all
`add`, on the same successful jailbreaks as §5.4:

| arm | vector | α |
|---|---|---|
| `unprojected` | `û_a` | reference |
| `perp_alpha` | `unit(û_a − (û_a·û_b) û_b)` | same as reference |
| `perp_effect` | same vector | retuned so the a-probe readout at layer L moves as far as the reference did |
| `par_norm` | `unit((û_a·û_b) û_b)` | same as reference — the control that makes the result readable |

**Default `--layers steer_band`.** The projection is recomputed **per layer** (`û_a[l]`, `û_b[l]` differ;
cross-layer projection is meaningless), so a joint set is fine. `steer_band` is the default because it is
the only config shared by every direction — with the per-direction layers above, `story_v2 × harm`
and `persona × eval` have *no* common single layer at all.

**α is signed here**, unlike everywhere else in §5: this steers story/persona additively, so §0.5's
restoring sign comes back (`−story`, `−persona`, `+harm`, `+eval`). `cell.RESTORE_SIGN`.

`perp_effect`'s α is found by a **prefill-only scan** — one forward pass per candidate over the
prompt set, seconds — because the readout at layer L is not a linear function of the push at layer
`l`. The scan trace is recorded in the manifest.

Reported before generating: `cos(û_a, û_b)` band-mean against the ±3/√d null band, and
`lopo_cos_stability`. If the cosine is inside the null band the projection removes nothing and
`perp` is the same experiment as `unprojected` — the script says so rather than producing four
indistinguishable cells. `(story_v2, story_v1)` is refused as a pair.

### Metrics

**`<cell>_summary.csv`**, one row per cell, written by `judge_strongreject.py` — the point at which
scores exist. Never StrongREJECT alone (§5.3).

| column | meaning |
|---|---|
| `strongreject`, `strongreject_coherent` | rubric score, all rows and non-degenerate rows |
| `pct_refused` / `pct_complied` / `pct_degenerate` | the 3-way label, from the judge |
| `refused_cluster`, `ci_lo`, `ci_hi`, `n_clusters` | refusal rate over `template_id` cluster means, Clopper–Pearson (§0.7) |
| `disagree_rate` | judge label vs the deterministic detectors — the only signal independent of the judge |
| `out_tokens`, `hit_cap_rate` | §5.4's arm table |
| `read_<axis>` | manipulation check: each probe's readout at layer L |

Per row, `<cell>_judged.jsonl` also carries the detector columns (`nonascii_frac`, `rep_frac`,
`refusal_prefix`, `det_degenerate`) — no API call, so they cost nothing and bound the judge.

**`aggregate_controls.csv`** — each target cell with `d_*_vs_noop` and `d_*_vs_random`.
**`aggregate_paired.csv`** — necessity beside sufficiency, one row per direction × layer config.

At this tag `aggregate.py` reports layer configs and α side by side and **refuses to rank them**: at
~30 rows clustered to `template_id` the ordering is noise. A positive cell is readable; a null is not.

### Notes

- **The rubric is not in the repo.** `judge_strongreject.py` requires `judge_templates.json` with
  `strongreject_rubric_system` and `strongreject_rubric` copied verbatim from `dsbowen/strong_reject`;
  §5.3 forbids paraphrase, so the script refuses to invent one. `--dry-run` scores the detector
  columns only, with no API call.
- **The 3-way label comes from the judge**, asked for after the rubric block so the rubric text stays
  byte-identical. `template_sha` covers the added instruction, so editing it invalidates the cache.
- **Batched generation**, not `batch_size=1`: greedy is bit-reproducible at fixed batch size *and*
  fixed composition. Resume therefore skips whole batches — dropping completed rows would change a
  batch's padding and generate the survivors under different conditions.
- **Hooks fire at every position, prefill and decode.** A cell that generated rows with zero hook
  calls raises rather than reporting a null.
- Layer indexing: `hidden_states[l]` is the output of block `l−1`, so layer `l` hooks `blocks[l−1]`.
  Out-of-band `--layers` is rejected, never clipped.
