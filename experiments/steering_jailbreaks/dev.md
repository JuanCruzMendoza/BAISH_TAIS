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

Layers, per direction: 3 single + `band` (L11–25). `cap` runs at `band` only, so it is comparable to
`ablate` at an identical layer set.

| direction | single layers | `band` | α | τ |
|---|---|---|---|---|
| `story_v2` | 15, 17, 18 | ✓ | 0.5, 1 | p75 (`cap` only) |
| `story_v1` | 15, 16, 20 | ✓ | 0.5, 1 | p75 (`cap` only) |
| `persona` | 17, 19, 21 | ✓ | 0.5, 1 | p75 (`cap` only) |
| `harm` | 20, 21, 22 | ✓ | 0.5, 1 | — |
| `eval` | 14, 15, 16 | ✓ | 0.5, 1 | — |

Absolute indices for L=28; another model needs them re-derived from §3. `ablate` and `cap` take no α.
**`band` is the only config shared across directions**, so cross-direction comparison reads there;
the singles are own-best sites and confound direction with layer.

≈ 90 cells, ≈ 4,400 generations. Greedy, `max_new_tokens=512`, batched at a pinned size.

### Run order

```bash
M=Qwen/Qwen2.5-7B-Instruct; T=50_per_direction
python gen_baseline.py $M --tag $T                                    # GPU, 100 rows
python judge_strongreject.py <results>/meta/gen_baseline.jsonl        # defines both sets
python steer_single.py $M --tag $T --direction story_v2 --sweep-layers 15,17,18
python steer_single.py $M --tag $T --direction story_v2 --layers band
python steer_single.py $M --tag $T --direction harm --layers band --alpha 0.5
python steer_single.py $M --tag $T --direction story_v2 --mode cap --layers band
python steer_single.py $M --tag $T --arm noop --layers band           # once per layer set
python steer_single.py $M --tag $T --direction story_v2 --arm random --layers band
python steer_induce.py $M --tag $T --direction story_v2 --sweep-layers 15,17,18 --alpha 0.5
python judge_strongreject.py <results>/meta/<cell>.jsonl              # per cell
python aggregate.py $M --tag $T
```

Run `story_v2 × ablate × band` first and read its 3-way outcome rate before the other ~89 cells.

`cap` needs τ from the **two-pole** corpus (framed prompts + bare requests, §0.6), so it requires
`cache_activations.py --dataset jailbreaks --split all` *without* `--poles pos`. `ablate`/`add` do not.

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
