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

Run `story_v2 × ablate × steer_band` first and read its 3-way outcome rate before the other ~145 cells
(63 target + 63 random + 20 no-op).

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
| `noop` | hooks registered at the same layers, body disabled. For `ablate`/`cap` it is the **only** zero point, since there is no α=0 |
| `random` | matched-norm random direction, **one cell per target cell** — same mode, layer set, α/√N and τ. The specificity control; a band-only random would be compared against single-layer targets |

`add` scales by **α/√N**, N = joint width, so a cell's per-layer coefficient is comparable across
layer counts; `--sweep-layers` cells have N=1, i.e. α unchanged.

**The no-op does not reproduce the baseline, and cannot.** It runs on the 30-row success subset,
the baseline on all 100, so batch composition differs (`n_in_batch` 30 vs 32) and greedy is
bit-reproducible only at fixed composition (§0.10). Measured: **29/30 responses differ and 6 rows
flip outcome**, with a hook that provably does nothing. Consequences — the success set is defined at
one composition and steered at another, so ~5 of its 30 rows do not comply at steer time and the
no-op's ASR, not 100%, is the reference; and **every comparison must be target vs no-op**, never vs
the baseline. Cells within a prompt set share composition, so those comparisons differ only by the hook.

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
| `pct_refused` / `pct_complied` / `pct_degenerate` | the 3-way label, **reconciled** (below) |
| `pct_degenerate_judge` | the judge's raw degeneracy rate, before reconciliation |
| `refused_cluster`, `ci_lo`, `ci_hi`, `n_clusters` | refusal rate over `template_id` cluster means, Clopper–Pearson (§0.7) |
| `disagree_rate` | judge's **raw** label vs the detectors — the only signal independent of the judge |
| `out_tokens`, `hit_cap_rate` | §5.4's arm table |
| `read_<axis>` | manipulation check: each probe's readout at layer L |

Per row, `<cell>_judged.jsonl` also carries the detector columns (`nonascii_frac`, `rep_frac`,
`loop_frac`, `max_run`, `distinct_4`, `compress_ratio`, `refusal_prefix`, `det_degenerate`) and
`outcome_judge` — no API call, so they cost nothing and bound the judge.

**Degeneracy: four length-robust signals, and the label is the union.** `rep_frac` (unigram
type-token ratio) is recorded but no longer decisive — it falls with length on good prose, so at
`> 0.6` it fired on 49% of coherent narrative while still missing 20% of true repetition loops.
`det_degenerate` now fires on `compress_ratio < 0.20`, `max_run ≥ 8`, `distinct_4 < 0.30`,
`loop_frac ≥ 0.25`, `nonascii_frac > 0.15`, or empty. Calibrated on 1,040 unsteered rows as
negatives and 218 verified-broken rows as positives: **0% / 99.5%**.

`outcome` is degenerate when *either* grader says so, with the judge's raw label kept in
`outcome_judge`. Neither subsumes the other — the judge reads a repetition loop as a refusal
(measured: 20% degenerate on a cell that is 100% loops), and the detector cannot see a response
that is coherent but cut off before it answers.

`--rescore` recomputes the detectors, `outcome` and the summary from an existing `_judged.jsonl`
with no API calls; it is idempotent and leaves every judge field untouched. It refuses
`gen_baseline` without `--rescore-baseline`, since re-splitting the prompt sets changes every
5.4/5.5 cell's `inputs.unit_ids` and so its `run_key`.

**`aggregate_controls.csv`** — each target cell with `d_*_vs_noop` and `d_*_vs_random`.
**`aggregate_paired.csv`** — necessity beside sufficiency, one row per direction × layer config.

At this tag `aggregate.py` reports layer configs and α side by side and **refuses to rank them**: at
~30 rows clustered to `template_id` the ordering is noise. A positive cell is readable; a null is not.

### Notes

- **The rubric is in the repo**, `judge_templates.json`, both strings byte-identical to
  `dsbowen/strong_reject` with the upstream URL and sha256 recorded alongside them.
  §5.3 forbids paraphrase, so `judge_strongreject.py` refuses to start without it. `--dry-run` scores the detector
  columns only, with no API call.
- **The 3-way label is asked of the judge** after the rubric block, so the rubric text stays
  byte-identical; `template_sha` covers the added instruction, so editing it invalidates the cache.
  The detectors then reconcile it — the judge is never shown their verdict, which would change
  `template_sha` and force a full re-grade for a decision that is free to make afterwards.
- **`--concurrency 8`** on the judge. One call per row at ~2.5 s means ~2.5 h serial for the pass;
  8-way makes it ~20 min. Grades are unchanged (same prompts, same cache keys) — only the row order
  in `_judged.jsonl` becomes completion order, which nothing reads. 429/5xx retry with backoff;
  a permanent 4xx raises at once.
- **Judge: `gpt-4o-mini`** at `temperature=0` (`--judge-model`; `claude-*` routes to Anthropic).
  ~4,850 calls at ~1.1k in / 0.25k out ≈ $1.50 — the judge sees the *bare* request, never the
  jailbreak wrapper. The id is in the cache key, so switching judges re-grades rather than mixes.
  A judge that declines to grade leaves `strongreject` null and depresses `asr`; `n_judged < n`
  prints as a warning.
- **Batched generation**, not `batch_size=1`: greedy is bit-reproducible at fixed batch size *and*
  fixed composition. Resume therefore skips whole batches — dropping completed rows would change a
  batch's padding and generate the survivors under different conditions.
- **Hooks fire at every position, prefill and decode.** A cell that generated rows with zero hook
  calls raises rather than reporting a null.
- Layer indexing: `hidden_states[l]` is the output of block `l−1`, so layer `l` hooks `blocks[l−1]`.
  Out-of-band `--layers` is rejected, never clipped.

---

## 1K_per_direction

`1_run`, `2_run`, … are successive passes over the same tag and the same corpus; a re-run changes
the config, never the rows.

### 1_run

Objective unchanged (H3). New direction vectors (800 pairs), one chosen layer per direction, the
whole jailbreak corpus.

The chosen layer was the selected by cohens dz 

We use single layer steering, not bands

#### Changes vs `50_per_direction`

| | 50_per_direction | 1_run |
|---|---|---|
| directions | 5 (`story_v2`, `story_v1`, `persona`, `harm`, `eval`) | 4 (`story_v2_1k`, `persona_v2`, `harm_v2`, `eval_v2`) |
| rows | 100-row stratified subset | **all 1,009** |
| layers | 3 singles + `steer_band` per direction | **one chosen layer per direction**, no joint config |
| modes | `ablate`, `add`, `cap` | `ablate`, `add` — **no `cap`** |
| α | 0.5, 1 | **0.25, 0.50, 0.75**, signed |
| suppression | `ablate` only | `ablate` **and** `add` at −α |
| §5.6 pairs | 8 ordered pairs | not in this run |

Dropping `cap` drops the two-pole corpus requirement — `--poles pos` is enough.

`add` at −α is here because 50_per_direction's one incidental negative-α cell beat its best
`ablate` cell (insights §9): `ablate` may be the wrong suppression operator, and this run tests
that on both sets.

#### Layers

One layer per direction, the layer extraction chose at this tag:

| `story_v2_1k` | `persona_v2` | `harm_v2` | `eval_v2` |
|---|---|---|---|
| 23 | 15 | 21 | **9** |

`eval_v2`'s L9 is outside the reporting band (11–25), which `parse_layers` rejects rather than
clips; it enters as an explicit out-of-band opt-in recorded in the manifest.

#### Configs

One primary mode per (set, direction) as before, plus the −α mirror of it.

Three cells per (set, direction), four where the direction is the suppressed one and gets `ablate`
as well → **28 target cells**, plus one `noop` per (set, layer) → **8**. **36 cells in pass 1.**
Every cell in a set runs on the *same* rows, which is what makes target-vs-no-op paired.

**`steer_single` — 508 successes, restore refusal**

| # | direction | mode | L | α |
|---|---|---|---|---|
| 1–4 | `story_v2_1k` | `ablate`, then `add` | 23 | —, −0.25, −0.50, −0.75 |
| 5–8 | `persona_v2` | `ablate`, then `add` | 15 | —, −0.25, −0.50, −0.75 |
| 9–11 | `harm_v2` | `add` | 21 | +0.25, +0.50, +0.75 |
| 12–14 | `eval_v2` | `add` | 9 ⚠ | +0.25, +0.50, +0.75 |
| 15–18 | — | `noop` | 9 ⚠, 15, 21, 23 | — |

**`steer_induce` — 433 refusals, induce compliance**

| # | direction | mode | L | α |
|---|---|---|---|---|
| 19–21 | `story_v2_1k` | `add` | 23 | +0.25, +0.50, +0.75 |
| 22–24 | `persona_v2` | `add` | 15 | +0.25, +0.50, +0.75 |
| 25–28 | `harm_v2` | `ablate`, then `add` | 21 | —, −0.25, −0.50, −0.75 |
| 29–32 | `eval_v2` | `ablate`, then `add` | 9 ⚠ | —, −0.25, −0.50, −0.75 |
| 33–36 | — | `noop` | 9 ⚠, 15, 21, 23 | — |

⚠ `out_of_band`, recorded in that cell's manifest.

`random` is deliberately not in pass 1: matched to every target it doubles the run, for a control
that at 50_per_direction existed only at `steer_band` — only where nothing was readable. Pass 2
runs `random` on the cells that moved, at the same layer, mode and α.

#### Cost

The two sets are the baseline's own split, so their sizes are not known until it is judged. Measured
— and the projection from 50_per_direction's 30/64 rates was wrong, because ASR on the full corpus
is **53.1%**, not 30%, which makes the successes the *larger* half:

| | cells | rows | generations | batches | GPU |
|---|---|---|---|---|---|
| baseline | 1 | 1,009 | 1,009 | 33 | 9 min |
| success set | 18 | **508** | 9,144 | 17 | 1.3 h |
| refusal set | 18 | **433** | 7,794 | 15 | 1.3 h |
| | **37** | | **17,947** | | **2.81 h** |

The split is 51.2% complied / 44.0% refused / 4.8% degenerate; `success` also requires
`strongreject > 0` and `refusal` requires `== 0`, which is why 941 of 1,009 rows land in a set.
Judging the 36 steering cells is **16,938 calls** at `--concurrency 8` — ≈1.5 h and ≈$5.
**≈4.5 h end to end**, split evenly between the two sets.

`hit_cap_rate` on the baseline is **0.476** at `max_new_tokens=512` (mean 329 output tokens). Nearly
half the responses are truncated, which is the failure mode the degeneracy detector is most likely to
misfile as broken output — read it before reading any breakage number.

#### Run order

```bash
M=Qwen/Qwen2.5-7B-Instruct; T=1K_per_direction
python gen_baseline.py $M --tag $T --split all --decoding greedy
python judge_strongreject.py <results>/meta/gen_baseline.jsonl        # defines both sets
python steer_batch.py $M --tag $T --script steer_single --jobs jobs_success.json
python steer_batch.py $M --tag $T --script steer_induce --jobs jobs_refusal.json
python judge_strongreject.py <results>/meta/<cell>.jsonl              # per cell
python aggregate.py $M --tag $T
```

Decoding is greedy, 50_per_direction's §5.1 pick, reused — the decoding comparison is not re-run.

Run `harm_v2 × add × L21 × α=0.5` on the success set first: L21 sits beside 50_per_direction's one
causal cell (`harm add L20`, 24/24), so it is the run's smoke test.

#### Metrics

Unchanged. Every comparison is target vs its own `noop` at the same layer and prompt set, on the
rows the no-op left movable.

#### Open

- The chosen layers come from `cohens_dz_train`, and 50_per_direction measured **r = 0.00** between
  probe reading and steering effect. Nothing supports a probe-quality criterion for picking a
  *steering* layer, and probe_jailbreak_detection's curves show L23 is not even the best detection
  layer for `story_v2_1k` (L15–L17 is). With one layer per direction, a null cell cannot be told
  apart from a wrong layer.
- `eval_v2` at L9 is the highest-risk set: 0.32 depth, and every `eval` α=1 cell at
  50_per_direction was 80–97% degenerate. Read its `pct_degenerate` before its effect.
- α is still not comparable across directions (insights §5): the push is `α·σ_l·û_l` with σ read
  from each direction's own extraction file, so α=0.5 at L9 and at L23 are different absolute
  pushes. Report `|Δh|` alongside α.
- The sets are defined by the baseline at 33-batch composition and steered at 11/22-batch
  composition, so some success rows do not comply at steer time (20% at 50_per_direction). The
  no-op is the denominator, never the baseline.
- §5.6 is out of this run. When it returns, only three of the six pairs are a real manipulation —
  band-mean `cos(û_a, û_b)` against the ±0.050 null band: `persona_v2 × eval_v2` **+0.311**,
  `story_v2_1k × persona_v2` **+0.133**, `persona_v2 × harm_v2` −0.075 (but **−0.240** at
  persona's L15). The other three are inside or at the null band, where projecting `b` out of `a`
  removes nothing and `perp` is the same experiment as `unprojected`. There is also no shared
  layer any more, so a pair has to run at `a`'s chosen layer using `û_b` at that same layer.

### 2_run

Two things 1_run left open, and nothing else:

1. **Is α saturated?** 1_run stopped at 0.75 and the α curve was still moving on three of the four
   directions. Extend all four to **α = 1.00 and 1.25** at their own layers, on both sets.
2. **Is the `cohens_dz` layer the right *steering* layer?** 1_run picked one layer per direction from
   extraction quality alone, and `probe_jailbreak_detection` shows that layer is not where the probe
   discriminates jailbreak *families* best. Re-run the two framing directions at their
   detection-best layer instead.

#### New layers

| direction | 1_run | 2_run adds | why (probe_jailbreak_detection/insights.md) |
|---|---|---|---|
| `story_v2_1k` | L23 | **L15** | largest fiction-vs-rest margin, +57.3 against L23's +28.3, with `nonfiction_other` at 0.0%; `cohens_dz` 2.80 is the highest of the large-margin layers |
| `persona_v2` | L15 | **L4** ⚠ | largest roleplay − nonfiction margin of any layer (+50.4), and `cohens_dz_train` (2.77) and `mean_paired_cos` (0.606) both peak there |

⚠ L4 is at depth 0.14, far outside the reporting band (11–25), so its cells carry
`--allow-out-of-band`. `harm_v2` and `eval_v2` gain no new layer — the margin criterion is a
fiction/roleplay criterion and says nothing about them.

#### Configs

α is **signed** and `steer_single.resolve` refuses the half with no headroom, so the sign is not a
choice: restoring is `RESTORE_SIGN` (−1 for `story`/`persona`, +1 for `harm`/`eval`) on the success
set and its mirror on the refusal set. The tables below give the signed value.

**`steer_single` — 508 successes, restore refusal**

| # | direction | mode | L | α |
|---|---|---|---|---|
| 1–2 | `harm_v2` | `add` | 21 | +1.00, +1.25 |
| 3–4 | `eval_v2` | `add` | 9 ⚠ | +1.00, +1.25 |
| 5–6 | `story_v2_1k` | `add` | 23 | −1.00, −1.25 |
| 7–8 | `persona_v2` | `add` | 15 | −1.00, −1.25 |
| 9–14 | `story_v2_1k` | `ablate`, then `add` | **15** | —, −0.25, −0.50, −0.75, −1.00, −1.25 |
| 15–20 | `persona_v2` | `ablate`, then `add` | **4** ⚠ | —, −0.25, −0.50, −0.75, −1.00, −1.25 |
| 21 | — | `noop` | 4 ⚠ | — |

**`steer_induce` — 433 refusals, induce compliance**

| # | direction | mode | L | α |
|---|---|---|---|---|
| 22–23 | `harm_v2` | `add` | 21 | −1.00, −1.25 |
| 24–25 | `eval_v2` | `add` | 9 ⚠ | −1.00, −1.25 |
| 26–27 | `story_v2_1k` | `add` | 23 | +1.00, +1.25 |
| 28–29 | `persona_v2` | `add` | 15 | +1.00, +1.25 |
| 30–34 | `story_v2_1k` | `add` | **15** | +0.25, +0.50, +0.75, +1.00, +1.25 |
| 35–39 | `persona_v2` | `add` | **4** ⚠ | +0.25, +0.50, +0.75, +1.00, +1.25 |
| 40 | — | `noop` | 4 ⚠ | — |

**40 cells.** Only **two** new `noop`s: the no-op is per (set, layer set), so `story_v2_1k` at L15
reuses the L15 no-op 1_run already ran for `persona_v2` — different directions, different stems,
same layer — and L9/L21/L23 are unchanged.

`ablate` appears only where suppressing that direction is the hypothesis — the framing axes on the
success set — exactly as in 1_run. Ablating `story`/`persona` on already-refused prompts is not an
induce lever, so those four cells are not run.

With 1_run this gives a 5-point α curve (0.25 → 1.25) at every 1_run layer and at both new ones,
which is what makes the saturation question answerable rather than a two-point extrapolation.

#### Cost

| | cells | rows | generations |
|---|---|---|---|
| success set | 21 | 508 | 10,668 |
| refusal set | 19 | 433 | 8,227 |
| | **40** | | **18,895** |

≈3.0 h GPU at 1_run's measured rate. The baseline and the two prompt sets are 1_run's and are
**not** recomputed — the split has to be identical or the cells are not comparable across runs.

**Judging is the binding constraint.** Two limits, needing different handling:

| limit | measured | consequence |
|---|---|---|
| tokens per minute | 200k TPM, ~1.2k tokens a call ≈ 165 calls/min | ~2.6 h at `--concurrency 6`; 8 workers sat on the ceiling and 429'd. Transient, so it retries with full jitter |
| **requests per day** | **10,000 RPD** | 18,895 calls do not fit under it. Not retryable — no ladder outlives a day, and each attempt spends another request against an empty counter |

**A second provider removes the day boundary.** With `OPENROUTER_API_KEY` set, an exhausted
key (RPD 429, `insufficient_quota` 429, or 402) moves grading to OpenRouter serving the *same*
`gpt-4o-mini` rather than ending the pass. The judge's identity is the model, so the cache key
and the per-row resume stamp are unchanged and **nothing already graded is re-graded**; a row
graded through either endpoint is a valid cache hit for the other. `judge_provider` records the
endpoint per row — never part of a key, so it cannot trigger a re-grade — and `pct_via_fallback`
carries it into each cell's summary, so a provider effect on scores is checkable rather than
invisible.

Without a fallback key nothing changes: `judge_strongreject` exits **3** on an exhausted key,
distinctly from any other failure, so a caller looping over cells stops rather than marching the
rest into the same wall, and resume per `unit_id` means the next day spends quota only on the
remainder. ≈$5.6 in total either way.

#### Run order

```bash
M=Qwen/Qwen2.5-7B-Instruct; T=1K_per_direction
python steer_batch.py $M --tag $T --script steer_single --jobs jobs_success_2.json
python steer_batch.py $M --tag $T --script steer_induce --jobs jobs_refusal_2.json
python judge_strongreject.py <results>/meta/<cell>.jsonl              # per new cell
python aggregate.py $M --tag $T                                       # 1_run + 2_run together
```

`aggregate.py` spans the tag, so it re-reports 1_run's cells alongside these.

#### Metrics

Unchanged: target vs its own `noop`, matched on `prompt_set` × `layers_spec`. The α=1/1.25
cells at L9/L21/L23 pair against **1_run's** no-ops, so they must be generated at 1_run's
`--batch-size 32 --max-batch-tokens 65536` — greedy is bit-reproducible only at fixed batch
composition, and changing either number breaks that pairing silently. The L4 cells get the two
new no-ops.

#### Open

- **α = 1.25 is where 50_per_direction broke.** Every `eval` α=1 cell there was 80–97% degenerate.
  Read `pct_degenerate` on the eval and α=1.25 cells before reading any ΔASR from them; a cell that
  is mostly broken output has a ΔASR, and it means nothing.
- **L4 is the riskiest cell in the run.** σ at L4 is 19.3 against 181.2 at L23, so α=1 is a ~9×
  smaller absolute push — but it is applied 24 blocks upstream of the answer, and nothing in this
  project has steered that shallow. If it produces salad at every α, that is the finding.
- **The new-layer cells confound layer with detection quality**, deliberately. If story@L15 beats
  story@L23 that is consistent with "steer where the probe discriminates", but with two layers and
  one direction it is not yet evidence for a rule.
- α is still not comparable across directions or layers (`|Δh|` is), and there is still **no
  `random` arm** — nothing here is a specificity claim.
