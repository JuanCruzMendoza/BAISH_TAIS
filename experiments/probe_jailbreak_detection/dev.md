# probe_jailbreak_detection — do the probes read jailbreaks as fiction? (H2)

Spec §3. **Objective:** for each direction, what percentage of 100 real jailbreak prompts does the
probe read as that direction — overall, per layer, and per jailbreak category?

## Run order

```bash
# GPU, 100 prompts. Writes into extraction's blob cache.
python experiments/extraction/cache_activations.py Qwen/Qwen2.5-7B-Instruct \
    --dataset jailbreaks --split all --subsample-n 100 --poles pos --tag 50_per_direction
# CPU
python jb_readout.py Qwen/Qwen2.5-7B-Instruct --tag 50_per_direction
python jb_metrics.py Qwen/Qwen2.5-7B-Instruct --tag 50_per_direction
```

Needs `extract_direction.py` per axis and its cached `train` + `heldout` views: the threshold is
placed between each axis's own pole distributions.

## The 100-row subset

Drawn wrapper-diverse, because §0.7 clusters by `template_id` and the corpus is lopsided: 400 of its
425 wrappers are `in_the_wild` singletons while `jailbreak_mimicry`'s 300 rows share **2**. Rules
(`views._jb_template_diverse`): family allocation 35/30/20/15; inside a family, sources get rows ∝
**distinct wrapper count** with a floor of 2, so `in_the_wild` cannot crowd out `strongreject`; inside
a source, round-robin over wrappers, ≤2 rows each counted globally, preferring unseen requests.

→ **100 rows, 93 wrappers, 38 requests**, 17/18 techniques, 95 clean JBB categories.

Rows whose `prompt` equals their `request` are dropped (8 rows, all `technique=bare_request`): they
carry no framing.

## The threshold

A probe outputs a number per prompt, `(h − μ)·û`, so "reads it as this direction" needs a cut. The cut
comes from the probe's own reference poles: score that axis's 65 positive and 65 negative extraction
prompts (pooled train + held-out) at the same layer, and place τ between the two piles.

**Default `midpoint`: τ = ½(mean(pos) + mean(neg))** — the same rule as `probe_select`'s
`acc_at_train_thr`, so both experiments cut in the same place. A jailbreak counts when its readout
lands on the positive pole's side. `--threshold` switches rule, permissive → strict:

| rule | τ | `pct_reads` then means |
|---|---|---|
| `neg_median` | median(neg) | above a typical negative-pole prompt — 50% FPR |
| `neg_p90` / `neg_p95` | quantile of neg | FPR pinned at 10% / 5% |
| `gap_mid` | ½(p95(neg) + p5(pos)) | max-margin: furthest from both poles, tail-insensitive |
| **`midpoint`** | ½(mean(pos) + mean(neg)) | positive side of the pole-mean bisector |
| `pos_p5` | p5(pos) | as extreme as a genuine positive-pole prompt |

`midpoint` uses pole means, which a long tail can drag; `gap_position` measures where τ actually
landed relative to the two rules that bracket it.

## Metrics

Two tables, **band layers only** (L11–25, §0.3).

**`jb_readout_rows.csv`** — the sample manifest, one row per jailbreak: `row_id`, `family`, `source`,
`technique`, `template_id`, `request_sha8`, `category`, `base_task_source`, `split`, `n_chars`,
`n_tokens`.

**`jb_metrics__<rule>_rate.csv`** — per probe × layer × slice.

| column | meaning |
|---|---|
| `probe`, `layer`, `depth` | the direction, the layer, and `layer / L` |
| `group_kind`, `group` | `all`/`all`, then each slice: `family`, `source`, `technique`, `category` (JBB rows only, §3.3). Cells need ≥5 rows |
| `n`, `n_reads` | group size, and how many cleared |
| `pct_reads` | **the headline**: % of the group's jailbreaks whose readout clears τ |
| `threshold` | the τ used, so a cell is auditable |
| `ref_tpr` | % of the *positive* pole clearing the same τ. Near 1.0 the bar is passable, so a low `pct_reads` is a real finding; low `ref_tpr` means τ is too strict to conclude anything |
| `ref_fpr` | % of the negative pole clearing τ. Pinned by the `neg_p*` rules, implicit under `midpoint` |
| `gap_position` | where τ sits between p95(neg) and p5(pos): 0 = permissive edge, 1 = strict edge. Outside [0,1] a tail is dragging the pole means; blank means the poles overlap at those quantiles |
| `n_ref` | reference points per pole (65) |

**`jb_metrics__<rule>_band.csv`** — the same collapsed over the band, per probe × slice.

| column | meaning |
|---|---|
| `probe`, `group_kind`, `group`, `n` | as above |
| `pct_reads_mean` | **the per-direction headline**: mean `pct_reads` over the band |
| `pct_reads_min`, `pct_reads_max` | spread across band layers — a wide range means the layer choice matters |
| `n_layers`, `band_lo`, `band_hi` | which layers were averaged |
| `ref_tpr_mean`, `ref_fpr_mean`, `gap_position_mean` | the three threshold diagnostics, band-averaged |

## Notes

- **`--poles`** on `cache_activations.py`; `build_view`/`load_view_matrix` accept single-arm views, and
  their pos/neg alignment and duplicate checks apply only when both arms exist.
- **`--max-batch-tokens`** (default 16384). The subset holds a 47,308-char prompt; batching it with
  seven short ones pads all eight to ~12k tokens over 29 layers. Length-sorted batching drops the peak
  from 94,616 padded tokens to 11,827.
- **The threshold is calibrated off-distribution:** the poles are ~70-word extraction prompts, the
  jailbreaks a median ~1,180 chars. Read the `length` probe's `pct_reads` first — jailbreaks are long,
  so a high value there means the axis is reading length rather than framing.
