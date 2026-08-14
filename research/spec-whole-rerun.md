# spec-whole-rerun — the `1K_per_direction` pipeline on a new model

Same tag (`1K_per_direction`), same datasets, same metrics; results are keyed by `<tag>/<model_slug>`
so nothing collides with the Qwen run. Only the model changes — and every absolute layer index with
it.

Run end to end by `notebooks/notebook_1K_gemma.py` (gemma-2-9b-it): every stage is guarded on its
artefacts, and the three manual decisions below are the three points it stops at.

**Dropped vs the Qwen run:** `ablate` (no config where it helped), `cap`, the `length` foil,
`compare_crossed` / §1.2a, the `random` arm, the §5.1 decoding comparison (greedy is reused), and the 2_run second-layer pass — one chosen layer per direction, one steering pass.

**α default everywhere: 0.25, 0.50, 0.75, 1.00**, signed by `cell.RESTORE_SIGN`.

## Preconditions

- `L` = `n_layers` of the new model; reporting band = `round(0.40L)`–`round(0.90L)`. Every layer
  number below is *derived*, never copied from the Qwen run.
- Datasets are model-independent and already built: `story_mode_v2/pairs_1k.jsonl`, `role_play_v2`,
  `eval_v2`, `harm_v2` (`pairs.jsonl` + `pairs_heldout.jsonl`), `jailbreaks/jailbreaks.jsonl`.
- `$BLOB_STORE`: leave unset, or point at a **per-model** path. Blobs are keyed by token ids only, so
  a store shared with a same-tokenizer model (Qwen2.5-7B vs -14B) would silently mix activations.
- `.env` with `OPENAI_API_KEY` **and** `OPENROUTER_API_KEY` — the judge pass exceeds 10k RPD on one
  key.
- `$ATTN_IMPL` where the architecture needs a specific attention kernel: gemma-2 soft-caps its
  attention logits and sdpa drops that, so it runs at `eager` — a different activation and a
  different generation, not a speed knob.
- `M=<model>; export RUN_TAG=1K_per_direction` for every command below.

## 1. extraction

**Needs.** GPU, one model load. 6,400 train + 1,600 held-out prompts.

**Configs.** 4 directions (`story_v2_1k`, `persona_v2`, `harm_v2`, `eval_v2`), 800 train / 200
held-out each, both splits. No `--append-task` (already in the v2 pairs), no `length`. Saturation
curve on: n ∈ {50,…,750} × 5 seeds, plus `cos(d_800_train, d_200_heldout)`.

```bash
python cache_activations.py $M --dataset story_v2_1k,persona_v2,eval_v2,harm_v2 --split train,heldout
python extract_direction.py $M --direction <axis> --curve      # x4
python probe_select.py      $M --direction <axis>               # x4
python plot_figures.py      $M
python -m experiments.common.check_stale $M
```

**Read.** `csv/probe_select__<axis>_rate.csv`: `cohens_dz_train`, `cohens_dz_heldout`,
`mean_paired_cos`, `lopo_ci_lo`. Layer selection is deferred to §2.

## 2. probe_jailbreak_detection

**Needs.** GPU for the readout cache (1,009 framed prompts, `--poles pos`), then CPU. Extraction's
vectors and cached train+heldout views (the threshold sits between each axis's own poles, n_ref =
1,000).

**Configs.** Whole corpus (1,017 − 8 `bare_request` rows), 4 probes, both thresholds (`midpoint`,
`gap_mid`). Run the **all-layers** sweep first — the layer is not chosen yet.

```bash
python experiments/extraction/cache_activations.py $M --dataset jailbreaks --split all --poles pos
python jb_readout.py $M --axes $A                                     # A=story_v2_1k,persona_v2,harm_v2,eval_v2
for R in midpoint gap_mid; do python jb_metrics.py $M --axes $A --threshold $R --all-layers; done
python plot_layer_curves.py $M --all-layers
```

**Read.** `jb_metrics__<rule>__all_rate.csv` per probe × layer × slice, `pct_reads` with `ref_tpr`
beside it (a low `pct_reads` at low `ref_tpr` is the threshold failing, not a reading), and the
per-family curves for the two framing axes.

### ▸ MANUAL GATE — one layer per direction

Pick `L_axis` for each of the four, then record them in `extraction/insights.md` (experiments 3–5
read them from there, never from a JSON):

- default criterion: max `cohens_dz_train`, smallest train↔held-out gap;
- for `story_v2_1k` prefer the **fiction − nonfiction `pct_reads` gap** from §2 where it disagrees —
  that is the layer that discriminates the jailbreak families, and 50_per_direction measured r = 0.00
  between probe quality and steering effect.

A layer outside the band is allowed but must be opted in explicitly (`--allow-out-of-band`, recorded
per manifest). Then re-run §2's headline files at the chosen layers:

```bash
L=story_v2_1k=<l>,persona_v2=<l>,harm_v2=<l>,eval_v2=<l>
for R in midpoint gap_mid; do python jb_metrics.py $M --axes $A --layers $L --threshold $R; done
python plot_layer_curves.py $M
```

→ `jb_metrics__<rule>_chosen.csv`, one row per probe × slice at that probe's own layer.

## 3. cross_probe_detection

**Needs.** No GPU; extraction's vectors + cached views.

**Configs.** 4×4 at the chosen layers, `--diag heldout` (the deployed vector on the 200 held-out
pairs; at n=800 LOPO moves `d_z` by ~0.005). `cohens_dz` emitted — AUROC saturates at n=1,000.
`_matched.csv` at depth 0.65 stays as the common-depth control. No `story_v1` positive control exists
at this tag.

```bash
python cross_auroc.py   $M --axes $A --layers $L --diag heldout
python geometry.py      $M --axes $A --layers $L
python plot_matrices.py $M
```

**Read.** `cross_auroc_chosen.csv` (`excess_over_null`, `cohens_dz_folded`, `delta_excluded`),
`geometry_cos_chosen.csv` (both conventions), `geometry_selfsplit.csv` as the cosine floor. Keep the
band-mean `cos` per pair — §6 selects pairs off it.

## 4. steering_jailbreaks — baseline + the α sweep

**Needs.** GPU, the bulk of the run. `--poles pos` is enough (no `cap`). Judge keys.

**Configs.** Greedy, `max_new_tokens=512`, batch size and `--max-batch-tokens` **pinned and identical
for the baseline and every cell** (Qwen 32 / 65536; gemma-2-9b 16 / 24576, its KV cache being ~6×
bigger per token). Never change them between cells — greedy is bit-reproducible only at fixed batch
composition, and each target is compared against its own no-op.

Single-layer steering, mode `add` only, one primary sign per (set, direction):

| prompt set | goal | `story_v2_1k`, `persona_v2` | `harm_v2`, `eval_v2` |
|---|---|---|---|
| **success** (`steer_single`) | restore refusal | `add` at −α | `add` at +α |
| **refusal** (`steer_induce`) | induce compliance | `add` at +α | `add` at −α |

α ∈ {0.25, 0.50, 0.75, 1.00} → **16 target cells per set**, plus one `noop` per (set, chosen layer)
→ 4 if the four layers are distinct. **≈40 cells + 1 baseline**; ≈20 × n_rows generations per set
(Qwen: 508 successes / 433 refusals → ≈18.8k generations, ≈3 h) and one judge call per row.

```bash
python gen_baseline.py $M --split all --decoding greedy --batch-size 32 --max-batch-tokens 65536
python judge_strongreject.py <results>/meta/gen_baseline.jsonl        # defines both sets
python steer_batch.py $M --script steer_single --jobs jobs_success.json
python steer_batch.py $M --script steer_induce --jobs jobs_refusal.json
python judge_strongreject.py <results>/meta/<cell>.jsonl              # per cell, --concurrency 6
python aggregate.py $M
```

`jobs_*.json` is one argv tail per cell: `["--direction", "<axis>", "--layers", "<l>", "--alpha",
"<±α>"]`, plus `["--arm", "noop", "--layers", "<l>"]` per layer, and `--allow-out-of-band` where the
chosen layer needs it.

**Read.** `aggregate_controls.csv` (`d_*_vs_noop`), the α curve per direction, and `pct_degenerate`
**before** any ΔASR — a mostly-broken cell has a ΔASR and it means nothing. Also `hit_cap_rate` on
the baseline. Report `|Δh|` beside α: α is not comparable across directions or layers.

Smoke test first: `harm_v2 × add × its layer × α=0.50` on the success set.

## 5. Narrativity check (`judge_narrativity.py`)

### ▸ MANUAL GATE — which story cells

Pick the α magnitudes worth judging from §4 (the cells with a readable effect and `pct_degenerate`
low), at `story_v2_1k`'s chosen layer. No generation, judge only.

**Configs.** Forced A/B against each cell's **own no-op** on the same row; both sets (sign resolved
per set); pairs where either side is degenerate excluded; both texts cut to 2,000 chars; `gpt-4o-mini`
at temperature 0, `--provider openrouter`, `--concurrency 8`.

```bash
python judge_narrativity.py $M --direction story_v2_1k --layer <l> --alphas <a1>,<a2> \
    --provider openrouter --concurrency 8
```

**Read.** `pct_cluster` (per `template_id`) with its CI against the 50% null; `pct_neither` and
`pct_picked_A` first — a high escape rate or position bias makes the win rate unreadable. Prediction:
steered side wins on the refusal set (α > 0), loses on the success set (α < 0).

## 6. steer_pairs (§5.6)

### ▸ MANUAL GATE — which pairs

Ordered pair `(a, b)` qualifies only if, at **a's own chosen layer**, `cos(û_a, û_b)` clears the
±3/√d null band (§3's geometry) **and** `a` has a §4 effect to decompose. Expect ≤2 pairs; anything
inside the null band makes `perp` the same experiment as `unprojected` and the script says so.

**Configs.** Two generated arms — `perp_alpha` (necessity) and `par_component`, not normalised
(sufficiency) — at a's chosen layer. `unprojected` is **not generated**: `single_twin` resolves it to
the §4 cell at the same direction, layer, α and set, so α must be one of the four swept there.
`perp_effect` is skipped — `α_eff = α/√(1−c²)` is under 5% for any cosine worth running.
2 arms × n_rows generations per (pair, set).

```bash
python steer_pairs.py $M --pair <a>,<b> --layers <l_a> --alpha <α> --prompt-set <success|refusal> \
    --arms perp_alpha,par_component --decoding greedy --batch-size 16 --max-batch-tokens 24576
python judge_strongreject.py <results>/meta/<cell>.jsonl
python aggregate.py $M
```

`--allow-out-of-band` where a's chosen layer is outside the band; batch parameters must be §4's, or
`single_twin` refuses the reference rather than pairing against one built differently.

**Read.** `perp_alpha` vs `unprojected` (necessity: does `a` still work re-pointed off `b`) and
`par_component` vs `unprojected` (sufficiency: does the b-content in the push carry the effect on its
own).

## Open / carried over

- The threshold in §2 is calibrated off-distribution and there is **no `length` foil** at this tag, so
  "is a high `pct_reads` just prompt length?" is unanswerable here.
- No `random` arm anywhere: nothing in §4–6 is a specificity claim.
- The two prompt sets are defined by the baseline's batch composition and steered at another, so some
  success rows do not comply at steer time. The no-op is the denominator, never the baseline.
- With one layer per direction, a null cell cannot be told apart from a wrong layer.
