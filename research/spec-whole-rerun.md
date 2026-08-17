# spec-whole-rerun — the `1K_per_direction` pipeline on a new model

Same tag (`1K_per_direction`), same datasets, same metrics; results are keyed by `<tag>/<model_slug>`
so nothing collides with the Qwen run. Only the model changes — and every absolute layer index with
it.

Run end to end by one notebook per model — `notebooks/notebook_1K_gemma.py` (gemma-2-9b-it),
`notebooks/notebook_1K_qwen32B.py` (Qwen2.5-32B-Instruct): every stage is guarded on its artefacts,
and the manual decisions below are the points it stops at.

**Only what needs the GPU runs on the GPU box** — plus whatever is free there because the data is
already resident. §J has the local commands:

- **every judge pass, always local** — API-bound, ~23k calls over 3–5 h, and a rented GPU idling
  through it is the most expensive hour in the run. No API key reaches the instance.
- **cross_probe_detection (§3), whichever side already holds the pole cache.** CPU-only and ~2 min,
  but it reads the pole cache (2.23 GiB on gemma, 4.79 on the 32B). So it is not pinned to a
  machine: it runs wherever that cache already is, and downloads it nowhere.

The two sides hand off through the Hub, each pushing its own scope.

**The pole cache moves as `acts/blobs.tar`, never as loose blobs.** Both directions of this are load-
bearing and both were wrong during the gemma run:

- *Pull* `['*/acts/blobs.tar', '*/acts/views/**']`, not `'*/acts/**'`. One request instead of 7,732.
  The request count is what exhausts the Xet read-token quota and 429s, and it is **the same 7,731 at
  a 32B as at a 7B** — only the bytes scale. `pack=True` unpacks the tar, so the tree the scripts
  read is identical either way.
- *Push* holds `*/acts/blobs/*` back unconditionally (`ckpt.push`), so the tar is the only stored
  form. It used to hold them back only under `pack=True`, and the extraction scope is also pushed
  unpacked for vectors and figures — so `msg="figures"` uploaded all 7,731 loose `.npy` beside the
  tar and the cache was stored **twice**: 4.46 GiB on the Hub where 2.23 GiB does, with every
  `'*/acts/**'` pull paying for both.

**Dropped vs the Qwen run:** `ablate` (no config where it helped), `cap`, the `length` foil,
`compare_crossed` / §1.2a, the `random` arm, and the §5.1 decoding comparison (greedy is reused).

**Chosen layers.** Per model, derived at gate 1 and never carried across:

| model | L | d | band | chosen |
|---|---|---|---|---|
| gemma-2-9b-it | 42 | 3584 | 17–38 | `story_v2_1k` **L28 + L15**, `persona_v2` L15, `harm_v2` L19, `eval_v2` L8 |
| Qwen2.5-32B-Instruct | 64 | 5120 | 26–58 | *undecided — gate 1* |

Story keeps two layers wherever its criteria disagree (on gemma by 13: L28 is the `cohens_dz_train`
peak, L15 the fiction − nonfiction `pct_reads` peak), because with one layer a null cell cannot be
told apart from a wrong layer. That is also the only replicated steering result so far — L15 beat
L28 on gemma and L11 beat L19 on the 7B, i.e. `cohens_dz` picked the worse layer both times — so it
is not an optional extra. A chosen layer outside the band carries `--allow-out-of-band` (gemma's L15
and L8 do).

**α default everywhere: 0.25, 0.50, 0.75, 1.00**, signed by `cell.RESTORE_SIGN`.

## Preconditions

- `L` = `n_layers` of the new model; reporting band = `round(0.40L)`–`round(0.90L)`. Every layer
  number below is *derived*, never copied from the Qwen run.
- Datasets are model-independent and already built: `story_mode_v2/pairs_1k.jsonl`, `role_play_v2`,
  `eval_v2`, `harm_v2` (`pairs.jsonl` + `pairs_heldout.jsonl`), `jailbreaks/jailbreaks.jsonl`.
- `$BLOB_STORE`: leave unset, or point at a **per-model** path. Blobs are keyed by token ids only, so
  a store shared with a same-tokenizer model would silently serve the wrong activations —
  **Qwen2.5-32B and Qwen2.5-7B share a tokenizer**, so this is a live risk on that run, not a
  hypothetical. Its notebook raises if the variable is set at all.
- `.env` **on the judging machine** with `OPENAI_API_KEY` and `OPENROUTER_API_KEY` — the sweep
  exceeds 10k RPD on one key. The GPU box needs only `HF_TOKEN`.
- `$ATTN_IMPL` where the architecture needs a specific attention kernel: gemma-2 soft-caps its
  attention logits and sdpa drops that, so it runs at `eager` — a different activation and a
  different generation, not a speed knob. Qwen2.5 needs none; leave it unset, which is also the only
  choice with an sm_120 kernel if the box is Blackwell.
- **Not Qwen3-32B**, though `Plan story-mode.md` names it. It is a hybrid-thinking model whose chat
  template defaults to thinking on, so `max_new_tokens=512` would be spent inside `<think>` and
  StrongREJECT would grade a truncated trace; disabling it means patching `templated()` and moving
  the read position for one model only. Qwen2.5-32B-Instruct is L=64 × 5120 either way, so every
  derived number is unchanged, and it makes the third model a clean **scale** control against the
  7B (same tokenizer, template and architecture) while gemma covers architecture.
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

### ▸ MANUAL GATE — the layer(s) per direction

Pick `L_axis` for each of the four — more than one where the criteria disagree — then record them in
`extraction/insights.md` (experiments 3–5 read them from there, never from a JSON):

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

## 3. cross_probe_detection — **wherever the pole cache already is**

**Needs.** No GPU, ~2 min of CPU, but it *does* need extraction's `acts/` — the off-diagonal AUROC
is computed on the cached pole activations, not on the vectors — plus the vectors themselves. That
cache is one `blobs.tar` of 7,731 blobs at `(L+1, d)` fp16 — **2.23 GiB on gemma, 4.79 GiB on the
32B**; only the bytes scale, the file count does not. Its size is what decides where this runs, and
the rule is that it never causes the download:

- **On the GPU box, at gate 1, if extraction ran in that same session.** The cache is on local disk,
  so this is free, and it follows the layers just entered. Best case — take it when you can.
- **Locally (§J.0) otherwise.** A fresh instance resuming from complete extraction manifests never
  pulls `acts/`, so the notebook cell skips rather than dragging it back.

The notebook implements exactly that: it checks `acts/views` and `acts/blobs` and does nothing if
either is empty. Both sides pull the (small) `cross_probe_detection` scope so each can tell
"computed at these layers" from "not computed here"; only the side that computes it pushes.

One host-RAM caveat for the GPU-box branch: axes load one at a time at ~0.8 GB each on a 7B, but
**~2.7 GB per axis at a 32B** (1,000 pairs × 2 poles × 65 layers × 5120 dims, fp32) — no longer free
alongside a resident 65.6 GB model. On a thin instance, take the local branch even when the cache
is there.

**Configs.** 4×4 at the chosen layers, `--diag heldout` (the deployed vector on the 200 held-out
pairs; at n=800 LOPO moves `d_z` by ~0.005). `cohens_dz` emitted — AUROC saturates at n=1,000.
`_matched.csv` at depth 0.65 stays as the common-depth control. No `story_v1` positive control exists
at this tag.

`--layers` takes `axis=layer[+layer]`, and a `+` adds a **second probe row** for that axis, not a
second axis. Story gets both of its steering layers (`story_v2_1k=28+15`), as at 4_run: the two are
*different read positions*, not two readings of one vector (cos +0.206 on Qwen, +0.219 here), so
each needs its own row before a steering cell at either can be interpreted. This is also what
answers "is story@L15 just persona?" — at 4_run the answer was **no** (cos +0.137, and the shared
behaviour was a steering result, not a direction). §6 reads `cos(story@15, persona@15)` from
`geometry_cos.csv`, which spans every band layer either way.

**Read.** `cross_auroc_chosen.csv` (`excess_over_null`, `cohens_dz_folded`, `delta_excluded`),
`geometry_cos_chosen.csv` (both conventions), `geometry_selfsplit.csv` as the cosine floor. Keep the
band-mean `cos` per pair — §6 selects pairs off it.

## 4. steering_jailbreaks — baseline + the α sweep

**Needs.** GPU, the bulk of the run. `--poles pos` is enough (no `cap`). **Two GPU sessions**, with a
local judge pass between them: the two prompt sets *are* the baseline's 3-way labels, so the sweep
cannot be built until the baseline is graded (§J.1).

**Configs.** Greedy, `max_new_tokens=512`, batch size and `--max-batch-tokens` **pinned and identical
for the baseline and every cell**. Never change them between cells — greedy is bit-reproducible only
at fixed batch composition, each target is compared against its own no-op, and the pin can never be
*lowered* afterwards, so it is sized to fit rather than to be fast:

| model | KV/token | weights (bf16) | pin | peak KV |
|---|---|---|---|---|
| Qwen2.5-7B | 56 KiB | 15 GB | 32 / 65536 | ~3.5 GiB |
| gemma-2-9b | 336 KiB | 18 GB | 16 / 24576 | ~10 GiB |
| Qwen2.5-32B | 256 KiB | 65.6 GB | 32 / 32768 | ~12 GiB (96 GB card) |

`--max-batch-tokens` bounds `len(batch) × longest_prompt`, so at 32768 a batch reaches 32 only when
its longest prompt is ≤1024 — `jailbreaks` is median 288, p95 1,063, max 8,688 tokens, and the long
tail falls into batches of 3 on its own.

Single-layer steering, mode `add` only, one primary sign per (set, direction):

| prompt set | goal | `story_v2_1k`, `persona_v2` | `harm_v2`, `eval_v2` |
|---|---|---|---|
| **success** (`steer_single`) | restore refusal | `add` at −α | `add` at +α |
| **refusal** (`steer_induce`) | induce compliance | `add` at +α | `add` at −α |

α ∈ {0.25, 0.50, 0.75, 1.00} at **each chosen (axis, layer)** → 5 pairs × 4 α = **20 target cells per
set**, plus one `noop` per (set, layer) — 4, since story@L15 and persona@L15 share theirs. **48 cells
+ 1 baseline**, ≈24 × n_rows generations per set (≈22.6k at Qwen's 508/433 split) and one judge call
per row.

```bash
# GPU session A
python gen_baseline.py $M --split all --decoding greedy --batch-size 32 --max-batch-tokens 65536
#            -> push, then §J.1 locally: judging it defines both prompt sets
# GPU session B, after §J.1 is back on the Hub
python steer_batch.py $M --script steer_single --jobs jobs_success.json
python steer_batch.py $M --script steer_induce --jobs jobs_refusal.json
#            -> push, then §J.2 locally (judge every cell + aggregate)
```

`jobs_*.json` is one argv tail per cell: `["--direction", "<axis>", "--layers", "<l>", "--alpha",
"<±α>"]`, plus `["--arm", "noop", "--layers", "<l>"]` per layer, and `--allow-out-of-band` where the
chosen layer needs it. An axis with two layers emits one cell per layer; the stems differ by `L<l>`,
so they never collide.

**§2 and §3 take one layer per probe** (`jb_metrics`, `cross_auroc`, `geometry` all do), so their
`_chosen` tables run at each axis's **first** layer — story L28. Story's L15 row is not lost: it is
in the per-layer files those runs also write (`jb_metrics__<rule>__all_rate.csv`,
`cross_auroc_tensor.csv`, `geometry_cos.csv`).

**Read.** `aggregate_controls.csv` (`d_*_vs_noop`), the α curve per direction, and `pct_degenerate`
**before** any ΔASR — a mostly-broken cell has a ΔASR and it means nothing. Also `hit_cap_rate` on
the baseline. Report `|Δh|` beside α: α is not comparable across directions or layers.

Smoke test first: `harm_v2 × add × its layer × α=0.50` on the success set.

## 5. Narrativity check (`judge_narrativity.py`) — local, no GPU

### ▸ MANUAL GATE — which story cells

Pick the α magnitudes worth judging from §4 (the cells with a readable effect and `pct_degenerate`
low), at `story_v2_1k`'s chosen layer. Judge only: it runs off the existing `_judged.jsonl`, so it
never touches the GPU box (§J.3).

**Configs.** Forced A/B against each cell's **own no-op** on the same row; both sets (sign resolved
per set), **each with its own α magnitude**, since the two sides peak at different α; pairs where
either side is degenerate excluded; both texts cut to 2,000 chars; `gpt-4o-mini` at temperature 0,
`--provider openrouter`, `--concurrency 8`.

```bash
python judge_narrativity.py $M --direction story_v2_1k --layer <l> \
    --alphas success=<a_restore>,refusal=<a_induce> --provider openrouter --concurrency 8
```

**Read.** `pct_cluster` (per `template_id`) with its CI against the 50% null; `pct_neither` and
`pct_picked_A` first — a high escape rate or position bias makes the win rate unreadable. Prediction:
steered side wins on the refusal set (α > 0), loses on the success set (α < 0).

## 6. steer_pairs (§5.6)

### ▸ MANUAL GATE — which pairs

Ordered pair `(a, b)` qualifies only if, at **the anchor's steered layer**, `cos(û_a, û_b)` clears
the ±3/√d null band (§3's geometry) **and** `a` has a §4 effect there to decompose. Expect ≤2 pairs;
anything inside the null band makes `perp` the same experiment as `unprojected` and the script says
so. An anchor with two layers names one (`a@L`) — the projection is same-layer, so it is not a free
parameter, and `story_v2_1k@15 × persona_v2` is the pair to expect: L15 is persona's own layer, so
both vectors are compared where both are deployed.

**Configs.** Two generated arms — `perp_alpha` (necessity) and `par_component`, not normalised
(sufficiency) — at a's chosen layer. `unprojected` is **not generated**: `single_twin` resolves it to
the §4 cell at the same direction, layer, α and set, so α must be one of the four swept there.
`perp_effect` is skipped — `α_eff = α/√(1−c²)` is under 5% for any cosine worth running.
2 arms × n_rows generations per (pair, set).

```bash
# GPU, one invocation per (pair, set)
python steer_pairs.py $M --pair <a>,<b> --layers <l_a> --alpha <α> --prompt-set <success|refusal> \
    --arms perp_alpha,par_component --decoding greedy --batch-size 16 --max-batch-tokens 24576
#            -> push, then §J.4 locally (judge the arms + aggregate)
```

`--allow-out-of-band` where a's chosen layer is outside the band; batch parameters must be §4's, or
`single_twin` refuses the reference rather than pairing against one built differently.

**Read.** `perp_alpha` vs `unprojected` (necessity: does `a` still work re-pointed off `b`) and
`par_component` vs `unprojected` (sufficiency: does the b-content in the push carry the effect on its
own).

## J. Off the GPU box — every judge pass, and cross-probe when the box skipped it

All CPU + API. Run in the repo on a machine with `.env` holding `OPENAI_API_KEY` and
`OPENROUTER_API_KEY`; the notebook prints the same commands with the paths filled in. Common
preamble, and `$D` is the steering results dir:

```bash
M=google/gemma-2-9b-it            # or Qwen/Qwen2.5-32B-Instruct
T=1K_per_direction; export RUN_TAG=$T
R=JuanCruzMendoza/BAISH_TAIS
D=experiments/steering_jailbreaks/results/$T/${M//\//_}
pull() { python -c "from experiments.common import ckpt; ckpt.pull('$R', experiment='steering_jailbreaks', tag='$T')"; }
push() { python -c "from experiments.common import ckpt; ckpt.push('$R', experiment='steering_jailbreaks', tag='$T', msg='$1')"; }
```

**J.0 — cross_probe_detection (§3), only if the GPU box skipped it.** Its gate-1 cell prints which
branch it took. Once, after gate 1; no GPU, no judge, ~2 min of compute — the `blobs.tar` is
the pull, and only the first time. Skip this whole block if the notebook already ran it.

```bash
python -c "from experiments.common import ckpt; ckpt.pull('$R', experiment='extraction', tag='$T', subpaths=['*/vectors/**', '*/meta/**', '*/acts/blobs.tar', '*/acts/views/**'], pack=True)"
A=story_v2_1k,persona_v2,harm_v2,eval_v2
L=story_v2_1k=28+15,persona_v2=15,harm_v2=19,eval_v2=8   # gate 1's output; `+` = a second probe row
python experiments/cross_probe_detection/cross_auroc.py   $M --tag $T --axes $A --layers $L --diag heldout
python experiments/cross_probe_detection/geometry.py      $M --tag $T --axes $A --layers $L
python experiments/cross_probe_detection/plot_matrices.py $M --tag $T
python -c "from experiments.common import ckpt; ckpt.push('$R', experiment='cross_probe_detection', tag='$T', msg='cross-probe')"
```

Nothing in §4 waits on this — run it while the GPU sweeps. §6's gate reads its
`geometry_cos.csv`, so it has to be done before the pairs are chosen, not before they are steered.

**J.1 — the baseline (blocks the sweep).** 1,009 calls, ~10 min.

```bash
pull
python experiments/steering_jailbreaks/judge_strongreject.py $D/meta/gen_baseline.jsonl --concurrency 6
push "baseline judged"
```

Read the split it prints (`pct_complied` / `pct_refused` / `pct_degenerate`) and `hit_cap_rate`
before anything else. Then re-run the notebook: it pulls this back and the sweep starts.

**J.2 — the sweep.** 48 cells, ~23k calls, 3–5 h, ≈$6.8. Over the 10k/day cap, so the OpenRouter
key is what keeps it one day.

```bash
pull
for f in $D/meta/steer_*.jsonl; do
  [ -e "$f" ] || continue                       # nullglob is off: an unmatched glob is literal
  case "$f" in *_judged.jsonl) continue;; esac
  python experiments/steering_jailbreaks/judge_strongreject.py "$f" --concurrency 6
done
python experiments/steering_jailbreaks/aggregate.py $M --tag $T
push "sweep judged"
```

`--concurrency 6`, not 8: the binding limit is 200k TPM ≈ 165 calls/min and 8 workers sit on it.
A cell whose `csv/<stem>_summary.csv` says every row was scored is skipped, and grading resumes per
row, so re-running the loop after an interruption costs only the remainder. Exit **3** is the daily
cap with no fallback left — stop and resume tomorrow rather than marching the rest into it.

**J.3 — narrativity (§5), after picking the α magnitudes off `aggregate_controls.csv`.**

```bash
for L in 28 15; do            # one invocation per story layer: the stem carries the layer
  python experiments/steering_jailbreaks/judge_narrativity.py $M --tag $T \
      --direction story_v2_1k --layer $L \
      --alphas success=<a_restore>,refusal=<a_induce> --provider openrouter --concurrency 8
done
push "narrativity"
```

**One invocation per layer, never two.** The stem is `judge_narrativity__<axis>__L<l>` and carries
neither the α nor the set, while both are in `config` — so a second invocation at the same layer has
a different `run_key`, and `Run.__enter__` archives the first one's `_pairs.jsonl` and
`_narrativity.csv` into `meta/_archive/`. Splitting the two sides across two commands silently keeps
only the second.

**Give each side its own magnitude.** They peak in different places: restore keeps working past the α
where induce has already turned over and gone degenerate. At gemma's story@L15, α=1.5 is 6.5%
degenerate on successes but **95%** on refusals, and degenerate pairs are dropped — so a shared
magnitude either wastes the restore side's headroom or spends calls on a cell with nothing coherent
left to judge. `--alphas 0.5,1.5` (no `=`) still applies both to every set; `success=1.25+1.5` lists
several for one side.

`--provider openrouter` deliberately: same model and same cache key, so it is not a different judge —
it just leaves the OpenAI day for the 20×-larger StrongREJECT pass.

**J.4 — the pairs (§6).** ~8 cells, well under a day's cap.

```bash
pull
for f in $D/meta/steer_pairs__*.jsonl; do
  [ -e "$f" ] || continue                       # nullglob is off: an unmatched glob is literal
  case "$f" in *_judged.jsonl) continue;; esac
  python experiments/steering_jailbreaks/judge_strongreject.py "$f" --concurrency 6
done
python experiments/steering_jailbreaks/aggregate.py $M --tag $T   # spans the tag: sweep + pairs
python -m experiments.common.check_stale $M $T
push "pairs judged"
```

`check_stale` flags the jailbreak activations `--view-only` leaves uncomputed — expected, not a
finding.

## Open / carried over

- The threshold in §2 is calibrated off-distribution and there is **no `length` foil** at this tag, so
  "is a high `pct_reads` just prompt length?" is unanswerable here.
- No `random` arm anywhere: nothing in §4–6 is a specificity claim.
- The two prompt sets are defined by the baseline's batch composition and steered at another, so some
  success rows do not comply at steer time. The no-op is the denominator, never the baseline.
- With one layer per direction a null cell cannot be told apart from a wrong layer, which is why
  story runs at two. The other three still carry that ambiguity.
- **Story's two layers confound layer with criterion, deliberately.** If L15 beats L28 that is
  consistent with "steer where the probe discriminates", but with two layers on one direction it is
  not yet evidence for a rule.
