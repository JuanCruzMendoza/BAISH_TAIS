# initial_tests — what has been run

Pilot experiments for the story-mode plan (`research/Plan story-mode.md`), Phase 1.
Goal: establish that the axes we want to steer exist, are linearly readable, and are
not artifacts — before scaling to 19.2k prompts and 3 model families.

All scripts: diff-in-means on the **last prompt token** (generation onset, chat
template applied), every layer, one forward pass per prompt. Model defaults to
`Qwen/Qwen2.5-7B-Instruct`; results so far are **Qwen2.5-3B-Instruct** (smoke test).
Layer `l` in the CSVs = `hidden_states[l]`, so `l=0` is embeddings and `l=36` is the
final block.

```bash
python extract_direction.py Qwen/Qwen2.5-3B-Instruct
python extract_direction_tier3.py Qwen/Qwen2.5-3B-Instruct
python deconfound_length_tier3.py Qwen/Qwen2.5-3B-Instruct
python residual_length_tier3.py Qwen/Qwen2.5-3B-Instruct
python steer_narrativity.py Qwen/Qwen2.5-3B-Instruct    # generates; needs 1-4 run first
```

Outputs go to `results/<model>/`, `/` replaced by `_`.

## Two different axes

The plan says "story-mode", which turned out to be two separable things. Keeping them
apart is the main conceptual result of these pilots:

- **fictionality** — invented vs real, *narrative form held fixed* (Tier-1/Tier-2)
- **narrativity** — request wrapped in a story vs stated plainly (Tier-3)

## Datasets (`data/initial_tests/`)

| file | pairs | contrast |
|---|---|---|
| `fiction_vs_real_pairs.jsonl` | 20 | fictionality. **Tier-1** (10): byte-identical body, provenance label swapped (novel/memoir). **Tier-2** (10): full rewrite, never uses the words novel/memoir |
| `tier3_story_vs_bare_pairs.jsonl` | 10 | narrativity. Same request embedded in a short story vs stated plainly. Confounded with length by construction |
| `length_filler_pairs.jsonl` | 30 | length/verbosity nuisance for Tier-3. Same 10 requests padded to story length with non-narrative prose, 3 styles: `expository` (topic-matched facts), `ambient` (topic-neutral description), `oblique` (document framing ending in `"... reads:"`, matching the story's speech act) |
| `story_jailbreaks.jsonl` | 3 | story-wrapped jailbreaks for the causal experiment, from `Casey27/JailbreakPrompts`. `prompt` = what the model sees, `request` = the plain ask kept separate so a judge can score later, `source` = jailbreak family. Grows over time; rows with `FILL_ME` are skipped |
| `length_control_pairs.jsonl` | 8 | legacy. Terse question vs request for a *verbose answer* — that is requested-verbosity, not prompt length. Wrong nuisance for Tier-3; superseded by `length_filler_pairs` |

Tier-3 and filler splits are aligned (requests 1–7 train, 8–10 test) so nuisance
axes are fitted on train requests and matched-eval negatives come from test requests.

## Experiments

### 1. `extract_direction.py` — does a fictionality direction exist and generalize?

Extracts fiction−real from train pairs; reports held-out AUROC, cross-tier transfer
(Tier-1 dir → all Tier-2 and vice versa), and cosine against the legacy length axis.
Cross-tier transfer is the test that matters: a Tier-1-only direction has never seen
a full rewrite, so if it still separates Tier-2 the axis is not a "novel/memoir" word
detector. → `metrics.csv`, `directions.pt`

### 2. `extract_direction_tier3.py` — is narrativity a different axis from fictionality?

Extracts story−bare, then applies it to the fictionality pairs. Near 0.5 there means
narrativity does not read fiction/real. Also reports cosine against the legacy length
axis. → `tier3_metrics.csv`, `tier3_direction.pt`

Known limitation, and the reason experiment 3 exists: story members are ~5× longer
than bare members, so `auroc_t3_heldout` is uninformative on its own.

### 3. `deconfound_length_tier3.py` — is the narrativity vector just length?

Keeps the Tier-3 dataset and the diff-in-means family; attacks length three ways.

| | fix | vector |
|---|---|---|
| 1 | regress the paired differences on Δtokens, take the intercept (contrast extrapolated to zero length delta) | `narrativity_ols` |
| 2 | project out the 3-dim length subspace spanned by the filler styles | `narrativity_orth` |
| 3 | keep the direction, change the statistic: `score ~ n_tokens + is_story`, plus within-token-bin AUROC | — |

Headline columns are the **length-matched** AUROCs (`*_M`): story vs filler-long, both
classes long. `*_M_obl` is sharpest — length, topic and speech act all matched.
`len_ho` is the foil: a pure length vector on the naive eval. Column-by-column legend
is in the script docstring. → `tier3_deconfounded_metrics.csv`,
`tier3_deconfounded_directions.pt`

Verified against a stubbed model with known ground truth
(`hidden = α·tokens·LEN + β·is_story·NAR`): at β=0 the naive metric still reads 1.00
while `ols_M`/`ort_M` correctly collapse to ~0.5; at β=0.5 the deconfounded vectors
recover signal the raw vector misses. That test is not in the repo.

### 4. `residual_length_tier3.py` — is length actually *gone* from `narrativity_orth`?

Experiment 3 showed narrativity **survives** length matching. It never showed length was
removed: `*_M` evals are length-matched, so a vector that still reads length scores the
same on them. And FIX 2's projection is *orthogonal*, which sets `d·b = 0` but not
`dᵀΣb` — the covariance between the narrativity readout and the length readout. Those
differ exactly when length and narrativity are correlated, which is this case
(`len_frac_raw` ≈ 0.5).

Test: a pure-length, no-narrative contrast — **filler-long vs filler-short**, test rows
only (subspace fitted on train rows). Neither class is a story, so any separation is
length. Reported for `narrativity_raw` / `length_pooled` (eval sanity + positive
control), `narrativity_orth` (the answer), and a Σ-orthogonal LEACE-style alternative
`d − Q(QᵀΣQ)⁻¹QᵀΣd`. Σ is shrunk toward `(tr/d)·I` and never materialized as a
hidden×hidden matrix. Also reports `sig_frac_*` (length share of the readout) against
`euc_frac_ort` (≈0 by construction) — a large gap between those two *is* the finding.

`p_ort` is an exact paired sign-flip test; with 9 filler pairs the p floor is 1/512, so
`p > 0.05` means "no leak detectable at this n", not "no leak". → `tier3_residual_length_metrics.csv`,
`tier3_residual_length_directions.pt` (`narrativity_leace`)

Outcomes: no leak → keep `narrativity_orth`, question closed. Leak + `leace_M` holds →
switch to `narrativity_leace`. Leak + `leace_M` collapses → not separable at this n,
escalate to the design-side controls in `research/deconfounding-length.md`.

### 5. `steer_narrativity.py` — does steering narrativity move refusal? (§7a + §7b)

**Not yet run.**

Story-wrapped jailbreaks in, generations out. α < 0 (away from narrativity) should restore
refusal on jailbreaks that succeeded; α > 0 should induce compliance on ones that didn't.
**Both arms come from one sweep** — every prompt runs at both signs plus a shared α=0
baseline, and the §7a/§7b split happens at analysis time by whether that baseline complied
or refused. So a prompt that already refuses is not a wasted row, it is the §7b case.

Sweeps **one layer at a time** so layers can be compared — `SIMULTANEOUS=1` injects at all
of them at once instead, which is a stronger and differently-interpreted intervention.
Default grid is 4 α × 4 layers + 1 baseline = **17 generations per request**;
`LAYERS=18-26 ALPHAS=-2,-1.5,-1,-0.5,0.5,1,1.5,2` restores the full 73-cell sweep.

- **Layers 20, 22, 24, 26.** `ort_M` saturates at 1.00 band-wide and cannot rank layers, so
  the choice comes from 2c's residual-length column. L22 = ~60% depth where steering usually
  bites, and the fictionality best layer, but the **highest** leakage of the four (dev
  −0.130); L24 = 2c's pick, cleanest inside the fictionality overlap L19–24 (§4 matched
  layers); L26 = cleanest overall; L20 fills the gap. The **spread is deliberate** —
  `resid_ort_layer` only works as a confound check if the tested layers differ in leakage.
- **α is symmetric** (`±1, ±2`). Beyond covering both arms, it gives the dose-response
  monotonicity §9 asks for and is the only way to see the damage failure mode: if −α and
  +α restore refusal *equally*, the effect is perturbation damage, not a signed direction.
  Two points per side is the minimum for both; widen α before widening layers if you have
  budget to spend.
- **α is in units of the layer's median activation norm**, not raw multiples of the
  direction. Load-bearing for a layer sweep: residual-stream norms grow with depth, so a
  fixed raw coefficient would make deep layers look weaker than they are.
- Injected at **all** positions (prefill + every decoded token). Greedy decoding, so
  differences across α are steering and not sampling. `MAX_NEW=1024` — high enough that
  `out_tokens` stays a measurement rather than a ceiling, which matters because that column
  is how the toward-shorter side effect from 2c gets checked. Verify the saturation rate at
  the cap before reading it.
- **Layer indexing:** layer `l` = `hidden_states[l]` = output of `blocks[l-1]`. Asserted
  against the direction tensor at startup.
- `nar_proj_final` is the manipulation check, read at the **final** layer — at the steered
  layer it is tautological (add `α·û`, project onto `û`, moves by exactly `α`).
- `resid_ort_layer` carries each layer's residual-length AUROC from experiment 4. **If
  steering power correlates with it across L18–26, the effect is length, not narrativity.**
  insights.md 2c found the leakage varies a lot inside this band, so the sweep is
  informative either way.
- Startup smoke test asserts the hook fires, that α=0 reproduces the baseline exactly, and
  that a large α changes the output — a hook that silently never fires would otherwise look
  like a clean null at every layer.
- Appends after each generation and skips `(id, layer, α)` already present, so an
  interrupted run resumes. → `steer_<direction>.jsonl`. `OUT_DIR` redirects output only —
  point it at mounted Drive on Colab so a disconnect loses nothing and a relaunch resumes.

Controls are one flag away, not run yet: `DIRECTION=length_pooled` (the named length rival)
and `DIRECTION=random` (matched-norm, must **not** restore refusal). No judge here — raw
responses only, so any rubric can score them later.

## What downstream phases should reuse

- fictionality: `directions.pt` → `fiction_tier1` (extract from Tier-1, validate on Tier-2 — never extract from Tier-2)
- narrativity: `tier3_deconfounded_directions.pt` → `narrativity_orth`, **not** `narrativity_raw`
- length as a *rival* direction in the §7 cross-steering matrix: `length_pooled` / `length_slope`.
  It scores 1.00 on the naive Tier-3 eval, so it is a real competitor, not a formality.
- band L19–28 for both axes (see `insights.md`)

## Not addressed here

- 3B only; the plan needs ≥2 families at 7B+.
- n is tiny (10–20 pairs). Held-out cells are 3×3 to 3×9 comparisons — read layer
  bands, never single-layer wiggles.
- Narrativity is still entangled with **fictional framing** (a story implies invented
  events). That is the fictionality axis, and no length control touches it.
- Early-layer results may be lexical (stories contain `found`, `she`, `his`; fillers
  do not). Resolved for fictionality via cross-tier transfer; unresolved for narrativity.
