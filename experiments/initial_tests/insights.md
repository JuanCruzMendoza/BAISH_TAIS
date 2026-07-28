# Initial tests — insights

All results: Qwen2.5-3B-Instruct, diff-in-means on the last prompt token. See
`dev.md` for what each script does.

# 1. Fictionality (fiction vs real, narrative form fixed)

**Pilot:** does a linear fiction/real direction exist & generalize? (Qwen2.5-3B-Instruct, 20 pairs, diff-in-means, `extract_direction.py`)

## Findings
- A fiction/real direction **exists and generalizes across topics** — held-out AUROC → 1.0 in mid layers.
- **Not a length artifact:** `cos_fiction_length` ≈ 0 (and slightly negative) in mid–late layers.
- **Cross-tier transfer is asymmetric:**
  - Tier-1 dir → all Tier-2: **0.8–0.9** (peak 0.95–0.97 @ L22–24)
  - Tier-2 dir → all Tier-1: **≤0.79** (mostly 0.6–0.77)
- Tier-1 (byte-identical body, provenance label swapped) = **clean** fiction/real axis. It transfers to Tier-2 even though Tier-2 never uses the words "novel/memoir" → a real concept, **not a lexical detector** (resolves the AUROC=0.83-at-layer-1 worry).
- Tier-2 (full rewrite) = fiction/real **+ genre/setting confound** → contaminated vector, poor transfer. Clincher: Tier-1 is the *easier* eval set, yet `dir_t2` scores only 0.6–0.79 on it while `dir_t1` scores ~0.9 on the *harder* Tier-2 set.

## Decision
- **Extract the direction from Tier-1** (form-matched); use **Tier-2 as held-out realistic validation** — never extract from Tier-2 alone.
- **Best layer ≈ 22** (usable band 19–24): strongest worst-case transfer, orthogonal to length.
- Scale the 32 story/realness prompts as **Tier-1-style form-matched contrasts**; keep narrative rewrites for validation only. (= the "fictionality direction, narrative form fixed" arm of the 2×2.)

# 2. Narrativity (story vs bare request)

## 2a. Baseline, `extract_direction_tier3.py`
- **Separate axis from fictionality ✓** (good decorrelation control): `cos_tier1` ≈ 0; `auroc_on_t1` ≈ 0.5 in the mid band → the story/bare axis does not read Tier-1 fiction/real.
- **Not a usable concept direction ✗**: `auroc_t3_heldout` = 1.0 at *every* layer (incl. L1) and `cos_length` = 0.45–0.63 at L1–3 → dominated by length/surface form, not abstract narrativity. Steering it would move verbosity, not story-mode.
- `auroc_on_t2` far from 0.5 (0.8–0.9 late layers) → Tier-2's fiction/real is itself partly a narrativity contrast → extra evidence Tier-2 is confounded.
- **Decision:** keep only as the narrativity⊥fictionality check. For a real narrativity vector, length-match the bare arm (expository passage = story length) → Tier-3b.

## 2b. Length-deconfounded, `deconfound_length_tier3.py` (Tier-3b — done)
Three fixes on the *same* Tier-3 pairs: OLS intercept on Δtokens (fix 1), projection out of a 3-dim length subspace from `length_filler_pairs.jsonl` (fix 2), length-controlled statistic (fix 3).

- **The confound is confirmed, and the naive metric is worthless**: `len_ho` = 1.00 at every layer — a *pure length* vector aces story-vs-bare exactly as well as the narrativity vector does. Any Tier-3 number without a length control means nothing.
- **Narrativity survives length matching ✓** — this **revises 2a's ✗**. On the length-matched eval (story vs filler-long, both classes long) `ort_M` = 1.00 at every layer, `ort_M_obl` = 1.00 even against oblique fillers (length + topic + speech act all matched), `auroc_binned` = 1.00 from L3, `t_is_story_ort` = 6–14. Three independent length controls agree. The raw vector was contaminated, not empty.
- **~half the raw vector was length**: `len_frac_raw` = 0.47–0.55 in the mid band, 0.73–0.81 at the extremes. Use `narrativity_orth`, never `narrativity_raw`.
- **The legacy length check understated contamination**: `cos_length` (vs `length_control_pairs.jsonl`) is ≤0.23 mid/late and read as clean, while the matched-padding subspace captures ~50% of the norm. Requested-verbosity ≠ prompt length — the 2a "clean in mid layers" reading was a false all-clear.
- **Fix 1 is not trustworthy at n=7**: `cos_slope_len` = 0.03–0.21, i.e. the length axis estimated inside Tier-3 is nearly orthogonal to the filler-estimated one — the two fixes remove different things. `loo_cos_ols` = 0.90–0.95 (borderline), `ols_M_obl` = 0.00 at L1–2. Fix 2 is load-bearing; treat `ols_*` as corroboration only.
- `len_rank` = 3 at every layer → the three filler styles span a genuinely 3-dim nuisance subspace; none was redundant. `raw_M_oblique` is the only per-style negative that ever dips (0.67 @ L1) → oblique is the hardest control, as designed.

**Decision:** narrativity is a usable axis. Steer `narrativity_orth` @ **L19–28** (`len_frac` minimal there, and it overlaps the L19–24 fictionality band → matched layers across directions, as §4 of the plan requires). Promote `length_pooled`/`length_slope` from a cosine check to a **named rival** in the §7 cross-steering matrix — with `len_ho` = 1.00 it is a real competitor.

## Caveats
- 3B smoke test → confirm on 7B+.
- Tiny N (transfer = 100 comparisons; deconfounded held-out cells are 3×9 = 27): read layer **bands**, not single-layer wiggles. A reported 1.00 means 27/27 and is indistinguishable from 0.95.
- `ort_M` = 1.00 at L1–2 is **not** evidence of a concept: stories contain narrative vocabulary (`found`, `she`, `his`) the fillers lack, so early layers may be a word detector. Resolved for fictionality via cross-tier transfer; unresolved for narrativity — another reason to read L19–28 only.
- `t_is_story` pools all 80 items including the 7 train stories → optimistic.
- Narrativity remains entangled with **fictional framing** (a story implies invented events). That is the fictionality axis; no length control touches it.
