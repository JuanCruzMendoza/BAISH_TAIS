# Initial tests — insights

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

## Tier-3 — narrativity (story vs bare request), `extract_direction_tier3.py`
- **Separate axis from fictionality ✓** (good decorrelation control): `cos_tier1` ≈ 0; `auroc_on_t1` ≈ 0.5 in the mid band → the story/bare axis does not read Tier-1 fiction/real.
- **Not a usable concept direction ✗**: `auroc_t3_heldout` = 1.0 at *every* layer (incl. L1) and `cos_length` = 0.45–0.63 at L1–3 → dominated by length/surface form, not abstract narrativity. Steering it would move verbosity, not story-mode.
- `auroc_on_t2` far from 0.5 (0.8–0.9 late layers) → Tier-2's fiction/real is itself partly a narrativity contrast → extra evidence Tier-2 is confounded.
- **Decision:** keep only as the narrativity⊥fictionality check. For a real narrativity vector, length-match the bare arm (expository passage = story length) → Tier-3b (TODO).

## Caveats
- 3B smoke test → confirm on 7B+.
- Tiny N (transfer = 100 comparisons): read layer **bands**, not single-layer wiggles.
