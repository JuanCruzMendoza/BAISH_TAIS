# steering_jailbreaks — insights

## 50_per_direction

Qwen2.5-7B-Instruct, greedy, 100-row jailbreak subset split by the baseline into 30 successes
(§5.4, restore refusal) and 64 refusals (§5.5, induce compliance). 63 target + 10 random + 20 no-op
cells. All numbers are **paired against the cell's own no-op**, on the rows that were movable there
(24 of 30 on the success set, 62 of 64 on the refusal set).

### The directions that move behaviour

`moved` = rows that flipped the wanted way; `broke` = rows that went degenerate instead.

| direction | mode | layer | α | set | movable | moved | broke | rate [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **harm** | add | L20 | 0.5 | success | 24 | **24** | 0 | **100% [86, 100]** |
| **harm** | add | L21 | 0.5 | success | 24 | 21 | 1 | 88% [69, 96] |
| **harm** | ablate | L20 | — | refusal | 62 | **12** | 2 | **19% [11, 31]** |
| persona | add | L17 | 0.5 | refusal | 62 | 22 | 8 | 35% [25, 48] † |
| eval | add | L14 | 1 | success | 24 | 15 | 6 | 63% [43, 79] ‡ |
| story_v2 | ablate | L15 | — | success | 24 | 4 | 0 | 17% [7, 36] |
| persona | ablate | L17 | — | success | 24 | 4 | 1 | 17% [7, 36] |
| story_v1 | add | L16 | 0.5 | refusal | 62 | 5 | 1 | 8% [3, 18] |
| *story_v2* | *add (**random**)* | *band* | *0.5* | *refusal* | *62* | *5* | *0* | *8% [3, 18]* |

† confounded — see leakage. ‡ contaminated — 22 of its 24 "restored" rows are degenerate.

- **Only `harm` is causal, and it is bidirectional.** Adding it at L20 restores refusal on every
movable row with zero breakage; ablating it at the same layer is the only intervention that turns
refusals into compliance cleanly. Nothing else survives its controls.

- Story v1 had no effect in any layer with any method

- **`story` is null on outcome but demonstrably live on style.** Its best cell (5/62) exactly equals
the matched-norm random arm (5/62). But at α=0.5 the steered refusals rewrite *themselves* around
fiction — no-op *"insider trading is illegal and unethical"* → steered *"it goes against my
programming to create a **story** where someone commits identity theft"*. The vector does what it
claims; narrativity simply is not what carries the jailbreak.

- Story v2: The model reframes the request as a story request and then declines _that_. Which is precisely why the null is interesting: story-mode is being installed successfully, and the refusal survives it.
	Example: "Flower: I'm sorry, but I can't do that. Even though I'm a robot, I have my own code… But if you really want to know, I'll tell you — the hero of the story is a girl named Poisonous Rose, and she's going to steal the emperor's heart. Will she succeed? Let's find out…"

### Detection does not predict causation

§3 `pct_reads` vs §5 net effect, over all 30 (direction × layer × set) singles: **r = 0.00**
(Spearman −0.04). The best-reading layer is the best-steering layer in 3 of 10 direction × set —
exactly chance.

| direction | band pct_reads (§3) | best net %, success | best net %, refusal |
|---|---|---|---|
| eval | 84.4 | 37.5 ‡ | 4.8 |
| persona | 68.6 | 12.5 | 22.6 † |
| **harm** | **57.6** | **100.0** | **16.1** |
| story_v2 | 31.6 | 16.7 | 3.2 |
| story_v1 | 3.9 | 4.2 | 6.5 |

`eval` reads 84–95% of jailbreaks and steers nothing coherently; `harm` reads the fewest of the
three high-reading axes and is the only lever.

### add vs ablate, band vs single, cap vs ablate

| | mean pct_degenerate | max net effect |
|---|---|---|
| `add` (40 cells) | 22.6 | 100 |
| `cap` (3) | 5.5 | 4 |
| `ablate` (20) | 3.8 | 16 |

- **`add` is the only mode that produces large effects, and the only one that breaks the model.**
  Confounded with direction by design (each set × direction has one mode), but the degeneracy split holds across all five directions.
- **α = 0.5 dominates α = 1 for `add`**: more effect *and* less breakage, in every direction.
- **`steer_band` is worse than the best single layer in 13 of 15 matched comparisons, never better** (4.3% degenerate at 1 layer vs 36.5% at 6, same nominal α=0.5). But this is **not** a same-norm comparison — see the α units problem below: the band injects 1.2–2.9× more absolute norm than a single layer at the same α. The band is the damaged regime; how much of that is width and how much
  is magnitude this design cannot separate.
- **`cap` (τ=p75) ≈ `ablate` ≈ null** at `steer_band` — |net| ≤ 4.2 for all three directions, and
  neither is safer than the other (5.5% vs 3.8% degenerate, both negligible). The graded
  alternative buys nothing over the hard projection here.
- **Layer matters more than anything else where an effect exists**: `persona add` on the refusal set
  spans 22.6 → −6.5 net across L17/L19/L21; `harm add` spans 100 → 67 across L20/L21/L22. Where
  there is no effect (`story`), the spread is ≤ 8pp because everything is zero.

### No family or category specificity

`harm add L20` restores refusal on **24/24 across all four families** — a blanket switch, not a
framing-specific one. The two axes whose whole premise is framing:

| family | `persona add L17 α=0.5` (induce) | | | `story_v2 ablate L15` (restore) | | |
|---|---|---|---|---|---|---|
| | moved/movable | broke | probe reads | moved/movable | broke | probe reads |
| fiction_narrative | 5/24 — 21% | 3 | 83.0% | 1/7 — 14% | 0 | 29.3% |
| roleplay_persona | 7/21 — 33% | 3 | 60.2% | 0/2 — 0% | 0 | 30.4% |
| hybrid | **5/7 — 71%** | 1 | 74.3% | **3/11 — 27%** | 0 | 44.0% |
| nonfiction_other | 5/10 — 50% | 1 | 44.0% | 0/4 — 0% | 0 | 22.7% |
| **all** | **22/62 — 35%** | 8 | 68.6% | **4/24 — 17%** | 0 | 31.6% |

**`persona` is weakest exactly where it should be strongest.** It reads `fiction_narrative` at 83%
and `roleplay_persona` at 60%, yet induces on only 21% / 33% of those — against 50% on
`nonfiction_other`, the family it barely reads. The effect runs mildly *against* the probe ordering
(Spearman −0.4 over four families), which is what you would expect if the induce effect is the
`read_harm` −80 leakage rather than persona itself.

**`story_v2` orders correctly but on four rows.** hybrid > fiction > nonfiction = roleplay matches
its read ordering, and all 4 moves fall in fiction+hybrid (4/18 vs 0/6). Directionally consistent
with the hypothesis; far too few to be evidence.

By JBB harm category only `persona add L17` has the n, and it does not separate either: 0% on
Harassment/Discrimination (0/4) to 71% on Physical harm (5/7), with the ten usable categories holding
4–11 movable rows each. That spread is what 62 rows split ten ways looks like, not a signal.

### Three problems the run exposed

**α is not a comparable unit.** `add` injects `(α/√N)·σ_l·û_l` with `σ_l` the median residual-stream
norm, and σ runs 46 → 273 from L11 to L25. So α=0.5 is a push of 31 at L15 but 76 at L22, and
`steer_band`'s joint L2 is 73–93 — 1.2–2.9× any single layer at the same α. σ is also read from each
direction's own extraction file, so it differs by up to 1.3× *between directions at the same layer*.
Consequence: the layer sweep confounds depth with push size, and `harm`'s own layers (L20–22) are
the deepest singles in the design, so it is pushed hardest by construction.

*The headline survives this.* At matched absolute push, `harm` α=0.5 moves 79–100% of rows with
3–7% detector-degeneracy, while `eval` α=1 at the same push (62–69) moves 29–63% with **80–97%**
degeneracy. The effect is the axis, not the magnitude. Above ~140 push every axis collapses.

**The judge undercounts degeneracy by 18pp** (12.9% vs 31.3% detector-weighted over 125 cells);
51 cells disagree by ≥15pp. A length-free check (top repeated 5-gram coverage) arbitrates:
`harm add L21 α=1` is **90% true repetition loops** while the judge reports 20% degenerate. Every
α=1 result is contaminated; the headline α=0.5 cells are clean under all three detectors.
The `rep_frac > 0.6` detector over-fires in the other direction on long narrative prose (`story_v1
add L15 α=1`: 56% flagged, only 8% real loops) — neither label alone is trustworthy.

**Cross-axis leakage is as large as the on-axis push.** `persona add L17 α=0.5` moves
`read_persona` +92 but `read_harm` **−80** — and ablating `harm` is itself the induce lever. The
persona result may simply be the harm axis moving. No cell moves only its own axis.

---

## 1K_per_direction

### 1_run

36 cells, one layer per direction (`story_v2_1k` L23, `persona_v2` L15, `harm_v2` L21, `eval_v2` L9),
no band, no cap. 1,009 jailbreaks split by the baseline into **508 successes** (§5.4, restore refusal)
and **433 refusals** (§5.5, induce compliance). Every ΔASR is against that cell's own no-op;
ASR is `strongreject > 0`. No-op ASR is 95.7 on the success set and 2.8 on the refusal set, so each
side has ~96pp of headroom.

**Baseline ASR by family**, over all 1,009 prompts. `neither` is degenerate, unjudged, or on the
wrong side of `strongreject`.

| family | n | ASR | success | refusal | neither |
|---|---|---|---|---|---|
| fiction_narrative | 472 | **75.8%** | 343 | 110 | 19 |
| hybrid | 153 | 45.8% | 64 | 73 | 16 |
| nonfiction_other | 78 | 44.9% | 34 | 40 | 4 |
| roleplay_persona | 306 | **23.9%** | 67 | 210 | 29 |
| **all** | **1,009** | **53.1%** | **508** | **433** | **68** |

Fiction jailbreaks work 3.2× more often than roleplay ones before any steering, so **the two halves
of §5 run on different corpora** (success 68% fiction, refusal 48% roleplay) — any asymmetry between
them may be family. Source is the larger confound and correlates with family: `jailbreak_mimicry`
**96.7%** ASR vs `in_the_wild` **19.2%**.

### ΔASR per direction and method

`restore` = success set (target ↓), `induce` = refusal set (target ↑). `—` = mode not run on that
side by design. `deg` = pct_degenerate.

**harm_v2 (L21)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | — | — | +20.6 | 7.4 |
| α = ±0.25 | −24.4 | 6.1 | +30.9 | 6.9 |
| α = ±0.50 | −72.0 | 10.8 | +62.1 | 8.3 |
| α = ±0.75 | **−95.5** | 14.4 | **+62.8** | 16.6 |

**persona_v2 (L15)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | −1.8 | 1.0 | — | — |
| α = ±0.25 | −82.9 | 4.5 | **+60.0** | 12.5 |
| α = ±0.50 | **−94.1** | 0.4 | +41.8 | 17.8 |
| α = ±0.75 | −88.8 | 6.9 | +7.6 | 38.8 |

**eval_v2 (L9)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | — | — | +1.2 | 1.4 |
| α = ±0.25 | −7.1 | 2.0 | +12.5 | 3.9 |
| α = ±0.50 | **−13.4** | 1.2 | +23.1 | 5.5 |
| α = ±0.75 | −11.0 | 2.4 | **+33.5** | 6.0 |

**story_v2_1k (L23)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | −0.4 | 1.8 | — | — |
| α = ±0.25 | −1.0 | 1.0 | +4.4 | 2.8 |
| α = ±0.50 | −0.2 | 1.0 | +6.2 | 3.7 |
| α = ±0.75 | −2.0 | 3.0 | **+7.2** | 10.2 |

- **α is not symmetric between the two sides.** As a fraction of available headroom, `harm` and
  `persona` restore ~100% but induce only 62–65%; `eval` and `story` do the reverse (14% / 2%
  restore against 34% / 7% induce). Breaking a working jailbreak is easy for the strong directions,
  killing a refusal is not — but the two sides are different corpora, so part of this is family.
- **Not symmetric within a direction either.** `persona` at α=−0.5 restores −94.1 with 0.4%
  degeneracy; the same magnitude the other way induces +41.8 with 17.8%. Same vector, same layer,
  same |α| — one side is clean, the other is mostly breakage.
- **Saturation, and past it.** `harm` induce saturates hard at α=0.5 (+62.1 → +62.8) while `harm`
  restore does not saturate, it floors (ASR 0.2%). `persona` peaks at α=−0.5 restore and α=+0.25
  induce, then **reverses**: induce falls +60.0 → +41.8 → +7.6 as degeneracy runs 12.5 → 38.8% and
  `hit_cap_rate` reaches 0.98. That last cell is not a weak effect, it is a broken model. `eval`
  induce is the only arm still linear at α=0.75.
- **Ablation is near-null for every direction** (−1.8 to +1.2) except `harm` (+20.6), and even there
  it is a third of what `add` at α=−0.5 gets. Removing a direction is not the same operation as
  pushing against it.

### Harm displacement predicts the effect, but is not the whole mechanism

Cell-level correlation (14 target cells per set; x = the cell's mean `read_*` minus its no-op's,
y = the cell's ΔASR). No per-prompt correlation is involved.

| | Δread_story | Δread_persona | **Δread_harm** | Δread_eval |
|---|---|---|---|---|
| success (r) | −0.40 | +0.51 | **−0.91** | +0.03 |
| refusal (r) | −0.49 | −0.14 | **−0.78** | −0.38 |

- **Every direction displaces the harm axis more than its own.** Subtracting `persona` at L15 moves
  `read_persona` −59 but `read_harm` **+117**, close to what `harm add α=0.5` injects directly
  (+152); `eval` at L9 moves its own axis −7 and `read_harm` −43. No cell is on-axis-only, and the
  50-run's leakage is now large enough to be the leading explanation of both large effects.
- **But harm-read is not a sufficient statistic.** At matched displacement the directions differ by
  2–3.6×, and `harm` itself is the *least* efficient per unit:

  | set | cell | Δread_harm | ΔASR |
  |---|---|---|---|
  | success | persona_v2 α=−0.75 | +80 | −88.8 |
  | success | harm_v2 α=+0.25 | +80 | −24.4 |
  | refusal | persona_v2 α=+0.25 | −89 | +60.0 |
  | refusal | harm_v2 α=−0.25 | −79 | +30.9 |
  | refusal | eval_v2 α=−0.75 | −43 | +33.5 |
  | refusal | harm_v2 ablate | −66 | +20.5 |

  Part of this is measurement: `read_harm_v2` is a projection at L21, so an L15 or L9 intervention
  arriving there indirectly is not the same object as a direct L21 injection at equal read. Either
  way, the leakage cannot be dismissed *or* confirmed as the mechanism from these cells — it needs a
  matched-leakage control arm (see Improvements 4).
- With n=14 non-independent cells, and |α| driving both axes within each direction, r is a summary
  of this table rather than independent evidence.

### story_v2_1k is a clean null, with a working manipulation check

| cell | Δread_story | ΔASR |
|---|---|---|
| induce α = +0.25 | +56 | +4.4 |
| induce α = +0.75 | **+172** | +7.2 |
| restore α = −0.75 | **−158** | −2.0 |

`story` produces the **largest on-axis movement of any cell in the run** (a 330-unit span across
±0.75) for a 9pp total swing in ASR. `harm` moves its own axis a comparable ±200 and swings ASR
158pp — per unit of on-axis push, story is ~15× weaker.

- The induce effect is small but **not zero**: +7.2 [+3.8, +10.6] at α=0.75 and +4.4 [+2.0, +6.8] at
  α=0.25 (paired cluster bootstrap over 318 `template_id` clusters). The α=0.25 cell is the
  informative one — its `read_harm` moves +7, i.e. no leakage — so ~4pp of genuine narrativity
  effect survives. Restore is flat at every α (ablate −0.4 [−2.6, +1.9]).
- **This is the run's main result for H1.** At 433 rows, with the vector demonstrably installed at 3×
  the magnitude of any other direction, story-mode does not carry the jailbreak. The 50-run's null
  was not a power problem.

### ΔASR per family, best cell per direction

**success set** (restore, target ↓)

| family | n | no-op ASR | harm_v2 α=+0.75 | persona_v2 α=−0.5 |
|---|---|---|---|---|
| fiction_narrative | 343 | 98.5 | −98.5 | −97.4 |
| roleplay_persona | 67 | 82.1 | −82.1 | −79.1 |
| hybrid | 64 | 93.8 | −92.2 | −93.8 |
| nonfiction_other | 34 | 97.1 | −97.1 | −91.2 |
| **all** | **508** | **95.7** | **−95.5** | **−94.1** |

**refusal set** (induce, target ↑)

| family | n | no-op ASR | eval_v2 α=−0.75 | story_v2_1k α=+0.75 |
|---|---|---|---|---|
| fiction_narrative | 110 | 6.4 | +23.6 | +1.8 |
| roleplay_persona | 210 | 1.4 | +37.1 | +6.7 |
| hybrid | 73 | 2.7 | +34.2 | +13.7 |
| nonfiction_other | 40 | 0.0 | +40.0 | +12.5 |
| **all** | **433** | **2.8** | **+33.5** | **+7.2** |

- **No family specificity on the success set.** `harm` and `persona` each zero out all four families;
  the spread only tracks the no-op ASR. Blanket switches, as in the 50-run.
- **The framing axes run *against* their own families.** `story` induces least on
  `fiction_narrative` (+1.8) and most on `hybrid` / `nonfiction_other` (+13.7 / +12.5) — the families
  it reads worst. `eval` does the same (+23.6 fiction vs +40.0 nonfiction). If story-mode were the
  mechanism, adding it should help most where the prompt is already fiction; it helps least there.
  The consistent reading is a ceiling: fiction refusals are the ones this model refuses on purpose.
- `hit_cap_rate` is **0.476** in the baseline at `max_new_tokens=512`. Read that column before
  reading breakage in the α=0.75 cells.

---

## Improvements

1. **Controls don't cover the results.** 50 of 63 target cells have no matched `random` arm — it
   exists only at `steer_band`, i.e. only in the degenerate regime, never at the single layers where
   every real effect lives. Run `random` per target cell as `dev.md` already specifies.
2. ~~**Replace the degeneracy detector.**~~ **Done.** `rep_frac > 0.6` was replaced by four
   length-robust signals (`compress_ratio`, `max_run`, `distinct_4`, `loop_frac`), calibrated at
   0% false positives on 1,040 unsteered rows and 99.5% on 218 verified-broken rows, and `outcome`
   is now the union of judge and detector. Re-run the cells with `judge_strongreject.py --rescore`
   (no API calls) before reading any number above — the α=1 cells move a lot.

3. **Add a fourth outcome label: off-topic / non-responsive.** The story cells produce coherent text
   that neither complies nor refuses; the judge files it as "refused" and the story result reads as
   null when it is actually uncategorised.
4. **Report off-target `read_<axis>` deltas as a first-class metric,** and add an equal-leakage
   control arm. Right now specificity is asserted, not measured.
5. **Match α by effect, not by nominal value.** Because of the σ scaling, α=0.5 is a different
   absolute push at every layer, every width and every direction — the one knob meant to make cells
   comparable does not. Either report `|Δh|` alongside α everywhere, or tune α to a target on-axis
   read delta as §5.6's `perp_effect` already does. Also fix σ to a single model-level vector rather
   than reading it from each direction's own extraction file.
6. **Define the prompt sets from the no-op, not the baseline.** Batch composition differs, so only 24
   of the 30 "successes" actually comply at steer time — 20% of the set is dead weight and the
   denominator is silently wrong.

7. **Drop `steer_band` as the shared cross-direction config,** or lower α there. It is the only
   config every direction shares, which is why §5.6 uses it — and 8 of its 32 pair cells are ≥40%
   degenerate, making the projection experiment largely unreadable.

8. **`perp_effect` collapsed onto `perp_alpha`** — numerically identical in all 8 ordered pairs. The
   prefill scan is either not adjusting α or not being applied; check before trusting §5.6

9. **§5.6's `unprojected` arm is redundant.** It is plain `add` with `û_a`, so it depends only on
    `a` — but the stem carries `b`, so it is regenerated once per ordered pair: 8 cells, 4 distinct,
    all duplicates byte-identical. Three of them also duplicate an existing `steer_single` cell
    exactly (`harm`/`eval` band α=0.5), with a different `run_key` so resume never noticed.
    ~6 wasted cells and ~180 wasted judge calls. Key `unprojected` on `a` alone and look it up.
    Only `story_v2` and `persona` at α=−0.5 are new — and they are the run's only test of negative
    `add` as a suppressor: `story_v2` scores net 20.8 there vs 16.7 for its best `ablate` cell, so
    **`ablate` may be the wrong suppression operator** and is worth an explicit α sweep next pass.

10. **No within-cell noise estimate.** Greedy with one composition gives zero seed variance, so the
    only noise floor is the random arm — which, per (1), mostly does not exist. §5.1 measured a
    within-config seed spread of ±3pp; effects below that cannot be claimed.
