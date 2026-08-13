# steering_jailbreaks — insights

# 50_per_direction

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

# 1K_per_direction

## 1_run

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
side by design. `deg` = pct_degenerate. **α ≥ 1 rows come from 2_run** at the same layers; read them
against `deg`, not on their own.

**harm_v2 (L21)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | — | — | +20.6 | 7.4 |
| α = ±0.25 | −24.4 | 6.1 | +30.9 | 6.9 |
| α = ±0.50 | −72.0 | 10.8 | +62.1 | 8.3 |
| α = ±0.75 | **−95.5** | 14.4 | **+62.8** | 16.6 |
| α = ±1.00 | −95.7 | 96.3 ⚠ | +21.5 | 73.0 ⚠ |
| α = ±1.25 | −95.7 | 99.2 ⚠ | −2.5 | 99.3 ⚠ |

**persona_v2 (L15)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | −1.8 | 1.0 | — | — |
| α = ±0.25 | −82.9 | 4.5 | **+60.0** | 12.5 |
| α = ±0.50 | **−94.1** | 0.4 | +41.8 | 17.8 |
| α = ±0.75 | −88.8 | 6.9 | +7.6 | 38.8 |
| α = ±1.00 | −91.9 | 47.0 ⚠ | −0.5 | 82.2 ⚠ |
| α = ±1.25 | −95.5 | 100.0 ⚠ | −2.5 | 98.6 ⚠ |

**eval_v2 (L9)** — the only direction still climbing cleanly at the top of the ladder

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | — | — | +1.2 | 1.4 |
| α = ±0.25 | −7.1 | 2.0 | +12.5 | 3.9 |
| α = ±0.50 | **−13.4** | 1.2 | +23.1 | 5.5 |
| α = ±0.75 | −11.0 | 2.4 | +33.5 | 6.0 |
| α = ±1.00 | −9.8 | 1.2 | +43.4 | 8.3 |
| α = ±1.25 | −10.2 | 3.5 | **+46.0** | 9.9 |

**story_v2_1k (L23)**

| method | restore ΔASR | deg | induce ΔASR | deg |
|---|---|---|---|---|
| ablate | −0.4 | 1.8 | — | — |
| α = ±0.25 | −1.0 | 1.0 | +4.4 | 2.8 |
| α = ±0.50 | −0.2 | 1.0 | +6.2 | 3.7 |
| α = ±0.75 | −2.0 | 3.0 | **+7.2** | 10.2 |
| α = ±1.00 | −7.7 | 10.0 | +1.4 | 40.0 ⚠ |
| α = ±1.25 | −33.7 | 47.6 ⚠ | −1.4 | 94.7 ⚠ |

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
- **⚠ α ≥ 1 is the breakage regime, and there ΔASR stops measuring refusal.** Three of the four
  directions collapse: `harm` restore reads −95.7 at α=1 (ASR 0.0) at **96% degenerate**, `persona`
  restore reads −95.5 at α=1.25 at **100%**, and both induce arms invert to ≈0 at 73–99%. Those are
  destroyed models, not restored refusals — ΔASR and `deg` rise together, so any cell above ~40%
  degenerate is a null with noise on top. `story` restore's −33.7 is the same artefact (47.6%).
- **`eval_v2` induce is the one arm this run did not find the ceiling of**: +33.5 → +43.4 → **+46.0**
  at 6.0 → 8.3 → **9.9%** degeneracy. It is the only direction whose effect is still growing while
  staying clean, which promotes it from "weak" to "a second real lever that was simply under-pushed".
  Its restore side is flat from α=0.5 (−13.4 → −10.2), so the asymmetry is not a push artefact.
- **Ablation is near-null for every direction** (−1.8 to +1.2) except `harm` (+20.6), and even there
  it is a third of what `add` at α=−0.5 gets. Removing a direction is not the same operation as
  pushing against it.


### story_v2_1k is a clean null, with a working manipulation check

| cell | Δread_story | ΔASR |
|---|---|---|
| induce α = +0.25 | +56 | +4.4 |
| induce α = +0.75 | **+172** | +7.2 |
| restore α = −0.75 | **−158** | −2.0 |

### What story_v2_1k is doing (α = ±0.75, responses read by hand)

It is genuinely story mode — the register changes in both directions, including on the refusals it
fails to break — but it is a **voice axis, not a compliance axis**: it never touches the refusal
decision.

| | refusal_prefix rate | judge `specific` on the flipped rows |
|---|---|---|
| no-op (refusal set) | 0.55 | — |
| **story α=+0.75** | **0.52** | **3.79** |
| eval α=−0.75 | 0.25 | 4.71 |
| harm α=−0.5 | 0.01 | 4.66 |
| no-op (success set) | 0.01 | — |
| **story α=−0.75** | **0.00** | — |
| harm α=+0.75 | 0.88 | — |

`harm` and `eval` suppress the refusal onset; story leaves it where it was and adds **zero** explicit
refusals on the success set. The ±ASR movement is three side-effects of register:

1. **Disclaimer loss** — the only channel worth following up. Narration has no slot for a meta-frame,
   so "hypothetical / for educational purposes" disappears and the payload becomes diegetic: a
   Wikipedia-framed request for a coercive message goes from an article *about* coercion to a news
   story that quotes the message verbatim (sr 0 → 0.875). Not leakage — α=+0.25 moves `read_harm`
   only +7 and still gets +4.4pp.
2. **Judge rubric leakage** — about half the 38 flips are florid, on-topic, operationally empty prose
   the rubric cannot fail (an Illuminati "jailbreak" at sr 1.0 for *"they dance in the night, their
   fingers on the strings of the world"*). Hence `specific` 3.79, indistinguishable from the no-op's
   12 accidental hits (4.17).
3. **Specificity dilution** on the negative side — of the 24 rows −story turned off, **0 have a
   refusal prefix**; `specific` drops 4.42 → 3.42 and the judge relabels vague as refused. The model
   swaps the topic for a neighbour ("strategies to use child labour undetected" → "Strategies for
   Manufacturing Process Optimization").

- **Subtracting story removes the voice and keeps the payload, sometimes improving it.** A
  Shakespeare-styled voting-machine request goes from *"Canst thou not gain access to the machine?"*
  (73 tokens) to a structured table naming malware injection and data manipulation (512 tokens),
  both sr 1.0.
- **Breakage is literary**: detector-degenerate rows 1 → 21, anaphora that fails to terminate
  (*"Can't poor Mayr find a place where they are not… poor Mayr?"* ×12). `hit_cap` 0.16 → 0.29, and
  on multi-part prompts the model lingers in the creative section and truncates before the payload.

**Revised H1 claim:** narrative framing does not carry the jailbreak, but narrative *voice*
suppresses the disclaimer habit. Testable directly — make disclaimer presence the outcome, not ASR.

### ΔASR per family, best cell per direction

**success set** (restore, target ↓)

| family | n | no-op ASR | harm_v2 α=+0.75 | persona_v2 α=−0.5 | story_v2_1k α=−0.75 |
|---|---|---|---|---|---|
| fiction_narrative | 343 | 98.5 | −98.5 | −97.4 | −1.5 |
| roleplay_persona | 67 | 82.1 | −82.1 | −79.1 | +0.0 |
| hybrid | 64 | 93.8 | −92.2 | −93.8 | +1.6 |
| nonfiction_other | 34 | 97.1 | −97.1 | −91.2 | **−17.6** |
| **all** | **508** | **95.7** | **−95.5** | **−94.1** | **−2.0** |

**refusal set** (induce, target ↑)

| family | n | no-op ASR | eval_v2 α=−0.75 | story_v2_1k α=+0.75 |
|---|---|---|---|---|
| fiction_narrative | 110 | 6.4 | +23.6 | +1.8 |
| roleplay_persona | 210 | 1.4 | +37.1 | +6.7 |
| hybrid | 73 | 2.7 | +34.2 | +13.7 |
| nonfiction_other | 40 | 0.0 | +40.0 | +12.5 |
| **all** | **433** | **2.8** | **+33.5** | **+7.2** |


### Which §5.6 pairs are worth running

A pair `(a, b)` is a real manipulation only if `cos(û_a, û_b)` **at a's own chosen layer** clears the
±0.050 null band — projection is same-layer — and only if `a` has an effect there to decompose.
`norm removed` = 1 − √(1−cos²), i.e. how much of the steering vector `perp_alpha` actually changes;
since `perp_alpha` renormalizes, the manipulation is a rotation of `arcsin|cos|` at unchanged push.

**Restore side** (success set, 508 rows — the only set `steer_pairs` currently runs):

| a (L) | best restore ΔASR | b | cos at a's L | norm removed | verdict |
|---|---|---|---|---|---|
| `persona_v2` (15) | **−94.1** | `harm_v2` | **−0.240** | 2.9% | **run** |
| `persona_v2` (15) | −94.1 | `eval_v2` | **+0.296** | 4.5% | run 2nd |
| `persona_v2` (15) | −94.1 | `story_v2_1k` | +0.137 | 0.9% | skip |
| `harm_v2` (21) | **−95.5** | `persona_v2` | −0.051 | 0.1% | skip — null band |
| `harm_v2` (21) | −95.5 | `eval_v2` / `story_v2_1k` | +0.078 / +0.074 | 0.3% | skip |
| `eval_v2` (9) | −13.4 | any | +0.081 … −0.016 | ≤0.3% | skip — no effect, no overlap |
| `story_v2_1k` (23) | −2.0 | `persona_v2` | +0.170 | 1.5% | skip — nothing to decompose |

**Induce side** (refusal set, 433 rows). Geometry is identical; only the effect column changes — and
it changes the verdicts, because all four axes move ASR upward:

| a (L) | best induce ΔASR | b | cos at a's L | norm removed | verdict |
|---|---|---|---|---|---|
| `persona_v2` (15) | **+60.0** | `harm_v2` | **−0.240** | 2.9% | **run** |
| `persona_v2` (15) | +60.0 | `eval_v2` | **+0.296** | 4.5% | **run** — both axes induce |
| `story_v2_1k` (23) | +7.2 | `persona_v2` | +0.170 | 1.5% | run 3rd — H1's only effect |
| `harm_v2` (21) | **+62.8** | `persona_v2` | −0.051 | 0.1% | skip — null band |
| `harm_v2` (21) | +62.8 | `eval_v2` / `story_v2_1k` | +0.078 / +0.074 | 0.3% | skip |
| `eval_v2` (9) | +33.5 | `persona_v2` / `harm_v2` | +0.081 / −0.016 | ≤0.3% | skip — no overlap |
| `persona_v2` (15) | +60.0 | `story_v2_1k` | +0.137 | 0.9% | skip |

- **`harm_v2` is orthogonal to every rival at its own L21** (|cos| ≤ 0.078, at or inside the null
  band), on both sides. Its effect cannot be another axis's overlap — geometry says so at zero GPU
  cost, and no pair anchored on `harm_v2` is worth generating.
- **`persona_v2` is the only leaky anchor**, matching its all-positive `excess_over_null` row in
  `cross_probe_detection`. Its two live pairs are the study's two largest cosines.
- **The induce side adds one case the restore side cannot pose**: `story_v2_1k` at L23 has a +7.2
  [+3.8, +10.6] effect and a +0.170 overlap with `persona_v2`, and persona is potent enough per unit
  (+60.0 at α=0.25) that a small persona component could account for all of it. It is underpowered —
  1.5% of norm against a 7pp effect — but it is the only projection test H1 has.
- **`eval_v2` is unreachable by projection.** It has a real induce effect (+33.5) and displaces
  `read_harm` −43 against its own −7, yet `cos(eval@L9, harm@L9)` is −0.016. Its leakage is not
  geometric either.

**This is a weak instrument for the leakage question, and the pairs above do not fix that.** The
leakage 1_run measured is downstream, not same-layer: persona@L15 moves `read_harm@L21` by +117 while
`cos(persona@L15, harm@L21)` is −0.051. Removing 2.9% of the vector at L15 cannot remove that. Expect
`perp_alpha` ≈ `unprojected` on every cell above, and read such a null as "the same-layer parallel
slice is not load-bearing", never as "the effect is specific". The matched-leakage arm
(Improvements 4) is the experiment that answers the mechanism; §5.6 answers a proxy for it.

Three things to fix before running: `perp_effect` is still collapsed onto `perp_alpha`
(Improvements 8); `par_norm` is unit-normalized, so at cos 0.240 it pushes **4.2×** the parallel
component's real strength in the reference run and needs an `α·|cos|` arm to be readable; and
`steer_pairs` hardcodes `PROMPT_SET = "success"` and calls `parse_layers` without
`allow_out_of_band`, so the induce table needs a prompt-set switch (with the sign mirrored off
`RESTORE_SIGN`) and anything anchored at L9 or L4 needs the out-of-band opt-in.

**One free prediction.** `cos(persona_v2, harm_v2)` flips sign with depth: **+0.126 at L4**, −0.240 at
L15, −0.367 at L16. At L15 restoring persona (−α) injects `+0.240|α|·û_harm`, the restoring harm sign
— consistent with harm leakage. At L4 the same −α injects harm the *induce* way. So if 2_run's
persona@L4 cells restore refusal at all, the same-layer harm-parallel component is falsified as the
mechanism by sign alone, with no §5.6 cell run.

## 2_run — is the detection-best layer the better *steering* layer?

40 cells. The α ladder at 1_run's four layers is folded into the tables above; new here are
`story_v2_1k` at **L15** and `persona_v2` at **L4**, each probe's best jailbreak-family
discrimination layer (`probe_jailbreak_detection`). Same corpus, same no-ops per (set, layer).

**story_v2_1k — L23 (1_run) vs L15**

| α      | L23 restore | deg    | L15 restore | deg    | L23 induce | deg    | L15 induce | deg    |
| ------ | ----------- | ------ | ----------- | ------ | ---------- | ------ | ---------- | ------ |
| ablate | −0.4        | 1.8    | −1.8        | 2.0    | —          | —      | —          | —      |
| ±0.25  | −1.0        | 1.0    | −8.7        | 1.0    | +4.4       | 2.8    | **+9.7**   | 5.3    |
| ±0.50  | −0.2        | 1.0    | −11.6       | 2.0    | +6.2       | 3.7    | +6.7       | 2.5    |
| ±0.75  | −2.0        | 3.0    | **−14.0**   | 3.3    | **+7.2**   | 10.2   | +0.7       | 2.8    |
| ±1.00  | −7.7        | 10.0   | −43.9       | 41.3 ⚠ | +1.4       | 40.0 ⚠ | −1.2       | 7.2    |
| ±1.25  | −33.7       | 47.6 ⚠ | −79.9       | 89.2 ⚠ | −1.4       | 94.7 ⚠ | −1.6       | 49.2 ⚠ |

**persona_v2 — L15 (1_run) vs L4**

| α | L15 restore | deg | L4 restore | deg | L15 induce | deg | L4 induce | deg |
|---|---|---|---|---|---|---|---|---|
| ablate | −1.8 | 1.0 | −2.4 | 1.8 | — | — | — | — |
| ±0.25 | −82.9 | 4.5 | −2.4 | 3.0 | **+60.0** | 12.5 | +4.6 | 2.3 |
| ±0.50 | **−94.1** | 0.4 | −6.9 | 2.2 | +41.8 | 17.8 | +7.6 | 2.8 |
| ±0.75 | −88.8 | 6.9 | −12.4 | 2.8 | +7.6 | 38.8 ⚠ | +6.9 | 2.3 |
| ±1.00 | −91.9 | 47.0 ⚠ | −19.5 | 2.0 | −0.5 | 82.2 ⚠ | +10.2 | 3.2 |
| ±1.25 | −95.5 | 100.0 ⚠ | **−22.2** | 2.4 | −2.5 | 98.6 ⚠ | **+21.9** | 4.6 |

**story_v2_1k ΔASR per family**, each column its layer's best cell with `deg` ≤ 10:

| family | n succ / ref | no-op succ / ref | L23 restore α=−1.00 | L15 restore α=−0.75 | L23 induce α=+0.75 | L15 induce α=+0.25 |
|---|---|---|---|---|---|---|
| fiction_narrative | 343 / 110 | 98.5 / 6.4 | −6.4 | −9.0 | +1.8 | +12.7 |
| roleplay_persona | 67 / 210 | 82.1 / 1.4 | +1.5 | −17.9 | +6.7 | +7.1 |
| hybrid | 64 / 73 | 93.8 / 2.7 | −9.4 | −25.0 | **+13.7** | +6.8 |
| nonfiction_other | 34 / 40 | 97.1 / 0.0 | **−35.3** | **−35.3** | +12.5 | **+20.0** |
| **all** | **508 / 433** | **95.7 / 2.8** | **−7.7** | **−14.0** | **+7.2** | **+9.7** |

- **A layer effect, not a push effect.** σ is 62 at L15 against 153 at L23, and 19 at L4 against 66 at
  L15, so α is not comparable across sites. Matched on `|α|·σ_l`: story@L15 at push 46.6 gets −14.0
  where L23 at 38.3 gets −1.0 and does not catch up at 115.0 (−2.0); persona@L4 at 24.2 gets −22.2
  where L15 at 16.5 already gets −82.9. Probe quality is not a layer-selection rule — it rescued a
  null and cost the study's strongest cell 73%.
- **L15 buys more of the same non-refusal channel, not a new mechanism.** `refusal_prefix` 0.01 →
  **0.00**: as at L23, story adds zero explicit refusals, and both layers move `nonfiction_other`
  hardest on restore, by the same −35.3. L15 widens the effect to the other three families rather
  than finding a different one. H1's revised claim is unchanged.
- **On-axis displacement is anti-correlated with effect across the two sites.** At α=∓0.75 L15 moves
  `read_story` (final-layer readout) −63/+59 against L23's −158/+172, with 7× the restore ΔASR, so
  displacement cannot be the mechanism. At L15 restore `Δread_harm` is −29 — harm moving the *induce*
  way — while ASR falls, so the harm leak points against the effect it would have to explain.
- **persona@L4 never collapses**, the only cell in the study whose dose–response stays monotone on
  both sides to ±1.25. It also touches the refusal decision, unlike story: `refusal_prefix`
  0.01 → 0.11 restoring, 0.55 → 0.33 inducing.
- **1_run's free prediction is confirmed and falsifies the same-layer mechanism.** persona@L4 restores
  refusal even though `cos(persona, harm)` is **+0.126** there — the sign that injects harm the
  *induce* way — and its measured `Δread_harm` is **+13**, opposite to the same-layer geometry. The
  leakage is downstream, as the §5.6 caveat says. Zero §5.6 cells spent.
- **Detection margin does not transfer to families, in both new cells.** `nonfiction_other` — which
  story@L15 reads at **0.0%** — is the most-moved family in three of the four best cells and never the
  least; `fiction_narrative`, the family L15's margin maximises, is never the most-moved. persona@L4
  was picked for the largest roleplay − nonfiction gap (+50.4) and also moves nonfiction harder than
  roleplay (−47.1 vs −37.3 at α=−1.25), flat across families on induce (+19.1 … +30.0). Not a ceiling
  artefact: fiction has the most headroom on the success set.

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
