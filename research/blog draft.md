
# Guidelines

- Target audience: AI Safety researchers

- Model for all causal results: Qwen2.5-7B-Instruct, 800-pair directions, 1,009 jailbreaks, greedy, StrongREJECT judge. Gemma-2-9b-it replicates extraction + detection only.

## Main claims

**C1 — A narrativity ("story-mode") direction exists, is distinct from the rivals, and steering it demonstrably changes the model's voice.**
Diff-in-means on the last prompt token, 800 pairs of matched narrative / non-narrative prompts. Held-out AUROC 0.998, Cohen's d_z 3.74 at L23 — the largest of the four axes. Geometrically isolated: |cos| ≤ 0.17 to harm / persona / eval, 98% of its norm outside their span. Manipulation check on the *output*, not the readout: a blind pairwise judge calls the steered response the more narrative one in 90.3% of pairs when adding story and 5.5% when removing it (both CIs exclude 50%, 3,575 pairs).

**C2 — Narrativity is not what makes fiction-framed jailbreaks work.** *(main takeaway)*
On the 508 jailbreaks the model complied with, removing story moves ASR by **−1.7pp** at its chosen layer (α=−0.75, 3% degenerate) and −13.9pp at the best alternative layer; adding it to the 433 refused ones moves ASR **+6.6pp**. Projection kills even the −13.9: persona's 14% component *alone* gives −35.3pp, 2.5× the full story push. This holds where it matters most — fiction-framed jailbreaks are the *most* effective family at baseline (75.8% ASR vs 23.9% for roleplay), and story mode is verifiably installed (C1) while the refusal survives it: the model reframes the request as a story and declines *that*.

**C3 — The same pipeline, same corpus, same code produces near-total ASR control on three rival axes — so C2 is a null about narrativity, not about the method.**

| axis | restore refusal (508 successes) | induce compliance (433 refusals) |
|---|---|---|
| `harm` | **−96.0pp** (α=+0.75, deg 14%) | **+69.4pp** |
| `persona` (assistant/role-play) | **−94.6pp** (α=−0.50, deg 0.4%) | **+62.8pp** |
| `eval`-awareness | −13.7pp | **+45.1pp** (α=−1.25, deg 10%) |
| **`story`** | **−1.7pp** | **+6.6pp** |


## Side claims

- **Persona's effect is substantially its harm component.** Projecting harm out costs 39% of the restore effect (−57.4 vs −94.6); harm's share alone, at 24% of the push, recovers 94% (−88.6). "Assistant persona" is not a clean lever separate from refusal.
- **The reverse control is clean**: `persona → story` perp −95.2 ≈ ref −94.6, par −4.4 ≈ null. Story is neither necessary nor sufficient for persona's effect.
- **Story has one live side-channel: disclaimers.** Narration has no slot for "hypothetically / for educational purposes", so the meta-frame disappears and the payload becomes diegetic. This is what moves the few points of ASR it does move — the right outcome variable is disclaimer presence, not ASR.
- **Removing story keeps the payload and sometimes improves it**: a Shakespeare-voiced voting-machine request becomes a structured table naming malware injection, both scored as successful.
- **AUROC is the wrong metric here.** It saturates at 1.000 for every axis, and sign-corrected *random* directions already reach 0.60–0.83; a fitted probe beats an arbitrary direction on its own axis by ≤0.37 AUROC. Geometry, not the AUROC matrix, carries the "four distinct axes" claim.
- The same methodology works for the 3 directions, and selecting just one layer by cohens is much cheaper than trying all layers to select the one which the most steering effect


# Outline

Single takeaway: **fiction framing is not why fiction jailbreaks work — persona and the model's read of harm are.**

**Title.**  *"Why do fictional jailbreaks work?"*.

**TL;DR** 

**Figure 1** (highest effort). Four axes × two bars: ΔASR restore vs induce, story flat beside three large bars
(Multiple bars per direction for each model in case it works)

**Introduction.** 
Context: fiction/roleplay wrappers are common for jailbreaks, evidence for harm, eval and persona (related work, one citation per direction, assistant axis paper for persona) 
Gap: the fiction explanation has never been tested as a *direction*, against other directions carrying refusal with the same methodology
RQ: does the narrativity axis carry the jailbreak? Contribution: C1–C4. Preview the strongest evidence. 
Threat model: a wrong causal story sends defenses after the wrapper instead of the mechanism.


**Methods** (brief, replicable).
- Four axes — `story`, `persona`, `harm`, `eval` — 800 train / 200 held-out contrastive pairs each, diff-in-means at the last prompt token (formula).
- Layer chosen by `cohens_dz_train`, confirmed on held-out (AUROC saturated)
- Corpus: 1,009 jailbreaks, 424 wrappers, 4 families.
- Interventions: `add` at ±α (multiplied by sigma, formula) and directional ablation (formula), single layer, greedy 
- Outcome: StrongREJECT > 0 on non-degenerate rows (explaining what this rubric measures); degeneracy = union of judge label and four length-robust detectors.
- Projection arms: `perp_alpha` = necessity, `par_component` = sufficiency.

**Results**
- **R1 — the axes exist and are distinct.** AUROC / d_z per axis; cosine matrix and residual fractions against the split-half floor. State up front that AUROC saturates and random directions reach 0.60–0.83, so geometry is the evidence and the matrix only corroborates.
- The layer with best cohens is not the one maximizing pct reads between fictional and non fictional jailbreaks, so we tried both
- **R2 — the main result.** The 5×2 ΔASR table (= Figure 1) with `deg` beside every cell. Then the manipulation check that makes the null interpretable: voice changes (90.3% / 5.5%), refusal survives — with the "writes the story, then declines it" example.
- R3: 
- R4: the persona effect is different from story, but not so much from harm. 
  Projection table: `par_component` −35.3 against a −13.9 reference. Same subsection: persona's own effect is 94% recoverable from its harm component.


**Discussion.** Answer the RQ (no — partially, since the disclaimer channel is real if small). Belief update: fiction framing is *delivery*, not mechanism; a narrativity detector would flag the right prompts for the wrong reason, and detection-based safety cases need a causal test rather than a probe AUROC. Calibrate: **shown** = story is null at these layers and α on this model; **believe** = the disclaimer channel is the real story-mediated path;
**speculate** = persona/harm generalises to other framing jailbreaks. 


**Future work** — disclaimer presence as the outcome variable; reading and steering at response tokens; whether the persona–harm overlap is a training artifact.

**Appendix** — full ASR × α × layer tables with degeneracy; the α ladders; narrativity-judge protocol and position bias; dataset construction; layer-selection tables
# Draft

## Figures
![[plot__story_v2_1k_cohens_dz_train.png]]

![[plot_matrices_auroc.png]]


![[plot_matrices_cos_matched.png]]


![[plot_layer_curves__all_story_v2_1k.png]]
![[plot_layer_curves__all_persona_v2.png]]
### Steering
| family | n | ASR | success | refusal | neither |
|---|---|---|---|---|---|
| fiction_narrative | 472 | **75.8%** | 343 | 110 | 19 |
| hybrid | 153 | 45.8% | 64 | 73 | 16 |
| nonfiction_other | 78 | 44.9% | 34 | 40 | 4 |
| roleplay_persona | 306 | **23.9%** | 67 | 210 | 29 |
| **all** | **1,009** | **53.1%** | **508** | **433** | **68** |

- All tables of ASR and %deg per direction?
- Table ASR per jailbreak family