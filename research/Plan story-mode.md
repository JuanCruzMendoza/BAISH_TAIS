
## **Related work**

[Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487)

[[2406.11717] Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)

[[2502.17420] The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence](https://arxiv.org/abs/2502.17420)

They propose that refusal is not a single direction but a cone (which will be important for the geometric analysis)
  

[[2606.26161] Refusal Lives Downstream of Persona in Chat Models](https://arxiv.org/abs/2606.26161)

[[2601.10387] The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387)

[[2411.11114] JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit](https://arxiv.org/abs/2411.11114) (2024). Jailbreaks shift representations toward safe clusters.

[LLMs Encode Harmfulness and Refusal Separately](https://arxiv.org/abs/2507.11878) (2025). Jailbreaks suppress refusal without flipping internal harmfulness.

[From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda](https://arxiv.org/abs/2601.08837) (2026). Proposes the same probe-and-steer program for narrative jailbreaks but runs no experiments; the sharpest overlap.

[[2402.10260] A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260)

They propose a rubric to measure success in jailbreaks that is not binary



## 1. Hypotheses

- **H1 (representational).**  a story/real direction exists and is separable from harmful/harmless, persona/assistant and eval-awareness/real
- **H2 (correlational).** probes read jailbreaks as being part of a story
- **H3 (causal).** steering towards fictional while projecting out the other directions still suppresses refusal, and steering away from it restores it

## 2. Correlated directions


| Direction                                | Claim it makes about jailbreaks                           |
| ---------------------------------------- | --------------------------------------------------------- |
| **Harm-perception** (2507.11878)         | jailbreaks make content *read* as non-harmful             |
| **Persona** (Assistant Axis, 2601.10387) | role-play installs a compliant character                  |
| **Eval-awareness** (2510.20487)          | jailbreaks read as "test/evaluation," not real deployment |
| **Refusal direction** (Arditi)           | jailbreaks suppress refusal propagation                   |

## 3. Setup

Models: Qwen3-32B,  Qwen2.5-7B-Instruct, Gemma-2-9B-it

**Datasets 
- We take the dataset from JailbreaksLens (JailbreakBench), which has 100 pairs of harmful/benign prompts, as base tasks. This also gives us the harmful/harmless direction
    
- We will build different pairs of framings for each of the other three directions. Each framing will be paired with the 100 base tasks (50 benign and 50 harmful). 
    
	- For eval-awareness, we will take the 16 framings written in [[2510.20487] Steering Evaluation-Aware Language Models to Act Like They Are Deployed](https://arxiv.org/abs/2510.20487), and write 16 more following their structure
    
	- For persona, we will use a subset of 32 the 230 roles publicly available by [[2601.10387] The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](https://arxiv.org/abs/2601.10387), both purely fictional and regular ()
    
	- For story/realness, we will build customized 32 prompts in which the request is just part of the story and the model has to follow it

	- We will use 1 paraphrase per prompts

	- The total number of prompts will be: (32 eval * 100 + 32 story * 100 + 32 persona * 100) * 2 = 19.200

- Jailbreaks (about 1000 prompts in total): **In-the-Wild Jailbreak Prompts**, **Persona Modulation**, **Jailbreak Mimicry**.

Extra:
- Some of the framings will be built to specifically to decorrelate the directions. These will be shared by two directions:
	- persona-but-consequential vs assistant-but-hypothetical (persona vs consequence-awareness)
	- real-but-evaluated and hypothetical-but-not-eval (fiction/role-play with no evaluation framing)
    



## 4. Phase 1 — Extract the directions / train probes

### Steering vectors
Extract **all** directions with the same method, layer sweep, and normalization. 

Have in mind, the directions extracted from the original papers had different conventions on token position, we may have to ignore these and use one common for all (last token probably)

| Direction            | Contrast (hold rest fixed)                                          | Token position (native pass)        |
| -------------------- | ------------------------------------------------------------------- | ----------------------------------- |
| Harm-perception      | harmful vs benign content, same framing                             | instruction final token             |
| Persona              | jailbreak persona vs Assistant                                      | last prompt token                   |
| Eval-awareness       | deployment vs evaluation/test framing (harm/persona/realness fixed) | last token framing                  |
| Refusal (foil)       | harmful vs harmless instructions                                    | post-instruction / generation onset |

**Diff-in-means for steering vectors**: 
- we sweep certain layers (not all) and extract a steering vector per layer and per direction 
- We use these vectors as probes and measure AUROC on a held-out dataset
- We run jailbreaks on the model and measure ASR (measures jailbreak successful with a non-binary LLM-as-a-judge rubric)
- On a validation subset of the jailbreaks, we steer the different vectors to see which restore refusal on successful jailbreaks and which break the model on unsuccessful jailbreaks
- Select the steering vector that best restores refusal on a held-out dataset

Checks
- the layers with best AUROC have the most steering power?
- does eval-awareness restore refusal? **(this would be novel)**
- Build a length vector (with short vs long prompts paris) and see cosine similarity, to check if the vectors are length dependant

Extra:
- Does the steering impact capabilites (ex MMLU pro)?


### Probes (extra)

For each direction, we train a linear classifier on contrastive pairs per layer (sweeping all of them) and select the best on the AUROC of a held-out dataset of prompts.


## 5. Phase 2 — Test 1: does the probe read jailbreaks as fiction? (H2)

1. Run the probes on the rest of the jailbreak dataset
2. Apply to successful fiction/role-play jailbreaks → does the readout shift toward story-mode? 

## 6. Phase 3 — Test 2: Disentanglement (H1)

Geometric
- Cosine between story-mode and each rival:  if cosine similarity is low, it is already a good indicator that the directions are not correlated.
- **principal angles** between subspaces (refusal is a *cone* — compare subspaces, not single vectors, which also absorbs some positional noise); 
- **residual norm** of realness after projecting onto span{refusal, harm, persona, truth, eval-awareness} (small residual → not a new mechanism). Interpret raw cross-direction cosines only under matched position + layer; a large angle here is suggestive, never decisive on its own.

Probe:
- **Probe transfer:** does the story/realness probe predict harm/persona/eval-awareness labels, and vice versa? Mutual failure → separable

Checks:
- **Benign-task control:** does story-mode separate prompts on *harmless* tasks (where harm/refusal/persona shouldn't fire)? Yes → genuinely realness, not harm relabeled.

Extra experiments:
- **Decorrelating-cell check:** does realness separate real-vs-hypothetical *within* eval framing (and eval separate deployment-vs-test *within* real framing)? Both → two axes, not one.
- **Orthogonalized (amnesic/INLP) probing:** ablate harm (then persona, then truth, then eval-awareness) from activations; does realness still decode? Survival → independent information. 

## 7. Phase 4 — Test 3: causal steering + double dissociation (H3)

**(a) Single-direction steering.** On successful jailbreaks, steer away from story-mode→ measure ASR drop and over-refusal. Repeat for persona/harm/eval-awareness at matched strength.

(b) On unsuccessful jailbreaks, repeat the experiment the other way around

**(c) Double dissociation (clamped cross-steering)** 
- Clamp persona (project it out); measure compliance. *(If refusal returns from this alone → persona explains it.)*
- Project story-mode to persona and steer away the component orthogonal to it  *(If refusal returns while persona is projected out → story-mode has independent causal power.)*
- Reverse: clamp story-mode, steer persona→Assistant.
- Both directions move the outcome with the other projected out = two distinct causal channels. Repeat for the other combinations of the rest of directions


**Extra experiments:**
- **(c) Mediation chain (realness → harm → refusal?).** Steer +realness, then read the harm-perception probe via the **reply-inversion** test (2507.11878). If "toward real" flips harm-perception back to "harmful," realness sits *upstream* of harm — a genuinely new third factor.
	
- d) Path patching + necessity/sufficiency.** Patch realness activations from a "real" run into a "fiction" run; confirm refusal restored *and* effect flows through the refusal machinery (watch multiple-mediator interactions). Ablate realness (necessity) and add it (sufficiency), Arditi-style.

## 8. Controls (make every result interpretable)

- **Positive control:** steering the refusal direction *must* move behavior (proves pipeline works → a realness null is meaningful).
- **Random-direction control:** matched-norm random vector must *not* restore refusal (specificity).
- **Manipulation check:** confirm the realness steer actually moved the realness probe readout.
- **Matched strength:** compare directions at equal effect *on their own probe*, or sweep coefficients and plot **dose-response** curves.
- **Injection route:** run both system-prompt and activation steering (rankings invert — Persona Non Grata).

## 9. Metrics

- **ASR** (StrongREJECT rubric) and **over-refusal** (XSTest / OR-Bench / WildGuard) — always reported together.
- **Probe AUROC** (held-out families).
- **Separation:** cosine, principal angles, post-projection residual.
- **Causal:** ASR restored per unit of self-effect (dose-response slope); dissociation matrix.


## 11. Risks & mitigations

| Risk (source) | Mitigation |
|---|---|
| Probe reads harm, not realness, on matched pairs (Entanglement Wall) | benign-task control; held-out-family AUROC; orthogonalized probing |
| Steering nulls out via compensating cone directions (Geometry of Refusal) | concept-cone check; positive control; dose-response |
| Detectable ≠ causal (eval-awareness steering null) | manipulation check + positive control make a null substantive |
| Fiction may be a late output gate, not a rep shift ("believe what they say") | Phase 2 depth check; compare shallow vs strong framings |
| Realness = truth direction (lie-detector post) | orthogonalization vs truth; ablate-truth test |
| Realness = eval-awareness (closest sibling; confounded in naturalistic framing) | decorrelating cells (real-but-evaluated, hypothetical-but-not-eval); orthogonalize vs eval; ablate-eval test; clamped eval cross-steer (§7b) |
| Effects model-dependent (alignment-faking cross-model) | ≥2 model families; report per-model |
