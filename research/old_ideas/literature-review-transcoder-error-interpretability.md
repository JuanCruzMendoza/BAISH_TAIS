# AI Safety Literature Review: Interpretability of Transcoder/SAE Reconstruction Error ("dark matter")

> Generated: 2026-07-18
> Scope: Is the reconstruction error left by sparse dictionary learning (SAEs, transcoders) structured, functional, and safety-relevant — or noise? What would make an "amplify-the-error" experiment on a Gemma Scope 2 transcoder more novel?
> Method: 5 parallel search agents (academic papers/surveys, open-problems/agendas, adjacent methods, LessWrong/AF, benchmarks/tooling). Abstracts/intros only. Two most consequential 2026 claims verified directly against arXiv.

## Summary

The narrow "is the error noise or structure?" question is **already settled: it is structured, not noise.** Gurnee showed the SAE error is causally pathological (perturbing along it hurts next-token prediction 2–4.5× more than a random vector of equal norm); Engels et al. showed ~half the error vector and >90% of its *norm* is linearly predictable from the input, leaving a smaller genuinely **nonlinear** residual; Kutsyk et al. showed error nodes carry real cross-layer computation. So the base experiment's headline question yields little new knowledge on its own.

What is **open** is the *behavioral/safety* content of the error, tested by *amplification*, on *transcoders* specifically. Prior steering work uses dictionary **features** (refusal, sycophancy), not the error term; prior error work uses ablation/probing, not forward-pass amplification; and it is almost all SAEs, not transcoders. The one strong exception — and the nearest-neighbor to guard against — is Cui et al. 2026 ("SAE Interventions are Unreliable"), which shows suppressed refusal behavior is recoverable **through the reconstruction residual** (95.8% recovery), with attribution localizing recovery to "the component left unexplained by the SAE." That is a partial scoop of "the residual carries safety behavior," so the project must differentiate on axis (transcoder vs SAE), mechanism (injection/steering vs recovery-after-clamping), and the ± direction-dependence test.

The single highest-leverage design change, recommended independently by four of five agents: **do not amplify the whole error — decompose it first (linear-predictable vs nonlinear) and amplify only the nonlinear residual.** Combined with a matched-norm random control and a degradation curve, this converts the study from "is the error noise?" into the sharper, more defensible "is the *nonlinear* MLP computation a transcoder misses a direction-dependent, safety-relevant input to the refusal circuit that a better dictionary would not absorb?"

## Top Sources

1. [Decomposing the Dark Matter of Sparse Autoencoders](https://arxiv.org/abs/2410.14670) — Engels, Riggs, Tegmark (2024). The load-bearing prior: splits the error into linear-predictable (~half the vector, >90% of norm) vs nonlinear components. Dictates decomposing before amplifying.
2. [SAE reconstruction errors are (empirically) pathological](https://www.alignmentforum.org/posts/rZPiuFxESMxCDHe4B) — Gurnee (2024). Empirical backbone for "not noise"; supplies the matched-norm random-vector control protocol.
3. [SAE Interventions are Unreliable: Post-Intervention Recovery of Suppressed Behavior](https://arxiv.org/abs/2606.18322) — Cui, Shen, Yang (2026). **Nearest neighbor / partial scoop.** Suppressed refusal behavior recovers through the reconstruction residual (95.8%); attribution localizes to the unexplained component. Must differentiate.
4. [What is the functional role of SAE errors?](https://www.lesswrong.com/posts/WzHPpMz2kRongsA7q) — Kutsyk, Hua, woog, Assis (2025). Closest problem-level prior; ablation+restoration on Gemma-2; cross-layer-superposition hypothesis; their **crosscoder analysis of the error failed to train** (an explicit open door).
5. [Transcoders Beat Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2501.18823) — Paulo & Belrose (2025). Skip/affine transcoders; frames the coder's error as "dark matter." The architecture Gemma Scope 2 ships.
6. [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496) — Sharkey et al. (TMLR 2025). §2.1.2c names error-node incompleteness as an open problem and calls error nodes "an inadequate solution" because they contain "everything else."
7. [Gemma Scope 2 (technical report + weights)](https://huggingface.co/google/gemma-scope-2-4b-it) — Google DeepMind (2025). The artifact: SAEs + transcoders **with and without affine skip** for every layer of Gemma 3, pretrained and instruct. Enables skip-vs-nonskip and base-vs-instruct ablations for free.
8. [Negative Results for SAEs on Downstream Tasks](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9) — GDM Mech Interp (2025). Probes on the residual beat probes on the SAE reconstruction OOD — the discarded info is task-relevant. Converts your amplification into a *causal* test of a correlational finding.

## Subfields and Research Directions

### 1. Structure of the reconstruction error ("dark matter")

**Description:** What the error is made of, and how much is reducible.

**Key findings/works:** Engels et al. ([2410.14670](https://arxiv.org/abs/2410.14670)) — linear-predictable vs nonlinear split; [Understanding SAE scaling in the presence of feature manifolds](https://arxiv.org/abs/2509.02565) — why some error is irreducible; [Ensembling SAEs](https://arxiv.org/abs/2505.16077); [A Unified Theory of Sparse Dictionary Learning](https://arxiv.org/abs/2512.05534) (spurious minima → residual). **Absorption/splitting** leaks concept mass into the error: [A is for Absorption](https://arxiv.org/abs/2409.14507) (Chanin et al.), [SAEs Do Not Find Canonical Units](https://arxiv.org/abs/2502.04878).

**Methodologies:** linear regression of error on input; feature-manifold analysis; scaling laws.

### 2. Functional / behavioral role of the error (most relevant)

**Description:** Does the error causally drive behavior, and can it be steered?

**Key works:** Gurnee (pathological); Kutsyk et al. (functional role, ablation+restoration); [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647) (Marks et al. — explicit "error nodes"); [e2e Sparse Dictionary Learning](https://arxiv.org/abs/2405.12241) (Braun et al. — functional ≠ reconstruction importance; KL-based objective); [Evidence for feature-specific error correction](https://www.lesswrong.com/posts/uDrsffSLzWD6cDnTt) (self-repair may damp amplification at 1×). Nearest-neighbor: Cui et al. ([2606.18322](https://arxiv.org/abs/2606.18322)).

### 3. Transcoders, crosscoders, and circuit tracing

**Description:** The specific decomposition family the experiment uses, and its faithfulness limits.

**Key works:** [Transcoders Find Interpretable Circuits](https://arxiv.org/abs/2406.11944) (Dunefsky et al.); Paulo & Belrose (skip transcoders); [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) (Anthropic — error nodes as a named limitation); [CLTs are incentivized to learn Unfaithful Circuits](https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor) (CLTs collapse multi-hop→single-hop, hiding computation); [Jacobian SAEs](https://arxiv.org/abs/2502.18147) ("MLPs are ~linear in the JSAE basis" — reframes the error as the irreducibly nonlinear MLP computation); crosscoders for chat-tuning ([2504.02922](https://arxiv.org/abs/2504.02922), Minder et al.).

### 4. Safety-relevant content that dictionaries miss

**Description:** Evidence that specifically safety-relevant computation lands in the residual/blind spot.

**Key works:** [SAEs are highly dataset dependent: the refusal direction](https://www.alignmentforum.org/posts/rtp6n7Z23uJpEH7od) (Kissane et al. — webtext SAEs fail to reconstruct refusal; it lands in the residual); GDM Negative Results (residual probes win OOD); [Auditing LMs for Hidden Objectives](https://www.lesswrong.com/posts/wSKPuBfgkkqfTpmWJ) (Marks et al.); [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (deceptive outputs without expected features firing).

### 5. Steering rigor and evaluation

**Description:** How to avoid mistaking degradation for steering, and how to make the LLM judge trustworthy.

**Key works:** [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717) (Arditi et al.) and [There Is More to Refusal than a Single Direction](https://arxiv.org/abs/2602.02132) (targets to project onto); "A Sober Look at Steering Vectors" and [multi-behavior/inverted-U study](https://arxiv.org/abs/2511.18284) (effect peaks at moderate scale while coherence declines monotonically — the exact 1×–5× confound); [Analysing the Safety Pitfalls of Steering Vectors](https://arxiv.org/abs/2603.24543) (protocol analog); LLM-judge reliability: [Reliability without Validity](https://arxiv.org/abs/2606.19544) (report chance-corrected κ, position/verbosity controls).

**Key organizations:** Anthropic, Google DeepMind, Apollo Research, EleutherAI, MIT/IAIFI, UK AISI, Stanford NLP.
**Key authors:** Neel Nanda, Lee Sharkey, Joshua Engels, Wes Gurnee, Arthur Conmy, Nora Belrose, Sam Marks, Andy Arditi, Lucius Bushnaq.

---

## Further Research: Novelty-Boosting Directions

Ordered by novelty-leverage per unit effort. Tier 1 = do these regardless; Tier 2 = elevates the claim from methodological to mechanistic/safety; Tier 3 = higher-effort, higher-payoff differentiators.

### Tier 1 — Sharpen the core (cheap, high leverage)

1. **Decompose-then-steer (the single biggest upgrade).** Regress the error on the input activation (Engels), subtract the linear-predictable part, and amplify **only the nonlinear residual**. *New question:* is the blind spot *fundamental* (nonlinear, no dictionary can absorb it) or *fixable* (a better transcoder would capture it)? Cleaner and more publishable than amplifying the whole vector — which mostly amplifies "fake dark matter."
2. **Skip vs non-skip transcoder error.** Gemma Scope 2 ships both per layer; the affine skip already removes the linear map, so its residual is a purer nonlinear object. Run the ± sweep on both and compare — a tooling-native, near-free way to operationalize #1.
3. **Matched-norm random control + dose-response curve.** For every ±k·ε injection, add a random vector of equal norm (Gurnee's protocol), judged blind; replace the 4 discrete scales with a finer sweep and jointly fit a behavioral-effect curve and a coherence/perplexity curve. *New question:* is the shift direction-specific and in a usable-coherence band (real steering, inverted-U) or generic monotonic degradation? This is the rigor floor that makes any positive result defensible; the ± symmetry already in the plan is the right instinct — this completes it.

### Tier 2 — Turn it into a mechanistic/safety claim

4. **Base-vs-instruct error diffing.** Compute and amplify the error on both pretrained and instruction-tuned Gemma 3 4B (both in Gemma Scope 2); use Latent Scaling (Minder et al.) to avoid crosscoder artifacts. *New question:* is the safety-relevant content **introduced by instruction tuning/RLHF** (present in instruct error, absent in base)? Localizes where alignment computation hides.
5. **Attribute the amplified error to the refusal circuit.** Project the error onto the Arditi refusal direction and/or feed it through a sparse-feature-circuit / attribution graph (Marks et al.; circuit-tracer). *New question:* does the error act *on* the known safety circuit, or through an orthogonal, previously-uncharacterized pathway?
6. **Safety-content targeting.** Rather than generic error, project onto known safety probes (deception, harmful-intent, sycophancy) and amplify only that projection. Reframes the contribution from "behavior shifts" to "audits miss *safety-relevant* computation in the residual" — directly answering the UK AISI / Oxford AIGI auditing agendas.

### Tier 3 — Higher-effort differentiators

7. **Meta-dictionary / crosscoder on the error (an explicitly failed open problem).** Kutsyk et al. tried a crosscoder analysis of the error and it *failed to train*. Train a secondary SAE/crosscoder over per-layer error vectors and steer with individual meta-latents. *New question:* does the error decompose into a few **nameable** safety-relevant directions, turning "noise vs signal" into "how many, and what are they?" Success here is a concrete novel contribution.
8. **Self-repair confound test.** A null at 1× may be downstream self-repair, not irrelevance. Repeat with downstream layers frozen/ablated ([feature-specific error correction](https://www.lesswrong.com/posts/uDrsffSLzWD6cDnTt)). *New question:* does the model actively cancel the error, and does that mask its functional content?
9. **Quantify circuit "dark matter."** Embed amplification in an attribution-graph pipeline (circuit-tracer): measure what fraction of behavior on your prompts routes through per-layer error nodes vs features, then amplify the error node in-graph. *New question:* do "uninterpretable" nodes carry safety-relevant computation?

### Scoop-avoidance: differentiate from Cui et al. 2026 ([2606.18322](https://arxiv.org/abs/2606.18322))

They show the residual **restores suppressed** behavior after feature clamping (recovery, SAE, optimization-based). Position this work as the injection/steering counterpart: amplifying the error **injects/steers** behavior with **no feature clamping**, on **transcoders** (nonlinear MLP computation, not SAE residual-stream), with a **± direction-dependence** test. Cite them as the suppression-side complement, not competition.

---

## Landscape Gaps

- **Behavioral content of the error via amplification:** unaddressed — steering uses features; error work uses ablation/probing.
- **Transcoders specifically:** dark-matter analysis is almost entirely SAEs; transcoder error (esp. skip/affine → nonlinear residue) is under-studied.
- **Nonlinear-only steering:** nobody isolates and amplifies just the nonlinear residual (Engels stops at structure/loss).
- **Where safety content in the error originates:** base-vs-instruct diffing of the error is unexplored.
- **Methodology/infrastructure gaps (must build):** an error-scaling forward hook (`err = true_out − transcoder_out`; reinsert `true_out + (k−1)·err`, correct affine-skip handling + IT chat template); a stratified ~50-prompt panel (no single benchmark covers refusal + over-refusal + sycophancy + benign at n≈50); a paired baseline↔intervention behavioral-shift rubric judge with κ + position/verbosity controls; a degeneration/coherence guard; and verification that circuit-tracer's presets actually wire Gemma Scope 2 4B-IT *skip*-transcoders.
- **Replication gap:** verify Gemma Scope 2 skip-vs-nonskip claims on the exact 4B-IT weights before building on them.

## Benchmarks & Tooling (quick reference)

**Prompts/metrics:** [SORRY-Bench](https://github.com/SORRY-Bench/SORRY-Bench) (category-balanced harmful — best for a diverse panel), [StrongREJECT](https://github.com/dsbowen/strong_reject) (harmful + 0–1 autograder — best refusal quantifier), [HarmBench](https://github.com/centerforaisafety/HarmBench) (classifier as second judge), [XSTest](https://huggingface.co/datasets/natolambert/xstest-v2-copy) + [OR-Bench](https://github.com/justincui03/or-bench) (over-refusal/caution — critical, since amplification may increase caution), [Anthropic sycophancy evals](https://github.com/meg-tong/sycophancy-eval), catalog: [SafetyPrompts.com](https://safetyprompts.com/). Protocol analog: [Analysing the Safety Pitfalls of Steering Vectors](https://arxiv.org/abs/2603.24543).

**Tooling:** [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) (models error nodes + feature interventions; Gemma-3 via nnsight — primary splicing harness), [SAELens](https://github.com/decoderesearch/SAELens) (official Gemma Scope 2 loader; error-term hook), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [nnsight/nnterp](https://github.com/ndif-team/nnsight), [pyvene](https://github.com/stanfordnlp/pyvene) (compare vs contrastive steering), [Inspect AI](https://inspect.aisi.org.uk) (eval harness — pairs with the local `inspect-read-logs` skill), [Neuronpedia Gemma Scope 2](https://www.neuronpedia.org/gemma-scope-2). Also note [ReSAE](https://arxiv.org/abs/2605.27819) (residualized SAEs) and [CRaFT](https://arxiv.org/abs/2604.01604) (CLT refusal-feature selection) as recent adjacent methods.

## All Sources

| Title | Type | Org | URL |
|---|---|---|---|
| Decomposing the Dark Matter of SAEs | paper | MIT/IAIFI | https://arxiv.org/abs/2410.14670 |
| SAE reconstruction errors are pathological | post | — | https://www.alignmentforum.org/posts/rZPiuFxESMxCDHe4B |
| SAE Interventions are Unreliable | paper (2026) | — | https://arxiv.org/abs/2606.18322 |
| What is the functional role of SAE errors? | post | AISC | https://www.lesswrong.com/posts/WzHPpMz2kRongsA7q |
| Transcoders Beat SAEs | paper | EleutherAI | https://arxiv.org/abs/2501.18823 |
| Transcoders Find Interpretable Circuits | paper | — | https://arxiv.org/abs/2406.11944 |
| Circuit Tracing (attribution graphs) | report | Anthropic | https://transformer-circuits.pub/2025/attribution-graphs/methods.html |
| CLTs incentivized to learn unfaithful circuits | post | — | https://www.lesswrong.com/posts/6CS2NDmoLCFcEJMor |
| Jacobian SAEs | paper | — | https://arxiv.org/abs/2502.18147 |
| Open Problems in Mechanistic Interpretability | survey | multi | https://arxiv.org/abs/2501.16496 |
| Gemma Scope 2 (weights) | artifact | Google DeepMind | https://huggingface.co/google/gemma-scope-2-4b-it |
| Negative Results for SAEs on Downstream Tasks | report | Google DeepMind | https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9 |
| Sparse Feature Circuits | paper | — | https://arxiv.org/abs/2403.19647 |
| e2e Sparse Dictionary Learning | paper | Apollo | https://arxiv.org/abs/2405.12241 |
| A is for Absorption | paper | — | https://arxiv.org/abs/2409.14507 |
| SAEs Do Not Find Canonical Units | paper | — | https://arxiv.org/abs/2502.04878 |
| Refusal is mediated by a single direction | paper | — | https://arxiv.org/abs/2406.11717 |
| More to Refusal than a Single Direction | paper (2026) | — | https://arxiv.org/abs/2602.02132 |
| SAEs are highly dataset dependent (refusal) | post | — | https://www.alignmentforum.org/posts/rtp6n7Z23uJpEH7od |
| Crosscoders for chat-tuning artifacts | paper | — | https://arxiv.org/abs/2504.02922 |
| Multi-behavior activation control (inverted-U) | paper | — | https://arxiv.org/abs/2511.18284 |
| Analysing the Safety Pitfalls of Steering Vectors | paper (2026) | — | https://arxiv.org/abs/2603.24543 |
| Reliability without Validity (LLM judge) | paper (2026) | — | https://arxiv.org/abs/2606.19544 |
| Understanding SAE scaling / feature manifolds | paper | — | https://arxiv.org/abs/2509.02565 |
| Evidence for feature-specific error correction | post | — | https://www.lesswrong.com/posts/uDrsffSLzWD6cDnTt |
| ReSAE: Residualized SAEs | paper | — | https://arxiv.org/abs/2605.27819 |
| CRaFT: Circuit-Guided Refusal via CLTs | paper (2026) | — | https://arxiv.org/abs/2604.01604 |

> Caveat: several 2026 arXiv IDs come from abstract-level reads this session; the two most consequential (2606.18322, the CLT-unfaithfulness post) were verified directly. Verify others before citing formally.
