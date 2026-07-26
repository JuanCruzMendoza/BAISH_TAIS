# Disentangling Language from Culture: Do Language-Specific Neurons Gate Cultural-Value Features?

**Field:** Mechanistic Interpretability (+ multilingual value-alignment evaluation)

## Research question

When a model answers the *same* values question in different languages, is it reading **one shared set of value representations** (re-expressed per language) or **different representations per language**? And do **language-specific neurons causally gate** which cultural values are read out?

## Three hypotheses 

- **H1 — Shared but language-gated:** the same value features fire across languages, but language neurons retune them so the answer tracks the query language's culture.
- **H2 — English-pivot default (the safety-relevant case):** the same features fire *and* the answer stays Western/English regardless of query language. Language changes the words, not the values.
- **H3 — Separate per language:** different value features fire per language; culture is genuinely stored per-language.

## Setup

- **Models:** Gemma-2-2B (prototype + circuit tracing) and Gemma-2-9B (main statistical run) — chosen for the complete public SAE suite (Gemma Scope) and `circuit-tracer` support.
- **Data:** ~15–20 World Values Survey (WVS) items across 2–3 value dimensions (e.g., traditional–secular, individual–collective), professionally translated into 4 value-divergent languages (English, Japanese, Arabic, Spanish). Answers constrained to a determinate token (Agree/Disagree) so they can be attributed.
- **Two lenses:** SAE **value features** (interpretable "concepts") and LAPE **language-specific neurons** (arXiv:2402.16438). Optionally reuse CAPE's culture-neuron map (arXiv:2508.02241) as step 0.
- **Value-feature identification (make-or-break step):** the *primary* selector is **attribution patching from the value-answer token → features** (causal and answer-specific — this is what makes them *value* features, not merely *culture* features). Cross-check with **mutual information** to value/country labels (descriptive — the selector introduced by CuE, arXiv:2603.23301) and a **contrastive** value-laden-vs-neutral filter. Take the intersection; report how many features survive.

## Experiments

| Move                                | Action                                                                                                                                                                  | What it settles                                                            |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **1 — Invariance (MVP)**            | Ask each item in 4 languages; compute cross-lingual **Jaccard overlap** of the value-feature sets                                                                       | Same features (H1/H2) vs. different (H3)                                   |
| **2 — Causal gating**               | Ablate the **query language's** LAPE neurons; measure change in value features **and** in the answer                                                                    | Values move with the switch (H1) vs. stay fixed (H2)                       |
| **3 — Default direction**           | With those language neurons off, measure drift of the answer (KL vs. WVS country ground truth)                                                                          | Does everything collapse toward the Western/English profile? (confirms H2) |
| **4 — Circuit deep-dive (stretch)** | For 2–3 culture-divergent items × 2 languages, build the **attribution graph** from the answer token; locate the language→value path; **validate by targeted ablation** | *How* the gating happens, at circuit level                                 |

**Metrics:** cross-lingual feature Jaccard (headline); causal effect of language-neuron ablation on value-feature activation and on the answer distribution (KL toward the English profile); % of value features that are language-invariant. **Control:** run the same pipeline on non-value factual questions to prove any invariance is value-specific.

**Design notes.** Move 3 has a sharper variant — *swap* in another language's neurons (e.g., activate English neurons on a Japanese query) to test that gating is causal and directional. Move 4 must be read **contrastively** (diff graphs across languages / value-vs-neutral), uses `circuit-tracer` / Neuronpedia on Gemma-2-2B, and treats each graph as a **hypothesis validated by intervention**, not proof.

## How this is new relative to CAPE (arXiv:2508.02241)

CAPE locates *where* culture lives; this project asks *what it encodes, whether it is shared across languages, and what it does to the model's actual values.*

| Axis | CAPE | This project |
|---|---|---|
| Unit of analysis | Raw neurons (entropy-based, opaque) | **Interpretable SAE value features** |
| Effect measured after intervention | **Perplexity** on cultural text | The model's **actual value answer** + value-feature activations |
| Across query languages | Neurons found per-language separately; **no invariance test** | **Cross-lingual feature invariance** (Jaccard) is the core metric |
| Language ↔ value link | Culture and language neurons treated as **parallel/separate** (set subtraction) | **Builds the causal bridge**: language neurons → value features/answer |
| "Whose values are default?" | Not asked | **English-pivot test** (Move 3) |
| WVS usage | As a **corpus** to detect neurons | As an **elicited opinion** that is scored |

This is a well-positioned **follow-up**, not a from-scratch discovery: CAPE's own future-work section calls for exactly this — *"future work should include more comprehensive behavioral and downstream assessments"* and *"effects of culture neuron interventions on downstream tasks and human-centered evaluation."* Novelty rests on the **interpretable-feature + behavioral-value + cross-lingual** axes; the *behavioral* English-pivot finding alone is already known (arXiv:2402.18120), so the contribution is the **mechanism**, not the bias.

## How this differs from CuE (arXiv:2603.23301)

CuE is the closest methodological work — it is the one paper that uses SAEs to represent culture. The overlap is real and is **credited, not re-claimed**: using SAE features for culture, selecting them by mutual information with country labels, and finding an Anglophone default are all CuE's. The contribution here is the two axes CuE never touches.

| Axis | CuE | This project |
|---|---|---|
| Feature selection | Features **correlated with a country** (MI-to-country, descriptive) | Features that **causally drive the value answer** (attribution from the answer token) |
| Query language varied? | **No** | **Yes** — cross-lingual invariance is the core test (Move 1) |
| Language neurons | **Never touched** | **Ablated to test gating** (Move 2) |
| Core question | *Localize and steer cultural knowledge* | *Is culture entangled with language — do language neurons gate values?* |

**One-line distinction:** CuE finds what is culturally salient and steers with it; this project finds what determines the model's stance on a value and tests whether that is language-invariant and language-gated. Framing is load-bearing: drift toward "find SAE cultural features + show an Anglophone default" collapses into CuE, so the cross-lingual and language-neuron axes must stay the headline.

## Why it matters for AI safety

- **Eval validity / alignment generalization (primary).** If safety-relevant values are shared features merely gated by language (H1), or a fixed Anglophone default (H2), then alignment and red-teaming performed **in English do not certify behavior in other languages**. English-only safety evaluation would give **false assurance** for most of the world's users — a concrete, testable failure mode of current safety practice. Move 1's invariance metric directly quantifies "does an alignment property established in English survive a change of query language at the feature level?"
- **Value imposition (secondary).** If H2 holds, non-English users silently receive Anglophone values regardless of the language they use — a concrete mechanism of cultural homogenization.
- **Method with independent value.** A validated way to test whether a value/safety property is language-invariant at the feature level is reusable for auditing any multilingual model, beyond this study.

## Key prior work

- CAPE — *Isolating Culture Neurons in Multilingual LLMs* (arXiv:2508.02241)
- LAPE — *Language-Specific Neurons* (arXiv:2402.16438)
- *Do Llamas Work in English?* (arXiv:2402.10588)
- *Steering LLMs for Culturally Localized Generation* / CuE (arXiv:2603.23301) — SAE cultural features + steering, no language-neuron link
- *Exploring Multilingual Concepts of Human Value* (arXiv:2402.18120) — behavioral English-pivot via concept vectors
- `circuit-tracer` (BlackboxNLP 2025) + Neuronpedia attribution graphs (Gemma-2-2B)
