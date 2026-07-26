# AI Safety Literature Review: Cultural & Linguistic Bias in Multilingual LLMs — and Novel Project Ideas

> Generated: 2026-07-17
> Seed: [`A4-disentangling-language-from-culture.md`](A4-disentangling-language-from-culture.md) (Do language-specific neurons gate cultural-value features?)
> Method: 3 parallel research agents (Alignment Forum/LW; open-problems & agendas; behavioral value alignment) + ~12 targeted web searches + abstract-level reads of the closest papers. Source policy: abstracts/landing pages only.

---

## Summary

The field this proposal sits in has **matured fast in 2025–2026** and is now crowded on some axes and wide open on others. Three research communities that used to be separate are converging:

1. **Behavioral cultural-value evaluation** — LLMs default to Western/English/Protestant-European values on the World Values Survey (WVS) and GlobalOpinionQA, and — critically — *translating a question into a language does not move the answer toward that language's speakers* (Anthropic's GlobalOpinionQA; Tao et al. PNAS Nexus). LLMs also show **higher-than-human cross-lingual similarity**, i.e., they *flatten* real cultural variation.
2. **Mechanistic interpretability of multilingual representation** — models converge to a **shared, English-centric latent space in middle layers** and re-diverge to language-specific space in late layers (Do Llamas Work in English?; Cross-Layer Transcoders; Transfer Neurons). This is established for **facts and translation**, but *contested for values*: activation-patching finds language-*agnostic* concepts (Dumas et al., EPFL), while concept-probing finds value concepts are language-*specific* (Xu et al., Tianjin).
3. **Cross-lingual safety** — safety behaviors are robust in high-resource languages and **fail in low-resource ones** (translating harmful prompts into low-resource languages jailbreaks GPT-4 ~79% of the time). A fresh mechanistic result — **"Refusal Direction is Universal Across Safety-Aligned Languages" (NeurIPS 2025)** — shows the *downstream* refusal direction is shared, which reframes the open question from "is there a shared refusal representation?" to "why isn't it triggered?"

**The single most important seam** (independently surfaced by the Alignment Forum evidence and the behavioral literature): **representations look increasingly language-invariant with scale, yet elicited values/safety behaviors are demonstrably NOT invariant.** Where in the stack does invariant representation become variant behavior — and is that gap the mechanism of both value imposition and multilingual jailbreaks? That question is under-explored and is where the highest-value, safety-relevant projects live.

**Consequence for the A4 proposal:** it remains viable but is no longer first-mover on "disentangle language from culture." It must now cite and differentiate against several 2025–2026 near-neighbors (below). Its surviving novelty is the **combination**: SAE *value* features selected by **causal attribution from the value-answer token** + **language-neuron ablation** + **English-pivot test** + **WVS-scored answer**. However, the biggest, least-crowded upside is to **pivot the same machinery from cultural *values* to *safety behaviors*** (refusal, harm-recognition, sycophancy), where stakes are higher and the interpretability field is thinner.

---

## Top Sources

1. [Refusal Direction is Universal Across Safety-Aligned Languages](https://arxiv.org/abs/2505.17306) (NeurIPS 2025) — the shared refusal direction result; reframes the cross-lingual-safety mechanism question.
2. [The Multilingual Divide and Its Impact on Global AI Safety](https://arxiv.org/abs/2505.21344) (Cohere Labs) — the definitive open-problems/agenda piece for this exact topic; English-centric safety, translationese undermining eval validity, localized-vs-translated evals.
3. [Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](https://deepmind.google/research/publications/225819/) (DeepMind, ICML 2026) — cultural-zone membership explains safety-rating variance beyond demographics; ~10% of safety items are culturally sensitive; LLMs can't substitute for diverse human raters.
4. [Language over Content: Tracing Cultural Understanding in Multilingual LLMs](https://arxiv.org/abs/2510.16565) — **closest neighbor**: disentangles language from culture via circuit path-overlap; finds *language dominates content*.
5. [Neuron-Level Analysis of Cultural Understanding in LLMs](https://arxiv.org/abs/2510.08284) (ICLR 2026) — culture-general vs culture-specific neurons (<1%, shallow-mid MLP); code: CULNIG.
6. [Entangled in Representations: Mechanistic Investigation of Cultural Biases](https://arxiv.org/abs/2508.08879) (Alice Oh, Isabelle Augenstein et al.) — "Culturescope"; cultural-flattening score; traces Western-dominance bias internally.
7. [Exploring Multilingual Concepts of Human Value](https://arxiv.org/abs/2402.18120) (Xu et al., Tianjin) — behavioral+concept English-pivot; value concepts language-specific, unidirectional high→low-resource transfer.
8. [Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations](https://arxiv.org/abs/2411.08745) (EPFL, West group) — the counter-evidence: concepts encoded separately from output language.
9. [A Roadmap to Pluralistic Alignment](https://arxiv.org/abs/2402.05070) (Sorensen et al., ICML 2024) — Overton / steerable / distributional framing; standard RLHF can *collapse* value diversity.
10. [Unintended Impacts of LLM Alignment on Global Representation](https://arxiv.org/abs/2402.15018) (Ryan, Held, Yang, Stanford) — RLHF/DPO degrade non-English/global representation as a side effect.

---

## Subfields and Research Directions

### 1. Behavioral cultural-value evaluation (the "what" of bias)

**Description:** Measures the values/opinions an LLM expresses, usually by replaying survey instruments (WVS, Pew) and comparing to human distributions per country.

**Key findings / open problems:**
- Default outputs cluster on US/Western/Protestant-European values; **question translation ≠ cultural localization** ([GlobalOpinionQA](https://arxiv.org/abs/2306.16388); [Tao et al.](https://arxiv.org/abs/2311.14096)).
- **Multilingual ≠ multicultural**: multilingual capability does not predict cultural alignment; self-consistency predicts it better ([Rystrøm, Kirk, Hale](https://arxiv.org/abs/2502.16534)).
- LLMs **flatten** cross-cultural moral variation (more uniform than humans) ([One Model, Many Morals](https://arxiv.org/abs/2509.21443)).
- **Measurement validity is weak**: "culture" is poorly operationalized in MCQ benchmarks ([Hire Your Anthropologist!](https://arxiv.org/abs/2510.05931)); measured "values" are an **artifact of elicitation method** (classification vs CoT vs open-ended) ([Rethinking AI Cultural Alignment](https://arxiv.org/abs/2501.07751), Barez et al.).

**Key orgs/authors:** Anthropic (Durmus, Ganguli); Cornell/KTH (Kizilcec, Viberg); Oxford Internet Institute (Kirk, Hale); Michigan (Jurgens); Oxford/UCL (Barez).
**Benchmarks:** WVS, GlobalOpinionQA, WorldValuesBench, CDEval, BLEnD, NormAd, CIVICS, Camellia, Global MMLU.

### 2. Mechanistic interpretability of multilingual representation (the "where/how")

**Description:** Locates and characterizes how multilingual models store language vs. concept, using neurons, SAE features, activation patching, and circuit/attribution graphs.

**Key findings / open problems:**
- **Shared middle-layer space, English-centric, language-specific late layers** ([Do Llamas Work in English?](https://arxiv.org/abs/2402.10588); [Cross-Layer Transcoders](https://arxiv.org/abs/2511.10840); [Transfer Neurons](https://arxiv.org/abs/2509.17030)).
- **Contested for concepts:** language-*agnostic* ([Separating Tongue from Thought](https://arxiv.org/abs/2411.08745)) vs language-*specific* for values ([Xu et al.](https://arxiv.org/abs/2402.18120)). *This contradiction, unresolved for values specifically, is the A4 opening.*
- **Language-specific neurons (LAPE)** exist ([2402.16438](https://arxiv.org/abs/2402.16438)) but a 2025 result argues they **do not facilitate cross-lingual transfer** ([2503.17456](https://arxiv.org/abs/2503.17456)) — a caution for any "ablate language neurons" design.
- **Multilingual SAEs** now exist ([SAEs capture language-specific concepts](https://arxiv.org/abs/2507.11230); [Multilingual Steering by Design](https://arxiv.org/abs/2605.23036)); **script can dominate linguistic structure** ([2604.05090](https://arxiv.org/abs/2604.05090); [One Language, Two Scripts](https://arxiv.org/abs/2603.08869)).
- **Tokenizer bias toward English** is a named mechanistic source of non-English underperformance ([Cross-Layer Transcoders](https://arxiv.org/abs/2511.10840)).

**Methods/tools:** Gemma Scope SAEs ([2408.05147](https://arxiv.org/abs/2408.05147)), SAELens, circuit-tracer, Neuronpedia, activation/attribution patching, LAPE, CULNIG.

### 3. Mechanistic interpretability of *culture/values* specifically (the crowded core)

**Description:** The direct precursors/competitors to A4 — localizing and steering cultural knowledge/values internally.

- [CAPE](https://arxiv.org/abs/2508.02241) — culture neurons (entropy-based); the paper A4 explicitly follows up.
- [Neuron-Level Analysis](https://arxiv.org/abs/2510.08284) (ICLR 2026) — culture-general vs -specific neurons; suppression drops cultural benchmarks up to 30%.
- [Entangled in Representations / Culturescope](https://arxiv.org/abs/2508.08879) — cultural-flattening score; low-resource cultures *less* biased (limited parametric knowledge).
- [Language over Content](https://arxiv.org/abs/2510.16565) — circuit path-overlap; **language dominates culture**; same-language cross-country > cross-language same-country overlap.
- [Cultural Value Alignment via Latent Activation Steering](https://arxiv.org/abs/2605.26365) — WVS + 300-dilemma behavioral probing; finds value dimensions are **coupled/entangled** (steering one shifts others).
- [Rethinking Cross-lingual Alignment: cultural erasure / Surgical Steering](https://arxiv.org/abs/2510.26024) — cross-lingual alignment improves facts *at the cost of* cultural localization; facts vs culture best steered at *different layers*.
- CuE — SAE cultural features + steering ([2603.23301](https://arxiv.org/abs/2603.23301), the methodological near-twin A4 already credits).

### 4. Cross-lingual safety (the higher-stakes, thinner-interpretability frontier)

**Description:** Whether safety/refusal behaviors survive a change of language, and why they fail.

- **Behavioral failure well-established:** low-resource jailbreaks (~79% on GPT-4); [Towards Safe Multilingual Frontier AI](https://arxiv.org/abs/2409.13708) (jailbreak vulnerability ∝ under-resourcedness across 24 EU languages); [Sycophancy as a Multilingual Alignment Failure](https://arxiv.org/abs/2606.08451).
- **Mechanistic:** [Refusal Direction is Universal Across Safety-Aligned Languages](https://arxiv.org/abs/2505.17306) (NeurIPS 2025) — the refusal *direction* is shared; [LASA](https://arxiv.org/abs/2604.12710) — language-agnostic alignment at a "semantic bottleneck."
- **Representation-level fixes (unaudited internally):** [Enforcing Multilingual Consistency](https://arxiv.org/abs/2602.16660); [Multilingual Safety Alignment via Self-Distillation](https://arxiv.org/abs/2605.02971); [Middle-Layer Representation Alignment](https://arxiv.org/abs/2502.14830).
- **SAE-for-safety (English-only or domain-only):** [Steering LM Refusal with SAEs](https://arxiv.org/abs/2411.11296); [Safe-SAIL](https://arxiv.org/abs/2509.18127); [BioRefusalAudit](https://arxiv.org/abs/2605.30162).

### 5. Homogenization, value imposition & pluralistic alignment (the "why it matters")

- [Unintended Impacts of Alignment on Global Representation](https://arxiv.org/abs/2402.15018); [The Homogenizing Effect of LLMs](https://arxiv.org/abs/2508.01491) (USC); [AI Suggestions Homogenize Writing Toward Western Styles](https://dl.acm.org/doi/10.1145/3706598.3713564) (CHI 2025); [Artificial Hivemind](https://arxiv.org/abs/2510.22954).
- **Agendas:** [Roadmap to Pluralistic Alignment](https://arxiv.org/abs/2402.05070); [DeepMind geo-cultural salience](https://deepmind.google/research/publications/225819/); [Cohere Multilingual Divide](https://arxiv.org/abs/2505.21344); [PlurVA-LLM @ AACL 2026](https://www.aclweb.org/portal/content/first-cfp-first-workshop-pluralistic-value-alignment-llms-plurva-llm-aacl-2026) (names "interpretability of value alignment" + "multilingual/multicultural" together — the closest agenda framing to A4).

---

## Landscape Gaps (where novelty is available)

- **Values vs. facts, mechanistically.** The shared-English-space evidence is almost all on facts/translation. Whether *values* live in that shared space is the contested, under-tested core.
- **Behavioral × mechanistic rarely combined on the same value construct.** Most work does one or the other. Linking an internal feature causally to cross-lingual value *expression* is rare.
- **Safety features ≠ cultural values.** Interpretability of *cultural values* is now crowded; interpretability of *cross-lingual safety* (refusal, harm-recognition, sycophancy) at the SAE-feature level, tied to language neurons, is thin.
- **Recognition vs. execution split.** The refusal *direction* is universal — so why do low-resource jailbreaks work? Nobody has cleanly separated **harm-recognition** (upstream) from **refusal-execution** (downstream) *across languages*.
- **The "fixes" are unaudited internally.** Cross-lingual safety-alignment methods (LASA, consistency loss, self-distillation) are evaluated behaviorally; nobody checks whether they create *genuinely shared* features (that generalize to unseen languages/attacks) or brittle per-language patches.
- **No scaling curve** for value/safety-feature cross-lingual invariance.
- **No standard metric/tool** for "is safety property P language-invariant at the feature level?" (infrastructure gap).
- **Script × language × culture** never fully dissociated for values/safety.
- **Elicitation-method sensitivity** (values shift with prompt format) is a validity threat almost no mechanistic study controls for.

---

## Novel Project Ideas (ranked; each with novelty positioning + safety relevance + method + feasibility)

> Ranking weights: (a) novelty against the 2025–2026 near-neighbors above, (b) AI-safety importance, (c) feasibility on Gemma-2 2B/9B/27B with Gemma Scope + circuit-tracer (the A4 toolkit). Ideas 1–2 are the strongest bets; 3–5 are high-value pivots of the A4 machinery; 6–8 are method/design contributions.

### ⭐ Idea 1 — Recognition vs. execution: a mechanistic account of *why* multilingual jailbreaks work
**Claim to test:** Multilingual jailbreaks succeed not because the refusal *representation* is missing in low-resource languages, but because **upstream harm-recognition features fail to fire**, so the (shared) refusal direction is never triggered.
**Why novel:** [Refusal Direction is Universal](https://arxiv.org/abs/2505.17306) shows the *execution* direction is shared but stops there; it doesn't separate recognition from execution, doesn't use SAE features, and doesn't test the low-resource *triggering* failure. No one has cleanly split "does the model *notice* the request is harmful" from "does it *refuse*" across languages.
**Method:** With Gemma Scope SAEs, isolate (a) harm-recognition features (contrastive harmful/benign prompts) and (b) the refusal-execution direction. Across high- and low-resource languages (translate AdvBench-style prompts), measure which stage degrades. Use LAPE language neurons to test whether they gate recognition. Validate causally: patch harm-recognition features from English into a low-resource prompt — does refusal recover?
**Safety relevance:** Targets the highest-stakes failure (jailbreaks), predicts *which* languages are vulnerable and *why*, and says *where* to intervene (boost recognition vs. execution). Directly operationalizes "English red-teaming doesn't certify other languages."
**Feasibility:** High. Same toolkit as A4; refusal-direction extraction is well-documented; AdvBench translations are cheap.

### ⭐ Idea 2 — Do cross-lingual safety-alignment methods create *shared* safety features or *brittle patches*?
**Claim to test:** Methods like [Enforcing Multilingual Consistency](https://arxiv.org/abs/2602.16660), [Self-Distillation](https://arxiv.org/abs/2605.02971), and [LASA](https://arxiv.org/abs/2604.12710) make outputs safer per-language without making the model *use the same internal safety features* — so they won't generalize to unseen languages/attacks.
**Why novel:** These methods are evaluated purely behaviorally. An **interpretability audit of the fixes themselves** (feature-level invariance before/after; held-out-language and held-out-attack generalization) is unclaimed territory — a "failure mode of the solutions" contribution.
**Method:** Reproduce a lightweight representational-consistency fine-tune (or audit a released aligned checkpoint). Before/after, compute cross-lingual Jaccard of active safety features (Idea-1 machinery) and OOD generalization on held-out languages/attack types. Diagnostic: genuine shared-feature reuse vs. per-language add-ons.
**Safety relevance:** Tells the field whether its current cross-lingual safety methods are mechanistically robust or Potemkin — high leverage for anyone deploying multilingually.
**Feasibility:** Medium (a fine-tune run, or scope to auditing an existing aligned model pair).

### ⭐ Idea 3 — The knowledge–stance gap: "knows the local norm, imposes its own value"
**Claim to test:** The model's **cultural knowledge** features (what group X believes) vary by language/culture, while its **value-stance** features (what the model itself endorses) stay English-default-invariant. That gap *is* the mechanism of value imposition.
**Why novel:** [Language over Content](https://arxiv.org/abs/2510.16565) measures cultural *knowledge* paths; [GlobalOpinionQA](https://arxiv.org/abs/2306.16388) shows behaviorally that translation doesn't shift the answer. Nobody has **mechanistically separated knowledge from stance** and shown knowledge-variance coexisting with stance-invariance. This is a sharper, safety-relevant refinement of A4's own thesis and cleanly differentiates it from Language-over-Content.
**Method:** Contrastive descriptive ("In Japan, people tend to…") vs. prescriptive ("The right thing to do is…") prompts; select knowledge-features vs. stance-features by attribution from the respective answer tokens; test each set's cross-lingual invariance separately.
**Safety relevance:** Value imposition + eval validity; quantifies "the model knows better but answers Western anyway."
**Feasibility:** High. Direct extension of the A4 pipeline.

### Idea 4 — The alignment fingerprint: is there one English-anchored "aligned" direction all languages inherit?
**Claim to test:** RLHF/DPO installs a largely **single, English-anchored "assistant/aligned" direction**; non-English queries inherit it (alignment = "respond as an aligned English assistant, re-expressed"). If so, that is the mechanistic basis of English-only-alignment false assurance.
**Why novel:** [Ryan et al.](https://arxiv.org/abs/2402.15018) show the *behavioral* degradation; the *representational* cause (one shared direction vs. per-language) is untested.
**Method:** Base vs. instruction-tuned Gemma pair; isolate the alignment-induced representational delta; test whether it's one shared direction or per-language; ablate/patch to confirm causal role in non-English behavior.
**Safety relevance:** Directly mechanizes the primary safety thesis of the A4 file.
**Feasibility:** Medium (needs a clean base/aligned pair; Gemma base vs. -it works).

### Idea 5 — Scaling curve: does scale fix or *entrench* the English default?
**Claim to test:** Value-feature cross-lingual invariance (and the English-pivot's strength) changes monotonically across 2B→9B→27B — and may get *worse* (homogenization), contradicting the optimistic "scale makes concepts language-invariant" reading.
**Why novel:** The Alignment-Forum evidence says concepts get more language-invariant with scale, but *elicited values* don't. No one has plotted a **scaling curve for value/safety-feature invariance**.
**Method:** Run A4's invariance metric across the Gemma-2 ladder (all covered by Gemma Scope), for both a value construct and a safety construct.
**Safety relevance:** Predicts whether the problem self-corrects with scale or compounds — a forecasting result for frontier models.
**Feasibility:** High. Gemma Scope covers 2B/9B/27B.

### Idea 6 — Triple dissociation: script vs. language vs. culture
**Claim to test:** Which axis actually gates value/safety features — the *script*, the *language*, or the *culture*?
**Why novel:** [Encode Script over Linguistic Structure](https://arxiv.org/abs/2604.05090) and [One Language, Two Scripts](https://arxiv.org/abs/2603.08869) show script matters; A4 only separates language from culture. The **full factorial** has not been run for values/safety.
**Method:** Stimuli that vary the axes independently — Serbian (Cyrillic vs Latin, same language/culture), Hindi vs Urdu (near-shared language, different script + culture), Japanese (kanji/kana/romaji). Measure feature invariance per axis.
**Safety relevance:** Tells you whether "safe in language L" claims are really "safe in script S" — a subtle eval-validity trap.
**Feasibility:** Medium (careful stimulus design).

### Idea 7 — A reusable "language-invariance certificate" for safety properties (method/infra)
**Claim to build:** A drop-in tool: given a probe/feature/direction for property P, output a **cross-lingual invariance score** with a matched non-P control, for any multilingual model.
**Why novel:** Fills a named infrastructure gap — there is no standard way to certify feature-level language-invariance of a safety/value property. This is A4's "method with independent value," generalized and packaged.
**Method:** Wrap the A4/Idea-1 pipeline; ship a small benchmark over a few properties (refusal, sycophancy, one WVS value) on Gemma + one other open multilingual model; release code + Neuronpedia links.
**Safety relevance:** Reusable auditing for any multilingual deployment; a concrete artifact reviewers value.
**Feasibility:** Medium-high (mostly engineering + packaging).

### Idea 8 — Affective inequity: does harm-sensitivity fire equally across languages?
**Claim to test:** Empathy/harm-sensitivity features (e.g., to self-harm or crisis disclosures) fire *less* in low-resource languages, producing differential care quality.
**Why novel:** The ICML pluralistic-alignment community flags "cultural affective inequity"; no feature-level cross-lingual study exists.
**Method:** Curated (ethically reviewed) crisis-style prompts across languages; measure activation of empathy/safety features and downstream response quality.
**Safety relevance:** Differential safety for vulnerable users in their own languages — a concrete, human-impact harm.
**Feasibility:** Medium (sensitive stimuli; needs care/ethics).

---

## What this means for the original A4 proposal

- **Still viable, but reposition.** A4 is no longer the first to "disentangle language from culture." Add explicit differentiation vs. [Language over Content](https://arxiv.org/abs/2510.16565) (uses circuit paths + cultural *knowledge*, no language-neuron ablation, no English-pivot, no WVS-scored *stance*), [Neuron-Level Analysis](https://arxiv.org/abs/2510.08284), [Entangled/Culturescope](https://arxiv.org/abs/2508.08879), and [Latent Activation Steering](https://arxiv.org/abs/2605.26365).
- **Surviving novelty of A4:** SAE *value* features chosen by **causal attribution from the value-answer token** (not MI-to-country correlation) + **language-neuron ablation** + **English-pivot/default-direction** test + **WVS-scored answer**. Keep those four as the headline; drift toward "find SAE cultural features + show Anglophone default" collapses into CuE/Culturescope.
- **De-risk one design choice:** [Language-specific Neurons Do Not Facilitate Cross-Lingual Transfer](https://arxiv.org/abs/2503.17456) warns LAPE-neuron ablation may not move behavior as hoped — pilot Move 2 early, and have Idea-3 (knowledge–stance) or Idea-1 (safety features) ready as the higher-upside pivot.
- **Highest-leverage pivot:** run the same pipeline on **safety behaviors** (Ideas 1–2), not just WVS values. Less crowded, higher stakes, same tools.

**Recommended next step:** run `/novelty-check` on **Idea 1** (recognition-vs-execution) and **Idea 3** (knowledge–stance) — the two that best combine open novelty with A4-toolkit feasibility — then `/brainstorm` to scope an MVP.

---

## All Sources

| Title | Type | Org / Authors | URL |
|---|---|---|---|
| Refusal Direction is Universal Across Safety-Aligned Languages | paper (NeurIPS 2025) | — | https://arxiv.org/abs/2505.17306 |
| The Multilingual Divide and Its Impact on Global AI Safety | agenda/report | Cohere Labs (Peppin, Kreutzer et al.) | https://arxiv.org/abs/2505.21344 |
| Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment | paper (ICML 2026) | DeepMind (Saakyan, Rastogi, Aroyo) | https://deepmind.google/research/publications/225819/ |
| Language over Content: Tracing Cultural Understanding in Multilingual LLMs | paper (mech-interp) | Cho, Ko, Hwang, Lee, Lee, Park | https://arxiv.org/abs/2510.16565 |
| Neuron-Level Analysis of Cultural Understanding in LLMs | paper (ICLR 2026) | Yamamoto, Kumon, Bollegala, Yanaka | https://arxiv.org/abs/2510.08284 |
| Entangled in Representations (Culturescope) | paper (mech-interp) | Yu, Jeong, Pawar, Shin, Jin, Myung, Oh, Augenstein | https://arxiv.org/abs/2508.08879 |
| Isolating Culture Neurons in Multilingual LLMs (CAPE) | paper | — | https://arxiv.org/abs/2508.02241 |
| Language-Specific Neurons (LAPE) | paper | — | https://arxiv.org/abs/2402.16438 |
| Do Llamas Work in English? | paper | Wendler et al. | https://arxiv.org/abs/2402.10588 |
| Separating Tongue from Thought (activation patching) | paper | Dumas, Wendler, Veselovsky, Monea, West (EPFL) | https://arxiv.org/abs/2411.08745 |
| Exploring Multilingual Concepts of Human Value | paper | Xu, Dong, Guo, Wu, Xiong (Tianjin) | https://arxiv.org/abs/2402.18120 |
| Tracing Multilingual Representations with Cross-Layer Transcoders | paper | Harrasse, Draye, Pandey, Jin, Schölkopf | https://arxiv.org/abs/2511.10840 |
| The Transfer Neurons Hypothesis | paper | — | https://arxiv.org/abs/2509.17030 |
| Language-specific Neurons Do Not Facilitate Cross-Lingual Transfer | paper | — | https://arxiv.org/abs/2503.17456 |
| SAEs Can Capture Language-Specific Concepts Across Diverse Languages | paper | — | https://arxiv.org/abs/2507.11230 |
| Multilingual Steering by Design (multilingual SAEs) | paper | — | https://arxiv.org/abs/2605.23036 |
| Multilingual LMs Encode Script Over Linguistic Structure | paper | — | https://arxiv.org/abs/2604.05090 |
| One Language, Two Scripts: Script-Invariance | paper | — | https://arxiv.org/abs/2603.08869 |
| Gemma Scope (SAE suite) | tool/paper | Google DeepMind | https://arxiv.org/abs/2408.05147 |
| Cultural Value Alignment via Latent Activation Steering | paper (ACL 2026 SRW) | Dang, Masud | https://arxiv.org/abs/2605.26365 |
| Rethinking Cross-lingual Alignment (cultural erasure / Surgical Steering) | paper | Han, Agrawal, Briakou | https://arxiv.org/abs/2510.26024 |
| CuE — Steering LLMs for Culturally Localized Generation | paper (SAE culture) | — | https://arxiv.org/abs/2603.23301 |
| Towards Measuring Subjective Global Opinions (GlobalOpinionQA) | paper+benchmark | Anthropic (Durmus et al.) | https://arxiv.org/abs/2306.16388 |
| Cultural Bias and Cultural Alignment of LLMs | paper (PNAS Nexus) | Tao, Viberg, Baker, Kizilcec | https://arxiv.org/abs/2311.14096 |
| Multilingual != Multicultural | paper | Rystrøm, Kirk, Hale (Oxford) | https://arxiv.org/abs/2502.16534 |
| One Model, Many Morals | paper | Farid et al. (Michigan) | https://arxiv.org/abs/2509.21443 |
| Rethinking AI Cultural Alignment (elicitation sensitivity) | position | Bravansky, Trhlik, Barez | https://arxiv.org/abs/2501.07751 |
| Hire Your Anthropologist! (culture-benchmark critique) | paper | — | https://arxiv.org/abs/2510.05931 |
| BLEnD (everyday cultural knowledge) | benchmark (NeurIPS 2024) | — | https://arxiv.org/abs/2406.09948 |
| CIVICS (culturally-informed values) | dataset | — | https://arxiv.org/abs/2405.13974 |
| Global MMLU | benchmark | Cohere Labs et al. | https://arxiv.org/abs/2412.03304 |
| Camellia (cultural bias, Asian languages) | benchmark | — | https://arxiv.org/abs/2510.05291 |
| LASA (language-agnostic semantic alignment, safety) | paper | — | https://arxiv.org/abs/2604.12710 |
| Enforcing Multilingual Consistency for LLM Safety Alignment | paper | — | https://arxiv.org/abs/2602.16660 |
| Multilingual Safety Alignment via Self-Distillation | paper | — | https://arxiv.org/abs/2605.02971 |
| Middle-Layer Representation Alignment for Cross-Lingual Transfer | paper (ACL 2025) | — | https://arxiv.org/abs/2502.14830 |
| Sycophancy as a Multilingual Alignment Failure | paper | — | https://arxiv.org/abs/2606.08451 |
| Towards Safe Multilingual Frontier AI (EU languages) | report | Kanepajs, Ivanov, Moulange | https://arxiv.org/abs/2409.13708 |
| Steering LM Refusal with SAEs | paper | — | https://arxiv.org/abs/2411.11296 |
| Safe-SAIL (fine-grained safety via SAE) | paper | — | https://arxiv.org/abs/2509.18127 |
| BioRefusalAudit (SAE refusal-depth audit) | paper | — | https://arxiv.org/abs/2605.30162 |
| Unintended Impacts of LLM Alignment on Global Representation | paper (ACL 2024) | Ryan, Held, Yang (Stanford) | https://arxiv.org/abs/2402.15018 |
| The Homogenizing Effect of LLMs | position | Sourati, Ziabari, Dehghani (USC) | https://arxiv.org/abs/2508.01491 |
| Artificial Hivemind | paper | — | https://arxiv.org/abs/2510.22954 |
| A Roadmap to Pluralistic Alignment | agenda (ICML 2024) | Sorensen et al. | https://arxiv.org/abs/2402.05070 |
| Operationalizing Pluralistic Values | paper | — | https://arxiv.org/abs/2511.14476 |
| Mechanistic Interpretability for LLM Alignment (survey) | survey | Naseem | https://arxiv.org/abs/2602.11180 |
| Tracing the Thoughts of a Large Language Model | blog (Anthropic) | Jermyn et al. | https://www.lesswrong.com/posts/zsr4rWRASxwmgXfmq/tracing-the-thoughts-of-a-large-language-model |
| Testing the Authoritarian Bias of LLMs | LW post | Jin, Strauss, Guzman Piedrahita, Samway | https://www.lesswrong.com/posts/s6JntqWQrNj5Gygzx/testing-the-authoritarian-bias-of-llms |
| PlurVA-LLM @ AACL 2026 (CfP) | workshop | — | https://www.aclweb.org/portal/content/first-cfp-first-workshop-pluralistic-value-alignment-llms-plurva-llm-aacl-2026 |

> Note: several IDs are 2026 preprints (beyond the training cutoff) verified via live search titles; a handful surfaced only as arXiv IDs during salvage and were excluded here unless a title was confirmed.
