# Jailbreak dataset

Runnable jailbreak prompts for the story-mode study, weighted towards fiction and
role-play framings. Every row in `jailbreaks.csv` / `jailbreaks.jsonl` is a complete
prompt that can be sent to a model as-is, so refusal and ASR can be measured directly.

## Composition

1,017 prompts. 92% carry a narrative or role-play framing; the remaining 8% are the
non-narrative contrast arm.

| source | n | fiction | hybrid | roleplay | nonfiction |
|---|---:|---:|---:|---:|---:|
| In-the-Wild | 400 | 86 | 105 | 209 | – |
| Jailbreak Mimicry | 300 | 300 | – | – | – |
| StrongREJECT | 144 | 24 | – | 48 | 72 |
| PAIR | 84 | 5 | 16 | 49 | 14 |
| DeepInception | 57 | 57 | – | – | – |
| JBC (AIM) | 32 | – | 32 | – | – |
| **total** | **1,017** | **472** | **153** | **306** | **86** |

Split 307 val / 710 test. 424 distinct `template_id`s, but only 22 are true reused
templates (StrongREJECT 15, DeepInception 6, JBC's `aim` 1) applied across 233 rows;
the rest — In-the-Wild, Jailbreak Mimicry, PAIR (784 rows) — are one-off prompts with
no literal reuse. Base tasks: 660 rows over 32 JBB
behaviors (17–23 uses each, all 10 JBB categories present), 357 rows over 339 distinct
AdvBench requests. Prompt length p50 1,012 / p90 3,619 / p99 8,886 chars.

`validate.py` re-runs the integrity, composition and coverage checks on the built file.

## Sources

| Source | Reference | What it contributes | Released form |
|---|---|---|---|
| In-the-Wild | Shen et al., *"Do Anything Now"*, CCS 2024, [2308.03825](https://arxiv.org/abs/2308.03825) | 1,405 community jailbreak wrappers, dominated by DAN-family personas | [verazuo/jailbreak_llms](https://github.com/verazuo/jailbreak_llms) CSV |
| Jailbreak Mimicry | Ntais, [2510.22085](https://arxiv.org/abs/2510.22085) | 729 LoRA-generated narrative jailbreaks — screenplay/scene scaffolds | [Kaggle](https://www.kaggle.com/datasets/pavlosntais/prompts/) (serves anonymously) |
| JailbreakBench artifacts — JBC | Chao et al., NeurIPS 2024 D&B | the AIM template applied to all 100 JBB behaviors, with per-model success labels | [JailbreakBench/artifacts](https://github.com/JailbreakBench/artifacts) |
| JailbreakBench artifacts — PAIR | Chao et al., NeurIPS 2024 D&B | LLM-generated attacks keyed to the same JBB behaviors, with success labels | same |
| DeepInception | Li et al., [2311.03191](https://arxiv.org/abs/2311.03191) | the canonical nested-fiction jailbreak (layer *i* creates layer *i+1*) | [tmlr-group/DeepInception](https://github.com/tmlr-group/DeepInception) |
| StrongREJECT | Souly et al., [2402.10260](https://arxiv.org/abs/2402.10260) | curated implementations of the well-known wrappers, plus the non-narrative contrast arm | [dsbowen/strong_reject](https://github.com/dsbowen/strong_reject) |

## Instantiation

Sources that ship instance prompts — PAIR, JBC, DeepInception, Jailbreak Mimicry — keep
them verbatim. Sources that ship reusable wrappers — In-the-Wild, StrongREJECT — are
instantiated with harmful goals, round-robin so goals stay balanced across wrappers.
StrongREJECT wrappers carry a `{forbidden_prompt}` slot; In-the-Wild wrappers mostly do
not, and the request is appended after the wrapper, which is the JailbreakHub protocol
(`has_slot` records which case applied).

Base tasks come from JBB (In-the-Wild, StrongREJECT, PAIR, JBC) or AdvBench
(DeepInception, Jailbreak Mimicry); `base_task_source` records which.

### Disjoint from the probe requests

**No prompt here uses a harmful request that the direction-extraction datasets use.** The
65 goals in `data/harm/harm_selected_pairs*.csv` (50 train + 15 heldout) are held back, so
the jailbreak set draws only on the remaining 32 JBB behaviors plus AdvBench.

The wrapper arms draw from the unfiltered JBB-100 rather than `jbb_pairs_filtered.csv`. 

## Framing labels

`family` is the axis the study cares about and uses the same four values everywhere:

| family | meaning |
|---|---|
| `fiction_narrative` | request sits inside a story, scene or simulated world; the model is author or narrator |
| `roleplay_persona` | the model is told to *become* a character with its own rules (DAN, evil confidant, …) |
| `hybrid` | both — the AIM pattern, a story in which a character is instructed |
| `nonfiction_other` | non-narrative attacks and the bare request; the contrast arm |


## Columns

| column | meaning |
|---|---|
| `id` | `ITW-`, `JM-`, `JBC-`, `PAIR-`, `DI-`, `SR-` prefix by source |
| `source`, `source_ref` | provenance |
| `family` | framing label, see above |
| `technique` | specific attack, e.g. `nested_fiction`, `aim`, `narrative_mimicry`, `refusal_suppression` |
| `base_task_source`, `jbb_index`, `category`, `request` | the harmful goal underneath the framing |
| `prompt` | the full runnable text |
| `target_model`, `jailbroken_ref` | for the JailbreakBench artifacts: the model attacked and that paper's success label, as JSON |
| `template_id`, `has_slot` | which wrapper produced the row, and whether the request was substituted into a slot or appended |
| `frame_note` | one-line description of the framing device |
| `split` | `val` (30%) / `test` (70%), stratified by source × family, seed 20260730 |

`jailbroken_ref` is that paper's label against *their* target model, not a label for the
models in this study — treat it as a prior on difficulty, not as ground truth.

