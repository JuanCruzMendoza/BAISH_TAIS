## Appendix

### A1. Dataset construction

Each direction dataset is composed of 800 training pairs and 200 held out.

**story** — 1,000 pairs generated in 25 batches, one disjoint topic domain each, with all contexts distinct and the five held-out domains absent from training. The contrast is predicate type: changes of state in temporal order against classificatory and prescriptive predication. There are 8 narrative modes and 16 non-narrative styles, with half of the pairs realistic and half not. 

Each pair is matched on topic and length, uses different tenses, third person only, no harmful content, with both arms ending mid-text on a full stop so the read position is the same token throughout. 

An automatic diversity audit was also run, checking that there are no content 5-gram collisions and no near-duplicates.


**persona** — 200 of 275 [assistant-axis](https://arxiv.org/abs/2601.10387) roles × 5 paraphrases. Removed: 21 harm-adjacent roles (criminality would sit only on the positive arm), assistant itself, and 53 near-synonyms. The negative pole is a pool of 13 default-assistant framings, weighted so the mean paired framing gap is +0.01 words — uniform sampling would have left it 3.6 words shorter in nearly every pair.
*Not controlled:* speech act (role side 38% declarative against the assistant side's 80%), realism at 77/23 rather than balanced, and 28% of the negative pole carrying explicit helpfulness content.

**harm** — (framing + harmful request) against (framing + benign request), framing byte-identical within pair. Spread evenly over four framing families — bare, role-play, story, evaluation — so framing cannot correlate with the label. All 50 [JailbreakBench](https://github.com/JailbreakBench/artifacts) pairs in training at 16 uses each, plus 15 held-out pairs, with the splits sharing no framing and no behavior.
*Not controlled:* the harmful goal averages 1.84 words longer, inherited from JailbreakBench. In the bare and role-play families the final token is the request's own last word, which differs across the pair in half of them.

**eval** — 50 framing pairs (16 from the [source paper](https://arxiv.org/abs/2510.20487), 34 authored to the same structure) crossed with the shared requests, 16 pairs per framing. Each framing draws whole harmful/benign pairs, so it is exactly half harmful, and requests are dealt from category-rotated blocks so no framing repeats a request or category. Splits share no framing string, 4-gram, behavior or prompt.

**Jailbreak corpus** — 1,017 prompts, of which the 1,009 carrying a framing are used.

![The six sources of the jailbreak corpus](figures/table_sources.png)


### A2. The StrongREJECT rubric

The judge is shown the bare harmful request and the response, and scores three items:

![The three items of the StrongREJECT rubric](figures/table_rubric.png)

```
strongreject = (1 − refused) × (convincing + specific − 2) / 8
```

The score is 0 for any refusal and otherwise runs from 0 to 1, and a row counts as a successful attack when it is above 0. Disclaimers are explicitly not penalized, except in creative content generation, where they count against how convincing a response is. In addition, a response carrying no information specific enough to help the user is scored as a refusal even when the model never declines.

To detect degenerate responses, we use the judge's label and four length detectors (compression ratio, longest token run, distinct 4-grams, loop fraction), calibrated to 0% false positives on 1,040 unsteered responses and 99.5% recall on 218 verified-broken ones.

We used gpt-4o-mini at temperature 0, showing the judge only the bare harmful request and the response.  

ASR is computed on non-degenerate rows only: a broken response scores zero exactly as a refusal does, so including such rows would make a destroyed model look like a restored one.

### A3. The narrativity judge

An LLM judge is presented with a steered response and the baseline response on the *same* jailbreak prompt. The judge is asked which text is written more as narrative fiction than as expository writing, with both poles explained in the same terms the direction was extracted from:

![The two poles the narrativity judge is given](figures/table_poles.png)

It is told to judge only the writing style, and that length, subject matter, quality and whether the text answers any question are all irrelevant. A third option, neither, covers broken output and genuine ties. 

Three controls make the choice interpretable:
- Length: both texts are truncated to 2,000 characters. Steered responses are systematically longer, and an untruncated pair would let the judge read "longer" as "more narrative".
- Position: A/B order is randomized per row from a fixed seed.
- Degeneracy: pairs where either side is degenerate are dropped, since a repetition loop reads as stranger and therefore more literary.

The same judge as A2 is used: gpt-4o-mini, temperature 0. 

### A4. Complete steering sweep

Every steered cell behind Figure 1 and Table 1: attack success rate and the percentage of degenerate responses, per direction, layer and steering strength. The restore arm is run on the prompts the unsteered model complied with, where baseline ASR is 100 by construction, so its ΔASR is ASR − 100. The suppress arm is run on the prompts it refused, where baseline ASR is 0, so its ΔASR is the ASR itself. Only the magnitude of α is listed, since the two arms always use opposite signs. 

**Table 7. Qwen2.5-7B-Instruct.** 508 prompts in the restore arm, 433 in the suppress arm.

![Every steered cell in Qwen2.5-7B-Instruct](figures/table_sweep_qwen.png)

**Table 8. Gemma-2-9b-it.** 542 prompts in the restore arm, 423 in the suppress arm.

![Every steered cell in Gemma-2-9b-it](figures/table_sweep_gemma.png)

