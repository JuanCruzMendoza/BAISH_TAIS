# Deconfounding length in the Tier-3 (narrativity) direction

**Problem.** Tier-3 contrasts a request embedded in a short story against the same request stated
plainly. Every story item is ~60 tokens longer than its bare pair, so `story − bare` diff-in-means
absorbs a length effect. `cos_length` is expected large by construction — the question is whether
anything *else* is in there.

This is not a hypothetical nuisance: length is an explicitly linear feature of the residual stream
(models linearly encode their own remaining output length, 2607.05316), and contrastive probes are
known to track whatever feature is most *prominent* rather than the intended one (Farquhar et al.).

Two things must be kept separate:
- **cleaning the direction** — remove length from the vector we steer with;
- **auditing the readout** — show the separation was never length-driven in the first place.

---

## Methods

### Estimator-side (change how the vector is computed)

| Method | What it does | Cost |
|---|---|---|
| **Length-partialled diff-in-means** | Regress the paired differences on Δtokens; use the intercept | free, no new data |
| **Whitened removal (LEACE / mass-mean)** | Remove length via an *oblique* projection defined by Σ | needs shrinkage estimate of Σ |
| **Cluster-norm** | Normalize activations within token-count bins before extracting | cheap |
| **INLP / amnesic** | Iteratively null the length subspace, check narrativity survives | moderate |

**Length-partialled diff-in-means.** Pairs are content-matched, so each gives `Δh_i = h(story_i) − h(bare_i)`,
and Δtokens varies across pairs. Per layer, fit

```
Δh_i = a + b·Δtok_i + ε_i
```

`b` is the per-token drift (a length direction estimated in-distribution, not from a proxy set); `a` is
the story effect extrapolated to zero length delta. Plain diff-in-means is `mean(Δh) = a + b·mean(Δtok)`
— it silently carries the average length effect. Using `a` drops that term. Identified only because
Δtok varies; unstable at n=10, fine at n=32.

**Whitened removal.** Naive orthogonal projection `d − (d·û_len)û_len` **under-removes** length whenever
length and narrativity are correlated in activation space. LEACE proves that erasing a binary concept
requires neutralizing exactly its diff-in-means direction, but the minimal-damage projection is oblique,
defined w.r.t. the covariance. Same correction appears as the mass-mean probe: use `Σ⁻¹(μ₊ − μ₋)`,
explicitly to mitigate interference from non-orthogonal features. Estimate Σ with Ledoit-Wolf shrinkage
(d ≈ 3.5k, n ≈ tens).

Diff-in-means itself is worst-case optimal for editing (Belrose) — the estimator is fine, it just needs
conditioning.

**Cluster-norm** (Laurito et al., EMNLP 2024). Cluster the *pair averages* — these isolate the salient
non-contrastive feature while staying blind to the contrast — then normalize per cluster before
extraction. For us: cluster on token count. This is stratification promoted from the evaluation into
the extraction step.

### Design-side (for the 32-prompt scale-up)

- **Right-pad pairs to equal token length.** ActAdd's algorithm does this as step 1 (`pad_right_same_token_len`).
- **Single-token contrast.** CAA makes pairs byte-identical except the final answer letter → zero length
  delta by construction. Tier-1 is already this shape; Tier-3 is not.
- **2×2 + histogram length matching.** *Is Evaluation Awareness Just Format Sensitivity?* (2603.19426) is
  the closest paper to this problem — same rival direction, same confound. They cross context × format,
  match length by histogram alignment, and pool across formats so surface structure cannot predict the
  label. Their result: eval-awareness probes only generalize once format is decorrelated. Read before
  writing the 32 story prompts.
- **Fix the length-control set.** `length_control_pairs.jsonl` currently contrasts
  `"Define photosynthesis"` vs `"Give a detailed step-by-step description…"` — that is *requested
  verbosity*, not prompt length. The nuisance actually present in Tier-3 is "bare request preceded by
  ~60 tokens of prose." Rebuild with neutral non-narrative filler (scenery, weather; no characters, no
  agent), several filler styles → gives a length *subspace*, not one vector.

### Statistic-side (audit the readout, vector unchanged)

- **Length-conditioned AUROC.** Bucket by token count and compute AUROC within buckets — no length
  variance left to explain.
- **GLM control.** Regress `projection ~ tokens + is_story`, report the `is_story` coefficient. Precedent:
  Length-Controlled AlpacaEval, which frames length as a *mediator* and reports the counterfactual
  "what if both had equal length."
- **Selectivity.** Report `AUROC(story dir) − AUROC(length dir)` on the same eval, not AUROC alone
  (Hewitt & Liang's control-task logic, transposed to directions).
- **Reversed-length eval set** (~20 pairs, eval-only): padded-bare vs one-sentence-compressed-story. If
  the direction still orders them correctly when length runs the *other* way, length isn't the mechanism.

### Causal (the decisive test)

Promote length from a cosine check to a **named rival** in Phase 3 (§6) and Phase 4 (§7) of the plan:
run the length vector through the full pipeline at matched self-effect strength. If length steering does
not move ASR while narrativity does — and narrativity survives the estimator-side fixes — the confound is
dead, whatever the raw cosine says.

Note a high `cos_length` does not by itself mean "the vector is length", and a near-zero one does not
clear it either: length can be multi-dimensional, and non-orthogonal features interfere.

---

## Recommendation

1. Length-partialled diff-in-means (intercept) as the default estimator for Tier-3.
2. Whitened/LEACE length removal instead of orthogonal projection; report Δ AUROC and Δ steering before/after.
3. Cluster-norm on token-count bins at extraction.
4. Length-conditioned AUROC + GLM readout as the standing audit.
5. Length as a matched-strength rival in the steering dissociation matrix.
6. Rebuild the length-control set with filler-prose pairs; add the reversed-length eval set.

**Caveat to state in the writeup.** No purely post-hoc adjustment separates two features that are
perfectly collinear in the extraction set. (1) works only because Δtok varies across pairs; (2)–(6) work
because they bring in data where length and narrativity decorrelate. That data can be evaluation-only and
small — far cheaper than redesigning extraction.

---

## Sources

| Work | Relevance |
|---|---|
| [LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/abs/2306.03819) | erasure = neutralize diff-in-means, oblique projection |
| [The Geometry of Truth](https://arxiv.org/abs/2310.06824) | mass-mean probe, `Σ⁻¹` correction for non-orthogonal features |
| [Diff-in-Means Concept Editing is Worst-Case Optimal](https://blog.eleuther.ai/diff-in-means/) | why the estimator is worth keeping |
| [Cluster-norm for Unsupervised Probing of Knowledge](https://arxiv.org/abs/2407.18712) | normalize within confound clusters before extraction |
| [Challenges with unsupervised LLM knowledge discovery](https://arxiv.org/abs/2312.10029) | probes track the most prominent feature, not the intended one |
| [Null It Out (INLP)](https://arxiv.org/abs/2004.07667) / [Amnesic Probing](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00359/98091/Amnesic-Probing-Behavioral-Explanation-with) | iterative nulling + behavioral check |
| [Improving Causal Interventions in Amnesic Probing](https://arxiv.org/pdf/2506.11673) | LEACE/mean-projection > INLP for the causal version |
| [Steering Language Models with Activation Engineering (ActAdd)](https://arxiv.org/pdf/2308.10248) | `pad_right_same_token_len` |
| [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) | single-token contrast design |
| [Is Evaluation Awareness Just Format Sensitivity?](https://arxiv.org/html/2603.19426) | 2×2 decorrelation + histogram length matching; closest prior |
| [Length-Controlled AlpacaEval](https://arxiv.org/abs/2404.04475) | GLM length control, length-as-mediator framing |
| [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/) | selectivity |
| [How Much is Left? LLMs Linearly Encode Their Remaining Output Length](https://arxiv.org/pdf/2607.05316) | length is strongly linearly encoded |
| [Analysing the Generalisation and Reliability of Steering Vectors](https://arxiv.org/abs/2407.12404) | spurious biases drive per-input steering effectiveness |
