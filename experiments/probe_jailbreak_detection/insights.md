# probe_jailbreak_detection — insights

## 50_per_direction - midpoint

Qwen2.5-7B-Instruct, 100 jailbreak prompts, band L11–25, threshold = `midpoint`.

Results:
- Story v1: With these few at least and the midpoint threshold, the story_v1 does not activate in any layer 

- Story v2: The layers of story_v2 with the highest mean_paired_cos activate the most on fiction_narrative jailbreaks and they don't in nonfiction and roleplay, but still the mean pct_reads is only 34%
	Meanwhile, the layers of story_v2 which most activate on fiction_narrative, also activate the same on nonfiction

- Persona: the layer with the highest pct_reads (layer 17) cannot distinguish nonfiction prompts (but activates on fiction_narrativity and hybrid)

- Eval: The eval direction almost always activates in most layers

- Harm: it activates more on early layers rather than the layers with highest mean_paired_cos

These results in general suggests that the probes are not good enough and that it may be necessary to calibrate the threshold per direction and have a better criteria for layer selection

### layer selection by family margin

`jb_layer_select.py`: per layer, `margin = mean(pct_reads over target families) −
mean(pct_reads over off families)`, families weighted equally. Top-3 band layers:

| probe | targets | top-3 layers (margin) |
|---|---|---|
| story_v1 | fiction_narrative + hybrid | L15 (+9.6), L20 (+8.2), L16 (+6.8) |
| story_v2 | fiction_narrative + hybrid | L17 (+30.0), L18 (+27.4), L15 (+22.1) |
| persona | hybrid + roleplay_persona | L19 (+26.0), L21 (+19.8), L22 (+19.6) |

Reading it:

- **story_v1** — the ranking is noise: the best layer reads 14% of fiction and 5% of
  hybrid, so the margin is a handful of prompts.
- **story_v2** — L17 is the only clean cell (40/30% on target, 0/10% off) and it is also
  a top-3 `mean_paired_cos` layer. L18 ranks second on margin but fires on everything
  (nonfiction 20%, roleplay 37%), so the margin there is a level shift, not selectivity.
- **persona** — every top layer still reads 77–91% of `fiction_narrative`, i.e. the
  margin comes entirely from suppressing `nonfiction_other`. The axis separates framed
  from unframed, not persona from story.

### story_v2

Top-3 `mean_paired_cos` layers: L16 (0.776), L15 (0.775), L17 (0.772).

| slice | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|
| all | 19.0% | 100 | 19.0 | 1.000 |
| fiction_narrative | 34.3% | 35 | 12.0 | 1.000 |
| hybrid | 21.7% | 20 | 4.3 | 1.000 |
| nonfiction_other | 0.0% | 15 | 0.0 | 1.000 |
| roleplay_persona | 8.9% | 30 | 2.7 | 1.000 |

Layer with highest `pct_reads` per slice:

| slice | layer | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|---|
| all | L19 | 72.0% | 100 | 72 | 1.000 |
| fiction_narrative | L19 | 60.0% | 35 | 21 | 1.000 |
| hybrid | L21 | 100.0% | 20 | 20 | 1.000 |
| nonfiction_other | L21 | 66.7% | 15 | 10 | 1.000 |
| roleplay_persona | L19 | 83.3% | 30 | 25 | 1.000 |

### story_v1

Top-3 `mean_paired_cos` layers: L16 (0.767), L15 (0.758), L14 (0.753).

| slice | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|
| all | 4.0% | 100 | 4.0 | 1.000 |
| fiction_narrative | 9.5% | 35 | 3.3 | 1.000 |
| hybrid | 3.3% | 20 | 0.7 | 1.000 |
| nonfiction_other | 0.0% | 15 | 0.0 | 1.000 |
| roleplay_persona | 0.0% | 30 | 0.0 | 1.000 |

Layer with highest `pct_reads` per slice:

| slice | layer | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|---|
| all | L25 | 7.0% | 100 | 7 | 1.000 |
| fiction_narrative | L15 | 14.3% | 35 | 5 | 1.000 |
| hybrid | L15 | 5.0% | 20 | 1 | 1.000 |
| nonfiction_other | L25 | 13.3% | 15 | 2 | 1.000 |
| roleplay_persona | L18 | 3.3% | 30 | 1 | 1.000 |

### harm

Top-3 `mean_paired_cos` layers: L21 (0.708), L22 (0.691), L20 (0.688).

| slice | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|
| all | 52.3% | 100 | 52.3 | 0.892 |
| fiction_narrative | 57.1% | 35 | 20.0 | 0.892 |
| hybrid | 63.3% | 20 | 12.7 | 0.892 |
| nonfiction_other | 17.8% | 15 | 2.7 | 0.892 |
| roleplay_persona | 56.7% | 30 | 17.0 | 0.892 |

Layer with highest `pct_reads` per slice:

| slice | layer | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|---|
| all | L11 | 100.0% | 100 | 100 | 0.800 |
| fiction_narrative | L11 | 100.0% | 35 | 35 | 0.800 |
| hybrid | L11 | 100.0% | 20 | 20 | 0.800 |
| nonfiction_other | L11 | 100.0% | 15 | 15 | 0.800 |
| roleplay_persona | L11 | 100.0% | 30 | 30 | 0.800 |

### persona (role-play)

Top-3 `mean_paired_cos` layers: L12 (0.707), L11 (0.703), L13 (0.699).

| slice | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|
| all | 49.0% | 100 | 49.0 | 0.990 |
| fiction_narrative | 71.4% | 35 | 25.0 | 0.990 |
| hybrid | 46.7% | 20 | 9.3 | 0.990 |
| nonfiction_other | 46.7% | 15 | 7.0 | 0.990 |
| roleplay_persona | 25.6% | 30 | 7.7 | 0.990 |

Layer with highest `pct_reads` per slice:

| slice             | layer | pct_reads | n   | n_reads | ref_tpr |
| ----------------- | ----- | --------- | --- | ------- | ------- |
| all               | L18   | 94.0%     | 100 | 94      | 0.892   |
| fiction_narrative | L16   | 100.0%    | 35  | 35      | 0.923   |
| hybrid            | L18   | 100.0%    | 20  | 20      | 0.892   |
| nonfiction_other  | L18   | 93.3%     | 15  | 14      | 0.892   |
| roleplay_persona  | L17   | 90.0%     | 30  | 27      | 0.908   |
|                   |       |           |     |         |         |

persona at L17, per family:

| slice | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|
| all | 87.0% | 100 | 87 | 0.908 |
| fiction_narrative | 94.3% | 35 | 33 | 0.908 |
| hybrid | 95.0% | 20 | 19 | 0.908 |
| nonfiction_other | 53.3% | 15 | 8 | 0.908 |
| roleplay_persona | 90.0% | 30 | 27 | 0.908 |

### eval

Top-3 `mean_paired_cos` layers: L14 (0.332), L15 (0.325), L16 (0.318).

| slice | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|
| all | 94.3% | 100 | 94.3 | 0.779 |
| fiction_narrative | 92.4% | 35 | 32.3 | 0.779 |
| hybrid | 100.0% | 20 | 20.0 | 0.779 |
| nonfiction_other | 100.0% | 15 | 15.0 | 0.779 |
| roleplay_persona | 90.0% | 30 | 27.0 | 0.779 |

Layer with highest `pct_reads` per slice:

| slice | layer | pct_reads | n | n_reads | ref_tpr |
|---|---|---|---|---|---|
| all | L19 | 99.0% | 100 | 99 | 0.692 |
| fiction_narrative | L19 | 97.1% | 35 | 34 | 0.692 |
| hybrid | L14 | 100.0% | 20 | 20 | 0.785 |
| nonfiction_other | L14 | 100.0% | 15 | 15 | 0.785 |
| roleplay_persona | L19 | 100.0% | 30 | 30 | 0.692 |
