
# Guidelines

- Target audience: AI Safety researchers

- Main claims:
	- I found a story mode direction which makes the model narrate but steering away from it in jailbreaks framed as fictional has little effect on the ASR, and steering towards it either
	- With the same methodology, the role-play/assistant, harmful/harmless and the eval-awareness had a big effect on ASR


Side claims:
- **`persona → story` behaved exactly as predicted**: `perp` −95.2 ≈ ref −94.6, `par` −4.4 ≈ null.
  Story is neither necessary nor sufficient for persona's effect, as the arithmetic said.


# Draft
![[plot__story_v2_1k_cohens_dz_train.png]]

![[plot_matrices_auroc.png]]


![[plot_matrices_cos_matched.png]]


![[plot_layer_curves__all_story_v2_1k.png]]
![[plot_layer_curves__all_persona_v2.png]]
## Steering
| family | n | ASR | success | refusal | neither |
|---|---|---|---|---|---|
| fiction_narrative | 472 | **75.8%** | 343 | 110 | 19 |
| hybrid | 153 | 45.8% | 64 | 73 | 16 |
| nonfiction_other | 78 | 44.9% | 34 | 40 | 4 |
| roleplay_persona | 306 | **23.9%** | 67 | 210 | 29 |
| **all** | **1,009** | **53.1%** | **508** | **433** | **68** |

- All tables of ASR and %deg per direction?
- Table ASR per jailbreak family