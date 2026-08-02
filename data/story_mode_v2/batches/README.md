Per-author batch files, 40 pairs each. Batches 01-20 are `train`, 21-25 are `heldout`.
Merged into `../pairs_1k.jsonl` / `../pairs_1k_heldout.jsonl` by `../merge_batches.py`.
Each batch is written against a disjoint topic domain, which is what makes context
collisions structurally impossible. See `../AUTHORING_SPEC.md`.
