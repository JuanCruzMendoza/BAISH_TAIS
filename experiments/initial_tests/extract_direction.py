"""
Pilot test: does a linear "fiction vs real" direction exist and generalize?

Extracts a diff-in-means direction from the TRAIN pairs (fiction - real) at every
layer, then reports, on the held-out TEST pairs, how well projection onto that
direction separates fiction from real (AUROC). Also reports cosine similarity
against a length direction (long - short) built from neutral prompts -- the
confound falsifier: if the fiction direction is really just encoding length,
this cosine will be large.

Usage:
    pip install torch transformers
    python extract_direction.py [model_name]
        default model: Qwen/Qwen2.5-7B-Instruct  (plan sec.3)

Run on a GPU box (e.g. RunPod). Reads .jsonl from data/initial_tests/ (override
with the DATA_DIR env var); writes results to experiments/initial_tests/results/.

What "the direction is useful" looks like (per-layer table + metrics.csv):
    - auroc_pooled near 1.0 at some middle layer  -> fiction/real is linearly encoded
    - |cos_fiction_length| near 0                 -> not a length artifact
    - auroc_t1_to_t2 and auroc_t2_to_t1 both high -> a shared fiction/real axis that
      transfers across lexical realizations, not a "novel/memoir" word detector
    - cos_t1_t2 near 1                            -> both tiers induce the same axis
"""
import csv
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo layout: this script is in experiments/initial_tests/, the .jsonl data in
# data/initial_tests/ (siblings under the repo root). Override with $DATA_DIR.
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(HERE, "..", "..", "data", "initial_tests")
)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"


def load(path):
    with open(os.path.join(DATA_DIR, path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


print(f"loading {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
dev = model.device


@torch.no_grad()
def last_token_reps(text):
    """Return [n_layers+1, hidden] hidden states at the final prompt token,
    at the position where the model is about to generate its response."""
    msgs = [{"role": "user", "content": text}]
    try:
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception:
        enc = tok(text, return_tensors="pt")
    enc = enc.to(dev)
    out = model(**enc, output_hidden_states=True)
    return torch.stack(out.hidden_states, dim=0)[:, 0, -1, :].float().cpu()


def reps_for(items, key):
    return torch.stack([last_token_reps(it[key]) for it in items], dim=0)


def auroc(pos, neg):
    """Mann-Whitney U estimate of P(pos > neg)."""
    tot = len(pos) * len(neg)
    c = 0.0
    for a in pos:
        for b in neg:
            c += 1.0 if a > b else (0.5 if a == b else 0.0)
    return c / tot


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


pairs = load("fiction_vs_real_pairs.jsonl")
lc = load("length_control_pairs.jsonl")

n_t1 = sum(p["tier"] == 1 for p in pairs)
n_t2 = sum(p["tier"] == 2 for p in pairs)
print(f"{len(pairs)} fiction/real pairs (Tier-1={n_t1}, Tier-2={n_t2}); {len(lc)} length pairs")
print("extracting activations (one forward pass per prompt) ...")

# reps for every pair once; slice by tier/split with index lists (no recompute)
fic_all = reps_for(pairs, "fiction")     # [N, n_layers+1, hidden]
real_all = reps_for(pairs, "real")
long_lc, short_lc = reps_for(lc, "long"), reps_for(lc, "short")

sel = lambda cond: [i for i, p in enumerate(pairs) if cond(p)]
tr = sel(lambda p: p["split"] == "train")                    # pooled train (both tiers)
te = sel(lambda p: p["split"] == "test")                     # pooled test  (both tiers)
te_t1 = sel(lambda p: p["split"] == "test" and p["tier"] == 1)
te_t2 = sel(lambda p: p["split"] == "test" and p["tier"] == 2)
tr_t1 = sel(lambda p: p["split"] == "train" and p["tier"] == 1)
tr_t2 = sel(lambda p: p["split"] == "train" and p["tier"] == 2)
all_t1 = sel(lambda p: p["tier"] == 1)                       # every Tier-1 pair
all_t2 = sel(lambda p: p["tier"] == 2)                       # every Tier-2 pair


def direction(idx):
    """diff-in-means fiction - real over the given pairs -> [n_layers+1, hidden]"""
    return fic_all[idx].mean(0) - real_all[idx].mean(0)


def auroc_at(dvec_l, idx, l):
    """AUROC separating fiction from real over `idx` at layer l, along direction dvec_l."""
    u = dvec_l / (dvec_l.norm() + 1e-8)
    pos = (fic_all[idx][:, l, :] @ u).tolist()
    neg = (real_all[idx][:, l, :] @ u).tolist()
    return auroc(pos, neg)


dir_pool = direction(tr)      # trained on both tiers (the original direction)
dir_t1 = direction(tr_t1)     # trained on Tier-1 only
dir_t2 = direction(tr_t2)     # trained on Tier-2 only
len_dir = long_lc.mean(0) - short_lc.mean(0)

# Column legend:
#   pooled : pooled dir on held-out test, both tiers      -- reproduces the original number
#   T1 / T2: pooled dir on the Tier-1 / Tier-2 test subset  (only 3+3 pairs -> coarse)
#   T1>T2  : dir from Tier-1 train, tested on ALL Tier-2     (10 pairs, never seen)
#   T2>T1  : dir from Tier-2 train, tested on ALL Tier-1     (10 pairs, never seen)
#   cosT12 : cosine between the Tier-1 and Tier-2 directions (does the axis agree?)
# The transfer columns + cosT12 are the real "concept, not lexical artifact" evidence:
# a Tier-1-only direction has never seen a full rewrite, so if it still separates all
# Tier-2 pairs the axis is shared across surface form, not the word "novel".
print(f"\n{'lyr':>3} {'pooled':>6} {'T1':>6} {'T2':>6} {'T1>T2':>6} {'T2>T1':>6} {'cosT12':>7} {'cosLen':>7}")
rows = []
n_layers = dir_pool.shape[0]
for l in range(1, n_layers):
    r = (
        l,
        auroc_at(dir_pool[l], te, l),
        auroc_at(dir_pool[l], te_t1, l),
        auroc_at(dir_pool[l], te_t2, l),
        auroc_at(dir_t1[l], all_t2, l),
        auroc_at(dir_t2[l], all_t1, l),
        cos(dir_t1[l], dir_t2[l]),
        cos(dir_pool[l], len_dir[l]),
    )
    rows.append(r)
    print(f"{r[0]:>3} {r[1]:>6.2f} {r[2]:>6.2f} {r[3]:>6.2f} {r[4]:>6.2f} {r[5]:>6.2f} {r[6]:>7.2f} {r[7]:>7.2f}")

# --- save results (namespaced per model so multiple runs don't overwrite) ---
out_dir = os.path.join(HERE, "results", MODEL.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "layer", "auroc_pooled", "auroc_pool_on_t1", "auroc_pool_on_t2",
        "auroc_t1_to_t2", "auroc_t2_to_t1", "cos_t1_t2", "cos_fiction_length",
    ])
    w.writerows(rows)
# direction vectors are the reusable artifact for Phase 2/3 steering & probes
dir_path = os.path.join(out_dir, "directions.pt")
torch.save(
    {"model": MODEL, "fiction_pooled": dir_pool, "fiction_tier1": dir_t1,
     "fiction_tier2": dir_t2, "length": len_dir},
    dir_path,
)
# best layer = strongest worst-case cross-tier transfer (i.e. most tier-invariant)
best = max(rows, key=lambda r: min(r[4], r[5]))
print(f"\nbest transfer layer: {best[0]}  T1>T2={best[4]:.2f}  T2>T1={best[5]:.2f}  "
      f"cosT12={best[6]:.2f}  cosLen={best[7]:.2f}")
print("note: T1/T2 cols are 3+3 pairs (coarse, ~0.11 steps); transfer cols are 10+10 (~0.01).")
print(f"saved metrics       -> {csv_path}")
print(f"saved directions.pt -> {dir_path}")
