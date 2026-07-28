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

What "the direction is useful" looks like:
    - test AUROC near 1.0 at some middle layer  -> fiction/real is linearly encoded
    - |cos(fiction, length)| near 0             -> not a length artifact
    - both Tier-1 and Tier-2 pairs separate      -> not a lexical/register artifact
      (inspect per-tier by filtering the jsonl; see --tier note below)
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
train = [p for p in pairs if p["split"] == "train"]
test = [p for p in pairs if p["split"] == "test"]
lc = load("length_control_pairs.jsonl")

print(f"train={len(train)} pairs  test={len(test)} pairs  length-control={len(lc)} pairs")
print("extracting activations ...")
fic_tr, real_tr = reps_for(train, "fiction"), reps_for(train, "real")
fic_te, real_te = reps_for(test, "fiction"), reps_for(test, "real")
long_lc, short_lc = reps_for(lc, "long"), reps_for(lc, "short")

fic_dir = fic_tr.mean(0) - real_tr.mean(0)      # [n_layers+1, hidden]
len_dir = long_lc.mean(0) - short_lc.mean(0)

print(f"\n{'layer':>5} {'test_AUROC':>11} {'cos(fic,len)':>13}")
rows = []
n_layers = fic_dir.shape[0]
for l in range(1, n_layers):
    d = fic_dir[l] / (fic_dir[l].norm() + 1e-8)
    pos = (fic_te[:, l, :] @ d).tolist()
    neg = (real_te[:, l, :] @ d).tolist()
    a, c = auroc(pos, neg), cos(fic_dir[l], len_dir[l])
    rows.append((l, a, c))
    print(f"{l:>5} {a:>11.3f} {c:>13.3f}")

# --- save results (namespaced per model so multiple runs don't overwrite) ---
out_dir = os.path.join(HERE, "results", MODEL.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["layer", "test_auroc", "cos_fiction_length"])
    w.writerows(rows)
# direction vectors are the reusable artifact for Phase 2/3 steering & probes
dir_path = os.path.join(out_dir, "directions.pt")
torch.save({"model": MODEL, "fiction": fic_dir, "length": len_dir}, dir_path)
best = max(rows, key=lambda r: r[1])
print(f"\nbest layer: {best[0]} (test AUROC {best[1]:.3f}, cos-with-length {best[2]:.3f})")
print(f"saved metrics       -> {csv_path}")
print(f"saved directions.pt -> {dir_path}")
