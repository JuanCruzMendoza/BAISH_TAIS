"""
Tier-3 pilot: the NARRATIVITY direction (request inside a short story vs bare request).

Tier-3 pairs contrast a request embedded in a short story ("story") against the
same request stated plainly ("bare"). That is the *narrativity* axis -- as opposed
to the *fictionality* axis of Tier-1/Tier-2 (fiction vs real, narrative form fixed).

We extract the Tier-3 diff-in-means direction (story - bare) from the Tier-3 train
pairs and, per layer, report:

    auroc_t3_heldout : Tier-3 dir on held-out Tier-3 test (story vs bare)
                       -> expected ~1.0 (trivially separable; story != bare request)
    auroc_on_t1      : Tier-3 dir applied to ALL Tier-1 pairs (fiction vs real)
    auroc_on_t2      : Tier-3 dir applied to ALL Tier-2 pairs (fiction vs real)
                       -> near 0.5 means narrativity does NOT read fiction/real,
                          i.e. it is a separate axis from fictionality
    cos_tier1        : cosine(Tier-3 narrativity dir, Tier-1 fiction/real dir)
    cos_length       : cosine(Tier-3 narrativity dir, length dir)
                       -> EXPECTED LARGE by construction: a story is long, a bare
                          request is short, so narrativity is confounded with length.
                          This number quantifies that confound.

Usage:
    python extract_direction_tier3.py [model_name]      # default Qwen/Qwen2.5-7B-Instruct
Reads .jsonl from data/initial_tests/ (override with $DATA_DIR); writes
experiments/initial_tests/results/<model>/tier3_metrics.csv.
"""
import csv
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
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


# --- data ---
t3 = load("tier3_story_vs_bare_pairs.jsonl")
fr = load("fiction_vs_real_pairs.jsonl")          # Tier-1 + Tier-2 (fiction/real)
lc = load("length_control_pairs.jsonl")

t3_tr = [p for p in t3 if p["split"] == "train"]
t3_te = [p for p in t3 if p["split"] == "test"]
t1 = [p for p in fr if p["tier"] == 1]            # all Tier-1 (never seen by Tier-3 dir)
t2 = [p for p in fr if p["tier"] == 2]            # all Tier-2

print(f"Tier-3: train={len(t3_tr)} test={len(t3_te)} | Tier-1={len(t1)} Tier-2={len(t2)} | length={len(lc)}")
print("extracting activations ...")
story_tr, bare_tr = reps_for(t3_tr, "story"), reps_for(t3_tr, "bare")
story_te, bare_te = reps_for(t3_te, "story"), reps_for(t3_te, "bare")
fic_t1, real_t1 = reps_for(t1, "fiction"), reps_for(t1, "real")
fic_t2, real_t2 = reps_for(t2, "fiction"), reps_for(t2, "real")
long_lc, short_lc = reps_for(lc, "long"), reps_for(lc, "short")

t3_dir = story_tr.mean(0) - bare_tr.mean(0)       # narrativity: story - bare
t1_dir = fic_t1.mean(0) - real_t1.mean(0)         # fictionality reference (all Tier-1)
len_dir = long_lc.mean(0) - short_lc.mean(0)


def auroc_at(dvec_l, pos_reps, neg_reps, l):
    """AUROC separating pos from neg at layer l, projecting onto direction dvec_l."""
    u = dvec_l / (dvec_l.norm() + 1e-8)
    pos = (pos_reps[:, l, :] @ u).tolist()
    neg = (neg_reps[:, l, :] @ u).tolist()
    return auroc(pos, neg)


# on_t1 / on_t2 use fiction as the positive class: ~0.5 => narrativity is blind to
# fiction/real (separate axis); far from 0.5 (either side) => the axes overlap.
print(f"\n{'lyr':>3} {'t3_held':>8} {'on_T1':>6} {'on_T2':>6} {'cosT1':>7} {'cosLen':>7}")
rows = []
n_layers = t3_dir.shape[0]
for l in range(1, n_layers):
    r = (
        l,
        auroc_at(t3_dir[l], story_te, bare_te, l),
        auroc_at(t3_dir[l], fic_t1, real_t1, l),
        auroc_at(t3_dir[l], fic_t2, real_t2, l),
        cos(t3_dir[l], t1_dir[l]),
        cos(t3_dir[l], len_dir[l]),
    )
    rows.append(r)
    print(f"{r[0]:>3} {r[1]:>8.2f} {r[2]:>6.2f} {r[3]:>6.2f} {r[4]:>7.2f} {r[5]:>7.2f}")

out_dir = os.path.join(HERE, "results", MODEL.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "tier3_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["layer", "auroc_t3_heldout", "auroc_on_t1", "auroc_on_t2", "cos_tier1", "cos_length"])
    w.writerows(rows)
dir_path = os.path.join(out_dir, "tier3_direction.pt")
torch.save(
    {"model": MODEL, "narrativity": t3_dir, "fictionality_t1": t1_dir, "length": len_dir},
    dir_path,
)
print(f"\nsaved metrics    -> {csv_path}")
print(f"saved direction  -> {dir_path}")
