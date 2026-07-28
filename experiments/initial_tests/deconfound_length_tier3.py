"""
Length-deconfounding for the Tier-3 NARRATIVITY direction (story vs bare request).

Tier-3 pairs are confounded by construction: the story member is ~5x longer than
the bare member, so `mean(story) - mean(bare)` may be a prompt-length vector wearing
a narrativity hat. This script keeps the Tier-3 dataset and the diff-in-means family
untouched and attacks the confound three ways:

  FIX 1 -- condition the estimator on length (OLS intercept).
    Per layer, regress the PAIRED difference on the pair's token delta:
        dh_i = a + b * dtok_i + eps_i        (dh_i in R^hidden, one row per pair)
    `a` is the story-bare contrast extrapolated to zero length delta; `b` IS the
    length axis, estimated in-distribution rather than from a proxy set. Same data,
    same contrast, same estimator family -- just partialled on length.
    CAVEAT: `a` is an out-of-range extrapolation (every observed dtok is >> 0). It is
    only meaningful if the length effect is ~linear in dtok over [0, max], and it is
    noisy at small n. `loo_cos` reports leave-one-out stability -- read it before
    trusting `ols_*`; < ~0.9 means n is too small for this fix to be worth anything.

  FIX 2 -- project out a length SUBSPACE built from matched non-narrative padding.
    length_filler_pairs.jsonl pads the SAME requests to story length with prose that
    has no characters, no events and no narrative time, in three styles:
        expository : topic-matched background facts   (controls topical elaboration too)
        ambient    : topic-neutral static description (pure padding)
        oblique    : document framing ending in "... reads:"
                     (matches the story's speech act, not just its length)
    One diff-in-means vector per style spans a k=3 length/verbosity subspace L;
    d_orth = d_raw - P_L d_raw. `len_frac` = ||P_L d_raw|| / ||d_raw|| quantifies how
    much of the raw direction lives in L. Note this deliberately replaces
    length_control_pairs.jsonl, which contrasts a terse question against a request
    for a *verbose answer* -- that measures requested-verbosity, not prompt length,
    and is the wrong nuisance for Tier-3.

  FIX 3 -- score the readout with length held fixed, same direction, no refit.
    Pool all four groups (story, bare, filler-long, filler-short), project onto the
    direction, and regress the scalar:
        score ~ 1 + n_tokens + is_story    -> report t on is_story
    The filler longs are what identify this: without them n_tokens predicts is_story
    almost perfectly and the coefficient is not separable (see `corr(tok, is_story)`
    printed at startup, Tier-3-only vs pooled). Also reports within-token-bin AUROC.

The headline numbers are the *length-matched* AUROCs (`*_M`): story vs filler-long,
both classes long. `raw_M_obl` is the sharpest single test -- length, topic and
speech act all matched, only narrativity differs.

Reading the table:
    raw_ho ~ 1.0 AND len_ho ~ 1.0   -> the naive Tier-3 eval is fully explainable by
                                       length; the confound is real, not hypothetical
    raw_M / ols_M / ort_M >> 0.5    -> narrativity survives length matching
    ols_M / ort_M collapse to ~0.5  -> the Tier-3 vector was mostly length
    *_M at or BELOW ~0.5            -> also "no narrativity", not an inverted signal.
                                       The fillers run slightly longer than the stories,
                                       so a pure-length vector lands near 0.0 on the
                                       matched eval; that residual gap biases *_M
                                       against narrativity, i.e. it is conservative.
    ols_M / ort_M >> raw_M          -> expected when narrativity is WEAK: length
                                       dominates the raw vector and masks it. Do not
                                       read raw_M as the ceiling for the other two.
    len_frac large                  -> lots of the raw vector sits in the length subspace
                                       (large len_frac + surviving ort_M is fine: the
                                       vector was contaminated, the residual is real)
    k (= len_rank) 1 not 3          -> the three filler styles induce ONE length axis;
                                       the extra basis rows were noise and were dropped
    t_story large with tokens in the model -> readout is not a length readout

Usage:
    pip install torch transformers
    python deconfound_length_tier3.py [model_name]     # default Qwen/Qwen2.5-7B-Instruct

Reads .jsonl from data/initial_tests/ (override with $DATA_DIR); writes
experiments/initial_tests/results/<model>/tier3_deconfounded_{metrics.csv,directions.pt}.

NOT fixed here: narrativity is still entangled with the *presence of a fictional
frame* (a story implies invented events). That is the Tier-1 fictionality axis and is
a separate contrast, not a length control.
"""
import csv
import json
import math
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(HERE, "..", "..", "data", "initial_tests")
)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
STYLES = ["expository", "ambient", "oblique"]
SHARPEST = "oblique"  # length + topic + speech act matched to the story


def load(path):
    with open(os.path.join(DATA_DIR, path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- model / activations
print(f"loading {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
dev = model.device


@torch.no_grad()
def last_token_reps(text):
    """([n_layers+1, hidden] at the final prompt token, prompt length in tokens).

    The token count comes from the same encoding path as the activations, so
    `dtok` below is the length the model actually saw (chat template included).
    """
    msgs = [{"role": "user", "content": text}]
    try:
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception:
        enc = tok(text, return_tensors="pt")
    enc = enc.to(dev)
    out = model(**enc, output_hidden_states=True)
    reps = torch.stack(out.hidden_states, dim=0)[:, 0, -1, :].float().cpu()
    return reps, int(enc["input_ids"].shape[-1])


def reps_for(items, key):
    """-> ([N, n_layers+1, hidden], LongTensor[N] of token counts)"""
    out = [last_token_reps(it[key]) for it in items]
    return torch.stack([r for r, _ in out], 0), torch.tensor(
        [n for _, n in out], dtype=torch.float32
    )


# ---------------------------------------------------------------- small stats helpers
def auroc(pos, neg):
    """Mann-Whitney U estimate of P(pos > neg)."""
    tot = len(pos) * len(neg)
    if tot == 0:
        return float("nan")
    c = 0.0
    for a in pos:
        for b in neg:
            c += 1.0 if a > b else (0.5 if a == b else 0.0)
    return c / tot


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def pearson(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    return float((xc @ yc) / (xc.norm() * yc.norm() + 1e-8))


def proj(reps, dvec_l, l):
    """Scalar readout: project layer-l reps onto the unit direction."""
    u = dvec_l / (dvec_l.norm() + 1e-8)
    return reps[:, l, :] @ u


def auroc_at(dvec_l, pos_reps, neg_reps, l):
    return auroc(proj(pos_reps, dvec_l, l).tolist(), proj(neg_reps, dvec_l, l).tolist())


# ---------------------------------------------------------------- FIX 1: OLS intercept
def ols_paired(dh, dtok):
    """Per-layer OLS of paired differences on token delta.

    dh: [n, n_layers+1, hidden] story-minus-bare differences; dtok: [n] token deltas.
    Returns (intercept, slope), each [n_layers+1, hidden]. Closed form, vectorised
    over layers and hidden units; intercept == plain diff-in-means iff slope == 0.
    """
    xc = dtok - dtok.mean()
    denom = (xc * xc).sum()
    if float(denom) < 1e-8:
        raise ValueError("token deltas have no variance -> length cannot be partialled out")
    slope = torch.einsum("n,nld->ld", xc, dh - dh.mean(0)) / denom
    intercept = dh.mean(0) - slope * dtok.mean()
    return intercept, slope


def loo_intercept_cos(dh, dtok, full_intercept):
    """Leave-one-out stability of the OLS intercept: mean cos(LOO fit, full fit) per layer.

    The intercept extrapolates outside the observed dtok range, so it is the fragile
    part of FIX 1. Low values here mean the fix needs more pairs, not that the
    narrativity signal is absent.
    """
    n = dh.shape[0]
    acc = torch.zeros(full_intercept.shape[0])
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        a_i, _ = ols_paired(dh[keep], dtok[keep])
        for l in range(full_intercept.shape[0]):
            acc[l] += cos(a_i[l], full_intercept[l])
    return acc / n


# ---------------------------------------------------------- FIX 2: subspace projection
def project_out(d, basis, rank_tol=0.05):
    """Remove span(basis) from d. basis: [k, hidden] rows.

    Truncated to the numerically meaningful rank: if every filler style induces the
    same length axis the basis rows are near-collinear, and a plain QR would still
    return k orthonormal columns whose 2nd and 3rd are fixed by sampling noise --
    projecting those out deletes real signal for nothing and inflates removed_frac.
    Keeps only singular directions above rank_tol * largest.
    -> (residual, removed_frac, effective_rank)
    """
    u, s, _ = torch.linalg.svd(basis.T, full_matrices=False)   # u: [hidden, k]
    q = u[:, s > rank_tol * s[0]]
    comp = q @ (q.T @ d)
    return d - comp, float(comp.norm() / (d.norm() + 1e-8)), int(q.shape[1])


# ------------------------------------------------- FIX 3: length-controlled regression
def partial_t_is_story(scores, toks, is_story):
    """t-statistic on is_story in  score ~ 1 + n_tokens + is_story.

    Scale-free in `scores`, so comparable across directions with different norms.
    pinv (not inverse) because tokens and is_story are still correlated; check the
    printed corr(tok, is_story) before reading this -- near +-1 means unidentified.
    """
    n = scores.shape[0]
    x = torch.stack([torch.ones(n), toks - toks.mean(), is_story], dim=1)
    xtx_inv = torch.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ scores)
    resid = scores - x @ beta
    dof = n - x.shape[1]
    s2 = float((resid * resid).sum()) / dof
    se = math.sqrt(max(s2 * float(xtx_inv[2, 2]), 1e-30))
    return float(beta[2]) / se


def binned_auroc(scores, toks, is_story, n_bins=4):
    """AUROC computed only WITHIN token-count bins, pooled by pair count.

    nan if no bin holds both classes -- which is exactly what happens on Tier-3
    alone (stories are long, bare requests are short, zero overlap). The filler
    longs are what give the long bins a negative class.
    """
    edges = torch.quantile(toks, torch.linspace(0, 1, n_bins + 1))
    acc, tot = 0.0, 0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (toks >= lo) & ((toks <= hi) if i == n_bins - 1 else (toks < hi))
        pos = scores[m & (is_story == 1)].tolist()
        neg = scores[m & (is_story == 0)].tolist()
        if not pos or not neg:
            continue
        w = len(pos) * len(neg)
        acc += w * auroc(pos, neg)
        tot += w
    return acc / tot if tot else float("nan")


# ---------------------------------------------------------------------------- data
t3 = load("tier3_story_vs_bare_pairs.jsonl")
lf = load("length_filler_pairs.jsonl")

t3_tr = [p for p in t3 if p["split"] == "train"]
t3_te = [p for p in t3 if p["split"] == "test"]
lf_by_style = {s: [r for r in lf if r["style"] == s] for s in STYLES}
missing = [s for s in STYLES if not lf_by_style[s]]
if missing:
    raise SystemExit(f"length_filler_pairs.jsonl has no rows for styles: {missing}")

print(f"Tier-3: train={len(t3_tr)} test={len(t3_te)} | "
      f"fillers: " + ", ".join(f"{s}={len(lf_by_style[s])}" for s in STYLES))
print("extracting activations (one forward pass per prompt) ...")

story_tr, tok_story_tr = reps_for(t3_tr, "story")
bare_tr, tok_bare_tr = reps_for(t3_tr, "bare")
story_te, tok_story_te = reps_for(t3_te, "story")
bare_te, tok_bare_te = reps_for(t3_te, "bare")

# filler reps per style, split the same way as Tier-3 so the length subspace is
# fitted on train requests and the matched eval negatives come from test requests
fl = {}   # style -> dict(long/short reps + token counts, train/test index lists)
for s in STYLES:
    rows = lf_by_style[s]
    lr, lt = reps_for(rows, "long")
    sr, st = reps_for(rows, "short")
    fl[s] = {
        "long": lr, "short": sr, "tok_long": lt, "tok_short": st,
        "tr": [i for i, r in enumerate(rows) if r["split"] == "train"],
        "te": [i for i, r in enumerate(rows) if r["split"] == "test"],
    }

# ------------------------------------------------------------- length-match diagnostic
dtok_tr = tok_story_tr - tok_bare_tr
dtok_te = tok_story_te - tok_bare_te
print(f"\nTier-3 dtok (story - bare): mean={dtok_tr.mean():.0f} "
      f"min={dtok_tr.min():.0f} max={dtok_tr.max():.0f}")
for s in STYLES:
    d = fl[s]["tok_long"] - fl[s]["tok_short"]
    print(f"filler dtok [{s:<10}]        : mean={d.mean():.0f} "
          f"min={d.min():.0f} max={d.max():.0f}")
tok_story_all = torch.cat([tok_story_tr, tok_story_te])
tok_flong_all = torch.cat([fl[s]["tok_long"] for s in STYLES])
ratio = float(tok_flong_all.mean() / tok_story_all.mean())
print(f"story len={tok_story_all.mean():.0f} tok, filler-long len={tok_flong_all.mean():.0f} tok "
      f"(ratio {ratio:.2f}) -> residual length gap biases *_M "
      f"{'AGAINST' if ratio > 1 else 'TOWARD'} narrativity")
if not 0.8 <= ratio <= 1.25:
    print("  WARNING: fillers are not length-matched to the stories. The *_M columns "
          "are not a length control until the filler prose is rewritten to match.")

# identifiability of FIX 3, before vs after adding the fillers
tk_t3 = torch.cat([tok_story_all, torch.cat([tok_bare_tr, tok_bare_te])])
is_t3 = torch.cat([torch.ones(len(tok_story_all)), torch.zeros(len(tok_story_all))])
tk_pool = torch.cat([tk_t3, tok_flong_all, torch.cat([fl[s]["tok_short"] for s in STYLES])])
is_pool = torch.cat([is_t3, torch.zeros(len(tok_flong_all) + sum(len(fl[s]["tok_short"]) for s in STYLES))])
print(f"corr(n_tokens, is_story): Tier-3 only={pearson(tk_t3, is_t3):+.2f}  "
      f"pooled with fillers={pearson(tk_pool, is_pool):+.2f}   "
      "(near +-1 => FIX 3 is not identified)")

# -------------------------------------------------------------------------- directions
dh_tr = story_tr - bare_tr                                    # [n_tr, L+1, hidden]
raw_dir = story_tr.mean(0) - bare_tr.mean(0)                  # confounded baseline
ols_dir, len_slope = ols_paired(dh_tr, dtok_tr)               # FIX 1
loo_cos = loo_intercept_cos(dh_tr, dtok_tr, ols_dir)

# FIX 2 basis: one length vector per filler style, fitted on train requests only
basis_tr = {
    s: fl[s]["long"][fl[s]["tr"]].mean(0) - fl[s]["short"][fl[s]["tr"]].mean(0)
    for s in STYLES
}
len_dir_pooled = torch.stack([basis_tr[s] for s in STYLES]).mean(0)   # foil direction

# full-data variants: the artifact to reuse in Phase 2/3 (no held-out left, so they
# are reported in the CSV only, never used for the AUROCs printed below)
story_all = torch.cat([story_tr, story_te])
bare_all = torch.cat([bare_tr, bare_te])
dtok_all = torch.cat([dtok_tr, dtok_te])
raw_dir_all = story_all.mean(0) - bare_all.mean(0)
ols_dir_all, _ = ols_paired(story_all - bare_all, dtok_all)
basis_all = {s: fl[s]["long"].mean(0) - fl[s]["short"].mean(0) for s in STYLES}

# -------------------------------------------------------- pooled items for FIX 3
pool_reps = torch.cat(
    [story_all, bare_all] + [fl[s]["long"] for s in STYLES] + [fl[s]["short"] for s in STYLES]
)
pool_tok = tk_pool
pool_is_story = is_pool

# length-matched eval sets: stories vs long fillers, both classes long
neg_te = {s: fl[s]["long"][fl[s]["te"]] for s in STYLES}
neg_te_all = torch.cat([neg_te[s] for s in STYLES])
neg_all = {s: fl[s]["long"] for s in STYLES}
neg_all_all = torch.cat([neg_all[s] for s in STYLES])

# ------------------------------------------------------------------------------ table
n_layers = raw_dir.shape[0]
print(f"\n{'lyr':>3} {'raw_ho':>7} {'len_ho':>7} {'raw_M':>6} {'ols_M':>6} {'ort_M':>6} "
      f"{'rawMob':>7} {'lenfrc':>7} {'k':>2} {'tStory':>7} {'looC':>6}")
rows = []
for l in range(1, n_layers):
    b = torch.stack([basis_tr[s][l] for s in STYLES])
    ort_dir_l, len_frac_raw, len_rank = project_out(raw_dir[l], b)
    _, len_frac_ols, _ = project_out(ols_dir[l], b)

    scores = proj(pool_reps, raw_dir[l], l)
    r = {
        "layer": l,
        # confounded baseline + the foil that shows it is a confound
        "raw_ho": auroc_at(raw_dir[l], story_te, bare_te, l),
        "len_ho": auroc_at(len_dir_pooled[l], story_te, bare_te, l),
        # length-matched: held-out stories vs held-out filler longs
        "raw_M": auroc_at(raw_dir[l], story_te, neg_te_all, l),
        "ols_M": auroc_at(ols_dir[l], story_te, neg_te_all, l),
        "ort_M": auroc_at(ort_dir_l, story_te, neg_te_all, l),
        "raw_M_obl": auroc_at(raw_dir[l], story_te, neg_te[SHARPEST], l),
        "ols_M_obl": auroc_at(ols_dir[l], story_te, neg_te[SHARPEST], l),
        "ort_M_obl": auroc_at(ort_dir_l, story_te, neg_te[SHARPEST], l),
        # FIX 2 diagnostics
        "len_frac_raw": len_frac_raw,
        "len_frac_ols": len_frac_ols,
        "len_rank": len_rank,
        # FIX 3 (on all three directions; t is scale-free so they are comparable)
        "t_is_story": partial_t_is_story(scores, pool_tok, pool_is_story),
        "t_is_story_ols": partial_t_is_story(
            proj(pool_reps, ols_dir[l], l), pool_tok, pool_is_story),
        "t_is_story_ort": partial_t_is_story(
            proj(pool_reps, ort_dir_l, l), pool_tok, pool_is_story),
        "auroc_binned": binned_auroc(scores, pool_tok, pool_is_story),
        # how far each fix moved the vector
        "cos_ols_raw": cos(ols_dir[l], raw_dir[l]),
        "cos_ort_raw": cos(ort_dir_l, raw_dir[l]),
        "cos_slope_len": cos(len_slope[l], len_dir_pooled[l]),
        "loo_cos_ols": float(loo_cos[l]),
        # full-data (positives seen; stable but optimistic) matched AUROCs
        "raw_M_full": auroc_at(raw_dir_all[l], story_all, neg_all_all, l),
        "ols_M_full": auroc_at(ols_dir_all[l], story_all, neg_all_all, l),
        "ort_M_full": auroc_at(
            project_out(raw_dir_all[l], torch.stack([basis_all[s][l] for s in STYLES]))[0],
            story_all, neg_all_all, l),
    }
    for s in STYLES:                       # per-style matched AUROC, raw direction
        r[f"raw_M_{s}"] = auroc_at(raw_dir[l], story_te, neg_te[s], l)
    rows.append(r)
    print(f"{l:>3} {r['raw_ho']:>7.2f} {r['len_ho']:>7.2f} {r['raw_M']:>6.2f} "
          f"{r['ols_M']:>6.2f} {r['ort_M']:>6.2f} {r['raw_M_obl']:>7.2f} "
          f"{r['len_frac_raw']:>7.2f} {r['len_rank']:>2} {r['t_is_story']:>7.1f} "
          f"{r['loo_cos_ols']:>6.2f}")

# ------------------------------------------------------------------------------ output
out_dir = os.path.join(HERE, "results", MODEL.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "tier3_deconfounded_metrics.csv")
fields = list(rows[0].keys())
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

ort_all_stack = torch.stack([
    project_out(raw_dir_all[l], torch.stack([basis_all[s][l] for s in STYLES]))[0]
    if l > 0 else torch.zeros_like(raw_dir_all[0])
    for l in range(n_layers)
])
dir_path = os.path.join(out_dir, "tier3_deconfounded_directions.pt")
torch.save({
    "model": MODEL,
    "narrativity_raw": raw_dir_all,        # confounded, for reference
    "narrativity_ols": ols_dir_all,        # FIX 1
    "narrativity_orth": ort_all_stack,     # FIX 2
    "length_slope": len_slope,             # in-distribution length axis (FIX 1 by-product)
    "length_basis": {s: basis_all[s] for s in STYLES},
    "length_pooled": len_dir_pooled,
}, dir_path)

# best layer = narrativity survives BOTH fixes on the length-matched eval
best = max(rows, key=lambda r: min(r["ols_M"], r["ort_M"]))
print(f"\nbest deconfounded layer: {best['layer']}  "
      f"ols_M={best['ols_M']:.2f} ort_M={best['ort_M']:.2f} raw_M={best['raw_M']:.2f} "
      f"| raw_ho={best['raw_ho']:.2f} len_ho={best['len_ho']:.2f} "
      f"len_frac={best['len_frac_raw']:.2f} looC={best['loo_cos_ols']:.2f}")
print(f"resolution: raw_ho/len_ho = {len(t3_te)}x{len(t3_te)} comparisons; "
      f"*_M = {len(t3_te)}x{neg_te_all.shape[0]}; read layer BANDS, not single layers.")
print(f"saved metrics    -> {csv_path}")
print(f"saved directions -> {dir_path}")
