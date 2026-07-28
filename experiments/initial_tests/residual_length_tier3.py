"""
Residual-length audit for `narrativity_orth` (the FIX 2 output of deconfound_length_tier3.py).

WHY THIS EXISTS
    deconfound_length_tier3.py showed narrativity SURVIVES length matching (`ort_M`,
    `ort_M_obl` = 1.00). It never showed length is GONE. Both `*_M` evals are
    length-matched by construction, so they cannot detect residual length in the
    vector -- a direction that still reads length would score exactly the same on them.

    FIX 2 removes length with an ORTHOGONAL projection (`project_out`: SVD basis,
    d - QQ^T d), which makes d^T b = 0 for every length vector b. But the thing we
    steer with is the READOUT x . d, and its covariance with the length readout x . b is

        cov(x.d, x.b) = d^T Sigma b            (Sigma = activation covariance)

    which orthogonality does NOT set to zero. Whenever length and narrativity are
    correlated in activation space -- exactly the situation here, `len_frac_raw` ~ 0.5 --
    d^T b = 0 leaves d^T Sigma b != 0 and the residual leaks into steering. Removing
    length from the readout requires Sigma-orthogonality, which is the LEACE / mass-mean
    correction (Belrose et al. 2023; Marks & Tegmark 2023). See
    `research/deconfounding-length.md`.

WHAT IT TESTS
    A PURE-LENGTH, NO-NARRATIVE contrast: filler-long vs filler-short. Neither class is
    a story, so any separation is length/verbosity and nothing else. Score it with each
    candidate direction.

        resid_ort ~ 0.5   -> orthogonal removal sufficed; LEACE is unnecessary; the
                             question raised in deconfounding-length.md is closed
        resid_ort >> 0.5  -> `narrativity_orth` still reads length; adopt
                             `narrativity_leace` if `leace_M` holds up
        resid_ort << 0.5  -> also leakage (sign-flipped), not a clean vector

    Held out properly: the length subspace is fitted on the TRAIN filler rows, the
    contrast is scored on the TEST filler rows. Scoring on train rows would be
    near-circular -- those rows define the subspace being removed.

COLUMNS
    resid_raw      AUROC(filler-long vs filler-short, test rows) for `narrativity_raw`
                   -> must be far from 0.5, else the eval cannot detect length at all
                      and every other column is meaningless
    resid_len      same for `length_pooled` -> positive control, expect ~1.0
    resid_ort      same for `narrativity_orth`                        <-- THE ANSWER
    resid_leace    same for the Sigma-orthogonal alternative
    p_ort/p_leace  two-sided exact paired sign-flip p-value against AUROC = 0.5.
                   With 3 test requests x 3 styles = 9 pairs the finest attainable
                   p is 1/512; read `p > 0.05` as "no detectable leak at this n",
                   NOT as "no leak".
    sig_frac_*     fraction of the direction's Sigma-norm inside the length subspace,
                   i.e. how much of the READOUT is length. This is `len_frac` measured
                   in the metric that matters. ~0 for leace by construction (sanity).
    euc_frac_ort   Euclidean length fraction of `narrativity_orth`; ~0 by construction.
                   euc_frac_ort ~ 0 while sig_frac_ort is large IS the whole point.
    ort_M/leace_M  length-matched narrativity AUROC (story vs filler-long, held out),
    *_obl          same as deconfound_length_tier3.py so the numbers are comparable.
                   If `leace_M` collapses while `ort_M` held, the two effects are not
                   separable at this n -- report that, do not pick the flattering one.
    cos_leace_ort  how far the Sigma correction moved the vector
    resid_ort_full in-sample variant (full-data artifact, basis from all filler rows).
                   Optimistic; reported for parity with the `*_full` columns upstream.

READING IT
    Decide on the L19-28 band (`BAND`), never a single layer. The verdict block at the
    end does this. Cells are 9x9 = 81 comparisons; a 1.00 is indistinguishable from 0.95.

Usage:
    python residual_length_tier3.py [model_name]        # default Qwen/Qwen2.5-7B-Instruct
    BAND=19-28 RHO=0.1 python residual_length_tier3.py Qwen/Qwen2.5-3B-Instruct

Reads .jsonl from data/initial_tests/ (override with $DATA_DIR); writes
results/<model>/tier3_residual_length_{metrics.csv,directions.pt}.
"""
import csv
import json
import math
import os
import random
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(HERE, "..", "..", "data", "initial_tests")
)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
STYLES = ["expository", "ambient", "oblique"]
SHARPEST = "oblique"
RHO = float(os.environ.get("RHO", 0.1))        # covariance shrinkage toward (tr/d)*I
BAND = os.environ.get("BAND", "19-28")
BAND_LO, BAND_HI = (int(x) for x in BAND.split("-"))


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
    """[n_layers+1, hidden] at the final prompt token (chat template applied)."""
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


def proj(reps, dvec_l, l):
    u = dvec_l / (dvec_l.norm() + 1e-8)
    return reps[:, l, :] @ u


def auroc_at(dvec_l, pos_reps, neg_reps, l):
    return auroc(proj(pos_reps, dvec_l, l).tolist(), proj(neg_reps, dvec_l, l).tolist())


def paired_signflip_p(pos, neg, max_exact=14, n_sample=4096):
    """Two-sided p for AUROC(pos, neg) != 0.5 under within-pair exchangeability.

    pos[i] and neg[i] are the long/short members of the SAME filler pair, so the null
    "this direction is blind to length" makes swapping them exchangeable. Exact
    enumeration of all 2^n sign flips while that is cheap, sampled otherwise (fixed
    seed -> reproducible). Statistic is |AUROC - 0.5|: leakage in either direction counts.
    """
    n = len(pos)
    obs = abs(auroc(pos, neg) - 0.5)
    if n <= max_exact:
        masks = range(1 << n)
    else:
        rng = random.Random(0)
        masks = [rng.getrandbits(n) for _ in range(n_sample)]
    hits = tot = 0
    for m in masks:
        p2 = [neg[i] if (m >> i) & 1 else pos[i] for i in range(n)]
        n2 = [pos[i] if (m >> i) & 1 else neg[i] for i in range(n)]
        hits += abs(auroc(p2, n2) - 0.5) >= obs - 1e-12
        tot += 1
    return hits / tot


# ------------------------------------------------------- projections: Euclidean vs Sigma
def basis_q(basis, rank_tol=0.05):
    """Orthonormal columns spanning the length subspace, rank-truncated.

    Same truncation as `project_out` in deconfound_length_tier3.py so the Euclidean
    arm here reproduces FIX 2 exactly and the Sigma arm differs only in the metric.
    """
    u, s, _ = torch.linalg.svd(basis.T, full_matrices=False)
    return u[:, s > rank_tol * s[0]]


def sigma_apply(xc, m):
    """(Sigma_shrunk @ m) without ever forming the hidden x hidden matrix.

    xc: [N, hidden] centered reps; m: [hidden, k]. Cost is O(N*hidden*k). Shrinkage is
    mandatory, not cosmetic: N is a few dozen and hidden is thousands, so the sample
    covariance is massively rank-deficient.
    """
    n, d = xc.shape
    sm = xc.T @ (xc @ m) / n
    tr_over_d = float((xc * xc).sum()) / (n * d)
    return (1 - RHO) * sm + RHO * tr_over_d * m


def sigma_project_out(d, q, xc):
    """Remove span(q) from d in the Sigma metric: the LEACE-style oblique projection.

    Solves for the unique c with (d - q c)^T Sigma q = 0, i.e. c = (q^T Sigma q)^-1 q^T Sigma d.
    Result has zero readout covariance with every length direction in span(q), which is
    what orthogonal projection fails to give.
    """
    sq = sigma_apply(xc, q)                       # [hidden, k]
    g = q.T @ sq                                  # [k, k] = q^T Sigma q
    c = torch.linalg.solve(g + 1e-6 * torch.eye(g.shape[0]), sq.T @ d)
    return d - q @ c


def sigma_frac(d, q, xc):
    """Fraction of d's Sigma-norm lying in span(q) -- length share of the READOUT."""
    sq = sigma_apply(xc, q)
    g = q.T @ sq
    v = sq.T @ d
    num = float(v @ torch.linalg.solve(g + 1e-6 * torch.eye(g.shape[0]), v))
    den = float(d @ sigma_apply(xc, d.unsqueeze(1)).squeeze(1))
    return math.sqrt(max(num, 0.0) / max(den, 1e-12))


def euclid_frac(d, q):
    return float((q @ (q.T @ d)).norm() / (d.norm() + 1e-8))


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
      + ", ".join(f"{s}={len(lf_by_style[s])}" for s in STYLES))
print("extracting activations (one forward pass per prompt) ...")

story_tr = reps_for(t3_tr, "story")
bare_tr = reps_for(t3_tr, "bare")
story_te = reps_for(t3_te, "story")
bare_te = reps_for(t3_te, "bare")

fl = {}
for s in STYLES:
    rows_s = lf_by_style[s]
    fl[s] = {
        "long": reps_for(rows_s, "long"),
        "short": reps_for(rows_s, "short"),
        "tr": [i for i, r in enumerate(rows_s) if r["split"] == "train"],
        "te": [i for i, r in enumerate(rows_s) if r["split"] == "test"],
    }

# -------------------------------------------------------------------------- directions
raw_dir = story_tr.mean(0) - bare_tr.mean(0)
basis_tr = {
    s: fl[s]["long"][fl[s]["tr"]].mean(0) - fl[s]["short"][fl[s]["tr"]].mean(0)
    for s in STYLES
}
len_dir_pooled = torch.stack([basis_tr[s] for s in STYLES]).mean(0)

# full-data variants: reproduce the saved artifact so `resid_ort_full` audits the exact
# vector downstream phases would load
story_all = torch.cat([story_tr, story_te])
bare_all = torch.cat([bare_tr, bare_te])
raw_dir_all = story_all.mean(0) - bare_all.mean(0)
basis_all = {s: fl[s]["long"].mean(0) - fl[s]["short"].mean(0) for s in STYLES}

# ------------------------------------------------------------------------ eval sets
# pure length, no narrative in EITHER class, held-out requests only
pos_len = torch.cat([fl[s]["long"][fl[s]["te"]] for s in STYLES])
neg_len = torch.cat([fl[s]["short"][fl[s]["te"]] for s in STYLES])
# narrativity, length-matched (same construction as deconfound_length_tier3.py)
neg_te = {s: fl[s]["long"][fl[s]["te"]] for s in STYLES}
neg_te_all = torch.cat([neg_te[s] for s in STYLES])
# in-sample counterpart for the *_full column
pos_len_all = torch.cat([fl[s]["long"] for s in STYLES])
neg_len_all = torch.cat([fl[s]["short"] for s in STYLES])
# pooled reps -> Sigma. Same population the readout will run on.
pool_reps = torch.cat(
    [story_all, bare_all] + [fl[s]["long"] for s in STYLES] + [fl[s]["short"] for s in STYLES]
)
print(f"Sigma from {pool_reps.shape[0]} pooled prompts, shrinkage rho={RHO} | "
      f"residual-length eval = {pos_len.shape[0]}x{neg_len.shape[0]} comparisons")

# ------------------------------------------------------------------------------ table
n_layers = raw_dir.shape[0]
print(f"\n{'lyr':>3} {'rsdRaw':>7} {'rsdLen':>7} {'rsdOrt':>7} {'p_ort':>6} "
      f"{'rsdLce':>7} {'sfOrt':>6} {'efOrt':>6} {'ort_M':>6} {'lceM':>6} {'cosLO':>6}")
rows = []
leace_stack = torch.zeros_like(raw_dir_all)
for l in range(1, n_layers):
    xc = pool_reps[:, l, :] - pool_reps[:, l, :].mean(0)
    q = basis_q(torch.stack([basis_tr[s][l] for s in STYLES]))
    q_all = basis_q(torch.stack([basis_all[s][l] for s in STYLES]))

    ort_l = raw_dir[l] - q @ (q.T @ raw_dir[l])              # FIX 2, reproduced
    lce_l = sigma_project_out(raw_dir[l], q, xc)             # Sigma-orthogonal rival
    ort_full_l = raw_dir_all[l] - q_all @ (q_all.T @ raw_dir_all[l])
    leace_stack[l] = sigma_project_out(raw_dir_all[l], q_all, xc)

    s_ort = (proj(pos_len, ort_l, l).tolist(), proj(neg_len, ort_l, l).tolist())
    s_lce = (proj(pos_len, lce_l, l).tolist(), proj(neg_len, lce_l, l).tolist())

    r = {
        "layer": l,
        # can this eval see length at all?
        "resid_raw": auroc_at(raw_dir[l], pos_len, neg_len, l),
        "resid_len": auroc_at(len_dir_pooled[l], pos_len, neg_len, l),
        # the answer
        "resid_ort": auroc(*s_ort),
        "resid_leace": auroc(*s_lce),
        "p_ort": paired_signflip_p(*s_ort),
        "p_leace": paired_signflip_p(*s_lce),
        # geometry: readout metric vs Euclidean
        "sig_frac_raw": sigma_frac(raw_dir[l], q, xc),
        "sig_frac_ort": sigma_frac(ort_l, q, xc),
        "sig_frac_leace": sigma_frac(lce_l, q, xc),
        "euc_frac_ort": euclid_frac(ort_l, q),
        # does narrativity survive the stronger removal?
        "ort_M": auroc_at(ort_l, story_te, neg_te_all, l),
        "leace_M": auroc_at(lce_l, story_te, neg_te_all, l),
        "ort_M_obl": auroc_at(ort_l, story_te, neg_te[SHARPEST], l),
        "leace_M_obl": auroc_at(lce_l, story_te, neg_te[SHARPEST], l),
        "cos_leace_ort": cos(lce_l, ort_l),
        # in-sample variant of the saved artifact (optimistic)
        "resid_ort_full": auroc_at(ort_full_l, pos_len_all, neg_len_all, l),
    }
    rows.append(r)
    print(f"{l:>3} {r['resid_raw']:>7.2f} {r['resid_len']:>7.2f} {r['resid_ort']:>7.2f} "
          f"{r['p_ort']:>6.3f} {r['resid_leace']:>7.2f} {r['sig_frac_ort']:>6.2f} "
          f"{r['euc_frac_ort']:>6.2f} {r['ort_M']:>6.2f} {r['leace_M']:>6.2f} "
          f"{r['cos_leace_ort']:>6.2f}")

# ------------------------------------------------------------------------------ output
out_dir = os.path.join(HERE, "results", MODEL.replace("/", "_"))
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "tier3_residual_length_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

dir_path = os.path.join(out_dir, "tier3_residual_length_directions.pt")
torch.save({"model": MODEL, "rho": RHO, "narrativity_leace": leace_stack}, dir_path)

# does the reproduced full-data orth vector match the artifact downstream phases load?
saved = os.path.join(out_dir, "tier3_deconfounded_directions.pt")
if os.path.exists(saved):
    ref = torch.load(saved)["narrativity_orth"]
    mid = [l for l in range(1, n_layers) if BAND_LO <= l <= BAND_HI]
    agree = sum(
        cos(raw_dir_all[l] - basis_q(torch.stack([basis_all[s][l] for s in STYLES]))
            @ (basis_q(torch.stack([basis_all[s][l] for s in STYLES])).T @ raw_dir_all[l]),
            ref[l]) for l in mid
    ) / len(mid)
    print(f"\nreproduces saved narrativity_orth over L{BAND}: mean cos = {agree:.4f} "
          f"(< 0.999 => this script and FIX 2 disagree, fix before reading anything else)")

# ------------------------------------------------------------------------------ verdict
band = [r for r in rows if BAND_LO <= r["layer"] <= BAND_HI]
mean = lambda k: sum(r[k] for r in band) / len(band)
leaky = [r["layer"] for r in band if r["p_ort"] < 0.05]
print(f"\n--- verdict over L{BAND} ({len(band)} layers) ---")
print(f"eval sanity : resid_raw={mean('resid_raw'):.2f}  resid_len={mean('resid_len'):.2f} "
      f"(both must be far from 0.50, else the eval is blind and nothing below counts)")
print(f"leakage     : resid_ort={mean('resid_ort'):.2f}  sig_frac_ort={mean('sig_frac_ort'):.2f} "
      f"vs euc_frac_ort={mean('euc_frac_ort'):.2f}")
print(f"              layers with p_ort < 0.05: {leaky if leaky else 'none'}")
print(f"survival    : ort_M={mean('ort_M'):.2f} leace_M={mean('leace_M'):.2f} | "
      f"ort_M_obl={mean('ort_M_obl'):.2f} leace_M_obl={mean('leace_M_obl'):.2f}")
if not leaky and mean("sig_frac_ort") < 0.15:
    print("=> no detectable residual length. Keep narrativity_orth; LEACE unnecessary.")
elif mean("leace_M") >= mean("ort_M") - 0.05:
    print("=> residual length detected AND narrativity survives the Sigma correction. "
          "Switch downstream phases to narrativity_leace.")
else:
    print("=> residual length detected BUT the Sigma correction also costs narrativity "
          "(leace_M << ort_M). Not separable at this n -- report both, escalate to the "
          "design-side controls in research/deconfounding-length.md.")
print(f"\nresolution: {pos_len.shape[0]}x{neg_len.shape[0]} comparisons per cell, "
      f"p floor = 1/{2 ** pos_len.shape[0]}; read the band, not single layers.")
print(f"saved metrics    -> {csv_path}")
print(f"saved directions -> {dir_path}")
