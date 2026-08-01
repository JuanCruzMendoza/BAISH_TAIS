"""Spec 1.6: is the v2 vector the same vector the crossed table gives?

Matched n — 50 against 50 — so any difference is construction and not sample size.
Cross-evaluation uses the other side's 50, not its 15 (spec 0.7.3): neither vector
was fitted on the other's data.

    python compare_crossed.py <model>
    python compare_crossed.py <model> --curve      # second run, needs v1_curve cached
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views

CURVE_N = [5, 10, 25, 50, 100, 250, 1000]


def paired_cos(pos, neg, u):
    delta = pos - neg
    return float(np.mean([met.cos(delta[i], u) for i in range(delta.shape[0])]))


def load(lay, axis):
    stem = mf.stem("directions", axis)
    path = lay.vectors / f"{stem}.pt"
    if not path.exists():
        raise SystemExit(f"run extract_direction.py --direction {axis} first")
    mf.load_upstream(lay.meta / f"{stem}_manifest.json")
    return torch.load(path, weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    lay = cfg.Layout("extraction", args.model, args.tag)

    v2, v1 = load(lay, "story_v2"), load(lay, "story_v1")
    d_v2, u_v2 = v2["d"].numpy(), v2["u"].numpy()
    d_v1, u_v1 = v1["d"].numpy(), v1["u"].numpy()

    v1_view = views.read_view(lay, "story_v1", "train")
    v2_view = views.read_view(lay, "story_v2", "train")
    m1 = acts.load_view_matrix(lay, v1_view)
    m2 = acts.load_view_matrix(lay, v2_view)

    # The audience arm is a reported cosine, not a saved artefact (spec 1.6).
    d_v1_aud = (met.diff_in_means(m1["pos"], m1["neg2"]) if "neg2" in v1_view["poles"] else None)

    Lp1 = d_v2.shape[0]
    rows = []
    for l in range(Lp1):
        # cross-evaluation, n=50 each way, neither vector fitted on the other's data
        a_pos = np.einsum("nd,d->n", m1["pos"][:, l, :], u_v2[l])
        a_neg = np.einsum("nd,d->n", m1["neg"][:, l, :], u_v2[l])
        b_pos = np.einsum("nd,d->n", m2["pos"][:, l, :], u_v1[l])
        b_neg = np.einsum("nd,d->n", m2["neg"][:, l, :], u_v1[l])
        ci_a, ci_b = met.auroc_ci(a_pos, a_neg), met.auroc_ci(b_pos, b_neg)
        row = {"layer": l, "depth": round(l / (Lp1 - 1), 4),
               "cos_v2_v1": met.cos(d_v2[l], d_v1[l]),
               "cos_null_band": met.random_cos_band(d_v2.shape[1]),
               "auroc_v2_on_v1": ci_a["auroc"], "v2_on_v1_ci_lo": ci_a["ci_lo"],
               "v2_on_v1_ci_hi": ci_a["ci_hi"], "n_v1": ci_a["n"],
               "auroc_v1_on_v2": ci_b["auroc"], "v1_on_v2_ci_lo": ci_b["ci_lo"],
               "v1_on_v2_ci_hi": ci_b["ci_hi"], "n_v2": ci_b["n"],
               "mean_paired_cos_v2": paired_cos(m2["pos"][:, l, :], m2["neg"][:, l, :], u_v2[l]),
               "mean_paired_cos_v1": paired_cos(m1["pos"][:, l, :], m1["neg"][:, l, :], u_v1[l])}
        if d_v1_aud is not None:
            row["cos_v2_v1_audience"] = met.cos(d_v2[l], d_v1_aud[l])
        rows.append(row)

    curve = {}
    if args.curve:
        curve = subsample_curve(lay)

    stem = mf.stem("compare_crossed")
    config = {"matched_n": True, "neg_pole": "prompt_expository", "curve": args.curve,
              "curve_n": CURVE_N if args.curve else [], "seed": cfg.SEED}
    inputs = {"v2_view_key": v2_view["view_key"], "v1_view_key": v1_view["view_key"],
              "v2_run_key": v2.get("run_key"), "v1_run_key": v1.get("run_key")}
    with mf.Run(lay, stem, config, inputs) as run:
        with run.artefact(".csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        if curve:
            run.artefact("_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")

        band = cfg.band(Lp1 - 1)
        inband = [r for r in rows if r["layer"] in band]
        print(f"v2 (n={m2['pos'].shape[0]})  vs  v1_50 (n={m1['pos'].shape[0]})   band {band[0]}-{band[-1]}")
        print(f"  cos(d_v2, d_v1_50)  band mean {np.mean([r['cos_v2_v1'] for r in inband]):+.3f}"
              f"   null band +/-{rows[0]['cos_null_band']:.3f}")
        print(f"  AUROC v2 -> v1 rows  {np.mean([r['auroc_v2_on_v1'] for r in inband]):.3f}")
        print(f"  AUROC v1 -> v2 pairs {np.mean([r['auroc_v1_on_v2'] for r in inband]):.3f}")
        print(f"  mean_paired_cos      v2 {np.mean([r['mean_paired_cos_v2'] for r in inband]):.3f}"
              f"   v1 {np.mean([r['mean_paired_cos_v1'] for r in inband]):.3f}")


def subsample_curve(lay):
    """cos(d_n, d_full) on the v1 curve view: where does the vector stop moving?"""
    view = views.read_view(lay, "v1_curve", "train")
    m = acts.load_view_matrix(lay, view)
    pos, neg = m["pos"], m["neg"]
    n_tot = pos.shape[0]
    d_full = met.diff_in_means(pos, neg)
    rng = np.random.default_rng(cfg.SEED)
    order = rng.permutation(n_tot)
    band = cfg.band(pos.shape[1] - 1)
    out = {}
    for n in CURVE_N:
        if n > n_tot:
            continue
        idx = order[:n]
        d_n = met.diff_in_means(pos[idx], neg[idx])
        out[str(n)] = {"band_mean_cos": float(np.mean([met.cos(d_n[l], d_full[l]) for l in band])),
                       "per_layer": {str(l): met.cos(d_n[l], d_full[l]) for l in band}}
    out["n_total"] = n_tot
    return out


if __name__ == "__main__":
    main()
