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


def extract_v1(model_id, out, view, neg_pole="neg"):
    """d_v1_50 from the 50-row matched subsample; saved so 1.2a can reuse it."""
    m = acts.load_view_matrix(model_id, view)
    pos, neg = m["pos"], m[neg_pole]
    dvec = met.diff_in_means(pos, neg)
    stem = mf.stem("directions", "v1_fair50" if neg_pole == "neg" else "v1_fair50_audience")
    config = {"source": "v1_matched_50", "neg_pole": neg_pole, "n_pairs": pos.shape[0],
              "estimator": "diff_in_means", "seed": cfg.SEED}
    with mf.Run(out, stem, config, {"view_key": view["view_key"]}) as run:
        torch.save({"model": model_id, "axis": "story_v1_50", "neg_pole": neg_pole,
                    "d": torch.from_numpy(dvec), "u": torch.from_numpy(met.unit(dvec)),
                    "lopo_d": torch.from_numpy(met.lopo_directions(pos, neg)),
                    "n_pairs": pos.shape[0], "view_key": view["view_key"],
                    "run_key": run.run_key}, run.artefact(".pt"))
    return dvec, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--curve", action="store_true")
    args = ap.parse_args()
    out = cfg.results_dir("extraction", args.model)

    v2_path = out / f"{mf.stem('directions', 'story')}.pt"
    if not v2_path.exists():
        raise SystemExit("run extract_direction.py --direction story first")
    v2 = torch.load(v2_path, weights_only=False)
    d_v2, u_v2 = v2["d"].numpy(), v2["u"].numpy()

    v1_view = views.read_view(args.model, "v1_fair50", "train")
    d_v1, m1 = extract_v1(args.model, out, v1_view, "neg")
    u_v1 = met.unit(d_v1)

    have_audience = "neg2" in v1_view["poles"]
    d_v1_aud = extract_v1(args.model, out, v1_view, "neg2")[0] if have_audience else None

    v2_view = views.read_view(args.model, "story", "train")
    m2 = acts.load_view_matrix(args.model, v2_view)

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
        curve = subsample_curve(args.model)

    stem = mf.stem("compare_crossed")
    config = {"matched_n": True, "neg_pole": "prompt_expository", "curve": args.curve,
              "curve_n": CURVE_N if args.curve else [], "seed": cfg.SEED}
    inputs = {"v2_view_key": v2_view["view_key"], "v1_view_key": v1_view["view_key"],
              "v2_run_key": v2.get("run_key")}
    with mf.Run(out, stem, config, inputs) as run:
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


def subsample_curve(model_id):
    """cos(d_n, d_full) on the v1 curve view: where does the vector stop moving?"""
    view = views.read_view(model_id, "v1_curve", "train")
    m = acts.load_view_matrix(model_id, view)
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
