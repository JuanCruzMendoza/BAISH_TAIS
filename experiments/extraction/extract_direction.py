"""Diff-in-means direction for one axis, every layer, from cached activations. CPU only.

    python extract_direction.py <model> --direction story_v2
    python extract_direction.py <model> --direction story_v2_1k --curve

Writes directions__<axis>.pt with the full vector, the LOPO vectors, pole means
and sigma_act, all of which sections 1.2, 2 and 5 reuse. `--curve` adds the
saturation curve as csv/directions__<axis>_curve.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views

CURVE_STEP = 50      # n = 50, 100, ... up to one step below the fit split
CURVE_SEEDS = 5      # a single draw at small n is noisy; report mean +/- sd


def subsample_curve(pos, neg, band, step=CURVE_STEP, seeds=CURVE_SEEDS):
    """cos(d_n, d_full) over nested subsamples of the fit split.

    One permutation per seed, prefixes taken from it, so the n's are nested and the
    curve is monotone by construction rather than by luck. Optimistic by design --
    the n pairs are inside d_full -- which is why the disjoint train/held-out cosine
    is reported beside it.

    Every layer is stored (the figures span 0..L); `band_mean_cos` aggregates `band`.
    """
    n, Lp1 = pos.shape[0], pos.shape[1]
    d_full = met.diff_in_means(pos, neg)
    orders = [np.random.default_rng(cfg.SEED + s).permutation(n) for s in range(seeds)]
    inband = [l for l in range(Lp1) if l in set(band)]
    out = {}
    for k in range(step, n, step):
        cos_sk = np.array([[met.cos(met.diff_in_means(pos[o[:k]], neg[o[:k]])[l], d_full[l])
                            for l in range(Lp1)] for o in orders])     # [seeds, L+1]
        out[str(k)] = {
            "band_mean_cos": float(cos_sk[:, inband].mean()),
            "band_mean_cos_sd": float(cos_sk[:, inband].mean(axis=1).std(ddof=1)),
            "per_layer": {str(l): float(cos_sk[:, l].mean()) for l in range(Lp1)},
            "per_layer_sd": {str(l): float(cos_sk[:, l].std(ddof=1)) for l in range(Lp1)}}
    return out


def disjoint_cos(d_train, d_ho, band):
    """cos(d_800_train, d_200_heldout): two vectors fitted on non-overlapping pairs."""
    per_layer = {str(l): met.cos(d_train[l], d_ho[l]) for l in range(d_train.shape[0])}
    return {"band_mean_cos": float(np.mean([per_layer[str(l)] for l in band])),
            "per_layer": per_layer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--direction", required=True, choices=views.DIRECTIONS)
    ap.add_argument("--curve", action="store_true",
                    help="cos(d_n, d_full) subsample curve + the train/held-out cosine")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    lay = cfg.Layout("extraction", args.model, args.tag)
    view = views.read_view(lay, args.direction, "train")
    m = acts.load_view_matrix(lay, view)
    pos, neg = m["pos"], m["neg"]                       # [n, L+1, d]
    n, Lp1, d = pos.shape
    band = cfg.band(Lp1 - 1)

    dvec = met.diff_in_means(pos, neg)                  # [L+1, d]
    lopo = met.lopo_directions(pos, neg)                # [n, L+1, d]
    pooled = np.concatenate([pos, neg])
    mu = pooled.mean(axis=0)                            # [L+1, d] pooled centre
    sigma = met.sigma_act(pooled)                       # [L+1]

    curve = {}
    if args.curve:
        curve = {"n_total": n, "step": CURVE_STEP, "seeds": CURVE_SEEDS,
                 "cos_null_band": met.random_cos_band(d), "n_layers": Lp1 - 1,
                 "band": band, "cos_n_vs_full": subsample_curve(pos, neg, band)}
        try:
            ho = acts.load_view_matrix(lay, views.read_view(lay, args.direction, "heldout"))
        except FileNotFoundError:
            curve["cos_train_vs_heldout"] = None
            print("! no held-out view cached: the disjoint-fit cosine is skipped")
        else:
            curve["n_heldout"] = int(ho["pos"].shape[0])
            curve["cos_train_vs_heldout"] = disjoint_cos(
                dvec, met.diff_in_means(ho["pos"], ho["neg"]), band)

    stem = mf.stem("directions", args.direction)
    config = {"direction": args.direction, "n_pairs": n, "position": "last_token",
              "estimator": "diff_in_means", "curve": args.curve,
              "curve_step": CURVE_STEP if args.curve else None,
              "curve_seeds": CURVE_SEEDS if args.curve else None, "seed": cfg.SEED}
    inputs = {"view_key": view["view_key"], "source_files": view["source_files"]}

    with mf.Run(lay, stem, config, inputs) as run:
        torch.save({"model": args.model, "axis": args.direction,
                    "d": torch.from_numpy(dvec),
                    "u": torch.from_numpy(met.unit(dvec)),
                    "mu": torch.from_numpy(mu),
                    "mu_neg": torch.from_numpy(neg.mean(axis=0)),
                    "sd_neg": torch.from_numpy(neg.std(axis=0, ddof=1)),
                    "sigma_act": torch.from_numpy(sigma),
                    "lopo_d": torch.from_numpy(lopo),
                    "pair_ids": m["pair_ids"], "n_pairs": n, "n_layers": Lp1 - 1,
                    "view_key": view["view_key"], "run_key": run.run_key},
                   run.artefact(".pt"))
        print(f"{args.direction}: n={n}  layers=0..{Lp1 - 1}  d={d}  band={band[0]}-{band[-1]}")
        print(f"  ||d|| mid-band {np.linalg.norm(dvec[band[len(band) // 2]]):.2f}   "
              f"sigma_act {sigma[band[len(band) // 2]]:.1f}")
        if curve:
            run.artefact("_curve.json", kind="csv").write_text(
                json.dumps(curve, indent=2), encoding="utf-8")
            c = curve["cos_n_vs_full"]
            shown = [k for k in c if int(k) % 100 == 0] or list(c)
            print("  cos(d_n, d_full) band-mean: "
                  + "  ".join(f"n{k}={c[k]['band_mean_cos']:.3f}" for k in shown))
            ho_cos = curve["cos_train_vs_heldout"]
            if ho_cos:
                print(f"  cos(d_train, d_heldout) band-mean {ho_cos['band_mean_cos']:+.3f}  "
                      f"(null band +/-{curve['cos_null_band']:.3f})")


if __name__ == "__main__":
    main()
