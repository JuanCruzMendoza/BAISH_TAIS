"""Diff-in-means direction for one axis, every layer, from cached activations. CPU only.

    python extract_direction.py <model> --direction story

Writes directions__<axis>.pt with the full vector, the 50 LOPO vectors, pole means
and sigma_act, all of which sections 1.2, 2 and 5 reuse.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--direction", required=True, choices=views.DIRECTIONS)
    args = ap.parse_args()

    out = cfg.results_dir("extraction", args.model)
    view = views.read_view(args.model, args.direction, "train")
    m = acts.load_view_matrix(args.model, view)
    pos, neg = m["pos"], m["neg"]                       # [n, L+1, d]
    n, Lp1, d = pos.shape

    dvec = met.diff_in_means(pos, neg)                  # [L+1, d]
    lopo = met.lopo_directions(pos, neg)                # [n, L+1, d]
    pooled = np.concatenate([pos, neg])
    mu = pooled.mean(axis=0)                            # [L+1, d] pooled centre
    sigma = met.sigma_act(pooled)                       # [L+1]

    stem = mf.stem("directions", args.direction)
    config = {"direction": args.direction, "n_pairs": n, "position": "last_token",
              "estimator": "diff_in_means", "seed": cfg.SEED}
    inputs = {"view_key": view["view_key"], "source_files": view["source_files"]}

    with mf.Run(out, stem, config, inputs) as run:
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
        band = cfg.band(Lp1 - 1)
        print(f"{args.direction}: n={n}  layers=0..{Lp1 - 1}  d={d}  band={band[0]}-{band[-1]}")
        print(f"  ||d|| mid-band {np.linalg.norm(dvec[band[len(band) // 2]]):.2f}   "
              f"sigma_act {sigma[band[len(band) // 2]]:.1f}")


if __name__ == "__main__":
    main()
