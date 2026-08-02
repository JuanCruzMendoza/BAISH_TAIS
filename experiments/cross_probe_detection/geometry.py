"""Spec 2.3: cosines against a within-axis floor, plus residual fractions.

    python geometry.py <model>

Geometry is the primary H1 evidence (spec 2.1): the axes' datasets are disjoint and
their prompt lengths differ by an order of magnitude, so an off-diagonal AUROC null
could be distribution shift. Cosines are distribution-free.

All comparisons are within-layer -- cosines across layers are meaningless.
"""
import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views

SIBLING = {"story_v2": "story_v1", "story_v1": "story_v2"}


def load_probe(src, axis):
    stem = mf.stem("directions", axis)
    path = src.vectors / f"{stem}.pt"
    if not path.exists():
        return None
    mf.load_upstream(src.meta / f"{stem}_manifest.json")
    return torch.load(path, weights_only=False)


def split_halves(pos, neg, seed=cfg.SEED):
    """Two diff-in-means from disjoint halves: the noise floor for every cosine."""
    n = pos.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    h1, h2 = idx[:n // 2], idx[n // 2:]
    return (met.diff_in_means(pos[h1], neg[h1]), met.diff_in_means(pos[h2], neg[h2]),
            min(len(h1), len(h2)))


# --------------------------------------------------------------------- tables

def selfsplit_rows(halves, Lp1):
    """The calibration table: how well does an axis agree with itself?"""
    rows = []
    for a, (d1, d2, nh) in halves.items():
        for l in range(Lp1):
            s = met.cos(d1[l], d2[l])
            rows.append({"axis": a, "layer": l, "depth": round(l / (Lp1 - 1), 4),
                         "n_half": nh, "split_cos": s, "reliability": met.spearman_brown(s)})
    return rows


def cos_rows(halves, probes, Lp1):
    rel = {a: [met.spearman_brown(met.cos(d1[l], d2[l])) for l in range(Lp1)]
           for a, (d1, d2, _) in halves.items()}
    rows = []
    for a, b in combinations(probes, 2):
        da, db = probes[a]["d"].numpy(), probes[b]["d"].numpy()
        for l in range(Lp1):
            c = met.cos(da[l], db[l])
            ra, rb = rel[a][l], rel[b][l]
            den = np.sqrt(ra * rb) if ra > 0 and rb > 0 else float("nan")
            rows.append({"axis_a": a, "axis_b": b, "layer": l,
                         "depth": round(l / (Lp1 - 1), 4), "cos": c,
                         "reliability_a": ra, "reliability_b": rb,
                         # Attenuation-corrected: the cosine between the *true* axes,
                         # given that each estimate is itself noisy.
                         "cos_disattenuated": float(c / den) if den == den else float("nan")})
    return rows


def residual_rows(probes, Lp1):
    """resid_frac of each axis against the others, and of each rival against story.

    Every `others` basis is the same size: a story variant's sibling is dropped, and
    `story_v1` never enters anyone else's basis either. Otherwise the rivals would be
    projected onto 5 axes and story onto 4, and the extra near-collinear dimension
    would depress their residuals for free.
    """
    rows = []
    names = list(probes)
    canon = [b for b in names if b != "story_v1"]
    for a in names:
        da = probes[a]["d"].numpy()
        rivals = [b for b in canon if b != a and SIBLING.get(a) != b]
        bases = [("others", rivals)]
        if a != "story_v2" and "story_v2" in names:
            bases.append(("story_v2", ["story_v2"]))
        for label, basis in bases:
            if not basis:
                continue
            for l in range(Lp1):
                B = np.stack([probes[b]["d"].numpy()[l] for b in basis])
                rows.append({"axis": a, "basis": label, "basis_axes": "+".join(basis),
                             "layer": l, "depth": round(l / (Lp1 - 1), 4),
                             "resid_frac": met.residual_frac(da[l], B)})
    return rows


# --------------------------------------------------------------------- helpers

def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def bandmean(rows, band, key, **eq):
    v = [r[key] for r in rows if r["layer"] in band and all(r[k] == x for k, x in eq.items())]
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--axes", default=",".join(views.DIRECTIONS))
    args = ap.parse_args()

    src = cfg.acts_layout(args.model, args.tag)
    lay = cfg.Layout("cross_probe_detection", args.model, args.tag, acts_cache=False)
    names = [a for a in args.axes.split(",") if a]

    probes = {a: p for a in names if (p := load_probe(src, a)) is not None}
    if not probes:
        raise SystemExit("no direction vectors: run extract_direction.py first")

    halves, view_keys, n_pairs = {}, {}, {}
    for a in probes:
        v = views.read_view(src, a, "train")
        m = acts.load_view_matrix(src, v)
        halves[a] = split_halves(m["pos"], m["neg"])
        view_keys[a], n_pairs[a] = v["view_key"], m["pos"].shape[0]

    Lp1, d_model = next(iter(probes.values()))["u"].shape
    band = cfg.band(Lp1 - 1)

    self_r = selfsplit_rows(halves, Lp1)
    cos_r = cos_rows(halves, probes, Lp1)
    res_r = residual_rows(probes, Lp1)

    stem = mf.stem("geometry")
    config = {"axes": list(probes), "floor": "split_half_cos", "seed": cfg.SEED,
              "disattenuation": "spearman_brown", "band": [band[0], band[-1]],
              "cos_null_band": met.random_cos_band(d_model)}
    inputs = {"view_keys": view_keys,
              "direction_run_keys": {a: probes[a].get("run_key") for a in probes}}

    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact("_selfsplit.csv"), self_r)
        write_csv(run.artefact("_cos.csv"), cos_r)
        write_csv(run.artefact("_residual.csv"), res_r)

        print(f"{len(probes)} axes, d={d_model}, band {band[0]}-{band[-1]}, "
              f"cos null band +/-{met.random_cos_band(d_model):.3f}")
        print("\nself-agreement (band means) -- everything below is read against this")
        print("  " + "axis".ljust(10) + "n".rjust(4) + "split_cos".rjust(11)
              + "reliab".rjust(9) + "resid_others".rjust(14))
        for a in probes:
            print("  " + a.ljust(10) + str(n_pairs[a]).rjust(4)
                  + f"{bandmean(self_r, band, 'split_cos', axis=a):+.3f}".rjust(11)
                  + f"{bandmean(self_r, band, 'reliability', axis=a):.3f}".rjust(9)
                  + f"{bandmean(res_r, band, 'resid_frac', axis=a, basis='others'):.3f}".rjust(14))

        print("\ncosine (band mean) -> disattenuated")
        for a, b in combinations(probes, 2):
            c = bandmean(cos_r, band, "cos", axis_a=a, axis_b=b)
            dis = bandmean(cos_r, band, "cos_disattenuated", axis_a=a, axis_b=b)
            flag = "  <- inside the null band" if abs(c) < met.random_cos_band(d_model) else ""
            print(f"  {a:9s} {b:9s} {c:+.3f}  ->  {dis:+.3f}{flag}")

        print("\nresidual after projecting out story_v2 (1.0 = orthogonal)")
        for a in probes:
            s = bandmean(res_r, band, "resid_frac", axis=a, basis="story_v2")
            if s == s:
                print(f"  {a:10s} {s:.3f}")


if __name__ == "__main__":
    main()
