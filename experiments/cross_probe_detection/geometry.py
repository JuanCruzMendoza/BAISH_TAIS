"""Spec 2.3: cosines against a within-axis floor, plus residual fractions.

    python geometry.py <model>
    python geometry.py <model> --tag 1K_per_direction \
        --layers story_v2_1k=23,persona_v2=15,harm_v2=21,eval_v2=9

Geometry is the primary H1 evidence (spec 2.1): the axes' datasets are disjoint and
their prompt lengths differ by an order of magnitude, so an off-diagonal AUROC null
could be distribution shift. Cosines are distribution-free.

The per-layer tables are within-layer only -- a cosine across layers has no basis
vector in common. `--layers` adds `_cos_chosen.csv`, which reports both conventions
side by side precisely so the cross-layer one is never read alone.
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
STORY = ("story_v2", "story_v2_1k")     # reverse-residual anchor; the one present wins


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


def reliabilities(halves, Lp1):
    return {a: [met.spearman_brown(met.cos(d1[l], d2[l])) for l in range(Lp1)]
            for a, (d1, d2, _) in halves.items()}


def cos_rows(halves, probes, Lp1):
    rel = reliabilities(halves, Lp1)
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


def cos_chosen_rows(probes, chosen, rel):
    """The chosen-layer cosine matrix under both conventions.

    `own_layer`: each vector at its own chosen layer, cos(d_row[L_row], d_col[L_col]).
    Two different bases, so it is the *deployed* comparison, not a geometric one.
    `matched_to_col`: cos(d_row[L_col], d_col[L_col]) -- both at the column's layer,
    the only convention in which the cosine has its usual meaning.
    """
    rows = []
    for a in chosen:                                     # row = probe
        da = probes[a]["d"].numpy()
        for b in chosen:                                 # column = axis at its layer
            db, lb = probes[b]["d"].numpy(), chosen[b]
            for conv, la in (("own_layer", chosen[a]), ("matched_to_col", lb)):
                c = met.cos(da[la], db[lb])
                ra, rb = rel[a][la], rel[b][lb]
                den = np.sqrt(ra * rb) if ra > 0 and rb > 0 else float("nan")
                rows.append({"axis_row": a, "axis_col": b, "convention": conv,
                             "layer_row": la, "layer_col": lb, "cos": c,
                             "reliability_row": ra, "reliability_col": rb,
                             "cos_disattenuated": float(c / den) if den == den
                                                  else float("nan")})
    return rows


def residual_rows(probes, Lp1, anchor=None):
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
        if anchor and a != anchor:
            bases.append((anchor, [anchor]))
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
    ap.add_argument("--layers", default=None,
                    help="axis=layer,... one chosen layer per direction (extraction "
                         "insights.md); adds _cos_chosen.csv")
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
    L = Lp1 - 1
    band = cfg.band(L)
    anchor = next((s for s in STORY if s in probes), None)

    chosen = cfg.parse_axis_layers(args.layers) if args.layers else None
    if chosen is not None:
        bad = [f"{a}=L{l}" for a, l in chosen.items() if not 0 <= l <= L]
        unknown = [a for a in chosen if a not in probes]
        missing = [a for a in probes if a not in chosen]
        if bad or unknown or missing:
            raise SystemExit(f"--layers: outside 0..{L} {bad}, unknown {unknown}, "
                             f"missing {missing}")

    self_r = selfsplit_rows(halves, Lp1)
    cos_r = cos_rows(halves, probes, Lp1)
    res_r = residual_rows(probes, Lp1, anchor)
    chosen_r = (cos_chosen_rows(probes, chosen, reliabilities(halves, Lp1))
                if chosen is not None else None)

    stem = mf.stem("geometry")
    config = {"axes": list(probes), "floor": "split_half_cos", "seed": cfg.SEED,
              "disattenuation": "spearman_brown", "band": [band[0], band[-1]],
              "residual_anchor": anchor, "chosen_layers": chosen,
              "cos_null_band": met.random_cos_band(d_model)}
    inputs = {"view_keys": view_keys,
              "direction_run_keys": {a: probes[a].get("run_key") for a in probes}}

    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact("_selfsplit.csv"), self_r)
        write_csv(run.artefact("_cos.csv"), cos_r)
        write_csv(run.artefact("_residual.csv"), res_r)
        if chosen_r is not None:
            write_csv(run.artefact("_cos_chosen.csv"), chosen_r)

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

        if anchor:
            print(f"\nresidual after projecting out {anchor} (1.0 = orthogonal)")
            for a in probes:
                s = bandmean(res_r, band, "resid_frac", axis=a, basis=anchor)
                if s == s:
                    print(f"  {a:12s} {s:.3f}")

        if chosen_r is not None:
            print("\nchosen-layer cosine, "
                  + " ".join(f"{a}=L{chosen[a]}" for a in chosen))
            for conv in ("own_layer", "matched_to_col"):
                print(f"  [{conv}]  " + "".join(a[:9].rjust(10) for a in chosen))
                for a in chosen:
                    cells = [next(r["cos"] for r in chosen_r
                                  if r["axis_row"] == a and r["axis_col"] == b
                                  and r["convention"] == conv) for b in chosen]
                    print("  " + a[:14].ljust(14)
                          + "".join(f"{c:+.3f}".rjust(10) for c in cells))


if __name__ == "__main__":
    main()
