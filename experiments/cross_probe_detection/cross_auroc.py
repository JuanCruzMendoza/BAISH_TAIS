"""Spec 2.2: probe x axis paired AUROC and Cohen's d_z, + a random null row (H1). CPU only.

    python cross_auroc.py <model>
    python cross_auroc.py <model> --tag 1K_per_direction \
        --layers story_v2_1k=23,persona_v2=15,harm_v2=21,eval_v2=9

Off-diagonal cells pool the target axis's train + held-out pairs (spec 0.7.3): the probe
was never fitted on any of that axis's data, so its train pairs are as out-of-sample as
its held-out ones. The diagonal is never the in-sample pooled number -- `--diag heldout`
scores the deployed vector on the held-out split alone, `--diag lopo` adds the LOPO row
that small n needs.

`--layers` fixes one layer per direction instead of the mean_paired_cos peak. Axes are
loaded and released one at a time: 1,000 pooled pairs x 29 layers x 3584 dims is ~0.8 GB
per axis.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views

MATCHED_DEPTH = 0.65      # spec 2.2(a): one common fractional depth for every probe
DELTAS = [0.60, 0.65, 0.70, 0.75]   # pre-registered equivalence bounds (spec 0.7)
N_NULL_DRAWS = 20         # one random direction has sd ~0.07 at n=65


# ------------------------------------------------------------------- upstream

def load_probe(src, axis):
    stem = mf.stem("directions", axis)
    path = src.vectors / f"{stem}.pt"
    if not path.exists():
        return None
    mf.load_upstream(src.meta / f"{stem}_manifest.json")
    return torch.load(path, weights_only=False)


def axis_mpc(ax, u_ax, Lp1):
    """mean_paired_cos per layer: how consistent this axis's contrast is.

    Train pairs only. It decides the own-best layer, and spec 0.7 keeps the held-out
    pairs out of every selection.
    """
    nt = ax["n_train"]
    return [float(np.mean(met.unit(ax["pos"][:nt, l, :] - ax["neg"][:nt, l, :]) @ u_ax[l]))
            for l in range(Lp1)]


def axis_meta(src, axis):
    """view_keys and pair counts, without touching the blobs."""
    tv, hv = views.read_view(src, axis, "train"), views.read_view(src, axis, "heldout")
    return {"view_keys": {"train": tv["view_key"], "heldout": hv["view_key"]}}


def load_axis(src, axis):
    """Pooled train + held-out, train first so the LOPO slice is [:n_train]."""
    tv, hv = views.read_view(src, axis, "train"), views.read_view(src, axis, "heldout")
    a, b = acts.load_view_matrix(src, tv), acts.load_view_matrix(src, hv)
    return {"pos": np.concatenate([a["pos"], b["pos"]]),
            "neg": np.concatenate([a["neg"], b["neg"]]),
            "n_train": a["pos"].shape[0]}


# --------------------------------------------------------------------- cells

def cell(pos, neg):
    """One cell. The Clopper-Pearson interval is computed and consumed here.

    It is not emitted: `delta_excluded` is the only thing the interval is read for,
    and CP is a pure function of (wins, ties, n), all three of which are columns --
    so any interval can be reconstructed exactly without carrying two more.

    Cohen's d_z is carried beside the AUROC because at 800 pairs the AUROC saturates
    and cannot rank anything; d_z keeps moving after it stops.
    """
    ci = met.auroc_ci(pos, neg)
    dz = met.cohens_dz(pos, neg)
    return {"auroc": ci["auroc"], "wins": ci["wins"], "ties": ci["ties"], "n": ci["n"],
            # Sign-free: a probe reading a rival at 0.17 reads it as strongly as at
            # 0.83, just inverted. Same for a d_z of -1.4.
            "auroc_folded": max(ci["auroc"], 1.0 - ci["auroc"]),
            "cohens_dz": dz, "cohens_dz_folded": abs(dz),
            "delta_excluded": delta_excluded(ci["ci_lo"], ci["ci_hi"])}


def delta_excluded(ci_lo, ci_hi):
    """Tightest pre-registered delta whose region [1-d, d] contains the interval.

    Two-sided, unlike spec 0.7's one-sided bound: absence has to exclude leakage of
    either sign, and the story axes read length *inverted* (extraction insights).
    """
    for d in DELTAS:
        if ci_lo >= 1.0 - d and ci_hi <= d:
            return d
    return None


def scores(m, u):
    """[n, L+1, d] x [L+1, d] -> [n, L+1]. Mean-free: paired AUROC is mu-invariant."""
    return np.einsum("nld,ld->nl", m, u)


def axis_rows(aname, ax, U, Lp1, nulls_a, diag="lopo"):
    """Every probe's rows for one axis. `nulls_a` is that axis's null row per layer."""
    nt = ax["n_train"]
    nf = {r["layer"]: (r["auroc_folded"], r["cohens_dz_folded"]) for r in nulls_a}
    rows = []
    for pname, u in U.items():
        sp, sn = scores(ax["pos"], u), scores(ax["neg"], u)
        for l in range(Lp1):
            n_a, n_dz = nf.get(l, (float("nan"), float("nan")))
            base = {"probe": pname, "axis": aname, "layer": l,
                    "depth": round(l / (Lp1 - 1), 4),
                    "null_folded": n_a, "null_dz_folded": n_dz,
                    # Side by side with the AUROC on purpose: a paired AUROC only
                    # needs a consistent *sign*, so a probe with cos 0.09 to an axis
                    # can still rank 64/65 pairs correctly (spec 2.1).
                    "cos_probe_axis": met.cos(u[l], U[aname][l]) if aname in U
                                      else float("nan")}
            if pname != aname:
                c = cell(sp[:, l], sn[:, l])
                rows.append({**base, "cell_type": "offdiag_pooled", **c,
                             # The reference for an off-diagonal cell is not 0.5:
                             # it is what a random direction already gets on this
                             # axis. Below zero = reads the rival axis *less* than
                             # an arbitrary direction does.
                             "excess_over_null": c["auroc_folded"] - n_a,
                             "excess_dz_over_null": c["cohens_dz_folded"] - n_dz})
                continue
            # Diagonal: never the pooled in-sample number. LOPO is recomputed here
            # from this axis's own train rows -- it is a closed-form update of the pole
            # sums, so it does not have to be carried in the probe .pt. At 800 train
            # pairs it moves d_z by ~0.005, ~100x below its SE, so `--diag heldout`
            # drops it and lets the deployed vector answer for itself.
            if diag == "lopo":
                lopo_ul = met.unit(met.lopo_directions(ax["pos"][:nt, l, :],
                                                       ax["neg"][:nt, l, :]))
                lo = cell(np.einsum("nd,nd->n", ax["pos"][:nt, l, :], lopo_ul),
                          np.einsum("nd,nd->n", ax["neg"][:nt, l, :], lopo_ul))
                rows.append({**base, "cell_type": "diag_lopo", **lo,
                             "excess_over_null": lo["auroc_folded"] - n_a,
                             "excess_dz_over_null": lo["cohens_dz_folded"] - n_dz,
                             "delta_excluded": None})  # absence is not a diagonal claim
            ho = cell(sp[nt:, l], sn[nt:, l])
            rows.append({**base, "cell_type": "diag_heldout", **ho,
                         "excess_over_null": ho["auroc_folded"] - n_a,
                         "excess_dz_over_null": ho["cohens_dz_folded"] - n_dz,
                         "delta_excluded": None})
    return sorted(rows, key=lambda r: (r["probe"], r["layer"], r["cell_type"]))


def null_rows(aname, ax, R, Lp1, mpc_a):
    """Random unit directions: what an axis gives up to *any* direction.

    The mean AUROC is 0.5 by symmetry and says nothing. The folded columns are the
    statistic, and they track `axis_mean_paired_cos`: the more consistent an axis's
    contrast, the more an arbitrary direction earns on it, because a paired statistic
    only needs a shared sign. That is the ceiling on what this matrix can say about H1.
    """
    rows = []
    for l in range(Lp1):
        a = np.empty(len(R))
        dz = np.empty(len(R))
        for k in range(len(R)):
            sp, sn = ax["pos"][:, l, :] @ R[k, l], ax["neg"][:, l, :] @ R[k, l]
            a[k], dz[k] = met.paired_auroc(sp, sn), met.cohens_dz(sp, sn)
        rows.append({"probe": f"random_{cfg.SEED}", "axis": aname, "layer": l,
                     "depth": round(l / (Lp1 - 1), 4), "cell_type": "null_pooled",
                     "auroc": float(a.mean()), "n": ax["pos"].shape[0],
                     "auroc_folded": float(np.maximum(a, 1 - a).mean()),
                     "cohens_dz": float(dz.mean()),
                     "cohens_dz_folded": float(np.abs(dz).mean()),
                     "auroc_sd": float(a.std(ddof=1)), "n_draws": len(R),
                     "axis_mean_paired_cos": mpc_a[l]})
    return rows


# ------------------------------------------------------------------- matrices

def matrix_rows(tensor, layer_of, band):
    """One row per probe x axis x cell_type at the chosen layer, + band summaries."""
    by_key = {}
    for r in tensor:
        by_key.setdefault((r["probe"], r["axis"], r["cell_type"]), []).append(r)
    out = []
    for (p, a, kind), rs in by_key.items():
        l = layer_of(p)
        pick = next((r for r in rs if r["layer"] == l), None)
        if pick is None:
            continue
        inband = [r for r in rs if r["layer"] in band]
        out.append({**pick,
                    "band_mean_auroc": float(np.mean([r["auroc"] for r in inband])),
                    "band_max_folded": float(np.max([r["auroc_folded"] for r in inband])),
                    "band_mean_cohens_dz": float(np.mean([r["cohens_dz"] for r in inband])),
                    "band_mean_excess": float(np.mean([r.get("excess_over_null", float("nan"))
                                                       for r in inband]))})
    return sorted(out, key=lambda r: (r["probe"], r["axis"], r["cell_type"]))


def print_matrix(rows, axes, title, diag_kind, key="auroc", fmt="{:.3f}"):
    kinds = {(r["probe"], r["axis"]): r for r in rows
             if not r["cell_type"].startswith("diag") or r["cell_type"] == diag_kind}
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    print(f"\n{title}")
    print("  " + "probe".ljust(14) + "L".rjust(4) + "".join(a[:9].rjust(10) for a in axes))
    for p in probes:
        l = next((r["layer"] for r in rows if r["probe"] == p), "")
        cells = []
        for a in axes:
            r = kinds.get((p, a))
            cells.append("-".rjust(10) if r is None else
                         (fmt.format(r[key]) + ("*" if p == a else " ")).rjust(10))
        print("  " + p[:14].ljust(14) + str(l).rjust(4) + "".join(cells))
    print(f"  * diagonal = {'LOPO on train' if diag_kind == 'diag_lopo' else 'held-out only'}"
          f"; off-diagonal = pooled train+heldout")


# --------------------------------------------------------------------- helpers

def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--matched-depth", type=float, default=MATCHED_DEPTH)
    ap.add_argument("--axes", default=",".join(views.DIRECTIONS))
    ap.add_argument("--layers", default=None,
                    help="axis=layer,... one chosen layer per direction (extraction "
                         "insights.md); default is the mean_paired_cos peak in band")
    ap.add_argument("--diag", choices=["lopo", "heldout"], default="lopo",
                    help="diagonal cell: LOPO on train + the held-out split, or the "
                         "deployed vector on the held-out split alone")
    args = ap.parse_args()

    src = cfg.acts_layout(args.model, args.tag)          # extraction: vectors + blobs
    lay = cfg.Layout("cross_probe_detection", args.model, args.tag, acts_cache=False)
    names = [a for a in args.axes.split(",") if a]

    # An axis needs both a fitted probe (for its diagonal) and cached train+heldout
    # views (for every column), so a direction that has not been run yet is skipped
    # rather than crashing the matrix.
    probes = {a: p for a in names if (p := load_probe(src, a)) is not None}
    meta = {}
    for a in list(probes):
        try:
            meta[a] = axis_meta(src, a)
        except FileNotFoundError as e:                                    # noqa: PERF203
            print(f"! skipping {a}: {e}")
            del probes[a]
    names = [a for a in names if a in probes]
    skipped = [a for a in args.axes.split(",") if a and a not in probes]
    if skipped:
        print(f"! not in the matrix (no vector or no cached views): {skipped}")

    Lp1, d_model = next(iter(probes.values()))["u"].shape
    L = Lp1 - 1
    band = cfg.band(L)
    matched_l = round(args.matched_depth * L)

    chosen = cfg.parse_axis_layers(args.layers) if args.layers else None
    if chosen is not None:
        bad = [f"{a}=L{l}" for a, l in chosen.items() if not 0 <= l <= L]
        unknown = [a for a in chosen if a not in probes]
        missing = [a for a in names if a not in chosen]
        if bad or unknown or missing:
            raise SystemExit(f"--layers: outside 0..{L} {bad}, unknown {unknown}, "
                             f"missing {missing}")

    U = {a: probes[a]["u"].numpy() for a in probes}
    # One draw set for every axis, so a null is comparable across columns.
    R = met.unit(np.random.default_rng(cfg.SEED).normal(size=(N_NULL_DRAWS, Lp1, d_model)))

    body, nulls, mpc, n_pooled = [], [], {}, {}
    for aname in names:
        ax = load_axis(src, aname)
        mpc[aname] = axis_mpc(ax, U[aname], Lp1)
        n_pooled[aname] = ax["pos"].shape[0]
        nul = null_rows(aname, ax, R, Lp1, mpc[aname])
        body += axis_rows(aname, ax, U, Lp1, nul, args.diag)
        nulls += nul
        del ax                       # ~0.8 GB per axis at 1,000 pooled pairs
    tensor = sorted(body, key=lambda r: (r["probe"], r["axis"], r["layer"],
                                         r["cell_type"])) + nulls

    peak = {a: max(band, key=lambda l: mpc[a][l]) for a in probes}
    probe_layers = chosen if chosen is not None else peak

    stem = mf.stem("cross_auroc")
    config = {"axes": names, "probes": sorted(probes), "matched_depth": args.matched_depth,
              "matched_layer": matched_l,
              "layer_rule": "explicit" if chosen is not None else "mpc_peak_in_band",
              "probe_layers": probe_layers, "band": [band[0], band[-1]],
              "offdiag_pool": "train+heldout",
              "diag": ["lopo_train", "heldout"] if args.diag == "lopo" else ["heldout"],
              "deltas": DELTAS, "n_null_draws": N_NULL_DRAWS,
              "interval": "clopper_pearson", "seed": cfg.SEED}
    inputs = {"view_keys": {a: meta[a]["view_keys"] for a in names},
              "direction_run_keys": {a: probes[a].get("run_key") for a in probes}}

    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact("_tensor.csv"), tensor)
        m_matched = matrix_rows(tensor, lambda p: matched_l, band)
        m_probe = matrix_rows(tensor, lambda p: probe_layers.get(p, matched_l), band)
        write_csv(run.artefact("_matched.csv"), m_matched)
        write_csv(run.artefact("_chosen.csv" if chosen is not None else "_ownbest.csv"),
                  m_probe)

        print(f"{len(probes)} probes x {len(names)} axes x {Lp1} layers   "
              f"band {band[0]}-{band[-1]}   n per axis: "
              + ", ".join(f"{a}={n_pooled[a]}" for a in names))
        dk = "diag_lopo" if args.diag == "lopo" else "diag_heldout"
        print_matrix(m_matched, names, f"(a) matched layer L{matched_l} "
                                      f"(depth {matched_l / L:.3f})", dk)
        rule = ("chosen layer (extraction insights.md): " if chosen is not None else
                "own-best layer = peak mean_paired_cos in band: ")
        head = rule + " ".join(f"{a}=L{probe_layers[a]}" for a in names)
        print_matrix(m_probe, names, "(b) AUROC, " + head, dk)
        print_matrix(m_probe, names, "(c) Cohen's d_z, " + head, dk,
                     key="cohens_dz", fmt="{:+.2f}")

        off = [r for r in m_matched if r["cell_type"] == "offdiag_pooled"]
        worst = max(off, key=lambda r: r["excess_over_null"])
        print(f"\n  strongest off-diagonal at the matched layer, net of the null: "
              f"{worst['probe']} -> {worst['axis']} {worst['auroc']:.3f}  "
              f"folded {worst['auroc_folded']:.3f} vs null {worst['null_folded']:.3f}")
        for d in DELTAS:
            k = sum(1 for r in off if (r["delta_excluded"] or 1.0) <= d)
            print(f"  equivalence: {k}/{len(off)} off-diagonal cells exclude delta={d:.2f}")
        k = sum(1 for r in off if r["excess_over_null"] <= 0)
        print(f"  {k}/{len(off)} off-diagonal cells read their rival axis no better than "
              f"a random direction")
        nul = [r for r in nulls if r["layer"] in band]
        print("\n  random-direction null (band means) -- the real reference for a cell")
        print("    " + "axis".ljust(14) + "folded".rjust(8) + "|dz|".rjust(8)
              + "axis_c".rjust(9))
        for a in names:
            rs = [r for r in nul if r["axis"] == a]
            v = np.mean([r["auroc_folded"] for r in rs])
            z = np.mean([r["cohens_dz_folded"] for r in rs])
            c = np.mean([r["axis_mean_paired_cos"] for r in rs])
            flag = "  <- any direction separates it" if v >= 0.75 else ""
            print(f"    {a:14s}{v:8.3f}{z:8.2f}{c:9.3f}{flag}")
        print("  folded null tracks axis_c: a consistent contrast is separable by any "
              "direction,\n  so this matrix cannot settle H1 on its own (spec 2.1) -- "
              "read geometry.py")


if __name__ == "__main__":
    main()
