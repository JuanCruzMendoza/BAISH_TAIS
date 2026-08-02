"""Spec 2.2: probe x axis paired AUROC + a random null row (H1). CPU only.

    python cross_auroc.py <model>

Off-diagonal cells pool the target axis's train + held-out pairs (65, spec 0.7.3):
the probe was never fitted on any of that axis's data, so its 50 train pairs are as
out-of-sample as its 15 held-out ones. Diagonal cells are LOPO on train (n=50) and
the held-out 15, reported separately -- never the in-sample pooled number.
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
    15 out of every selection.
    """
    nt = ax["n_train"]
    return [float(np.mean(met.unit(ax["pos"][:nt, l, :] - ax["neg"][:nt, l, :]) @ u_ax[l]))
            for l in range(Lp1)]


def load_axis(src, axis):
    """Pooled train + held-out, train first so the LOPO slice is [:n_train]."""
    tv, hv = views.read_view(src, axis, "train"), views.read_view(src, axis, "heldout")
    a, b = acts.load_view_matrix(src, tv), acts.load_view_matrix(src, hv)
    return {"pos": np.concatenate([a["pos"], b["pos"]]),
            "neg": np.concatenate([a["neg"], b["neg"]]),
            "n_train": a["pos"].shape[0],
            "view_keys": {"train": tv["view_key"], "heldout": hv["view_key"]}}


# --------------------------------------------------------------------- cells

def cell(pos, neg):
    """One AUROC cell: exact interval + the sign-free leakage magnitude."""
    ci = met.auroc_ci(pos, neg)
    return {"auroc": ci["auroc"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "wins": ci["wins"], "ties": ci["ties"], "n": ci["n"],
            # Sign-free: a probe reading a rival at 0.17 reads it as strongly as at
            # 0.83, just inverted.
            "auroc_folded": max(ci["auroc"], 1.0 - ci["auroc"])}


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


def tensor_rows(probes, axes, Lp1, null_folded):
    rows = []
    for pname, probe in probes.items():
        u = probe["u"].numpy()
        lopo_u = met.unit(probe["lopo_d"].numpy()) if "lopo_d" in probe else None
        for aname, ax in axes.items():
            sp, sn = scores(ax["pos"], u), scores(ax["neg"], u)
            nt = ax["n_train"]
            u_ax = probes[aname]["u"].numpy() if aname in probes else None
            for l in range(Lp1):
                nf = null_folded.get((aname, l), float("nan"))
                base = {"probe": pname, "axis": aname, "layer": l,
                        "depth": round(l / (Lp1 - 1), 4), "null_folded": nf,
                        # Side by side with the AUROC on purpose: a paired AUROC only
                        # needs a consistent *sign*, so a probe with cos 0.09 to an axis
                        # can still rank 64/65 pairs correctly (spec 2.1).
                        "cos_probe_axis": (met.cos(u[l], u_ax[l]) if u_ax is not None
                                           else float("nan"))}
                if pname != aname:
                    c = cell(sp[:, l], sn[:, l])
                    rows.append({**base, "cell_type": "offdiag_pooled", **c,
                                 # The reference for an off-diagonal cell is not 0.5:
                                 # it is what a random direction already gets on this
                                 # axis. Below zero = reads the rival axis *less* than
                                 # an arbitrary direction does.
                                 "excess_over_null": c["auroc_folded"] - nf,
                                 "delta_excluded": delta_excluded(c["ci_lo"], c["ci_hi"])})
                    continue
                # Diagonal: never the pooled in-sample number.
                lo = cell(np.einsum("nd,nd->n", ax["pos"][:nt, l, :], lopo_u[:, l, :]),
                          np.einsum("nd,nd->n", ax["neg"][:nt, l, :], lopo_u[:, l, :]))
                rows.append({**base, "cell_type": "diag_lopo", **lo,
                             "excess_over_null": lo["auroc_folded"] - nf,
                             "delta_excluded": None})
                ho = cell(sp[nt:, l], sn[nt:, l])
                rows.append({**base, "cell_type": "diag_heldout", **ho,
                             "excess_over_null": ho["auroc_folded"] - nf,
                             "delta_excluded": None})
    return rows


def null_rows(axes, Lp1, d_model, mpc, seed=cfg.SEED):
    """Random unit directions: what an axis gives up to *any* direction.

    The mean is 0.5 by symmetry and says nothing. `auroc_folded` is the statistic,
    and it tracks `axis_mean_paired_cos`: the more consistent an axis's contrast, the
    higher the AUROC an arbitrary direction earns on it, because paired AUROC only
    needs a shared sign. That is the ceiling on what this matrix can say about H1.
    """
    rng = np.random.default_rng(seed)
    r = met.unit(rng.normal(size=(N_NULL_DRAWS, Lp1, d_model)))
    rows = []
    for aname, ax in axes.items():
        for l in range(Lp1):
            a = np.array([met.paired_auroc(ax["pos"][:, l, :] @ r[k, l],
                                           ax["neg"][:, l, :] @ r[k, l])
                          for k in range(N_NULL_DRAWS)])
            rows.append({"probe": f"random_{seed}", "axis": aname, "layer": l,
                         "depth": round(l / (Lp1 - 1), 4), "cell_type": "null_pooled",
                         "auroc": float(a.mean()), "n": ax["pos"].shape[0],
                         "auroc_folded": float(np.maximum(a, 1 - a).mean()),
                         "auroc_sd": float(a.std(ddof=1)), "n_draws": N_NULL_DRAWS,
                         "axis_mean_paired_cos": mpc.get(aname, [float("nan")] * Lp1)[l]})
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
                    "band_mean_excess": float(np.mean([r.get("excess_over_null", float("nan"))
                                                       for r in inband]))})
    return sorted(out, key=lambda r: (r["probe"], r["axis"], r["cell_type"]))


def print_matrix(rows, axes, title):
    kinds = {(r["probe"], r["axis"]): r for r in rows if r["cell_type"] != "diag_heldout"}
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    print(f"\n{title}")
    print("  " + "probe".ljust(12) + "L".rjust(4) + "".join(a[:8].rjust(9) for a in axes))
    for p in probes:
        l = next((r["layer"] for r in rows if r["probe"] == p), "")
        cells = []
        for a in axes:
            r = kinds.get((p, a))
            cells.append("-".rjust(9) if r is None else
                         f"{r['auroc']:.3f}{'*' if p == a else ' '}".rjust(9))
        print("  " + p[:12].ljust(12) + str(l).rjust(4) + "".join(cells))
    print("  * diagonal = LOPO on train (n=50); off-diagonal = pooled train+heldout")


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
    args = ap.parse_args()

    src = cfg.acts_layout(args.model, args.tag)          # extraction: vectors + blobs
    lay = cfg.Layout("cross_probe_detection", args.model, args.tag, acts_cache=False)
    names = [a for a in args.axes.split(",") if a]

    # An axis needs both a fitted probe (for its diagonal) and cached train+heldout
    # views (for every column), so a direction that has not been run yet is skipped
    # rather than crashing the matrix.
    probes = {a: p for a in names if (p := load_probe(src, a)) is not None}
    axes = {}
    for a in list(probes):
        try:
            axes[a] = load_axis(src, a)
        except FileNotFoundError as e:                                    # noqa: PERF203
            print(f"! skipping {a}: {e}")
            del probes[a]
    names = [a for a in names if a in axes]
    skipped = [a for a in args.axes.split(",") if a and a not in axes]
    if skipped:
        print(f"! not in the matrix (no vector or no cached views): {skipped}")

    Lp1 = next(iter(probes.values()))["u"].shape[0]
    d_model = next(iter(probes.values()))["u"].shape[1]
    L = Lp1 - 1
    band = cfg.band(L)
    matched_l = round(args.matched_depth * L)

    # Own-best layer is decided here, at run time: the peak of mean_paired_cos inside
    # the reporting band. probe_select's `primary` is not used -- it maximises the same
    # quantity but only inside the length-gated band, and at 9 length pairs that gate
    # admits 1-2 layers, so it lands up to 9 layers off the peak.
    mpc = {a: axis_mpc(axes[a], probes[a]["u"].numpy(), Lp1) for a in probes}
    peak = {a: max(band, key=lambda l: mpc[a][l]) for a in probes}

    nulls = null_rows(axes, Lp1, d_model, mpc)
    tensor = tensor_rows(probes, axes, Lp1,
                         {(r["axis"], r["layer"]): r["auroc_folded"] for r in nulls}) + nulls

    stem = mf.stem("cross_auroc")
    config = {"axes": names, "probes": sorted(probes), "matched_depth": args.matched_depth,
              "matched_layer": matched_l, "layer_rule": "mpc_peak_in_band",
              "ownbest_layers": peak, "band": [band[0], band[-1]],
              "offdiag_pool": "train+heldout", "diag": ["lopo_train", "heldout"],
              "deltas": DELTAS, "n_null_draws": N_NULL_DRAWS,
              "interval": "clopper_pearson", "seed": cfg.SEED}
    inputs = {"view_keys": {a: axes[a]["view_keys"] for a in names},
              "direction_run_keys": {a: probes[a].get("run_key") for a in probes}}

    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact("_tensor.csv"), tensor)
        m_matched = matrix_rows(tensor, lambda p: matched_l, band)
        m_ownbest = matrix_rows(tensor, lambda p: peak.get(p, matched_l), band)
        write_csv(run.artefact("_matched.csv"), m_matched)
        write_csv(run.artefact("_ownbest.csv"), m_ownbest)

        print(f"{len(probes)} probes x {len(axes)} axes x {Lp1} layers   "
              f"band {band[0]}-{band[-1]}   n per axis: "
              + ", ".join(f"{a}={axes[a]['pos'].shape[0]}" for a in names))
        print_matrix(m_matched, names, f"(a) matched layer L{matched_l} "
                                      f"(depth {matched_l / L:.3f})")
        print_matrix(m_ownbest, names,
                     "(b) own-best layer = peak mean_paired_cos in band: "
                     + " ".join(f"{a}=L{peak[a]}({mpc[a][peak[a]]:.2f})" for a in probes))

        off = [r for r in m_matched if r["cell_type"] == "offdiag_pooled"]
        worst = max(off, key=lambda r: r["excess_over_null"])
        print(f"\n  strongest off-diagonal at the matched layer, net of the null: "
              f"{worst['probe']} -> {worst['axis']} {worst['auroc']:.3f} "
              f"[{worst['ci_lo']:.3f}, {worst['ci_hi']:.3f}]  "
              f"folded {worst['auroc_folded']:.3f} vs null {worst['null_folded']:.3f}")
        for d in DELTAS:
            k = sum(1 for r in off if (r["delta_excluded"] or 1.0) <= d)
            print(f"  equivalence: {k}/{len(off)} off-diagonal cells exclude delta={d:.2f}")
        k = sum(1 for r in off if r["excess_over_null"] <= 0)
        print(f"  {k}/{len(off)} off-diagonal cells read their rival axis no better than "
              f"a random direction")
        nul = [r for r in tensor if r["cell_type"] == "null_pooled" and r["layer"] in band]
        print("\n  random-direction null (band means) -- the real reference for a cell")
        print("    " + "axis".ljust(10) + "folded".rjust(8) + "axis_c".rjust(9))
        for a in names:
            v = np.mean([r["auroc_folded"] for r in nul if r["axis"] == a])
            c = np.mean([r["axis_mean_paired_cos"] for r in nul if r["axis"] == a])
            flag = "  <- any direction separates it" if v >= 0.75 else ""
            print(f"    {a:10s}{v:8.3f}{c:9.3f}{flag}")
        print("  folded null tracks axis_c: a consistent contrast is separable by any "
              "direction,\n  so this matrix cannot settle H1 on its own (spec 2.1) -- "
              "read geometry.py")


if __name__ == "__main__":
    main()
