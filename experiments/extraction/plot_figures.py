"""Two figures per direction, from csv/ into figures/. No recomputation.

    python plot_figures.py <model>                                  # every direction present
    python plot_figures.py <model> --direction story_v2_1k
    python plot_figures.py <model> --direction story_v2_1k --layers 0,8,16,24,28

`_cos_curve.png`      cos(d_n, d_full) vs n pairs, one curve per layer (10 by default,
                      uniform over 0..L). Needs `extract_direction.py --curve`.
`_cohens_dz_train.png` cohens_dz_train vs layer.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf, metrics as met, views

N_CURVES = 10        # layers drawn on the cosine figure


def default_layers(L, k=N_CURVES):
    """k layers uniformly spaced over 0..L inclusive, both endpoints kept."""
    return sorted(dict.fromkeys(int(round(x)) for x in np.linspace(0, L, k)))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(r, key):
    v = r.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def plot_cos_curve(curve, axis, layers, path):
    ns = sorted((int(k) for k in curve["cos_n_vs_full"]))
    band = curve["band"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    colours = plt.get_cmap("viridis")(np.linspace(0, 0.92, len(layers)))
    for c, l in zip(colours, layers):
        y = [curve["cos_n_vs_full"][str(n)]["per_layer"][str(l)] for n in ns]
        ax.plot(ns, y, marker="o", ms=3, lw=1.4, color=c,
                label=f"L{l}" + ("*" if l in band else ""))
    ax.set_xlabel(f"pairs in the subsample (fit split n={curve['n_total']})")
    ax.set_ylabel(r"cos($d_n$, $d_\mathrm{full}$)")
    ax.set_title(f"{axis} — subsample saturation ({curve['seeds']} seeds per n)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, ncol=2, title=f"layer (* in band {band[0]}–{band[-1]})",
              title_fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cohens(rows, axis, band, path, with_heldout=False):
    xs = [int(r["layer"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axvspan(band[0], band[-1], color="0.85", zorder=0,
               label=f"band {band[0]}–{band[-1]}")
    ax.plot(xs, [num(r, "cohens_dz_train") for r in rows], marker="o", ms=3.5, lw=1.6,
            color="#1f4e79", label="cohens_dz_train")
    if with_heldout and "cohens_dz_heldout" in rows[0]:
        ax.plot(xs, [num(r, "cohens_dz_heldout") for r in rows], marker="s", ms=3,
                lw=1.2, ls="--", color="#c0504d", label="cohens_dz_heldout")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"Cohen's $d_z$")
    ax.set_title(f"{axis} — paired effect size by layer")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cos_vs_anchor(u, axis, anchor, band, null, path):
    """cos(d_anchor, d_l) across depth: is the axis one direction or several?"""
    L = u.shape[0] - 1
    ys = [met.cos(u[anchor], u[l]) for l in range(L + 1)]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axhspan(-null, null, color="0.88", zorder=0, label=f"null ±{null:.3f}")
    ax.axvspan(band[0], band[-1], color="#e8eef5", zorder=0,
               label=f"band {band[0]}–{band[-1]}")
    ax.axhline(0, lw=0.6, color="0.4")
    ax.plot(range(L + 1), ys, marker="o", ms=3.5, lw=1.6, color="#1f4e79")
    ax.axvline(anchor, ls=":", lw=1.2, color="#c0504d")
    ax.annotate(f"L{anchor}", (anchor, 1.0), textcoords="offset points",
                xytext=(4, -10), color="#c0504d", fontsize=8)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"cos(d$_{{L{anchor}}}$, d$_l$)")
    ax.set_title(f"{axis} — how far the axis rotates from L{anchor}")
    ax.set_ylim(min(-0.1, min(ys) - 0.05), 1.05)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return ys


def anchor_figure(lay, axis, anchor):
    pt = lay.vectors / f"{mf.stem('directions', axis)}.pt"
    if not pt.exists():
        raise SystemExit(f"{axis}: no {pt.name}; run extract_direction.py first")
    mf.load_upstream(lay.meta / f"{mf.stem('directions', axis)}_manifest.json")
    u = torch.load(pt, weights_only=False)["u"].numpy()
    L, d = u.shape[0] - 1, u.shape[1]
    if not 0 <= anchor <= L:
        raise SystemExit(f"{axis}: anchor L{anchor} outside 0..{L}")
    band, null = cfg.band(L), met.random_cos_band(d)

    stem = mf.stem("plot_anchor", axis)
    inputs = {"directions_run_key": mf.load_upstream(
        lay.meta / f"{mf.stem('directions', axis)}_manifest.json")["run_key"]}
    with mf.Run(lay, stem, {"direction": axis, "anchor": anchor}, inputs) as run:
        p = run.artefact(f"_cos_vs_L{anchor}.png")
        ys = plot_cos_vs_anchor(u, axis, anchor, band, null, p)
    inband = [ys[l] for l in band]
    print(f"  {Path(p).relative_to(lay.root).as_posix()}")
    print(f"    band-mean cos {np.mean(inband):+.3f}   min in band "
          f"L{band[int(np.argmin(inband))]}={min(inband):+.3f}   "
          f"L0={ys[0]:+.3f}  L{L}={ys[L]:+.3f}   null ±{null:.3f}")
    return p


def figures_for(lay, axis, args):
    table = lay.csv / f"{mf.stem('probe_select', axis)}.csv"
    if not table.exists():
        return None
    rows = sorted(read_rows(table), key=lambda r: int(r["layer"]))
    L = int(rows[-1]["layer"])
    summary = json.loads(
        (lay.csv / f"{mf.stem('probe_select', axis)}_summary.json").read_text(encoding="utf-8"))
    band = summary["band"]

    curve_path = lay.csv / f"{mf.stem('directions', axis)}_curve.json"
    curve = json.loads(curve_path.read_text(encoding="utf-8")) if curve_path.exists() else None
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else default_layers(L, args.n_curves))
    bad = [l for l in layers if not 0 <= l <= L]
    if bad:
        raise SystemExit(f"{axis}: layers {bad} outside 0..{L}")

    # The upstream run_keys, so check_stale flags a figure whose table was refitted.
    inputs = {"probe_select_run_key":
              mf.load_upstream(lay.meta / f"{mf.stem('probe_select', axis)}_manifest.json")["run_key"]}
    if curve is not None:
        inputs["directions_run_key"] = mf.load_upstream(
            lay.meta / f"{mf.stem('directions', axis)}_manifest.json")["run_key"]

    stem = mf.stem("plot", axis)
    config = {"direction": axis, "layers": layers, "with_heldout": args.with_heldout}
    made = []
    with mf.Run(lay, stem, config, inputs) as run:
        p = run.artefact("_cohens_dz_train.png")
        plot_cohens(rows, axis, band, p, with_heldout=args.with_heldout)
        made.append(p)
        if curve is None:
            run.notes.append("no _curve.json: cosine figure skipped")
            print(f"  ! {axis}: no curve json (extract_direction.py --curve), cosine figure skipped")
        else:
            p = run.artefact("_cos_curve.png")
            plot_cos_curve(curve, axis, layers, p)
            made.append(p)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--direction", action="append", choices=views.DIRECTIONS,
                    help="repeatable; default is every direction with a probe_select table")
    ap.add_argument("--layers", default=None,
                    help="comma-separated layers for the cosine figure (default: "
                         f"{N_CURVES} uniform over 0..L)")
    ap.add_argument("--n-curves", type=int, default=N_CURVES)
    ap.add_argument("--with-heldout", action="store_true",
                    help="add cohens_dz_heldout as a second series")
    ap.add_argument("--anchor", type=int, default=None,
                    help="instead of the usual pair, plot cos(d_anchor, d_l) across depth "
                         "for the chosen layer (own stem, does not touch the others)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    lay = cfg.Layout("extraction", args.model, args.tag)
    axes = args.direction or [a for a in views.DIRECTIONS
                              if (lay.csv / f"{mf.stem('probe_select', a)}.csv").exists()]
    if not axes:
        raise SystemExit(f"no probe_select tables under {lay.csv}")

    if args.anchor is not None:
        if len(axes) != 1:
            raise SystemExit("--anchor needs exactly one --direction (it is per-axis)")
        anchor_figure(lay, axes[0], args.anchor)
        return

    for axis in axes:
        made = figures_for(lay, axis, args)
        if made is None:
            print(f"  ! {axis}: no probe_select table, skipped")
            continue
        for p in made:
            print(f"  {Path(p).relative_to(lay.root).as_posix()}")


if __name__ == "__main__":
    main()
