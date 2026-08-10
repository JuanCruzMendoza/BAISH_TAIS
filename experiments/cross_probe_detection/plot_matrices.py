"""The four chosen-layer matrices, from csv/ into figures/. No recomputation.

    python plot_matrices.py <model> --tag 1K_per_direction

Needs `cross_auroc.py --layers` and `geometry.py --layers`.

`_auroc.png` / `_cohens_dz.png`   probe (row, at its own layer) x axis (column)
`_excess_over_null.png`           the same AUROC net of the random-direction null
`_cos_own.png`                    cos(d_row[L_row], d_col[L_col]) -- two different bases
`_cos_matched.png`                cos(d_row[L_col], d_col[L_col]) -- one basis
"""
import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def heatmap(M, row_lab, col_lab, title, sub, cbar, path, vmin, vmax, fmt="{:+.3f}"):
    n = len(row_lab)
    fig, ax = plt.subplots(figsize=(1.55 * n + 2.6, 1.35 * n + 2.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n), col_lab, fontsize=8)
    ax.set_yticks(range(n), row_lab, fontsize=8)
    ax.set_xlabel("axis (evaluated on)", fontsize=9)
    ax.set_ylabel("probe", fontsize=9)
    span = max(vmax - vmin, 1e-9)
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if v != v:
                continue
            far = abs((v - 0.5 * (vmin + vmax)) / (0.5 * span))
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if far > 0.65 else "0.1",
                    fontweight="bold" if i == j else "normal")
    for i in range(n):                       # the diagonal is not a cross-axis claim
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="0.15", lw=1.6))
    # ~11 chars per inch at this font size; a one-line subtitle overruns the axes.
    wrapped = textwrap.fill(sub, int(11 * fig.get_figwidth()))
    ax.set_title(title, fontsize=11, pad=12 + 11 * (wrapped.count("\n") + 1))
    ax.text(0.5, 1.01, wrapped, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color="0.3")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cell_matrices(rows, axes, layers, diag_kind, keys):
    """{key: [probe x axis]} at the probe's own chosen layer."""
    idx = {a: i for i, a in enumerate(axes)}
    out = {k: np.full((len(axes), len(axes)), np.nan) for k in keys}
    for r in rows:
        p, a = r["probe"], r["axis"]
        if p not in idx or a not in idx:
            continue
        if r["cell_type"] not in ("offdiag_pooled", diag_kind):
            continue
        if int(r["layer"]) != layers[p]:
            continue
        for k in keys:
            out[k][idx[p], idx[a]] = float(r[k])
    return out


def cos_matrix(rows, axes, convention):
    idx = {a: i for i, a in enumerate(axes)}
    M = np.full((len(axes), len(axes)), np.nan)
    for r in rows:
        if r["convention"] != convention:
            continue
        M[idx[r["axis_row"]], idx[r["axis_col"]]] = float(r["cos"])
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    lay = cfg.Layout("cross_probe_detection", args.model, args.tag, acts_cache=False)
    ca = mf.load_upstream(lay.meta / "cross_auroc_manifest.json")
    ge = mf.load_upstream(lay.meta / "geometry_manifest.json")
    if ca["config"]["layer_rule"] != "explicit":
        raise SystemExit("cross_auroc was not run with --layers; nothing to plot")
    if not ge["config"].get("chosen_layers"):
        raise SystemExit("geometry was not run with --layers; no _cos_chosen.csv")

    layers = {a: int(l) for a, l in ca["config"]["probe_layers"].items()}
    axes = [a for a in ca["config"]["axes"] if a in layers]
    if layers != {a: int(l) for a, l in ge["config"]["chosen_layers"].items()}:
        raise SystemExit("cross_auroc and geometry disagree on the chosen layers")

    diag = ca["config"]["diag"]
    diag_kind = "diag_lopo" if "lopo_train" in diag else "diag_heldout"
    diag_note = ("diagonal = LOPO on train" if diag_kind == "diag_lopo" else
                 "diagonal = the deployed vector on the held-out split")
    M = cell_matrices(read_rows(lay.csv / "cross_auroc_chosen.csv"), axes, layers,
                      diag_kind, ["auroc", "cohens_dz", "excess_over_null"])
    A, D, E = M["auroc"], M["cohens_dz"], M["excess_over_null"]
    cos_rows = read_rows(lay.csv / "geometry_cos_chosen.csv")
    C_own = cos_matrix(cos_rows, axes, "own_layer")
    C_mat = cos_matrix(cos_rows, axes, "matched_to_col")

    # The layer belongs on the axis it is operative for. In a cell matrix everything
    # happens at the *row's* layer, so labelling the columns with theirs would name a
    # number the off-diagonal cells never use.
    lab = [f"{a}\nL{layers[a]}" for a in axes]
    plain = list(axes)
    null = ge["config"]["cos_null_band"]
    dz_max = float(np.nanmax(np.abs(D)))
    ex_max = float(np.nanmax(np.abs(E)))
    stem = mf.stem("plot_matrices")
    config = {"axes": axes, "chosen_layers": layers}
    inputs = {"cross_auroc_run_key": ca["run_key"], "geometry_run_key": ge["run_key"]}

    with mf.Run(lay, stem, config, inputs) as run:
        made = [
            (run.artefact("_auroc.png"), A, lab, plain, "paired AUROC at the chosen layers",
             f"every cell read at the row's layer; {diag_note}, "
             f"off-diagonal = pooled train+heldout", "AUROC", 0.0, 1.0, "{:.3f}"),
            (run.artefact("_excess_over_null.png"), E, lab, plain,
             "AUROC net of the random-direction null",
             "folded AUROC − what 20 random unit directions score on that axis at the "
             "row's layer; ≤0 = no better than an arbitrary direction",
             "excess", -ex_max, ex_max, "{:+.3f}"),
            (run.artefact("_cohens_dz.png"), D, lab, plain,
             "Cohen's $d_z$ at the chosen layers",
             "same cells as the AUROC matrix; sign is the direction of the read",
             r"$d_z$", -dz_max, dz_max, "{:+.2f}"),
            (run.artefact("_cos_own.png"), C_own, lab, lab,
             "cos between the chosen vectors",
             f"each vector at its own chosen layer — different bases; "
             f"null ±{null:.3f}", "cosine", -1.0, 1.0, "{:+.3f}"),
            (run.artefact("_cos_matched.png"), C_mat, plain, lab,
             "cos at the column's chosen layer",
             f"row vector re-read at the column's layer — one basis; "
             f"null ±{null:.3f}", "cosine", -1.0, 1.0, "{:+.3f}"),
        ]
        for path, M, rl, cl, title, sub, cbar, lo, hi, fmt in made:
            heatmap(M, rl, cl, title, sub, cbar, path, lo, hi, fmt)
            print(f"  {Path(path).relative_to(lay.root).as_posix()}")

    print("\n  chosen layers: " + " ".join(f"{a}=L{layers[a]}" for a in axes))
    print(f"  strongest off-diagonal |cos| (matched): "
          f"{np.nanmax(np.abs(C_mat - np.eye(len(axes)))):.3f}   null ±{null:.3f}")


if __name__ == "__main__":
    main()
