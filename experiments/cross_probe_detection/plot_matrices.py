"""The four chosen-layer matrices, from csv/ into figures/. No recomputation.

    python plot_matrices.py <model> --tag 1K_per_direction

Needs `cross_auroc.py --layers` and `geometry.py --layers`.

A probe is a (vector, layer) pair, so an axis given two chosen layers (`a=L1+L2`) is two
probe rows of the same vector. Which side of a matrix it lands on depends on what the
cell reads:

`_auroc.png` / `_cohens_dz.png`   probe (row, at its own layer) x axis (column: a
                                  *dataset*, so a second layer is a row only)
`_excess_over_null.png`           the same AUROC net of the random-direction null
`_cos_own.png`                    cos(d_row[L_row], d_col[L_col]) -- two different bases;
                                  probe x probe, so a second layer is row and column
`_cos_matched.png`                cos(d_row[L_col], d_col[L_col]) -- one basis; the row's
                                  layer is unused, so a second layer is a column only
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def heatmap(M, row_lab, col_lab, title, cbar, path, vmin, vmax, fmt="{:+.3f}",
            boxes=()):
    """`boxes` = the self-cells: same axis on both sides, not a cross-axis claim.

    Not the literal diagonal any more -- a second chosen layer makes the matrices
    non-square, and in `_cos_matched` a row can have two self-cells.
    """
    nr, nc = len(row_lab), len(col_lab)
    boxes = set(boxes)
    fig, ax = plt.subplots(figsize=(1.55 * nc + 2.6, 1.35 * nr + 2.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(nc), col_lab, fontsize=8)
    ax.set_yticks(range(nr), row_lab, fontsize=8)
    ax.set_xlabel("axis (evaluated on)", fontsize=9)
    ax.set_ylabel("probe", fontsize=9)
    span = max(vmax - vmin, 1e-9)
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if v != v:
                continue
            far = abs((v - 0.5 * (vmin + vmax)) / (0.5 * span))
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                    color="white" if far > 0.65 else "0.1",
                    fontweight="bold" if (i, j) in boxes else "normal")
    for i, j in boxes:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="0.15", lw=1.6))
    ax.set_title(title, fontsize=11, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cell_matrices(rows, entries, axes, diag_kind, keys):
    """{key: [probe x axis]}, each cell read at the row probe's own chosen layer."""
    ridx = {e: i for i, e in enumerate(entries)}
    cidx = {a: j for j, a in enumerate(axes)}
    out = {k: np.full((len(entries), len(axes)), np.nan) for k in keys}
    for r in rows:
        e, a = (r["probe"], int(r["layer"])), r["axis"]
        if e not in ridx or a not in cidx:
            continue
        if r["cell_type"] not in ("offdiag_pooled", diag_kind):
            continue
        for k in keys:
            out[k][ridx[e], cidx[a]] = float(r[k])
    return out


def cos_matrix(rows, row_keys, col_entries, convention, keyed_rows):
    """`keyed_rows` = rows are (axis, layer) probes; otherwise rows are bare axes."""
    ridx = {e: i for i, e in enumerate(row_keys)}
    cidx = {e: j for j, e in enumerate(col_entries)}
    M = np.full((len(row_keys), len(col_entries)), np.nan)
    for r in rows:
        if r["convention"] != convention:
            continue
        rk = (r["axis_row"], int(r["layer_row"])) if keyed_rows else r["axis_row"]
        ck = (r["axis_col"], int(r["layer_col"]))
        if rk in ridx and ck in cidx:
            M[ridx[rk], cidx[ck]] = float(r["cos"])
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

    # {axis: layer} in older manifests, {axis: [layers]} once an axis could have two.
    def as_entries(d, order):
        out = []
        for a in order:
            ls = d.get(a)
            for l in ([ls] if isinstance(ls, (int, str)) else ls or []):
                out.append((a, int(l)))
        return out

    axes = [a for a in ca["config"]["axes"] if a in ca["config"]["probe_layers"]]
    entries = as_entries(ca["config"]["probe_layers"], axes)
    if entries != as_entries(ge["config"]["chosen_layers"], axes):
        raise SystemExit("cross_auroc and geometry disagree on the chosen layers")

    diag = ca["config"]["diag"]
    diag_kind = "diag_lopo" if "lopo_train" in diag else "diag_heldout"
    diag_note = ("diagonal = LOPO on train" if diag_kind == "diag_lopo" else
                 "diagonal = the deployed vector on the held-out split")
    M = cell_matrices(read_rows(lay.csv / "cross_auroc_chosen.csv"), entries, axes,
                      diag_kind, ["auroc", "cohens_dz", "excess_over_null"])
    A, D, E = M["auroc"], M["cohens_dz"], M["excess_over_null"]
    cos_rows = read_rows(lay.csv / "geometry_cos_chosen.csv")
    C_own = cos_matrix(cos_rows, entries, entries, "own_layer", keyed_rows=True)
    C_mat = cos_matrix(cos_rows, axes, entries, "matched_to_col", keyed_rows=False)

    # The layer belongs on the axis it is operative for. In a cell matrix everything
    # happens at the *row's* layer, so labelling the columns with theirs would name a
    # number the off-diagonal cells never use.
    lab = [f"{a}\nL{l}" for a, l in entries]
    plain = list(axes)
    # Self-cells: the same axis on both sides, wherever it sits. With two chosen layers
    # these are no longer the literal diagonal.
    box_cell = [(i, axes.index(a)) for i, (a, _) in enumerate(entries)]
    box_own = [(i, j) for i, e in enumerate(entries) for j, f in enumerate(entries)
               if e == f]
    box_mat = [(axes.index(a), j) for j, (a, _) in enumerate(entries)]
    null = ge["config"]["cos_null_band"]
    dz_max = float(np.nanmax(np.abs(D)))
    ex_max = float(np.nanmax(np.abs(E)))
    stem = mf.stem("plot_matrices")
    config = {"axes": axes, "chosen_layers": ca["config"]["probe_layers"]}
    inputs = {"cross_auroc_run_key": ca["run_key"], "geometry_run_key": ge["run_key"]}

    with mf.Run(lay, stem, config, inputs) as run:
        made = [
            (run.artefact("_auroc.png"), A, lab, plain, box_cell,
             "paired AUROC at the chosen layers", "AUROC", 0.0, 1.0, "{:.3f}"),
            (run.artefact("_excess_over_null.png"), E, lab, plain, box_cell,
             "AUROC net of the random-direction null",
             "excess", -ex_max, ex_max, "{:+.3f}"),
            (run.artefact("_cohens_dz.png"), D, lab, plain, box_cell,
             "Cohen's $d_z$ at the chosen layers",
             r"$d_z$", -dz_max, dz_max, "{:+.2f}"),
            (run.artefact("_cos_own.png"), C_own, lab, lab, box_own,
             "cos between the chosen vectors", "cosine", -1.0, 1.0, "{:+.3f}"),
            (run.artefact("_cos_matched.png"), C_mat, plain, lab, box_mat,
             "cosine similarity at the column's chosen layer",
             "cosine", -1.0, 1.0, "{:+.3f}"),
        ]
        for path, M, rl, cl, bx, title, cbar, lo, hi, fmt in made:
            heatmap(M, rl, cl, title, cbar, path, lo, hi, fmt, bx)
            print(f"  {Path(path).relative_to(lay.root).as_posix()}")

    # The subtitles are gone from the figures, so the reading conventions they carried are
    # printed here instead -- they are what the off-diagonal cells mean.
    print(f"\n  cells read at the row's layer; {diag_note}, "
          f"off-diagonal = pooled train+heldout")
    print(f"  cos null band ±{null:.3f}")
    print("  chosen layers: " + " ".join(f"{a}=L{l}" for a, l in entries))
    off = C_mat.copy()
    for i, j in box_mat:
        off[i, j] = np.nan
    print(f"  strongest off-diagonal |cos| (matched): {np.nanmax(np.abs(off)):.3f}   "
          f"null ±{null:.3f}")


if __name__ == "__main__":
    main()
