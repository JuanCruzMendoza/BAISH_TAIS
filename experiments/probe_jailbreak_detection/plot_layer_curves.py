"""`pct_reads` vs layer, from csv/ into figures/. No recomputation.

    python plot_layer_curves.py <model> --tag 1K_per_direction

One figure per probe. `--per-family` probes get one curve per jailbreak family; every
other probe gets the `all` slice alone. `ref_tpr` rides along as a dashed grey line: it is
also a percentage of a set clearing the same tau, so it shares the axis, and where it
collapses the curve above it is not a claim about jailbreaks.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf

# Fixed so the two per-family figures are read against each other, and ordered
# most- to least-narrative rather than alphabetically.
FAMILIES = [("fiction_narrative", "#3b4cc0"), ("hybrid", "#00929c"),
            ("roleplay_persona", "#e8871a"), ("nonfiction_other", "#9a9a9a")]


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def series(rows, probe, kind, group):
    xy = sorted((int(r["layer"]), float(r["pct_reads"]), float(r["ref_tpr"]))
                for r in rows if r["probe"] == probe
                and r["group_kind"] == kind and r["group"] == group)
    return [p[0] for p in xy], [p[1] for p in xy], [100 * p[2] for p in xy]


def curves(rows, probe, per_family):
    if not per_family:
        x, y, t = series(rows, probe, "all", "all")
        return [(x, y, "all families", "#3b4cc0", 2.0)], t
    out, tpr = [], None
    for fam, colour in FAMILIES:
        x, y, t = series(rows, probe, "family", fam)
        if x:
            out.append((x, y, fam, colour, 1.7))
            tpr = t
    return out, tpr


def plot(rows, probe, layer, per_family, rule, band, path):
    cs, tpr = curves(rows, probe, per_family)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for x, y, label, colour, lw in cs:
        ax.plot(x, y, marker="o", ms=3.2, lw=lw, color=colour, label=label)
    if tpr:
        x0 = cs[0][0]
        ax.plot(x0, tpr, ls="--", lw=1.1, color="0.45", label="ref_tpr (the bar)")
    # A gap in x means a layer outside the band was scored because it was chosen.
    if layer is not None:
        ax.axvline(layer, color="0.2", lw=1.0, ls=":", zorder=0)
        # Inside the axes and rotated: at the band edge a centred label above the axes
        # runs into the title.
        ax.annotate(f" chosen L{layer}", (layer, 99), fontsize=8, color="0.25",
                    ha="left", va="top", rotation=90)
    ax.set_xlabel("layer")
    ax.set_ylabel(f"% of jailbreaks clearing $\\tau$  ({rule})")
    ax.set_title(f"{probe} — `pct_reads` vs layer", fontsize=11)
    ax.set_ylim(-3, 103)
    ax.set_xticks([l for l in cs[0][0] if l % 2 == int(band[0]) % 2] or cs[0][0])
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threshold", default="midpoint")
    ap.add_argument("--per-family", default="story_v2_1k,persona_v2",
                    help="probes drawn as one curve per family; the rest get `all`")
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    up = mf.load_upstream(lay.meta / f"jb_metrics__{args.threshold}_manifest.json")
    rows = read_rows(lay.csv / f"jb_metrics__{args.threshold}_rate.csv")

    probes = up["config"]["probes"]
    chosen = {a: int(l) for a, l in (up["config"].get("probe_layers") or {}).items()}
    band = up["config"]["band"]
    fam = [p for p in args.per_family.split(",") if p]
    unknown = [p for p in fam if p not in probes]
    if unknown:
        raise SystemExit(f"--per-family: not in the run: {unknown}; it holds {probes}")

    stem = mf.stem("plot_layer_curves")
    config = {"probes": probes, "threshold_rule": args.threshold, "per_family": fam,
              "chosen_layers": chosen or None}
    inputs = {"jb_metrics_run_key": up["run_key"]}

    with mf.Run(lay, stem, config, inputs) as run:
        for p in probes:
            path = run.artefact(f"_{p}.png")
            plot(rows, p, chosen.get(p), p in fam, args.threshold, band, path)
            print(f"  {Path(path).relative_to(lay.root).as_posix()}"
                  f"   {'4 families' if p in fam else 'all families'}")


if __name__ == "__main__":
    main()
