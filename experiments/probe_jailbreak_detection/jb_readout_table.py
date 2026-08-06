"""Spec 3: one row per jailbreak, one column per (direction, layer set) readout.

    python jb_readout_table.py <model> --sweep story_v2=15,17,18 --sweep harm=20,21,22
    python jb_readout_table.py <model> --band frac:0.70-0.90

The per-jailbreak view the other two scripts do not give: jb_readout.pt holds the raw
[probe, row, layer] tensor and is gitignored, jb_metrics aggregates to pct_reads per
probe x layer x group. This writes the numbers as a table keyed by row_id, with the
same layer sets steering_jailbreaks steers, so a cell's manipulation check and the
probe's own reading of that prompt sit in one place.

A single layer's column is its readout. A band's column is the **mean** over the
band's layers: raw projections are not comparable across depth, so that mean leans on
the deepest layers. It is the band's readout, not a depth-invariant score.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf

META_COLS = ("family", "source", "technique", "template_id", "category", "n_tokens")


def parse_sweep(specs, probes, L):
    """['story_v2=15,17,18', ...] -> {direction: [layers]}, order preserved.

    The spec is read with the layer grammar, so 15-18 and frac: work too, but every
    layer becomes its own column -- a sweep is one cell per layer (spec 5.4.0).
    """
    out = {}
    for s in specs or []:
        if "=" not in s:
            raise SystemExit(f"--sweep wants <direction>=<layers>, got {s!r}")
        name, spec = s.split("=", 1)
        name = name.strip()
        if name not in probes:
            raise SystemExit(f"no probe {name!r} in jb_readout.pt; have {probes}")
        out.setdefault(name, [])
        for l in cfg.parse_layers(spec.strip(), L):
            if l not in out[name]:
                out[name].append(l)
    return out


def build(framed, probes, rows, sweep, band_layers, band_stem, L):
    """-> (column names, [{row_id, ...meta, <dir>__<set>: readout}])."""
    idx = {a: i for i, a in enumerate(probes)}
    cols = []
    for a in sweep:
        cols += [(a, f"{a}__{cfg.layer_stem(str(l))}", [l]) for l in sweep[a]]
        cols.append((a, f"{a}__{band_stem}", band_layers))
    out = []
    for j, r in enumerate(rows):
        row = {"row_id": r["row_id"], **{c: r[c] for c in META_COLS if c in r}}
        for a, name, layers in cols:
            row[name] = round(float(framed[idx[a], j, layers].mean()), 4)
        out.append(row)
    return [c[1] for c in cols], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sweep", action="append", metavar="DIR=LAYERS",
                    help="repeatable; one column per layer. " + cfg.LAYER_SPEC)
    ap.add_argument("--band", default="steer_band",
                    help="one extra column per direction, the mean over these layers")
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    up = lay.vectors / "jb_readout.pt"
    if not up.exists():
        raise SystemExit("run jb_readout.py first")
    mf.load_upstream(lay.meta / "jb_readout_manifest.json")
    R = torch.load(up, weights_only=False)

    probes, framed, L = R["probes"], R["framed"], R["n_layers"]
    with (lay.csv / "jb_readout_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row_id"] for r in rows] == list(R["row_ids"]), "row order drifted"

    band_layers = cfg.parse_layers(args.band, L)
    sweep = parse_sweep(args.sweep, probes, L) or {a: [] for a in probes}
    band_stem = cfg.layer_stem(args.band)
    cols, table = build(framed, probes, rows, sweep, band_layers, band_stem, L)

    stem = mf.stem("jb_readout_table", band_stem)
    config = {"directions": list(sweep), "sweep_layers": sweep,
              "band_spec": args.band, "band_layers": band_layers,
              "band_column": "mean over band_layers", "n_rows": len(table),
              "readout": "(h - mu) . u_hat", "seed": cfg.SEED}
    inputs = {"jb_view_key": R["jb_view_key"], "jb_readout_run_key": R.get("run_key")}

    with mf.Run(lay, stem, config, inputs) as run:
        path = run.artefact(".csv")
        fields = ["row_id", *[c for c in META_COLS if c in rows[0]], *cols]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(table)

        print(f"{len(table)} jailbreaks x {len(cols)} readout columns -> {path.name}")
        print(f"  band {band_stem} = L{band_layers[0]}-{band_layers[-1]} "
              f"({len(band_layers)} layers), column is their mean\n")
        print("  " + "direction".ljust(10) + "single layers".ljust(18) + "band mean"
              + "  (band-mean readout over all rows)")
        for a in sweep:
            singles = ",".join(f"L{l}" for l in sweep[a]) or "-"
            bm = sum(r[f"{a}__{band_stem}"] for r in table) / len(table)
            print("  " + a.ljust(10) + singles.ljust(18) + f"{bm:+9.2f}")
        print("\n  Raw projections, uncentred across depth: a band column leans on its\n"
              "  deepest layers, and a readout is only 'reads as this direction' once cut\n"
              "  at a threshold -- jb_metrics owns that cut (tau is per probe x layer).")


if __name__ == "__main__":
    main()
