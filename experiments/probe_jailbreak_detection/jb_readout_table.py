"""Spec 3: one row per jailbreak, one column per (direction, layer set) readout.

    python jb_readout_table.py <model> --sweep story_v2=15,17,18 --sweep harm=20,21,22
    python jb_readout_table.py <model> --band frac:0.70-0.90

The per-jailbreak view the other two scripts do not give: jb_readout.pt holds the raw
[probe, row, layer] tensor and is gitignored, jb_metrics aggregates to pct_reads per
probe x layer x group. This writes the numbers as a table keyed by row_id, with the
same layer sets steering_jailbreaks steers, so a cell's manipulation check and the
probe's own reading of that prompt sit in one place. `prompt_head` carries the opening
sentences so a row can be read rather than only looked up.

A single layer's column is its readout. A band's column is the **mean** over the
band's layers: raw projections are not comparable across depth, so that mean leans on
the deepest layers. It is the band's readout, not a depth-invariant score.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf, views

META_COLS = ("family", "category", "n_tokens")
HEAD_CHARS = 300


def prompt_head(text, limit=HEAD_CHARS):
    """First sentences, whitespace collapsed so the cell stays one readable line.

    Cut at the last sentence end that fits; if none does, hard-cut and mark it. Most
    of these prompts are a wall of framing, so the head is what identifies the wrapper.
    """
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= limit:
        return t
    cut = max(t.rfind(c, 0, limit + 1) for c in (". ", "! ", "? "))
    return t[: cut + 1] if cut > limit // 3 else t[:limit].rstrip() + "..."


def source_state(man):
    """Whether the corpus is still the bytes jb_readout.py sampled from."""
    out = {}
    for f in man["inputs"]["source_files"]:
        p = cfg.REPO / f["path"]
        if not p.exists():
            raise SystemExit(f"missing {f['path']}, which jb_readout.py sampled from")
        out[f["path"]] = mf.sha256_file(p) == f["sha256"]
    return out


def prompt_heads(man, rows, limit):
    """{row_id: head}, looked up by row_id -- never re-sampled.

    jb_readout.py already fixed which rows it used, so replaying the subsample recipe
    would only add a way to disagree with it. What has to hold is that *these* rows
    still carry the same prompt, so each is checked against the n_chars, family and
    category jb_readout_rows.csv recorded for it. That is per-row and exact, where the
    corpus-wide sha is neither: the file can move for reasons that miss all 100 rows.

    Read from tracked artefacts only (meta/ + csv/), since acts/views/ is gitignored.
    """
    _, pairs = views.load_pairs("jailbreaks", man["config"]["split"])
    by_id = {p["pair_id"]: p for p in pairs}
    heads, bad = {}, []
    for r in rows:
        p = by_id.get(r["row_id"])
        if p is None:
            bad.append(f"{r['row_id']}: absent from the corpus")
            continue
        got, want = len(p["pos"]), str(r.get("n_chars", "")).strip()
        if want and str(got) != want:
            bad.append(f"{r['row_id']}: prompt is {got} chars, readout recorded {want}")
            continue
        drift = [k for k in ("family", "category")
                 if k in r and (p["meta"].get(k) or "") != (r[k] or "")]
        if drift:
            bad.append(f"{r['row_id']}: {'/'.join(drift)} differs from the readout")
            continue
        heads[r["row_id"]] = prompt_head(p["pos"], limit)
    if bad:
        raise SystemExit(f"{len(bad)} of {len(rows)} rows no longer match what "
                         f"jb_readout.py read:\n  " + "\n  ".join(bad[:5]))
    return heads


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


def build(framed, probes, rows, sweep, band_layers, band_stem, heads):
    """-> (column names, [{row_id, prompt_head, ...meta, <dir>__<set>: readout}])."""
    idx = {a: i for i, a in enumerate(probes)}
    cols = []
    for a in sweep:
        cols += [(a, f"{a}__{cfg.layer_stem(str(l))}", [l]) for l in sweep[a]]
        cols.append((a, f"{a}__{band_stem}", band_layers))
    out = []
    for j, r in enumerate(rows):
        row = {"row_id": r["row_id"], "prompt_head": heads[r["row_id"]],
               **{c: r[c] for c in META_COLS if c in r}}
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
    ap.add_argument("--head-chars", type=int, default=HEAD_CHARS,
                    help="length budget for the prompt_head column")
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    up = lay.vectors / "jb_readout.pt"
    if not up.exists():
        raise SystemExit("run jb_readout.py first")
    man = mf.load_upstream(lay.meta / "jb_readout_manifest.json")
    R = torch.load(up, weights_only=False)

    probes, framed, L = R["probes"], R["framed"], R["n_layers"]
    with (lay.csv / "jb_readout_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row_id"] for r in rows] == list(R["row_ids"]), "row order drifted"

    band_layers = cfg.parse_layers(args.band, L)
    sweep = parse_sweep(args.sweep, probes, L) or {a: [] for a in probes}
    band_stem = cfg.layer_stem(args.band)
    src_ok = source_state(man)
    heads = prompt_heads(man, rows, args.head_chars)
    cols, table = build(framed, probes, rows, sweep, band_layers, band_stem, heads)

    stem = mf.stem("jb_readout_table", band_stem)
    config = {"directions": list(sweep), "sweep_layers": sweep,
              "band_spec": args.band, "band_layers": band_layers,
              "band_column": "mean over band_layers", "n_rows": len(table),
              "head_chars": args.head_chars,
              "readout": "(h - mu) . u_hat", "seed": cfg.SEED}
    inputs = {"jb_view_key": R["jb_view_key"], "jb_readout_run_key": R.get("run_key"),
              # Recorded, not enforced: prompt text is verified per row instead.
              "source_sha_matches_readout": src_ok}

    with mf.Run(lay, stem, config, inputs) as run:
        path = run.artefact(".csv")
        fields = ["row_id", "prompt_head",
                  *[c for c in META_COLS if c in rows[0]], *cols]
        # utf-8-sig, unlike every other table here: this one holds prompt text and is
        # meant to be read, and a BOM is what makes Excel show the non-ASCII correctly.
        # Every reader in the repo already opens CSVs as utf-8-sig, so it costs nothing.
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(table)

        print(f"{len(table)} jailbreaks x {len(cols)} readout columns -> {path.name}")
        for f, ok in src_ok.items():
            if not ok:
                print(f"  note: {f} is not the bytes jb_readout.py sampled from, but all "
                      f"{len(table)} rows verify per row (n_chars, family, category)")
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
