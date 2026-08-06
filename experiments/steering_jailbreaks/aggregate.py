"""Spec 5.8: the cross-cell join, and nothing else. CPU.

    python aggregate.py <model> --tag 50_per_direction

Reads every <stem>_summary.csv (never the JSONLs) and recomputes none of it, so the cell
tables and this one cannot disagree. Three jobs: one table over all cells; each cell
beside its no-op and matched random arms; necessity (5.4) beside sufficiency (5.5) per
direction. Cells whose manifest is not `complete` are excluded and listed.

At tag 50_per_direction it refuses to rank layer configs or alphas: at ~30 rows clustered
to template_id the ordering is noise (spec 5.0).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf
from experiments.steering_jailbreaks import sets

NO_RANK_TAGS = ("50_per_direction",)


def load_cells(lay):
    """-> (rows, excluded). A summary with an incomplete manifest is not consumable."""
    rows, excluded = [], []
    for path in sorted(Path(lay.csv).glob("*_summary.csv")):
        stem = path.name[: -len("_summary.csv")]
        man = Path(lay.meta) / f"{stem}_manifest.json"
        if not man.exists():
            excluded.append((stem, "no manifest"))
            continue
        status = json.loads(man.read_text(encoding="utf-8")).get("status")
        if status != "complete":
            excluded.append((stem, f"status={status}"))
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows, excluded


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def key_of(r):
    """A cell's arm-independent identity: what its no-op and random arms match."""
    return (r["prompt_set"], r["mode"], r["layers_spec"], r["alpha"], r["tau_q"])


def with_controls(rows):
    """Each target cell plus the deltas against its own controls."""
    noop = {(r["prompt_set"], r["layers_spec"]): r for r in rows if r["arm"] == "noop"}
    rand = {key_of(r): r for r in rows if r["arm"] == "random"}
    out = []
    for r in rows:
        if r["arm"] != "target":
            continue
        n = noop.get((r["prompt_set"], r["layers_spec"]))
        q = rand.get(key_of(r))
        row = dict(r)
        for label, ref in (("noop", n), ("random", q)):
            for col in ("strongreject_coherent", "pct_refused", "pct_degenerate"):
                a, b = num(r[col]), (num(ref[col]) if ref else None)
                row[f"d_{col}_vs_{label}"] = None if b is None else round(a - b, 4)
        out.append(row)
    return out


def paired(rows):
    """Necessity beside sufficiency: one row per direction x layer config."""
    by = {}
    for r in rows:
        by.setdefault((r["direction"], r["layers_spec"]), {})[r["prompt_set"]] = r
    out = []
    for (d, spec), sides in sorted(by.items()):
        s, f = sides.get("success"), sides.get("refusal")
        out.append({
            "direction": d, "layers_spec": spec,
            "success_mode": s["mode"] if s else None,
            "restore_d_refused_vs_noop": s["d_pct_refused_vs_noop"] if s else None,
            "restore_d_refused_vs_random": s["d_pct_refused_vs_random"] if s else None,
            "restore_degenerate_pct": s["pct_degenerate"] if s else None,
            "refusal_mode": f["mode"] if f else None,
            "induce_d_sr_vs_noop": f["d_strongreject_coherent_vs_noop"] if f else None,
            "induce_d_sr_vs_random": f["d_strongreject_coherent_vs_random"] if f else None,
            "induce_degenerate_pct": f["pct_degenerate"] if f else None})
    return out


def decoding(rows):
    """Spec 5.1: ASR per decoding config, averaged over seeds. The pick reads this."""
    by = {}
    for r in rows:
        if r["prompt_set"] == "decoding":
            by.setdefault(r["decoding"], []).append(r)
    out = []
    for label, cells in sorted(by.items()):
        asr = [num(c["asr"]) for c in cells]
        out.append({
            "decoding": label, "n_seeds": len(cells), "n_rows": cells[0]["n"],
            "asr_mean": round(sum(asr) / len(asr), 1),
            "asr_min": min(asr), "asr_max": max(asr),
            "asr_spread": round(max(asr) - min(asr), 1),
            "strongreject_mean": round(sum(num(c["strongreject"]) for c in cells) / len(cells), 4),
            "pct_degenerate_mean": round(
                sum(num(c["pct_degenerate"]) for c in cells) / len(cells), 1),
            "hit_cap_rate_mean": round(
                sum(num(c["hit_cap_rate"]) for c in cells) / len(cells), 4),
            "deterministic": label == "greedy"})
    return out


def write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    lay = cfg.Layout(sets.EXPERIMENT, args.model, args.tag, acts_cache=False)
    cells, excluded = load_cells(lay)
    if not cells:
        raise SystemExit("no *_summary.csv: run judge_strongreject.py on each cell first")

    steering = [r for r in cells if r["prompt_set"] != "decoding"]
    targets = with_controls(steering)
    dec = decoding(cells)
    stem = mf.stem("aggregate")
    config = {"tag": cfg.tag(args.tag), "n_cells": len(cells),
              "n_excluded": len(excluded), "rank_layers": cfg.tag(args.tag) not in NO_RANK_TAGS}
    inputs = {"cell_run_keys": sorted(r["run_key"] for r in cells)}

    with mf.Run(lay, stem, config, inputs) as run:
        write(run.artefact("_cells.csv"), cells)
        write(run.artefact("_controls.csv"), targets)
        write(run.artefact("_paired.csv"), paired(targets))
        write(run.artefact("_decoding.csv"), dec)

        print(f"{len(cells)} cells -> {len(targets)} target cells with controls")
        for s, why in excluded:
            print(f"  ! excluded {s}: {why}")
        if dec:
            print("\ndecoding comparison (spec 5.1) -- pick one and pass it to every "
                  "later script:")
            print("  " + "config".ljust(12) + "seeds  ASR (mean/min-max)   degen%  "
                  "determ.")
            for r in dec:
                print(f"  {r['decoding']:12}{r['n_seeds']:5}  "
                      f"{r['asr_mean']:5.1f}% {r['asr_min']:5.1f}-{r['asr_max']:<5.1f} "
                      f"{r['pct_degenerate_mean']:8.1f}  {str(r['deterministic'])}")
            print("  Higher ASR leaves more headroom to restore refusal; greedy makes a "
                  "steering delta\n  steering rather than sampling. A sampled pick turns "
                  "ASR into a rate over n>=5 samples\n  per cell and multiplies 5.4 "
                  "(spec 5.1).")
        if cfg.tag(args.tag) in NO_RANK_TAGS:
            print(f"\ntag {cfg.tag(args.tag)}: layer configs and alphas are reported side by "
                  f"side and NOT ranked.\n  At ~30 rows clustered to template_id the ordering "
                  f"is noise (spec 5.0/0.7); a positive cell is readable, a null is not.")


if __name__ == "__main__":
    main()
