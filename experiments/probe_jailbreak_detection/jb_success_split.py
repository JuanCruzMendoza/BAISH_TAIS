"""Spec 3.5: does a probe's readout separate the jailbreaks that worked from those refused?

    python jb_success_split.py <model>     # AFTER gen_baseline.py + judge_strongreject.py

Joins experiment 5's baseline judge labels onto experiment 3's readouts. The two sets are
exactly the ones 5.4 and 5.5 steer, imported from `steering_jailbreaks.sets` rather than
re-derived, so "successful jailbreak" cannot mean two things in two experiments.

Observational. A separation here says the axis tracks an outcome the model already had;
only 5 can say the axis causes it.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf, metrics as met
from experiments.probe_jailbreak_detection import jb_metrics as jbm
from experiments.steering_jailbreaks import sets

MIN_SIDE = 5        # a group needs this many rows on *both* sides to be worth a number

GROUPS = ("family",)


def slices(rows):
    yield "all", "all", list(range(len(rows)))
    for kind in GROUPS:
        groups = {}
        for j, r in enumerate(rows):
            groups.setdefault(r[kind], []).append(j)
        for g, idx in sorted(groups.items()):
            yield kind, g, idx


def split_rows(rows, judged):
    """-> per-row label in {success, refusal, degenerate, unjudged}, in readout order.

    `unjudged` covers both a row the baseline never generated and one whose scores did not
    parse -- neither is a degenerate response, and pooling them would overstate how much
    of the set the model broke on.
    """
    succ = set(sets.outcome_ids(judged, "success"))
    ref = set(sets.outcome_ids(judged, "refusal"))
    out = []
    for r in rows:
        uid = str(r["row_id"])
        if uid in succ:
            out.append("success")
        elif uid in ref:
            out.append("refusal")
        elif judged.get(uid, {}).get("outcome") == "degenerate":
            out.append("degenerate")
        else:
            out.append("unjudged")
    return out


def rate_rows(probes, framed, ref, rows, label, band, Lp1, rule):
    tau_fn = jbm.THRESHOLDS[rule]
    lab = np.array(label)
    out = []
    for i, a in enumerate(probes):
        pos, neg = ref[a]["pos"].numpy(), ref[a]["neg"].numpy()
        for l in band:
            tau = tau_fn(pos[:, l], neg[:, l])
            x = framed[i, :, l]
            for kind, g, idx in slices(rows):
                sel = np.array(idx)
                s, f = sel[lab[sel] == "success"], sel[lab[sel] == "refusal"]
                if len(s) < MIN_SIDE or len(f) < MIN_SIDE:
                    continue
                out.append({
                    "probe": a, "layer": l, "depth": round(l / (Lp1 - 1), 4),
                    "group_kind": kind, "group": g,
                    "n_success": len(s), "n_refusal": len(f),
                    # > 0.5: the jailbreaks that worked read *higher* on this axis.
                    # Unpaired, so no exact interval applies (metrics.unpaired_auroc).
                    "auroc": met.unpaired_auroc(x[s], x[f]),
                    "mean_success": float(x[s].mean()),
                    "mean_refusal": float(x[f].mean()),
                    "pct_reads_success": 100.0 * float((x[s] > tau).mean()),
                    "pct_reads_refusal": 100.0 * float((x[f] > tau).mean()),
                    "threshold": tau})
    return out


def band_rows(rate, rows, label, band):
    """Band mean per probe x slice, plus the length control for that slice."""
    ntok = np.array([float(r["n_tokens"] or 0) for r in rows])
    lab = np.array(label)
    len_auroc = {}
    for kind, g, idx in slices(rows):
        sel = np.array(idx)
        s, f = sel[lab[sel] == "success"], sel[lab[sel] == "refusal"]
        if len(s) >= MIN_SIDE and len(f) >= MIN_SIDE:
            len_auroc[(kind, g)] = met.unpaired_auroc(ntok[s], ntok[f])

    by = {}
    for r in rate:
        by.setdefault((r["probe"], r["group_kind"], r["group"]), []).append(r)
    out = []
    for (a, kind, g), rs in by.items():
        au = np.array([r["auroc"] for r in rs])
        out.append({
            "probe": a, "group_kind": kind, "group": g,
            "n_success": rs[0]["n_success"], "n_refusal": rs[0]["n_refusal"],
            "auroc_mean": float(au.mean()), "auroc_min": float(au.min()),
            "auroc_max": float(au.max()),
            "pct_reads_success_mean": float(np.mean([r["pct_reads_success"] for r in rs])),
            "pct_reads_refusal_mean": float(np.mean([r["pct_reads_refusal"] for r in rs])),
            # The same contrast run on prompt length alone. Length is confounded with
            # family (spec 3.2), so a probe whose auroc_mean does not beat this is not
            # separating the two sets by anything the direction contributes.
            "auroc_len": len_auroc.get((kind, g), float("nan")),
            "n_layers": len(rs), "band_lo": band[0], "band_hi": band[-1]})
    return sorted(out, key=lambda r: (r["group_kind"] != "all", r["group_kind"],
                                      r["group"], r["probe"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threshold", default="midpoint", choices=list(jbm.THRESHOLDS))
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    up = lay.vectors / "jb_readout.pt"
    if not up.exists():
        raise SystemExit("run jb_readout.py first")
    mf.load_upstream(lay.meta / "jb_readout_manifest.json")
    R = torch.load(up, weights_only=False)

    probes, framed = R["probes"], R["framed"].numpy()
    Lp1 = R["n_layers"] + 1
    band = cfg.band(R["n_layers"])
    with (lay.csv / "jb_readout_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row_id"] for r in rows] == list(R["row_ids"]), "row order drifted"

    slay = cfg.Layout(sets.EXPERIMENT, args.model, args.tag, acts_cache=False)
    bpath = slay.meta / f"{mf.stem('gen_baseline')}_manifest.json"
    if not bpath.exists():
        raise SystemExit(f"missing {bpath.name}: this script runs after experiment 5's "
                         f"baseline (spec 3.5). Run gen_baseline.py then "
                         f"judge_strongreject.py on its .jsonl.")
    bman = mf.load_upstream(bpath)
    judged = sets.baseline_judged(args.model, args.tag)

    # What makes the join valid is that both sides used the same *rows*, not the same
    # view_key. Caching the bare-request arm for `cap` adds `neg` rows to the view, which
    # moves view_key while leaving the 100 framed prompts identical -- and jailbreak_rows
    # filters to pole == "pos" anyway. So compare row identity and only note the keys.
    same_view = bman["inputs"]["jb_view_key"] == R["jb_view_key"]
    missing = [r["row_id"] for r in rows if str(r["row_id"]) not in judged]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(rows)} readout rows are absent from the judged "
            f"baseline (e.g. {missing[:3]}). The baseline did not generate for these "
            f"rows, so there is nothing to join"
            + ("" if same_view else "; the two jb_view_keys also differ, so the "
                                    "baseline likely ran on a different subsample"))
    if not same_view:
        print(f"  note: view_key differs from jb_readout's (expected if the bare-request "
              f"arm was cached for cap); all {len(rows)} rows matched by row_id")

    label = split_rows(rows, judged)
    counts = {k: label.count(k) for k in ("success", "refusal", "degenerate", "unjudged")}
    if min(counts["success"], counts["refusal"]) < MIN_SIDE:
        raise SystemExit(f"one side is too small to split: {counts}")

    rate = rate_rows(probes, framed, R["ref"], rows, label, band, Lp1, args.threshold)
    bands = band_rows(rate, rows, label, band)

    stem = mf.stem("jb_success_split", args.threshold)
    config = {"probes": probes, "threshold_rule": args.threshold,
              "band": [band[0], band[-1]], "layers": band, "min_side": MIN_SIDE,
              "groups": ["all", *GROUPS], "counts": counts,
              "decoding": bman["config"].get("decoding"), "seed": cfg.SEED}
    # The judged rows carry labels but not the judge id -- that lives in the summary the
    # judge writes beside them. Pinned here because the split is a function of it.
    bsum = slay.csv / f"{mf.stem('gen_baseline')}_summary.csv"
    judge_model = None
    if bsum.exists():
        with bsum.open(encoding="utf-8-sig", newline="") as f:
            judge_model = next(csv.DictReader(f), {}).get("judge_model")
    inputs = {"jb_view_key": R["jb_view_key"],
              "baseline_jb_view_key": bman["inputs"]["jb_view_key"],
              "jb_readout_run_key": R.get("run_key"),
              "baseline_run_key": bman["run_key"], "judge_model": judge_model}

    with mf.Run(lay, stem, config, inputs) as run:
        jbm.write_csv(run.artefact("_rate.csv"), rate)
        jbm.write_csv(run.artefact("_band.csv"), bands)

        cl = {k: len({r["template_id"] for r, x in zip(rows, label) if x == k})
              for k in ("success", "refusal")}
        print(f"{len(rows)} jailbreaks: {counts['success']} success / "
              f"{counts['refusal']} refusal / {counts['degenerate']} degenerate"
              + (f" / {counts['unjudged']} unjudged" if counts["unjudged"] else ""))
        print(f"  template_id clusters (spec 0.7): {cl['success']} / {cl['refusal']}   "
              f"band L{band[0]}-{band[-1]}, threshold = {args.threshold}\n")
        print("readout of successes vs refusals, band mean")
        print("  " + "probe".ljust(10) + "auroc".rjust(7) + "min-max".rjust(14)
              + "reads S".rjust(10) + "reads R".rjust(10))
        for a in probes:
            r = next(x for x in bands if x["probe"] == a and x["group_kind"] == "all")
            print("  " + a.ljust(10) + f"{r['auroc_mean']:6.3f}"
                  + f"{r['auroc_min']:.2f}-{r['auroc_max']:.2f}".rjust(14)
                  + f"{r['pct_reads_success_mean']:.0f}%".rjust(10)
                  + f"{r['pct_reads_refusal_mean']:.0f}%".rjust(10))
        allr = next(x for x in bands if x["group_kind"] == "all")
        print(f"\n  length control, same contrast on n_tokens alone: {allr['auroc_len']:.3f}")
        print("  auroc > 0.5: the jailbreaks that worked read higher on the axis. No exact\n"
              "  interval applies (unpaired), and at these n a value near 0.5 is unreadable.\n"
              "  Observational: experiment 5 is the causal test.")

        for kind in GROUPS:
            sub = [x for x in bands if x["group_kind"] == kind]
            if not sub:
                continue
            gs = list(dict.fromkeys(x["group"] for x in sub))
            print(f"\nby {kind} (band-mean auroc)")
            print("  " + "probe".ljust(10) + "".join(g[:13].rjust(16) for g in gs))
            for a in probes:
                cells = []
                for g in gs:
                    m = next((x for x in sub if x["probe"] == a and x["group"] == g), None)
                    cells.append(("-" if m is None else
                                  f"{m['auroc_mean']:.3f} ({m['n_success']}/"
                                  f"{m['n_refusal']})").rjust(16))
                print("  " + a.ljust(10) + "".join(cells))


if __name__ == "__main__":
    main()
