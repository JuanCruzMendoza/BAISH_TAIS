"""Spec 3: what fraction of the 100 jailbreaks does each probe read as its direction?

    python jb_metrics.py <model>
    python jb_metrics.py <model> --threshold neg_p90

One number per cell: `pct_reads`, the percentage of jailbreak prompts whose readout
clears the probe's threshold. Reported per probe x band layer, sliced by jailbreak
group, and averaged over the band.

**Threshold.** Default `midpoint`: tau = (mean(pos) + mean(neg)) / 2 on the probe's
own reference poles, pooled train + held-out (65 points each). Same rule as
probe_select's `acc_at_train_thr`, so the two experiments cut in the same place. A
jailbreak counts when its readout falls on the positive pole's side of that boundary.

Its one weakness, worth watching in `ref_tpr`: the mean of a pole is pulled by its
tail, so a wide positive pole drags tau upward and makes the bar stricter for a reason
unrelated to where the boundary should sit. `--threshold` switches to the alternatives
below, from permissive to strict: `neg_median`, `neg_p90`, `neg_p95`, `gap_mid`,
`midpoint`, `pos_p5`.

An accuracy-maximising threshold is *not* available: the extraction diagonals are
AUROC 1.000, so the poles are perfectly separated and every tau inside the gap ties.

Band layers only (spec 0.3).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf

MIN_GROUP = 5        # below this a group percentage is not worth printing

# tau from the probe's own reference poles, permissive -> strict. With the poles
# perfectly separated, `neg_p95` sits at the bottom edge of the empty gap and `pos_p5`
# at the top, so those two bracket the answer; `gap_mid` is the max-margin point
# between them and `midpoint` (the default) bisects the pole *means*.
THRESHOLDS = {
    "neg_median": lambda pos, neg: float(np.median(neg)),
    "neg_p90": lambda pos, neg: float(np.quantile(neg, 0.90)),
    "neg_p95": lambda pos, neg: float(np.quantile(neg, 0.95)),
    "gap_mid": lambda pos, neg: float(0.5 * (np.quantile(neg, 0.95)
                                             + np.quantile(pos, 0.05))),
    "midpoint": lambda pos, neg: float(0.5 * (pos.mean() + neg.mean())),
    "pos_p5": lambda pos, neg: float(np.quantile(pos, 0.05)),
}

GROUPS = {"family": "family", "source": "source", "technique": "technique"}


def slices(rows):
    """(group_kind, group, row indices). `all` first, then each grouping field.

    `category` is restricted to base_task_source == 'jbb': the rest are AdvBench-
    derived, 300 of them with an empty category (spec 3.3).
    """
    yield "all", "all", list(range(len(rows)))
    for kind, key in GROUPS.items():
        groups = {}
        for j, r in enumerate(rows):
            groups.setdefault(r[key], []).append(j)
        for g, idx in sorted(groups.items()):
            yield kind, g, idx
    jbb = {}
    for j, r in enumerate(rows):
        if r["base_task_source"] == "jbb" and r["category"]:
            jbb.setdefault(r["category"], []).append(j)
    for g, idx in sorted(jbb.items()):
        yield "category", g, idx


def _gap_position(pos, neg, tau):
    """Where tau falls in [p95(neg), p5(pos)]: 0 permissive edge, 1 strict edge.

    nan when p5(pos) <= p95(neg), i.e. the poles overlap at those quantiles and there
    is no empty gap to place tau inside. Real extraction axes read AUROC 1.000 so a gap
    exists; a nan here says the reference separation is weaker than assumed.
    """
    lo, hi = float(np.quantile(neg, 0.95)), float(np.quantile(pos, 0.05))
    return float((tau - lo) / (hi - lo)) if hi > lo else float("nan")


def rate_rows(probes, framed, ref, rows, band, Lp1, rule):
    tau_fn = THRESHOLDS[rule]
    out = []
    for i, a in enumerate(probes):
        pos, neg = ref[a]["pos"].numpy(), ref[a]["neg"].numpy()
        for l in band:
            tau = tau_fn(pos[:, l], neg[:, l])
            x = framed[i, :, l]
            diag = {"threshold": tau,
                    # What the bar costs on the reference set. ref_tpr low means tau is
                    # too strict for anything to pass, which a low pct_reads would
                    # otherwise be misread as "jailbreaks are not narrative".
                    "ref_fpr": float((neg[:, l] > tau).mean()),
                    "ref_tpr": float((pos[:, l] > tau).mean()), "n_ref": pos.shape[0],
                    # Where tau sits inside the empty gap between the poles: 0 = the
                    # permissive edge (top of the negative pole), 1 = the strict edge
                    # (bottom of the positive pole). Shows how much the rule matters.
                    "gap_position": _gap_position(pos[:, l], neg[:, l], tau)}
            for kind, g, idx in slices(rows):
                if len(idx) < MIN_GROUP:
                    continue
                hit = int((x[idx] > tau).sum())
                out.append({"probe": a, "layer": l, "depth": round(l / (Lp1 - 1), 4),
                            "group_kind": kind, "group": g, "n": len(idx),
                            "n_reads": hit, "pct_reads": 100.0 * hit / len(idx), **diag})
    return out


def band_rows(rate, band):
    """Mean pct_reads over the band, per probe x slice."""
    by = {}
    for r in rate:
        by.setdefault((r["probe"], r["group_kind"], r["group"]), []).append(r)
    out = []
    for (a, kind, g), rs in by.items():
        p = np.array([r["pct_reads"] for r in rs])
        out.append({"probe": a, "group_kind": kind, "group": g, "n": rs[0]["n"],
                    "pct_reads_mean": float(p.mean()),
                    "pct_reads_min": float(p.min()), "pct_reads_max": float(p.max()),
                    "n_layers": len(rs), "band_lo": band[0], "band_hi": band[-1],
                    "ref_tpr_mean": float(np.mean([r["ref_tpr"] for r in rs])),
                    "ref_fpr_mean": float(np.mean([r["ref_fpr"] for r in rs])),
                    "gap_position_mean": float(np.mean([r["gap_position"] for r in rs]))})
    return sorted(out, key=lambda r: (r["group_kind"] != "all", r["group_kind"],
                                      r["group"], r["probe"]))


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
    ap.add_argument("--threshold", default="midpoint", choices=list(THRESHOLDS))
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    up = lay.vectors / "jb_readout.pt"
    if not up.exists():
        raise SystemExit("run jb_readout.py first")
    mf.load_upstream(lay.meta / "jb_readout_manifest.json")
    R = torch.load(up, weights_only=False)

    probes = R["probes"]
    framed = R["framed"].numpy()
    Lp1 = R["n_layers"] + 1
    band = cfg.band(R["n_layers"])
    with (lay.csv / "jb_readout_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row_id"] for r in rows] == list(R["row_ids"]), "row order drifted"

    rate = rate_rows(probes, framed, R["ref"], rows, band, Lp1, args.threshold)
    bands = band_rows(rate, band)

    stem = mf.stem("jb_metrics", args.threshold)
    config = {"probes": probes, "threshold_rule": args.threshold,
              "reference": "pooled train+heldout poles", "band": [band[0], band[-1]],
              "layers": band, "min_group": MIN_GROUP, "arms": ["framed"],
              "seed": cfg.SEED}
    inputs = {"jb_view_key": R["jb_view_key"], "jb_readout_run_key": R.get("run_key")}

    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact("_rate.csv"), rate)
        write_csv(run.artefact("_band.csv"), bands)

        print(f"{len(rows)} jailbreak prompts, band L{band[0]}-{band[-1]} "
              f"({len(band)} layers), threshold = {args.threshold} "
              f"on {rate[0]['n_ref']} reference points per pole\n")
        print("pct of jailbreaks the probe reads as its direction, band mean")
        print("  " + "probe".ljust(10) + "pct".rjust(7) + "min-max".rjust(14)
              + "ref_tpr".rjust(9) + "ref_fpr".rjust(9) + "gap_pos".rjust(9))
        for a in probes:
            r = next(x for x in bands if x["probe"] == a and x["group_kind"] == "all")
            print("  " + a.ljust(10) + f"{r['pct_reads_mean']:6.1f}%"
                  + f"{r['pct_reads_min']:.0f}-{r['pct_reads_max']:.0f}%".rjust(14)
                  + f"{r['ref_tpr_mean']:.2f}".rjust(9)
                  + f"{r['ref_fpr_mean']:.2f}".rjust(9)
                  + (f"{r['gap_position_mean']:+.2f}"
                     if r["gap_position_mean"] == r["gap_position_mean"]
                     else "no gap").rjust(9))
        print("  ref_tpr: near 1.0 the bar is passable, so a low pct is a real finding; "
              "low ref_tpr\n  means tau is too strict to conclude anything. gap_pos: "
              "where tau sits between the\n  poles, 0 = permissive edge, 1 = strict edge; "
              "outside [0,1] the pole means are being\n  dragged by a tail, 'no gap' means "
              "the poles overlap at the 5/95 quantiles.")

        for kind in ("family", "source", "category"):
            sub = [x for x in bands if x["group_kind"] == kind]
            if not sub:
                continue
            gs = list(dict.fromkeys(x["group"] for x in sub))
            print(f"\nby {kind} (band mean pct)")
            print("  " + "probe".ljust(10) + "".join(g[:13].rjust(15) for g in gs))
            for a in probes:
                cells = []
                for g in gs:
                    m = next((x for x in sub if x["probe"] == a and x["group"] == g), None)
                    cells.append(("-" if m is None
                                  else f"{m['pct_reads_mean']:.0f}% (n={m['n']})").rjust(15))
                print("  " + a.ljust(10) + "".join(cells))


if __name__ == "__main__":
    main()
