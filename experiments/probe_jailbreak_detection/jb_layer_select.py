"""Pick the band layers whose `pct_reads` best separates a probe's target families.

    python jb_layer_select.py <model> --tag 50_per_direction

Per probe, score each band layer by

    margin = mean(pct_reads over target families) - mean(pct_reads over off families)

story_v1 / story_v2 target {fiction_narrative, hybrid} against {nonfiction_other,
roleplay_persona}; persona targets {hybrid, roleplay_persona} against the other two.
Families are weighted equally, not by n. Reads the `_rate.csv` written by jb_metrics.py;
prints only.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg

FAMILIES = ["fiction_narrative", "hybrid", "nonfiction_other", "roleplay_persona"]

TARGETS = {
    "story_v1": ["fiction_narrative", "hybrid"],
    "story_v2": ["fiction_narrative", "hybrid"],
    "persona": ["hybrid", "roleplay_persona"],
}

TOP_K = 3


def score_layers(rate, probe, target):
    off = [g for g in FAMILIES if g not in target]
    by_layer = {}
    for r in rate:
        if r["probe"] != probe or r["group_kind"] != "family":
            continue
        by_layer.setdefault(int(r["layer"]), {})[r["group"]] = float(r["pct_reads"])
    out = []
    for l, pct in sorted(by_layer.items()):
        if any(g not in pct for g in FAMILIES):
            continue
        tgt = sum(pct[g] for g in target) / len(target)
        neg = sum(pct[g] for g in off) / len(off)
        out.append({"layer": l, "margin": tgt - neg, "target_mean": tgt,
                    "off_mean": neg, **pct})
    return sorted(out, key=lambda r: -r["margin"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threshold", default="midpoint")
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    path = lay.csv / f"jb_metrics__{args.threshold}_rate.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run jb_metrics.py --threshold {args.threshold}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        rate = list(csv.DictReader(f))

    print(f"threshold = {args.threshold}, top {TOP_K} band layers by margin\n")
    for probe, target in TARGETS.items():
        ranked = score_layers(rate, probe, target)
        if not ranked:
            print(f"{probe}: no family rows\n")
            continue
        off = [g for g in FAMILIES if g not in target]
        print(f"{probe}: + {' '.join(target)}   - {' '.join(off)}")
        print("  " + "layer".ljust(7) + "margin".rjust(8)
              + "".join(g[:16].rjust(18) for g in FAMILIES))
        for r in ranked[:TOP_K]:
            print("  " + f"L{r['layer']}".ljust(7) + f"{r['margin']:+7.1f}"
                  + "".join(f"{r[g]:.1f}%".rjust(18) for g in FAMILIES))
        print()


if __name__ == "__main__":
    main()
