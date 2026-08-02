"""Spec 3.1-3.4: aggregate the jailbreak readouts. CPU only.

    python jb_metrics.py <model>

3.1 is the primary test and it is paired within-row: framed `prompt` vs bare
`request`. Spec 0.7 requires cluster-mean aggregation *before* testing, so every
paired number is reported three ways -- by row, by `template_id`, by `request` -- and
the smallest n is the one that counts.

3.2's family test is between-group and therefore confounded with source and length;
it carries no interval (Clopper-Pearson does not apply to a Mann-Whitney statistic)
and is reported with both confounds measured alongside it.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, manifest as mf, metrics as met

FICTION = ("fiction_narrative", "hybrid")     # spec 3.2's positive side
MIN_GROUP = 5                                 # below this a group cell says nothing
CLUSTERS = {"row": None, "template": "template_id", "request": "request_sha8"}


def paired_cell(pos, neg, cluster_ids=None):
    """Cluster-mean first, then test (spec 0.7). Never anti-conservative."""
    if cluster_ids is not None:
        _, pos = met.cluster_means(pos, cluster_ids)
        _, neg = met.cluster_means(neg, cluster_ids)
    ci = met.auroc_ci(pos, neg)
    return {"auroc": ci["auroc"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "wins": ci["wins"], "ties": ci["ties"], "n": ci["n"],
            "cohens_dz": met.cohens_dz(pos, neg),
            "mean_delta": float(np.mean(np.asarray(pos) - np.asarray(neg)))}


# --------------------------------------------------------------- 3.1 layer profile

def layer_rows(probes, framed, bare, rows, Lp1):
    out = []
    for i, a in enumerate(probes):
        for l in range(Lp1):
            for label, key in CLUSTERS.items():
                ids = [r[key] for r in rows] if key else None
                out.append({"probe": a, "layer": l, "depth": round(l / (Lp1 - 1), 4),
                            "cluster": label,
                            **paired_cell(framed[i, :, l], bare[i, :, l], ids)})
    return out


# ------------------------------------------------------------------- 3.1 subgroups

def group_rows(probes, framed, bare, rows, band, best):
    """Per family / source / technique / JBB category, at the probe's own-best layer.

    `category` is restricted to `base_task_source == 'jbb'`: the other rows are
    AdvBench-derived, 300 of them with an empty category (spec 3.3).
    """
    kinds = {"family": lambda r: r["family"], "source": lambda r: r["source"],
             "technique": lambda r: r["technique"],
             "category": lambda r: r["category"] if r["base_task_source"] == "jbb" else None}
    out = []
    for i, a in enumerate(probes):
        l = best[a]
        for kind, keyf in kinds.items():
            groups = {}
            for j, r in enumerate(rows):
                g = keyf(r)
                if g:
                    groups.setdefault(g, []).append(j)
            for g, idx in sorted(groups.items()):
                if len(idx) < MIN_GROUP:
                    continue
                sub = [rows[j] for j in idx]
                for label, key in CLUSTERS.items():
                    ids = [r[key] for r in sub] if key else None
                    cell = paired_cell(framed[i, idx, l], bare[i, idx, l], ids)
                    out.append({"probe": a, "layer": l, "group_kind": kind, "group": g,
                                "cluster": label, "n_rows": len(idx), **cell,
                                "band_mean_auroc": float(np.mean(
                                    [paired_cell(framed[i, idx, k], bare[i, idx, k],
                                                 ids)["auroc"] for k in band]))})
    return out


# ---------------------------------------------------------- 3.2 family + confounds

def family_rows(probes, framed, bare, rows, best):
    """Between-group: does fiction framing move the probe more than nonfiction?

    Run on the within-row delta, not the absolute readout, so the prompt-distribution
    offset that 2.1 warns about cancels before the groups are compared.
    """
    fic = [j for j, r in enumerate(rows) if r["family"] in FICTION]
    non = [j for j, r in enumerate(rows) if r["family"] == "nonfiction_other"]
    ntok = np.array([float(r["n_tokens_framed"] or 0) for r in rows])
    out = []
    for i, a in enumerate(probes):
        l = best[a]
        d = framed[i, :, l] - bare[i, :, l]
        rec = {"probe": a, "layer": l, "scope": "pooled",
               "n_fiction": len(fic), "n_nonfiction": len(non),
               "wrappers_fiction": len({rows[j]["template_id"] for j in fic}),
               "wrappers_nonfiction": len({rows[j]["template_id"] for j in non}),
               "auroc_delta": met.unpaired_auroc(d[fic], d[non]),
               "auroc_absolute": met.unpaired_auroc(framed[i, fic, l], framed[i, non, l]),
               # Length is confounded with family (spec 3.2): median chars 973 fiction
               # vs 409 nonfiction. Both controls travel with the number.
               "spearman_delta_ntokens": met.spearman(d, ntok),
               "median_ntok_fiction": float(np.median(ntok[fic])),
               "median_ntok_nonfiction": float(np.median(ntok[non]))}
        out.append(rec)
        for src in sorted({r["source"] for r in rows}):
            f = [j for j in fic if rows[j]["source"] == src]
            n = [j for j in non if rows[j]["source"] == src]
            if not f or not n:
                continue
            out.append({"probe": a, "layer": l, "scope": f"within_source:{src}",
                        "n_fiction": len(f), "n_nonfiction": len(n),
                        "wrappers_fiction": len({rows[j]["template_id"] for j in f}),
                        "wrappers_nonfiction": len({rows[j]["template_id"] for j in n}),
                        "auroc_delta": met.unpaired_auroc(d[f], d[n]),
                        "auroc_absolute": met.unpaired_auroc(framed[i, f, l],
                                                             framed[i, n, l]),
                        "spearman_delta_ntokens": float("nan"),
                        "median_ntok_fiction": float(np.median(ntok[f])),
                        "median_ntok_nonfiction": float(np.median(ntok[n]))})
    return out


def length_rows(probes, framed, bare, rows, best):
    """AUROC inside n_token deciles (spec 3.2): does the paired result survive length?"""
    ntok = np.array([float(r["n_tokens_framed"] or 0) for r in rows])
    out = []
    for i, a in enumerate(probes):
        l = best[a]
        for b in met.auroc_within_bins(framed[i, :, l], bare[i, :, l], ntok):
            out.append({"probe": a, "layer": l, **b})
    return out


# ------------------------------------------------------------- 3.4 calibrated form

def calibrated_rows(probes, framed, bare, ref, band):
    """Ordinal only, per spec 3.4: a 65-point reference cannot carry a percentile.

    For `harm` this is the whole test (every jailbreak row is harmful, so the probe has
    no contrast here): if framed readouts sit inside the harmful pole's range,
    jailbreaks do not flip internal harmfulness (2507.11878).
    """
    out = []
    for i, a in enumerate(probes):
        pos, neg = ref[a]["pos"].numpy(), ref[a]["neg"].numpy()
        for l in band:
            row = {"probe": a, "layer": l, "n_ref": pos.shape[0]}
            for name, x in (("framed", framed[i, :, l]), ("bare", bare[i, :, l])):
                for pole, r in (("pos", pos[:, l]), ("neg", neg[:, l])):
                    sd = r.std(ddof=1)
                    row[f"z_{name}_vs_{pole}"] = float((x.mean() - r.mean())
                                                       / sd) if sd > 0 else float("nan")
                    row[f"frac_{name}_above_{pole}_median"] = float(
                        (x > np.median(r)).mean())
                    row[f"frac_{name}_inside_{pole}_range"] = float(
                        ((x >= r.min()) & (x <= r.max())).mean())
            out.append(row)
    return out


# --------------------------------------------------------------------- helpers

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
    args = ap.parse_args()

    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)
    up = lay.vectors / "jb_readout.pt"
    if not up.exists():
        raise SystemExit("run jb_readout.py first")
    mf.load_upstream(lay.meta / "jb_readout_manifest.json")
    R = torch.load(up, weights_only=False)

    probes = R["probes"]
    framed, bare = R["framed"].numpy(), R["bare"].numpy()
    Lp1 = R["n_layers"] + 1
    band = cfg.band(R["n_layers"])
    best = {a: int(R["best_layer"][a]) for a in probes}
    with (lay.csv / "jb_readout_rows.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["row_id"] for r in rows] == list(R["row_ids"]), "row order drifted"

    prof = layer_rows(probes, framed, bare, rows, Lp1)
    grp = group_rows(probes, framed, bare, rows, band, best)
    fam = family_rows(probes, framed, bare, rows, best)
    ln = length_rows(probes, framed, bare, rows, best)
    cal = calibrated_rows(probes, framed, bare, R["ref"], band)

    stem = mf.stem("jb_metrics")
    config = {"probes": probes, "best_layers": best, "clusters": list(CLUSTERS),
              "min_group": MIN_GROUP, "fiction_families": list(FICTION),
              "band": [band[0], band[-1]], "interval": "clopper_pearson", "seed": cfg.SEED}
    inputs = {"jb_view_key": R["jb_view_key"], "jb_readout_run_key": R.get("run_key")}

    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact("_paired.csv"), prof)
        write_csv(run.artefact("_groups.csv"), grp)
        write_csv(run.artefact("_family.csv"), fam)
        write_csv(run.artefact("_length.csv"), ln)
        write_csv(run.artefact("_calibrated.csv"), cal)

        print(f"{len(rows)} rows, band {band[0]}-{band[-1]}\n")
        print("3.1 primary paired test, framed vs bare, at each probe's own-best layer")
        print("  " + "probe".ljust(10) + "L".rjust(4)
              + "".join(f"{c} auroc (n)".rjust(26) for c in CLUSTERS))
        for a in probes:
            cells = []
            for c in CLUSTERS:
                r = next(x for x in prof if x["probe"] == a and x["layer"] == best[a]
                         and x["cluster"] == c)
                cells.append(f"{r['auroc']:.3f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}] "
                             f"n={r['n']}".rjust(26))
            print("  " + a.ljust(10) + str(best[a]).rjust(4) + "".join(cells))

        print("\n3.2 family test on the within-row delta (no interval: Mann-Whitney)")
        print("  " + "probe".ljust(10) + "pooled".rjust(9) + "wrappers".rjust(12)
              + "rho(delta,ntok)".rjust(17) + "  within-source cells")
        for a in probes:
            p = next(x for x in fam if x["probe"] == a and x["scope"] == "pooled")
            ws = [f"{x['scope'].split(':')[1]} {x['auroc_delta']:.2f} "
                  f"({x['n_fiction']}v{x['n_nonfiction']})"
                  for x in fam if x["probe"] == a and x["scope"] != "pooled"]
            print("  " + a.ljust(10) + f"{p['auroc_delta']:.3f}".rjust(9)
                  + f"{p['wrappers_fiction']}v{p['wrappers_nonfiction']}".rjust(12)
                  + f"{p['spearman_delta_ntokens']:+.3f}".rjust(17)
                  + "  " + ("; ".join(ws) or "none with both sides"))

        print("\n3.4 calibrated (band means): where do framed readouts sit vs the poles?")
        print("  " + "probe".ljust(10) + "z vs pos".rjust(10) + "z vs neg".rjust(10)
              + "frac>pos med".rjust(14) + "frac in pos range".rjust(19))
        for a in probes:
            c = [x for x in cal if x["probe"] == a]
            m = lambda k: float(np.mean([x[k] for x in c]))
            print("  " + a.ljust(10) + f"{m('z_framed_vs_pos'):+.2f}".rjust(10)
                  + f"{m('z_framed_vs_neg'):+.2f}".rjust(10)
                  + f"{m('frac_framed_above_pos_median'):.2f}".rjust(14)
                  + f"{m('frac_framed_inside_pos_range'):.2f}".rjust(19))
        print("\n  3.5 (success split) needs experiment 4's judge labels and runs later.")


if __name__ == "__main__":
    main()
