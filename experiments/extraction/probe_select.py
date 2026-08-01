"""Per-layer probe metrics and the layer-band selection rule (spec 1.2).

    python probe_select.py <model> --direction story_v2
    python probe_select.py <model> --direction story_v2 --transfer v1_nofiller100   # 1.2a

LOPO on the 50 train pairs selects; the 15 held-out pairs report. Intervals are
Clopper-Pearson, never bootstrap (spec 0.7).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views

BAND_TOL = 0.05      # CP lower bound must be within this of the best lopo auroc
LEN_TOL = 0.10       # |resid_len_auroc - 0.5| gate


def load_direction(lay, axis):
    stem = mf.stem("directions", axis)
    path = lay.vectors / f"{stem}.pt"
    if not path.exists():
        return None
    mf.load_upstream(lay.meta / f"{stem}_manifest.json")
    return torch.load(path, weights_only=False)


def proj(h, u):
    """h [n, L+1, d], u [L+1, d] -> [n, L+1]"""
    return np.einsum("nld,ld->nl", h, u)


# ------------------------------------------------------------------ 1.2 tables


def layer_metrics(args, lay, run):
    dirn = load_direction(lay, args.direction)
    if dirn is None:
        raise SystemExit(f"run extract_direction.py --direction {args.direction} first")

    d_full = dirn["d"].numpy()
    u_full = dirn["u"].numpy()
    lopo = dirn["lopo_d"].numpy()
    sigma = dirn["sigma_act"].numpy()
    Lp1 = d_full.shape[0]

    train_view = views.read_view(lay, args.direction, "train")
    train = acts.load_view_matrix(lay, train_view)
    tp, tn = train["pos"], train["neg"]

    # Spec 0.2(a): if a task was appended, split the readout by task harmfulness.
    tmeta = train_view.get("meta", {})
    harmful = np.array([bool(tmeta.get(p, {}).get("task_harmful", False)) for p in train["pair_ids"]])
    has_tasks = any("task_harmful" in tmeta.get(p, {}) for p in train["pair_ids"])

    try:
        ho = acts.load_view_matrix(lay, views.read_view(lay, args.direction, "heldout"))
    except FileNotFoundError:
        ho = None
        print("! no held-out view cached: heldout columns will be blank")

    # length foil + len_frac (spec 1.2). Absent -> gate skipped, recorded in the CSV.
    len_dir = load_direction(lay, "length")
    try:
        len_ho = acts.load_view_matrix(lay, views.read_view(lay, "length", "heldout"))
    except FileNotFoundError:
        len_ho = None
    if len_ho is None:
        print("! no length heldout view: resid_len_auroc blank, selection gate SKIPPED")

    rows = []
    for l in range(Lp1):
        u_l = u_full[l]
        # LOPO: pair i scored by the vector fitted without it.
        u_lopo = met.unit(lopo[:, l, :])                      # [n, d]
        s_pos = np.einsum("nd,nd->n", tp[:, l, :], u_lopo)
        s_neg = np.einsum("nd,nd->n", tn[:, l, :], u_lopo)
        lopo_ci = met.auroc_ci(s_pos, s_neg)

        delta = tp[:, l, :] - tn[:, l, :]
        mpc = float(np.mean([met.cos(delta[i], u_l) for i in range(delta.shape[0])]))
        stab = float(np.mean([met.cos(lopo[i, l, :], d_full[l]) for i in range(lopo.shape[0])]))

        row = {"layer": l, "depth": round(l / (Lp1 - 1), 4),
               "lopo_auroc": lopo_ci["auroc"], "lopo_ci_lo": lopo_ci["ci_lo"],
               "lopo_ci_hi": lopo_ci["ci_hi"], "lopo_wins": lopo_ci["wins"],
               "lopo_ties": lopo_ci["ties"], "n_train": lopo_ci["n"],
               "lopo_sign_p": met.sign_test_p(s_pos, s_neg),
               "mean_paired_cos": mpc, "lopo_cos_stability": stab,
               "cohens_dz_train": met.cohens_dz(s_pos, s_neg),
               "norm": float(np.linalg.norm(d_full[l])),
               "norm_over_sigma": float(np.linalg.norm(d_full[l]) / max(sigma[l], 1e-9))}

        if has_tasks:
            # Does the framing axis read the same way over harmful and benign requests?
            # A large gap means the vector is framing x harm, not framing.
            row |= {"lopo_auroc_task_harmful": met.paired_auroc(s_pos[harmful], s_neg[harmful]),
                    "lopo_auroc_task_benign": met.paired_auroc(s_pos[~harmful], s_neg[~harmful]),
                    "n_task_harmful": int(harmful.sum()), "n_task_benign": int((~harmful).sum())}

        if ho is not None:
            hp = proj(ho["pos"][:, l:l + 1, :], u_full[l:l + 1])[:, 0]
            hn = proj(ho["neg"][:, l:l + 1, :], u_full[l:l + 1])[:, 0]
            ci = met.auroc_ci(hp, hn)
            thr = 0.5 * (np.einsum("nd,d->n", tp[:, l, :], u_l).mean()
                         + np.einsum("nd,d->n", tn[:, l, :], u_l).mean())
            row |= {"heldout_auroc": ci["auroc"], "heldout_ci_lo": ci["ci_lo"],
                    "heldout_ci_hi": ci["ci_hi"], "n_heldout": ci["n"],
                    "acc_at_train_thr": float(((hp > thr).sum() + (hn <= thr).sum())
                                              / (2 * len(hp)))}

        if len_ho is not None:
            lp = np.einsum("nd,d->n", len_ho["pos"][:, l, :], u_l)
            ln = np.einsum("nd,d->n", len_ho["neg"][:, l, :], u_l)
            row["resid_len_auroc"] = met.paired_auroc(lp, ln)
            row["n_length"] = len(lp)
        if len_dir is not None:
            ul = len_dir["u"].numpy()[l]
            row["len_frac"] = abs(met.cos(d_full[l], ul))

        rows.append(row)

    sel = select_band(rows, gate=len_ho is not None)
    write_csv(run.artefact(".csv"), rows)
    return rows, sel


def _longest_run(rows, ok):
    runs, cur = [], []
    for r, good in zip(rows, ok):
        if good:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return max(runs, key=len) if runs else []


def select_band(rows, gate=True):
    """Fixed in advance (spec 1.2), so it is not post-hoc.

    The criterion compares each layer's CP lower bound against the *best lower
    bound*, not against the best point estimate: at a saturated 50/50 the point
    estimate is 1.00 while its own CP lower bound is 0.929, so a
    `ci_lo >= max(auroc) - 0.05` rule would reject every layer including the
    best one. Comparing like with like also rewards precision, and the argmax
    always qualifies, so the band is never empty.
    """
    best_lo = max(r["lopo_ci_lo"] for r in rows)
    auroc_ok = [r["lopo_ci_lo"] >= best_lo - BAND_TOL for r in rows]
    gated = [ok and abs(r.get("resid_len_auroc", 0.5) - 0.5) <= LEN_TOL
             for r, ok in zip(rows, auroc_ok)] if gate else auroc_ok

    band, gate_failed = _longest_run(rows, gated), False
    if not band:
        # Every layer that reads its own axis also reads length. Report the
        # AUROC band and say so loudly rather than returning nothing.
        band, gate_failed = _longest_run(rows, auroc_ok), True

    top = max(b["mean_paired_cos"] for b in band)
    primary = min((b for b in band if b["mean_paired_cos"] == top), key=lambda b: b["layer"])
    return {"band": [b["layer"] for b in band], "primary": primary["layer"],
            "best_lopo_auroc": max(r["lopo_auroc"] for r in rows),
            "best_lopo_ci_lo": best_lo, "band_threshold": best_lo - BAND_TOL,
            "gate_applied": gate and not gate_failed, "gate_failed": gate_failed,
            "n_pass_length_gate": int(sum(gated)) if gate else None,
            "primary_depth": primary["depth"],
            "primary_lopo_auroc": primary["lopo_auroc"],
            "primary_resid_len_auroc": primary.get("resid_len_auroc"),
            "primary_mean_paired_cos": primary["mean_paired_cos"]}


# ---------------------------------------------------------------- 1.2a transfer


def transfer_report(args, lay):
    """Both story vectors + the length foil on the filler-free v1 prompts."""
    view = views.read_view(lay, args.transfer, "train")
    m = acts.load_view_matrix(lay, view)
    meta = view["meta"]

    probes = {}
    for axis, label in [(args.direction, "d_v2"), ("length", "d_length")]:
        dd = load_direction(lay, axis)
        if dd is not None:
            probes[label] = dd["u"].numpy()
    v1 = load_direction(lay, "story_v1")
    if v1 is not None:
        probes["d_v1_50"] = v1["u"].numpy()
    if "d_length" not in probes:
        print("! d_length missing: this test is UNINTERPRETABLE without the length foil (spec 1.2a)")

    dn = np.array([float(meta[p]["n_words_story"]) - float(meta[p]["n_words_bare"])
                   for p in m["pair_ids"]])
    Lp1 = next(iter(probes.values())).shape[0]

    rows, deciles = [], {}
    for label, u in probes.items():
        for l in range(Lp1):
            sp = np.einsum("nd,d->n", m["pos"][:, l, :], u[l])
            sn = np.einsum("nd,d->n", m["neg"][:, l, :], u[l])
            ci = met.auroc_ci(sp, sn)
            rows.append({"probe": label, "layer": l, "depth": round(l / (Lp1 - 1), 4),
                         "auroc": ci["auroc"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                         "n": ci["n"], "cohens_dz": met.cohens_dz(sp, sn),
                         "spearman_readout_dwords": met.spearman(sp - sn, dn)})
            deciles.setdefault(label, {})[l] = met.auroc_within_bins(sp, sn, dn)

    stem = mf.stem("probe_select", args.direction, f"transfer_{args.transfer}")
    config = {"direction": args.direction, "transfer": args.transfer,
              "probes": sorted(probes), "seed": cfg.SEED}
    inputs = {"view_key": view["view_key"], "source_files": view["source_files"]}
    with mf.Run(lay, stem, config, inputs) as run:
        write_csv(run.artefact(".csv"), rows)
        run.artefact("_deciles.json").write_text(json.dumps(deciles, indent=2), encoding="utf-8")
        band = cfg.band(Lp1 - 1)
        print(f"transfer {args.transfer}: n={rows[0]['n']} pairs, probes {sorted(probes)}")
        for label in probes:
            mid = [r for r in rows if r["probe"] == label and r["layer"] in band]
            print(f"  {label:10s} band-mean AUROC {np.mean([r['auroc'] for r in mid]):.3f}")
        print("  read: high d_v2 AND ~0.5 d_length is a transfer result; high for both is not")


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
    ap.add_argument("--direction", required=True, choices=views.DIRECTIONS)
    ap.add_argument("--transfer", default=None, help="spec 1.2a, e.g. v1_nofiller100")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    lay = cfg.Layout("extraction", args.model, args.tag)

    if args.transfer:
        transfer_report(args, lay)
        return

    stem = mf.stem("probe_select", args.direction)
    view = views.read_view(lay, args.direction, "train")
    config = {"direction": args.direction, "band_tol": BAND_TOL, "len_tol": LEN_TOL,
              "selector": "lopo_paired_auroc", "interval": "clopper_pearson",
              "seed": cfg.SEED}
    inputs = {"view_key": view["view_key"]}
    with mf.Run(lay, stem, config, inputs) as run:
        rows, sel = layer_metrics(args, lay, run)
        run.artefact("_selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
        band = sel["band"]
        print(f"{args.direction}: best lopo AUROC {sel['best_lopo_auroc']:.3f} "
              f"(CP lower {sel['best_lopo_ci_lo']:.3f}, band threshold "
              f"{sel['band_threshold']:.3f})")
        print(f"  band {band[0]}-{band[-1]} ({len(band)} layers), primary L{sel['primary']} "
              f"(depth {sel['primary_depth']}, resid_len {sel['primary_resid_len_auroc']})")
        if sel["gate_failed"]:
            print("  ! NO layer passed the length gate — band is AUROC-only and provisional")
        elif not sel["gate_applied"]:
            print("  ! length gate not applied (no length heldout view) — band is provisional")


if __name__ == "__main__":
    main()
