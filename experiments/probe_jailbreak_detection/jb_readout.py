"""Spec 3: project every probe onto the jailbreak subset. CPU only.

    python jb_readout.py <model>

Two readouts per row: the framed jailbreak `prompt` and the bare `request` underneath
it, request byte-identical. That within-row contrast is spec 3.1's primary test.

Readouts are (h - mu) . u_hat with experiment 1's stored mu (spec 0.6), so the
jailbreak numbers and the reference pole distributions share one origin and 3.4's
z-scores mean something.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, metrics as met, views


def load_probe(src, axis):
    stem = mf.stem("directions", axis)
    path = src.vectors / f"{stem}.pt"
    if not path.exists():
        return None
    mf.load_upstream(src.meta / f"{stem}_manifest.json")
    return torch.load(path, weights_only=False)


def readout(h, u, mu):
    """[n, L+1, d] -> [n, L+1] with the probe's own centre subtracted."""
    return np.einsum("nld,ld->nl", h - mu, u)


def reference_poles(src, axis, probe, band):
    """Spec 3.4 reference distributions, plus the probe's own-best layer.

    Pooled train + held-out = 65 points. Fifteen held-out points alone cannot carry a
    percentile at all; 65 gives resolution 1/65, which is why 3.4 restricts this to
    ordinal statements.

    The own-best layer is the peak of mean_paired_cos on train inside the band -- the
    same run-time rule experiment 2 uses, so the two experiments quote one layer per
    probe.
    """
    u, mu = probe["u"].numpy(), probe["mu"].numpy()
    mats = {s: acts.load_view_matrix(src, views.read_view(src, axis, s))
            for s in ("train", "heldout")}
    poles = {p: np.concatenate([readout(mats[s][p], u, mu) for s in ("train", "heldout")])
             for p in ("pos", "neg")}
    tr = mats["train"]
    mpc = [float(np.mean(met.unit(tr["pos"][:, l, :] - tr["neg"][:, l, :]) @ u[l]))
           for l in range(u.shape[0])]
    return poles, max(band, key=lambda l: mpc[l]), mpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--axes", default=",".join(views.DIRECTIONS))
    args = ap.parse_args()

    src = cfg.acts_layout(args.model, args.tag)      # extraction: vectors + blob cache
    lay = cfg.Layout("probe_jailbreak_detection", args.model, args.tag, acts_cache=False)

    view = views.read_view(src, "jailbreaks", args.split)
    m = acts.load_view_matrix(src, view)
    meta, ids = view["meta"], m["pair_ids"]
    ntok = {r["row_id"]: {} for r in view["rows"]}
    for r in view["rows"]:
        ntok[r["row_id"]][r["pole"]] = r.get("n_tokens")

    names = [a for a in args.axes.split(",") if a]
    probes, skipped = {}, []
    for a in names:
        p = load_probe(src, a)
        (probes.__setitem__(a, p) if p is not None else skipped.append(a))
    if not probes:
        raise SystemExit("no direction vectors: run extract_direction.py first")
    if skipped:
        print(f"! no vector for {skipped}: not in the readout")

    Lp1 = next(iter(probes.values()))["u"].shape[0]
    band = cfg.band(Lp1 - 1)
    framed = np.stack([readout(m["pos"], probes[a]["u"].numpy(), probes[a]["mu"].numpy())
                       for a in probes])            # [n_probes, n_rows, L+1]
    bare = np.stack([readout(m["neg"], probes[a]["u"].numpy(), probes[a]["mu"].numpy())
                     for a in probes])
    ref, best, mpc = {}, {}, {}
    for a in probes:
        ref[a], best[a], mpc[a] = reference_poles(src, a, probes[a], band)

    rows = [{"row_id": i, "family": meta[i]["family"], "source": meta[i]["source"],
             "technique": meta[i]["technique"], "template_id": meta[i]["template_id"],
             "request_sha8": mf.sha256_obj(meta[i]["request"])[:8],
             "category": meta[i]["category"],
             "base_task_source": meta[i]["base_task_source"],
             "split": meta[i].get("split"), "n_chars": meta[i].get("n_chars"),
             "n_tokens_framed": ntok[i].get("pos"), "n_tokens_bare": ntok[i].get("neg")}
            for i in ids]

    stem = mf.stem("jb_readout")
    config = {"axes": list(probes), "split": args.split, "n_rows": len(ids),
              "subsample": view.get("subsample"), "position": "last_token",
              "readout": "(h - mu) . u_hat", "reference": "train+heldout poles",
              "layer_rule": "mpc_peak_in_band", "best_layers": best, "seed": cfg.SEED}
    inputs = {"jb_view_key": view["view_key"], "source_files": view["source_files"],
              "direction_run_keys": {a: probes[a].get("run_key") for a in probes}}

    with mf.Run(lay, stem, config, inputs) as run:
        torch.save({"probes": list(probes), "row_ids": ids, "n_layers": Lp1 - 1,
                    "framed": torch.from_numpy(framed.astype("float32")),
                    "bare": torch.from_numpy(bare.astype("float32")),
                    "ref": {a: {p: torch.from_numpy(v.astype("float32"))
                                for p, v in ref[a].items()} for a in ref},
                    "best_layer": best, "mean_paired_cos": mpc,
                    "jb_view_key": view["view_key"], "run_key": run.run_key},
                   run.artefact(".pt"))
        with run.artefact("_rows.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        nw = len({r["template_id"] for r in rows})
        nq = len({r["request_sha8"] for r in rows})
        print(f"{len(ids)} rows, {nw} wrappers, {nq} requests, {len(probes)} probes, "
              f"{Lp1} layers   band {band[0]}-{band[-1]}")
        print("  effective n after spec 0.7 clustering: "
              f"{nw} by template_id, {nq} by request")
        for i, a in enumerate(probes):
            d = (framed[i] - bare[i])[:, band].mean()
            print(f"  {a:10s} L{best[a]:<3d} band-mean framed-minus-bare delta {d:+8.2f}")


if __name__ == "__main__":
    main()
