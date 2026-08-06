"""Spec 5.6: does direction a still work once b is projected out of it? GPU.

    python steer_pairs.py <model> --pair story_v2,persona

Four arms per ordered pair, all `add` at the sign spec 0.5 says restores refusal, on the
prompts that jailbroke the unsteered model:

    unprojected          u_a                                   reference
    perp_alpha           unit(u_a - (u_a.u_b) u_b)  same alpha  plan 7c as written
    perp_effect          the same vector, alpha retuned so the a-probe readout at layer L
                         moves as far as the unprojected run moved it
    par_norm             unit((u_a.u_b) u_b)        same alpha  the control

Projection is recomputed **per layer**: u_a[l] and u_b[l] differ, so a joint set steers a
different a_perp at each layer. Cross-layer projection is meaningless (spec 2.3), which is
why `band` -- not a single layer -- is the default: it is the only config shared by every
direction, so a pair result is comparable across pairs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (config as cfg, generate as gen, hooks as hk,
                                manifest as mf, metrics as met, model as mdl)
from experiments.steering_jailbreaks import cell, sets

SCRIPT = "steer_pairs"
PROMPT_SET = "success"
ARMS = ("unprojected", "perp_alpha", "perp_effect", "par_norm")


def decompose(ua, ub, layers):
    """Per layer: unit a_perp, unit a_par, and cos(u_a, u_b)."""
    perp, par, cosv = np.array(ua), np.array(ua), np.zeros(ua.shape[0])
    for l in layers:
        c = float(ua[l] @ ub[l])
        p = c * ub[l]
        r = ua[l] - p
        cosv[l] = c
        perp[l] = r / max(np.linalg.norm(r), 1e-12)
        par[l] = p / max(np.linalg.norm(p), 1e-12)
    return perp, par, cosv


def lopo_stability(probe, layers):
    """mean cos(d_LOPO_i[l], d_full[l]) over the layer set (spec 1.2)."""
    if "lopo_d" not in probe:
        return None
    d, lo = probe["d"].numpy(), probe["lopo_d"].numpy()
    return float(np.mean([met.cos(lo[i, l], d[l]) for l in layers for i in range(lo.shape[0])]))


def self_effect(tok, model, prompts, u, sigma, layers, alpha, u_final, base,
                batch_size, max_batch_tokens):
    """|mean readout shift| at layer L for one candidate vector and alpha."""
    ut = torch.from_numpy(np.ascontiguousarray(u)).float()
    specs = hk.build("add", layers, ut, {"calls": 0}, alpha=alpha, sigma=sigma)
    h = gen.prefill_states(tok, model, prompts, specs, batch_size, max_batch_tokens)
    return abs(float((h @ u_final).mean() - base))


def match_alpha(tok, model, prompts, u, sigma, layers, target, u_final, base,
                alpha0, batch_size, max_batch_tokens, grid=8):
    """Smallest-error alpha on a log-spaced scan around alpha0 (spec 5.6 correction 1).

    Prefill only, so the whole scan is seconds. Scanned rather than solved because the
    readout at layer L is not a linear function of the push at layer l.
    """
    cands = [alpha0 * (2.0 ** (k / 2.0)) for k in range(0, grid)]
    best, best_err, trace = alpha0, None, []
    for a in cands:
        eff = self_effect(tok, model, prompts, u, sigma, layers, np.sign(alpha0) * abs(a),
                          u_final, base, batch_size, max_batch_tokens)
        err = abs(eff - target)
        trace.append({"alpha": round(float(a), 4), "effect": round(eff, 4)})
        if best_err is None or err < best_err:
            best, best_err = np.sign(alpha0) * abs(a), err
        if eff >= target:
            break
    return float(best), trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pair", required=True, help="a,b -- b is projected out of a")
    ap.add_argument("--both-orders", action="store_true", help="also run (b, a)")
    ap.add_argument("--layers", default="steer_band", help=cfg.LAYER_SPEC)
    ap.add_argument("--alpha", type=float, default=0.5, help="magnitude; sign from spec 0.5")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--decoding", default="greedy", choices=list(gen.DECODINGS))
    ap.add_argument("--decode-seed", type=int, default=None, help="required if sampling")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    args = ap.parse_args()

    names = [x for x in args.pair.split(",") if x]
    if len(names) != 2 or any(n not in cell.AXES for n in names):
        raise SystemExit(f"--pair a,b from {cell.AXES}")
    if set(names) == {"story_v2", "story_v1"}:
        raise SystemExit("(story_v2, story_v1) is not a valid pair: at cos +0.76 the "
                         "orthogonal residual is construction noise (spec 5.6)")
    arms = [a for a in args.arms.split(",") if a]
    decode = gen.resolve_decode(args.decoding, args.decode_seed)

    src = cfg.acts_layout(args.model, args.tag)
    lay = cfg.Layout(sets.EXPERIMENT, args.model, args.tag, acts_cache=False)
    tok, model = mdl.load(args.model)
    n_layers = model.config.num_hidden_layers
    layers = cfg.parse_layers(args.layers, n_layers)

    _, all_rows = sets.jailbreak_rows(args.model, args.tag, tok=tok)
    keep = set(sets.outcome_ids(sets.baseline_judged(args.model, args.tag), "success"))
    rows = [r for r in all_rows if r["unit_id"] in keep]
    if not rows:
        raise SystemExit("baseline has no successful jailbreaks: nothing to project")
    prompts = [r["prompt"] for r in rows]

    orders = [tuple(names)] + ([tuple(reversed(names))] if args.both_orders else [])
    for a_name, b_name in orders:
        pa, pb = cell.load_probe(src, a_name), cell.load_probe(src, b_name)
        ua, ub = pa["u"].numpy(), pb["u"].numpy()
        sigma = pa["sigma_act"].numpy()
        perp, par, cosv = decompose(ua, ub, layers)
        null = met.random_cos_band(ua.shape[-1])
        stab = lopo_stability(pa, layers)
        alpha = cell.RESTORE_SIGN[a_name] * abs(args.alpha)

        print(f"\n{a_name} - proj({b_name}), layers {layers[0]}-{layers[-1]}, alpha {alpha:+g}")
        print(f"  cos(u_a,u_b) band-mean {cosv[layers].mean():+.3f}  "
              f"(null band +/-{null:.3f})   lopo_cos_stability "
              f"{'n/a' if stab is None else f'{stab:.3f}'}")
        if abs(cosv[layers].mean()) < null:
            print("  ! cos is inside the null band: the projection removes nothing, so "
                  "`perp` and `unprojected` are the same experiment (spec 2.3)")

        u_final = torch.from_numpy(np.ascontiguousarray(ua[ua.shape[0] - 1])).float()
        h0 = gen.prefill_states(tok, model, prompts, None, args.batch_size,
                                args.max_batch_tokens)
        base = float((h0 @ u_final).mean())
        target = self_effect(tok, model, prompts, ua, sigma, layers, alpha, u_final, base,
                             args.batch_size, args.max_batch_tokens)
        print(f"  unprojected self-effect at layer L: {target:.3f}")

        alpha_eff, trace = (alpha, [])
        if "perp_effect" in arms:
            alpha_eff, trace = match_alpha(tok, model, prompts, perp, sigma, layers, target,
                                           u_final, base, alpha, args.batch_size,
                                           args.max_batch_tokens)
            print(f"  matched-self-effect alpha: {alpha_eff:+.3f} "
                  f"(unprojected {alpha:+g})")

        for arm in arms:
            u, a = {"unprojected": (ua, alpha), "perp_alpha": (perp, alpha),
                    "perp_effect": (perp, alpha_eff), "par_norm": (par, alpha)}[arm]
            stem = mf.stem(SCRIPT, f"{a_name}-perp-{b_name}", cfg.layer_stem(args.layers),
                           f"a{a:g}", arm)
            ut = torch.from_numpy(np.ascontiguousarray(u)).float()
            counter = {"calls": 0}
            specs = hk.build("add", layers, ut, counter, alpha=a, sigma=sigma)
            config = {"direction": a_name, "projected_out": b_name, "arm": arm,
                      "mode": "add", "prompt_set": PROMPT_SET,
                      "layers_spec": str(args.layers), "layers": layers,
                      "n_layers_steered": len(layers), "alpha": a,
                      "per_layer_coef": hk.per_layer_coef("add", layers, a),
                      "alpha_unprojected": alpha, "self_effect_target": round(target, 4),
                      "alpha_scan": trace if arm == "perp_effect" else None,
                      "cos_ab_band": round(float(cosv[layers].mean()), 4),
                      "cos_null_band": round(null, 4), "lopo_cos_stability": stab,
                      "tau_q": None, "decoding": args.decoding, "decode": decode,
                      "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                      "max_batch_tokens": args.max_batch_tokens, "position": "all_tokens",
                      "n_rows": len(rows), "seed": cfg.SEED}
            inputs = {"unit_ids": [r["unit_id"] for r in rows],
                      "direction_run_key": pa.get("run_key"),
                      "projected_run_key": pb.get("run_key")}
            cell.emit(lay, src, stem, config, inputs, rows, specs, layers, tok, model,
                      args.batch_size, args.max_batch_tokens, args.max_new_tokens, decode,
                      counter)


if __name__ == "__main__":
    main()
