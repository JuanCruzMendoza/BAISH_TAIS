"""Spec 5.6: does direction a still work once b is projected out of it? GPU.

    python steer_pairs.py <model> --pair story_v2,persona

Four arms per ordered pair, all `add` at the sign spec 0.5 says restores refusal, on the
prompts that jailbroke the unsteered model:

    unprojected          u_a                                   reference
    perp_alpha           unit(u_a - (u_a.u_b) u_b)  same alpha  plan 7c as written
    perp_effect          the same vector, alpha retuned so the a-probe readout at layer L
                         moves as far as the unprojected run moved it
    par_component        (u_a.u_b) u_b              same alpha  the control -- NOT unit
                         norm, so it delivers the b-content the reference actually has

Projection is recomputed **per layer**: u_a[l] and u_b[l] differ, so a joint set steers a
different a_perp at each layer. Cross-layer projection is meaningless (spec 2.3), which is
why `band` -- not a single layer -- is the default: it is the only config shared by every
direction, so a pair result is comparable across pairs.

`unprojected` is plain `add` with u_a, so it depends on **a alone**. Its stem therefore drops
the `-perp-b` half and it is generated once per (a, layers, alpha) rather than once per
ordered pair -- at 50_per_direction the old naming produced 8 byte-identical cells for 4
configs. For the directions whose 5.4 mode is already `add` (harm, eval) it also reproduces
that steer_single cell exactly, so it is skipped when the twin is present; `--force-unprojected`
regenerates it anyway.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (config as cfg, generate as gen, hooks as hk,
                                manifest as mf, metrics as met, model as mdl)
from experiments.steering_jailbreaks import cell, sets

SCRIPT = "steer_pairs"
ARMS = ("unprojected", "perp_alpha", "perp_effect", "par_component")

# Which 5.4/5.5 script owns each prompt set, and so which stem `unprojected` would duplicate.
OWNER = {"success": "steer_single", "refusal": "steer_induce"}
# The sign of `add` that pushes the wanted way on each set: 0.5's restoring sign on the
# successes, its mirror on the refusals. Same rule as steer_single/steer_induce's mode map.
SET_SIGN = {"success": +1.0, "refusal": -1.0}


def decompose(ua, ub, layers):
    """Per layer: unit a_perp, the a_par *component*, and cos(u_a, u_b).

    `par` is deliberately **not** unit-normalised: |p| = |cos|, so pushing it at the
    reference alpha delivers exactly the b-content the reference push delivers. Normalising
    it would inject 1/|cos| times that -- 4.2x at cos 0.240 -- and a sufficiency arm given
    four times the dose it is meant to model answers nothing.
    """
    perp, par, cosv = np.array(ua), np.array(ua), np.zeros(ua.shape[0])
    for l in layers:
        c = float(ua[l] @ ub[l])
        p = c * ub[l]
        r = ua[l] - p
        cosv[l] = c
        perp[l] = r / max(np.linalg.norm(r), 1e-12)
        par[l] = p
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


def single_twin(lay, a_name, layer_spec, alpha, config, batch_size, max_batch_tokens):
    """The 5.4/5.5 cell `unprojected` would duplicate, if it exists and matches.

    Same vector, mode, layers, alpha and prompt set means the same generations -- byte
    identical, since composition is fixed by the prompt set. Compared on the semantic
    knobs only: steer_single's config carries cap/tau keys this script never sets.

    Batching is compared too, unlike the semantic knobs: greedy is bit-reproducible only at
    fixed batch size *and* composition (spec 0.10), so a twin generated at another batch
    size is not a reference the perp arms can be paired against.
    """
    twin_script = OWNER[config["prompt_set"]]
    stem = cell.stem_for(twin_script, a_name, "add", layer_spec, alpha, None, "target")
    path = Path(lay.meta) / f"{stem}_manifest.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("status") != "complete":
        return None
    keys = ("direction", "mode", "prompt_set", "layers", "alpha", "decoding", "decode",
            "max_new_tokens", "n_rows")
    prior = m.get("config", {})
    if any(prior.get(k) != config.get(k) for k in keys):
        return None
    if (prior.get("batch_size"), prior.get("max_batch_tokens")) != (batch_size,
                                                                   max_batch_tokens):
        print(f"  ! {stem} matches semantically but was generated at batch "
              f"{prior.get('batch_size')}/{prior.get('max_batch_tokens')}, not "
              f"{batch_size}/{max_batch_tokens} -- not reusable as a reference")
        return None
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--pair", required=True, help="a,b -- b is projected out of a")
    ap.add_argument("--both-orders", action="store_true", help="also run (b, a)")
    ap.add_argument("--layers", default="steer_band", help=cfg.LAYER_SPEC)
    ap.add_argument("--prompt-set", default="success", choices=list(OWNER),
                    help="success = restore refusal (5.4), refusal = induce compliance (5.5)")
    ap.add_argument("--alpha", type=float, default=0.5, help="magnitude; sign from spec 0.5")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--allow-out-of-band", action="store_true",
                    help="permit an anchor's chosen layer outside the reporting band")
    ap.add_argument("--force-unprojected", action="store_true",
                    help="generate the reference arm even when steer_single already has it")
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
    layers = cfg.parse_layers(args.layers, n_layers, args.allow_out_of_band)
    # Same escape as steer_single's: the projection runs at the anchor's *chosen* layer,
    # which a second model can put outside the band.
    out_of_band = sorted(set(layers) - set(cfg.band(n_layers)))

    _, all_rows = sets.jailbreak_rows(args.model, args.tag, tok=tok)
    keep = set(sets.outcome_ids(sets.baseline_judged(args.model, args.tag), args.prompt_set))
    rows = [r for r in all_rows if r["unit_id"] in keep]
    if not rows:
        raise SystemExit(f"baseline has no {args.prompt_set} rows: nothing to project")
    prompts = [r["prompt"] for r in rows]

    orders = [tuple(names)] + ([tuple(reversed(names))] if args.both_orders else [])
    for a_name, b_name in orders:
        pa, pb = cell.load_probe(src, a_name), cell.load_probe(src, b_name)
        ua, ub = pa["u"].numpy(), pb["u"].numpy()
        sigma = pa["sigma_act"].numpy()
        perp, par, cosv = decompose(ua, ub, layers)
        null = met.random_cos_band(ua.shape[-1])
        stab = lopo_stability(pa, layers)
        alpha = cell.RESTORE_SIGN[a_name] * SET_SIGN[args.prompt_set] * abs(args.alpha)

        print(f"\n{a_name} - proj({b_name}) on {args.prompt_set}, "
              f"layers {layers[0]}-{layers[-1]}, alpha {alpha:+g}")
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
                    "perp_effect": (perp, alpha_eff), "par_component": (par, alpha)}[arm]
            # |u| is 1 except for par_component, where it is |cos|. alpha alone therefore
            # does not state that arm's push, so record the fraction it actually delivers.
            push_frac = round(float(np.mean([np.linalg.norm(u[l]) for l in layers])), 4)
            # `unprojected` is u_a alone: no b in the stem, and none in the run_key
            # either, or the second pair would archive the first pair's manifest and
            # regenerate the identical rows.
            solo = arm == "unprojected"
            # The prompt set is not in the stem and does not need to be: SET_SIGN flips the
            # sign of `a` between the two sets, so a success cell and a refusal cell at the
            # same |alpha| always stem differently (`a-0.75` vs `a0.75`).
            stem = mf.stem(SCRIPT, a_name if solo else f"{a_name}-perp-{b_name}",
                           cfg.layer_stem(args.layers), f"a{a:g}", arm)
            ut = torch.from_numpy(np.ascontiguousarray(u)).float()
            counter = {"calls": 0}
            specs = hk.build("add", layers, ut, counter, alpha=a, sigma=sigma)
            config = {"direction": a_name, "projected_out": None if solo else b_name,
                      "arm": arm,
                      "mode": "add", "prompt_set": args.prompt_set,
                      "layers_spec": str(args.layers), "layers": layers,
                      "n_layers_steered": len(layers), "alpha": a,
                      "push_frac": push_frac,
                      "per_layer_coef": hk.per_layer_coef("add", layers, a),
                      "alpha_unprojected": None if solo else alpha,
                      "self_effect_target": None if solo else round(target, 4),
                      "alpha_scan": trace if arm == "perp_effect" else None,
                      "cos_ab_band": None if solo else round(float(cosv[layers].mean()), 4),
                      "cos_null_band": None if solo else round(null, 4),
                      "lopo_cos_stability": stab,
                      "tau_q": None, "decoding": args.decoding, "decode": decode,
                      "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                      "max_batch_tokens": args.max_batch_tokens, "position": "all_tokens",
                      "n_rows": len(rows), "seed": cfg.SEED}
            # Added only when it fires, so an in-band cell keeps the config dict -- and so
            # the run_key -- it already had, and no completed cell is invalidated by the
            # escape existing. Same rule as cell.run's.
            if out_of_band:
                config["out_of_band"] = out_of_band
            inputs = {"unit_ids": [r["unit_id"] for r in rows],
                      "direction_run_key": pa.get("run_key"),
                      "projected_run_key": None if solo else pb.get("run_key")}
            if solo and not args.force_unprojected:
                twin = single_twin(lay, a_name, args.layers, a, config,
                                   args.batch_size, args.max_batch_tokens)
                if twin is not None:
                    print(f"  {arm}: identical to {twin} -- skipped. Read that cell as "
                          f"the reference (--force-unprojected to regenerate)")
                    continue
            cell.emit(lay, src, stem, config, inputs, rows, specs, layers, tok, model,
                      args.batch_size, args.max_batch_tokens, args.max_new_tokens, decode,
                      counter)


if __name__ == "__main__":
    main()
