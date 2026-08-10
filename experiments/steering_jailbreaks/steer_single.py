"""Spec 5.4: restore refusal on jailbreaks that succeeded. GPU.

    python steer_single.py <model> --direction story_v2 --mode ablate --sweep-layers 15,17,18
    python steer_single.py <model> --direction harm --mode add --layers band --alpha 0.5

--layers is one joint cell; --sweep-layers is one cell per element (spec 5.4.0). Mode
defaults to the direction's primary for this set: ablate for story/persona, add at +alpha
for harm/eval.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, generate as gen, model as mdl
from experiments.steering_jailbreaks import cell, sets

SCRIPT = "steer_single"
PROMPT_SET = "success"


def add_cell_args(ap):
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--direction", default=None, choices=cell.AXES)
    ap.add_argument("--mode", default=None, choices=["ablate", "add", "cap"])
    ap.add_argument("--arm", default="target", choices=["target", "noop", "random"])
    ap.add_argument("--layers", default=None, help=f"one joint cell: {cfg.LAYER_SPEC}")
    ap.add_argument("--sweep-layers", default=None, help="one cell per element; --width N for joint windows")
    ap.add_argument("--width", type=int, default=1, help="joint width of each swept cell")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="add only; signed -- the half with headroom on this set (spec 5.4b)")
    ap.add_argument("--tau-q", type=float, default=75.0, help="cap only; percentile of the two-pole corpus")
    ap.add_argument("--allow-out-of-band", action="store_true",
                    help="permit a chosen layer outside the reporting band (eval_v2 L9)")
    ap.add_argument("--decoding", default="greedy", choices=list(gen.DECODINGS),
                    help="spec 5.1; must match gen_baseline's, or the sets do not apply")
    ap.add_argument("--decode-seed", type=int, default=None, help="required if sampling")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    return ap


def cell_specs(args, n_layers):
    """-> [layer_spec], one per cell."""
    if bool(args.layers) == bool(args.sweep_layers):
        raise SystemExit("give exactly one of --layers / --sweep-layers (spec 5.4.0)")
    if args.layers:
        return [args.layers]
    members = cfg.parse_layers(args.sweep_layers, n_layers, args.allow_out_of_band)
    if args.width <= 1:
        return [str(l) for l in members]
    return [f"{l}-{l + args.width - 1}" for l in members
            if l + args.width - 1 <= members[-1]]


def resolve(args, prompt_set):
    direction = args.direction
    if args.arm != "noop" and direction is None:
        raise SystemExit("--direction is required unless --arm noop")
    mode = args.mode or (cell.PRIMARY[prompt_set].get(direction) if direction else "noop")
    if mode == "cap":
        if prompt_set != "success" or direction not in cell.CAP_SLOT:
            raise SystemExit("cap is scoped to story_v2/story_v1/persona on successes "
                             "(spec 5.4a); harm/eval are never capped")
    if mode == "add":
        if args.alpha == 0:
            raise SystemExit("alpha 0 is the noop arm, not a target cell (--arm noop)")
        # Only one sign has headroom on a given set: spec 0.5's restoring sign on the
        # successes, its mirror on the refusals. The magnitude is free, the sign is not --
        # the old guard pinned alpha positive, which is the same rule only for the mode each
        # set's PRIMARY mapping happens to assign. 1K_per_direction suppresses with -alpha
        # as well as with `ablate`, so the rule has to be stated on the sign itself.
        want = cell.RESTORE_SIGN[direction] * (1.0 if prompt_set == "success" else -1.0)
        if args.alpha * want < 0:
            raise SystemExit(
                f"--alpha {args.alpha:+g} pushes {direction} the way that has no headroom "
                f"on the {prompt_set} set; it needs alpha {'> 0' if want > 0 else '< 0'}")
    return direction, mode


def run(args, script, prompt_set, want, tok=None, model=None):
    """tok/model injected by steer_batch.py, which drives many cells under one load."""
    direction, mode = resolve(args, prompt_set)
    decode = gen.resolve_decode(args.decoding, args.decode_seed)
    if model is None:
        tok, model = mdl.load(args.model)
    n_layers = model.config.num_hidden_layers

    _, all_rows = sets.jailbreak_rows(args.model, args.tag, tok=tok)
    keep = set(sets.outcome_ids(sets.baseline_judged(args.model, args.tag), want))
    rows = [r for r in all_rows if r["unit_id"] in keep]
    if not rows:
        raise SystemExit(f"baseline has no {want} rows: nothing to steer")

    specs = cell_specs(args, n_layers)
    print(f"{script}: {direction or 'noop'} x {mode} x {len(specs)} cell(s) "
          f"on {len(rows)}/{len(all_rows)} {want} rows")
    for spec in specs:
        cell.run(script, args.model, args.tag, rows, prompt_set, direction, mode, spec,
                 args.arm, args.alpha, args.tau_q, n_layers, tok, model,
                 args.batch_size, args.max_batch_tokens, args.max_new_tokens,
                 decoding=args.decoding, decode=decode,
                 allow_out_of_band=args.allow_out_of_band)


if __name__ == "__main__":
    run(add_cell_args(argparse.ArgumentParser()).parse_args(), SCRIPT, PROMPT_SET, "success")
