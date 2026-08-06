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

from experiments.common import config as cfg, model as mdl
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
    ap.add_argument("--alpha", type=float, default=0.5, help="add only; positive throughout (spec 5.4b)")
    ap.add_argument("--tau-q", type=float, default=75.0, help="cap only; percentile of the two-pole corpus")
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
    members = cfg.parse_layers(args.sweep_layers, n_layers)
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
    if mode == "add" and args.alpha <= 0:
        raise SystemExit("alpha is positive throughout (spec 5.4b): the negative half has "
                         "no headroom on this prompt set")
    return direction, mode


def run(args, script, prompt_set, want):
    direction, mode = resolve(args, prompt_set)
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
                 args.batch_size, args.max_batch_tokens, args.max_new_tokens)


if __name__ == "__main__":
    run(add_cell_args(argparse.ArgumentParser()).parse_args(), SCRIPT, PROMPT_SET, "success")
