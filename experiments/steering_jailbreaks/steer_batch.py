"""Run many 5.4 / 5.5 cells under one model load. GPU.

    python steer_batch.py <model> --tag T --script steer_single --jobs jobs.json

`jobs.json` is a list of argv tails, each exactly what you would pass to
steer_single.py / steer_induce.py after the model:

    [["--direction", "story_v2", "--sweep-layers", "15,17,18"],
     ["--arm", "noop", "--layers", "steer_band"]]

Scheduling only. Every job goes through the same parser, the same `resolve` and the
same `cell.run`, so a cell produced here is indistinguishable from the same cell
invoked alone -- same stem, same run_key, same per-cell resume.

All jobs are parsed and validated **before** the model is loaded, so a typo costs a
second rather than a load.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, generate as gen, model as mdl
from experiments.steering_jailbreaks import steer_single

# script -> (stem prefix, prompt_set, baseline outcome to steer)
SCRIPTS = {"steer_single": ("steer_single", "success", "success"),
           "steer_induce": ("steer_induce", "refusal", "refusal")}


def shared(args):
    """Flags every job inherits. A job may override any of them: it is parsed last."""
    out = ["--tag", args.tag] if args.tag else []
    out += ["--decoding", args.decoding,
            "--max-new-tokens", str(args.max_new_tokens),
            "--batch-size", str(args.batch_size),
            "--max-batch-tokens", str(args.max_batch_tokens)]
    if args.decode_seed is not None:
        out += ["--decode-seed", str(args.decode_seed)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--script", required=True, choices=list(SCRIPTS))
    ap.add_argument("--jobs", required=True, help="JSON list of argv tails")
    ap.add_argument("--decoding", default="greedy", choices=list(gen.DECODINGS))
    ap.add_argument("--decode-seed", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-batch-tokens", type=int, default=65536)
    args = ap.parse_args()

    script, prompt_set, want = SCRIPTS[args.script]
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not all(isinstance(j, list) for j in jobs):
        raise SystemExit(f"{args.jobs}: expected a JSON list of argv tails")

    # Validate everything first: a bad job should not cost a model load. The layer
    # checks need the depth, so read it from the config -- that is a small JSON, not
    # the 15 GB of weights.
    from transformers import AutoConfig
    n_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers

    base = shared(args)
    parsed = []
    for i, job in enumerate(jobs):
        p = steer_single.add_cell_args(argparse.ArgumentParser(prog=f"job[{i}]"))
        a = p.parse_args([args.model, *base, *[str(x) for x in job]])
        steer_single.resolve(a, prompt_set)          # illegal mode/direction/alpha
        cells = steer_single.cell_specs(a, n_layers)  # exactly one of --layers/--sweep
        for spec in cells:
            # cell_specs passes --layers through as a string; this is the same call
            # cell.run makes, so an out-of-band layer is caught here, not after a load.
            try:
                cfg.parse_layers(spec, n_layers, a.allow_out_of_band)
            except ValueError as e:
                raise SystemExit(f"job[{i}] {' '.join(str(x) for x in job)}: {e}")
        parsed.append((job, a, len(cells)))
    print(f"{len(parsed)} jobs validated for {args.script} ({prompt_set} set) "
          f"-> {sum(c for _, _, c in parsed)} cells, L={n_layers}, "
          f"batch {args.batch_size} / {args.max_batch_tokens} tokens")

    t0 = time.time()
    tok, model = mdl.load(args.model)
    print(f"model loaded once in {time.time() - t0:.0f}s\n")

    for i, (job, a, n) in enumerate(parsed, 1):
        print(f"[{i}/{len(parsed)}] {' '.join(str(x) for x in job)}  ({n} cell(s))")
        steer_single.run(a, script, prompt_set, want, tok=tok, model=model)

    print(f"\n{len(parsed)} jobs in {(time.time() - t0) / 60:.1f} min, one model load")


if __name__ == "__main__":
    main()
