"""Spec 5.2: unsteered greedy generations on the 100-row jailbreak subset. GPU.

    python gen_baseline.py <model> --tag 50_per_direction

Defines the two steering sets: judge this output, then 5.4 runs on its successes and
5.5 on its refusals. Resumable at batch granularity.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, generate as gen, manifest as mf, model as mdl
from experiments.steering_jailbreaks import sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="all", choices=["all", "val", "test"])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    args = ap.parse_args()

    lay = cfg.Layout(sets.EXPERIMENT, args.model, args.tag, acts_cache=False)
    print(f"run {lay}")
    tok, model = mdl.load(args.model)
    view, rows = sets.jailbreak_rows(args.model, args.tag, tok=tok, split=args.split)

    stem = mf.stem("gen_baseline")
    config = {"split": args.split, "decoding": "greedy", "max_new_tokens": args.max_new_tokens,
              "batch_size": args.batch_size, "max_batch_tokens": args.max_batch_tokens,
              "n_rows": len(rows), "seed": cfg.SEED}
    inputs = {"jb_view_key": view["view_key"], "source_files": view["source_files"],
              "chat_template_sha": mdl.chat_template_sha(tok)}

    with mf.Run(lay, stem, config, inputs, resumable=True) as run:
        done = run.resume_from(".jsonl")
        with run.open_append(".jsonl") as fh:
            info = gen.run(tok, model, rows, gen.Sink(fh), done,
                           args.batch_size, args.max_batch_tokens, args.max_new_tokens,
                           progress=lambda i, n: print(f"  {i}/{n}", end="\r"))
        print(f"\n{len(rows)} rows, {len(done)} cached, {info['n_rows_run']} generated "
              f"in {info['n_batches_run']}/{info['n_batches']} batches")
        print(f"  next: judge_strongreject.py {run.artefact('.jsonl')}")


if __name__ == "__main__":
    main()
