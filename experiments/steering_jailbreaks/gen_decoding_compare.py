"""Spec 5.1: which decoding do the steering runs use? GPU.

    python gen_decoding_compare.py <model> --tag 50_per_direction

50 rows of the jailbreak subset, stratified by source x family, under greedy plus two
sampled configs at three seeds each: 7 cells, 350 generations. Judge every cell, then read
`aggregate.py`'s decoding table and pick.

The decision the table informs, and the two criteria conflict: a sampled config with
higher ASR leaves more headroom for refusal to be restored, while greedy is deterministic,
so a steering delta is steering and not sampling. Whatever is chosen has to be passed to
every later script, because `decoding` is inside every cell's run_key -- changing it
correctly invalidates downstream cells rather than silently mixing decodings. Note that a
sampled choice makes ASR a rate over n>=5 samples per cell, which multiplies section 5.4.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import config as cfg, generate as gen, manifest as mf, model as mdl
from experiments.steering_jailbreaks import sets

# Seeds per config from gen.DECODINGS. Greedy is deterministic, so one run is the cell.
N_SEEDS = {"greedy": 1, "t0.7p0.9": 3, "t1.0p0.95": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-rows", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    args = ap.parse_args()

    lay = cfg.Layout(sets.EXPERIMENT, args.model, args.tag, acts_cache=False)
    src = cfg.acts_layout(args.model, args.tag)
    print(f"run {lay}")
    tok, model = mdl.load(args.model)
    view, all_rows = sets.jailbreak_rows(args.model, args.tag, tok=tok)
    rows = sets.stratified(all_rows, args.n_rows, cfg.SEED)

    cells = sum(N_SEEDS.values())
    print(f"{len(rows)}/{len(all_rows)} rows, {len(N_SEEDS)} configs, {cells} cells, "
          f"{cells * len(rows)} generations")
    print(f"  strata: {len({(r['source'], r['family']) for r in rows})} source x family cells")

    for label, n_seeds in N_SEEDS.items():
        for i in range(n_seeds):
            decode = gen.resolve_decode(label, cfg.SEED + i if n_seeds > 1 else None)
            stem = mf.stem("gen_decoding_compare", label,
                           *((f"s{i}",) if n_seeds > 1 else ()))
            config = {"prompt_set": "decoding", "mode": "baseline", "arm": "target",
                      "direction": None, "decoding": label, "decode": decode,
                      "seed_index": i if n_seeds > 1 else None,
                      "layers_spec": None, "n_layers_steered": 0, "alpha": None,
                      "per_layer_coef": None, "tau_q": None,
                      "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
                      "max_batch_tokens": args.max_batch_tokens,
                      "n_rows": len(rows), "n_rows_pool": len(all_rows),
                      "stratify": "source x family", "seed": cfg.SEED}
            inputs = {"unit_ids": [r["unit_id"] for r in rows],
                      "jb_view_key": view["view_key"],
                      "source_files": view["source_files"],
                      "chat_template_sha": mdl.chat_template_sha(tok)}

            with mf.Run(lay, stem, config, inputs, resumable=True) as run:
                done = run.resume_from(".jsonl")
                with run.open_append(".jsonl") as fh:
                    info = gen.run(tok, model, rows, gen.Sink(fh), done, args.batch_size,
                                   args.max_batch_tokens, args.max_new_tokens,
                                   decode=decode,
                                   progress=lambda i_, n_: print(f"    {i_}/{n_}", end="\r"))
                print(f"\n  {stem}: {info['n_rows_run']} generated, {len(done)} cached")

    print("\nnext: judge every cell, then read aggregate.py's _decoding.csv")
    print(f"  for f in {lay.meta}/gen_decoding_compare*.jsonl; do "
          f"python judge_strongreject.py \"$f\"; done")
    print(f"  python aggregate.py {args.model} --tag {cfg.tag(args.tag)}")
    print("\nThe chosen decoding must then be passed to gen_baseline.py and every steer_* "
          "run; it is inside run_key, so mixing decodings cannot happen silently.")


if __name__ == "__main__":
    main()
