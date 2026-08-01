"""Cache last-token activations for one dataset. The only script that touches the GPU.

Resumable by construction (spec 0.11): blobs are content-addressed per prompt, the
view is written before any forward pass and doubles as the work list.

    python cache_activations.py <model> --dataset story --split train
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import acts, config as cfg, manifest as mf, model as mdl, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train", choices=["train", "heldout", "all", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--append-task", action="store_true",
                    help="spec 0.2(a): append a rotated base task to persona/eval framings")
    args = ap.parse_args()

    out = cfg.results_dir("extraction", args.model)
    tok, model = mdl.load(args.model)
    L = model.config.num_hidden_layers
    hash_fn = mdl.prompt_hasher(tok)

    view, texts = views.build_view(args.dataset, args.split, hash_fn,
                                   append_task=args.append_task)
    stem = mf.stem("cache_activations", args.dataset, args.split)
    config = {"dataset": args.dataset, "split": args.split, "batch_size": args.batch_size,
              "append_task": args.append_task, "position": "last_token",
              "dtype": "float16", "seed": cfg.SEED}
    inputs = {"view_key": view["view_key"], "source_files": view["source_files"],
              "chat_template_sha": mdl.chat_template_sha(tok)}

    with mf.Run(out, stem, config, inputs) as run:
        views.write_view(args.model, view)          # work list first: no GPU needed
        acts.write_acts_manifest(args.model, {
            "model_id": args.model, "n_layers": L,
            "d_model": model.config.hidden_size, "dtype": "float16",
            "position": "last_token", "chat_template_sha": mdl.chat_template_sha(tok),
            "batch_size": args.batch_size})

        todo = acts.missing(args.model, [r["prompt_sha16"] for r in view["rows"]])
        run.resumed_from = len(set(r["prompt_sha16"] for r in view["rows"])) - len(todo)
        print(f"{args.dataset}/{args.split}: {view['n_pairs']} pairs, "
              f"{len(view['rows'])} prompts, {len(todo)} to compute "
              f"({run.resumed_from} cached)")

        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            h = mdl.last_token_hidden(tok, model, [texts[s] for s in chunk],
                                      batch_size=args.batch_size)
            for sha, row in zip(chunk, h):
                acts.write(args.model, sha, row)
            print(f"  {min(i + len(chunk), len(todo))}/{len(todo)}", end="\r")

        run.artefact(".txt").write_text(
            f"view_key {view['view_key']}\nprompts {len(view['rows'])}\n", encoding="utf-8")
        print(f"\nview_key {view['view_key'][:16]}  ->  {views.view_path(args.model, args.dataset, args.split).name}")


if __name__ == "__main__":
    main()
