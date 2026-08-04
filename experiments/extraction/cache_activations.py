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


def plan_batches(todo, ntok, batch_size, max_batch_tokens):
    """Length-sorted batches under a padded-token budget.

    A fixed batch size is fine for the extraction tables but not for jailbreaks: one
    47k-char prompt batched with seven short ones pads all eight to its length, and
    output_hidden_states materialises every layer over the padded sequence before the
    last position is sliced out. Sorting by length keeps batches homogeneous and the
    budget caps the one prompt that has to run alone.
    """
    order = sorted(todo, key=lambda s: ntok.get(s, 0))
    batches, cur = [], []
    for s in order:
        trial = cur + [s]
        longest = max(ntok.get(x, 0) for x in trial)
        if cur and (len(trial) > batch_size or len(trial) * longest > max_batch_tokens):
            batches.append(cur)
            cur = [s]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train", choices=["train", "heldout", "all", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag", default=None, help="run label; also read from $RUN_TAG (default 'base')")
    ap.add_argument("--append-task", action="store_true",
                    help="spec 0.2(a): append a rotated base task to persona/eval framings")
    ap.add_argument("--subsample-n", type=int, default=None,
                    help="spec 3: subsample the dataset to n pairs (jailbreaks only)")
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="padded-token budget per batch; jailbreaks span 57-47k chars")
    ap.add_argument("--poles", default=None,
                    help="comma-separated arms to cache; default every arm the loader has. "
                         "'pos' makes a single-arm view with no contrast (spec 3)")
    args = ap.parse_args()
    poles = [p for p in (args.poles or "").split(",") if p] or None

    subsample = None if args.subsample_n is None else {
        "n": args.subsample_n, "strategy": "template_diverse", "seed": cfg.SEED,
        "max_per_template": views.JB_MAX_PER_TEMPLATE,
        "family_alloc": views.JB_FAMILY_ALLOC, "filter": views.JB_FILTER}

    lay = cfg.Layout("extraction", args.model, args.tag)
    print(f"run {lay}")
    tok, model = mdl.load(args.model)
    L = model.config.num_hidden_layers
    hash_fn = mdl.prompt_hasher(tok)

    view, texts = views.build_view(args.dataset, args.split, hash_fn,
                                   append_task=args.append_task, subsample=subsample,
                                   token_info=mdl.token_info_fn(tok), poles=poles)
    # No subsample knob in the stem: the view pointer views/<ds>__<split>.json is
    # single-valued per tag, so two subsamples of one table cannot coexist anyway.
    # A changed subsample archives the prior manifest through the normal run_key path.
    stem = mf.stem("cache_activations", args.dataset, args.split)
    config = {"dataset": args.dataset, "split": args.split, "batch_size": args.batch_size,
              "append_task": args.append_task, "subsample": subsample,
              "poles": view["poles"], "position": "last_token",
              "dtype": "float16", "seed": cfg.SEED}
    inputs = {"view_key": view["view_key"], "source_files": view["source_files"],
              "chat_template_sha": mdl.chat_template_sha(tok)}

    with mf.Run(lay, stem, config, inputs) as run:
        views.write_view(lay, view)                 # work list first: no GPU needed
        acts.write_acts_manifest(lay, {
            "model_id": args.model, "n_layers": L,
            "d_model": model.config.hidden_size, "dtype": "float16",
            "position": "last_token", "chat_template_sha": mdl.chat_template_sha(tok),
            "batch_size": args.batch_size})

        todo = acts.missing(lay, [r["prompt_sha16"] for r in view["rows"]])
        run.resumed_from = len(set(r["prompt_sha16"] for r in view["rows"])) - len(todo)
        ntok = {r["prompt_sha16"]: r.get("n_tokens") or 0 for r in view["rows"]}
        batches = plan_batches(todo, ntok, args.batch_size, args.max_batch_tokens)
        print(f"{args.dataset}/{args.split}: {view['n_pairs']} pairs, "
              f"{len(view['rows'])} prompts, {len(todo)} to compute "
              f"({run.resumed_from} cached) in {len(batches)} batches, "
              f"longest {max(ntok.values(), default=0)} tokens")

        done = 0
        for chunk in batches:
            h = mdl.last_token_hidden(tok, model, [texts[s] for s in chunk],
                                      batch_size=len(chunk))
            for sha, row in zip(chunk, h):
                acts.write(lay, sha, row)
            done += len(chunk)
            print(f"  {done}/{len(todo)}", end="\r")

        print(f"\nview_key {view['view_key'][:16]}  ->  "
              f"{views.view_path(lay, args.dataset, args.split).name}")


if __name__ == "__main__":
    main()
