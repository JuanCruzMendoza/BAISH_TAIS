"""Dataset loaders and activation views (spec 0.8).

A view is the identity of a dataset or a subsample of one: an ordered list of
{row_id, pole, prompt_sha16}. view_key hashes that list, so if the sampling code
changes and the same {n, seed} yields different rows, the key changes and every
downstream result correctly goes stale.
"""
import csv
import json
import random
import sys
from pathlib import Path  # noqa: F401  (used by the v1 loaders' error messages)

from . import config as cfg
from . import manifest as mf
from . import prompts as pr

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

DIRECTIONS = ["story_v2", "story_v1", "harm", "harm_v2", "persona", "eval", "length"]

# ------------------------------------------------------------------- readers


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def _jsonl(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


# ---------------------------------------------------------- direction loaders
# Each returns [{pair_id, pos, neg, meta}], already rendered to final prompt text.


def _story_v2(split):
    name = "pairs.jsonl" if split == "train" else "pairs_heldout.jsonl"
    src = cfg.DATA / "story_mode_v2" / name
    return src, [{"pair_id": r["pair_id"],
                  "pos": pr.continuation(r["text_narrative"]),
                  "neg": pr.continuation(r["text_nonnarrative"]),
                  "meta": {k: r[k] for k in ("narr_mode", "nonnarr_style", "realism",
                                             "tense_polarity", "context")}}
                 for r in _jsonl(src)]


def _harm(split):
    name = "harm_selected_pairs.csv" if split == "train" else "harm_selected_pairs_heldout.csv"
    src = cfg.DATA / "harm" / name
    return src, [{"pair_id": f"harm-{r['jbb_index']}",
                  "pos": r["harmful_goal"], "neg": r["benign_goal"],
                  "meta": {"jbb_index": r["jbb_index"], "category": r["category"]}}
                 for r in _csv(src)]


def _harm_v2(split):
    """v1 `harm` under four framing families; its `bare` cell is v1 unchanged."""
    name = "pairs.jsonl" if split == "train" else "pairs_heldout.jsonl"
    src = cfg.DATA / "harm_v2" / name
    return src, [{"pair_id": r["pair_id"],
                  "pos": r["prompt_harmful"], "neg": r["prompt_benign"],
                  "meta": {"family": r["family"], "arm": r["arm"],
                           "framing_id": r["framing_id"], "jbb_index": r["jbb_index"],
                           "category": r["category"]}}
                 for r in _jsonl(src)]


def _attach(rows, tasks):
    """Spec 0.2(a): one task per pair, byte-identical across the pair."""
    if not tasks:
        return rows
    if len(tasks) < len(rows):
        raise RuntimeError(f"need {len(rows)} tasks, task pool has {len(tasks)}")
    for r, t in zip(rows, tasks):
        r["pos"], r["neg"] = pr.with_task(r["pos"], t["text"]), pr.with_task(r["neg"], t["text"])
        r["meta"] = {**r["meta"], "task_harmful": t["harmful"], "task_id": t["task_id"]}
    return rows


def _persona(split, tasks=None):
    name = ("roles_subset_paraphrased.csv" if split == "train"
            else "roles_subset_paraphrased_heldout.csv")
    src = cfg.DATA / "role_play" / name
    rows = [{"pair_id": f"persona-{r['role']}",
             "pos": r["pos_instruction"], "neg": r["neg_instruction_padded"],
             "meta": {"role": r["role"], "stratum": r["stratum"],
                      "neg_variant": r["neg_variant"]}}
            for r in _csv(src)]
    return src, _attach(rows, tasks)


def _eval(split, tasks=None):
    name = ("eval_awareness_pairs.csv" if split == "train"
            else "eval_awareness_pairs_heldout.csv")
    src = cfg.DATA / "eval_aware" / name
    rows = [{"pair_id": f"eval-{r['id']}", "pos": r["evaluation"], "neg": r["deployment"],
             "meta": {"id": r["id"], "source": r["source"], "person": r["person"]}}
            for r in _csv(src)]
    return src, _attach(rows, tasks)


def _length(split):
    src = cfg.DATA / "initial_tests" / "length_filler_pairs.jsonl"
    want = "train" if split == "train" else "test"
    return src, [{"pair_id": r["id"], "pos": r["long"], "neg": r["short"],
                  "meta": {"style": r["style"], "request_id": r["request_id"],
                           "domain": r["domain"]}}
                 for r in _jsonl(src) if r["split"] == want]


# ------------------------------------------------- v1 story tables (1.6 / 1.2a)

_V1_MATCHED = cfg.DATA / "story_mode" / "story_mode_prompts_matched.csv"
_V1_MATCHED_HO = cfg.DATA / "story_mode" / "story_mode_prompts_matched_heldout.csv"
_V1_PLAIN = cfg.DATA / "story_mode" / "story_mode_prompts.csv"
_V1_WRAPPERS = cfg.DATA / "story_mode" / "story_wrappers.csv"
_V1_WRAPPERS_HO = cfg.DATA / "story_mode" / "story_wrappers_heldout.csv"


def _wrapper_ids(path=None):
    return [r["id"] for r in _csv(path or _V1_WRAPPERS)]


def _requests_in(path):
    return {r["request"] for r in _csv(path)}


def _request_split():
    """One canonical request list, permuted once, split 50/50.

    First half -> 1.6 extraction, second half -> 1.2a transfer, so the v1
    vector has never seen a 1.2a row.
    """
    shared = sorted(_requests_in(_V1_PLAIN) & _requests_in(_V1_MATCHED))
    rng = random.Random(cfg.SEED)
    perm = shared[:]
    rng.shuffle(perm)
    half = len(perm) // 2
    return perm[:half], perm[half:]


def _story_v1(split):
    """v1 matched, as a first-class direction: 50 train / 15 heldout pairs.

    One row per wrapper, each with a different request, so wrapper is not confounded
    with request. Train and held-out files share no wrapper and no request, so the
    held-out 15 test framing *and* request generalisation - the same shape as v2's
    disjoint context families.

    Train requests come from the first half of `_request_split`, leaving the second
    half reserved for `v1_nofiller100` (spec 1.2a).
    """
    if split == "train":
        src, wrappers = _V1_MATCHED, _wrapper_ids()
        requests, _ = _request_split()
    else:
        src, wrappers = _V1_MATCHED_HO, _wrapper_ids(_V1_WRAPPERS_HO)
        requests = sorted({r["request"] for r in _csv(_V1_MATCHED_HO)})
        random.Random(cfg.SEED).shuffle(requests)
    if len(requests) < len(wrappers):
        raise RuntimeError(f"{split}: need >= {len(wrappers)} requests, have {len(requests)}")

    want = {(w, requests[i]): None for i, w in enumerate(wrappers)}
    found = {}
    for r in _csv(src):
        key = (r["story_id"], r["request"])
        if key in want and key not in found:
            found[key] = r
    missing = set(want) - set(found)
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} wrapper x request cells absent from "
                           f"{Path(src).name}, e.g. {sorted(missing)[:3]}")
    rows = []
    for i, w in enumerate(wrappers):
        r = found[(w, requests[i])]
        rows.append({"pair_id": f"v1-{w}", "pos": r["prompt_story"],
                     "neg": r["prompt_expository"], "neg2": r["prompt_audience"],
                     "meta": {"story_id": w, "request": r["request"], "genre": r["genre"],
                              "jbb_index": r["jbb_index"], "label": r["label"],
                              "n_words_story": r["n_words_story"],
                              "n_words_expository": r["n_words_expository"]}})
    return src, rows


def _v1_nofiller100():
    """Spec 1.2a: 100 pairs, 50 reserved requests x 2 wrappers, preamble added."""
    wrappers = _wrapper_ids()
    _, held = _request_split()
    want = {}
    for i, req in enumerate(held):
        for j in range(2):
            want[(wrappers[(2 * i + j) % len(wrappers)], req)] = None
    found = {}
    for r in _csv(_V1_PLAIN):
        key = (r["story_id"], r["request"])
        if key in want and key not in found:
            found[key] = r
    missing = set(want) - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} wrapper x request cells absent from plain table")
    rows = []
    for (w, req) in sorted(want):
        r = found[(w, req)]
        # This table is built with PREAMBLE = "" (build_main_dataset.py:23); add it
        # to BOTH arms so the prefix matches v2 and the matched table (spec 1.2a).
        rows.append({"pair_id": f"v1n-{w}-{r['jbb_index']}",
                     "pos": pr.continuation(r["prompt_story"]),
                     "neg": pr.continuation(r["prompt_bare"]),
                     "meta": {"story_id": w, "request": req, "jbb_index": r["jbb_index"],
                              "n_words_story": r["n_words_story"],
                              "n_words_bare": r["n_words_bare"]}})
    return _V1_PLAIN, rows


def _v1_curve(n=1000):
    """Spec 1.6 second run: larger matched subsample for the cos(d_n, d_full) curve."""
    rows = [r for r in _csv(_V1_MATCHED)]
    rng = random.Random(cfg.SEED)
    by_wrapper = {}
    for r in rows:
        by_wrapper.setdefault(r["story_id"], []).append(r)
    picked = []
    per = max(1, n // max(1, len(by_wrapper)))
    for w in sorted(by_wrapper):
        pool = by_wrapper[w]
        rng.shuffle(pool)
        picked += pool[:per]
    picked = picked[:n]
    return _V1_MATCHED, [{"pair_id": f"v1c-{r['prompt_id']}", "pos": r["prompt_story"],
                          "neg": r["prompt_expository"],
                          "meta": {"story_id": r["story_id"], "request": r["request"]}}
                         for r in picked]


def _jailbreaks(split):
    src = cfg.DATA / "jailbreaks" / "jailbreaks.csv"
    rows = [r for r in _csv(src) if split == "all" or r["split"] == split]
    return src, [{"pair_id": r["id"], "pos": r["prompt"], "neg": r["request"],
                  "meta": {"family": r["family"], "source": r["source"],
                           "technique": r["technique"], "template_id": r["template_id"],
                           "request": r["request"], "category": r["category"],
                           "base_task_source": r["base_task_source"]}}
                 for r in rows]


# ------------------------------------------------------------------ dispatch

_LOADERS = {
    "story_v2": _story_v2, "story_v1": _story_v1, "harm": _harm, "harm_v2": _harm_v2,
    "persona": _persona, "eval": _eval, "length": _length,
}
_SINGLETONS = {"v1_nofiller100": _v1_nofiller100, "v1_curve": _v1_curve}
FRAMING_ONLY = {"persona", "eval"}


def base_tasks(split="train"):
    """Balanced harmful/benign task pool for spec 0.2(a).

    One task per harm row, alternating pole after a seeded shuffle, so the pool is
    half harmful by construction and no goal repeats. Train tasks come from the harm
    train rows and held-out tasks from the held-out rows, so a task a persona pair
    was fitted with never reappears in its own evaluation set.
    """
    name = "harm_selected_pairs.csv" if split == "train" else "harm_selected_pairs_heldout.csv"
    rows = list(_csv(cfg.DATA / "harm" / name))
    order = list(range(len(rows)))
    random.Random(cfg.SEED).shuffle(order)
    pool = []
    for rank, i in enumerate(order):
        harmful = rank % 2 == 0
        r = rows[i]
        pool.append({"text": r["harmful_goal"] if harmful else r["benign_goal"],
                     "harmful": harmful,
                     "task_id": f"jbb{r['jbb_index']}-{'h' if harmful else 'b'}"})
    return pool


def load_pairs(dataset, split="train", append_task=False):
    if dataset in _SINGLETONS:
        return _SINGLETONS[dataset]()
    if dataset == "jailbreaks":
        return _jailbreaks(split)
    if dataset not in _LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}")
    if dataset in FRAMING_ONLY and append_task:
        return _LOADERS[dataset](split, tasks=base_tasks(split))
    return _LOADERS[dataset](split)


# --------------------------------------------------------------------- views


def build_view(dataset, split, hash_fn, append_task=False, subsample=None, token_info=None):
    """Ordered rows + content-derived view_key. Written before any forward pass.

    `token_info(text) -> dict` is merged into each row. Used to record the final
    token id and length at the read position, so a saturated AUROC can be checked
    against the trivial explanation that the two poles end on different tokens.
    """
    src, pairs = load_pairs(dataset, split, append_task=append_task)
    poles = ["pos", "neg"] + (["neg2"] if pairs and "neg2" in pairs[0] else [])
    rows, texts = [], {}
    for p in pairs:
        for pole in poles:
            sha = hash_fn(p[pole])
            row = {"row_id": p["pair_id"], "pole": pole, "prompt_sha16": sha}
            if token_info is not None:
                row.update(token_info(p[pole]))
            rows.append(row)
            texts[sha] = p[pole]
    view = {"dataset": dataset, "split": split,
            "source_files": [{"path": str(Path(src).relative_to(cfg.REPO)),
                              "sha256": mf.sha256_file(src)}],
            "subsample": subsample, "append_task": bool(append_task),
            "poles": poles, "n_pairs": len(pairs), "rows": rows,
            "meta": {p["pair_id"]: p["meta"] for p in pairs}}
    view["view_key"] = mf.sha256_obj(rows)
    return view, texts


def view_path(layout, dataset, split, view_key=None):
    """Plain name is the current pointer; the keyed name is history.

    Without the keyed copy, changing a dataset overwrites the only record of the
    previous view and every earlier result becomes an unresolvable view_key.
    """
    d = layout.acts / "views"
    return d / (f"{dataset}__{split}__{view_key[:8]}.json" if view_key
                else f"{dataset}__{split}.json")


def write_view(layout, view):
    for path in (view_path(layout, view["dataset"], view["split"], view["view_key"]),
                 view_path(layout, view["dataset"], view["split"])):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(view, indent=2), encoding="utf-8")
        tmp.replace(path)
    return view_path(layout, view["dataset"], view["split"])


def read_view(layout, dataset, split="train"):
    path = view_path(layout, dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"no view for {dataset}/{split}: run cache_activations.py first")
    return json.loads(path.read_text(encoding="utf-8"))
