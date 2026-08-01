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
from pathlib import Path

from . import config as cfg
from . import manifest as mf
from . import prompts as pr

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

DIRECTIONS = ["story", "harm", "persona", "eval", "length"]

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


def _story(split):
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


def _persona(split, tasks=None):
    name = ("roles_subset_paraphrased.csv" if split == "train"
            else "roles_subset_paraphrased_heldout.csv")
    src = cfg.DATA / "role_play" / name
    rows = []
    for i, r in enumerate(_csv(src)):
        pos, neg = r["pos_instruction"], r["neg_instruction_padded"]
        if tasks:
            t = tasks[i % len(tasks)]
            pos, neg = pr.with_task(pos, t), pr.with_task(neg, t)
        rows.append({"pair_id": f"persona-{r['role']}", "pos": pos, "neg": neg,
                     "meta": {"role": r["role"], "stratum": r["stratum"],
                              "neg_variant": r["neg_variant"]}})
    return src, rows


def _eval(split, tasks=None):
    name = ("eval_awareness_pairs.csv" if split == "train"
            else "eval_awareness_pairs_heldout.csv")
    src = cfg.DATA / "eval_aware" / name
    rows = []
    for i, r in enumerate(_csv(src)):
        pos, neg = r["evaluation"], r["deployment"]
        if tasks:
            t = tasks[i % len(tasks)]
            pos, neg = pr.with_task(pos, t), pr.with_task(neg, t)
        rows.append({"pair_id": f"eval-{r['id']}", "pos": pos, "neg": neg,
                     "meta": {"id": r["id"], "source": r["source"], "person": r["person"]}})
    return src, rows


def _length(split):
    src = cfg.DATA / "initial_tests" / "length_filler_pairs.jsonl"
    want = "train" if split == "train" else "test"
    return src, [{"pair_id": r["id"], "pos": r["long"], "neg": r["short"],
                  "meta": {"style": r["style"], "request_id": r["request_id"],
                           "domain": r["domain"]}}
                 for r in _jsonl(src) if r["split"] == want]


# ------------------------------------------------- v1 story tables (1.6 / 1.2a)

_V1_MATCHED = cfg.DATA / "story_mode" / "story_mode_prompts_matched.csv"
_V1_PLAIN = cfg.DATA / "story_mode" / "story_mode_prompts.csv"
_V1_WRAPPERS = cfg.DATA / "story_mode" / "story_wrappers.csv"


def _wrapper_ids():
    return [r["id"] for r in _csv(_V1_WRAPPERS)]


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


def _v1_fair50():
    """Spec 1.6: 50 wrappers x 1 distinct request each, matched table."""
    wrappers = _wrapper_ids()
    fair, _ = _request_split()
    if len(fair) < len(wrappers):
        raise RuntimeError(f"need >= {len(wrappers)} shared requests, have {len(fair)}")
    want = {(w, fair[i]) for i, w in enumerate(wrappers)}
    found = {}
    for r in _csv(_V1_MATCHED):
        key = (r["story_id"], r["request"])
        if key in want and key not in found:
            found[key] = r
    missing = want - set(found)
    if missing:
        raise RuntimeError(f"{len(missing)} wrapper x request cells absent from matched table, "
                           f"e.g. {sorted(missing)[:3]}")
    rows = []
    for i, w in enumerate(wrappers):
        r = found[(w, fair[i])]
        rows.append({"pair_id": f"v1f-{w}", "pos": r["prompt_story"],
                     "neg": r["prompt_expository"], "neg2": r["prompt_audience"],
                     "meta": {"story_id": w, "request": r["request"],
                              "genre": r["genre"], "jbb_index": r["jbb_index"]}})
    return _V1_MATCHED, rows


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
    "story": _story, "harm": _harm, "persona": _persona, "eval": _eval, "length": _length,
}
_SINGLETONS = {
    "v1_fair50": _v1_fair50, "v1_nofiller100": _v1_nofiller100, "v1_curve": _v1_curve,
}
FRAMING_ONLY = {"persona", "eval"}


def base_tasks():
    """The 50 harm goals, for spec 0.2(a) task appending."""
    _, rows = _harm("train")
    return [r["pos"] for r in rows]


def load_pairs(dataset, split="train", append_task=False):
    if dataset in _SINGLETONS:
        return _SINGLETONS[dataset]()
    if dataset == "jailbreaks":
        return _jailbreaks(split)
    if dataset not in _LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}")
    if dataset in FRAMING_ONLY and append_task:
        return _LOADERS[dataset](split, tasks=base_tasks())
    return _LOADERS[dataset](split)


# --------------------------------------------------------------------- views


def build_view(dataset, split, hash_fn, append_task=False, subsample=None):
    """Ordered rows + content-derived view_key. Written before any forward pass."""
    src, pairs = load_pairs(dataset, split, append_task=append_task)
    poles = ["pos", "neg"] + (["neg2"] if pairs and "neg2" in pairs[0] else [])
    rows, texts = [], {}
    for p in pairs:
        for pole in poles:
            sha = hash_fn(p[pole])
            rows.append({"row_id": p["pair_id"], "pole": pole, "prompt_sha16": sha})
            texts[sha] = p[pole]
    view = {"dataset": dataset, "split": split,
            "source_files": [{"path": str(Path(src).relative_to(cfg.REPO)),
                              "sha256": mf.sha256_file(src)}],
            "subsample": subsample, "append_task": bool(append_task),
            "poles": poles, "n_pairs": len(pairs), "rows": rows,
            "meta": {p["pair_id"]: p["meta"] for p in pairs}}
    view["view_key"] = mf.sha256_obj(rows)
    return view, texts


def view_path(model_id, dataset, split):
    return cfg.acts_dir(model_id) / "views" / f"{dataset}__{split}.json"


def write_view(model_id, view):
    path = view_path(model_id, view["dataset"], view["split"])
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(view, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def read_view(model_id, dataset, split="train"):
    path = view_path(model_id, dataset, split)
    if not path.exists():
        raise FileNotFoundError(f"no view for {dataset}/{split}: run cache_activations.py first")
    return json.loads(path.read_text(encoding="utf-8"))
