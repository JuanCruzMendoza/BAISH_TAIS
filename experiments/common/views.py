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
from collections import Counter
from pathlib import Path  # noqa: F401  (used by the v1 loaders' error messages)

from . import config as cfg
from . import manifest as mf
from . import prompts as pr

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

DIRECTIONS = ["story_v2", "story_v1", "harm", "harm_v2", "persona", "eval", "length",
              # 1K_per_direction: 800 train / 200 held-out each. story_v2_1k is a separate
              # name rather than a bigger story_v2, so the 50-pair tag stays reproducible.
              "story_v2_1k", "persona_v2", "eval_v2"]

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


# ------------------------------------------------- 1K_per_direction (800 / 200)
# The framing axes carry their task inside the dataset, so `--append-task` is refused
# rather than ignored (TASK_BAKED below).


def _story_v2_1k(split):
    name = "pairs_1k.jsonl" if split == "train" else "pairs_1k_heldout.jsonl"
    src = cfg.DATA / "story_mode_v2" / name
    return src, [{"pair_id": r["pair_id"],
                  "pos": pr.continuation(r["text_narrative"]),
                  "neg": pr.continuation(r["text_nonnarrative"]),
                  "meta": {k: r[k] for k in ("narr_mode", "nonnarr_style", "realism",
                                             "tense_polarity", "context", "domain", "genre")}}
                 for r in _jsonl(src)]


def _persona_v2(split):
    name = "pairs.jsonl" if split == "train" else "pairs_heldout.jsonl"
    src = cfg.DATA / "role_play_v2" / name
    return src, [{"pair_id": r["pair_id"],
                  "pos": pr.with_task(r["role_framing"], r["task"]),
                  "neg": pr.with_task(r["assistant_framing"], r["task"]),
                  "meta": {"role": r["role"], "stratum": r["stratum"],
                           "role_form": r["role_form"], "neg_variant": r["neg_variant"],
                           "neg_form": r["neg_form"], "jbb_index": r["jbb_index"],
                           "task_id": r["task_id"], "category": r["category"],
                           "task_harmful": r["label"] == "harmful"}}
                 for r in _jsonl(src)]


def _eval_v2(split):
    name = "pairs.jsonl" if split == "train" else "pairs_heldout.jsonl"
    src = cfg.DATA / "eval_v2" / name
    return src, [{"pair_id": r["pair_id"],
                  "pos": pr.with_task(r["framing_evaluation"], r["request"]),
                  "neg": pr.with_task(r["framing_deployment"], r["request"]),
                  "meta": {"framing_id": r["framing_id"], "framing_source": r["framing_source"],
                           "framing_person": r["framing_person"], "jbb_index": r["jbb_index"],
                           "task_id": r["task_id"], "category": r["category"],
                           "task_harmful": r["label"] == "harmful"}}
                 for r in _jsonl(src)]


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


# Spec 3: the 100-row subset. Family allocation lifts nonfiction_other from its
# proportional 8 to 15, because it is the contrast class of 3.2's family test.
JB_FAMILY_ALLOC = {"fiction_narrative": 35, "roleplay_persona": 30,
                   "hybrid": 20, "nonfiction_other": 15}
JB_MAX_PER_TEMPLATE = 2


def _largest_remainder(weights, total, floor=0):
    """Integer allocation summing to `total`, each key >= min(floor, its weight)."""
    keys = sorted(weights)
    base = {k: min(floor, weights[k]) for k in keys}
    left = total - sum(base.values())
    wsum = sum(max(weights[k] - base[k], 0) for k in keys) or 1
    exact = {k: left * max(weights[k] - base[k], 0) / wsum for k in keys}
    out = {k: base[k] + int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: -(exact[k] % 1)):
        if sum(out.values()) >= total:
            break
        out[k] += 1
    return {k: min(v, weights[k]) for k, v in out.items()}


def _take_wrappers(by_tpl, want, cap, rng, seen_req, used_tpl):
    """Round-robin over shuffled wrappers, preferring rows with an unseen request.

    `used_tpl` is global: a template_id that spans two families is still one cluster,
    so the cap has to be counted across families, not inside each.
    """
    order = sorted(by_tpl)
    rng.shuffle(order)
    pool = {t: list(by_tpl[t]) for t in order}
    picked = []
    for _ in range(cap):
        for t in order:
            if len(picked) >= want or not pool[t] or used_tpl[t] >= cap:
                continue
            fresh = [r for r in pool[t] if r["request"] not in seen_req]
            pick = rng.choice(fresh or pool[t])
            pool[t] = [r for r in pool[t] if r is not pick]
            seen_req.add(pick["request"])
            used_tpl[t] += 1
            picked.append(pick)
    return picked


def _jb_template_diverse(rows, n, seed, alloc=None, cap=JB_MAX_PER_TEMPLATE):
    """Wrapper-diverse subsample (spec 0.7 clustering).

    template_id is very concentrated -- 400 of the corpus's 425 wrappers are
    in_the_wild singletons while jailbreak_mimicry's 300 rows share 2 -- so a
    row-proportional sample of 100 would collapse to ~20 clusters. Three rules:

    - family allocation lifts nonfiction_other above its proportional share, since it
      is the contrast class of 3.2's family test;
    - inside a family, sources are allocated proportional to their **distinct wrapper
      count** with a floor, so in_the_wild's 86 fiction wrappers cannot crowd out
      strongreject's 2 -- that pair is 3.2's only clean within-source family cell;
    - inside a source, round-robin over wrappers at most `cap` rows each, preferring
      rows whose request has not been used.

    Request coverage is bounded by the corpus, not by n: in_the_wild's 400 rows sit on
    ~32 JBB behaviours, so any wrapper-rich sample is request-poor (spec 0.7).
    """
    alloc = alloc or JB_FAMILY_ALLOC
    rng = random.Random(seed)
    seen_req, picked = set(), []
    used_tpl = Counter()
    fam_alloc = _largest_remainder({f: sum(1 for r in rows if r["family"] == f)
                                    for f in alloc if any(r["family"] == f for r in rows)},
                                   n)
    fam_alloc = {f: min(alloc[f], fam_alloc.get(f, 0)) if f in alloc else 0 for f in alloc}
    short = n - sum(fam_alloc.values())
    for f in sorted(fam_alloc, key=lambda f: -alloc[f]):        # spend any rounding slack
        room = min(alloc[f], sum(1 for r in rows if r["family"] == f)) - fam_alloc[f]
        take = max(0, min(short, room))
        fam_alloc[f] += take
        short -= take

    for fam in sorted(fam_alloc):
        fam_rows = [r for r in rows if r["family"] == fam]
        by_src = {}
        for r in fam_rows:
            by_src.setdefault(r["source"], {}).setdefault(r["template_id"], []).append(r)
        wrappers = {s: len(t) for s, t in by_src.items()}
        # Headroom, not raw availability: a wrapper already used by another family
        # cannot supply `cap` more rows.
        room_of = lambda t: {s: sum(min(len(v), cap - used_tpl[k])
                                    for k, v in t[s].items()) for s in t}
        rooms = room_of(by_src)
        src_alloc = _largest_remainder(wrappers, fam_alloc[fam], floor=2)
        for s in sorted(src_alloc):
            src_alloc[s] = min(src_alloc[s], rooms[s])
        got = sum(src_alloc.values())
        for s in sorted(src_alloc, key=lambda s: -wrappers[s]):  # give slack to the widest
            take = max(0, min(fam_alloc[fam] - got, rooms[s] - src_alloc[s]))
            src_alloc[s] += take
            got += take
        for s in sorted(src_alloc):
            picked += _take_wrappers(by_src[s], src_alloc[s], cap, rng, seen_req, used_tpl)
    return sorted(picked, key=lambda r: r["id"])


JB_FILTER = "prompt != request"


def _jailbreaks(split, subsample=None):
    """Spec 3.1's contrast is framed `prompt` vs bare `request`, so a row whose prompt
    *is* its request carries no framing and is dropped: 8 rows, all
    technique=bare_request, and all nonfiction_other -- i.e. all in 3.2's contrast
    class, where a forced zero delta would bias the family test toward fiction.
    """
    src = cfg.DATA / "jailbreaks" / "jailbreaks.csv"
    rows = [r for r in _csv(src) if split == "all" or r["split"] == split]
    rows = [r for r in rows if r["prompt"].strip() != r["request"].strip()]
    if subsample:
        if subsample.get("strategy", "template_diverse") != "template_diverse":
            raise ValueError(f"unknown strategy {subsample['strategy']!r}")
        rows = _jb_template_diverse(rows, subsample["n"],
                                    subsample.get("seed", cfg.SEED),
                                    cap=subsample.get("max_per_template",
                                                      JB_MAX_PER_TEMPLATE))
    return src, [{"pair_id": r["id"], "pos": r["prompt"], "neg": r["request"],
                  "meta": {"family": r["family"], "source": r["source"],
                           "technique": r["technique"], "template_id": r["template_id"],
                           "request": r["request"], "category": r["category"],
                           "base_task_source": r["base_task_source"],
                           "split": r["split"], "n_chars": r["n_chars"]}}
                 for r in rows]


# ------------------------------------------------------------------ dispatch

_LOADERS = {
    "story_v2": _story_v2, "story_v1": _story_v1, "harm": _harm, "harm_v2": _harm_v2,
    "persona": _persona, "eval": _eval, "length": _length,
    "story_v2_1k": _story_v2_1k, "persona_v2": _persona_v2, "eval_v2": _eval_v2,
}
_SINGLETONS = {"v1_nofiller100": _v1_nofiller100, "v1_curve": _v1_curve}
FRAMING_ONLY = {"persona", "eval"}
TASK_BAKED = {"persona_v2", "eval_v2"}     # task is in the pairs file, not appended here


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


def load_pairs(dataset, split="train", append_task=False, subsample=None):
    if dataset in _SINGLETONS:
        return _SINGLETONS[dataset]()
    if dataset == "jailbreaks":
        return _jailbreaks(split, subsample)
    if dataset not in _LOADERS:
        raise KeyError(f"unknown dataset {dataset!r}")
    if dataset in TASK_BAKED and append_task:
        raise ValueError(f"{dataset}: the task is already in the pairs file; "
                         f"--append-task would append a second one")
    if dataset in FRAMING_ONLY and append_task:
        return _LOADERS[dataset](split, tasks=base_tasks(split))
    return _LOADERS[dataset](split)


# --------------------------------------------------------------------- views


def build_view(dataset, split, hash_fn, append_task=False, subsample=None, token_info=None,
               poles=None):
    """Ordered rows + content-derived view_key. Written before any forward pass.

    `token_info(text) -> dict` is merged into each row. Used to record the final
    token id and length at the read position, so a saturated AUROC can be checked
    against the trivial explanation that the two poles end on different tokens.

    `subsample` both drives the sampling and is recorded, so a changed sampler moves
    the view_key even when {n, seed} are unchanged (spec 0.8).

    `poles` restricts which arms are cached; default is every arm the loader supplies.
    Passing ["pos"] makes a single-arm view -- no contrast, so downstream can only read
    absolute levels, but it also caches nothing it will not use.
    """
    src, pairs = load_pairs(dataset, split, append_task=append_task, subsample=subsample)
    avail = [p for p in ("pos", "neg", "neg2") if pairs and p in pairs[0]]
    poles = [p for p in (poles or avail) if p in avail]
    if not poles:
        raise ValueError(f"{dataset}/{split}: no requested pole exists, have {avail}")
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
