"""Activation blob cache: content-addressed per prompt, atomic writes (spec 0.8 / 0.11)."""
import json
import os

import numpy as np


def blob_path(layout, sha):
    return layout.blobs / f"{sha}.npy"


def has(layout, sha):
    return blob_path(layout, sha).exists()


def write(layout, sha, arr):
    """Atomic: a kill mid-write must not leave a truncated but present blob."""
    path = blob_path(layout, sha)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.ascontiguousarray(arr, dtype=np.float16), allow_pickle=False)
    os.replace(tmp, path)


def read(layout, sha):
    return np.load(blob_path(layout, sha), allow_pickle=False)


def missing(layout, shas):
    return [s for s in dict.fromkeys(shas) if not has(layout, s)]


def acts_manifest_path(layout):
    return layout.acts / "acts_manifest.json"


def write_acts_manifest(layout, payload):
    path = acts_manifest_path(layout)
    prior = {}
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        for k in ("chat_template_sha", "n_layers", "d_model", "position", "dtype"):
            if k in prior and prior[k] != payload.get(k):
                raise RuntimeError(
                    f"acts cache {k} changed ({prior[k]!r} -> {payload.get(k)!r}); "
                    f"existing blobs are invalid: delete acts/blobs/ and re-cache")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({**prior, **payload}, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_view_matrix(layout, view):
    """-> {pole: [n_pairs, L+1, d] float32}, rows in view order.

    A single-pole view is allowed (spec 3 reads jailbreak prompts with no contrast
    arm); the pos/neg checks below are contrast checks and only apply when both exist.
    """
    order = {}
    for r in view["rows"]:
        order.setdefault(r["pole"], []).append(r)
    ids = [r["row_id"] for r in order["pos"]]
    out = {}
    for pole, rows in order.items():
        # Paired metrics are only meaningful if row i of every pole is the same pair.
        got = [r["row_id"] for r in rows]
        if got != ids:
            raise RuntimeError(f"pole {pole!r} is misaligned with 'pos' in view "
                               f"{view['dataset']}/{view['split']}")
        if len({r["prompt_sha16"] for r in rows}) == 1 and len(rows) > 1:
            raise RuntimeError(f"pole {pole!r} is a single repeated prompt")
        out[pole] = np.stack([read(layout, r["prompt_sha16"]) for r in rows]).astype("float32")
    if "neg" in order:
        dup = sum(a == b for a, b in zip([r["prompt_sha16"] for r in order["pos"]],
                                         [r["prompt_sha16"] for r in order["neg"]]))
        if dup:
            raise RuntimeError(f"{dup} pairs have identical pos and neg prompts")
    out["pair_ids"] = ids
    return out
