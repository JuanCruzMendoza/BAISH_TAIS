"""Activation blob cache: content-addressed per prompt, atomic writes (spec 0.8 / 0.11)."""
import json
import os

import numpy as np

from . import config as cfg


def blob_path(model_id, sha):
    return cfg.acts_dir(model_id) / "blobs" / f"{sha}.npy"


def has(model_id, sha):
    return blob_path(model_id, sha).exists()


def write(model_id, sha, arr):
    """Atomic: a kill mid-write must not leave a truncated but present blob."""
    path = blob_path(model_id, sha)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, np.ascontiguousarray(arr, dtype=np.float16), allow_pickle=False)
    os.replace(tmp, path)


def read(model_id, sha):
    return np.load(blob_path(model_id, sha), allow_pickle=False)


def missing(model_id, shas):
    return [s for s in dict.fromkeys(shas) if not has(model_id, s)]


def acts_manifest_path(model_id):
    return cfg.acts_dir(model_id) / "acts_manifest.json"


def write_acts_manifest(model_id, payload):
    path = acts_manifest_path(model_id)
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


def load_view_matrix(model_id, view):
    """-> {pole: [n_pairs, L+1, d] float32}, rows in view order."""
    order = {}
    for r in view["rows"]:
        order.setdefault(r["pole"], []).append(r)
    out = {}
    for pole, rows in order.items():
        stack = [read(model_id, r["prompt_sha16"]) for r in rows]
        out[pole] = np.stack(stack).astype("float32")
    out["pair_ids"] = [r["row_id"] for r in order["pos"]]
    return out
