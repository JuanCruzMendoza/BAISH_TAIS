"""One steering cell, shared by steer_single.py (5.4) and steer_induce.py (5.5).

A cell is direction x mode x layer set x alpha/tau x arm on one prompt set, and it is
the unit of the stem, the run_key and resume (spec 0.1 / 0.10 / 0.11). Arms are separate
cells rather than extra rows in one file, so the no-op is run once per layer set instead
of once per direction and a completed arm cache-hits.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common import (acts, config as cfg, generate as gen, hooks as hk,
                                manifest as mf, views)
from experiments.steering_jailbreaks import sets

# Spec 5.4a: one mode per (set, direction).
PRIMARY = {
    "success": {"story_v2": "ablate", "story_v1": "ablate", "persona": "ablate",
                "harm": "add", "eval": "add"},
    "refusal": {"story_v2": "add", "story_v1": "add", "persona": "add",
                "harm": "ablate", "eval": "ablate"},
}
# cap occupies exactly one slot: story/persona on successes, ceiling only.
CAP_SLOT = ("story_v2", "story_v1", "persona")
CLAMP = {"story_v2": "ceil", "story_v1": "ceil", "persona": "ceil",
         "harm": "floor", "eval": "floor"}
AXES = ["story_v2", "story_v1", "persona", "harm", "eval"]


def load_probe(src, axis):
    stem = mf.stem("directions", axis)
    path = src.vectors / f"{stem}.pt"
    if not path.exists():
        raise SystemExit(f"no vector for {axis}: run extract_direction.py --direction {axis}")
    mf.load_upstream(src.meta / f"{stem}_manifest.json")
    return torch.load(path, weights_only=False)


def final_layer_probes(src, axes):
    """{axis: u at layer L} for the manipulation check (spec 5.4)."""
    out = {}
    for a in axes:
        p = src.vectors / f"{mf.stem('directions', a)}.pt"
        if p.exists():
            u = torch.load(p, weights_only=False)["u"]
            out[a] = u[u.shape[0] - 1].float()
    return out


def tau_percentile(src, u, layers, q):
    """tau[l] = q-th percentile of raw <h, u_hat> over the two-pole corpus (spec 0.6).

    Framed prompts *plus* their bare requests: the percentile has to sit between two
    modes, which a framed-only corpus does not have.
    """
    view = views.read_view(src, "jailbreaks", "all")
    if "neg" not in view.get("poles", []):
        raise SystemExit(
            "cap needs tau, and tau needs the two-pole corpus (spec 0.6). Re-cache with\n"
            "  python experiments/extraction/cache_activations.py <model> "
            "--dataset jailbreaks --split all --tag <tag>\n"
            "(i.e. without --poles pos), then re-run this cell.")
    m = acts.load_view_matrix(src, view)
    h = np.concatenate([m["pos"], m["neg"]])
    return {l: float(np.percentile(h[:, l, :] @ u[l], q)) for l in layers}, len(h)


def random_unit(u_shape, seed):
    """Matched-norm random direction, one per layer (spec 5.4 specificity arm)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(u_shape)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def stem_for(script, direction, mode, layer_spec, alpha, tau_q, arm):
    parts = [] if arm == "noop" else [direction]
    parts += [mode if arm != "noop" else "noop", cfg.layer_stem(layer_spec)]
    if mode == "add" and arm != "noop":
        parts.append(f"a{alpha:g}")
    if mode == "cap" and arm != "noop":
        parts.append(f"q{tau_q:g}")
    if arm == "random":
        parts.append(f"random{cfg.SEED}")
    return mf.stem(script, *parts)


def run(script, model_id, tag, rows, prompt_set, direction, mode, layer_spec, arm,
        alpha, tau_q, n_layers, tok, model, batch_size, max_batch_tokens, max_new_tokens):
    """Run one cell and return its manifest run_key."""
    src = cfg.acts_layout(model_id, tag)
    lay = cfg.Layout(sets.EXPERIMENT, model_id, tag, acts_cache=False)
    layers = cfg.parse_layers(layer_spec, n_layers)

    probe = None if arm == "noop" else load_probe(src, direction)
    sigma = probe["sigma_act"].numpy() if probe is not None else None
    if arm == "random":
        u = random_unit(probe["u"].shape, cfg.SEED)
    elif probe is not None:
        u = probe["u"].numpy()
    else:
        u = None

    tau, n_ref = (None, None)
    if mode == "cap" and arm != "noop":
        tau, n_ref = tau_percentile(src, u, layers, tau_q)

    counter = {"calls": 0}
    ut = None if u is None else torch.from_numpy(np.ascontiguousarray(u)).float()
    specs = hk.build("noop" if arm == "noop" else mode, layers, ut, counter,
                     alpha=alpha, sigma=sigma, tau=tau,
                     clamp=CLAMP.get(direction, "ceil"))

    stem = stem_for(script, direction, mode, layer_spec, alpha, tau_q, arm)
    config = {"direction": None if arm == "noop" else direction,
              "mode": mode, "arm": arm, "prompt_set": prompt_set,
              "layers_spec": str(layer_spec), "layers": layers, "n_layers_steered": len(layers),
              "alpha": alpha if mode == "add" else None,
              "per_layer_coef": hk.per_layer_coef(mode, layers, alpha),
              "tau_q": tau_q if mode == "cap" else None,
              "tau": tau, "tau_n_ref": n_ref,
              "clamp": CLAMP.get(direction) if mode == "cap" else None,
              "decoding": "greedy", "max_new_tokens": max_new_tokens,
              "batch_size": batch_size, "max_batch_tokens": max_batch_tokens,
              "position": "all_tokens", "n_rows": len(rows), "seed": cfg.SEED}
    inputs = {"unit_ids": [r["unit_id"] for r in rows],
              "direction_run_key": None if probe is None else probe.get("run_key")}

    probes = final_layer_probes(src, AXES)
    with mf.Run(lay, stem, config, inputs, resumable=True) as run_:
        done = run_.resume_from(".jsonl")
        with run_.open_append(".jsonl") as fh:
            info = gen.run(tok, model, rows, gen.Sink(fh), done, batch_size,
                           max_batch_tokens, max_new_tokens, specs=specs, probes=probes,
                           progress=lambda i, n: print(f"    {i}/{n}", end="\r"))
        if info["n_rows_run"] and info["hook_calls_last"] == 0:
            raise RuntimeError(f"{stem}: hooks never fired -- the intervention did nothing")
        print(f"\n  {stem}: {info['n_rows_run']} generated, {len(done)} cached, "
              f"N={len(layers)} layers {layers[0]}-{layers[-1]}")
        return run_.run_key
