"""Batched greedy generation with batch-granular resume (spec 0.10 / 0.11).

Batching is the rule, not `batch_size=1`: greedy decoding is bit-reproducible at a
*fixed* batch size and fixed composition, which is what the length sort plus the token
budget give. Resume therefore skips whole batches -- dropping completed rows from a
batch would change its padding and generate the survivors under different conditions.
"""
import torch

from . import hooks as hk
from . import model as mdl


def plan_batches(units, ntok, batch_size, max_batch_tokens):
    """Length-sorted batches under a padded-token budget.

    Planned over the *full* unit set so the plan is a function of the cell's inputs and
    not of what is left to do (spec 0.11).
    """
    order = sorted(units, key=lambda x: ntok.get(x, 0))
    batches, cur = [], []
    for x in order:
        trial = cur + [x]
        longest = max(ntok.get(y, 0) for y in trial)
        if cur and (len(trial) > batch_size or len(trial) * longest > max_batch_tokens):
            batches.append(cur)
            cur = [x]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    return batches


GREEDY = {"do_sample": False}

# Spec 5.1's candidate decodings. One registry, so gen_decoding_compare measures exactly
# what the later scripts can be told to use -- a label that only one script understands is
# how a comparison stops applying to the runs it was meant to inform.
DECODINGS = {
    "greedy": GREEDY,
    "t0.7p0.9": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    "t1.0p0.95": {"do_sample": True, "temperature": 1.0, "top_p": 0.95},
}


def resolve_decode(label, seed=None):
    if label not in DECODINGS:
        raise SystemExit(f"unknown decoding {label!r}; choose from {list(DECODINGS)}")
    kw = dict(DECODINGS[label])
    if kw["do_sample"]:
        if seed is None:
            raise SystemExit(f"decoding {label!r} samples, so it needs an explicit seed: "
                             f"without one the run is not reproducible (spec 0.10)")
        kw["seed"] = seed
    return kw


@torch.no_grad()
def _one_batch(tok, model, texts, max_new_tokens, decode=None, batch_index=0):
    """decode: {do_sample, temperature, top_p, seed} -- GREEDY when None.

    The seed is offset by the batch's index in the plan, so a sampled run is reproducible
    given the same seed and the same batching plan (the plan is a function of the full row
    set, spec 0.11) *without* every batch starting from the same RNG state -- which would
    correlate the draw for row j of every batch.
    """
    decode = dict(decode or GREEDY)
    seed = decode.pop("seed", None)
    if seed is not None:
        torch.manual_seed(seed + batch_index)
    enc = tok([mdl.templated(tok, t) for t in texts], return_tensors="pt",
              padding=True, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    assert enc["attention_mask"][:, -1].all(), "right padding would steer a pad tail"
    n_in = enc["input_ids"].shape[-1]
    out = model.generate(**enc, max_new_tokens=max_new_tokens, **decode,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    eos = tok.eos_token_id
    rows = []
    for seq in out[:, n_in:]:
        ids = seq.tolist()
        stop = ids.index(eos) if eos in ids else len(ids)
        rows.append({"response": tok.decode(ids[:stop], skip_special_tokens=True),
                     "out_tokens": stop, "hit_cap": int(eos not in ids)})
    return rows


@torch.no_grad()
def prefill_states(tok, model, prompts, specs=None, batch_size=8, max_batch_tokens=16384):
    """Final-layer state at the last prompt token, under `specs`. No generation.

    One forward pass per batch, so a whole prompt set costs seconds. This is what makes
    spec 5.6's matched-self-effect arm measurable: scan alpha until the probe readout
    moves as far as the unprojected run moved it, without generating at every candidate.
    """
    ntok = {p: len(mdl.token_ids(tok, p)) for p in dict.fromkeys(prompts)}
    # Keyed by prompt, then expanded back to the caller's order: a repeated prompt is
    # computed once and returned at every position it occupies, rather than leaving a hole.
    got = {}
    cap = hk.FinalCapture()
    with hk.installed(model, specs or [], capture=cap):
        for batch in plan_batches(list(ntok), ntok, batch_size, max_batch_tokens):
            enc = tok([mdl.templated(tok, p) for p in batch], return_tensors="pt",
                      padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            cap.h = None
            model(**enc)
            if cap.h is None:
                raise RuntimeError("final-layer capture did not fire: no prefill pass seen")
            for j, p in enumerate(batch):
                got[p] = cap.h[j]
    return torch.stack([got[p] for p in prompts])


def run(tok, model, rows, sink, done, batch_size, max_batch_tokens, max_new_tokens,
        specs=None, probes=None, progress=None, decode=None):
    """Generate for every row not already in `done`, appending one JSONL line each.

    rows: [{unit_id, prompt, n_tokens, ...}] -- extra keys are copied to the output.
    specs: hook specs from hooks.build, or None for the unsteered pass.
    probes: {name: u_final [d]} for the manipulation check at layer L (spec 5.4).
    decode: sampling config, or None for greedy (spec 5.1).
    """
    ntok = {r["unit_id"]: r.get("n_tokens") or 0 for r in rows}
    by_id = {r["unit_id"]: r for r in rows}
    batches = plan_batches(list(by_id), ntok, batch_size, max_batch_tokens)
    # Carry each batch's index in the *full* plan, so a resumed run seeds its batches
    # exactly as an uninterrupted one would.
    todo = [(i, b) for i, b in enumerate(batches) if any(u not in done for u in b)]
    counter = {"calls": 0}
    capture = hk.FinalCapture() if probes else None
    n_done = 0

    with hk.installed(model, specs or [], capture=capture):
        for bi, batch in todo:
            counter["calls"] = 0
            if capture is not None:
                capture.h = None            # never attribute a prior batch's states
            got = _one_batch(tok, model, [by_id[u]["prompt"] for u in batch],
                             max_new_tokens, decode, batch_index=bi)
            readouts = [{} for _ in batch]
            if capture is not None and capture.h is not None:
                readouts = [{f"read_{k}": float(capture.h[i] @ v) for k, v in probes.items()}
                            for i in range(len(batch))]
            for u, g, rd in zip(batch, got, readouts):
                sink.write_row({**{k: v for k, v in by_id[u].items() if k != "prompt"},
                                **g, **rd, "hook_calls": counter["calls"],
                                "n_in_batch": len(batch)})
            n_done += len(batch)
            if progress:
                progress(n_done, sum(len(b) for _, b in todo))
    return {"n_batches": len(batches), "n_batches_run": len(todo), "n_rows_run": n_done,
            "hook_calls_last": counter["calls"]}


class Sink:
    """Append-only JSONL, flushed and fsynced per row (spec 0.11)."""

    def __init__(self, fh):
        self.fh = fh
        self.n = 0

    def write_row(self, row):
        import json
        import os
        self.fh.write(json.dumps(row, default=str) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.n += 1
