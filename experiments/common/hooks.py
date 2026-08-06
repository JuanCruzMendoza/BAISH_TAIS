"""Steering hooks: ablate / add / cap, every token position (spec 5.4).

hidden_states[l] is the output of decoder block l-1, so the hook for layer l goes on
blocks[l-1]. Layer 0 is the embedding output and is not steerable; the band starts far
above it anyway.
"""
import contextlib
import math

import torch

MODES = ("ablate", "add", "cap", "noop")


def decoder_blocks(model):
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise RuntimeError("cannot locate decoder blocks on this model")


def _hs(output):
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output, hs):
    return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


def _ablate(u, counter):
    def hook(module, args, output):
        hs = _hs(output)
        v = u.to(hs.dtype).to(hs.device)
        counter["calls"] += 1
        return _rewrap(output, hs - torch.einsum("...d,d->...", hs, v).unsqueeze(-1) * v)
    return hook


def _add(u, coef, counter):
    def hook(module, args, output):
        hs = _hs(output)
        v = u.to(hs.dtype).to(hs.device)
        counter["calls"] += 1
        return _rewrap(output, hs + coef * v)
    return hook


def _cap(u, tau, clamp, counter):
    def hook(module, args, output):
        hs = _hs(output)
        v = u.to(hs.dtype).to(hs.device)
        excess = torch.einsum("...d,d->...", hs, v) - tau
        excess = excess.clamp(min=0) if clamp == "ceil" else excess.clamp(max=0)
        counter["calls"] += 1
        return _rewrap(output, hs - excess.unsqueeze(-1) * v)
    return hook


def _noop(counter):
    def hook(module, args, output):
        counter["calls"] += 1
        return output
    return hook


def build(mode, layers, u, counter, alpha=None, sigma=None, tau=None, clamp="ceil"):
    """-> [(layer, hook_fn)] for one cell. u/sigma/tau are indexed by layer.

    `add` uses alpha / sqrt(N) with N the joint width (spec 5.4b), so the per-layer
    coefficient is comparable across layer counts and the pilot's single-layer alpha
    calibration transfers.
    """
    n = len(layers)
    specs = []
    for l in layers:
        if mode == "ablate":
            specs.append((l, _ablate(u[l], counter)))
        elif mode == "add":
            specs.append((l, _add(u[l], alpha / math.sqrt(n) * float(sigma[l]), counter)))
        elif mode == "cap":
            specs.append((l, _cap(u[l], float(tau[l]), clamp, counter)))
        elif mode == "noop":
            specs.append((l, _noop(counter)))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return specs


def per_layer_coef(mode, layers, alpha):
    """The number that makes cells at different N readable side by side (spec 5.4b)."""
    return None if mode != "add" else alpha / math.sqrt(len(layers))


class FinalCapture:
    """Final-layer state at the last prompt position, prefill pass only.

    The manipulation check (spec 5.4): read the probes at layer L, where an additive
    push is not tautological. Left padding puts the last real token at -1.
    """

    def __init__(self):
        self.h = None

    def hook(self, module, args, output):
        hs = _hs(output)
        if hs.shape[1] > 1:
            self.h = hs[:, -1, :].detach().float().cpu()


@contextlib.contextmanager
def installed(model, specs, capture=None):
    blocks = decoder_blocks(model)
    handles = []
    try:
        if capture is not None:
            handles.append(blocks[-1].register_forward_hook(capture.hook))
        for l, fn in specs:
            if l < 1:
                raise ValueError(f"layer {l} is the embedding output, not steerable")
            handles.append(blocks[l - 1].register_forward_hook(fn))
        yield
    finally:
        for h in handles:
            h.remove()
