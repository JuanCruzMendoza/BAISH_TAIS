"""Paths, seed, layer band.

Layout:
    experiments/<exp>/results/<tag>/<model_slug>/
        csv/       what you read: metrics tables + *_selection.json
        vectors/   *.pt consumed by later experiments (gitignored)
        acts/      blobs/ + views/ activation cache (gitignored)
        meta/      manifests, runs.csv, _archive/, *_deciles.json
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

SEED = 20260731
DEFAULT_TAG = "base"

BAND_LO, BAND_HI = 0.40, 0.90

# The steering window, matched to the Assistant Axis paper's depth fraction: it caps
# Qwen3-32B at layers 46-53 of 64 (0.72-0.83) and Llama-3.3-70B at 56-71 of 80
# (0.70-0.89). 0.70-0.90 spans both. This is the widest *joint* config (spec 5.4.0);
# the reporting band above stays 0.40-0.90 and remains the ceiling on any layer spec.
STEER_LO, STEER_HI = 0.70, 0.90


def band(L):
    """Reporting band (spec 0.3). Every layer is still swept."""
    return list(range(round(BAND_LO * L), round(BAND_HI * L) + 1))


def steer_band(L):
    """Widest joint steering config (spec 5.4.0), at the paper's depth fraction."""
    return list(range(round(STEER_LO * L), round(STEER_HI * L) + 1))


LAYER_SPEC = ("22 | 18-25 | 18,22,25 | frac:0.70-0.90 | steer_band | band"
              "   (no 'all'; the reporting band is the ceiling)")


def parse_layers(spec, L):
    """Layer-spec grammar (spec 5.4.0) -> sorted layers. Out-of-band specs are rejected.

    Clipping would record the set the caller asked for while steering another one.
    """
    s = str(spec).strip()
    b = band(L)
    if s == "all":
        raise ValueError(f"'all' is not a layer spec: band is the ceiling ({b[0]}-{b[-1]}), "
                         f"spec 5.4.0")
    if s == "steer_band":
        out = steer_band(L)
    elif s == "band":
        out = b
    elif s.startswith("frac:"):
        lo, hi = s[5:].split("-")
        out = list(range(round(float(lo) * L), round(float(hi) * L) + 1))
    elif "," in s:
        out = [int(x) for x in s.split(",")]
    elif "-" in s:
        lo, hi = s.split("-")
        out = list(range(int(lo), int(hi) + 1))
    else:
        out = [int(s)]
    out = sorted(dict.fromkeys(out))
    outside = [l for l in out if not b[0] <= l <= b[-1]]
    if not out or outside:
        raise ValueError(f"layer spec {spec!r} -> {out or '[]'}; {outside} outside the band "
                         f"{b[0]}-{b[-1]} (spec 5.4.0 rejects rather than clips)")
    return out


def layer_stem(spec):
    """The unresolved spec, as it appears in a stem (spec 0.1)."""
    s = str(spec).strip()
    if s in ("band", "steer_band"):
        return s
    if s.startswith("frac:"):
        return "f" + s[5:]
    return "L" + s.replace(",", "_")


def model_slug(model_id):
    return model_id.replace("/", "_")


def tag(explicit=None):
    return explicit or os.environ.get("RUN_TAG") or DEFAULT_TAG


def run_dir(experiment, model_id, tag_=None):
    return REPO / "experiments" / experiment / "results" / tag(tag_) / model_slug(model_id)


class Layout:
    """The four subdirectories of one <tag>/<model> run."""

    def __init__(self, experiment, model_id, tag_=None, acts_cache=True):
        self.root = run_dir(experiment, model_id, tag_)
        self.csv = self.root / "csv"
        self.vectors = self.root / "vectors"
        self.meta = self.root / "meta"
        self.acts = self.root / "acts"
        for d in (self.csv, self.vectors, self.meta):
            d.mkdir(parents=True, exist_ok=True)
        # Experiments 2+ only read the cache, which lives under extraction; an empty
        # acts/ under their own results dir would suggest otherwise.
        if acts_cache:
            (self.acts / "views").mkdir(parents=True, exist_ok=True)
            self.blobs.mkdir(parents=True, exist_ok=True)

    @property
    def blobs(self):
        """BLOB_STORE lets several tags share one content-addressed store."""
        shared = os.environ.get("BLOB_STORE")
        return Path(shared) if shared else self.acts / "blobs"

    def __repr__(self):
        return f"Layout({self.root.relative_to(REPO).as_posix()})"


def acts_layout(model_id, tag_=None):
    """The cache always lives under extraction, whichever experiment reads it."""
    return Layout("extraction", model_id, tag_)
