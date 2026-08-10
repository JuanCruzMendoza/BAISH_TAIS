"""Paths, seed, layer band.

Layout:
    experiments/<exp>/results/<tag>/<model_slug>/
        csv/       what you read: metrics tables + *_summary.json / *_selection.json
        figures/   *.png plotted from csv/
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


def load_env(path=None):
    """Read REPO/.env into os.environ. Gitignored; holds API keys.

    A variable already set in the environment always wins, so a shell export beats a
    stale .env rather than being silently replaced by it. Returns the names loaded --
    never the values, which must not reach a log or a manifest.
    """
    p = Path(path) if path else REPO / ".env"
    if not p.exists():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


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


def parse_axis_layers(spec):
    """'story_v2_1k=23,persona_v2=15' -> {axis: layer}: one chosen layer per direction.

    Manual, not derived: at 1K_per_direction the layers are read off `cohens_dz_train`
    and live in extraction/insights.md, so they enter as an explicit argument.
    """
    out = {}
    for part in str(spec).split(","):
        if not part.strip():
            continue
        axis, sep, layer = part.partition("=")
        if not sep:
            raise ValueError(f"--layers: expected axis=layer, got {part!r}")
        out[axis.strip()] = int(layer)
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
    """The subdirectories of one <tag>/<model> run."""

    def __init__(self, experiment, model_id, tag_=None, acts_cache=True):
        self.root = run_dir(experiment, model_id, tag_)
        self.csv = self.root / "csv"
        self.figures = self.root / "figures"
        self.vectors = self.root / "vectors"
        self.meta = self.root / "meta"
        self.acts = self.root / "acts"
        for d in (self.csv, self.figures, self.vectors, self.meta):
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
