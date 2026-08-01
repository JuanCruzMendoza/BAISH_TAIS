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


def band(L):
    """Reporting band (spec 0.3). Every layer is still swept."""
    return list(range(round(BAND_LO * L), round(BAND_HI * L) + 1))


def model_slug(model_id):
    return model_id.replace("/", "_")


def tag(explicit=None):
    return explicit or os.environ.get("RUN_TAG") or DEFAULT_TAG


def run_dir(experiment, model_id, tag_=None):
    return REPO / "experiments" / experiment / "results" / tag(tag_) / model_slug(model_id)


class Layout:
    """The four subdirectories of one <tag>/<model> run."""

    def __init__(self, experiment, model_id, tag_=None):
        self.root = run_dir(experiment, model_id, tag_)
        self.csv = self.root / "csv"
        self.vectors = self.root / "vectors"
        self.meta = self.root / "meta"
        self.acts = self.root / "acts"
        for d in (self.csv, self.vectors, self.meta, self.acts):
            d.mkdir(parents=True, exist_ok=True)
        (self.acts / "views").mkdir(exist_ok=True)
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
