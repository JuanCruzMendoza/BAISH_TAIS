"""Paths, seed, layer band."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

SEED = 20260731

BAND_LO, BAND_HI = 0.40, 0.90


def band(L):
    """Reporting band (spec 0.3). Every layer is still swept."""
    return list(range(round(BAND_LO * L), round(BAND_HI * L) + 1))


def model_slug(model_id):
    return model_id.replace("/", "_")


def results_dir(experiment, model_id):
    d = REPO / "experiments" / experiment / "results" / model_slug(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def acts_dir(model_id):
    """One cache per model, shared by every experiment."""
    d = REPO / "experiments" / "extraction" / "results" / model_slug(model_id) / "acts"
    (d / "blobs").mkdir(parents=True, exist_ok=True)
    (d / "views").mkdir(parents=True, exist_ok=True)
    return d
