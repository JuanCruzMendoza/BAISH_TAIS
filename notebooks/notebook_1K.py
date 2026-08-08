import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import pathlib, subprocess, sys
    import marimo as mo

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    TAG = "1K_per_direction"
    HF_REPO = "JuanCruzMendoza/BAISH_TAIS"

    # This file lives in notebooks/, so the repo root is above it when opened from a
    # checkout; standalone in molab there is no checkout and we clone beside the notebook.
    NB = mo.notebook_dir()
    ROOTS = [d for d in [NB, *NB.parents]
             if (d / "experiments" / "common" / "ckpt.py").exists()]
    REPO = str(ROOTS[0]) if ROOTS else str(NB / "BAISH_TAIS")

    def sh(*a, cwd=None, allow_fail=False):
        p = subprocess.run(a, cwd=cwd or REPO, capture_output=True, text=True)
        print(p.stdout[-4000:], p.stderr[-2000:])
        if p.returncode and not allow_fail:
            raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, a))}\n"
                               f"{(p.stderr or p.stdout)[-1500:]}")
        return p

    if not pathlib.Path(REPO, "experiments").exists():
        sh("git", "clone", "https://github.com/JuanCruzMendoza/BAISH_TAIS.git", REPO,
           cwd=str(NB))
    else:
        sh("git", "fetch", "origin")
        # Never `reset --hard`: after a restart the tree holds results pulled back from
        # the Hub, and a rebase onto a dirty tree is refused rather than destructive.
        if subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip():
            print("! worktree dirty (restored results?) -- fetched, not rebased")
        else:
            sh("git", "rebase", "origin/main")
    sh("git", "log", "--oneline", "-1")
    sh("pip", "install", "-q", "transformers", "accelerate", "numpy", "matplotlib",
       "huggingface_hub", "hf_xet")
    sh("python", "-c",
       "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))")

    sys.path.insert(0, REPO)
    return HF_REPO, MODEL, REPO, TAG, mo, sh


@app.cell(hide_code=True)
def _(TAG, mo):
    mo.md(f"""
    # {TAG} — extraction

    Spec: `experiments/extraction/dev.md`, *1K_per_direction*. Four directions at 800
    train / 200 held-out, `--curve` for the saturation measurement, **manual** layer
    choice. No length foil, no `--append-task` (baked into the v2 pairs), no
    `compare_crossed` and no v1 transfer — both are v1 arms.

    ## Checkpoint

    Paste an HF **write** token. The results go to a private dataset repo on an hourly
    timer, so a molab shutdown costs at most the last hour. Restarting this notebook
    pulls them back and every finished unit cache-hits.
    """)
    return


@app.cell
def _(mo):
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    HF_TOKEN
    return (HF_TOKEN,)


@app.cell
def _(HF_REPO, HF_TOKEN, mo, sh):
    from experiments.common import ckpt

    mo.stop(not HF_TOKEN.value, mo.md("*Paste the HF token to start.*"))
    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    restored = ckpt.pull(HF_REPO, token=HF_TOKEN.value)
    sh("git", "status", "--short", "experiments")
    TIMER = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value)
    print("restored from Hub:", restored, "| hourly checkpoint armed")
    return TIMER, ckpt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cache the activations — GPU

    ~6,400 train + 1,600 held-out prompts. The only cell that touches the GPU;
    everything below reads the blob cache. All 8 cells run in **one process** — the model
    load dominated the wall time at 8 invocations — and checkpoint in **one commit**, since
    every push is a commit and the Hub's quota is a 5-minute window. A rate-limited
    checkpoint warns and moves on rather than killing the cell; re-running resumes per
    prompt from the blobs already on disk.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, TAG, TIMER, ckpt, sh):
    DIRECTIONS = ["story_v2_1k", "persona_v2", "eval_v2", "harm_v2"]

    # One process for all 8 cells: the model load dominates, and 8 invocations paid it
    # 8 times. Resume is per prompt (content-addressed blobs), so a crash still costs
    # only the prompts not yet written, not the whole list.
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", ",".join(DIRECTIONS), "--split", "train,heldout")
    ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="acts")
    cached = TIMER is not None
    return DIRECTIONS, cached


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Directions and the per-layer table — CPU

    `--curve` writes `csv/directions__<axis>_curve.json`: `cos(d_n, d_800)` for
    n ∈ {50, …, 750} at 5 seeds, plus the disjoint `cos(d_800_train, d_200_heldout)`.
    `probe_select` emits the table and stops — no band rule, no primary layer.
    """)
    return


@app.cell
def _(DIRECTIONS, HF_REPO, HF_TOKEN, MODEL, TAG, cached, ckpt, sh):
    assert cached
    for _d in DIRECTIONS:
        sh("python", "experiments/extraction/extract_direction.py", MODEL,
           "--tag", TAG, "--direction", _d, "--curve")
        sh("python", "experiments/extraction/probe_select.py", MODEL,
           "--tag", TAG, "--direction", _d)
    ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="directions + probe_select")
    extracted = True
    return (extracted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figures

    8 total, two per direction: the cos curve and `cohens_dz_train`.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, TAG, ckpt, extracted, sh):
    assert extracted
    sh("python", "experiments/extraction/plot_figures.py", MODEL, "--tag", TAG,
       "--with-heldout")
    ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="figures")
    figured = True
    return (figured,)


@app.cell
def _(MODEL, TAG, TIMER, figured, sh):
    assert figured
    # After the push, and never fatal: check_stale exits 1 on any finding, which would
    # otherwise abort the cell over results that are already on the Hub.
    _stale = sh("python", "-m", "experiments.common.check_stale", MODEL, TAG,
                allow_fail=True).returncode
    TIMER.set()          # nothing writes results/ from here, so stop the hourly push
    print(("! check_stale reported findings above" if _stale else "all artefacts current")
          + " | hourly checkpoint stopped")
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    mo.md(f"""
    ## Choose the layers

    Read `cohens_dz_train` off `csv/probe_select__<axis>.csv`, confirm against
    `cohens_dz_heldout`, `mean_paired_cos` and `lopo_ci_lo`. The chosen layers go in
    `insights.md` — experiments 2–5 read them from there, not from any JSON.

    Results are on the Hub under `extraction/results/{TAG}/<model_slug>/`; pull them
    locally with `snapshot_download` to write up.

    **The checkpoint excludes `vectors/*.pt`** — `lopo_d` is ~335 MB per direction at
    n=800, half the payload, and rebuilds from the blob cache in seconds. So a pulled
    snapshot has the tables and figures but no vectors, and the first thing that reads a
    direction fails rather than warns. Rebuild them after any pull:

    ```bash
    for d in story_v2_1k persona_v2 eval_v2 harm_v2; do
      python experiments/extraction/extract_direction.py $MODEL --tag {TAG} --direction $d --curve
    done
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
