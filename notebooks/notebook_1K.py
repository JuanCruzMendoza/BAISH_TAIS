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
    timer, so a shutdown costs at most the last hour. Restarting pulls them back and every
    finished unit cache-hits.

    **Leave `re-run extraction` off** unless you mean to recompute it. Extraction is done
    and on the Hub; off, the four cells below skip their work and the pull narrows to the
    three directories the probes need (`acts/`, `vectors/`, `meta/directions__*`).
    On, they re-run *and re-push* — and a push costs commits per file *considered*, 256 per
    commit, so re-uploading ~7,700 identical blobs three times is ~90 commits for nothing.
    """)
    return


@app.cell
def _(mo):
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    # Off = extraction is done and on the Hub: pull what the probes need and skip its
    # four cells. Leaving them on costs ~90 commits re-uploading identical files, because
    # a push costs commits per file *considered* (256 per commit), not per file changed.
    RUN_EXTRACTION = mo.ui.checkbox(
        label="re-run extraction (off: pull its results from the Hub)")
    mo.vstack([HF_TOKEN, RUN_EXTRACTION])
    return HF_TOKEN, RUN_EXTRACTION


@app.cell
def _(HF_REPO, HF_TOKEN, RUN_EXTRACTION, TAG, mo, sh):
    from experiments.common import ckpt

    mo.stop(not HF_TOKEN.value, mo.md("*Paste the HF token to start.*"))
    # SCOPE is not cosmetic: upload_folder emits one commit per 256 files considered, so
    # an unscoped push also walks every other experiment's results (spec: ckpt.scope).
    SCOPE = {"experiment": "extraction", "tag": TAG}
    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    # Downstream, extraction is three directories. `acts/` because the thresholds are
    # calibrated on the pole *activations*, not just on the vectors -- jb_readout reloads
    # all 8,000 pole prompts, so the blobs are not optional. `vectors/` for the probes and
    # `meta/directions__*` for the manifest they are validated against. Skipped: csv/,
    # figures/, probe_select's tables and ~30 MB of meta/*.jsonl resume partials.
    _need = None if RUN_EXTRACTION.value else ["*/acts/**", "*/vectors/**",
                                              "*/meta/directions__*"]
    restored = ckpt.pull(HF_REPO, token=HF_TOKEN.value, subpaths=_need, **SCOPE)
    # NEW EXTRACTION RUN (another model / tag): add pack=True to the pull above and
    # uncomment the matching push in the cache cell. pack=True stores acts/blobs as a
    # single tar -- 2 commits instead of ~30 -- and unpacks it here so the scripts see the
    # same tree. Both sides or neither. This tag's blobs are loose on the Hub already, and
    # pack=True is a no-op when no tar is present, so it is safe to leave enabled.
    sh("git", "status", "--short", "experiments")
    # Only arm the timer when something will write extraction results. With the box off
    # nothing here does, and the jailbreak blobs are deliberately not checkpointed.
    TIMER = (ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **SCOPE)
             if RUN_EXTRACTION.value else None)
    print("restored from Hub:", restored,
          "| extraction:", "re-running, hourly checkpoint armed" if TIMER
          else "pulled, cells skipped")
    return SCOPE, TIMER, ckpt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cache the activations — GPU

    Skipped unless `re-run extraction` is on, as are the next three cells.

    ~6,400 train + 1,600 held-out prompts. The first cell that touches the GPU; everything
    below it reads the blob cache. All 8 cells run in **one process** — the model load
    dominated the wall time at 8 invocations — and checkpoint once at the end.

    That push is still ~30 commits: `upload_folder` emits one per 256 files, and 7,731
    blobs is 7,731 files. Commits are the rationed resource, so the push is scoped to this
    experiment and tag. A rate-limited checkpoint warns and moves on rather than killing
    the cell; re-running resumes per prompt from the blobs already on disk.

    **For a new model or tag**, uncomment the `pack=True` lines in this cell *and* the
    checkpoint cell: the blob cache goes up as one tar, ~2 commits instead of ~30. Not for
    steering — its ~500 files already cost ~3 commits, and its resume partials have to stay
    individually addressable. Trade-off: a packed push re-sends the whole tar whenever any
    blob changes, so pack a finished cache, not a resuming one.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_EXTRACTION, SCOPE, TAG, ckpt, sh):
    DIRECTIONS = ["story_v2_1k", "persona_v2", "eval_v2", "harm_v2"]

    if RUN_EXTRACTION.value:
        # One process for all 8 cells: the model load dominates, and 8 invocations paid it
        # 8 times. Resume is per prompt (content-addressed blobs), so a crash still costs
        # only the prompts not yet written, not the whole list.
        sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
           "--dataset", ",".join(DIRECTIONS), "--split", "train,heldout")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="acts", **SCOPE)
        # NEW EXTRACTION RUN: use this instead, together with pack=True on the pull above.
        # ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="acts", pack=True, **SCOPE)
    cached = True                # the blobs are either just written or just pulled
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
def _(DIRECTIONS, HF_REPO, HF_TOKEN, MODEL, RUN_EXTRACTION, SCOPE, TAG, cached, ckpt, sh):
    assert cached
    if RUN_EXTRACTION.value:
        for _d in DIRECTIONS:
            sh("python", "experiments/extraction/extract_direction.py", MODEL,
               "--tag", TAG, "--direction", _d, "--curve")
            sh("python", "experiments/extraction/probe_select.py", MODEL,
               "--tag", TAG, "--direction", _d)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value,
                      msg="directions + probe_select", **SCOPE)
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
def _(HF_REPO, HF_TOKEN, MODEL, RUN_EXTRACTION, SCOPE, TAG, ckpt, extracted, sh):
    assert extracted
    if RUN_EXTRACTION.value:
        sh("python", "experiments/extraction/plot_figures.py", MODEL, "--tag", TAG,
           "--with-heldout")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="figures", **SCOPE)
    figured = True
    return (figured,)


@app.cell
def _(MODEL, RUN_EXTRACTION, TAG, TIMER, figured, sh):
    assert figured
    if not RUN_EXTRACTION.value:
        # check_stale spans the whole tag, so running it here would only report the
        # jailbreak artefacts the cells below have not written yet.
        print("extraction pulled from the Hub -- nothing to check or stop")
    else:
        # After the push, and never fatal: check_stale exits 1 on any finding, which would
        # otherwise abort the cell over results that are already on the Hub.
        _stale = sh("python", "-m", "experiments.common.check_stale", MODEL, TAG,
                    allow_fail=True).returncode
        TIMER.set()      # nothing writes results/ from here, so stop the hourly push
        print(("! check_stale reported findings" if _stale else "all artefacts current")
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

    The checkpoint **includes `vectors/*.pt`** (~1.3 GB, one commit), so experiments 2–5
    get the directions straight from a scoped pull instead of downloading 1.6 GB of blobs
    and re-running `extract_direction` first.
    """)
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    mo.md(f"""
    # {TAG} — probe_jailbreak_detection (H2)

    Spec: `experiments/probe_jailbreak_detection/dev.md`, *{TAG}*. The four chosen-layer
    probes against **all 1,009** jailbreak prompts, at two thresholds.

    Reads extraction's `vectors/directions__<axis>.pt` and writes the jailbreak
    activations into its blob cache, so it needs extraction pulled — not re-run.

    ## Checkpoint

    Its own scope, and a **separate** one from extraction's — a push scoped to
    `extraction/` would walk all ~7,700 blobs and cost ~30 commits before writing anything.
    This experiment's results are ~6 small files, so each push here is one commit.

    The 1,009 new blobs are therefore **not** checkpointed. They are only an input to
    `jb_readout.py`; once its `.pt` is on the Hub they are disposable, and the price is
    that a kill *during* the GPU cell costs that pass rather than resuming from it.

    **A jailbreak-only session** is: the setup cell, the token cell with the box off, the
    extraction checkpoint cell, then the four below — **2 commits** total, and the only GPU
    work is the 1,009-prompt pass.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, TAG, ckpt, mo):
    mo.stop(not HF_TOKEN.value, mo.md("*Paste the HF token to start.*"))
    JB_SCOPE = {"experiment": "probe_jailbreak_detection", "tag": TAG}
    JB_AXES = "story_v2_1k,persona_v2,harm_v2,eval_v2"
    # extraction/insights.md, section 1K: max cohens_dz_train, min train<->heldout gap.
    JB_LAYERS = "story_v2_1k=23,persona_v2=15,harm_v2=21,eval_v2=9"
    print("restored from Hub:",
          ckpt.pull(HF_REPO, token=HF_TOKEN.value, **JB_SCOPE))
    return JB_AXES, JB_LAYERS, JB_SCOPE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cache the jailbreak activations — GPU

    1,009 prompts (1,017 minus the 8 whose `prompt` is its `request`), framed arm only
    (`--poles pos`), no subsample. Written into extraction's blob cache, so `--tag` has to
    match the extraction run.

    `--max-batch-tokens` matters here more than at 100 rows: the corpus holds a
    47,308-char prompt and spans 57 chars to that, so length-sorted batching is what keeps
    the padded peak near the budget instead of near `batch_size x longest`.
    """)
    return


@app.cell
def _(MODEL, TAG, extracted, sh):
    assert extracted                      # the probes come from the extraction cells
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", "jailbreaks", "--split", "all", "--poles", "pos")
    jb_cached = True
    return (jb_cached,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Readout and metrics — CPU

    `jb_readout.py` projects each probe onto all 1,009 prompts at every layer and stores
    the two reference pole distributions beside them (1,000 points per pole here, pooled
    train + held-out). Everything downstream reads that one `.pt`, ~470 KB.

    `jb_metrics.py` runs twice, once per threshold rule. `midpoint` bisects the pole
    *means* and `gap_mid` the empty gap between p95(neg) and p5(pos); the stems differ, so
    the two never share a file. `--layers` makes the headline `_chosen.csv` — one row per
    probe x slice at that probe's own layer — and adds `eval_v2`'s L9 to the scored set,
    since it sits outside the band.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, JB_AXES, JB_SCOPE, MODEL, TAG, ckpt, jb_cached, sh):
    assert jb_cached
    sh("python", "experiments/probe_jailbreak_detection/jb_readout.py", MODEL,
       "--tag", TAG, "--axes", JB_AXES)
    # The one push that matters: after this the blobs above are disposable.
    ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_readout", **JB_SCOPE)
    jb_read = True
    return (jb_read,)


@app.cell
def _(HF_REPO, HF_TOKEN, JB_AXES, JB_LAYERS, JB_SCOPE, MODEL, TAG, ckpt, jb_read, sh):
    assert jb_read
    for _rule in ("midpoint", "gap_mid"):
        sh("python", "experiments/probe_jailbreak_detection/jb_metrics.py", MODEL,
           "--tag", TAG, "--axes", JB_AXES, "--layers", JB_LAYERS, "--threshold", _rule)
    ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_metrics", **JB_SCOPE)
    jb_done = True
    return (jb_done,)


@app.cell
def _(MODEL, TAG, jb_done, sh):
    assert jb_done
    sh("python", "-m", "experiments.common.check_stale", MODEL, TAG, allow_fail=True)
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    mo.md(f"""
    ## Read the results

    `probe_jailbreak_detection/results/{TAG}/<model_slug>/csv/`:
    `jb_metrics__midpoint_chosen.csv` and `jb_metrics__gap_mid_chosen.csv` are the
    headline, `_rate.csv` the per-layer context around it.

    Read `ref_tpr` before `pct_reads`: near 1.0 the bar is passable and a low `pct_reads`
    is a real finding, low `ref_tpr` means tau is too strict to conclude anything. There is
    **no `length` foil at this tag**, so the 50-pair check — is a high `pct_reads` just
    prompt length? — cannot be run here.
    """)
    return


if __name__ == "__main__":
    app.run()
