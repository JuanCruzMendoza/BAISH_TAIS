import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    # cell 1
    import os, pathlib, subprocess, sys
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
        # A restart pulls results back from the Hub, and meta/runs.csv plus the manifests
        # churn on every re-run -- so tracked files under experiments/ differ, the rebase
        # refuses, and the clone silently strands on an old commit while the notebook's own
        # cells (molab keeps its own copy of them) look current. Discarding those is free:
        # the Hub is authoritative for them and cell 3 re-pulls. Still not `reset --hard`,
        # and `checkout --` leaves untracked and ignored files -- acts/, vectors/, the hf
        # download cache -- untouched.
        sh("git", "checkout", "--", "experiments")
        _dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
        if _dirty:
            print(f"! tracked files still modified, not rebased:\n{_dirty[:500]}")
        else:
            sh("git", "rebase", "origin/main", allow_fail=True)
    sh("git", "log", "--oneline", "-1")
    sh("pip", "install", "-q", "transformers", "accelerate", "numpy", "matplotlib",
       "huggingface_hub", "hf_xet")
    sh("python", "-c",
       "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))")

    sys.path.insert(0, REPO)
    return HF_REPO, MODEL, REPO, TAG, mo, os, sh


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 2
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
    # cell 3
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    # Off = extraction is done and on the Hub: pull what the probes need and skip its
    # four cells. Leaving them on costs ~90 commits re-uploading identical files, because
    # a push costs commits per file *considered* (256 per commit), not per file changed.
    RUN_EXTRACTION = mo.ui.checkbox(
        label="re-run extraction (off: pull its results from the Hub)")
    mo.vstack([HF_TOKEN, RUN_EXTRACTION])
    return HF_TOKEN, RUN_EXTRACTION


@app.cell
def _(HF_REPO, HF_TOKEN, RUN_EXTRACTION, TAG, mo, os, sh):
    # cell 4
    from experiments.common import ckpt

    mo.stop(not HF_TOKEN.value, mo.md("*Paste the HF token to start.*"))
    if not RUN_EXTRACTION.value:
        # Xet buys chunk-level dedup on *upload*, which a pull-only session never uses,
        # and its read-token endpoint is what returned 429 when the blobs were 7,731
        # separate requests. Not set when extraction re-runs -- its pushes want the dedup.
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    # SCOPE is not cosmetic: upload_folder emits one commit per 256 files considered, so
    # an unscoped push also walks every other experiment's results (spec: ckpt.scope).
    SCOPE = {"experiment": "extraction", "tag": TAG}
    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    # Downstream, extraction is three directories. `acts/` because the thresholds are
    # calibrated on the pole *activations*, not just on the vectors -- jb_readout reloads
    # all 8,000 pole prompts, so the blobs are not optional. `vectors/` for the probes and
    # `meta/directions__*` for the manifest they are validated against. Skipped: csv/,
    # figures/ and probe_select's tables.
    _need = None if RUN_EXTRACTION.value else ["*/acts/**", "*/vectors/**",
                                              "*/meta/directions__*"]
    # pack=True on BOTH sides or neither. The blob cache is stored on the Hub as a single
    # acts/blobs.tar: 7,731 loose .npy was one read request each and hit the quota, and one
    # commit per 256 files made every push ~30. Unpacked here, so the scripts see the same
    # tree either way and run_key resume still cache-hits.
    restored = ckpt.pull(HF_REPO, token=HF_TOKEN.value, subpaths=_need, pack=True, **SCOPE)
    sh("git", "status", "--short", "experiments")
    # Only arm the timer when something will write extraction results. With the box off
    # nothing here does, and the jailbreak blobs are deliberately not checkpointed.
    TIMER = (ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, pack=True, **SCOPE)
             if RUN_EXTRACTION.value else None)
    print("restored from Hub:", restored,
          "| extraction:", "re-running, hourly checkpoint armed" if TIMER
          else "pulled, cells skipped")
    return SCOPE, TIMER, ckpt


@app.cell(hide_code=True)
def _(mo):
    # cell 5
    mo.md(r"""
    ## Cache the activations — GPU

    Skipped unless `re-run extraction` is on, as are the next three cells.

    ~6,400 train + 1,600 held-out prompts. The first cell that touches the GPU; everything
    below it reads the blob cache. All 8 cells run in **one process** — the model load
    dominated the wall time at 8 invocations — and checkpoint once at the end.

    That push is `pack=True`, so the cache goes up as a single `acts/blobs.tar`: **2
    commits**, against ~30 when the 7,731 blobs went loose (`upload_folder` emits one per
    256 files) — and a loose cache also cost one *read* request per blob on the way back
    down, which is what returned 429. A rate-limited checkpoint warns and moves on rather
    than killing the cell; re-running resumes per prompt from the blobs on disk.

    Trade-off: a packed push re-sends the whole 1.6 GB tar whenever any blob changes, so
    the hourly timer re-tars on every tick. Pack a finished cache, not a resuming one — and
    not steering, whose ~500 files already cost ~3 commits and whose resume partials have to
    stay individually addressable.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_EXTRACTION, SCOPE, TAG, ckpt, sh):
    # cell 6
    DIRECTIONS = ["story_v2_1k", "persona_v2", "eval_v2", "harm_v2"]

    if RUN_EXTRACTION.value:
        # One process for all 8 cells: the model load dominates, and 8 invocations paid it
        # 8 times. Resume is per prompt (content-addressed blobs), so a crash still costs
        # only the prompts not yet written, not the whole list.
        sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
           "--dataset", ",".join(DIRECTIONS), "--split", "train,heldout")
        # pack=True to match the pull: pushing the blobs loose here would leave both
        # representations on the Hub, and they would drift apart on the next run.
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="acts", pack=True, **SCOPE)
    cached = True                # the blobs are either just written or just pulled
    return DIRECTIONS, cached


@app.cell(hide_code=True)
def _(mo):
    # cell 7
    mo.md(r"""
    ## Directions and the per-layer table — CPU

    `--curve` writes `csv/directions__<axis>_curve.json`: `cos(d_n, d_800)` for
    n ∈ {50, …, 750} at 5 seeds, plus the disjoint `cos(d_800_train, d_200_heldout)`.
    `probe_select` emits the table and stops — no band rule, no primary layer.
    """)
    return


@app.cell
def _(DIRECTIONS, HF_REPO, HF_TOKEN, MODEL, RUN_EXTRACTION, SCOPE, TAG, cached, ckpt, sh):
    # cell 8
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
    # cell 9
    mo.md(r"""
    ## Figures

    8 total, two per direction: the cos curve and `cohens_dz_train`.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_EXTRACTION, SCOPE, TAG, ckpt, extracted, sh):
    # cell 10
    assert extracted
    if RUN_EXTRACTION.value:
        sh("python", "experiments/extraction/plot_figures.py", MODEL, "--tag", TAG,
           "--with-heldout")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="figures", **SCOPE)
    figured = True
    return (figured,)


@app.cell
def _(MODEL, RUN_EXTRACTION, TAG, TIMER, figured, sh):
    # cell 11
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
    # cell 12
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
    # cell 13
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

    **Leave `re-run probe_jailbreak_detection` off** once its results are on the Hub. Off,
    the activation cell runs `--view-only`: it writes `acts/views/jailbreaks__all.json`
    from the tokenizer alone — seconds, on CPU, no weights — and the readout and metrics
    cells skip. That view is the one thing **steering** needs from this section, and it is
    the one thing not on the Hub, because it is written into *extraction's* tree while the
    pushes here are scoped to `probe_jailbreak_detection/`. Rebuilding it is cheaper than
    storing it: pushing it would re-tar extraction's 1.6 GB cache and drag the 1,009
    jailbreak blobs in with it, +206 MB on every future extraction pull, for activations
    steering never reads. Verified: the CPU path reproduces the GPU path's `view_key` and
    every row exactly, so no downstream `run_key` moves.
    """)
    return


@app.cell
def _(mo):
    # Off = this experiment is done and on the Hub: build the view on CPU and skip the
    # readout and metrics cells. On = the full 1,009-prompt GPU pass.
    # cell 14
    RUN_JB = mo.ui.checkbox(
        label="re-run probe_jailbreak_detection (off: view-only, CPU, then skip)")
    RUN_JB
    return (RUN_JB,)


@app.cell
def _(HF_REPO, HF_TOKEN, TAG, ckpt, mo):
    # cell 15
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
    # cell 16
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
def _(MODEL, RUN_JB, TAG, extracted, sh):
    # cell 17
    assert extracted                      # the probes come from the extraction cells
    # Off: write the view and stop. It is what steering reads, it costs seconds on CPU, and
    # skipping the weights is what lets a judge-only session run without a GPU at all.
    _vo = [] if RUN_JB.value else ["--view-only"]
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", "jailbreaks", "--split", "all", "--poles", "pos", *_vo)
    jb_cached = True
    return (jb_cached,)


@app.cell(hide_code=True)
def _(mo):
    # cell 18
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
def _(HF_REPO, HF_TOKEN, JB_AXES, JB_SCOPE, MODEL, RUN_JB, TAG, ckpt, jb_cached, sh):
    # cell 19
    assert jb_cached
    if RUN_JB.value:                      # needs the blobs, which --view-only does not write
        sh("python", "experiments/probe_jailbreak_detection/jb_readout.py", MODEL,
           "--tag", TAG, "--axes", JB_AXES)
        # The one push that matters: after this the blobs above are disposable.
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_readout", **JB_SCOPE)
    jb_read = True
    return (jb_read,)


@app.cell
def _(HF_REPO, HF_TOKEN, JB_AXES, JB_LAYERS, JB_SCOPE, MODEL, RUN_JB, TAG, ckpt, jb_read,
      sh):
    # cell 20
    assert jb_read
    if RUN_JB.value:
        for _rule in ("midpoint", "gap_mid"):
            sh("python", "experiments/probe_jailbreak_detection/jb_metrics.py", MODEL,
               "--tag", TAG, "--axes", JB_AXES, "--layers", JB_LAYERS,
               "--threshold", _rule)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_metrics", **JB_SCOPE)
    jb_done = True
    return (jb_done,)


@app.cell
def _(MODEL, RUN_JB, TAG, jb_done, sh):
    # cell 21
    assert jb_done
    if RUN_JB.value:
        sh("python", "-m", "experiments.common.check_stale", MODEL, TAG, allow_fail=True)
    else:
        # --view-only leaves the jailbreak blobs uncomputed on purpose, which check_stale
        # would report as a finding against results that are already on the Hub.
        print("probe_jailbreak_detection pulled from the Hub -- nothing to check")
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 22
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


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 23
    mo.md(f"""
    # Steering jailbreak - 1

    Spec: `experiments/steering_jailbreaks/dev.md`, *{TAG} / 1_run*. **36 cells** — 28
    target + 8 no-op — one chosen layer per direction, no joint config, no `cap`, no §5.6
    pairs. α is **signed**: `harm_v2`/`eval_v2` are added on the successes and subtracted on
    the refusals, `story_v2_1k`/`persona_v2` the other way round, and each of the four is
    also suppressed by `ablate` on the set where suppressing it is the hypothesis.

    Measured at 1_run: **17,947 generations in 2.81 h GPU**, plus ≈1.5 h of judging over
    16,938 calls. The two sets came out at **508 successes / 433 refusals** — ASR is 53.1%
    on the full corpus against 30% on the 50-row subset, so the *successes* are the larger
    half and the two `steer_batch` calls cost about the same, ~4.5 min per cell.

    Depends on the jailbreak cell above for `acts/views/jailbreaks__all.json` — `sets.py`
    rebuilds the prompts from that view and checks them against its `prompt_sha16`. The view
    lives in *extraction's* tree and was never pushed there (the run above is scoped to
    `probe_jailbreak_detection/`), so it is re-made locally rather than pulled.

    ## Checkpoint

    Its own scope again, and `pack=False` — unlike extraction. Steering writes a few hundred
    small files whose resume partials have to stay individually addressable, and a packed
    push re-sends the whole tar on every tick of a multi-hour run. ~152 files per push, so
    1 commit each: 13 for the whole 1_run generation pass.

    The hourly timer is **one per scope**: re-running the cell that arms it replaces its
    timer rather than adding one. Before that it leaked — every pasted token or toggled
    checkbox left another thread running, and 1_run ended up with three.

    **Leave `run steering` off** unless you mean to spend the hours: marimo runs every cell
    on load, so an unguarded cell here would start the sweep in a session opened for
    something else.
    """)
    return


@app.cell
def _(mo):
    # cell 24
    OPENAI_KEY = mo.ui.text(label="OPENAI_API_KEY (judge, spec 5.3)", kind="password",
                            full_width=True)
    RUN_STEERING = mo.ui.checkbox(label="run steering (36 cells, ~4 h GPU + ~1.5 h judging)")
    mo.vstack([OPENAI_KEY, RUN_STEERING])
    return OPENAI_KEY, RUN_STEERING


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, OPENAI_KEY, REPO, RUN_STEERING, TAG, ckpt, mo, os):
    # cell 25
    import pathlib as _pl

    mo.stop(not HF_TOKEN.value, mo.md("*Paste the HF token to start.*"))
    ST_SCOPE = {"experiment": "steering_jailbreaks", "tag": TAG}
    # extraction/insights.md, section 1K. eval_v2's L9 is outside the reporting band
    # (11-25), so its cells carry --allow-out-of-band and record it in their manifest.
    ST_CHOSEN = {"story_v2_1k": 23, "persona_v2": 15, "harm_v2": 21, "eval_v2": 9}
    ST_META = _pl.Path(REPO, "experiments/steering_jailbreaks/results", TAG,
                       MODEL.replace("/", "_"), "meta")
    # judge_strongreject reads it from the environment of the subprocess, which inherits
    # this one. Nothing else in the notebook needs an API key.
    if OPENAI_KEY.value:
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY.value
    elif RUN_STEERING.value:
        print("! no OPENAI_API_KEY: generation will run, judging will not")
    print("restored from Hub:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))
    # The timer is what makes the checkpoint mid-run: steer_batch is one process for 18
    # cells, so a per-script push never fires inside it.
    ST_TIMER = (ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
                if RUN_STEERING.value else None)
    return ST_CHOSEN, ST_META, ST_SCOPE, ST_TIMER


@app.cell(hide_code=True)
def _(mo):
    # cell 26
    mo.md(r"""
    ## Baseline, and the split it defines — GPU

    Unsteered greedy over all 1,009 prompts. Judging it is what defines the two sets:
    §5.4 runs on the rows it complied with, §5.5 on the rows it refused, and a degenerate
    row falls in neither.

    The sizes are not known until this is judged. Measured at 1_run: **508 / 433**, from
    51.2% complied / 44.0% refused / 4.8% degenerate, with one row the judge declined to
    grade. Both are the *baseline's* split, so some success rows will not comply at steer
    time — batch composition differs (33 batches here, 17 and 15 there) and greedy is
    bit-reproducible only at fixed composition. The no-op is the denominator, never this.

    `hit_cap_rate` was **0.476** at 1_run: nearly half the baseline responses ran into
    `max_new_tokens=512`, mean 329 output tokens. A truncated-but-coherent response is the
    case the degeneracy detector is most likely to misfile, so read that column before
    reading breakage anywhere downstream.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_STEERING, ST_SCOPE, TAG, ckpt, jb_cached, sh):
    # Not cosmetic: that cell writes acts/views/jailbreaks__all.json, which sets.py reads.
    # cell 27
    assert jb_cached
    if RUN_STEERING.value:
        sh("python", "experiments/steering_jailbreaks/gen_baseline.py", MODEL,
           "--tag", TAG, "--split", "all", "--decoding", "greedy",
           "--batch-size", "32", "--max-batch-tokens", "65536")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="gen_baseline", **ST_SCOPE)
    st_baseline = True
    return (st_baseline,)


@app.cell
def _(HF_REPO, HF_TOKEN, RUN_STEERING, ST_META, ST_SCOPE, ckpt, sh, st_baseline):
    # cell 28
    assert st_baseline
    if RUN_STEERING.value:
        sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
           str(ST_META / "gen_baseline.jsonl"), "--concurrency", "6")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="gen_baseline judged", **ST_SCOPE)
    st_split = True
    return (st_split,)


@app.cell(hide_code=True)
def _(mo):
    # cell 29
    mo.md(r"""
    ## The cell list

    Per (set, direction): three α cells, plus `ablate` where that direction is the one being
    suppressed on that set — 14 per set. Plus one `noop` per (set, layer); the four chosen
    layers are all distinct, so that is 4 per set.

    α's sign is not free. Only one half has headroom on a given set — spec 0.5's restoring
    sign on the successes, its mirror on the refusals — and `steer_single.resolve` refuses
    the other, so a sign slip fails before the model loads rather than producing a floor.
    """)
    return


@app.cell
def _(MODEL, ST_CHOSEN, ST_META, mo):
    # cell 30
    import json as _json
    import tempfile as _tf
    from pathlib import Path as _P

    from transformers import AutoConfig as _AC

    from experiments.common import config as _cfg
    from experiments.steering_jailbreaks import cell as _cell

    _ALPHAS = (0.25, 0.50, 0.75)
    # Read from the config, not hardcoded: the band -- and so which layer needs the
    # out-of-band opt-in -- is a function of depth. A small JSON, not the weights.
    _BAND = _cfg.band(_AC.from_pretrained(MODEL).num_hidden_layers)

    def _jobs(prompt_set):
        """14 target + 4 noop argv tails for one prompt set."""
        out = []
        for axis, layer in ST_CHOSEN.items():
            oob = ["--allow-out-of-band"] if layer not in _BAND else []
            sign = _cell.RESTORE_SIGN[axis] * (1 if prompt_set == "success" else -1)
            if _cell.PRIMARY[prompt_set][axis] == "ablate":
                out.append(["--direction", axis, "--layers", str(layer), *oob])
            for a in _ALPHAS:
                # --mode add is explicit: on the set where this axis is suppressed its
                # PRIMARY is `ablate`, and the -alpha arm is the alternative to that.
                out.append(["--direction", axis, "--mode", "add", "--layers", str(layer),
                            "--alpha", f"{sign * a:g}", *oob])
        for layer in sorted(set(ST_CHOSEN.values())):
            oob = ["--allow-out-of-band"] if layer not in _BAND else []
            out.append(["--arm", "noop", "--layers", str(layer), *oob])
        return out

    ST_JOBS = {}
    _lines, _total = [], 0
    for _set in ("success", "refusal"):
        _js = _jobs(_set)
        _total += len(_js)
        _p = _P(_tf.gettempdir()) / f"st_jobs_{_set}.json"
        _p.write_text(_json.dumps(_js, indent=1), encoding="utf-8")
        ST_JOBS[_set] = str(_p)
        _lines.append(f"**{_set}** — {len(_js)} cells\n\n```\n"
                      + "\n".join(" ".join(j) for j in _js) + "\n```")

    _EXPECT = {"steer_single": len(_jobs("success")),
               "steer_induce": len(_jobs("refusal"))}

    def ST_PENDING(script):
        """-> (cells with a complete manifest, expected).

        Lets the two steer cells skip `steer_batch` outright when their whole set is
        already generated. Without it a resumed session pays a model load each, purely to
        define the markers the judge cell asserts on -- and resume inside `steer_batch`
        generates nothing anyway. Superseded manifests live in meta/_archive, so the live
        directory holds only current ones.
        """
        n = 0
        for f in ST_META.glob(f"{script}__*_manifest.json"):
            try:
                n += _json.loads(f.read_text(encoding="utf-8")).get("status") == "complete"
            except ValueError:                     # torn tail mid-push
                pass
        return n, _EXPECT[script]

    mo.md(f"{_total} cells\n\n" + "\n\n".join(_lines))
    return ST_JOBS, ST_PENDING


@app.cell(hide_code=True)
def _(mo):
    # cell 31
    mo.md(r"""
    ## Steer — GPU

    One `steer_batch` process per set, so the model loads twice rather than 36 times. Every
    job is parsed and validated before the load, so a typo costs a second. Resume is per
    cell *and* per batch inside a cell, so a kill costs at most one batch.

    Each cell **skips `steer_batch` entirely** once all 18 of its manifests are complete, so
    a resumed session pays no model load to get past here. Without that, resume costs two
    loads purely to define the markers the judge cell asserts on.

    `harm_v2 × add × L21` is the run's smoke test — L21 sits beside 50_per_direction's one
    causal cell (`harm add L20`, 24/24).
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_STEERING, ST_JOBS, ST_PENDING, ST_SCOPE, TAG, ckpt,
      sh, st_split):
    # cell 32
    assert st_split                      # the success set is defined by the judged baseline
    _n, _want = ST_PENDING("steer_single")
    if RUN_STEERING.value and _n < _want:
        sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
           "--script", "steer_single", "--jobs", ST_JOBS["success"],
           "--batch-size", "32", "--max-batch-tokens", "65536")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="steer_single cells", **ST_SCOPE)
    else:
        print(f"steer_single: {_n}/{_want} cells complete"
              + (" -- skipped, no model load" if RUN_STEERING.value else " (not running)"))
    st_success = True
    return (st_success,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_STEERING, ST_JOBS, ST_PENDING, ST_SCOPE, TAG, ckpt,
      sh, st_success):
    # cell 33
    assert st_success
    _n, _want = ST_PENDING("steer_induce")
    if RUN_STEERING.value and _n < _want:
        sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
           "--script", "steer_induce", "--jobs", ST_JOBS["refusal"],
           "--batch-size", "32", "--max-batch-tokens", "65536")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="steer_induce cells", **ST_SCOPE)
    else:
        print(f"steer_induce: {_n}/{_want} cells complete"
              + (" -- skipped, no model load" if RUN_STEERING.value else " (not running)"))
    st_steered = True
    return (st_steered,)


@app.cell(hide_code=True)
def _(mo):
    # cell 34
    mo.md(r"""
    ## Judge — API

    One call per row — **16,938 calls** over the 36 cells, ≈$5. Resumable per row, and the
    raw responses are cached in `meta/judge_cache.jsonl` by (request, response, model,
    template_sha), so re-running this cell is free for anything already graded.

    **`--concurrency 6`, not 8.** The binding limit is tokens per minute, not requests:
    measured 200k TPM on `gpt-4o-mini` at ~1.2k tokens a call, i.e. ~165 calls/min, and 8
    workers sat exactly on that ceiling and 429'd. 6 leaves ~25% headroom. The floor is
    ~1.7 h whatever the concurrency — the TPM bucket sets it, not parallelism — so expect
    ~2.3 h at 6.

    It judges every `meta/*.jsonl` that **has a sibling manifest**, which is what
    `judge_strongreject` itself requires. `judge_cache.jsonl` — the judge's own response
    cache — also lives in `meta/`, and a blocklist that named only `_judged` and
    `gen_baseline` picked it up and died on it before grading anything.

    The judge sees the **bare request**, never the jailbreak wrapper, and the deterministic
    detectors run alongside it at no cost — `outcome` is degenerate when *either* says so.
    That matters most for the α=0.75 cells: at 50_per_direction the judge undercounted
    degeneracy by 18pp and every α=1 result was contaminated.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, RUN_STEERING, ST_META, ST_SCOPE, ckpt, sh, st_steered):
    # cell 35
    assert st_steered
    if RUN_STEERING.value:
        # A generations file is one that has a sibling manifest -- which is exactly what
        # judge_strongreject loads first. meta/ also holds judge_cache.jsonl (the judge's
        # own response cache) and the _judged.jsonl outputs; neither has a manifest, and a
        # blocklist missed the cache. gen_baseline is excluded because it is judged above.
        _todo = sorted(p for p in ST_META.glob("*.jsonl")
                       if not p.name.endswith("_judged.jsonl")
                       and not p.name.startswith("gen_baseline")
                       and (ST_META / f"{p.stem}_manifest.json").exists())
        print(f"judging {len(_todo)} cells")
        # allow_fail: judging is resumable per row, so one bad cell should cost a retry of
        # that cell rather than the remaining hour of the pass.
        _failed = []
        for _i, _p in enumerate(_todo, 1):
            print(f"[{_i}/{len(_todo)}] {_p.stem}")
            if sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
                  str(_p), "--concurrency", "6", allow_fail=True).returncode:
                _failed.append(_p.stem)
        # One push for all of them: the hourly timer has been carrying the partials.
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="judged", **ST_SCOPE)
        if _failed:
            print(f"! {len(_failed)} cells did not judge, re-run this cell: {_failed}")
    st_judged = True
    return (st_judged,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, RUN_STEERING, ST_SCOPE, ST_TIMER, TAG, ckpt, sh, st_judged):
    # cell 36
    assert st_judged
    if not RUN_STEERING.value:
        print("steering not run -- nothing to aggregate or stop")
    else:
        sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
        _ok = ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="aggregate", **ST_SCOPE)
        _stale = sh("python", "-m", "experiments.common.check_stale", MODEL, TAG,
                    allow_fail=True).returncode
        # Only stop the timer once the results are actually up: try_push returns None when
        # it was rate-limited, and stopping on a skipped push strands them on local disk.
        if _ok is not None:
            ST_TIMER.set()
        print(("! check_stale reported findings" if _stale else "all artefacts current")
              + (" | hourly checkpoint stopped" if _ok is not None
                 else " | push SKIPPED, timer left running"))
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 37
    mo.md(f"""
    ## Read the results

    `steering_jailbreaks/results/{TAG}/<model_slug>/csv/`: `aggregate_cells.csv` is every
    cell, `aggregate_controls.csv` each target with `d_*_vs_noop`, `aggregate_paired.csv`
    necessity beside sufficiency.

    Read in this order, or the numbers mislead:

    1. **`pct_degenerate` before any effect.** `eval_v2` at L9 is the risk — 0.32 depth,
       and every `eval` α=1 cell at 50_per_direction was 80–97% broken.
    2. **`d_*_vs_noop`, never vs the baseline.** Different batch composition; a provably
       inert hook flipped 6 of 30 rows at 50_per_direction.
    3. **`read_<axis>` for all four axes.** No cell moved only its own axis last time —
       `persona add` moved `read_harm` by −80, and ablating harm is itself the induce lever.

    There is **no `random` arm at this tag**. Pass 2 adds it on whatever moved, so nothing
    here is a specificity claim yet, and `|Δh|` is the comparable magnitude — not α, which
    is scaled by a σ that differs per layer and per direction.
    """)
    return


if __name__ == "__main__":
    app.run()
