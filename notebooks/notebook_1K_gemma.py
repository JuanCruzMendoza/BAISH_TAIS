import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    # cell 1
    import os, pathlib, subprocess, sys
    from collections import deque

    import marimo as mo

    MODEL = "google/gemma-2-9b-it"
    TAG = "1K_per_direction"
    HF_REPO = "JuanCruzMendoza/BAISH_TAIS"

    # Pinned once and used by the baseline, every steering cell and every pair arm. Greedy
    # is bit-reproducible only at fixed batch composition, so a target and its no-op have to
    # be generated under the same numbers -- changing these once anything is generated
    # breaks every target-vs-noop comparison at this tag, silently. Lower than Qwen's
    # 32/65536 because gemma-2-9b's KV cache is ~6x bigger per token (42 layers x 8 KV heads
    # x 256 vs 28 x 4 x 128); raise it on an 80 GB card, but only before the baseline runs.
    BATCH = ("--batch-size", "16", "--max-batch-tokens", "24576")

    # This file lives in notebooks/, so the repo root is above it when opened from a
    # checkout; standalone in molab there is no checkout and we clone beside the notebook.
    NB = mo.notebook_dir()
    ROOTS = [d for d in [NB, *NB.parents]
             if (d / "experiments" / "common" / "ckpt.py").exists()]
    REPO = str(ROOTS[0]) if ROOTS else str(NB / "BAISH_TAIS")

    def sh(*a, cwd=None, allow_fail=False):
        """Stream the subprocess's output while it runs, keeping the tail for the error.

        Captured output arrives only when the process exits, so "which cell is generating
        now" reached the screen hours after it stopped being the answer -- for a
        `steer_batch` driving 40 cells under one load, that is the whole run. `-u` because
        a piped child block-buffers its own stdout, so the line naming a cell would
        otherwise land 8 KB later, next to a different cell.
        """
        a = list(a)
        if a[:1] == ["python"]:
            a.insert(1, "-u")
        p = subprocess.Popen(a, cwd=cwd or REPO, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace",
                             bufsize=1)
        tail = deque(maxlen=60)
        for line in p.stdout:
            line = line.rstrip()
            tail.append(line)
            print(line)
        p.wait()
        if p.returncode and not allow_fail:
            raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, a))}\n"
                               + "\n".join(tail))
        return p

    if not pathlib.Path(REPO, "experiments").exists():
        sh("git", "clone", "https://github.com/JuanCruzMendoza/BAISH_TAIS.git", REPO,
           cwd=str(NB))
    else:
        sh("git", "fetch", "origin")
        # A restart pulls results back from the Hub, and meta/runs.csv plus the manifests
        # churn on every re-run -- so tracked files under experiments/ differ, the rebase
        # refuses, and the clone silently strands on an old commit. Discarding them is
        # free: the Hub is authoritative and cell 6 re-pulls. Not `reset --hard`, so
        # untracked and ignored files -- acts/, vectors/, the hf cache -- survive.
        sh("git", "checkout", "--", "experiments")
        _dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip()
        if _dirty:
            print(f"! tracked files still modified, not rebased:\n{_dirty[:500]}")
        else:
            sh("git", "rebase", "origin/main", allow_fail=True)
    sh("git", "log", "--oneline", "-1")
    # transformers >= 4.42 for gemma-2; `openai` is the judge's SDK, and OpenRouter speaks
    # the same protocol so the fallback needs no second package.
    sh("pip", "install", "-q", "transformers>=4.44", "accelerate", "numpy", "matplotlib",
       "huggingface_hub", "hf_xet", "openai")
    sh("python", "-c",
       "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))")

    sys.path.insert(0, REPO)
    return BATCH, HF_REPO, MODEL, REPO, TAG, mo, os, sh


@app.cell(hide_code=True)
def _(MODEL, TAG, mo):
    # cell 2
    mo.md(f"""
    # {TAG} on `{MODEL}`

    Spec: `research/spec-whole-rerun.md`. The GPU half of the pipeline for a second model:
    extraction → probe_jailbreak_detection → steering (baseline + α sweep) → projection
    pairs. Cross-probe, every judge pass and the narrativity check run locally.

    **Same tag, different model slug**, so nothing here can collide with the Qwen run.
    Dropped from it: `ablate`, `cap`, the `length` foil, the `random` arm, the decoding
    comparison (greedy, reused) and the second-layer pass. **α = 0.25 / 0.50 / 0.75 / 1.00**
    for every direction, sign derived from `cell.RESTORE_SIGN`.

    ## Run all cells

    Every stage is guarded on an artefact, not on a checkbox: what is already complete is
    skipped **before** the model load, so re-running the notebook after a kill is the way to
    resume.

    **This notebook does what needs the GPU, plus whatever is free because the data is
    already here.** It hands off through the Hub — pushes, prints the commands, stops:

    - **the judge passes always run on your own machine**, ~23k API calls over 3–5 h, during
      which a rented GPU would be idle and billed, and no API key needs to reach the instance;
    - **cross_probe_detection (§3) runs here only if the pole cache is already on this box**
      — i.e. extraction ran in *this* session. It is CPU-only, so when the 2.2 GiB is on local
      disk it is ~2 min and free; on a fresh instance that resumed from complete manifests it
      is skipped rather than pulling that cache back, since that download is what kept hitting
      the Xet read-token 429.

    Five points stop and wait for you:

    | stop | what to read first | what to do |
    |---|---|---|
    | **gate 1 — layers** | already decided: story L28+L15, persona L15, harm L19, eval L8 | nothing — the box is prefilled |
    | **judge the baseline** | — | run it locally, push, re-run here |
    | **judge the sweep** | — | run it locally + `aggregate.py` |
    | **gate 3 — pairs** | `geometry_cos_chosen.csv` (cos at a's layer vs the ±3/√d null band) + the α curve | type the ordered pairs and their α |
    | **judge the pairs** | — | run it locally + `aggregate.py` |

    The baseline stop is a real dependency, not a convenience: the two prompt sets *are* the
    baseline's 3-way labels, so the sweep cannot be built until it is graded. Cross-probe and
    the narrativity check block nothing — whichever machine runs them, the GPU keeps working.
    `research/spec-whole-rerun.md` §J lists every local command in order.

    ## Gemma-2 specifics

    - **Gated repo**: the HF token below is what downloads the weights, so it needs the
      licence accepted on `{MODEL}`.
    - **`ATTN_IMPL=eager`**, set in cell 5. Gemma-2 soft-caps the attention logits and the
      sdpa kernel drops that — a different activation and a different generation, not a
      speed knob.
    - **L = 42**, so every absolute layer is re-derived here; none of Qwen's carry over.
    """)
    return


@app.cell
def _(mo):
    # cell 3
    # The only credential this notebook needs. No judge key: grading runs off the GPU box
    # (see the hand-off cells), so an API key never reaches the rented instance.
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write, and gemma-2 licence accepted)",
                          kind="password", full_width=True)
    HF_TOKEN
    return (HF_TOKEN,)


@app.cell(hide_code=True)
def _(mo):
    # cell 4
    mo.md(r"""
    ## Checkpoint, and what it costs

    Commits are the scarce resource, not bytes: `upload_folder` emits **one commit per 256
    files considered** — considered, not changed. So every push is scoped to the experiment
    that wrote it, and nothing is pushed twice.

    | scope | pack | pushed when |
    |---|---|---|
    | `extraction` | **yes** (`acts/blobs.tar`) | inside the extraction stage only |
    | `probe_jailbreak_detection` | no | after the readout, after the metrics |
    | `cross_probe_detection` | no | at gate 1, **only if** it ran here (cache present) |
    | `steering_jailbreaks` | no | after each generation stage |

    `cross_probe_detection` is pulled unconditionally — it is a handful of small csv and
    manifests, and gate 1 needs them to tell "already computed at these layers" from "not
    computed on this box". It is pushed only in the branch that actually computed it.

    **The extraction scope is never pushed after the jailbreak activations are cached.**
    Those 1,009 blobs land in *extraction's* tree, and a packed push would re-tar them into
    the 2.2 GiB cache for activations nothing downstream reads.

    The hourly timer is armed **only around a stage that is actually generating** and
    stopped when it returns, so an idle session does not spend commits re-uploading an
    unchanged tree.

    The push is staged — the scoped tree is mirrored (hard links) and the append-only
    `.jsonl` files snapshotted and trimmed to their last complete row — so no file is
    measured while a subprocess is still appending to it.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, REPO, TAG, mo, os):
    # cell 5
    import json as _json
    from pathlib import Path as _P

    from transformers import AutoConfig as _AC

    from experiments.common import ckpt, config as cfg

    mo.stop(not HF_TOKEN.value,
            mo.md("*Paste the HF token to start. It downloads the gated weights and "
                  "checkpoints every result; nothing runs without it.*"))
    # from_pretrained in every subprocess authenticates with $HF_TOKEN -- the same token
    # that checkpoints the results, and gemma-2 is a gated repo.
    os.environ["HF_TOKEN"] = HF_TOKEN.value
    # Gemma-2 soft-caps attention logits; sdpa and flash drop that. model.load reads this,
    # and unset means transformers' own default, so no Qwen artefact moves.
    os.environ["ATTN_IMPL"] = "eager"

    AXES = ["story_v2_1k", "persona_v2", "harm_v2", "eval_v2"]
    ALPHAS = (0.25, 0.50, 0.75, 1.00)
    RULES = ("midpoint", "gap_mid")

    EX_SCOPE = {"experiment": "extraction", "tag": TAG}
    JB_SCOPE = {"experiment": "probe_jailbreak_detection", "tag": TAG}
    CP_SCOPE = {"experiment": "cross_probe_detection", "tag": TAG}
    ST_SCOPE = {"experiment": "steering_jailbreaks", "tag": TAG}

    def _root(experiment):
        return _P(REPO, "experiments", experiment, "results", TAG,
                  MODEL.replace("/", "_"))

    # cross_probe_detection is written here only when the pole cache happens to be on this
    # box already (see the gate-1 cell); its scope exists either way so the push is available
    # in the branch that takes it, and is simply never called in the branch that does not.
    EX_ROOT, JB_ROOT, CP_ROOT, ST_ROOT = (
        _root("extraction"), _root("probe_jailbreak_detection"),
        _root("cross_probe_detection"), _root("steering_jailbreaks"))

    # A small JSON, not the weights -- but a gated one, hence $HF_TOKEN above.
    N_LAYERS = _AC.from_pretrained(MODEL).num_hidden_layers
    BAND = cfg.band(N_LAYERS)

    def DONE(root, stem, **want):
        """Complete manifest whose config matches `want`.

        The `want` half is what makes a gate re-runnable: `jb_metrics`, `cross_auroc` and
        `geometry` record the chosen layers in their config, so revising the gate makes
        their manifests stop matching and the stage runs again instead of leaving a
        complete-looking table computed at the old layers.
        """
        try:
            m = _json.loads(_P(root, "meta", f"{stem}_manifest.json")
                            .read_text(encoding="utf-8"))
        except (OSError, ValueError):            # missing, or a torn tail mid-push
            return False
        return (m.get("status") == "complete"
                and all(m.get("config", {}).get(k) == v for k, v in want.items()))

    def ST_COMPLETE(stem):
        """A steering cell: complete manifest **and** its rows on disk (spec 0.11).

        Both halves matter once the scope passes 256 files and a push splits into two
        commits: a 429 between the parts can land a manifest without its `.jsonl`, and a
        manifest-only skip would drop that cell from the judge pass and from `aggregate`
        without saying so.
        """
        return (ST_ROOT / "meta" / f"{stem}.jsonl").exists() and DONE(ST_ROOT, stem)

    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    print(f"{MODEL}: L={N_LAYERS}, band {BAND[0]}-{BAND[-1]}, alphas {ALPHAS}")
    return (ALPHAS, AXES, BAND, CP_ROOT, CP_SCOPE, DONE, EX_ROOT, EX_SCOPE, JB_ROOT,
            JB_SCOPE, N_LAYERS, RULES, ST_COMPLETE, ST_ROOT, ST_SCOPE, ckpt)


@app.cell
def _(AXES, CP_SCOPE, DONE, EX_ROOT, EX_SCOPE, HF_REPO, HF_TOKEN, JB_ROOT, JB_SCOPE,
      ST_SCOPE, ckpt):
    # cell 6
    # Small trees first, so the pending checks below can be made without the pole cache.
    # `subpaths` is glob-relative to <model_slug>'s parent. cross_probe_detection is pulled
    # too: it is small, and the gate-1 cell needs its manifests to tell "already computed at
    # these layers" from "not computed here".
    print("extraction (small):",
          ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                    subpaths=["*/vectors/**", "*/meta/**", "*/csv/**", "*/figures/**"]))
    print("probe_jailbreak_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **JB_SCOPE))
    print("cross_probe_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **CP_SCOPE))
    print("steering_jailbreaks:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))

    # Only two things here *require* activations: extraction itself, to finish caching, and
    # jb_readout, which re-projects onto all 8,000 pole prompts. `jb_metrics` reads the
    # readout `.pt` and steering reads `vectors/` + the jailbreak view, so once those two are
    # done the 2.2 GiB never comes down again -- which is what removes the Xet read-token 429
    # a packed pull of it kept hitting.
    #
    # cross_auroc / geometry are the third reader, but they never justify the download: they
    # run at gate 1 only if the cache is *already* here, and are deferred to the local pass
    # if it is not. So this condition stays keyed on the two that would otherwise fail.
    EX_CACHED = all(DONE(EX_ROOT, f"cache_activations__{_d}__{_s}")
                    for _d in AXES for _s in ("train", "heldout"))
    if not (EX_CACHED and DONE(JB_ROOT, "jb_readout")):
        # The tar, NOT `*/acts/**`. pack=True unpacks acts/blobs.tar into acts/blobs/, so
        # the scripts see the same tree either way and run_key resume still cache-hits --
        # but asking for the whole directory also fetches the 7,731 loose .npy, which is
        # the same bytes a second time and, far worse, 7,731 requests against the Xet
        # read-token quota instead of one. views/ is small and has no packed form.
        print("extraction acts:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                                            subpaths=["*/acts/blobs.tar", "*/acts/views/**"],
                                            pack=True))
    else:
        print("extraction acts: nothing on this box requires them -- skipped (2.2 GiB)")
    pulled = True
    return EX_CACHED, pulled


@app.cell(hide_code=True)
def _(mo):
    # cell 7 (markdown)
    mo.md(r"""
    # 1 — extraction

    Four directions at 800 train / 200 held-out, `--curve` for the saturation measurement,
    **manual** layer choice. No length foil, no `--append-task` (baked into the v2 pairs),
    no `compare_crossed` and no v1 transfer — both are v1 arms.

    ~6,400 train + 1,600 held-out prompts, one model load. Everything below the caching cell
    reads the blob cache, so re-running the analysis costs seconds.
    """)
    return


@app.cell
def _(AXES, EX_CACHED, EX_SCOPE, HF_REPO, HF_TOKEN, MODEL, TAG, ckpt, pulled, sh):
    # cell 8
    assert pulled
    if EX_CACHED:
        # cache_activations calls mdl.load() before it consults the blob cache, so a
        # finished pass still costs 18 GB of weights. This is the skip that matters.
        print("extraction activations complete -- skipped, no model load")
    else:
        # The timer only for the duration of the pass: a packed tick re-tars the whole
        # cache, so it must not outlive the stage that needs the protection.
        _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, pack=True,
                               **EX_SCOPE)
        try:
            # 8 cells in one process; cache_activations prints `[i/8] <dataset>/<split>`
            # as it enters each and a heartbeat inside it.
            print(f"=== caching activations: {', '.join(AXES)} x train,heldout ===")
            sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
               "--dataset", ",".join(AXES), "--split", "train,heldout")
        finally:
            _timer.set()
        # pack=True is what puts the cache on the Hub at all: ckpt.push now holds loose
        # blobs back unconditionally, so blobs.tar is the only stored form. That guard used
        # to live under `if pack:`, which left the *other* pushes of this same scope
        # (vectors, figures -- both unpacked) free to upload all 7,731 .npy beside the tar.
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="acts", pack=True, **EX_SCOPE)
    ex_cached = True
    return (ex_cached,)


@app.cell
def _(AXES, DONE, EX_ROOT, EX_SCOPE, HF_REPO, HF_TOKEN, MODEL, TAG, ckpt, ex_cached, sh):
    # cell 9
    assert ex_cached
    # `--curve` writes csv/directions__<axis>_curve.json: cos(d_n, d_800) for
    # n in {50..750} at 5 seeds, plus the disjoint cos(d_800_train, d_200_heldout).
    # probe_select emits the per-layer table and stops -- no band rule, no primary layer.
    _todo = [a for a in AXES if not (DONE(EX_ROOT, f"directions__{a}")
                                     and DONE(EX_ROOT, f"probe_select__{a}"))]
    print(f"to extract: {_todo or 'nothing'}"
          + (f" | already done: {[a for a in AXES if a not in _todo]}" if _todo else ""))
    for _i, _a in enumerate(_todo, 1):
        print(f"\n=== [{_i}/{len(_todo)}] extract_direction {_a} (+ curve) ===")
        sh("python", "experiments/extraction/extract_direction.py", MODEL,
           "--tag", TAG, "--direction", _a, "--curve")
        print(f"\n=== [{_i}/{len(_todo)}] probe_select {_a} ===")
        sh("python", "experiments/extraction/probe_select.py", MODEL,
           "--tag", TAG, "--direction", _a)
    if _todo:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="directions + probe_select",
                      **EX_SCOPE)
    else:
        print("directions and probe_select complete for all four axes -- skipped")
    ex_vectors = True
    return (ex_vectors,)


@app.cell
def _(AXES, DONE, EX_ROOT, EX_SCOPE, HF_REPO, HF_TOKEN, MODEL, TAG, ckpt, ex_vectors, sh):
    # cell 10
    assert ex_vectors
    # 8 figures, two per direction: the cos curve and cohens_dz_train.
    if all(DONE(EX_ROOT, f"plot__{_a}") for _a in AXES):
        print("extraction figures complete -- skipped")
    else:
        sh("python", "experiments/extraction/plot_figures.py", MODEL, "--tag", TAG,
           "--with-heldout")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="figures", **EX_SCOPE)
        # check_stale spans the tag, so it also reports whatever the later stages have not
        # written yet. Never fatal for that reason.
        sh("python", "-m", "experiments.common.check_stale", MODEL, TAG, allow_fail=True)
    ex_done = True
    return (ex_done,)


@app.cell(hide_code=True)
def _(mo):
    # cell 11 (markdown)
    mo.md(r"""
    # 2 — probe_jailbreak_detection (H2)

    The four probes against **all 1,009** jailbreak prompts (1,017 minus the 8 whose
    `prompt` is its `request`), at two thresholds. Swept over **every layer** here, because
    the layer is not chosen yet — gate 1 below reads these curves.

    The activation cell writes into *extraction's* blob cache (`--tag` must match) and its
    1,009 blobs are deliberately **not** checkpointed: they are only an input to
    `jb_readout.py`, and once its 470 KB `.pt` is on the Hub they are disposable. Once that
    `.pt` exists the cell drops to `--view-only` — the view is a tokenizer artefact,
    seconds on CPU, and it is what `sets.py` rebuilds the steering prompt sets from.
    """)
    return


@app.cell
def _(DONE, JB_ROOT, MODEL, TAG, ex_done, sh):
    # cell 12
    assert ex_done                        # the probes come from the extraction cells
    # --view-only writes acts/views/jailbreaks__all.json from the tokenizer alone: no
    # weights, and it reproduces the GPU path's view_key exactly, so no run_key moves.
    _vo = ["--view-only"] if DONE(JB_ROOT, "jb_readout") else []
    print("=== jailbreaks/all: "
          + ("view only, no weights ===" if _vo else "caching 1,009 activations ==="))
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", "jailbreaks", "--split", "all", "--poles", "pos", *_vo)
    jb_cached = True
    return (jb_cached,)


@app.cell
def _(AXES, DONE, HF_REPO, HF_TOKEN, JB_ROOT, JB_SCOPE, MODEL, TAG, ckpt, jb_cached, sh):
    # cell 13
    assert jb_cached
    if DONE(JB_ROOT, "jb_readout"):
        print("jb_readout complete -- skipped")
    else:
        print(f"=== jb_readout: {', '.join(AXES)} over 1,009 prompts, every layer ===")
        sh("python", "experiments/probe_jailbreak_detection/jb_readout.py", MODEL,
           "--tag", TAG, "--axes", ",".join(AXES))
        # The one push that matters here: after it the 1,009 jailbreak blobs are disposable.
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_readout", **JB_SCOPE)
    jb_read = True
    return (jb_read,)


@app.cell
def _(AXES, DONE, HF_REPO, HF_TOKEN, JB_ROOT, JB_SCOPE, MODEL, RULES, TAG, ckpt, jb_read,
      sh):
    # cell 14
    assert jb_read
    # Every layer 0..L, both threshold rules, into the parallel `__all` stem. No --layers:
    # the chosen ones do not exist yet, and this sweep is what they are read off.
    _todo = [r for r in RULES if not DONE(JB_ROOT, f"jb_metrics__{r}__all")]
    for _rule in _todo:
        print(f"\n=== jb_metrics --all-layers, threshold {_rule} ===")
        sh("python", "experiments/probe_jailbreak_detection/jb_metrics.py", MODEL,
           "--tag", TAG, "--axes", ",".join(AXES), "--threshold", _rule, "--all-layers")
    if _todo or not DONE(JB_ROOT, "plot_layer_curves__all"):
        print("\n=== plot_layer_curves --all-layers ===")
        sh("python", "experiments/probe_jailbreak_detection/plot_layer_curves.py", MODEL,
           "--tag", TAG, "--all-layers")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_metrics all-layers", **JB_SCOPE)
    else:
        print("all-layer metrics and curves complete -- skipped")
    jb_all = True
    return (jb_all,)


@app.cell(hide_code=True)
def _(BAND, TAG, mo):
    # cell 15 (markdown)
    mo.md(f"""
    ## ▸ GATE 1 — the chosen layers (already filled in)

    **Decided, from `extraction/insights.md` §Gemma 9B:** `story_v2_1k` **L28 and L15**,
    `persona_v2` L15, `harm_v2` L19, `eval_v2` L8. Story keeps both because its two criteria
    disagree by 13 layers — L28 is the `cohens_dz_train` peak (3.62), L15 the
    fiction − nonfiction `pct_reads` peak (+77.5 against L28's +15.5) — and with one layer a
    null cell cannot be told apart from a wrong layer. `a=L1+L2` gives an axis two cells;
    everything downstream runs each (axis, layer) as its own cell.

    Edit the box only to revise that. What each criterion is, if you do:

    - `extraction/results/{TAG}/<model_slug>/csv/probe_select__<axis>.csv` —
      **`cohens_dz_train`**, confirmed against `cohens_dz_heldout` (200 pairs),
      `mean_paired_cos` and `lopo_ci_lo`. This is the default criterion.
    - `probe_jailbreak_detection/.../csv/jb_metrics__midpoint__all_rate.csv` — for
      **`story_v2_1k`** prefer the layer with the largest **fiction − nonfiction
      `pct_reads`** gap where it disagrees with `cohens_dz`: that is the layer that
      discriminates the jailbreak families, and 50_per_direction measured r = 0.00 between
      probe quality and steering effect. Read `ref_tpr` beside it — a low `pct_reads` at low
      `ref_tpr` is the threshold failing, not a reading.

    The reporting band is **L{BAND[0]}–L{BAND[-1]}**, so **L15 and L8 are outside it** and
    every cell that uses them carries `--allow-out-of-band`, recorded in its manifest.

    One caveat on the `_chosen` tables below: **`jb_metrics` takes one layer per probe**, so
    it runs at each axis's **first** layer (story L28); story's L15 row is in the per-layer
    file the same run writes (`jb_metrics__<rule>__all_rate.csv`). `cross_auroc` and
    `geometry` do **not** have that limit — `a=L1+L2` gives them a second probe row, and they
    get both of story's layers.
    """)
    return


@app.cell
def _(mo):
    # cell 16
    # Prefilled with the decision, not blank: gate 1 is answered, and a blank box would stop
    # the notebook on a question that has already been settled. `+` gives an axis two layers.
    CHOSEN_IN = mo.ui.text(
        label="chosen layers (axis=layer, `+` for a second layer, comma-separated)",
        full_width=True, value="story_v2_1k=28+15,persona_v2=15,harm_v2=19,eval_v2=8")
    CHOSEN_IN
    return (CHOSEN_IN,)


@app.cell
def _(AXES, BAND, CHOSEN_IN, N_LAYERS, jb_all, mo):
    # cell 17
    assert jb_all
    mo.stop(not CHOSEN_IN.value.strip(),
            mo.md("*Gate 1: enter the layers, `axis=L` or `axis=L1+L2`.*"))

    # Not cfg.parse_axis_layers: that maps an axis to exactly one layer, and story has two.
    # Order matters -- the first is the axis's primary, the one the single-layer tables use.
    CHOSEN = {}
    for _part in CHOSEN_IN.value.replace(" ", "").split(","):
        if not _part:
            continue
        _ax, _sep, _ls = _part.partition("=")
        if not _sep or _ax not in AXES:
            raise ValueError(f"{_part!r}: expected axis=layer with axis in {AXES}")
        _layers = [int(x) for x in _ls.split("+") if x]
        if not _layers or any(not 0 <= l <= N_LAYERS for l in _layers):
            raise ValueError(f"{_part!r}: layers must be 0..{N_LAYERS}")
        CHOSEN.setdefault(_ax, [])
        CHOSEN[_ax] += [l for l in _layers if l not in CHOSEN[_ax]]
    if set(CHOSEN) != set(AXES):
        raise ValueError(f"name exactly {AXES}; missing {sorted(set(AXES) - set(CHOSEN))}")

    # The single-layer consumers (jb_metrics, cross_auroc, geometry) take one per probe.
    PRIMARY = {a: ls[0] for a, ls in CHOSEN.items()}
    # Layers, not axes: a no-op is per (set, layer), and story@15 and persona@15 share one.
    OOB = {l for ls in CHOSEN.values() for l in ls if l not in BAND}
    LAYER_ARG = ",".join(f"{a}={l}" for a, l in PRIMARY.items())
    # cross_auroc / geometry take `axis=L1+L2`, where `+` adds a second *probe row* for that
    # axis -- the same vector read at another layer -- rather than a second axis. Story gets
    # both of its steering layers, which is what tells a null cell from a wrong layer.
    PROBE_ARG = ",".join(a + "=" + "+".join(map(str, ls)) for a, ls in CHOSEN.items())
    _cells = sum(len(ls) for ls in CHOSEN.values())
    _distinct = sorted({l for ls in CHOSEN.values() for l in ls})
    # Plain concatenation, not a nested f-string: PEP 701 is 3.12+ and this has to parse
    # on whatever the instance ships.
    print("chosen: " + ", ".join(a + "=L" + "+L".join(map(str, ls))
                                 for a, ls in CHOSEN.items()))
    print(f"{_cells} (axis, layer) pairs over {len(_distinct)} distinct layers {_distinct}; "
          f"primary for the single-layer tables: {LAYER_ARG}")
    print(f"out of band (L{BAND[0]}-{BAND[-1]}), explicit opt-in: "
          f"{sorted(OOB) or 'none'}")
    return CHOSEN, LAYER_ARG, OOB, PRIMARY, PROBE_ARG


@app.cell
def _(AXES, DONE, HF_REPO, HF_TOKEN, JB_ROOT, JB_SCOPE, LAYER_ARG, MODEL, PRIMARY, RULES,
      TAG, ckpt, sh):
    # cell 18
    # The headline files: one row per probe x slice at that probe's own layer -- PRIMARY,
    # since the script takes one layer per probe. Re-runs whenever the gate changes, because
    # `probe_layers` is part of what DONE compares.
    _todo = [r for r in RULES if not DONE(JB_ROOT, f"jb_metrics__{r}", probe_layers=PRIMARY)]
    for _rule in _todo:
        print(f"\n=== jb_metrics at {LAYER_ARG}, threshold {_rule} ===")
        sh("python", "experiments/probe_jailbreak_detection/jb_metrics.py", MODEL,
           "--tag", TAG, "--axes", ",".join(AXES), "--layers", LAYER_ARG,
           "--threshold", _rule)
    if _todo:
        print("\n=== plot_layer_curves (band) ===")
        sh("python", "experiments/probe_jailbreak_detection/plot_layer_curves.py", MODEL,
           "--tag", TAG)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="jb_metrics chosen", **JB_SCOPE)
    else:
        print("chosen-layer metrics current for these layers -- skipped")
    jb_chosen = True
    return (jb_chosen,)


@app.cell(hide_code=True)
def _(HF_REPO, MODEL, PROBE_ARG, TAG, jb_chosen, mo):
    # cell 19 (markdown)
    assert jb_chosen
    mo.md(f"""
    # 3 — cross_probe_detection (H1) — **here iff the cache is already here**

    4×4 at the chosen layers. No GPU and ~2 min of CPU, but its `--diag heldout` AUROC is
    computed on the **cached pole activations**, not on the vectors — so the only thing that
    decides where it runs is whether `acts/` is on this box:

    - **it is** — extraction ran in *this* session, so the cache is on local disk and the
      next cell computes the whole thing for free. This is the good case: edit gate 1 above
      and re-run, and cross-probe follows the layers you just chose.
    - **it is not** — a fresh instance that resumed from complete extraction manifests never
      downloaded the cache (cell 6 skips it deliberately). The next cell then does **nothing**
      rather than pull 2.2 GiB whose per-chunk read tokens are what hit the Xet 429. Run it on
      your own machine instead, where the cache is downloaded once and kept:

    ```bash
    M={MODEL}; T={TAG}; export RUN_TAG=$T
    python -c "from experiments.common import ckpt; ckpt.pull('{HF_REPO}', experiment='extraction', tag='$T', subpaths=['*/vectors/**', '*/meta/**', '*/acts/blobs.tar', '*/acts/views/**'], pack=True)"
    A=story_v2_1k,persona_v2,harm_v2,eval_v2; L={PROBE_ARG}
    python experiments/cross_probe_detection/cross_auroc.py $M --tag $T --axes $A --layers $L --diag heldout
    python experiments/cross_probe_detection/geometry.py    $M --tag $T --axes $A --layers $L
    python experiments/cross_probe_detection/plot_matrices.py $M --tag $T
    python -c "from experiments.common import ckpt; ckpt.push('{HF_REPO}', experiment='cross_probe_detection', tag='$T', msg='cross-probe')"
    ```

    Either way `--layers` is `{PROBE_ARG}`: `+` gives story a **second probe row**, the same
    vector read at L15, not a second axis. Gate 3 reads `cos(story@15, persona@15)` from
    `geometry_cos.csv`, which spans every band layer regardless.

    Nothing in §4 waits on this — the sweep runs either way.
    """)
    return


@app.cell
def _(AXES, CHOSEN, CP_ROOT, CP_SCOPE, DONE, EX_ROOT, HF_REPO, HF_TOKEN, MODEL, PROBE_ARG,
      TAG, ckpt, jb_chosen, sh):
    # cell 20
    assert jb_chosen
    # The one thing that decides: is the pole cache on this box? `views/` is what read_view
    # raises on and `blobs/` is the 2.2 GiB itself, so both have to be there -- a tree pulled
    # with `*/meta/**` alone has the manifests and neither.
    _has_acts = (any(EX_ROOT.joinpath("acts", "views").glob("*"))
                 and any(EX_ROOT.joinpath("acts", "blobs").glob("*.npy")))
    # cross_auroc records the layers as `probe_layers`, geometry and plot_matrices as
    # `chosen_layers`; both are the {axis: [layers]} dict, so revising gate 1 makes all three
    # stop matching and they recompute instead of leaving tables built at the old layers.
    _current = (DONE(CP_ROOT, "cross_auroc", probe_layers=CHOSEN)
                and DONE(CP_ROOT, "geometry", chosen_layers=CHOSEN)
                and DONE(CP_ROOT, "plot_matrices", chosen_layers=CHOSEN))

    if _current:
        print(f"cross-probe: already computed at {PROBE_ARG} -- skipped")
    elif not _has_acts:
        print("cross-probe: no pole cache on this box, and it is not worth 2.2 GiB to make "
              "one -- skipped. Run the local block above (§J.0).")
    else:
        for _script, _extra in (("cross_auroc", ["--diag", "heldout"]),
                                ("geometry", []), ("plot_matrices", None)):
            print(f"\n=== {_script} at {PROBE_ARG} ===")
            _args = (["--axes", ",".join(AXES), "--layers", PROBE_ARG, *_extra]
                     if _extra is not None else [])
            sh("python", f"experiments/cross_probe_detection/{_script}.py", MODEL,
               "--tag", TAG, *_args)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="cross-probe", **CP_SCOPE)
    return


@app.cell(hide_code=True)
def _(mo):
    # cell 21 (markdown)
    mo.md(r"""
    # 4 — steering_jailbreaks (H3): generation only

    Greedy over all 1,009 prompts for the baseline, then **`add` only, α ∈ 0.25 / 0.50 /
    0.75 / 1.00** at each chosen (axis, layer), signed: restoring is `RESTORE_SIGN` (−1 for
    the framing axes, +1 for the refusal ones) on the success set and its mirror on the
    refusal set. `steer_single.resolve` refuses the half with no headroom, so a sign slip
    fails before the model loads.

    With story at two layers that is **5 (axis, layer) pairs × 4 α = 20 target cells per
    set**, plus one no-op per (set, layer) — 4, since story@L15 and persona@L15 share
    theirs — so **24 per set, 48 in total**, ≈22.6k generations.

    **Judging does not happen here.** It is API-bound, ~19k calls over 3–5 h, and a rented
    GPU idling through it is the most expensive thing in the run — so it is a local step, and
    no API key is ever pasted into this instance. Two consequences:

    - **`gen_baseline` judged is a hard dependency**, because the two prompt sets *are* its
      3-way labels. The gate below stops the notebook until the judged file comes back from
      the Hub, so the baseline and the sweep are two GPU sessions with a local pass between.
    - the sweep cells end at generation; the hand-off cell prints exactly what to run locally.

    The baseline does not depend on gate 1, so it runs while you are still choosing layers.
    """)
    return


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, ST_COMPLETE, ST_SCOPE, TAG, ckpt, jb_cached, sh):
    # cell 22
    assert jb_cached                      # that cell writes the view sets.py reads
    if ST_COMPLETE("gen_baseline"):
        # gen_baseline calls mdl.load() before it reads its own resume, so re-entering a
        # finished baseline still costs the weights.
        print("gen_baseline complete -- skipped, no model load")
    else:
        _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
        try:
            print("=== gen_baseline: unsteered greedy over all 1,009 prompts ===")
            sh("python", "experiments/steering_jailbreaks/gen_baseline.py", MODEL,
               "--tag", TAG, "--split", "all", "--decoding", "greedy", *BATCH)
        finally:
            _timer.set()
        # The guarantee before the gate below sends you to another machine: the rows have to
        # be on the Hub for the local pass to grade them.
        ckpt.push(HF_REPO, token=HF_TOKEN.value, msg="gen_baseline", **ST_SCOPE)
    st_baseline = True
    return (st_baseline,)


@app.cell
def _(HF_REPO, MODEL, ST_ROOT, TAG, mo, st_baseline):
    # cell 23
    assert st_baseline
    # The two prompt sets ARE the baseline's 3-way labels, so nothing below can be built
    # until it is graded -- and grading is a local step now. This is the hand-off: it stops
    # here, you run the pass on your own machine, push, and re-run the notebook.
    _judged = ST_ROOT / "meta" / "gen_baseline_judged.jsonl"
    mo.stop(not _judged.exists(), mo.md(f"""
    ### ▸ Judge the baseline locally, then re-run this notebook

    `gen_baseline.jsonl` is on the Hub; `gen_baseline_judged.jsonl` is not here yet. On your
    own machine, in the repo, with `OPENAI_API_KEY` (and ideally `OPENROUTER_API_KEY`) in
    `.env`:

    ```bash
    M={MODEL}; export RUN_TAG={TAG}
    python -c "from experiments.common import ckpt; ckpt.pull('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}')"
    python experiments/steering_jailbreaks/judge_strongreject.py \\
        experiments/steering_jailbreaks/results/{TAG}/{MODEL.replace('/', '_')}/meta/gen_baseline.jsonl \\
        --concurrency 6
    python -c "from experiments.common import ckpt; ckpt.push('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}', msg='baseline judged')"
    ```

    1,009 calls, ~10 min, well under the daily cap. Then re-run here: cell 6 pulls it back,
    every finished GPU stage skips before its model load, and the sweep starts.
    """))
    st_split = True
    return (st_split,)


@app.cell
def _(ALPHAS, CHOSEN, MODEL, N_LAYERS, OOB, ST_COMPLETE, TAG, mo, st_split):
    # cell 24
    import argparse as _ap
    import json as _j
    import tempfile as _tf
    from pathlib import Path as _P

    from experiments.steering_jailbreaks import cell as _cell, steer_single as _ss

    assert st_split

    def _oob(layer):
        return ["--allow-out-of-band"] if layer in OOB else []

    def _jobs(prompt_set):
        """argv tails for one set: one cell per (axis, layer, alpha), plus a no-op per layer.

        No `ablate`: it did not help in any config at the Qwen run, so suppression is
        `add` at -alpha. The sign is derived, never typed.
        """
        out, flip = [], 1.0 if prompt_set == "success" else -1.0
        for axis, layers in CHOSEN.items():
            for layer in layers:            # story runs at both L28 and L15
                for a in ALPHAS:
                    out.append(["--direction", axis, "--mode", "add",
                                "--layers", str(layer),
                                "--alpha", f"{_cell.RESTORE_SIGN[axis] * flip * a:g}",
                                *_oob(layer)])
        # One no-op per (set, layer), not per direction: the pairing is on
        # prompt_set x layers_spec, so story@L15 and persona@L15 share the L15 no-op.
        for layer in sorted({l for ls in CHOSEN.values() for l in ls}):
            out.append(["--arm", "noop", "--layers", str(layer), *_oob(layer)])
        return out

    def _stems(script, prompt_set, job):
        """The stems this argv tail writes -- the same call cell.run makes.

        `resolve` runs here too, so an illegal sign or mode surfaces while the list is
        built rather than after a model load.
        """
        a = _ss.add_cell_args(_ap.ArgumentParser(prog="job")).parse_args(
            [MODEL, "--tag", TAG, *[str(x) for x in job]])
        direction, mode = _ss.resolve(a, prompt_set)
        return [_cell.stem_for(script, direction, mode, spec, a.alpha, a.tau_q, a.arm)
                for spec in _ss.cell_specs(a, N_LAYERS)]

    ST_JOBS, ST_ALL_STEMS, _lines = {}, [], []
    for _set, _script in (("success", "steer_single"), ("refusal", "steer_induce")):
        _all = _jobs(_set)
        _st = [_stems(_script, _set, j) for j in _all]
        ST_ALL_STEMS += [s for ss in _st for s in ss]
        _pend = [j for j, ss in zip(_all, _st) if not all(map(ST_COMPLETE, ss))]
        _p = _P(_tf.gettempdir()) / f"gemma_jobs_{_set}.json"
        _p.write_text(_j.dumps(_pend, indent=1), encoding="utf-8")
        ST_JOBS[_set] = (str(_p), len(_pend))
        _lines.append(f"**{_set}** — {len(_all)} cells, **{len(_pend)} pending**\n\n```\n"
                      + "\n".join(" ".join(j) for j in _all) + "\n```")

    mo.md(f"{len(ST_ALL_STEMS)} steering cells\n\n" + "\n\n".join(_lines))
    return ST_ALL_STEMS, ST_JOBS


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, ST_JOBS, ST_SCOPE, TAG, ckpt, sh):
    # cell 25
    # One steer_batch process per set, so the model loads at most twice rather than 40
    # times, and only the *pending* jobs are handed to it -- a set with nothing left skips
    # the process entirely and pays no load. Resume is also per cell and per batch inside
    # steer_batch, so a kill costs at most one batch.
    _path, _n = ST_JOBS["success"]
    if _n:
        _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
        try:
            # steer_batch names each cell by its stem as it enters it, and cell.emit
            # repeats it with a quarter-by-quarter row count.
            print(f"=== steer_single: {_n} pending cell(s) on the success set ===")
            sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
               "--script", "steer_single", "--jobs", _path, *BATCH)
        finally:
            _timer.set()
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="steer_single cells", **ST_SCOPE)
    else:
        print("steer_single: all cells complete -- skipped, no model load")
    st_success = True
    return (st_success,)


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, ST_JOBS, ST_SCOPE, TAG, ckpt, sh, st_success):
    # cell 26
    assert st_success
    _path, _n = ST_JOBS["refusal"]
    if _n:
        _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
        try:
            print(f"=== steer_induce: {_n} pending cell(s) on the refusal set ===")
            sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
               "--script", "steer_induce", "--jobs", _path, *BATCH)
        finally:
            _timer.set()
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="steer_induce cells", **ST_SCOPE)
    else:
        print("steer_induce: all cells complete -- skipped, no model load")
    st_steered = True
    return (st_steered,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, ST_ALL_STEMS, ST_ROOT, ST_SCOPE, TAG, ckpt, mo, st_steered):
    # cell 27
    import csv as _csv

    assert st_steered
    # Everything generated is on the Hub before this prints: the two steer cells pushed, and
    # a cell that is not up there cannot be graded anywhere else.
    _ok = ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="sweep generated", **ST_SCOPE)

    def _graded(stem):
        """True when this cell's summary exists and every row in it was scored."""
        p = ST_ROOT / "csv" / f"{stem}_summary.csv"
        if not p.exists():
            return False
        with p.open(encoding="utf-8-sig", newline="") as f:
            r = next(_csv.DictReader(f), None)
        return bool(r) and r.get("n") not in (None, "") and r["n"] == r.get("n_judged")

    _todo = [s for s in ST_ALL_STEMS if not _graded(s)]
    _slug = MODEL.replace("/", "_")
    mo.md(f"""
    ## ▸ GPU work done — judge the sweep locally

    {len(ST_ALL_STEMS) - len(_todo)}/{len(ST_ALL_STEMS)} cells are already graded;
    **{len(_todo)} are not**. {'Everything is on the Hub.' if _ok is not None else
    '**The last push was rate-limited — re-run this cell before shutting the instance down.**'}

    ~{len(_todo) * 470:,} calls at ~1.2k tokens each: `--concurrency 6` (200k TPM ≈ 165
    calls/min; 8 workers sit on the ceiling and 429), and over 10k calls it crosses
    `gpt-4o-mini`'s daily request cap, so an `OPENROUTER_API_KEY` in `.env` is what keeps it
    a one-day job — same model, same cache key, so nothing already graded is re-graded.

    On your own machine, in the repo:

    ```bash
    M={MODEL}; export RUN_TAG={TAG}
    python -c "from experiments.common import ckpt; ckpt.pull('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}')"
    for f in experiments/steering_jailbreaks/results/{TAG}/{_slug}/meta/steer_*.jsonl; do
      [ -e "$f" ] || continue
      case "$f" in *_judged.jsonl) continue;; esac
      python experiments/steering_jailbreaks/judge_strongreject.py "$f" --concurrency 6
    done
    python experiments/steering_jailbreaks/aggregate.py $M --tag {TAG}
    python -c "from experiments.common import ckpt; ckpt.push('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}', msg='sweep judged')"
    ```

    Judging is resumable per row and skips a cell whose `_summary.csv` says every row was
    scored, so re-running the loop after an interruption costs only what is left. Then read
    `aggregate_controls.csv` and come back for gate 3 — the pairs need the α curve.

    **The narrativity check (§5) is local too**, and needs no GPU at all: it runs off the
    `_judged.jsonl` files, so run it there once you have picked the story α magnitudes.
    """)
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 28 (markdown)
    mo.md(f"""
    # 5 — projection pairs (§5.6) ▸ GATE 3

    Does direction `a` still move behaviour once `b` is projected out of it **at a's own
    layer**? A pair is worth a cell only if both hold:

    - `cos(û_a, û_b)` at a's chosen layer clears the ±3/√d null band —
      `cross_probe_detection/results/{TAG}/<model_slug>/csv/geometry_cos_chosen.csv`;
    - `a` has an effect at that layer to decompose (`aggregate_controls.csv`).

    Two arms per (pair, set): **`perp_alpha`** — `(û_a − c·û_b)/√(1−c²)`, unit norm, asks
    *necessity* — and **`par_component`** — `c·û_b`, norm `|c|`, asks *sufficiency*.
    `unprojected` is **not generated**: it resolves to the `steer_single` / `steer_induce`
    cell the α sweep already produced at the same direction, layer, α and set, so the
    reference is free — which is why **α must be one of 0.25 / 0.50 / 0.75 / 1.00**.
    `perp_effect` is skipped: `α_eff = α/√(1−c²)` is under 5% for any cosine worth running.

    One line per ordered pair, `a>b` then the α magnitude per set (omit a set to skip it).
    The anchor has two layers where story does, so name one with `a@L`; without it the
    anchor's primary layer is used, and `b` is always read at the same layer as `a`:

    ```
    story_v2_1k@15>persona_v2 success=0.75 refusal=0.25
    persona_v2>eval_v2 success=0.5
    ```

    `story_v2_1k@15 × persona_v2` is the pair to expect: L15 is persona's own layer, so the
    two vectors are compared where both are deployed. Blank skips the section.
    """)
    return


@app.cell
def _(mo):
    # cell 29
    PAIRS_IN = mo.ui.text_area(
        label="pairs: `a[@layer]>b set=alpha ...`, one per line", full_width=True, rows=4,
        placeholder="story_v2_1k@15>persona_v2 success=0.75 refusal=0.25")
    PAIRS_IN
    return (PAIRS_IN,)


@app.cell
def _(ALPHAS, AXES, CHOSEN, PAIRS_IN, ST_COMPLETE, mo, st_steered):
    # cell 30
    from experiments.common import config as _cfg2, manifest as _mf
    from experiments.steering_jailbreaks import cell as _cell2, steer_pairs as _sp

    # Only the *generated* sweep is needed to build these stems; the α that picks each pair
    # comes from the local aggregate, which is why gate 3 is a text box and not a rule.
    assert st_steered
    mo.stop(not PAIRS_IN.value.strip(),
            mo.md("*Gate 3: enter the ordered pairs to run, or leave blank to skip.*"))

    PAIR_ARMS = ("perp_alpha", "par_component")

    def _parse(line):
        head, *rest = line.split()
        a, sep, b = head.partition(">")
        # `a@L` picks which of the anchor's chosen layers to project at -- story has two,
        # and the projection is same-layer, so this is not a free parameter.
        a, at, want = a.partition("@")
        if not sep or a not in AXES or b not in AXES or a == b:
            raise ValueError(f"{line!r}: expected `a>b` with two different axes from {AXES}")
        layer = int(want) if at else CHOSEN[a][0]
        if layer not in CHOSEN[a]:
            raise ValueError(f"{line!r}: {a} was steered at {CHOSEN[a]}, not L{layer} -- "
                             f"there is no `unprojected` twin at that layer")
        mags = {}
        for tok in rest:
            k, s, v = tok.partition("=")
            if s == "" or k not in ("success", "refusal"):
                raise ValueError(f"{line!r}: expected success=<alpha> / refusal=<alpha>")
            mags[k] = abs(float(v))
            if mags[k] not in ALPHAS:
                # Without a twin at that alpha there is no `unprojected` reference, and the
                # two arms are unreadable on their own.
                raise ValueError(f"{line!r}: alpha {v} has no twin; use one of {ALPHAS}")
        if not mags:
            raise ValueError(f"{line!r}: name at least one set")
        return a, b, layer, mags

    PAIRS = [_parse(l.strip()) for l in PAIRS_IN.value.splitlines() if l.strip()]

    def _alpha(pset, a, mag):
        """The signed alpha steer_pairs will compute for this (set, anchor)."""
        return _cell2.RESTORE_SIGN[a] * _sp.SET_SIGN[pset] * mag

    PAIR_JOBS, PAIR_STEMS, _rows, _warn = {"success": [], "refusal": []}, [], [], []
    for _a, _b, _layer, _mags in PAIRS:
        for _pset, _mag in _mags.items():
            _al = _alpha(_pset, _a, _mag)
            _stems = [_mf.stem("steer_pairs", f"{_a}-perp-{_b}",
                               _cfg2.layer_stem(str(_layer)), f"a{_al:g}", _arm)
                      for _arm in PAIR_ARMS]
            PAIR_STEMS += _stems
            _done = sum(map(ST_COMPLETE, _stems))
            if _done < len(PAIR_ARMS):
                PAIR_JOBS[_pset].append((_a, _b, _layer, _mag))
            # The reference the whole experiment is read against, and it is not generated
            # here; and the no-op the arms pair against, which steer_pairs does not emit.
            _twin = _cell2.stem_for(_sp.OWNER[_pset], _a, "add", str(_layer), _al, None,
                                    "target")
            if not ST_COMPLETE(_twin):
                _warn.append(f"missing `unprojected` twin for {_pset} {_a}: {_twin}")
            _noop = f"{_sp.OWNER[_pset]}__noop__L{_layer}"
            if not ST_COMPLETE(_noop):
                _warn.append(f"missing no-op for {_pset}: {_noop}")
            _rows.append(f"| `{_a}` → `{_b}` | {_pset} | L{_layer} | {_al:+g} | "
                         f"{_done}/{len(PAIR_ARMS)} |")

    mo.md(f"{len(PAIR_STEMS)} pair cells "
          f"({len(PAIRS)} ordered pairs x {len(PAIR_ARMS)} arms)\n\n"
          "| a → b | set | layer | alpha | arms done |\n|---|---|---|---|---|\n"
          + "\n".join(_rows)
          + ("\n\n**! " + "**\n\n**! ".join(dict.fromkeys(_warn)) + "**" if _warn else
             "\n\nAll `unprojected` twins and no-ops are present."))
    return PAIR_JOBS, PAIR_STEMS


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, OOB, PAIR_JOBS, ST_SCOPE, TAG, ckpt, sh):
    # cell 31
    # There is no steer_batch driver for steer_pairs, so the model loads once per pair --
    # ~1 min against ~20 min of generation per pair. Pairs whose arms are both complete are
    # dropped before that, so a resumed session pays only for what is left. Inside an
    # invocation the usual guarantees hold: resume per row, and a complete arm cache-hits.
    _jobs = PAIR_JOBS["success"]
    for _i, (_a, _b, _layer, _mag) in enumerate(_jobs, 1):
        print(f"\n=== [{_i}/{len(_jobs)}] steer_pairs success: {_a} - proj({_b}) "
              f"at L{_layer}, |alpha| {_mag:g} ===")
        sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL, "--tag", TAG,
           "--prompt-set", "success", "--pair", f"{_a},{_b}", "--layers", str(_layer),
           "--alpha", f"{_mag:g}", "--arms", "perp_alpha,par_component",
           *(["--allow-out-of-band"] if _layer in OOB else []), *BATCH)
    if _jobs:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="pairs success", **ST_SCOPE)
    else:
        print("steer_pairs success: all cells complete -- skipped, no model load")
    pairs_success = True
    return (pairs_success,)


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, OOB, PAIR_JOBS, ST_SCOPE, TAG, ckpt, pairs_success,
      sh):
    # cell 32
    assert pairs_success
    _jobs = PAIR_JOBS["refusal"]
    for _i, (_a, _b, _layer, _mag) in enumerate(_jobs, 1):
        print(f"\n=== [{_i}/{len(_jobs)}] steer_pairs refusal: {_a} - proj({_b}) "
              f"at L{_layer}, |alpha| {_mag:g} ===")
        sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL, "--tag", TAG,
           "--prompt-set", "refusal", "--pair", f"{_a},{_b}", "--layers", str(_layer),
           "--alpha", f"{_mag:g}", "--arms", "perp_alpha,par_component",
           *(["--allow-out-of-band"] if _layer in OOB else []), *BATCH)
    if _jobs:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="pairs refusal", **ST_SCOPE)
    else:
        print("steer_pairs refusal: all cells complete -- skipped, no model load")
    pairs_steered = True
    return (pairs_steered,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, PAIR_STEMS, ST_SCOPE, TAG, ckpt, mo, pairs_steered):
    # cell 33
    assert pairs_steered
    _ok = ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="pairs generated", **ST_SCOPE)
    _slug = MODEL.replace("/", "_")
    mo.md(f"""
    ## ▸ GPU work done — judge the pairs locally

    {len(PAIR_STEMS)} arm cells generated.
    {'On the Hub.' if _ok is not None else
     '**The push was rate-limited — re-run this cell before shutting the instance down.**'}
    ~{len(PAIR_STEMS) * 470:,} calls, well under a day's cap.

    ```bash
    M={MODEL}; export RUN_TAG={TAG}
    python -c "from experiments.common import ckpt; ckpt.pull('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}')"
    for f in experiments/steering_jailbreaks/results/{TAG}/{_slug}/meta/steer_pairs__*.jsonl; do
      [ -e "$f" ] || continue
      case "$f" in *_judged.jsonl) continue;; esac
      python experiments/steering_jailbreaks/judge_strongreject.py "$f" --concurrency 6
    done
    python experiments/steering_jailbreaks/aggregate.py $M --tag {TAG}
    python -m experiments.common.check_stale $M {TAG}
    python -c "from experiments.common import ckpt; ckpt.push('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}', msg='pairs judged')"
    ```

    `aggregate.py` spans the tag, so it re-reports the α sweep alongside the pairs.
    `check_stale` will flag the jailbreak activations `--view-only` deliberately left
    uncomputed — expected, not a finding.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # cell 34 (markdown)
    mo.md(r"""
    ## Read the pairs

    1. **`pct_degenerate` first**, especially on `par_component`: it pushes an axis at a
       magnitude the anchor's own α ladder never tested.
    2. **Each arm against its `unprojected` twin**, not against the no-op directly — the
       twin is the reference the projection is defined relative to. `steer_pairs` prints the
       stem it resolved to, and `cos_ab_band` / `push_frac` in each manifest say what was
       removed and at what magnitude.
    3. **`perp_alpha` ≈ reference ⇒ `b` is not necessary. `par_component` ≈ reference ⇒
       `b`'s share is sufficient.** Not an algebraic split: both can hold, or neither.
    4. **`read_<axis>` for all four axes.** On `perp_alpha` the pushed vector has *zero* `b`
       content by construction — if `read_b` still moves, the contamination was never
       geometric and no same-layer projection could remove it.

    Still **no `random` arm** at this tag, so nothing here is a specificity claim.
    """)
    return


if __name__ == "__main__":
    app.run()
