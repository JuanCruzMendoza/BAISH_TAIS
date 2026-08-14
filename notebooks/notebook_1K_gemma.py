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

    Spec: `research/spec-whole-rerun.md`. The whole pipeline for a second model, one
    notebook: extraction → probe_jailbreak_detection → cross_probe_detection → steering
    (baseline + α sweep) → narrativity → projection pairs.

    **Same tag, different model slug**, so nothing here can collide with the Qwen run.
    Dropped from it: `ablate`, `cap`, the `length` foil, the `random` arm, the decoding
    comparison (greedy, reused) and the second-layer pass. **α = 0.25 / 0.50 / 0.75 / 1.00**
    for every direction, sign derived from `cell.RESTORE_SIGN`.

    ## Run all cells

    Every stage is guarded on an artefact, not on a checkbox: what is already complete is
    skipped **before** the model load, so re-running the notebook after a kill is the way to
    resume, and a judge-only session needs no GPU at all.

    Three points stop and wait for you — the notebook runs everything above them and halts:

    | gate | what to read first | what to type |
    |---|---|---|
    | **1 — layers** | `probe_select__<axis>.csv` (`cohens_dz_train`) and `jb_metrics__*__all_rate.csv` (fiction − nonfiction `pct_reads`, for story) | one layer per direction |
    | **2 — narrativity α** | `aggregate_controls.csv`: story cells with an effect and low `pct_degenerate` | the α magnitudes to judge |
    | **3 — pairs** | `geometry_cos_chosen.csv` (cos at a's layer vs the ±3/√d null band) + the α curve | the ordered pairs and their α |

    Nothing below a gate runs until it is filled in. Everything above it is already on the
    Hub by then.

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
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write, and gemma-2 licence accepted)",
                          kind="password", full_width=True)
    OPENAI_KEY = mo.ui.text(label="OPENAI_API_KEY (judge, spec 5.3)", kind="password",
                            full_width=True)
    # Optional but close to necessary here: ~19k judge calls do not fit under a 10,000/day
    # cap, so when the OpenAI key is spent the judge moves to OpenRouter and keeps going.
    # Same model, so nothing already graded is re-graded.
    OPENROUTER_KEY = mo.ui.text(label="OPENROUTER_API_KEY (fallback judge, optional)",
                                kind="password", full_width=True)
    mo.vstack([HF_TOKEN, OPENAI_KEY, OPENROUTER_KEY])
    return HF_TOKEN, OPENAI_KEY, OPENROUTER_KEY


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
    | `cross_probe_detection` | no | once, ~10 small files |
    | `steering_jailbreaks` | no | per generation stage, then **every 5 cells** while judging |

    **The extraction scope is never pushed after the jailbreak activations are cached.**
    Those 1,009 blobs land in *extraction's* tree, and a packed push would re-tar them into
    the 2.5 GB cache for activations nothing downstream reads.

    The hourly timer is armed **only around a stage that is actually generating** and
    stopped when it returns, so an idle session does not spend commits re-uploading an
    unchanged tree. The judge pass does not use it: it pushes every `JUDGE_PUSH_EVERY`
    cells instead, because graded rows cost real money and daily quota and should not sit
    on local disk for an hour.

    The push is staged — the scoped tree is mirrored (hard links) and the append-only
    `.jsonl` files snapshotted and trimmed to their last complete row — so no file is
    measured while a subprocess is still appending to it.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, OPENAI_KEY, OPENROUTER_KEY, REPO, TAG, mo, os):
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
    # Set only when non-empty: an empty string reads as *present* downstream, so a blank
    # box would arm a fallback that cannot authenticate.
    for _var, _val in (("OPENAI_API_KEY", OPENAI_KEY.value),
                       ("OPENROUTER_API_KEY", OPENROUTER_KEY.value)):
        if _val:
            os.environ[_var] = _val
        else:
            os.environ.pop(_var, None)
    keys_ok = bool(OPENAI_KEY.value or OPENROUTER_KEY.value)

    AXES = ["story_v2_1k", "persona_v2", "harm_v2", "eval_v2"]
    ALPHAS = (0.25, 0.50, 0.75, 1.00)
    RULES = ("midpoint", "gap_mid")
    # The judge's daily request cap, measured on gpt-4o-mini. Tier-dependent, and it only
    # sizes the estimate printed before a judge pass -- the stop itself is the provider's
    # own 429, since this process cannot know what the account already spent today.
    JUDGE_RPD = 10_000
    # Push the judge pass every this many cells: at ~4 min a cell that caps what a
    # disconnection can cost at ~20 min of grading, for ~2 commits a push.
    JUDGE_PUSH_EVERY = 5

    EX_SCOPE = {"experiment": "extraction", "tag": TAG}
    JB_SCOPE = {"experiment": "probe_jailbreak_detection", "tag": TAG}
    CP_SCOPE = {"experiment": "cross_probe_detection", "tag": TAG}
    ST_SCOPE = {"experiment": "steering_jailbreaks", "tag": TAG}

    def _root(experiment):
        return _P(REPO, "experiments", experiment, "results", TAG,
                  MODEL.replace("/", "_"))

    EX_ROOT, JB_ROOT = _root("extraction"), _root("probe_jailbreak_detection")
    CP_ROOT, ST_ROOT = _root("cross_probe_detection"), _root("steering_jailbreaks")

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
    print("judge keys:", ", ".join(k for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")
                                   if os.environ.get(k)) or "none")
    return (ALPHAS, AXES, BAND, CP_ROOT, CP_SCOPE, DONE, EX_ROOT, EX_SCOPE, JB_ROOT,
            JB_SCOPE, JUDGE_PUSH_EVERY, JUDGE_RPD, N_LAYERS, RULES, ST_COMPLETE, ST_ROOT,
            ST_SCOPE, cfg, ckpt, keys_ok)


@app.cell
def _(AXES, CP_ROOT, CP_SCOPE, DONE, EX_ROOT, EX_SCOPE, HF_REPO, HF_TOKEN, JB_ROOT,
      JB_SCOPE, RULES, ST_SCOPE, ckpt):
    # cell 6
    # Small trees first, so the pending checks below can be made without the pole cache.
    # `subpaths` is glob-relative to <model_slug>'s parent.
    print("extraction (small):",
          ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                    subpaths=["*/vectors/**", "*/meta/**", "*/csv/**", "*/figures/**"]))
    print("probe_jailbreak_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **JB_SCOPE))
    print("cross_probe_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **CP_SCOPE))
    print("steering_jailbreaks:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))

    # Who actually reads activations: extraction itself (to finish caching), jb_readout (it
    # re-projects onto all 8,000 pole prompts), and cross_auroc / geometry (pooled AUROC is
    # computed on the cached activations, not on the vectors). A steering- or judge-only
    # session reads none of them, and skipping the 2.5 GB is most of its startup.
    EX_CACHED = all(DONE(EX_ROOT, f"cache_activations__{_d}__{_s}")
                    for _d in AXES for _s in ("train", "heldout"))
    _jb_done = (DONE(JB_ROOT, "jb_readout")
                and all(DONE(JB_ROOT, f"jb_metrics__{_r}__all") for _r in RULES))
    if not (EX_CACHED and _jb_done
            and DONE(CP_ROOT, "cross_auroc") and DONE(CP_ROOT, "geometry")):
        # pack=True unpacks acts/blobs.tar into acts/blobs/, so the scripts see the same
        # tree either way and run_key resume still cache-hits. A no-op with no tar there.
        print("extraction acts:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                                            subpaths=["*/acts/**"], pack=True))
    else:
        print("extraction acts: not needed by anything still pending -- skipped (2.5 GB)")
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
        # pack=True to match the pull: pushing the blobs loose would leave both
        # representations on the Hub and they would drift apart on the next run.
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
    ## ▸ GATE 1 — one layer per direction

    Everything above is on the Hub. Pull it locally and read, per direction:

    - `extraction/results/{TAG}/<model_slug>/csv/probe_select__<axis>.csv` —
      **`cohens_dz_train`**, confirmed against `cohens_dz_heldout` (200 pairs),
      `mean_paired_cos` and `lopo_ci_lo`. This is the default criterion.
    - `probe_jailbreak_detection/.../csv/jb_metrics__midpoint__all_rate.csv` — for
      **`story_v2_1k`** prefer the layer with the largest **fiction − nonfiction
      `pct_reads`** gap where it disagrees with `cohens_dz`: that is the layer that
      discriminates the jailbreak families, and 50_per_direction measured r = 0.00 between
      probe quality and steering effect. Read `ref_tpr` beside it — a low `pct_reads` at low
      `ref_tpr` is the threshold failing, not a reading.

    The reporting band is **L{BAND[0]}–L{BAND[-1]}**. A layer outside it is allowed and gets
    `--allow-out-of-band` automatically, recorded in every manifest that uses it.

    Record the four in `extraction/insights.md` as well — experiments 2–5 read them from
    there, not from any JSON.
    """)
    return


@app.cell
def _(mo):
    # cell 16
    CHOSEN_IN = mo.ui.text(
        label="chosen layers (axis=layer, comma-separated)", full_width=True,
        placeholder="story_v2_1k=23,persona_v2=15,harm_v2=21,eval_v2=9")
    CHOSEN_IN
    return (CHOSEN_IN,)


@app.cell
def _(AXES, BAND, CHOSEN_IN, N_LAYERS, cfg, jb_all, mo):
    # cell 17
    assert jb_all
    mo.stop(not CHOSEN_IN.value.strip(),
            mo.md("*Gate 1: read the tables above and enter one layer per direction. "
                  "Nothing below this runs until then.*"))
    CHOSEN = cfg.parse_axis_layers(CHOSEN_IN.value)
    _missing, _bad = set(AXES) - set(CHOSEN), {a: l for a, l in CHOSEN.items()
                                               if not 0 <= l <= N_LAYERS}
    if _missing or _bad or set(CHOSEN) - set(AXES):
        raise ValueError(f"name exactly {AXES} with 0 <= layer <= {N_LAYERS}; "
                         f"missing {sorted(_missing)}, "
                         f"unknown {sorted(set(CHOSEN) - set(AXES))}, out of range {_bad}")
    OOB = {a: l for a, l in CHOSEN.items() if l not in BAND}
    LAYER_ARG = ",".join(f"{a}={l}" for a, l in CHOSEN.items())
    print(f"chosen: {LAYER_ARG}")
    print(f"out of band (L{BAND[0]}-{BAND[-1]}), explicit opt-in: {OOB or 'none'}")
    return CHOSEN, LAYER_ARG, OOB


@app.cell
def _(AXES, CHOSEN, DONE, HF_REPO, HF_TOKEN, JB_ROOT, JB_SCOPE, LAYER_ARG, MODEL, RULES,
      TAG, ckpt, sh):
    # cell 18
    # The headline files: one row per probe x slice at that probe's own layer. Re-runs
    # whenever the gate changes, because `probe_layers` is part of what DONE compares.
    _todo = [r for r in RULES if not DONE(JB_ROOT, f"jb_metrics__{r}", probe_layers=CHOSEN)]
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
def _(mo):
    # cell 19 (markdown)
    mo.md(r"""
    # 3 — cross_probe_detection (H1)

    4×4 at the chosen layers, no GPU. `--diag heldout`: the deployed vector on the 200
    held-out pairs, since at 800 train pairs LOPO moves `d_z` by ~0.005. `cohens_dz` is
    emitted because AUROC saturates at n=1,000 and cannot rank cells.

    No positive control exists at this tag — `story_v1` is a 50-pair arm — so a matrix of
    nulls is not self-validating here.
    """)
    return


@app.cell
def _(AXES, CHOSEN, CP_ROOT, CP_SCOPE, DONE, HF_REPO, HF_TOKEN, LAYER_ARG, MODEL, TAG,
      ckpt, jb_chosen, sh):
    # cell 20
    assert jb_chosen
    _run = False
    if not DONE(CP_ROOT, "cross_auroc", probe_layers=CHOSEN):
        print(f"\n=== cross_auroc at {LAYER_ARG} ===")
        sh("python", "experiments/cross_probe_detection/cross_auroc.py", MODEL, "--tag", TAG,
           "--axes", ",".join(AXES), "--layers", LAYER_ARG, "--diag", "heldout")
        _run = True
    if not DONE(CP_ROOT, "geometry", chosen_layers=CHOSEN):
        print(f"\n=== geometry at {LAYER_ARG} ===")
        sh("python", "experiments/cross_probe_detection/geometry.py", MODEL, "--tag", TAG,
           "--axes", ",".join(AXES), "--layers", LAYER_ARG)
        _run = True
    if _run:
        print("\n=== plot_matrices ===")
        sh("python", "experiments/cross_probe_detection/plot_matrices.py", MODEL, "--tag", TAG)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="cross_probe_detection", **CP_SCOPE)
    else:
        print("cross_auroc and geometry current for these layers -- skipped")
    cp_done = True
    return (cp_done,)


@app.cell(hide_code=True)
def _(mo):
    # cell 21 (markdown)
    mo.md(r"""
    # 4 — steering_jailbreaks (H3)

    Greedy over all 1,009 prompts for the baseline; judging it defines the two prompt sets
    (§5.4 runs on the rows it complied with, §5.5 on the rows it refused, a degenerate row
    falls in neither). Sizes are not known until it is judged.

    Then **`add` only, one chosen layer per direction, α ∈ 0.25 / 0.50 / 0.75 / 1.00**,
    signed: restoring is `RESTORE_SIGN` (−1 for the framing axes, +1 for the refusal ones)
    on the success set and its mirror on the refusal set. `steer_single.resolve` refuses the
    half with no headroom, so a sign slip fails before the model loads. **32 target cells +
    one no-op per (set, layer)**.

    The baseline does not depend on gate 1, so it runs while you are still choosing layers.
    """)
    return


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, ST_COMPLETE, ST_ROOT, ST_SCOPE, TAG, ckpt, jb_cached,
      keys_ok, mo, sh):
    # cell 22
    assert jb_cached                      # that cell writes the view sets.py reads
    mo.stop(not keys_ok,
            mo.md("*Paste a judge key (OpenAI and/or OpenRouter). The baseline defines both "
                  "prompt sets, and it does that only once it is graded.*"))
    _jsonl = ST_ROOT / "meta" / "gen_baseline.jsonl"
    if ST_COMPLETE("gen_baseline") and (ST_ROOT / "meta" / "gen_baseline_judged.jsonl").exists():
        # gen_baseline calls mdl.load() before it reads its own resume, so re-entering a
        # finished baseline still costs the weights.
        print("baseline present and judged -- skipped, no model load")
    else:
        if not ST_COMPLETE("gen_baseline"):
            _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
            try:
                print("=== gen_baseline: unsteered greedy over all 1,009 prompts ===")
                sh("python", "experiments/steering_jailbreaks/gen_baseline.py", MODEL,
                   "--tag", TAG, "--split", "all", "--decoding", "greedy", *BATCH)
            finally:
                _timer.set()
        print("\n=== judging gen_baseline -- this is what defines the two prompt sets ===")
        sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
           str(_jsonl), "--concurrency", "6")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="gen_baseline", **ST_SCOPE)
    st_split = True
    return (st_split,)


@app.cell
def _(ALPHAS, CHOSEN, MODEL, N_LAYERS, OOB, ST_COMPLETE, TAG, mo, st_split):
    # cell 23
    import argparse as _ap
    import json as _j
    import tempfile as _tf
    from pathlib import Path as _P

    from experiments.steering_jailbreaks import cell as _cell, steer_single as _ss

    assert st_split

    def _oob(layer):
        return ["--allow-out-of-band"] if layer in OOB.values() else []

    def _jobs(prompt_set):
        """argv tails for one set: 4 axes x 4 alphas, plus a no-op per layer.

        No `ablate`: it did not help in any config at the Qwen run, so suppression is
        `add` at -alpha. The sign is derived, never typed.
        """
        out, flip = [], 1.0 if prompt_set == "success" else -1.0
        for axis, layer in CHOSEN.items():
            for a in ALPHAS:
                out.append(["--direction", axis, "--mode", "add", "--layers", str(layer),
                            "--alpha", f"{_cell.RESTORE_SIGN[axis] * flip * a:g}",
                            *_oob(layer)])
        # One no-op per (set, layer) -- two directions sharing a layer share its no-op,
        # since the pairing is on prompt_set x layers_spec, not on the direction.
        for layer in sorted(set(CHOSEN.values())):
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
    # cell 24
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
    # cell 25
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


@app.cell(hide_code=True)
def _(mo):
    # cell 26 (markdown)
    mo.md(r"""
    ## Judge — API, and two rate limits

    One call per **ungraded** row. Two ceilings, needing different handling:

    | limit | measured | response |
    |---|---|---|
    | tokens per minute | 200k TPM on `gpt-4o-mini`, ~1.2k tokens a call ≈ 165 calls/min | `--concurrency 6`; 8 workers sit on the ceiling and 429 |
    | **requests per day** | **10,000 RPD** | not retryable — with an OpenRouter key the judge switches provider and continues on the *same* `gpt-4o-mini`; without one it exits **3** and the loop stops cleanly |

    Nothing already graded is re-graded: the cache key and the per-row resume stamp are the
    model, not the endpoint, so a row graded through either provider is a valid cache hit
    for the other. A cell is skipped when its `csv/<stem>_summary.csv` says every row was
    scored, so a cell killed midway re-enters and resumes per row.

    Exit 3 breaks the loop; a single other failure is one bad cell and the loop carries on;
    two in a row is a wall and also stops. Pushed every `JUDGE_PUSH_EVERY` cells.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, JUDGE_PUSH_EVERY, JUDGE_RPD, ST_ROOT, ST_SCOPE, ckpt, sh):
    # cell 27
    import csv as _csv
    import json as _js
    import os as _os

    _META, _CSV = ST_ROOT / "meta", ST_ROOT / "csv"

    def _graded(stem):
        """True when this cell's summary exists and every row in it was scored."""
        p = _CSV / f"{stem}_summary.csv"
        if not p.exists():
            return False
        with p.open(encoding="utf-8-sig", newline="") as f:
            r = next(_csv.DictReader(f), None)
        return bool(r) and r.get("n") not in (None, "") and r["n"] == r.get("n_judged")

    def _ids(path):
        """Distinct unit_ids in a jsonl, torn tail discarded (spec 0.11)."""
        out = set()
        if not path.exists():
            return out
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.add(_js.loads(line)["unit_id"])
            except (ValueError, KeyError):
                break
        return out

    def JUDGE(label, mine):
        """Grade every generated-but-ungraded cell. True when the pass finished.

        A generations file is one with a sibling manifest -- exactly what
        judge_strongreject loads first. meta/ also holds judge_cache.jsonl (the judge's own
        response cache) and the _judged.jsonl outputs; neither has a manifest, and a
        blocklist naming only those two picked up the cache and died on it. gen_baseline is
        judged in its own cell.
        """
        mine = set(mine)
        todo = sorted(p for p in _META.glob("*.jsonl")
                      if not p.name.endswith("_judged.jsonl")
                      and not p.name.startswith("gen_baseline")
                      and (_META / f"{p.stem}_manifest.json").exists()
                      and not _graded(p.stem))
        left = {p.stem: len(_ids(p) - _ids(_META / f"{p.stem}_judged.jsonl")) for p in todo}
        need = sum(left.values())
        new = sum(p.stem in mine for p in todo)
        print(f"[{label}] judging {len(todo)} cells, {new} of them this stage's "
              f"({len(mine) - new}/{len(mine)} already graded)\n"
              f"{need:,} rows still ungraded, against a {JUDGE_RPD:,}/day request cap "
              f"-> {-(-need // JUDGE_RPD)} day(s)")
        if need > JUDGE_RPD and not _os.environ.get("OPENROUTER_API_KEY"):
            print(f"! this pass CANNOT finish today on the OpenAI key alone. It will grade "
                  f"~{JUDGE_RPD:,} rows, stop on the daily cap, and resume here tomorrow -- "
                  f"re-run the notebook, or paste an OpenRouter key to carry straight on.")

        # allow_fail because judging is resumable per row, so one bad cell costs a retry of
        # that cell rather than the rest of the pass. Exit 3 is the *daily* cap, which every
        # remaining cell would hit in turn, so it breaks out. The belt to that brace: exit 3
        # rests on a regex over the error string, so two consecutive failures of ANY kind
        # also stop -- one bad cell is a bad cell, two in a row is a wall.
        failed, capped, done_rows, streak, since = [], False, 0, 0, 0
        for i, p in enumerate(todo, 1):
            # One line per cell, and it is the only per-cell progress there is: the
            # judge's own per-row counter is terminal-only (cfg.LIVE), so a 500-row cell
            # prints this line and its summary, not 500 numbered lines.
            print(f"\n=== [{i}/{len(todo)}] judging {p.stem}  "
                  f"({left[p.stem]:,} rows ungraded) ===")
            rc = sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
                    str(p), "--concurrency", "6", allow_fail=True).returncode
            # Before the break checks: a cell that hit the cap still graded rows on the way.
            since += 1
            if rc == 3:
                capped = True
                break
            if rc:
                failed.append(p.stem)
                streak += 1
                if streak == 2:
                    print(f"\n! two cells failed back to back ({failed[-2:]}) -- stopping. "
                          f"That is a wall, not a bad cell: check the last traceback for a "
                          f"quota, key or network problem before re-running.")
                    break
            else:
                done_rows += left[p.stem]
                streak = 0
            if since >= JUDGE_PUSH_EVERY:
                ckpt.try_push(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE,
                              msg=f"{label} judged, {i}/{len(todo)} cells")
                since = 0
        # Whatever the last partial batch left, including the cells a break stopped after.
        # Nothing to push when the loop never ran, so a verification re-run does not spend
        # commits walking an unchanged tree.
        if since:
            ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg=f"{label} judged", **ST_SCOPE)
        if failed:
            print(f"! {len(failed)} cells did not judge, re-run this cell: {failed}")
        if capped:
            also = (" Both the OpenAI key and the OpenRouter fallback are spent."
                    if _os.environ.get("OPENROUTER_API_KEY") else
                    " Paste an OpenRouter key to carry on without waiting.")
            print(f"\n! DAILY REQUEST CAP -- stopped cleanly after ~{done_rows:,} rows, "
                  f"~{need - done_rows:,} to go.{also} Everything graded is on the Hub; "
                  f"re-run the notebook and the GPU cells will all skip.")
        return not capped and not failed
    return (JUDGE,)


@app.cell
def _(JUDGE, ST_ALL_STEMS, st_steered):
    # cell 28
    assert st_steered
    st_judged = JUDGE("steering", ST_ALL_STEMS)
    return (st_judged,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, ST_SCOPE, TAG, ckpt, sh, st_judged):
    # cell 29
    # An aggregate over a half-graded run is a table that looks final and is not, so it is
    # not written at all until every cell is scored.
    if not st_judged:
        print("judging did not finish -- aggregate SKIPPED. Re-run the notebook; every GPU "
              "cell will skip and judging resumes.")
    else:
        print("=== aggregate over every cell at this tag ===")
        sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="aggregate", **ST_SCOPE)
    st_agg = True
    return (st_agg,)


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 30 (markdown)
    mo.md(f"""
    ## Read the α sweep, then ▸ GATE 2

    `steering_jailbreaks/results/{TAG}/<model_slug>/csv/`: `aggregate_cells.csv` is every
    cell, `aggregate_controls.csv` each target with `d_*_vs_noop`.

    Read in this order, or the numbers mislead:

    1. **`pct_degenerate` before any effect.** A mostly-broken cell still has a ΔASR and it
       means nothing. An out-of-band layer and the top of the α ladder are where this bites.
    2. **`d_*_vs_noop`, matched on `prompt_set` × `layers_spec`** — never against the
       baseline, whose batch composition differs.
    3. **`|Δh|`, not α.** The push is `α·σ_l·û_l` and σ differs per layer and direction.
    4. **`read_<axis>` for all four axes.** At the Qwen run no cell moved only its own axis.

    **Gate 2** — the narrativity check is the manipulation check on the output side: is the
    steered response the more narrative of the pair, judged against its own no-op on the
    same row? Enter the `story_v2_1k` α **magnitudes** worth judging: the cells with a
    readable effect and low `pct_degenerate`. Blank skips the section.
    """)
    return


@app.cell
def _(mo):
    # cell 31
    NARR_ALPHAS = mo.ui.text(label="narrativity: story alpha magnitudes", full_width=True,
                             placeholder="0.25,0.75")
    NARR_ALPHAS
    return (NARR_ALPHAS,)


@app.cell
def _(CHOSEN, DONE, HF_REPO, HF_TOKEN, MODEL, NARR_ALPHAS, ST_ROOT, ST_SCOPE, TAG, ckpt,
      mo, os, sh, st_agg):
    # cell 32
    assert st_agg
    mo.stop(not NARR_ALPHAS.value.strip(),
            mo.md("*Gate 2: enter the story α magnitudes to judge, or leave blank to skip.*"))
    _mags = [float(x) for x in NARR_ALPHAS.value.replace(" ", "").split(",") if x]
    _layer = CHOSEN["story_v2_1k"]
    _stem = f"judge_narrativity__story_v2_1k__L{_layer}"
    # Forced onto OpenRouter when a key is there: the same model and the same cache key, so
    # it is not a different judge -- it just keeps the OpenAI daily cap for the
    # StrongREJECT pass, which is 20x larger.
    _prov = ["--provider", "openrouter"] if os.environ.get("OPENROUTER_API_KEY") else []
    if DONE(ST_ROOT, _stem, alpha_mags=_mags, prompt_sets=["success", "refusal"]):
        print(f"narrativity at L{_layer} for alphas {_mags} complete -- skipped")
    else:
        # Runs off the existing _judged.jsonl -- no generation. Each steered cell against
        # its own no-op at the same layer and set; pairs where either side is degenerate are
        # excluded, because a repetition loop reads as more literary.
        print(f"=== narrativity: story_v2_1k at L{_layer}, alphas {_mags}, both sets ===")
        sh("python", "experiments/steering_jailbreaks/judge_narrativity.py", MODEL,
           "--tag", TAG, "--direction", "story_v2_1k", "--layer", str(_layer),
           "--alphas", ",".join(f"{m:g}" for m in _mags), "--concurrency", "8", *_prov)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="narrativity", **ST_SCOPE)
    narr_done = True
    return (narr_done,)


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 33 (markdown)
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

    One line per ordered pair, `a>b` then the α magnitude per set (omit a set to skip it):

    ```
    story_v2_1k>persona_v2 success=0.75 refusal=0.25
    persona_v2>eval_v2 success=0.5
    ```

    Blank skips the section.
    """)
    return


@app.cell
def _(mo):
    # cell 34
    PAIRS_IN = mo.ui.text_area(
        label="pairs: `a>b set=alpha ...`, one per line", full_width=True, rows=4,
        placeholder="story_v2_1k>persona_v2 success=0.75 refusal=0.25")
    PAIRS_IN
    return (PAIRS_IN,)


@app.cell
def _(ALPHAS, AXES, CHOSEN, PAIRS_IN, ST_COMPLETE, mo, st_agg):
    # cell 35
    from experiments.common import config as _cfg2, manifest as _mf
    from experiments.steering_jailbreaks import cell as _cell2, steer_pairs as _sp

    assert st_agg
    mo.stop(not PAIRS_IN.value.strip(),
            mo.md("*Gate 3: enter the ordered pairs to run, or leave blank to skip.*"))

    PAIR_ARMS = ("perp_alpha", "par_component")

    def _parse(line):
        head, *rest = line.split()
        a, sep, b = head.partition(">")
        if not sep or a not in AXES or b not in AXES or a == b:
            raise ValueError(f"{line!r}: expected `a>b` with two different axes from {AXES}")
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
        return a, b, mags

    PAIRS = [_parse(l.strip()) for l in PAIRS_IN.value.splitlines() if l.strip()]

    def _alpha(pset, a, mag):
        """The signed alpha steer_pairs will compute for this (set, anchor)."""
        return _cell2.RESTORE_SIGN[a] * _sp.SET_SIGN[pset] * mag

    PAIR_JOBS, PAIR_STEMS, _rows, _warn = {"success": [], "refusal": []}, [], [], []
    for _a, _b, _mags in PAIRS:
        _layer = CHOSEN[_a]
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
    # cell 36
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
           *(["--allow-out-of-band"] if _layer in OOB.values() else []), *BATCH)
    if _jobs:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="pairs success", **ST_SCOPE)
    else:
        print("steer_pairs success: all cells complete -- skipped, no model load")
    pairs_success = True
    return (pairs_success,)


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, OOB, PAIR_JOBS, ST_SCOPE, TAG, ckpt, pairs_success,
      sh):
    # cell 37
    assert pairs_success
    _jobs = PAIR_JOBS["refusal"]
    for _i, (_a, _b, _layer, _mag) in enumerate(_jobs, 1):
        print(f"\n=== [{_i}/{len(_jobs)}] steer_pairs refusal: {_a} - proj({_b}) "
              f"at L{_layer}, |alpha| {_mag:g} ===")
        sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL, "--tag", TAG,
           "--prompt-set", "refusal", "--pair", f"{_a},{_b}", "--layers", str(_layer),
           "--alpha", f"{_mag:g}", "--arms", "perp_alpha,par_component",
           *(["--allow-out-of-band"] if _layer in OOB.values() else []), *BATCH)
    if _jobs:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="pairs refusal", **ST_SCOPE)
    else:
        print("steer_pairs refusal: all cells complete -- skipped, no model load")
    pairs_steered = True
    return (pairs_steered,)


@app.cell
def _(HF_REPO, HF_TOKEN, JUDGE, MODEL, PAIR_STEMS, ST_SCOPE, TAG, ckpt, pairs_steered, sh):
    # cell 38
    assert pairs_steered
    _ok = JUDGE("pairs", PAIR_STEMS)
    if not _ok:
        print("judging did not finish -- aggregate SKIPPED. Re-run the notebook.")
    else:
        # aggregate.py spans the tag, so this re-reports the alpha sweep alongside the pairs.
        print("=== aggregate: the alpha sweep and the pairs together ===")
        sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="pairs aggregate", **ST_SCOPE)
        # check_stale spans the whole tag, so it also reports the jailbreak activations that
        # --view-only deliberately leaves uncomputed. Never fatal for that reason.
        sh("python", "-m", "experiments.common.check_stale", MODEL, TAG, allow_fail=True)
    pairs_done = _ok
    return (pairs_done,)


@app.cell(hide_code=True)
def _(mo, pairs_done):
    # cell 39 (markdown)
    mo.md(f"""
    ## Read the pairs — {'complete' if pairs_done else 'not finished'}

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
