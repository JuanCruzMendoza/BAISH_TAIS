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
        # refuses, and the clone silently strands on an old commit. Discarding them is
        # free: the Hub is authoritative and cell 5 re-pulls. Not `reset --hard`, so
        # untracked and ignored files -- acts/, vectors/, the hf cache -- survive.
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
    # {TAG} — steering_jailbreaks, **2_run**

    Spec: `experiments/steering_jailbreaks/dev.md`, *{TAG} / 2_run*. **40 cells**, and
    nothing else: extraction and `probe_jailbreak_detection` are done and are only *read*
    here.

    Two questions 1_run left open:

    1. **Is α saturated?** All four 1_run layers — `harm_v2` L21, `eval_v2` L9,
       `story_v2_1k` L23, `persona_v2` L15 — get **α = 1.00 and 1.25** on top of 1_run's
       0.25 / 0.50 / 0.75, on both sets.
    2. **Is the `cohens_dz` layer the right *steering* layer?** `story_v2_1k` is re-run at
       **L15** and `persona_v2` at **L4** — their best jailbreak-*detection* layers — over
       the full grid 0.25 / 0.50 / 0.75 / 1.00 / 1.25 plus `ablate`.

    21 cells on the successes, 19 on the refusals: **18,895 generations**, ≈3.0 h GPU and
    ≈2.6 h of judging.

    ## Resume

    Every cell below is guarded on an artefact, not on a checkbox: a finished steering cell
    is skipped by its manifest, a graded cell by its `_summary.csv`. **Re-running the whole
    notebook after a kill is the intended way to resume** — and when a whole stage is
    already done it is skipped *before* the model load, so a judge-only session needs no GPU
    at all.

    ## Keys

    Paste both. Nothing downstream runs until they are set, which is also what stops a
    session opened for something else from starting a 4-hour sweep: marimo runs every cell
    on load, and a password field is empty on a fresh load.
    """)
    return


@app.cell
def _(mo):
    # cell 3
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    OPENAI_KEY = mo.ui.text(label="OPENAI_API_KEY (judge, spec 5.3)", kind="password",
                            full_width=True)
    mo.vstack([HF_TOKEN, OPENAI_KEY])
    return HF_TOKEN, OPENAI_KEY


@app.cell(hide_code=True)
def _(mo):
    # cell 4
    mo.md(r"""
    ## Pull — only what 2_run needs

    Two scopes, both narrow.

    **extraction**: `vectors/` and `meta/directions__*` only — the four direction files and
    the manifests they are validated against, ~8 MB. **Not `acts/`**: 1_run's notebook
    needed the 1.6 GB pole cache because `jb_readout.py` re-projects onto it, and nothing
    here does. That also removes the read-quota 429 that `HF_HUB_DISABLE_XET=1` was meant
    to dodge there.

    **The push is staged, and that is not cosmetic.** `upload_folder` stats a file and then
    uploads it, so a row landing in between commits a size that does not describe the
    content — and the strict download path then refuses that file from every machine,
    permanently. It happened on 2_run before the fix:
    `steer_induce__persona_v2__add__L15__a1.jsonl` went up recorded at 238,254 bytes with
    335,267 stored, byte 238,254 landing *mid-row*, because the hourly timer read it while
    `steer_batch` was appending. `push`'s lock serialises pushes against each other and did
    nothing about a subprocess writing underneath one. A fresh clone reproduced it exactly:
    the damage is server-side.

    `push` now mirrors the scoped tree into a staging directory first and uploads from
    there, so every file has stopped changing before it is measured. The `.jsonl` files are
    snapshotted and trimmed to their last complete row — a mid-row cut costs nothing, since
    spec 0.11's resume reads what is there and regenerates the rest. Everything else is
    hard-linked, so the mirror is directory entries rather than bytes.

    `pull` covers the damage already on the Hub: it clears local `.incomplete` partials,
    retries, and if a mismatch survives that, fetches file by file and **skips** the
    unfetchable ones, naming them. `ST_COMPLETE` requires the `.jsonl` beside the manifest,
    so a skipped cell regenerates rather than being silently counted as done.

    It does **not** try `HF_HUB_DISABLE_XET`. Measured against this very file on hub 1.7.1:
    Xet fetched all 335,267 bytes and the *plain* path raised `Consistency check failed`,
    since it compares against the size the HEAD reports. Xet is the lenient path on some
    versions and the strict one on others, so disabling it can only lose a download that
    would have worked.

    **steering_jailbreaks**: everything. The judged baseline defines both prompt sets and
    must be 1_run's exact split; 1_run's 36 manifests are what tell the cells below what is
    already done; and `judge_cache.jsonl` makes a re-judge free.

    `pack=False` on both sides, as in 1_run: steering's resume partials have to stay
    individually addressable, and a packed push re-sends the whole tar on every tick.

    ## Commits — 2_run crosses the batch boundary

    Commits are the scarce resource, not bytes, and `upload_folder` emits **one commit per
    256 files considered** — considered, not changed, so an unchanged tree still costs the
    full walk. Steering keeps 4 files per cell (`.jsonl`, `_manifest.json`,
    `_judged.jsonl`, `_summary.csv`):

    | | cells | files in scope | commits per push |
    |---|---|---|---|
    | 1_run | 36 | 154 | **1** |
    | 1_run + 2_run | 76 | 314 | **2** |

    So every push here costs twice what the same push cost in 1_run, and the crossover
    happens *during* this run. Four manual pushes plus ~6 hourly ticks over a ~5.6 h session
    ≈ **18–20 commits** — comfortable, but the split has a consequence: a 429 landing between
    part 1 and part 2 can put a manifest on the Hub without its `.jsonl`, because the batching
    is by file, not by stem. `ST_COMPLETE` therefore requires both, or a torn push would make
    a cell look finished and silently drop it from the judge pass and from `aggregate`.

    `meta/_archive/*` is in `IGNORE`, so re-running a cell does not grow the walk.

    **Xet is on, and it was on in 1_run too** — despite that notebook setting
    `HF_HUB_DISABLE_XET=1`. `constants.HF_HUB_DISABLE_XET` is read from the environment
    **once, at import time**, and 1_run's cell 4 imported `ckpt` — and through it
    `huggingface_hub` — on its first line, before setting the variable. So the flag only
    ever reached the `sh()` subprocesses, never the notebook's own pull and push. Setting
    it here would need to happen in cell 1, ahead of every HF import. Left on deliberately:
    the `.jsonl` files are LFS-tracked, so chunk dedup means only their appended tails move
    on each hourly tick, and 1_run pushed this way without trouble.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, OPENAI_KEY, REPO, TAG, mo, os):
    # cell 5
    import json as _json
    import pathlib as _pl

    from experiments.common import ckpt

    mo.stop(not HF_TOKEN.value or not OPENAI_KEY.value,
            mo.md("*Paste **both** keys to start. The HF token restores the run and "
                  "checkpoints it; the OpenAI key grades it.*"))
    # judge_strongreject reads it from the environment of its subprocess, which inherits
    # this one. Nothing else in the notebook needs an API key.
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY.value

    EX_SCOPE = {"experiment": "extraction", "tag": TAG}
    ST_SCOPE = {"experiment": "steering_jailbreaks", "tag": TAG}
    # The judge's *daily* request cap, the limit that actually binds this run: 2_run needs
    # ~18,900 calls and gpt-4o-mini's RPD was measured at 10,000 ("Limit 10000, Used 10000")
    # when 1_run hit it. Tier-dependent -- raise it here if the account tier has changed,
    # since it only sizes the estimate printed before the pass, not the pass itself.
    JUDGE_RPD = 10_000
    ST_ROOT = _pl.Path(REPO, "experiments/steering_jailbreaks/results", TAG,
                       MODEL.replace("/", "_"))
    ST_META, ST_CSV = ST_ROOT / "meta", ST_ROOT / "csv"

    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    # ckpt.pull handles a size mismatch itself, inside its own attempts loop, so there is no
    # wrapper here: local `.incomplete` partials are deleted (their metadata kept, so files
    # that did land are not re-fetched), and a mismatch that survives that is a broken record
    # on the Hub, which it works around by fetching file by file and skipping those.
    print("extraction:", ckpt.pull(HF_REPO, token=HF_TOKEN.value,
                                   subpaths=["*/vectors/**", "*/meta/directions__*"],
                                   **EX_SCOPE))
    print("steering:  ", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))
    # The timer is what makes the checkpoint mid-run: steer_batch is one process for many
    # cells, so a per-script push never fires inside it. Idempotent per scope, so
    # re-running this cell replaces its timer instead of leaking another thread.
    ST_TIMER = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
    print("hourly checkpoint armed for steering_jailbreaks")

    def ST_COMPLETE(stem):
        """`complete` manifest **and** its rows on disk (spec 0.11).

        Both halves are load-bearing now. 2_run takes the steering scope past 256 files, so
        every push splits into two commits -- and a 429 between the parts can land a
        manifest without the `.jsonl` beside it, since `upload_folder` batches by file, not
        by stem. A manifest-only skip would then drop that cell from the judge pass and from
        `aggregate` without saying so. At 154 files 1_run pushed atomically and could not hit
        this.
        """
        if not (ST_META / f"{stem}.jsonl").exists():
            return False
        try:
            return _json.loads((ST_META / f"{stem}_manifest.json")
                               .read_text(encoding="utf-8")).get("status") == "complete"
        except (OSError, ValueError):        # missing, or a torn tail mid-push
            return False

    st_pulled = True
    return (JUDGE_RPD, ST_COMPLETE, ST_CSV, ST_META, ST_SCOPE, ST_TIMER, ckpt, st_pulled)


@app.cell(hide_code=True)
def _(mo):
    # cell 6
    mo.md(r"""
    ## The prompt view — CPU, seconds

    `sets.py` rebuilds both prompt sets from `acts/views/jailbreaks__all.json` and checks
    every prompt against its `prompt_sha16`. That view lives in *extraction's* tree and was
    never pushed there (1_run's jailbreak run was scoped to `probe_jailbreak_detection/`),
    so it is rebuilt locally rather than pulled: `--view-only` writes it from the tokenizer
    alone, no weights, and reproduces the GPU path's `view_key` exactly.
    """)
    return


@app.cell
def _(MODEL, TAG, sh, st_pulled):
    # cell 7
    assert st_pulled
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", "jailbreaks", "--split", "all", "--poles", "pos", "--view-only")
    view_ready = True
    return (view_ready,)


@app.cell(hide_code=True)
def _(mo):
    # cell 8
    mo.md(r"""
    ## The baseline, and the split it defines

    **1_run's, unchanged** — 508 successes / 433 refusals. It is pulled, not recomputed: the
    two sets have to be identical to 1_run's or the new cells are not comparable to the old
    ones.

    The cell below regenerates it only if the Hub had nothing, which would mean the pull went
    wrong, so it says so loudly first. `run_key` makes the regenerated cell the same cell,
    but it costs 9 min of GPU and 1,009 judge calls.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, ST_COMPLETE, ST_META, ST_SCOPE, TAG, ckpt, sh, view_ready):
    # cell 9
    assert view_ready                     # sets.py reads the view that cell writes
    _jsonl = ST_META / "gen_baseline.jsonl"
    if ST_COMPLETE("gen_baseline") and (ST_META / "gen_baseline_judged.jsonl").exists():
        # gen_baseline calls mdl.load() before it reads its own resume, so re-entering a
        # finished baseline still costs 15 GB of weights. This is the skip that matters.
        print("baseline present and judged -- skipped, no model load")
    else:
        print("! the judged baseline did not come back from the Hub; regenerating it. "
              "The split must match 1_run's 508/433 -- check that before reading anything.")
        if not ST_COMPLETE("gen_baseline"):
            sh("python", "experiments/steering_jailbreaks/gen_baseline.py", MODEL,
               "--tag", TAG, "--split", "all", "--decoding", "greedy",
               "--batch-size", "32", "--max-batch-tokens", "65536")
        sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
           str(_jsonl), "--concurrency", "6")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="gen_baseline", **ST_SCOPE)
    st_split = True
    return (st_split,)


@app.cell(hide_code=True)
def _(mo):
    # cell 10
    mo.md(r"""
    ## The cell list

    α is **signed** and `steer_single.resolve` refuses the half with no headroom, so the
    sign is derived, never typed: `RESTORE_SIGN` is −1 for the framing axes (`story`,
    `persona`) and +1 for the refusal axes (`harm`, `eval`) on the success set, mirrored on
    the refusal set. "α = 1.25 restore" is therefore **−1.25** for story and **+1.25** for
    harm; a hand-written sign slip fails at parse time rather than producing a floor.

    Only **two** new `noop`s. The no-op is per (set, layer set), so `story_v2_1k` at L15
    reuses the one 1_run already ran there for `persona_v2`, and L9 / L21 / L23 are
    unchanged.

    `ablate` appears only where suppressing that direction is the hypothesis — the framing
    axes on the success set, as in 1_run. Ablating `story` / `persona` on already-refused
    prompts is not an induce lever.
    """)
    return


@app.cell
def _(MODEL, ST_COMPLETE, TAG, mo):
    # cell 11
    import argparse as _ap
    import json as _j
    import tempfile as _tf
    from pathlib import Path as _P

    from transformers import AutoConfig as _AC

    from experiments.common import config as _cfg
    from experiments.steering_jailbreaks import cell as _cell, steer_single as _ss

    # 1_run's layers (extraction/insights.md: max cohens_dz_train, min train<->heldout gap).
    # The saturation extension runs at all four of them, two more alphas each -- one dict
    # rather than two that have to agree, and it also says which no-ops already exist.
    ST_1RUN, WIDE = {"harm_v2": 21, "eval_v2": 9, "story_v2_1k": 23,
                     "persona_v2": 15}, (1.00, 1.25)
    # The detection-best layers, full grid. probe_jailbreak_detection/insights.md: story L15
    # has the largest fiction-vs-rest margin, persona L4 the largest roleplay-vs-nonfiction
    # one plus a cohens_dz peak. story L15 and persona L15 are different directions at the
    # same layer, so their stems differ and the L15 no-op serves both.
    NEW_AXES, GRID = ({"story_v2_1k": 15, "persona_v2": 4},
                      (0.25, 0.50, 0.75, 1.00, 1.25))

    # Read from the config, not hardcoded: the band -- and so which layer needs the
    # out-of-band opt-in -- is a function of depth. A small JSON, not the weights.
    _N_LAYERS = _AC.from_pretrained(MODEL).num_hidden_layers
    _BAND = _cfg.band(_N_LAYERS)

    def _oob(layer):
        return ["--allow-out-of-band"] if layer not in _BAND else []

    def _add(axis, layer, alpha):
        # --mode add is explicit: on the set where this axis is suppressed its PRIMARY is
        # `ablate`, and the -alpha arm is the alternative to it.
        return ["--direction", axis, "--mode", "add", "--layers", str(layer),
                "--alpha", f"{alpha:g}", *_oob(layer)]

    def _jobs(prompt_set):
        """argv tails for one prompt set, exactly as steer_single.py would take them."""
        out, flip = [], 1.0 if prompt_set == "success" else -1.0
        for axis, layer in ST_1RUN.items():
            for a in WIDE:
                out.append(_add(axis, layer, _cell.RESTORE_SIGN[axis] * flip * a))
        for axis, layer in NEW_AXES.items():
            if _cell.PRIMARY[prompt_set][axis] == "ablate":
                out.append(["--direction", axis, "--layers", str(layer), *_oob(layer)])
            for a in GRID:
                out.append(_add(axis, layer, _cell.RESTORE_SIGN[axis] * flip * a))
        for layer in sorted(set(NEW_AXES.values()) - set(ST_1RUN.values())):
            out.append(["--arm", "noop", "--layers", str(layer), *_oob(layer)])
        return out

    def _stems(script, prompt_set, job):
        """The stems this argv tail writes -- the same call cell.run makes.

        `resolve` runs here too, so an illegal sign or mode surfaces while the list is being
        built rather than after a model load.
        """
        a = _ss.add_cell_args(_ap.ArgumentParser(prog="job")).parse_args(
            [MODEL, "--tag", TAG, *[str(x) for x in job]])
        direction, mode = _ss.resolve(a, prompt_set)
        return [_cell.stem_for(script, direction, mode, spec, a.alpha, a.tau_q, a.arm)
                for spec in _ss.cell_specs(a, _N_LAYERS)]

    ST_JOBS, ST_ALL_STEMS, _lines = {}, [], []
    for _set, _script in (("success", "steer_single"), ("refusal", "steer_induce")):
        _all = _jobs(_set)
        _st = [_stems(_script, _set, j) for j in _all]
        ST_ALL_STEMS += [s for ss in _st for s in ss]
        _pend = [j for j, ss in zip(_all, _st) if not all(map(ST_COMPLETE, ss))]
        _p = _P(_tf.gettempdir()) / f"st2_jobs_{_set}.json"
        _p.write_text(_j.dumps(_pend, indent=1), encoding="utf-8")
        ST_JOBS[_set] = (str(_p), len(_pend))
        _lines.append(f"**{_set}** — {len(_all)} cells, **{len(_pend)} pending**\n\n```\n"
                      + "\n".join(" ".join(j) for j in _all) + "\n```")

    mo.md(f"{len(ST_ALL_STEMS)} cells in 2_run\n\n" + "\n\n".join(_lines))
    return ST_ALL_STEMS, ST_JOBS


@app.cell(hide_code=True)
def _(mo):
    # cell 12
    mo.md(r"""
    ## Steer — GPU

    One `steer_batch` process per set, so the model loads at most twice rather than 32
    times, and only the **pending** jobs are handed to it — a set with nothing left skips the
    process entirely and pays no load. Resume is also per cell and per batch inside
    `steer_batch` itself, so a kill costs at most one batch either way.

    **`--batch-size 32 --max-batch-tokens 65536`, matching 1_run exactly.** Greedy is
    bit-reproducible only at fixed batch composition, and the α = 1 / 1.25 cells at
    L9 / L21 / L23 are paired against no-ops 1_run generated under those numbers. Changing
    them would silently break that pairing.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, ST_JOBS, ST_SCOPE, TAG, ckpt, sh, st_split):
    # cell 13
    assert st_split                      # the success set is defined by the judged baseline
    _path, _n = ST_JOBS["success"]
    if _n:
        sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
           "--script", "steer_single", "--jobs", _path,
           "--batch-size", "32", "--max-batch-tokens", "65536")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="2_run steer_single", **ST_SCOPE)
    else:
        print("steer_single: all 2_run cells complete -- skipped, no model load")
    st_success = True
    return (st_success,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, ST_JOBS, ST_SCOPE, TAG, ckpt, sh, st_success):
    # cell 14
    assert st_success
    _path, _n = ST_JOBS["refusal"]
    if _n:
        sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
           "--script", "steer_induce", "--jobs", _path,
           "--batch-size", "32", "--max-batch-tokens", "65536")
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="2_run steer_induce", **ST_SCOPE)
    else:
        print("steer_induce: all 2_run cells complete -- skipped, no model load")
    st_steered = True
    return (st_steered,)


@app.cell(hide_code=True)
def _(mo):
    # cell 15
    mo.md(r"""
    ## Judge — API, and **two** rate limits

    One call per row, **18,895** over the 40 new cells, ≈$5.6. Two separate ceilings bind,
    and they need different handling:

    | limit | measured | what it costs | response |
    |---|---|---|---|
    | **tokens per minute** | 200k TPM on `gpt-4o-mini`, ~1.2k tokens a call ≈ 165 calls/min | the pass runs at ~2.6 h whatever the concurrency | `--concurrency 6`. 8 workers sat exactly on the ceiling and 429'd; 6 leaves ~25% headroom. Transient 429/5xx retry with full jitter, 7 attempts reaching ~61s — one full window |
    | **requests per day** | **10,000 RPD** | 18,895 calls **cannot finish in one day** | not retryable: no backoff ladder outlives a day, and every attempt spends another request against a counter that is already empty. `judge_strongreject` raises and exits **3** |

    So this pass is a **two-day job**, and the notebook is built for it. The cell prints the
    ungraded-row count against the cap before it starts, then stops cleanly the moment it
    sees exit 3 — rather than marching the remaining cells into the same wall one fast
    failure at a time. Exit 3 breaks the loop; a single other non-zero exit is one bad cell
    and the loop carries on; two in a row is treated as a wall and also stops.

    **The stop is reactive, and deliberately so.** `JUDGE_RPD` below only sizes the estimate
    — nothing counts requests client-side, because this process cannot know how much of
    today's cap `1_run`, the baseline, or anything else on the account already spent. A
    counter starting at zero each session would stop early and waste quota, so the
    provider's own 429 is the authority. Learning it from the error costs the ~6 in-flight
    calls, which are rejected rather than billed, and the interrupted cell's summary is
    never written — so it is simply re-judged tomorrow, per row.

    Cells are judged in sorted-stem order, which puts all 19 `steer_induce` cells (8,227
    rows) ahead of the 21 `steer_single` ones. The day boundary therefore lands cleanly:
    day one finishes the whole refusal set plus a few success cells, day two the rest.

    **Day two is just re-running the notebook.** Everything already generated skips before
    the model load, every graded row is skipped by `unit_id`, and the judge's own response
    cache (`meta/judge_cache.jsonl`, keyed by request + response + model + `template_sha`)
    makes an already-seen call free. Only the ungraded remainder spends quota.

    A cell is skipped when its `csv/<stem>_summary.csv` says every row was graded. That is
    what makes a re-run cheap without trusting a file list: the summary is written only at
    the end of a pass, so a cell killed midway re-enters and resumes per row. 1_run's 36
    cells are skipped by the same test.

    The judge sees the **bare request**, never the jailbreak wrapper, and the deterministic
    detectors run alongside it at no cost — `outcome` is degenerate when *either* says so.
    That matters most here: α = 1.25, and `eval_v2` at L9, are exactly where
    50_per_direction produced 80–97% broken output.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, JUDGE_RPD, ST_ALL_STEMS, ST_CSV, ST_META, ST_SCOPE, ckpt, sh,
      st_steered):
    # cell 16
    import csv as _csv
    import json as _js

    assert st_steered

    def _graded(stem):
        """True when this cell's summary exists and every row in it was scored."""
        p = ST_CSV / f"{stem}_summary.csv"
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

    # A generations file is one with a sibling manifest -- exactly what judge_strongreject
    # loads first. meta/ also holds judge_cache.jsonl (the judge's own response cache) and
    # the _judged.jsonl outputs; neither has a manifest, and a blocklist naming only those
    # two picked up the cache and died on it. gen_baseline is judged in its own cell above.
    _mine = set(ST_ALL_STEMS)
    _todo = sorted(p for p in ST_META.glob("*.jsonl")
                   if not p.name.endswith("_judged.jsonl")
                   and not p.name.startswith("gen_baseline")
                   and (ST_META / f"{p.stem}_manifest.json").exists()
                   and not _graded(p.stem))
    # One API call per *ungraded* row -- a resumed day pays only for what is left, since
    # judge_strongreject skips unit_ids already in _judged.jsonl by the same judge.
    _left = {p.stem: len(_ids(p) - _ids(ST_META / f"{p.stem}_judged.jsonl"))
             for p in _todo}
    _need = sum(_left.values())
    _new = sum(p.stem in _mine for p in _todo)
    print(f"judging {len(_todo)} cells, {_new} of them 2_run's "
          f"({len(_mine) - _new}/{len(_mine)} of 2_run already graded)\n"
          f"{_need:,} rows still ungraded, against a {JUDGE_RPD:,}/day request cap "
          f"-> {-(-_need // JUDGE_RPD)} day(s)")
    if _need > JUDGE_RPD:
        print(f"! this pass CANNOT finish today. It will grade ~{JUDGE_RPD:,} rows, stop on "
              f"the daily cap, and resume here tomorrow -- just re-run the notebook.")

    # Two different failures, two different responses. `allow_fail` because judging is
    # resumable per row, so one bad cell should cost a retry of that cell rather than the
    # rest of the pass -- but exit 3 is the *daily* cap, which every remaining cell would hit
    # in turn, so it breaks out instead of failing 40 cells in a row.
    #
    # Nothing counts requests client-side: the stop is the provider's 429, because this
    # process cannot know how much of today's cap 1_run, the baseline, or anything else on
    # the account already spent. A counter starting at zero each session would stop early and
    # waste quota. Cost of learning it from the error: the ~6 in-flight calls get rejected,
    # unbilled, and the cell's summary is simply not written -- so it resumes tomorrow.
    #
    # The belt to that brace: exit 3 depends on a regex over the error string
    # (`requests per day|RPD`), so a reworded 429 would read as transient, grind 7 retries a
    # row, and exit 1. Two consecutive failures of ANY kind therefore also stop the pass --
    # one bad cell is a bad cell, two in a row is a wall, whatever it is called.
    _failed, _capped, _done_rows, _streak = [], False, 0, 0
    for _i, _p in enumerate(_todo, 1):
        print(f"[{_i}/{len(_todo)}] {_p.stem}  ({_left[_p.stem]:,} rows left)")
        _rc = sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
                 str(_p), "--concurrency", "6", allow_fail=True).returncode
        if _rc == 3:
            _capped = True
            break
        if _rc:
            _failed.append(_p.stem)
            _streak += 1
            if _streak == 2:
                print(f"\n! two cells failed back to back ({_failed[-2:]}) -- stopping. "
                      f"That is a wall, not a bad cell: check the last traceback above for "
                      f"a quota, key or network problem before re-running.")
                break
        else:
            _done_rows += _left[_p.stem]
            _streak = 0
    # One push for all of them: the hourly timer has been carrying the partials. Guarded,
    # because a verification re-run with nothing left to grade would otherwise spend two
    # commits walking 314 unchanged files.
    if _todo:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="2_run judged", **ST_SCOPE)
    if _failed:
        print(f"! {len(_failed)} cells did not judge, re-run this cell: {_failed}")
    if _capped:
        print(f"\n! DAILY REQUEST CAP -- stopped cleanly after ~{_done_rows:,} rows, "
              f"~{_need - _done_rows:,} to go. Everything graded is on the Hub. Re-run the "
              f"notebook after the quota resets; the GPU cells will all skip.")
    st_judged = not _capped and not _failed
    return (st_judged,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, ST_SCOPE, ST_TIMER, TAG, ckpt, sh, st_judged):
    # cell 17
    # An aggregate over a half-graded run is a table that looks final and is not, so it is
    # not written at all until every cell is scored. The timer stays armed, since the run
    # is not over.
    if not st_judged:
        print("judging did not finish -- aggregate SKIPPED and the hourly checkpoint left "
              "running. Re-run the notebook; every GPU cell will skip and judging resumes.")
    else:
        # aggregate.py spans the tag, so this re-reports 1_run's cells alongside 2_run's.
        sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
        _ok = ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="2_run aggregate", **ST_SCOPE)
        # check_stale spans the whole tag, so it also reports the jailbreak activations that
        # --view-only deliberately left uncomputed. Never fatal for that reason.
        _stale = sh("python", "-m", "experiments.common.check_stale", MODEL, TAG,
                    allow_fail=True).returncode
        # Only stop the timer once the results are actually up: try_push returns None when
        # it was rate-limited, and stopping on a skipped push strands them on local disk.
        if _ok is not None:
            ST_TIMER.set()
        print(("! check_stale reported findings (the uncached jailbreak blobs are expected)"
               if _stale else "all artefacts current")
              + (" | hourly checkpoint stopped" if _ok is not None
                 else " | push SKIPPED, timer left running"))
    return


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 18
    mo.md(f"""
    ## Read the results

    `steering_jailbreaks/results/{TAG}/<model_slug>/csv/`: `aggregate_cells.csv` is every
    cell of both runs, `aggregate_controls.csv` each target with `d_*_vs_noop`.

    Read in this order:

    1. **`pct_degenerate` before any effect.** α = 1.25, and `eval_v2` at L9, are where
       50_per_direction broke — 80–97% degenerate at α=1. A mostly-broken cell still has a
       ΔASR, and it means nothing.
    2. **`d_*_vs_noop`, matched on `prompt_set` × `layers_spec`.** The L4 cells have their
       own new no-op; the L9 / L21 / L23 cells pair against 1_run's.
    3. **`|Δh|`, not α.** The push is `α·σ_l·û_l` and σ runs 19.3 at L4 to 181.2 at L23, so
       the same α is a ~9× different absolute push between the new persona cells and the
       story ones.
    4. **`read_<axis>` for all four axes.** No 1_run cell moved only its own axis.

    Still **no `random` arm** at this tag, so nothing here is a specificity claim.
    """)
    return


if __name__ == "__main__":
    app.run()
