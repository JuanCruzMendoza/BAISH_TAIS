import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    # cell 1
    import os, pathlib, subprocess, sys
    from collections import deque

    import marimo as mo

    MODEL = "Qwen/Qwen2.5-7B-Instruct"
    TAG = "1K_per_direction"
    HF_REPO = "JuanCruzMendoza/BAISH_TAIS"

    # IDENTICAL to runs 1-4 at this tag. The new cells are scored against no-ops generated
    # here, but they also have to sit on the same α curve as story@L15/L23 -- and greedy is
    # bit-reproducible only at fixed batch size *and* composition. Moving either number
    # would make L7/L18 incomparable to the layers they exist to be compared against,
    # silently and with no error.
    BATCH = ("--batch-size", "32", "--max-batch-tokens", "65536")

    NB = mo.notebook_dir()
    ROOTS = [d for d in [NB, *NB.parents]
             if (d / "experiments" / "common" / "ckpt.py").exists()]
    REPO = str(ROOTS[0]) if ROOTS else str(NB / "BAISH_TAIS")

    def sh(*a, cwd=None, allow_fail=False):
        """Stream the subprocess's output while it runs, keeping the tail for the error."""
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
    return BATCH, HF_REPO, MODEL, REPO, TAG, mo, os, sh


@app.cell(hide_code=True)
def _(MODEL, TAG, mo):
    # cell 2
    mo.md(f"""
    # {TAG} on `{MODEL}` — story at **L7 and L18**

    A **continuation** of runs 1–4 at the same tag and model slug. Nothing here re-runs:
    extraction, `probe_jailbreak_detection`, cross-probe, the baseline and all 96 existing
    cells stay as they are, and every stem this notebook writes is new.

    **Why.** On gemma-2-9b story@L15 restores refusal by **+39.7 points at 0.7% degenerate**;
    on Qwen the best story cell — L15 — reaches **+16.4 at 3.3%**, and dies at α = −1.00
    (41.3% degenerate). Qwen has only ever steered story at two layers, both in band. Two
    untried layers are the candidates for the gap:

    | L | frac | why this layer |
    |---|---|---|
    | **18** | 0.64 | the story probe's read-out *profile* at gemma L15 (fiction 84 / roleplay 34 / hybrid 60 / nonfic 6) is a broad reader. Qwen L15 is narrow — hybrid **13**, roleplay 3. Qwen L18 is the broad one: 68 / 14 / **32** / 5 |
    | **7** | 0.25 | gemma's winning layer is *out of band*. Qwen's out-of-band shallow peak is L7 — fic−nonfic margin 59.0, just above the L10–L13 dead zone where the probe reads ~0 |

    L15 and L23 are **not** re-run; they are what these are read against.

    ## Cells

    | direction | L | α magnitudes | success arm | refusal arm |
    |---|---|---|---|---|
    | `story_v2_1k` | 7 | 0.25, 0.50, 0.75, 1.00 | `add` −α | `add` +α |
    | `story_v2_1k` | 18 | 0.25, 0.50, 0.75, 1.00 | `add` −α | `add` +α |

    The same ladder gemma ran at its L15/L28, so the two models line up rung for rung.
    Signs are `RESTORE_SIGN['story_v2_1k'] × (+1 on successes, −1 on refusals)`, derived
    from `cell`, never typed. `ablate` is not run — story's success-set primary *is* `ablate`,
    but α is what was asked for and the α ladder is what gemma is comparable on.

    **20 cells, ≈9.4k generations** (16 `add` + **4 no-ops**), ≈1.6 h GPU. Unlike the α-tail
    passes, L7 and L18 have **no no-op yet** — it is per (prompt set, layer) — so one is
    emitted per set per layer, which is also why the batch numbers above must not move.

    L7 is outside the band (L11–25), so its cells carry `--allow-out-of-band`. L18 is inside.

    ## Run all cells

    Guarded on artefacts, so re-running after a kill resumes. **Judging is not here** — it is
    API-bound and runs on your own machine; the last cell prints the commands.
    """)
    return


@app.cell
def _(mo):
    # cell 3
    # The only credential this notebook needs. No judge key: grading runs off the GPU box.
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    HF_TOKEN
    return (HF_TOKEN,)


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, REPO, TAG, mo, os):
    # cell 4
    import json as _json
    from pathlib import Path as _P

    from transformers import AutoConfig as _AC

    from experiments.common import ckpt, config as cfg

    mo.stop(not HF_TOKEN.value,
            mo.md("*Paste the HF token to start. Nothing runs without it.*"))
    os.environ["HF_TOKEN"] = HF_TOKEN.value

    # The new cells, as (axis, layer) -> α magnitudes. Signs are derived from
    # cell.RESTORE_SIGN and the prompt set, never typed here.
    NEW = {("story_v2_1k", 7): (0.25, 0.50, 0.75, 1.00),
           ("story_v2_1k", 18): (0.25, 0.50, 0.75, 1.00)}

    EX_SCOPE = {"experiment": "extraction", "tag": TAG}
    JB_SCOPE = {"experiment": "probe_jailbreak_detection", "tag": TAG}
    ST_SCOPE = {"experiment": "steering_jailbreaks", "tag": TAG}

    def _root(experiment):
        return _P(REPO, "experiments", experiment, "results", TAG,
                  MODEL.replace("/", "_"))

    JB_ROOT, ST_ROOT = _root("probe_jailbreak_detection"), _root("steering_jailbreaks")

    N_LAYERS = _AC.from_pretrained(MODEL).num_hidden_layers
    BAND = cfg.band(N_LAYERS)

    def DONE(root, stem, **want):
        """Complete manifest whose config matches `want`."""
        try:
            m = _json.loads(_P(root, "meta", f"{stem}_manifest.json")
                            .read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (m.get("status") == "complete"
                and all(m.get("config", {}).get(k) == v for k, v in want.items()))

    def ST_COMPLETE(stem):
        """A steering cell: complete manifest **and** its rows on disk (spec 0.11)."""
        return (ST_ROOT / "meta" / f"{stem}.jsonl").exists() and DONE(ST_ROOT, stem)

    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    _oob = sorted({l for _, l in NEW if l not in BAND})
    print(f"{MODEL}: L={N_LAYERS}, band {BAND[0]}-{BAND[-1]}")
    print("new cells: " + ", ".join(f"{a}@L{l} a={'/'.join(f'{m:g}' for m in ms)}"
                                    for (a, l), ms in NEW.items()))
    print(f"out of band, explicit opt-in: {_oob or 'none'}")
    return (BAND, DONE, EX_SCOPE, JB_ROOT, JB_SCOPE, NEW, N_LAYERS, ST_COMPLETE,
            ST_ROOT, ST_SCOPE, ckpt)


@app.cell
def _(EX_SCOPE, HF_REPO, HF_TOKEN, JB_SCOPE, ST_SCOPE, ckpt):
    # cell 5
    # Steering reads `vectors/` and the jailbreak view -- never the 2.2 GiB pole cache. So
    # extraction is pulled *small*: no `*/acts/**`, and in particular not blobs.tar. The
    # jailbreak view is rebuilt from the tokenizer in the next cell, weights not required.
    # `directions__story_v2_1k.pt` holds every layer, so L7 and L18 need no re-extraction.
    print("extraction (vectors + meta):",
          ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                    subpaths=["*/vectors/**", "*/meta/**"]))
    print("probe_jailbreak_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **JB_SCOPE))
    # Brings back gen_baseline_judged.jsonl (the prompt sets) and every finished cell, so
    # ST_COMPLETE can skip runs 1-4 and only the 20 new ones are built.
    print("steering_jailbreaks:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))
    pulled = True
    return (pulled,)


@app.cell
def _(DONE, JB_ROOT, MODEL, TAG, pulled, sh):
    # cell 6
    assert pulled
    # --view-only writes acts/views/jailbreaks__all.json from the tokenizer alone: no
    # weights, no activations, and it reproduces the GPU path's view_key exactly, so no
    # run_key moves. jb_readout is complete at this tag, so this is always the cheap path --
    # the branch stays only so the notebook is correct on a tag where it is not.
    _vo = ["--view-only"] if DONE(JB_ROOT, "jb_readout") else []
    print("=== jailbreaks/all: "
          + ("view only, no weights ===" if _vo else "caching 1,009 activations ==="))
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", "jailbreaks", "--split", "all", "--poles", "pos", *_vo)
    jb_cached = True
    return (jb_cached,)


@app.cell
def _(HF_REPO, MODEL, ST_ROOT, TAG, jb_cached, mo):
    # cell 7
    assert jb_cached                      # that cell writes the view sets.py reads
    # The two prompt sets ARE the baseline's 3-way labels -- 508 successes / 433 refusals,
    # 1_run's split. It is already judged at this tag, so this gate should pass on the first
    # run; it is here because every cell below silently steers the wrong rows if it does not.
    _judged = ST_ROOT / "meta" / "gen_baseline_judged.jsonl"
    mo.stop(not _judged.exists(), mo.md(f"""
    ### ▸ `gen_baseline_judged.jsonl` is missing

    It should have come back with cell 5's pull — it was graded during 1_run. If it really is
    gone, regrade it locally and push:

    ```bash
    M={MODEL}; export RUN_TAG={TAG}
    python -c "from experiments.common import ckpt; ckpt.pull('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}')"
    python experiments/steering_jailbreaks/judge_strongreject.py \\
        experiments/steering_jailbreaks/results/{TAG}/{MODEL.replace('/', '_')}/meta/gen_baseline.jsonl \\
        --concurrency 6
    python -c "from experiments.common import ckpt; ckpt.push('{HF_REPO}', experiment='steering_jailbreaks', tag='{TAG}', msg='baseline judged')"
    ```
    """))
    st_split = True
    return (st_split,)


@app.cell
def _(BAND, MODEL, NEW, N_LAYERS, ST_COMPLETE, TAG, mo, st_split):
    # cell 8
    import argparse as _ap
    import json as _j
    import tempfile as _tf
    from pathlib import Path as _P

    from experiments.steering_jailbreaks import cell as _cell, steer_single as _ss

    assert st_split

    def _oob(layer):
        return ["--allow-out-of-band"] if layer not in BAND else []

    def _jobs(prompt_set):
        """argv tails for one set: one `add` cell per (axis, layer, magnitude), plus no-ops.

        The sign is `RESTORE_SIGN[axis] * (+1 on successes, -1 on refusals)` — the only sign
        `resolve` accepts on that set. `--mode add` is explicit because story's success-set
        PRIMARY is `ablate`, so the -α arm is the alternative to it, not the default.

        A no-op is emitted per layer these targets steer, and only where one is not already
        complete: it is per (set, layer), and L7 and L18 are new at this tag, so unlike the
        α-tail passes both really do need one.
        """
        out, flip = [], 1.0 if prompt_set == "success" else -1.0
        for (axis, layer), mags in NEW.items():
            for a in mags:
                out.append(["--direction", axis, "--mode", "add", "--layers", str(layer),
                            "--alpha", f"{_cell.RESTORE_SIGN[axis] * flip * a:g}",
                            *_oob(layer)])
        _script = "steer_single" if prompt_set == "success" else "steer_induce"
        for layer in sorted({int(j[j.index("--layers") + 1]) for j in out}):
            if not ST_COMPLETE(f"{_script}__noop__L{layer}"):
                out.append(["--arm", "noop", "--layers", str(layer), *_oob(layer)])
        return out

    def _stems(script, prompt_set, job):
        """The stems this argv tail writes -- the same call cell.run makes.

        `resolve` runs here too, so an illegal sign surfaces while the list is built rather
        than after a model load.
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
        _p = _P(_tf.gettempdir()) / f"qwen5_jobs_{_set}.json"
        _p.write_text(_j.dumps(_pend, indent=1), encoding="utf-8")
        ST_JOBS[_set] = (str(_p), len(_pend))
        _lines.append(f"**{_set}** — {len(_all)} cells, **{len(_pend)} pending**\n\n```\n"
                      + "\n".join(" ".join(j) for j in _all) + "\n```")

    mo.md(f"{len(ST_ALL_STEMS)} steering cells\n\n" + "\n\n".join(_lines))
    return ST_ALL_STEMS, ST_JOBS


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, ST_JOBS, ST_SCOPE, TAG, ckpt, sh):
    # cell 9
    # One steer_batch process per set, so the model loads at most twice; only the *pending*
    # jobs are handed to it, and a set with nothing left pays no load at all.
    _path, _n = ST_JOBS["success"]
    if _n:
        _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
        try:
            print(f"=== steer_single: {_n} pending cell(s) on the success set ===")
            sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
               "--script", "steer_single", "--jobs", _path, *BATCH)
        finally:
            _timer.set()
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="story L7/L18 success", **ST_SCOPE)
    else:
        print("steer_single: all cells complete -- skipped, no model load")
    st_success = True
    return (st_success,)


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, MODEL, ST_JOBS, ST_SCOPE, TAG, ckpt, sh, st_success):
    # cell 10
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
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="story L7/L18 refusal", **ST_SCOPE)
    else:
        print("steer_induce: all cells complete -- skipped, no model load")
    st_steered = True
    return (st_steered,)


@app.cell
def _(HF_REPO, HF_TOKEN, ST_ALL_STEMS, ST_COMPLETE, ST_SCOPE, ckpt, st_steered):
    # cell 11
    assert st_steered
    # push(), not try_push(): the gate below sends you to another machine, so the rows have
    # to be on the Hub before this notebook is done with them.
    _missing = [s for s in ST_ALL_STEMS if not ST_COMPLETE(s)]
    if _missing:
        raise RuntimeError(f"{len(_missing)} cell(s) did not complete: {_missing[:6]}")
    print(f"all {len(ST_ALL_STEMS)} cells complete; pushing")
    print(ckpt.push(HF_REPO, token=HF_TOKEN.value, msg="story L7/L18 complete", **ST_SCOPE))
    st_done = True
    return (st_done,)


@app.cell(hide_code=True)
def _(HF_REPO, MODEL, ST_ALL_STEMS, TAG, mo, st_done):
    # cell 12 (markdown)
    assert st_done
    _slug = MODEL.replace("/", "_")
    _stems = " \\\n      ".join(sorted(ST_ALL_STEMS))
    mo.md(f"""
    # ▸ Judge these locally, then re-aggregate

    {len(ST_ALL_STEMS)} new cells, ≈9.4k rows, ≈9.4k judge calls (~1.6 h at `--concurrency 6`,
    ≈$3). The list is explicit rather than a `steer_*` glob: runs 1–4 left 96 graded cells in
    the same directory, and re-entering each of them costs a process load to discover it has
    nothing to do.

    **Run one judge process at a time.** `judge_strongreject` serialises its appends with a
    `threading.Lock`, which is per *process* — two of them on the same cell interleave their
    writes and corrupt the `_judged.jsonl`.

    ```bash
    M={MODEL}; T={TAG}; export RUN_TAG=$T
    R={HF_REPO}
    D=experiments/steering_jailbreaks/results/$T/{_slug}
    python -c "from experiments.common import ckpt; ckpt.pull('$R', experiment='steering_jailbreaks', tag='$T')"

    for s in \\
      {_stems}
    do
      python experiments/steering_jailbreaks/judge_strongreject.py "$D/meta/$s.jsonl" --concurrency 6
    done

    python experiments/steering_jailbreaks/aggregate.py $M --tag $T
    python -c "from experiments.common import ckpt; ckpt.push('$R', experiment='steering_jailbreaks', tag='$T', msg='story L7/L18 judged')"
    ```

    `aggregate.py` spans the whole tag, so it rebuilds `aggregate_controls.csv` over all
    **116** cells at once — story's α curve gains two layers rather than forking.

    ## Read it against L15, not against gemma

    1. **`pct_degenerate` first.** A degenerate response scores `strongreject ≈ 0`, which is
       what a refusal scores too, so a cell that broke the model is indistinguishable from one
       that restored refusal on the Δ column alone. Story@L15 was 3.3% at α = −0.75 and 41.3%
       at −1.00; L23 was 99.6% degenerate at its top rung. Anything past ~15% measures
       degeneration.
    2. **`d_*_vs_noop`, matched on `prompt_set` × `layers_spec`** — L7 and L18 have their own
       no-ops from this pass and must be paired with those, not with L15's.
    3. **The question is the frontier, not the peak**: largest Δrefusal at ≤5% degenerate.
       L15 tops out at **+16.4**; gemma L15 reaches **+39.7**. Either L7 or L18 closing that
       gap says Qwen's story effect was layer-limited; neither closing it says it is not.
    4. **`read_story_v2_1k` per α.** Qwen L15 displaces the read-out ~3.5× further per unit α
       than gemma L15 does, which is why its α window shuts sooner. If L7 or L18 restores as
       much at a *smaller* read-out displacement, the layer was the problem.

    Still **no `random` arm** at this tag, so nothing here is a specificity claim — and the
    existing pair arms at L15 already show story's persona-parallel component restoring
    **more** (38.0% refused) than story itself (20.3%).
    """)
    return


if __name__ == "__main__":
    app.run()
