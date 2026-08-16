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

    # IDENTICAL to notebook_1K_gemma, and that is the whole point: these cells are compared
    # against no-ops generated there. Greedy is bit-reproducible only at fixed batch size
    # *and* composition, so changing either number here would make every new cell
    # incomparable to the L15/L8 no-ops it is scored against -- silently, with no error.
    BATCH = ("--batch-size", "16", "--max-batch-tokens", "24576")

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
    # {TAG} on `{MODEL}` — the α tail

    A **continuation** of `notebook_1K_gemma.py`, same tag and same model slug, adding the
    high-α cells the first sweep stopped short of. Nothing here re-runs: extraction,
    probe_jailbreak_detection, cross-probe, the baseline and all 48 original cells stay as
    they are, and every stem this notebook writes is new.

    **Why these two.** The first sweep ran α ≤ 1.00 and two directions never turned over:

    - **`story_v2_1k` @ L15** is still climbing at α = −1.00 (ΔSR −0.440) at **0.7%
      degenerate** — it has headroom, and it is the layer that beat L28. → **α 1.25, 1.50**
    - **`eval_v2` @ L8** is near-null at α = 1.00 (ΔSR −0.108 restore, +0.081 induce) at ~3%
      degenerate — far too clean to be at its limit, so "no effect" is not yet established.
      → **α 1.50, 2.00**

    `persona_v2` and `harm_v2` are **not** extended: both already saturate (99.8% / 99.4%
    refused) and break into degeneration by α = 1.00, so a larger α measures nothing.
    `story_v2_1k` @ **L28** is not extended either — it is 99.6% degenerate at α = −1.00
    already.

    **"Both positive and negative" is what the two arms already are.** `steer_single.resolve`
    permits exactly one sign per prompt set — the restoring sign on successes, its mirror on
    refusals — and exits before the model loads otherwise. So `eval_v2` runs **+α on the
    success set and −α on the refusal set**, which is both directions of the axis; asking for
    both signs *on one set* is not a config the code accepts.

    ## Cells

    | direction | L | α magnitudes | success arm | refusal arm |
    |---|---|---|---|---|
    | `story_v2_1k` | 15 | 1.25, 1.50 | `add` −α | `add` +α |
    | `eval_v2` | 8 | 1.50, 2.00 | `add` +α | `add` −α |

    **8 cells**, ≈3.9k generations. No new no-ops: the no-op is per (prompt set, layer) and
    L15 and L8 both already have theirs from the first sweep — which is also why the batch
    numbers above must not move.

    Both layers are outside the band (L17–38), so every cell carries `--allow-out-of-band`.

    ## Run all cells

    Guarded on artefacts, so re-running after a kill resumes. **Judging is not here** — it is
    API-bound and runs on your own machine; the last cell prints the commands.
    """)
    return


@app.cell
def _(mo):
    # cell 3
    # The only credential this notebook needs. No judge key: grading runs off the GPU box.
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write, and gemma-2 licence accepted)",
                          kind="password", full_width=True)
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
    # Gemma-2 soft-caps attention logits; sdpa drops that. Same value as the first notebook,
    # for the same reason as BATCH -- a different kernel is a different generation.
    os.environ["ATTN_IMPL"] = "eager"

    # The α tail, as (axis, layer) -> magnitudes. Signs are derived from cell.RESTORE_SIGN
    # and the prompt set, never typed here.
    EXTRA = {("story_v2_1k", 15): (1.25, 1.50),
             ("eval_v2", 8): (1.50, 2.00)}

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
    _oob = sorted({l for _, l in EXTRA if l not in BAND})
    print(f"{MODEL}: L={N_LAYERS}, band {BAND[0]}-{BAND[-1]}")
    print("extra cells: " + ", ".join(f"{a}@L{l} a={'/'.join(f'{m:g}' for m in ms)}"
                                      for (a, l), ms in EXTRA.items()))
    print(f"out of band, explicit opt-in: {_oob or 'none'}")
    return (BAND, DONE, EXTRA, EX_SCOPE, JB_ROOT, JB_SCOPE, N_LAYERS, ST_COMPLETE,
            ST_ROOT, ST_SCOPE, ckpt)


@app.cell
def _(EX_SCOPE, HF_REPO, HF_TOKEN, JB_SCOPE, ST_SCOPE, ckpt):
    # cell 5
    # Steering reads `vectors/` and the jailbreak view -- never the 2.2 GiB pole cache. So
    # extraction is pulled *small*: no `*/acts/**`, and in particular not blobs.tar. The
    # jailbreak view is rebuilt from the tokenizer in the next cell, weights not required.
    print("extraction (vectors + meta):",
          ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                    subpaths=["*/vectors/**", "*/meta/**"]))
    print("probe_jailbreak_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **JB_SCOPE))
    # Brings back gen_baseline_judged.jsonl (the prompt sets) and every finished cell, so
    # ST_COMPLETE can skip the 48 already done and only the 8 new ones are built.
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
    # The two prompt sets ARE the baseline's 3-way labels. It is already judged at this tag,
    # so this gate should pass on the first run -- it is here because every cell below
    # silently steers the wrong rows if it does not.
    _judged = ST_ROOT / "meta" / "gen_baseline_judged.jsonl"
    mo.stop(not _judged.exists(), mo.md(f"""
    ### ▸ `gen_baseline_judged.jsonl` is missing

    It should have come back with cell 5's pull — it was graded during the first sweep. If
    it really is gone, regrade it locally and push:

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
def _(BAND, EXTRA, MODEL, N_LAYERS, ST_COMPLETE, TAG, mo, st_split):
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
        """argv tails for one set: one `add` cell per (axis, layer, magnitude).

        The sign is `RESTORE_SIGN[axis] * (+1 on successes, -1 on refusals)` — the only sign
        `resolve` accepts on that set. No no-op jobs: the no-op is per (set, layer) and L15
        and L8 already have complete ones from the first sweep, so emitting them here would
        only produce stems ST_COMPLETE immediately filters out.
        """
        out, flip = [], 1.0 if prompt_set == "success" else -1.0
        for (axis, layer), mags in EXTRA.items():
            for a in mags:
                out.append(["--direction", axis, "--mode", "add", "--layers", str(layer),
                            "--alpha", f"{_cell.RESTORE_SIGN[axis] * flip * a:g}",
                            *_oob(layer)])
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
        _p = _P(_tf.gettempdir()) / f"gemma2_jobs_{_set}.json"
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
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="steer_single alpha tail", **ST_SCOPE)
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
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="steer_induce alpha tail", **ST_SCOPE)
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
    print(ckpt.push(HF_REPO, token=HF_TOKEN.value, msg="alpha tail complete", **ST_SCOPE))
    st_done = True
    return (st_done,)


@app.cell(hide_code=True)
def _(EXTRA, HF_REPO, MODEL, TAG, mo, st_done):
    # cell 12 (markdown)
    assert st_done
    _slug = MODEL.replace("/", "_")
    _n = 2 * sum(len(m) for m in EXTRA.values())
    mo.md(f"""
    # ▸ Judge these locally, then re-aggregate

    {_n} new cells, ≈3.9k rows, ≈3.9k judge calls (~40 min at `--concurrency 6`). The loop
    skips any cell whose `csv/<stem>_summary.csv` says it is already scored, so re-running it
    after an interruption costs only the remainder.

    **Run one judge process at a time.** `judge_strongreject` serialises its appends with a
    `threading.Lock`, which is per *process* — two of them on the same cell interleave their
    writes and corrupt the `_judged.jsonl`.

    ```bash
    M={MODEL}; T={TAG}; export RUN_TAG=$T
    R={HF_REPO}
    D=experiments/steering_jailbreaks/results/$T/{_slug}
    python -c "from experiments.common import ckpt; ckpt.pull('$R', experiment='steering_jailbreaks', tag='$T')"

    for f in $D/meta/steer_*.jsonl; do
      [ -e "$f" ] || continue                       # nullglob is off: an unmatched glob is literal
      case "$f" in *_judged.jsonl) continue;; esac
      python experiments/steering_jailbreaks/judge_strongreject.py "$f" --concurrency 6
    done

    python experiments/steering_jailbreaks/aggregate.py $M --tag $T
    python -c "from experiments.common import ckpt; ckpt.push('$R', experiment='steering_jailbreaks', tag='$T', msg='alpha tail judged')"
    ```

    `aggregate.py` spans the whole tag, so it rebuilds `aggregate_controls.csv` over all
    **56** cells at once — the α curve per direction extends rather than forking.

    **Read `pct_degenerate` before any Δ.** These are the largest α at this tag: a degenerate
    response scores `strongreject ≈ 0`, which is exactly what a refusal scores, so a cell
    that broke the model is indistinguishable from one that restored refusal on the Δ column
    alone. At α = 1.00 story@L15 was 0.7% and eval@L8 ~3%; if either passes ~15% here, that
    cell measures degeneration, not steering.
    """)
    return


if __name__ == "__main__":
    app.run()
