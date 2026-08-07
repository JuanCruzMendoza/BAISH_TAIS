import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import subprocess, pathlib, sys
    TAG = "50_per_direction"
    REPO, MODEL = "/marimo/BAISH_TAIS", "Qwen/Qwen2.5-7B-Instruct"

    ROOT = str(pathlib.Path(REPO).parent)
    def sh(*a, cwd=REPO):
        p = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
        print(p.stdout[-4000:], p.stderr[-2000:])
        if p.returncode:
            # In the exception, not just printed: marimo shows the traceback and the
            # cell output separately, and `assert ..., a` carried only the argv.
            raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, a))}\n"
                               f"{(p.stderr or p.stdout)[-1500:]}")
        return p
    if pathlib.Path(REPO).exists():
        sh("git", "fetch", "origin"); sh("git", "reset", "--hard", "origin/main")
    else:
        sh("git", "clone", "https://github.com/JuanCruzMendoza/BAISH_TAIS.git", REPO, cwd=ROOT)
    sh("git", "log", "--oneline", "-1")
    sh("python", "-c", "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))")
    return MODEL, REPO, TAG, pathlib, sh, subprocess, sys


@app.cell
def _(sh):
    sh("pip", "install", "-q", "transformers", "accelerate", "numpy", "openai")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 50_per_direction
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extraction
    """)
    return


@app.cell
def _(MODEL, TAG, sh):
    DATASETS = [("story_v2","train"),("story_v2","heldout"),("story_v1","train"),("story_v1","heldout"),
                ("harm","train"),("harm","heldout"),("persona","train"),("persona","heldout"),
                ("eval","train"),("eval","heldout"),("length","train"),("length","heldout"),
                ("v1_nofiller100","train")]
    for ds, sp in DATASETS:
        extra = ["--append-task"] if ds in ("persona", "eval") else []
        sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
           "--dataset", ds, "--split", sp, *extra)
    return


@app.cell
def _(MODEL, TAG, sh):
    for d in ["story_v2", "story_v1", "harm", "persona", "eval", "length"]:
        sh("python", "experiments/extraction/extract_direction.py", MODEL, "--tag", TAG, "--direction", d)
        sh("python", "experiments/extraction/probe_select.py", MODEL, "--tag", TAG, "--direction", d)
    sh("python", "experiments/extraction/compare_crossed.py", MODEL, "--tag", TAG)
    sh("python", "experiments/extraction/probe_select.py", MODEL, "--tag", TAG,
       "--direction", "story_v2", "--transfer", "v1_nofiller100")
    sh("python", "-m", "experiments.common.check_stale", MODEL, TAG)
    return


@app.cell
def _(TAG, mo, sh):
    sh("tar", "-cf", f"/tmp/extraction_{TAG}.tar", "--exclude=_archive", "--exclude=vectors",
       "-C", "experiments/extraction/results", TAG)
    mo.download(open(f"/tmp/extraction_{TAG}.tar", "rb").read(), filename=f"extraction_{TAG}.tar")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Probe jailbreak detection
    """)
    return


@app.cell
def _(MODEL, TAG, sh):
    # No --poles: cache BOTH arms. jb_readout reads only m["pos"], so the two-pole view
    # serves it fine, and cap's tau needs the bare requests (spec 0.6). Caching pos-only
    # here and re-caching later would move the view_key between jb_readout and the
    # steering runs -- which is exactly the mismatch that has to be reasoned about twice.
    sh("python", "experiments/extraction/cache_activations.py", MODEL, "--tag", TAG,
       "--dataset", "jailbreaks", "--split", "all", "--subsample-n", "100")
    return


@app.cell
def _(MODEL, REPO, subprocess, sys):
    subprocess.run([sys.executable, f"{REPO}/experiments/probe_jailbreak_detection/jb_readout.py",
                     MODEL, "--tag", "50_per_direction"], cwd=REPO, check=True)
    subprocess.run([sys.executable, f"{REPO}/experiments/probe_jailbreak_detection/jb_metrics.py",
                     MODEL, "--tag", "50_per_direction"], cwd=REPO, check=True)
    return


@app.cell
def _(MODEL, TAG, sh):
    # Per-jailbreak table, at the same layer sets section 5.4 steers.
    sh("python", "experiments/probe_jailbreak_detection/jb_readout_table.py", MODEL, "--tag", TAG,
       "--sweep", "story_v2=15,17,18", "--sweep", "story_v1=15,16,20",
       "--sweep", "persona=17,19,21", "--sweep", "harm=20,21,22", "--sweep", "eval=14,15,16",
       "--band", "steer_band")
    return


@app.cell
def _(TAG, mo, sh):
    sh("tar", "-cf", f"/tmp/jb_{TAG}.tar", "--exclude=_archive",
       "-C", "experiments/probe_jailbreak_detection/results", TAG)
    mo.download(open(f"/tmp/jb_{TAG}.tar", "rb").read(), filename=f"jb_{TAG}.tar")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steering jailbreaks (spec 5)

    Run order below is **top to bottom**; every cell is one script.

    | cell | needs |
    |---|---|
    | two-pole re-cache, `gen_decoding_compare`, `gen_baseline`, `steer_*` | **GPU** |
    | `judge_strongreject` (3 cells, marked) | **OpenAI API key** |
    | `aggregate`, `jb_success_split` | CPU only |

    The judge is the only thing that calls an API. Everything else is local.

    5.4 and 5.5 go through `steer_batch.py`, which validates all its jobs and *then*
    loads the model once — 2 loads for 93 cells instead of 47. Batch size is **32 /
    65536 padded tokens** everywhere: it sits inside every cell's `run_key`, so it has
    to be identical across scripts and must not change once the first cell has run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prerequisite — the jailbreaks cache

    Already done by the probe-detection section, which caches **both** poles. `cap`'s
    threshold sits on the two-pole corpus (framed prompts *and* bare requests, spec 0.6),
    so there is no separate re-cache step and no `view_key` drift between what
    `jb_readout` read and what the steering runs read.

    Nothing to run here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Paths and the API key

    Paste the key into the field. It is kept in the kernel process only — it is not
    written into this notebook file, and `.env` is gitignored anyway.
    """)
    return


@app.cell
def _(MODEL, REPO, TAG, mo, pathlib):
    SLUG = MODEL.replace("/", "_")
    SJ = pathlib.Path(REPO) / "experiments/steering_jailbreaks/results" / TAG / SLUG
    SJ_META, SJ_CSV = SJ / "meta", SJ / "csv"

    API_KEY = mo.ui.text(label="OPENAI_API_KEY", kind="password", full_width=True)
    API_KEY
    return API_KEY, SJ, SJ_CSV, SJ_META


@app.cell
def _(API_KEY):
    import os
    JUDGE_MODEL = "gpt-4o-mini"
    # Judging is one API call per row and was serial: ~2.5 s/row, so ~2.5 h for the
    # full pass. 8-way brings that to ~20 min. Raise only if the account's rate limit
    # allows -- 429s are retried with backoff, but a wall of them just wastes time.
    JUDGE_CONC = "8"
    if API_KEY.value:
        os.environ["OPENAI_API_KEY"] = API_KEY.value
        print("key set;", JUDGE_MODEL, "will grade")
    else:
        print("no key yet - the judge cells will refuse until one is pasted above")
    return JUDGE_CONC, JUDGE_MODEL


@app.cell
def _(SJ_META):
    def ungraded(pattern="*.jsonl"):
        """Generation files under meta/, excluding the judge's own outputs."""
        return sorted(p for p in SJ_META.glob(pattern)
                      if not p.name.endswith("_judged.jsonl")
                      and p.name != "judge_cache.jsonl")
    return (ungraded,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.1 — `gen_decoding_compare.py`  ·  GPU

    50 rows × 7 cells = 350 generations. Fixes the decoding under which
    "successful jailbreak" is defined.
    """)
    return


@app.cell
def _(MODEL, TAG, sh):
    sh("python", "experiments/steering_jailbreaks/gen_decoding_compare.py", MODEL, "--tag", TAG,
       "--batch-size", "32", "--max-batch-tokens", "65536")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 — judge the 7 decoding cells  ·  **API**

    One call per row; ~350 calls. Resumable: a re-run only grades what is missing, and
    identical responses hit the response-keyed cache.
    """)
    return


@app.cell
def _(JUDGE_CONC, JUDGE_MODEL, sh, ungraded):
    for _f in ungraded("gen_decoding_compare*.jsonl"):
        sh("python", "experiments/steering_jailbreaks/judge_strongreject.py", str(_f),
           "--judge-model", JUDGE_MODEL, "--concurrency", JUDGE_CONC)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pick the decoding  ·  CPU

    `aggregate.py` writes `aggregate_decoding.csv`: `asr_mean` with `asr_min`/`asr_max`
    across seeds. Read it, then set `DECODING` in the next cell. Greedy is the
    recommendation — a steering delta should be steering, not sampling — and a sampled
    pick turns ASR into a rate over ≥5 samples per cell, multiplying 5.4 by 5.
    """)
    return


@app.cell
def _(MODEL, SJ_CSV, TAG, sh):
    sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
    print((SJ_CSV / "aggregate_decoding.csv").read_text(encoding="utf-8"))
    return


@app.cell
def _():
    # Set from aggregate_decoding.csv above. Sampled configs also need DECODE_SEED.
    DECODING, DECODE_SEED = "greedy", None
    DECODE_ARGS = ["--decoding", DECODING] + (
        [] if DECODE_SEED is None else ["--decode-seed", str(DECODE_SEED)])
    print("downstream cells will use:", DECODE_ARGS)
    return (DECODE_ARGS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.2 — `gen_baseline.py`  ·  GPU

    100 unsteered generations. This is what **defines both prompt sets**: 5.4 runs on the
    rows it complied with, 5.5 on the rows it refused.
    """)
    return


@app.cell
def _(BATCH, DECODE_ARGS, MODEL, TAG, sh):
    sh("python", "experiments/steering_jailbreaks/gen_baseline.py", MODEL, "--tag", TAG,
       *BATCH, *DECODE_ARGS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 — judge the baseline  ·  **API**

    ~100 calls. Nothing below can run until this finishes: `steer_single` and
    `steer_induce` both read the two sets out of this file.
    """)
    return


@app.cell
def _(JUDGE_CONC, JUDGE_MODEL, SJ_META, sh):
    sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
       str(SJ_META / "gen_baseline.jsonl"), "--judge-model", JUDGE_MODEL,
       "--concurrency", JUDGE_CONC)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.5 — `jb_success_split.py`  ·  CPU

    Joins the baseline labels onto experiment 3's readouts: does a probe already read the
    jailbreaks that worked differently from the ones refused? Unblocked by the cell above.
    """)
    return


@app.cell
def _(MODEL, TAG, sh):
    sh("python", "experiments/probe_jailbreak_detection/jb_success_split.py", MODEL, "--tag", TAG)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The cell grid

    `--mode` is deliberately **not** passed for the primary jobs: `cell.PRIMARY` maps
    (prompt set, direction) → mode, so the notebook cannot disagree with the code about
    which direction is ablated and which is added. `cap` is the one explicit mode, since
    it is an alternative rather than the primary.
    """)
    return


@app.cell
def _():
    LAYERS = {"story_v2": "15,17,18", "story_v1": "15,16,20", "persona": "17,19,21",
              "harm": "20,21,22", "eval": "14,15,16"}
    STORY = ["story_v2", "story_v1", "persona"]      # ablate on successes, add on refusals
    HARMEV = ["harm", "eval"]                        # add on successes, ablate on refusals
    ALPHAS = ["0.5", "1"]
    NOOP_SWEEP = "14,15,16,17,18,19,20,21,22"        # every single layer used above

    def jobs(alpha_dirs, plain_dirs, with_cap):
        """One argv tail per invocation. A sweep is 3 cells, a band 1 (spec 5.4.0)."""
        out = []
        for _d in plain_dirs:                        # ablate: no alpha
            out.append(["--direction", _d, "--sweep-layers", LAYERS[_d]])
            out.append(["--direction", _d, "--layers", "steer_band"])
        for _d in alpha_dirs:                        # add: one cell per alpha
            for _a in ALPHAS:
                out.append(["--direction", _d, "--sweep-layers", LAYERS[_d], "--alpha", _a])
                out.append(["--direction", _d, "--layers", "steer_band", "--alpha", _a])
        for _d in (STORY if with_cap else []):       # graded alternative to ablate
            out.append(["--direction", _d, "--mode", "cap", "--layers", "steer_band",
                        "--tau-q", "75"])
        # specificity control: one random cell per target cell, since aggregate matches it
        # on (mode, layers_spec, alpha, tau_q) -- a band-only random leaves every
        # single-layer and every alpha=1 cell without one (spec 5.4, "same layer set and
        # same alpha/sqrt(N)").
        out += [_j + ["--arm", "random"] for _j in list(out)]
        out.append(["--arm", "noop", "--sweep-layers", NOOP_SWEEP])   # zero point per layer set
        out.append(["--arm", "noop", "--layers", "steer_band"])
        return out

    SUCCESS_JOBS = jobs(HARMEV, STORY, with_cap=True)     # 5.4
    REFUSAL_JOBS = jobs(STORY, HARMEV, with_cap=False)    # 5.5, modes mirrored
    print(f"5.4: {len(SUCCESS_JOBS)} jobs   5.5: {len(REFUSAL_JOBS)} jobs")
    return REFUSAL_JOBS, SUCCESS_JOBS


@app.cell
def _(REFUSAL_JOBS, SUCCESS_JOBS):
    import json
    # steer_batch reads a JSON list of argv tails and drives them all under one load.
    for _name, _js in (("success", SUCCESS_JOBS), ("refusal", REFUSAL_JOBS)):
        open(f"/tmp/jobs_{_name}.json", "w").write(json.dumps(_js))
    BATCH = ["--batch-size", "32", "--max-batch-tokens", "65536"]
    print("job files written;", " ".join(BATCH))
    return (BATCH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.4 — `steer_single.py`, on baseline **successes**  ·  GPU

    Restore refusal: ablate story/persona, add +α to harm/eval, plus `cap` as the graded
    alternative, the no-op zero point and the matched-random control.
    """)
    return


@app.cell
def _(BATCH, DECODE_ARGS, MODEL, TAG, sh):
    sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
       "--script", "steer_single", "--jobs", "/tmp/jobs_success.json", *BATCH, *DECODE_ARGS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.5 — `steer_induce.py`, on baseline **refusals**  ·  GPU

    The mirror image: add story/persona at +α, ablate harm/eval. No `cap` — it is scoped
    to successes only (spec 5.4a).
    """)
    return


@app.cell
def _(BATCH, DECODE_ARGS, MODEL, TAG, sh):
    sh("python", "experiments/steering_jailbreaks/steer_batch.py", MODEL, "--tag", TAG,
       "--script", "steer_induce", "--jobs", "/tmp/jobs_refusal.json", *BATCH, *DECODE_ARGS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.6 — `steer_pairs.py`  ·  GPU

    Does direction `a` still work once `b` is projected out of it? Four arms per ordered
    pair at `steer_band`. `(story_v2, story_v1)` is refused as a pair — they are the same
    axis. The script reports `cos(a, b)` against the null band first and stops if the
    projection removes nothing.
    """)
    return


@app.cell
def _(BATCH, DECODE_ARGS, MODEL, TAG, sh):
    for _p in ["story_v2,persona", "story_v2,harm", "story_v2,eval", "persona,harm"]:
        sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL, "--tag", TAG,
           "--pair", _p, "--both-orders", "--layers", "steer_band", *BATCH, *DECODE_ARGS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.3 — judge every steered cell  ·  **API**

    The big one: ~6,900 calls at ~1.1k in / 0.25k out, roughly $2.30 on `gpt-4o-mini`.
    The judge sees the *bare* request, never the jailbreak wrapper.

    Safe to interrupt and re-run. Rows already graded by this judge are skipped, and a
    row whose API call landed but whose write did not comes back from the cache for free.
    Watch for `n_judged < n` in the run line — that is the judge declining to grade, and
    it depresses ASR rather than showing up as an error.
    """)
    return


@app.cell
def _(JUDGE_CONC, JUDGE_MODEL, sh, ungraded):
    _gens = ungraded()
    print(f"{len(_gens)} generation files to grade")
    for _i, _f in enumerate(_gens, 1):
        print(f"[{_i}/{len(_gens)}] {_f.name}")
        sh("python", "experiments/steering_jailbreaks/judge_strongreject.py", str(_f),
           "--judge-model", JUDGE_MODEL, "--concurrency", JUDGE_CONC)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5.8 — `aggregate.py`  ·  CPU

    The cross-cell join, reading `*_summary.csv` only. Writes `aggregate_cells.csv`,
    `_controls.csv` (each target beside its no-op and random arms), `_paired.csv`
    (necessity beside sufficiency) and `_decoding.csv`. At this tag it refuses to rank
    layer configs or αs: at ~30 rows clustered to `template_id` the ordering is noise.
    """)
    return


@app.cell
def _(MODEL, SJ_CSV, TAG, sh):
    sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
    print((SJ_CSV / "aggregate_controls.csv").read_text(encoding="utf-8")[:4000])
    return


@app.cell
def _(TAG, mo, sh):
    sh("tar", "-cf", f"/tmp/steering_{TAG}.tar", "--exclude=_archive",
       "-C", "experiments/steering_jailbreaks/results", TAG)
    mo.download(open(f"/tmp/steering_{TAG}.tar", "rb").read(), filename=f"steering_{TAG}.tar")
    return


if __name__ == "__main__":
    app.run()
