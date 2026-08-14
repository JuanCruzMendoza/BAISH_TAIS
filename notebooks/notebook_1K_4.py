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
    # `openai` is the judge's SDK, and OpenRouter speaks the same protocol so the fallback
    # needs no second package. Named explicitly rather than relied on: molab happens to ship
    # it, and 1_run's judging only worked because of that.
    sh("pip", "install", "-q", "transformers", "accelerate", "numpy", "matplotlib",
       "huggingface_hub", "hf_xet", "openai")
    sh("python", "-c",
       "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))")

    sys.path.insert(0, REPO)
    return HF_REPO, MODEL, REPO, TAG, mo, os, sh


@app.cell(hide_code=True)
def _(TAG, mo):
    # cell 2
    mo.md(f"""
    # {TAG} — steering_jailbreaks, **4_run** (projection, §5.6)

    Spec: `experiments/steering_jailbreaks/dev.md`, *{TAG} → 4_run*. Extraction,
    `probe_jailbreak_detection` and runs 1–3 are done and are only *read* here.

    **The question.** Does direction `a` still move behaviour once `b` is projected out of
    it at the same layer? The pair that motivates the pass is `story_v2_1k` × `persona_v2`
    at **L15**: story@L15 is the only story cell with a non-trivial effect (−13.9 restore,
    +9.1 induce), it sits at persona's own chosen layer, and `cos = +0.137` there — so a
    persona contaminant is a live alternative explanation for H1's one positive result.

    **16 cells, 7,528 generations**, ≈1.3 h GPU and ≈1 h of judging. Four ordered pairs ×
    two prompt sets × two arms:

    | arm | vector | ‖v‖ | asks |
    |---|---|---|---|
    | `perp_alpha` | `(û_a − c·û_b) / √(1−c²)` | 1 | **necessity** — does `a` work with `b` gone? |
    | `par_component` | `c·û_b` | **\\|c\\|** | **sufficiency** — does `b`'s share alone do it? |

    `unprojected` is **not generated**: `single_twin` resolves it to the `steer_single` /
    `steer_induce` cell runs 1–2 already produced at the same direction, layer, α and prompt
    set, so the reference is free. `perp_effect` is not run either — `α_eff = α₀/√(1−c²)` is
    +1% to +5% here, under the ±3pp noise floor.

    **Batch parameters are pinned to `32 / 65536`.** `single_twin` compares them, and greedy
    is bit-reproducible only at fixed batch composition — a mismatch makes it refuse the
    twin (loudly) rather than pair the arms against a reference built differently.

    ## Resume

    Every cell below is guarded on an artefact, not on a checkbox: a finished steering cell
    is skipped by its manifest, a graded cell by its `_summary.csv`. **Re-running the whole
    notebook after a kill is the intended way to resume** — and when a whole stage is
    already done it is skipped *before* the model load, so a judge-only session needs no GPU
    at all.

    ## Keys

    Paste the HF token and at least one judge key. Nothing downstream runs until they are
    set, which is also what stops a session opened for something else from starting a
    multi-hour sweep: marimo runs every cell on load, and a password field is empty on a
    fresh load.

    7,528 calls fit under a 10,000/day cap — but only if nothing else on the account spent
    it first. Paste the OpenRouter key and that stops mattering: when the OpenAI key is
    spent the judge continues on the *same* `gpt-4o-mini`, so the cache and the per-row
    resume stamps are unchanged and nothing already graded is re-graded.
    """)
    return


@app.cell
def _(mo):
    # cell 3
    HF_TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    OPENAI_KEY = mo.ui.text(label="OPENAI_API_KEY (judge, spec 5.3)", kind="password",
                            full_width=True)
    OPENROUTER_KEY = mo.ui.text(label="OPENROUTER_API_KEY (fallback judge, optional)",
                                kind="password", full_width=True)
    mo.vstack([HF_TOKEN, OPENAI_KEY, OPENROUTER_KEY])
    return HF_TOKEN, OPENAI_KEY, OPENROUTER_KEY


@app.cell(hide_code=True)
def _(mo):
    # cell 4
    mo.md(r"""
    ## Pull — only what 4_run needs

    Two scopes, both narrow, and both unchanged from 2_run.

    **extraction**: `vectors/` and `meta/directions__*` only, ~8 MB. 4_run needs **all four**
    direction files rather than the one being steered — `steer_pairs` loads `û_b` too, and
    reads it at `a`'s layer.

    **steering_jailbreaks**: everything. The judged baseline defines both prompt sets and
    must be 1_run's exact split; the manifests from runs 1–3 are what let `single_twin`
    resolve `unprojected` for free and what tell the judge loop below what is already
    graded; and `judge_cache.jsonl` makes a re-judge free.

    `pack=False` on both sides: steering's resume partials have to stay individually
    addressable, and a packed push re-sends the whole tar on every tick.

    ## Commits — still 2 per push

    Commits are the scarce resource, not bytes, and `upload_folder` emits **one commit per
    256 files considered** — considered, not changed. Steering keeps 4 files per cell
    (`.jsonl`, `_manifest.json`, `_judged.jsonl`, `_summary.csv`):

    | | cells | files in scope | commits per push |
    |---|---|---|---|
    | 1_run | 36 | 154 | 1 |
    | + 2_run | 76 | 314 | 2 |
    | + 3_run + **4_run** | **96** | **394** | **2** |

    394 is still under 512, so this pass costs exactly what 2_run's pushes cost. Budget:
    3 manual pushes (2 steer + 1 aggregate) + ~4 judge pushes + ~2 hourly ticks ≈ **18
    commits** over a ~2.5 h session.

    Because the batching is by file and not by stem, a 429 between the two parts can land a
    manifest on the Hub without its `.jsonl`. `ST_COMPLETE` therefore requires **both**, or a
    torn push would make a cell look finished and silently drop it from the judge pass and
    from `aggregate`.

    `meta/_archive/*` is in `IGNORE`, so re-running a cell does not grow the walk. The push
    is staged — mirrored into a staging directory and `.jsonl` files trimmed to their last
    complete row — so no file is measured while `steer_pairs` is still appending to it.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, MODEL, OPENAI_KEY, OPENROUTER_KEY, REPO, TAG, mo, os):
    # cell 5
    import json as _json
    import pathlib as _pl

    from experiments.common import ckpt

    # Either judge key is enough to start -- the OpenAI one may already be spent for the
    # day, and refusing to run then would defeat the fallback.
    mo.stop(not HF_TOKEN.value or not (OPENAI_KEY.value or OPENROUTER_KEY.value),
            mo.md("*Paste the HF token and at least one judge key. The HF token restores "
                  "the run and checkpoints it; the judge key grades it.*"))
    # judge_strongreject reads these from the environment of its subprocess, which inherits
    # this one. Set only when non-empty: an empty string reads as *present* downstream, so
    # a blank box would arm a fallback that cannot authenticate.
    for _var, _val in (("OPENAI_API_KEY", OPENAI_KEY.value),
                       ("OPENROUTER_API_KEY", OPENROUTER_KEY.value)):
        if _val:
            os.environ[_var] = _val
        else:
            os.environ.pop(_var, None)
    print("judge keys:", ", ".join(k for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")
                                   if os.environ.get(k)) or "none")

    EX_SCOPE = {"experiment": "extraction", "tag": TAG}
    ST_SCOPE = {"experiment": "steering_jailbreaks", "tag": TAG}
    # The judge's *daily* request cap. 4_run needs ~7,530 calls, which fits -- but only if
    # nothing else on the account spent today's quota first, which is why the fallback is
    # still worth arming. It only sizes the estimate printed before the pass.
    JUDGE_RPD = 10_000
    # Push the judge pass every this many cells. Judged rows are the costliest artefact in
    # the run -- real money and daily quota -- so they should not sit only on local disk
    # waiting for the hourly timer. At 16 cells this is 3-4 pushes, ~7 commits.
    JUDGE_PUSH_EVERY = 5
    ST_ROOT = _pl.Path(REPO, "experiments/steering_jailbreaks/results", TAG,
                       MODEL.replace("/", "_"))
    ST_META, ST_CSV = ST_ROOT / "meta", ST_ROOT / "csv"

    ckpt.setup(HF_REPO, token=HF_TOKEN.value)
    # ckpt.pull handles a size mismatch itself, inside its own attempts loop: local
    # `.incomplete` partials are deleted (their metadata kept, so files that did land are not
    # re-fetched), and a mismatch that survives that is a broken record on the Hub, which it
    # works around by fetching file by file and skipping those.
    print("extraction:", ckpt.pull(HF_REPO, token=HF_TOKEN.value,
                                   subpaths=["*/vectors/**", "*/meta/directions__*"],
                                   **EX_SCOPE))
    print("steering:  ", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))
    # The timer is what makes the checkpoint mid-run: a steer_pairs invocation generates two
    # cells over ~15 min and pushes only when it returns. Idempotent per scope, so re-running
    # this cell replaces its timer instead of leaking another thread.
    ST_TIMER = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
    print("hourly checkpoint armed for steering_jailbreaks")

    def ST_COMPLETE(stem):
        """`complete` manifest **and** its rows on disk (spec 0.11).

        Both halves are load-bearing: the steering scope is past 256 files, so every push
        splits into two commits -- and a 429 between the parts can land a manifest without
        the `.jsonl` beside it, since `upload_folder` batches by file, not by stem. A
        manifest-only skip would then drop that cell from the judge pass and from
        `aggregate` without saying so.
        """
        if not (ST_META / f"{stem}.jsonl").exists():
            return False
        try:
            return _json.loads((ST_META / f"{stem}_manifest.json")
                               .read_text(encoding="utf-8")).get("status") == "complete"
        except (OSError, ValueError):        # missing, or a torn tail mid-push
            return False

    st_pulled = True
    return (JUDGE_PUSH_EVERY, JUDGE_RPD, ST_COMPLETE, ST_CSV, ST_META, ST_SCOPE, ST_TIMER,
            ckpt, st_pulled)


@app.cell(hide_code=True)
def _(mo):
    # cell 6
    mo.md(r"""
    ## The prompt view — CPU, seconds

    `sets.py` rebuilds both prompt sets from `acts/views/jailbreaks__all.json` and checks
    every prompt against its `prompt_sha16`. That view lives in *extraction's* tree and was
    never pushed there, so it is rebuilt locally rather than pulled: `--view-only` writes it
    from the tokenizer alone, no weights, and reproduces the GPU path's `view_key` exactly.
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
    two sets have to be identical to 1_run's, both because the new cells must be comparable
    to the old ones and because `single_twin` compares `n_rows` before it will hand back a
    reference.

    The cell below regenerates it only if the Hub had nothing, which would mean the pull went
    wrong, so it says so loudly first.
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

    Four ordered pairs, all at **L15**, all `add`. α is the anchor's largest `deg`-clean ASR
    delta from runs 1–2 — story −0.75 / +0.25, persona −0.50 / +0.25 — and its **sign is
    derived, never typed**: `RESTORE_SIGN[a] × SET_SIGN[prompt_set]`, imported from `cell`
    and `steer_pairs` so it cannot drift from what the script itself computes. That is also
    what keeps a success cell and a refusal cell at the same |α| from colliding on one stem
    (`a-0.5` vs `a0.5`).

    The cell below computes each stem the same way `steer_pairs` does and checks three
    things before any GPU work:

    1. **which arms are already complete**, so a pair with both done is dropped from the run
       list entirely and costs no model load;
    2. **that the `unprojected` twin exists** for every config — it is the reference the whole
       experiment is read against, and it is not regenerated here;
    3. **that L15 has its no-op on both sets**. `steer_pairs` emits no `noop` arm, so the
       pairing is against 1_run's, which must be present.

    A missing twin or no-op is printed, not raised: the arms are still worth generating, but
    the analysis is not readable until the reference is back.
    """)
    return


@app.cell
def _(MODEL, ST_COMPLETE, mo):
    # cell 11
    from experiments.common import config as _cfg, manifest as _mf
    from experiments.steering_jailbreaks import cell as _cell, steer_pairs as _sp

    LAYER = 15
    # perp_alpha and par_component only. `unprojected` resolves to the existing steer_single
    # / steer_induce twin (single_twin), and perp_effect is alpha0/sqrt(1-c^2) away from
    # perp_alpha -- +1% to +5% at these cosines, under the noise floor.
    ARMS = ("perp_alpha", "par_component")
    # (a, b): b is projected out of a, read at a's layer. cos at L15 -- story/persona
    # +0.137, persona/eval +0.296, persona/harm -0.240; all clear the +/-0.050 null band.
    PAIRS = (("story_v2_1k", "persona_v2"), ("persona_v2", "story_v2_1k"),
             ("persona_v2", "eval_v2"), ("persona_v2", "harm_v2"))
    # |alpha| per (prompt set, anchor): the largest ASR delta that anchor reached at <=15%
    # degeneracy in runs 1-2. story restore stops at 0.75 because 1.00/1.25 are 41%/89%
    # degenerate; persona induce stops at 0.25 because 0.50 is already past its peak.
    AMAG = {("success", "story_v2_1k"): 0.75, ("success", "persona_v2"): 0.50,
            ("refusal", "story_v2_1k"): 0.25, ("refusal", "persona_v2"): 0.25}
    # Pinned, and compared by single_twin: greedy is bit-reproducible only at fixed batch
    # composition, so the reference and the arms have to be generated under the same numbers.
    BATCH = ("--batch-size", "32", "--max-batch-tokens", "65536")

    def _alpha(pset, a):
        """The signed alpha steer_pairs will compute for this (set, anchor)."""
        return _cell.RESTORE_SIGN[a] * _sp.SET_SIGN[pset] * AMAG[(pset, a)]

    def _stem(pset, a, b, arm):
        al = _alpha(pset, a)
        return _mf.stem("steer_pairs", f"{a}-perp-{b}", _cfg.layer_stem(str(LAYER)),
                        f"a{al:g}", arm)

    def _twin(pset, a):
        """The 5.4/5.5 cell steer_pairs will reuse as `unprojected`."""
        return _cell.stem_for(_sp.OWNER[pset], a, "add", str(LAYER), _alpha(pset, a),
                              None, "target")

    PAIR_JOBS, ST_ALL_STEMS, _lines, _warn = {}, [], [], []
    for _pset in ("success", "refusal"):
        _pending, _rows = [], []
        for _a, _b in PAIRS:
            _al = _alpha(_pset, _a)
            _stems = [_stem(_pset, _a, _b, _arm) for _arm in ARMS]
            ST_ALL_STEMS += _stems
            _done = sum(map(ST_COMPLETE, _stems))
            if _done < len(ARMS):
                _pending.append((_a, _b, abs(_al)))
            if not ST_COMPLETE(_twin(_pset, _a)):
                _warn.append(f"missing `unprojected` twin for {_pset} {_a}: "
                             f"{_twin(_pset, _a)}")
            _rows.append(f"| `{_a}` -> `{_b}` | {_al:+g} | {_done}/{len(ARMS)} |")
        _noop = f"{'steer_single' if _pset == 'success' else 'steer_induce'}__noop__L{LAYER}"
        if not ST_COMPLETE(_noop):
            _warn.append(f"missing no-op for {_pset}: {_noop} -- nothing to pair against")
        PAIR_JOBS[_pset] = _pending
        _lines.append(f"**{_pset}** — {len(PAIRS)} pairs, **{len(_pending)} pending**\n\n"
                      "| a -> b | alpha | arms done |\n|---|---|---|\n" + "\n".join(_rows))

    mo.md(f"{len(ST_ALL_STEMS)} cells in 4_run ({len(PAIRS)} pairs x 2 sets x "
          f"{len(ARMS)} arms)\n\n" + "\n\n".join(_lines)
          + ("\n\n**! " + "**\n\n**! ".join(_warn) + "**" if _warn else
             "\n\nAll `unprojected` twins and both L15 no-ops are present."))
    return ARMS, BATCH, LAYER, PAIR_JOBS, ST_ALL_STEMS


@app.cell(hide_code=True)
def _(mo):
    # cell 12
    mo.md(r"""
    ## Steer — GPU

    One `steer_pairs` invocation per pending (set, ordered pair), each generating its two
    arms. **This is the one place the notebook is less efficient than runs 1–3**: there is no
    `steer_batch` driver for `steer_pairs`, so the model loads once per pair rather than once
    per set — up to 8 loads, ~1 min each against ~80 min of generation. Pairs whose arms are
    both complete are dropped before that, so a resumed session pays only for what is left.

    Inside an invocation the usual guarantees hold: `mf.Run(resumable=True)` resumes per row,
    so a kill costs at most one batch, and an arm that is already complete cache-hits its
    rows instead of regenerating them.

    Each invocation also runs two prefill passes over the prompt set before generating — the
    unsteered baseline readout and the reference's self-effect, both recorded in the
    manifest. Seconds, no sampling.
    """)
    return


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, LAYER, MODEL, PAIR_JOBS, ST_SCOPE, TAG, ckpt, sh,
      st_split):
    # cell 13
    assert st_split                      # the success set is defined by the judged baseline
    _jobs = PAIR_JOBS["success"]
    for _i, (_a, _b, _mag) in enumerate(_jobs, 1):
        print(f"[{_i}/{len(_jobs)}] success  {_a} - proj({_b})  |alpha| {_mag:g}")
        sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL,
           "--tag", TAG, "--prompt-set", "success", "--pair", f"{_a},{_b}",
           "--layers", str(LAYER), "--alpha", f"{_mag:g}",
           "--arms", "perp_alpha,par_component", *BATCH)
    if _jobs:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="4_run pairs success", **ST_SCOPE)
    else:
        print("steer_pairs success: all cells complete -- skipped, no model load")
    st_pairs_success = True
    return (st_pairs_success,)


@app.cell
def _(BATCH, HF_REPO, HF_TOKEN, LAYER, MODEL, PAIR_JOBS, ST_SCOPE, TAG, ckpt, sh,
      st_pairs_success):
    # cell 14
    assert st_pairs_success
    _jobs = PAIR_JOBS["refusal"]
    for _i, (_a, _b, _mag) in enumerate(_jobs, 1):
        print(f"[{_i}/{len(_jobs)}] refusal  {_a} - proj({_b})  |alpha| {_mag:g}")
        sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL,
           "--tag", TAG, "--prompt-set", "refusal", "--pair", f"{_a},{_b}",
           "--layers", str(LAYER), "--alpha", f"{_mag:g}",
           "--arms", "perp_alpha,par_component", *BATCH)
    if _jobs:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="4_run pairs refusal", **ST_SCOPE)
    else:
        print("steer_pairs refusal: all cells complete -- skipped, no model load")
    st_steered = True
    return (st_steered,)


@app.cell(hide_code=True)
def _(mo):
    # cell 15
    mo.md(r"""
    ## Judge — API, and **two** rate limits

    One call per ungraded row — **7,528** for 4_run's sixteen cells, ≈$2.4. Two separate
    ceilings bind, and they need different handling:

    | limit | measured | what it costs | response |
    |---|---|---|---|
    | **tokens per minute** | 200k TPM on `gpt-4o-mini`, ~1.2k tokens a call ≈ 165 calls/min | the pass runs at ~1 h whatever the concurrency | `--concurrency 6`. 8 workers sat exactly on the ceiling and 429'd; 6 leaves ~25% headroom. Transient 429/5xx retry with full jitter, 7 attempts reaching ~61s — one full window |
    | **requests per day** | **10,000 RPD** | 7,528 fits — *if* nothing else spent today's quota | no backoff outlives a day, so it is not retried. **With an OpenRouter key it is not fatal either.** Without one, `judge_strongreject` exits **3** |

    **The fallback keeps this a one-day job even if the cap is part-spent.** On an RPD 429,
    an `insufficient_quota` 429 or a 402, the judge switches to OpenRouter and continues with
    the *same* `gpt-4o-mini`. Because the judge's identity is the model, the cache key
    (`[request, response, model, template_sha]`) and the per-row resume stamp
    (`judge_model`, `template_sha`) are untouched — so **nothing already graded is
    re-graded**, and a row graded through one endpoint is a valid cache hit for the other.
    `judge_provider` records the endpoint per row and never enters a key.

    Exit 3 breaks the loop; a single other non-zero exit is one bad cell and the loop carries
    on; two in a row is treated as a wall and also stops.

    **The stop is reactive, deliberately so.** `JUDGE_RPD` only sizes the estimate — nothing
    counts requests client-side, because this process cannot know how much of today's cap
    runs 1–3 or anything else on the account already spent. A counter starting at zero each
    session would stop early and waste quota, so the provider's own 429 is the authority.

    A cell is skipped when its `csv/<stem>_summary.csv` says every row was graded. The
    summary is written only at the end of a pass, so a cell killed midway re-enters and
    resumes per row. Runs 1–3's cells are skipped by the same test.

    **Checkpointed every `JUDGE_PUSH_EVERY` cells.** Judged rows are the costliest artefact
    in the run — real money and daily quota — so leaving them on local disk until the hourly
    timer fires puts up to an hour of grading on a disconnection. At 16 cells that is 3–4
    pushes, ~7 commits.

    The judge sees the **bare request**, never the jailbreak wrapper, and the deterministic
    detectors run alongside it at no cost — `outcome` is degenerate when *either* says so.
    Read `pct_degenerate` first on the `par_component` cells: they push a different axis than
    the anchor, at a magnitude no earlier cell tested.
    """)
    return


@app.cell
def _(HF_REPO, HF_TOKEN, JUDGE_PUSH_EVERY, JUDGE_RPD, ST_ALL_STEMS, ST_CSV, ST_META,
      ST_SCOPE, ckpt, sh, st_steered):
    # cell 16
    import csv as _csv
    import json as _js
    import os as _os

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
    # the _judged.jsonl outputs; neither has a manifest. gen_baseline is judged in its own
    # cell above.
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
    print(f"judging {len(_todo)} cells, {_new} of them 4_run's "
          f"({len(_mine) - _new}/{len(_mine)} of 4_run already graded)\n"
          f"{_need:,} rows still ungraded, against a {JUDGE_RPD:,}/day request cap "
          f"-> {-(-_need // JUDGE_RPD)} day(s)")
    if _need > JUDGE_RPD and not _os.environ.get("OPENROUTER_API_KEY"):
        print(f"! this pass CANNOT finish today on the OpenAI key alone. It will grade "
              f"~{JUDGE_RPD:,} rows, stop on the daily cap, and resume here tomorrow -- "
              f"re-run the notebook, or paste an OpenRouter key to carry straight on.")
    elif _need > JUDGE_RPD:
        print(f"  over the {JUDGE_RPD:,}/day cap, so expect a switch to OpenRouter partway "
              f"through; it grades the remainder with the same model.")

    # Two different failures, two different responses. `allow_fail` because judging is
    # resumable per row, so one bad cell should cost a retry of that cell rather than the
    # rest of the pass -- but exit 3 is the *daily* cap, which every remaining cell would hit
    # in turn, so it breaks out instead of failing every cell in a row.
    #
    # The belt to that brace: exit 3 depends on a regex over the error string
    # (`requests per day|RPD`), so a reworded 429 would read as transient, grind 7 retries a
    # row, and exit 1. Two consecutive failures of ANY kind therefore also stop the pass --
    # one bad cell is a bad cell, two in a row is a wall, whatever it is called.
    #
    # `_since` counts cells graded but not yet pushed; pushing every JUDGE_PUSH_EVERY caps
    # what a disconnection can cost to roughly that many cells of grading.
    _failed, _capped, _done_rows, _streak, _since = [], False, 0, 0, 0
    for _i, _p in enumerate(_todo, 1):
        print(f"[{_i}/{len(_todo)}] {_p.stem}  ({_left[_p.stem]:,} rows left)")
        _rc = sh("python", "experiments/steering_jailbreaks/judge_strongreject.py",
                 str(_p), "--concurrency", "6", allow_fail=True).returncode
        # Before the break checks: a cell that hit the daily cap still graded rows on its
        # way there, and those are exactly the ones worth not losing.
        _since += 1
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
        if _since >= JUDGE_PUSH_EVERY:
            ckpt.try_push(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE,
                          msg=f"4_run judged, {_i}/{len(_todo)} cells")
            _since = 0
    # Whatever the last partial batch left, including the cells a break stopped after.
    # Nothing to push when the loop never ran, so a verification re-run does not spend two
    # commits walking 394 unchanged files.
    if _since:
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="4_run judged", **ST_SCOPE)
    if _failed:
        print(f"! {len(_failed)} cells did not judge, re-run this cell: {_failed}")
    if _capped:
        _also = (" Both the OpenAI key and the OpenRouter fallback are spent."
                 if _os.environ.get("OPENROUTER_API_KEY") else
                 " Paste an OpenRouter key to carry on without waiting.")
        print(f"\n! DAILY REQUEST CAP -- stopped cleanly after ~{_done_rows:,} rows, "
              f"~{_need - _done_rows:,} to go.{_also} Everything graded is on the Hub; "
              f"re-run the notebook and the GPU cells will all skip.")
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
        # aggregate.py spans the tag, so this re-reports runs 1-3 alongside 4_run's pairs.
        sh("python", "experiments/steering_jailbreaks/aggregate.py", MODEL, "--tag", TAG)
        _ok = ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="4_run aggregate", **ST_SCOPE)
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
    cell of all four runs, `aggregate_controls.csv` each target with `d_*_vs_noop`.

    Read in this order:

    1. **`pct_degenerate` first**, as always — and here especially on the `par_component`
       cells, which push an axis the anchor's own α ladder never tested.
    2. **Each arm against its `unprojected` twin**, not against the no-op directly: the twin
       is the reference the projection is defined relative to. `steer_pairs` prints which
       stem it resolved to, and `cos_ab_band` / `push_frac` in each manifest say what was
       removed and at what magnitude.
    3. **`perp_alpha` ≈ reference ⇒ `b` is not necessary. `par_component` ≈ reference ⇒ `b`'s
       share is sufficient.** They are not an algebraic split, so both can be true or neither.
    4. **`read_<axis>` for all four axes.** On `perp_alpha` the pushed vector has *zero* `b`
       content by construction — if `read_b` still moves, the contamination was never
       geometric and no same-layer projection could have removed it.

    Two readings fixed in advance:

    - **`persona_v2` → `eval_v2` is a sign control.** `cos = +0.296`, but eval is a refusal
      axis, so restoring persona pushes eval at −0.148 — eval's *inducing* sign, against the
      −94.6 it would have to explain. `par_component` should move ASR the **wrong** way. If
      it does not, the arm is not doing what the decomposition says.
    - **`persona_v2` → `story_v2_1k` is a method control**, answered by arithmetic already
      (story alone at full α gets −13.9, so 14% of it cannot produce −94.6).

    The persona-anchored references are saturated (ASR 1.6 at α=−0.5), so partial attribution
    is unrecoverable there — only the qualitative necessity read is, and it is bimodal enough
    to survive: effect intact ⇒ ASR stays near 2, effect carried by `b` ⇒ ASR jumps toward 96.

    Still **no `random` arm** at this tag, so nothing here is a specificity claim.
    """)
    return


if __name__ == "__main__":
    app.run()
