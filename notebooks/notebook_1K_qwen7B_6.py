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

    # IDENTICAL to runs 1-5, and here it is load-bearing twice over: `single_twin` compares
    # batch parameters before it will hand back a reference, and greedy is bit-reproducible
    # only at fixed batch composition. steer_pairs defaults to 8 / 16384, so these must be
    # passed explicitly or it refuses the twin -- loudly, but only after a model load.
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
    # {TAG} on `{MODEL}` — projection at **L18** (§5.6)

    A **continuation** of runs 1–5. Nothing here re-runs; every stem this notebook writes is new.

    **The question.** 5_run found story@**L18** restores **−46.5 ΔASR** at 2.4% degeneracy — 3.3×
    story@L15 and the largest story cell in the study. 4_run had already shown that story@L15's
    smaller effect was largely its **14% persona component**: `par_component` alone reached −35.3
    against the full push's −13.9. L18 is the cell that most needs the same test, and the component
    it would be carried by is *bigger* here:

    | convention | cos |
    |---|---|
    | `story@L18 · persona@L18` — what `steer_pairs` uses | **+0.177** |
    | `story@L15 · persona@L15` — 4_run's | +0.137 |

    `steer_pairs` reads `û_b` at `a`'s layer, so the projection removes a 17.7%-norm persona
    component. If L18 behaves like L15, that sliver alone will beat the full story vector again.

    ## Cells

    One ordered pair, `story_v2_1k` − proj(`persona_v2`), at L18, two sets, two arms:

    | arm | vector | ‖v‖ | asks |
    |---|---|---|---|
    | `perp_alpha` | `(û_a − c·û_b) / √(1−c²)` | 1 | **necessity** — does story work with persona gone? |
    | `par_component` | `c·û_b` | **0.177** | **sufficiency** — does persona's share alone do it? |

    | set | n | α | arms |
    |---|---|---|---|
    | success | 508 | **−0.75** | `perp_alpha`, `par_component` |
    | refusal | 433 | **+0.25** | `perp_alpha`, `par_component` |

    **4 cells, 1,882 generations**, ≈25 min GPU. α magnitudes are 5_run's two best `deg`-clean
    cells (restore −46.5 at 2.4%, induce +11.3 at 5.1%) — the same two cells the §5.9 narrativity
    check ran on, so all three measurements sit on one pair of cells. Signs are derived from
    `cell.RESTORE_SIGN × steer_pairs.SET_SIGN`, never typed.

    **Nothing else is generated.** `unprojected` resolves to 5_run's `steer_single` /
    `steer_induce` twin at the same direction, layer, α and set, so the reference is free;
    `perp_effect` is `α₀/√(1−c²)` from `perp_alpha`, **+1.6%** at this cosine and under the noise
    floor; and both L18 no-ops already exist. L18 is in band (L11–25), so no out-of-band opt-in.

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

    LAYER = 18
    # (a, b): b is projected out of a, read at a's layer.
    PAIR = ("story_v2_1k", "persona_v2")
    # perp_alpha and par_component only -- see the overview for why the other two are skipped.
    ARMS = ("perp_alpha", "par_component")
    # |alpha| per set: story@L18's largest deg-clean cell from 5_run.
    AMAG = {"success": 0.75, "refusal": 0.25}

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
    print(f"{MODEL}: L={N_LAYERS}, band {BAND[0]}-{BAND[-1]}")
    print(f"pair: {PAIR[0]} - proj({PAIR[1]})  at L{LAYER}"
          f"{'' if LAYER in BAND else '  (OUT OF BAND)'}")
    print("arms: " + ", ".join(ARMS)
          + " | |alpha| " + ", ".join(f"{k} {v:g}" for k, v in AMAG.items()))
    return (AMAG, ARMS, BAND, DONE, EX_SCOPE, JB_ROOT, JB_SCOPE, LAYER, N_LAYERS, PAIR,
            ST_COMPLETE, ST_ROOT, ST_SCOPE, ckpt)


@app.cell
def _(EX_SCOPE, HF_REPO, HF_TOKEN, JB_SCOPE, ST_SCOPE, ckpt):
    # cell 5
    # Steering reads `vectors/` and the jailbreak view -- never the 2.2 GiB pole cache. So
    # extraction is pulled *small*: no `*/acts/**`, and in particular not blobs.tar.
    # BOTH direction files are needed, not just the one being steered: steer_pairs loads
    # `û_b` as well and reads it at `a`'s layer.
    print("extraction (vectors + meta):",
          ckpt.pull(HF_REPO, token=HF_TOKEN.value, **EX_SCOPE,
                    subpaths=["*/vectors/**", "*/meta/**"]))
    print("probe_jailbreak_detection:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **JB_SCOPE))
    # Brings back gen_baseline_judged.jsonl (the prompt sets) and every finished cell, so
    # `single_twin` can resolve `unprojected` against 5_run's L18 cells for free.
    print("steering_jailbreaks:", ckpt.pull(HF_REPO, token=HF_TOKEN.value, **ST_SCOPE))
    pulled = True
    return (pulled,)


@app.cell
def _(DONE, JB_ROOT, MODEL, TAG, pulled, sh):
    # cell 6
    assert pulled
    # --view-only writes acts/views/jailbreaks__all.json from the tokenizer alone: no
    # weights, no activations, and it reproduces the GPU path's view_key exactly, so no
    # run_key moves. jb_readout is complete at this tag, so this is always the cheap path.
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
    # 1_run's split. `single_twin` compares `n_rows` before it will hand back a reference, so
    # a different split here would not silently mis-pair: it would refuse.
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
def _(AMAG, ARMS, LAYER, PAIR, ST_COMPLETE, mo, st_split):
    # cell 8
    from experiments.common import config as _cfg, manifest as _mf
    from experiments.steering_jailbreaks import cell as _cell, steer_pairs as _sp

    assert st_split

    _a, _b = PAIR

    def _alpha(pset):
        """The signed alpha steer_pairs will compute for this set."""
        return _cell.RESTORE_SIGN[_a] * _sp.SET_SIGN[pset] * AMAG[pset]

    ST_PENDING, ST_ALL_STEMS, _lines, _warn = {}, [], [], []
    for _pset in ("success", "refusal"):
        _al = _alpha(_pset)
        _stems = [_mf.stem("steer_pairs", f"{_a}-perp-{_b}", _cfg.layer_stem(str(LAYER)),
                           f"a{_al:g}", _arm) for _arm in ARMS]
        ST_ALL_STEMS += _stems
        _done = sum(map(ST_COMPLETE, _stems))
        ST_PENDING[_pset] = _done < len(ARMS)
        # The two artefacts this run reads but does not build. A missing one is printed, not
        # raised: the arms are still worth generating, but they are unreadable without the
        # reference they are defined relative to.
        _twin = _cell.stem_for(_sp.OWNER[_pset], _a, "add", str(LAYER), _al, None, "target")
        _noop = f"{_sp.OWNER[_pset]}__noop__L{LAYER}"
        for _what, _s in (("`unprojected` twin", _twin), ("no-op", _noop)):
            if not ST_COMPLETE(_s):
                _warn.append(f"missing {_what} for {_pset}: `{_s}`")
        _lines.append(f"**{_pset}** — α {_al:+g}, {_done}/{len(ARMS)} arms done\n\n"
                      + "\n".join(f"- `{s}`" for s in _stems)
                      + f"\n- reference: `{_twin}`\n- no-op: `{_noop}`")

    mo.md(f"{len(ST_ALL_STEMS)} cells, {sum(ST_PENDING.values())} set(s) pending\n\n"
          + "\n\n".join(_lines)
          + ("\n\n**! " + "**\n\n**! ".join(_warn) + "**" if _warn else
             "\n\nBoth twins and both L18 no-ops are present."))
    return ST_ALL_STEMS, ST_PENDING


@app.cell(hide_code=True)
def _(mo):
    # cell 9
    mo.md(r"""
    ## Steer — GPU

    One `steer_pairs` invocation per pending set, each generating both arms. There is no
    `steer_batch` driver for `steer_pairs`, so the model loads once per set rather than once
    for the notebook — 2 loads, ~1 min each against ~25 min of generation. A set whose arms are
    both complete is skipped before the load.

    Inside an invocation `mf.Run(resumable=True)` resumes per row, so a kill costs at most one
    batch. Each invocation also runs two prefill passes first — the unsteered baseline read-out
    and the reference's self-effect, both recorded in the manifest. Seconds, no sampling.
    """)
    return


@app.cell
def _(AMAG, BATCH, HF_REPO, HF_TOKEN, LAYER, MODEL, PAIR, ST_PENDING, ST_SCOPE, TAG, ckpt,
      sh):
    # cell 10
    _timer = ckpt.autopush(HF_REPO, every=3600, token=HF_TOKEN.value, **ST_SCOPE)
    try:
        for _pset in ("success", "refusal"):
            if not ST_PENDING[_pset]:
                print(f"steer_pairs {_pset}: both arms complete -- skipped, no model load")
                continue
            print(f"\n=== {_pset}: {PAIR[0]} - proj({PAIR[1]}) at L{LAYER}, "
                  f"|alpha| {AMAG[_pset]:g} ===")
            sh("python", "experiments/steering_jailbreaks/steer_pairs.py", MODEL,
               "--tag", TAG, "--prompt-set", _pset, "--pair", ",".join(PAIR),
               "--layers", str(LAYER), "--alpha", f"{AMAG[_pset]:g}",
               "--arms", "perp_alpha,par_component", *BATCH)
    finally:
        _timer.set()
    if any(ST_PENDING.values()):
        ckpt.try_push(HF_REPO, token=HF_TOKEN.value, msg="L18 pairs", **ST_SCOPE)
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
        raise RuntimeError(f"{len(_missing)} cell(s) did not complete: {_missing}")
    print(f"all {len(ST_ALL_STEMS)} cells complete; pushing")
    print(ckpt.push(HF_REPO, token=HF_TOKEN.value, msg="L18 pairs complete", **ST_SCOPE))
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

    {len(ST_ALL_STEMS)} cells, 1,882 rows, ≈1,882 judge calls (~20 min at `--concurrency 6`,
    ≈$0.6). The list is explicit rather than a `steer_*` glob: runs 1–5 left 116 graded cells in
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
    python -c "from experiments.common import ckpt; ckpt.push('$R', experiment='steering_jailbreaks', tag='$T', msg='L18 pairs judged')"
    ```

    ## How to read it

    1. **`pct_degenerate` first, and hardest on `par_component`** — it pushes persona, an axis
       whose own α ladder at L15 was ≥96% degenerate by α=1.25, at a magnitude no L18 cell has
       tested.
    2. **Each arm against its `unprojected` twin, not against the no-op.** The twin is what the
       projection is defined relative to: `steer_single__story_v2_1k__add__L18__a-0.75`
       (−46.5 ΔASR) and `steer_induce__story_v2_1k__add__L18__a0.25` (+11.3). `steer_pairs`
       prints which stem it resolved to, and `cos_ab_band` / `push_frac` in each manifest say
       what was removed and at what magnitude.
    3. **`perp_alpha` ≈ twin ⇒ persona is not necessary. `par_component` ≈ twin ⇒ persona's
       share is sufficient.** Not an algebraic split — both can hold, or neither. 4_run found
       them strongly **sub-additive** at L15 (perp −10.3 and par −35.3 against a −13.9
       reference, each exceeding the whole), so do not add them or read either as a percentage.
    4. **`read_persona_v2` on `perp_alpha`.** That vector has *zero* persona content by
       construction; if `read_persona` still moves, the contamination was never geometric and no
       same-layer projection could have removed it.

    **The prediction, fixed in advance.** At L15 the 14% persona sliver beat the full story push
    in 3 of 4 comparisons. At L18 the sliver is **17.7%** and the reference is 3.3× larger. If
    `par_component` again beats the twin, L18's −46.5 is persona reached through a wider overlap
    and 5_run's headline does not survive. If `perp_alpha` holds near −46.5, story@L18 is the
    first genuinely story-carried steering result at this tag.

    Still **no `random` arm**, so neither outcome is a specificity claim — a surviving
    `perp_alpha` would still not separate story from any direction of that norm at L18.
    """)
    return


if __name__ == "__main__":
    app.run()
