import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    mo.md(r"""
    # HF checkpoint mechanism — test

    Proves four things about `experiments/common/ckpt.py` on a throwaway dataset repo,
    with a synthetic results tree that mimics the real layout. No GPU, no model.

    | # | claim |
    |---|---|
    | 1 | `.npy` blobs and `.jsonl` partials are **dedup-backed** on the Hub |
    | 2 | the **second push is far cheaper** than the first (only new bytes move) |
    | 3 | `pull()` restores the tree **byte-identical** after a wipe |
    | 4 | `autopush()` checkpoints **mid-run**, with no explicit call |

    Nothing runs until you press the button. `push()` is additive by design — it never
    deletes on the Hub — so delete the test repo at the bottom before a second run.
    """)
    return (mo,)


@app.cell
def _(mo):
    import hashlib, pathlib, shutil, subprocess, sys, time

    NB = mo.notebook_dir()
    MOLAB = pathlib.Path("/marimo").exists()
    REPO = str(NB / "BAISH_TAIS") if MOLAB else str(NB)

    def sh(*a, cwd=REPO):
        p = subprocess.run(a, cwd=cwd, capture_output=True, text=True)
        print(p.stdout[-2000:], p.stderr[-1000:])
        if p.returncode:
            raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, a))}")
        return p

    if MOLAB and not pathlib.Path(REPO).exists():
        sh("git", "clone", "https://github.com/JuanCruzMendoza/BAISH_TAIS.git", REPO,
           cwd=str(NB))
    sh("pip", "install", "-q", "huggingface_hub", "hf_xet", "numpy")

    sys.path.insert(0, REPO)
    import numpy as np
    from experiments.common import ckpt
    print("repo:", REPO, "| molab:", MOLAB)
    return REPO, ckpt, hashlib, np, pathlib, shutil, time


@app.cell
def _(mo, pathlib):
    # A dataset repo of your own; created private, deleted at the bottom.
    TEST_REPO = "JuanCruzMendoza/baish-tais-ckpt-test"

    # 200 x ~200 KB = ~40 MB. The real acts/ is 915 blobs / 184 MB -- raise this to 915
    # if you want the true first-push cost.
    N_BLOBS, N_NEW = 200, 20

    ROOT = pathlib.Path(mo.notebook_dir()) / "_ckpt_test"
    RUN = ROOT / "extraction/results/ckpt_test/TestModel"

    TOKEN = mo.ui.text(label="HF_TOKEN (write)", kind="password", full_width=True)
    RUN_BTN = mo.ui.run_button(label="Run test")
    mo.vstack([TOKEN, RUN_BTN])
    return N_BLOBS, N_NEW, ROOT, RUN, RUN_BTN, TEST_REPO, TOKEN


@app.cell
def _(N_BLOBS, RUN, RUN_BTN, ROOT, TEST_REPO, TOKEN, ckpt, mo, np, shutil):
    mo.stop(not RUN_BTN.value, mo.md("*Press **Run test**.*"))

    api = ckpt.setup(TEST_REPO, token=TOKEN.value)
    print(".gitattributes committed; *.npy is LFS-tracked")

    def blob(i):
        """Deterministic in i, so the post-pull comparison is meaningful."""
        return np.random.default_rng(i).standard_normal((100, 512), dtype=np.float32)

    def rows(i, lo, hi):
        """meta/*.jsonl is the resume state and is append-only, ~120 KB per cell."""
        return "".join(f'{{"unit_id": "{i}_{k}", "text": "{"x" * 200}"}}\n'
                       for k in range(lo, hi))

    def build(lo, hi):
        for d in ("csv", "meta", "acts/blobs", "acts/views"):
            (RUN / d).mkdir(parents=True, exist_ok=True)
        for i in range(lo, hi):
            np.save(RUN / "acts/blobs" / f"{i:016x}.npy", blob(i))
            (RUN / "acts/views" / f"view_{i}.json").write_text(f'{{"blob": {i}}}')
            (RUN / "csv" / f"cell_{i}_summary.csv").write_text(f"cell,asr\n{i},0.5\n")
            (RUN / "meta" / f"cell_{i}_manifest.json").write_text(f'{{"run_key": "{i}"}}')
            (RUN / "meta" / f"cell_{i}.jsonl").write_text(rows(i, 0, 500))

    def grow(lo, hi, n=500):
        """Cells in flight when the timer fires: the tail is new, the head is not."""
        for i in range(lo, hi):
            with (RUN / "meta" / f"cell_{i}.jsonl").open("a") as f:
                f.write(rows(i, n, n + 100))

    shutil.rmtree(ROOT, ignore_errors=True)
    build(0, N_BLOBS)
    mb = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) / 1e6
    print(f"built {N_BLOBS} cells, {mb:.1f} MB")
    return api, blob, build, grow, mb


@app.cell
def _(ROOT, TEST_REPO, TOKEN, api, ckpt, mb, time):
    t0 = time.time()
    ckpt.push(TEST_REPO, root=ROOT, token=TOKEN.value, msg="push 1")
    T1 = time.time() - t0
    print(f"push 1: {T1:.1f}s for {mb:.1f} MB ({len(api.list_repo_files(TEST_REPO, repo_type='dataset'))} files)")
    return (T1,)


@app.cell
def _(T1, TEST_REPO, api, mo):
    # Claim 1. A blob stored plainly is re-sent in every commit -- all 184 MB of it.
    # Xet is the current backend and LFS the older one; either gives cross-commit dedup.
    rel = "extraction/results/ckpt_test/TestModel"
    probe = {"acts/blobs/*.npy": f"{rel}/acts/blobs/{0:016x}.npy",
             "meta/*.jsonl": f"{rel}/meta/cell_0.jsonl",
             "csv/*.csv": f"{rel}/csv/cell_0_summary.csv"}
    info = {p.path: p for p in
            api.get_paths_info(TEST_REPO, list(probe.values()), repo_type="dataset")}

    def deduped(f):
        return f.lfs is not None or getattr(f, "xet_hash", None) is not None

    rowsmd = "\n".join(
        f"| `{k}` | {info[v].size:,} | `{info[v].lfs is not None}` | "
        f"`{getattr(info[v], 'xet_hash', None) is not None}` | **{deduped(info[v])}** |"
        for k, v in probe.items())
    ok_lfs = all(deduped(info[probe[k]]) for k in ("acts/blobs/*.npy", "meta/*.jsonl"))
    mo.md(f"""
    **Claim 1 — {'PASS' if ok_lfs else 'FAIL'}**

    | path | size | lfs | xet | deduped |
    |---|---|---|---|---|
    {rowsmd}

    `.npy` (written once) and `.jsonl` (append-only, 30 MB in the real run) must both be
    deduped. The csv need not be — a few hundred tiny files. Baseline push 1 = {T1:.1f}s.
    """)
    return (ok_lfs,)


@app.cell
def _(N_BLOBS, N_NEW, ROOT, T1, TEST_REPO, TOKEN, build, ckpt, grow, mo, ok_lfs, time):
    # Claim 2: the state a real tick sees -- some cells finished, some grew mid-flight.
    build(N_BLOBS, N_BLOBS + N_NEW)
    grow(N_BLOBS - 10, N_BLOBS)
    t0 = time.time()
    ckpt.push(TEST_REPO, root=ROOT, token=TOKEN.value, msg="push 2")
    T2 = time.time() - t0
    ok_dedup = T2 < T1 / 2
    mo.md(f"""
    **Claim 2 — {'PASS' if ok_dedup else 'FAIL'}**

    push 1 = **{T1:.1f}s** ({N_BLOBS} cells) → push 2 = **{T2:.1f}s** (+{N_NEW} cells,
    10 jsonl grown). Ratio {T2 / T1:.2f}. Near 1 means the payload is being re-sent whole
    — check claim 1 ({ok_lfs}).
    """)
    return (T2,)


@app.cell
def _(N_BLOBS, N_NEW, ROOT, T2, TEST_REPO, TOKEN, ckpt, hashlib, mo, shutil):
    # Claim 3: kill the machine, get the tree back.
    def digest(root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.rglob("*")) if p.is_file()
                and ".cache" not in p.parts}

    before = digest(ROOT)
    shutil.rmtree(ROOT)
    assert not ROOT.exists()
    ckpt.pull(TEST_REPO, root=ROOT, token=TOKEN.value)
    after = digest(ROOT)

    missing = sorted(set(before) - set(after))
    differing = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    ok_pull = not missing and not differing
    mo.md(f"""
    **Claim 3 — {'PASS' if ok_pull else 'FAIL'}**

    {len(before)} files wiped ({5 * (N_BLOBS + N_NEW)} expected), {len(after)} restored.
    Missing: {len(missing)}. Content mismatches: {len(differing)}.
    Extras from an earlier run: {len(set(after) - set(before))}.
    {('First missing: `' + missing[0] + '`') if missing else ''}
    """)
    return (ok_pull,)


@app.cell
def _(ROOT, RUN, TEST_REPO, TOKEN, api, ckpt, mo, ok_pull, time):
    # Claim 4: a 20s timer, standing in for the 600s one a real run would use.
    stop = ckpt.autopush(TEST_REPO, every=20, root=ROOT, token=TOKEN.value)
    (RUN / "csv" / "written_mid_run_summary.csv").write_text("cell,asr\nmidrun,0.9\n")
    time.sleep(30)
    stop.set()

    landed = any(f.endswith("written_mid_run_summary.csv")
                 for f in api.list_repo_files(TEST_REPO, repo_type="dataset"))
    mo.md(f"""
    **Claim 4 — {'PASS' if landed else 'FAIL'}**

    A file written with no `push()` call reached the Hub within one 20s tick: `{landed}`.
    Timer stopped. Prior claim 3 = {ok_pull}.
    """)
    return (landed,)


@app.cell
def _(TEST_REPO, mo):
    KILL = mo.ui.run_button(label=f"Delete {TEST_REPO} and the local tree")
    KILL
    return (KILL,)


@app.cell
def _(KILL, ROOT, TEST_REPO, TOKEN, mo, shutil):
    mo.stop(not KILL.value, mo.md("*Test repo still on the Hub.*"))
    from huggingface_hub import HfApi
    HfApi(token=TOKEN.value or None).delete_repo(TEST_REPO, repo_type="dataset")
    shutil.rmtree(ROOT, ignore_errors=True)
    mo.md(f"Deleted `{TEST_REPO}` and `{ROOT.name}/`.")
    return


if __name__ == "__main__":
    app.run()
