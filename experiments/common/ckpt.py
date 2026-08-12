"""Checkpoint results/ to a private HF dataset repo, mid-run.

molab kills a notebook at 12h and after 90min idle, and /marimo reaches its R2 store
only on a graceful stop -- so a crash loses whatever never left the machine. push() is
the guarantee: the results are on the Hub the moment it returns. pull() puts them back
in the tree the scripts read, so run_key resume skips the cells already done.
"""
import os
import tarfile
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from experiments.common import config as cfg

# Across commits the Hub only dedups LFS/Xet-backed files; plain files are re-sent every
# time. Neither extension is in the Hub's default .gitattributes and both sit under the
# 10 MB auto-LFS threshold, so these lines have to be committed before the first push.
#   *.npy   acts/blobs, ~204 KB x 915: written once, so this is pure re-upload avoided.
#   *.jsonl meta/, the resume partials, 30 MB and append-only: Xet dedups at chunk level,
#           so only the appended tail moves instead of all 30 MB on every tick.
# blobs.tar needs no line: at 1.6 GB it is far over the auto-LFS threshold.
LFS_LINES = ["*.npy filter=lfs diff=lfs merge=lfs -text",
             "*.jsonl filter=lfs diff=lfs merge=lfs -text"]

ALLOW = ["*/results/**"]                # every experiment; scope() narrows it
# vectors/*.pt are INCLUDED: 2.1 MB each at n=800, 8.4 MB for four, one commit. They were
# excluded back when lopo_d was stored in them (~335 MB each) and the constraint looked like
# bytes; it is commits. The alternative was a 1.6 GB blob download plus an extract_direction
# rebuild at the start of every downstream session, which is what experiments 2-5 would
# otherwise pay to get a direction they cannot compute themselves.
#
# *.tar is ignored in the folder walk because pack=True uploads it by itself, and because a
# stale export must not ride along; *.tar does not match *.tar.gz.
IGNORE = ["*/meta/_archive/*", "*.tar", "*.tar.gz"]

# Two overlapping pushes race on the commit they build.
_lock = threading.Lock()

# One live timer per scope. A reactive notebook re-runs the cell that arms the timer
# whenever any UI input it reads changes -- a pasted token, a toggled checkbox -- and each
# run used to start another thread while the previous one kept its own reference and was
# never stopped. Measured on the 1_run steering pass: three threads, 8 hourly commits where
# there should have been 3. Keyed by scope so extraction and steering can each hold one.
_timers = {}


def scope(experiment=None, tag=None, subpaths=None):
    """Which paths a push or pull touches. Always pass the experiment being run.

    upload_folder does NOT make one commit per call: it batches at
    _commit_api.UPLOAD_BATCH_MAX_NUM_FILES (256) and emits one commit per batch, titled
    "<msg> (part N)". Measured on the 1K extraction run, 12 push() calls produced 167
    commits, growing 8 -> 30 per call as the tree grew. Commits are the scarce resource
    (an undocumented hourly quota, and a 429 mid-split leaves a cell half-uploaded), so
    what matters is how many files a push considers.

    ALLOW spans every experiment, so an unscoped push from steering_jailbreaks also
    considers extraction's ~7,700 blobs and costs ~30 commits before writing anything of
    its own. Scoped, its ~500 files cost ~3.

    `subpaths` narrows below the tag, each one glob-relative to <model_slug>'s parent, e.g.
    "*/vectors/**". Only useful for pull(): a downstream experiment needs a few of an
    upstream one's directories and none of its csv/ or figures/. Never narrow a push --
    what a scoped push omits, it also cannot restore.
    """
    if experiment is None:
        return ALLOW
    base = f"{experiment}/results/{tag}/" if tag else f"{experiment}/results/"
    return [base + s for s in subpaths] if subpaths else [base + "**"]


def _token(token=None):
    if token:
        return token
    cfg.load_env()
    return os.environ.get("HF_TOKEN")


def setup(repo_id, token=None):
    api = HfApi(token=_token(token))
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    try:
        p = api.hf_hub_download(repo_id, ".gitattributes", repo_type="dataset")
        cur = Path(p).read_text(encoding="utf-8")
    except EntryNotFoundError:
        cur = ""
    missing = [l for l in LFS_LINES if l not in cur]
    if missing:
        body = ("\n".join([cur.rstrip("\n"), *missing]) + "\n").lstrip("\n")
        api.upload_file(path_or_fileobj=body.encode(), path_in_repo=".gitattributes",
                        repo_id=repo_id, repo_type="dataset",
                        commit_message="lfs-track the checkpoint payload")
    return api


# ------------------------------------------------------- packed blob cache
# One tar is one file is one commit, versus ~30 for 7,700 loose blobs. Use it for a new
# extraction run (another model, another tag); do NOT use it for steering, whose ~500
# files already cost ~3 commits and whose resume partials must stay individually
# addressable. push(pack=True) and pull(pack=True) are a pair -- packing one side only
# leaves a repo nobody can restore.
BLOB_TAR = "blobs.tar"


def _blob_dirs(root, experiment=None, tag=None):
    pat = f"{experiment or '*'}/results/{tag or '*'}/*/acts/blobs"
    return [p for p in root.glob(pat) if p.is_dir()]


def pack_blobs(root=None, experiment=None, tag=None):
    """acts/blobs/*.npy -> acts/blobs.tar. Uncompressed: float16 barely gzips."""
    root = Path(root or cfg.REPO / "experiments")
    out = []
    for d in _blob_dirs(root, experiment, tag):
        tar = d.parent / BLOB_TAR
        files = sorted(d.glob("*.npy"))
        tmp = tar.with_suffix(".tar.tmp")
        with tarfile.open(tmp, "w") as tf:
            for p in files:
                tf.add(p, arcname=p.name)
        os.replace(tmp, tar)
        out.append((tar, len(files)))
    return out


def unpack_blobs(root=None, experiment=None, tag=None, keep_tar=False):
    """acts/blobs.tar -> acts/blobs/, skipping blobs already on disk.

    Idempotent, and never overwrites: a partially cached local run keeps its own blobs
    and only gains the ones it was missing.
    """
    root = Path(root or cfg.REPO / "experiments")
    out = []
    for tar in root.glob(f"{experiment or '*'}/results/{tag or '*'}/*/acts/{BLOB_TAR}"):
        d = tar.parent / "blobs"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        with tarfile.open(tar) as tf:
            for m in tf:
                if m.isfile() and not (d / m.name).exists():
                    tf.extract(m, d, filter="data")
                    n += 1
        if not keep_tar:
            tar.unlink()
        out.append((tar, n))
    return out


def push(repo_id, root=None, token=None, msg="ckpt", experiment=None, tag=None,
         pack=False):
    root = Path(root or cfg.REPO / "experiments")
    api = HfApi(token=_token(token))
    ignore = list(IGNORE)
    with _lock:
        if pack:
            # The tar goes up on its own (IGNORE already drops *.tar from the folder
            # walk), and the loose blobs are held back so the two never diverge.
            for tar, n in pack_blobs(root, experiment, tag):
                api.upload_file(path_or_fileobj=str(tar),
                                path_in_repo=tar.relative_to(root).as_posix(),
                                repo_id=repo_id, repo_type="dataset",
                                commit_message=f"{msg}: {BLOB_TAR} ({n} blobs)")
                tar.unlink()
            ignore.append("*/acts/blobs/*")
        return api.upload_folder(
            folder_path=str(root), repo_id=repo_id, repo_type="dataset",
            allow_patterns=scope(experiment, tag), ignore_patterns=ignore,
            commit_message=msg)


def try_push(repo_id, root=None, token=None, msg="ckpt", experiment=None, tag=None,
             pack=False):
    """push() that warns instead of raising.

    Every push is one commit, and the Hub enforces an undocumented per-window quota on
    commits: `upload_folder` waits out the RateLimit header and then gives up after 5
    retries with a 429. That must not kill a cell whose GPU pass already finished -- the
    blobs are on local disk and the next push picks them up, so a skipped checkpoint
    costs nothing unless the instance dies first. Keep push() for callers that want the
    guarantee.
    """
    try:
        return push(repo_id, root=root, token=token, msg=msg,
                    experiment=experiment, tag=tag, pack=pack)
    except Exception as e:                                            # noqa: BLE001
        print(f"! checkpoint SKIPPED ({msg}): {type(e).__name__}: {str(e)[:300]}")
        return None


def _is_rate_limit(e):
    """429 reaches us two ways, and only one of them is an HfHubHTTPError.

    The Xet path is Rust and surfaces as a bare RuntimeError -- "Network error: Request
    error: HTTP status client error (429 Too Many Requests) ... /xet-read-token/<sha>" --
    so the type is not a reliable discriminator and the string has to be.
    """
    return "429" in str(e) or "too many requests" in str(e).lower()


def _is_size_mismatch(e):
    """A download that came out the wrong length. Xet's wording and the plain path's.

    Measured on a resumed 2_run session: "Task error: File size mismatch: expected 238254
    bytes but downloaded 335267 bytes", and 335267 - 238254 = 97013 -- exactly the bytes
    already sitting in the .incomplete file. The Xet writer is Rust and reconstructs the
    whole file, but `_download_to_tmp_and_move` hands it an incomplete_path it does not
    truncate, so a leftover partial is prepended to the real content.
    """
    s = str(e).lower()
    return "size mismatch" in s or "consistency check failed" in s


def _pull_one_by_one(repo_id, root, token, patterns, ignore):
    """Fetch each file alone and skip any whose stored object is unusable -> [skipped].

    `snapshot_download` is all-or-nothing, so a single broken record on the Hub stops the
    whole pull -- and the premise of the checkpoint scheme is that a killed session can just
    be re-run.

    A record breaks this way when a push reads a file a *subprocess* is appending to: the
    size is stat'd mid-row and committed against a pointer covering the longer content, so
    the object downloads at its real length and fails the check. Measured on 2_run:
    steer_induce__persona_v2__add__L15__a1.jsonl recorded 238,254 with 335,267 stored, and
    byte 238,254 lands mid-line. By construction such a file is the resume partial of a cell
    that was still generating, so dropping it costs a regeneration, not a result -- and its
    manifest is `in_progress`, which no consumer treats as complete anyway.
    """
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi(token=token)
    want = filter_repo_objects(api.list_repo_files(repo_id, repo_type="dataset"),
                               allow_patterns=patterns, ignore_patterns=ignore)
    skipped = []
    for f in want:
        try:
            api.hf_hub_download(repo_id, f, repo_type="dataset", local_dir=str(root))
        except Exception as e:                                        # noqa: BLE001
            if not _is_size_mismatch(e):
                raise
            skipped.append(f)
    return skipped


def _clear_incomplete(root):
    """Delete the local-dir download partials. Returns how many went.

    They have to be *deleted*, not retried: the partial is keyed by etag, so an unchanged
    remote file resolves to the same .incomplete every time and the mismatch reproduces
    exactly. Only download bookkeeping lives here -- never a result -- and the sibling
    metadata is left alone so completed files are not re-fetched.
    """
    n = 0
    for p in Path(root).glob(".cache/huggingface/download/**/*.incomplete"):
        p.unlink(missing_ok=True)
        n += 1
    return n


def pull(repo_id, root=None, token=None, experiment=None, tag=None, pack=False,
         subpaths=None, attempts=6, max_workers=4):
    """False if nothing is checkpointed yet, so a first run falls through to computing.

    Scope this too: reads are cheap per request but not unlimited, and a loose blob cache
    is one request per blob -- extraction's 7,731 of them exceed the read quota on a free
    account and come back 429, from the xet-read-token endpoint rather than from the file
    downloads. Hence `max_workers=4` (below the library's 8, to spread the burst) and the
    retry: snapshot_download resumes, skipping whatever already landed, so each attempt
    starts where the last one stopped and the download completes across several windows.

    A size mismatch is handled in two steps -- clear the local partials, then, if the record
    on the Hub is what is broken, fetch file by file and skip the unfetchable. See
    `_is_size_mismatch` and `_pull_one_by_one`.

    The real fix for a blob cache this size is not to store it loose: see `pack`.

    `subpaths` narrows further inside one experiment.

    `pack=True` unpacks any acts/blobs.tar into acts/blobs/ afterwards, so the tree the
    scripts read is the same either way and run_key resume still cache-hits. Safe on a
    repo that stores loose blobs: with no tar present it is a no-op.
    """
    root = Path(root or cfg.REPO / "experiments")
    root.mkdir(parents=True, exist_ok=True)
    patterns, ignore = scope(experiment, tag, subpaths), [".gitattributes"]
    repaired = False
    for i in range(attempts):
        try:
            snapshot_download(repo_id, repo_type="dataset", local_dir=str(root),
                              token=_token(token), max_workers=max_workers,
                              allow_patterns=patterns, ignore_patterns=ignore)
            break
        except RepositoryNotFoundError:
            return False
        except Exception as e:                                        # noqa: BLE001
            if _is_size_mismatch(e):
                # The cheap cause first: a killed session leaves an .incomplete partial
                # that a resumed download can prepend to the real bytes. Deleting it costs
                # nothing when that was not the cause.
                #
                # Do NOT reach for HF_HUB_DISABLE_XET here. Measured against the broken
                # 2_run file on hub 1.7.1: Xet fetched all 335,267 bytes and the *plain*
                # path raised "Consistency check failed", because it compares against the
                # size the HEAD reports. Xet is lenient on some versions and strict on
                # others, so turning it off can only lose a download that would have
                # worked. (The variable is also read into constants at import, so setting
                # os.environ from a notebook cell never took effect anyway.)
                if not repaired and i < attempts - 1:
                    repaired = True
                    n = _clear_incomplete(root)
                    print(f"! size mismatch ({i + 1}/{attempts}): deleted {n} "
                          f".incomplete partial(s), retrying. {str(e)[:160]}")
                    continue
                # Nothing stale left locally, so it is the record on the Hub that is
                # broken. Take everything else rather than lose the session.
                skipped = _pull_one_by_one(repo_id, root, _token(token), patterns, ignore)
                print(f"! {len(skipped)} file(s) have a broken size record on the Hub and "
                      f"were SKIPPED; everything else is local now:")
                for f in skipped:
                    print(f"    {f}")
                print("  A record breaks this way only on a file a push read while a "
                      "subprocess was appending to it, so each is the resume partial of a "
                      "cell that never finished. Those cells regenerate.")
                break
            if not _is_rate_limit(e) or i == attempts - 1:
                raise
            wait = min(300, 30 * 2 ** i)
            print(f"! rate-limited ({i + 1}/{attempts}), resuming in {wait}s: "
                  f"{str(e)[:160]}")
            time.sleep(wait)
    if pack:
        for tar, n in unpack_blobs(root, experiment, tag):
            print(f"unpacked {n} new blobs from {tar.name}")
    return True


def autopush(repo_id, every=600, root=None, token=None, experiment=None, tag=None,
             pack=False):
    """Returns an Event; set it to stop. A timer is what makes the checkpoint mid-run:
    steer_batch.py is one process for 93 cells, so a per-script push never fires inside
    it.

    `pack` has to match whatever the manual pushes for this experiment use. A packed repo
    and an unpacked tick would leave both representations on the Hub, and pull() would
    then unpack a tar over blobs that no longer came from it. The cost is re-tarring the
    cache on every tick, which is why `every` should not be small when packing.

    Idempotent per scope: arming a second timer for the same (experiment, tag) stops the
    first, so re-running the arming cell replaces its timer instead of adding one.
    """
    key = (experiment, tag)
    prior = _timers.pop(key, None)
    if prior is not None and not prior.is_set():
        prior.set()
        print(f"  timer for {experiment}/{tag or '-'} re-armed; stopped the previous one")
    stop = threading.Event()
    _timers[key] = stop

    def loop():
        while not stop.wait(every):
            try:
                push(repo_id, root=root, token=token, msg="auto",
                     experiment=experiment, tag=tag, pack=pack)
            except Exception as e:
                print("ckpt failed:", type(e).__name__, e)

    threading.Thread(target=loop, daemon=True).start()
    return stop
