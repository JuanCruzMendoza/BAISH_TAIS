"""Checkpoint results/ to a private HF dataset repo, mid-run.

molab kills a notebook at 12h and after 90min idle, and /marimo reaches its R2 store
only on a graceful stop -- so a crash loses whatever never left the machine. push() is
the guarantee: the results are on the Hub the moment it returns. pull() puts them back
in the tree the scripts read, so run_key resume skips the cells already done.
"""
import os
import threading
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
LFS_LINES = ["*.npy filter=lfs diff=lfs merge=lfs -text",
             "*.jsonl filter=lfs diff=lfs merge=lfs -text"]

ALLOW = ["*/results/**"]
# vectors/*.pt rebuild from the cache in ~5s, but they are LFS by default and immutable,
# so they cost one upload and nothing after -- cheaper than remembering to re-run
# extract_direction. The tarballs are stale exports; *.tar does not match *.tar.gz.
IGNORE = ["*/meta/_archive/*", "*.tar", "*.tar.gz"]

# upload_folder builds one commit, so two overlapping pushes race on it.
_lock = threading.Lock()


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


def push(repo_id, root=None, token=None, msg="ckpt"):
    root = Path(root or cfg.REPO / "experiments")
    with _lock:
        return HfApi(token=_token(token)).upload_folder(
            folder_path=str(root), repo_id=repo_id, repo_type="dataset",
            allow_patterns=ALLOW, ignore_patterns=IGNORE, commit_message=msg)


def pull(repo_id, root=None, token=None):
    """False if nothing is checkpointed yet, so a first run falls through to computing."""
    root = Path(root or cfg.REPO / "experiments")
    root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id, repo_type="dataset", local_dir=str(root),
                          token=_token(token), ignore_patterns=[".gitattributes"])
    except RepositoryNotFoundError:
        return False
    return True


def autopush(repo_id, every=600, root=None, token=None):
    """Returns an Event; set it to stop. A timer is what makes the checkpoint mid-run:
    steer_batch.py is one process for 93 cells, so a per-script push never fires inside
    it."""
    stop = threading.Event()

    def loop():
        while not stop.wait(every):
            try:
                push(repo_id, root=root, token=token, msg="auto")
            except Exception as e:
                print("ckpt failed:", type(e).__name__, e)

    threading.Thread(target=loop, daemon=True).start()
    return stop
