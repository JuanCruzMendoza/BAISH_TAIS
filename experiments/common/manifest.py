"""Run keys, archive-on-write, run log, resume gate (spec 0.10 / 0.11)."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

CANON = dict(sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, **CANON).encode()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def git_env():
    diff = _git("diff", "HEAD")
    return {"git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(diff.strip()),
            "git_diff_sha": hashlib.sha256(diff.encode()).hexdigest() if diff.strip() else None}


def stem(script, *parts):
    """<script>__<knob>__<knob> (spec 0.1). Semantic knobs only."""
    keep = [str(p) for p in parts if p not in (None, "")]
    return "__".join([Path(script).stem, *keep])


class Run:
    """Manifest lifecycle: in_progress at start, complete at exit.

    Also owns archive-on-write and the resume gate. Use as a context manager;
    an exception leaves the manifest at in_progress, which is what
    check_stale.py reports as an interrupted run.
    """

    def __init__(self, out_dir, stem_, config, inputs, resumable=False):
        self.dir = Path(out_dir)
        self.stem = stem_
        self.config = config
        self.inputs = inputs
        self.resumable = resumable
        self.run_key = sha256_obj({"config": config, "inputs": inputs})
        self.path = self.dir / f"{self.stem}_manifest.json"
        self.t0 = time.time()
        self.resumed_from = 0
        self.notes = []

    # ------------------------------------------------------------- lifecycle

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        prior = self._read_manifest()
        if prior is not None and prior.get("run_key") != self.run_key:
            self._archive(prior)
        self._write("in_progress")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self._write("complete")
            self._log("written" if self.resumed_from == 0 else "resumed")
        else:
            self._log("failed")
        return False

    # ------------------------------------------------------------ artefacts

    def artefact(self, suffix):
        return self.dir / f"{self.stem}{suffix}"

    def resume_from(self, suffix=".jsonl"):
        """Completed unit ids from a matching partial (spec 0.11).

        Returns an empty set unless the partial's run_key equals ours; a
        mismatched partial has already been archived by __enter__.
        """
        path = self.artefact(suffix)
        if not (self.resumable and path.exists()):
            return set()
        done = set()
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["unit_id"])
                except (json.JSONDecodeError, KeyError):
                    break  # truncated tail: stop, redo from here
        self.resumed_from = len(done)
        return done

    def open_append(self, suffix=".jsonl"):
        return self.artefact(suffix).open("a", encoding="utf-8")

    # -------------------------------------------------------------- internals

    def _read_manifest(self):
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _archive(self, prior):
        arch = self.dir / "_archive"
        arch.mkdir(exist_ok=True)
        tag = (prior.get("run_key") or "unknown")[:8]
        for p in self.dir.glob(f"{self.stem}*"):
            if p.is_dir():
                continue
            suffix = p.name[len(self.stem):]
            os.replace(p, arch / f"{self.stem}__{tag}{suffix}")
        self.notes.append(f"archived prior run {tag}")

    def _write(self, status):
        payload = {"stem": self.stem, "status": status, "run_key": self.run_key,
                   "config": self.config, "inputs": self.inputs,
                   "env": {**git_env(), "argv": sys.argv, "python": sys.version.split()[0],
                           "platform": platform.platform(), "host": platform.node(),
                           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "wall_s": round(time.time() - self.t0, 2),
                           "resumed_from": self.resumed_from},
                   "notes": self.notes}
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, **{k: v for k, v in CANON.items()
                                                        if k != "separators"}), encoding="utf-8")
        os.replace(tmp, self.path)

    def _log(self, outcome):
        log = self.dir / "runs.csv"
        new = not log.exists()
        env = git_env()
        with log.open("a", encoding="utf-8", newline="") as f:
            if new:
                f.write("timestamp,script,stem,run_key,git_sha,git_dirty,outcome,"
                        "resumed_from,wall_s,argv\n")
            f.write(",".join([time.strftime("%Y-%m-%dT%H:%M:%S"), Path(sys.argv[0]).name,
                              self.stem, self.run_key[:16], env["git_sha"][:12],
                              str(env["git_dirty"]), outcome, str(self.resumed_from),
                              f"{time.time() - self.t0:.1f}",
                              '"' + " ".join(sys.argv[1:]).replace('"', "'") + '"']) + "\n")


def load_upstream(path):
    """Read an upstream manifest and refuse it unless status == complete."""
    path = Path(path)
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("status") != "complete":
        raise RuntimeError(f"{path.name}: status={m.get('status')!r}, not consumable (spec 0.11)")
    return m
