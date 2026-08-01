"""Report artefacts that are stale, interrupted, or built on a missing upstream (spec 0.11).

    python -m experiments.common.check_stale <model> [tag]
"""
import json
import sys
from pathlib import Path

from . import config as cfg
from . import manifest as mf


def scan(model_id, tag=None):
    root = cfg.REPO / "experiments"
    pat = f"*/results/{cfg.tag(tag)}/{cfg.model_slug(model_id)}"

    manifests = {}
    for p in root.glob(f"{pat}/meta/*_manifest.json"):
        try:
            manifests[p] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifests[p] = {"status": "unreadable"}

    by_key = {m.get("run_key") for m in manifests.values() if m.get("run_key")}
    view_keys = set()
    for p in root.glob(f"{pat}/acts/views/*.json"):
        try:
            view_keys.add(json.loads(p.read_text(encoding="utf-8"))["view_key"])
        except (json.JSONDecodeError, KeyError):
            pass

    interrupted, stale, orphaned = [], [], []
    for p, m in manifests.items():
        if m.get("status") != "complete":
            interrupted.append((p, m.get("status")))
            continue
        inputs = m.get("inputs") or {}
        for field, val in inputs.items():
            if field.endswith("view_key") and isinstance(val, str) and val not in view_keys:
                orphaned.append((p, f"{field}={val[:12]} not on disk"))
            if field.endswith("run_key") and isinstance(val, str) and val not in by_key:
                stale.append((p, f"{field}={val[:12]} no longer current"))
        for f in inputs.get("source_files", []):
            src = cfg.REPO / f["path"]
            if not src.exists():
                orphaned.append((p, f"missing input {f['path']}"))
            elif mf.sha256_file(src) != f["sha256"]:
                stale.append((p, f"input changed: {f['path']}"))
    return manifests, interrupted, stale, orphaned


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m experiments.common.check_stale <model> [tag]")
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    manifests, interrupted, stale, orphaned = scan(sys.argv[1], tag)
    print(f"tag={cfg.tag(tag)}  model={sys.argv[1]}  artefacts={len(manifests)}")
    for title, items in (("INTERRUPTED (not consumable)", interrupted),
                         ("STALE (upstream changed)", stale),
                         ("ORPHANED (upstream missing)", orphaned)):
        print(f"\n{title}: {len(items)}")
        for p, why in items:
            print(f"  {Path(p).name}: {why}")
    if not manifests:
        print("\nno artefacts found for this tag")
        return 1
    if not (interrupted or stale or orphaned):
        print("\nall artefacts current")
    return 1 if (interrupted or stale or orphaned) else 0


if __name__ == "__main__":
    sys.exit(main())
