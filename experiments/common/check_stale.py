"""Report artefacts that are stale, interrupted, or built on a missing upstream (spec 0.11).

    python -m experiments.common.check_stale <model>
"""
import json
import sys
from pathlib import Path

from . import config as cfg


def scan(model_id):
    root = cfg.REPO / "experiments"
    manifests = {}
    for p in root.glob(f"*/results/{cfg.model_slug(model_id)}/*_manifest.json"):
        try:
            manifests[p] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifests[p] = {"status": "unreadable"}

    by_key = {m.get("run_key"): p for p, m in manifests.items() if m.get("run_key")}
    views = {}
    for p in root.glob(f"*/results/{cfg.model_slug(model_id)}/acts/views/*.json"):
        v = json.loads(p.read_text(encoding="utf-8"))
        views[v["view_key"]] = p

    interrupted, stale, orphaned = [], [], []
    for p, m in manifests.items():
        if m.get("status") != "complete":
            interrupted.append((p, m.get("status")))
            continue
        inputs = m.get("inputs") or {}
        for field, val in inputs.items():
            if field.endswith("view_key") and val and val not in views:
                orphaned.append((p, f"{field}={val[:12]} not on disk"))
            if field.endswith("run_key") and val and val not in by_key:
                stale.append((p, f"{field}={val[:12]} no longer current"))
        for f in inputs.get("source_files", []):
            src = cfg.REPO / f["path"]
            if not src.exists():
                orphaned.append((p, f"missing input {f['path']}"))
            else:
                from . import manifest as mf
                if mf.sha256_file(src) != f["sha256"]:
                    stale.append((p, f"input changed: {f['path']}"))
    return interrupted, stale, orphaned


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m experiments.common.check_stale <model>")
    interrupted, stale, orphaned = scan(sys.argv[1])
    for title, items in (("INTERRUPTED (not consumable)", interrupted),
                         ("STALE (upstream changed)", stale),
                         ("ORPHANED (upstream missing)", orphaned)):
        print(f"\n{title}: {len(items)}")
        for p, why in items:
            print(f"  {Path(p).name}: {why}")
    if not (interrupted or stale or orphaned):
        print("\nall artefacts current")
    return 1 if (interrupted or stale or orphaned) else 0


if __name__ == "__main__":
    sys.exit(main())
