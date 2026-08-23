"""
Flatten a steer_<direction>.jsonl into a spreadsheet CSV, with the response scored.

One row per generation. Drops the full response (kept as a 120-char preview) and adds the
readout used to judge the run in docs/initial_tests/insights.md 3:
    saturated      out_tokens hit the max_new cap -> length is a ceiling, not a measurement
    nonascii_frac  fraction of non-ASCII chars   -> collapse into non-English character salad
    rep_frac       1 - unique_words/total_words  -> repetition loop
    degenerate     nonascii_frac > 0.15 or rep_frac > 0.6 (auto; UNDERcounts, see insights 3)
Also writes a <name>_by_cell.csv pivot: mean proj / out_tokens / degenerate-count per
(layer, alpha), which is what the tables in docs/initial_tests/insights.md 3 are built from.

Usage:
    python steer_to_csv.py [steer_narrativity_orth.jsonl | /path/to.jsonl] [model_name]
Bare filename resolves under results/<model>/. Writes <name>.csv next to the input.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARG = sys.argv[1] if len(sys.argv) > 1 else "steer_narrativity_orth.jsonl"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-3B-Instruct"

path = ARG if os.path.isabs(ARG) or os.path.exists(ARG) else os.path.join(
    HERE, "results", MODEL.replace("/", "_"), ARG
)
if not os.path.exists(path):
    raise SystemExit(f"not found: {path}")
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def nonascii_frac(t):
    return sum(ord(c) > 127 for c in t) / max(len(t), 1)


def rep_frac(t):
    w = re.findall(r"\S+", t)
    return 1 - len(set(w)) / max(len(w), 1)


FIELDS = [
    "id", "source", "direction", "layer", "alpha", "raw_scale",
    "out_tokens", "saturated", "nar_proj_final", "hook_calls", "resid_ort_layer",
    "nonascii_frac", "rep_frac", "degenerate", "response_preview",
]

out = []
for r in rows:
    resp = r.get("response", "")
    na, rep = nonascii_frac(resp), rep_frac(resp)
    out.append({
        "id": r["id"], "source": r.get("source"), "direction": r.get("direction"),
        "layer": r["layer"], "alpha": r["alpha"], "raw_scale": r.get("raw_scale"),
        "out_tokens": r["out_tokens"],
        "saturated": int(r["out_tokens"] >= r.get("max_new_tokens", 10 ** 9)),
        "nar_proj_final": round(r["nar_proj_final"], 2), "hook_calls": r.get("hook_calls"),
        "resid_ort_layer": r.get("resid_ort_layer"),
        "nonascii_frac": round(na, 3), "rep_frac": round(rep, 3),
        "degenerate": int(na > 0.15 or rep > 0.6),
        "response_preview": " ".join(resp.split())[:120],
    })

out.sort(key=lambda x: (x["id"], (x["layer"] is None, x["layer"] or 0), x["alpha"]))
csv_path = os.path.splitext(path)[0] + ".csv"
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # -sig so Excel reads UTF-8
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(out)
print(f"{len(out)} rows -> {csv_path}")

# ---- pivot by (layer, alpha), steered cells only ----
steered = [x for x in out if x["layer"] is not None]
alphas = sorted(set(x["alpha"] for x in steered))
layers = sorted(set(x["layer"] for x in steered))
pivot_path = os.path.splitext(path)[0] + "_by_cell.csv"
with open(pivot_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["layer", "alpha", "n", "mean_proj", "mean_out_tokens", "n_degenerate", "n_saturated"])
    for L in layers:
        for a in alphas:
            m = [x for x in steered if x["layer"] == L and x["alpha"] == a]
            if not m:
                continue
            w.writerow([
                L, a, len(m),
                round(sum(x["nar_proj_final"] for x in m) / len(m), 2),
                round(sum(x["out_tokens"] for x in m) / len(m), 1),
                sum(x["degenerate"] for x in m),
                sum(x["saturated"] for x in m),
            ])
print(f"pivot -> {pivot_path}")
