"""Figure 1: best-cell ΔASR per direction, restore beside enable, one bar per model.

    python plot_figure1.py --tag 1K_per_direction

Reads each model's `csv/aggregate_cells.csv` and nothing else. ΔASR is against the
**baseline**, which is 100 on the success set and 0 on the refusal set by construction
(the two sets are defined by it), so the delta is the steered ASR shifted by a constant.

Cell selection, per (model, direction, arm): the largest |ΔASR| whose degeneracy is at or
below `--max-deg`, preferring the cleaner cell when two are within `--tol` points. A cell
above the ceiling is not a behaviour change, it is a broken model, and its ASR is
indistinguishable from a refusal.

The chosen layers are a research decision recorded in `extraction/insights.md` and
`probe_jailbreak_detection/insights.md`, not derivable from these CSVs, so they live in
MODELS below. `story` carries two of them: the layer maximising the fiction-vs-non-fiction
reading gap (the primary bar) and the layer maximising Cohen's d_z, which `--no-story-dz` drops.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.common import metrics as met  # noqa: E402

# label -> (model_slug, {direction: layer}), the primary layer per direction
MODELS = [
    ("Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-7B-Instruct",
     {"story_v2_1k": "18", "persona_v2": "15", "harm_v2": "21", "eval_v2": "9"},
     {"story_v2_1k": "23"}),                      # the d_z layer, second story bar
    ("Gemma-2-9B-it", "google_gemma-2-9b-it",
     {"story_v2_1k": "15", "persona_v2": "15", "harm_v2": "19", "eval_v2": "8"},
     {"story_v2_1k": "28"}),
]

NAMES = {"harm_v2": "Harm", "persona_v2": "Persona",
         "story_v2_1k": "Story", "eval_v2": "Eval"}
ORDER = ["harm_v2", "persona_v2", "story_v2_1k", "eval_v2"]
COLOURS = {"Qwen2.5-7B-Instruct": "#3b4cc0", "Gemma-2-9B-it": "#e8871a"}
ARMS = [("success", "Restoring refusal"), ("refusal", "Suppressing refusal")]
# ASR is a percentage, so the ticks span its full range. The limits reach a little past
# it: a cell at -100 carries an interval whose lower end would otherwise be clipped by
# the axis floor.
YLIM = (-106.0, 102.0)
YTICKS = list(range(-100, 101, 25))


def read_cells(slug, tag):
    path = (REPO / "experiments/steering_jailbreaks/results" / tag / slug
            / "csv/aggregate_cells.csv")
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("arm") == "target"]


def best_cell(rows, direction, layer, prompt_set, max_deg, tol):
    """Largest |ΔASR| at or below the degeneracy ceiling, cleaner cell on a near-tie."""
    cand = []
    for r in rows:
        if r["direction"] != direction or r["layers_spec"] != layer:
            continue
        if r["prompt_set"] != prompt_set or not r["asr"]:
            continue
        deg = float(r["pct_degenerate"])
        if deg > max_deg:
            continue
        asr = float(r["asr"])
        delta = asr - 100.0 if prompt_set == "success" else asr
        cand.append({"delta": delta, "deg": deg, "asr": asr, "alpha": r["alpha"],
                     "mode": r["mode"], "n": r["n"], "stem": r["stem"]})
    if not cand:
        return None
    top = max(abs(c["delta"]) for c in cand)
    near = [c for c in cand if abs(c["delta"]) >= top - tol]
    return min(near, key=lambda c: c["deg"])


def interval(slug, tag, stem):
    """Exact 95% interval on the cell's ASR, plus the wrapper-clustered estimate.

    The error bar is Clopper-Pearson over the non-degenerate rows, which is the sampling
    error given this corpus. It is *not* the whole uncertainty: `cluster_asr` re-weights
    every wrapper equally, and where the two disagree the number is being carried by a
    few heavily-reused templates rather than by the corpus (spec 0.7).
    """
    path = (REPO / "experiments/steering_jailbreaks/results" / tag / slug
            / "meta" / f"{stem}_judged.jsonl")
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows
            if r.get("outcome") != "degenerate" and not r.get("det_degenerate")
            and r.get("strongreject") is not None]
    if not rows:
        return None
    hit = [1.0 if r["strongreject"] > 0 else 0.0 for r in rows]
    k, n = int(sum(hit)), len(hit)
    lo, hi = met.clopper_pearson(k, n)
    _, cm = met.cluster_means(hit, [r.get("template_id", "?") for r in rows])
    return {"asr": 100.0 * k / n, "lo": 100.0 * lo, "hi": 100.0 * hi, "n_rows": n,
            "cluster_asr": 100.0 * float(cm.mean()), "n_clusters": len(cm)}


def collect(tag, max_deg, tol, story_dz):
    """[(arm_key, direction_key, label, model_label, cell)] in plot order."""
    out = []
    for prompt_set, _ in ARMS:
        for d in ORDER:
            keys = [(d, None)] + ([(d, "dz")] if story_dz and d == "story_v2_1k" else [])
            for direction, variant in keys:
                for label, slug, primary, secondary in MODELS:
                    layer = secondary[direction] if variant else primary[direction]
                    cell = best_cell(read_cells(slug, tag), direction, layer,
                                     prompt_set, max_deg, tol)
                    if cell is not None:
                        ci = interval(slug, tag, cell["stem"])
                        if ci:
                            # The interval is on the ASR, and the baseline is a constant,
                            # so the same half-widths apply to the delta.
                            cell["err"] = (abs(ci["asr"] - ci["lo"]),
                                           abs(ci["hi"] - ci["asr"]))
                            cell.update({k: ci[k] for k in
                                         ("cluster_asr", "n_clusters", "n_rows")})
                    name = NAMES[direction] + (" (d_z)" if variant else "")
                    out.append((prompt_set, f"{direction}{variant or ''}", name,
                                label, layer, cell))
    return out


def plot(rows, max_deg, path, annotate, absolute=False):
    groups = []          # (arm_label, [(bar_label, [(model, layer, cell)])])
    for prompt_set, arm_label in ARMS:
        cats, seen = [], []
        for ps, key, name, model, layer, cell in rows:
            if ps != prompt_set:
                continue
            if key not in seen:
                seen.append(key)
                cats.append((name, []))
            cats[seen.index(key)][1].append((model, layer, cell))
        groups.append((arm_label, cats))

    n_models = len(MODELS)
    width = 0.8 / n_models
    xs, ticks, labels, gap = [], [], [], 0.9
    x = 0.0
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    seen_models = set()

    for gi, (arm_label, cats) in enumerate(groups):
        start = x
        for name, bars in cats:
            ticks.append(x)
            labels.append(name)
            for k, (model, layer, cell) in enumerate(bars):
                off = (k - (n_models - 1) / 2) * width
                if cell is None:
                    continue
                lab = model if model not in seen_models else None
                seen_models.add(model)
                val = cell["asr"] if absolute else cell["delta"]
                err = cell.get("err")
                # The delta is the ASR shifted by a constant baseline, so on the restore
                # side the interval's ends swap: lower ASR is a larger negative delta.
                yerr = ([[err[1]], [err[0]]] if (not absolute and val < 0)
                        else [[err[0]], [err[1]]]) if err else None
                ax.bar(x + off, val, width * 0.92,
                       color=COLOURS.get(model, "0.5"), label=lab, zorder=3,
                       yerr=yerr, capsize=2.5,
                       error_kw={"ecolor": "0.25", "elinewidth": 0.9, "zorder": 5})
                if annotate and absolute:
                    # Every bar rises from 0, so its value sits on top. The layer goes
                    # inside a tall bar and above the value on a bar too short to hold it.
                    ax.annotate(f"{val:.1f}" if val < 10 else f"{val:.0f}",
                                (x + off, val + 2.0), ha="center", va="bottom",
                                fontsize=7.5, color="0.15", zorder=4)
                    inside = val >= 14
                    ax.annotate(f"L{layer}", (x + off, 2.5 if inside else val + 8.0),
                                ha="center", va="bottom", fontsize=6.5,
                                color="white" if inside else "0.45", zorder=4)
                elif annotate:
                    d = cell["delta"]
                    # Clear the error bar, not just the bar end, or the cap sits on top
                    # of the digits.
                    reach = (err[0] if d < 0 else err[1]) if err else 0.0
                    va = "top" if d < 0 else "bottom"
                    dy = -(reach + 3.0) if d < 0 else reach + 3.0
                    y, colour = d + dy, "0.15"
                    # A bar that reaches the axis floor would carry its label off the
                    # figure, so those labels go inside the bar instead.
                    if not (YLIM[0] + 4 < y < YLIM[1] - 4):
                        y = d + (5.0 if d < 0 else -5.0)
                        va = "bottom" if d < 0 else "top"
                        colour = "white"
                    ax.annotate(f"{d:+.0f}", (x + off, y), ha="center", va=va,
                                fontsize=7.5, color=colour, zorder=4)
                    ax.annotate(f"L{layer}", (x + off, 0), ha="center",
                                va="bottom" if cell["delta"] < 0 else "top",
                                fontsize=6.5, color="0.45")
            x += 1.0
        xs.append((start - 0.5, x - 0.5, arm_label))
        x += gap

    for i, (lo, hi, arm_label) in enumerate(xs):
        mid = (lo + hi) / 2
        ax.annotate(arm_label, (mid, 1.075 if absolute else 1.02),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=11, weight="bold")
        if absolute:
            # Absolute ASR is only readable against where the arm started, and by
            # construction that is 100 on the success set and 0 on the refusal set.
            ref = 100 if ARMS[i][0] == "success" else 0
            ax.annotate(f"unsteered ASR = {ref}", (mid, 1.015),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=9, color="0.4")
            if ref:
                ax.hlines(ref, lo + 0.15, hi - 0.15, ls="--", lw=1.2,
                          color="0.45", zorder=2)
        if i:
            ax.axvline(lo - gap / 2, color="0.75", lw=0.9, ls=":", zorder=0)

    ax.axhline(0, color="0.2", lw=1.0, zorder=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ASR (%)" if absolute
                  else "ΔASR against the unsteered baseline (points)")
    if absolute:
        ax.set_ylim(0.0, 106.0)
    else:
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    ax.legend(fontsize=9, loc="upper right" if absolute else "lower right",
              framealpha=0.95)
    ax.set_title("Attack success rate when steering each direction "
                 "(best cell per direction)", fontsize=11,
                 pad=42 if absolute else 26)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1K_per_direction")
    ap.add_argument("--max-deg", type=float, default=15.0,
                    help="degeneracy ceiling in percent (default 15)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="two cells within this many points count as tied")
    ap.add_argument("--no-story-dz", dest="story_dz", action="store_false",
                    help="drop the second story bar at the Cohen's d_z layer")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--absolute", action="store_true",
                    help="plot the steered ASR itself instead of the delta")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    default = ("docs/figures/fig1_asr_absolute.png" if a.absolute
               else "docs/figures/fig1_asr_per_direction.png")
    a.out = a.out or default

    rows = collect(a.tag, a.max_deg, a.tol, a.story_dz)
    out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
    plot(rows, a.max_deg, out, not a.no_annotate, a.absolute)

    side = out.with_suffix(".csv")
    with side.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "direction", "label", "model", "layer", "mode", "alpha",
                    "asr", "delta_asr", "ci_lo", "ci_hi", "cluster_asr", "n_clusters",
                    "pct_degenerate", "n"])
        for ps, key, name, model, layer, c in rows:
            if c is None:
                w.writerow([ps, key, name, model, layer] + [""] * 10)
                print(f"  !! no cell under the ceiling: {name} {model} {ps}",
                      file=sys.stderr)
                continue
            err = c.get("err")
            lo = f"{c['asr'] - err[0]:.1f}" if err else ""
            hi = f"{c['asr'] + err[1]:.1f}" if err else ""
            w.writerow([ps, key, name, model, layer, c["mode"], c["alpha"], c["asr"],
                        f"{c['delta']:.1f}", lo, hi,
                        f"{c.get('cluster_asr', float('nan')):.1f}",
                        c.get("n_clusters", ""), c["deg"], c["n"]])
            gap = abs(c["asr"] - c.get("cluster_asr", c["asr"]))
            flag = "  <-- wrapper-weighted estimate differs" if gap >= 10 else ""
            print(f"  {ps:8s} {name:16s} {model:20s} L{layer:<3s} "
                  f"a={c['alpha'] or '-':>5s}  dASR={c['delta']:+7.1f} "
                  f"[{lo or '?':>5s},{hi or '?':>6s}]  deg={c['deg']:>4}  "
                  f"clust={c.get('cluster_asr', float('nan')):5.1f}{flag}")
    print(f"\n  {out.relative_to(REPO).as_posix()}")
    print(f"  {side.relative_to(REPO).as_posix()}   (the cells behind every bar)")


if __name__ == "__main__":
    main()
