"""
Causal steering on story-wrapped jailbreaks: does moving AWAY from narrativity restore refusal?

Plan sections 7(a) and 7(b). Inputs are prompts whose harmful request is already inside a
narrative frame. Steering away from narrativity (alpha < 0) should restore refusal on the
ones that succeeded; steering toward it (alpha > 0) should induce compliance on the ones
that did not. Both arms come out of ONE sweep: every prompt is run at both signs plus a
shared alpha=0 baseline, and the split into 7(a) and 7(b) happens at analysis time
according to whether that baseline complied or refused.

The alpha grid is SYMMETRIC for two reasons beyond covering both arms. It gives the
dose-response monotonicity that section 9 asks for, and it is the only way to see the
damage failure mode: if -alpha and +alpha restore refusal *equally*, the effect is
perturbation damage rather than a signed directional one, and an asymmetric grid hides that.

Steers ONE layer at a time and compares them -- not all layers at once. SIMULTANEOUS=1
switches to injecting at every listed layer in a single forward pass, which is a much
stronger and differently-interpreted intervention.

LAYER CHOICE (default 20,22,24,26)
    `ort_M` saturates at 1.00 across the whole band, so it cannot rank layers; the
    selection comes from the residual-length column of docs/initial_tests/insights.md 2c instead.
        L22  ~60% depth, where behavioural steering usually bites, and the fictionality
             best layer. HIGHEST residual-length leakage of the four (dev -0.130).
        L24  2c's primary pick: cleanest residual inside the fictionality overlap L19-24,
             so section 4's matched-layers requirement holds.
        L26  cleanest overall (dev -0.056); tests whether going deeper than the
             fictionality band buys more steering power.
        L20  fills the gap so the leakage-vs-power correlation has four spread points.
    The SPREAD is deliberate. `resid_ort_layer` is only useful as a confound check if the
    tested layers differ in leakage; four clean layers would throw that away. Skipped
    L19/L21/L23/L25 are all high-leak layers adjacent to ones kept, and adjacent layers
    are highly correlated. L18 is outside the established band.
    LAYERS=18-26 restores the full sweep.

COEFFICIENT UNITS
    `narrativity_orth` is a difference of means with a subspace removed, so its norm is
    arbitrary and raw multipliers are not comparable across layers. The direction is
    unit-normalised and alpha is expressed in units of the layer's MEDIAN ACTIVATION NORM,
    measured over all prompt token positions of this request set:

        h <- h + alpha * median_norm(layer) * unit(narrativity_orth[layer])

    This is load-bearing for a layer sweep, not cosmetic: residual-stream norms grow with
    depth, so a fixed raw coefficient would make deep layers look weaker than they are.
    alpha = -1 means "subtract one median activation norm of narrativity". Both alpha and
    the resolved raw scalar are logged.

LAYER INDEXING -- the easy thing to get wrong
    The CSVs in results/ index by `hidden_states[l]`, where l=0 is the embedding output.
    So layer l is the OUTPUT of decoder block l-1, and the hook goes on blocks[l-1].
    Asserted at startup against the direction tensor's leading dimension.

WHAT IS INJECTED WHERE
    All positions: the prompt prefill and every decoded token. Refusal is decided at
    generation onset, so the prompt positions are the ones that matter; keeping it on
    during decode stops the effect decaying as the response grows.

MANIPULATION CHECK
    `nar_proj_final` is the readout at the FINAL layer, last prompt token. Reading it at
    the steered layer would be tautological -- we add alpha * unit(d) and project onto
    unit(d), so it moves by exactly the raw scalar by construction. At a downstream layer
    it measures whether the intervention actually propagated.

CONFOUND CHECK THE SWEEP GIVES YOU FREE
    `resid_ort_layer` carries each layer's residual-length AUROC from
    tier3_residual_length_metrics.csv when present. If steering power tracks residual-length
    leakage across L18-26, the effect is length-driven; if it does not, that is evidence
    against. docs/initial_tests/insights.md 2c found the leakage varies a lot inside this band (worst at
    L19, L21-23), so the sweep is informative either way.

CONTROLS (plan section 8) -- not run here, one flag away
    --direction length_pooled   the named length rival from docs/initial_tests/insights.md 2c
    --direction random          matched-norm random vector; must NOT restore refusal
    Same sweep, same units, so effects are comparable at matched strength.

NOT IN SCOPE
    No judge, no ASR scoring. Raw responses are saved so any rubric can run over them later.
    Decoding is greedy (do_sample=False) so differences across alpha are attributable to
    steering rather than sampling noise.

Usage:
    python steer_narrativity.py [model_name]
    LAYERS=18-26 ALPHAS=-2,-1.5,-1,-0.5,0.5,1,1.5,2 MAX_NEW=1024 python steer_narrativity.py Qwen/Qwen2.5-3B-Instruct
    LAYERS=22,24 python steer_narrativity.py Qwen/Qwen2.5-3B-Instruct        # cut further
    DIRECTION=length_pooled python steer_narrativity.py Qwen/Qwen2.5-3B-Instruct
    SIMULTANEOUS=1 python steer_narrativity.py Qwen/Qwen2.5-3B-Instruct

Reads data/initial_tests/story_jailbreaks.jsonl (override with $DATA_DIR) and
results/<model>/tier3_deconfounded_directions.pt. Appends to
results/<model>/steer_<direction>.jsonl after every generation, and skips
(id, layer, alpha) rows already present -- so an interrupted run resumes.

$OUT_DIR redirects the output only (the .pt and the residual-length CSV are still read from
results/<model>/). On Colab, point it at mounted Drive so results survive a disconnect and a
relaunch resumes from them -- better than downloading at the end, since the run is what gets
interrupted:

    from google.colab import drive; drive.mount('/content/drive')
    !cd /content/BAISH_TAIS && OUT_DIR=/content/drive/MyDrive/baish_results \
        python experiments/initial_tests/steer_narrativity.py Qwen/Qwen2.5-3B-Instruct
"""
import csv
import json
import os
import statistics
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR", os.path.join(HERE, "..", "..", "data", "initial_tests")
)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
DIRECTION = os.environ.get("DIRECTION", "narrativity_orth")
LAYERS = os.environ.get("LAYERS", "20,22,24,26")
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "-2,-1,1,2").split(",")]
MAX_NEW = int(os.environ.get("MAX_NEW", 1024))
SIMULTANEOUS = os.environ.get("SIMULTANEOUS", "") not in ("", "0", "false")
SEED = int(os.environ.get("SEED", 0))

def parse_layers(spec):
    """"20,22,24,26" | "18-26" | "20,22-24" -> sorted unique layer list."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = (int(x) for x in part.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


LAYER_LIST = parse_layers(LAYERS)


def load_jsonl(path):
    """One JSON object per line, with a decode error that says what to fix.

    Jailbreak prompts are pasted in as raw text, and raw text trips JSON in two
    predictable ways. The stdlib message ("Expecting ',' delimiter") names neither.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                near = line[max(0, e.colno - 45):e.colno + 25].replace("\n", "")
                raise SystemExit(
                    f"{os.path.basename(path)} line {i}, column {e.colno}: {e.msg}\n"
                    f"  near: ...{near}...\n"
                    "  Pasted jailbreak text almost always means one of:\n"
                    '    a literal "  inside a string  -> escape it as \\"\n'
                    "    a literal tab or newline      -> escape as \\t / \\n, or delete it\n"
                    "  Reliable fix: build the row in Python and json.dumps() it rather than\n"
                    "  typing the quotes by hand."
                ) from None
    return rows


# ------------------------------------------------------------------------------- data
items = load_jsonl(os.path.join(DATA_DIR, "story_jailbreaks.jsonl"))
todo = [it for it in items if "FILL_ME" not in (it["prompt"] + it["request"])]
if not todo:
    raise SystemExit(
        f"all {len(items)} rows in story_jailbreaks.jsonl are still templates -- "
        "fill `prompt` and `request` (and delete the template rows) before running"
    )
if len(todo) < len(items):
    print(f"skipping {len(items) - len(todo)} unfilled template row(s)")

# --------------------------------------------------------------------------- model
print(f"loading {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
dev = model.device


def get_blocks(m):
    for attr in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = m
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise SystemExit(f"cannot locate decoder blocks on {type(m).__name__}")


blocks = get_blocks(model)

# ---------------------------------------------------------------------- directions
dir_path = os.path.join(
    HERE, "results", MODEL.replace("/", "_"), "tier3_deconfounded_directions.pt"
)
if not os.path.exists(dir_path):
    raise SystemExit(f"missing {dir_path} -- run deconfound_length_tier3.py first")
bundle = torch.load(dir_path)

if DIRECTION == "random":
    g = torch.Generator().manual_seed(SEED)
    ref = bundle["narrativity_orth"]
    vecs = torch.randn(ref.shape, generator=g)          # matched-norm control
elif DIRECTION in bundle:
    vecs = bundle[DIRECTION]
else:
    raise SystemExit(
        f"'{DIRECTION}' not in {dir_path}; available: "
        + ", ".join(k for k, v in bundle.items() if torch.is_tensor(v))
    )

# layer l in the CSVs == hidden_states[l] == output of blocks[l-1]
n_layers = len(blocks)
if vecs.shape[0] != n_layers + 1:
    raise SystemExit(
        f"direction has {vecs.shape[0]} layers but model has {n_layers} blocks "
        f"(expected {n_layers + 1} = embeddings + blocks). Wrong model for this .pt?"
    )
if vecs.shape[1] != model.config.hidden_size:
    raise SystemExit(f"direction hidden={vecs.shape[1]} vs model {model.config.hidden_size}")
bad = [l for l in LAYER_LIST if not 1 <= l <= n_layers]
if bad:
    raise SystemExit(f"layers {bad} outside 1..{n_layers}")

units = {l: (vecs[l] / vecs[l].norm()).to(dev) for l in LAYER_LIST}

# ------------------------------------------------------- per-layer median activation norm
# Median over ALL prompt token positions of this request set: the perturbation is applied
# at every position, so a last-token-only norm would misstate the relative scale.
print("measuring per-layer activation norms ...")
norms = {l: [] for l in LAYER_LIST}


def encode(text):
    msgs = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception:
        return tok(text, return_tensors="pt")


with torch.no_grad():
    for it in todo:
        out = model(**encode(it["prompt"]).to(dev), output_hidden_states=True)
        for l in LAYER_LIST:
            norms[l] += out.hidden_states[l][0].float().norm(dim=-1).tolist()
median_norm = {l: statistics.median(norms[l]) for l in LAYER_LIST}
print("  " + "  ".join(f"L{l}:{median_norm[l]:.0f}" for l in LAYER_LIST))

# ------------------------------------------------------------------------------ hooks
state = {"calls": 0, "final": None}


def make_add_hook(unit_vec, raw_scale):
    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + raw_scale * unit_vec.to(hs.dtype)
        state["calls"] += 1
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
    return hook


def capture_final_hook(module, args, output):
    """Grab the last-position final-layer state on the PREFILL pass only (seq > 1)."""
    hs = output[0] if isinstance(output, tuple) else output
    if hs.shape[1] > 1:
        state["final"] = hs[0, -1, :].detach().float().cpu()


unit_final = (vecs[n_layers] / vecs[n_layers].norm()).cpu()


@torch.no_grad()
def generate(prompt, cfg):
    """cfg: list of (layer, raw_scale). Empty -> unsteered baseline."""
    state["calls"], state["final"] = 0, None
    handles = [blocks[-1].register_forward_hook(capture_final_hook)]
    for l, raw in cfg:
        handles.append(blocks[l - 1].register_forward_hook(make_add_hook(units[l], raw)))
    try:
        enc = encode(prompt).to(dev)
        out = model.generate(
            **enc, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        new = out[0, enc["input_ids"].shape[-1]:]
        proj = float(state["final"] @ unit_final) if state["final"] is not None else float("nan")
        return tok.decode(new, skip_special_tokens=True), int(new.shape[0]), proj, state["calls"]
    finally:
        for h in handles:
            h.remove()


# ----------------------------------------------------------- residual-length per layer
resid = {}
resid_csv = os.path.join(
    HERE, "results", MODEL.replace("/", "_"), "tier3_residual_length_metrics.csv"
)
if os.path.exists(resid_csv):
    with open(resid_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            resid[int(row["layer"])] = float(row["resid_ort"])
else:
    print(f"note: {os.path.basename(resid_csv)} not found -> resid_ort_layer will be null")

# ------------------------------------------------------------------------- resume state
out_dir = os.environ.get(
    "OUT_DIR", os.path.join(HERE, "results", MODEL.replace("/", "_"))
)
out_path = os.path.join(out_dir, f"steer_{DIRECTION}.jsonl")
os.makedirs(out_dir, exist_ok=True)
done = set()
if os.path.exists(out_path):
    for row in load_jsonl(out_path):
        done.add((row["id"], row["layer"], row["alpha"]))
    print(f"resuming: {len(done)} generations already in {os.path.basename(out_path)}")

sink = open(out_path, "a", encoding="utf-8")


def emit(row):
    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    sink.flush()


# ------------------------------------------------------------------------ smoke test
# A hook that never fires, or one whose arithmetic is wrong, would silently produce
# baseline text at every alpha. Check both before spending the run.
probe = todo[0]["prompt"]
base_txt, _, base_proj, base_calls = generate(probe, [])
l0 = LAYER_LIST[len(LAYER_LIST) // 2]
zero_txt, _, _, zero_calls = generate(probe, [(l0, 0.0)])
big_txt, _, big_proj, big_calls = generate(probe, [(l0, -2.0 * median_norm[l0])])
print(f"\nsmoke test @ L{l0}:")
print(f"  hook fires            : {zero_calls > 0 and big_calls > 0} "
      f"(baseline calls={base_calls}, steered calls={big_calls})")
print(f"  alpha=0 == baseline   : {zero_txt == base_txt}")
print(f"  alpha=-2 != baseline  : {big_txt != base_txt}")
print(f"  final-layer readout   : {base_proj:+.1f} -> {big_proj:+.1f} (propagation)")
if not (zero_calls > 0 and zero_txt == base_txt):
    raise SystemExit("hook is broken (did not fire, or alpha=0 changed the output) -- stopping")
if big_txt == base_txt:
    print("  WARNING: alpha=-2 left the output unchanged. Either this layer has no "
          "steering power or the direction is wrong; inspect before trusting a null.")

# ------------------------------------------------------------------------------- run
plan = ["baseline"] + [(l, a) for l in LAYER_LIST for a in ALPHAS]
if SIMULTANEOUS:
    plan = ["baseline"] + [("all", a) for a in ALPHAS]
total = len(todo) * len(plan) - len(done)
print(f"\n{len(todo)} requests x {len(plan)} configs = {len(todo) * len(plan)} generations "
      f"({total} remaining), max_new={MAX_NEW}, greedy, direction={DIRECTION}"
      + (" [SIMULTANEOUS]" if SIMULTANEOUS else ""))

n = 0
for it in todo:
    for cfg in plan:
        if cfg == "baseline":
            layer, alpha, raw, hooks = None, 0.0, None, []
        elif SIMULTANEOUS:
            layer, alpha = "all", cfg[1]
            hooks = [(l, alpha * median_norm[l]) for l in LAYER_LIST]
            raw = None
        else:
            layer, alpha = cfg
            raw = alpha * median_norm[layer]
            hooks = [(layer, raw)]
        if (it["id"], layer, alpha) in done:
            continue
        text, ntok, proj, calls = generate(it["prompt"], hooks)
        emit({
            "model": MODEL, "direction": DIRECTION, "layer": layer, "alpha": alpha,
            "raw_scale": raw, "id": it["id"], "source": it.get("source"),
            "request": it["request"], "prompt": it["prompt"],
            "response": text, "out_tokens": ntok,
            "nar_proj_final": proj, "hook_calls": calls,
            "resid_ort_layer": resid.get(layer) if isinstance(layer, int) else None,
            "max_new_tokens": MAX_NEW, "greedy": True,
        })
        n += 1
        print(f"[{n}/{total}] {it['id']} L{layer} a={alpha:+.1f} -> {ntok} tok, "
              f"proj={proj:+.1f} | {text[:60]!r}")

sink.close()
print(f"\nwrote {n} new generations -> {out_path}")
print("next: score `response` with a refusal/StrongREJECT rubric, then plot ASR vs alpha "
      "per layer. Check whether steering power correlates with `resid_ort_layer` -- if it "
      "does, the effect is length, not narrativity.")
