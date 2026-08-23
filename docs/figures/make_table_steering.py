"""Table 1 (delta-ASR per axis and model) as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

# ASCII hyphen, not U+2212: cmr10 has no minus-sign glyph, and routing the number
# through mathtext to get one would forfeit the bold face (mathtext ignores
# \mathbf on digits).
HEAD = ("Axis", "Restore", "Suppress", "Restore", "Suppress")
ROWS = [
    ("harm", "-99.8", "+64.9", "-99.4", "+41.8"),
    ("persona", "-98.4", "+62.8", "-99.6", "+51.1"),
    ("story, max gap", "-50.8", "+14.1", "-73.4", "+29.3"),
    (r"story, max $d_z$", "-12.0", "+9.0", "-5.9", "+6.6"),
    ("eval", "-17.7", "+48.7", "-44.1", "+15.1"),
]
# The cells that all but saturate their arm, as bolded in the draft.
BOLD = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4)]
COLS = ((0.035, "left", 20), (0.400, "center", 12), (0.560, "center", 12),
        (0.760, "center", 12), (0.915, "center", 12))
GROUPS = (("Qwen2.5-7B-Instruct", 0.330, 0.630),
          ("Gemma-2-9b-it", 0.690, 0.980))

render("docs/figures/table_steering.png", HEAD, ROWS, COLS, bold=BOLD,
       groups=GROUPS, fig_w=7.6)
