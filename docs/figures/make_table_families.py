"""Table 5 (delta-ASR by jailbreak family) as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

HEAD = ("Family", ["n successful", "/ refused"], [r"Restore", r"$\Delta$ASR"],
        ["Suppress", r"$\Delta$ASR"])
ROWS = [
    ("Fiction / Narrative", "343 / 110", "-40.2", "+10.0"),
    ("Role-play / Persona", "67 / 210", "-73.1", "+14.8"),
    ("Hybrid", "64 / 73", "-75.0", "+17.8"),
    ("Non-fiction / Other", "34 / 40", "-67.6", "+15.0"),
    ("All", "508 / 433", "-50.8", "+14.1"),
]
BOLD = [(2, 2), (2, 3), (4, 0), (4, 1), (4, 2), (4, 3)]
COLS = ((0.040, "left", 24), (0.450, "center", 16), (0.680, "center", 12),
        (0.890, "center", 12))

render("docs/figures/table_families.png", HEAD, ROWS, COLS, bold=BOLD,
       rules_after=(3,), fig_w=7.0)
