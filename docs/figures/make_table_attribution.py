"""The steering attribution table (Table 6) as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

ARROW = r" $\rightarrow$ "
HEAD = ("Pair", "Cos sim", "Arm", "Restore", "Suppress")
ROWS = [
    ("story" + ARROW + "persona (L18)", "+0.177", "Unprojected", "100%", "100%"),
    ("", "", "Perpendicular", "103.1%", "106.4%"),
    ("", "", "Parallel", "32.5%", "75.2%"),
    ("persona (L15)", "", "Unprojected", "100%", "100%"),
    ("persona" + ARROW + "story", "+0.137", "Perpendicular", "100.2%", "100.8%"),
    ("", "", "Parallel", "8.4%", "10.4%"),
    ("persona" + ARROW + "eval", "+0.296", "Perpendicular", "94.0%", "89.8%"),
    ("", "", "Parallel", "25.8%", "25.0%"),
    ("persona" + ARROW + "harm", "-0.240", "Perpendicular", "60.8%", "80.6%"),
    ("", "", "Parallel", "93.4%", "47.1%"),
]
# Wrap widths are loose on the pair column so mathtext is never split.
COLS = ((0.030, "left", 60), (0.325, "center", 12), (0.410, "left", 18),
        (0.720, "center", 12), (0.885, "center", 12))

render("docs/figures/table_attribution.png", HEAD, ROWS, COLS, rules_after=(2,))
