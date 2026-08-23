"""Table 4 (the narrativity judge) as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

HEAD = ("Layer", "Arm", r"$\alpha$", "Steered wins")
ROWS = [
    ("L18", "restore", "-0.75", "3.4%"),
    ("L18", "suppress", "+0.25", "87.0%"),
    ("L23", "restore", "-0.75", "13.2%"),
    ("L23", "suppress", "+0.25", "63.5%"),
]
BOLD = [(0, 3), (1, 3)]
COLS = ((0.050, "left", 10), (0.250, "left", 12), (0.540, "center", 10),
        (0.830, "center", 14))

render("docs/figures/table_narrativity.png", HEAD, ROWS, COLS, bold=BOLD,
       fig_w=6.4)
