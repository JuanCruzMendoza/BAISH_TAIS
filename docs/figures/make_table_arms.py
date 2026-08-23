"""The Methodology steering-attribution arms table as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

HEAD = ("Arm", "Vector", "Norm", "Question")
ROWS = [
    ("Unprojected", r"$\hat{u}_a$", "1", ["the reference cell"]),
    ("Perpendicular", r"$(\hat{u}_a - c\,\hat{u}_b)\,/\,\sqrt{1-c^2}$", "1",
     [r"$\mathbf{necessity}$: does $a$ still work",
      r"with $b$ removed?"]),
    ("Parallel", r"$c\,\hat{u}_b$", r"$|c|$",
     [r"$\mathbf{sufficiency}$: does $b$'s share of",
      r"the reference push reproduce it?"]),
]
COLS = ((0.040, "left", 16), (0.215, "left", 60), (0.510, "center", 8),
        (0.600, "left", 60))

render("docs/figures/table_arms.png", HEAD, ROWS, COLS, fig_w=8.0)
