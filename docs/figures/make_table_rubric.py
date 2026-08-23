"""The A2 StrongREJECT rubric table as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

from md_table import parse

HEAD, ROWS, BOLD = parse("Item")
COLS = ((0.045, "left", 16), (0.270, "center", 10), (0.420, "left", 48))

render("docs/figures/table_rubric.png", HEAD, ROWS, COLS, bold=BOLD, fig_w=7.0)
