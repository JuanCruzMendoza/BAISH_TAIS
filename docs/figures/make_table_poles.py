"""The A3 narrativity-judge poles table as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

from md_table import parse

HEAD, ROWS, BOLD = parse("Pole")
COLS = ((0.035, "left", 20), (0.310, "left", 62))

render("docs/figures/table_poles.png", HEAD, ROWS, COLS, bold=BOLD, fig_w=8.6)
