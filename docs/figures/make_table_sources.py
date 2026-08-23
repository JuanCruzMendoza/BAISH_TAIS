"""The A1 jailbreak-corpus source table as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

from md_table import parse

HEAD, ROWS, BOLD = parse("Source")
COLS = ((0.040, "left", 24), (0.330, "center", 8), (0.420, "left", 52))

render("docs/figures/table_sources.png", HEAD, ROWS, COLS, bold=BOLD,
       fig_w=8.2)
