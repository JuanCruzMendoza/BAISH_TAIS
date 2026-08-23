"""Table 2 (declines and specificity on the successful jailbreaks) as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

HEAD = ("Cell", "ASR", ["Explicit", "declines"], ["Judge", "specificity"])
ROWS = [
    ("baseline", "100", "0%", "4.87"),
    (r"story L18, $\alpha$ = -0.75", "49.2", "0%", "2.73"),
    (r"persona L15, $\alpha$ = -0.5", "1.6", "31%", "4.10"),
    (r"harm L21, $\alpha$ = +0.75", "0.2", "90%", "4.08"),
    (r"eval L9, $\alpha$ = +0.5", "82.3", "5%", "4.69"),
]
BOLD = [(1, 2), (1, 3)]
COLS = ((0.040, "left", 40), (0.480, "center", 10), (0.650, "center", 14),
        (0.860, "center", 14))

render("docs/figures/table_specificity.png", HEAD, ROWS, COLS, bold=BOLD,
       fig_w=8.0)
