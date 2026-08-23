"""Table 3 (the alpha sweep at story L18) as a PNG."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

HEAD = (r"$\alpha$", r"$\Delta$ASR", "Degeneracy", r"$\Delta$ASR", "Degeneracy")
ROWS = [
    (r"$\pm$0.25", "-15.6", "1.2", "+14.1", "5.1"),
    (r"$\pm$0.50", "-29.1", "1.0", "+8.3", "5.8"),
    (r"$\pm$0.75", "-50.8", "2.4", "+1.8", "3.2"),
    (r"$\pm$1.00", "-87.2", "28.0", "+1.4", "9.0"),
]
BOLD = [(0, 3), (2, 1)]
COLS = ((0.045, "left", 12), (0.330, "center", 12), (0.500, "center", 12),
        (0.700, "center", 12), (0.900, "center", 12))
GROUPS = (("Restoring refusal", 0.260, 0.575),
          ("Suppressing refusal", 0.630, 0.965))

render("docs/figures/table_monotonic.png", HEAD, ROWS, COLS, bold=BOLD,
       groups=GROUPS, fig_w=7.4)
