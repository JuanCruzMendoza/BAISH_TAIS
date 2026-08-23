"""Tables 7 and 8 (the full steering sweep) as PNGs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

from md_table import parse

# Wrap widths are loose everywhere: every cell is short, and splitting one that
# holds mathtext would leave each half to render as raw OT1 characters.
COLS = ((0.035, "left", 24), (0.190, "center", 24), (0.300, "center", 24),
        (0.450, "center", 24), (0.610, "center", 24),
        (0.760, "center", 24), (0.930, "center", 24))
GROUPS = (("Restoring refusal", 0.380, 0.680),
          ("Suppressing refusal", 0.700, 0.985))

for out, occurrence in [("table_sweep_qwen.png", 0), ("table_sweep_gemma.png", 1)]:
    head, rows, bold = parse("Direction", occurrence=occurrence)
    head = ("Direction", "Layer", r"$|\alpha|$", "ASR", "Degeneracy",
            "ASR", "Degeneracy")
    render("docs/figures/" + out, head, rows, COLS, bold=bold, groups=GROUPS,
           fig_w=8.4)
