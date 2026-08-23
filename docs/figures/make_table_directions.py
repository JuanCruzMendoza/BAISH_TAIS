"""The Methodology directions table as a PNG, for platforms with no tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from latex_table import render

HEAD = ("Axis", "Positive arm", "Negative arm")
ROWS = [
    ("story", "an invented narration",
     "a non-narrative text on the same topic"),
    ("persona", "200 role framings, 5 paraphrases each",
     "a pool of 13 default-assistant framings"),
    ("harm", "a harmful request", "its content-matched benign twin"),
    ("eval", "an evaluation framing", "a deployment framing"),
]
COLS = ((0.035, "left", 14), (0.190, "left", 42), (0.605, "left", 40))

render("docs/figures/table_directions.png", HEAD, ROWS, COLS)
