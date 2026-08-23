"""Pull a markdown table out of the draft, so long ones are not retyped."""
import re
from pathlib import Path

DRAFT = Path(__file__).resolve().parents[2] / "docs" / "draft.md"
# cmr10 has no glyph for these, and they only ever appear as decoration here.
SWAP = {chr(8722): "-", chr(8211): "-", chr(8212): "-", chr(215): "x"}


def clean(cell):
    cell = re.sub(r"\[([^]]*)]\([^)]*\)", r"\1", cell).strip()   # drop links
    for k, v in SWAP.items():
        cell = cell.replace(k, v)
    return cell


def parse(first_cell, path=DRAFT, occurrence=0):
    """Returns (head, rows, bold) for the first table whose header cell matches.

    `bold` is the set of (data-row, column) pairs that were **starred**.
    `occurrence` picks among repeated headers, as the two A4 sweeps share one.
    """
    lines = path.read_text(encoding="utf-8").split(chr(10))
    hits = [k for k, l in enumerate(lines)
            if l.startswith("| " + first_cell + " ")]
    i = hits[occurrence]
    block = []
    while i < len(lines) and lines[i].startswith("|"):
        block.append([c.strip() for c in lines[i].strip("|").split("|")])
        i += 1
    head = [clean(c) for c in block[0]]
    bold, rows = set(), []
    for r, raw in enumerate(block[2:]):                  # block[1] is the rule
        row = []
        for c, cell in enumerate(raw):
            if cell.startswith("**") and cell.endswith("**"):
                bold.add((r, c))
                cell = cell[2:-2]
            row.append(clean(cell))
        rows.append(row)
    return head, rows, bold
