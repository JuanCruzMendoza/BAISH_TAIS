"""Shared booktabs-style table renderer for the docs figures."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# cmr10 and cmb10 ship with matplotlib, so none of this needs a LaTeX install.
# mathtext is set to cm so that a \rightarrow inside a cell matches the body type.
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"],
                     "mathtext.fontset": "cm", "axes.unicode_minus": False})

LINE_H = 0.235      # inches per text line
ROW_PAD = 0.088     # inches above and below a row block
RULE_PAD = 0.075    # booktabs \abovetopsep / \belowbottomsep


def render(path, head, rows, cols, rules_after=(), bold=(), groups=(),
           fig_w=9.0, dpi=200, fontsize=12):
    r"""`cols` = one (x, ha, wrap_chars) per column, x in figure fraction.

    `rules_after` = data-row indices to draw a \midrule under, for grouping.
    `bold` = (data-row, column) index pairs to set in the bold face.
    `groups` = (label, x_lo, x_hi) spanning headers, each over its own
    \cmidrule, drawn in a row above the column names. Extents are given rather
    than derived from `cols`, since a column carries an anchor and not a width.

    Wrapping counts source characters, so a cell holding mathtext should be
    passed as a list of lines instead of relying on the wrap width.
    """

    def lines_of(t, wrap):
        if isinstance(t, (list, tuple)):    # already split by the caller
            return list(t)
        return textwrap.wrap(t, wrap) or [""]

    cells = [[lines_of(t, c[2]) for t, c in zip(r, cols)]
             for r in [head] + list(rows)]
    heights = [max(len(c) for c in row) * LINE_H + 2 * ROW_PAD for row in cells]
    group_h = (LINE_H + 2 * ROW_PAD) if groups else 0.0
    fig_h = sum(heights) + group_h + 2 * RULE_PAD
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")
    bold = set(bold)

    def rule(y, lw, lo=0.02, hi=0.98):
        fig.add_artist(plt.Line2D([lo, hi], [y, y], color="black", lw=lw,
                                  solid_capstyle="butt"))

    y = 1.0 - RULE_PAD / fig_h
    rule(y, 1.7)                                    # \toprule
    if groups:
        base = y - (ROW_PAD + 0.74 * LINE_H) / fig_h
        for label, lo, hi in groups:
            fig.text(0.5 * (lo + hi), base, label, fontsize=fontsize,
                     va="baseline", ha="center", color="black",
                     family=["cmb10"])
        y -= group_h / fig_h
        for _, lo, hi in groups:
            rule(y, 0.8, lo, hi)                    # \cmidrule
    for i, (row, h) in enumerate(zip(cells, heights)):
        top = y - ROW_PAD / fig_h
        for j, (lines, (x, ha, _)) in enumerate(zip(row, cols)):
            heavy = i == 0 or (i - 1, j) in bold
            for k, line in enumerate(lines):
                fig.text(x, top - (k + 0.74) * LINE_H / fig_h, line,
                         fontsize=fontsize, va="baseline", ha=ha, color="black",
                         family=["cmb10"] if heavy else ["cmr10"])
        y -= h / fig_h
        if i == 0 or (i - 1) in rules_after:
            rule(y, 0.9)                            # \midrule
    rule(y, 1.7)                                    # \bottomrule
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight",
                pad_inches=0.14)
    print("wrote", path)
