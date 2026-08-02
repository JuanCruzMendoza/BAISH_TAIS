"""Paired AUROC, Clopper-Pearson, LOPO, effect sizes, subspace angles.

No bootstrap anywhere (spec 0.7): paired AUROC is a proportion over n pairs,
so Clopper-Pearson on the win count is exact, seedless, and correct at the
boundary where a bootstrap over the same outcomes degenerates.
"""
import math

import numpy as np

# ---------------------------------------------------------------- paired AUROC


def paired_outcomes(pos, neg):
    """Per-pair win / tie indicators. pos, neg: [n] readouts of the same pairs."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    wins = int((pos > neg).sum())
    ties = int((pos == neg).sum())
    return wins, ties, len(pos)


def paired_auroc(pos, neg):
    wins, ties, n = paired_outcomes(pos, neg)
    return (wins + 0.5 * ties) / n if n else float("nan")


def cohens_dz(pos, neg):
    d = np.asarray(pos, float) - np.asarray(neg, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


# ------------------------------------------------------- Clopper-Pearson exact


def _betacf(a, b, x, itmax=300, eps=1e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b). Self-contained: no scipy."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k, n, alpha=0.05, side="two"):
    """Exact interval for k successes in n. side: 'two' | 'upper' | 'lower'."""
    if n == 0:
        return float("nan"), float("nan")
    a = alpha if side != "two" else alpha / 2.0
    lo = 0.0 if (k == 0 or side == "upper") else _beta_ppf(a, k, n - k + 1)
    hi = 1.0 if (k == n or side == "lower") else _beta_ppf(1.0 - a, k + 1, n - k)
    return lo, hi


def auroc_ci(pos, neg, alpha=0.05, side="two"):
    """CP interval on the paired AUROC. Ties are folded in as half-wins."""
    wins, ties, n = paired_outcomes(pos, neg)
    k = int(round(wins + 0.5 * ties))
    lo, hi = clopper_pearson(k, n, alpha, side)
    return {"auroc": (wins + 0.5 * ties) / n if n else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "wins": wins, "ties": ties, "n": n}


def sign_test_p(pos, neg):
    """Exact one-sided binomial p for 'positive scores above negative'."""
    wins, ties, n = paired_outcomes(pos, neg)
    k, m = wins, n - ties
    if m == 0:
        return 1.0
    return sum(math.comb(m, i) for i in range(k, m + 1)) / (2 ** m)


# ------------------------------------------------------------------------ LOPO


def diff_in_means(pos, neg):
    """pos, neg: [n, ...] -> mean(pos) - mean(neg) over axis 0."""
    return pos.mean(axis=0) - neg.mean(axis=0)


def lopo_directions(pos, neg):
    """[n, ...] -> [n, ...]; row i is the vector fitted without pair i."""
    n = pos.shape[0]
    ps, ns = pos.sum(axis=0), neg.sum(axis=0)
    return (ps - pos) / (n - 1) - (ns - neg) / (n - 1)


def unit(v, axis=-1, eps=1e-12):
    nrm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(nrm, eps)


def readout(h, u, mu=None):
    """(h - mu) . u"""
    x = h if mu is None else h - mu
    return np.einsum("...d,...d->...", x, np.broadcast_to(u, x.shape))


def sigma_act(h):
    """Median residual-stream norm per layer: steering units for experiment 4."""
    return np.median(np.linalg.norm(np.asarray(h, "float32"), axis=-1), axis=0)


# ---------------------------------------------------------------- geometry


def principal_angles(A, B):
    """Angles in degrees between the column spans of A [d, p] and B [d, q]."""
    qa = np.linalg.qr(np.asarray(A, float))[0]
    qb = np.linalg.qr(np.asarray(B, float))[0]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def cos(a, b, eps=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), eps))


def random_cos_band(d, k=3.0):
    """+/- k/sqrt(d): the cosine null band for two random vectors."""
    return k / math.sqrt(d)


def residual_frac(v, basis):
    """||v - P_span(basis) v|| / ||v|| (spec 2.3). basis: [k, d] rows."""
    v = np.asarray(v, float)
    q = np.linalg.qr(np.asarray(basis, float).T)[0]
    r = v - q @ (q.T @ v)
    return float(np.linalg.norm(r) / max(np.linalg.norm(v), 1e-12))


def spearman_brown(s):
    """Split-half cosine -> reliability of the double-length estimate, 2s/(1+s).

    A split-half cosine is the reliability of a *half*-sized vector; the full
    vector is more reliable than that, and correcting a cross-axis cosine by the
    uncorrected half value would over-correct.
    """
    return 2.0 * s / (1.0 + s) if s > 0 else float("nan")


# ------------------------------------------------------- clustered aggregation


def cluster_means(values, cluster_ids):
    """Spec 0.7: collapse to one value per cluster before testing."""
    buckets = {}
    for v, c in zip(values, cluster_ids):
        buckets.setdefault(c, []).append(v)
    keys = sorted(buckets)
    return keys, np.array([np.mean(buckets[k]) for k in keys])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def auroc_within_bins(pos, neg, bin_key, n_bins=10):
    """Paired AUROC inside deciles of bin_key. Length control for spec 1.2a/3.2."""
    pos, neg, bin_key = map(lambda a: np.asarray(a, float), (pos, neg, bin_key))
    edges = np.quantile(bin_key, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (bin_key >= lo) & (bin_key <= hi if i == n_bins - 1 else bin_key < hi)
        if m.sum() >= 3:
            out.append({"bin": i, "n": int(m.sum()), "lo": float(lo), "hi": float(hi),
                        "auroc": paired_auroc(pos[m], neg[m])})
    return out
