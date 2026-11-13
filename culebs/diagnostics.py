"""Small pure helpers for data diagnostics: run lengths, masks, quantile summaries.

Everything returns aggregates. Nothing here should ever return or write a frame-level table --
those are derived copies of licensed data (RULES.md, "The data").
"""

import numpy as np

from games import GAME, GAMES, half_video_blocks


def runs(mask):
    """[(start, length)] of the True runs of a 1-D boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    return np.stack([starts, ends - starts], axis=1)


def graded_mask(game=GAME):
    """True on the video frames the scorer grades (both halves, per games.half_video_blocks)."""
    mask = np.zeros(GAMES[game]["n_video_frames"], dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def half_masks(game=GAME):
    """{half: mask} over video frames."""
    n = GAMES[game]["n_video_frames"]
    out = {}
    for half, first, last in half_video_blocks(game):
        mask = np.zeros(n, dtype=bool)
        mask[first:last + 1] = True
        out[half] = mask
    return out


def quantiles(values, qs=(0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)):
    """{'p50': ...} summary of finite values; empty dict if none."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {"p{:g}".format(q * 100): round(float(np.quantile(values, q)), 4) for q in qs}


def pct(numerator, denominator):
    return round(100.0 * float(numerator) / float(denominator), 3) if denominator else None
