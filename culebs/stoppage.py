"""Short-stoppage and restart-onset evidence (Sprint 3 research candidate, off by default).

Mechanism, from the external diagnosis (docs/SPRINT3_EXTERNAL_AUDIT.md):
  - the decoders detect almost every annotated stoppage on the original Metrica tracking, but
    leave D 1-2 s after the true restart (late restart timing);
  - the conservative dead model, which does not use ball missingness, misses ~40 % of stoppages.

Two tracking-only signals address exactly those two failures:

1. Line crossing. A stoppage starts where the ball crosses a field line: either the ball is
   observed past a line, or -- when it disappears -- the last reliable trajectory reaches a line
   within max_cross_s at its observed velocity. Uses position and velocity only.
2. Restart onset. A stoppage ends when the ball is kicked: an observed ball moving >= kick_speed
   after >= min_still_frames of being missing or nearly static.

A candidate stoppage interval is [crossing, next onset or the ball back inside the pitch) if
it lasts min_s..max_s. Evidence:
  dead += w_interval inside the interval, and dead -= w_onset for onset_frames after each onset
  (anywhere), so the decoder leaves D at the kick instead of later. Nothing is fitted; the
  thresholds are physical priors, and the weights are fixed before external evaluation.
"""

import numpy as np

from culebs.diagnostics import runs
from games import GAME, GAMES

DEFAULTS = {
    "max_cross_s": 1.0,         # the extrapolated crossing must happen within 1 s of last sight
    "min_toward_speed": 1.0,    # m/s toward the line, so a ball rolling along it does not count
    "kick_speed": 2.0,          # m/s; a struck or thrown ball
    "still_speed": 0.5,         # m/s; a placed ball
    "min_still_frames": 12,     # ~0.5 s missing/static before an onset counts
    "min_s": 1.0,               # stoppage interval bounds (seconds)
    "max_s": 90.0,
    "w_interval": 2.5,          # nats of dead evidence inside a candidate stoppage
    "w_onset": 3.0,             # nats of alive evidence right after a restart onset
    "onset_frames": 25,         # 1 s
}


def params_with_defaults(params):
    out = dict(DEFAULTS)
    out.update(params or {})
    return out


def crossings(ball, p):
    """Frames where the ball crosses a line: observed past it, or extrapolated at a gap start."""
    length, width = GAMES[GAME]["pitch"]
    xy, have, out = ball["xy"], ball["have"], ball["out"]
    n = len(have)
    frames = set(np.flatnonzero(out & ~np.concatenate(([False], out[:-1]))).tolist())
    for start, gap in runs(~have):
        last = start - 1
        if last < 1 or not have[last] or not have[last - 1]:
            continue
        v = (xy[last] - xy[last - 1]) * 25.0
        x, y = xy[last]
        # time to each line along the current velocity, only when moving toward it
        times = []
        for dist, toward in ((x, -v[0]), (length - x, v[0]), (y, -v[1]), (width - y, v[1])):
            if toward >= p["min_toward_speed"]:
                times.append(max(0.0, dist) / toward)
        if times and min(times) <= p["max_cross_s"]:
            frames.add(min(n - 1, last + 1 + int(round(min(times) * 25.0))))
    return np.array(sorted(frames), dtype=int)


def onsets(ball, p):
    """Frames where an observed ball starts moving >= kick_speed after being missing/static."""
    have, speed = ball["have"], np.nan_to_num(ball["speed"], nan=0.0)
    quiet = ~have | (speed < p["still_speed"])
    c = np.concatenate(([0], np.cumsum(quiet)))
    t = np.arange(len(have))
    w = int(p["min_still_frames"])
    quiet_before = (c[t] - c[np.maximum(0, t - w)]) >= w
    moving = have & (speed >= p["kick_speed"])
    first = moving & ~np.concatenate(([False], moving[:-1]))
    return np.flatnonzero(first & quiet_before)


def stoppage_evidence(geo, params=None):
    """{'dead_add': per-frame additive dead log-odds, 'intervals': [(a, b)], 'summary'}."""
    p = params_with_defaults(params)
    ball = geo["ball"]
    n = len(ball["have"])
    cross = crossings(ball, p)
    kicks = onsets(ball, p)
    dead_add = np.zeros(n)
    intervals = []
    # The ball re-observed inside the pitch for >= min_still_frames also ends a stoppage: the
    # evidence is about the time the ball was out or missing, so an undetected kick must not
    # let an interval swallow live play (first-version defect, games 1-2 diagnosis).
    inside = ball["have"] & ~ball["out"]
    c = np.concatenate(([0], np.cumsum(inside)))
    w = int(p["min_still_frames"])
    back_in = np.flatnonzero((c[w:] - c[:-w]) == w)          # window [t, t+w) fully inside
    for a in cross:
        later = kicks[kicks > a]
        seen = back_in[back_in > a]
        ends = [int(later[0])] if len(later) else []
        ends += [int(seen[0])] if len(seen) else []
        if not ends:
            continue
        b = min(ends)
        if p["min_s"] * 25 <= b - a <= p["max_s"] * 25:
            intervals.append((int(a), b))
            dead_add[a:b] += p["w_interval"]
    for k in kicks:
        dead_add[k:k + int(p["onset_frames"])] -= p["w_onset"]
    summary = {"crossings": int(len(cross)), "onsets": int(len(kicks)),
               "intervals": len(intervals),
               "interval_s": (np.percentile([(b - a) / 25.0 for a, b in intervals], [10, 50, 90]).round(1).tolist()
                              if intervals else [])}
    return {"dead_add": dead_add, "intervals": intervals, "onsets": kicks, "summary": summary}
