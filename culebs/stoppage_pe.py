"""Positive-evidence stoppage detection for the majority-vote family (Sprint 4, off by default).

Goal: raise dead-ball recall of e108 without re-introducing the failure that made the mpe models
lose on other matches -- stoppages invented inside live play, which split true possessions.
So nothing here treats a missing ball as dead, and an interval is only emitted when several
INDEPENDENT signals agree.

Signals (each can be disabled for ablation):

  observed_out    the ball is observed past a touch/goal line (beyond out_margin)
  extrapolation   the ball disappears while its last reliable trajectory (>= 3 consistent
                  observations) reaches a line within max_cross_s; confidence falls with the
                  time to the line. A ball lost mid-flight away from a line gives NOTHING.
  placement       an observed ball static (< still_speed) for >= placement_frames at a restart
                  location: touchline band, corner, goal area, centre spot (strong), or elsewhere
                  (weak: a free kick, but also a keeper holding the ball)
  onset           a confirmed restart: an observed ball reaching kick_speed after >= quiet_frames
                  missing or static, WITH a player within contact_m in the frames just before.
                  The ball reappearing on its own is not an onset.
  set_piece       Smart Tagging SET PIECES restart frames (competition match only; there is no
                  equivalent tagging on the external matches, so it is never evaluated there)
  motion_support  collective motion inside the candidate interval below motion_ratio x the
                  block median -- SUPPORT only, it can never create an interval on its own

Assembly: an interval runs from an anchor (crossing, placement start or set-piece window start)
to the next confirmed onset within max_s. If sustained live play (>= 2 s of an observed moving
ball) appears before that onset, the candidate is DROPPED -- the restart was missed, and
stretching the interval would swallow live play (first-version defect, games 1-2). Its score is the sum of the distinct signal types it
contains: crossing (observed 1.0, extrapolated = confidence), placement (strength), onset 1.0,
set piece 1.0, motion support 0.5. Intervals scoring >= min_score add w_dead nats of dead
evidence over [start, onset); a confirmed onset that closes an interval adds -w_onset for
onset_frames (fixes late restarts). With min_score 2.0 an onset alone, or a crossing alone,
never creates a stoppage.

All thresholds are physical priors fixed before any evaluation; see DEFAULTS.
"""

import numpy as np

from culebs.diagnostics import runs
from games import GAME, GAMES

DEFAULTS = {
    # components
    "use_observed_out": True,
    "use_extrapolation": True,
    "use_placement": True,
    "use_onset_contact": True,      # require a player at the ball for an onset
    "use_onset_alive": True,        # alive evidence right after a confirmed onset
    "use_set_piece": False,         # needs Smart Tagging (competition only)
    "use_motion_support": True,
    # physics / geometry priors
    "min_out_frames": 3,            # an observed crossing must persist (>= 0.12 s past the line)
    "max_cross_s": 0.6,             # crossing must be reached within 0.6 s of last sight
    "min_toward_speed": 2.0,        # m/s toward the line
    "max_turn_deg": 30.0,           # trajectory consistency over the last 3 observations
    "still_speed": 0.5,             # m/s: a placed ball
    "placement_frames": 25,         # 1 s static
    "touch_band_m": 1.5,            # throw-in band
    "corner_m": 2.5,
    "goal_area_depth_m": 6.0, "goal_area_half_width_m": 10.0,
    "centre_m": 2.0,
    "placement_strength": {"touchline": 1.0, "corner": 1.0, "centre": 1.0, "goal_area": 0.5, "open": 0.3},
    "kick_speed": 2.0,              # m/s
    "quiet_frames": 12,             # ~0.5 s missing or static before an onset
    "contact_m": 2.5,               # a player this close to the ball just before the kick
    "contact_lookback": 5,          # frames
    "max_s": 90.0,
    "live_frames": 50,              # >= 2 s of an observed, moving ball = play has resumed
    "live_speed": 1.0,              # m/s
    "motion_ratio": 0.7,
    # evidence
    "min_score": 2.0,
    "w_dead": 3.0,
    "w_onset": 3.0,
    "onset_frames": 25,
}


def params_with_defaults(params):
    out = dict(DEFAULTS)
    out.update(params or {})
    return out


def _pitch():
    return GAMES[GAME]["pitch"]


def observed_out_frames(ball, p=None):
    """First frame of each run of >= min_out_frames observed beyond a line (a single-frame
    excursion is tracking noise at the line: dev diagnosis, games 1-2)."""
    k = int((p or DEFAULTS)["min_out_frames"])
    return np.array([s for s, n in runs(ball["out"]) if n >= k], dtype=int)


def extrapolated_crossings(ball, p):
    """[(frame, confidence)] for gaps whose last reliable trajectory reaches a line in time."""
    length, width = _pitch()
    xy, have = ball["xy"], ball["have"]
    n = len(have)
    out = []
    for start, _ in runs(~have):
        last = start - 1
        if last < 2 or not have[last - 2:last + 1].all():
            continue
        v1 = xy[last] - xy[last - 1]
        v0 = xy[last - 1] - xy[last - 2]
        n0, n1 = np.hypot(*v0), np.hypot(*v1)
        if n0 == 0 or n1 == 0:
            continue
        turn = np.degrees(np.arccos(np.clip(np.dot(v0, v1) / (n0 * n1), -1, 1)))
        if turn > p["max_turn_deg"]:
            continue                                      # inconsistent trajectory: no claim
        v = v1 * 25.0
        x, y = xy[last]
        best = None
        for dist, toward in ((x, -v[0]), (length - x, v[0]), (y, -v[1]), (width - y, v[1])):
            if toward >= p["min_toward_speed"]:
                t = max(0.0, dist) / toward
                best = t if best is None else min(best, t)
        if best is None or best > p["max_cross_s"]:
            continue                                      # lost mid-flight away from a line
        conf = float(np.clip(1.0 - best / p["max_cross_s"], 0.0, 1.0)) * (1.0 - turn / (2 * p["max_turn_deg"]))
        out.append((min(n - 1, last + 1 + int(round(best * 25))), round(conf, 3)))
    return out


def placements(ball, p):
    """[(start, end, kind, strength)] of static observed balls at a restart location."""
    length, width = _pitch()
    xy, have = ball["xy"], ball["have"]
    speed = np.nan_to_num(ball["speed"], nan=np.inf)
    still = have & (speed < p["still_speed"])
    out = []
    for s, k in runs(still):
        if k < p["placement_frames"]:
            continue
        x, y = np.nanmedian(xy[s:s + k], axis=0)
        edge_y = min(y, width - y)
        edge_x = min(x, length - x)
        if edge_x <= p["corner_m"] and edge_y <= p["corner_m"]:
            kind = "corner"
        elif edge_y <= p["touch_band_m"]:
            kind = "touchline"
        elif edge_x <= p["goal_area_depth_m"] and abs(y - width / 2) <= p["goal_area_half_width_m"]:
            kind = "goal_area"
        elif np.hypot(x - length / 2, y - width / 2) <= p["centre_m"]:
            kind = "centre"
        else:
            kind = "open"
        out.append((int(s), int(s + k), kind, float(p["placement_strength"][kind])))
    return out


def confirmed_onsets(ball, nearest_m, p):
    """Frames where a quiet ball is kicked, with a player at it just before (if required)."""
    have = ball["have"]
    speed = np.nan_to_num(ball["speed"], nan=0.0)
    quiet = ~have | (speed < p["still_speed"])
    c = np.concatenate(([0], np.cumsum(quiet)))
    t = np.arange(len(have))
    w = int(p["quiet_frames"])
    quiet_before = (c[t] - c[np.maximum(0, t - w)]) >= w
    moving = have & (speed >= p["kick_speed"])
    first = moving & ~np.concatenate(([False], moving[:-1])) & quiet_before
    cand = np.flatnonzero(first)
    if not p["use_onset_contact"]:
        return cand
    near = np.nan_to_num(nearest_m, nan=np.inf, posinf=np.inf) <= p["contact_m"]
    lb = int(p["contact_lookback"])
    return np.array([k for k in cand if near[max(0, k - lb):k + 1].any()], dtype=int)


def positive_stoppage_evidence(geo, params=None, set_piece_frames=None, blocks=None):
    """{'dead_add', 'intervals': [(start, end, score, signals)], 'summary'}."""
    p = params_with_defaults(params)
    ball = geo["ball"]
    n = len(ball["have"])
    nearest = np.minimum(geo["control"]["d_home"], geo["control"]["d_away"])

    anchors = []   # (frame, signal, value)
    if p["use_observed_out"]:
        anchors += [(int(f), "crossing", 1.0) for f in observed_out_frames(ball, p)]
    if p["use_extrapolation"]:
        anchors += [(f, "crossing", c) for f, c in extrapolated_crossings(ball, p) if c > 0]
    place = placements(ball, p) if p["use_placement"] else []
    anchors += [(s, "placement", st) for s, e, kind, st in place]
    sp = sorted(set_piece_frames or []) if p["use_set_piece"] else []
    anchors += [(max(0, int(f) - 75), "set_piece", 1.0) for f in sp]   # window opens 3 s before
    anchors.sort()
    onsets = confirmed_onsets(ball, nearest, p)

    motion = geo.get("motion")
    live_median = float(np.nanmedian(motion)) if motion is not None else None

    # Sustained live play: the first frame of every >= live_frames run of an observed moving ball.
    # Not used as a restart (reappearance is never a restart); used to DROP a candidate whose
    # confirmed onset would only come after play had visibly resumed.
    speed = np.nan_to_num(ball["speed"], nan=0.0)
    moving = ball["have"] & (speed >= p["live_speed"])
    live_starts = np.array([s for s, k in runs(moving) if k >= p["live_frames"]], dtype=int)

    dead_add = np.zeros(n)
    intervals, used_onsets = [], set()
    dropped_live = 0
    for a, _, _ in anchors:
        later = onsets[(onsets > a) & (onsets - a <= p["max_s"] * 25)]
        if not len(later):
            continue
        b = int(later[0])
        resumed = live_starts[(live_starts > a) & (live_starts < b - int(p["contact_lookback"]))]
        if len(resumed):
            dropped_live += 1
            continue                                     # play resumed before any confirmed onset
        if intervals and a < intervals[-1][1]:
            continue                                     # inside an interval already emitted
        signals = {"onset": 1.0}
        for f, kind, val in anchors:
            if a <= f < b + 1:
                signals[kind] = max(signals.get(kind, 0.0), val)
        for s, e, kind, st in place:
            if s < b and e > a:
                signals["placement"] = max(signals.get("placement", 0.0), st)
        for f in sp:
            if a <= f <= b + 50:
                signals["set_piece"] = 1.0
        if p["use_motion_support"] and live_median and b > a:
            if np.nanmedian(motion[a:b]) < p["motion_ratio"] * live_median:
                signals["motion_support"] = 0.5
        score = round(sum(signals.values()), 3)
        if score >= p["min_score"]:
            dead_add[a:b] += p["w_dead"]
            intervals.append((int(a), b, score, sorted(signals)))
            used_onsets.add(b)
    if p["use_onset_alive"]:
        for b in used_onsets:
            dead_add[b:b + int(p["onset_frames"])] -= p["w_onset"]

    kinds = {}
    for _, _, _, sig in intervals:
        key = "+".join(sig)
        kinds[key] = kinds.get(key, 0) + 1
    summary = {"anchors": len(anchors), "confirmed_onsets": int(len(onsets)),
               "dropped_play_resumed_first": dropped_live,
               "intervals": len(intervals), "signal_combinations": kinds,
               "interval_s_p10_p50_p90": (np.percentile([(b - a) / 25.0 for a, b, _, _ in intervals],
                                                        [10, 50, 90]).round(1).tolist() if intervals else []),
               "placements_by_kind": {k: sum(1 for _, _, kk, _ in place if kk == k) for k in p["placement_strength"]}}
    return {"dead_add": dead_add, "intervals": intervals, "summary": summary}
