"""Provider-agnostic learning representation of one match (numpy only).

Everything is derived from a Tracking tuple through the same reliability, de-duplication and ball
kinematics as e110a (culebs.features), then sampled every STRIDE frames (5 fps by default).

Arrays per sampled frame t (m = ceil(n / STRIDE)):

    common (m, C)        team-symmetric scalars: ball observed / interpolated / missing masks,
                         ball speed (0 + mask when missing), log time since the ball was last seen
                         and until it is next seen, distance of the reference point to the goal and
                         touch lines, out / static / exit-gap / PES evidence, collective motion,
                         in-flight flag, stale-reference flag
    team   (m, 2, T)     per team [home, away]: k nearest distances to the reference point, counts
                         within 2 / 5 / 10 m, closest player's speed and speed relative to the ball,
                         team centroid distance, mean speed, the e110a control evidence pointing to
                         this team (max(+-llr, 0))
    nodes  (m, 2, K, N)  per team the K players nearest the reference point: dx, dy (metres / 10),
                         distance, vx, vy relative to the ball (or absolute when the ball is missing),
                         speed, valid -- unordered sets for the graph model
    y      (m,)          labels 0 H / 1 A / 2 D / 3 unknown (ignored)

The reference point is the ball where the tracking has it, else the LAST SEEN ball position with a
"stale" flag -- a missing ball is never presented as an observation and never zero-filled.
Symmetries: swapping the team axis (and H<->A in y) is an exact team swap; negating dx/vx or
dy/vy in `nodes` is a pitch reflection (everything else is reflection-invariant by construction).
"""

import warnings

import numpy as np

from culebs.decode import params_with_defaults as decode_defaults
from culebs.features import _metres, geometric_evidence
from culebs.stoppage_pe import positive_stoppage_evidence

STRIDE = 5
MAX_BALL_MS = 40.0
K = 8
COMMON = ("observed", "interpolated", "missing", "ball_speed", "log_since", "log_until", "goal_line_m",
          "touch_line_m", "out", "static", "exited", "pes", "motion", "in_flight", "stale")
TEAM = ("d1", "d2", "d3", "n2", "n5", "n10", "near_speed", "near_rel_speed", "centroid_d", "mean_speed", "llr_for")
NODE = ("dx", "dy", "dist", "vx", "vy", "speed", "valid")
E110A_DECODE = {"dead_model": "conservative", "smooth_window": 50, "positive_stoppage": True}


def _velocity(pos, half=2):
    """Central difference over +/- half frames (m/s); NaN where either end is missing."""
    v = np.full(pos.shape, np.nan)
    v[half:-half] = (pos[2 * half:] - pos[:-2 * half]) * (25.0 / (2 * half))
    return v


def _fill_last(have):
    idx = np.where(have, np.arange(len(have)), -1)
    last = np.maximum.accumulate(idx)
    return last


def build(trk, graded, stride=STRIDE, k=K, geo=None):
    """dict(common, team, nodes, frames, n) for one match. `graded` marks frames inside play."""
    if geo is None:
        geo = geometric_evidence(trk, graded, None)
    players, _, pitch = _metres(trk)
    length, width = pitch
    n = len(players)
    ball = geo["ball"]
    have, xy = ball["have"], ball["xy"]
    usable = geo["usable"]

    last = _fill_last(have)
    nxt = _fill_last(have[::-1])[::-1]
    nxt = np.where(nxt >= 0, n - 1 - nxt, -1)
    t = np.arange(n)
    since = np.where(last >= 0, (t - last) / 25.0, 120.0)
    until = np.where(nxt >= 0, (nxt - t) / 25.0, 120.0)
    ref = np.where(have[:, None], xy, np.where((last >= 0)[:, None], xy[np.clip(last, 0, n - 1)],
                                              np.array([length / 2, width / 2])))
    stale = ~have

    dp = decode_defaults(E110A_DECODE)
    pes = positive_stoppage_evidence(geo, dp["pes_params"], None)["dead_add"]
    pv = _velocity(np.where(usable[:, :, None], players, np.nan))
    bv = _velocity(np.where(have[:, None], xy, np.nan))
    llr = geo["control"]["llr"]

    f = np.arange(0, n, stride)
    m = len(f)
    P = np.where(usable[f][:, :, None], players[f], np.nan)          # (m, S, 2)
    V = pv[f]
    R = ref[f]
    BV = np.nan_to_num(bv[f])
    sp = np.hypot(BV[:, 0], BV[:, 1])
    BV = BV * np.minimum(1.0, MAX_BALL_MS / np.maximum(sp, 1e-6))[:, None]   # physical cap
    d = np.hypot(P[:, :, 0] - R[:, None, 0], P[:, :, 1] - R[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    spd = np.nan_to_num(np.hypot(V[:, :, 0], V[:, :, 1]))
    rel = np.hypot(np.nan_to_num(V[:, :, 0]) - BV[:, None, 0], np.nan_to_num(V[:, :, 1]) - BV[:, None, 1])

    team = np.zeros((m, 2, len(TEAM)), np.float32)
    nodes = np.zeros((m, 2, k, len(NODE)), np.float32)
    for side in (0, 1):
        sel = np.flatnonzero(trk.team_of_slot == side)
        ds = d[:, sel]
        order = np.argsort(ds, axis=1, kind="stable")
        dsort = np.take_along_axis(ds, order, axis=1)
        fin = np.isfinite(dsort)
        dcl = np.where(fin, np.minimum(dsort, 60.0), 60.0)
        for j in range(3):
            team[:, side, j] = (dcl[:, j] if dcl.shape[1] > j else 60.0) / 10.0
        team[:, side, 3] = (ds < 2).sum(1)
        team[:, side, 4] = (ds < 5).sum(1)
        team[:, side, 5] = (ds < 10).sum(1) / 2.0
        first = sel[order[:, 0]]
        team[:, side, 6] = spd[np.arange(m), first] / 5.0
        team[:, side, 7] = np.where(have[f], np.minimum(rel[np.arange(m), first], 50.0) / 10.0, 0.0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx = np.nanmean(P[:, sel, 0], axis=1)
            cy = np.nanmean(P[:, sel, 1], axis=1)
            team[:, side, 9] = np.nan_to_num(np.nanmean(np.where(np.isfinite(ds), spd[:, sel], np.nan), axis=1)) / 5.0
        team[:, side, 8] = np.nan_to_num(np.hypot(cx - R[:, 0], cy - R[:, 1]), nan=60.0) / 10.0
        team[:, side, 10] = np.clip((llr[f] if side == 0 else -llr[f]), 0, None)
        kk = min(k, len(sel))
        top = sel[order[:, :kk]]
        rows = np.arange(m)[:, None]
        valid = np.isfinite(d[rows, top])
        dx = np.where(valid, P[rows, top, 0] - R[:, None, 0], 0.0)
        dy = np.where(valid, P[rows, top, 1] - R[:, None, 1], 0.0)
        vx = np.nan_to_num(V[rows, top, 0]) - np.where(have[f], BV[:, 0], 0.0)[:, None]
        vy = np.nan_to_num(V[rows, top, 1]) - np.where(have[f], BV[:, 1], 0.0)[:, None]
        vx, vy = np.clip(vx, -50, 50), np.clip(vy, -50, 50)
        block = np.stack([dx / 10.0, dy / 10.0, np.hypot(dx, dy) / 10.0, vx / 10.0, vy / 10.0,
                          spd[rows, top] / 5.0, valid.astype(float)], axis=-1)
        nodes[:, side, :kk] = np.where(valid[..., None], block, 0.0)

    obs, itp = ball["observed"][f], ball["interpolated"][f]
    goal = np.minimum(R[:, 0], length - R[:, 0])
    touch = np.minimum(R[:, 1], width - R[:, 1])
    common = np.stack([
        obs, itp, ~have[f], np.where(have[f], np.clip(np.nan_to_num(ball["speed"][f]), 0, MAX_BALL_MS), 0.0) / 10.0,
        np.log1p(since[f]), np.log1p(until[f]), np.clip(goal, -5, 60) / 10.0, np.clip(touch, -5, 40) / 10.0,
        ball["out"][f], geo["static"][f], geo["exited"][f], pes[f] / 3.0,
        np.nan_to_num(geo["motion"][f], nan=0.0) / 3.0, np.nan_to_num(geo["control"]["in_flight"][f]).astype(float),
        stale[f]], axis=1).astype(np.float32)
    return {"common": common, "team": team, "nodes": nodes, "frames": f, "n": n, "stride": stride}


def defreeze(trk, min_frames=4):
    """Mark repeated frozen-ball coordinates as missing (external-data preprocessing).

    Exact repeated positions can encode a provider artifact instead of a ball
    observation. Runs below the fixed threshold are unchanged. The threshold
    and source-data provenance are recorded in the external research protocol.
    """
    from culebs.atd import Tracking
    from culebs.diagnostics import runs
    b = trk.ball.copy()
    same = np.zeros(len(b), bool)
    same[1:] = np.all(b[1:, :2] == b[:-1, :2], axis=1) & ~np.isnan(b[1:, 0])
    hide = np.zeros(len(b), bool)
    for s, n in runs(same):
        if n + 1 >= min_frames:
            hide[s - 1:s + n] = True
    b[hide] = np.nan
    return Tracking(trk.players, b, trk.team_of_slot, trk.team_names, trk.source), hide


def swap(rec, y=None):
    """Exact team swap: exchange the team axis; H<->A in labels."""
    out = dict(rec)
    out["team"] = rec["team"][:, ::-1].copy()
    out["nodes"] = rec["nodes"][:, ::-1].copy()
    if y is None:
        return out
    y2 = y.copy()
    y2[y == 0], y2[y == 1] = 1, 0
    return out, y2


def upsample(values, frames, n):
    """Per-sampled-frame values -> per-frame (nearest sample, centred)."""
    stride = int(frames[1] - frames[0]) if len(frames) > 1 else 1
    idx = np.clip(np.rint(np.arange(n) / stride).astype(int), 0, len(frames) - 1)
    return values[idx]
