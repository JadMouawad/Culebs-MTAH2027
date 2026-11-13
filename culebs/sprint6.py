"""Sprint 6 candidates on top of frozen e110a (all optional, independently ablatable).

C1  gap_consensus_evidence   missing-ball stoppage detection that ABSTAINS unless independent
                             signals agree. For every ball-missing gap >= min_gap_s inside play:
        exit        the ball was last seen within exit_m of a line moving outward, or the last
                    reliable trajectory reaches a line within max_cross_s (stoppage_pe)
        slowdown    collective motion during the gap < slow_ratio x the motion in the
                    pre_s seconds before it (change relative to the previous structure)
        restart_pos the ball reappears static at a restart location (touchline band, corner,
                    goal area, centre spot) -- stoppage_pe.placements on the frames after the gap
        restart_kick a confirmed restart onset (quiet ball kicked with a player at it) within
                    kick_window_s after the ball reappears
    The gap gets w_gap nats of dead evidence only if >= min_signals of the enabled signals agree;
    otherwise nothing (abstention). Ball absence alone, or slow players alone, can never assert a
    stoppage. The interval runs from the gap start to the restart kick (or the reappearance).

D1  persistence_llr          llr' = (1 - a) llr + a * mean(llr over +/- half_window frames)
D2  velocity_gated_llr       llr' = llr * g, g = with_gain when the ball moves with the nearest
                             player (relative speed < with_ms), away_gain when it moves away from
                             every player (relative speed > away_ms to both nearest players), else 1.
                             NOTE: majority-vote models use only the SIGN of llr, so gains > 0 are
                             inert there (first design: output identical to e110a). The evaluated
                             D2 uses away_gain = 0: evidence is removed (ball loose, previous team
                             holds) when the ball moves away from both teams' nearest players.

Parameters are physical priors fixed before evaluation (docs/SPRINT6_MODEL_SELECTION.md).
"""

import numpy as np

from culebs import stoppage_pe as pe
from culebs.diagnostics import runs
from games import GAME, GAMES

GAP_DEFAULTS = {"min_gap_s": 1.0, "exit_m": 3.0, "slow_ratio": 0.7, "pre_s": 5.0, "kick_window_s": 3.0,
                "min_signals": 3, "w_gap": 3.0, "max_gap_s": 90.0,
                "use_exit": True, "use_slowdown": True, "use_restart_pos": True, "use_restart_kick": True}
PERSIST_DEFAULTS = {"a": 0.5, "half_window": 25}
VELOCITY_DEFAULTS = {"with_ms": 2.0, "away_ms": 6.0, "with_gain": 1.5, "away_gain": 0.5}


def gap_consensus_evidence(geo, params=None):
    p = dict(GAP_DEFAULTS)
    p.update(params or {})
    L, W = GAMES[GAME]["pitch"]
    ball = geo["ball"]
    have, xy, speed = ball["have"], ball["xy"], np.nan_to_num(ball["speed"], nan=0.0)
    motion = geo["motion"]
    n = len(have)
    nearest = np.minimum(geo["control"]["d_home"], geo["control"]["d_away"])
    cross = {f for f, c in pe.extrapolated_crossings(ball, pe.DEFAULTS) if c > 0}
    onsets = pe.confirmed_onsets(ball, nearest, pe.DEFAULTS)
    place = pe.placements(ball, pe.DEFAULTS)
    restart_kinds = {"touchline", "corner", "goal_area", "centre"}
    dead_add = np.zeros(n)
    counts = {"gaps": 0, "asserted": 0, "abstained": 0, "signal_hits": {}}
    for s, m in runs(~have):
        if m < p["min_gap_s"] * 25 or m > p["max_gap_s"] * 25 or s < 2 or s + m >= n:
            continue
        counts["gaps"] += 1
        sig = {}
        if p["use_exit"]:
            x, y = xy[s - 1]
            dist = [x, L - x, y, W - y]
            side = int(np.argmin(dist))
            v = xy[s - 1] - xy[s - 2] if have[s - 2] else np.zeros(2)
            outward = (-v[0], v[0], -v[1], v[1])[side] > 0
            near_crossing = any(s - 1 <= f <= s + 25 for f in cross)
            sig["exit"] = bool((dist[side] <= p["exit_m"] and outward) or near_crossing)
        if p["use_slowdown"]:
            pre = motion[max(0, s - int(p["pre_s"] * 25)):s]
            during = motion[s:s + m]
            if np.isfinite(pre).any() and np.isfinite(during).any():
                sig["slowdown"] = bool(np.nanmedian(during) < p["slow_ratio"] * np.nanmedian(pre))
            else:
                sig["slowdown"] = False
        end = s + m
        if p["use_restart_pos"]:
            sig["restart_pos"] = any(a_ >= end and a_ <= end + 25 * 2 and kind in restart_kinds
                                     for a_, _, kind, _ in place)
        kick = None
        if p["use_restart_kick"]:
            later = onsets[(onsets >= end) & (onsets <= end + int(p["kick_window_s"] * 25))]
            kick = int(later[0]) if len(later) else None
            sig["restart_kick"] = kick is not None
        for k, v in sig.items():
            counts["signal_hits"][k] = counts["signal_hits"].get(k, 0) + int(v)
        if sum(sig.values()) >= p["min_signals"]:
            dead_add[s:(kick if kick is not None else end)] += p["w_gap"]
            counts["asserted"] += 1
        else:
            counts["abstained"] += 1
    return dead_add, counts


def persistence_llr(llr, params=None):
    p = dict(PERSIST_DEFAULTS)
    p.update(params or {})
    k = int(p["half_window"])
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    return (1 - p["a"]) * llr + p["a"] * np.convolve(llr, kern, mode="same")


def velocity_gated_llr(llr, trk, geo, params=None):
    p = dict(VELOCITY_DEFAULTS)
    p.update(params or {})
    L, W = GAMES[GAME]["pitch"]
    pm = np.where(geo["usable"][:, :, None], trk.players[:, :, :2] * [L, W], np.nan)
    ball = geo["ball"]["xy"]
    vel_p = np.vstack([np.full((1,) + pm.shape[1:], np.nan), np.diff(pm, axis=0) * 25.0])
    vel_b = np.vstack([[np.nan, np.nan], np.diff(ball, axis=0) * 25.0])
    d = np.hypot(pm[:, :, 0] - ball[:, None, 0], pm[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    rel = []
    for team in (0, 1):
        dt = np.where((trk.team_of_slot == team)[None, :], d, np.inf)
        k = np.argmin(dt, axis=1)
        r = np.hypot(*(vel_b - vel_p[np.arange(len(ball)), k]).T)
        rel.append(np.where(np.isfinite(dt.min(axis=1)), r, np.nan))
    rel_near = np.where(llr > 0, rel[0], rel[1])          # the team the evidence points to
    gain = np.ones(len(llr))
    gain[np.nan_to_num(rel_near, nan=np.inf) < p["with_ms"]] = p["with_gain"]
    away = (np.nan_to_num(rel[0], nan=-1) > p["away_ms"]) & (np.nan_to_num(rel[1], nan=-1) > p["away_ms"])
    gain[away] = p["away_gain"]
    return llr * gain
