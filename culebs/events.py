"""Smart Tagging as soft, noisy evidence (Experiment B).

The tags are automatic phases, not possessions (DATA.md). They are used in three bounded ways:

1. Team context. Each code's Team tag may mean "the team in possession" (BUILD UP) or "the team
   defending" (DEFENDING IN ...). Which one -- and how reliable -- is *measured*, per code, as the
   agreement between the tag's team and Experiment A's geometric control inside the tag, on
   frames where geometry is confident. No labels are involved; the sign and weight come out of
   the unlabelled match. A code whose agreement is near 50 % gets weight ~0.
2. Conflicts. Tags are combined as signed, weighted evidence, so the paired codes the tagger
   emits (BUILD UP for one side + DEFENDING IN ATTACKING THIRD for the other) reinforce rather
   than conflict. A frame is a genuine conflict only when, after weighting, some active tags
   point home and others point away; those contributions cancel and the frame is counted.
3. Restarts. A SET PIECES instance brackets a restart. The restart frame is detected inside the
   window from the ball itself (static -> moving); dead-ball evidence is added before it, and --
   if the tag names a team -- a small team prior after it, because the restarting side has the
   ball by the laws of the game. N/A set pieces (all throw-ins and kick-offs) give no team.

Tags outside graded play (19 halftime false positives, see the audit) are dropped, and the team
contribution per frame is capped so tags can never outvote clear ball-control geometry.
"""

import numpy as np

from starter import load_smart_tagging as lst

HOME_NAMES = ("barcelona",)
NA_VALUES = ("", "n/a", "na", "none")

DEFAULTS = {
    "team_nats_per_frame": 0.08,     # prior: weak; ~100 frames (4 s) of tag alone to beat 8 nats
    "team_cap_nats": 0.15,           # |sum of tag team evidence| per frame
    "min_confident_llr": 2.0,        # geometric frames used to measure a code's semantics
    "min_measured_frames": 250,      # below this a code's semantics are unknown -> weight 0
    "restart_kick_speed": 2.0,       # m/s; ball leaves a placed position
    "restart_dead_nats": 1.5,        # per frame, before the detected restart inside the window
    "restart_team_nats": 0.5,        # per frame, for restart_team_frames after it, if Team known
    "restart_team_frames": 50,       # 2 s
}


def params_with_defaults(params):
    out = dict(DEFAULTS)
    out.update(params or {})
    return out


def team_sign(value):
    """+1 home, -1 away, 0 unknown/ambiguous."""
    v = (value or "").strip().lower()
    if v in NA_VALUES:
        return 0
    return 1 if v in HOME_NAMES else -1


def load_events(graded):
    """Instances clipped to the video, with frames inside graded play only. Others dropped."""
    n = len(graded)
    out, dropped = [], 0
    for e in lst.load():
        s, t = max(0, e["start_frame"]), min(n, e["end_frame"])
        if t <= s or not graded[s:t].any():
            dropped += 1
            continue
        e = dict(e)
        e["start_frame"], e["end_frame"] = s, t
        e["sign"] = team_sign(e["tags"].get("Team"))
        out.append(e)
    return out, dropped


def code_semantics(events, geo_llr, graded, p):
    """{code: (weight, agreement, n_frames)}; weight in [-1, 1], sign = possessing(+)/defending(-)."""
    stats = {}
    for e in events:
        if e["code"] == "SET PIECES" or e["sign"] == 0:
            continue
        sl = slice(e["start_frame"], e["end_frame"])
        llr = geo_llr[sl][graded[sl]]
        llr = llr[np.abs(llr) >= p["min_confident_llr"]]
        agree = int(np.sum(np.sign(llr) == e["sign"]))
        a, n = stats.get(e["code"], (0, 0))
        stats[e["code"]] = (a + agree, n + len(llr))
    out = {}
    for code, (agree, n) in sorted(stats.items()):
        if n < p["min_measured_frames"]:
            out[code] = (0.0, None, n)
            continue
        rate = agree / float(n)
        out[code] = (round(2 * rate - 1, 4), round(rate, 4), n)
    return out


def restart_frame(e, ball_speed, ball_have, p):
    """First frame in the window where a present ball goes from slow to >= kick speed; else the
    window midpoint. Algorithmic, per instance -- never a hand-picked frame."""
    s, t = e["start_frame"], e["end_frame"]
    sp = np.where(ball_have[s:t], ball_speed[s:t], np.nan)
    moving = np.nan_to_num(sp, nan=0.0) >= p["restart_kick_speed"]
    slow_before = np.concatenate(([False], np.nan_to_num(sp[:-1], nan=np.inf) < p["restart_kick_speed"]))
    hits = np.flatnonzero(moving & slow_before)
    return (s + int(hits[0]), True) if len(hits) else ((s + t) // 2, False)


def event_evidence(events, geo, graded, params=None):
    """Per-frame additive evidence: team log-odds (home +) and dead-ball log-odds, plus summary."""
    p = params_with_defaults(params)
    n = len(graded)
    semantics = code_semantics(events, geo["control"]["llr"], graded, p)

    team = np.zeros(n)
    to_home = np.zeros(n)
    to_away = np.zeros(n)
    both_tagged = np.zeros((2, n), dtype=bool)
    for e in events:
        if e["code"] == "SET PIECES" or e["sign"] == 0:
            continue
        weight = semantics.get(e["code"], (0.0, None, 0))[0]
        sl = slice(e["start_frame"], e["end_frame"])
        contribution = p["team_nats_per_frame"] * weight * e["sign"]
        team[sl] += contribution
        if contribution > 0:
            to_home[sl] += contribution
        elif contribution < 0:
            to_away[sl] -= contribution
        both_tagged[0 if e["sign"] > 0 else 1, sl] = True
    team = np.clip(team, -p["team_cap_nats"], p["team_cap_nats"])

    dead = np.zeros(n)
    restarts, detected = [], 0
    for e in events:
        if e["code"] != "SET PIECES":
            continue
        r, found = restart_frame(e, geo["ball"]["speed"], geo["ball"]["have"], p)
        detected += found
        dead[e["start_frame"]:r] += p["restart_dead_nats"]
        if e["sign"]:
            after = slice(r, min(n, r + p["restart_team_frames"]))
            team[after] += p["restart_team_nats"] * e["sign"]
        restarts.append(r)

    team[~graded] = 0.0
    dead[~graded] = 0.0
    conflict = (to_home > 0) & (to_away > 0) & graded
    both = both_tagged[0] & both_tagged[1] & graded
    summary = {
        "params": p,
        "instances_used": len(events),
        "code_semantics": {c: {"weight": w, "agreement": a, "confident_frames": k}
                           for c, (w, a, k) in semantics.items()},
        "both_teams_tagged_graded_pct": round(100 * float(both.sum() / graded.sum()), 3),
        "weighted_evidence_conflict_graded_pct": round(100 * float(conflict.sum() / graded.sum()), 3),
        "set_pieces": sum(1 for e in events if e["code"] == "SET PIECES"),
        "set_pieces_team_na": sum(1 for e in events if e["code"] == "SET PIECES" and e["sign"] == 0),
        "restarts_detected_from_ball": int(detected),
    }
    return {"team": team, "dead": dead, "conflict": conflict, "restarts": restarts,
            "summary": summary}
