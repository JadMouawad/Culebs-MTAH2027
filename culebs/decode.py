"""Per-frame emissions over {H, A, D} and a deterministic duration-aware decoder (Experiment C).

Emissions
---------
    ball visible:  p_dead = sigmoid(dead_prior + w_out*out + w_motion*log LR_dead(motion)
                                    + w_static*static + event dead evidence)
    ball missing:  p_dead = sigmoid(w_missing*logit(P(dead | motion, missing)) + w_exit*exited
                                    + event dead evidence)
    where P(dead | motion, missing) is the label-free mixture-proportion estimate in
    culebs.features.missing_ball_dead_probability (it already conditions on motion, so the
    generic prior and motion terms are not added again on those frames).
    q_home  = sigmoid(team log-odds)                      geometry (+ events)

    e_H = log(1 - p_dead) + log(2 q_home)
    e_A = log(1 - p_dead) + log(2 (1 - q_home))
    e_D = log(p_dead)

The team term is a likelihood ratio against "team unknown" (q = 0.5 -> 0), so a frame with no
ball reads as *no evidence about the team*, not as two half-likely alternatives that D would beat.

Decoder
-------
Viterbi over a left-to-right chain per label: a label must be held for min_frames[label] before
it can be left, and leaving it costs a switch penalty (nats). H<->A needs accumulated control
evidence worth switch_nats; alive->D and D->alive have their own costs. Decoding runs inside each
graded block separately and never across halftime; ungraded frames are written X (never scored).
Exact ties inside the Viterbi resolve to the lowest state index, so the output is a pure function
of the inputs; a block with no team evidence at all is written X rather than defaulting to H.

Why no X inside graded frames: the official scorer forward-fills X before computing spellF1
(possession_metrics.fill_abstentions), so an X inside a segment is identical to repeating the
segment's label on the ranking metric, and it is charged as a miss by macroF1.
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

DEFAULTS = {
    "dead_prior": -0.5,     # prior: ~38 % dead (effective playing time ~55-60 of ~95 minutes)
    "w_out": 4.0,           # ball centre past a line: strong evidence of dead ball
    "w_motion": 1.0,        # x log f_dead/f_live of collective motion (label-free estimate)
    "w_static": 1.5,        # a placed ball (static >= 1 s)
    "w_missing": 1.0,       # x logit of the unsupervised P(dead | motion, ball missing)
    "w_exit": 2.0,          # the gap began with the ball leaving the pitch
    "dead_clip": 6.0,       # |dead log-odds| cap per frame
    "team_clip": 4.0,       # |team log-odds| cap per frame after fusion
    "switch_nats": 24.0,    # H<->A: ~6 frames of unambiguous control, or ~1 s of moderate
    "stop_nats": 12.0,      # alive -> D
    "restart_nats": 12.0,   # D -> alive
    "min_possession_frames": 5,   # prior: 0.2 s anti-flicker only; short spells stay possible
    "min_dead_frames": 25,        # prior: no stoppage shorter than ~1 s
    "smooth_window": 50,          # for the undecoded (majority-vote) variants
    # Majority-vote tie rule. "hold" (default since Sprint 3 review): a tied vote keeps the
    # previous output label -- possession stays with the last team (CHALLENGE.md) -- and is
    # neutral between the teams. "home": the Sprint 1-3 rule (argmax order H, A, D), which sent
    # 39-225 tied frames per match mostly to H; kept only so published configs reproduce.
    "majority_ties": "hold",
    # Part B: which dead-ball evidence is used
    #   "mpe"          (Sprint 1) motion likelihood ratio + P(dead | motion, ball missing)
    #   "conservative" positive evidence only: ball past a line, exit gaps, placed ball, restart
    #                  windows from events. Ball missingness and motion carry no dead evidence.
    "dead_model": "mpe",
    # Part C: restart-team model
    #   "none"   a restart may go to either team at the same cost (Sprint 1)
    #   "cause"  D is split by the team that had the ball; restarting with the same team or the
    #            opponent costs -log P(relation | cause), cause = "exit" if the ball left the pitch
    #            within restart_window_frames, else "whistle"
    "restart_model": "none",
    "p_opponent_after_exit": 0.90,     # prior from the laws (throw-in/goal kick/corner go to the
                                       # side that did not touch last); calibrated on other matches
    "p_same_after_whistle": 0.60,      # prior: a foul is usually on the team in possession
    "restart_window_frames": 1125,     # 45 s: an exit this recent explains the stoppage
    # Sprint 3 candidate (culebs/stoppage.py): line-crossing -> kick-onset stoppage intervals and
    # restart-onset evidence. False = off (Sprints 1-2 behaviour, byte-identical).
    "short_stoppage": False,
    # Sprint 4 candidate (culebs/stoppage_pe.py): multi-signal positive-evidence stoppages for
    # the majority-vote family. False = off (all earlier configs byte-identical). Component
    # switches and thresholds go in "pes_params".
    "positive_stoppage": False,
    "pes_params": None,
    "restart_min_frames": 0,           # >0: after a restart the restarting team holds the ball
                                       # at least this long (the taker must play it), so the
                                       # restart team is decided on summed evidence over the
                                       # window, not on its first frame. 0 = off (Sprint 1).
}


def params_with_defaults(params):
    out = dict(DEFAULTS)
    out.update(params or {})
    return out


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def emissions(geo, ev=None, params=None):
    """(n, 3) log emission scores for H, A, D, and the fused team / dead log-odds."""
    p = params_with_defaults(params)
    team = geo["control"]["llr"].copy()
    if p["dead_model"] == "mpe":
        visible = (p["dead_prior"] + p["w_out"] * geo["ball"]["out"]
                   + p["w_motion"] * geo["motion_log_lr"] + p["w_static"] * geo["static"])
        missing = (p["w_missing"] * _logit(np.nan_to_num(geo["p_dead_missing"], nan=0.5))
                   + p["w_exit"] * geo["exited"])
    elif p["dead_model"] == "conservative":
        visible = p["dead_prior"] + p["w_out"] * geo["ball"]["out"] + p["w_static"] * geo["static"]
        missing = p["dead_prior"] + p["w_exit"] * geo["exited"]
    else:
        raise ValueError("unknown dead_model {!r}".format(p["dead_model"]))
    dead = np.where(geo["ball"]["have"], visible, missing)
    if ev is not None:
        team = team + ev["team"]
        dead = dead + ev["dead"]
    if p["short_stoppage"]:
        from culebs.stoppage import stoppage_evidence
        dead = dead + stoppage_evidence(geo, p.get("stoppage_params"))["dead_add"]
    if p["positive_stoppage"]:
        from culebs.stoppage_pe import positive_stoppage_evidence
        restarts = ev["restarts"] if ev is not None else None
        dead = dead + positive_stoppage_evidence(geo, p["pes_params"], restarts)["dead_add"]
    team = np.clip(team, -p["team_clip"], p["team_clip"])
    dead = np.clip(dead, -p["dead_clip"], p["dead_clip"])

    alive = _log_sigmoid(-dead)
    e = np.empty((len(team), 3))
    e[:, H] = alive + np.log(2.0) + _log_sigmoid(team)
    e[:, A] = alive + np.log(2.0) + _log_sigmoid(-team)
    e[:, D] = _log_sigmoid(dead)
    return e, team, dead


def viterbi(e, min_frames, switch, extra=None):
    """Best label path for one contiguous block. e: (T, K); min_frames: K ints >= 1;
    switch: (K, K) penalties in nats (diagonal ignored, np.inf forbids a transition);
    extra: optional list of (from, to, cost array of length T) added on top of switch at t."""
    T, K = e.shape
    sizes = [int(max(1, m)) for m in min_frames]
    base = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(int)
    last = base + np.array(sizes) - 1
    M = int(sum(sizes))
    label_of = np.repeat(np.arange(K), sizes)
    extra_into = {k: [(j, c) for j, kk, c in (extra or []) if kk == k] for k in range(K)}

    delta = np.full(M, -np.inf)
    delta[last] = e[0]                     # a block may start in any label, already "held"
    back = np.zeros((T, M), dtype=np.int32)
    for t in range(1, T):
        new = np.full(M, -np.inf)
        ptr = np.zeros(M, dtype=np.int32)
        ends = delta[last]
        for k in range(K):
            b, l = base[k], last[k]
            # enter label k from the held end of another label
            cand = ends - switch[:, k]
            for j, cost in extra_into[k]:
                cand[j] -= cost[t]
            cand[k] = -np.inf
            j = int(np.argmax(cand))
            enter_val, enter_from = cand[j], last[j]
            if sizes[k] == 1:
                if delta[l] >= enter_val:
                    new[l], ptr[l] = delta[l], l
                else:
                    new[l], ptr[l] = enter_val, enter_from
                continue
            new[b], ptr[b] = enter_val, enter_from
            if sizes[k] > 2:
                new[b + 1:l] = delta[b:l - 1]
                ptr[b + 1:l] = np.arange(b, l - 1)
            # the held end: stay, or arrive from the last chain step
            if delta[l] >= delta[l - 1]:
                new[l], ptr[l] = delta[l], l
            else:
                new[l], ptr[l] = delta[l - 1], l - 1
        new += e[t, label_of]
        delta, back[t] = new, ptr

    state = int(last[int(np.argmax(delta[last]))])
    path = np.empty(T, dtype=np.int8)
    for t in range(T - 1, -1, -1):
        path[t] = label_of[state]
        state = back[t, state]
    return path


def switch_matrix(p):
    s = np.zeros((3, 3))
    s[H, A] = s[A, H] = p["switch_nats"]
    s[H, D] = s[A, D] = p["stop_nats"]
    s[D, H] = s[D, A] = p["restart_nats"]
    return s


def recent_exit(geo, window):
    """True where the ball was past a line, or a gap began with it leaving, within `window`
    frames up to and including t -- the stoppage is then most likely an exit (throw-in, goal
    kick, corner), after which the laws give the restart to the other side."""
    hit = (geo["ball"]["out"] | geo["exited"]).astype(np.int64)
    c = np.concatenate(([0], np.cumsum(hit)))
    t = np.arange(len(hit))
    return (c[t + 1] - c[np.maximum(0, t + 1 - window)]) > 0


def restart_costs(geo, p):
    """(cost_same, cost_opponent) per frame, in nats, for a D -> alive restart."""
    exit_ = recent_exit(geo, int(p["restart_window_frames"]))
    po = float(np.clip(p["p_opponent_after_exit"], 1e-3, 1 - 1e-3))
    ps = float(np.clip(p["p_same_after_whistle"], 1e-3, 1 - 1e-3))
    same = np.where(exit_, -np.log(1 - po), -np.log(ps))
    opp = np.where(exit_, -np.log(po), -np.log(1 - ps))
    return same, opp, exit_


def decode(e, blocks, params=None, geo=None):
    """Labels for the whole video: decoded inside each (first, last) block, X elsewhere."""
    p = params_with_defaults(params)
    out = np.full(len(e), "X", dtype="<U1")
    mins = (p["min_possession_frames"], p["min_possession_frames"], p["min_dead_frames"])
    if p["restart_model"] == "none":
        for first, last in blocks:
            path = viterbi(e[first:last + 1], mins, switch_matrix(p))
            out[first:last + 1] = LABELS[path]
        return _unknown_team_to_x(out, e, blocks)
    if p["restart_model"] != "cause":
        raise ValueError("unknown restart_model {!r}".format(p["restart_model"]))
    if geo is None:
        raise ValueError("restart_model 'cause' needs the geometric evidence")
    same, opp, _ = restart_costs(geo, p)
    hold = int(p["restart_min_frames"])
    if hold <= 0:
        # labels H, A, D-after-H, D-after-A
        DH, DA = 2, 3
        s = np.full((4, 4), np.inf)
        s[H, A] = s[A, H] = p["switch_nats"]
        s[H, DH] = s[A, DA] = p["stop_nats"]
        s[DH, H] = s[DH, A] = s[DA, A] = s[DA, H] = p["restart_nats"]
        e4 = np.concatenate([e, e[:, D:D + 1]], axis=1)
        to_label = np.array(["H", "A", "D", "D"])
        sizes = mins + (p["min_dead_frames"],)
        extra_for = lambda sl: [(DH, H, same[sl]), (DA, A, same[sl]),  # noqa: E731
                                (DH, A, opp[sl]), (DA, H, opp[sl])]
    else:
        # labels H, A, D-after-H, D-after-A, H-restarting, A-restarting. A restart enters HR/AR,
        # which must be held for `hold` frames and then continues as ordinary H/A at no cost.
        DH, DA, HR, AR = 2, 3, 4, 5
        s = np.full((6, 6), np.inf)
        s[H, A] = s[A, H] = s[HR, A] = s[AR, H] = p["switch_nats"]
        s[H, DH] = s[A, DA] = s[HR, DH] = s[AR, DA] = p["stop_nats"]
        s[DH, HR] = s[DH, AR] = s[DA, AR] = s[DA, HR] = p["restart_nats"]
        s[HR, H] = s[AR, A] = 0.0
        e4 = np.concatenate([e, e[:, D:D + 1], e[:, H:H + 1], e[:, A:A + 1]], axis=1)
        to_label = np.array(["H", "A", "D", "D", "H", "A"])
        sizes = mins + (p["min_dead_frames"], hold, hold)
        extra_for = lambda sl: [(DH, HR, same[sl]), (DA, AR, same[sl]),  # noqa: E731
                                (DH, AR, opp[sl]), (DA, HR, opp[sl])]
    for first, last in blocks:
        sl = slice(first, last + 1)
        path = viterbi(e4[sl], sizes, s, extra_for(sl))
        out[sl] = to_label[path]
    return _unknown_team_to_x(out, e, blocks)


def _unknown_team_to_x(out, e, blocks):
    """A block whose H and A emissions are identical everywhere carries no team evidence, so
    the decoder's H/A choice there is only its lowest-index tie-break. Write X instead of
    defaulting to home. (Never triggered on the competition or Metrica matches; a guard.)"""
    for first, last in blocks:
        sl = slice(first, last + 1)
        if np.allclose(e[sl, H], e[sl, A]):
            block = out[sl]
            block[(block == "H") | (block == "A")] = "X"
    return out


def frame_labels(team, dead, blocks):
    """Undecoded per-frame labels: D where p_dead > 0.5, else the team the evidence points to.
    Where team evidence is exactly neutral the last evidenced team is held (possession stays with
    the last team -- CHALLENGE.md), back-filled at the head of a block. Never biased to H."""
    codes = np.full(len(team), D, dtype=np.int8)
    for first, last in blocks:
        t = team[first:last + 1]
        side = np.where(t > 0, H, np.where(t < 0, A, -1))
        idx = np.where(side >= 0, np.arange(len(t)), -1)
        carry = np.maximum.accumulate(idx)
        known = np.flatnonzero(side >= 0)
        carry[carry < 0] = known[0] if len(known) else 0
        # no team evidence anywhere in the block: the team is unknown (-1), never home by default
        held = side[carry] if len(known) else np.full(len(t), -1)
        codes[first:last + 1] = np.where(dead[first:last + 1] > 0, D, held)
    return codes


def majority(labels3, window, blocks, ties="hold"):
    """Centred +/-window majority over H/A/D codes (-1 = unknown, not counted) inside each block.

    ties="hold": a frame whose top count is shared keeps the previous output label; tied frames
    at the head of a block take the first untied label after them; a block with no untied frame
    at all is X. ties="home": legacy argmax order H, A, D.
    """
    n = len(labels3)
    out = np.full(n, "X", dtype="<U1")
    for first, last in blocks:
        seg = labels3[first:last + 1]
        cum = [np.concatenate(([0], np.cumsum(seg == k))) for k in range(3)]
        i = np.arange(len(seg))
        lo, hi = np.maximum(0, i - window), np.minimum(len(seg), i + window + 1)
        counts = np.stack([c[hi] - c[lo] for c in cum], axis=1)
        winner = np.argmax(counts, axis=1)
        if ties == "home":
            out[first:last + 1] = LABELS[winner]
            continue
        if ties != "hold":
            raise ValueError("unknown majority tie rule {!r}".format(ties))
        top = counts.max(axis=1)
        tied = ((counts == top[:, None]).sum(axis=1) > 1) | (top == 0)
        untied = np.flatnonzero(~tied)
        if not len(untied):
            continue                                   # stays X: nothing to decide on
        idx = np.maximum.accumulate(np.where(~tied, i, -1))
        idx[idx < 0] = untied[0]                       # head of the block: back-fill
        out[first:last + 1] = LABELS[winner[idx]]
    return out
