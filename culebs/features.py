"""Team-level geometric evidence from the ATD (Experiment A).

Nothing here follows a player identity: slots are tracklets, so every quantity is "the nearest
reliable player of each team", "how many opponents are close", "how fast is everyone moving".
That is what survives identity switches.

Pipeline, all vectorised over frames:

    reliability_mask      drop samples on very short tracklets and around single-frame teleports
    duplicate_mask        drop a same-team detection within dup_radius of another (tracker clones)
    ball_kinematics       metres, velocity from consecutive valid positions, short-gap interpolation
    team_control          nearest distance per team, separation, opposition pressure -> log-odds
    dead-ball evidence    ball off the pitch, a static ball, collective motion as an anchored
                          likelihood ratio, and P(dead | motion, ball missing) -- both from the
                          label-free mixture-proportion estimate

Every threshold is a named parameter with a physical meaning; DEFAULTS documents where each value
comes from. None of them was fitted to labels of this match.
"""

import warnings

import numpy as np

from culebs.diagnostics import runs
from games import GAME, GAMES

HOME, AWAY = 0, 1

# Parameter origins. "prior" = football / physics knowledge fixed before looking at any labels.
DEFAULTS = {
    # tracking reliability
    "min_segment_frames": 13,     # prior: a tracklet shorter than ~0.5 s is mostly re-ID noise
    "max_player_step_m": 1.0,     # prior: 1 m in 40 ms = 25 m/s, beyond any human sprint
    "dup_radius_m": 0.5,          # prior: two bodies cannot be closer than ~0.5 m centre to centre
    # ball
    "max_ball_step_m": 1.6,       # prior: 40 m/s, about the hardest shot, per 40 ms frame
    "max_interp_frames": 5,       # prior: bridge only gaps <= 0.2 s; longer gaps stay unknown
    "max_interp_speed": 30.0,     # m/s implied across a bridged gap
    "out_margin_m": 0.5,          # prior: ball centre this far past a line = out (tracking noise)
    # control
    "control_sigma_m": 1.5,       # prior: playing distance ~1-2 m in the possession literature
    "control_max_m": 4.0,         # beyond this from everyone, nobody controls the ball
    "pressure_radius_m": 3.0,     # prior: common pressing radius
    "pressure_gamma": 0.25,       # each pressing opponent shrinks control confidence
    "flight_speed": 12.0,         # prior: m/s; above this the ball is travelling, not controlled
    "llr_clip": 4.0,              # nats; no single frame is allowed to be decisive
    # dead ball
    "motion_window_frames": 25,   # 1 s smoothing of collective motion
    "static_speed": 0.3,          # m/s; a placed ball
    "static_min_frames": 25,      # 1 s static before it counts as a placed ball
    # missing ball
    "mpe_bins": 40,               # histogram bins on log motion for the mixture-proportion estimate
    "mpe_high_quantile": 0.75,    # live-play anchor: motion above this quantile of ball-visible frames
    "motion_lr_clip": 3.0,        # nats; motion alone is never decisive
    "mpe_min_dead_share": 0.05,   # below this estimated dead share of missing frames, motion is uninformative
    "exit_dist_m": 2.0,           # last seen this close to (or past) a line, moving outward = exit
    "min_gap_frames": 25,         # exits are only credited to gaps >= 1 s
}


def params_with_defaults(params):
    out = dict(DEFAULTS)
    out.update(params or {})
    return out


def _metres(trk):
    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width])
    return trk.players[:, :, :2] * scale, trk.ball[:, :2] * scale, (length, width)


def reliability_mask(players_m, p):
    """(n, slots) True where a player sample is trusted: on a tracklet long enough, and not on
    either side of a physically impossible jump (identity switch / re-acquisition)."""
    present = ~np.isnan(players_m[:, :, 0])
    reliable = present.copy()
    for slot in range(present.shape[1]):
        for start, length in runs(present[:, slot]):
            if length < p["min_segment_frames"]:
                reliable[start:start + length, slot] = False
    step = np.hypot(*np.moveaxis(np.diff(players_m, axis=0), 2, 0))
    jump = np.nan_to_num(step, nan=0.0) > p["max_player_step_m"]
    reliable[1:] &= ~jump
    reliable[:-1] &= ~jump
    return reliable


def duplicate_mask(players_m, usable, team_of_slot, p, chunk=20000):
    """(n, slots) True for a detection that duplicates a lower-index same-team detection."""
    dup = np.zeros(usable.shape, dtype=bool)
    r2 = p["dup_radius_m"] ** 2
    for team in (HOME, AWAY):
        idx = np.flatnonzero(team_of_slot == team)
        for a in range(0, len(players_m), chunk):
            xy = players_m[a:a + chunk][:, idx]                       # (c, k, 2)
            ok = usable[a:a + chunk][:, idx]
            d2 = np.sum((xy[:, :, None] - xy[:, None]) ** 2, axis=3)  # (c, k, k)
            close = (d2 < r2) & ok[:, :, None] & ok[:, None, :]
            close &= np.tril(np.ones((len(idx), len(idx)), dtype=bool), -1)[None]  # j < i only
            dup[a:a + chunk, idx] = close.any(axis=2)
    return dup


def ball_kinematics(ball_m, p, pitch):
    """Ball position (with short gaps bridged), speed from positions, and quality flags."""
    n = len(ball_m)
    xy = ball_m.copy()
    present = ~np.isnan(xy[:, 0])

    # A single-frame teleport is an observation error: drop the sample that jumps away and back.
    step = np.hypot(*(np.diff(xy, axis=0).T))
    bad = np.zeros(n, dtype=bool)
    big = np.nan_to_num(step, nan=0.0) > p["max_ball_step_m"]
    bad[1:-1] = big[:-1] & big[1:]
    xy[bad] = np.nan
    observed = present & ~bad

    interpolated = np.zeros(n, dtype=bool)
    for start, length in runs(~observed):
        before, after = start - 1, start + length
        if length > p["max_interp_frames"] or before < 0 or after >= n:
            continue
        gap_m = np.hypot(*(xy[after] - xy[before]))
        if gap_m / ((length + 1) / 25.0) > p["max_interp_speed"]:
            continue
        w = np.arange(1, length + 1)[:, None] / (length + 1.0)
        xy[start:after] = xy[before] + w * (xy[after] - xy[before])
        interpolated[start:after] = True
    have = observed | interpolated

    # Speed from consecutive valid positions only (the ATD's own speed channel is implausible --
    # max 7.3 m/s -- see docs/AUDIT_2026-10-26.md).
    both = have[1:] & have[:-1]
    d = np.diff(xy, axis=0) * 25.0
    fwd = np.full((n, 2), np.nan)
    fwd[:-1][both] = d[both]
    bwd = np.full((n, 2), np.nan)
    bwd[1:][both] = d[both]
    vel = np.where(np.isnan(fwd), bwd, np.where(np.isnan(bwd), fwd, (fwd + bwd) / 2))
    speed = np.hypot(vel[:, 0], vel[:, 1])

    length, width = pitch
    m = p["out_margin_m"]
    out = have & ((xy[:, 0] < -m) | (xy[:, 0] > length + m) | (xy[:, 1] < -m) | (xy[:, 1] > width + m))
    return {"xy": xy, "have": have, "observed": observed, "interpolated": interpolated,
            "teleport": bad, "speed": speed, "out": out}


def team_control(players_m, usable, team_of_slot, ball, p):
    """Per-frame team-control log-odds (positive = home) and its ingredients. 0 where unknown."""
    n = len(players_m)
    d = np.hypot(players_m[:, :, 0] - ball["xy"][:, None, 0],
                 players_m[:, :, 1] - ball["xy"][:, None, 1])
    d = np.where(usable & ball["have"][:, None], d, np.inf)
    home, away = team_of_slot == HOME, team_of_slot == AWAY
    d_home, d_away = d[:, home].min(axis=1), d[:, away].min(axis=1)
    near = np.minimum(d_home, d_away)
    far = np.maximum(d_home, d_away)

    # Gaussian control kernel: LLR = (d_far^2 - d_near^2) / (2 sigma^2), signed toward the nearer.
    s2 = 2.0 * p["control_sigma_m"] ** 2
    far_c = np.minimum(far, p["control_max_m"] * 2)        # cap so an absent team is not infinite
    llr = (far_c ** 2 - np.minimum(near, far_c) ** 2) / s2
    llr = np.clip(llr, 0, p["llr_clip"])
    nearest_home = d_home <= d_away

    # Opposition pressure on the nearest player: opponents (deduplicated) within the radius.
    near_slot = np.argmin(d, axis=1)
    carrier = players_m[np.arange(n), near_slot]
    dc = np.hypot(players_m[:, :, 0] - carrier[:, None, 0], players_m[:, :, 1] - carrier[:, None, 1])
    opp = np.where(nearest_home[:, None], away[None, :], home[None, :])
    pressing = (opp & usable & (dc < p["pressure_radius_m"])).sum(axis=1)
    llr = llr / (1.0 + p["pressure_gamma"] * pressing)

    # No evidence when nobody is close, when the ball is in flight, or when there is no ball.
    valid = ball["have"] & np.isfinite(near) & (near <= p["control_max_m"])
    in_flight = ball["speed"] > p["flight_speed"]
    valid &= ~np.nan_to_num(in_flight, nan=False)
    signed = np.where(nearest_home, llr, -llr)
    signed[~valid] = 0.0
    return {"llr": signed, "valid": valid, "d_home": d_home, "d_away": d_away,
            "pressing": pressing, "in_flight": in_flight & ball["have"]}


def collective_motion(players_m, usable, p):
    """Median speed of usable players, smoothed over motion_window_frames (m/s)."""
    step = np.hypot(*np.moveaxis(np.diff(players_m, axis=0), 2, 0)) * 25.0
    step = np.vstack([step[:1], step])
    step = np.where(usable, step, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN frames -> NaN, handled below
        med = np.nanmedian(step, axis=1)
    w = p["motion_window_frames"]
    filled = np.where(np.isnan(med), 0.0, med)
    weight = (~np.isnan(med)).astype(float)
    k = np.ones(2 * (w // 2) + 1)
    num = np.convolve(filled, k, mode="same")
    den = np.convolve(weight, k, mode="same")
    with np.errstate(all="ignore"):
        return np.where(den > 0, num / den, np.nan)


def two_component_split(values, iters=200):
    """Deterministic 1-D two-Gaussian EM on log values. Returns (posterior_low_fn, summary).

    Unsupervised: it sees the unlabelled motion of the match and nothing else. Initialised at the
    20th / 80th percentiles so the result does not depend on a random seed.
    """
    x = np.log(np.asarray(values)[np.isfinite(values)] + 1e-3)
    if x.size < 2:                                   # no reliable motion at all: diagnostic unavailable
        return (lambda v: np.full(len(np.asarray(v)), 0.5)), {"means_ms": None, "note": "no motion samples"}
    mu = np.percentile(x, [20, 80]).astype(float)
    sd = np.array([x.std(), x.std()]) / 2
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        ll -= ll.max(axis=1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0)
        w = nk / nk.sum()
        mu = (r * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-6
    lo = int(np.argmin(mu))

    def posterior_low(v):
        z = np.log(np.asarray(v) + 1e-3)
        ll = -0.5 * ((z[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        ll -= ll.max(axis=1, keepdims=True)
        r = np.exp(ll)
        out = r[:, lo] / r.sum(axis=1)
        return np.where(np.isfinite(z), out, 0.5)

    summary = {"means_ms": np.exp(mu).round(3).tolist(), "sds_log": sd.round(3).tolist(),
               "weights": w.round(3).tolist(), "low": lo}
    return posterior_low, summary


def static_ball(ball, p):
    """True on frames inside a run of >= static_min_frames where the observed ball barely moves."""
    slow = ball["have"] & (np.nan_to_num(ball["speed"], nan=np.inf) < p["static_speed"])
    out = np.zeros(len(slow), dtype=bool)
    for start, length in runs(slow):
        if length >= p["static_min_frames"]:
            out[start:start + length] = True
    return out


def missing_ball_dead_probability(motion, have, graded, p):
    """P(dead | collective motion, ball missing), estimated without labels.

    Mixture-proportion estimation (the "anchor" argument of Blanchard et al. 2010 / Elkan & Noto
    2008): take ball-visible frames as a sample of live-play motion f_live, and model motion on
    ball-missing frames as  f_miss = pi * f_live + (1 - pi) * f_dead.  Where stoppages are rare --
    high collective motion -- f_dead ~ 0, so pi ~ f_miss / f_live there. Then
        P(dead | m, missing) = 1 - pi * f_live(m) / f_miss(m).
    Visible-ball frames include some stoppages (placed balls), which biases f_live toward low
    motion and makes this estimate conservative (it under-, not over-, states dead ball).
    """
    m = np.log(np.asarray(motion) + 1e-3)
    ok = np.isfinite(m) & graded
    live, miss = ok & have, ok & ~have
    if not live.any() or not miss.any():
        # Degenerate input (no usable players, or the ball never / always seen): nothing to
        # estimate from, so no motion evidence either way. Real matches never reach this branch.
        return (np.where(have, np.nan, 0.5), np.zeros(len(m)),
                {"live_share_of_missing_pi": None, "mean_p_dead_missing_graded": None,
                 "motion_log_lr_by_bin": None, "degenerate": True})
    lo, hi = np.percentile(m[ok], [0.5, 99.5])
    edges = np.linspace(lo, hi, int(p["mpe_bins"]) + 1)
    f_live = np.histogram(np.clip(m[live], lo, hi), edges)[0].astype(float) + 1.0
    f_miss = np.histogram(np.clip(m[miss], lo, hi), edges)[0].astype(float) + 1.0
    f_live /= f_live.sum()
    f_miss /= f_miss.sum()
    ratio = f_miss / f_live
    centres = (edges[:-1] + edges[1:]) / 2
    anchor = centres >= np.quantile(m[live], p["mpe_high_quantile"])
    pi = float(np.clip(np.median(ratio[anchor]), 0.0, 1.0))
    p_dead_bin = np.clip(1.0 - pi / ratio, 0.0, 1.0)
    idx = np.clip(np.digitize(np.clip(m, lo, hi), edges) - 1, 0, len(centres) - 1)
    p_dead = np.where(np.isfinite(m), p_dead_bin[idx], 1.0 - pi)
    p_dead = np.where(have, np.nan, p_dead)
    # The same decomposition gives the motion likelihood ratio f_dead/f_live for ANY frame:
    #   f_dead = (f_miss - pi f_live) / (1 - pi)   =>   LR(m) = (r(m) - pi) / (1 - pi).
    # Degeneracy guard: when (1 - pi) is tiny the estimator has found no stoppage mass among
    # ball-missing frames, and dividing by it would turn histogram noise into strong evidence.
    if 1.0 - pi < p["mpe_min_dead_share"]:
        log_lr = np.zeros(len(ratio))
    else:
        lr_bin = np.clip((ratio - pi) / (1.0 - pi), 1e-3, None)
        log_lr = np.clip(np.log(lr_bin), -p["motion_lr_clip"], p["motion_lr_clip"])
    motion_log_lr = np.where(np.isfinite(m), log_lr[idx], 0.0)
    return p_dead, motion_log_lr, {
        "live_share_of_missing_pi": round(pi, 4),
        "mean_p_dead_missing_graded": round(float(np.nanmean(p_dead[graded & ~have])), 4),
        "motion_log_lr_by_bin": {"centres_ms": np.exp(centres).round(2).tolist(),
                                 "log_lr": log_lr.round(2).tolist()}}


def exit_gaps(ball, graded, pitch, p):
    """True over each ball gap (>= min_gap_frames) whose last observation is at/past a line and
    not moving back into the pitch -- the ball most likely left the field of play."""
    length, width = pitch
    xy, have = ball["xy"], ball["have"]
    out = np.zeros(len(have), dtype=bool)
    n_exit = 0
    for start, n in runs(~have):
        if n < p["min_gap_frames"] or start == 0 or not graded[start]:
            continue
        x, y = xy[start - 1]
        dist = np.array([x, length - x, y, width - y])
        side = int(np.argmin(dist))
        if dist[side] > p["exit_dist_m"]:
            continue
        # outward direction for that side, from the last two observations
        if start >= 2 and have[start - 2]:
            v = xy[start - 1] - xy[start - 2]
            outward = (-v[0], v[0], -v[1], v[1])[side]
            if outward < 0:
                continue
        out[start:start + n] = True
        n_exit += 1
    return out, n_exit


def geometric_evidence(trk, graded, params=None):
    """Everything Experiment A produces, as a dict of per-frame arrays plus a summary."""
    p = params_with_defaults(params)
    players_m, ball_m, pitch = _metres(trk)
    reliable = reliability_mask(players_m, p)
    dup = duplicate_mask(players_m, reliable, trk.team_of_slot, p)
    usable = reliable & ~dup
    ball = ball_kinematics(ball_m, p, pitch)
    control = team_control(players_m, usable, trk.team_of_slot, ball, p)
    motion = collective_motion(players_m, usable, p)
    # Diagnostic only, not used in emissions: an unanchored mixture can split live play into two
    # modes (it did on synthetic data), so the anchored likelihood ratio below is used instead.
    _, motion_fit = two_component_split(motion[graded])
    static = static_ball(ball, p)
    p_dead_missing, motion_log_lr, mpe = missing_ball_dead_probability(motion, ball["have"], graded, p)
    exited, n_exit = exit_gaps(ball, graded, pitch, p)

    present = ~np.isnan(players_m[:, :, 0])
    summary = {
        "params": p,
        "reliable_share_of_samples": round(float(reliable[graded].sum() / max(1, present[graded].sum())), 4),
        "duplicates_removed_graded": int(dup[graded].sum()),
        "ball_observed_graded_pct": round(100 * float(ball["observed"][graded].mean()), 3),
        "ball_interpolated_graded_frames": int(ball["interpolated"][graded].sum()),
        "ball_teleports_dropped": int(ball["teleport"].sum()),
        "ball_out_graded_frames": int(ball["out"][graded].sum()),
        "ball_in_flight_graded_frames": int(control["in_flight"][graded].sum()),
        "control_evidence_graded_pct": round(100 * float(control["valid"][graded].mean()), 3),
        "static_ball_graded_frames": int(static[graded].sum()),
        "motion_mixture_diagnostic_only": motion_fit,
        "missing_ball_mixture_proportion": mpe,
        "exit_gaps": n_exit,
        "exit_gap_graded_frames": int(exited[graded].sum()),
        "derived_ball_speed_graded": _q(ball["speed"][graded]),
        "nearest_distance_graded_m": _q(np.minimum(control["d_home"], control["d_away"])[graded & ball["have"]]),
    }
    return {"ball": ball, "control": control, "motion": motion, "motion_log_lr": motion_log_lr,
            "static": static, "usable": usable, "p_dead_missing": p_dead_missing,
            "exited": exited, "summary": summary}


def _q(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return {}
    return {k: round(float(np.quantile(v, q)), 3)
            for k, q in (("p5", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95), ("p99", .99))}
