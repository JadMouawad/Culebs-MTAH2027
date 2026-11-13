"""Candidate possession models, each a pure function of (tracking, graded mask, events, params).

    nearest_player_smooth   the official baseline (reference)
    geometric_smooth        A: geometric evidence, per-frame decision (team held through neutral
                            frames), same majority vote as baseline
    geometric_events_smooth A+B: plus Smart Tagging evidence, same majority vote
    geometric_decoded       A+C: geometric evidence, duration-aware decoder
    full_decoded            A+B+C

The smoothed variants share the baseline's window so that A and B can be compared with the
baseline on equal temporal footing; C replaces the smoothing altogether.

Params are namespaced: {"features": {...}, "events": {...}, "decode": {...}}.
"""

import numpy as np

from culebs import decode as dec
from culebs.baselines import nearest_player_codes, smooth
from culebs.diagnostics import runs
from culebs.events import event_evidence
from culebs.features import geometric_evidence


def blocks_of(graded):
    return [(int(s), int(s + n - 1)) for s, n in runs(graded)]


def _ungraded_to_x(labels, graded):
    labels = np.asarray(labels).copy()
    labels[~graded] = "X"
    return labels


def predict(method, trk, graded, events=None, params=None):
    """(labels, info). labels: 'H'/'A'/'D' on graded frames, 'X' elsewhere (except the baseline,
    which is reproduced exactly as the organisers write it)."""
    params = params or {}
    if method == "nearest_player_smooth":
        return smooth(nearest_player_codes(trk), int(params.get("window", 50))), {}

    geo = geometric_evidence(trk, graded, params.get("features"))
    info = {"features": geo["summary"]}
    ev = None
    if method in ("geometric_events_smooth", "full_decoded"):
        if events is None:
            raise ValueError("{} needs Smart Tagging events".format(method))
        ev = event_evidence(events, geo, graded, params.get("events"))
        info["events"] = ev["summary"]

    dp = dec.params_with_defaults(params.get("decode"))
    info["decode_params"] = dp
    e, team, dead = dec.emissions(geo, ev, dp)
    blocks = blocks_of(graded)
    if method in SPRINT5_METHODS:
        labels = _sprint5(method, trk, geo, team, dead, blocks, dp, params, info)
    elif method == "geometric_s6":
        labels = _sprint6(trk, geo, team, dead, blocks, dp, params, info)
    elif method.endswith("_smooth"):
        labels = dec.majority(dec.frame_labels(team, dead, blocks), int(dp["smooth_window"]), blocks,
                              dp["majority_ties"])
    elif method.endswith("_decoded"):
        labels = dec.decode(e, blocks, dp, geo)
    else:
        raise KeyError(method)

    frame_argmax = dec.LABELS[np.argmax(e, axis=1)]
    info["evidence"] = {
        "frames_argmax_disagrees_with_output_pct":
            round(100 * float(np.mean(frame_argmax[graded] != labels[graded])), 3),
        "team_evidence_nonzero_graded_pct": round(100 * float(np.mean(team[graded] != 0)), 3),
        "mean_p_dead_graded": round(float(np.mean(1 / (1 + np.exp(-dead[graded])))), 4),
    }
    return _ungraded_to_x(labels, graded), info


SPRINT5_METHODS = ("geometric_adaptive", "geometric_cusum", "geometric_supervised")


def _sprint5(method, trk, geo, team, dead, blocks, dp, params, info):
    """Sprint 5 candidates (docs/SPRINT5_BOUNDARY_RESEARCH.md). None is a promoted contender."""
    from culebs import temporal
    raw = dec.frame_labels(team, dead, blocks)
    if method == "geometric_adaptive":
        return temporal.adaptive_majority(raw, team, blocks, params.get("temporal"))
    base = dec.majority(raw, int(dp["smooth_window"]), blocks, dp["majority_ties"])
    if method == "geometric_cusum":
        return temporal.cusum_team(base, team, blocks, params.get("temporal"))
    # geometric_supervised: a team model trained ONLY on the DFL development matches
    from culebs import supervised as sup
    model, train_info = sup.train_on_dfl(tuple(params.get("groups", sup.ALL_GROUPS)))
    info["supervised_training"] = train_info
    a, ctx = sup.antisym_features(trk, geo)
    llr = sup.predict_llr(model, sup.design(a, ctx, tuple(params.get("groups", sup.ALL_GROUPS))))
    return dec.majority(dec.frame_labels(llr, dead, blocks), int(dp["smooth_window"]), blocks, dp["majority_ties"])


def _sprint6(trk, geo, team, dead, blocks, dp, params, info):
    """Sprint 6 candidates on top of the decode params given (e110a's in the configs): optional
    gap-consensus stoppages (C1) and control-evidence transforms (D1 persistence, D2 velocity)."""
    from culebs import sprint6
    s6 = params.get("s6", {})
    if s6.get("gap_consensus"):
        add, counts = sprint6.gap_consensus_evidence(geo, s6.get("gap_params"))
        dead = np.clip(dead + add, -dp["dead_clip"], dp["dead_clip"])
        info["gap_consensus"] = counts
    if s6.get("llr") == "persistence":
        team = sprint6.persistence_llr(team, s6.get("llr_params"))
    elif s6.get("llr") == "velocity":
        team = sprint6.velocity_gated_llr(team, trk, geo, s6.get("llr_params"))
    return dec.majority(dec.frame_labels(team, dead, blocks), int(dp["smooth_window"]), blocks, dp["majority_ties"])


METHODS = ("nearest_player_smooth", "geometric_smooth", "geometric_events_smooth",
           "geometric_decoded", "full_decoded") + SPRINT5_METHODS + ("geometric_s6",)
