"""Learned release method `learned_ens6` (ens6_rescue80): six DFL-trained TCN members, numpy only.

Called by culebs.release.predict for configs whose method starts with "learned_". The config's params give the weights
folder (repository-relative), each member file with its sha256, and the decoder. A member whose bytes differ from the
frozen hash is refused before anything is predicted. No ground truth, no torch.

The record built here is the one scripts/sprint11_numpy_inference.competition_record builds (Sprint 13-15 path, output
af29a2bd...), from the Tracking that the release pipeline already parsed.
"""

import hashlib
import os

import numpy as np

from culebs import ROOT


class WeightsError(Exception):
    """Missing or altered model weights; the message says which file and what to do."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_members(params, root=ROOT):
    from culebs.dl import npinfer12 as N12
    folder = os.path.join(root, params["weights_dir"])
    out = []
    for m in params["members"]:
        path = os.path.join(folder, m["file"])
        if not os.path.isfile(path):
            raise WeightsError("model weights not found: {} (expected in {}; see {}/ATTRIBUTION.md for their origin)".format(
                m["file"], params["weights_dir"], params["weights_dir"]))
        got = _sha256(path)
        if got != m["sha256"]:
            raise WeightsError("model weights {} changed: sha256 {} != frozen {}".format(m["file"], got[:12], m["sha256"][:12]))
        out.append(N12.load(path, m["sha256"]))
    return out


def record(trk, graded):
    from culebs.dl import features as F
    from culebs.features import geometric_evidence
    idx = np.flatnonzero(np.diff(np.concatenate(([0], graded.astype(np.int8), [0]))))
    blocks = [(int(a), int(b) - 1) for a, b in zip(idx[::2], idx[1::2])]
    geo = geometric_evidence(trk, graded, None)
    rec = F.build(trk, graded, geo=geo)
    rec["blocks"] = np.array(blocks)
    return rec, blocks


def predict_ens6(trk, graded, params, root=ROOT):
    """H/A/D/X labels (one per frame) of the six-member ensemble decoded with params['decoder']."""
    from culebs.dl import decoders13 as D
    from culebs.dl import npinfer as NP
    from culebs.dl import npinfer12 as N12
    members = load_members(params, root)
    rec, blocks = record(trk, graded)
    p = np.mean([N12.predict_member(W, rec) for W in members], axis=0)
    return D.decode(params["decoder"], NP.upsample(p, rec["frames"], len(graded)), blocks)


def load_readout(params, root=ROOT):
    """The ens6_dead33 readout parameters, refused if the file is missing or its bytes differ from the frozen hash."""
    r = params["readout"]
    path = os.path.join(root, r["weights_dir"], r["file"])
    if not os.path.isfile(path):
        raise WeightsError("readout weights not found: {} (expected in {}; see {}/ATTRIBUTION.md)".format(
            r["file"], r["weights_dir"], r["weights_dir"]))
    got = _sha256(path)
    if got != r["sha256"]:
        raise WeightsError("readout weights {} changed: sha256 {} != frozen {}".format(r["file"], got[:12], r["sha256"][:12]))
    with np.load(path) as z:
        return {k: z[k] for k in ("weight", "bias", "mean", "sd")}


def predict_ens6_dead33(trk, graded, params, root=ROOT):
    """ens6_dead33: the ens6 probabilities with the Sprint 33 dead readout applied, decoded with params['decoder'].

    Same computation as the research rehearsal (culebs-mtah2027 scripts/sprint33_candidate.py, output aead8668...):
    float32 5 Hz ens6 probabilities, readout dead probability at 5 Hz, nearest-sample upsampling, team ratio kept."""
    from culebs import dead_readout33 as R
    from culebs.dl import decoders13 as D
    from culebs.dl import npinfer as NP
    from culebs.dl import npinfer12 as N12
    members = load_members(params, root)
    par = load_readout(params, root)
    rec, blocks = record(trk, graded)
    p5 = np.mean([N12.predict_member(W, rec) for W in members], axis=0).astype(np.float32)
    full = NP.upsample(p5, rec["frames"], len(graded))
    d = {"common": rec["common"], "team": rec["team"], "nodes": rec["nodes"], "teacher": p5}
    dead, _ = R.predict_dead(par, R.features(d), p5[:, 2], params["readout"]["logit_cap"])
    deadu = NP.upsample(dead[:, None].astype(np.float32), rec["frames"], len(graded))[:, 0]
    return D.decode(params["decoder"], R.combine(full, deadu), blocks)


def two_head_labels(rec, blocks, n, params, root=ROOT):
    """x_t1_50_dead37 labels from a record, as evaluated (culebs-mtah2027 scripts/sprint37_repro.py, research/sprint-37-robust-clean).

    Team ratio from the six DFL-head members; dead probability from the six members whose dead logit averages both label heads;
    Sprint 33 dead readout on that factorised teacher; 5 Hz float32, nearest-sample upsampling, team ratio kept."""
    from culebs import dead_readout33 as R
    from culebs.dl import decoders13 as D
    from culebs.dl import npinfer as NP
    from culebs.dl import npinfer12 as N12
    weights = {"weights_dir": params["weights_dir"]}
    team_members = load_members(dict(weights, members=params["team_members"]), root)
    dead_members = load_members(dict(weights, members=params["dead_members"]), root)
    par = load_readout(params, root)
    team = np.mean([N12.predict_member(W, rec) for W in team_members], axis=0)
    dead = np.mean([N12.predict_member(W, rec) for W in dead_members], axis=0)[:, 2]
    alive = team[:, :2].sum(1)
    ratio = np.divide(team[:, 0], alive, out=np.full(len(team), 0.5), where=alive > 0)
    p5 = np.column_stack(((1 - dead) * ratio, (1 - dead) * (1 - ratio), dead)).astype(np.float32)
    d = {"common": rec["common"], "team": rec["team"], "nodes": rec["nodes"], "teacher": p5}
    dead2, _ = R.predict_dead(par, R.features(d), p5[:, 2], params["readout"]["logit_cap"])
    full = NP.upsample(p5, rec["frames"], n)
    deadu = NP.upsample(dead2[:, None], rec["frames"], n)[:, 0]
    return D.decode(params["decoder"], R.combine(full, deadu), blocks)


def predict_x_t1_50(trk, graded, params, root=ROOT):
    rec, blocks = record(trk, graded)
    return two_head_labels(rec, blocks, len(graded), params, root)
