"""Sprint 12: numpy-only inference for TCN seed ensembles and the short-run merge decoder
(experimental; NO torch). culebs.dl.npinfer (Sprint 11) is reused and left unchanged.

A weights file (.npz, exported by scripts/sprint12_release_candidate.py, kept outside Git) holds one
member: float32 tensors under canonical names (inp, blocks.<i>.c1/c2/n1/n2, out), the training-set
normalisation, the zeroed feature lists of no-ball-state members, and the configuration. Members
of the Sprint 10 (tcn.* / heads.*) and Sprint 11/12 (inp / blocks.<i>.n1.n / out) layouts are
mapped to the same names at export. The architecture is the A_hier_rf12 TCN of npinfer.

Inference is per period block (the Sprint 11 fix), team-swap symmetrised, then the members'
H/A/D probabilities are averaged. Decoding: e110a's rule (D if p_dead > 0.5, else the team;
+/-50-frame vote), then optionally every H/A/D run shorter than `min_run` frames is merged into
its longer neighbour, shortest first (Sprint 12 decoder `vote50_min_25`); or, instead of the vote,
a 3-state Viterbi path with penalties (H<->A, alive<->dead) (Sprint 12 decoder `viterbi_4_16`).
"""

import json

import numpy as np

from culebs import decode as dec
from culebs.dl import npinfer as NP

LABELS = NP.LABELS


def canonical(name):
    """Sprint 10 / Sprint 11-12 state-dict key -> canonical key."""
    for a, b in (("tcn.", ""), ("heads.out.", "out."), (".n1.n.", ".n1."), (".n2.n.", ".n2.")):
        name = name.replace(a, b)
    return name


def load(path, expect_sha=None):
    if expect_sha is not None and NP.sha256(path) != expect_sha:
        raise ValueError("weights sha256 mismatch: " + path)
    with np.load(path, allow_pickle=False) as z:
        W = {k: z[k] for k in z.files}
    W["cfg"] = json.loads(str(W.pop("cfg_json")))
    cfg = W["cfg"]
    if cfg.get("kind", "hier") != "hier" or cfg.get("graph") or cfg.get("norm", "group") != "group" \
            or cfg.get("gated") or cfg.get("heads", 1) != 1 or cfg.get("multiscale") or cfg.get("attention") \
            or cfg.get("extra") or cfg.get("masked_time"):
        raise ValueError("numpy port implements the plain hier TCN (GroupNorm, one head) only")
    return W


def inputs(rec, W):
    """(x, x_swapped) float64 (T, C): normalised common + both team blocks, zeroed groups applied."""
    c = ((rec["common"] - W["norm.cm"]) / W["norm.cs"]).astype(np.float32).astype(np.float64)
    t = ((rec["team"] - W["norm.tm"]) / W["norm.ts"]).astype(np.float32).astype(np.float64)
    c[:, W["zero_common"].astype(int)] = 0.0
    t[:, :, W["zero_team"].astype(int)] = 0.0
    return (np.concatenate([c, t.reshape(len(t), -1)], 1),
            np.concatenate([c, t[:, ::-1].reshape(len(t), -1)], 1))


def forward(W, x):
    """x (T, C) normalised -> logits (T, 2), float64."""
    f64 = lambda k: W[k].astype(np.float64)  # noqa: E731
    h = NP.conv1d(x.T, f64("inp.weight"), f64("inp.bias"), 1)
    for i, d in enumerate(W["cfg"]["dilations"]):
        p = "blocks.{}.".format(i)
        z = NP.gelu(NP.group_norm(NP.conv1d(h, f64(p + "c1.weight"), f64(p + "c1.bias"), d), f64(p + "n1.weight"), f64(p + "n1.bias")))
        z = NP.gelu(NP.group_norm(NP.conv1d(z, f64(p + "c2.weight"), f64(p + "c2.bias"), d), f64(p + "n2.weight"), f64(p + "n2.bias")))
        h = h + z
    return h.T @ f64("out.weight").T + f64("out.bias")


def predict_member(W, rec):
    """(m, 3) H/A/D on the 5 Hz frames; each period block alone, team-swap symmetrised."""
    x, xs = inputs(rec, W)
    p = np.zeros((len(x), 3))
    p[:, 2] = 1.0
    for a, b in NP.sample_blocks(rec["frames"], [tuple(map(int, v)) for v in rec["blocks"]]):
        p1 = NP._probs(forward(W, x[a:b]))
        p2 = NP._probs(forward(W, xs[a:b]))
        p[a:b] = (p1 + p2[:, [1, 0, 2]]) / 2
    return p


def predict(members, rec):
    """Mean of the members' probabilities (a single member is a list of one)."""
    return np.mean([predict_member(W, rec) for W in members], axis=0)


def merge_short(lab, m, blocks):
    """Merge every H/A/D run shorter than m frames into its longer neighbour, shortest first
    (ties: the earliest run; neighbour ties: the left one). Frames outside the blocks are kept."""
    out = np.asarray(lab).copy()
    for a, b in blocks:
        seg = out[a:b + 1]
        if not len(seg):
            continue
        edge = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], edge))
        rr = [[str(seg[s]), int(e - s)] for s, e in zip(starts, np.concatenate((edge, [len(seg)])))]
        while len(rr) > 1:
            k = min(range(len(rr)), key=lambda i: rr[i][1])
            if rr[k][1] >= m:
                break
            j = max([i for i in (k - 1, k + 1) if 0 <= i < len(rr)], key=lambda i: rr[i][1])
            rr[j][1] += rr[k][1]
            del rr[k]
            merged = [rr[0]]
            for v, n in rr[1:]:
                if v == merged[-1][0]:
                    merged[-1][1] += n
                else:
                    merged.append([v, n])
            rr = merged
        out[a:b + 1] = np.repeat([v for v, _ in rr], [n for _, n in rr])
    return out


def viterbi(logp, pen):
    """logp (n, 3) H/A/D log-probabilities, pen (3, 3) switch penalties -> best label path (codes)."""
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    v = logp[0].copy()
    for i in range(1, n):
        cand = v[:, None] - pen
        back[i] = np.argmax(cand, 0)
        v = cand[back[i], np.arange(3)] + logp[i]
    path = np.empty(n, np.int8)
    path[-1] = int(np.argmax(v))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode_viterbi(pf, blocks, team_pen, dead_pen):
    """Per-frame probabilities (n, 3) -> labels; X outside the blocks."""
    pen = np.array([[0, team_pen, dead_pen], [team_pen, 0, dead_pen], [dead_pen, dead_pen, 0]], float)
    lab = np.full(len(pf), "X", "<U1")
    lp = np.log(np.clip(pf, 1e-6, 1))
    for a, b in blocks:
        lab[a:b + 1] = LABELS[viterbi(lp[a:b + 1], pen)]
    return lab


def decode(rec, p, window=50, min_run=0, viterbi_pen=None):
    n = len(rec["e110a_team"])
    pf = NP.upsample(p, rec["frames"], n)
    blocks = [tuple(map(int, x)) for x in rec["blocks"]]
    if viterbi_pen is not None:
        return decode_viterbi(pf, blocks, *viterbi_pen)
    team = NP._logit(pf[:, 0] / np.maximum(pf[:, 0] + pf[:, 1], 1e-6))
    dead = np.clip(NP._logit(pf[:, 2]), -6, 6)
    lab = dec.majority(dec.frame_labels(team, dead, blocks), window, blocks, "hold")
    return merge_short(lab, min_run, blocks) if min_run else lab
