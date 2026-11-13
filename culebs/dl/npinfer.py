"""Sprint 11 Part 9: numpy-only forward pass of the frozen Sprint 10 TCN (experimental; NO torch).

The weights file (.npz, exported by scripts/sprint11_numpy_inference.py export, kept outside Git)
holds the float32 tensors of the checkpoint exactly as trained, the training-set normalisation
statistics and the configuration; every computation here is float64. Architecture (A_hier_rf12):

    x (T, 37) normalised  -> Conv1d k=1 (37 -> 64)
    5 x residual block, dilation d in (1, 2, 4, 8, 16):
        h = GELU(GroupNorm8(Conv1d k=3, dilation d, zero padding d (x)))      dropout = identity
        h = GELU(GroupNorm8(Conv1d k=3, dilation d, zero padding d (h)))
        x = x + h
    Linear (64 -> 2): dead logit, home-given-alive logit
    P(H) = (1 - pd) ph, P(A) = (1 - pd)(1 - ph), P(D) = pd; symmetrised with the team-swapped input

GroupNorm statistics are taken over the whole sequence given (8 groups x T), as in PyTorch, so
the `context` argument (full / per_half / windowed, see culebs.dl.s11) changes the output; GELU is
the exact erf form (math.erf, the platform libm). Decoding is e110a's own rule, re-implemented
from culebs.decode with no torch dependency.
"""

import hashlib
import json
import math

import numpy as np

from culebs import decode as dec

LABELS = np.array(["H", "A", "D", "X"])
_ERF = np.frompyfunc(math.erf, 1, 1)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load(path, expect_sha=None):
    if expect_sha is not None and sha256(path) != expect_sha:
        raise ValueError("weights sha256 mismatch")
    with np.load(path, allow_pickle=False) as z:
        W = {k: z[k] for k in z.files}
    W["cfg"] = json.loads(str(W.pop("cfg_json")))
    if W["cfg"]["kind"] != "hier" or W["cfg"]["graph"]:
        raise ValueError("numpy port implements the hier TCN without the graph encoder only")
    return W


def gelu(x):
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)).astype(np.float64))


def conv1d(x, w, b, dilation):
    """x (C_in, T), w (C_out, C_in, k), zero padding = dilation * (k - 1) / 2 -> (C_out, T)."""
    k = w.shape[2]
    pad = dilation * (k - 1) // 2
    T = x.shape[1]
    xp = np.zeros((x.shape[0], T + 2 * pad))
    xp[:, pad:pad + T] = x
    out = np.empty((w.shape[0], T))
    out[:] = b[:, None]
    for j in range(k):
        out += w[:, :, j] @ xp[:, j * dilation:j * dilation + T]
    return out


def group_norm(x, gamma, beta, groups=8, eps=1e-5):
    C, T = x.shape
    g = x.reshape(groups, (C // groups) * T)
    mu = g.mean(1, keepdims=True)
    var = g.var(1, keepdims=True)
    y = ((g - mu) / np.sqrt(var + eps)).reshape(C, T)
    return y * gamma[:, None] + beta[:, None]


def forward(W, common, team):
    """common (T, Cc), team (T, 2, Ct), both normalised -> logits (T, 2), float64."""
    f64 = lambda k: W[k].astype(np.float64)  # noqa: E731
    x = np.concatenate([common, team.reshape(len(team), -1)], 1).astype(np.float64).T
    h = conv1d(x, f64("tcn.inp.weight"), f64("tcn.inp.bias"), 1)
    for i, d in enumerate(W["cfg"]["dilations"]):
        p = "tcn.blocks.{}.".format(i)
        z = gelu(group_norm(conv1d(h, f64(p + "c1.weight"), f64(p + "c1.bias"), d), f64(p + "n1.weight"), f64(p + "n1.bias")))
        z = gelu(group_norm(conv1d(z, f64(p + "c2.weight"), f64(p + "c2.bias"), d), f64(p + "n2.weight"), f64(p + "n2.bias")))
        h = h + z
    return h.T @ f64("heads.out.weight").T + f64("heads.out.bias")


def _probs(logits):
    pd = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    ph = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    return np.stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd], 1)


def normed(rec, W):
    c = (rec["common"] - W["norm.cm"]) / W["norm.cs"]
    t = (rec["team"] - W["norm.tm"]) / W["norm.ts"]
    return c.astype(np.float32).astype(np.float64), t.astype(np.float32).astype(np.float64)


def _sym(W, c, t):
    p1 = _probs(forward(W, c, t))
    p2 = _probs(forward(W, c, t[:, ::-1]))
    return (p1 + p2[:, [1, 0, 2]]) / 2


def sample_blocks(frames, blocks):
    out = []
    for a, b in blocks:
        idx = np.flatnonzero((frames >= a) & (frames <= b))
        if len(idx):
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def predict(W, rec, context="full", window=256):
    """(m, 3) H/A/D probabilities on the 5 Hz frames; the same contexts as culebs.dl.s11.predict."""
    c, t = normed(rec, W)
    p = _sym(W, c, t)
    if context == "full":
        return p
    for a, b in sample_blocks(rec["frames"], [tuple(map(int, x)) for x in rec["blocks"]]):
        n, h = b - a, window // 2
        if context == "per_half" or n <= window:
            p[a:b] = _sym(W, c[a:b], t[a:b])
            continue
        starts = list(range(0, n - window, h)) + [n - window]
        for s in starts:
            q = _sym(W, c[a + s:a + s + window], t[a + s:a + s + window])
            lo = 0 if s == 0 else h // 2
            hi = window if s == n - window else h // 2 + h
            p[a + s + lo:a + s + hi] = q[lo:hi]
    return p


def upsample(values, frames, n):
    stride = int(frames[1] - frames[0]) if len(frames) > 1 else 1
    idx = np.clip(np.rint(np.arange(n) / stride).astype(int), 0, len(frames) - 1)
    return values[idx]


def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def decode(rec, p, variant="nn_vote", window=50):
    """Per-frame H/A/D/X labels with e110a's decision rule (D if p_dead > .5; +/-window vote)."""
    n = len(rec["e110a_team"])
    pf = upsample(p, rec["frames"], n)
    blocks = [tuple(map(int, x)) for x in rec["blocks"]]
    team = _logit(pf[:, 0] / np.maximum(pf[:, 0] + pf[:, 1], 1e-6))
    dead = np.clip(_logit(pf[:, 2]), -6, 6)
    if variant == "nn_vote":
        lab3 = dec.frame_labels(team, dead, blocks)
    elif variant == "nn_team+e110a_dead":
        lab3 = dec.frame_labels(team, rec["e110a_dead"], blocks)
    else:
        raise ValueError(variant)
    return dec.majority(lab3, window, blocks, "hold")
