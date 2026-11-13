"""Sprint 13: modular possession decoders on fixed per-frame evidence (numpy only; experimental).

Every decoder takes the same input -- per-frame H/A/D probabilities pf (n, 3) at 25 Hz and the
inclusive period blocks [(first, last), ...] -- and returns labels "H"/"A"/"D", "X" outside the
blocks. Nothing crosses a block edge, so the half-time hole can never be smoothed over. The
grids and the selection rule are fixed in experiments/sprint13_protocol.json. `decode(name)`
parses a grid name such as "viterbi_4_16" or "rescue_0.8_25".

The e110a decision rule (D if p_dead > 0.5, else the team the evidence points to, last evidenced
team held where it is neutral) and its +/-w majority vote are reused from culebs.decode; the
short-run merge and the 3-state Viterbi from culebs.dl.npinfer12 (both unchanged).
"""

import numpy as np

from culebs import decode as dec
from culebs.dl.npinfer12 import merge_short, viterbi

LABELS = np.array(["H", "A", "D", "X"])
EPS = 1e-6


# ------------------------------------------------------------------------------------ helpers
def _logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def team_dead(pf):
    """Team logit (H vs A among the alive mass) and dead logit, as in culebs.dl.train."""
    return _logit(pf[:, 0] / np.maximum(pf[:, 0] + pf[:, 1], EPS)), np.clip(_logit(pf[:, 2]), -6, 6)


def e110a_rule(pf, blocks):
    team, dead = team_dead(pf)
    return dec.frame_labels(team, dead, blocks)


def _blank(n):
    return np.full(n, "X", "<U1")


def _per_block(pf, blocks, fn):
    """Apply fn(pf_block) -> codes (len,) to each block; X elsewhere."""
    out = _blank(len(pf))
    for a, b in blocks:
        if b >= a:
            out[a:b + 1] = LABELS[fn(pf[a:b + 1])]
    return out


def _box(x, h):
    """Centred box mean over +/-h rows (edges: mean of what exists), x (n, k)."""
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, 0)])
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - h), np.minimum(len(x), i + h + 1)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def runs_of(seg):
    """[(start, stop)] constant runs of a 1-D array (half-open, local indices)."""
    if not len(seg):
        return []
    edge = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    s = np.concatenate(([0], edge))
    e = np.concatenate((edge, [len(seg)]))
    return list(zip(s.tolist(), e.tolist()))


def onehot(codes, eps=1e-3):
    """Ground-truth codes (0..3) -> pseudo-probabilities; X frames become uniform."""
    p = np.full((len(codes), 3), eps)
    ok = codes < 3
    p[np.flatnonzero(ok), codes[ok]] = 1 - 2 * eps
    p[~ok] = 1.0 / 3
    return p


def from_e110a(team, dead):
    """e110a evidence (team, dead logits) -> pseudo-probabilities whose e110a_rule is e110a's."""
    pd = 1 / (1 + np.exp(-np.clip(dead, -30, 30)))
    ph = 1 / (1 + np.exp(-np.clip(team, -30, 30)))
    return np.stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd], 1)


# ------------------------------------------------------------------------- conventional filters
def argmax(pf, blocks):
    return _per_block(pf, blocks, lambda p: np.argmax(p, 1))


def vote(pf, blocks, w):
    return dec.majority(e110a_rule(pf, blocks), w, blocks, "hold")


def vote_merge(pf, blocks, w, m):
    return merge_short(vote(pf, blocks, w), m, blocks)


def box_mean(pf, blocks, h):
    return _per_block(pf, blocks, lambda p: np.argmax(_box(p, h), 1))


def ema(pf, blocks, tau_s, fps=25.0):
    """Forward-backward exponential smoothing (zero phase), then argmax."""
    alpha = 1 - np.exp(-1.0 / (tau_s * fps))

    def f(p):
        fw = np.empty_like(p)
        bw = np.empty_like(p)
        fw[0], bw[-1] = p[0], p[-1]
        for i in range(1, len(p)):
            fw[i] = fw[i - 1] + alpha * (p[i] - fw[i - 1])
        for i in range(len(p) - 2, -1, -1):
            bw[i] = bw[i + 1] + alpha * (p[i] - bw[i + 1])
        return np.argmax(fw + bw, 1)
    return _per_block(pf, blocks, f)


def _median(x, h):
    pad = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1), axis=1)


def median(pf, blocks, h):
    team, dead = team_dead(pf)
    t2, d2 = team.copy(), dead.copy()
    for a, b in blocks:
        t2[a:b + 1] = _median(team[a:b + 1], h)
        d2[a:b + 1] = _median(dead[a:b + 1], h)
    lab = LABELS[dec.frame_labels(t2, d2, blocks)]
    return np.where(_blocks_mask(len(pf), blocks), lab, "X")


def _blocks_mask(n, blocks):
    m = np.zeros(n, bool)
    for a, b in blocks:
        m[a:b + 1] = True
    return m


def gauss(pf, blocks, sigma_s, fps=25.0):
    s = sigma_s * fps
    r = int(np.ceil(3 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)

    def f(p):
        lp = np.log(np.clip(p, EPS, 1))
        num = np.stack([np.convolve(lp[:, j], k, mode="same") for j in range(3)], 1)
        den = np.convolve(np.ones(len(p)), k, mode="same")[:, None]
        return np.argmax(num / den, 1)
    return _per_block(pf, blocks, f)


def hysteresis3(pf, blocks, margin):
    """Switch state only when another state's 1 s-mean probability beats the current by margin."""
    def f(p):
        q = _box(p, 12)
        out = np.empty(len(q), np.int8)
        s = int(np.argmax(q[0]))
        for i in range(len(q)):
            j = int(np.argmax(q[i]))
            if j != s and q[i, j] - q[i, s] > margin:
                s = j
            out[i] = s
        return out
    return _per_block(pf, blocks, f)


def bilateral(pf, blocks, h):
    """Box mean of probabilities weighted by per-frame confidence (1 - normalised entropy)."""
    def f(p):
        ent = -(np.clip(p, EPS, 1) * np.log(np.clip(p, EPS, 1))).sum(1) / np.log(3)
        w = np.clip(1 - ent, 1e-3, 1)[:, None]
        return np.argmax(_box(p * w, h) / _box(w, h), 1)
    return _per_block(pf, blocks, f)


# ------------------------------------------------------------------------- adaptive smoothing
def _majority_var(codes, lo_w, blocks):
    """dec.majority with a per-frame half-width (same 'hold' tie rule); codes -1 = unknown."""
    out = _blank(len(codes))
    for a, b in blocks:
        seg = codes[a:b + 1]
        cum = [np.concatenate(([0], np.cumsum(seg == k))) for k in range(3)]
        i = np.arange(len(seg))
        w = lo_w[a:b + 1]
        lo, hi = np.maximum(0, i - w), np.minimum(len(seg), i + w + 1)
        counts = np.stack([c[hi] - c[lo] for c in cum], 1)
        winner = np.argmax(counts, 1)
        top = counts.max(1)
        tied = ((counts == top[:, None]).sum(1) > 1) | (top == 0)
        untied = np.flatnonzero(~tied)
        if not len(untied):
            continue
        idx = np.maximum.accumulate(np.where(~tied, i, -1))
        idx[idx < 0] = untied[0]
        out[a:b + 1] = LABELS[winner[idx]]
    return out


def adaptive_vote(pf, blocks, wmin, wmax):
    """Vote whose half-width shrinks where the 1 s-mean top probability is high."""
    conf = np.zeros(len(pf))
    for a, b in blocks:
        conf[a:b + 1] = _box(pf[a:b + 1].max(1, keepdims=True), 12)[:, 0]
    frac = np.clip((conf - 0.6) / 0.3, 0, 1)
    w = np.rint(wmax - frac * (wmax - wmin)).astype(int)
    return _majority_var(e110a_rule(pf, blocks), w, blocks)


def split_window(pf, blocks, wd, wt):
    """Dead/alive by a +/-wd vote; the team of alive frames by a +/-wt vote among alive frames."""
    codes = e110a_rule(pf, blocks)
    dead = np.where(codes == 2, 2, np.where(codes >= 0, 0, -1)).astype(np.int8)
    dmask = _majority_var(dead, np.full(len(pf), wd), blocks) == "D"
    team = np.where(codes == 2, -1, codes).astype(np.int8)
    tl = _majority_var(team, np.full(len(pf), wt), blocks)
    # a window without a single alive frame has no team: fall back to the plain team argmax
    fill = (tl == "X") & _blocks_mask(len(pf), blocks)
    tl[fill] = np.where(pf[fill, 0] >= pf[fill, 1], "H", "A")
    return np.where(dmask, "D", tl)


# -------------------------------------------------------------------- probabilistic decoders
def pen3(t, d):
    return np.array([[0, t, d], [t, 0, d], [d, d, 0]], float)


def viterbi_dec(pf, blocks, t, d, log_prior=None):
    lp = np.log(np.clip(pf, EPS, 1))
    if log_prior is not None:
        lp = lp - log_prior
    return _per_block(lp, blocks, lambda x: viterbi(x, pen3(t, d)))


def hsmm(pf, blocks, k, t):
    """Explicit minimum-duration semi-Markov Viterbi (expanded states).

    Each label is a chain of k sub-states that must be traversed in order; the last sub-state
    self-loops or exits to the first sub-state of another label at cost t. A run is therefore at
    least k frames long, except at block edges (a run may enter or leave through an edge). With
    k = 1 this is the plain Viterbi with a uniform switch penalty t (the Potts / change-point
    segmentation with cost t per change point).
    """
    def f(p):
        lp = np.log(np.clip(p, EPS, 1))
        n = len(lp)
        v = np.full((3, k), -np.inf)
        v[:, k - 1] = lp[0]                      # a run may already be under way at the edge
        back_in = np.zeros((n, 3), np.int8)      # sub-state 0 of c entered from label back_in
        back_last = np.zeros((n, 3), bool)       # last sub-state: True = stayed, False = advanced
        for i in range(1, n):
            last = v[:, k - 1]
            nv = np.empty_like(v)
            # entering label c from the best other label's last sub-state
            cand = last[None, :] - t + np.where(np.eye(3, dtype=bool), -np.inf, 0.0)
            src = np.argmax(cand, 1)
            nv[:, 0] = cand[np.arange(3), src]
            back_in[i] = src
            if k > 1:
                nv[:, 1:] = v[:, :-1]
                stay = v[:, k - 1] >= v[:, k - 2]
                nv[:, k - 1] = np.where(stay, v[:, k - 1], v[:, k - 2])
                back_last[i] = stay
            else:
                stay = last >= nv[:, 0]
                nv[:, 0] = np.where(stay, last, nv[:, 0])
                back_last[i] = stay
            v = nv + lp[i][:, None]
        c, s = np.unravel_index(int(np.argmax(v)), v.shape)
        path = np.empty(n, np.int8)
        for i in range(n - 1, -1, -1):
            path[i] = c
            if i == 0:
                break
            if s == k - 1 and back_last[i, c]:
                continue                         # stayed in the last sub-state
            if k > 1 and s > 0:
                s -= 1
                continue
            c, s = int(back_in[i, c]), k - 1     # entered c at frame i from another label
        return path
    return _per_block(pf, blocks, f)


# ---------------------------------------------------------------------- spell-level decoders
def refine(lab, pf, blocks, r):
    """Move each boundary within +/-r frames (never past half of an adjacent run) to the position
    that maximises the log-likelihood of the two labels either side."""
    out = lab.copy()
    lp = np.log(np.clip(pf, EPS, 1))
    idx = {"H": 0, "A": 1, "D": 2}
    for a, b in blocks:
        seg = out[a:b + 1]
        rr = runs_of(seg)
        if len(rr) < 2:
            continue
        cuts = [s for s, _ in rr[1:]]
        new = list(cuts)
        for q, c in enumerate(cuts):
            left_run, right_run = rr[q], rr[q + 1]
            la, lb = idx[seg[c - 1]], idx[seg[c]]
            lo = max(c - r, left_run[0] + (left_run[1] - left_run[0]) // 2 + 1)
            hi = min(c + r, right_run[0] + (right_run[1] - right_run[0]) // 2)
            if hi <= lo:
                continue
            # boundary at p in [lo, hi]: frames lo..p-1 take la, p..hi-1 take lb; relative score
            # sum_{lo <= j < p} (log p_la - log p_lb)
            gain = np.concatenate(([0.0], np.cumsum(lp[a + lo:a + hi, la] - lp[a + lo:a + hi, lb])))
            new[q] = lo + int(np.argmax(gain))
        labs = [seg[s] for s, _ in rr]
        bounds = [0] + new + [len(seg)]
        rebuilt = np.empty(len(seg), "<U1")
        for q, v in enumerate(labs):
            rebuilt[bounds[q]:bounds[q + 1]] = v
        out[a:b + 1] = rebuilt
    return out


def rescue(base, fine, pf, blocks, tau, min_len):
    """Restore runs of a fine decoding that the base decoding removed (label differs on >= 50 %
    of the run) when the run's mean probability of its own label is >= tau and it is >= min_len
    frames long."""
    out = base.copy()
    idx = {"H": 0, "A": 1, "D": 2}
    for a, b in blocks:
        seg = fine[a:b + 1]
        for s, e in runs_of(seg):
            v = seg[s]
            if v not in idx or e - s < min_len:
                continue
            if np.mean(base[a + s:a + e] != v) < 0.5:
                continue
            if pf[a + s:a + e, idx[v]].mean() >= tau:
                out[a + s:a + e] = v
    return out


def conf_merge(lab, pf, blocks, m, tau):
    """merge_short restricted to runs shorter than m whose mean probability of their own label is
    below tau (confident short runs are kept, whatever their length)."""
    out = np.asarray(lab).copy()
    idx = {"H": 0, "A": 1, "D": 2}
    for a, b in blocks:
        seg = out[a:b + 1]
        if not len(seg):
            continue
        rr = []
        for s, e in runs_of(seg):
            conf = pf[a + s:a + e, idx[seg[s]]].mean() if seg[s] in idx else 1.0
            rr.append([str(seg[s]), e - s, conf])
        while len(rr) > 1:
            cand = [i for i in range(len(rr)) if rr[i][1] < m and rr[i][2] < tau]
            if not cand:
                break
            k = min(cand, key=lambda i: rr[i][1])
            j = max([i for i in (k - 1, k + 1) if 0 <= i < len(rr)], key=lambda i: rr[i][1])
            rr[j][1] += rr[k][1]
            rr[j][2] = np.inf                      # a merged run is never a merge candidate again
            del rr[k]
            merged = [rr[0]]
            for v, n, c in rr[1:]:
                if v == merged[-1][0]:
                    merged[-1][1] += n
                    merged[-1][2] = np.inf
                else:
                    merged.append([v, n, c])
            rr = merged
        out[a:b + 1] = np.repeat([v for v, _, _ in rr], [n for _, n, _ in rr])
    return out


# ---------------------------------------------------------------------------------- dispatch
def decode(name, pf, blocks, log_prior=None, memo=None):
    """Grid name -> labels. See experiments/sprint13_protocol.json for every family. `memo` (a dict
    for one pf) caches the merge25 and viterbi_4_16 decodings the spell-level families build on."""
    if memo is not None and name in memo:
        return memo[name]
    if memo is not None and name.split("_")[0] in ("refine", "rescue", "cmerge"):
        for base in ("vote50_min_25", "viterbi_4_16"):
            if base not in memo:
                memo[base] = decode(base, pf, blocks)
        k = name.split("_")
        if k[0] == "refine":
            return refine(memo["vote50_min_25"], pf, blocks, int(k[1]))
        if k[0] == "rescue":
            return rescue(memo["vote50_min_25"], memo["viterbi_4_16"], pf, blocks, float(k[1]), int(k[2]))
        return conf_merge(memo["viterbi_4_16"], pf, blocks, int(k[1]), float(k[2]))
    k = name.split("_")
    f = k[0]
    if name == "argmax":
        return argmax(pf, blocks)
    if f == "vote" and len(k) == 2:
        return vote(pf, blocks, int(k[1]))
    if f.startswith("vote") and k[1] == "min":
        return vote_merge(pf, blocks, int(f[4:]), int(k[2]))
    if f == "box":
        return box_mean(pf, blocks, int(k[1]))
    if f == "ema":
        return ema(pf, blocks, float(k[1]))
    if f == "median":
        return median(pf, blocks, int(k[1]))
    if f == "gauss":
        return gauss(pf, blocks, float(k[1]))
    if f == "hyst3":
        return hysteresis3(pf, blocks, float(k[1]))
    if f == "bilat":
        return bilateral(pf, blocks, int(k[1]))
    if f == "split":
        return split_window(pf, blocks, int(k[1]), int(k[2]))
    if f == "avote":
        return adaptive_vote(pf, blocks, int(k[1]), int(k[2]))
    if f == "viterbi":
        return viterbi_dec(pf, blocks, float(k[1]), float(k[2]))
    if f == "vprior":
        return viterbi_dec(pf, blocks, float(k[1]), float(k[2]), log_prior)
    if f == "hsmm":
        return hsmm(pf, blocks, int(k[1]), float(k[2]))
    if f == "refine":
        return refine(vote_merge(pf, blocks, 50, 25), pf, blocks, int(k[1]))
    if f == "rescue":
        return rescue(vote_merge(pf, blocks, 50, 25), viterbi_dec(pf, blocks, 4, 16), pf, blocks,
                      float(k[1]), int(k[2]))
    if f == "cmerge":
        return conf_merge(viterbi_dec(pf, blocks, 4, 16), pf, blocks, int(k[1]), float(k[2]))
    raise ValueError(name)


GRID = (["argmax"]
        + ["vote_{}".format(w) for w in (12, 25, 37, 50, 75)]
        + ["vote50_min_{}".format(m) for m in (12, 25, 50)] + ["vote25_min_{}".format(m) for m in (12, 25)]
        + ["box_{}".format(h) for h in (6, 12, 25, 50)]
        + ["ema_{}".format(t) for t in (0.25, 0.5, 1, 2)]
        + ["median_{}".format(h) for h in (12, 25, 50)]
        + ["gauss_{}".format(s) for s in (0.25, 0.5, 1, 2)]
        + ["hyst3_{}".format(m) for m in (0.1, 0.2, 0.3, 0.4)]
        + ["bilat_{}".format(h) for h in (12, 25, 50)]
        + ["split_{}_{}".format(wd, wt) for wd in (50, 25) for wt in (12, 25)]
        + ["avote_{}_{}".format(a, b) for a, b in ((12, 50), (25, 50), (12, 75))]
        + ["viterbi_{}_{}".format(t, d) for t in (2, 4, 8, 16) for d in (4, 8, 16, 32)]
        + ["vprior_{}_{}".format(t, d) for t in (4, 8, 16) for d in (8, 16, 32)]
        + ["hsmm_{}_{}".format(k, t) for k in (6, 12, 25) for t in (2, 4, 8)]
        + ["refine_{}".format(r) for r in (12, 25, 50)]
        + ["rescue_{}_{}".format(t, n) for t in (0.6, 0.7, 0.8, 0.9) for n in (12, 25)]
        + ["cmerge_{}_{}".format(m, t) for m in (25, 50) for t in (0.7, 0.8, 0.9)])

FAMILIES = ("argmax", "vote", "vote50_min", "vote25_min", "box", "ema", "median", "gauss", "hyst3", "bilat",
            "split", "avote", "viterbi", "vprior", "hsmm", "refine", "rescue", "cmerge")


def family(name):
    if name == "argmax":
        return "argmax"
    if "_min_" in name:
        return name.split("_min_")[0] + "_min"
    return name.split("_")[0]
