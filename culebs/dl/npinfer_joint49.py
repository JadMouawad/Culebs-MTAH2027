"""NumPy-only team expert for S49C, with the fixed S47 player-set encoder.

Normalisation rounds to float32, as in training. Neural arithmetic is float64,
using the same convolution/GroupNorm/exact-GELU kernels as frozen S37 inference.
The network sees the entire sampled record, exactly as the research harness did.
No fitting, period-dependent adaptation, torch, labels or video is used here.
"""
import json

import numpy as np

from culebs.dl import npinfer as NP
from culebs.dl import npinfer12 as N12


def load(path, expected_sha256=None):
    if expected_sha256 is not None and NP.sha256(path) != expected_sha256:
        raise ValueError("joint expert weight hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    w["cfg"] = json.loads(str(w.pop("cfg_json")))
    c = w["cfg"]
    if c.get("gated") or c.get("norm", "group") != "group" or c.get("heads") != 2 or any(
            c.get(k) for k in ("attention", "multiscale", "extra", "masked_time")):
        raise ValueError("unsupported joint expert architecture")
    if w["out.weight"].shape[0] != 4:
        raise ValueError("joint expert must retain its two hierarchical heads")
    return w


def node_inputs(nodes, w):
    """The original S47 float32 normalisation, followed by float64 computation."""
    x = nodes.astype(np.float32).copy()
    valid = x[..., 6:7] > .5
    x[..., :6] = np.where(valid, (x[..., :6] - w["node_norm.nm"]) / w["node_norm.ns"], 0.0)
    return x.astype(np.float64)


def encode_nodes(nodes, w):
    x = node_inputs(nodes, w)
    valid = x[..., 6:7] > .5
    h = NP.gelu(x @ w["enc.phi.0.weight"].astype(np.float64).T + w["enc.phi.0.bias"].astype(np.float64))
    h = (h @ w["enc.phi.2.weight"].astype(np.float64).T + w["enc.phi.2.bias"].astype(np.float64)) * valid
    count = np.maximum(valid.sum(-2), 1)
    mean = h.sum(-2) / count
    maximum = np.where(valid, h, -1e4).max(-2)
    maximum = np.where(valid.any(-2), maximum, 0.0)
    return np.concatenate([mean, maximum], -1).reshape(len(x), -1)


def forward(w, x):
    f64 = lambda key: w[key].astype(np.float64)
    h = NP.conv1d(x.T, f64("inp.weight"), f64("inp.bias"), 1)
    for i, dilation in enumerate(w["cfg"]["dilations"]):
        stem = "blocks.{}.".format(i)
        z = NP.gelu(NP.group_norm(NP.conv1d(h, f64(stem + "c1.weight"), f64(stem + "c1.bias"), dilation),
                                  f64(stem + "n1.weight"), f64(stem + "n1.bias")))
        z = NP.gelu(NP.group_norm(NP.conv1d(z, f64(stem + "c2.weight"), f64(stem + "c2.bias"), dilation),
                                  f64(stem + "n2.weight"), f64(stem + "n2.bias")))
        h = h + z
    return h.T @ f64("out.weight").T + f64("out.bias")


def predict_ratio(w, rec):
    x, _ = N12.inputs(rec, w)
    if w["cfg"]["use_nodes"]:
        x = np.concatenate([x, encode_nodes(rec["nodes"], w)], -1)
    logits = forward(w, x)[:, 1] * w["cfg"]["logit_scale"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def ensemble_ratio(members, rec):
    # Preserve the research pipeline's probability rounding and seed averaging.
    return np.mean([predict_ratio(w, rec).astype(np.float32) for w in members], axis=0).astype(np.float64)
