"""Local kinematic invariants before external-only learned normalization."""
import numpy as np

from culebs.dl import npinfer_joint49 as N


def invariants(nodes):
    x = np.asarray(nodes, np.float64)
    if x.shape[-1] != 7:
        raise ValueError("Expected the seven original player-node fields")
    valid = x[..., 6] > .5
    if not np.isfinite(x[valid, :6]).all():
        raise ValueError("Non-finite valid player kinematics")
    x = np.where(valid[..., None], x, 0.)
    dx, dy, vx, vy = x[..., 0], x[..., 1], x[..., 3], x[..., 4]
    radius = np.hypot(dx, dy)
    dot = dx*vx + dy*vy
    denominator = np.maximum(radius, 1e-6)
    result = np.stack((radius, dot/denominator, np.abs(dx*vy-dy*vx)/denominator,
                       np.hypot(vx, vy), x[..., 5], dot, valid.astype(float)), axis=-1)
    return np.where(valid[..., None], result, 0.).astype(np.float32)


def load(path, expected_sha256=None):
    w = N.load(path, expected_sha256)
    if w["cfg"].get("node_transform") != "radial49":
        raise ValueError("Weight file does not declare the radial representation")
    return w


def transform_record(rec):
    return dict(rec, nodes=invariants(rec["nodes"]))


def ensemble_ratio(members, rec):
    return N.ensemble_ratio(members, transform_record(rec))
