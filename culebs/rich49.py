"""Enriched ownership and the frozen optional gap specialist, NumPy only."""
import os

import numpy as np

from culebs import ROOT, learned, composite49 as F
from culebs.dl import npinfer_rich49 as N
from culebs.dl.features import COMMON
from culebs.radial49 import labels_from_evidence


def load_students(params, root=ROOT):
    members = [N.load(os.path.join(root, params["weights_dir"], m["file"]), m["sha256"])
               for m in params["members"]]
    if len(members) != 2:
        raise ValueError("Expected exactly the two frozen training seeds")
    for weights in members:
        shared = bool(weights["cfg"].get("shared_team", False))
        if shared != bool(params.get("shared_team", False)):
            raise ValueError("Shared ownership configuration/weight mismatch")
        if shared:
            for key in ("out.weight", "out.bias"):
                if not np.array_equal(weights[key][1], weights[key][3]):
                    raise ValueError("Shared ownership output rows diverged")
    return members


def predict(trk, graded, params, root=ROOT):
    base = params["base"]
    par = learned.load_readout(base, root)
    members = load_students(params["students"], root)
    rec, blocks = learned.record(trk, graded)
    teacher = F.teacher(rec, base, root)
    z, ratio = F.readout(rec, teacher, par, base["readout"]["logit_cap"])
    expert = N.ensemble_ratio(members, rec)
    kwargs = {}
    if params.get("specialist"):
        specialist = F.teacher(rec, params["specialist"], root)
        z2, r2 = F.readout(rec, specialist, par, base["readout"]["logit_cap"])
        kwargs = dict(specialist_z=z2, specialist_ratio=r2,
                      observed=rec["common"][:, COMMON.index("observed")] > .5)
    own, robust = labels_from_evidence(F.R.logit(teacher[:, 2]).astype(np.float32), z, ratio, expert,
                                      rec["frames"], len(graded), blocks, base["decoder"], **kwargs)
    return robust if kwargs else own
