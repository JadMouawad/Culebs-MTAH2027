"""Frozen invariant ownership and optional clean gap specialist, NumPy only."""
import os

import numpy as np

from culebs import ROOT, learned, composite49 as F, joint49 as C
from culebs.dl import npinfer_radial49 as N
from culebs.dl.features import COMMON


def labels_from_evidence(teacher_dead,z,ratio,student,frames,n,blocks,decoder,
                         specialist_z=None,specialist_ratio=None,observed=None):
    ownership=C.labels_from_logits(teacher_dead,z,ratio,student,frames,n,blocks,decoder)
    if specialist_z is None:
        return ownership,ownership
    if specialist_ratio is None or observed is None:
        raise ValueError("The gap specialist requires aligned ratio and observation arrays")
    fused_z,fused_ratio=F.fuse(z,ratio,student,np.asarray(specialist_z,np.float64),
                               np.asarray(specialist_ratio,np.float64),observed)
    robust=F.labels(teacher_dead,fused_z,fused_ratio,frames,n,blocks,decoder)
    return ownership,robust


def predict(trk,graded,params,root=ROOT):
    base=params["base"]
    par=learned.load_readout(base,root)
    student=params["students"]
    members=[N.load(os.path.join(root,student["weights_dir"],m["file"]),m["sha256"]) for m in student["members"]]
    rec,blocks=learned.record(trk,graded)
    teacher=F.teacher(rec,base,root)
    z,ratio=F.readout(rec,teacher,par,base["readout"]["logit_cap"])
    expert=N.ensemble_ratio(members,rec)
    kwargs={}
    if params.get("specialist"):
        specialist=F.teacher(rec,params["specialist"],root)
        z2,r2=F.readout(rec,specialist,par,base["readout"]["logit_cap"])
        kwargs=dict(specialist_z=z2,specialist_ratio=r2,
                    observed=rec["common"][:,COMMON.index("observed")]>.5)
    own,robust=labels_from_evidence(F.R.logit(teacher[:,2]).astype(np.float32),z,ratio,expert,
                                   rec["frames"],len(graded),blocks,base["decoder"],**kwargs)
    return robust if kwargs else own
