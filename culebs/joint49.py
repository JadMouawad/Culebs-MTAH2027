"""Frozen S37 dead evidence plus the fixed S49C ownership expert; NumPy only.

The original teacher is used for the dead readout BEFORE ownership fusion. This
matters because the readout includes the teacher's maximum team probability.
"""
import os

import numpy as np

from culebs import ROOT
from culebs import dead_readout33 as R
from culebs import learned
from culebs.dl import decoders13 as D
from culebs.dl import npinfer12 as N12
from culebs.dl import npinfer_joint49 as N
from culebs.dl.features import upsample


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z,np.float64),-30,30)))


def labels_from_logits(teacher_dead_logit, dead_logit, base_ratio, student_ratio, frames, n, blocks, decoder):
    """The exact evidence-to-label path used by the prospective external test."""
    td = _sigmoid(teacher_dead_logit)
    ratio = _sigmoid((R.logit(np.asarray(base_ratio,np.float64))+R.logit(student_ratio))/2)
    teacher = np.column_stack(((1-td)*ratio,(1-td)*(1-ratio),td))
    dead = _sigmoid(dead_logit)
    posterior = R.combine(upsample(teacher,frames,n),upsample(dead[:,None],frames,n)[:,0])
    return D.decode(decoder,posterior,[tuple(map(int,b)) for b in blocks])


def predict(trk, graded, params, root=ROOT):
    """One algorithmic label per frame; all weights verified before inference."""
    base = params["base"]
    weights = dict(weights_dir=base["weights_dir"])
    team_members = learned.load_members(dict(weights,members=base["team_members"]),root)
    dead_members = learned.load_members(dict(weights,members=base["dead_members"]),root)
    par = learned.load_readout(base,root)
    students = params["students"]
    members = [N.load(os.path.join(root,students["weights_dir"],m["file"]),m["sha256"]) for m in students["members"]]
    rec,blocks = learned.record(trk,graded)
    team = np.mean([N12.predict_member(w,rec) for w in team_members],axis=0)
    dead = np.mean([N12.predict_member(w,rec) for w in dead_members],axis=0)[:,2]
    alive = team[:,:2].sum(1)
    ratio = np.divide(team[:,0],alive,out=np.full(len(team),.5),where=alive>0)
    p5 = np.column_stack(((1-dead)*ratio,(1-dead)*(1-ratio),dead)).astype(np.float32)
    # Retain S37's original teacher confidence in the readout.
    d = dict(common=rec["common"],team=rec["team"],nodes=rec["nodes"],teacher=p5)
    dead2,_ = R.predict_dead(par,R.features(d),p5[:,2],base["readout"]["logit_cap"])
    alive = p5[:,:2].sum(1)
    base_ratio = np.divide(p5[:,0],alive,out=np.full(len(p5),.5),where=alive>0).astype(np.float32)
    return labels_from_logits(R.logit(p5[:,2]).astype(np.float32),R.logit(dead2).astype(np.float32),
                              base_ratio,N.ensemble_ratio(members,rec),rec["frames"],len(graded),blocks,base["decoder"])
