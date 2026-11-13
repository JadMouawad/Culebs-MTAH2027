"""Fixed S49F composition: S37, the joint ownership expert and clean T2 gaps."""
import os

import numpy as np

from culebs import ROOT, learned
from culebs import dead_readout33 as R
from culebs.dl import decoders13 as D
from culebs.dl import npinfer12 as N12
from culebs.dl import npinfer_joint49 as N
from culebs.dl.features import COMMON, upsample


def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(np.asarray(z,np.float64),-30,30)))


def fuse(base_z,base_ratio,student_ratio,t2_z,t2_ratio,observed):
    ratio_c=sigmoid((R.logit(np.asarray(base_ratio,np.float64))+R.logit(student_ratio))/2)
    index=np.arange(len(observed))
    last=np.maximum.accumulate(np.where(observed,index,-1))
    elapsed=np.where(observed,-1,index-last-1)
    mask=elapsed>=5  # one second at the fixed 5 Hz feature rate
    z=np.where(mask,(np.asarray(base_z,np.float64)+t2_z)/2,base_z)
    ratio=np.where(mask,sigmoid((R.logit(ratio_c)+R.logit(t2_ratio))/2),ratio_c)
    return z,ratio


def labels(teacher_dead_logit,z,ratio,frames,n,blocks,decoder):
    td=sigmoid(teacher_dead_logit)
    teacher=np.column_stack(((1-td)*ratio,(1-td)*(1-ratio),td))
    posterior=R.combine(upsample(teacher,frames,n),upsample(sigmoid(z)[:,None],frames,n)[:,0])
    return D.decode(decoder,posterior,[tuple(map(int,b)) for b in blocks])


def teacher(rec,params,root):
    weights=dict(weights_dir=params["weights_dir"])
    team=learned.load_members(dict(weights,members=params["team_members"]),root)
    dead=learned.load_members(dict(weights,members=params["dead_members"]),root)
    t=np.mean([N12.predict_member(w,rec) for w in team],axis=0)
    d=np.mean([N12.predict_member(w,rec) for w in dead],axis=0)[:,2]
    alive=t[:,:2].sum(1)
    ratio=np.divide(t[:,0],alive,out=np.full(len(t),.5),where=alive>0)
    return np.column_stack(((1-d)*ratio,(1-d)*(1-ratio),d)).astype(np.float32)


def readout(rec,p5,par,cap):
    d=dict(common=rec["common"],team=rec["team"],nodes=rec["nodes"],teacher=p5)
    dead,_=R.predict_dead(par,R.features(d),p5[:,2],cap)
    alive=p5[:,:2].sum(1)
    ratio=np.divide(p5[:,0],alive,out=np.full(len(p5),.5),where=alive>0).astype(np.float32)
    return R.logit(dead).astype(np.float32),ratio


def predict(trk,graded,params,root=ROOT):
    base=params["base"]
    par=learned.load_readout(base,root)  # supplied by the frozen S37 release base
    students=params["students"]
    members=[N.load(os.path.join(root,students["weights_dir"],m["file"]),m["sha256"]) for m in students["members"]]
    rec,blocks=learned.record(trk,graded)
    p5=teacher(rec,base,root)
    z,r=readout(rec,p5,par,base["readout"]["logit_cap"])
    t2=teacher(rec,params["specialist"],root)
    z2,r2=readout(rec,t2,par,base["readout"]["logit_cap"])
    z,r=fuse(z,r,N.ensemble_ratio(members,rec),z2.astype(np.float64),r2.astype(np.float64),
             rec["common"][:,COMMON.index("observed")]>.5)
    return labels(R.logit(p5[:,2]).astype(np.float32),z,r,rec["frames"],len(graded),blocks,base["decoder"])
