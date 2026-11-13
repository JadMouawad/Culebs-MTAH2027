"""All original node channels plus six fixed kinematic quantities; NumPy only."""
import numpy as np

from culebs.dl import npinfer_joint49 as N
from culebs.dl import npinfer12 as N12
from culebs.dl import npinfer_radial49 as I


def enrich(nodes):
    x=np.asarray(nodes,np.float32)
    if x.shape[-1]!=7: raise ValueError("Expected original seven-channel nodes")
    invariant=I.invariants(x)
    valid=x[...,6:7]>.5
    return np.where(valid,np.concatenate((x[...,:6],invariant[...,:6],valid.astype(np.float32)),-1),0.).astype(np.float32)


def normalize(nodes,stats):
    x=np.asarray(nodes,np.float32).copy()
    if x.shape[-1]!=13: raise ValueError("Expected enriched thirteen-channel nodes")
    valid=x[...,-1:]>.5
    x[...,:12]=np.where(valid,(x[...,:12]-stats["nm"])/stats["ns"],0.)
    return x


def node_inputs(nodes,w):
    return normalize(nodes,dict(nm=w["node_norm.nm"],ns=w["node_norm.ns"])).astype(np.float64)


def encode_nodes(nodes,w):
    x=node_inputs(nodes,w)
    valid=x[...,-1:]>.5
    h=N.NP.gelu(x@w["enc.phi.0.weight"].astype(np.float64).T+w["enc.phi.0.bias"].astype(np.float64))
    h=(h@w["enc.phi.2.weight"].astype(np.float64).T+w["enc.phi.2.bias"].astype(np.float64))*valid
    mean=h.sum(-2)/np.maximum(valid.sum(-2),1)
    maximum=np.where(valid,h,-1e4).max(-2)
    maximum=np.where(valid.any(-2),maximum,0.)
    return np.concatenate((mean,maximum),-1).reshape(len(x),-1)


def load(path,expected_sha256=None):
    w=N.load(path,expected_sha256)
    if w["cfg"].get("node_transform")!="rich49" or w["enc.phi.0.weight"].shape!=(32,13) or w["node_norm.nm"].shape!=(12,):
        raise ValueError("Weight file does not declare the enriched representation")
    return w


def predict_ratio(w,rec):
    x,_=N12.inputs(rec,w)
    x=np.concatenate((x,encode_nodes(enrich(rec["nodes"]),w)),-1)
    logits=N.forward(w,x)[:,1]*w["cfg"]["logit_scale"]
    return (1/(1+np.exp(-np.clip(logits,-30,30)))).astype(np.float32)


def ensemble_ratio(members,rec):
    return np.mean([predict_ratio(w,rec) for w in members],0).astype(np.float64)
