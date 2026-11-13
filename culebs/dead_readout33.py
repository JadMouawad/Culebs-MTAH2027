"""Sprint 33 PFF-free compact live/dead readout on top of a frozen leader (experiments/sprint33_spellf1_protocol.json, A).

Design follows the published architecture of Marc's ens6_dead_readout_v1 (bounded binary dead correction added to the
teacher's dead log-odds; teacher team ratio kept; missing-live cost; L2 on the correction; KL to confident teacher frames)
but contains NONE of its learned parts: no PFF-trained encoder, no PFF normalisation, no PFF-derived prediction. The
encoder is replaced by fixed, label-free multi-scale summaries of the competition-format tracking features.
"""

import numpy as np

from culebs.dl.features import COMMON, TEAM

SCALES = (10, 30, 75)                    # +/- 2 s, 6 s, 15 s at 5 Hz
CTX = ("observed", "missing", "ball_speed", "static", "motion", "out", "pes", "exited", "stale")


def _box(x, h):
    """Centred moving mean over +/- h samples (edge-normalised), column-wise."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), 0)
    n = len(x)
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def features(d):
    """(n5, F) float32 from one entry's 5 Hz features and teacher probabilities; names in feature_names()."""
    cm = np.nan_to_num(d["common"].astype(np.float64))
    tm = np.nan_to_num(d["team"].astype(np.float64))
    sym = tm.mean(1)                                                   # H/A-symmetric team summaries
    vis = d["nodes"][..., -1].reshape(len(cm), -1).mean(1, keepdims=True)
    tl = logit(d["teacher"][:, 2])[:, None]
    tmax = d["teacher"][:, :2].max(1, keepdims=True)
    base = np.hstack([cm, sym[:, [TEAM.index(k) for k in ("mean_speed", "near_speed", "centroid_d", "n5")]], vis, tl, tmax])
    ctx = np.hstack([cm[:, [COMMON.index(k) for k in CTX]], sym[:, [TEAM.index("mean_speed")]], tl])
    parts = [base] + [_box(ctx, h) for h in SCALES]
    return np.hstack(parts).astype(np.float32)


def feature_names():
    base = list(COMMON) + ["mean_speed", "near_speed", "centroid_d", "n5", "visible", "teacher_dead_logit", "teacher_team_max"]
    ctx = list(CTX) + ["mean_speed", "teacher_dead_logit"]
    return base + ["{}@{}".format(c, h) for h in SCALES for c in ctx]


def norm_stats(items):
    """Provider-balanced mean / sd (each provider's matches averaged first)."""
    means, sq = [], []
    for pv in sorted({it["provider"] for it in items}):
        ch = [it for it in items if it["provider"] == pv]
        means.append(np.mean([it["x"].mean(0, dtype=np.float64) for it in ch], 0))
        sq.append(np.mean([np.square(it["x"].astype(np.float64)).mean(0) for it in ch], 0))
    mu = np.mean(means, 0)
    sd = np.sqrt(np.maximum(np.mean(sq, 0) - mu * mu, 0)) + 1e-3
    return mu.astype(np.float32), sd.astype(np.float32)


def fit(items, cfg):
    """items: dict(provider, x (n,F), y (n,) in {0,1,2}, missing (n,) bool, teacher_dead (n,)). Frames with y outside
    {0,1,2} must already be removed. Returns parameters (weight, bias, mean, sd) and a loss history."""
    import torch
    from torch.nn import functional as fn
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    mu, sd = norm_stats(items)
    nf = items[0]["x"].shape[1]
    layer = torch.nn.Linear(nf, 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    opt = torch.optim.AdamW(layer.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    prep = []
    for it in items:
        w = np.ones(len(it["y"]), np.float32)
        w[(it["y"] < 2) & it["missing"]] *= cfg["missing_live_cost"]
        p = np.clip(it["teacher_dead"], 1e-6, 1 - 1e-6).astype(np.float32)
        prep.append(dict(provider=it["provider"], x=((it["x"] - mu) / sd).astype(np.float32), y=(it["y"] == 2).astype(np.float32),
                         w=w, lp=np.log(p / (1 - p)).astype(np.float32), pt=p))
    groups = [[i for i, it in enumerate(prep) if it["provider"] == pv] for pv in sorted({it["provider"] for it in prep})]
    hist = []
    per = cfg["batch"] // 12
    for step in range(cfg["steps"]):
        xs, ys, ws, lps, pts = [], [], [], [], []
        for _ in range(12):                                  # equal provider, then equal match sampling
            it = prep[int(rng.choice(groups[int(rng.integers(len(groups)))]))]
            idx = rng.integers(len(it["y"]), size=per)
            xs.append(it["x"][idx]); ys.append(it["y"][idx]); ws.append(it["w"][idx]); lps.append(it["lp"][idx]); pts.append(it["pt"][idx])
        x, y, w, lp, pt = [torch.from_numpy(np.concatenate(v)) for v in (xs, ys, ws, lps, pts)]
        delta = cfg["logit_cap"] * torch.tanh(layer(x)[:, 0])
        z = lp + delta
        loss = (fn.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum()
        loss = loss + cfg["l2_weight"] * delta.square().mean()
        conf = (pt >= 0.9) | (pt <= 0.1)
        if conf.any():
            loss = loss + cfg["kl_weight"] * fn.binary_cross_entropy_with_logits(z[conf], pt[conf])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step + 1 == cfg["steps"]:
            hist.append((step + 1, round(float(loss.detach()), 5)))
    return dict(weight=layer.weight.detach().numpy()[0].astype(np.float32), bias=np.float32(layer.bias.detach()[0]),
                mean=mu, sd=sd), hist


def predict_dead(par, x, teacher_dead, cap):
    delta = cap * np.tanh(((x - par["mean"]) / par["sd"]) @ par["weight"] + par["bias"])
    z = logit(teacher_dead) + delta
    return 1 / (1 + np.exp(-np.clip(z, -30, 30))), delta


def combine(teacher, dead):
    """Teacher (n,3) H/A/D and a new dead probability -> (n,3) with the teacher's conditional team ratio."""
    alive = teacher[:, :2].sum(1)
    ratio = np.divide(teacher[:, 0], alive, out=np.full(len(teacher), 0.5), where=alive > 0)
    return np.column_stack(((1 - dead) * ratio, (1 - dead) * (1 - ratio), dead)).astype(np.float32)
