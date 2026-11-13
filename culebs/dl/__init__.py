"""Sprint 10: learned possession models. EXPERIMENTAL -- not part of any frozen contender.

Never imported by culebs.models, culebs.release or scripts/make_submission.py (tested). Only
culebs.dl.features is numpy-only; culebs.dl.nets and culebs.dl.train need PyTorch, which lives in
a separate research environment and is never a release dependency.

    culebs.dl.features   provider-agnostic, team-symmetric, permutation-invariant frame features
                         from any Tracking tuple (DFL, SkillCorner, Metrica, the competition ATD)
    culebs.dl.nets       Model A (TCN, direct / hierarchical / multi-task heads), Model B (player
                         set/graph attention encoder + TCN), the ball-recovery network
    culebs.dl.train      deterministic training, augmentation, inference, decoding, fusion with e110a
"""
