"""Write and check a frame,label submission.

Always written with LF line endings, whatever the platform, so the same code produces the same
bytes on every teammate's machine (RULES.md rule 3: "not approximately -- the same file").
Validation is delegated to the official evaluation/possession_metrics.py, never re-implemented.
"""

import hashlib
import os
import sys

import numpy as np

from culebs import ROOT
from games import GAME, GAMES

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

VALID_LABELS = ("H", "A", "D", "X")


def write_submission(path, labels):
    """Write labels (one per video frame) as frame,label with LF endings. Returns the sha256."""
    labels = np.asarray(labels)
    n_expected = GAMES[GAME]["n_video_frames"]
    if len(labels) != n_expected:
        raise ValueError("{} labels, the submission needs exactly {}".format(len(labels), n_expected))
    bad = set(np.unique(labels)) - set(VALID_LABELS)
    if bad:
        raise ValueError("labels outside H/A/D/X: {}".format(sorted(bad)))

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    body = "frame,label\n" + "".join("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    data = body.encode("ascii")
    with open(path, "wb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def validate(path):
    """None if the official checks pass, otherwise the official one-line reason."""
    table = pm.read_label_csv(path)
    return pm.validate_submission(table, GAMES[GAME]["n_video_frames"])


def normalised_sha256(path):
    """sha256 of the file with CRLF folded to LF -- compares files across git autocrlf settings."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()
