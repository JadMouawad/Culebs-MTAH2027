"""Team Culebs research code for More than a Hack 2027, Phase 1 (ball possession).

Sits beside the official repository files (games.py, evaluation/, starter/) and imports from them
rather than copying them, so the contract -- frame indexing, asset paths, the scorer -- has exactly
one definition.

    culebs.atd          fast ATD reader -> numpy arrays, with an optional local cache
    culebs.baselines    the official nearest-player baseline, re-expressed on those arrays
    culebs.submission   write / validate a frame,label file
    culebs.guard        the only sanctioned door to the public 5-minute ground truth
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Everything generated lives under here and is gitignored.
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
