"""The official reference baseline, re-expressed on the cached arrays.

Same algorithm as starter/baseline.py -- nearest tracked player to the ball, then a centred
majority vote -- vectorised over frames. scripts/make_baseline.py checks that it writes files
identical to the official ones, so this is the anchor every experiment is compared against.
"""

import numpy as np

from culebs.atd import AWAY, HOME
from games import GAME, GAMES

NOBODY = -1


def nearest_player_codes(trk, pitch=None):
    """Per-frame team code of the player nearest the ball (metres), NOBODY where no ball."""
    length, width = pitch or GAMES[GAME]["pitch"]
    scale = np.array([length, width], dtype=np.float64)
    players = trk.players[:, :, :2].astype(np.float64) * scale
    ball = trk.ball[:, None, :2].astype(np.float64) * scale

    d2 = np.sum((players - ball) ** 2, axis=2)                  # (n, slots), NaN where missing
    d2 = np.where(np.isnan(d2), np.inf, d2)
    nearest = np.argmin(d2, axis=1)                              # first minimum, as the official loop
    codes = trk.team_of_slot[nearest].astype(np.int8)
    no_answer = np.isnan(trk.ball[:, 0]) | np.isinf(d2[np.arange(len(d2)), nearest])
    codes[no_answer] = NOBODY
    return codes


def smooth(codes, window):
    """Centred +/-window majority vote; D where nobody is seen. Ties go home. Mirrors
    starter/baseline.smooth exactly (tests/test_baselines.py holds it to that)."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_cum = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_cum = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels
