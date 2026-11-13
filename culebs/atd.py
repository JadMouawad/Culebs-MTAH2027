"""Read the raw ATD into numpy arrays, once, and cache them locally.

The official starter streams the text file every run. That is fine for one baseline and slow for a
research loop, so this module parses it into dense arrays and keeps them in artifacts/cache/
(gitignored -- a frame-level copy of licensed data, never to be committed or shared).

    from culebs.atd import load_tracking
    trk = load_tracking()               # tactical
    trk.players.shape                   # (199319, 46, 3)  x, y normalised 0-1; speed m/s (float64)
    trk.ball.shape                      # (199319, 3)
    trk.team_of_slot                    # (46,) 0 = home, 1 = away

Row i is video frame i (ATD frame_id i + 1), checked on load: a file that is not one contiguous
record per frame from 1 raises rather than being silently re-indexed.
"""

import hashlib
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np

from culebs import ARTIFACTS_DIR
from games import GAME, require, source_frames

CACHE_DIR = os.path.join(ARTIFACTS_DIR, "cache")
CACHE_VERSION = 1

HOME, AWAY = 0, 1

Tracking = namedtuple("Tracking", "players ball team_of_slot team_names source")


def read_meta(meta_path):
    """(team_of_slot, (home_name, away_name)). First team in the file is home -- see
    starter/baseline.py, which this must agree with."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    team_of_slot = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return team_of_slot, (teams[0][1], teams[1][1])


def parse_raw(raw_path, n_slots, n_frames):
    """Dense (players, ball) arrays from the raw text. NaN where the tracker saw nothing.

    Raises if any record has the wrong number of fields or the frame ids are not exactly 1..n.
    """
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float64)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    expected_id = 1
    with open(raw_path) as handle:
        for line in handle:
            frame_part, player_part, ball_part = line.rstrip("\n").split(":")
            frame_id = int(frame_part)
            if frame_id != expected_id:
                raise ValueError("{}: frame_id {} where {} was expected".format(
                    raw_path, frame_id, expected_id))
            values = player_part.replace(";", ",").split(",")
            if len(values) != n_slots * 3:
                raise ValueError("{}: frame_id {} has {} player values, expected {}".format(
                    raw_path, frame_id, len(values), n_slots * 3))
            row = frame_id - 1
            players[row] = np.array(values, dtype=np.float64).reshape(n_slots, 3)
            ball[row] = np.array(ball_part.split(","), dtype=np.float64)
            expected_id += 1
    if expected_id - 1 != n_frames:
        raise ValueError("{}: {} records, expected {}".format(raw_path, expected_id - 1, n_frames))
    return players, ball


def _file_key(path):
    """Cheap identity of a source file: name, size and mtime. Enough to invalidate a cache."""
    stat = os.stat(path)
    text = "{}|{}|{}|{}".format(os.path.basename(path), stat.st_size, int(stat.st_mtime),
                                CACHE_VERSION)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def load_tracking(source="tactical", game=GAME, data_dir=None, use_cache=True):
    """The ATD of one source as a Tracking tuple. Parsed once, then read from the local cache."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team_of_slot, team_names = read_meta(meta_path)
    n_frames = source_frames(source, game)

    cache_path = os.path.join(CACHE_DIR, "{}_{}_{}.npz".format(game, source, _file_key(raw_path)))
    if use_cache and os.path.isfile(cache_path):
        with np.load(cache_path) as cached:
            players, ball = cached["players"], cached["ball"]
    else:
        players, ball = parse_raw(raw_path, len(team_of_slot), n_frames)
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(cache_path, players=players, ball=ball)
    return Tracking(players, ball, team_of_slot, team_names, source)
