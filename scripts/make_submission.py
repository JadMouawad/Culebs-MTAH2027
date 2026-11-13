"""One command from the competition bundle to the final <team_name>.csv.

    $env:MTAH_DATA_DIR = "D:\\path\\to\\mtah2027-data"        # bash: export MTAH_DATA_DIR=...
    .venv\\Scripts\\python scripts/make_submission.py              # frozen primary (e110a)
    .venv\\Scripts\\python scripts/make_submission.py --candidate e108_geometric_conservative_smooth50

Steps, each a hard stop with a one-line reason (exit code 2):
  1. the frozen registry (submission/contenders.json) and the candidate's config, hash-checked
  2. the team name from submission/team.json; it must be confirmed as the registered name
  3. the pinned environment (requirements-lock.txt: Python, numpy, pandas) and a clean git
     checkout: the final file must come from a committed revision
  4. MTAH_DATA_DIR and the required match files (never downloaded); hashes checked against the
     team-only docs/data_manifest.sha256 when it is present
  5. parse the tactical ATD (no cache) and run the frozen candidate; no ground truth is read
  6. exactly 199,319 rows of H/A/D/X; the official validator; the frozen sha256
  7. write <out>/<team_name>.csv (an existing different file is never replaced without --force)
     and <team_name>.manifest.json (code revision, config, environment, input hashes; relative
     paths only)

FINAL vs non-final (Sprint 9). "final": true only when NONE of these hold, decided before
anything is written and listed in the manifest's "nonfinal_reasons":
  any development flag (--allow-unconfirmed-team-name, --allow-env-mismatch, --allow-dirty,
  --skip-input-hashes, --no-checksum); --contenders / --team-config pointing anywhere but the
  committed submission/ files; a dirty tree; an unconfirmed name; an environment mismatch; frozen
  source files that differ from the registry's source_sha256; a candidate whose role is not
  primary or fallback. A non-final run never writes into a folder that holds a final submission.

Default output folder: outputs/submission/ (gitignored).
"""

import argparse
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from culebs import OUTPUTS_DIR  # noqa: E402
from culebs import release as rel  # noqa: E402


def run(args, environ=None, load=None):
    """The pipeline; returns the manifest. `load(data_dir)` -> Tracking (tests pass a fixture)."""
    started = time.perf_counter()
    reg = rel.load_contenders(args.contenders)
    cand = rel.select_candidate(reg, args.candidate)
    config = rel.load_config(cand)
    name, confirmed = rel.team_name(args.team_config, args.team_name, args.allow_unconfirmed_team_name)
    versions, mismatch = rel.check_environment(args.allow_env_mismatch)
    tree = rel.check_source_tree(args.allow_dirty)
    root = rel.data_dir(environ)
    inputs, input_status = rel.check_inputs(root, cand["assets"], verify=not args.skip_input_hashes)
    if load is None:
        from culebs.atd import load_tracking

        def load(d):
            return load_tracking(data_dir=d, use_cache=False)
    trk = load(root)
    out = args.out or os.path.join(OUTPUTS_DIR, "submission")
    drift = rel.source_drift(reg)
    reasons = rel.nonfinal_reasons(args, cand, confirmed, tree, mismatch, drift,
                                   rel.trusted_config_reasons(args.contenders, args.team_config))
    rel.protect_final_output(out, name, final=not reasons)     # before anything is written
    result = rel.build(cand, config, trk, out, name, force=args.force,
                       expected=False if args.no_checksum else None)
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "team_name": name, "team_name_confirmed": confirmed,
        "candidate": {k: cand[k] for k in ("id", "role", "config", "config_sha256", "method")},
        "config": config,
        "submission": result,
        "code": rel.code_state(reg.get("source_files", [])),
        "frozen_code_drift": drift,
        "environment": versions, "environment_mismatch": mismatch,
        "inputs_sha256": inputs, "inputs_status": input_status,
        "ground_truth_read": False,
        "final": bool(not reasons and result["frozen_sha256_match"]),
        "nonfinal_reasons": reasons,
        "runtime_s": round(time.perf_counter() - started, 1),
    }
    rel.write_new(os.path.join(out, name + ".manifest.json"),
                  (json.dumps(manifest, indent=1) + "\n").encode("utf-8"), force=True)
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--candidate", help="contender id (default: the registry's primary)")
    p.add_argument("--out", help="output folder (default outputs/submission)")
    p.add_argument("--force", action="store_true", help="replace an existing, different <team>.csv")
    p.add_argument("--team-name", help="explicit team name; confirmed only if it equals the confirmed name in "
                                       "submission/team.json, otherwise needs --allow-unconfirmed-team-name")
    p.add_argument("--allow-unconfirmed-team-name", action="store_true",
                   help="non-final runs only: accept a team name not yet confirmed as registered")
    p.add_argument("--allow-env-mismatch", action="store_true", help="non-final runs only")
    p.add_argument("--allow-dirty", action="store_true",
                   help="non-final runs only: accept uncommitted changes or a non-git copy")
    p.add_argument("--skip-input-hashes", action="store_true",
                   help="non-final runs only: do not check input hashes against the team data manifest")
    # development-only, hidden: synthetic dry runs and tests; any of them makes the run non-final
    p.add_argument("--no-checksum", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--contenders", default=rel.CONTENDERS, help=argparse.SUPPRESS)
    p.add_argument("--team-config", default=rel.TEAM_CONFIG, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        m = run(args)
    except rel.ReleaseError as exc:
        print("FAIL  {}".format(exc))
        return 2
    s = m["submission"]
    print("OK    {}  sha256 {}  rows {}  frozen match {}  ({} s)".format(
        s["file"], s["sha256"], s["rows"], s["frozen_sha256_match"], m["runtime_s"]))
    print("      candidate {} ({}); validator: {}".format(m["candidate"]["id"], m["candidate"]["role"],
                                                         s["official_validator"]))
    if not m["final"]:
        print("NOTE  NOT A FINAL RUN; do not upload this file. Reasons: {}".format(
            "; ".join(m["nonfinal_reasons"]) or "frozen checksum not matched"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
