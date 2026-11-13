"""Final submission pipeline: frozen contender -> <team_name>.csv, checked end to end.

Used by scripts/make_submission.py; every check is a function here so tests can exercise it with
synthetic fixtures. The pipeline never downloads anything, never reads a ground truth (it does not
import culebs.guard and the frozen methods need only the tactical ATD), and refuses to finish
unless the file passes the official validator AND matches the frozen checksum.

Failures raise ReleaseError with a one-line reason; the CLI turns that into exit code 2.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile

import numpy as np

from culebs import ROOT
from games import GAME, GAMES, asset_paths

sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

CONTENDERS = os.path.join(ROOT, "submission", "contenders.json")
TEAM_CONFIG = os.path.join(ROOT, "submission", "team.json")
APPROVALS = os.path.join(ROOT, "submission", "APPROVALS.json")
LOCK = os.path.join(ROOT, "requirements-lock.txt")
DATA_MANIFEST = os.path.join(ROOT, "docs", "data_manifest.sha256")
VALID_LABELS = ("H", "A", "D", "X")
DECISIONS = ("team_name", "weights_public", "licence_mit", "primary_switch")
REQUIRED_MEMBERS = 4                     # fixed here: a record cannot lower it
CANDIDATE_KEYS = ("id", "role", "config", "config_sha256", "method", "expected_sha256", "assets")
TEAM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# Pinned environment of the frozen runs; anything else must be explicitly allowed.
PINNED = ("python", "numpy", "pandas")


class ReleaseError(Exception):
    """A check failed; the message is the reason, written for the person running the command."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _lf_sha(path):
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read().replace(b"\r\n", b"\n"))


# ------------------------------------------------------------------------------- configuration
def load_contenders(path=CONTENDERS):
    """The frozen registry, validated. A corrupt or incomplete file is an error, never a default."""
    try:
        with open(path, "rb") as handle:
            reg = json.loads(handle.read())
    except FileNotFoundError:
        raise ReleaseError("contender registry not found: {}".format(os.path.relpath(path, ROOT)))
    except ValueError as exc:
        raise ReleaseError("contender registry is not valid JSON: {}".format(exc))
    cands = reg.get("candidates") if isinstance(reg, dict) else None
    if not isinstance(cands, list) or not cands:
        raise ReleaseError("contender registry has no 'candidates' list")
    for c in cands:
        missing = [k for k in CANDIDATE_KEYS if not isinstance(c, dict) or k not in c]
        if missing:
            raise ReleaseError("contender {!r} is missing {}".format(
                c.get("id") if isinstance(c, dict) else c, missing))
    ids = [c["id"] for c in cands]
    if len(set(ids)) != len(ids):
        raise ReleaseError("contender ids are not unique")
    primary = [c["id"] for c in cands if c["role"] == "primary"]
    if len(primary) != 1:
        raise ReleaseError("exactly one contender must have role 'primary', found {}".format(primary))
    return reg


def select_candidate(reg, cid=None):
    cands = {c["id"]: c for c in reg["candidates"]}
    if cid is None:
        return next(c for c in reg["candidates"] if c["role"] == "primary")
    if cid not in cands:
        raise ReleaseError("unknown candidate {!r}; frozen ones are {}".format(cid, sorted(cands)))
    return cands[cid]


def load_config(cand):
    """The candidate's experiment config, verified against its frozen hash."""
    path = os.path.join(ROOT, cand["config"])
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        raise ReleaseError("config not found: {}".format(cand["config"]))
    got = sha256_bytes(raw.replace(b"\r\n", b"\n"))
    if got != cand["config_sha256"]:
        raise ReleaseError("config {} changed since it was frozen (sha256 {} != {})".format(
            cand["config"], got[:12], cand["config_sha256"][:12]))
    try:
        config = json.loads(raw)
    except ValueError as exc:
        raise ReleaseError("config {} is not valid JSON: {}".format(cand["config"], exc))
    if config.get("method") != cand["method"] or "params" not in config:
        raise ReleaseError("config {} is incomplete or names a different method".format(cand["config"]))
    return config


def team_name(path=TEAM_CONFIG, override=None, allow_unconfirmed=False):
    """(name, confirmed). The name comes from explicit configuration, never from a default.

    submission/team.json holds {"team_name": ..., "registered_name_confirmed": bool}. The final
    file is named after the team (SUBMISSION.md), and that name must be the one every member gave
    in the registration form, which this code cannot check. So an unconfirmed name is refused
    unless the caller says, explicitly, that this run is not the final upload.

    An explicit override (--team-name) never confirms anything by itself: it counts as confirmed
    only when it equals the confirmed name in team.json. Any other name is a non-final run.
    """
    try:
        with open(path, "rb") as handle:
            cfg = json.loads(handle.read())
    except FileNotFoundError:
        raise ReleaseError("team configuration not found: {}".format(os.path.relpath(path, ROOT)))
    except ValueError as exc:
        raise ReleaseError("team configuration is not valid JSON: {}".format(exc))
    if not isinstance(cfg, dict):
        raise ReleaseError("team configuration must be a JSON object")
    registered = cfg.get("team_name")
    name = registered if override is None else override
    confirmed = bool(cfg.get("registered_name_confirmed")) and name == registered
    if not isinstance(name, str) or not name.strip():
        raise ReleaseError("team name is empty; set team_name in submission/team.json")
    if not TEAM_NAME_RE.match(name):
        raise ReleaseError("team name {!r} must be letters, digits, '-' or '_' (it becomes a file name)".format(name))
    if not confirmed and not allow_unconfirmed:
        raise ReleaseError("team name {!r} is not confirmed as the registered competition name; set "
                           "registered_name_confirmed in submission/team.json after checking the "
                           "registration form (an explicit --team-name must equal that confirmed name)".format(name))
    return name, confirmed


# ------------------------------------------------------------------------------- environment
def pinned_versions(lock=LOCK):
    want = {}
    with open(lock) as handle:
        for line in handle:
            m = re.match(r"^\s*(numpy|pandas)==([^\s#]+)", line)
            if m:
                want[m.group(1)] = m.group(2)
            m = re.search(r"Python (\d+\.\d+\.\d+)", line)
            if m:
                want["python"] = m.group(1)
    return want


def current_versions():
    import pandas
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__}


def check_environment(allow_mismatch=False, lock=LOCK):
    want, have = pinned_versions(lock), current_versions()
    bad = {k: (have[k], want.get(k)) for k in PINNED if want.get(k) and have[k] != want[k]}
    if bad and not allow_mismatch:
        raise ReleaseError("environment differs from requirements-lock.txt: {}; create the pinned "
                           "venv (docs/SETUP.md) or pass --allow-env-mismatch for a non-final run".format(
                               ", ".join("{} {} (pinned {})".format(k, v[0], v[1]) for k, v in bad.items())))
    return have, bad


def data_dir(environ=None):
    """MTAH_DATA_DIR, required and checked. The pipeline never falls back to a guess."""
    env = os.environ if environ is None else environ
    value = env.get("MTAH_DATA_DIR")
    if not value or not value.strip():
        raise ReleaseError("MTAH_DATA_DIR is not set; point it at the unpacked competition bundle (DATA.md)")
    if not os.path.isdir(value):
        raise ReleaseError("MTAH_DATA_DIR does not name an existing directory")
    return value


def check_inputs(root, assets, manifest=DATA_MANIFEST, verify=True):
    """Relative path -> sha256 for each required asset. Missing files are an error; nothing is
    downloaded. If the team-only data manifest is present, every hash must match it."""
    paths = asset_paths(GAME, root)
    missing, rel = [], {}
    for name in assets:
        if name not in paths:
            raise ReleaseError("unknown asset {!r}".format(name))
        if not os.path.isfile(paths[name]):
            missing.append(os.path.relpath(paths[name], root).replace(os.sep, "/"))
        rel[name] = os.path.relpath(paths[name], root).replace(os.sep, "/")
    if missing:
        raise ReleaseError("required match files are missing under MTAH_DATA_DIR: {} (they are licensed "
                           "and never downloaded by this pipeline; see DATA.md)".format(", ".join(missing)))
    hashes = {rel[n]: sha256_file(paths[n]) for n in assets}
    status = "not checked"
    if verify and os.path.isfile(manifest):
        known = {}
        with open(manifest) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 3 and not line.startswith("#"):
                    known[parts[2]] = parts[0]
        wrong = [p for p, h in hashes.items() if p in known and known[p] != h]
        if wrong:
            raise ReleaseError("input files differ from docs/data_manifest.sha256: {}".format(wrong))
        status = "matches docs/data_manifest.sha256" if all(p in known for p in hashes) else "partly listed"
    return hashes, status


# ------------------------------------------------------------------------------- prediction
def predict(config, trk):
    from culebs.diagnostics import graded_mask
    if config["method"] in ("learned_ens6", "learned_ens6_dead33", "learned_x_t1_50"):
        from culebs.learned import WeightsError, predict_ens6, predict_ens6_dead33, predict_x_t1_50
        fn = {"learned_ens6": predict_ens6, "learned_ens6_dead33": predict_ens6_dead33, "learned_x_t1_50": predict_x_t1_50}[config["method"]]
        try:
            return np.asarray(fn(trk, graded_mask(), config["params"]))
        except WeightsError as exc:
            raise ReleaseError(str(exc))
    if config["method"] == "learned_shared49":
        from culebs import rich49
        from culebs.learned import WeightsError
        try:
            return np.asarray(rich49.predict(trk, graded_mask(), config["params"]))
        except (WeightsError, ValueError, FileNotFoundError) as exc:
            raise ReleaseError(str(exc))
    from culebs import models
    labels, _ = models.predict(config["method"], trk, graded_mask(), None, config.get("params", {}))
    return np.asarray(labels)


def check_labels(labels):
    n = GAMES[GAME]["n_video_frames"]
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n:
        raise ReleaseError("model produced {} labels; the submission needs exactly {}".format(len(labels), n))
    bad = sorted(set(np.unique(labels).tolist()) - set(VALID_LABELS))
    if bad:
        raise ReleaseError("model produced labels outside H/A/D/X: {}".format(bad))
    return labels


def render(labels):
    """The exact bytes of the submission: LF endings, official header, frames 0..n-1 in order."""
    return ("frame,label\n" + "".join("{},{}\n".format(i, lab) for i, lab in enumerate(labels))).encode("ascii")


def official_validate(path):
    """Run evaluation/validate_submission.py in a subprocess: the exact checks the scorer runs."""
    script = os.path.join(ROOT, "evaluation", "validate_submission.py")
    proc = subprocess.run([sys.executable, script, path], capture_output=True, text=True, cwd=ROOT)
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
    if proc.returncode != 0 or reason is not None:
        raise ReleaseError("official validator rejected the file: {}".format(
            reason or (proc.stdout + proc.stderr).strip().splitlines()[-1:]))
    keep = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().startswith(("OK", "reports"))]
    return "; ".join(keep)


def git_status(root=ROOT):
    """(head, changed paths) of the checkout, or (None, None) when it is not a git checkout."""
    try:
        head = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        porcelain = subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True,
                                            stderr=subprocess.DEVNULL)
    except Exception:
        return None, None
    return head, [ln[3:] for ln in porcelain.splitlines() if ln.strip()]


def check_source_tree(allow_dirty=False, status=git_status):
    """The final file must come from a committed revision: refuse a dirty or unknown tree."""
    head, changed = status()
    if head is None:
        if not allow_dirty:
            raise ReleaseError("not a git checkout, so the code revision cannot be recorded; run from a clone "
                               "or pass --allow-dirty for a non-final run")
        return {"head": None, "dirty": None}
    if changed and not allow_dirty:
        raise ReleaseError("source tree has {} uncommitted change(s) (e.g. {}); commit them first, or pass "
                           "--allow-dirty for a non-final run".format(len(changed), changed[0]))
    return {"head": head, "dirty": bool(changed)}


def _same_file(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def git_tracked(path, root=ROOT):
    """True when `path` is a file committed in the checkout at `root` (inside it, known to git)."""
    try:
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    except ValueError:                                   # another drive on Windows
        return False
    if rel.startswith(".."):
        return False
    try:
        subprocess.check_output(["git", "-C", root, "ls-files", "--error-unmatch", "--", rel],
                                stderr=subprocess.DEVNULL)
    except Exception:
        return False
    return True


def trusted_config_reasons(contenders, team_config, tracked=git_tracked):
    """Why the registry / team configuration in use cannot back a final run ([] = trusted).

    Trusted means: the repository's own submission/contenders.json and submission/team.json, and
    both committed to git (the clean-tree check then guarantees they are unmodified). The approval
    record, when present, must be committed too: an ignored file would pass the clean-tree check."""
    out = []
    for label, path, default in (("contender registry", contenders, CONTENDERS),
                                 ("team configuration", team_config, TEAM_CONFIG)):
        if not _same_file(path, default):
            out.append("{} overridden (development-only)".format(label))
        elif not tracked(path):
            out.append("{} is not committed to git".format(label))
    if os.path.isfile(APPROVALS) and not tracked(APPROVALS):
        out.append("approval record is not committed to git")
    return out


def source_drift(reg):
    """Frozen source files whose LF-normalised sha256 differs from the registry (or is missing)."""
    want = reg.get("source_sha256", {})
    files = reg.get("source_files", [])
    drift = [f for f in files if want.get(f) != (_lf_sha(os.path.join(ROOT, f))
                                                 if os.path.isfile(os.path.join(ROOT, f)) else None)]
    return sorted(drift) if files else ["<registry lists no source files>"]


# "release_candidate": fully gated and able to produce a final file, but not the registered primary (Sprint 29)
RELEASE_ROLES = ("primary", "fallback", "release_candidate")
# CLI flag -> attribute on the parsed args; each one makes a run non-final
DEV_FLAGS = (("--allow-unconfirmed-team-name", "allow_unconfirmed_team_name"),
             ("--allow-env-mismatch", "allow_env_mismatch"), ("--allow-dirty", "allow_dirty"),
             ("--skip-input-hashes", "skip_input_hashes"), ("--no-checksum", "no_checksum"))


def approved_candidates(path=APPROVALS):
    """Ids the team's approval record names: its selected candidate and that candidate's fallbacks."""
    try:
        with open(path, "rb") as handle:
            sel = json.loads(handle.read()).get("selected_candidate") or {}
    except (FileNotFoundError, ValueError):
        return set()
    return {sel.get("id")} | set(sel.get("fallbacks", []))


WEIGHT_SUFFIXES = (".npz", ".npy", ".pt", ".pth", ".ckpt", ".onnx", ".pkl", ".joblib", ".safetensors", ".h5", ".bin")
WEIGHT_HINT = re.compile(r"weights_dir|\.(npz|npy|pt|pth|ckpt|onnx|pkl|joblib|safetensors|h5|bin)\b")


def _collect_weights(node, folder, out):
    if isinstance(node, dict):
        folder = node.get("weights_dir", folder)
        entry = isinstance(node.get("file"), str)
        if entry:                                        # no recorded sha256: a record can never match it
            out["{}/{}".format(folder, node["file"]) if folder else node["file"]] = node.get("sha256")
        elif "file" in node:
            out["<malformed file entry>"] = None
        for key, value in node.items():
            if key != "file" and not (entry and key.startswith("source_")):   # the provenance of a listed file is not a loaded file
                _collect_weights(value, folder, out)
    elif isinstance(node, list):
        for value in node:
            _collect_weights(value, folder, out)
    elif isinstance(node, str) and node.lower().endswith(WEIGHT_SUFFIXES):
        out.setdefault(node, None)                       # no recorded sha256: a record can never match it


def candidate_weights(cand, root=ROOT):
    """{repository path: sha256} of every weight file named anywhere in a candidate's config ({} for a candidate without weights).
    Recursive, so a new config layout cannot hide a file; a bare path or an entry without a hash has sha256 None."""
    with open(os.path.join(root, cand["config"]), "rb") as handle:
        params = json.loads(handle.read()).get("params") or {}
    out = {}
    _collect_weights(params, None, out)
    return out


def approval_problems(root=ROOT):
    """Everything that stops APPROVALS.json from authorising its selected candidate ([] = complete). Consents bind to
    decisions_version and to the exact candidate id, CSV sha256 and the full set of weights the candidate and its fallbacks load."""
    with open(os.path.join(root, "submission", "APPROVALS.json"), encoding="utf-8") as handle:
        ap = json.load(handle)
    out = []
    members = ap.get("members", [])
    if ap.get("required_members", REQUIRED_MEMBERS) != REQUIRED_MEMBERS:
        out.append("required_members must be {}".format(REQUIRED_MEMBERS))
    if len(members) != REQUIRED_MEMBERS:
        out.append("expected {} members, found {}".format(REQUIRED_MEMBERS, len(members)))
    names = [m.get("name") for m in members]
    if len(set(n for n in names if n)) != len([n for n in names if n]):
        out.append("duplicate member names")
    sel = ap.get("selected_candidate") or {}
    version = ap.get("decisions_version")
    if not sel.get("id") or not version or "YYYY" in version:
        out.append("APPROVALS has no selected_candidate / decisions_version")
    else:
        out += _selection_problems(ap, sel, root)
    for i, m in enumerate(members):
        who = m.get("name") or "member {}".format(i + 1)
        if not m.get("name"):
            out.append("{}: name missing".format(who))
        for d in DECISIONS:
            if m.get(d) is not True:
                out.append("{}: no recorded consent to '{}'".format(who, d))
        if m.get("decisions_version") != version:
            out.append("{}: consent not recorded for decisions_version {}".format(who, version))
        for k in ("date", "evidence"):
            if not m.get(k):
                out.append("{}: {} missing".format(who, k))
    return out


def _selection_problems(ap, sel, root):
    out = []
    with open(os.path.join(root, "submission", "contenders.json"), encoding="utf-8") as handle:
        cands = {c["id"]: c for c in json.load(handle)["candidates"]}
    cand = cands.get(sel["id"])
    listed = {w.get("file"): w for w in sel.get("weights", [])}
    if len(listed) != len(sel.get("weights", [])):
        out.append("a weight file is listed twice")
    if cand is None:
        out.append("selected candidate {} is not registered".format(sel["id"]))
    else:
        if cand.get("expected_sha256") != sel.get("expected_sha256"):
            out.append("selected candidate checksum differs from the registry")
        text = (ap.get("decisions") or {}).get("primary_switch") or ""
        if sel["id"] not in text or str(sel.get("expected_sha256")) not in text:
            out.append("the primary_switch decision text does not name the selected candidate and its checksum")
    need = {}
    for cid in [sel["id"]] + list(sel.get("fallbacks", [])):
        if cid not in cands:
            if cid != sel["id"]:
                out.append("fallback {} is not registered".format(cid))
            continue
        try:
            need.update(candidate_weights(cands[cid], root))
        except (OSError, ValueError, KeyError):
            out.append("config of {} cannot be read".format(cid))
    for f, sha in sorted(need.items()):
        if f not in listed:
            out.append("selected_candidate.weights does not list {}".format(f))
        elif listed[f].get("sha256") != sha:
            out.append("listed sha256 differs from the config: {}".format(f))
    for f, w in sorted(listed.items(), key=lambda kv: str(kv[0])):
        if f not in need:
            out.append("weight file not used by the selected candidate or its fallbacks: {}".format(f))
        path = os.path.join(root, f or "")
        if not f or not os.path.isfile(path):
            out.append("weight file missing: {}".format(f))
        elif sha256_file(path) != w.get("sha256"):
            out.append("weight file changed: {}".format(f))
        if not w.get("licence") or "TEAM TO CONFIRM" in w["licence"]:
            out.append("no licence recorded for {}".format(f))
        if not w.get("attribution") or not os.path.isfile(os.path.join(root, w["attribution"])):
            out.append("attribution file missing for {}".format(f))
    return out


def requires_approval(cand, root=ROOT):
    """Protected unless every signal agrees there is no trained model; an unreadable config counts as protected (Sprint 39)."""
    if str(cand.get("method", "")).startswith("learned") or cand.get("uses_trained_model") or cand.get("weights"):
        return True
    try:
        if candidate_weights(cand, root):
            return True
        with open(os.path.join(root, cand["config"]), "rb") as handle:   # anywhere in the file: top level, or a dictionary key
            return bool(WEIGHT_HINT.search(handle.read().decode("utf-8", "replace").lower()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return True


def approval_reasons(cand, root=ROOT):
    """A candidate with trained weights is final only when APPROVALS names it and is complete, whatever its role says (Sprint 37c)."""
    if not requires_approval(cand, root):
        return []
    if cand.get("id") not in approved_candidates(os.path.join(root, "submission", "APPROVALS.json")):
        return ["candidate {} is not the selected candidate or a fallback in submission/APPROVALS.json".format(cand.get("id"))]
    problems = approval_problems(root)
    if problems:
        return ["submission/APPROVALS.json is incomplete for {}: {} item(s), e.g. {}".format(cand.get("id"), len(problems), problems[0])]
    return []


def nonfinal_reasons(args, cand, confirmed, tree, mismatch, drift, trust):
    """Every reason this run cannot produce the file to upload; final <=> the list is empty.
    Decided BEFORE anything is written."""
    out = ["{} given (development-only)".format(flag) for flag, attr in DEV_FLAGS if getattr(args, attr, False)]
    if not confirmed:
        out.append("team name not confirmed as the registered name")
    if tree.get("dirty") is not False:
        out.append("source tree not a clean git checkout")
    if mismatch:
        out.append("environment differs from requirements-lock.txt")
    if drift:
        out.append("frozen source files differ from the registry: {}".format(", ".join(drift)))
    if cand["role"] not in RELEASE_ROLES:
        out.append("candidate role {!r} is not a release role {}".format(cand["role"], RELEASE_ROLES))
    return out + approval_reasons(cand) + list(trust)


def protect_final_output(out_dir, name, final):
    """A non-final run never touches a folder that holds a final submission (not even with --force)."""
    if final:
        return
    path = os.path.join(out_dir, name + ".manifest.json")
    try:
        with open(path, "rb") as handle:
            existing = json.loads(handle.read())
    except (FileNotFoundError, ValueError):
        return
    if isinstance(existing, dict) and existing.get("final") is True:
        raise ReleaseError("{} holds a FINAL submission; a non-final run will not write there (use another "
                           "--out)".format(os.path.basename(os.path.normpath(out_dir)) or out_dir))


def code_state(files):
    """git HEAD, dirty flag, and sha256 (LF-normalised) of every file the frozen pipeline runs."""
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", ROOT] + list(args), text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"head": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": {f: _lf_sha(os.path.join(ROOT, f)) for f in files}}


def write_new(path, data, force=False):
    """Write atomically. An existing different file is never replaced silently."""
    if os.path.exists(path):
        with open(path, "rb") as handle:
            same = handle.read() == data
        if not same and not force:
            raise ReleaseError("{} already exists with different content; pass --force to replace it".format(
                os.path.basename(path)))
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def build(cand, config, trk, out_dir, name, force=False, expected=None):
    """Predict, check, write <name>.csv, validate it officially and compare the frozen checksum.

    expected: the sha256 to require (default the candidate's frozen one); pass False to skip the
    comparison (synthetic dry runs only). Returns the manifest dict (no absolute paths in it).
    """
    labels = check_labels(predict(config, trk))
    data = render(labels)
    digest = sha256_bytes(data)
    want = cand["expected_sha256"] if expected is None else expected
    if want and digest != want:
        raise ReleaseError("reproducibility mismatch: produced sha256 {} but {} is frozen at {}; the file "
                           "was not written".format(digest[:12], cand["id"], want[:12]))
    # validate a staged copy first, then publish it under the final name
    stage_dir = tempfile.mkdtemp()
    staged = os.path.join(stage_dir, name + ".csv")
    with open(staged, "wb") as handle:
        handle.write(data)
    try:
        validator = official_validate(staged)
    finally:
        os.remove(staged)
        os.rmdir(stage_dir)
    path = os.path.join(out_dir, name + ".csv")
    write_new(path, data, force=force)
    counts = {k: int(np.sum(labels == k)) for k in VALID_LABELS}
    return {"file": name + ".csv", "sha256": digest, "rows": int(len(labels)), "label_counts": counts,
            "frozen_sha256_match": bool(want) and digest == want, "official_validator": validator}
