"""Submit runs to the nanobrain leaderboard on Labless.

uv run python scripts/submit.py output/NAME [output/NAME2 ...]

Each run's label and notes come from its config (`name`, `notes`). Checks the eval results, signs in
with GitHub once, and posts the scores with the code as run: git remote and commit, plus the source
files and config from the run's snapshot. Each payload is also saved to
output/NAME/labless_submission.json.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import platform
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

API_URL = "https://api.labless.dev"
PROJECT = "nanobrain-smri"
MAX_NAME_LENGTH = 20
TOKEN_PATH = Path.home() / ".cache" / "nanobrain" / "labless_token.json"
# brainmarks-smri task group scored by the leaderboard. keep in sync with scripts/train.sbatch
TASK_GROUP = "open"
# source files sent for review, diffs and reproduction. the big image list is left out
REVIEW_SUFFIXES = (".py", ".yaml", ".yml")


def main(args: argparse.Namespace):
    if args.login_only:
        github_sign_in()
        return

    # recorded for maintainers. validation reruns the eval with the pinned brainmarks-smri
    brainmarks_url = importlib.metadata.distribution("brainmarks-smri").read_text("direct_url.json")
    installed_commit = json.loads(brainmarks_url).get("vcs_info", {}).get("commit_id")

    # check all runs before signing in
    payloads = []
    has_errors = False
    for run_dir in args.run_dirs:
        payload = make_payload(run_dir, installed_commit)
        payloads.append((run_dir, payload))
        for error in payload["run"]["validation_errors"]:
            print(f"error: {error}", file=sys.stderr)
            has_errors = True
    if has_errors:
        sys.exit(1)
    if args.dry_run:
        print("dry run, not submitted")
        return

    token, login = github_sign_in()
    for run_dir, payload in payloads:
        payload["contributor"]["login"] = login
        (run_dir / "labless_submission.json").write_text(json.dumps(payload, indent=2) + "\n")
        result = api_request("POST", f"/api/nano-projects/{PROJECT}/submissions", payload, token)
        print(f"submitted {run_dir}: {json.dumps(result)}")


def make_payload(run_dir: Path, brainmarks_commit: str | None) -> dict:
    """Submission payload for one run, with its validation errors. Also saved in the run dir."""
    config = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"))
    summary = json.loads((run_dir / "summary.json").read_text())
    scores = json.loads((run_dir / "eval" / "aggregate.json").read_text())["summary"]

    # code as run, by repo-relative path. the run config, with its overrides, goes in as
    # config.yaml so it reproduces with --config
    review_files = {}
    with tarfile.open(run_dir / "source.tar.gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(REVIEW_SUFFIXES):
                review_files[member.name] = tar.extractfile(member).read().decode()
    review_files["config.yaml"] = (run_dir / "config.yaml").read_text()
    name = config["name"]
    notes = config["notes"] or ""

    errors = []
    # brainmarks-smri only scores a group when all its tasks are done
    if TASK_GROUP not in scores:
        errors.append(f"{run_dir}: no {TASK_GROUP} score, the eval is incomplete")
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"{run_dir}: name {name!r} is longer than {MAX_NAME_LENGTH} characters")

    # scores per task and task group, 0 at chance and at most 1
    # all is the same as the task group here
    metrics = {task: row["score"] for task, row in scores.items() if task != "all"}

    # payload follows the nanopath labless submission format
    git = summary["git"] or {}
    payload = {
        "version": 1,
        "title": name,
        "status": "completed",
        "notes": notes,
        "contributor": {"login": None},
        "repo": {
            "remote": git.get("remote"),
            "branch": git.get("branch"),
            "commit": git.get("sha"),
            "dirty": git.get("dirty"),
            "review_files": {"source": "source.tar.gz", "files": review_files},
        },
        "run": {
            "name": run_dir.name,
            "label": name,
            "tier": "full",
            "family": "nanobrain",
            "command": "uv run python -m nanobrain.train --config config.yaml",
            "hardware": f"{summary['gpu']} (single GPU, {summary['cpus']} CPUs)",
            "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config": config,
            "summary": summary,
            "metrics": metrics,
            "environment": {
                "python": platform.python_version(),
                "brainmarks_smri_commit": brainmarks_commit,
            },
            "validation_errors": errors,
        },
        "artifacts": [],
    }
    payload_bytes = json.dumps(payload, sort_keys=True).encode()
    payload["submission_id"] = hashlib.sha256(payload_bytes).hexdigest()[:10]

    payload_path = run_dir / "labless_submission.json"
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    score = metrics.get(TASK_GROUP, float("nan"))
    print(
        f"{run_dir}: {name}, {TASK_GROUP} score {score:.3f}, {git.get('remote')} {git.get('sha')}"
    )
    return payload


def api_request(method: str, path: str, payload: dict | None = None, token: str | None = None):
    headers = {"Accept": "application/json", "User-Agent": "nanobrain-submit/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    request = urllib.request.Request(API_URL + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {path} failed: {error.code} {error.read().decode()}")


def github_sign_in() -> tuple[str, str]:
    """GitHub sign-in through Labless. Returns (access token, GitHub login).

    Uses the cached token if Labless still accepts it, otherwise the GitHub device flow: open a URL
    in any browser and enter the printed code. The new token is cached for next time.
    """
    if TOKEN_PATH.exists():
        access_token = json.loads(TOKEN_PATH.read_text())["access_token"]
        try:
            login = api_request("GET", "/api/auth/github/me", token=access_token)["login"]
            print(f"signed in as {login} (cached token {TOKEN_PATH})")
            return access_token, login
        except RuntimeError:
            print("cached token was rejected, signing in again")

    device = api_request("POST", "/api/auth/github/device", {})
    print(f"Sign in with GitHub: open {device['verification_uri']} and enter {device['user_code']}")
    interval = int(device["interval"])
    deadline = time.monotonic() + int(device["expires_in"])
    while time.monotonic() < deadline:
        time.sleep(interval)
        token = api_request(
            "POST", "/api/auth/github/device/token", {"device_code": device["device_code"]}
        )
        if token["status"] == "authorized":
            access_token = token["access_token"]
            login = api_request("GET", "/api/auth/github/me", token=access_token)["login"]
            print(f"signed in as {login}")
            # readable only by the user
            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_PATH.touch(mode=0o600)
            TOKEN_PATH.write_text(json.dumps({"access_token": access_token, "login": login}))
            return access_token, login
        if token["status"] == "slow_down":
            interval += 5
        elif token["status"] != "authorization_pending":
            raise RuntimeError(f"GitHub sign-in failed: {token}")
    raise RuntimeError("GitHub sign-in code expired")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="*", help="run output folders, output/NAME")
    parser.add_argument("--dry-run", action="store_true", help="check and save, don't submit")
    parser.add_argument("--login-only", action="store_true", help="sign in once, for slurm jobs")
    args = parser.parse_args()
    main(args)
