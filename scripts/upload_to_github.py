#!/usr/bin/env python3
"""
upload_to_github.py -- publish this project (code + dataset) to GitHub.

The dataset is ~2.5 GB across ~110 chunk files.  GitHub rejects any single
file over 100 MB and any single push over ~2 GB, so this script commits the
data **year by year** and pushes after each commit.  Every chunk here is
15-40 MB, comfortably under the file limit.

Usage
-----
    export GITHUB_TOKEN=ghp_xxx
    python scripts/upload_to_github.py --repo pridictior_snow4

    # code only, skip the dataset
    python scripts/upload_to_github.py --repo pridictior_snow4 --no-data

The token needs the ``repo`` scope.  It is read from ``--token`` or the
``GITHUB_TOKEN`` / ``GH_TOKEN`` environment variable and is never written to
disk: the remote is configured without credentials and the token is injected
per-push through a temporary credential helper.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from typing import List, Optional

API = "https://api.github.com"
MAX_FILE_MB = 95


def api(path: str, token: str, method: str = "GET",
        payload: Optional[dict] = None) -> dict:
    req = urllib.request.Request(
        f"{API}{path}", method=method,
        data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json",
                 "User-Agent": "btcpred-uploader"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise RuntimeError(f"GitHub API {method} {path} -> {e.code}: {detail}") from None


def run(cmd: List[str], cwd: str, check: bool = True,
        quiet: bool = False) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if not quiet and p.stdout.strip():
        print(p.stdout.strip()[-2000:])
    if p.returncode != 0:
        msg = (p.stderr or p.stdout).strip()[-2000:]
        if check:
            raise RuntimeError(f"$ {' '.join(cmd)}\n{msg}")
        if not quiet:
            print(msg)
    return p


def ensure_repo(token: str, owner: str, name: str, private: bool,
                description: str) -> dict:
    try:
        return api(f"/repos/{owner}/{name}", token)
    except RuntimeError as e:
        if "404" not in str(e):
            raise
    print(f"[github] creating repository {owner}/{name}")
    return api("/user/repos", token, "POST",
               {"name": name, "private": private, "description": description,
                "auto_init": False, "has_issues": True, "has_wiki": False})


def oversized(root: str) -> List[str]:
    bad = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in files:
            p = os.path.join(d, f)
            try:
                if os.path.getsize(p) > MAX_FILE_MB * 1_000_000:
                    bad.append(f"{os.path.relpath(p, root)} "
                               f"({os.path.getsize(p)/1e6:.0f} MB)")
            except OSError:
                pass
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="pridictior_snow4")
    ap.add_argument("--owner", default=None, help="default: the token's user")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), ".."))
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN")
                    or os.environ.get("GH_TOKEN"))
    ap.add_argument("--branch", default="main")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-data", action="store_true",
                    help="push code only, leave data/ out")
    ap.add_argument("--data-dir", default="data/btcusdt_1s")
    ap.add_argument("--force", action="store_true",
                    help="force-push (overwrite remote history)")
    args = ap.parse_args()

    if not args.token:
        print("!! no token: pass --token or set GITHUB_TOKEN")
        return 2
    root = os.path.abspath(args.root)
    me = api("/user", args.token)
    owner = args.owner or me["login"]
    print(f"[github] authenticated as {me['login']}")

    big = oversized(root)
    if big:
        print("!! files over GitHub's 100 MB limit:")
        for b in big:
            print("   ", b)
        return 3

    ensure_repo(args.token, owner, args.repo, args.private,
                "Second-by-second Bitcoin price predictor: live market "
                "simulator, CPU market analyser, GPU price predictor, "
                "Snowflake notebook. Trains fully offline.")

    if not os.path.isdir(os.path.join(root, ".git")):
        run(["git", "init", "-q"], root)
    cfg = {
        "user.email": f"{me['login']}@users.noreply.github.com",
        "user.name": me["login"],
        "http.postBuffer": "524288000",
        # The chunks are already zlib-compressed, so delta-compressing them is
        # pure waste -- and `pack-objects` will OOM on a small box trying.
        # These settings keep peak RSS at a few hundred MB.
        "core.bigFileThreshold": "8m",
        "core.compression": "0",
        "pack.threads": "1",
        "pack.window": "0",
        "pack.depth": "1",
        "pack.windowMemory": "64m",
        "pack.deltaCacheSize": "32m",
        "pack.packSizeLimit": "512m",
    }
    for k, v in cfg.items():
        run(["git", "config", k, v], root)
    run(["git", "checkout", "-q", "-B", args.branch], root)

    remote = f"https://github.com/{owner}/{args.repo}.git"
    run(["git", "remote", "remove", "origin"], root, check=False, quiet=True)
    run(["git", "remote", "add", "origin", remote], root)

    # token lives only in this process's environment, never in .git/config
    env_remote = f"https://{me['login']}:{args.token}@github.com/{owner}/{args.repo}.git"

    data_rel = args.data_dir.replace("\\", "/")
    first_push = [True]

    def push(msg: str, pathspec: Optional[List[str]] = None) -> None:
        run(["git", "add", "-A", "--"] + (pathspec or ["."]), root, quiet=True)
        st = run(["git", "status", "--porcelain"], root, quiet=True)
        if not st.stdout.strip():
            print(f"[github] nothing to commit for '{msg}'")
            return
        run(["git", "commit", "-q", "-m", msg], root)
        cmd = ["git", "push", "-q"]
        if args.force and first_push[0]:
            cmd.append("--force")          # only the first push may rewrite
        cmd += [env_remote, f"HEAD:refs/heads/{args.branch}"]
        print(f"[github] pushing: {msg}")
        run(cmd, root)
        first_push[0] = False

    # ---- 1. code first, so the repo is usable even if the data stalls -----
    #        pathspec magic keeps the dataset out without touching .gitignore
    push("Bitcoin price predictor: simulator, analyser, predictor, "
         "Snowflake notebook",
         pathspec=[".", f":(exclude){data_rel}/*.npz"])

    if args.no_data:
        print(f"[github] done (code only): {remote}")
        return 0

    # ---- 2. dataset, one commit per year ----------------------------------
    full = os.path.join(root, args.data_dir)
    files = sorted(f for f in os.listdir(full) if f.endswith(".npz")) \
        if os.path.isdir(full) else []
    if not files:
        print(f"[github] no chunks in {full}; run the downloader first")
        return 0
    by_year = defaultdict(list)
    for f in files:
        m = re.match(r"(\d{4})-\d{2}\.npz", f)
        by_year[m.group(1) if m else "misc"].append(f)

    total_mb = sum(os.path.getsize(os.path.join(full, f)) for f in files) / 1e6
    print(f"[github] uploading {len(files)} chunks, {total_mb:,.0f} MB, "
          f"one commit per year")
    for year in sorted(by_year):
        mb = sum(os.path.getsize(os.path.join(full, f))
                 for f in by_year[year]) / 1e6
        spec = [f"{data_rel}/{f}" for f in by_year[year]]
        push(f"dataset: BTCUSDT 1s bars for {year} "
             f"({len(by_year[year])} chunks, {mb:,.0f} MB)", pathspec=spec)

    push("dataset manifest + remaining assets")
    print(f"[github] done -> {remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
