#!/usr/bin/env python3
"""Fetch and lock the official Diffusion Policy artifacts used by the bounded run."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MIN_FREE_BYTES = 200 * 1024 ** 3


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def parse_headers(text):
    result = {}
    for raw_line in text.replace("\r", "").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        result[key.strip().lower()] = value.strip()
    return result


def head(url):
    proc = run(
        ["curl", "--fail", "--location", "--head", "--silent", "--show-error", "--max-time", "60", url],
        stdout=subprocess.PIPE,
    )
    return parse_headers(proc.stdout)


def tmux_session_active(session):
    command = ["tmux"]
    socket = os.environ.get("FAIL_DETECT_TMUX_SOCKET")
    if socket:
        command.extend(["-L", socket])
    exists = subprocess.run(
        command + ["has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode != 0:
        return False
    panes = subprocess.run(
        command + ["list-panes", "-t", session, "-F", "#{pane_dead}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if panes.returncode != 0:
        return True
    return any(line.strip() == "0" for line in panes.stdout.splitlines())


def require_resource_gate(gpu_index, sessions):
    blockers = []
    for session in sessions:
        if tmux_session_active(session):
            blockers.append("tmux:" + session)

    proc = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        blockers.append("gpu{}:unreadable".format(gpu_index))
    else:
        pids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if pids:
            blockers.append("gpu{}:compute_pids={}".format(gpu_index, ",".join(pids)))
    if blockers:
        raise RuntimeError("resource gate closed: " + " ".join(blockers))


def validate_remote(record):
    headers = head(record["url"])
    actual_length = int(headers.get("content-length", "-1"))
    if actual_length != int(record["content_length"]):
        raise RuntimeError("Content-Length changed for {}: {} != {}".format(record["url"], actual_length, record["content_length"]))
    for field, header_name in (("etag", "etag"), ("last_modified", "last-modified")):
        actual = headers.get(header_name)
        if actual != record[field]:
            raise RuntimeError("{} changed for {}: {!r} != {!r}".format(header_name, record["url"], actual, record[field]))
    return {
        "content_length": actual_length,
        "etag": headers["etag"],
        "last_modified": headers["last-modified"],
    }


def fetch(repo_root, name, record, previous_lock):
    destination = repo_root / record["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(record["content_length"])
    if destination.exists():
        if destination.stat().st_size != expected_size:
            raise RuntimeError("existing artifact has wrong size: {}".format(destination))
    else:
        part = destination.with_name(destination.name + ".part")
        if part.exists() and part.stat().st_size > expected_size:
            raise RuntimeError("partial download exceeds expected size: {}".format(part))
        run([
            "curl", "--fail", "--location", "--retry", "8", "--retry-all-errors",
            "--continue-at", "-", "--output", str(part), record["url"],
        ])
        if part.stat().st_size != expected_size:
            raise RuntimeError("downloaded artifact has wrong size: {}".format(part))
        os.replace(str(part), str(destination))

    digest = sha256_file(destination)
    old_digest = previous_lock.get(name, {}).get("sha256")
    if old_digest and old_digest != digest:
        raise RuntimeError("SHA-256 changed for {}: {} != {}".format(destination, digest, old_digest))
    return destination, digest


def extract_transport(repo_root, archive, record, previous_lock):
    destination = repo_root / record["extracted_destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    member = record["archive_member"]
    with zipfile.ZipFile(str(archive), "r", allowZip64=True) as bundle:
        try:
            info = bundle.getinfo(member)
        except KeyError:
            raise RuntimeError("official archive does not contain exact member: {}".format(member))
        if destination.exists():
            if destination.stat().st_size != info.file_size:
                raise RuntimeError("existing extracted dataset has wrong size: {}".format(destination))
        else:
            part = destination.with_name(destination.name + ".part")
            if part.exists():
                raise RuntimeError("stale extraction partial requires inspection: {}".format(part))
            with bundle.open(info, "r") as source, part.open("xb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if part.stat().st_size != info.file_size:
                raise RuntimeError("extracted dataset has wrong size: {}".format(part))
            os.replace(str(part), str(destination))

    digest = sha256_file(destination)
    old_digest = previous_lock.get("transport_image_abs", {}).get("sha256")
    if old_digest and old_digest != digest:
        raise RuntimeError("SHA-256 changed for extracted dataset")
    return destination, info.file_size, digest


def ensure_symlink(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(str(target), str(link.parent))
    if os.path.lexists(str(link)):
        if not link.is_symlink() or os.readlink(str(link)) != relative_target:
            raise RuntimeError("refusing to replace existing artifact path: {}".format(link))
        return
    link.symlink_to(relative_target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=5)
    parser.add_argument("--spr-session", default="dynamac_spr_full_20260821")
    parser.add_argument("--guardian-session", default="dynamac_guardian_full_20260821")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    require_resource_gate(args.gpu_index, [args.spr_session, args.guardian_session])
    free_bytes = shutil.disk_usage(str(repo_root)).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError("at least {} free bytes required; found {}".format(MIN_FREE_BYTES, free_bytes))

    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    previous = {}
    if args.lock_file.exists():
        with args.lock_file.open("r", encoding="utf-8") as handle:
            previous = json.load(handle).get("artifacts", {})

    locked = {}
    resolved = {}
    for name in ("robomimic_image", "transport_ph_dp_checkpoint"):
        record = manifest["artifacts"][name]
        remote = validate_remote(record)
        path, digest = fetch(repo_root, name, record, previous)
        locked[name] = {
            "url": record["url"],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "remote": remote,
        }
        resolved[name] = path

    dataset_record = manifest["artifacts"]["robomimic_image"]
    dataset, dataset_size, dataset_digest = extract_transport(
        repo_root, resolved["robomimic_image"], dataset_record, previous
    )
    locked["transport_image_abs"] = {
        "archive_member": dataset_record["archive_member"],
        "path": str(dataset),
        "bytes": dataset_size,
        "sha256": dataset_digest,
    }

    ensure_symlink(repo_root / dataset_record["upstream_link"], dataset)
    checkpoint_record = manifest["artifacts"]["transport_ph_dp_checkpoint"]
    ensure_symlink(repo_root / checkpoint_record["upstream_link"], resolved["transport_ph_dp_checkpoint"])

    payload = {
        "schema": "dynamac-fail-detect-quant-artifact-lock-v1",
        "protocol_label": manifest["protocol_label"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": locked,
    }
    atomic_json(args.lock_file, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("artifact preparation failed: {}".format(exc), file=sys.stderr)
        raise
