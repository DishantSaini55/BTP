"""Safely probe and acquire approved IDD-PeD development video archives."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath


ALLOWED_SETS = {"0001", "0002", "0004", "0007"}
URL_TEMPLATE = "https://cvit.iiit.ac.in/images/datasets/IDDPed/Videos/gp_set_{set_id}.tar"
CHUNK_BYTES = 1024 * 1024
RETRIES = 5


class RedirectRecorder(urllib.request.HTTPRedirectHandler):
    def __init__(self) -> None:
        super().__init__()
        self.redirects: list[dict[str, object]] = []

    def redirect_request(self, request, response, code, message, headers, new_url):
        self.redirects.append({
            "from": request.full_url,
            "to": new_url,
            "status": code,
        })
        return super().redirect_request(request, response, code, message, headers, new_url)


def parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+)", value.strip(), re.IGNORECASE)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def probe_url(url: str, timeout: int = 20) -> dict[str, object]:
    redirects = RedirectRecorder()
    opener = urllib.request.build_opener(redirects)
    result: dict[str, object] = {
        "url": url,
        "head_status": None,
        "head_content_length_bytes": None,
        "range_status": None,
        "range_total_bytes": None,
        "range_response_bytes_read": 0,
        "range_supported": False,
        "accept_ranges": None,
        "etag": None,
        "last_modified": None,
        "redirects": redirects.redirects,
        "error": None,
    }

    try:
        request = urllib.request.Request(url, method="HEAD")
        with opener.open(request, timeout=timeout) as response:
            result["head_status"] = response.status
            length = response.headers.get("Content-Length")
            result["head_content_length_bytes"] = int(length) if length and length.isdigit() else None
            result["accept_ranges"] = response.headers.get("Accept-Ranges")
            result["etag"] = response.headers.get("ETag")
            result["last_modified"] = response.headers.get("Last-Modified")
    except urllib.error.HTTPError as error:
        result["head_status"] = error.code
        length = error.headers.get("Content-Length")
        result["head_content_length_bytes"] = int(length) if length and length.isdigit() else None
        result["accept_ranges"] = error.headers.get("Accept-Ranges")
        result["etag"] = error.headers.get("ETag")
        result["last_modified"] = error.headers.get("Last-Modified")
        error.close()
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        result["error"] = f"HEAD: {error}"

    try:
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with opener.open(request, timeout=timeout) as response:
            result["range_status"] = response.status
            result["range_response_bytes_read"] = len(response.read(1))
            parsed = parse_content_range(response.headers.get("Content-Range"))
            if parsed:
                start, end, total = parsed
                result["range_total_bytes"] = total
                result["range_supported"] = response.status == 206 and start == 0 and end == 0
            result["accept_ranges"] = response.headers.get("Accept-Ranges") or result["accept_ranges"]
            result["etag"] = response.headers.get("ETag") or result["etag"]
            result["last_modified"] = response.headers.get("Last-Modified") or result["last_modified"]
    except urllib.error.HTTPError as error:
        result["range_status"] = error.code
        result["accept_ranges"] = error.headers.get("Accept-Ranges") or result["accept_ranges"]
        error.close()
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        range_error = f"Range: {error}"
        result["error"] = "; ".join(filter(None, (result["error"], range_error)))

    result["redirects"] = list(redirects.redirects)
    return result


def expected_size(probe: dict[str, object]) -> int | None:
    range_size = probe.get("range_total_bytes")
    if isinstance(range_size, int) and range_size >= 0:
        return range_size
    head_size = probe.get("head_content_length_bytes")
    if isinstance(head_size, int) and head_size >= 0:
        return head_size
    return None


def require_d_drive(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"Dataset destinations must be on D:; got {resolved}")
    return resolved


def require_development_archive(path: Path) -> Path:
    resolved = require_d_drive(path)
    if resolved.name not in {f"gp_set_{set_id}.tar" for set_id in ALLOWED_SETS}:
        raise ValueError(f"Only approved development-set archives may be inspected/extracted: {resolved.name}")
    return resolved


def safe_member_name(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(name)
    if (
        not normalized
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return posix_path


def verify_tar_readable(archive_path: Path) -> list[dict[str, object]]:
    members_out: list[dict[str, object]] = []
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive.getmembers():
                safe_member_name(member.name)
                if member.isdir():
                    members_out.append({"name": member.name, "size": 0, "type": "directory"})
                    continue
                if not member.isfile():
                    raise ValueError(f"Unsupported non-regular tar member: {member.name}")
                stream = archive.extractfile(member)
                if stream is None:
                    raise tarfile.ReadError(f"Cannot read member: {member.name}")
                read_bytes = 0
                with stream:
                    while True:
                        chunk = stream.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        read_bytes += len(chunk)
                if read_bytes != member.size:
                    raise tarfile.ReadError(
                        f"Truncated member {member.name}: read {read_bytes} of {member.size} bytes"
                    )
                members_out.append({"name": member.name, "size": member.size, "type": "file"})
    except (tarfile.TarError, OSError) as error:
        raise tarfile.ReadError(f"Archive is unreadable: {error}") from error
    return members_out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_partial_state(
    state_path: Path,
    url: str,
    size: int | None,
    etag: str | None,
    last_modified: str | None,
) -> None:
    if not state_path.exists():
        raise FileExistsError(
            f"Found an orphan partial download without resume metadata: {state_path.parent}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("url") != url or state.get("expected_bytes") != size:
        raise ValueError("Partial download source or expected size changed; preserving partial file.")
    old_validator = state.get("etag") or state.get("last_modified")
    new_validator = etag or last_modified
    if not old_validator or old_validator != new_validator:
        raise ValueError("Cannot safely resume: remote validator is absent or changed.")


def _finish_download(
    partial: Path,
    state_path: Path,
    archive_path: Path,
    size: int | None,
    state: dict[str, object],
    elapsed_seconds: float,
    transferred_bytes: int,
) -> dict[str, object]:
    actual_size = partial.stat().st_size
    if size is not None and actual_size != size:
        raise ValueError(f"Size mismatch: received {actual_size} bytes, expected {size}.")

    print(f"Verifying tar readability: {partial}", flush=True)
    members = verify_tar_readable(partial)
    digest = sha256_file(partial)
    average_speed = transferred_bytes / elapsed_seconds if elapsed_seconds > 0 else None
    state.update({
        "status": "completed",
        "bytes": actual_size,
        "sha256": digest,
        "trusted_checksum_match": False,
        "archive_members": len(members),
        "elapsed_seconds": elapsed_seconds,
        "average_bytes_per_second": average_speed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    metadata_path = archive_path.with_name(archive_path.name + ".metadata.json")
    if archive_path.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing archive or its metadata.")
    atomic_json(state_path, state)
    partial.rename(archive_path)
    state_path.rename(metadata_path)
    print(
        f"COMPLETED {archive_path} bytes={actual_size} sha256={digest} "
        f"elapsed={elapsed_seconds:.1f}s "
        f"average_speed={(average_speed or 0) / (1024 * 1024):.2f} MiB/s "
        "trusted_checksum_match=false",
        flush=True,
    )
    return state


def download_archive(
    set_id: str,
    destination: Path,
    max_bytes: int,
    allow_unknown_size: bool,
    execute: bool,
    timeout: int = 60,
    url: str | None = None,
) -> dict[str, object]:
    if set_id not in ALLOWED_SETS:
        raise ValueError(f"Only approved development sets may be downloaded: {set_id}")
    if max_bytes <= 0:
        raise ValueError("--max-bytes must be a positive integer.")
    url = url or URL_TEMPLATE.format(set_id=set_id)
    probe = probe_url(url, timeout=min(timeout, 20))
    size = expected_size(probe)
    result: dict[str, object] = {
        "set_id": set_id,
        "url": url,
        "probe": probe,
        "status": "blocked",
    }
    if not execute:
        result["reason"] = "Dry run only. Add --execute after explicit approval."
        return result
    if probe.get("error") and probe.get("range_status") is None:
        result["reason"] = f"Remote probe failed: {probe['error']}"
        return result
    if not probe.get("range_supported"):
        result["reason"] = "Server did not confirm byte-range support; safe resume is unavailable."
        return result
    if not (probe.get("etag") or probe.get("last_modified")):
        result["reason"] = "No stable ETag/Last-Modified validator; safe resume cannot be guaranteed."
        return result
    if size is None and not allow_unknown_size:
        result["reason"] = "Remote size is unknown; pass --allow-unknown-size with a hard --max-bytes cap."
        return result
    if size is not None and size > max_bytes:
        result["reason"] = f"Remote size {size} exceeds --max-bytes={max_bytes}."
        return result

    destination = require_d_drive(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"gp_set_{set_id}.tar"
    partial = destination / f"gp_set_{set_id}.tar.partial"
    state_path = destination / f"gp_set_{set_id}.tar.partial.json"
    metadata_path = destination / f"gp_set_{set_id}.tar.metadata.json"
    if archive_path.exists() or metadata_path.exists():
        result["reason"] = f"Refusing to overwrite existing archive/metadata for GP{set_id}."
        result["status"] = "already_exists"
        return result
    if partial.exists():
        _validate_partial_state(
            state_path, url, size,
            str(probe.get("etag")) if probe.get("etag") else None,
            str(probe.get("last_modified")) if probe.get("last_modified") else None,
        )
    elif state_path.exists():
        raise FileExistsError(f"Resume metadata exists without a partial file: {state_path}")

    current_size = partial.stat().st_size if partial.exists() else 0
    limit = size if size is not None else max_bytes
    if current_size > limit:
        raise ValueError(f"Partial file {current_size} exceeds permitted limit {limit}.")
    free_bytes = shutil.disk_usage(destination).free
    if free_bytes < limit - current_size:
        result["reason"] = f"Insufficient D: space: need {limit-current_size}, have {free_bytes}."
        result["status"] = "insufficient_space"
        return result

    validator = probe.get("etag") or probe.get("last_modified")
    state: dict[str, object] = {
        "status": "running",
        "set_id": set_id,
        "url": url,
        "expected_bytes": size,
        "size_source": "Content-Range" if probe.get("range_total_bytes") is not None else "Content-Length",
        "etag": probe.get("etag"),
        "last_modified": probe.get("last_modified"),
        "validator": validator,
        "trusted_checksum_match": False,
    }
    if not partial.exists():
        atomic_json(state_path, state)

    started = time.monotonic()
    attempts = 0
    while attempts < RETRIES:
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={offset}-",
                "If-Range": str(validator),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise ValueError(
                        f"Expected HTTP 206 for byte range at {offset}; received {response.status}."
                    )
                content_range = parse_content_range(response.headers.get("Content-Range"))
                if not content_range or content_range[0] != offset:
                    raise ValueError(f"Invalid Content-Range for resume offset {offset}.")
                if size is not None and content_range[2] != size:
                    raise ValueError("Remote size changed during transfer; partial file preserved.")
                response_etag = response.headers.get("ETag")
                response_modified = response.headers.get("Last-Modified")
                if (response_etag or response_modified) != validator:
                    raise ValueError("Remote validator changed during transfer; partial file preserved.")
                mode = "ab" if offset else "xb"
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while True:
                        block = response.read(CHUNK_BYTES)
                        if not block:
                            break
                        new_size = output.tell() + len(block)
                        if new_size > max_bytes or (size is not None and new_size > size):
                            raise ValueError("Transfer exceeded its configured size limit.")
                        output.write(block)
                        output.flush()
                        now = time.monotonic()
                        if now - last_report >= 1.0:
                            elapsed = max(now - started, 1e-9)
                            speed = (new_size - current_size) / elapsed / (1024 * 1024)
                            progress = (
                                f"{new_size}/{size} bytes ({min(100.0, new_size * 100.0 / size):.1f}%)"
                                if size is not None
                                else f"{new_size} bytes (cap {max_bytes})"
                            )
                            print(
                                f"\rGP{set_id}: {progress}, {speed:.2f} MiB/s, {elapsed:.0f}s",
                                end="",
                                flush=True,
                            )
                            last_report = now
                actual_size = partial.stat().st_size
                state["bytes"] = actual_size
                atomic_json(state_path, state)
                if size is not None and actual_size != size:
                    raise http.client.IncompleteRead(b"", size - actual_size)
                print("", flush=True)
                elapsed = max(time.monotonic() - started, 1e-9)
                return _finish_download(
                    partial,
                    state_path,
                    archive_path,
                    size,
                    state,
                    elapsed,
                    max(0, actual_size - current_size),
                )
        except (urllib.error.URLError, TimeoutError, socket.timeout, http.client.HTTPException) as error:
            attempts += 1
            state["bytes"] = partial.stat().st_size if partial.exists() else 0
            state["last_error"] = str(error)
            atomic_json(state_path, state)
            if attempts >= RETRIES:
                result["reason"] = f"FAILED after {attempts} attempts; partial retained: {error}"
                result["status"] = "failed_partial"
                return result
            delay = min(2 ** (attempts - 1), 30)
            print(f"GP{set_id}: retry {attempts}/{RETRIES} in {delay}s: {error}", flush=True)
            time.sleep(delay)
    result["reason"] = "Download did not complete."
    result["status"] = "failed_partial"
    return result


def inspect_archive(archive_path: Path) -> dict[str, object]:
    archive_path = require_development_archive(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    members = verify_tar_readable(archive_path)
    return {
        "path": str(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
        "trusted_checksum_match": False,
        "members": members,
    }


def extract_members(archive_path: Path, member_names: list[str], destination: Path) -> list[str]:
    archive_path = require_development_archive(archive_path)
    destination = require_d_drive(destination)
    members = verify_tar_readable(archive_path)
    by_name = {str(member["name"]): member for member in members}
    selected: list[str] = []
    for name in member_names:
        safe_member_name(name)
        if name not in by_name or by_name[name]["type"] != "file":
            raise ValueError(f"Selected member is missing or not a regular file: {name}")
        if not name.lower().endswith(".mp4"):
            raise ValueError(f"Selective extraction is limited to MP4 members: {name}")
        if PurePosixPath(name.replace("\\", "/")).parts[0] != archive_path.stem:
            raise ValueError(f"Selected member is outside the archive's set directory: {name}")
        selected.append(name)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate archive member requested.")

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(archive_path, mode="r:") as archive:
        for name in selected:
            relative = safe_member_name(name)
            target = destination.joinpath(*relative.parts)
            if target.exists() or target.with_name(target.name + ".partial").exists():
                raise FileExistsError(f"Refusing to overwrite extraction target: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            current = target.parent
            while current != destination.parent:
                if current.is_symlink():
                    raise ValueError(f"Symlink encountered in extraction path: {current}")
                if current == destination:
                    break
                current = current.parent
            member = archive.getmember(name)
            source = archive.extractfile(member)
            if source is None:
                raise tarfile.ReadError(f"Cannot read selected member: {name}")
            temporary = target.with_name(target.name + ".partial")
            with source, temporary.open("xb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_BYTES)
                output.flush()
                os.fsync(output.fileno())
            if temporary.stat().st_size != member.size:
                raise tarfile.ReadError(f"Extracted size mismatch for {name}")
            temporary.rename(target)
            extracted.append(str(target))
    return extracted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="HEAD and one-byte range probe only")
    probe_parser.add_argument("--sets", nargs="+", choices=sorted(ALLOWED_SETS), required=True)
    probe_parser.add_argument("--timeout", type=int, default=20)

    download_parser = subparsers.add_parser("download", help="Download only with explicit --execute")
    download_parser.add_argument("--sets", nargs="+", choices=sorted(ALLOWED_SETS), required=True)
    download_parser.add_argument(
        "--destination",
        type=Path,
        default=Path(r"D:\IDD-PeD-Evaluation\archives\phase5a"),
    )
    download_parser.add_argument("--max-bytes", type=int, required=True)
    download_parser.add_argument("--allow-unknown-size", action="store_true")
    download_parser.add_argument("--execute", action="store_true")
    download_parser.add_argument("--timeout", type=int, default=60)

    inspect_parser = subparsers.add_parser("inspect", help="Read and hash a downloaded tar without extracting")
    inspect_parser.add_argument("--archive", type=Path, required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract explicitly selected MP4 members only")
    extract_parser.add_argument("--archive", type=Path, required=True)
    extract_parser.add_argument("--members", nargs="+", required=True)
    extract_parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "probe":
            results = {
                set_id: probe_url(URL_TEMPLATE.format(set_id=set_id), timeout=args.timeout)
                for set_id in args.sets
            }
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0 if all(value["range_status"] == 206 for value in results.values()) else 1
        if args.command == "download":
            results = [
                download_archive(
                    set_id, args.destination, args.max_bytes,
                    args.allow_unknown_size, args.execute, timeout=args.timeout,
                )
                for set_id in args.sets
            ]
            print(json.dumps(results, indent=2, sort_keys=True))
            return 0 if all(item.get("status") == "completed" for item in results) else 1
        if args.command == "inspect":
            print(json.dumps(inspect_archive(args.archive), indent=2, sort_keys=True))
            return 0
        if args.command == "extract":
            print(json.dumps(
                {"extracted": extract_members(args.archive, args.members, args.destination)},
                indent=2,
                sort_keys=True,
            ))
            return 0
    except (OSError, ValueError, tarfile.TarError, urllib.error.URLError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
