#!/usr/bin/env python3
"""Create a consistent, read-only iMessage database snapshot on macOS.

This script intentionally uses only Python's standard library so the Mac does not need
PyTorch or the Windows project's dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote


def readonly_uri(path: Path) -> str:
    return f"file:{quote(path.resolve().as_posix(), safe='/:')}?mode=ro"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely export an iMessage SQLite snapshot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "imessage-snapshot",
        help="Private output folder (default: ~/Desktop/imessage-snapshot)",
    )
    args = parser.parse_args()
    source = (Path.home() / "Library" / "Messages" / "chat.db").resolve()
    output = args.output.expanduser().resolve()
    destination = output / "chat.db"
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    if not source.exists():
        raise SystemExit(f"Messages database not found: {source}")

    before = source.stat()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output, prefix=".chat-snapshot.", suffix=".db"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.chmod(0o600)
    try:
        try:
            with (
                sqlite3.connect(readonly_uri(source), uri=True, timeout=30) as source_db,
                sqlite3.connect(temporary) as snapshot_db,
            ):
                source_db.execute("PRAGMA query_only=ON")
                source_db.backup(snapshot_db, pages=2048)
                snapshot_db.commit()
        except sqlite3.Error as error:
            raise SystemExit(
                "Could not read Messages. Give Terminal Full Disk Access in System Settings > "
                "Privacy & Security, completely restart Terminal, and rerun this script."
            ) from error

        with sqlite3.connect(readonly_uri(temporary), uri=True) as check_db:
            quick_check = check_db.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise SystemExit(f"Snapshot integrity check failed: {quick_check}")
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise SystemExit(
                "Messages changed during export. Wait a moment and run the script again."
            )

        os.replace(temporary, destination)
        destination.chmod(0o600)
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "source_size": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "snapshot_size": destination.stat().st_size,
            "snapshot_sha256": sha256_file(destination),
            "sqlite_version": sqlite3.sqlite_version,
            "quick_check": quick_check,
            "source_open_mode": "read-only",
            "exporter": "texts-to-transformer-windows",
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o600)
        print(f"Snapshot ready: {output}")
        print("Transfer the entire folder privately to the Windows PC.")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
