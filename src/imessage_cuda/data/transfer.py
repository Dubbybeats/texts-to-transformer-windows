from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from imessage_cuda.data.snapshot import readonly_uri
from imessage_cuda.utils import ensure_private_dir, sha256_file, write_json


def import_snapshot(
    database: str | Path,
    destination: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(database).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination_path:
        raise ValueError("Imported snapshot source and destination must differ")

    expected_hash = None
    source_manifest: dict[str, Any] = {}
    if manifest_path is not None:
        manifest_file = Path(manifest_path).expanduser().resolve()
        source_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        expected_hash = source_manifest.get("snapshot_sha256")
        if not expected_hash:
            raise ValueError("The supplied manifest has no snapshot_sha256 value")

    actual_hash = sha256_file(source)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("Snapshot SHA-256 does not match the Mac export manifest")

    with sqlite3.connect(readonly_uri(source), uri=True) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
    if quick_check != "ok":
        raise ValueError(f"Snapshot SQLite integrity check failed: {quick_check}")

    destination_dir = ensure_private_dir(destination_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_dir, prefix=".imported-chat.", suffix=".db"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != actual_hash:
            raise OSError("Snapshot changed while being copied")
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        **source_manifest,
        "snapshot_sha256": actual_hash,
        "snapshot_size": destination_path.stat().st_size,
        "quick_check": quick_check,
        "import_verified": True,
        "import_source_name": source.name,
    }
    write_json(destination_dir / "manifest.json", manifest)
    return manifest
