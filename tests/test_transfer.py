import json
import sqlite3
from pathlib import Path

import pytest

from imessage_cuda.data.transfer import import_snapshot
from imessage_cuda.utils import sha256_file


def test_import_snapshot_verifies_hash_and_integrity(tmp_path: Path) -> None:
    source = tmp_path / "export" / "chat.db"
    source.parent.mkdir()
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE message (text TEXT)")
        connection.execute("INSERT INTO message VALUES ('synthetic only')")
    manifest = source.parent / "manifest.json"
    manifest.write_text(
        json.dumps({"snapshot_sha256": sha256_file(source), "source_open_mode": "read-only"}),
        encoding="utf-8",
    )

    destination = tmp_path / "work" / "snapshot" / "chat.db"
    result = import_snapshot(source, destination, manifest)

    assert result["import_verified"] is True
    assert sha256_file(destination) == sha256_file(source)
    assert result["quick_check"] == "ok"


def test_import_snapshot_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "chat.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE message (text TEXT)")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"snapshot_sha256": "0" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        import_snapshot(source, tmp_path / "destination" / "chat.db", manifest)
