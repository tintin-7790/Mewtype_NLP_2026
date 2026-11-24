import json
import os
import shutil
import tempfile
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def backup_existing_file(path: Path, run_id: str, timestamp: str) -> Path | None:
    """Copy an existing development artifact before a profile-driven rerun."""
    if not path.is_file():
        return None
    base = path.with_name(f"{path.name}.backup.{timestamp}.{run_id[:8]}")
    candidate = base
    index = 1
    while candidate.exists():
        candidate = path.with_name(f"{base.name}.{index}")
        index += 1
    shutil.copy2(path, candidate)
    return candidate
