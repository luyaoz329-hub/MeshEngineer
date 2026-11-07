"""Read and hash the same file bytes before adapting an asset."""

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import cast

from meshengineer.core.assets import Asset, AssetRef, AssetSource, JsonValue
from meshengineer.core.imports import AssetImportError
from meshengineer.io.stl import read_stl


def parse_json(text: str, context: str) -> JsonValue:
    """Reject duplicate keys and non-JSON constants at a JSON boundary."""
    def pairs(items: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{context}: duplicate field {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"{context}: invalid JSON constant {value}")

    try:
        return cast(JsonValue, json.loads(text, object_pairs_hook=pairs, parse_constant=constant))
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: {error.msg}") from error


def read_storage(path: Path) -> dict[str, Path]:
    """Read explicit source roots; fail before processing any assets."""
    value = parse_json(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(value, dict):
        raise ValueError("Storage configuration must be a JSON object")
    roots: dict[str, Path] = {}
    for key, directory in value.items():
        if not key or not isinstance(directory, str):
            raise ValueError(f"Invalid storage root for source {key}")
        root = Path(directory)
        if not root.is_absolute() or not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
            raise ValueError(f"Storage root is not an accessible absolute directory: {root}")
        roots[key] = root
    return roots


def validate_relative_path(value: str) -> None:
    """Validate portable manifest path syntax without accessing files."""
    if not value or "\\" in value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        raise ValueError("relative_path must be a relative '/' path without '..' or backslashes")


def load_asset(ref: AssetRef, roots: Mapping[str, Path]) -> Asset:
    """Load one asset without altering its input or repairing geometry."""
    if ref.source_id not in roots:
        raise ValueError(f"Source is not configured: {ref.source_id}")
    root = roots[ref.source_id]
    if not root.is_absolute():
        raise ValueError(f"Storage root must be absolute: {root}")
    try:
        validate_relative_path(ref.relative_path)
    except ValueError as error:
        raise AssetImportError(ref, "invalid_path", str(error)) from error
    try:
        resolved_root = root.resolve(strict=True)
        path = (resolved_root / ref.relative_path).resolve(strict=True)
        if not path.is_relative_to(resolved_root):
            raise AssetImportError(ref, "invalid_path", "Resolved asset escapes its storage root")
        if not path.is_file():
            raise AssetImportError(ref, "invalid_path", "Asset must be an ordinary file")
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AssetImportError(ref, "invalid_path", "Opened asset is not an ordinary file")
            raw = stream.read()
            after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != after.st_size:
            raise AssetImportError(ref, "read_error", "Asset changed while being read")
    except OSError as error:
        raise AssetImportError(ref, "read_error", f"Cannot read {ref.relative_path}: {error}") from error
    actual = hashlib.sha256(raw).hexdigest()
    if ref.expected_sha256 is not None and actual != ref.expected_sha256:
        raise AssetImportError(ref, "hash_mismatch", f"SHA-256 mismatch: expected {ref.expected_sha256}, actual {actual}", actual)
    suffix = PurePosixPath(ref.relative_path).suffix.lower()
    if suffix == ".stl":
        return read_stl(raw, AssetSource(ref, "stl", path, actual))
    if suffix == ".glb":
        from meshengineer.io.glb import read_glb
        return read_glb(raw, AssetSource(ref, "glb", path, actual))
    raise AssetImportError(ref, "unsupported_asset", f"Unsupported asset suffix: {suffix or '<none>'}", actual)
