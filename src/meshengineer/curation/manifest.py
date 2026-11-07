"""Strict whole-manifest preflight without loading geometry."""

from collections.abc import Mapping
from pathlib import Path

from meshengineer.core.assets import AssetRef
from meshengineer.io.loader import parse_json, validate_relative_path


def read_manifest(path: Path, roots: Mapping[str, Path]) -> list[AssetRef]:
    refs: list[AssetRef] = []
    identities: dict[tuple[str, str], int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        context = f"{path}:{line_number}"
        value = parse_json(line, context)
        if not isinstance(value, dict):
            raise ValueError(f"{context}: manifest row must be a JSON object")
        unknown = value.keys() - {"source_id", "asset_id", "relative_path", "expected_sha256"}
        if unknown:
            raise ValueError(f"{context}: unknown fields: {sorted(unknown)}")
        for name in ("source_id", "asset_id", "relative_path"):
            if not isinstance(value.get(name), str) or not value[name]:
                raise ValueError(f"{context}: {name} must be a nonempty string")
        source, asset, relative = value["source_id"], value["asset_id"], value["relative_path"]
        assert isinstance(source, str) and isinstance(asset, str) and isinstance(relative, str)
        expected = value.get("expected_sha256")
        if "expected_sha256" in value and not isinstance(expected, str):
            raise ValueError(f"{context}: expected_sha256 must be a hexadecimal string")
        assert expected is None or isinstance(expected, str)
        try:
            ref = AssetRef(source, asset, relative, expected)
            validate_relative_path(relative)
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error
        if source not in roots:
            raise ValueError(f"{context}: source_id is not configured: {source}")
        identity = (source, asset)
        if identity in identities:
            raise ValueError(f"{context}: duplicate asset identity {identity}; first seen at line {identities[identity]}")
        identities[identity] = line_number
        refs.append(ref)
    if not refs:
        raise ValueError(f"{path}: manifest must contain at least one asset")
    return refs
