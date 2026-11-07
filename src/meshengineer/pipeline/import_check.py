"""Single-asset import result construction shared by batch callers."""

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
import platform

from meshengineer.core.assets import AssetRef
from meshengineer.core.imports import AssetImportError, ImportRecord
from meshengineer.io.loader import load_asset


def environment_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("meshengineer", "numpy", "pygltflib"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def check_asset(ref: AssetRef, roots: Mapping[str, Path], environment: dict[str, str]) -> ImportRecord:
    path = str(roots[ref.source_id] / ref.relative_path)
    try:
        asset = load_asset(ref, roots)
    except AssetImportError as error:
        record = ImportRecord(ref, "failed", path, error.actual_sha256, code=error.code, message=error.message,
                              environment=environment)
        suffix = PurePosixPath(ref.relative_path).suffix.lower()
        if suffix == ".stl": record.format = "stl"
        elif suffix == ".glb": record.format = "glb"
        return record
    return ImportRecord(ref, "ok", str(asset.source.resolved_path), asset.source.actual_sha256, asset.source.format,
                        len(asset.meshes), sum(len(m.vertices) for m in asset.meshes),
                        sum(len(m.faces) for m in asset.meshes), sum(bool(n.mesh_indices) for n in asset.nodes),
                        asset.source.selected_scene_index, notes=list(asset.source.notes), environment=environment)
