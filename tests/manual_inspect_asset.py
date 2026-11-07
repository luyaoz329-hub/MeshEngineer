"""Inspect an asset through the installed public Python API."""

import argparse
from pathlib import Path

from meshengineer.core.assets import AssetRef
from meshengineer.io.loader import load_asset


def inspect_one(file_path: Path) -> None:
    path = file_path.resolve()
    asset = load_asset(AssetRef("manual", path.stem, path.name), {"manual": path.parent})
    print("source:", asset.source.format, asset.source.actual_sha256)
    print("meshes:", len(asset.meshes), "nodes:", len(asset.nodes))
    for index, mesh in enumerate(asset.meshes):
        print("mesh:", index, "vertices:", mesh.vertices.shape, mesh.vertices.dtype,
              "faces:", mesh.faces.shape, mesh.faces.dtype)
        print("first_triangle:", mesh.vertices[mesh.faces[0]].tolist())
        print("vertex_normals:", None if mesh.vertex_normals is None else mesh.vertex_normals.shape,
              "face_normals:", None if mesh.face_normals is None else mesh.face_normals.shape,
              "uv_sets:", sorted(mesh.uv_sets))
    for note in asset.source.notes:
        print("note:", note)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    inspect_one(parser.parse_args().path)
