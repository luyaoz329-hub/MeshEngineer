"""Minimal STL syntax and exact numeric-coordinate indexing."""

import re
import struct

import numpy as np

from meshengineer.core.assets import Asset, AssetSource, MeshData, NodeData
from meshengineer.core.imports import AssetImportError, ImportErrorCode

_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def read_stl(raw: bytes, source: AssetSource) -> Asset:
    """Adapt one ordinary binary or complete ASCII STL."""
    def fail(message: str, code: ImportErrorCode = "invalid_asset") -> None:
        raise AssetImportError(source.ref, code, message, source.actual_sha256)

    if len(raw) >= 84 and 84 + 50 * struct.unpack_from("<I", raw, 80)[0] == len(raw):
        source.stl_header = raw[:80]
        if b"COLOR=" in raw[:80]:
            fail("Binary STL COLOR= marker is not supported", "unsupported_asset")
        binary_corners: list[list[float]] = []
        binary_normals: list[list[float]] = []
        for values in struct.iter_unpack("<12fH", raw[84:]):
            if values[12] != 0:
                fail("Nonzero binary STL facet attributes are not supported", "unsupported_asset")
            row = [float(value) for value in values[:12]]
            if not np.isfinite(row).all():
                fail("Binary STL contains non-finite coordinates or normals")
            binary_normals.append(row[:3])
            binary_corners.extend([row[3:6], row[6:9], row[9:12]])
        if not binary_normals:
            fail("STL contains no triangles", "unsupported_asset")
        return _build_asset(binary_corners, binary_normals, source)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        fail("STL is not valid ASCII or a supported binary STL")
        raise AssertionError("unreachable")
    lines = [(i + 1, line.strip()) for i, line in enumerate(text.splitlines()) if line.strip()]
    cursor = 0

    def take(prefix: tuple[str, ...], numbers: bool = False) -> list[float]:
        nonlocal cursor
        if cursor >= len(lines):
            fail(f"Unexpected end of STL; expected {' '.join(prefix)}")
        line_number, line = lines[cursor]
        cursor += 1
        tokens = line.split()
        if tuple(tokens[:len(prefix)]) != prefix or len(tokens) != len(prefix) + (3 if numbers else 0):
            fail(f"Invalid STL syntax at line {line_number}; expected {' '.join(prefix)}")
        values: list[float] = []
        if numbers:
            for token in tokens[len(prefix):]:
                if not _NUMBER.fullmatch(token):
                    fail(f"Invalid number at STL line {line_number}")
                values.append(float(token))
            if not np.isfinite(values).all():
                fail(f"Non-finite number at STL line {line_number}")
        return values

    if not lines or lines[0][1].split()[0] != "solid":
        fail("ASCII STL must start with solid")
    source.stl_header = lines[0][1].encode("ascii")
    cursor = 1
    corners: list[list[float]] = []
    normals: list[list[float]] = []
    while cursor < len(lines) and lines[cursor][1].split()[0] != "endsolid":
        normals.append(take(("facet", "normal"), True))
        take(("outer", "loop"))
        for _ in range(3):
            corners.append(take(("vertex",), True))
        take(("endloop",))
        take(("endfacet",))
    if cursor >= len(lines):
        fail("ASCII STL is missing endsolid")
    cursor += 1
    if cursor < len(lines):
        if lines[cursor][1].split()[0] == "solid":
            fail("Multiple STL solid blocks are not supported", "unsupported_asset")
        fail(f"Trailing data after endsolid at line {lines[cursor][0]}")
    if not normals:
        fail("STL contains no triangles", "unsupported_asset")
    return _build_asset(corners, normals, source)


def _build_asset(corners: list[list[float]], normals: list[list[float]], source: AssetSource) -> Asset:
    vertices: list[list[float]] = []
    index: dict[tuple[float, float, float], int] = {}
    flat_faces: list[int] = []
    for point in corners:
        key = (point[0], point[1], point[2])
        if key not in index:
            index[key] = len(vertices)
            vertices.append(point)
        flat_faces.append(index[key])
    mesh = MeshData(np.asarray(vertices, dtype=np.float64), np.asarray(flat_faces, dtype=np.int64).reshape(-1, 3),
                    face_normals=np.asarray(normals, dtype=np.float64))
    source.notes.append("STL vertices share indices only for exactly equal numeric coordinates; source topology is ambiguous.")
    return Asset(source, [mesh], [NodeData(None, [0], np.eye(4, dtype=np.float64))])
