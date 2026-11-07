"""Verify STL behavior through the public import boundary."""

from pathlib import Path
import unittest

import numpy as np

from meshengineer.core.assets import Asset, AssetRef, MeshData
from meshengineer.io.loader import load_asset
from meshengineer.core.imports import AssetImportError
from tests.base.helpers import ascii_stl, binary_stl, load_bytes

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class StlTests(unittest.TestCase):
    def test_s10_glb_and_both_stl_encodings_share_the_geometry_contract(self) -> None:
        expected=[[[0,0,0],[1,0,0],[1,1,0]],[[0,0,0],[1,1,0],[0,1,0]]]
        for name in ['simple.glb','ascii-square.stl','binary-square.stl']:
            with self.subTest(name=name):
                asset=load_asset(AssetRef('fixture',name,name),{'fixture':FIXTURES})
                self.assertIsInstance(asset,Asset);mesh=asset.meshes[0];self.assertIsInstance(mesh,MeshData)
                np.testing.assert_array_equal(mesh.vertices[mesh.faces],expected)
                self.assertEqual(mesh.vertices.dtype,np.dtype('float64'));self.assertEqual(mesh.faces.dtype,np.dtype('int64'))
                self.assertEqual(mesh.vertices.shape,(4,3));self.assertEqual(mesh.faces.shape,(2,3))
                self.assertIsNone(mesh.vertex_normals)
                if name.endswith('.stl'):np.testing.assert_array_equal(mesh.face_normals,[[0,0,1]]*2)
                else:self.assertIsNone(mesh.face_normals)

    def test_s02_asymmetric_binary_normals_header_and_float32_values(self) -> None:
        triangles=[[(.1,2,3),(4,5,6),(7,8,9)]];normals=[(.25,-.5,2)]
        asset=load_bytes(binary_stl(triangles,normals,header=b'source header'))
        mesh=asset.meshes[0]
        np.testing.assert_array_equal(mesh.vertices[mesh.faces],np.asarray(triangles,dtype=np.float32).astype(np.float64))
        np.testing.assert_array_equal(mesh.face_normals,normals)
        self.assertEqual(asset.source.stl_header,b'source header'.ljust(80,b'\0'))

    def test_s05_numeric_text_spelling_does_not_change_exact_sharing(self) -> None:
        triangles=[[(0,0,0),(1,0,0),(0,1,0)],[(1.0,0,0),(0,1,0),(0,0,0)]]
        raw=ascii_stl(triangles).replace(b'vertex 1.0 0 0',b'vertex 1e0 0 0')
        mesh=load_bytes(raw).meshes[0]
        self.assertEqual(len(mesh.vertices),3)
        np.testing.assert_array_equal(mesh.vertices[mesh.faces],triangles)

    def test_s02_s03_binary_geometry_and_solid_header(self) -> None:
        triangles=[[(0,0,0),(2,0,0),(0,3,0)]]
        for header in [b"binary",b"solid looks like text"]:
            mesh=load_bytes(binary_stl(triangles,header=header)).meshes[0]
            np.testing.assert_array_equal(mesh.vertices[mesh.faces],triangles)
            np.testing.assert_array_equal(mesh.face_normals,[[0,0,1]])

    def test_s08_binary_color_and_nonzero_attributes_are_unsupported(self) -> None:
        triangles=[[(0,0,0),(2,0,0),(0,3,0)]]
        for raw in [binary_stl(triangles,header=b"COLOR="),binary_stl(triangles,attribute=1),binary_stl(triangles,attribute=0x8000)]:
            with self.subTest(raw=raw),self.assertRaises(AssetImportError) as result: load_bytes(raw)
            self.assertEqual(result.exception.code,"unsupported_asset")

    def test_s09_binary_truncation_nonfinite_and_empty(self) -> None:
        raw=binary_stl([[(0,0,0),(2,0,0),(0,3,0)]])
        for broken in [raw[:-1],raw+b"x",binary_stl([[(float("nan"),0,0),(1,0,0),(0,1,0)]])]:
            with self.subTest(raw=broken),self.assertRaises(AssetImportError) as result: load_bytes(broken)
            self.assertEqual(result.exception.code,"invalid_asset")
        with self.assertRaises(AssetImportError) as result: load_bytes(binary_stl([]))
        self.assertEqual(result.exception.code,"unsupported_asset")

    def test_s01_ascii_square_preserves_corners_and_normals(self) -> None:
        asset = load_asset(AssetRef("fixture", "square", "ascii-square.stl", "380f6f494798cfe09fcf99bc940c41da7dffbd33a217019ed61cfb743b1b874e"), {"fixture": FIXTURES})
        mesh = asset.meshes[0]
        self.assertEqual(mesh.vertices.shape, (4, 3))
        self.assertEqual(mesh.vertices.dtype, np.dtype("float64"))
        self.assertEqual(mesh.faces.dtype, np.dtype("int64"))
        self.assertEqual(mesh.vertices[mesh.faces].tolist(), [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        ])
        np.testing.assert_array_equal(mesh.face_normals, [[0, 0, 1], [0, 0, 1]])
        self.assertIsNone(mesh.vertex_normals)
        np.testing.assert_array_equal(asset.nodes[0].local_transform, np.eye(4))

    def test_s04_high_precision_ascii_is_not_rounded_to_float32(self) -> None:
        raw = ascii_stl([[(1, 0, 0), (1.0000000000000002, 0, 0), (0, 1, 0)]])
        mesh = load_bytes(raw).meshes[0]
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(mesh.vertices[mesh.faces][0, 1, 0], 1.0000000000000002)

    def test_s05_only_numerically_identical_coordinates_share_indices(self) -> None:
        raw = ascii_stl([[(0,0,0),(1,0,0),(0,1,0)],[(0,0,0),(0,1,0),(1e-12,0,0)]])
        mesh = load_bytes(raw).meshes[0]
        self.assertEqual(len(mesh.vertices), 4)
        self.assertEqual(mesh.faces[0,0], mesh.faces[1,0])
        self.assertNotEqual(mesh.faces[0,0], mesh.faces[1,2])

    def test_s06_defects_and_original_normals_are_not_repaired(self) -> None:
        triangles = [[(0,0,0),(0,0,0),(1,0,0)],[(0,0,0),(0,0,0),(1,0,0)],[(8,0,0),(9,0,0),(8,1,0)]]
        normals = [(0,0,0),(0,0,2),(0,0,-1)]
        mesh = load_bytes(ascii_stl(triangles, normals)).meshes[0]
        np.testing.assert_array_equal(mesh.vertices[mesh.faces], triangles)
        np.testing.assert_array_equal(mesh.face_normals, normals)
        self.assertEqual(mesh.faces.shape, (3,3))

    def test_s01_whitespace_scientific_notation_and_uppercase_suffix(self) -> None:
        raw = ascii_stl([[(0,0,0),(1,0,0),(0,1,0)]]).replace(b"vertex 1 ", b"vertex 1e0 ")
        mesh = load_bytes(b"\n" + raw + b"\n", ".STL").meshes[0]
        self.assertEqual(mesh.vertices[mesh.faces][0,1].tolist(), [1,0,0])

    def test_s07_multiple_solid_blocks_are_rejected(self) -> None:
        raw = ascii_stl([[(0,0,0),(1,0,0),(0,1,0)]])
        with self.assertRaises(AssetImportError) as result:
            load_bytes(raw + raw)
        self.assertEqual(result.exception.code, "unsupported_asset")

    def test_s09_each_syntax_fault_is_rejected_independently(self) -> None:
        raw = ascii_stl([[(0,0,0),(1,0,0),(0,1,0)]])
        variants = [b"", raw.replace(b"facet normal 0 0 1",b"facet normal 0 1"),
                    raw.replace(b"vertex 1 0 0\n", b""), raw.replace(b"endloop", b"vertex 2 2 2\nendloop"),
                    raw.replace(b"endfacet\n", b""), raw.replace(b"endsolid sample", b""),raw+b"junk\n"]
        for value in variants:
            with self.subTest(raw=value), self.assertRaises(AssetImportError) as result:
                load_bytes(value)
            self.assertEqual(result.exception.code, "invalid_asset")

    def test_s09_nonfinite_vertices_and_normals_are_rejected(self) -> None:
        for value in [float("nan"),float("inf"),-float("inf"),"1e999"]:
            for raw in [ascii_stl([[(value,0,0),(1,0,0),(0,1,0)]]),ascii_stl([[(0,0,0),(1,0,0),(0,1,0)]],[(value,0,0)])]:
                with self.subTest(value=value), self.assertRaises(AssetImportError) as result:
                    load_bytes(raw)
                self.assertEqual(result.exception.code,"invalid_asset")

    def test_s09_structurally_complete_empty_solid_is_unsupported(self) -> None:
        with self.assertRaises(AssetImportError) as result:
            load_bytes(b"solid empty\nendsolid empty\n")
        self.assertEqual(result.exception.code,"unsupported_asset")
