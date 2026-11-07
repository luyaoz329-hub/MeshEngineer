"""Exercise source protection, file boundaries and reference contracts."""

from contextlib import contextmanager
import gc
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from meshengineer.core.assets import AssetRef
from meshengineer.core.imports import AssetImportError
from meshengineer.io.loader import load_asset
from tests.base.helpers import ascii_stl

RAW = ascii_stl([[(0,0,0),(2,0,0),(0,3,0)]])


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "mesh.stl"
        self.path.write_bytes(RAW)
        self.ref = AssetRef("local","mesh","mesh.stl")
        self.roots = {"local":self.root}

    def test_p01_invalid_paths_are_rejected_without_open(self) -> None:
        for value in ["/tmp/x.stl","../x.stl","x/../mesh.stl","x\\mesh.stl"]:
            with self.subTest(path=value), patch.object(Path,"open") as opened, self.assertRaises(AssetImportError) as result:
                load_asset(AssetRef("local","x",value), self.roots)
            self.assertEqual(result.exception.code,"invalid_path")
            opened.assert_not_called()

    def test_p02_links_inside_root_and_linked_roots_work(self) -> None:
        (self.root/"link.stl").symlink_to(self.path)
        asset = load_asset(AssetRef("local","x","link.stl"),self.roots)
        self.assertEqual(asset.meshes[0].faces.shape,(1,3))
        linked = self.root/"root-link"
        linked.symlink_to(self.root,target_is_directory=True)
        self.assertEqual(load_asset(self.ref,{"local":linked}).meshes[0].faces.shape,(1,3))

    def test_p03_link_escape_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as outside:
            target=Path(outside)/"target.stl"; target.write_bytes(RAW)
            (self.root/"escape.stl").symlink_to(target)
            with patch.object(Path,"open") as opened, self.assertRaises(AssetImportError) as result:
                load_asset(AssetRef("local","escape","escape.stl"),self.roots)
            self.assertEqual(result.exception.code,"invalid_path")
            opened.assert_not_called()

    def test_p04_missing_unreadable_and_nonregular_targets(self) -> None:
        with self.assertRaises(AssetImportError) as result:
            load_asset(AssetRef("local","x","missing.stl"),self.roots)
        self.assertEqual(result.exception.code,"read_error")
        with patch.object(Path,"open",side_effect=PermissionError("denied")), self.assertRaises(AssetImportError) as result:
            load_asset(self.ref,self.roots)
        self.assertEqual(result.exception.code,"read_error")
        with self.assertRaises(AssetImportError) as result:
            load_asset(AssetRef("local","dir","."),self.roots)
        self.assertEqual(result.exception.code,"invalid_path")

    def test_p05_unicode_paths_and_moving_roots_preserve_content(self) -> None:
        other=self.root/"目录 space";other.mkdir();(other/"mesh.stl").write_bytes(RAW)
        asset=load_asset(self.ref,{"local":other})
        self.assertEqual(asset.source.actual_sha256,hashlib.sha256(RAW).hexdigest())

    def test_p06_matching_optional_and_uppercase_hash(self) -> None:
        actual=hashlib.sha256(RAW).hexdigest()
        for expected in [None,actual,actual.upper()]:
            asset=load_asset(AssetRef("local","x","mesh.stl",expected),self.roots)
            self.assertEqual(asset.source.actual_sha256,actual)
            self.assertEqual(self.path.read_bytes(),RAW)

    def test_p06_mismatch_stops_before_parsing_and_records_both_hashes(self) -> None:
        self.path.write_bytes(b"not an STL")
        with self.assertRaises(AssetImportError) as result:
            load_asset(AssetRef("local","x","mesh.stl","0"*64),self.roots)
        self.assertEqual(result.exception.code,"hash_mismatch")
        self.assertEqual(result.exception.actual_sha256,hashlib.sha256(b"not an STL").hexdigest())
        self.assertIn("0"*64,result.exception.message)

    def test_b07_parser_and_hash_use_the_same_bytes_without_reopening(self) -> None:
        real_open=Path.open; reads=[]
        @contextmanager
        def opening(path,*args,**kwargs):
            with real_open(path,*args,**kwargs) as stream:
                reads.append(str(path));yield stream
            with real_open(path,"wb") as stream:
                stream.write(ascii_stl([[(0,0,0),(99,0,0),(0,1,0)]]))
        with patch.object(Path,"open",opening):
            asset=load_asset(self.ref,self.roots)
        self.assertEqual(len(reads),1)
        self.assertEqual(asset.source.actual_sha256,hashlib.sha256(RAW).hexdigest())
        self.assertEqual(asset.meshes[0].vertices[asset.meshes[0].faces][0,1].tolist(),[2,0,0])

    def test_b08_changed_stat_is_a_read_error(self) -> None:
        fstat=os.fstat;calls=[]
        def changed(fd):
            value=fstat(fd);calls.append(fd)
            return SimpleNamespace(st_size=value.st_size,st_mtime_ns=value.st_mtime_ns+len(calls),st_mode=value.st_mode)
        with patch("os.fstat",side_effect=changed),self.assertRaises(AssetImportError) as result:
            load_asset(self.ref,self.roots)
        self.assertEqual(result.exception.code,"read_error")

    def test_b10_returned_arrays_outlive_the_source_file(self) -> None:
        asset=load_asset(self.ref,self.roots);self.path.unlink();gc.collect()
        self.assertEqual(asset.meshes[0].vertices[asset.meshes[0].faces][0,2].tolist(),[0,3,0])

    def test_u02_unsupported_suffix_and_unknown_program_errors(self) -> None:
        (self.root/"mesh.obj").write_bytes(RAW)
        with self.assertRaises(AssetImportError) as result:
            load_asset(AssetRef("local","x","mesh.obj"),self.roots)
        self.assertEqual(result.exception.code,"unsupported_asset")
        with patch.object(Path,"open",side_effect=RuntimeError("program defect")),self.assertRaisesRegex(RuntimeError,"program defect"):
            load_asset(self.ref,self.roots)

    def test_configuration_errors_and_reference_values_are_explicit(self) -> None:
        with self.assertRaises(ValueError): load_asset(self.ref,{})
        with self.assertRaises(ValueError): load_asset(self.ref,{"local":Path("relative")})
        for expected in ["", "abc", "g"*64]:
            with self.subTest(expected=expected), self.assertRaises(ValueError): AssetRef("local","x","mesh.stl",expected)
