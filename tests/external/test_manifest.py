"""Explicit opt-in data regression; selected missing inputs are errors."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from meshengineer.core.imports import AssetImportError
from meshengineer.curation.manifest import read_manifest
from meshengineer.io.loader import load_asset, read_storage


class ExternalManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keys=['MESHENGINEER_TEST_MANIFEST','MESHENGINEER_TEST_STORAGE','MESHENGINEER_TEST_EXPECTATIONS']
        missing=[key for key in keys if not os.environ.get(key)]
        if missing: raise RuntimeError('Selected external suite requires '+', '.join(missing))
        cls.manifest=Path(os.environ[keys[0]]);cls.storage=Path(os.environ[keys[1]])
        cls.roots=read_storage(cls.storage);cls.refs=read_manifest(cls.manifest,cls.roots)
        rows=json.loads(Path(os.environ[keys[2]]).read_text())['assets']
        cls.expected={(r['source_id'],r['asset_id']):r for r in rows}
        if len(rows)!=len(cls.expected) or set(cls.expected)!={(r.source_id,r.asset_id) for r in cls.refs}:
            raise RuntimeError('Expectation identities and selected manifest do not match exactly')
        if any(r.expected_sha256 is None for r in cls.refs): raise RuntimeError('Fixed data requires expected_sha256')
        cls.artifacts=Path(os.environ['MESHENGINEER_TEST_ARTIFACT_DIR']) if os.environ.get('MESHENGINEER_TEST_ARTIFACT_DIR') else None
        if cls.artifacts is not None:
            cls.artifacts.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(cls.manifest,cls.artifacts/'selected-manifest.jsonl')
            shutil.copyfile(Path(os.environ[keys[2]]),cls.artifacts/'selected-expectations.json')

    def test_x01_x02_selected_assets_against_independent_source_values(self):
        executed=imported=rejected=0
        for ref in self.refs:
            executed+=1
            with self.subTest(source=ref.source_id,asset=ref.asset_id):
                truth=self.expected[(ref.source_id,ref.asset_id)]
                if truth['status']=='failed':
                    with self.assertRaises(AssetImportError) as result: load_asset(ref,self.roots)
                    self.assertEqual(result.exception.code,truth['code'])
                else:
                    asset=load_asset(ref,self.roots)
                    self.assertEqual(asset.source.actual_sha256,ref.expected_sha256)
                    self.assertEqual(asset.source.format,truth['format'])
                    actual={'mesh_count':len(asset.meshes),'vertex_count':sum(len(m.vertices) for m in asset.meshes),
                            'face_count':sum(len(m.faces) for m in asset.meshes),'instance_count':sum(bool(n.mesh_indices) for n in asset.nodes),
                            'material_count':len(asset.materials),'selected_scene_index':asset.source.selected_scene_index}
                    for key,value in actual.items():
                        if key in truth:self.assertEqual(value,truth[key],key)
                    mesh=asset.meshes[0]
                    if 'primitive_materials' in truth:
                        self.assertEqual([m.material_index for m in asset.meshes],truth['primitive_materials'])
                        self.assertEqual([hashlib.sha256(i.data).hexdigest() for i in asset.images],truth['image_sha256'])
                    np.testing.assert_array_equal(mesh.vertices[mesh.faces[0]],truth['first_triangle'])
                    if 'first_normal' in truth:np.testing.assert_array_equal(mesh.face_normals[0],truth['first_normal'])
                    for mesh in asset.meshes:
                        self.assertEqual(mesh.vertices.dtype,np.dtype('float64'))
                        self.assertEqual(mesh.faces.dtype,np.dtype('int64'))
                path=self.roots[ref.source_id]/ref.relative_path
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),ref.expected_sha256,'Source changed')
                if truth['status']=='failed': rejected+=1
                else: imported+=1
        result={'selected_assets':len(self.refs),'executed_assets':executed,'imported':imported,
                'correctly_rejected':rejected,'unexpected_failures':executed-imported-rejected,
                'not_executed':len(self.refs)-executed,'manifest_sha256':hashlib.sha256(self.manifest.read_bytes()).hexdigest()}
        print(json.dumps(result))
        if self.artifacts is not None:
            (self.artifacts/'assets-summary.json').write_text(json.dumps(result,indent=2)+'\n')
        self.assertEqual(imported+rejected,len(self.refs))

    def test_x04_same_frozen_refs_use_a_different_root_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            links={}
            for i,(source,root) in enumerate(self.roots.items()):
                link=Path(tmp)/str(i);link.symlink_to(root,target_is_directory=True);links[source]=link
            for ref in self.refs:
                with self.subTest(asset=ref.asset_id):
                    truth=self.expected[(ref.source_id,ref.asset_id)]
                    if truth['status']=='failed':
                        with self.assertRaises(AssetImportError) as result:load_asset(ref,links)
                        self.assertEqual(result.exception.code,truth['code'])
                    else:
                        asset=load_asset(ref,links)
                        self.assertEqual(asset.source.ref,ref);self.assertEqual(asset.source.actual_sha256,ref.expected_sha256)
                        np.testing.assert_array_equal(asset.meshes[0].vertices[asset.meshes[0].faces[0]],truth['first_triangle'])

    def test_x03_x04_cli_completes_exactly_the_selected_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=(self.artifacts if self.artifacts is not None else Path(tmp))/'import.jsonl'
            result=subprocess.run([sys.executable,'-m','meshengineer.cli.main','import-check','--manifest',str(self.manifest),
                                   '--storage',str(self.storage),'--report',str(report)],capture_output=True,text=True)
            failed=sum(self.expected[(r.source_id,r.asset_id)]['status']=='failed' for r in self.refs)
            self.assertEqual(result.returncode,1 if failed else 0,result.stderr)
            summary=json.loads(result.stdout)
            if self.artifacts is not None:
                (self.artifacts/'batch-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            self.assertTrue(summary['completed']);self.assertEqual(summary['processed'],len(self.refs))
            self.assertEqual(summary['reported'],len(self.refs));self.assertEqual(summary['failed'],failed)
            records=[json.loads(line) for line in report.read_text().splitlines()]
            self.assertEqual([(r['ref']['source_id'],r['ref']['asset_id']) for r in records],[(r.source_id,r.asset_id) for r in self.refs])
            for record in records:
                truth=self.expected[(record['ref']['source_id'],record['ref']['asset_id'])]
                self.assertEqual(record['status'],truth['status'])
