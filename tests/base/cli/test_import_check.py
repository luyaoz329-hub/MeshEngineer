"""CLI behavior, preflight and report protection."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import os
import subprocess
import sys
from unittest.mock import patch

from meshengineer.cli.main import main
from tests.base.helpers import ascii_stl, material_glb, compression_glb, pack_glb


class ImportCheckTests(unittest.TestCase):
    def test_b08_changed_file_is_one_failure_then_next_input_continues(self):
        original=os.fstat;calls=[]
        def fstat(fd):
            value=original(fd);calls.append(fd)
            return SimpleNamespace(st_mode=value.st_mode,st_size=value.st_size,
                                   st_mtime_ns=value.st_mtime_ns+(1 if len(calls)==2 else 0))
        with patch('os.fstat',fstat):
            code,out,_=self.run_cli([self.good_row(),self.good_row(asset_id='next')])
        rows=[json.loads(line) for line in self.report.read_text().splitlines()]
        self.assertEqual(code,1);self.assertTrue(json.loads(out)['completed'])
        self.assertEqual([r['status'] for r in rows],['failed','ok']);self.assertEqual(rows[0]['code'],'read_error')

    def test_e04_selected_missing_asset_or_bad_hash_is_not_a_green_suite(self):
        truth=self.root/'expected.json';truth.write_text(json.dumps({'assets':[{'source_id':'local','asset_id':'one','status':'ok'}]}))
        for path,digest in [('missing.stl','0'*64),('a.stl','0'*64)]:
            self.manifest.write_text(json.dumps(self.good_row(relative_path=path,expected_sha256=digest))+'\n')
            environment=dict(os.environ)
            environment.pop('MESHENGINEER_TEST_ARTIFACT_DIR',None)
            environment.update(MESHENGINEER_TEST_MANIFEST=str(self.manifest),MESHENGINEER_TEST_STORAGE=str(self.storage),MESHENGINEER_TEST_EXPECTATIONS=str(truth))
            result=subprocess.run([sys.executable,'-m','unittest','tests.external.test_manifest.ExternalManifestTests.test_x01_x02_selected_assets_against_independent_source_values','-q'],
                                  cwd=Path(__file__).resolve().parents[3],env=environment,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertNotIn('skipped',result.stderr.lower())

    def test_b02_each_asset_error_allows_the_next_asset(self):
        (self.root/'bad.stl').write_bytes(b'broken')
        (self.root/'other.obj').write_bytes(b'geometry')
        with tempfile.TemporaryDirectory() as outside:
            target=Path(outside)/'outside.stl';target.write_bytes((self.root/'a.stl').read_bytes())
            (self.root/'escape.stl').symlink_to(target)
            faults=[('invalid_path','escape.stl',{}),('read_error','missing.stl',{}),
                    ('hash_mismatch','a.stl',{'expected_sha256':'0'*64}),
                    ('invalid_asset','bad.stl',{}),('unsupported_asset','other.obj',{})]
            for code,path,fields in faults:
                with self.subTest(code=code):
                    report=self.root/f'{code}.jsonl'
                    result,_,err=self.run_cli([self.good_row(asset_id='before'),
                                               self.good_row(asset_id='bad',relative_path=path,**fields),
                                               self.good_row(asset_id='after')],report)
                    self.assertEqual(result,1,err)
                    rows=[json.loads(line) for line in report.read_text().splitlines()]
                    self.assertEqual([r['status'] for r in rows],['ok','failed','ok'])
                    self.assertEqual(rows[1]['code'],code)

    def test_b05_manifest_and_file_aliases_cannot_be_reports(self):
        self.manifest.write_text(json.dumps(self.good_row())+'\n')
        targets=[self.manifest]
        for alias,original in [('link',self.root/'a.stl'),('hardlink',self.storage)]:
            path=self.root/alias
            if alias=='link':path.symlink_to(original)
            else:os.link(original,path)
            targets.append(path)
        for target in targets:
            before=target.read_bytes()
            result,_,_=self.invoke(target)
            self.assertEqual(result,2);self.assertEqual(target.read_bytes(),before)

    def test_b06_flush_failure_is_an_incomplete_batch(self):
        real_open=Path.open;reads=[]
        class Writer:
            def __init__(self,stream):self.stream=stream;self.flushes=0
            def write(self,value):return self.stream.write(value)
            def flush(self):
                self.flushes+=1
                if self.flushes==2:raise OSError('flush failed')
                return self.stream.flush()
            def close(self):return self.stream.close()
        def opening(path,*args,**kwargs):
            stream=real_open(path,*args,**kwargs)
            if path==self.report and args and args[0]=='x':return Writer(stream)
            if path.name=='a.stl':
                if reads:self.assertEqual(len(self.report.read_text().splitlines()),1)
                reads.append(path)
            return stream
        with patch.object(Path,'open',opening):
            code,out,_=self.run_cli([self.good_row(asset_id=str(i)) for i in range(3)])
        result=json.loads(out)
        self.assertEqual(code,1);self.assertFalse(result['completed'])
        self.assertEqual((result['processed'],result['reported'],len(reads)),(2,1,2))

    def test_b09_reports_keep_source_notes_counts_errors_and_versions(self):
        for name,builder in [('rich',material_glb),('fallback',lambda:compression_glb('EXT_meshopt_compression'))]:
            doc,blob=builder();(self.root/f'{name}.glb').write_bytes(pack_glb(doc,blob))
        code,_,err=self.run_cli([self.good_row(relative_path='rich.glb'),
                                self.good_row(asset_id='missing',relative_path='missing.glb'),
                                self.good_row(asset_id='changed',expected_sha256='0'*64),
                                self.good_row(asset_id='fallback',relative_path='fallback.glb')])
        self.assertEqual(code,1,err)
        rows=[json.loads(line) for line in self.report.read_text().splitlines()]
        self.assertEqual((rows[0]['mesh_count'],rows[0]['face_count'],rows[0]['instance_count']),(2,2,2))
        self.assertEqual(rows[1]['code'],'read_error');self.assertIsNone(rows[1]['actual_sha256'])
        self.assertEqual(rows[2]['code'],'hash_mismatch');self.assertIsNotNone(rows[2]['actual_sha256'])
        self.assertTrue(any('pixel' in note for note in rows[0]['notes']))
        self.assertTrue(any('fallback' in note for note in rows[3]['notes']))
        for row in rows:
            self.assertEqual(set(row['environment']),{'python','meshengineer','numpy','pygltflib'})
            self.assertNotIn('vertices',row);self.assertNotIn('images',row)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/"a.stl").write_bytes(ascii_stl([[(0,0,0),(1,0,0),(0,1,0)]]))
        self.storage=self.root/"storage.json";self.storage.write_text(json.dumps({"local":str(self.root)}))
        self.manifest=self.root/"assets.jsonl"
        self.report=self.root/"report.jsonl"

    def run_cli(self,rows, report=None):
        self.manifest.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
        return self.invoke(report)

    def invoke(self,report=None):
        out=io.StringIO();err=io.StringIO()
        with redirect_stdout(out),redirect_stderr(err):
            code=main(["import-check","--manifest",str(self.manifest),"--storage",str(self.storage),"--report",str(report or self.report)])
        return code,out.getvalue(),err.getvalue()

    def test_b01_b02_success_and_failure_share_a_sequential_report(self):
        rows=[{"source_id":"local","asset_id":str(i),"relative_path":p} for i,p in enumerate(["a.stl","missing.stl","a.stl"])]
        code,out,err=self.run_cli(rows)
        self.assertEqual(code,1,err)
        records=[json.loads(line) for line in self.report.read_text().splitlines()]
        self.assertEqual([r["status"] for r in records],["ok","failed","ok"])
        self.assertEqual(records[1]["code"],"read_error")
        self.assertIsNone(records[1]["vertex_count"])
        summary=json.loads(out)
        self.assertEqual((summary["processed"],summary["ok"],summary["failed"]),(3,2,1))
        self.assertTrue(summary["completed"])

    def good_row(self, **changes):
        return {"source_id":"local","asset_id":"one","relative_path":"a.stl",**changes}

    def test_m01_m08_separate_sources_and_same_content_remain_distinct(self):
        self.storage.write_text(json.dumps({"local":str(self.root),"second":str(self.root)}))
        code,_,err=self.run_cli([self.good_row(),self.good_row(source_id="second"),self.good_row(asset_id="two")])
        self.assertEqual(code,0,err)
        self.assertEqual(len(self.report.read_text().splitlines()),3)

    def test_m03_m05_bad_json_rows_blank_comments_and_duplicate_keys(self):
        variants=["", "\n", "[]\n", "null\n", "# comment\n", "{broken}\n",
                  '{"source_id":"local","source_id":"other","asset_id":"x","relative_path":"a.stl"}\n',
                  json.dumps(self.good_row())+"\n\n"]
        for value in variants:
            self.manifest.write_text(value)
            with self.subTest(value=value):
                code,_,err=self.invoke()
                self.assertEqual(code,2,err)
                self.assertFalse(self.report.exists())

    def test_m04_m05_m06_every_field_constraint(self):
        variants=[self.good_row(unrecognized=1)]
        for field in ["source_id","asset_id","relative_path"]:
            for bad in [None,"",42,[]]: variants.append(self.good_row(**{field:bad}))
            row=self.good_row();row.pop(field);variants.append(row)
        for bad in [None,"",42,"z"*64,"a"*63,"a"*65]: variants.append(self.good_row(expected_sha256=bad))
        for row in variants:
            with self.subTest(row=row):
                code,_,err=self.run_cli([row]);self.assertEqual(code,2,err)
                self.assertFalse(self.report.exists())

    def test_m07_duplicate_identity_reports_both_positions(self):
        for change in [{},{"relative_path":"missing.stl"},{"expected_sha256":"0"*64}]:
            code,_,err=self.run_cli([self.good_row(),self.good_row(**change)])
            self.assertEqual(code,2);self.assertIn(":2",err);self.assertIn("line 1",err)

    def test_m09_storage_and_source_failures_are_startup_errors(self):
        for configuration in [[],{"local":"relative"},{"local":str(self.root/"missing")},{"local":42},{"other":str(self.root)}]:
            self.storage.write_text(json.dumps(configuration))
            code,_,err=self.run_cli([self.good_row()]);self.assertEqual(code,2,err)
            self.assertFalse(self.report.exists())

    def test_m10_late_preflight_error_prevents_all_asset_reads(self):
        real_open=Path.open;reads=[]
        def opening(path,*args,**kwargs):
            if path.name=="a.stl": reads.append(path)
            return real_open(path,*args,**kwargs)
        with patch.object(Path,"open",opening):
            code,_,err=self.run_cli([self.good_row(),self.good_row(asset_id="two",expected_sha265="oops")])
        self.assertEqual(code,2,err);self.assertEqual(reads,[])

    def test_b05_existing_reports_and_input_aliases_are_protected(self):
        for target in [self.root/"a.stl",self.storage]:
            before=target.read_bytes()
            code,_,err=self.run_cli([self.good_row()],target)
            self.assertEqual(code,2,err);self.assertEqual(target.read_bytes(),before)
        self.report.write_text("sentinel")
        code,_,_=self.run_cli([self.good_row()])
        self.assertEqual(code,2);self.assertEqual(self.report.read_text(),"sentinel")

    def test_b04_unexpected_errors_preserve_the_original_exception(self):
        real_open=Path.open
        for error in [RuntimeError("bug"),ValueError("bug")]:
            def opening(path,*args,**kwargs):
                if path.name=="a.stl": raise error
                return real_open(path,*args,**kwargs)
            with patch.object(Path,"open",opening),self.assertRaises(type(error)):
                self.run_cli([self.good_row()],self.root/f"{type(error).__name__}.jsonl")

    def test_b03_root_failure_records_current_item_and_stops(self):
        real_open=Path.open;gone=self.root.with_name(self.root.name+"-offline");moved=[]
        @contextmanager
        def asset_opening(path,*args,**kwargs):
            with real_open(path,*args,**kwargs) as stream: yield stream
            if path.name=="a.stl" and not moved:
                self.root.rename(gone);moved.append(True)
        def opening(path,*args,**kwargs):
            if path.name=="a.stl": return asset_opening(path,*args,**kwargs)
            return real_open(path,*args,**kwargs)
        try:
            with patch.object(Path,"open",opening):
                code,out,_=self.run_cli([self.good_row(asset_id=str(i)) for i in range(3)])
        finally:
            if gone.exists(): gone.rename(self.root)
        summary=json.loads(out)
        self.assertEqual(code,1);self.assertFalse(summary["completed"])
        self.assertEqual(summary["processed"],2)
        self.assertEqual(len(self.report.read_text().splitlines()),2)

    def test_b06_write_failure_stops_and_prior_record_is_flushed(self):
        real_open=Path.open;writes=[];reads=[]
        class Writer:
            def __init__(self,stream): self.stream=stream
            def write(self,value):
                writes.append(value)
                if len(writes)==2: raise OSError("disk full")
                return self.stream.write(value)
            def flush(self): return self.stream.flush()
            def close(self): return self.stream.close()
        def opening(path,*args,**kwargs):
            stream=real_open(path,*args,**kwargs)
            if path==self.report and args and args[0]=="x": return Writer(stream)
            if path.name=="a.stl":
                if reads: self.assertEqual(len(self.report.read_text().splitlines()),1)
                reads.append(path)
            return stream
        with patch.object(Path,"open",opening):
            code,out,_=self.run_cli([self.good_row(asset_id=str(i)) for i in range(3)])
        summary=json.loads(out)
        self.assertEqual(code,1);self.assertFalse(summary["completed"])
        self.assertEqual(summary["reported"],1);self.assertEqual(len(reads),2)

    def test_e02_missing_arguments_exit_two(self):
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
            main(["import-check"])
        self.assertEqual(result.exception.code,2)

    def test_m02_optional_and_canonical_hash_in_reports(self):
        actual=hashlib.sha256((self.root/'a.stl').read_bytes()).hexdigest()
        for i,expected in enumerate([None,actual,actual.upper()]):
            row=self.good_row()
            if expected is not None:row['expected_sha256']=expected
            report=self.root/f'hash-{i}.jsonl'
            code,_,err=self.run_cli([row],report)
            self.assertEqual(code,0,err)
            result=json.loads(report.read_text())
            self.assertEqual(result['actual_sha256'],actual)
            self.assertEqual(result['ref']['expected_sha256'],actual if expected is not None else None)

    def test_m03_invalid_utf8_is_a_preflight_failure(self):
        self.manifest.write_bytes(b'\xff\n')
        code,_,_=self.invoke();self.assertEqual(code,2);self.assertFalse(self.report.exists())

    def test_s10_all_four_frozen_inputs_use_the_same_pipeline(self):
        tests=Path(__file__).resolve().parents[2]
        self.storage.write_text(json.dumps({'fixture':str(tests/'fixtures')}))
        rows=[json.loads(line) for line in (tests/'manifests/base.jsonl').read_text().splitlines()]
        expected=json.loads((tests/'fixtures/expected.json').read_text())
        code,out,err=self.run_cli(rows);self.assertEqual(code,0,err)
        records=[json.loads(line) for line in self.report.read_text().splitlines()]
        self.assertEqual(len(records),4)
        for record in records:
            truth=expected[record['ref']['asset_id']]
            for key,value in truth.items():
                self.assertEqual(record['actual_sha256'] if key=='expected_sha256' else record[key],value)
        self.assertEqual(json.loads(out)['reported'],4)

    def test_e02_installed_entry_works_outside_the_project_directory(self):
        executable=Path(sys.executable).with_name('meshengineer')
        result=subprocess.run([str(executable),'--help'],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('import-check',result.stdout)
        self.manifest.write_text(json.dumps(self.good_row())+'\n')
        result=subprocess.run([str(executable),'import-check','--manifest',str(self.manifest),'--storage',str(self.storage),'--report',str(self.report)],
                              cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(result.stdout)['reported'],1)
        result=subprocess.run([str(executable),'import-check','--unknown-option'],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)

    def test_e04_selected_external_suite_missing_data_fails_instead_of_skipping(self):
        environment=dict(os.environ)
        environment.update({name:str(self.root/'missing.json') for name in ['MESHENGINEER_TEST_MANIFEST','MESHENGINEER_TEST_STORAGE','MESHENGINEER_TEST_EXPECTATIONS']})
        result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/external','-t','.','-q'],
                              cwd=Path(__file__).resolve().parents[3],env=environment,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('skipped',result.stderr.lower())
