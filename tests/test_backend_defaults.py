"""CLI defaults select the qualified backend; API and explicit rollback stay stable."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import compile_file
TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent/'upstream/emmy-hackathon-2026'))


class BackendDefaultTests(unittest.TestCase):
    def cli(self, path, flags=None):
        env = {k:v for k,v in os.environ.items() if not k.startswith('EMMY_PTX_')}
        env.update(flags or {})
        return subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],
                              env=env,capture_output=True,text=True)

    def test_default_and_explicit_compact_match(self):
        path = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        expected = compile_file(path, contract_fma='auto', index_reuse='auto',
                                constant_division=True, unroll_loops=True, shared_loads='compact')
        baseline = compile_file(path, contract_fma='auto', index_reuse='auto')
        self.assertNotEqual(expected, baseline)
        for flags in ({}, {'EMMY_PTX_CONSTANT_DIVISION':'1','EMMY_PTX_UNROLL':'1','EMMY_PTX_SHARED_LOADS':'compact'}):
            result = self.cli(path, flags)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(result.stderr)
            self.assertEqual(result.stdout, expected)

    def test_explicit_off_recovers_previous_cli_and_api_defaults_stay_off(self):
        path = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        flags = {'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off'}
        result = self.cli(path, flags)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,compile_file(path,contract_fma='auto',index_reuse='auto'))
        self.assertEqual(compile_file(path),compile_file(path,constant_division=False,unroll_loops=False,shared_loads='off'))

    def test_exact_tolerance_keeps_strict_rounding_with_backend_enabled(self):
        source = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/source.name;path.write_bytes(source.read_bytes())
            meta = json.loads(source.with_name('meta.json').read_text());meta['atol']=0
            path.with_name('meta.json').write_text(json.dumps(meta))
            expected = compile_file(path,constant_division=True,unroll_loops=True,shared_loads='compact')
            result = self.cli(path)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout,expected)

    def test_flags_are_independent_and_invalid_modes_do_not_emit_ptx(self):
        path = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        cases=[({'EMMY_PTX_CONSTANT_DIVISION':'0'},dict(constant_division=False,unroll_loops=True,shared_loads='compact')),
               ({'EMMY_PTX_UNROLL':'0'},dict(constant_division=True,unroll_loops=False,shared_loads='compact')),
               ({'EMMY_PTX_SHARED_LOADS':'off'},dict(constant_division=True,unroll_loops=True,shared_loads='off'))]
        for flags,options in cases:
            result=self.cli(path,flags)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout,compile_file(path,contract_fma='auto',index_reuse='auto',**options))
        for flag in ('EMMY_PTX_CONSTANT_DIVISION','EMMY_PTX_UNROLL','EMMY_PTX_SHARED_LOADS'):
            for value in ('','yes','AUTO'):
                result=self.cli(path,{flag:value})
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(result.stdout)

    def test_explanation_reports_actual_default_ptx(self):
        path=ROOT/'examples/async_copy/async_copy.ir.json'
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'report.json'
            result=subprocess.run([sys.executable,str(ROOT/'tools/explain.py'),str(path),'--report',str(report)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout,self.cli(path).stdout)
            info=json.loads(report.read_text())
            self.assertEqual(info['policy']['shared_loads'],'compact')
            self.assertTrue(info['policy']['constant_division'])
            self.assertTrue(info['policy']['unroll_loops'])
            self.assertIn('constant_division',{x['pass'] for x in info['decisions']})


if __name__=='__main__':unittest.main()
