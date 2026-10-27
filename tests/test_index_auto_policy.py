"""Automatic reuse must not enter unmeasured strict or tree combinations."""
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


class IndexAutoPolicyTests(unittest.TestCase):
    def fixture(self):
        source = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        path = Path(temp.name)/source.name
        path.write_bytes(source.read_bytes())
        meta = json.loads(source.with_name('meta.json').read_text())
        path.with_name('meta.json').write_text(json.dumps(meta))
        return path, meta

    def cli(self, path, flags=None):
        env = {k:v for k,v in os.environ.items() if not k.startswith('EMMY_PTX_')}
        env.update(EMMY_PTX_CONSTANT_DIVISION='0',EMMY_PTX_UNROLL='0',EMMY_PTX_SHARED_LOADS='off')
        env.update(flags or {})
        return subprocess.run([sys.executable, str(ROOT/'compile.py'), str(path)],
                              cwd=path.parent, env=env, capture_output=True, text=True)

    def test_unset_and_auto_preserve_qualified_loop_output(self):
        path, _ = self.fixture()
        expected = compile_file(path, contract_fma='auto', index_reuse='loop')
        report = []
        self.assertEqual(compile_file(path, contract_fma='auto', index_reuse='auto',
                                      optimization_report=report), expected)
        self.assertEqual(report[0]['action'], 'loop')
        self.assertTrue(report[-1]['selected'])
        for flags in ({}, {'EMMY_PTX_INDEX_REUSE':'auto'}):
            result = self.cli(path, flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(result.stderr); self.assertEqual(result.stdout, expected)

    def test_ineligible_tolerances_preserve_strict_bytes_even_when_loop_would_select(self):
        path, original = self.fixture()
        mutations = [('atol', value) for value in (0, -1, float('nan'), float('inf'), True, '1e-6', None)]
        mutations += [('rtol', value) for value in (-1, float('nan'), float('inf'), True, '1e-6', None)]
        mutations += [('atol', 'missing'), ('rtol', 'missing')]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                meta = dict(original)
                if value == 'missing': meta.pop(key, None)
                else: meta[key] = value
                path.with_name('meta.json').write_text(json.dumps(meta))
                strict = compile_file(path)
                # Ensure this witness would actually expose a missing guard.
                self.assertNotEqual(compile_file(path, index_reuse='loop'), strict)
                report = []
                self.assertEqual(compile_file(path, contract_fma='auto', index_reuse='auto',
                                              optimization_report=report), strict)
                self.assertEqual(report, [{'pass':'index_reuse_policy', 'mode':'auto',
                    'action':'disabled', 'reason':'FMA disabled or ineligible'}])
                result = self.cli(path)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(result.stderr); self.assertEqual(result.stdout, strict)

    def test_explicit_fma_off_disables_only_automatic_reuse(self):
        path, _ = self.fixture(); strict = compile_file(path)
        for mode in (None, 'auto'):
            flags = {'EMMY_PTX_CONTRACT_FMA':'0'}
            if mode is not None: flags['EMMY_PTX_INDEX_REUSE'] = mode
            result = self.cli(path, flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, strict)
        for mode, api in (('loop', 'loop'), ('1', True)):
            result = self.cli(path, {'EMMY_PTX_CONTRACT_FMA':'0', 'EMMY_PTX_INDEX_REUSE':mode})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, compile_file(path, index_reuse=api))
            self.assertNotEqual(result.stdout, strict)

    def test_short_tree_disables_automatic_reuse_without_silencing_explicit_choice(self):
        path, _ = self.fixture()
        tree_only = compile_file(path, contract_fma='auto', short_tree=True)
        combined = compile_file(path, contract_fma='auto', short_tree=True, index_reuse='loop')
        self.assertNotEqual(combined, tree_only)
        report = []
        self.assertEqual(compile_file(path, contract_fma='auto', short_tree=True,
                                      index_reuse='auto', optimization_report=report), tree_only)
        self.assertEqual(report[0]['action'], 'disabled')
        self.assertEqual(report[0]['reason'], 'short-tree enabled')
        for mode in (None, 'auto', 'loop'):
            flags = {'EMMY_PTX_SHORT_TREE':'1'}
            if mode is not None: flags['EMMY_PTX_INDEX_REUSE'] = mode
            result = self.cli(path, flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, combined if mode == 'loop' else tree_only)

    def test_disabled_mode_and_api_defaults_preserve_baseline(self):
        path, _ = self.fixture()
        result = self.cli(path, {'EMMY_PTX_INDEX_REUSE':'0'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, compile_file(path, contract_fma='auto'))
        self.assertEqual(compile_file(path), compile_file(path, contract_fma=False, index_reuse=False))
        self.assertEqual(compile_file(path, index_reuse='auto'), compile_file(path))


if __name__ == '__main__': unittest.main()
