"""Dataflow guards, numerical differences and CLI selection for guarded FMA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import compile_file, CompileError
from ptxgen.optimizations import contract_fma
from test_gaussian_activations import fma32

TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent / 'upstream/emmy-hackathon-2026'))
MUL = '    mul.rn.f32 %f3, %f1, %f2;'
ADD = '    add.rn.f32 %f0, %f0, %f3;'


class FmaContractionTests(unittest.TestCase):
    def test_accumulator_alias_and_both_operand_orders(self):
        for add in (ADD, '    add.rn.f32 %f0, %f3, %f0;'):
            code = ['L0:', MUL, add, '    @%p0 bra L0;']
            self.assertEqual(contract_fma(code), [
                'L0:', '    fma.rn.f32 %f0, %f1, %f2, %f0;', '    @%p0 bra L0;'])
        # Destination may alias an original input: fma reads all inputs first.
        self.assertEqual(contract_fma([MUL, '    add.rn.f32 %f1, %f0, %f3;']),
                         ['    fma.rn.f32 %f1, %f1, %f2, %f0;'])

    def test_typed_provenance_gate_is_positional_and_empty_means_off(self):
        code = [MUL, ADD]
        self.assertEqual(contract_fma(code, allowed_adds=set()), code)
        self.assertEqual(contract_fma(code, allowed_adds={0}), code)
        self.assertEqual(contract_fma(code, allowed_adds={1}), contract_fma(code))

    def test_auto_keeps_pointwise_cancellation_separately_rounded(self):
        from dataclasses import replace
        from emmy.compiler.graph import Graph
        from emmy.compiler.dtype import F32
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.stmt.leaves import Let, Assign
        import test_init_scope as scope
        src = next((TASK/'problems/p00_scalar_add').glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(src.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        body = (Let('a', Literal(1048576.125), F32),
                Let('b', Literal(1048575.875), F32),
                Let('c', Literal(-1099511627776.0), F32),
                Assign('product', ElementwiseImpl('multiply'), ('a','b'), F32),
                Assign('answer', ElementwiseImpl('add'), ('product','c'), F32),
                scope.write('answer'))
        node.op = replace(node.op, body=(replace(node.op.body[0], body=body),))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'anonymous.ir.json'
            path.write_text(json.dumps(graph.to_dict(), default=repr))
            meta=json.loads(src.with_name('meta.json').read_text())
            meta.update(rtol=1e-6,atol=1e-6)
            path.with_name('meta.json').write_text(json.dumps(meta))
            strict=compile_file(path)
            self.assertEqual(compile_file(path,contract_fma='auto'), strict)
            self.assertNotEqual(compile_file(path,contract_fma=True), strict)
            self.assertIn('fma.rn.f32',compile_file(path,contract_fma=True))
            self.assertNotIn('fma.rn.f32',compile_file(path,contract_fma='auto'))
            result=subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],
                env={**{k:v for k,v in os.environ.items() if not k.startswith('EMMY_PTX_')},'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off'},
                capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout,strict)

    def test_no_cross_control_flow_or_intervening_instruction(self):
        for barrier in ('L1:', '    bra L1;', '    bar.sync 0;',
                        '    mov.f32 %f1, %f4;', '    // boundary', ''):
            code = [MUL, barrier, ADD]
            self.assertEqual(contract_fma(code), code)
        for mul, add in (('    @%p0 '+MUL.strip(), ADD),
                         (MUL, '    @%p0 '+ADD.strip()),
                         (MUL.replace('mul.rn', 'mul.rz'), ADD),
                         (MUL, ADD.replace('add.rn', 'sub.rn'))):
            self.assertEqual(contract_fma([mul, add]), [mul, add])

    def test_temporary_reuse_redefinitions_and_inplace_are_not_fused(self):
        cases = [
            [MUL, ADD, '    st.global.f32 [%rd0], %f3;'],
            ['    mov.f32 %f3, %f9;', MUL, ADD],
            [MUL, ADD, '    mov.b32 %r0, %f3;'],
            [MUL, ADD, '    st.global.v2.f32 [%rd0], {%f3, %f4};'],
            [MUL.replace('%f1', '%f3'), ADD],
            [MUL, '    add.rn.f32 %f3, %f0, %f3;'],
            [MUL, '    add.rn.f32 %f0, %f3, %f3;'],
        ]
        for code in cases:
            self.assertEqual(contract_fma(code), code)
        # %f30 must not be mistaken for %f3.
        self.assertEqual(len(contract_fma([MUL, ADD, '    mov.f32 %f30, %f9;'])), 2)

    def test_literal_bearing_formulas_are_outside_scope(self):
        for mul, add in ((MUL.replace('%f1','0f3F000000'),ADD),
                         (MUL.replace('%f2','0f3F000000'),ADD),
                         (MUL,'    add.rn.f32 %f0, %f3, 0f3F800000;'),
                         (MUL,'    add.rn.f32 %f0, 0f3F800000, %f3;')):
            self.assertEqual(contract_fma([mul, add]),[mul, add])

    def test_numerical_change_is_explicit_including_overflow(self):
        code = contract_fma([MUL, ADD])
        self.assertEqual(len(code), 1)
        a, b = np.float32(1 + 2**-23), np.float32(1 - 2**-23)
        c = np.float32(-1)
        separate = np.float32(np.float32(a*b)+c)
        fused = fma32(a, b, c)
        self.assertEqual(separate, 0)
        self.assertEqual(fused, np.float32(-2**-46))
        maximum = np.finfo(np.float32).max
        with np.errstate(all='ignore'):
            separate = np.float32(np.float32(maximum*np.float32(2))-maximum)
            fused = fma32(maximum, np.float32(2), -maximum)
        self.assertTrue(np.isinf(separate))
        self.assertEqual(fused, maximum)

    def test_official_matmul_has_fusions_and_default_is_unchanged(self):
        path = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        baseline = compile_file(path)
        candidate = compile_file(path, contract_fma=True)
        self.assertGreater(candidate.count('fma.rn.f32'), baseline.count('fma.rn.f32'))
        self.assertEqual(candidate, '\n'.join(contract_fma(baseline.splitlines()))+'\n')
        self.assertEqual(compile_file(path, contract_fma=False), baseline)

    def test_exact_missing_invalid_tolerances_refuse(self):
        src = next((TASK/'problems/p00_scalar_add').glob('*.ir.json'))
        original = json.loads(src.with_name('meta.json').read_text())
        for rtol, atol in ((0,0), (None,1e-3), (-1,1e-3), (float('nan'),0),
                           (float('inf'),0), (True,0), ('0.001',0)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/src.name
                path.write_bytes(src.read_bytes())
                meta = dict(original, rtol=rtol, atol=atol)
                path.with_name('meta.json').write_text(json.dumps(meta))
                with self.assertRaisesRegex(CompileError, 'positive atol'):
                    compile_file(path, contract_fma=True)
                self.assertEqual(compile_file(path, contract_fma='auto'), compile_file(path))

    def test_relative_only_tolerance_preserves_separate_rounding(self):
        # Cancellation makes a relative-only comparison against a separately
        # rounded zero fail, even though the fused result is mathematically closer.
        a, b = np.float32(1+2**-23), np.float32(1-2**-23)
        separate = np.float32(np.float32(a*b)-np.float32(1))
        fused = fma32(a,b,np.float32(-1))
        self.assertFalse(np.isclose(fused,separate,rtol=1e-6,atol=0))
        src = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/src.name;path.write_bytes(src.read_bytes())
            meta=json.loads(src.with_name('meta.json').read_text())
            meta.update(rtol=1e-6,atol=0)
            path.with_name('meta.json').write_text(json.dumps(meta))
            baseline=compile_file(path)
            self.assertEqual(compile_file(path,contract_fma='auto'),baseline)
            with self.assertRaisesRegex(CompileError,'positive atol'):
                compile_file(path,contract_fma=True)
            meta['atol']=1e-6
            path.with_name('meta.json').write_text(json.dumps(meta))
            self.assertNotEqual(compile_file(path,contract_fma='auto'),baseline)

    def test_unset_cli_mode_uses_guarded_auto(self):
        env={**{k:v for k,v in os.environ.items() if k!='EMMY_PTX_CONTRACT_FMA'},'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off'}
        # Test FMA policy independently; guarded reuse has its own CLI matrix.
        env['EMMY_PTX_INDEX_REUSE']='0'
        for case in ('m01_matmul_512','p00_scalar_add'):
            path=next((TASK/'problems'/case).glob('*.ir.json'))
            result=subprocess.run([sys.executable,str(ROOT/'compile.py'),str(path)],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stderr,'')
            self.assertEqual(result.stdout,compile_file(path,contract_fma='auto'))

    def test_cli_optin_and_invalid_mode(self):
        path = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        for mode in ('0', '1', 'auto', 'yes'):
            result = subprocess.run([sys.executable, str(ROOT/'compile.py'), str(path)],
                env={**os.environ,'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off', 'EMMY_PTX_CONTRACT_FMA':mode, 'EMMY_PTX_INDEX_REUSE':'0'}, capture_output=True, text=True)
            if mode == 'yes':
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertIn('must be 0, 1, or auto', result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, '')
                self.assertEqual(result.stdout, compile_file(path, contract_fma=mode!='0'))
        exact = next((TASK/'problems/p00_scalar_add').glob('*.ir.json'))
        result = subprocess.run([sys.executable, str(ROOT/'compile.py'), str(exact)],
            env={**os.environ,'EMMY_PTX_CONSTANT_DIVISION':'0','EMMY_PTX_UNROLL':'0','EMMY_PTX_SHARED_LOADS':'off','EMMY_PTX_CONTRACT_FMA':'auto'},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,compile_file(exact))

    def test_invalid_api_mode_is_refused(self):
        path = next((TASK/'problems/m01_matmul_512').glob('*.ir.json'))
        for mode in ('on','off',1,None):
            with self.assertRaisesRegex(CompileError,'invalid FMA mode'):
                compile_file(path,contract_fma=mode)


if __name__ == '__main__':
    unittest.main()
