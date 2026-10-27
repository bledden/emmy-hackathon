"""Whole-kernel fragment lowering and adversarial contract checks, all offline."""
from dataclasses import replace
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Let
from emmy.compiler.ir.pure.lam import Lambda
from ptxgen.compiler import CompileError, Emitter, compile_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from make_tensor_examples import example
from fragment_model import execute


def build(graph, meta):
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    return Emitter(graph, node, meta['kernels'][0]).compile()


def edit(graph, change):
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
    tile = node.op.body[0]
    node.op = replace(node.op, body=(change(tile),))


class TensorFragmentTests(unittest.TestCase):
    def test_checker_rejects_numerically_identical_wrong_register_type(self):
        graph, meta, data, _ = example()
        ptx = build(graph, meta).replace('.reg .b32 %b', '.reg .s32 %b')
        with self.assertRaisesRegex(AssertionError, 'packed b32'):
            execute(ptx, data, meta)

    def test_exact_results_from_emitted_instructions_changed_inputs_and_strides(self):
        for seed in (0, 11, 627):
            for chunks in (1, 2, 3):
                for padded in (False, True):
                    graph, meta, data, expected = example(chunks=chunks, padded=padded, seed=seed,
                                                        names=('renamed_a', 'renamed_b', 'out'))
                    actual = execute(build(graph, meta), data, meta)
                    np.testing.assert_array_equal(actual['out'], expected['out'])

    def test_checker_detects_reversed_packed_halves(self):
        graph, meta, data, expected = example(seed=821)
        ptx = build(graph, meta)
        broken, count = re.subn(r'(mov.b32 %\w+, \{)(%h\d+), (%h\d+)(\};)',
                               r'\1\3, \2\4', ptx, count=1)
        self.assertEqual(count, 1)
        self.assertFalse(np.array_equal(execute(broken, data, meta)['result'], expected['result']))

    def test_checker_detects_duplicate_store_owner_and_wrong_accumulator(self):
        graph, meta, data, expected = example()
        ptx = build(graph, meta)
        store = next(line for line in ptx.splitlines() if 'st.global.f32' in line)
        with self.assertRaisesRegex(AssertionError, 'duplicate store owner'):
            execute(ptx.replace(store, store+'\n'+store), data, meta)
        broken = re.sub(r'(st.global.f32 \[%\w+\], )%f\d+;', r'\g<1>0f00000000;', ptx)
        self.assertFalse(np.array_equal(execute(broken, data, meta)['result'], expected['result']))

    def test_fragment_lifetime_role_and_store_overlap(self):
        cases = [lambda b: b[:3]+b[4:],  # missing A load
                 lambda b: b[:2]+b[3:],  # missing C declaration
                 lambda b: (b[0],)+b,  # duplicate declaration
                 lambda b: b[:-2]+(replace(b[-2], a_frag='fb'), b[-1]),
                 lambda b: b+(b[-1],)]
        for change in cases:
            graph, meta, _, _ = example()
            edit(graph, lambda t: replace(t, body=change(t.body)))
            with self.assertRaises(CompileError): build(graph, meta)

    def test_every_unsupported_field_refuses(self):
        changes = {
            0: {'count': 2, 'nregs': 4, 'dtype': 'bf16', 'shape': (16, 8, 8), 'role': 'd'},
            3: {'swizzle': 'B128', 'staged': True, 'gmem_guard': (Literal(1),),
                'k_zero': Literal(0), 'b_trans': True, 'pair_frag': 'fa', 'byte_slab': True,
                'scale_buffer': 'left', 'scale_index': (Literal(0),), 'scale_ldm': 1,
                'fragment_layout': 'm16n8k8', 'fragment_index': 1, 'ldm': 17},
            5: {'shape': (16, 8, 8), 'ab_dtype': 'bf16', 'c_dtype': 'f16',
                'b_row_major': True, 'sfa_frag': 'fa', 'sfb_frag': 'fb'},
            6: {'shape': (16, 8, 8), 'epilogue': Lambda(('x',), (), ('x',)), 'extra_frags': ('fc',),
                'm_guard': (Literal(0), Literal(16)), 'n_guard': (Literal(0), Literal(8)),
                'atomic': True, 'fragment_layout': 'm16n8k8', 'volta_interleaved': True,
                'fragment_index': (1, 0), 'swizzle': 'B32', 'row_dim': 0, 'col_dim': 1, 'ldm': 9}}
        for index, fields in changes.items():
            for name, value in fields.items():
                with self.subTest(index=index, field=name):
                    graph, meta, _, _ = example()
                    edit(graph, lambda t: replace(t, body=tuple(
                        replace(s, **{name: value}) if i == index else s for i, s in enumerate(t.body))))
                    with self.assertRaises(CompileError): build(graph, meta)

    def test_bounds_constant_indices_dtype_and_buffer_direction(self):
        for field, value in [('src_index', (Literal(-1, 'int'), Literal(0, 'int'))),
                             ('src_index', (Literal(1, 'int'), Literal(0, 'int'))),
                             ('src_index', (Var('lane'), Literal(0, 'int'))),
                             ('src_index', (Literal(0, 'f32'), Literal(0, 'int'))),
                             ('src_buffer', 'result')]:
            graph, meta, _, _ = example()
            edit(graph, lambda t: replace(t, body=t.body[:3]+(replace(t.body[3], **{field:value}),)+t.body[4:]))
            with self.assertRaises(CompileError): build(graph, meta)

    def test_participation_control_flow_and_scalar_mixing_refuse(self):
        for change in [lambda t: replace(t, axes=(replace(t.axes[0], extent=Dim(31)),)),
                       lambda t: replace(t, axes=(replace(t.axes[0], name='lane'),)),
                       lambda t: replace(t, raster_group=1),
                       lambda t: replace(t, body=(Cond(Literal(1), t.body, ()),)),
                       lambda t: replace(t, body=(Loop(t.axes[0], t.body),)),
                       lambda t: replace(t, body=(Let('x', Literal(0)),)+t.body)]:
            graph, meta, _, _ = example()
            edit(graph, change)
            with self.assertRaises(CompileError): build(graph, meta)
        for key, value in [('block', [31,1,1]), ('block', [64,1,1]), ('grid', [2,1,1])]:
            graph, meta, _, _ = example()
            meta['kernels'][0][key] = value
            with self.assertRaises(CompileError): build(graph, meta)

    def test_modes_and_real_cli_match_saved_examples(self):
        for folder in sorted((ROOT/'examples').glob('mma_*')):
            path = folder/'tensor_tile.ir.json'
            data, expected = dict(np.load(folder/'inputs.npz')), dict(np.load(folder/'expected.npz'))
            meta = json.loads((folder/'meta.json').read_text())
            for options in ({}, {'contract_fma':'auto','index_reuse':'auto'}, {'index_reuse': True},
                            {'index_reuse':'loop'}, {'short_tree':True,'index_reuse':True},
                            {'contract_fma':'auto','index_reuse':'auto','constant_division':True,
                             'unroll_loops':True,'shared_loads':'compact'}):
                ptx = compile_file(path, **options)
                for name, values in execute(ptx, data, meta).items():
                    np.testing.assert_array_equal(values, expected[name])
            cli = subprocess.run([sys.executable, str(ROOT/'compile.py'), str(path)],
                                 capture_output=True, text=True, check=True)
            self.assertEqual(cli.stdout, compile_file(path, contract_fma='auto', index_reuse='auto',
                constant_division=True, unroll_loops=True, shared_loads='compact'))

    def test_explanation_sidecar_preserves_stdout(self):
        path = ROOT/'examples/mma_tile/tensor_tile.ir.json'
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/'report.json'
            proc = subprocess.run([sys.executable, str(ROOT/'tools/explain.py'), str(path),
                                   '--report', str(report)], capture_output=True, text=True, check=True)
            self.assertEqual(proc.stdout, compile_file(path, contract_fma='auto', index_reuse='auto',
                constant_division=True, unroll_loops=True, shared_loads='compact'))
            info = json.loads(report.read_text())
            self.assertEqual(info['decisions'][0]['action'], 'disabled')


if __name__ == '__main__':
    unittest.main()
