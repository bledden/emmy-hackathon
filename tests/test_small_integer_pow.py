"""Bounded positive powers: provenance, stale bindings, aliasing and numerics."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from emmy.compiler.graph import Graph
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.stmt.leaves import Assign, Let
from ptxgen.compiler import CompileError, Emitter, compile_file
import test_init_scope as scope


class SmallIntegerPowerTests(unittest.TestCase):
    def body(self, exponent, middle=()):
        return (Let('x', Literal(-2.0), F32), Let('e', Literal(exponent), F32),
                *middle, Assign('x', ElementwiseImpl('pow'), ('x', 'e'), F32), scope.write('x'))

    def test_literal_exponent_and_general_fallback(self):
        helper = scope.InitScopeTests()
        for k in range(9):
            self.assertIn('mov.f32', helper.compile_body(self.body(float(k))))
        for k in (-2.0, -1.0, 0.5, 9.0, float('nan'), float('inf')):
            self.assertIn('div.rn.f64', helper.compile_body(self.body(k)))
        for middle in ((Assign('e', ElementwiseImpl('copy'), ('x',), F32),),
                       (Cond(Literal(1, 'int'), ()),)):
            self.assertIn('div.rn.f64', helper.compile_body(self.body(3.0, middle)))

    def test_real_constant_load_requires_matching_metadata(self):
        source = next((scope.TASK/'problems/p02_scale_shift').glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(source.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        # Change the real generated multiply(x, constant) into pow(x, constant).
        body = tuple(replace(s, op=ElementwiseImpl('pow'), args=(s.args[1], s.args[0]))
                     if type(s).__name__ == 'Assign' and s.op.name == 'multiply' else s
                     for s in node.op.body[0].body)
        node.op = replace(node.op, body=(replace(node.op.body[0], body=body),))
        meta = json.loads(source.with_name('meta.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/source.name
            path.write_text(json.dumps(graph.to_dict(), default=repr))
            path.with_name('meta.json').write_text(json.dumps(meta))
            self.assertIn('mul.rn.f32', compile_file(path))
            arg = next(a for a in meta['kernels'][0]['args'] if a.get('data') == [2.0])
            for value in (3.0, 1e300):
                arg['data'] = [value]
                path.with_name('meta.json').write_text(json.dumps(meta))
                self.assertIn('div.rn.f64', compile_file(path))

    def test_wrong_arity_refuses(self):
        for arguments in (('x',), ('x', 'x', 'x')):
            emitter = Emitter(None, SimpleNamespace(op=None), {})
            emitter.bind('x'); emitter.bind('runtime')
            stmt = Assign('result', ElementwiseImpl('pow'), arguments, F32)
            with self.assertRaises(CompileError):
                emitter.statement(stmt)

    def test_emitted_chain_against_high_precision_and_special_values(self):
        rng = np.random.default_rng(22026)
        special = np.asarray([0., -0., 1., -1., np.inf, -np.inf, np.nan,
                              np.finfo(np.float32).max, -np.finfo(np.float32).max,
                              np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny,
                              2**-149, -2**-149], dtype=np.float32)
        x = np.concatenate([special, rng.uniform(-100,100,5000).astype(np.float32)])
        for k in range(9):
            for alias in (False, True):
                emitter = Emitter(None, SimpleNamespace(op=None), {})
                source = emitter.reg('f32'); destination = source if alias else emitter.reg('f32')
                emitter.positive_integer_power(destination, source, float(k))
                registers = {source: x.copy()}
                with np.errstate(all='ignore'):
                    for line in emitter.code:
                        opcode, operands = line.strip().rstrip(';').split(' ',1)
                        dst, *args = operands.split(', ')
                        values = [np.asarray(1.,dtype=np.float32) if a=='0f3F800000' else registers[a] for a in args]
                        if opcode == 'mul.rn.f32': value = np.float32(values[0]*values[1])
                        elif opcode == 'mov.f32': value = np.broadcast_to(values[0],x.shape).copy()
                        else: raise AssertionError(opcode)
                        registers[dst] = value
                    expected = np.power(x.astype(np.float64),k).astype(np.float32)
                actual = registers[destination]
                self.assertTrue(np.array_equal(np.isnan(actual),np.isnan(expected)))
                self.assertTrue(np.array_equal(np.isinf(actual),np.isinf(expected)))
                exact_special = ~np.isnan(expected[:len(special)])
                self.assertTrue(np.array_equal(actual[:len(special)][exact_special].view(np.uint32),
                                               expected[:len(special)][exact_special].view(np.uint32)))
                finite = np.isfinite(expected)
                np.testing.assert_allclose(actual[finite],expected[finite],rtol=8*np.finfo(np.float32).eps,atol=8*2**-149)


if __name__ == '__main__':
    unittest.main()
