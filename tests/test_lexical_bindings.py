"""Execute emitted scalar scope witnesses, independent of register allocation.

This small interpreter deliberately refuses instructions outside its vocabulary;
it checks scalar control flow, not GPU participation, loads or numerical accuracy.
"""
import struct
import unittest
from types import SimpleNamespace

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt.blocks import Cond, Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let
from ptxgen.compiler import CompileError, Emitter, definite_conditional_folds


def evaluate(code):
    lines = [line.strip().rstrip(';') for line in code]
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(':')}
    registers = {}

    def read(token):
        if token.startswith('%'):
            return registers[token]
        if token.startswith('0f'):
            return struct.unpack('>f', bytes.fromhex(token[2:]))[0]
        return int(token)

    pc = 0
    for _ in range(10000):
        if pc == len(lines):
            return registers
        line = lines[pc]
        pc += 1
        if line.endswith(':'):
            continue
        if line.startswith('@'):
            pred, branch, target = line.split()
            assert branch == 'bra'
            take = read(pred.lstrip('@!'))
            if take != pred.startswith('@!'):
                pc = labels[target]
            continue
        if line.startswith('bra '):
            pc = labels[line.split()[1]]
            continue
        opcode, operands = line.split(' ', 1)
        dst, *args = operands.split(', ')
        args = [read(arg) for arg in args]
        if opcode.startswith('mov.'):
            result = args[0]
        elif opcode in ('add.s32', 'add.rn.f32'):
            result = args[0] + args[1]
        elif opcode == 'mul.rn.f32':
            result = args[0] * args[1]
        elif opcode == 'neg.f32':
            result = -args[0]
        elif opcode == 'setp.ge.s32':
            result = args[0] >= args[1]
        elif opcode == 'setp.ne.s32':
            result = args[0] != args[1]
        else:
            raise AssertionError('unmodelled instruction: ' + line)
        if opcode.endswith('.f32'):
            result = struct.unpack('f', struct.pack('f', result))[0]
        registers[dst] = result
    raise AssertionError('instruction budget exhausted')


def loop(axis, trips, body, seed=True):
    return Loop(Axis(axis, Dim(trips)), body, seed=seed)


class LexicalBindingTests(unittest.TestCase):
    def run_body(self, body, observed):
        emitter = Emitter(None, SimpleNamespace(op=None), {})
        emitter.body(body)
        names = [emitter.read(name) for name in observed]
        values = evaluate(emitter.code)
        return [values[name] for name in names]

    def test_shadow_rhs_reads_outer_then_updates_inner(self):
        for trips in (0, 1, 3):
            for outer in (-7.0, 0.0, 7.0):
                body = (Let('x', Literal(outer), F32),
                        loop('i', trips, (
                            Assign('x', ElementwiseImpl('negative'), ('x',), F32),
                            Assign('x', ElementwiseImpl('negative'), ('x',), F32))),
                        Let('after', Var('x'), F32))
                self.assertEqual(self.run_body(body, ['x', 'after']), [outer, outer])

    def test_nested_shadow_restores_each_level(self):
        for trips in (1, 3):
            body = (Let('x', Literal(7.0), F32), Let('one', Literal(1.0), F32),
                    loop('outer', trips, (
                        Let('x', Literal(2.0), F32),
                        Cond(Literal(1, 'int'), (Let('x', Literal(9.0), F32),)),
                        Accum('total', 'x', axes=('outer',)))),
                    Let('after', Var('x'), F32))
            self.assertEqual(self.run_body(body, ['after', 'total']), [7.0, 2.0 * trips])

    def test_sibling_different_types_and_nested_type_shadow(self):
        for condition in (0, 1):
            body = (Let('x', Literal(7.0), F32),
                    Cond(Literal(condition, 'int'), (Let('x', Literal(2, 'int')),),
                         (Let('x', Literal(3.0), F32),)),
                    Cond(Literal(1, 'int'), (Let('local', Literal(4, 'int')),)),
                    Cond(Literal(1, 'int'), (Let('local', Literal(5.0), F32),)),
                    Let('after', Var('x'), F32))
            self.assertEqual(self.run_body(body, ['after']), [7.0])

    def test_seeded_nested_fold_does_not_overwrite_outer_state(self):
        body = (Init('total', 7.0, dtype=F32), Let('v', Literal(2.0), F32),
                loop('outer', 2, (
                    loop('inner', 3, (Accum('total', 'v', axes=('inner',)),)),
                    Accum('observed', 'total', axes=('outer',)))),
                Let('after', Var('total'), F32))
        self.assertEqual(self.run_body(body, ['after', 'observed']), [7.0, 12.0])

    def test_conditional_shadow_does_not_overwrite_owned_fold(self):
        for condition in (0, 1):
            body = (Let('v', Literal(2.0), F32),
                    loop('i', 3, (Accum('total', 'v', axes=('i',)),
                        Cond(Literal(condition, 'int'), (
                            Assign('total', ElementwiseImpl('multiply'), ('total', 'v'), F32),)))),)
            self.assertEqual(self.run_body(body, ['total']), [6.0])

    def test_expired_locals_never_become_visible_and_same_scope_type_errors_remain(self):
        for body in (
            (Cond(Literal(1, 'int'), (Let('x', Literal(2.0), F32),)), Let('bad', Var('x'), F32)),
            (Let('x', Literal(2, 'int')), Let('x', Literal(2.0), F32)),
        ):
            with self.assertRaises(CompileError):
                self.run_body(body, [])

    def test_literal_conditional_fold_exports_selected_arm_including_empty_loop(self):
        for condition in (0, 1):
            for trips in (0, 1, 3):
                fold = loop('i', trips, (Accum('total', 'v', axes=('i',)),))
                cond = Cond(Literal(condition, 'int'), (fold,) if condition else (),
                            () if condition else (fold,))
                body = (Let('v', Literal(2.0), F32), cond)
                self.assertEqual(self.run_body(body, ['total']), [2.0 * trips])

    def test_both_dynamic_arms_export_to_one_binding(self):
        for condition in (0, 1):
            left = loop('i', 3, (Accum('total', 'v', axes=('i',)),))
            right = loop('j', 2, (Accum('total', 'v', axes=('j',)),))
            body = (Let('flag', Literal(condition, 'int')), Let('v', Literal(2.0), F32),
                    Cond(Var('flag'), (left,), (right,)))
            self.assertEqual(self.run_body(body, ['total']), [6.0 if condition else 4.0])

    def test_missing_conditional_initialization_still_refuses(self):
        fold = loop('i', 3, (Accum('total', 'v', axes=('i',)),))
        for condition in (Literal(0, 'int'), Var('flag')):
            body = (Let('flag', Literal(1, 'int')), Let('v', Literal(2.0), F32),
                    Cond(condition, (fold,)), Let('after', Var('total'), F32))
            with self.assertRaisesRegex(CompileError, 'value used before definition: total'):
                self.run_body(body, [])

    def test_conditional_export_does_not_escape_an_enclosing_loop(self):
        fold = loop('i', 3, (Accum('total', 'v', axes=('i',)),))
        body = (Let('v', Literal(2.0), F32),
                loop('outer', 2, (Cond(Literal(1, 'int'), (fold,)),)),
                Let('after', Var('total'), F32))
        with self.assertRaisesRegex(CompileError, 'value used before definition: total'):
            self.run_body(body, [])

    def test_literal_definite_assignment_uses_emitted_precision(self):
        fold = loop('i', 3, (Accum('total', 'v', axes=('i',)),))
        for value in (0.0, -0.0, 1e-100, -1e-100):
            self.assertEqual(definite_conditional_folds(Cond(Literal(value), (fold,))), set())
        for value in (1.0, -1.0, 2**-149, float('nan'), float('inf')):
            self.assertEqual(definite_conditional_folds(Cond(Literal(value), (fold,))), {'total'})

    def test_general_branch_sequences_and_nested_conditionals(self):
        for outer in (0, 1):
            for inner in (0, 1):
                for trips in (0, 1, 3):
                    left = (Let('v', Literal(2.0), F32),
                            loop('i', trips, (Accum('total', 'v', axes=('i',)),)),
                            Let('local', Literal(99.0), F32))
                    right = (Let('v', Literal(3.0), F32),
                             loop('j', trips, (Accum('total', 'v', axes=('j',)),)))
                    body = (Let('outer', Literal(outer, 'int')),
                            Let('inner', Literal(inner, 'int')),
                            Cond(Var('outer'), (
                                Let('before', Literal(5.0), F32),
                                Cond(Var('inner'), left, right),
                                Let('after', Var('total'), F32)), right))
                    expected = (2.0 if outer and inner else 3.0) * trips
                    self.assertEqual(self.run_body(body, ['total']), [expected])

    def test_multiple_exports_and_inner_fold_shadowing(self):
        for flag in (0, 1):
            body = (Let('flag', Literal(flag, 'int')), Let('v', Literal(2.0), F32),
                    Cond(Literal(1, 'int'), (
                        loop('i', 3, (Accum('a', 'v', axes=('i',)),)),
                        Cond(Var('flag'),
                             (loop('j', 8, (Accum('a', 'v', axes=('j',)),)),),
                             (loop('k', 9, (Accum('a', 'v', axes=('k',)),)),)),
                        loop('l', 2, (Accum('b', 'a', axes=('l',)),))),))
            self.assertEqual(self.run_body(body, ['a', 'b']), [6.0, 12.0])

    def test_export_does_not_make_a_read_before_seed_legal(self):
        fold = loop('i', 2, (Accum('total', 'v', axes=('i',)),))
        body = (Let('v', Literal(2.0), F32), Cond(Literal(1, 'int'),
                (Let('early', Var('total'), F32), fold)))
        with self.assertRaisesRegex(CompileError, 'value used before definition: total'):
            self.run_body(body, [])

    def test_ordinary_redefinition_removes_fold_export(self):
        fold = loop('i', 2, (Accum('total', 'v', axes=('i',)),))
        body = (Let('v', Literal(2.0), F32), Cond(Literal(1, 'int'),
                (fold, Let('total', Literal(99.0), F32))), Let('bad', Var('total'), F32))
        with self.assertRaisesRegex(CompileError, 'value used before definition: total'):
            self.run_body(body, [])


if __name__ == '__main__':
    unittest.main()
