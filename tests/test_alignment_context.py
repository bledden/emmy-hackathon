"""Alignment proofs must survive wrap, but never a mutable induction binding."""
import math
import random
import unittest
from types import SimpleNamespace

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr as B, Literal as L, TernaryExpr as T, Var as V
from emmy.compiler.ir.kernel.ir import CpAsyncCopy
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Let
from ptxgen.compiler import Emitter, guaranteed_multiple


def s32(n):
    return (n + 2**31) % 2**32 - 2**31


def evaluate(expr, env):
    if isinstance(expr, L):
        return s32(int(expr.value))
    if isinstance(expr, V):
        return env[expr.name]
    if isinstance(expr, T):
        return evaluate(expr.if_true if evaluate(expr.cond, env) else expr.if_false, env)
    a, b = evaluate(expr.left, env), evaluate(expr.right, env)
    return s32({'+': lambda:a+b, '-':lambda:a-b, '*':lambda:a*b, '%':lambda:a%b}[expr.op]())


def emitter():
    e = Emitter(None, SimpleNamespace(op=None), {'block': [32, 1, 1]})
    e.buffers = {'src': ('%rd900', (4096,), 'global'), 'dst': ('%rd901', (4096,), 'shared')}
    e.buffer_kinds = {'src': 'f32', 'dst': 'f32'}
    return e


def copy_loop(start, step, extra=()):
    copy = CpAsyncCopy('dst', (V('i'),), 'src', (V('i'),), nbytes=16)
    return StridedLoop(Axis('i', Dim(64)), start, step, (copy, *extra), seed=False)


class ContextAlignmentTests(unittest.TestCase):
    def test_random_low_bits_with_negative_values_and_wrap(self):
        rng = random.Random(430043)
        def tree(depth):
            if depth == 0:
                return rng.choice([V('x'), V('y'), L(rng.choice([0, 1, -1, 6, 64, 2**30, -2**31]), 'int')])
            op = rng.choice(['+', '-', '*', '%', 'select'])
            if op == 'select':
                return T(V('y'), tree(depth-1), tree(depth-1))
            return B(op, tree(depth-1), L(rng.choice([3, -3, 6, -6, 64, -64, -2**31]), 'int') if op == '%' else tree(depth-1))
        for _ in range(5000):
            facts = {'x': 2**rng.randrange(0, 32), 'y': 2**rng.randrange(0, 32)}
            expr = tree(3)
            divisor = guaranteed_multiple(expr, facts)
            self.assertTrue(divisor == 0 or (divisor <= 2**32 and divisor & (divisor-1) == 0))
            for _ in range(5):
                env = {n:s32(rng.randrange(-2**31, 2**31)*d) for n,d in facts.items()}
                result = evaluate(expr, env)
                self.assertEqual(result % divisor if divisor else result, 0)

    def test_remainder_uses_both_operands(self):
        self.assertEqual(guaranteed_multiple(B('%', B('*', V('x'), L(4, 'int')), L(6, 'int'))), 2)
        self.assertEqual(guaranteed_multiple(B('%', V('x'), L(64, 'int')), {'x':16}), 16)
        self.assertEqual(guaranteed_multiple(B('%', V('x'), V('y')), {'x':16}), 1)

    def test_start_and_step_both_limit_induction_alignment(self):
        for start, step, wide in [(0,4,True), (8,12,True), (1,4,False), (2,4,False), (0,3,False)]:
            e = emitter(); e.loop(copy_loop(L(start,'int'), L(step,'int')))
            self.assertEqual(any('cp.async.cg' in line for line in e.code), wide)
            self.assertFalse(e.integer_multiples)

    def test_late_and_nested_mutations_decline_for_entire_loop(self):
        update = Accum('i', 'one', dtype='i32')
        for extra in [(update,), (Cond(L(1,'int'), (update,)),)]:
            e = emitter();e.statement(Let('one', L(1,'int')))
            e.loop(copy_loop(L(0,'int'), L(4,'int'), extra))
            self.assertFalse(any('cp.async.cg' in line for line in e.code))

    def test_runtime_step_and_reused_axis_do_not_inherit_facts(self):
        e = emitter();e.bind('step','s32')
        e.loop(copy_loop(L(0,'int'), V('step')))
        self.assertFalse(any('cp.async.cg' in line for line in e.code))
        e = emitter();e.loop(copy_loop(L(0,'int'), L(4,'int')))
        split = len(e.code)
        e.loop(copy_loop(L(1,'int'), L(4,'int')))
        self.assertFalse(any('cp.async.cg' in line for line in e.code[split:]))

    def test_register_rebinding_kills_alignment(self):
        e = emitter();reg = e.bind('x','s32');e.integer_multiples[reg] = 16
        self.assertEqual(e.index_multiple(V('x')),16)
        e.bind('x','s32')
        self.assertEqual(e.index_multiple(V('x')),1)


if __name__ == '__main__':
    unittest.main()
