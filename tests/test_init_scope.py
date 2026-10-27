"""Alpha-renaming an inner Init must preserve the outer binding.

These tests compare complete emitted programs for equivalent scoped IR, and
reject escape of a block-local declaration. They require no GPU or renderer.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Init, Let, Load, Write
from emmy.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('EMMY_COMPILER_ROOT', str(ROOT)))
from ptxgen.compiler import CompileError, compile_file

TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent/'upstream/emmy-hackathon-2026'))


def write(name):
    return Write('add', (BinaryExpr('*', Var('a0'), Literal(2, 'int')),),
                 values=(name, name), value_dtype=F32)


class InitScopeTests(unittest.TestCase):
    def compile_body(self, body, problem='p00_scalar_add'):
        source = next((TASK/'problems'/problem).glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(source.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        node.op = replace(node.op, body=(replace(node.op.body[0], body=body),))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/source.name
            path.write_text(json.dumps(graph.to_dict(), default=repr))
            path.with_name('meta.json').write_bytes(source.with_name('meta.json').read_bytes())
            return compile_file(path)

    def assert_alpha_equal(self, build):
        self.assertEqual(self.compile_body(build('state')), self.compile_body(build('inner')))

    def test_loop_strided_and_cond_declaration_alpha_equivalence(self):
        for kind in ('loop', 'strided', 'cond'):
            for trips in (0, 2):
                with self.subTest(kind=kind, trips=trips):
                    def build(name):
                        # The pre-declaration read checks declaration is not hoisted.
                        body = (write('state'), Init(name, 2.0, dtype=F32), write(name))
                        if kind == 'loop':
                            block = Loop(Axis('j', Dim(trips)), body)
                        elif kind == 'strided':
                            block = StridedLoop(Axis('j', Dim(trips)), Literal(0, 'int'), Literal(1, 'int'), body)
                        else:
                            block = Cond(Literal(trips, 'int'), body)
                        return (Init('state', 7.0, dtype=F32), block, write('state'))
                    self.assert_alpha_equal(build)

    def test_nested_declarations_restore_each_enclosing_binding(self):
        def build(name):
            deeper = 'state' if name == 'state' else 'deep'
            return (Init('state', 7.0, dtype=F32),
                    Loop(Axis('j', Dim(2)), (Init(name, 2.0, dtype=F32),
                         Cond(Literal(1, 'int'), (Init(deeper, 3.0, dtype=F32), write(deeper))),
                         write(name))), write('state'))
        self.assert_alpha_equal(build)

    def test_both_arms_have_separate_declarations(self):
        def build(name):
            other = 'state' if name == 'state' else 'other'
            return (Init('state', 7.0, dtype=F32),
                    Cond(BinaryExpr('<', Var('a0'), Literal(32, 'int')),
                         (Init(name, 2.0, dtype=F32), write(name)),
                         (write('state'), Init(other, 3.0, dtype=F32), write(other))),
                    write('state'))
        self.assert_alpha_equal(build)

    def test_local_init_does_not_escape(self):
        for wrapper in (lambda body: Loop(Axis('j', Dim(2)), body),
                        lambda body: Cond(Literal(1, 'int'), body)):
            with self.assertRaisesRegex(CompileError, 'value used before definition: local'):
                self.compile_body((wrapper((Init('local', 2.0, dtype=F32),)), write('local')))

    def test_seed_false_updates_correct_outer_or_shadowed_state(self):
        def build(name):
            fold = Loop(Axis('k', Dim(3)),
                        (Accum(name, 'v', ElementwiseImpl('add'), F32, axes=('k',)),), seed=False)
            return (Init('state', 7.0, dtype=F32), Let('v', Literal(1.0), F32),
                    Loop(Axis('j', Dim(2)), (Init(name, 2.0, dtype=F32), fold, write(name))),
                    write('state'))
        self.assert_alpha_equal(build)
        # An update with no inner declaration must continue to reuse outer state.
        ptx = self.compile_body((Init('state', 7.0, dtype=F32), Let('v', Literal(1.0), F32),
                                 Loop(Axis('k', Dim(3)), (Accum('state', 'v', axes=('k',)),), seed=False),
                                 write('state')))
        self.assertIn('add.rn.f32 %f0, %f0, %f1;', ptx)

    def test_other_shadowing_definitions_restore_outer_binding(self):
        for definition in ('let', 'assign', 'load'):
            for wrapper in ('cond', 'loop'):
                def build(name):
                    statement = {'let': Let(name, Literal(2.0), F32),
                        'assign': Assign(name, ElementwiseImpl('negative'), ('state',), F32),
                        'load': Load(names=(name,), input='x0', index=(Var('a0'),), dtype=F32)}[definition]
                    inner = (statement, write(name))
                    block = Cond(Literal(1, 'int'), inner) if wrapper == 'cond' else Loop(Axis('j', Dim(2)), inner)
                    return (Let('state', Literal(1.0), F32), block, write('state'))
                with self.subTest(definition=definition, wrapper=wrapper):
                    self.assert_alpha_equal(build)

    def test_non_accum_locals_and_loop_axis_do_not_escape(self):
        for definition in (Let('local', Literal(2.0), F32),
                           Load(names=('local',), input='x0', index=(Var('a0'),), dtype=F32)):
            for block in (Cond(Literal(1, 'int'), (definition,), (definition,)),
                          Loop(Axis('j', Dim(2)), (definition,))):
                with self.assertRaisesRegex(CompileError, 'value used before definition: local'):
                    self.compile_body((block, write('local')))
        with self.assertRaisesRegex(CompileError, 'value used before definition: j'):
            self.compile_body((Loop(Axis('j', Dim(2)), ()), Let('late', Var('j'))))
        with self.assertRaisesRegex(CompileError, 'unsupported lexical shadowing: j'):
            self.compile_body((Loop(Axis('j', Dim(2)), (Let('j', Literal(1, 'int')),)),))

    def test_same_scope_updates_and_local_fold_export_remain_valid(self):
        body = (Let('v', Literal(1.0), F32),
                Assign('v', ElementwiseImpl('negative'), ('v',), F32),
                Loop(Axis('j', Dim(3)), (Accum('sum', 'v', axes=('j',)),)), write('sum'))
        self.assertIn('add.rn.f32', self.compile_body(body))
        # A nested fold's final value belongs to its enclosing body, not every ancestor.
        with self.assertRaisesRegex(CompileError, 'value used before definition: sum'):
            self.compile_body((Let('v', Literal(1.0), F32),
                               Loop(Axis('outer', Dim(2)), (body[2],)), write('sum')))

    def test_nested_seed_shadow_restores_outer_and_mixed_fold_operations_refuse(self):
        def build(name):
            fold = Loop(Axis('j', Dim(3)), (Accum(name, 'v', axes=('j',)),))
            return (Init('state', 7.0, dtype=F32), Let('v', Literal(1.0), F32),
                    Loop(Axis('outer', Dim(2)), (fold, write(name))), write('state'))
        self.assert_alpha_equal(build)
        for seed in (False, True):
            with self.assertRaisesRegex(CompileError, 'mixed reduction operations for state'):
                self.compile_body((Init('state', 7.0, dtype=F32), Let('v', Literal(1.0), F32),
                    Loop(Axis('j', Dim(3)), (Accum('state', 'v', axes=('j',)),
                         Accum('state', 'v', ElementwiseImpl('maximum'), axes=('j',))), seed=seed),
                    write('state')))

    def test_owned_fold_name_is_not_a_local_assignment_exemption(self):
        for definition in (Let('sum', Literal(9.0), F32),
                           Assign('sum', ElementwiseImpl('multiply'), ('sum', 'v'), F32),
                           Load(names=('sum',), input='x0', index=(Var('a0'),), dtype=F32)):
            with self.subTest(definition=type(definition).__name__):
                with self.assertRaisesRegex(CompileError, 'unsupported lexical shadowing: sum'):
                    self.compile_body((Let('v', Literal(2.0), F32),
                        Loop(Axis('j', Dim(3)), (Accum('sum', 'v', axes=('j',)), definition)),
                        write('sum')))

    def test_collective_temporaries_and_partners_do_not_escape_or_shadow(self):
        combine = (Assign('tmp', ElementwiseImpl('add'), ('state', 'peer'), F32),
                   Assign('state', ElementwiseImpl('copy'), ('tmp',), F32))
        for kind in ('shuffle', 'tree'):
            state = (Let('state', Literal(1.0), F32),)
            if kind == 'shuffle':
                collective = WarpShuffle(('state',), ('peer',), combine, 32)
            else:
                state += (Smem('scratch', (128,)),
                          Write('scratch', (Var('a1_co'),), values=('state',), value_dtype=F32), Sync())
                collective = TreeHalve(('scratch',), ('state',), ('peer',), combine, 128, 'a1_co')
            observe = lambda name: Let('after', Var(name), F32)
            # Repeated collectives reuse cached locals without making them visible.
            self.compile_body((*state, collective, collective, observe('state')), 'r04_row_sum')
            self.compile_body((*state, Cond(Literal(1, 'int'), (collective,)), observe('state')),
                              'r04_row_sum')
            for local in ('tmp', 'peer'):
                with self.subTest(kind=kind, local=local, behavior='escape'):
                    with self.assertRaisesRegex(CompileError, 'value used before definition: ' + local):
                        self.compile_body((*state, collective, observe(local)), 'r04_row_sum')
                with self.subTest(kind=kind, local=local, behavior='shadow'):
                    with self.assertRaisesRegex(CompileError, 'unsupported lexical shadowing: ' + local):
                        self.compile_body((*state, Let(local, Literal(9.0), F32), collective,
                                           observe(local)), 'r04_row_sum')

    def test_uniform_cond_collective_can_update_outer_state(self):
        source = next((TASK/'problems/r04_row_sum').glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(source.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        tile = node.op.body[0]
        combine = Assign('state', ElementwiseImpl('add'), ('state', 'peer'), F32)
        # Use a whole-CTA tile so this reaches scope validation, not the excess-thread guard.
        body = (Let('state', Literal(1.0), F32),
                Cond(Literal(1, 'int'), (WarpShuffle(('state',), ('peer',), (combine,), 32),)))
        node.op = replace(node.op, body=(replace(tile, body=body),))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/source.name
            path.write_text(json.dumps(graph.to_dict(), default=repr))
            path.with_name('meta.json').write_bytes(source.with_name('meta.json').read_bytes())
            self.assertIn('shfl.sync.bfly', compile_file(path))


if __name__ == '__main__':
    unittest.main()
