"""Legacy folds without axes own the nearest enclosing loop, including masks.

Compare complete PTX with the same fold explicitly annotated, so a misplaced
seed (inside a branch, on every outer iteration, or despite seed=False) fails.
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
from emmy.compiler.ir.stmt.leaves import Accum, Init, Let, Write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import compile_file

TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent / 'upstream/emmy-hackathon-2026'))


class ImplicitAccumAxesTests(unittest.TestCase):
    def compile_body(self, body):
        source = next((TASK / 'problems/p00_scalar_add').glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(source.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        tile = node.op.body[0]
        node.op = replace(node.op, body=(replace(tile, body=body),))
        return self.compile_graph(graph, source)

    @staticmethod
    def compile_graph(graph, source):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / source.name
            path.write_text(json.dumps(graph.to_dict(), default=repr))
            path.with_name('meta.json').write_bytes(source.with_name('meta.json').read_bytes())
            return compile_file(path)

    def test_public_non_drain_folds_match_without_redundant_annotations(self):
        for case in ('r01_array_sum', 'r02_big_sum', 'r03_mean', 'r04_row_sum',
                     'f01_rmsnorm', 'f02_row_softmax', 'f03_l2norm'):
            with self.subTest(case=case):
                source = next((TASK / 'problems' / case).glob('*.ir.json'))
                graph = Graph.from_dict(json.loads(source.read_text()))
                original = compile_file(source)
                changed = 0

                def strip(body, owner=None):
                    nonlocal changed
                    result = []
                    for stmt in body:
                        tag = type(stmt).__name__
                        if tag == 'Accum':
                            # Never strip information from an outer-owned drain.
                            self.assertEqual(stmt.axes, (owner,))
                            stmt = replace(stmt, axes=())
                            changed += 1
                        if hasattr(stmt, 'body'):
                            nearest = stmt.axis.name if tag in ('Loop', 'StridedLoop') else owner
                            stmt = replace(stmt, body=strip(stmt.body, nearest))
                        if hasattr(stmt, 'else_body'):
                            stmt = replace(stmt, else_body=strip(stmt.else_body, owner))
                        result.append(stmt)
                    return tuple(result)

                for node in graph.nodes.values():
                    if type(node.op).__name__ == 'KernelOp':
                        node.op = replace(node.op, body=strip(node.op.body))
                self.assertGreater(changed, 0)
                self.assertEqual(self.compile_graph(graph, source), original)

    def assert_equivalent(self, build):
        explicit = self.compile_body(build(True))
        implicit = self.compile_body(build(False))
        self.assertEqual(implicit, explicit)
        return implicit

    @staticmethod
    def fold(explicit, op='add'):
        return Accum('sum', 'value', ElementwiseImpl(op), F32,
                     axes=('inner',) if explicit else ())

    @staticmethod
    def finish(loop):
        return (Let('value', Literal(2.0), F32), loop,
                Write('add', (Var('a0'),), values=('sum',), value_dtype=F32))

    def test_ordinary_and_strided_folds_including_empty_loops(self):
        for strided in (False, True):
            for count in (0, 3):
                for op in ('add', 'multiply', 'maximum', 'minimum'):
                    with self.subTest(strided=strided, count=count, op=op):
                        def build(explicit):
                            axis = Axis('inner', Dim(count))
                            body = (self.fold(explicit, op),)
                            loop = (StridedLoop(axis, Literal(0, 'int'), Literal(2, 'int'), body)
                                    if strided else Loop(axis, body))
                            return self.finish(loop)
                        self.assert_equivalent(build)

    def test_nearest_loop_owns_seed_not_outer_loop(self):
        def build(explicit):
            inner = Loop(Axis('inner', Dim(3)), (self.fold(explicit),))
            # The inner fold exports to the outer loop BODY, not past that
            # enclosing scope. Keep the seed-placement observation in scope.
            return (Let('value', Literal(2.0), F32),
                    Loop(Axis('outer', Dim(2)), (inner,
                         Write('add', (Var('a0'),), values=('sum',), value_dtype=F32))))
        ptx = self.assert_equivalent(build)
        # Exactly one zero identity initialization, emitted inside the outer loop.
        self.assertEqual(ptx.count(', 0f00000000;'), 1)

    def test_conditional_fold_seed_is_outside_condition(self):
        for else_arm in (False, True):
            with self.subTest(else_arm=else_arm):
                def build(explicit):
                    folds = (self.fold(explicit),)
                    condition = BinaryExpr('==', Var('inner'), Literal(1, 'int'))
                    masked = Cond(condition, () if else_arm else folds,
                                  folds if else_arm else ())
                    return self.finish(Loop(Axis('inner', Dim(3)), (masked,)))
                self.assert_equivalent(build)

    def test_seed_false_preserves_external_initialization(self):
        def build(explicit):
            loop = Loop(Axis('inner', Dim(3)), (self.fold(explicit),), seed=False)
            # The outer seed=True must not claim the unannotated inner fold.
            body = self.finish(Loop(Axis('outer', Dim(2)), (loop,)))
            return (Init('sum', 7.0, dtype=F32), *body)
        ptx = self.assert_equivalent(build)
        self.assertNotIn(', 0f00000000;', ptx)


if __name__ == '__main__':
    unittest.main()
