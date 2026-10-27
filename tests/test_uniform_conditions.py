"""CTA-wide predicates may guard collectives; any varying ancestor must refuse."""
from dataclasses import replace
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.kernel.ir import Sync
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Let, Load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import CompileError, compile_file, uniform_tile_names
TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent / 'upstream/emmy-hackathon-2026'))


class UniformConditionTests(unittest.TestCase):
    def stage(self, change, case='r04_row_sum'):
        source = next((TASK / 'problems' / case).glob('*.ir.json'))
        graph = Graph.from_dict(json.loads(source.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        node.op = replace(node.op, body=(change(node.op.body[0]),))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        ir = Path(temporary.name) / source.name
        ir.write_text(json.dumps(graph.to_dict(), default=repr))
        ir.with_name('meta.json').write_bytes(source.with_name('meta.json').read_bytes())
        return ir

    def test_row_uniform_predicates_allow_both_collective_branches(self):
        expressions = [Literal(0, 'int'), Literal(1, 'int'),
                       BinaryExpr('<', Var('a0'), Literal(32, 'int')),
                       BinaryExpr('==', BinaryExpr('&', Var('a0'), Literal(1, 'int')),
                                  Literal(0, 'int')),
                       TernaryExpr(Var('a0'), Literal(1, 'int'), Literal(0, 'int'))]
        for expr in expressions:
            for else_arm in (False, True):
                with self.subTest(expr=expr, else_arm=else_arm):
                    def wrap(tile):
                        # Storage stays outside branches, as in a row-specialized schedule.
                        prefix = tuple(s for s in tile.body if type(s).__name__ == 'Smem')
                        body = tuple(s for s in tile.body if type(s).__name__ != 'Smem')
                        return replace(tile, body=(*prefix, Cond(expr, () if else_arm else body,
                                                                body if else_arm else ())))
                    ptx = compile_file(self.stage(wrap))
                    self.assertIn('shfl.sync', ptx)
                    self.assertIn('bar.sync', ptx)

    def test_divergent_ancestor_still_refuses_uniform_descendant(self):
        for name in ('lane', 'warp', 'a1_co'):
            with self.subTest(name=name):
                path = self.stage(lambda t: replace(t, body=(Cond(Var(name),
                                  (Cond(Literal(1, 'int'), (Sync(),)),)),)))
                with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
                    compile_file(path)

    def test_every_conditional_ancestor_is_checked(self):
        path = self.stage(lambda t: replace(t, body=(Cond(Var('a0'),
                          (Cond(Var('lane'), (Sync(),)),)),)))
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(path)

    def test_writes_anywhere_invalidate_coordinate_uniformity(self):
        for placement in ('before', 'after', 'inside', 'loop', 'load'):
            with self.subTest(placement=placement):
                def change(t):
                    altered = Let('a0', Var('lane'))
                    if placement == 'load':
                        altered = Load(name='a0', input='x', index=(Var('a0'), Var('a1_co')), dtype=F32)
                    cond = Cond(Var('a0'), (Sync(),))
                    if placement == 'inside':
                        cond = replace(cond, body=(altered, Sync()))
                        body = (cond,)
                    elif placement == 'loop':
                        body = (cond, Loop(Axis('a0', Dim(2)), ()))
                    elif placement == 'before':
                        body = (altered, cond)
                    else:
                        body = (cond, altered)
                    return replace(t, body=body)
                with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
                    compile_file(self.stage(change))

    def test_uniform_alias_and_loop_index_allow_collectives(self):
        alias = self.stage(lambda t: replace(t, body=(Let('row_alias', Var('a0')),
                           Cond(Var('row_alias'), (Sync(),)))))
        induction = self.stage(lambda t: replace(t, body=(Loop(Axis('i', Dim(3)),
                               (Cond(Var('i'), (Sync(),)),)),)))
        for path in (alias, induction):
            self.assertIn('bar.sync', compile_file(path))

    def test_uniform_facts_follow_bindings_and_expire_on_writes(self):
        for varying in (Var('lane'), Var('a1_co')):
            path = self.stage(lambda t: replace(t, body=(Let('alias', Var('a0')),
                Let('alias', varying), Cond(Var('alias'), (Sync(),)))))
            with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
                compile_file(path)
        # A uniform inner shadow must not make an outer varying value uniform.
        path = self.stage(lambda t: replace(t, body=(Let('alias', Var('lane')),
            Cond(Literal(1, 'int'), (Let('alias', Var('a0')),)),
            Cond(Var('alias'), (Sync(),)))))
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(path)

    def test_carried_update_invalidates_earlier_predicate_on_backedge(self):
        path = self.stage(lambda t: replace(t, body=(Let('flag', Literal(1.0), F32),
            Load(name='v', input='x', index=(Var('a0'), Var('a1_co')), dtype=F32),
            Loop(Axis('i', Dim(2)), (
                Cond(Var('flag'), (Sync(),)), Accum('flag', 'v', axes=('i',))), seed=False))))
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(path)

    def test_uniform_trip_count_does_not_imply_uniform_induction(self):
        # Every lane performs four iterations, but its induction value differs.
        path = self.stage(lambda t: replace(t, body=(StridedLoop(Axis('i', Dim(512)),
            Var('a1_co'), Literal(128, 'int'), (Cond(Var('i'), (Sync(),)),)),)))
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(path)

    def test_uniform_alias_as_strided_loop_start(self):
        path = self.stage(lambda t: replace(t, body=(Let('start', Literal(0, 'int')),
            StridedLoop(Axis('i', Dim(4)), Var('start'), Literal(1, 'int'),
                        (Cond(Var('i'), (Sync(),)),)),)))
        self.assertIn('bar.sync', compile_file(path))

    def test_uniform_condition_cannot_rescue_nonuniform_loop(self):
        path = self.stage(lambda t: replace(t, body=(StridedLoop(Axis('i', Dim(257)),
                          Var('a1_co'), Literal(128, 'int'),
                          (Cond(Var('a0'), (Sync(),)),)),)))
        with self.assertRaisesRegex(CompileError, 'proven uniform'):
            compile_file(path)

    def test_refusal_cli_has_no_partial_ptx(self):
        path = self.stage(lambda t: replace(t, body=(Cond(Var('a1_co'), (Sync(),)),)))
        result = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(path)],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertIn('collective inside conditional', result.stderr)

    def test_raster_overrides_and_duplicate_axis_names_refuse(self):
        raster = self.stage(lambda t: replace(t, body=(Cond(Var('a0_b'), (Sync(),)),)),
                            case='a01_attention')
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(raster)
        duplicate = self.stage(lambda t: replace(t, axes=(t.axes[0], replace(t.axes[1], name='a0'))))
        with self.assertRaisesRegex(CompileError, 'duplicate Tile axis'):
            compile_file(duplicate)

    def test_outer_coordinate_that_splits_a_cta_refuses(self):
        # Match BOTH IR and metadata: a 256-thread CTA spans two 128-lane rows.
        # This must fail at the uniform-predicate proof, not launch validation.
        path = self.stage(lambda t: replace(t, block_threads=256,
                          body=(Cond(Var('a0'), (Sync(),)),)))
        meta = json.loads(path.with_name('meta.json').read_text())
        meta['kernels'][0]['block'] = [256, 1, 1]
        meta['kernels'][0]['grid'] = [32, 1, 1]
        path.with_name('meta.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(path)

    def test_accepted_coordinate_layouts_have_one_value_per_cta(self):
        # Enumerate independently decoded physical threads, including non-power-of-two
        # axes and CTA boundaries that cut through an axis. No PTX-emitter arithmetic.
        for shape in itertools.product((1, 2, 3, 4, 5, 8), repeat=3):
            cells = math.prod(shape)
            for block in (1, 2, 3, 4, 8, 16, 32):
                if cells % block:
                    continue
                axes = tuple(Axis(f'a{i}', Dim(n)) for i, n in enumerate(shape))
                accepted = uniform_tile_names(axes, block, set())
                for pos, size in enumerate(shape):
                    stride = math.prod(shape[pos + 1:])
                    if axes[pos].name not in accepted:
                        continue
                    for start in range(0, cells, block):
                        values = {(gid // stride) % size for gid in range(start, start + block)}
                        self.assertEqual(len(values), 1, (shape, block, pos, start))


if __name__ == '__main__':
    unittest.main()
