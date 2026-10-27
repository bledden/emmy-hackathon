"""Execute emitted integer raster code and prove finite-domain bijections."""
import itertools
from types import SimpleNamespace
import unittest

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import Smem, Sync
from emmy.compiler.ir.stmt.blocks import Cond
from emmy.compiler.ir.expr import Literal
from ptxgen.compiler import CompileError, Emitter
from index_model import execute


class ScalarRasterTests(unittest.TestCase):
    def make(self, m, n, inner, group, orient, body=()):
        e = Emitter(None, SimpleNamespace(op=None), {'block': [128, 1, 1]})
        axes = tuple(Axis(name, Dim(size)) for name, size in
                     zip(('outer', 'm', 'n', 'inner'), (2, m, n, inner)))
        tile = SimpleNamespace(axes=axes, raster_axes=('m', 'n'),
                               raster_group=group, raster_orient=orient, body=body)
        gid = e.reg('s32')
        e.raster(tile, gid, [2, m, n, inner])
        return e, gid

    def test_bijection_preserves_outer_and_inner_coordinates(self):
        for m, n, inner, group, orient in itertools.product(
                (3, 4), (5, 12), (1, 2, 3), (1, 3, 8), ('m', 'n')):
            emitter, gid = self.make(m, n, inner, group, orient)
            seen = []
            for flat in range(2*m*n*inner):
                regs, _, trace = execute(emitter.code, {gid: flat})
                seen.append((flat//(m*n*inner), regs[emitter.read('m')],
                             regs[emitter.read('n')], flat % inner))
                self.assertEqual(trace, [])
            expected = set(itertools.product(range(2), range(m), range(n), range(inner)))
            self.assertEqual(set(seen), expected)
            self.assertEqual(len(seen), len(set(seen)))

    def test_shared_memory_and_nested_collectives_keep_whole_cta_requirement(self):
        for body in ((Smem('scratch', (128,)),), (Sync(),),
                     (Cond(Literal(1, 'int'), (Sync(),)),)):
            with self.assertRaisesRegex(CompileError, 'whole CTAs'):
                self.make(3, 5, 1, 3, 'm', body)
            self.make(3, 5, 128, 3, 'm', body)


if __name__ == '__main__':
    unittest.main()
