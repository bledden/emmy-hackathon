"""Launch-site selection and ABI compatibility for a reused entry."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import CompileError, compile_file
TASK = Path(os.environ.get('EMMY_TASK_ROOT', ROOT.parent / 'upstream/emmy-hackathon-2026'))


class MetaSelectionTests(unittest.TestCase):
    def stage(self, mutate=lambda m: None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        original = next((TASK / 'problems/p00_scalar_add').glob('*.ir.json'))
        path = folder / original.name
        path.write_bytes(original.read_bytes())
        meta = json.loads((original.parent / 'meta.json').read_text())
        mutate(meta)
        (folder / 'meta.json').write_text(json.dumps(meta))
        return path

    @staticmethod
    def add_site(meta):
        chosen = meta['kernels'][0]
        other = copy.deepcopy(chosen)
        names = {'x0': 'y0', 'x1': 'y1', 'add': 'add_0'}
        other.update(node='add_0', final=False, arg_order=[names[n] for n in chosen['arg_order']])
        for arg in other['args']:
            arg['name'] = names[arg['name']]
        other['zero_outputs'] = [names[n] for n in other['zero_outputs']]
        meta['kernels'] = [other, chosen]

    def test_reused_kernel_selects_its_site_in_either_order(self):
        baseline = compile_file(self.stage())
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                def mutate(m):
                    self.add_site(m)
                    if reverse:
                        m['kernels'].reverse()
                self.assertEqual(compile_file(self.stage(mutate)), baseline)

    def test_reused_kernel_falls_back_to_unique_buffer_names(self):
        def mutate(m):
            self.add_site(m)
            for k in m['kernels']:
                k.pop('node')
        self.assertEqual(compile_file(self.stage(mutate)), compile_file(self.stage()))

    def test_reused_kernel_preserves_permuted_positional_abi(self):
        def permute(m):
            k = m['kernels'][0]
            k['arg_order'].reverse()
            k['args'].reverse()
        expected = compile_file(self.stage(permute))
        def mutate(m):
            permute(m)
            self.add_site(m)
        self.assertEqual(compile_file(self.stage(mutate)), expected)

    def test_reused_kernel_incompatible_launches_refuse(self):
        changes = {
            'grid': lambda k: k.update(grid=[2, 1, 1]),
            'block': lambda k: k.update(block=[128, 1, 1]),
            'smem': lambda k: k.update(smem_bytes=16),
            'shape': lambda k: k['args'][0].update(shape=[129]),
            'dtype': lambda k: k['args'][0].update(dtype='i32'),
            'role': lambda k: k['args'][0].update(output=True),
        }
        for name, change in changes.items():
            with self.subTest(field=name):
                def mutate(m):
                    self.add_site(m)
                    change(m['kernels'][0])
                with self.assertRaisesRegex(CompileError, 'reused kernel launches disagree'):
                    compile_file(self.stage(mutate))

    def test_other_launch_malformed_abi_also_refuses(self):
        changes = {
            'order': lambda k: k['arg_order'].reverse(),
            'scalar': lambda k: k['args'][0].update(scalar=True),
            'duplicate': lambda k: k['arg_order'].__setitem__(1, k['arg_order'][0]),
        }
        for name, change in changes.items():
            with self.subTest(field=name):
                def mutate(m):
                    self.add_site(m)
                    change(m['kernels'][0])
                with self.assertRaises(CompileError):
                    compile_file(self.stage(mutate))

    def test_other_launch_noninteger_dimensions_refuse(self):
        # Python compares float/bool dimensions equal to their integer values;
        # validate every site's types before comparing launch signatures.
        for field, position, value in [('grid', 0, 1.0), ('grid', 0, True),
                                       ('block', 0, 256.0), ('block', 1, 1.0)]:
            with self.subTest(field=field, position=position, value=value):
                def mutate(m):
                    self.add_site(m)
                    m['kernels'][0][field][position] = value
                result = subprocess.run([sys.executable, ROOT / 'compile.py', self.stage(mutate)],
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertIn('launch dimensions must be integers', result.stderr)

    def test_ambiguous_or_missing_entries_refuse_without_stdout(self):
        def duplicate(m):
            m['kernels'].append(copy.deepcopy(m['kernels'][0]))
        def ambiguous_buffers(m):
            duplicate(m)
            for k in m['kernels']:
                k.pop('node')
        def missing(m):
            m['kernels'][0]['entry'] = 'another_entry'
        def no_site(m):
            self.add_site(m)
            m['kernels'][1] = copy.deepcopy(m['kernels'][0])
        for mutate in (duplicate, ambiguous_buffers, missing, no_site):
            with self.subTest(case=mutate.__name__):
                result = subprocess.run([sys.executable, ROOT / 'compile.py', self.stage(mutate)],
                                        text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                self.assertIn('compile error', result.stderr)
                if mutate is no_site:
                    self.assertIn('no metadata launch site matches this KernelOp', result.stderr)

    def test_large_launch_widens_and_noninteger_launch_refuses(self):
        for block, grid in [(256.0, 1), (True, 64)]:
            with self.subTest(block=block, grid=grid):
                def mutate(m):
                    m['kernels'][0].update(block=[block, 1, 1], grid=[grid, 1, 1])
                with self.assertRaisesRegex(CompileError, 'launch (thread indices|dimensions)'):
                    compile_file(self.stage(mutate))
        wide = self.stage(lambda m: m['kernels'][0].update(grid=[2**23+1, 1, 1]))
        self.assertIn('setp.ge.u64', compile_file(wide))
        # At most 2^31 threads still has a maximum physical index of 2^31-1.
        path = self.stage(lambda m: m['kernels'][0].update(grid=[2**23, 1, 1]))
        self.assertIn('.entry', compile_file(path))

    def test_tile_extent_outside_signed_index_contract_refuses(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.graph import Graph
        path = self.stage(lambda m: m['kernels'][0].update(grid=[2**23, 1, 1]))
        graph = Graph.from_dict(json.loads(path.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == 'KernelOp')
        tile = node.op.body[0]
        node.op = replace(node.op, body=(replace(tile, axes=(replace(tile.axes[0], extent=Dim(2**31+1)),)),))
        path.write_text(json.dumps(graph.to_dict(), default=repr))
        with self.assertRaisesRegex(CompileError, 'Tile axis values must fit signed 32-bit'):
            compile_file(path)


if __name__ == '__main__':
    unittest.main()
