from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptxgen.compiler import CompileError, compile_file

TASK = Path(os.environ.get("EMMY_TASK_ROOT", ROOT.parent / "upstream/emmy-hackathon-2026"))


class ContractTests(unittest.TestCase):
    def fixture(self, name="p00_scalar_add"):
        original = next((TASK / "problems" / name).glob("*.ir.json"))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        ir = Path(temporary.name) / original.name
        ir.write_bytes(original.read_bytes())
        (ir.parent / "meta.json").write_bytes((original.parent / "meta.json").read_bytes())
        return ir

    def change_meta(self, path, mutate):
        metadata = path.parent / "meta.json"
        data = json.loads(metadata.read_text())
        mutate(data)
        metadata.write_text(json.dumps(data))

    def change_kernel(self, path, mutate):
        from emmy.compiler.graph import Graph
        graph = Graph.from_dict(json.loads(path.read_text()))
        node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
        node.op = mutate(node.op)
        path.write_text(json.dumps(graph.to_dict(), default=repr))

    def test_missing_metadata_refuses_without_stdout(self):
        path = self.fixture()
        (path.parent / "meta.json").unlink()
        proc = subprocess.run([sys.executable, ROOT / "compile.py", path], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, "")
        self.assertIn("metadata missing", proc.stderr)

    def test_untyped_let_infers_integer_bindings(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
        from emmy.compiler.ir.stmt.leaves import Let
        path = self.fixture()
        def mutate(k):
            tile = k.body[0]
            bindings = (Let("zero", Literal(0, "int")),
                        Let("alias", Var("a0")),
                        Let("computed", BinaryExpr("+", Var("alias"), Var("zero"))),
                        Let("selected", TernaryExpr(
                            BinaryExpr("<", Var("a0"), Literal(32, "int")),
                            Var("computed"), Var("alias"))))
            body = tuple(replace(s, index=(Var("selected"),))
                         if type(s).__name__ == "Load" else s for s in tile.body)
            return replace(k, body=(replace(tile, body=(*bindings, *body)),))
        self.change_kernel(path, mutate)
        ptx = compile_file(path)
        self.assertIn("selp.s32", ptx)

    def test_let_rejects_unsupported_explicit_dtype(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.stmt.leaves import Let
        path = self.fixture()
        self.change_kernel(path, lambda k: replace(k, body=(replace(
            k.body[0], body=(Let("bad", Literal(1), "complex128"), *k.body[0].body)),)))
        with self.assertRaisesRegex(CompileError, "unsupported dtype"):
            compile_file(path)

    def test_expression_type_boundaries_refuse_without_stdout(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr
        from emmy.compiler.ir.stmt.leaves import Let
        cases = [
            BinaryExpr("+", Literal(1, "int"), Literal(2.0)),
            BinaryExpr("<", Literal(1, "int"), Literal(2.0)),
            TernaryExpr(Literal(1, "int"), Literal(1, "int"), Literal(2.0)),
            *(BinaryExpr(op, Literal(3.0), Literal(2.0)) for op in ("/", "//", "%", "&")),
        ]
        for expression in cases:
            with self.subTest(expression=expression):
                path = self.fixture()
                self.change_kernel(path, lambda k: replace(k, body=(replace(
                    k.body[0], body=(Let("bad", expression), *k.body[0].body)),)))
                run = subprocess.run([sys.executable, ROOT / "compile.py", path],
                                     capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(run.stdout, "")
                self.assertIn("CompileError", run.stderr)

    def test_float_expression_does_not_silently_become_an_index(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import BinaryExpr, Literal
        from emmy.compiler.ir.stmt.leaves import Let
        path = self.fixture()
        expression = BinaryExpr("+", Literal(1.0), Literal(2.0))
        self.change_kernel(path, lambda k: replace(k, body=(replace(
            k.body[0], body=(Let("bad", expression, "int"), *k.body[0].body)),)))
        with self.assertRaisesRegex(CompileError, "result type disagrees"):
            compile_file(path)

    def test_ambiguous_entry_refuses(self):
        path = self.fixture()
        self.change_meta(path, lambda m: m["kernels"].append(copy.deepcopy(m["kernels"][0])))
        with self.assertRaisesRegex(CompileError, "ambiguous metadata"):
            compile_file(path)

    def test_metadata_controls_parameter_order(self):
        path = self.fixture()
        original = compile_file(path)
        def reverse(m):
            m["kernels"][0]["args"].reverse()
            m["kernels"][0]["arg_order"].reverse()
        self.change_meta(path, reverse)
        changed = compile_file(path)
        self.assertNotEqual(original, changed)
        # Output is now the first argument: its base pointer is rd0, not rd2.
        stores = changed.index("st.global.v2.f32")
        self.assertIn(", %rd0,", changed[changed.rfind("add.u64", 0, stores):stores])

    def test_metadata_shape_mismatch_refuses(self):
        path = self.fixture()
        self.change_meta(path, lambda m: m["kernels"][0]["args"][0].update(shape=[129]))
        with self.assertRaisesRegex(CompileError, "shape disagrees"):
            compile_file(path)

    def test_partial_block_collective_refuses(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.graph import Graph
        path = self.fixture("r01_array_sum")
        graph = Graph.from_dict(json.loads(path.read_text()))
        node = graph.nodes["sum_1"]
        tile = node.op.body[0]
        axis = replace(tile.axes[0], extent=Dim(127))
        node.op = replace(node.op, body=(replace(tile, axes=(axis,)),))
        path.write_text(json.dumps(graph.to_dict(), default=repr))
        with self.assertRaisesRegex(CompileError, "partial-block collectives"):
            compile_file(path)

    def test_activation_wrong_arities_refuse(self):
        from dataclasses import replace
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        for op in ('rsqrt', 'sigmoid', 'silu', 'tanh', 'log', 'square',
                   'floor', 'ceil', 'trunc', 'rint', 'sign', 'exp2', 'log2', 'log10', 'softplus',
                   'erf', 'gelu', 'gelu_tanh'):
            for count in (0, 2):
                with self.subTest(op=op, count=count):
                    path = self.fixture()
                    def mutate(k):
                        tile = k.body[0]
                        body = tuple(replace(s, op=ElementwiseImpl(op), args=s.args[:count])
                                     if type(s).__name__ == 'Assign' else s for s in tile.body)
                        return replace(k, body=(replace(tile, body=body),))
                    self.change_kernel(path, mutate)
                    with self.assertRaisesRegex(CompileError, 'bad arity'):
                        compile_file(path)

    def test_where_wrong_arities_refuse(self):
        from dataclasses import replace
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        for count in (0, 1, 2, 4):
            with self.subTest(count=count):
                path = self.fixture()
                def mutate(k):
                    tile = k.body[0]
                    body = tuple(replace(s, op=ElementwiseImpl('where'),
                                         args=(s.args * 2)[:count])
                                 if type(s).__name__ == 'Assign' else s for s in tile.body)
                    return replace(k, body=(replace(tile, body=body),))
                self.change_kernel(path, mutate)
                with self.assertRaisesRegex(CompileError, 'bad arity for where'):
                    compile_file(path)

    def test_unimplemented_activation_still_refuses(self):
        from dataclasses import replace
        from emmy.compiler.ir.elementwise import ElementwiseImpl
        path = self.fixture()
        def mutate(k):
            tile = k.body[0]
            body = tuple(replace(s, op=ElementwiseImpl('tan'), args=s.args[:1])
                         if type(s).__name__ == 'Assign' else s for s in tile.body)
            return replace(k, body=(replace(tile, body=body),))
        self.change_kernel(path, mutate)
        with self.assertRaisesRegex(CompileError, 'unsupported elementwise operation tan'):
            compile_file(path)

    def test_insufficient_dynamic_shared_memory_refuses(self):
        path = self.fixture("m01_matmul_512")
        self.change_meta(path, lambda m: m["kernels"][0].update(smem_bytes=256))
        with self.assertRaisesRegex(CompileError, "exceed launch dynamic"):
            compile_file(path)

    def test_invalid_raster_group_refuses(self):
        from dataclasses import replace
        path = self.fixture("a02_attention_128")
        self.change_kernel(path, lambda k: replace(k, body=(replace(k.body[0], raster_group=0),)))
        with self.assertRaisesRegex(CompileError, "invalid raster group"):
            compile_file(path)

    def test_unknown_swizzled_async_copy_refuses(self):
        from dataclasses import replace
        path = self.fixture("m01_matmul_512")
        def mutate(k):
            tile = k.body[0]
            body = tuple(replace(s, swizzle="B256") if type(s).__name__ == "CpAsyncCopy"
                         else s for s in tile.body)
            return replace(k, body=(replace(tile, body=body),))
        self.change_kernel(path, mutate)
        with self.assertRaisesRegex(CompileError, "unsupported software swizzle"):
            compile_file(path)

    def test_barrier_in_thread_varying_loop_refuses(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.kernel.ir import Sync
        path = self.fixture("f01_rmsnorm")
        def mutate(k):
            tile = k.body[0]
            loop = tile.body[0]
            loop = replace(loop, axis=replace(loop.axis, extent=Dim(257)),
                           body=(*loop.body, Sync()))
            return replace(k, body=(replace(tile, body=(loop, *tile.body[1:])),))
        self.change_kernel(path, mutate)
        # Threads 0, 1, and 127 take 3, 2, and 2 iterations: no uniform barrier.
        self.assertEqual([len(range(i, 257, 128)) for i in (0, 1, 127)], [3, 2, 2])
        with self.assertRaisesRegex(CompileError, "collective loop requires a proven uniform"):
            compile_file(path)

    def test_nonuniform_ancestor_of_shuffle_refuses(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import Literal, Var
        from emmy.compiler.ir.kernel.ir import WarpShuffle
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
        for varied in ('start', 'step', 'end'):
            with self.subTest(varied=varied):
                path = self.fixture('r01_array_sum')
                def mutate(k):
                    inner = Loop(Axis('inner', Dim(2)),
                                 (WarpShuffle(('unbound',), ('peer',), (), 32),))
                    bounds = dict(start=Literal(0, 'int'), step=Literal(1, 'int'),
                                  end=Literal(3, 'int'))
                    bounds[varied] = Var('lane')
                    outer = StridedLoop(Axis('outer', Dim(3)), body=(inner,), **bounds)
                    return replace(k, body=(replace(k.body[0], body=(outer,)),))
                self.change_kernel(path, mutate)
                # Reject at the ancestor's participation proof, before reading state.
                with self.assertRaisesRegex(CompileError, 'collective loop requires a proven uniform'):
                    compile_file(path)

    def test_equal_trip_coordinate_barrier_compiles(self):
        from dataclasses import replace
        from emmy.compiler.ir.kernel.ir import Sync
        for name in ('f01_rmsnorm', 'f03_l2norm', 'r04_row_sum'):
            with self.subTest(name=name):
                path = self.fixture(name)
                def mutate(k):
                    tile = k.body[0]
                    loop = replace(tile.body[0], body=(*tile.body[0].body, Sync()))
                    return replace(k, body=(replace(tile, body=(loop, *tile.body[1:])),))
                self.change_kernel(path, mutate)
                self.assertIn('bar.sync 0;', compile_file(path))

    def test_coordinate_range_is_invalidated_by_writes_anywhere(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import Literal, Var
        from emmy.compiler.ir.kernel.ir import Sync, WarpShuffle
        from emmy.compiler.ir.stmt.blocks import Cond, Loop
        from emmy.compiler.ir.stmt.leaves import Let
        for location in ('before', 'after', 'body', 'branch', 'backedge', 'combine', 'axis'):
            with self.subTest(location=location):
                path = self.fixture('f03_l2norm')
                def mutate(k):
                    tile = k.body[0]
                    loop = replace(tile.body[0], body=(*tile.body[0].body, Sync()))
                    changed = Let('a1_co', Literal(512, 'int'))
                    if location == 'axis':
                        changed = Loop(Axis('a1_co', Dim(1)), ())
                    if location == 'combine':
                        changed = WarpShuffle(('state',), ('peer',), (changed,), 32)
                    if location == 'branch':
                        changed = Cond(Var('lane'), (changed,))
                    if location == 'body':
                        body = (replace(loop, body=(*loop.body, changed)),)
                    elif location == 'before':
                        body = (changed, loop)
                    else:
                        body = (loop, changed)
                    if location == 'backedge':
                        body = (Loop(Axis('outer', Dim(2)), body),)
                    return replace(k, body=(replace(tile, body=body),))
                self.change_kernel(path, mutate)
                if location == 'before':
                    # The overwritten coordinate is now a proven literal, not
                    # the original per-lane range. All threads take zero trips.
                    self.assertIn('bar.sync', compile_file(path))
                else:
                    with self.assertRaisesRegex(CompileError, 'proven uniform'):
                        compile_file(path)

    def test_collective_loop_index_mutation_refuses(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import Literal, Var
        from emmy.compiler.ir.kernel.ir import Sync
        from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop
        from emmy.compiler.ir.stmt.leaves import Let
        for strided in (False, True):
            path = self.fixture('r01_array_sum')
            def mutate(k):
                body = (Let('i', Var('lane')), Sync())
                axis = Axis('i', Dim(4))
                loop = (StridedLoop(axis, Literal(0, 'int'), Literal(1, 'int'), body)
                        if strided else Loop(axis, body))
                return replace(k, body=(replace(k.body[0], body=(loop,)),))
            self.change_kernel(path, mutate)
            with self.assertRaisesRegex(CompileError, 'proven uniform'):
                compile_file(path)

    def test_conditional_ancestor_of_loop_collective_refuses(self):
        from dataclasses import replace
        from emmy.compiler.dim import Dim
        from emmy.compiler.ir.axis import Axis
        from emmy.compiler.ir.expr import Var
        from emmy.compiler.ir.kernel.ir import WarpShuffle
        from emmy.compiler.ir.stmt.blocks import Cond, Loop
        path = self.fixture('r01_array_sum')
        loop = Loop(Axis('step', Dim(2)), (WarpShuffle(('x',), ('peer',), (), 32),))
        self.change_kernel(path, lambda k: replace(k, body=(replace(
            k.body[0], body=(Cond(Var('lane'), (loop,)),)),)))
        with self.assertRaisesRegex(CompileError, 'collective inside conditional'):
            compile_file(path)

    def test_reduction_combine_rejects_non_assign_statements(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.kernel.ir import Sync
        from emmy.compiler.ir.stmt.leaves import Let
        for tag in ('WarpShuffle', 'TreeHalve'):
            for statement in (Sync(), Let('temporary', Literal(1.0))):
                with self.subTest(tag=tag, statement=type(statement).__name__):
                    path = self.fixture('r01_array_sum')
                    def mutate(k):
                        tile = k.body[0]
                        body = tuple(replace(s, combine_states=(statement,))
                                     if type(s).__name__ == tag else s for s in tile.body)
                        return replace(k, body=(replace(tile, body=body),))
                    self.change_kernel(path, mutate)
                    with self.assertRaisesRegex(CompileError, 'combine supports only Assign'):
                        compile_file(path)

    def test_tree_layout_boundaries_refuse(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import Literal
        from emmy.compiler.ir.stmt.leaves import Let
        for case, message in [('mutated', 'immutable CTA-local'),
                              ('scale', 'scale must match'),
                              ('overlap', 'independent writer slots'),
                              ('storage', 'buffer is missing or too small'),
                              ('length', 'exceeds participating coordinates')]:
            with self.subTest(case=case):
                path = self.fixture('r01_array_sum')
                def mutate(k):
                    tile = k.body[0]
                    body = []
                    for s in tile.body:
                        if type(s).__name__ == 'TreeHalve':
                            if case == 'scale': s = replace(s, inner=('lane', 16))
                            if case == 'overlap': s = replace(s, inner=('warp', 4))
                            if case == 'length': s = replace(s, length=8)
                        if type(s).__name__ == 'Smem' and case == 'storage':
                            s = replace(s, extents=(1,))
                        body.append(s)
                    if case == 'mutated':
                        body.append(Let('warp', Literal(0, 'int')))
                    return replace(k, body=(replace(tile, body=body),))
                self.change_kernel(path, mutate)
                with self.assertRaisesRegex(CompileError, message):
                    compile_file(path)

    def test_reduction_state_partner_alias_refuses(self):
        from dataclasses import replace
        for tag in ('WarpShuffle', 'TreeHalve'):
            path = self.fixture('r01_array_sum')
            def mutate(k):
                tile = k.body[0]
                body = tuple(replace(s, state_b=s.state) if type(s).__name__ == tag
                             else s for s in tile.body)
                return replace(k, body=(replace(tile, body=body),))
            self.change_kernel(path, mutate)
            with self.assertRaisesRegex(CompileError, 'partner names must be distinct'):
                compile_file(path)

    def test_misaligned_vector_store_scalarizes(self):
        from dataclasses import replace
        from emmy.compiler.ir.expr import BinaryExpr, Literal
        path = self.fixture()
        def mutate(k):
            tile = k.body[0]
            body = tuple(replace(s, index=(BinaryExpr("+", s.index[0], Literal(1, "int")),))
                         if type(s).__name__ == "Write" else s for s in tile.body)
            return replace(k, body=(replace(tile, body=body),))
        self.change_kernel(path, mutate)
        ptx = compile_file(path)
        self.assertNotIn("st.global.v2", ptx)
        self.assertEqual(ptx.count("st.global.f32"), 2)

    def test_open_ptx_hygiene_and_determinism(self):
        for path in sorted((TASK / "problems").glob("*/*.ir.json")):
            with self.subTest(case=path.parent.name):
                text = compile_file(path)
                self.assertEqual(text, compile_file(path))
                self.assertTrue(text.isascii())
                self.assertTrue(text.startswith(".version 8.0\n.target sm_89\n.address_size 64\n"))
                self.assertNotRegex(text, r"(?m)^\s*(?:add|sub|mul)\.f32")
                labels = re.findall(r"^\s*(L\d+):", text, re.M)
                self.assertEqual(len(labels), len(set(labels)))
                for kind, prefix, size in re.findall(r"\.reg \.(\w+) %(rd|r|f|p)<(\d+)>;", text):
                    indices = [int(n) for n in re.findall(r"%" + prefix + r"(\d+)\b", text)]
                    self.assertLess(max(indices), int(size))


if __name__ == "__main__":
    unittest.main()
