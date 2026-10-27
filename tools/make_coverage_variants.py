"""Independent GPU witnesses for arithmetic, collectives, memory and index binding.

Only typed IR construction/serialization and NumPy references; no renderer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import Smem, Sync, WarpShuffle
from emmy.compiler.ir.stmt.leaves import Assign, Let, Load, Write

from make_variants import inputs, load, reshape, save


def lit(n):
    return Literal(n, "int")


def binary(op, a, b):
    return BinaryExpr(op, a, b)


def template(upstream, size=128, cells=128):
    graph, meta = load(upstream, "p00_scalar_add")
    node = graph.nodes["add"]
    tile = node.op.body[0]
    tile = replace(tile, axes=(replace(tile.axes[0], extent=Dim(cells)),),
                   block_threads=128)
    reshape(graph, {name: [size] for name in ("x0", "x1", "add")})
    kernel = meta["kernels"][0]
    kernel["block"], kernel["grid"] = [128, 1, 1], [(cells + 127) // 128, 1, 1]
    for arg in kernel["args"]:
        arg["shape"] = [size]
        if "seed" in arg:
            arg["seed"] += 191
    meta["code"] = "Independent coverage witness; see generator for operation."
    return graph, meta, node, tile


def read(names, source, index, dtype):
    return Load(names=names, input=source, index=index, dtype=dtype)


def write(name, index, values):
    return Write(output=name, index=(index,), values=tuple(values), value_dtype=F32)


def emit(out, name, graph, meta, node, tile, body, expected):
    node.op = replace(node.op, body=(replace(tile, body=tuple(body)),))
    save(out, name, graph, meta, {"add": np.asarray(expected, dtype=np.float32)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    for operation in ("minimum", "negative", "abs", "reciprocal"):
        g, m, node, tile = template(args.upstream, size=257, cells=257)
        data = inputs(m)
        x, y = data["x0"], data["x1"]
        op_args = ("x", "y") if operation == "minimum" else ("x",)
        body = [read(("x",), "x0", (Var("a0"),), F32),
                read(("y",), "x1", (Var("a0"),), F32),
                Assign("result", ElementwiseImpl(operation), op_args, F32),
                write("add", Var("a0"), ("result",))]
        expected = {"minimum": lambda: np.minimum(x, y), "negative": lambda: -x,
                    "abs": lambda: np.abs(x), "reciprocal": lambda: np.float32(1) / x}[operation]()
        if operation == "reciprocal":
            m["rtol"], m["atol"] = 2e-7, 1e-7
        emit(args.out, "arithmetic_" + operation, g, m, node, tile, body, expected)

    # Whole-warp participation, but each subgroup must get only its own sum.
    for width in (2, 4, 8, 16):
        g, m, node, tile = template(args.upstream)
        x = inputs(m)["x0"]
        combine = Assign("state", ElementwiseImpl("add"), ("state", "other"), F32)
        body = [read(("state",), "x0", (Var("a0"),), F32),
                WarpShuffle(("state",), ("other",), (combine,), width),
                write("add", Var("a0"), ("state",))]
        expected = np.repeat(x.reshape(-1, width).sum(axis=1, dtype=np.float64), width)
        m["rtol"], m["atol"] = 2e-6, 2e-6
        emit(args.out, f"shuffle_{width}", g, m, node, tile, body, expected)

    # Unique shared writers; all readers exchange values after the correct scope.
    for scope in ("warp", "named_cta"):
        g, m, node, tile = template(args.upstream)
        x = inputs(m)["x0"]
        index = Var("a0")
        if scope == "warp":
            source = binary("+", binary("*", binary("/", index, lit(32)), lit(32)),
                            binary("-", lit(31), binary("%", index, lit(32))))
            sync = Sync(warp=True)
            expected = x.reshape(-1, 32)[:, ::-1].reshape(-1)
        else:
            source = binary("-", lit(127), index)
            sync = Sync(barrier_id=7, count=128)
            expected = x[::-1]
        body = [Smem("exchange", (128,), "f32", 16),
                read(("value",), "x0", (index,), F32),
                write("exchange", index, ("value",)), sync,
                read(("other",), "exchange", (source,), F32),
                write("add", index, ("other",))]
        emit(args.out, "barrier_" + scope, g, m, node, tile, body, expected)

    g, m, node, tile = template(args.upstream, size=512)
    index = binary("*", Var("a0"), lit(4))
    body = [read(tuple(f"x{i}" for i in range(4)), "x0", (index,), F32),
            write("add", index, tuple(f"x{i}" for i in range(4)))]
    emit(args.out, "aligned_vector4", g, m, node, tile, body, inputs(m)["x0"])

    # Untyped Let must infer integer type from its expression, then bind an index.
    g, m, node, tile = template(args.upstream)
    body = [Let("reverse_index", binary("-", lit(127), Var("a0"))),
            read(("value",), "x0", (Var("reverse_index"),), F32),
            write("add", Var("a0"), ("value",))]
    emit(args.out, "let_integer_index", g, m, node, tile, body, inputs(m)["x0"][::-1])

    for operation in ("multiply", "minimum"):
        for size in (17, 257):
            g, m = load(args.upstream, "r01_array_sum")
            node = g.nodes["sum_1"]
            tile = node.op.body[0]
            op = ElementwiseImpl(operation)
            loop = tile.body[0]
            loop = replace(loop, axis=replace(loop.axis, extent=Dim(size)),
                           body=tuple(replace(s, op=op) if type(s).__name__ == "Accum" else s
                                      for s in loop.body))
            body = [loop]
            for s in tile.body[1:]:
                if type(s).__name__ in ("WarpShuffle", "TreeHalve"):
                    s = replace(s, combine_states=tuple(replace(a, op=op) for a in s.combine_states))
                body.append(s)
            node.op = replace(node.op, body=(replace(tile, body=tuple(body)),))
            reshape(g, {"x": [size]})
            arg = m["kernels"][0]["args"][0]
            arg["shape"] = [size]
            # Products stay finite and far from zero: empty-lane zero identity
            # would fail decisively instead of hiding behind an absolute tolerance.
            values = np.random.default_rng(317 + size).uniform(.98, 1.02, size).astype(np.float32)
            if operation == "minimum":
                values = np.random.default_rng(619 + size).uniform(2, 7, size).astype(np.float32)
            arg.pop("seed")
            arg["data"] = values.tolist()
            expected = (np.prod(values.astype(np.float64)) if operation == "multiply" else values.min())
            m["rtol"], m["atol"] = 2e-5, 1e-6
            m["code"] = f"Independent {operation} fold of {size} explicit f32 inputs."
            save(args.out, f"fold_{operation}_{size}", g, m,
                 {"sum_1": np.array([expected], dtype=np.float32)})


if __name__ == "__main__":
    main()
