"""Changed-shape validation for online reductions, rasterization and async copies.

Typed fixture mutation only; independent NumPy references, no Emmy renderer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.ir.expr import BinaryExpr, Literal

from make_variants import inputs, load, reshape, save


def shapes(graph, meta, mapping):
    reshape(graph, mapping)
    for arg in meta["kernels"][0]["args"]:
        if arg["name"] in mapping:
            arg["shape"] = mapping[arg["name"]]
        if "seed" in arg:
            arg["seed"] += 73


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    for columns in (17, 257, 4097):
        graph, meta = load(args.upstream, "f02_row_softmax")
        node = graph.nodes["softmax"]
        tile = node.op.body[0]
        body = tuple(replace(s, axis=replace(s.axis, extent=Dim(columns)))
                     if type(s).__name__ == "StridedLoop" else s for s in tile.body)
        axes = (replace(tile.axes[0], extent=Dim(3)), tile.axes[1])
        node.op = replace(node.op, body=(replace(tile, axes=axes, body=body),))
        shapes(graph, meta, {"x": [3, columns], "softmax": [3, columns]})
        meta["kernels"][0]["grid"] = [3, 1, 1]
        x = inputs(meta)["x"].astype(np.float64)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        save(args.out, f"softmax_{columns}", graph, meta,
             {"softmax": (e / e.sum(axis=1, keepdims=True)).astype(np.float32)})

    # Both raster axes vary; both orientations have incomplete final stripes.
    for orient in ("m", "n"):
        graph, meta = load(args.upstream, "a02_attention_128")
        node = graph.nodes["matmul_1"]
        tile = node.op.body[0]
        axes = (replace(tile.axes[0], extent=Dim(10)), replace(tile.axes[1], extent=Dim(5)),
                *tile.axes[2:])
        node.op = replace(node.op, body=(replace(tile, axes=axes, raster_group=3,
                                               raster_orient=orient),))
        shapes(graph, meta, {"q": [10, 32], "v": [128, 160], "matmul_1": [10, 160]})
        meta["kernels"][0]["grid"] = [50, 1, 1]
        data = inputs(meta)
        scale = np.float32(meta["kernels"][0]["args"][0]["data"][0])
        scores = data["q"].astype(np.float64) @ data["kt"].astype(np.float64) / float(scale)
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        result = (exps / exps.sum(axis=1, keepdims=True)) @ data["v"].astype(np.float64)
        save(args.out, f"attention_raster_{orient}", graph, meta,
             {"matmul_1": result.astype(np.float32)})

    # Preserve the staged K loop but change both output tile counts and seeds.
    graph, meta = load(args.upstream, "m01_matmul_512")
    node = graph.nodes["matmul"]
    tile = node.op.body[0]
    axes = (replace(tile.axes[0], extent=Dim(3)), replace(tile.axes[1], extent=Dim(3)),
            *tile.axes[2:])
    node.op = replace(node.op, body=(replace(tile, axes=axes),))
    shapes(graph, meta, {"x0": [96, 512], "x1": [512, 192], "matmul": [96, 192]})
    meta["kernels"][0]["grid"] = [9, 1, 1]
    data = inputs(meta)
    result = data["x0"].astype(np.float64) @ data["x1"].astype(np.float64)
    save(args.out, "matmul_staged_rect", graph, meta, {"matmul": result.astype(np.float32)})

    # Misaligned two-value loads/stores, with an untouched zero halo at each end.
    graph, meta = load(args.upstream, "p00_scalar_add")
    node = graph.nodes["add"]
    tile = node.op.body[0]
    body = []
    for s in tile.body:
        tag = type(s).__name__
        if tag == "Load":
            if s.names[0].endswith("__u1"):
                continue
            s = replace(s, names=(s.names[0], s.names[0].replace("__u0", "__u1")))
        if tag in ("Load", "Write"):
            s = replace(s, index=(BinaryExpr("+", s.index[0], Literal(1, "int")),))
        body.append(s)
    node.op = replace(node.op, body=(replace(tile, body=tuple(body)),))
    shapes(graph, meta, {n: [130] for n in ("x0", "x1", "add")})
    meta["kernels"][0]["zero_outputs"] = ["add"]
    data = inputs(meta)
    result = np.zeros(130, dtype=np.float32)
    result[1:129] = data["x0"][1:129] + data["x1"][1:129]
    save(args.out, "misaligned_vectors", graph, meta, {"add": result})


if __name__ == "__main__":
    main()
