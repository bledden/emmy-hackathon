"""Create independent test shapes from typed IR, outside the submission repo.

No Emmy renderer or compiler is called. This only changes data objects, serializes
them, and computes NumPy expectations. Inputs are the organizer's open fixtures.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph


def load(root, case):
    folder = root / "problems" / case
    graph = Graph.from_dict(json.loads(next(folder.glob("*.ir.json")).read_text()))
    return graph, json.loads((folder / "meta.json").read_text())


def save(root, name, graph, meta, expected):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=False)
    node = next(n for n in graph.nodes.values() if type(n.op).__name__ == "KernelOp")
    node.op = replace(node.op, name="variant_" + name)
    meta["id"] = name
    meta["kernels"][0]["entry"] = node.op.name
    (folder / (node.op.name + ".ir.json")).write_text(json.dumps(graph.to_dict(), default=repr, indent=2))
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    np.savez(folder / "expected.npz", **expected)


def reshape(graph, shapes):
    for node in graph.nodes.values():
        node.outputs = tuple(replace(t, shape=tuple(Dim(n) for n in shapes[t.name]))
                             if t.name in shapes else t for t in node.outputs)


def inputs(meta):
    return {arg["name"]: np.random.default_rng(arg["seed"]).standard_normal(arg["shape"]).astype(np.float32)
            for arg in meta["kernels"][0]["args"] if "seed" in arg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    # Original 64-cell computation, but twice as many launched threads.
    graph, meta = load(args.upstream, "p00_scalar_add")
    meta["kernels"][0]["block"] = [512, 1, 1]
    data = inputs(meta)
    save(args.out, "oversized_block", graph, meta, {"add": data["x0"] + data["x1"]})

    # New length, partial final CTA, renamed entry, and reversed positional ABI.
    graph, meta = load(args.upstream, "p00_scalar_add")
    node = graph.nodes["add"]
    tile = node.op.body[0]
    node.op = replace(node.op, body=(replace(tile, axes=(replace(tile.axes[0], extent=Dim(70)),)),))
    reshape(graph, {n: [140] for n in ("x0", "x1", "add")})
    kernel = meta["kernels"][0]
    kernel["block"], kernel["grid"] = [32, 1, 1], [3, 1, 1]
    for arg in kernel["args"]:
        arg["shape"] = [140]
        if "seed" in arg:
            arg["seed"] += 41
    kernel["arg_order"].reverse()
    kernel["args"].reverse()
    data = inputs(meta)
    save(args.out, "tail_reordered_abi", graph, meta, {"add": data["x0"] + data["x1"]})

    # Convert pairs of scalar loads to typed vector loads, preserving computation.
    graph, meta = load(args.upstream, "p00_scalar_add")
    node = graph.nodes["add"]
    tile = node.op.body[0]
    body = []
    for stmt in tile.body:
        if type(stmt).__name__ == "Load":
            if stmt.names[0].endswith("__u1"):
                continue
            stmt = replace(stmt, names=(stmt.names[0], stmt.names[0].replace("__u0", "__u1")))
        body.append(stmt)
    node.op = replace(node.op, body=(replace(tile, body=tuple(body)),))
    data = inputs(meta)
    save(args.out, "vector_loads", graph, meta, {"add": data["x0"] + data["x1"]})

    # Most threads have zero loop trips, but every thread reaches the collective.
    graph, meta = load(args.upstream, "r01_array_sum")
    node = graph.nodes["sum_1"]
    tile = node.op.body[0]
    loop = tile.body[0]
    loop = replace(loop, axis=replace(loop.axis, extent=Dim(17)))
    node.op = replace(node.op, body=(replace(tile, body=(loop, *tile.body[1:])),))
    reshape(graph, {"x": [17]})
    meta["kernels"][0]["args"][0]["shape"] = [17]
    data = inputs(meta)
    save(args.out, "short_reduction", graph, meta,
         {"sum_1": np.asarray(data["x"].sum(dtype=np.float64), dtype=np.float32)})

    # Non-power-of-two row length with a final single element in the stride loop.
    graph, meta = load(args.upstream, "r04_row_sum")
    node = graph.nodes["sum_1"]
    tile = node.op.body[0]
    loop = tile.body[0]
    loop = replace(loop, axis=replace(loop.axis, extent=Dim(257)))
    axes = (replace(tile.axes[0], extent=Dim(3)), tile.axes[1])
    node.op = replace(node.op, body=(replace(tile, axes=axes, body=(loop, *tile.body[1:])),))
    reshape(graph, {"x": [3, 257], "sum_1": [3]})
    kernel = meta["kernels"][0]
    kernel["grid"] = [3, 1, 1]
    for arg in kernel["args"]:
        arg["shape"] = [3, 257] if arg["name"] == "x" else [3]
    data = inputs(meta)
    save(args.out, "odd_rows", graph, meta,
         {"sum_1": data["x"].sum(axis=1, dtype=np.float64).astype(np.float32)})


if __name__ == "__main__":
    main()
