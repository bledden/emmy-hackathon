"""Public entry point: load typed IR, bind metadata, resolve policy, emit PTX.

The compatibility imports below preserve the existing Python API. Internal
modules import their dependencies directly, never through this entry point.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .errors import CompileError, require, identifier
from .ir_analysis import (
    COLLECTIVES, S32_MIN, S32_MAX,
    extent,
    walk,
    loop_accumulators,
    written_names,
    definite_body_folds,
    literal_truth,
    definite_conditional_folds,
    warp_only_collectives,
    counted_region,
    uniform_trip_count,
    uniform_tile_names,
    uniform_expression,
    guaranteed_multiple,
)
from .scalars import BYTES, KINDS, PREFIX, instruction_type, literal as scalar_literal, f32
from .emitter import Emitter


def compile_file(path: Path, *, contract_fma: bool | str = False,
                 short_tree: bool = False, index_reuse: bool | str = False,
                 optimization_report=None, unroll_loops: bool = False,
                 constant_division: bool = False, shared_loads: str = 'off') -> str:
    require(type(contract_fma) is bool or contract_fma == "auto", "invalid FMA mode")
    require(type(short_tree) is bool, "invalid short-tree mode")
    require(type(unroll_loops) is bool, "invalid loop-unroll mode")
    require(type(constant_division) is bool, "invalid constant-division mode")
    require(shared_loads in ('off','address','vector','compact'), "invalid shared-load mode")
    require(type(index_reuse) is bool or (type(index_reuse) is str and index_reuse in ("loop", "auto")),
            "invalid index-reuse mode")
    # The required loader reconstructs typed statements; we never parse the repr.
    from emmy.compiler.graph import Graph

    graph = Graph.from_dict(json.loads(path.read_text()))
    nodes = [(nid, n) for nid, n in graph.nodes.items() if type(n.op).__name__ == "KernelOp"]
    require(len(nodes) == 1, "expected exactly one KernelOp")
    node_id, node = nodes[0]
    metadata_path = path.parent / "meta.json"
    require(metadata_path.is_file(), f"required adjacent metadata missing: {metadata_path}")
    meta = json.loads(metadata_path.read_text())
    matches = [k for k in meta["kernels"] if k["entry"] == node.op.name]
    require(matches, "meta.json has no matching entry")
    if len(matches) == 1:
        chosen = matches[0]
    else:
        # Emmy may reuse one entry at several sites. The per-entry IR file
        # describes one of those sites; bind its buffers, not the first site.
        same_node = [k for k in matches if k.get("node") == node_id]
        buffers = set(node.inputs) | {t.name for t in node.outputs}
        choices = same_node or [k for k in matches if set(k["arg_order"]) == buffers]
        require(choices, "no metadata launch site matches this KernelOp")
        require(len(choices) == 1, "ambiguous metadata for reused kernel")
        chosen = choices[0]

        def signature(k):
            require(all(type(n) is int for n in (*k["block"], *k["grid"])),
                    "launch dimensions must be integers")
            order, args = k["arg_order"], k["args"]
            require(len(order) == len(set(order)), "duplicate metadata argument")
            require(order == [a["name"] for a in args], "arg_order and args disagree")
            require(not any(a.get("scalar") for a in args),
                    "scalar-by-value arguments unsupported yet")
            # Names and input data may differ by launch site. The positional
            # buffer contract and geometry must be identical for this PTX.
            return (k["block"], k["grid"], k["smem_bytes"],
                    [(a["dtype"], a["shape"], bool(a.get("output"))) for a in args])

        expected = signature(chosen)
        require(all(signature(k) == expected for k in matches),
                "reused kernel launches disagree")
    automatic_fma = contract_fma == "auto"
    if contract_fma:
        tolerances = (meta.get("rtol"), meta.get("atol"))
        eligible = (all(type(t) in (int, float) and math.isfinite(t) and t >= 0
                        for t in tolerances) and tolerances[1] > 0)
        if contract_fma == "auto":
            contract_fma = eligible
        else:
            require(eligible, "FMA contraction requires finite nonnegative tolerances and positive atol")
    if index_reuse == "auto":
        # Stay within the reviewed policy combination. This is an evidence
        # boundary, not a mathematical dependence of integer CSE on FMA.
        enabled = bool(contract_fma) and not short_tree
        if optimization_report is not None:
            optimization_report.append({
                "pass": "index_reuse_policy", "mode": "auto",
                "action": "loop" if enabled else "disabled",
                "reason": ("guarded FMA enabled; short-tree disabled" if enabled
                           else "short-tree enabled" if short_tree else "FMA disabled or ineligible"),
            })
        index_reuse = "loop" if enabled else False
    emitter = Emitter(graph, node, chosen)
    emitter.contract_fma = contract_fma
    emitter.fma_reduction_only = automatic_fma
    emitter.short_tree = short_tree
    emitter.unroll_loops = unroll_loops
    emitter.constant_division = constant_division
    emitter.shared_loads = shared_loads
    emitter.index_reuse = index_reuse
    emitter.optimization_report = optimization_report
    return emitter.compile()
