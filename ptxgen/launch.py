"""Kernel ABI, physical coordinates, and final PTX assembly.

The driver in Emitter.compile makes the ordering explicit: validate the launch,
bind parameters, establish coordinates, emit the body, then finish and render.
"""
import math

from .errors import CompileError, require, identifier
from .ir_analysis import COLLECTIVES, S32_MAX, extent, walk, written_names, uniform_tile_names
from .scalars import BYTES, PREFIX, instruction_type


def validate_launch(e):
    require(not e.kernel.serial, "serial launch coordinates unsupported")
    require(len(e.kernel.body) == 1 and type(e.kernel.body[0]).__name__ == "Tile",
            "expected a single top-level Tile")
    tile = e.kernel.body[0]
    require(type(tile.aux_threads) is int and tile.aux_threads >= 0,
            "invalid auxiliary thread count")
    require(len({a.name for a in tile.axes}) == len(tile.axes), "duplicate Tile axis name")
    for axis in tile.axes:
        e.validate_axis(axis)
    block, grid = e.meta["block"], e.meta["grid"]
    require(len(block) == len(grid) == 3 and block[1:] == grid[1:] == [1, 1],
            "only one-dimensional launches supported")
    require(all(type(n) is int for n in (*block, *grid)), "launch dimensions must be integers")
    require(0 < block[0] <= 1024 and 0 < grid[0] <= S32_MAX, "invalid launch geometry")
    logical = block[0] - tile.aux_threads
    require(logical > 0 and (tile.block_threads is None or
            type(tile.block_threads) is int and tile.block_threads == logical),
            "launch block disagrees with Tile")
    require(not tile.aux_threads or tile.block_threads is not None,
            "auxiliary threads require explicit compute geometry")
    shape = [extent(a.extent) for a in tile.axes]
    require(all(0 < n <= S32_MAX + 1 for n in shape),
            "Tile axis values must fit signed 32-bit indices")
    cells = math.prod(shape)
    require(cells <= logical * grid[0], "launch does not cover Tile")
    wide_id = logical * grid[0] > S32_MAX + 1 or cells > S32_MAX
    require(isinstance(e.meta["smem_bytes"], int) and 0 <= e.meta["smem_bytes"] <= 99 * 1024,
            "dynamic shared memory exceeds runner limit")
    return tile, block, grid, logical, shape, cells, wide_id


def bind_parameters(e):
    order, args = e.meta["arg_order"], e.meta["args"]
    require(len(order) == len(set(order)), "duplicate metadata argument")
    require(order == [a["name"] for a in args], "arg_order and args disagree")
    tensors = {t.name: t for n in e.graph.nodes.values() for t in n.outputs}
    required = set(e.node.inputs) | {t.name for t in e.node.outputs}
    require(set(order) == required, "metadata arguments disagree with KernelOp buffers")
    parameters = []
    for index, (name, arg) in enumerate(zip(order, args)):
        identifier(name)
        require(not arg.get("scalar"), "scalar-by-value arguments unsupported yet")
        kind = e.dtype(arg["dtype"])
        require(kind == e.dtype(tensors[name].dtype), f"metadata dtype disagrees for {name}")
        buffer_shape = tuple(extent(n) for n in tensors[name].shape)
        # Scalar tensors have shape () in the graph but [1] in runner metadata.
        require(buffer_shape == tuple(arg["shape"]) or (not buffer_shape and arg["shape"] == [1]),
                f"metadata shape disagrees for {name}")
        parameters.append(f"    .param .u64 param_{index}")
        pointer = e.reg("u64")
        e.emit(f"ld.param.u64 {pointer}, [param_{index}];")
        e.emit(f"cvta.to.global.u64 {pointer}, {pointer};")
        e.buffers[name] = (pointer, buffer_shape, "global")
        e.buffer_kinds[name] = kind
    return parameters


def emit_coordinates(e, tile, block, grid, logical, shape, cells, wide_id):
    written_buffers = {s.output for s in walk(tile.body) if type(s).__name__ == 'Write'}
    written_buffers.update(s.dst for s in walk(tile.body) if type(s).__name__ == 'ZeroPrologue')
    written_buffers.update(s.dst_buffer for s in walk(tile.body) if type(s).__name__ == 'RegStore')
    e.readonly_global_inputs = frozenset(set(e.node.inputs) - written_buffers -
                                            {t.name for t in e.node.outputs})
    thread, cta, threads = e.reg("s32"), e.reg("s32"), e.reg("s32")
    e.thread = thread
    e.emit(f"mov.u32 {thread}, %tid.x;")
    e.emit(f"mov.u32 {cta}, %ctaid.x;")
    e.emit(f"mov.u32 {threads}, %ntid.x;")
    if wide_id:
        from .wide_tile import global_id
        gid = global_id(e, cta, thread, logical, tile.aux_threads)
    else:
        gid = (e.integer("+", e.integer("*", cta, str(logical)),
                            e.integer("%", thread, str(logical))) if tile.aux_threads else
               e.integer("+", e.integer("*", cta, threads), thread))
    e.emit(f"and.b32 {e.bind('lane', 's32')}, {thread}, 31;")
    e.emit(f"shr.u32 {e.bind('warp', 's32')}, {thread}, 5;")
    collectives = [type(s).__name__ for s in walk(tile.body)
                   if type(s).__name__ in COLLECTIVES]
    require(not collectives or cells % logical == 0,
            "partial-block collectives unsupported; refusing an unsafe barrier/shuffle mask")
    require(block[0] % 32 == 0 or all(tag == "WarpShuffle" for tag in collectives),
            "partial-warp blocks currently support only shuffle collectives")
    exit_label = e.label()
    if cells < logical * grid[0]:
        outside = e.reg("pred")
        e.emit(f"setp.ge.{'u64' if wide_id else 's32'} {outside}, {gid}, {cells};")
        e.emit(f"@{outside} bra {exit_label};")
    stride = cells
    for axis, size in zip(tile.axes, shape):
        stride //= size
        if wide_id:
            from .wide_tile import arithmetic, narrow
            value = narrow(e, arithmetic(e, '%', arithmetic(e, '/', gid, str(stride)), str(size)))
        else:
            value = e.integer("%", e.integer("/", gid, str(stride)), str(size))
        e.emit(f"mov.s32 {e.bind(axis.name, 's32')}, {value};")
    e.raster(tile, gid, shape)
    changed = written_names(tile.body) | set(tile.raster_axes or ())
    e.uniform_names = uniform_tile_names(tile.axes, logical, changed)
    # Global mixed-radix boundaries align to physical warp starts only if
    # the block is a multiple of 32. A partial block still has a builtin
    # warp coordinate, but its global axes need a separate proof.
    e.warp_uniform_names = (uniform_tile_names(tile.axes, 32, changed)
                               if block[0] % 32 == 0 and logical % 32 == 0 else set())
    if 'warp' not in changed | {a.name for a in tile.axes}:
        e.warp_uniform_names.add('warp')
        e.integer_ranges[e.read('warp', 's32')] = (0, (block[0] - 1) // 32)
    tile_names = {axis.name for axis in tile.axes}
    e.thread_layouts = {name: layout for name, layout in
                           (("lane", (1, 32)), ("warp", (32, block[0] // 32)))
                           if name not in changed | tile_names}
    # The global mixed-radix interval is conservative across all CTAs.
    # Exclude raster overrides and every coordinate written anywhere in
    # the body; no branch merging, alias or induction analysis is assumed.
    e.coordinate_ranges = {axis.name: (0, size - 1)
                              for axis, size in zip(tile.axes, shape)
                              if axis.name not in changed}
    stride = cells
    for axis, size in zip(tile.axes, shape):
        stride //= size
        if axis.name not in changed:
            e.coordinate_decodes[e.read(axis.name, 's32')] = (stride, size, logical, block[0])
        if axis.name not in changed and logical % (stride * size) == 0:
            e.thread_layouts[axis.name] = (stride, size)
    # Rasterization is a bounded permutation of these same coordinates.
    # It invalidates physical-neighbour/uniformity facts, not numeric bounds.
    # Keep those proof domains separate; body writes still exclude a name.
    changed_indices = written_names(tile.body)
    for axis, size in zip(tile.axes, shape):
        if axis.name not in changed_indices:
            e.integer_ranges[e.read(axis.name, "s32")] = (0, size - 1)
    e.active_threads = frozenset(range(block[0]))
    for name, (stride, size) in e.thread_layouts.items():
        e.thread_values[e.read(name, 's32')] = tuple(
            t // 32 if name == 'warp' and name not in tile_names else (t // stride) % size
            for t in range(block[0]))
    return exit_label


def finish_body(e, tile, exit_label):
    from .barrier_lifetimes import validate as validate_barrier_lifetimes
    try:
        validate_barrier_lifetimes(e.barrier_trace)
    except ValueError as error:
        raise CompileError(str(error)) from error
    if any(type(s).__name__ == "CpAsyncCopy" for s in walk(tile.body)):
        e.emit("cp.async.wait_all;")
    e.emit(exit_label + ":")
    e.emit("ret;")
    require(e.meta["smem_bytes"] or
            sum(math.prod(s) * BYTES[e.buffer_kinds[name]]
                for name, (_, s, space) in e.buffers.items() if space == "shared") <= 49152,
            "static shared memory exceeds 48 KiB")


def render(e, parameters):
    header = [".version 8.0", ".target sm_89", ".address_size 64", ""]
    if e.dynamic_smem_used:
        header.append(f".extern .shared .align {e.dynamic_smem_align} .b8 sm_dynamic[];")
    header += [
              f".visible .entry {identifier(e.kernel.name)}(", ",\n".join(parameters), ")", "{"]
    code = e.code
    if e.contract_fma:
        from .optimizations import contract_fma
        code = contract_fma(code, allowed_adds=(e.fma_accumulation_lines
                            if e.fma_reduction_only else None))
    if e.index_reuse:
        from .index_reuse import reuse_integer_expressions, reuse_loop_integer_expressions
        reuse = reuse_loop_integer_expressions if e.index_reuse == "loop" else reuse_integer_expressions
        code = reuse(code, e.optimization_report)
    if e.shared_loads in ('vector','compact'):
        from .shared_loads import combine
        code = combine(code, e.shared_address_info, e.optimization_report)
    if e.unroll_loops:
        from .loop_unroll import expand_annotated_loops
        options = (dict(allocate=e.reg, register_kinds=e.register_kinds,
                        address_proofs=e.shared_address_info) if e.shared_loads in ('vector','compact') else {})
        code = expand_annotated_loops(code, e.unroll_regions, e.optimization_report, **options)
        if e.shared_loads in ('vector','compact'):
            code = combine(code, e.shared_address_info, e.optimization_report)
    if e.shared_loads in ('address','compact'):
        from .shared_loads import rebase
        code = rebase(code, e.shared_address_info, e.optimization_report)
    if e.constant_division:
        from .constant_division import lower
        code = lower(code, e.reg, e.optimization_report)
    declarations = []
    for kind, count in e.counts.items():
        prefix = PREFIX[kind]
        declarations.append(f"    .reg .{instruction_type(kind)} %{prefix}<{count}>;")
    result = "\n".join(header + declarations + e.declarations + code + ["}", ""])
    require(result.isascii(), "non-ASCII PTX")
    return result
