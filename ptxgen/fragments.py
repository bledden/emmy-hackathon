"""One-warp dense tensor-core lowering, with a deliberately closed contract.

The global-load variant of Emmy's LdmatrixLoad fills PTX register fragments;
it does not issue a shared-memory ldmatrix instruction. Lane packing follows
PTX ISA 8.8, section 9.7.14.5.8 (mma.m16n8k16 floating-point fragments).
This module never promotes an ordinary scalar matmul to reduced precision.
"""
from dataclasses import dataclass, fields

from .errors import require, identifier
from .ir_analysis import S32_MAX, walk

TAGS = frozenset(('RegFragment', 'LdmatrixLoad', 'MmaSyncPtx', 'RegStore'))
SHAPE = (16, 8, 16)


def preflight(e, tile, block, grid, cells):
    if not any(type(s).__name__ in TAGS for s in walk(tile.body)):
        return
    require(all(type(s).__name__ in TAGS for s in tile.body),
            'tensor fragments require a straight-line fragment-only Tile')
    require(block == [32, 1, 1] and grid == [1, 1, 1] and cells == 32
            and tile.aux_threads == 0 and tile.raster_axes is None and tile.raster_group is None,
            'tensor fragments require one complete 32-thread CTA without rasterization')
    require(not ({a.name for a in tile.axes} & {'lane', 'warp'}),
            'tensor fragment Tile axes cannot shadow lane or warp')
    require(not (set(e.node.inputs) & {t.name for t in e.node.outputs}),
            'tensor fragment inputs and outputs must be distinct buffers')
    e.fragments = {}
    e.fragment_stores = []


def contract(stmt, required, defaults):
    """Fail closed on new schema fields as well as unsupported nondefault values."""
    require({f.name for f in fields(stmt)} == set(required) | defaults.keys(),
            f'unsupported {type(stmt).__name__} schema')
    for name, expected in defaults.items():
        value = getattr(stmt, name)
        require(type(value) is type(expected) and value == expected,
                f'{type(stmt).__name__}.{name} unsupported: {value!r}')


def shape(value):
    require(type(value) is tuple and value == SHAPE and all(type(n) is int for n in value),
            'tensor fragments support only m16n8k16')


@dataclass
class Fragment:
    role: str
    registers: tuple
    ready: bool = False


def get(e, name, role, *, ready=True):
    require(name in e.fragments, f'undeclared tensor fragment {name}')
    fragment = e.fragments[name]
    require(fragment.role == role, f'tensor fragment {name} requires role {role}')
    require(not ready or fragment.ready, f'tensor fragment {name} used before load')
    return fragment


def region(e, name, indices, ldm, rows, cols, kind, *, output=False):
    allowed = {t.name for t in e.node.outputs} if output else set(e.node.inputs)
    require(name in allowed and name in e.buffers, 'tensor fragment buffer has wrong direction')
    base, dimensions, space = e.buffers[name]
    require(space == 'global' and len(dimensions) == 2 and e.buffer_kinds[name] == kind,
            f'tensor fragment requires a rank-2 global {kind} buffer')
    require(type(ldm) is int and ldm in (0, dimensions[1]),
            'tensor fragment leading dimension must match physical row stride')
    require(len(indices) == 2 and all(type(x).__name__ == 'Literal' and
            type(x.value) is int and x.dtype in ('int', 'i32') for x in indices),
            'tensor fragment base indices must be constant s32 literals')
    row, col = (x.value for x in indices)
    require(row >= 0 and col >= 0 and row + rows <= dimensions[0]
            and col + cols <= dimensions[1], 'tensor fragment tile is out of bounds')
    require((row + rows - 1) * dimensions[1] + col + cols - 1 <= S32_MAX,
            'tensor fragment flattened index exceeds s32')
    return base, dimensions[1], row, col


def address(e, region_info, row, col, size):
    base, stride, r0, c0 = region_info
    row = e.integer('+', row, str(r0))
    col = e.integer('+', col, str(c0))
    flat = e.integer('+', e.integer('*', row, str(stride)), col)
    offset, pointer = e.reg('u64'), e.reg('u64')
    e.emit(f'mul.wide.s32 {offset}, {flat}, {size};')
    e.emit(f'add.u64 {pointer}, {base}, {offset};')
    return pointer


def coordinates(e):
    # Use the physical thread register, never a possibly rebound IR name.
    return e.integer('/', e.thread, '4'), e.integer('*', e.integer('%', e.thread, '4'), '2')


def declare(e, stmt):
    contract(stmt, ('name', 'role', 'shape', 'dtype'), {'count': 1, 'nregs': None})
    shape(stmt.shape)
    identifier(stmt.name)
    require(stmt.name not in e.fragments and stmt.name not in e.values,
            f'duplicate tensor fragment {stmt.name}')
    require(stmt.role in ('a', 'b', 'c'), 'unknown tensor fragment role')
    kind = 'f32' if stmt.role == 'c' else 'f16'
    require(e.dtype(stmt.dtype) == kind, f'tensor fragment {stmt.role} requires {kind}')
    regs = tuple(e.reg('f32' if stmt.role == 'c' else 'b32')
                 for _ in range(2 if stmt.role == 'b' else 4))
    e.fragments[stmt.name] = Fragment(stmt.role, regs, stmt.role == 'c')
    if stmt.role == 'c':
        for reg in regs:
            e.emit(f'mov.f32 {reg}, 0f00000000;')


def load(e, stmt):
    contract(stmt, ('frag', 'src_buffer', 'src_index', 'role', 'ldm'), {
        'swizzle': 'NONE', 'staged': False, 'gmem_guard': None, 'k_zero': None,
        'b_trans': False, 'pair_frag': None, 'byte_slab': False, 'scale_buffer': None,
        'scale_index': (), 'scale_ldm': 0, 'fragment_layout': 'm16n8k16', 'fragment_index': 0})
    require(stmt.role in ('a', 'b'), 'tensor fragment load requires role a or b')
    frag = get(e, stmt.frag, stmt.role, ready=False)
    info = region(e, stmt.src_buffer, stmt.src_index, stmt.ldm,
                  16, 16 if stmt.role == 'a' else 8, 'f16')
    group, column_pair = coordinates(e)
    for j, packed in enumerate(frag.registers):
        halves = []
        for half in range(2):
            if stmt.role == 'a':
                row = e.integer('+', group, str(8 * (j % 2)))
                col = e.integer('+', column_pair, str(8 * (j // 2) + half))
            else:
                row = e.integer('+', column_pair, str(8 * j + half))
                col = group
            pointer = address(e, info, row, col, 2)
            value = e.reg('f16')
            e.emit(f'ld.global.b16 {value}, [{pointer}];')
            halves.append(value)
        e.emit(f'mov.b32 {packed}, {{{", ".join(halves)}}};')
    frag.ready = True


def mma(e, stmt):
    contract(stmt, ('c_frag', 'a_frag', 'b_frag', 'shape'), {
        'ab_dtype': 'f16', 'c_dtype': 'f32', 'b_row_major': False,
        'sfa_frag': None, 'sfb_frag': None})
    shape(stmt.shape)
    c, a, b = (get(e, name, role) for name, role in
               ((stmt.c_frag, 'c'), (stmt.a_frag, 'a'), (stmt.b_frag, 'b')))
    operands = ['{' + ', '.join(f.registers) + '}' for f in (c, a, b, c)]
    e.emit('mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ' + ', '.join(operands) + ';')


def store(e, stmt):
    contract(stmt, ('dst_buffer', 'dst_index', 'frag', 'shape', 'ldm'), {
        'epilogue': None, 'extra_frags': (), 'm_guard': None, 'n_guard': None,
        'atomic': False, 'fragment_layout': 'm16n8k16', 'volta_interleaved': False,
        'fragment_index': (0, 0), 'swizzle': 'NONE', 'row_dim': None, 'col_dim': None})
    shape(stmt.shape)
    frag = get(e, stmt.frag, 'c')
    info = region(e, stmt.dst_buffer, stmt.dst_index, stmt.ldm, 16, 8, 'f32', output=True)
    _, _, row, col = info
    for buffer, r, c in e.fragment_stores:
        require(buffer != stmt.dst_buffer or row + 16 <= r or r + 16 <= row
                or col + 8 <= c or c + 8 <= col, 'overlapping tensor fragment stores unsupported')
    e.fragment_stores.append((stmt.dst_buffer, row, col))
    group, column_pair = coordinates(e)
    for j, value in enumerate(frag.registers):
        r = e.integer('+', group, str(8 * (j // 2)))
        c = e.integer('+', column_pair, str(j % 2))
        pointer = address(e, info, r, c, 4)
        e.emit(f'st.global.f32 [{pointer}], {value};')


def statement(e, stmt):
    require(hasattr(e, 'fragments'), 'tensor fragments require validated kernel context')
    {'RegFragment': declare, 'LdmatrixLoad': load,
     'MmaSyncPtx': mma, 'RegStore': store}[type(stmt).__name__](e, stmt)
