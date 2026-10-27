"""Pinned Kernel IR software slab layouts, in element-index units.

The modern B modes XOR bits from the selected row field into bits3..5.
No renderer/helper is called; this module emits ordinary integer PTX.
"""
import re


def parse(mode):
    if mode is None or mode == 'NONE':
        return None
    if mode in ('V70A', 'V70B'):
        return mode
    match = re.fullmatch(r'(B32|B64|B128)(?:@([0-9]+))?', mode) if isinstance(mode,str) else None
    if match is None:
        raise ValueError(f'unsupported software swizzle {mode!r}')
    shift = int(match[2]) if match[2] is not None else 6
    if not 0 <= shift <= 31:
        raise ValueError('software swizzle shift must be in [0, 31]')
    return shift, {'B32':1,'B64':3,'B128':7}[match[1]]


def emit(emitter, flat, layout, shape=()):
    if layout is None:
        return flat
    if isinstance(layout, str):
        return volta(emitter, flat, layout, shape)
    shift, mask = layout
    row = emitter.integer('>>', flat, str(shift))
    chunk = emitter.integer('&', row, str(mask))
    delta = emitter.integer('<<', chunk, '3')
    return emitter.integer('^', flat, delta)


def volta(emitter, flat, layout, shape):
    """Map the last two slab axes in element units, preserving leading slabs.

    V70A exchanges four-element column groups with permuted row bits;
    V70B exchanges row bits 0:1 with column bits 3:4. These are address
    permutations only: no tensor-core instruction or external compiler is used.
    """
    if len(shape) < 2:
        raise ValueError('V70 layout requires at least two shared slab dimensions')
    rows, cols = shape[-2:]
    complete = (rows % 16 == 0 and cols % 4 == 0 if layout == 'V70A'
                else rows % 4 == 0 and (cols % 32 == 0 or cols == 8))
    if rows <= 0 or cols <= 0 or not complete:
        raise ValueError(f'{layout} requires a complete bijective shared slab shape')
    op = emitter.integer
    def bit(value, position):
        return op('&', op('>>', value, str(position)), '1')
    slab_size = rows * cols
    leading = op('*', op('/', flat, str(slab_size)), str(slab_size))
    local = op('%', flat, str(slab_size))
    row, col = op('/', local, str(cols)), op('%', local, str(cols))
    if layout == 'V70A':
        p = op('&', row, '-16')
        for position, value in enumerate((op('^', bit(row, 2), bit(col, 3)),
                op('^', op('^', bit(row, 3), bit(row, 4)), bit(col, 4)),
                bit(row, 0), bit(row, 1))):
            p = op('|', p, op('<<', value, str(position)))
        offset = op('+', op('*', p, '4'), op('&', col, '3'))
        offset = op('+', offset, op('*', op('/', col, '4'), str(rows * 4)))
    else:
        r = op('|', op('&', row, '-4'), op('&', op('>>', col, '3'), '3'))
        c = op('&', col, '-25')  # retain all but column bits 3 and 4
        for position in (0, 1):
            v = op('^', bit(row, position), bit(col, position + 3))
            c = op('|', c, op('<<', v, str(position + 3)))
        offset = op('+', op('*', r, str(cols)), c)
    return op('+', leading, offset)


def preserves_alignment(layout, shape, width):
    """Whether a logically aligned start remains aligned after relocation."""
    if not isinstance(layout, str):
        return True
    low_run = 4 if layout == 'V70A' else 8
    return len(shape) >= 2 and width <= low_run and shape[-1] % width == 0
