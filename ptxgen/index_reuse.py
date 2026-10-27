"""Exact local reuse of integer expressions in this emitter's PTX subset.

No load reuse, code motion, range assumptions, or floating-point rewriting.
Unknown instructions and control-flow boundaries discard all facts.
"""
from collections import Counter
import re

INTEGER = re.compile(r'%r(?:d)?[0-9]+\Z')
DECIMAL = re.compile(r'-?[0-9]+\Z')
# Opcode -> (result width, operand width). Exact spellings retain type/rounding.
PURE = {op: (32, 32) for op in (
    'add.s32', 'sub.s32', 'mul.lo.s32', 'and.b32', 'or.b32', 'xor.b32',
    'shl.b32', 'shr.s32', 'shr.u32', 'div.s32', 'rem.s32')}
PURE.update({'mul.wide.s32': (64, 32), 'add.u64': (64, 64)})
# These instructions have one explicit destination. They are NOT reused;
# integer destinations receive fresh identities. No implicit CC state is used.
SINGLE = {
    'mov.s32': 2, 'mov.u32': 2, 'mov.u64': 2, 'mov.b32': 2, 'mov.b64': 2,
    'cvt.u32.u64': 2, 'cvta.to.global.u64': 2, 'ld.param.u64': 2,
    'selp.s32': 4, 'selp.b32': 4, 'min.s32': 3,
    'mov.f32': 2, 'abs.f32': 2, 'neg.f32': 2, 'ex2.approx.f32': 2,
    'lg2.approx.f32': 2, 'rcp.rn.f32': 2, 'sqrt.rn.f32': 2,
    'rsqrt.approx.f32': 2, 'cvt.rmi.f32.f32': 2, 'cvt.rni.f32.f32': 2,
    'cvt.rpi.f32.f32': 2, 'cvt.rzi.f32.f32': 2,
    'add.rn.f32': 3, 'sub.rn.f32': 3, 'mul.rn.f32': 3, 'div.rn.f32': 3,
    'max.NaN.f32': 3, 'min.NaN.f32': 3, 'copysign.f32': 3,
    'fma.rn.f32': 4, 'selp.f32': 4, 'and.pred': 3, 'or.pred': 3,
}
for suffix in ('eq.f32', 'eq.s32', 'ge.f32', 'ge.s32', 'gt.f32', 'gt.s32',
               'le.f32', 'le.s32', 'lt.f32', 'lt.s32', 'ne.s32', 'ne.f32',
               'neu.f32', 'num.f32'):
    SINGLE['setp.' + suffix] = 3
DESTINATION = re.compile(r'%(?:rd|r|f|p)[0-9]+\Z')
FLOAT_MEMORY = re.compile(
    r'(?:ld|st)\.(?:global|shared)(?:\.v[24])?\.f32\Z')


def register_width(name):
    return 64 if name.startswith('%rd') else 32


def candidate(opcode, args):
    if opcode not in PURE or len(args) != 3:
        return False
    result_width, operand_width = PURE[opcode]
    if not INTEGER.fullmatch(args[0]) or register_width(args[0]) != result_width:
        return False
    for operand in args[1:]:
        if INTEGER.fullmatch(operand):
            if register_width(operand) != operand_width:
                return False
        elif not DECIMAL.fullmatch(operand):
            return False
        elif not -(1 << (operand_width - 1)) <= int(operand) < (1 << operand_width):
            return False
    if opcode in ('div.s32', 'rem.s32'):
        # Do not reuse zero/unknown/negative-divisor exceptional operations.
        # This retains the exact signed opcode, including its negative-input
        # behavior; it does not replace division with a shift or change rounding.
        return bool(DECIMAL.fullmatch(args[2]) and 0 < int(args[2]) < 2**31)
    if opcode.startswith(('shl.', 'shr.')):
        return bool(DECIMAL.fullmatch(args[2]) and 0 <= int(args[2]) < 32)
    return True


def reuse_integer_expressions(code, report=None):
    """Replace repeated integer calculations with width-matched register moves.

    Keys use operand value identities, not register spellings. A representative
    must still hold that value. Preserve every destination assignment and every
    non-replaced instruction. Predicates, labels, collectives and unrecognized
    instructions are hard boundaries; nothing moves across them.
    """
    values, holders, available = {}, {}, {}
    serial = 0
    output, changes = [], []

    def fresh():
        nonlocal serial
        serial += 1
        return serial

    def assign(register, number):
        previous = values.get(register)
        if previous is not None:
            holders[previous].discard(register)
        values[register] = number
        holders.setdefault(number, set()).add(register)

    def value(operand):
        if INTEGER.fullmatch(operand):
            if operand not in values:
                assign(operand, fresh())
            return ('value', values[operand])
        return ('literal', operand)

    def clear():
        values.clear()
        holders.clear()
        available.clear()

    for line_number, line in enumerate(code, 1):
        text = line.strip()
        if not text.endswith(';') or text.startswith(('@', '.', '//')):
            clear()
            output.append(line)
            continue
        words = text[:-1].split(None, 1)
        opcode = words[0] if words else ''
        args = [a.strip() for a in words[1].split(',')] if len(words) == 2 else []
        if candidate(opcode, args):
            key = (opcode, tuple(value(arg) for arg in args[1:]))
            number = available.get(key)
            representatives = holders.get(number, set())
            if representatives:
                representative = min(representatives)
                width = PURE[opcode][0]
                replacement = f'    mov.b{width} {args[0]}, {representative};'
                changes.append({'line': line_number, 'before': line, 'after': replacement})
                line = replacement
            else:
                number = fresh()
                available[key] = number
            # Read inputs and find representatives before overwriting destination.
            assign(args[0], number)
        elif opcode in PURE and args and INTEGER.fullmatch(args[0]):
            # Known arithmetic, but not eligible (e.g. variable division).
            # Clear rather than infer anything from a partially parsed form.
            clear()
        elif (opcode in SINGLE and len(args) == SINGLE[opcode]
              and DESTINATION.fullmatch(args[0])):
            if INTEGER.fullmatch(args[0]):
                assign(args[0], fresh())
        elif FLOAT_MEMORY.fullmatch(opcode):
            # Stores write no registers. Loads may only preserve integer facts
            # when ALL their destinations are floating registers, including vectors.
            if opcode.startswith('ld.'):
                match = re.fullmatch(
                    r'ld\.(?:global|shared)(?:\.v([24]))?\.f32\s+'
                    r'(%f[0-9]+|\{(?:%f[0-9]+, )*%f[0-9]+\}), \[%rd[0-9]+\];', text)
                if (not match or len(re.findall(r'%f[0-9]+', match[2]))
                        != int(match[1] or 1)):
                    clear()
        else:
            clear()
        output.append(line)
    if report is not None:
        report.append({'pass': 'index_reuse', 'replacements': len(changes),
                       'by_opcode': dict(Counter(c['before'].strip().split()[0] for c in changes)),
                       'changes': changes})
    return output


def reuse_loop_integer_expressions(code, report=None):
    """Select full exact reuse when it removes div/rem in a backward-edge span.

    This is a structural scoping rule, not a profitability, reachability or trip-count
    proof. It reads emitted instructions only: no kernel names, metadata,
    dimensions or timing lookup. The transformation is the existing full pass;
    selecting a kernel never expands that pass's correctness assumptions.
    """
    details = []
    optimized = reuse_integer_expressions(code, details)
    result = details[0]
    labels, edges = {}, []
    valid = True
    label_pattern = r'[A-Za-z_$][A-Za-z_0-9$]*'
    branch_pattern = rf'(?:@!?%p[0-9]+\s+)?bra(?:\.uni)?\s+({label_pattern});'
    for number, line in enumerate(code, 1):
        match = re.fullmatch(rf'({label_pattern}):', line.strip())
        if match:
            if match[1] in labels:
                valid = False
            labels[match[1]] = number
    for number, line in enumerate(code, 1):
        text = line.strip()
        match = re.fullmatch(branch_pattern, text)
        if match:
            if match[1] not in labels:
                valid = False
            elif labels[match[1]] < number:
                edges.append((labels[match[1]], number))
        elif re.search(r'\bbra(?:\.|\s)', text):
            # Do not infer a loop from an unsupported branch spelling.
            valid = False
    witnesses = [change['line'] for change in result['changes']
                 if change['before'].strip().split()[0] in ('div.s32', 'rem.s32')
                 and any(start < change['line'] < end for start, end in edges)]
    selected = valid and bool(witnesses)
    if report is not None:
        report.append({**result, 'policy': 'loop', 'selected': selected,
                       'considered_replacements': result['replacements'],
                       'backward_edges': edges, 'loop_divrem_lines': witnesses,
                       'recognized_control_flow': valid,
                       'replacements': result['replacements'] if selected else 0,
                       'by_opcode': result['by_opcode'] if selected else {},
                       'changes': result['changes'] if selected else []})
    return optimized if selected else list(code)
