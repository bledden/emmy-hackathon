"""Exact signed-32 truncating division by positive compile-time constants.

For non-powers of two, m=ceil(2**32/d). For a=abs(n)<=2**31,
q=high32(a*m) is floor(a/d) or one greater: multiplication error is
less than a/2**32 <= 1/2. Comparing q*d with a corrects the latter.
q*d <= a+d < 2**32, so the unsigned comparison does not wrap.
Sign restoration is modulo 2**32, including n=INT_MIN. Emmy floor
semantics remain the responsibility of the caller's existing correction.
"""
import re
from types import SimpleNamespace


def lower(code, allocate, report=None):
    """Run after reuse selection, retaining all existing optimization decisions."""
    result, changes = [], []
    builder = SimpleNamespace(reg=allocate, emit=lambda text: result.append('    '+text))
    pattern = re.compile(r'\s*(div|rem)\.s32 (%r[0-9]+), (%r[0-9]+|-?[0-9]+), ([0-9]+);')
    for line in code:
        match = pattern.fullmatch(line)
        if match is None or not 2 <= int(match[4]) <= 2**31-1:
            result.append(line)
            continue
        value = emit(builder, match[3], int(match[4]), remainder=match[1]=='rem')
        builder.emit(f'mov.s32 {match[2]}, {value};')
        changes.append({'opcode':match[1], 'divisor':int(match[4])})
    if report is not None:
        report.append({'pass':'constant_division','replacements':len(changes),'changes':changes})
    return result


def emit(e, numerator, divisor, *, remainder=False):
    assert type(divisor) is int and 2 <= divisor <= 2**31-1
    sign = e.reg('s32')
    e.emit(f'shr.s32 {sign}, {numerator}, 31;')
    quotient = e.reg('s32')
    if divisor & (divisor-1) == 0:
        shift = divisor.bit_length()-1
        bias, adjusted = e.reg('s32'), e.reg('s32')
        e.emit(f'shr.u32 {bias}, {sign}, {32-shift};')
        e.emit(f'add.s32 {adjusted}, {numerator}, {bias};')
        e.emit(f'shr.s32 {quotient}, {adjusted}, {shift};')
    else:
        magnitude, product, correction = [e.reg('s32') for _ in range(3)]
        too_large = e.reg('pred')
        multiplier = ((1 << 32) + divisor - 1) // divisor
        e.emit(f'xor.b32 {magnitude}, {numerator}, {sign};')
        e.emit(f'sub.u32 {magnitude}, {magnitude}, {sign};')
        e.emit(f'mul.hi.u32 {quotient}, {magnitude}, {multiplier};')
        e.emit(f'mul.lo.u32 {product}, {quotient}, {divisor};')
        e.emit(f'setp.gt.u32 {too_large}, {product}, {magnitude};')
        e.emit(f'selp.u32 {correction}, 1, 0, {too_large};')
        e.emit(f'sub.u32 {quotient}, {quotient}, {correction};')
        e.emit(f'xor.b32 {quotient}, {quotient}, {sign};')
        e.emit(f'sub.s32 {quotient}, {quotient}, {sign};')
    if not remainder:
        return quotient
    product, result = e.reg('s32'), e.reg('s32')
    e.emit(f'mul.lo.s32 {product}, {quotient}, {divisor};')
    e.emit(f'sub.s32 {result}, {numerator}, {product};')
    return result
