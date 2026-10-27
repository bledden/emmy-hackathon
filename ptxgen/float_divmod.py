"""Floating floor quotient and divisor-sign remainder without rounded-divide shortcuts.

Binary-scaled subtraction computes fmod before the NumPy-style sign/quotient
adjustments. No FTZ, device library, or external compiler is used. The finite
loops are bounded by the input format's exponent range, not quotient magnitude.
"""
import math
from .scalars import literal, convert


def emit(e, destination, op, args, kind):
    from .errors import require
    require(len(args) == 2, f'bad arity for {op}')
    wide = 'f32' if kind == 'f16' else kind
    a, b = [convert(e, v, kind, wide) for v in args]
    quotient = op == 'floor_divide'
    zero, one, half, two, infinity, nan = [literal(x, wide) for x in (0., 1., .5, 2., math.inf, math.nan)]
    result, rem, divisor, scale, limit = [e.reg(wide) for _ in range(5)]
    p, negative_a, negative_b, different = [e.reg('pred') for _ in range(4)]
    divzero, invalid, up, down, signed, remzero, adjusted, finish = [e.label() for _ in range(8)]
    def inst(opcode, dst, *values):e.emit(f'{opcode}.{wide} {dst}, '+', '.join(values)+';')
    def cmp(relation, left, right):inst('setp.'+relation, p, left, right)
    def jump(label, invert=False):e.emit(f'@{"!" if invert else ""}{p} bra {label};')
    def mark(label):e.emit(label+':')
    def go(label):e.emit(f'bra {label};')

    # No write to destination until all reads finish, including either alias.
    cmp('eq', b, zero);jump(divzero)
    cmp('nan', a, b);jump(invalid)
    inst('abs', rem, a);inst('abs', divisor, b)
    cmp('eq', rem, infinity);jump(invalid)
    inst('mov', scale, divisor)
    cmp('eq', divisor, infinity);jump(signed)
    cmp('lt', rem, divisor);jump(signed)
    inst('mul.rn', limit, rem, half)
    mark(up)
    cmp('gt', scale, limit);jump(down)
    inst('mul.rn', scale, scale, two);go(up)
    mark(down)
    cmp('lt', scale, divisor);jump(signed)
    cmp('ge', rem, scale)
    e.emit(f'@{p} sub.rn.{wide} {rem}, {rem}, {scale};')
    inst('mul.rn', scale, scale, half);go(down)

    mark(signed)
    # PTX copysign takes sign first, magnitude second.
    inst('copysign', rem, a, rem)
    if quotient:
        inst('sub.rn', result, a, rem);inst('div.rn', result, result, b)
    cmp('eq', rem, zero);jump(remzero)
    inst('setp.lt', negative_a, a, zero);inst('setp.lt', negative_b, b, zero)
    e.emit(f'xor.pred {different}, {negative_a}, {negative_b};')
    e.emit(f'@{different} add.rn.{wide} {rem}, {rem}, {b};')
    if quotient:e.emit(f'@{different} sub.rn.{wide} {result}, {result}, {one};')
    go(adjusted)
    mark(remzero);inst('copysign', rem, b, zero)
    mark(adjusted)
    if quotient:
        qzero, rounded = e.label(), e.reg(wide)
        cmp('eq', result, zero);jump(qzero)
        inst('abs', limit, result);cmp('eq', limit, infinity);jump(finish)
        e.emit(f'cvt.rmi.{wide}.{wide} {rounded}, {result};')
        inst('sub.rn', limit, result, rounded);cmp('gt', limit, half)
        e.emit(f'@{p} add.rn.{wide} {rounded}, {rounded}, {one};')
        inst('mov', result, rounded);go(finish)
        mark(qzero)
        inst('div.rn', limit, a, b);inst('copysign', result, limit, zero);go(finish)
    else:
        inst('mov', result, rem);go(finish)
    mark(divzero)
    if quotient:inst('div.rn', result, a, b)
    else:inst('mov', result, nan)
    go(finish)
    mark(invalid);inst('mov', result, nan)
    mark(finish)
    result = convert(e, result, wide, kind)
    e.emit(f'mov.{"b16" if kind == "f16" else kind} {destination}, {result};')
