"""Fixed-width integer sign and nonnegative integral powers, without floats."""


def emit(e, destination, op, args, kind):
    from .errors import require
    require(len(args) == (1 if op == 'sign' else 2), f'bad arity for {op}')
    if op == 'sign':
        negative, zero = e.reg('pred'), e.reg('pred')
        # Read both predicates before writing an aliased destination.
        e.emit(f'setp.lt.{kind} {negative}, {args[0]}, 0;')
        e.emit(f'setp.eq.{kind} {zero}, {args[0]}, 0;')
        e.emit(f'selp.{kind} {destination}, -1, 1, {negative};')
        e.emit(f'@{zero} mov.{kind} {destination}, 0;')
        return

    # Repeated squaring in Z/(2**width) agrees with fixed-width multiplication
    # for every nonnegative signed exponent, including 0**0 == 1. Copy inputs
    # before touching the result so either operand can alias the destination.
    base, exponent, result, bit = [e.reg(kind) for _ in range(4)]
    predicate = e.reg('pred')
    begin, finish = e.label(), e.label()
    e.emit(f'mov.{kind} {base}, {args[0]};')
    e.emit(f'mov.{kind} {exponent}, {args[1]};')
    e.emit(f'mov.{kind} {result}, 0;')
    # Negative integer exponents are outside NumPy integer-power's domain.
    # A bounded zero result keeps invalid runtime input from spinning; it is
    # not a floating reciprocal and is not claimed to implement that domain.
    e.emit(f'setp.lt.{kind} {predicate}, {exponent}, 0;')
    e.emit(f'@{predicate} bra {finish};')
    e.emit(f'mov.{kind} {result}, 1;')
    e.emit(begin + ':')
    e.emit(f'setp.eq.{kind} {predicate}, {exponent}, 0;')
    e.emit(f'@{predicate} bra {finish};')
    e.emit(f'and.b{kind[1:]} {bit}, {exponent}, 1;')
    e.emit(f'setp.ne.{kind} {predicate}, {bit}, 0;')
    e.emit(f'@{predicate} mul.lo.{kind} {result}, {result}, {base};')
    e.emit(f'mul.lo.{kind} {base}, {base}, {base};')
    e.emit(f'shr.u{kind[1:]} {exponent}, {exponent}, 1;')
    e.emit(f'bra {begin};')
    e.emit(finish + ':')
    e.emit(f'mov.{kind} {destination}, {result};')
