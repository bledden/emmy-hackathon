"""Scalar PTX types, exact literals and explicit typed arithmetic.

Storage width and arithmetic kind are never inferred from an f32 default.
Binary16 arithmetic widens exact operands, rounds each result back to binary16,
and therefore remains distinct from mixed-precision f32 accumulators.
"""
import math
import struct

KINDS = {None:'f32', 'float':'f32', 'f32':'f32', 'f16':'f16', 'f64':'f64',
         'int':'s32', 'i32':'s32', 'i64':'s64'}
BYTES = {'f16':2, 'f32':4, 'f64':8, 's32':4, 's64':8}
PREFIX = {'f16':'h', 'f32':'f', 'f64':'fd', 's32':'r', 's64':'rl', 'u64':'rd', 'pred':'p', 'b32':'b'}

def promoted_kind(op, kinds):
    """Pinned Emmy's default Assign rule; explicit dtype always overrides it."""
    integer_ops = {'add', 'subtract', 'multiply', 'floor_divide', 'remainder',
                   'mod', 'bitwise_and', 'bitwise_or', 'bitwise_xor',
                   'left_shift', 'right_shift', 'bitwise_count'}
    if kinds and op in integer_ops and all(k in ('s32', 's64') for k in kinds):
        return 's64' if 's64' in kinds else 's32'
    if kinds and all(k == 'f16' for k in kinds):
        return 'f16'
    return 'f32'

def instruction_type(kind):
    return 'b16' if kind=='f16' else kind


def literal(value, kind):
    if kind in ('s32','s64'):
        bits=BYTES[kind]*8
        return str((int(value)+(1 << (bits-1))) % (1 << bits)-(1 << (bits-1)))
    if kind=='f16':
        try: data=struct.pack('>e',float(value))
        except OverflowError: data=struct.pack('>e',math.copysign(math.inf,value))
        return str(int.from_bytes(data,'big'))
    if kind=='f64':return '0d'+struct.pack('>d',float(value)).hex().upper()
    return '0f'+struct.pack('>f',float(value)).hex().upper()


def convert(e, value, source, target):
    if source==target:return value
    if source == 'f16' and not value.startswith('%'):
        # Half immediates are raw b16 literals, not a PTX floating immediate.
        # Conversions require a register operand with those exact bits.
        temporary = e.reg('f16')
        e.emit(f'mov.b16 {temporary}, {value};')
        value = temporary
    out=e.reg(target)
    if source in ('s32','s64') and target in ('s32','s64'):
        rounding=''
    elif target.startswith('s'):
        rounding='.rzi'
    elif source=='f16' and target in ('f32','f64') or source=='f32' and target=='f64':
        rounding=''
    else:rounding='.rn'
    e.emit(f'cvt{rounding}.{target}.{source} {out}, {value};')
    return out


def arithmetic(e, destination, op, args, kind):
    from .errors import require, CompileError
    if kind == 'f64':
        from .wide_math import emit_math
        if emit_math(e, destination, op, args):
            return
    if kind=='f16':
        # Half add/multiply are exact in f32 before final half rounding. Division
        # uses f64 to avoid a midpoint double-rounding error through binary32.
        wide='f64' if op in ('divide','sqrt','reciprocal') else 'f32'
        operands=[convert(e,a,'f16',wide) for a in args]
        result=e.reg(wide)
        if op=='pow':
            from .power import emit_power
            require(len(args)==2,'bad arity for pow')
            emit_power(e,result,*operands)
        else:e.arithmetic(result,op,operands)
        e.emit(f'cvt.rn.f16.{wide} {destination}, {result};')
        return
    integer=kind.startswith('s')
    if integer and op in ('sign', 'pow'):
        from .integer_math import emit
        emit(e, destination, op, args, kind)
        return
    if integer and op in ('floor_divide', 'remainder', 'mod'):
        require(len(args)==2,f'bad arity for {op}')
        q,r,signs,correction=[e.reg(kind) for _ in range(4)]
        different,nonzero,adjust=[e.reg('pred') for _ in range(3)]
        # Quotient overflow (MIN/-1) wraps in fixed-width NumPy integer
        # arithmetic. Bypass PTX's undefined quotient for that one pair.
        minimum=-(1<<(8*BYTES[kind]-1));a_min,b_minus,overflow=[e.reg('pred') for _ in range(3)]
        special,finish=e.label(),e.label()
        e.emit(f'setp.eq.{kind} {a_min}, {args[0]}, {minimum};')
        e.emit(f'setp.eq.{kind} {b_minus}, {args[1]}, -1;')
        e.emit(f'and.pred {overflow}, {a_min}, {b_minus};')
        e.emit(f'@{overflow} bra {special};')
        e.emit(f'rem.{kind} {r}, {args[0]}, {args[1]};')
        e.emit(f'xor.b{8*BYTES[kind]} {signs}, {args[0]}, {args[1]};')
        e.emit(f'setp.lt.{kind} {different}, {signs}, 0;')
        e.emit(f'setp.ne.{kind} {nonzero}, {r}, 0;')
        e.emit(f'and.pred {adjust}, {different}, {nonzero};')
        if op=='floor_divide':
            e.emit(f'div.{kind} {q}, {args[0]}, {args[1]};')
            e.emit(f'selp.{kind} {correction}, 1, 0, {adjust};')
            e.emit(f'sub.{kind} {destination}, {q}, {correction};')
        else:
            e.emit(f'selp.{kind} {correction}, {args[1]}, 0, {adjust};')
            e.emit(f'add.{kind} {destination}, {r}, {correction};')
        e.emit(f'bra {finish};');e.emit(special+':')
        e.emit(f'mov.{kind} {destination}, {minimum if op=="floor_divide" else 0};')
        e.emit(finish+':')
        return
    if integer and op in ('bitwise_and','bitwise_or','bitwise_xor','left_shift','right_shift','bitwise_count'):
        require(len(args)==(1 if op=='bitwise_count' else 2),f'bad arity for {op}')
        bits=8*BYTES[kind]
        if op=='bitwise_count':
            # NumPy counts the absolute value of a signed integer.
            magnitude=e.reg(kind);count=e.reg('s32')
            e.emit(f'abs.{kind} {magnitude}, {args[0]};')
            e.emit(f'popc.b{bits} {count}, {magnitude};')
            result=convert(e,count,'s32',kind)
            e.emit(f'mov.{kind} {destination}, {result};')
        elif op in ('left_shift','right_shift'):
            negative,large,clipped=[e.reg('pred') for _ in range(3)]
            e.emit(f'setp.lt.{kind} {negative}, {args[1]}, 0;')
            e.emit(f'setp.ge.{kind} {large}, {args[1]}, {bits};')
            e.emit(f'or.pred {clipped}, {negative}, {large};')
            count=convert(e,args[1],kind,'s32')
            bounded=e.reg('s32')
            e.emit(f'selp.s32 {bounded}, {bits}, {count}, {clipped};')
            inst=f'shl.b{bits}' if op=='left_shift' else f'shr.{kind}'
            e.emit(f'{inst} {destination}, {args[0]}, {bounded};')
        else:
            inst={'bitwise_and':'and','bitwise_or':'or','bitwise_xor':'xor'}[op]
            e.emit(f'{inst}.b{bits} {destination}, {args[0]}, {args[1]};')
        return
    binary={'add':'add','subtract':'sub','multiply':'mul.lo' if integer else 'mul.rn',
            'maximum':'max','minimum':'min'}
    if not integer:binary.update(add='add.rn',subtract='sub.rn',divide='div.rn')
    if op in binary:
        require(len(args)==2,f'bad arity for {op}')
        nan = None
        if not integer and op in ('maximum', 'minimum'):
            # PTX's .NaN modifier is f32-only; explicit propagation is required
            # for NumPy-style f64 maximum/minimum, including in-place updates.
            nan = e.reg('pred')
            e.emit(f'setp.nan.f64 {nan}, {args[0]}, {args[1]};')
        e.emit(f'{binary[op]}.{kind} {destination}, {args[0]}, {args[1]};')
        if nan is not None:
            e.emit(f'selp.f64 {destination}, {literal(math.nan, kind)}, {destination}, {nan};')
    elif op in ('copy','pad'):
        require(len(args)==1,f'bad arity for {op}')
        e.emit(f'mov.{kind} {destination}, {args[0]};')
    elif op in ('negative','abs','sqrt','reciprocal'):
        require(len(args)==1,f'bad arity for {op}')
        require(not integer or op in ('negative','abs'),f'{op} requires floating result dtype')
        inst={'negative':'neg','abs':'abs','sqrt':'sqrt.rn','reciprocal':'rcp.rn'}[op]
        e.emit(f'{inst}.{kind} {destination}, {args[0]};')
    elif op=='square':
        require(len(args)==1,'bad arity for square')
        e.emit(f"{'mul.lo' if integer else 'mul.rn'}.{kind} {destination}, {args[0]}, {args[0]};")
    elif op=='relu':
        require(len(args)==1,'bad arity for relu')
        arithmetic(e,destination,'maximum',[args[0],literal(0,kind)],kind)
    elif op=='where':
        require(len(args)==3,'bad arity for where')
        p=e.reg('pred');cmp='ne' if integer else 'neu'
        e.emit(f'setp.{cmp}.{kind} {p}, {args[0]}, {literal(0,kind)};')
        e.emit(f'selp.{kind} {destination}, {args[1]}, {args[2]}, {p};')
    elif op in ('floor','ceil','trunc','rint'):
        require(len(args)==1,f'bad arity for {op}')
        if integer:e.emit(f'mov.{kind} {destination}, {args[0]};')
        else:
            rounding={'floor':'rmi','ceil':'rpi','trunc':'rzi','rint':'rni'}[op]
            e.emit(f'cvt.{rounding}.{kind}.{kind} {destination}, {args[0]};')
    else:raise CompileError(f'unsupported {kind} elementwise operation {op}')


def f32(value):
    return "0f" + struct.pack(">f", float(value)).hex().upper()
