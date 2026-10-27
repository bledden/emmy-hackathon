"""Unsigned wide arithmetic for bounded, nonnegative launch coordinates only."""

def arithmetic(e, op, left, right):
    if op in ('+', '-') and right == '0' or op in ('*', '/') and right == '1':return left
    if op == '+' and left == '0':return right
    if op == '*' and (left == '0' or right == '0') or op == '%' and right == '1':return '0'
    result=e.reg('u64')
    opcode={'+':'add','-':'sub','*':'mul.lo','/':'div','%':'rem'}[op]
    e.emit(f'{opcode}.u64 {result}, {left}, {right};')
    return result


def narrow(e, value):
    # Callers prove the decoded axis value lies in [0, 2**31-1].
    result=e.reg('s32');e.emit(f'cvt.u32.u64 {result}, {value};');return result


def global_id(e, cta, thread, logical, auxiliary):
    base, local, result=[e.reg('u64') for _ in range(3)]
    if auxiliary:thread=e.integer('%',thread,str(logical))
    e.emit(f'mul.wide.u32 {base}, {cta}, {logical};')
    e.emit(f'cvt.u64.u32 {local}, {thread};')
    e.emit(f'add.u64 {result}, {base}, {local};')
    return result
