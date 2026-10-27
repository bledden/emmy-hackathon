"""Small integer PTX test executor; independent of the reuse implementation.

Finite bit-vector arithmetic and address/store traces, not a GPU simulator.
"""
import re


def execute(code, initial, memory=None, division='truncate', symbolic_float=False):
    regs = dict(initial)
    memory = dict(memory or {})
    trace = []
    labels = {line.strip()[:-1]: i for i, line in enumerate(code) if line.strip().endswith(':')}
    pc = 0

    def read(token):
        if symbolic_float and token.startswith('0f'):
            return ('f32_bits', token[2:])
        return regs[token] if token.startswith('%') else int(token)

    def signed(value, bits):
        value &= (1 << bits) - 1
        return value - (1 << bits) if value & (1 << (bits - 1)) else value

    def put(name, value):
        if symbolic_float and name.startswith('%f'):
            regs[name] = value
        elif name.startswith('%p'):
            regs[name] = bool(value)
        else:
            regs[name] = value & ((1 << (64 if name.startswith(('%rd', '%rl')) else 32)) - 1)

    for _ in range(10000):
        if pc == len(code):
            return regs, memory, trace
        text = code[pc].strip()
        pc += 1
        if not text or text.endswith(':') or text.startswith('//'):
            continue
        if text.startswith('@'):
            predicate, text = text.split(None, 1)
            take = bool(regs[predicate.lstrip('@!')])
            if predicate.startswith('@!'):
                take = not take
            if not take:
                continue
        assert text.endswith(';'), text
        op, _, arguments = text[:-1].partition(' ')
        args = [a.strip() for a in arguments.split(',')]
        if symbolic_float and op.endswith('.f32') and op.split('.')[0] in ('add','sub','mul','fma'):
            put(args[0], (op, *[read(arg) for arg in args[1:]]))
            continue
        if symbolic_float and op in ('ld.shared.v2.f32','ld.shared.v4.f32'):
            match = re.fullmatch(r'\{([^}]+)\}, \[(%rd[0-9]+)\]',arguments)
            assert match, text
            destinations = match[1].split(', ')
            address = read(match[2])
            assert address % (len(destinations)*4) == 0, ('misaligned vector',text,address)
            for index,destination in enumerate(destinations):
                pointer=address+index*4
                put(destination,memory[pointer])
                trace.append(('load','ld.shared.f32',pointer,memory[pointer]))
            continue
        if op in ('bra', 'bra.uni'):
            pc = labels[args[0]]
            continue
        if op == 'ret':
            return regs, memory, trace
        if op == 'bar.sync':
            trace.append(('barrier', args[0]))
            continue
        if op.startswith(('ld.', 'st.')):
            address = args[1] if op.startswith('ld.') else args[0]
            address = read(address[1:-1])
            if op.startswith('ld.'):
                put(args[0], memory[address])
                trace.append(('load', op, address, memory[address]))
            else:
                memory[address] = read(args[1])
                trace.append(('store', op, address, memory[address]))
            continue
        if op.startswith('setp.'):
            width = 64 if op.endswith('64') else 32
            a, b = (signed(read(x),width) if '.s' in op else read(x) & ((1 << width)-1)
                    for x in args[1:])
            cmp = op.split('.')[1]
            put(args[0], {'lt':a<b, 'eq':a==b, 'ge':a>=b, 'ne':a!=b, 'gt':a>b}[cmp])
            continue
        if op.startswith('mov.'):
            put(args[0], read(args[1]))
            continue
        if op.startswith('selp.'):
            put(args[0], read(args[1]) if read(args[3]) else read(args[2]))
            continue
        if op.startswith('cvt.'):
            put(args[0], signed(read(args[1]), int(op.split('.')[-1][1:])))
            continue
        if op.startswith(('abs.', 'neg.', 'popc.')):
            width = int(op.split('.')[-1][1:])
            value = signed(read(args[1]), width)
            result = abs(value) if op.startswith('abs.') else -value
            if op.startswith('popc.'):
                result = (read(args[1]) & ((1 << width)-1)).bit_count()
            put(args[0], result)
            continue
        assert len(args) == 3, text
        a, b = map(read,args[1:])
        base = op.split('.')[0]
        width = 64 if op.endswith('64') else 32
        if op.endswith(('s32', 's64')):
            a, b = signed(a,width), signed(b,width)
        if base == 'add': result = a+b
        elif base == 'sub': result = a-b
        elif base == 'mul': result = (a*b >> width) if '.hi.' in op else a*b
        elif base == 'min': result = min(a,b)
        elif base == 'and': result = a&b
        elif base == 'or': result = a|b
        elif base == 'xor': result = a^b
        elif base == 'shl': result = a << (b & 0xffffffff) if (b & 0xffffffff) < width else 0
        elif base == 'shr': result = a >> min(b & 0xffffffff, width)
        elif base in ('div','rem'):
            assert b != 0
            quotient = a//b if division=='floor' else (abs(a)//abs(b))*(-1 if (a<0)!=(b<0) else 1)
            result = quotient if base=='div' else a-quotient*b
        else: raise AssertionError(('unsupported model operation',text))
        put(args[0],result)
    raise AssertionError('execution bound exceeded')
