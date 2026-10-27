"""Offline instruction/ownership checker for the straight-line MMA examples.

Reads actual emitted addresses and packed operands. Matrix-to-lane inverses
are specified independently of the emitter. Exact integer-valued FP16 inputs
make arithmetic exact; this is NOT a model of hardware MMA rounding, timing,
denormals, NaNs, or concurrency. Unknown instructions fail rather than skip.
"""
import re
import struct

import numpy as np


def execute(ptx, inputs, meta):
    launch = meta['kernels'][0]
    assert launch['block'] == [32, 1, 1] and launch['grid'] == [1, 1, 1]
    registers = [{'%tid.x': lane, '%ctaid.x': 0, '%ntid.x': 32} for lane in range(32)]
    declared = {}
    for kind, prefix, count in re.findall(r'\.reg \.(\w+) (%\w+)<(\d+)>;', ptx):
        declared.update({f'{prefix}{i}': kind for i in range(int(count))})
    memory, outputs, parameters, writes = {}, {}, {}, set()
    for index, arg in enumerate(launch['args']):
        base = (index+1) << 32
        parameters[f'param_{index}'] = base
        count = int(np.prod(arg['shape']))
        if arg.get('output'):
            assert arg['dtype'] == 'f32'
            outputs[arg['name']] = (base, arg['shape'])
            memory.update({base+4*i: 0.0 for i in range(count)})
        else:
            data = np.asarray(inputs[arg['name']])
            assert data.dtype == np.float16 and tuple(arg['shape']) == data.shape
            assert np.isfinite(data).all() and (data == np.rint(data)).all()
            assert np.abs(data).max() <= 4
            memory.update({base+2*i: int(v) for i, v in enumerate(data.view(np.uint16).ravel())})

    def value(token, lane):
        if token.startswith('%'):
            return registers[lane][token]
        if token.startswith('0f'):
            return struct.unpack('>f', bytes.fromhex(token[2:]))[0]
        return int(token)

    def half(reg, lane, high):
        bits = (value(reg, lane) >> (16*high)) & 65535
        return struct.unpack('<e', struct.pack('<H', bits))[0]

    for line in ptx.splitlines():
        line = line.strip()
        if (not line or line.startswith(('.', '//')) or line in ('{', '}') or
                line.startswith(')') or line.endswith(':')):
            continue
        assert line.endswith(';'), line
        op, _, operands = line[:-1].partition(' ')
        if op == 'ret':
            break
        if op.startswith('mma.'):
            assert op == 'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32'
            groups = [s.split(', ') for s in re.findall(r'\{([^}]+)\}', operands)]
            assert [len(g) for g in groups] == [4, 4, 2, 4]
            dest, a, b, c = groups
            assert all(declared[r] == 'b32' for r in a+b), 'MMA operands require packed b32 registers'
            assert all(declared[r] == 'f32' for r in dest+c), 'MMA accumulators require f32 registers'
            # Inverse maps: for each logical matrix cell, find its owner.
            aa = np.array([[half(a[(k//8)*2 + r//8], (r%8)*4 + (k%8)//2, k%2)
                            for k in range(16)] for r in range(16)], dtype=np.float64)
            bb = np.array([[half(b[k//8], n*4 + (k%8)//2, k%2)
                            for n in range(8)] for k in range(16)], dtype=np.float64)
            result = aa @ bb
            updates = []
            for r in range(16):
                for n in range(8):
                    lane, slot = (r%8)*4+n//2, (r//8)*2+n%2
                    v = result[r, n] + value(c[slot], lane)
                    assert abs(v) <= 2**24 and int(v) == v
                    updates.append((lane, dest[slot], float(v)))
            for lane, reg, v in updates:
                registers[lane][reg] = v
            continue
        packed = re.fullmatch(r'(%\w+), \{(%\w+), (%\w+)\}', operands)
        if packed:
            assert op == 'mov.b32'
            d, lo, hi = packed.groups()
            for lane in range(32):
                registers[lane][d] = value(lo, lane) | (value(hi, lane) << 16)
            continue
        args = [a.strip() for a in operands.split(',')]
        for lane in range(32):
            regs = registers[lane]
            if op == 'ld.param.u64':
                regs[args[0]] = parameters[args[1][1:-1]]
            elif op.startswith('mov.') or op == 'cvta.to.global.u64':
                regs[args[0]] = value(args[1], lane)
            elif op == 'ld.global.b16':
                regs[args[0]] = memory[value(args[1][1:-1], lane)]
            elif op == 'st.global.f32':
                address = value(args[0][1:-1], lane)
                assert any(base <= address < base+4*np.prod(dims) and (address-base)%4 == 0
                           for base, dims in outputs.values()), 'store outside output'
                assert address not in writes, 'duplicate store owner'
                writes.add(address)
                memory[address] = value(args[1], lane)
            else:
                assert op in ('add.s32', 'add.u64', 'mul.lo.s32', 'mul.wide.s32',
                              'and.b32', 'shr.u32', 'shr.s32', 'sub.s32', 'div.s32', 'rem.s32', 'shl.b32'), op
                a, b = value(args[1], lane), value(args[2], lane)
                if op.startswith('add.'): v = a+b
                elif op.startswith('mul.'): v = a*b
                elif op.startswith('and.'): v = a & b
                elif op == 'shr.s32': v = ((int(a) + 2**31) % 2**32 - 2**31) >> b
                elif op == 'shr.u32': v = (int(a) & 0xffffffff) >> b
                elif op == 'sub.s32': v = (int(a) - int(b) + 2**31) % 2**32 - 2**31
                elif op.startswith('shl.'): v = a << b
                elif op.startswith('div.'): v = a//b
                else: v = a%b
                regs[args[0]] = v
    return {name: np.array([memory[base+4*i] for i in range(int(np.prod(dims)))],
                           dtype=np.float32).reshape(dims)
            for name, (base, dims) in outputs.items()}
