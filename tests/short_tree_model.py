"""Test-only symbolic executor for the emitted shared-tree instruction subset.

Checks ordered FP expression trees, physical shuffle rendezvous, shared accesses
and final values. It is not a general PTX emulator or a hardware/ISA proof.
"""
import random
import re


def execute(code, registers, memory, seed=0):
    labels, instructions = {}, []
    for raw in code:
        text = raw.strip()
        if text.endswith(':'):
            labels[text[:-1]] = len(instructions)
        else:
            assert text.endswith(';'), text
            instructions.append(text[:-1])
    registers = [dict(r) for r in registers]
    memory = dict(memory)
    count = len(registers)
    pc = [0] * count
    accesses = {}
    rng = random.Random(seed)

    def value(tid, token):
        if token.startswith('%'):
            assert token in registers[tid], ('undefined register', tid, token)
            return registers[tid][token]
        if token.startswith('0f'):
            return ('f32_bits', token[2:])
        return int(token, 0)

    def address(tid, token):
        assert token.startswith('[') and token.endswith(']')
        result = value(tid, token[1:-1])
        assert result in memory, ('out of bounds shared access', result)
        return result

    def touch(tid, ptr, write):
        for other, other_write in accesses.get(ptr, []):
            assert tid == other or not (write or other_write), ('shared race', tid, other, ptr)
        accesses.setdefault(ptr, []).append((tid, write))

    for _ in range(20000):
        if all(p == len(instructions) for p in pc):
            return registers, memory
        progress = False
        order = list(range(count))
        rng.shuffle(order)
        for tid in order:
            if pc[tid] == len(instructions):
                continue
            text = instructions[pc[tid]]
            if text.startswith('@'):
                pred, text = text.split(None, 1)
                predicate = pred[2:] if pred.startswith('@!') else pred[1:]
                take = bool(value(tid, predicate))
                if pred.startswith('@!'):
                    take = not take
                if not take:
                    pc[tid] += 1
                    progress = True
                    continue
            op, _, rest = text.partition(' ')
            args = [s.strip() for s in rest.split(',')]
            if op in ('bra', 'bra.uni'):
                if op == 'bra.uni':
                    # Conditional .uni branches must take the same direction
                    # in each physical warp (the source predicates are immutable).
                    if instructions[pc[tid]].startswith('@'):
                        pred=instructions[pc[tid]].split()[0].lstrip('@!')
                        group=range(tid//32*32,min(count,(tid//32+1)*32))
                        assert len({bool(value(t,pred)) for t in group})==1, 'nonuniform bra.uni'
                pc[tid] = labels[rest]
                progress = True
                continue
            if op == 'bar.sync':
                assert args == ['0'], 'model only covers full-CTA barrier zero'
                if not all(p == pc[tid] for p in pc):
                    continue
                pc = [p + 1 for p in pc]
                accesses.clear()
                progress = True
                continue
            if op == 'shfl.sync.bfly.b32':
                dst, src, distance, clamp, mask = args
                mask, distance, clamp = (value(tid, x) for x in (mask, distance, clamp))
                base = tid // 32 * 32
                members = [base + lane for lane in range(32) if mask & (1 << lane)]
                assert tid in members and all(t < count for t in members), 'invalid member mask'
                if not all(pc[t] == pc[tid] for t in members):
                    continue
                captured = {}
                for t in members:
                    lane = t % 32
                    segmask = (clamp >> 8) & 31
                    max_lane = (lane & segmask) | ((clamp & 31) & ~segmask)
                    source = lane ^ distance
                    source = source if source <= max_lane else lane
                    assert base + source in members, 'shuffle source outside participants'
                    captured[t] = value(base + source, src)
                for t in members:
                    registers[t][dst] = captured[t]
                    pc[t] += 1
                progress = True
                continue
            if op == 'ld.shared.f32':
                ptr = address(tid, args[1]); touch(tid, ptr, False)
                registers[tid][args[0]] = memory[ptr]
            elif op == 'st.shared.f32':
                ptr = address(tid, args[0]); touch(tid, ptr, True)
                memory[ptr] = value(tid, args[1])
            elif op in ('add.s32', 'add.u64', 'mul.lo.s32', 'mul.wide.s32'):
                a, b = (value(tid, s) for s in args[1:])
                result = a + b if op.startswith('add') else a * b
                if op.endswith('s32') and 'wide' not in op:
                    result = (result + 2**31) % 2**32 - 2**31
                registers[tid][args[0]] = result
            elif op in ('setp.lt.s32', 'setp.eq.s32'):
                a, b = (value(tid, s) for s in args[1:])
                registers[tid][args[0]] = a < b if '.lt.' in op else a == b
            elif op == 'and.pred':
                registers[tid][args[0]] = value(tid, args[1]) and value(tid, args[2])
            elif op in ('add.rn.f32', 'sub.rn.f32', 'mul.rn.f32', 'max.NaN.f32', 'min.NaN.f32',
                        'ex2.approx.f32', 'fma.rn.f32'):
                # Preserve operand order, opcode and rounding; no reassociation.
                registers[tid][args[0]] = (op, *(value(tid, arg) for arg in args[1:]))
            elif op in ('mov.f32', 'mov.b32', 'mov.b64'):
                registers[tid][args[0]] = value(tid, args[1])
            else:
                raise AssertionError(('unsupported model opcode', op))
            pc[tid] += 1
            progress = True
        assert progress, ('collective deadlock', pc)
    raise AssertionError('model step bound exceeded')
