"""Bounded expansion of emitter-owned, explicitly annotated serial loops.

Only straight-line bodies with a known 0..N unit-step induction qualify. The
already-lowered body is copied in execution order, after numerical policy and
integer reuse have run. No load or arithmetic operation moves between iterations.
"""
import re

MAX_TRIPS = 64
MAX_EXPANDED_LINES = 16384
STRAIGHT_LINE = frozenset(('mov', 'cvt', 'add', 'sub', 'mul', 'mad', 'div',
                          'rem', 'neg', 'abs', 'and', 'or', 'xor', 'not', 'shl',
                          'shr', 'shf', 'setp', 'selp', 'ld', 'st', 'fma',
                          'min', 'max', 'rcp', 'sqrt', 'rsqrt', 'ex2', 'lg2',
                          'copysign', 'sin', 'cos'))


def expand_annotated_loops(code, regions, report=None, *, allocate=None,
                           register_kinds=None, address_proofs=None):
    output = list(code)
    changes = []
    growth = 0
    for begin, end, index, predicate, count in regions:
        if type(count) is not int or not 2 <= count <= MAX_TRIPS:
            continue
        start_label, end_label = '    ' + begin + ':', '    ' + end + ':'
        if output.count(start_label) != 1 or output.count(end_label) != 1:
            continue
        start, finish = output.index(start_label), output.index(end_label)
        head = [f'    setp.ge.s32 {predicate}, {index}, {count};',
                f'    @{predicate} bra {end};']
        tail = [f'    add.s32 {index}, {index}, 1;', f'    bra {begin};']
        if (start < 1 or finish <= start + 4 or
                output[start - 1] != f'    mov.s32 {index}, 0;' or
                output[start + 1:start + 3] != head or output[finish - 2:finish] != tail):
            continue
        body = output[start + 3:finish - 2]
        # Reject nested control flow and any opaque operation. Regions originate
        # in the typed emitter, but keeping this boundary explicit makes a future
        # emitter extension decline rather than silently widen the pass.
        bad = False
        for line in body:
            text = line.strip()
            if not text.endswith(';') or text.startswith(('@', '.', '//')) or ':' in text:
                bad = True
                break
            opcode, _, operands = text[:-1].partition(' ')
            if opcode.split('.')[0] not in STRAIGHT_LINE:
                bad = True
                break
            if re.match(re.escape(index) + r'(?:\s*,|\s*$)', operands):
                bad = True
                break
        extra = count * (len(body) + 1) + 2
        if bad or growth + extra > MAX_EXPANDED_LINES:
            continue
        expanded = []
        rename = set()
        if allocate is not None:
            outside = '\n'.join(output[:start-1] + output[finish+1:])
            definitions = {}
            for line in body:
                text = line.strip()
                if text.startswith('ld.shared.'):
                    destinations = re.findall(r'%f[0-9]+', text.split('[')[0])
                else:
                    first = text.split(None,1)[1].split(',')[0].strip()
                    destinations = [first] if re.fullmatch(r'%rd[0-9]+', first) else []
                for register in destinations:
                    definitions[register] = definitions.get(register, 0)+1
            for register, definitions_count in definitions.items():
                token = re.compile(re.escape(register)+r'(?![0-9])')
                first_use = next(line for line in body if token.search(line))
                first_destination = first_use.strip().split(None,1)[1].split(',')[0].strip()
                defined_first = (first_use.strip().startswith('ld.shared.') and register in
                                 re.findall(r'%f[0-9]+', first_use.split('[')[0]) or
                                 first_destination == register)
                if not first_use.strip().startswith('ld.shared.'):
                    defined_first &= not token.search(first_use.split(',',1)[1] if ',' in first_use else '')
                if definitions_count == 1 and defined_first and not token.search(outside):
                    rename.add(register)
        for value in range(count):
            mapping = {register:allocate(register_kinds[register]) for register in sorted(rename)}
            expanded.append(f'    mov.s32 {index}, {value};')
            expanded.extend(re.sub(r'%[A-Za-z]+[0-9]+', lambda m:mapping.get(m[0],m[0]), line)
                            for line in body)
            if address_proofs is not None:
                from .shared_loads import specialize_proof
                for old,new in mapping.items():
                    if old in address_proofs:
                        address_proofs[new] = specialize_proof(address_proofs[old], index, value, mapping)
        # Preserve terminal induction and guard values even though their lexical
        # scope ends here; downstream allocation may eliminate the dead writes.
        expanded.extend([f'    mov.s32 {index}, {count};', head[0]])
        output[start - 1:finish + 1] = expanded
        growth += extra
        changes.append({'header': begin, 'iterations': count, 'body_lines': len(body),
                        'expanded_lines': len(expanded)})
    if report is not None:
        report.append({'pass': 'annotated_loop_unroll', 'loops': len(changes), 'changes': changes})
    return output
