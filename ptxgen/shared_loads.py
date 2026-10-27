"""Proved affine shared addresses and bounded scalar-to-vector load formation.

Address algebra is in Z/(2**32): normalization never assumes signed overflow
away. Lifting a constant byte offset out of the index happens only after bounds
prove both the variable sum and the final sum fit s32 and the access is in bounds.
Load motion is confined to straight-line, store-free regions with stable inputs
and no intervening use of the destination whose definition would move earlier.
"""
import math
import re
from .integer_semantics import MIN, MAX


def s32(value):
    return (value + 2**31) % 2**32 - 2**31


def address(e, name, indices, offset=0):
    base, shape, space = e.buffers[name]
    if space != 'shared' or e.buffer_kinds.get(name) != 'f32' or any(size <= 0 for size in shape):
        return None
    atoms, terms, constant = {}, {}, 0

    def key(expr):
        tag = type(expr).__name__
        if tag == 'Literal':
            return ('literal', s32(int(expr.value)))
        if tag == 'Var':
            return ('register', e.read(expr.name, 's32'))
        if tag == 'BinaryExpr':
            left, right = key(expr.left), key(expr.right)
            return ('binary', expr.op, left, right) if left and right else None
        return None

    def collect(expr, scale):
        nonlocal constant
        tag = type(expr).__name__
        if tag == 'Literal':
            constant = s32(constant + scale * int(expr.value))
            return True
        if tag == 'BinaryExpr':
            if expr.op in ('+', '-'):
                return collect(expr.left, scale) and collect(expr.right, scale if expr.op == '+' else -scale)
            if expr.op == '*':
                for literal, other in ((expr.left, expr.right), (expr.right, expr.left)):
                    if type(literal).__name__ == 'Literal':
                        return collect(other, s32(scale * int(literal.value)))
        identity = key(expr)
        if identity is None:
            return False
        atoms[identity] = expr
        terms[identity] = s32(terms.get(identity, 0) + scale)
        return True

    stride = math.prod(shape)
    for expr, size in zip(indices, shape):
        stride //= size
        if not collect(expr, stride):
            return None
    constant = s32(constant + offset)
    terms = {k:v for k,v in terms.items() if v}
    low = high = 0
    alignment = 16
    term_alignment = {}
    for identity, coefficient in terms.items():
        bounds = e.index_interval(atoms[identity])
        if bounds is None:
            return None
        a,b = sorted(coefficient * value for value in bounds)
        low,high = low+a,high+b
        term_alignment[identity] = coefficient * e.index_multiple(atoms[identity]) * 4
        alignment = math.gcd(alignment, term_alignment[identity])
    if not (MIN <= low <= high <= MAX and high+constant <= MAX and
            0 <= low+constant <= high+constant < math.prod(shape)):
        return None

    # The original address has already been lowered, so all typed validation and
    # arithmetic refusal checks have run before any canceled term disappears.
    variable = '0'
    for identity, coefficient in sorted(terms.items()):
        value = e.expression(atoms[identity])
        variable = e.integer('+', variable, e.integer('*', value, str(coefficient)))
    byte_offset, start, pointer = e.reg('u64'), e.reg('u64'), e.reg('u64')
    e.emit(f'mul.wide.s32 {byte_offset}, {variable}, 4;')
    e.emit(f'add.u64 {start}, {base}, {byte_offset};')
    e.emit(f'add.u64 {pointer}, {start}, {constant*4};')

    def registers(identity):
        if identity[0] == 'register':
            return {identity[1]}
        if identity[0] == 'binary':
            return registers(identity[2]) | registers(identity[3])
        return set()

    dependencies = {base}
    for identity in terms:
        dependencies.update(registers(identity))
    e.shared_address_info[pointer] = {
        'key':(base, tuple(sorted(terms.items()))), 'offset':constant*4,
        'alignment':alignment, 'dependencies':frozenset(dependencies),
        'term_alignment':term_alignment,
    }
    return pointer, space


LOAD = re.compile(r'\s*ld\.shared\.f32 (%f[0-9]+), \[(%rd[0-9]+)\];')
ANY_LOAD = re.compile(r'\s*ld\.shared\.(?:v[24]\.)?f32 (.+), \[(%rd[0-9]+)\];')
REGISTER = re.compile(r'%[A-Za-z]+[0-9]+')
# Exact opcodes emitted by this backend. Memory effects, control flow, unusual
# modifiers and unknown operations terminate the region rather than widen it.
PURE = frozenset((
    'mov.s32','mov.b32','mov.u32','mov.u64','mov.b64','mov.f32',
    'add.s32','sub.s32','mul.lo.s32','mul.wide.s32','add.u64',
    'and.b32','or.b32','xor.b32','shl.b32','shr.s32','shr.u32',
    'div.s32','rem.s32','cvt.u32.u64',
    'add.rn.f32','sub.rn.f32','mul.rn.f32','fma.rn.f32',
))


def rebase(code, proofs, report=None):
    """Reuse proved affine pointers while their value dependencies stay fixed.

    All equalities come from bounds-checked forms, not distribution through
    unproved s32 wrap. Writes to roots or the saved pointer kill the fact;
    unknown operations clear all facts.
    """
    available, result, changes = {}, [], []
    for number,line in enumerate(code,1):
        text=line.strip()
        load=ANY_LOAD.fullmatch(line)
        words=text.rstrip(';').split(None,1)
        if load:
            destinations=REGISTER.findall(load[1])
        elif text.endswith(';') and len(words)==2 and words[0] in PURE:
            destination=words[1].split(',')[0].strip()
            if not REGISTER.fullmatch(destination):
                available.clear();result.append(line);continue
            destinations=[destination]
        else:
            available.clear();result.append(line);continue
        for key,(register,_,dependencies) in list(available.items()):
            if set(destinations).intersection(dependencies | {register}):
                del available[key]
        if len(destinations)==1 and destinations[0] in proofs and words[0] in ('add.u64','mov.b64'):
            destination=destinations[0];proof=proofs[destination]
            saved=available.get(proof['key'])
            if saved is not None:
                register,offset,_=saved
                replacement=f'    add.u64 {destination}, {register}, {proof["offset"]-offset};'
                changes.append({'line':number,'before':line,'after':replacement})
                line=replacement
            available[proof['key']]=(destination,proof['offset'],proof['dependencies'])
        result.append(line)
    if report is not None:
        report.append({'pass':'shared_address_rebase','replacements':len(changes),'changes':changes})
    return result


def live_load_proofs(code, proofs):
    """Bind each proof to its first pointer definition, not its register name.

    A proof is true at the emitter-owned definition. Later writes to the pointer
    or any dependency kill it; they cannot revive that same static proof. Record
    facts at loads before their destinations are written. Unknown lines clear
    availability, including labels, branches and predicated instructions.
    """
    available, defined, at_load = {}, set(), {}
    for number, line in enumerate(code):
        text = line.strip()
        load = ANY_LOAD.fullmatch(line)
        words = text.rstrip(';').split(None, 1)
        if load:
            if load[2] in available:
                at_load[number] = available[load[2]]
            destinations = REGISTER.findall(load[1])
        elif text.endswith(';') and len(words) == 2 and words[0] in PURE:
            destination = words[1].split(',')[0].strip()
            if not REGISTER.fullmatch(destination):
                available.clear()
                defined.update(set(REGISTER.findall(text)).intersection(proofs))
                continue
            destinations = [destination]
        else:
            available.clear()
            # An unrecognized write must not be mistaken for an untouched
            # pointer when a later recognized assignment uses the same name.
            defined.update(set(REGISTER.findall(text)).intersection(proofs))
            continue
        written = set(destinations)
        for pointer, proof in list(available.items()):
            if pointer in written or written.intersection(proof['dependencies']):
                del available[pointer]
        for destination in destinations:
            if (destination in proofs and destination not in defined and
                    words[0] in ('add.u64', 'mov.b64')):
                available[destination] = proofs[destination]
            defined.add(destination)
    return at_load


def combine(code, proofs, report=None):
    output = list(code)
    replacements = []
    live = live_load_proofs(code, proofs)
    for first, line in enumerate(code):
        a = LOAD.fullmatch(line)
        if a is None or output[first] != line or first not in live:
            continue
        proof = live[first]
        if proof['alignment'] % 8 or proof['offset'] % 8:
            continue
        touched, written = set(), set()
        group = [(first,a[1])]
        limit = 4 if proof['alignment'] % 16 == 0 and proof['offset'] % 16 == 0 else 2
        # Bound both search work and how far a load's lifetime may be extended.
        for second in range(first+1, min(len(code),first+513)):
            text = code[second].strip()
            b = LOAD.fullmatch(code[second])
            if b and output[second] == code[second] and second in live:
                other = live[second]
                if (other['key'] == proof['key'] and
                        other['offset'] == proof['offset']+4*len(group) and b[1] not in {r for _,r in group} and
                        b[1] not in touched and
                        not written.intersection(proof['dependencies'] | other['dependencies'])):
                    group.append((second,b[1]))
                    if len(group) == limit:
                        break
            load = ANY_LOAD.fullmatch(code[second])
            if load:
                destinations = REGISTER.findall(load[1])
            else:
                words = text.rstrip(';').split(None,1)
                if (not text.endswith(';') or len(words)!=2 or words[0] not in PURE):
                    break
                destination = words[1].split(',')[0].strip()
                if not REGISTER.fullmatch(destination):
                    break
                destinations = [destination]
            touched.update(REGISTER.findall(text))
            written.update(destinations)
        width = 4 if len(group) == 4 else 2 if len(group) >= 2 else 0
        if width:
            chosen = group[:width]
            output[first] = f'    ld.shared.v{width}.f32 {{{", ".join(r for _,r in chosen)}}}, [{a[2]}];'
            for line,_ in chosen[1:]:output[line] = None
            replacements.append({'lines':[line+1 for line,_ in chosen],
                                 'width':width,'offset':proof['offset']})
    if report is not None:
        report.append({'pass':'combine_shared_loads','groups':len(replacements),
                       'removed_loads':sum(c['width']-1 for c in replacements),'changes':replacements})
    return [line for line in output if line is not None]


def specialize_proof(proof, index, value, renaming):
    """Specialize an address for an explicitly expanded induction value.

    Only a direct affine atom is substituted. Nested expressions retain the
    induction dependency, preventing combination across its intervening writes.
    """
    base, old_terms = proof['key']
    terms, offset = [], proof['offset']
    alignment = 16
    dependencies = set(proof['dependencies'])
    direct = ('register',index)
    for identity, coefficient in old_terms:
        if identity == direct:
            offset += coefficient*value*4
        else:
            terms.append((identity,coefficient))
            alignment = math.gcd(alignment,proof['term_alignment'][identity])
    # Drop the dependency only when no remaining opaque atom mentions it.
    def mentions(identity):
        return identity == direct or (identity[0]=='binary' and
                                      (mentions(identity[2]) or mentions(identity[3])))
    if not any(mentions(identity) for identity,_ in terms):
        dependencies.discard(index)
    return {**proof,'key':(renaming.get(base,base),tuple(terms)),
            'offset':offset,'alignment':alignment,
            'dependencies':frozenset(renaming.get(r,r) for r in dependencies)}
