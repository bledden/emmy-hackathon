"""Boolean SSA results use canonical s32 zero/one, preserving operand types.

This is scalar computation, not support for bool buffer arguments/storage.
"""
COMPARISONS = {'equal': 'eq', 'not_equal': 'ne', 'less': 'lt',
               'less_equal': 'le', 'greater': 'gt', 'greater_equal': 'ge'}


def emit_assign(e, stmt, kind, bool_result):
    from emmy.compiler.ir.expr import Var
    from .errors import require
    from .scalars import convert, instruction_type
    op = stmt.op.name
    bitwise_bool = {'bitwise_and':'logical_and', 'bitwise_or':'logical_or',
                    'bitwise_xor':'logical_xor', 'bitwise_not':'logical_not'}
    if bool_result and op in bitwise_bool:
        # Torch spells boolean masks with bitwise operators. Do not confuse
        # integer zero/one with bool: ~int(1) and ~bool(True) differ.
        require(all(e.read(n) in e.boolean_registers for n in stmt.args),
                'boolean bitwise operands must be boolean SSA values')
        op = bitwise_bool[op]
    logical = op in ('logical_and', 'logical_or', 'logical_xor', 'logical_not')
    if op not in COMPARISONS and not logical and not (bool_result and op in ('copy', 'where')):
        return False
    arity = 1 if op in ('copy', 'logical_not') else 3 if op == 'where' else 2
    require(len(stmt.args) == arity, f'bad arity for {op}')
    uniform = all(e.is_uniform(Var(n)) for n in stmt.args)
    warp_uniform = all(e.is_uniform(Var(n), warp=True) for n in stmt.args)
    predicate = e.reg('pred')
    if op in COMPARISONS:
        kinds = [e.values[n][1] for n in stmt.args]
        # NumPy scalar-array promotion for the five supported numeric kinds.
        # A comparison's RESULT type must never round its numeric operands.
        floating = any(k.startswith('f') for k in kinds)
        integer = any(k.startswith('s') for k in kinds)
        common = ('f64' if 'f64' in kinds or floating and integer else
                  'f32' if floating else 's64' if 's64' in kinds else 's32')
        a,b = [e.value_as(n, common) for n in stmt.args]
        cmp = 'neu' if op == 'not_equal' and floating else COMPARISONS[op]
        e.emit(f'setp.{cmp}.{common} {predicate}, {a}, {b};')
    else:
        args = [e.predicate(Var(n)) for n in stmt.args]
        if op == 'logical_not':
            e.emit(f'not.pred {predicate}, {args[0]};')
        elif op == 'copy':
            predicate = args[0]
        elif op == 'where':
            neg, yes, no = [e.reg('pred') for _ in range(3)]
            e.emit(f'not.pred {neg}, {args[0]};')
            e.emit(f'and.pred {yes}, {args[0]}, {args[1]};')
            e.emit(f'and.pred {no}, {neg}, {args[2]};')
            e.emit(f'or.pred {predicate}, {yes}, {no};')
        else:
            instruction = {'logical_and':'and', 'logical_or':'or', 'logical_xor':'xor'}[op]
            e.emit(f'{instruction}.pred {predicate}, {args[0]}, {args[1]};')
    value = e.reg('s32')
    e.emit(f'selp.s32 {value}, 1, 0, {predicate};')
    result = convert(e, value, 's32', kind)
    destination = e.declare(stmt.name, kind)
    e.emit(f'mov.{instruction_type(kind)} {destination}, {result};')
    if bool_result:
        e.boolean_registers.add(destination)
    if kind == 's32':
        e.integer_ranges[destination] = (0, 1)
    if uniform:
        e.uniform_registers.add(destination)
    if warp_uniform:
        e.warp_uniform_registers.add(destination)
    return True
