"""Hoist total, pure s32 divisions/remainders out of innermost loops.

No memory read, floating-point operation, variable divisor or written binding
qualifies. This makes speculative evaluation safe even for zero-trip loops.
"""
from dataclasses import fields, is_dataclass
from .ir_analysis import walk, written_names


def key(expr):
    tag = type(expr).__name__
    if tag == 'Var':
        return ('var', expr.name)
    if tag == 'Literal':
        dtype = getattr(expr.dtype, 'name', expr.dtype)
        if dtype in ('int', 'i32') and type(expr.value) is int and -2**31 <= expr.value < 2**31:
            return ('int', expr.value)
    if tag == 'BinaryExpr' and expr.op in ('+', '-', '*', '/', '//', '%'):
        a, b = key(expr.left), key(expr.right)
        if a is None or b is None:
            return None
        if expr.op in ('/', '//', '%') and not (b[0] == 'int' and b[1] > 0):
            return None
        return (expr.op, a, b)
    return None


def variables(k):
    if k[0] == 'var':
        return {k[1]}
    if k[0] == 'int':
        return set()
    return variables(k[1]) | variables(k[2])


def candidates(body, immutable_s32):
    statements = list(walk(body))
    if any(type(s).__name__ in ('Loop', 'StridedLoop') for s in statements):
        return []
    allowed = immutable_s32 - written_names(body)
    result, seen = [], set()

    def visit(expr):
        tag = type(expr).__name__
        if tag == 'BinaryExpr':
            visit(expr.left)
            visit(expr.right)
        elif tag == 'TernaryExpr':
            visit(expr.cond)
            visit(expr.if_true)
            visit(expr.if_false)
        k = key(expr)
        if (k is not None and k[0] in ('/', '//', '%') and k[2][1] > 1
                and variables(k) <= allowed and k not in seen):
            seen.add(k)
            result.append(expr)

    for stmt in statements:
        if is_dataclass(stmt):
            for field in fields(stmt):
                value = getattr(stmt, field.name)
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    if type(item).__name__ in ('Var', 'Literal', 'BinaryExpr', 'TernaryExpr'):
                        visit(item)
    return result
