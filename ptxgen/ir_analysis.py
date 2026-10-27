"""Pure IR facts used by scope, alignment and collective checks.

These functions neither allocate registers nor emit instructions.
"""
import math

from .errors import require
from .scalars import KINDS, literal as scalar_literal

COLLECTIVES = frozenset(("Sync", "WarpShuffle", "TreeHalve"))
S32_MIN, S32_MAX = -(2**31), 2**31 - 1


def extent(value):
    if isinstance(value, int):
        return value
    require(getattr(value, "is_static", False), f"dynamic dimension: {value}")
    return int(value.value)



def walk(body):
    for stmt in body:
        yield stmt
        yield from walk(getattr(stmt, "body", ()))
        yield from walk(getattr(stmt, "else_body", ()))



def loop_accumulators(body, axis, nested_loop=False):
    """Find this loop's folds; legacy empty axes mean the nearest loop.

    Explicit axes can name an outer owner through nested drains. Missing axes
    carry no such information, so never infer ownership across another loop.
    Conditions do not change the owner: seed even when no branch folds a value.
    """
    for stmt in body:
        tag = type(stmt).__name__
        if tag == "Accum" and (axis in stmt.axes or (not stmt.axes and not nested_loop)):
            yield stmt
        inner = nested_loop or tag in ("Loop", "StridedLoop")
        yield from loop_accumulators(getattr(stmt, "body", ()), axis, inner)
        yield from loop_accumulators(getattr(stmt, "else_body", ()), axis, inner)



def written_names(body):
    """Conservative destinations, including later writes and loop backedges."""
    names = set()
    for stmt in walk(body):
        if type(stmt).__name__ in ("Let", "Assign", "Init", "Accum"):
            names.add(stmt.name)
        for field in ("names", "state", "state_b"):
            names.update(getattr(stmt, field, ()))
        axis = getattr(stmt, "axis", None)
        if axis is not None:
            names.add(axis.name)
        names.update(written_names(getattr(stmt, "combine_states", ())))
    return names



def definite_body_folds(body):
    """Fold bindings defined at the end of this lexical statement sequence.

    A seeded loop defines its own folds even on a zero-trip path. A nested
    conditional contributes only its definite exports; an enclosing loop does
    not export a different inner loop's locals. Ordinary declarations replace
    fold provenance, and therefore cannot escape under the Accum exception.
    """
    names = set()
    for stmt in body:
        tag = type(stmt).__name__
        if tag in ('Loop', 'StridedLoop') and stmt.seed:
            names.update(a.name for a in loop_accumulators(stmt.body, stmt.axis.name))
        elif tag == 'Cond':
            names.update(definite_conditional_folds(stmt))
        elif tag in ('Let', 'Assign', 'Init'):
            names.discard(stmt.name)
        elif tag == 'Load':
            names.difference_update(stmt.names)
    return names



def literal_truth(expression):
    """Truth of the emitted literal, or None for a runtime expression."""
    if type(expression).__name__ == 'Literal':
        dtype = getattr(expression.dtype, 'name', expression.dtype)
        kind = KINDS.get(dtype)
        require(kind is not None, f"unsupported dtype {dtype}")
        encoded = scalar_literal(expression.value, kind)
        if kind in ('s32', 's64'):
            truth = bool(int(encoded))
        elif kind == 'f16':
            truth = bool(int(encoded) & 0x7fff)
        else:
            # Test the emitted literal, including precision-specific underflow
            # and signed zero, rather than Python's unrounded truth value.
            mask = 0x7fffffff if kind == 'f32' else 0x7fffffffffffffff
            truth = bool(int(encoded[2:], 16) & mask)
        return truth
    return None



def definite_conditional_folds(stmt):
    """Export folds defined on every possible path, without inventing a seed."""
    truth = literal_truth(stmt.cond)
    if truth is not None:
        return definite_body_folds(stmt.body if truth else stmt.else_body)
    return definite_body_folds(stmt.body) & definite_body_folds(stmt.else_body)



def warp_only_collectives(body):
    """No operation in this region synchronizes different physical warps."""
    return all(type(s).__name__ == 'WarpShuffle' or
               (type(s).__name__ == 'Sync' and s.warp)
               for s in walk(body) if type(s).__name__ in COLLECTIVES)



def counted_region(body):
    """Only warp-local operations and explicit named counted barriers."""
    return all(type(s).__name__ == 'WarpShuffle' or
               (type(s).__name__ == 'Sync' and
                (s.warp or (s.barrier_id > 0 and s.count is not None))) or
               (type(s).__name__ == 'TreeHalve' and s.barrier_id > 0 and
                s.barrier_count is not None)
               for s in walk(body) if type(s).__name__ in COLLECTIVES)



def uniform_trip_count(low, high, stop, step):
    """Return a proven common trip count, or None (including s32 wrap).

    Trips are monotone in start, so equal endpoint counts cover the interval.
    Check the final increment too: it executes before the terminating test.
    """
    if not (S32_MIN <= low <= high <= S32_MAX and
            S32_MIN <= stop <= S32_MAX and 0 < step <= S32_MAX):
        return None
    trips = lambda start: max(0, (stop - start + step - 1) // step)
    count = trips(low)
    if count != trips(high) or high + count * step > S32_MAX:
        return None
    return count



def uniform_tile_names(axes, block, changed):
    """Immutable mixed-radix coordinates whose boundaries cannot split a CTA."""
    stride, names = 1, set()
    for axis in reversed(axes):
        size = extent(axis.extent)
        if axis.name not in changed and (size == 1 or stride % block == 0):
            names.add(axis.name)
        stride *= size
    return names



def uniform_expression(expr, names, quotient=None):
    """Structural proof using the currently proven uniform bindings.

    Type and operation validity are checked by the ordinary expression emitter.
    Every operand must agree across the CTA, including both ternary arms.
    """
    tag = type(expr).__name__
    if tag == "Literal":
        return True
    if tag == "Var":
        return expr.name in names
    if tag == "BinaryExpr":
        return ((quotient is not None and quotient(expr)) or
                (uniform_expression(expr.left, names, quotient) and
                 uniform_expression(expr.right, names, quotient)))
    if tag == "TernaryExpr":
        return all(uniform_expression(e, names, quotient) for e in (expr.cond, expr.if_true, expr.if_false))
    return False



def guaranteed_multiple(expr, bindings=None):
    """Power-of-two divisibility valid even after signed 32-bit wrap.

    Only low bits matter for vector alignment. Odd factors are discarded:
    e.g. wrap(6*x) is always even, but need not remain divisible by six.
    Zero means a constant zero; 2**32 means all low 32 bits are zero.
    Optional bindings are proven powers of two for the current s32 bindings.
    """
    tag = type(expr).__name__
    if tag == "Literal":
        value = int(expr.value) & 0xffffffff
        return value & -value
    if tag == "Var":
        return (bindings or {}).get(expr.name, 1)
    if tag == "BinaryExpr":
        left = guaranteed_multiple(expr.left, bindings)
        right = guaranteed_multiple(expr.right, bindings)
        if expr.op in ("+", "-"):
            return math.gcd(left, right)
        if expr.op == "*":
            return math.gcd(left * right, 2**32)
        if (expr.op == "%" and type(expr.right).__name__ == "Literal"
                and int(expr.right.value) != 0):
            # Both truncating and floor remainders are a - q*b. Either
            # therefore preserves the common low zero bits, including wrap.
            return math.gcd(left, right)
    if tag == "TernaryExpr":
        return math.gcd(guaranteed_multiple(expr.if_true, bindings),
                        guaranteed_multiple(expr.if_false, bindings))
    return 1
