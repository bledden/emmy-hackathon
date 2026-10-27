"""Small non-wrapping interval analysis for integer division semantics.

Unknown and potentially wrapping arithmetic returns no fact. Facts about
bindings are supplied by the emitter, which owns scope and invalidation.
"""
MIN, MAX = -(2**31), 2**31-1


def interval(expr, variables):
    tag = type(expr).__name__
    if tag == "Literal":
        value = int(expr.value)
        return (value, value) if MIN <= value <= MAX else None
    if tag == "Var":
        return variables.get(expr.name)
    if tag == "TernaryExpr":
        a, b = interval(expr.if_true, variables), interval(expr.if_false, variables)
        return (min(a[0], b[0]), max(a[1], b[1])) if a and b else None
    if tag != "BinaryExpr":
        return None
    a, b = interval(expr.left, variables), interval(expr.right, variables)
    if expr.op in ("==", "!=", "<", "<=", ">", ">=", "and", "or", "&&", "||"):
        return (0, 1)
    if expr.op == "&":
        for bound in (a, b):
            if bound is not None and bound[0] == bound[1] and bound[0] >= 0:
                return (0, bound[0])
    if a is None or b is None:
        return None
    if expr.op == "+":
        result = a[0]+b[0], a[1]+b[1]
    elif expr.op == "-":
        result = a[0]-b[1], a[1]-b[0]
    elif expr.op == "*":
        values = [x*y for x in a for y in b];result = min(values), max(values)
    elif expr.op in ("/", "//", "%") and not b[0] <= 0 <= b[1]:
        if expr.op == "%":
            result = (0, b[1]-1) if b[0] > 0 else (b[0]+1, 0)
        else:
            values = [x//y for x in a for y in b];result = min(values), max(values)
    else:
        return None
    return result if MIN <= result[0] <= result[1] <= MAX else None
