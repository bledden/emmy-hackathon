"""Exact physical-thread facts for counted barrier participation.

No memory values are guessed. Integer facts follow registers, not source names.
A failed proof declines the extension; it never weakens the ordinary CTA proof.
"""
MIN, MAX = -(1 << 31), (1 << 31) - 1


def whole_warps(threads):
    return all(all(t in threads for t in range(w * 32, w * 32 + 32))
               for w in {t // 32 for t in threads})


def values(expr, bindings, block):
    tag = type(expr).__name__
    if tag == 'Literal':
        dtype = getattr(expr.dtype, 'name', expr.dtype)
        if dtype not in ('int', 'i32'):
            return None
        n = (int(expr.value) + (1 << 31)) % (1 << 32) - (1 << 31)
        return (n,) * block
    if tag == 'Var':
        return bindings.get(expr.name)
    if tag == 'TernaryExpr':
        parts = [values(x, bindings, block) for x in (expr.cond, expr.if_true, expr.if_false)]
        if any(x is None for x in parts):
            return None
        return tuple(a if c else b for c, a, b in zip(*parts))
    if tag != 'BinaryExpr':
        return None
    a, b = values(expr.left, bindings, block), values(expr.right, bindings, block)
    if a is None or b is None:
        return None
    op = expr.op
    if op in ('<<', '>>'):
        # PTX uses an unsigned shift count and clamps it to the word width.
        # Preserve signed arithmetic right shift and wrap the left result.
        def shift(x, y):
            n = min(y & 0xffffffff, 32)
            result = x << n if op == '<<' else x >> n
            return (result + (1 << 31)) % (1 << 32) - (1 << 31)
        return tuple(shift(x, y) for x, y in zip(a, b))
    operators = {'+': lambda x,y:x+y, '-':lambda x,y:x-y, '*':lambda x,y:x*y,
                 '&':lambda x,y:x&y, '|':lambda x,y:x|y, '^':lambda x,y:x^y,
                 '==':lambda x,y:x==y, '!=':lambda x,y:x!=y,
                 '<':lambda x,y:x<y, '<=':lambda x,y:x<=y,
                 '>':lambda x,y:x>y, '>=':lambda x,y:x>=y,
                 'and':lambda x,y:bool(x) and bool(y), 'or':lambda x,y:bool(x) or bool(y),
                 '&&':lambda x,y:bool(x) and bool(y), '||':lambda x,y:bool(x) or bool(y)}
    if op in ('/', '//', '%'):
        if any(y == 0 or (x == MIN and y == -1) for x,y in zip(a,b)):
            return None
        return tuple(x % y if op == '%' else x // y for x,y in zip(a,b))
    if op not in operators:
        return None
    return tuple((int(operators[op](x,y)) + (1 << 31)) % (1 << 32) - (1 << 31)
                 for x,y in zip(a,b))


def trips(starts, stops, step):
    if starts is None or stops is None or not 0 < step <= MAX:
        return None
    result = []
    for start, stop in zip(starts, stops):
        if not MIN <= start <= MAX or not MIN <= stop <= MAX:
            return None
        n = max(0, (stop-start+step-1)//step)
        if start+n*step > MAX:
            return None
        result.append(n)
    return tuple(result)


def same_on(values, threads):
    return values is not None and len({values[t] for t in threads}) <= 1


def per_warp(values, threads):
    return values is not None and all(same_on(values, {t for t in threads if t//32 == w})
                                     for w in {t//32 for t in threads})
