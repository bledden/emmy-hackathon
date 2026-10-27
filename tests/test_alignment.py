from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

from emmy.compiler.ir.expr import BinaryExpr, Literal, Var

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptxgen.compiler import guaranteed_multiple


def signed(value):
    return (value + 2**31) % 2**32 - 2**31


def evaluate(expr, x):
    if isinstance(expr, Literal):
        return signed(int(expr.value))
    if isinstance(expr, Var):
        return x
    a, b = evaluate(expr.left, x), evaluate(expr.right, x)
    if expr.op == "%":
        quotient = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        return a - quotient * b  # PTX rem truncates quotient toward zero.
    return signed({"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b}[expr.op]())


class AlignmentTests(unittest.TestCase):
    def test_signed_wrap_and_remainder_preserve_proven_low_bits(self):
        x = Var("x")
        literal = lambda n: Literal(n, "int")
        exprs = [
            BinaryExpr("%", BinaryExpr("*", x, literal(4)), literal(64)),
            BinaryExpr("%", BinaryExpr("*", x, literal(6)), literal(8)),
            BinaryExpr("*", BinaryExpr("*", x, literal(2**30)), literal(4)),
            BinaryExpr("+", BinaryExpr("*", x, literal(12)), literal(4)),
            BinaryExpr("-", BinaryExpr("*", x, literal(8)), literal(2)),
        ]
        rng = random.Random(817)
        values = [-2**31, -2**31 + 1, -1, 0, 1, 2**31 - 1]
        values += [rng.randrange(-2**31, 2**31) for _ in range(1000)]
        for expr in exprs:
            divisor = guaranteed_multiple(expr)
            self.assertEqual(divisor & (divisor - 1), 0)
            for value in values:
                result = evaluate(expr, value)
                self.assertEqual(result % divisor if divisor else result, 0)

    def test_unknown_variables_do_not_claim_alignment(self):
        x = Var("x")
        self.assertEqual(guaranteed_multiple(Literal(6, "int")), 2)
        self.assertEqual(guaranteed_multiple(x), 1)


if __name__ == "__main__":
    unittest.main()
