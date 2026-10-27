"""Binary32 instructions and stable activation formulas.

The emitter supplies register allocation and instruction storage; these
functions do not inspect fixtures, metadata or lexical bindings.
"""
import math

from .errors import CompileError, require
from .scalars import f32


def emit_math(emitter, destination, op, args):
    binary = {"add": "add.rn.f32", "subtract": "sub.rn.f32", "multiply": "mul.rn.f32",
              "divide": "div.rn.f32", "maximum": "max.NaN.f32", "minimum": "min.NaN.f32"}
    if op in binary:
        require(len(args) == 2, f"bad arity for {op}")
        emitter.emit(f"{binary[op]} {destination}, {args[0]}, {args[1]};")
    elif op in ("copy", "pad"):
        # Emmy 0.3.11's scalar pad spelling is only zero-width padding.
        # Nonempty padding is not represented by this unary operation.
        require(len(args) == 1, f"bad arity for {op}")
        emitter.emit(f"mov.f32 {destination}, {args[0]};")
    elif op == "where":
        require(len(args) == 3, "bad arity for where")
        condition = emitter.reg("pred")
        # NumPy truth: both signed zeros are false; NaN is true.
        emitter.emit(f"setp.neu.f32 {condition}, {args[0]}, {f32(0)};")
        emitter.emit(f"selp.f32 {destination}, {args[1]}, {args[2]}, {condition};")
    elif op in ("floor", "ceil", "trunc", "rint"):
        require(len(args) == 1, f"bad arity for {op}")
        rounding = {"floor": "rmi", "ceil": "rpi", "trunc": "rzi", "rint": "rni"}[op]
        # Round to an integral f32 value, without an s32 range restriction
        # or FTZ (which would change floor/ceil of subnormal inputs).
        emitter.emit(f"cvt.{rounding}.f32.f32 {destination}, {args[0]};")
    elif op in ("sqrt", "reciprocal", "negative", "abs"):
        require(len(args) == 1, f"bad arity for {op}")
        opcode = {"sqrt": "sqrt.rn", "reciprocal": "rcp.rn", "negative": "neg", "abs": "abs"}[op]
        emitter.emit(f"{opcode}.f32 {destination}, {args[0]};")
    elif op == "relu":
        require(len(args) == 1, "bad arity for relu")
        emitter.emit(f"max.NaN.f32 {destination}, {args[0]}, {f32(0)};")
    elif op in ("exp", "exp_fast"):
        # The pinned typed metadata gives exp_fast the same np.exp
        # semantics; both use this backend's existing approximate exp.
        require(len(args) == 1, f"bad arity for {op}")
        scaled = emitter.reg("f32")
        emitter.emit(f"mul.rn.f32 {scaled}, {args[0]}, {f32(math.log2(math.e))};")
        emitter.emit(f"ex2.approx.f32 {destination}, {scaled};")
    elif op in ("exp2", "log2", "log10", "sign"):
        require(len(args) == 1, f"bad arity for {op}")
        x = args[0]
        if op in ("exp2", "log2"):
            opcode = "ex2" if op == "exp2" else "lg2"
            emitter.emit(f"{opcode}.approx.f32 {destination}, {x};")
        elif op == "log10":
            base2 = emitter.reg("f32")
            emitter.emit(f"lg2.approx.f32 {base2}, {x};")
            emitter.emit(f"mul.rn.f32 {destination}, {base2}, {f32(math.log10(2))};")
        else:
            # NumPy sign maps BOTH zeros to +0 and propagates NaN.
            # Keep x intact until every comparison is done (d may alias x).
            positive, negative, number = [emitter.reg("pred") for _ in range(3)]
            nonnegative, signed = emitter.reg("f32"), emitter.reg("f32")
            emitter.emit(f"setp.gt.f32 {positive}, {x}, {f32(0)};")
            emitter.emit(f"setp.lt.f32 {negative}, {x}, {f32(0)};")
            emitter.emit(f"selp.f32 {nonnegative}, {f32(1)}, {f32(0)}, {positive};")
            emitter.emit(f"selp.f32 {signed}, {f32(-1)}, {nonnegative}, {negative};")
            emitter.emit(f"setp.num.f32 {number}, {x}, {x};")
            emitter.emit(f"selp.f32 {destination}, {signed}, {x}, {number};")
    elif op in ("sin", "cos"):
        require(len(args) == 1, f"bad arity for {op}")
        from .trigonometry import emit_trig
        emit_trig(emitter, destination, args[0], cosine=op == "cos")
    elif op == "softplus":
        require(len(args) == 1, "bad arity for softplus")
        softplus(emitter, destination, args[0])
    elif op in ("erf", "gelu", "gelu_tanh"):
        require(len(args) == 1, f"bad arity for {op}")
        if op == "gelu_tanh":
            x = args[0]
            square, factor, inner, exponent = [emitter.reg("f32") for _ in range(4)]
            # x * sigmoid(2u), u=sqrt(2/pi)*x*(1+0.044715*x*x).
            # Avoid 1+tanh(u) cancellation in the negative tail.
            emitter.emit(f"mul.rn.f32 {square}, {x}, {x};")
            emitter.emit(f"fma.rn.f32 {factor}, {f32(0.044715)}, {square}, {f32(1)};")
            emitter.emit(f"mul.rn.f32 {inner}, {x}, {factor};")
            emitter.emit(f"mul.rn.f32 {exponent}, {inner}, {f32(2 * math.sqrt(2 / math.pi))};")
            activation(emitter, destination, "silu", x, exponent)
        else:
            erf_activation(emitter, destination, op, args[0])
    elif op in ("rsqrt", "log", "square", "sigmoid", "silu", "tanh"):
        require(len(args) == 1, f"bad arity for {op}")
        x = args[0]
        if op == "rsqrt":
            emitter.emit(f"rsqrt.approx.f32 {destination}, {x};")
        elif op == "square":
            emitter.emit(f"mul.rn.f32 {destination}, {x}, {x};")
        elif op == "log":
            base2 = emitter.reg("f32")
            emitter.emit(f"lg2.approx.f32 {base2}, {x};")
            emitter.emit(f"mul.rn.f32 {destination}, {base2}, {f32(math.log(2))};")
        else:
            activation(emitter, destination, op, x)
    else:
        raise CompileError(f"unsupported elementwise operation {op}")



def softplus(emitter, destination, x):
    # logaddexp(0,x) = max(x,0) + log1p(exp(-abs(x))).
    # log1p(e) = 2 atanh(e/(2+e)); u <= 1/3 allows a fixed series.
    magnitude, scaled, e, denominator, u, u2, p, q, base = [
        emitter.reg("f32") for _ in range(9)]
    positive, small = emitter.reg("pred"), emitter.reg("pred")
    emitter.emit(f"abs.f32 {magnitude}, {x};")
    emitter.emit(f"mul.rn.f32 {scaled}, {magnitude}, {f32(-math.log2(math.e))};")
    emitter.emit(f"ex2.approx.f32 {e}, {scaled};")
    emitter.emit(f"add.rn.f32 {denominator}, {e}, {f32(2)};")
    emitter.emit(f"div.rn.f32 {u}, {e}, {denominator};")
    emitter.emit(f"mul.rn.f32 {u2}, {u}, {u};")
    emitter.emit(f"mul.rn.f32 {p}, {u2}, {f32(1 / 13)};")
    emitter.emit(f"add.rn.f32 {p}, {p}, {f32(1 / 11)};")
    for divisor in (9, 7, 5, 3, 1):
        emitter.emit(f"mul.rn.f32 {p}, {p}, {u2};")
        emitter.emit(f"add.rn.f32 {p}, {p}, {f32(1 / divisor)};")
    emitter.emit(f"mul.rn.f32 {q}, {u}, {p};")
    emitter.emit(f"add.rn.f32 {q}, {q}, {q};")
    # Avoid halving subnormals and losing their low bit. Here replacing
    # log1p(e) by e costs at most e/2 <= 2^-25 relative in real arithmetic.
    emitter.emit(f"setp.le.f32 {small}, {e}, {f32(2 ** -24)};")
    emitter.emit(f"selp.f32 {q}, {e}, {q}, {small};")
    # Read x before the only destination write: in-place assignment is safe.
    emitter.emit(f"setp.gt.f32 {positive}, {x}, {f32(0)};")
    emitter.emit(f"selp.f32 {base}, {x}, {f32(0)}, {positive};")
    emitter.emit(f"add.rn.f32 {destination}, {base}, {q};")



def polynomial(emitter, variable, coefficients):
    """Horner evaluation, coefficients in ascending order; separate RN steps."""
    result = emitter.reg("f32")
    emitter.emit(f"mov.f32 {result}, {f32(coefficients[-1])};")
    for coefficient in reversed(coefficients[:-1]):
        emitter.emit(f"mul.rn.f32 {result}, {result}, {variable};")
        emitter.emit(f"add.rn.f32 {result}, {result}, {f32(coefficient)};")
    return result



def erf_activation(emitter, destination, op, x):
    # Small z: erf(z)=z*sum_n 2/sqrt(pi)*(-1)^n*z^(2n)/(n!*(2n+1)).
    # Large z: compute erfc directly, preserving GELU's negative tail.
    z = x
    if op == "gelu":
        z = emitter.reg("f32")
        emitter.emit(f"mul.rn.f32 {z}, {x}, {f32(1 / math.sqrt(2))};")
    magnitude, square, series = [emitter.reg("f32") for _ in range(3)]
    small = emitter.reg("pred")
    emitter.emit(f"abs.f32 {magnitude}, {z};")
    emitter.emit(f"setp.lt.f32 {small}, {magnitude}, {f32(0.5)};")
    emitter.emit(f"mul.rn.f32 {square}, {z}, {z};")
    p = polynomial(emitter, square, [2 / math.sqrt(math.pi) * (-1)**n /
                                (math.factorial(n) * (2*n + 1)) for n in range(7)])
    emitter.emit(f"mul.rn.f32 {series}, {z}, {p};")
    if op == "gelu":
        emitter.emit(f"fma.rn.f32 {series}, {f32(0.5)}, {series}, {f32(0.5)};")
        emitter.emit(f"mul.rn.f32 {series}, {x}, {series};")

    # Numerical Recipes, 2nd ed., sec.6.2: erfc(z) ~ t*exp(-z*z+c(t)).
    # Mathematical coefficients only; the PTX lowering is implemented here.
    half, denominator, t, scaled_c, scaled_x, exponent, complement = [
        emitter.reg("f32") for _ in range(7)]
    emitter.emit(f"mul.rn.f32 {half}, {magnitude}, {f32(0.5)};")
    emitter.emit(f"add.rn.f32 {denominator}, {half}, {f32(1)};")
    emitter.emit(f"div.rn.f32 {t}, {f32(1)}, {denominator};")
    c = polynomial(emitter, t, [-1.26551223, 1.00002368, 0.37409196, 0.09678418,
                           -0.18628806, 0.27886807, -1.13520398, 1.48851587,
                           -0.82215223, 0.17087277])
    emitter.emit(f"mul.rn.f32 {scaled_c}, {c}, {f32(math.log2(math.e))};")
    scale = -math.log2(math.e) * (0.5 if op == "gelu" else 1)
    emitter.emit(f"mul.rn.f32 {scaled_x}, {x}, {f32(scale)};")
    # Use the original x, avoiding the extra rounding from squaring z.
    emitter.emit(f"fma.rn.f32 {exponent}, {scaled_x}, {x}, {scaled_c};")
    emitter.emit(f"ex2.approx.f32 {complement}, {exponent};")
    emitter.emit(f"mul.rn.f32 {complement}, {t}, {complement};")
    general = emitter.reg("f32")
    if op == "erf":
        emitter.emit(f"sub.rn.f32 {general}, {f32(1)}, {complement};")
        # PTX copies the FIRST source's sign onto the SECOND source's magnitude.
        emitter.emit(f"copysign.f32 {general}, {x}, {general};")
    else:
        negative = emitter.reg("pred")
        tail, positive = emitter.reg("f32"), emitter.reg("f32")
        emitter.emit(f"mul.rn.f32 {tail}, {x}, {f32(0.5)};")
        emitter.emit(f"mul.rn.f32 {tail}, {tail}, {complement};")
        emitter.emit(f"fma.rn.f32 {positive}, {f32(-0.5)}, {complement}, {f32(1)};")
        emitter.emit(f"mul.rn.f32 {positive}, {x}, {positive};")
        emitter.emit(f"setp.lt.f32 {negative}, {x}, {f32(0)};")
        emitter.emit(f"selp.f32 {general}, {tail}, {positive}, {negative};")
    # All x reads precede the only destination write, including in-place use.
    emitter.emit(f"selp.f32 {destination}, {series}, {general}, {small};")



def activation(emitter, destination, op, x, exponent_input=None):
    # Only nonpositive exp arguments: avoid overflow for finite inputs.
    magnitude, scaled, z, denominator = [emitter.reg("f32") for _ in range(4)]
    negative = emitter.reg("pred")
    emitter.emit(f"abs.f32 {magnitude}, {x if exponent_input is None else exponent_input};")
    factor = -math.log2(math.e) * (2 if op == "tanh" else 0.5 if op == "silu" else 1)
    emitter.emit(f"mul.rn.f32 {scaled}, {magnitude}, {f32(factor)};")
    emitter.emit(f"ex2.approx.f32 {z}, {scaled};")
    exponential = z
    if op == "silu":
        exponential = emitter.reg("f32")
        emitter.emit(f"mul.rn.f32 {exponential}, {z}, {z};")
    emitter.emit(f"add.rn.f32 {denominator}, {f32(1)}, {exponential};")
    emitter.emit(f"setp.lt.f32 {negative}, {x}, {f32(0)};")
    if op == "silu":
        # At x=-100 sigmoid(x) is already subnormal. Multiplying that
        # rounded value by x loses useful bits. Split the exponential so
        # (x*exp(x/2))*exp(x/2) underflows only at the final product.
        tail, numerator = emitter.reg("f32"), emitter.reg("f32")
        emitter.emit(f"mul.rn.f32 {tail}, {x}, {z};")
        emitter.emit(f"mul.rn.f32 {tail}, {tail}, {z};")
        emitter.emit(f"selp.f32 {numerator}, {tail}, {x}, {negative};")
        emitter.emit(f"div.rn.f32 {destination}, {numerator}, {denominator};")
    elif op == "sigmoid":
        numerator, sigmoid = emitter.reg("f32"), emitter.reg("f32")
        emitter.emit(f"selp.f32 {numerator}, {z}, {f32(1)}, {negative};")
        emitter.emit(f"div.rn.f32 {sigmoid}, {numerator}, {denominator};")
        emitter.emit(f"mov.f32 {destination}, {sigmoid};")
    else:
        # tanh.approx's documented error bound is too loose for our
        # tolerance. This exp ratio avoids overflow; its near-zero
        # cancellation is replaced by the degree-five Taylor polynomial.
        numerator, positive, negated, general = [emitter.reg("f32") for _ in range(4)]
        emitter.emit(f"sub.rn.f32 {numerator}, {f32(1)}, {z};")
        emitter.emit(f"div.rn.f32 {positive}, {numerator}, {denominator};")
        emitter.emit(f"neg.f32 {negated}, {positive};")
        emitter.emit(f"selp.f32 {general}, {negated}, {positive}, {negative};")
        x2, term, poly = [emitter.reg("f32") for _ in range(3)]
        emitter.emit(f"mul.rn.f32 {x2}, {x}, {x};")
        emitter.emit(f"mul.rn.f32 {term}, {x2}, {f32(2 / 15)};")
        emitter.emit(f"add.rn.f32 {term}, {term}, {f32(-1 / 3)};")
        emitter.emit(f"mul.rn.f32 {term}, {term}, {x2};")
        emitter.emit(f"add.rn.f32 {term}, {term}, {f32(1)};")
        emitter.emit(f"mul.rn.f32 {poly}, {x}, {term};")
        small = emitter.reg("pred")
        emitter.emit(f"setp.le.f32 {small}, {magnitude}, {f32(1 / 16)};")
        emitter.emit(f"selp.f32 {destination}, {poly}, {general}, {small};")
