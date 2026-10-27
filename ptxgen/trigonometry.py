"""Self-contained f32 sine/cosine with full-range integer argument reduction.

For |x| >= 1/2, multiply the exact 24-bit significand by floor(2/pi * 2**256).
Keep the quadrant and 64 fractional bits, rounding to the nearest quadrant.
The reciprocal truncation contributes less than 2**-128 turns even at FLT_MAX;
extracting 64 fraction bits adds less than 2**-64 turns. No large floating-point
subtraction or float-to-int overflow is involved. Evaluate Taylor polynomials
in f64 on [-pi/4, pi/4] and convert once to f32. Smaller inputs skip reduction.

This is an accuracy-oriented lowering, not a fast SFU approximation or a claim
of correctly rounded results for all 2**32 inputs. It needs hardware qualification.
"""
import math
import struct

# First 256 fractional bits of 2/pi. Tests independently regenerate these with
# high-precision arithmetic; runtime compilation needs only the standard library.
TWO_OVER_PI = int("a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebbc561", 16)


def f64(value):
    return "0d" + struct.pack(">d", value).hex().upper()


def emit_trig(e, destination, x, *, cosine):
    reg, emit = e.reg, e.emit
    bits, magnitude, exponent, mantissa, shift, word, offset = [reg("s32") for _ in range(7)]
    quadrant, sign = reg("s32"), reg("s32")
    special, small, negative, odd = [reg("pred") for _ in range(4)]
    small_label, polynomial, invalid, end = [e.label() for _ in range(4)]
    reduced, square, accumulator, sine, cosine_value = [reg("f64") for _ in range(5)]
    emit(f"mov.b32 {bits}, {x};")
    emit(f"and.b32 {magnitude}, {bits}, 2147483647;")
    emit(f"setp.ge.u32 {special}, {magnitude}, 2139095040;")
    emit(f"@{special} bra {invalid};")
    emit(f"setp.lt.u32 {small}, {magnitude}, 1056964608;")  # |x| < 0.5
    emit(f"@{small} bra {small_label};")
    emit(f"shr.u32 {exponent}, {magnitude}, 23;")
    emit(f"and.b32 {mantissa}, {magnitude}, 8388607;")
    emit(f"or.b32 {mantissa}, {mantissa}, 8388608;")
    # x = mantissa * 2**(biased_exponent - 150). The binary point in
    # mantissa * TWO_OVER_PI is at bit 406 - biased_exponent (152..280).
    emit(f"sub.s32 {shift}, 406, {exponent};")
    emit(f"sub.s32 {word}, {shift}, 64;")
    emit(f"shr.u32 {word}, {word}, 5;")
    emit(f"and.b32 {offset}, {shift}, 31;")
    product = [reg("s32") for _ in range(10)]
    carry, wide = reg("u64"), reg("u64")
    high = reg("s32")
    emit(f"mov.u64 {carry}, 0;")
    for i in range(8):
        limb = (TWO_OVER_PI >> (32 * i)) & 0xffffffff
        emit(f"mul.wide.u32 {wide}, {mantissa}, {limb};")
        emit(f"add.u64 {wide}, {wide}, {carry};")
        emit(f"mov.b64 {{{product[i]}, {high}}}, {wide};")
        emit(f"cvt.u64.u32 {carry}, {high};")
    emit(f"mov.b32 {product[8]}, {high};")
    emit(f"mov.u32 {product[9]}, 0;")
    window = [reg("s32") for _ in range(4)]
    select = reg("pred")
    for d in window:
        emit(f"mov.u32 {d}, 0;")
    # word is 2..6. Explicit predication avoids local memory/table addresses.
    for i in range(2, 7):
        emit(f"setp.eq.u32 {select}, {word}, {i};")
        for j, d in enumerate(window):
            emit(f"@{select} mov.b32 {d}, {product[i+j]};")
    extracted = [reg("s32") for _ in range(3)]
    for i, d in enumerate(extracted):
        emit(f"mov.b64 {wide}, {{{window[i]}, {window[i+1]}}};")
        emit(f"shr.u64 {wide}, {wide}, {offset};")
        emit(f"cvt.u32.u64 {d}, {wide};")
    emit(f"and.b32 {quadrant}, {extracted[2]}, 3;")
    emit(f"shr.u32 {sign}, {extracted[1]}, 31;")
    emit(f"add.u32 {quadrant}, {quadrant}, {sign};")
    emit(f"mov.b64 {wide}, {{{extracted[0]}, {extracted[1]}}};")
    # Reinterpret the fraction as signed: upper half becomes fraction - 1.
    emit(f"cvt.rn.f64.s64 {reduced}, {wide};")
    emit(f"mul.rn.f64 {reduced}, {reduced}, {f64(math.ldexp(math.pi / 2, -64))};")
    emit(f"bra {polynomial};")
    emit(small_label + ":")
    positive = reg("f32")
    emit(f"mov.b32 {positive}, {magnitude};")
    emit(f"cvt.f64.f32 {reduced}, {positive};")
    emit(f"mov.u32 {quadrant}, 0;")
    emit(polynomial + ":")
    emit(f"mul.rn.f64 {square}, {reduced}, {reduced};")
    # sin(r) = r * (1 + z * (-1/3! + z * (1/5! + ... + z/13!))).
    emit(f"mov.f64 {accumulator}, {f64(1 / math.factorial(13))};")
    for n in (11, 9, 7, 5, 3):
        coefficient = (-1 if ((n-1)//2) % 2 else 1) / math.factorial(n)
        emit(f"mul.rn.f64 {accumulator}, {accumulator}, {square};")
        emit(f"add.rn.f64 {accumulator}, {accumulator}, {f64(coefficient)};")
    emit(f"mul.rn.f64 {accumulator}, {accumulator}, {square};")
    emit(f"add.rn.f64 {accumulator}, {accumulator}, {f64(1)};")
    emit(f"mul.rn.f64 {sine}, {accumulator}, {reduced};")
    # cos(r) = 1 + z * (-1/2! + ... + z**5/12!).
    emit(f"mov.f64 {accumulator}, {f64(1 / math.factorial(12))};")
    for n in (10, 8, 6, 4, 2):
        coefficient = (-1 if (n//2) % 2 else 1) / math.factorial(n)
        emit(f"mul.rn.f64 {accumulator}, {accumulator}, {square};")
        emit(f"add.rn.f64 {accumulator}, {accumulator}, {f64(coefficient)};")
    emit(f"mul.rn.f64 {accumulator}, {accumulator}, {square};")
    emit(f"add.rn.f64 {cosine_value}, {accumulator}, {f64(1)};")
    if cosine:
        emit(f"add.u32 {quadrant}, {quadrant}, 1;")
    emit(f"and.b32 {sign}, {quadrant}, 1;")
    emit(f"setp.ne.u32 {odd}, {sign}, 0;")
    emit(f"selp.f64 {accumulator}, {cosine_value}, {sine}, {odd};")
    emit(f"and.b32 {sign}, {quadrant}, 2;")
    emit(f"shl.b32 {sign}, {sign}, 30;")
    if not cosine:
        emit(f"and.b32 {bits}, {bits}, 2147483648;")
        emit(f"xor.b32 {sign}, {sign}, {bits};")
    emit(f"cvt.rn.f32.f64 {destination}, {accumulator};")
    emit(f"mov.b32 {bits}, {destination};")
    emit(f"xor.b32 {bits}, {bits}, {sign};")
    emit(f"mov.b32 {destination}, {bits};")
    emit(f"bra {end};")
    emit(invalid + ":")
    emit(f"mov.f32 {destination}, 0f7FC00000;")
    emit(end + ":")
