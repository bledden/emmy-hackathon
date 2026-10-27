"""Binary64 sine/cosine with 1152-bit reciprocal-pi argument reduction.

The 53-bit significand multiplies a constant integer using 32-bit limbs.
A 128-bit fraction is complemented as an integer before floating conversion,
so a fraction extremely close to the next quadrant does not cancel to zero.
At DBL_MAX, reciprocal truncation contributes <2**-128 turns. This finite
precision bound is not a universal correct-rounding claim.
"""
import math
from .wide_math import Math64, const

TWO_OVER_PI_BITS = 1152
TWO_OVER_PI = int(
    'a2f9836e4e441529fc2757d1f534ddc0db6295993c439041fe5163abdebbc561'
    'b7246e3a424dd2e006492eea09d1921cfe1deb1cb129a73ee88235f52ebb4484'
    'e99c7026b45f7e413991d639835339f49c845f8bbdf9283b1ff897ffde05980fe'
    'f2f118b5a0a6d1f6d367ecf27cb09b74f463f669e5fea2d7527bac7ebe5f17b'
    '3d0739f78a5292ea6bfb5fb11f8d5d08',16)


def emit_trig64(e,dst,x,cosine=False):
    m=Math64(e);r=e.reg;emit=e.emit
    low,high,maghigh,exponent,ml,mh,shift,word,offset,quadrant,sign=[r('s32') for _ in range(11)]
    small,invalid,polynomial,end=[e.label() for _ in range(4)]
    emit(f'mov.b64 {{{low}, {high}}}, {x};')
    magnitude=m.op('abs',x)
    m.jump(m.compare('nan',x,x),invalid)
    m.jump(m.compare('eq',magnitude,const(math.inf)),invalid)
    m.jump(m.compare('lt',magnitude,const(.5)),small)
    emit(f'and.b32 {maghigh}, {high}, 2147483647;')
    emit(f'shr.u32 {exponent}, {maghigh}, 20;')
    emit(f'and.b32 {mh}, {maghigh}, 1048575;')
    emit(f'or.b32 {mh}, {mh}, 1048576;')
    emit(f'mov.b32 {ml}, {low};')
    emit(f'sub.s32 {shift}, {TWO_OVER_PI_BITS+1075}, {exponent};')
    emit(f'sub.s32 {word}, {shift}, 128;')
    emit(f'shr.u32 {word}, {word}, 5;')
    emit(f'and.b32 {offset}, {shift}, 31;')
    product=[r('s32') for _ in range(39)]
    carry,wide,addend=r('u64'),r('u64'),r('u64');hi=r('s32')
    emit(f'mov.u64 {carry}, 0;')
    for i in range(36):
        limb=(TWO_OVER_PI>>(32*i))&0xffffffff
        emit(f'mul.wide.u32 {wide}, {ml}, {limb};')
        emit(f'add.u64 {wide}, {wide}, {carry};')
        emit(f'mov.b64 {{{product[i]}, {hi}}}, {wide};')
        emit(f'cvt.u64.u32 {carry}, {hi};')
    emit(f'mov.b32 {product[36]}, {hi};')
    emit(f'mov.u64 {carry}, 0;')
    for i in range(36):
        limb=(TWO_OVER_PI>>(32*i))&0xffffffff
        emit(f'mul.wide.u32 {wide}, {mh}, {limb};')
        emit(f'cvt.u64.u32 {addend}, {product[i+1]};')
        emit(f'add.u64 {wide}, {wide}, {addend};')
        emit(f'add.u64 {wide}, {wide}, {carry};')
        emit(f'mov.b64 {{{product[i+1]}, {hi}}}, {wide};')
        emit(f'cvt.u64.u32 {carry}, {hi};')
    emit(f'mov.b32 {product[37]}, {hi};');emit(f'mov.u32 {product[38]}, 0;')
    window=[r('s32') for _ in range(6)];select=r('pred')
    for d in window:emit(f'mov.u32 {d}, 0;')
    for i in range(1,34):
        emit(f'setp.eq.u32 {select}, {word}, {i};')
        for j,d in enumerate(window):emit(f'@{select} mov.b32 {d}, {product[i+j]};')
    extracted=[r('s32') for _ in range(5)]
    for i,d in enumerate(extracted):
        emit(f'mov.b64 {wide}, {{{window[i]}, {window[i+1]}}};')
        emit(f'shr.u64 {wide}, {wide}, {offset};')
        emit(f'cvt.u32.u64 {d}, {wide};')
    emit(f'and.b32 {quadrant}, {extracted[4]}, 3;')
    emit(f'shr.u32 {sign}, {extracted[3]}, 31;')
    emit(f'add.u32 {quadrant}, {quadrant}, {sign};')
    negative=r('pred');emit(f'setp.ne.u32 {negative}, {sign}, 0;')
    emit(f'cvt.u64.u32 {carry}, {sign};')
    for d in extracted[:4]:
        emit(f'@{negative} not.b32 {d}, {d};')
        emit(f'cvt.u64.u32 {wide}, {d};')
        emit(f'add.u64 {wide}, {wide}, {carry};')
        emit(f'mov.b64 {{{d}, {hi}}}, {wide};')
        emit(f'cvt.u64.u32 {carry}, {hi};')
    upper,lower,reduced=[r('f64') for _ in range(3)]
    for d,a,b in [(upper,extracted[2],extracted[3]),(lower,extracted[0],extracted[1])]:
        emit(f'mov.b64 {wide}, {{{a}, {b}}};')
        emit(f'cvt.rn.f64.u64 {d}, {wide};')
    emit(f'mul.rn.f64 {upper}, {upper}, {const(2.0**-64)};')
    emit(f'mul.rn.f64 {lower}, {lower}, {const(2.0**-128)};')
    emit(f'add.rn.f64 {reduced}, {upper}, {lower};')
    emit(f'@{negative} neg.f64 {reduced}, {reduced};')
    emit(f'mul.rn.f64 {reduced}, {reduced}, {const(math.pi/2)};')
    m.go(polynomial)
    m.mark(small);m.move(reduced,magnitude);emit(f'mov.u32 {quadrant}, 0;')
    m.mark(polynomial)
    square=m.op('mul.rn',reduced,reduced)
    sine=m.op('mul.rn',reduced,m.polynomial(square,[(-1)**n/math.factorial(2*n+1) for n in range(11)]))
    cos=m.polynomial(square,[(-1)**n/math.factorial(2*n) for n in range(12)])
    if cosine:emit(f'add.u32 {quadrant}, {quadrant}, 1;')
    emit(f'and.b32 {sign}, {quadrant}, 1;')
    odd=r('pred');emit(f'setp.ne.u32 {odd}, {sign}, 0;')
    answer=m.select(odd,cos,sine)
    emit(f'and.b32 {sign}, {quadrant}, 2;')
    emit(f'shl.b32 {sign}, {sign}, 30;')
    if not cosine:
        emit(f'and.b32 {high}, {high}, 2147483648;')
        emit(f'xor.b32 {sign}, {sign}, {high};')
    emit(f'setp.ne.u32 {negative}, {sign}, 0;')
    emit(f'@{negative} neg.f64 {answer}, {answer};')
    m.move(dst,answer);m.go(end)
    m.mark(invalid);m.move(dst,const(math.nan));m.mark(end)
