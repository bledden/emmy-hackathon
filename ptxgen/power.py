"""Direct f32 power lowering, with f64 log/exp polynomials and IEEE cases.

For finite nonzero |x|, normalize x exactly, range-reduce its mantissa around
one, and evaluate log via 2*atanh((m-1)/(m+1)). Compute y*log2(x), then exp2
with an integer exponent and a Taylor polynomial on [0, ln(2)). All internal
arithmetic uses explicit f64 rounding; only the final result rounds to f32.
No external math function, renderer or device library is called. This is not
an exhaustive correct-rounding guarantee; numerical evidence is separate.
"""
import math
from .trigonometry import f64


def emit_power(e, destination, x, y):
    reg, emit = e.reg, e.emit
    xb, ax, yb, ay, yi, parity, exponent, scale, high, low = [reg('s32') for _ in range(10)]
    test, negative, ynegative, integer, odd, signed = [reg('pred') for _ in range(6)]
    labels = {k:e.label() for k in ('one','invalid','yinf','parity_done','xzero','xinf',
                                    'normalized','mantissa','zero','infinity','signed','end')}
    def jump(pred, label, invert=False):emit(f"@{'!' if invert else ''}{pred} bra {labels[label]};")
    def go(label):emit(f"bra {labels[label]};")
    def mark(label):emit(labels[label]+':')
    xf, yf, result = reg('f32'),reg('f32'),reg('f32')
    m,z,z2,p,logarithm,product,nf,remainder,answer,factor = [reg('f64') for _ in range(10)]
    emit(f'mov.b32 {xb}, {x};');emit(f'mov.b32 {yb}, {y};')
    emit(f'and.b32 {ax}, {xb}, 2147483647;');emit(f'and.b32 {ay}, {yb}, 2147483647;')
    emit(f'setp.lt.s32 {negative}, {xb}, 0;');emit(f'setp.lt.s32 {ynegative}, {yb}, 0;')
    emit(f'setp.ne.u32 {odd}, 0, 0;')
    emit(f'setp.eq.u32 {test}, {ay}, 0;');jump(test,'one')
    emit(f'setp.eq.u32 {test}, {xb}, 1065353216;');jump(test,'one')
    for bits in (ax,ay):
        emit(f'setp.gt.u32 {test}, {bits}, 2139095040;');jump(test,'invalid')
    emit(f'setp.eq.u32 {test}, {ay}, 2139095040;');jump(test,'yinf')
    # Every finite binary32 number of magnitude >= 2**24 is an even integer.
    emit(f'setp.eq.u32 {integer}, 0, 0;')
    emit(f'setp.ge.u32 {test}, {ay}, 1266679808;');jump(test,'parity_done')
    emit(f'cvt.rzi.s32.f32 {yi}, {y};')
    emit(f'cvt.rn.f32.s32 {yf}, {yi};')
    emit(f'setp.eq.f32 {integer}, {yf}, {y};')
    emit(f'and.b32 {parity}, {yi}, 1;')
    emit(f'setp.ne.u32 {odd}, {parity}, 0;')
    emit(f'and.pred {odd}, {odd}, {integer};')
    mark('parity_done')
    emit(f'setp.eq.u32 {test}, {ax}, 0;');jump(test,'xzero')
    emit(f'setp.eq.u32 {test}, {ax}, 2139095040;');jump(test,'xinf')
    emit(f'not.pred {test}, {integer};');emit(f'and.pred {test}, {test}, {negative};');jump(test,'invalid')
    # Exact normalization including subnormal bases. Keep m close to one so
    # log2(x) near x=1 never comes from subtracting nearly equal large terms.
    emit(f'mov.b32 {xf}, {ax};');emit(f'mov.s32 {scale}, 0;')
    emit(f'setp.ge.u32 {test}, {ax}, 8388608;');jump(test,'normalized')
    emit(f'mul.rn.f32 {xf}, {xf}, 0f4B800000;')  # exact *2**24
    emit(f'mov.b32 {ax}, {xf};');emit(f'mov.s32 {scale}, 24;')
    mark('normalized')
    emit(f'shr.u32 {exponent}, {ax}, 23;')
    emit(f'sub.s32 {exponent}, {exponent}, 127;')
    emit(f'sub.s32 {exponent}, {exponent}, {scale};')
    emit(f'and.b32 {parity}, {ax}, 8388607;')
    emit(f'or.b32 {parity}, {parity}, 1065353216;')
    emit(f'mov.b32 {xf}, {parity};');emit(f'cvt.f64.f32 {m}, {xf};')
    emit(f'setp.le.f64 {test}, {m}, {f64(math.sqrt(2))};');jump(test,'mantissa')
    emit(f'mul.rn.f64 {m}, {m}, {f64(.5)};')
    emit(f'add.s32 {exponent}, {exponent}, 1;')
    mark('mantissa')
    emit(f'sub.rn.f64 {z}, {m}, {f64(1.)};')
    emit(f'add.rn.f64 {p}, {m}, {f64(1.)};')
    emit(f'div.rn.f64 {z}, {z}, {p};');emit(f'mul.rn.f64 {z2}, {z}, {z};')
    emit(f'mov.f64 {p}, {f64(1/31)};')
    for k in range(29,0,-2):
        emit(f'mul.rn.f64 {p}, {p}, {z2};');emit(f'add.rn.f64 {p}, {p}, {f64(1/k)};')
    emit(f'mul.rn.f64 {logarithm}, {p}, {z};')
    emit(f'mul.rn.f64 {logarithm}, {logarithm}, {f64(2/math.log(2))};')
    emit(f'cvt.rn.f64.s32 {nf}, {exponent};')
    emit(f'add.rn.f64 {logarithm}, {logarithm}, {nf};')
    emit(f'cvt.f64.f32 {m}, {y};');emit(f'mul.rn.f64 {product}, {m}, {logarithm};')
    emit(f'setp.ge.f64 {test}, {product}, {f64(128.)};');jump(test,'infinity')
    emit(f'setp.le.f64 {test}, {product}, {f64(-150.)};');jump(test,'zero')
    emit(f'cvt.rmi.s32.f64 {exponent}, {product};')
    emit(f'cvt.rn.f64.s32 {nf}, {exponent};')
    emit(f'sub.rn.f64 {remainder}, {product}, {nf};')
    emit(f'mul.rn.f64 {remainder}, {remainder}, {f64(math.log(2))};')
    emit(f'mov.f64 {answer}, {f64(1/math.factorial(18))};')
    for k in range(17,-1,-1):
        emit(f'mul.rn.f64 {answer}, {answer}, {remainder};')
        emit(f'add.rn.f64 {answer}, {answer}, {f64(1/math.factorial(k))};')
    emit(f'add.s32 {high}, {exponent}, 1023;');emit(f'shl.b32 {high}, {high}, 20;')
    emit(f'mov.u32 {low}, 0;');emit(f'mov.b64 {factor}, {{{low}, {high}}};')
    emit(f'mul.rn.f64 {answer}, {answer}, {factor};');emit(f'cvt.rn.f32.f64 {result}, {answer};');go('signed')
    mark('yinf')
    emit(f'setp.eq.u32 {test}, {ax}, 1065353216;');jump(test,'one')
    emit(f'setp.gt.u32 {test}, {ax}, 1065353216;')
    emit(f'xor.pred {test}, {test}, {ynegative};');jump(test,'infinity');go('zero')
    mark('xzero');jump(ynegative,'infinity');go('zero')
    mark('xinf');jump(ynegative,'zero');go('infinity')
    mark('zero');emit(f'mov.f32 {result}, 0f00000000;');go('signed')
    mark('infinity');emit(f'mov.f32 {result}, 0f7F800000;')
    mark('signed')
    emit(f'and.pred {signed}, {negative}, {odd};')
    emit(f'mov.b32 {parity}, {result};')
    emit(f'@{signed} xor.b32 {parity}, {parity}, 2147483648;')
    emit(f'mov.b32 {destination}, {parity};');go('end')
    mark('invalid');emit(f'mov.f32 {destination}, 0f7FC00000;');go('end')
    mark('one');emit(f'mov.f32 {destination}, 0f3F800000;')
    mark('end')
