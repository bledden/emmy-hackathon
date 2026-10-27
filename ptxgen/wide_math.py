"""Direct binary64 scalar math; no f32 detour or external device library.

Range-reduced exp uses a split ln(2), a degree-20 Taylor polynomial, and two
normal powers of two so subnormal results round only at the final scaling.
Log normalizes the exact significand and evaluates 2*atanh((m-1)/(m+1)).
These are tolerance-oriented implementations, not correct-rounding claims.
"""
import math
from decimal import Decimal, localcontext
from .scalars import literal


def const(x):
    return literal(x, 'f64')


with localcontext() as ctx:
    ctx.prec = 80
    _ln2 = Decimal(2).ln()
    LN2_HI = float.fromhex('0x1.62e42fefa0000p-1')
    LN2_LO = float(_ln2 - Decimal(LN2_HI))


class Math64:
    def __init__(self, emitter):
        self.e = emitter

    def reg(self):return self.e.reg('f64')
    def emit(self, s):self.e.emit(s)
    def move(self, dst, src):self.emit(f'mov.f64 {dst}, {src};')
    def op(self, op, a, b=None):
        result = self.reg()
        self.emit(f'{op}.f64 {result}, {a}'+(f', {b}' if b is not None else '')+';')
        return result
    def compare(self, cmp, a, b):
        p = self.e.reg('pred');self.emit(f'setp.{cmp}.f64 {p}, {a}, {b};');return p
    def select(self, p, a, b):
        result=self.reg();self.emit(f'selp.f64 {result}, {a}, {b}, {p};');return result
    def jump(self, p, label, invert=False):self.emit(f'@{"!" if invert else ""}{p} bra {label};')
    def label(self):return self.e.label()
    def mark(self, label):self.emit(label+':')
    def go(self, label):self.emit(f'bra {label};')

    def exp(self, dst, x, base2=False, tail=None):
        invalid, zero, infinity, end = [self.label() for _ in range(4)]
        self.jump(self.compare('nan', x, x), invalid)
        if tail is None:
            self.jump(self.compare('ge', x, const(1024 if base2 else 710)), infinity)
            self.jump(self.compare('le', x, const(-1075 if base2 else -746)), zero)
        else:
            # A compensated log product may round to an exponent boundary
            # while its residual lies on the finite/nonzero side of it.
            self.jump(self.compare('gt',x,const(1024)),infinity)
            self.jump(self.compare('lt',x,const(-1075)),zero)
            ordinary=self.label()
            self.jump(self.compare('eq',x,const(-1075)),ordinary,True)
            self.move(dst,self.select(self.compare('gt',tail,const(0)),const(2**-1074),const(0)))
            self.go(end);self.mark(ordinary)
        scaled = x if base2 else self.op('mul.rn', x, const(1/math.log(2)))
        n = self.e.reg('s32');nf=self.reg()
        self.emit(f'cvt.rni.s32.f64 {n}, {scaled};')
        self.emit(f'cvt.rn.f64.s32 {nf}, {n};')
        if base2:
            fraction=self.op('sub.rn',x,nf)
            if tail is not None:fraction=self.op('add.rn',fraction,tail)
            r=self.op('mul.rn',fraction,const(math.log(2)))
        else:
            r=self.op('sub.rn',x,self.op('mul.rn',nf,const(LN2_HI)))
            r=self.op('sub.rn',r,self.op('mul.rn',nf,const(LN2_LO)))
        answer=self.reg();self.move(answer,const(1/math.factorial(20)))
        for k in range(19,-1,-1):
            self.emit(f'mul.rn.f64 {answer}, {answer}, {r};')
            self.emit(f'add.rn.f64 {answer}, {answer}, {const(1/math.factorial(k))};')
        # First scale exponent stays in [-1021,1023]. With |r|<=ln2/2,
        # this product is normal; second scale performs any subnormal rounding.
        first,second,high,low=[self.e.reg('s32') for _ in range(4)]
        self.emit(f'max.s32 {first}, {n}, -1021;')
        self.emit(f'min.s32 {first}, {first}, 1023;')
        self.emit(f'sub.s32 {second}, {n}, {first};')
        self.emit(f'mov.u32 {low}, 0;')
        for exponent in (first,second):
            factor=self.reg()
            self.emit(f'add.s32 {high}, {exponent}, 1023;')
            self.emit(f'shl.b32 {high}, {high}, 20;')
            self.emit(f'mov.b64 {factor}, {{{low}, {high}}};')
            self.emit(f'mul.rn.f64 {answer}, {answer}, {factor};')
        self.move(dst,answer);self.go(end)
        self.mark(zero);self.move(dst,const(0));self.go(end)
        self.mark(infinity);self.move(dst,const(math.inf));self.go(end)
        self.mark(invalid);self.move(dst,const(math.nan));self.mark(end)

    def log(self, dst, x, base='e', residual=None):
        invalid,zero,infinity,normalized,reduced,end=[self.label() for _ in range(6)]
        self.jump(self.compare('nan',x,x),invalid)
        self.jump(self.compare('eq',x,const(0)),zero)
        self.jump(self.compare('lt',x,const(0)),invalid)
        self.jump(self.compare('eq',x,const(math.inf)),infinity)
        m=self.reg();self.move(m,x)
        lo,hi,exponent,adjust=[self.e.reg('s32') for _ in range(4)]
        self.emit(f'mov.s32 {adjust}, 0;')
        self.jump(self.compare('ge',m,const(2.0**-1022)),normalized)
        self.emit(f'mul.rn.f64 {m}, {m}, {const(2.0**54)};')
        self.emit(f'mov.s32 {adjust}, 54;')
        self.mark(normalized)
        self.emit(f'mov.b64 {{{lo}, {hi}}}, {m};')
        self.emit(f'shr.u32 {exponent}, {hi}, 20;')
        self.emit(f'sub.s32 {exponent}, {exponent}, 1023;')
        self.emit(f'sub.s32 {exponent}, {exponent}, {adjust};')
        self.emit(f'and.b32 {hi}, {hi}, 1048575;')
        self.emit(f'or.b32 {hi}, {hi}, 1072693248;')
        self.emit(f'mov.b64 {m}, {{{lo}, {hi}}};')
        self.jump(self.compare('le',m,const(math.sqrt(2))),reduced)
        self.emit(f'mul.rn.f64 {m}, {m}, {const(.5)};')
        self.emit(f'add.s32 {exponent}, {exponent}, 1;')
        self.mark(reduced)
        z=self.op('div.rn',self.op('sub.rn',m,const(1)),self.op('add.rn',m,const(1)))
        zz=self.op('mul.rn',z,z);p=self.reg();self.move(p,const(1/31))
        for k in range(29,0,-2):
            self.emit(f'mul.rn.f64 {p}, {p}, {zz};')
            self.emit(f'add.rn.f64 {p}, {p}, {const(1/k)};')
        core=self.op('mul.rn',self.op('mul.rn',p,z),const(2))
        ef=self.reg();self.emit(f'cvt.rn.f64.s32 {ef}, {exponent};')
        if base=='2':
            fraction=self.op('mul.rn',core,const(1/math.log(2)))
            answer=self.op('add.rn',fraction,ef)
            if residual is not None:
                # FastTwoSum: |integer exponent| >= |fraction|, except e=0
                # where addition is exact. Preserve the small mantissa term.
                self.move(residual,self.op('add.rn',self.op('sub.rn',ef,answer),fraction))
        else:
            answer=self.op('add.rn',core,self.op('mul.rn',ef,const(LN2_LO)))
            answer=self.op('add.rn',answer,self.op('mul.rn',ef,const(LN2_HI)))
            if base=='10':answer=self.op('mul.rn',answer,const(1/math.log(10)))
        self.move(dst,answer);self.go(end)
        self.mark(zero);self.move(dst,const(-math.inf));self.go(end)
        self.mark(infinity);self.move(dst,const(math.inf));self.go(end)
        self.mark(invalid);self.move(dst,const(math.nan));self.mark(end)

    def polynomial(self, x, coefficients):
        p=self.reg();self.move(p,const(coefficients[-1]))
        for c in reversed(coefficients[:-1]):
            self.emit(f'mul.rn.f64 {p}, {p}, {x};')
            self.emit(f'add.rn.f64 {p}, {p}, {const(c)};')
        return p

    def activation(self, dst, op, x):
        magnitude=self.op('abs',x);negative=self.compare('lt',x,const(0))
        z=self.reg()
        self.exp(z,self.op('mul.rn',magnitude,const(-.5 if op=='silu' else -2 if op=='tanh' else -1)))
        exponential=self.op('mul.rn',z,z) if op=='silu' else z
        denominator=self.op('add.rn',const(1),exponential)
        if op=='softplus':
            # log1p(z)=2*atanh(z/(2+z)), z<=1. Preserve tiny z directly.
            u=self.op('div.rn',z,self.op('add.rn',const(2),z))
            p=self.polynomial(self.op('mul.rn',u,u),[1/k for k in range(1,40,2)])
            small=self.op('mul.rn',self.op('mul.rn',p,u),const(2))
            small=self.select(self.compare('le',z,const(2**-53)),z,small)
            base=self.select(self.compare('gt',x,const(0)),x,const(0))
            result=self.op('add.rn',base,small)
        elif op=='tanh':
            general=self.op('div.rn',self.op('sub.rn',const(1),z),denominator)
            general=self.op('copysign',x,general)
            coefficients=[1.]
            for n in range(1,13):
                coefficients.append(-sum(coefficients[i]*coefficients[n-1-i] for i in range(n))/(2*n+1))
            small=self.op('mul.rn',x,self.polynomial(self.op('mul.rn',x,x),coefficients))
            result=self.select(self.compare('lt',magnitude,const(.125)),small,general)
        else:
            if op=='silu':
                tail=self.op('mul.rn',self.op('mul.rn',x,z),z)
                numerator=self.select(negative,tail,x)
            else:numerator=self.select(negative,z,const(1))
            result=self.op('div.rn',numerator,denominator)
        self.move(dst,result)

    def gaussian(self, dst, op, x):
        # NIST DLMF 7.6.1 power series for |z|<1.5; 7.9.1's equivalent
        # continued fraction erfc(z)=exp(-z*z)/(sqrt(pi)*(z+(.5/(z+1/(...))))).
        # 96 levels are evaluated backwards for positive z>=1.5. This avoids
        # subtracting erf from one to obtain the small negative GELU tail.
        z=x if op=='erf' else self.op('mul.rn',x,const(1/math.sqrt(2)))
        a=self.op('abs',z);square=self.op('mul.rn',z,z)
        p=self.polynomial(square,[2/math.sqrt(math.pi)*(-1)**n/(math.factorial(n)*(2*n+1)) for n in range(28)])
        small=self.op('mul.rn',z,p)
        if op=='gelu':small=self.op('mul.rn',x,self.op('add.rn',const(.5),self.op('mul.rn',const(.5),small)))
        cf=self.reg();self.move(cf,const(0))
        for k in range(96,0,-1):
            denominator=self.op('add.rn',a,cf)
            self.emit(f'div.rn.f64 {cf}, {const(k/2)}, {denominator};')
        weight=self.op('div.rn',const(1/math.sqrt(math.pi)),self.op('add.rn',a,cf))
        half_exp=self.reg();self.exp(half_exp,self.op('mul.rn',square,const(-.5)))
        complement=self.op('mul.rn',self.op('mul.rn',weight,half_exp),half_exp)
        if op=='erf':general=self.op('copysign',x,self.op('sub.rn',const(1),complement))
        else:
            prefactor=self.op('mul.rn',self.op('mul.rn',x,const(.5)),weight)
            tail=self.op('mul.rn',self.op('mul.rn',prefactor,half_exp),half_exp)
            positive=self.op('mul.rn',x,self.op('sub.rn',const(1),self.op('mul.rn',const(.5),complement)))
            general=self.select(self.compare('lt',x,const(0)),tail,positive)
        self.move(dst,self.select(self.compare('lt',a,const(1.5)),small,general))

    def gelu_tanh(self,dst,x):
        square=self.op('mul.rn',x,x)
        cubic=self.op('mul.rn',self.op('mul.rn',square,x),const(.044715))
        argument=self.op('mul.rn',self.op('add.rn',x,cubic),const(2*math.sqrt(2/math.pi)))
        # x*sigmoid(argument), with a split exponential to retain the tail.
        magnitude=self.op('abs',argument);half=self.reg()
        self.exp(half,self.op('mul.rn',magnitude,const(-.5)))
        denominator=self.op('add.rn',const(1),self.op('mul.rn',half,half))
        tail=self.op('mul.rn',self.op('mul.rn',x,half),half)
        numerator=self.select(self.compare('lt',argument,const(0)),tail,x)
        self.move(dst,self.op('div.rn',numerator,denominator))

    def power(self,dst,x,y):
        one,copy,invalid,yinf,xzero,xinf,zero,infinity,signed,end=[self.label() for _ in range(10)]
        self.jump(self.compare('eq',y,const(0)),one)
        self.jump(self.compare('eq',x,const(1)),one)
        self.jump(self.compare('eq',y,const(1)),copy)
        self.jump(self.compare('nan',x,y),invalid)
        ax=self.op('abs',x);ay=self.op('abs',y)
        self.jump(self.compare('eq',ay,const(math.inf)),yinf)
        # Testing integrality in f64 avoids overflowing a float-to-int cast.
        integral=self.compare('eq',self.op('cvt.rzi.f64',y),y)
        half_floor=self.op('cvt.rmi.f64',self.op('mul.rn',y,const(.5)))
        odd=self.compare('eq',self.op('sub.rn',y,self.op('mul.rn',half_floor,const(2))),const(1))
        self.emit(f'and.pred {odd}, {odd}, {integral};')
        lo,hi=[self.e.reg('s32') for _ in range(2)]
        self.emit(f'mov.b64 {{{lo}, {hi}}}, {x};')
        negative=self.e.reg('pred');self.emit(f'setp.lt.s32 {negative}, {hi}, 0;')
        flip=self.e.reg('pred');self.emit(f'and.pred {flip}, {negative}, {odd};')
        self.jump(self.compare('eq',ax,const(0)),xzero)
        self.jump(self.compare('eq',ax,const(math.inf)),xinf)
        not_integer=self.e.reg('pred');bad=self.e.reg('pred')
        self.emit(f'not.pred {not_integer}, {integral};');self.emit(f'and.pred {bad}, {negative}, {not_integer};')
        self.jump(bad,invalid)
        logarithm,loglow,answer=self.reg(),self.reg(),self.reg();self.log(logarithm,ax,'2',residual=loglow)
        product=self.op('mul.rn',y,logarithm)
        self.jump(self.compare('gt',product,const(2048)),infinity)
        self.jump(self.compare('lt',product,const(-2048)),zero)
        error=self.reg();negative_product=self.op('neg',product)
        self.emit(f'fma.rn.f64 {error}, {y}, {logarithm}, {negative_product};')
        correction=self.op('add.rn',error,self.op('mul.rn',y,loglow))
        self.exp(answer,product,True,tail=correction);self.go(signed)
        self.mark(yinf)
        self.jump(self.compare('eq',ax,const(1)),one)
        large=self.compare('gt',ax,const(1));yn=self.compare('lt',y,const(0));grow=self.e.reg('pred')
        self.emit(f'xor.pred {grow}, {large}, {yn};')
        self.emit(f'setp.ne.u32 {flip}, 0, 0;')
        self.jump(grow,infinity);self.go(zero)
        self.mark(xzero);self.jump(self.compare('lt',y,const(0)),infinity);self.go(zero)
        self.mark(xinf);self.jump(self.compare('lt',y,const(0)),zero);self.go(infinity)
        self.mark(zero);self.move(answer,const(0));self.go(signed)
        self.mark(infinity);self.move(answer,const(math.inf))
        self.mark(signed);self.emit(f'@{flip} neg.f64 {answer}, {answer};');self.move(dst,answer);self.go(end)
        self.mark(one);self.move(dst,const(1));self.go(end)
        self.mark(copy);self.move(dst,x);self.go(end)
        self.mark(invalid);self.move(dst,const(math.nan));self.mark(end)


def emit_math(e, destination, op, args):
    from .errors import require
    supported={'sign','rsqrt','exp','exp_fast','exp2','log','log2','log10','sigmoid','silu','tanh','softplus','erf','gelu','gelu_tanh','pow','sin','cos'}
    if op not in supported:return False
    require(len(args)==(2 if op=='pow' else 1),f'bad arity for {op}')
    m=Math64(e);x=args[0]
    if op in ('sin','cos'):
        from .wide_trig import emit_trig64
        emit_trig64(e,destination,x,op=='cos')
    elif op=='pow':m.power(destination,*args)
    elif op in ('exp','exp_fast','exp2'):m.exp(destination,x,op=='exp2')
    elif op in ('log','log2','log10'):m.log(destination,x,{'log':'e','log2':'2','log10':'10'}[op])
    elif op in ('sigmoid','silu','tanh','softplus'):m.activation(destination,op,x)
    elif op in ('erf','gelu'):m.gaussian(destination,op,x)
    elif op=='gelu_tanh':m.gelu_tanh(destination,x)
    elif op=='rsqrt':m.move(destination,m.op('rcp.rn',m.op('sqrt.rn',x)))
    else:
        positive=m.compare('gt',x,const(0));negative=m.compare('lt',x,const(0));nan=m.compare('nan',x,x)
        result=m.select(positive,const(1),const(0));result=m.select(negative,const(-1),result)
        m.move(destination,m.select(nan,const(math.nan),result))
    return True
