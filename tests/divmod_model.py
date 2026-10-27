"""Instruction-level vector executor with independent per-lane program counters.

This interpreter implements PTX arithmetic primitives via NumPy. It contains
no floor-divmod formula; tests compare the emitted program to native ufuncs.
"""
import re, struct
import numpy as np


def execute(code, inputs, destination, budget=200000):
    regs={name:np.asarray(value).copy() for name,value in inputs.items()}
    n=len(next(iter(regs.values())))
    def dtype(name):
        return np.bool_ if name.startswith('%p') else np.float64 if name.startswith('%fd') else np.float16 if name.startswith('%h') else np.float32
    def read(token):
        if token.startswith('%'):
            assert token in regs, f'uninitialized register {token}'
            return regs[token]
        if token.startswith('0d'):return np.full(n,struct.unpack('>d',bytes.fromhex(token[2:]))[0])
        if token.startswith('0f'):return np.full(n,struct.unpack('>f',bytes.fromhex(token[2:]))[0],np.float32)
        return np.full(n,int(token))
    labels={line.strip()[:-1]:i for i,line in enumerate(code) if line.strip().endswith(':')}
    pc=np.zeros(n,dtype=np.int64);steps=0
    while np.any(pc<len(code)):
        steps+=1
        assert steps<=budget,'instruction budget exceeded'
        i=int(pc[pc<len(code)].min());mask=pc==i;pc[mask]=i+1;line=code[i].strip()
        if line.endswith(':'):continue
        if line.startswith('@'):
            pred,line=line.split(' ',1);neg=pred.startswith('@!');pv=read(pred[2:] if neg else pred[1:]);mask &= ~pv if neg else pv
            if not mask.any():continue
        op,tail=line.rstrip(';').split(' ',1)
        if op=='bra':pc[mask]=labels[tail];continue
        dest,*src=re.findall(r'[^,\s]+',tail)
        with np.errstate(all='ignore'):
            if op.startswith('mov.'):value=read(src[0])
            elif op.startswith('cvt.'):
                value=np.floor(read(src[0])) if '.rmi.' in op else read(src[0])
            elif op.startswith('setp.'):
                a,b=map(read,src);cmp=op.split('.')[1]
                value={'eq':np.equal,'lt':np.less,'gt':np.greater,'ge':np.greater_equal,'nan':lambda a,b:np.isnan(a)|np.isnan(b)}[cmp](a,b)
            elif op=='xor.pred':value=np.logical_xor(*map(read,src))
            elif op.startswith('abs.'):value=np.abs(read(src[0]))
            elif op.startswith('copysign.'):value=np.copysign(read(src[1]),read(src[0]))
            else:
                a,b=map(read,src);base=op.split('.')[0]
                value={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}[base](a,b)
            if dest not in regs:regs[dest]=np.zeros(n,dtype=dtype(dest))
            regs[dest][mask]=value[mask]
    return regs[destination]
