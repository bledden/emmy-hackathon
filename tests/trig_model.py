"""Vector interpreter for the emitted scalar trig instruction subset, not GPU proof."""
import re
import struct
import numpy as np


def execute(code, inputs, destination, extra_inputs=None):
    n=len(inputs);regs={'%f0':np.asarray(inputs,dtype=np.float32).copy()}
    regs.update({k:np.asarray(v,dtype=np.float32).copy() for k,v in (extra_inputs or {}).items()})
    def kind(name):
        return np.float64 if name.startswith('%fd') else np.float32 if name.startswith('%f') else np.bool_ if name.startswith('%p') else np.uint64
    def read(token):
        if token.startswith('%'):
            assert token in regs, f'uninitialized {token}'
            return regs[token]
        if token.startswith('0d'):return np.full(n,struct.unpack('>d',bytes.fromhex(token[2:]))[0])
        if token.startswith('0f'):return np.full(n,struct.unpack('>f',bytes.fromhex(token[2:]))[0],dtype=np.float32)
        return np.full(n,int(token,0),dtype=np.uint64)
    def write(name,v,mask):
        if name not in regs:regs[name]=np.zeros(n,dtype=kind(name))
        regs[name][mask]=v[mask] if isinstance(v,np.ndarray) else v
    def bits32(token):
        a=read(token)
        return a.view(np.uint32).astype(np.uint64) if a.dtype==np.float32 else a & np.uint64(0xffffffff)
    def signed32(token):
        return bits32(token).astype(np.uint32).view(np.int32)
    labels={line.strip()[:-1]:i for i,line in enumerate(code) if line.strip().endswith(':')}
    pc=np.zeros(n,dtype=np.int64)
    for i,line in enumerate(code):
        mask=pc==i;pc[mask]=i+1
        line=line.strip()
        if not mask.any() or line.endswith(':'):continue
        if line.startswith('@'):
            pred,line=line.split(' ',1);neg=pred.startswith('@!');pv=read(pred[2:] if neg else pred[1:]);mask &= ~pv if neg else pv
            if not mask.any():continue
        op,tail=line.rstrip(';').split(' ',1)
        if op=='bra':pc[mask]=labels[tail];continue
        args=re.findall(r'\{[^}]+\}|[^,\s]+',tail)
        dest,*src=args
        if op=='mov.b64':
            if dest.startswith('{'):
                lo,hi=[t.strip() for t in dest[1:-1].split(',')];v=read(src[0]);write(lo,v&np.uint64(0xffffffff),mask);write(hi,v>>np.uint64(32),mask)
            else:
                lo,hi=[t.strip() for t in src[0][1:-1].split(',')]
                v=bits32(lo)|(bits32(hi)<<np.uint64(32))
                write(dest,v.view(np.float64) if kind(dest)==np.float64 else v,mask)
            continue
        if op=='mov.b32':
            v=bits32(src[0]);v=v.astype(np.uint32).view(np.float32) if kind(dest)==np.float32 else v
        elif op.startswith('mov.'):v=read(src[0])
        elif op=='cvt.rn.f64.s64':v=read(src[0]).view(np.int64).astype(np.float64)
        elif op=='cvt.rn.f64.s32':v=signed32(src[0]).astype(np.float64)
        elif op=='cvt.rn.f32.s32':v=signed32(src[0]).astype(np.float32)
        elif op in ('cvt.rzi.s32.f32','cvt.rmi.s32.f64'):
            v=(np.trunc if '.rzi.' in op else np.floor)(read(src[0])).astype(np.int64).astype(np.uint64)&np.uint64(0xffffffff)
        elif op=='cvt.f64.f32':v=read(src[0]).astype(np.float64)
        elif op=='cvt.rn.f32.f64':v=read(src[0]).astype(np.float32)
        elif op=='cvt.u64.u32':v=bits32(src[0])
        elif op=='cvt.u32.u64':v=read(src[0])&np.uint64(0xffffffff)
        elif op.startswith('setp.'):
            a,b=map(signed32 if op.endswith('.s32') else read,src);relation=op.split('.')[1];v={'ge':np.greater_equal,'gt':np.greater,'le':np.less_equal,'lt':np.less,'eq':np.equal,'ne':np.not_equal,'neu':np.not_equal}[relation](a,b)
        elif op.startswith('selp.'):v=np.where(read(src[2]),read(src[0]),read(src[1]))
        elif op=='not.pred':v=~read(src[0])
        else:
            a,b=map(read,src);base=op.split('.')[0]
            with np.errstate(over='ignore',invalid='ignore'):
                if base=='add':v=a+b
                elif base=='sub':v=a-b
                elif base=='mul':v=a*b
                elif base=='div':v=a/b
                elif base=='and':v=a&b
                elif base=='or':v=a|b
                elif base=='xor':v=a^b
                elif base=='shr':v=a>>b
                elif base=='shl':v=a<<b
                else:raise AssertionError(op)
            if op.endswith(('u32','s32','b32')) and op!='mul.wide.u32':v &= np.uint64(0xffffffff)
        write(dest,v,mask)
    assert np.all(pc==len(code)), 'unexecuted control path'
    return regs[destination]
