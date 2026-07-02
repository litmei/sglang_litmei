#!/usr/bin/env python3
"""
npu_grouped_matmul — A5 adaptation test.

Rules established:
  - int8 w: scale must be 2D (E,N), not 3D
  - int32 w (int4-packed): scale must be 3D (E,1,N_logical)
  - A5: npu_grouped_matmul does NOT support int8×int4 (int32-packed w)
  - Both A3/A5: int8×int8 + bf16 2D scale + int32 bias + pts WORKS

Target fix for A5 (方案 B): use int8 weight + bf16 2D scale instead of int32-packed.
"""

import argparse, sys, traceback
import numpy as np
import torch
import torch_npu  # noqa: F401

def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def blue(s):  return f"\033[94m{s}\033[0m"

M, K = 48, 256; E = 2
N_INT8 = 256       # for non-packed int8 weight
N_ORIG = 128       # for int32-packed weight
N_LOGICAL = 256    # = N_ORIG//4 * 8

GL = torch.tensor([M//3, M - M//3], device='npu', dtype=torch.int64)

def run_one(name, fn, v):
    print(f"\n{blue('─'*55)}")
    print(f"  [{name}]")
    try:
        ok, msg = fn(v)
        print(f"  {green('PASS')}  {msg}")
        return True
    except Exception as e:
        print(f"  {red('FAIL')}")
        if v: traceback.print_exc()
        else: print(f"  {type(e).__name__}: {str(e).split(chr(10))[0][:350]}")
        return False

def f32_to_i64(t): 
    return torch.from_numpy(t.cpu().numpy().view(np.uint32).astype(np.int64)).npu()

# ============ Baselines ============
def case_fp16(verbose):
    x=[torch.randn(M,K,device='npu',dtype=torch.float16)]
    w=[torch.randn(E,K,N_INT8,device='npu',dtype=torch.float16)]
    out=torch.ops.npu.npu_grouped_matmul(x,w,group_list=GL,split_item=2,group_type=0,group_list_type=0,output_dtype=torch.float16)
    return True,f"fp16 glt=0"

def case_bf16(verbose):
    x=[torch.randn(M,K,device='npu',dtype=torch.bfloat16)]
    w=[torch.randn(E,K,N_INT8,device='npu',dtype=torch.bfloat16)]
    out=torch.ops.npu.npu_grouped_matmul(x,w,group_list=GL,split_item=2,group_type=0,group_list_type=0,output_dtype=torch.bfloat16)
    return True,f"bf16 glt=0"

# ============ 方案 B: int8 weight + bf16 2D scale (A5 target) ============
def case_int8w_bf16scale_i32bias_pts_glt0(verbose):
    """int8×int8, bf16 2D scale, int32 bias, +pts, glt=0 — A3/A5 both PASS."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            bias=[torch.zeros(E,N_INT8,device='npu',dtype=torch.int32)],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=0,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, i32 bias, +pts, glt=0"

def case_int8w_bf16scale_i32bias_pts_glt1(verbose):
    """Same but glt=1 — test if glt=1 works with int8 weight."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            bias=[torch.zeros(E,N_INT8,device='npu',dtype=torch.int32)],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=1,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, i32 bias, +pts, glt=1"

def case_int8w_bf16scale_nobias_pts_glt0(verbose):
    """int8×int8, bf16 2D scale, no bias, +pts, glt=0."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=0,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, no bias, +pts, glt=0"

def case_int8w_bf16scale_nobias_pts_glt1(verbose):
    """int8×int8, bf16 2D scale, no bias, +pts, glt=1."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=1,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, no bias, +pts, glt=1"

# ============ 方案 B extended: int8 weight + bf16 2D scale + f32 bias (matching actual code) ============
def case_int8w_bf16scale_f32bias_pts_glt0(verbose):
    """int8×int8, bf16 2D scale, f32 bias, +pts, glt=0."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            bias=[torch.randn(E,N_INT8,device='npu',dtype=torch.float32)],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=0,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, f32 bias, +pts, glt=0"

def case_int8w_bf16scale_f32bias_pts_glt1(verbose):
    """int8×int8, bf16 2D scale, f32 bias, +pts, glt=1 — KEY for A5 adaptation."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            bias=[torch.randn(E,N_INT8,device='npu',dtype=torch.float32)],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=1,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, f32 bias, +pts, glt=1"

def case_int8w_bf16scale_bf16bias_pts_glt1(verbose):
    """int8×int8, bf16 2D scale, bf16 bias, +pts, glt=1."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w=[torch.randint(-128,127,(E,K,N_INT8),device='npu',dtype=torch.int8)]
    kw=dict(scale=[torch.randn(E,N_INT8,device='npu',dtype=torch.bfloat16).abs()+0.01],
            bias=[torch.zeros(E,N_INT8,device='npu',dtype=torch.bfloat16)],
            per_token_scale=[torch.ones(M,device='npu',dtype=torch.float32)],
            group_list=GL,split_item=2,group_type=0,group_list_type=1,output_dtype=torch.bfloat16)
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,**kw)
    return True,f"int8×int8, bf16 2D scale, bf16 bias, +pts, glt=1"

# ============ A3 exact repro (int32 w + i64 3D scale) — works on A3, fails on A5 ============
def case_int32w_i64_3dscale_f32bias_pts_glt1(verbose):
    """A3 debug repro: int32 w, i64 3D scale, f32 bias, +pts, glt=1."""
    x=[torch.randint(-128,127,(M,K),device='npu',dtype=torch.int8)]
    w_int8=torch.randint(-128,127,(E,K,N_ORIG),device='npu',dtype=torch.int8)
    w=[w_int8.view(torch.int32).contiguous()]
    scale_i64=[f32_to_i64(torch.randn(E,1,N_LOGICAL,device='npu',dtype=torch.float32).abs()+0.01)]
    bias=[torch.randn(E,N_LOGICAL,device='npu',dtype=torch.float32)]
    pts=[torch.ones(M,device='npu',dtype=torch.float32)]
    if verbose:
        print(f"    x={x[0].dtype}{list(x[0].shape)} w={w[0].dtype}{list(w[0].shape)}")
        print(f"    scale={scale_i64[0].dtype}{list(scale_i64[0].shape)} bias={bias[0].dtype}{list(bias[0].shape)}")
    out=torch.ops.npu.npu_grouped_matmul(x=x,weight=w,scale=scale_i64,bias=bias,
        per_token_scale=pts,group_list=GL,split_item=2,group_type=0,group_list_type=1,output_dtype=torch.bfloat16)
    return True,f"A3 EXACT REPRO (int32w, i64 3D, glt=1), out={out[0].shape}"

# ============ Main ============
def main():
    p=argparse.ArgumentParser(); p.add_argument("-v","--verbose",action="store_true"); a=p.parse_args()
    print("="*55)
    print("  A5 Adaptation Test")
    print(f"  torch_npu: {torch_npu.__version__}, device: {torch.npu.get_device_name(0)}")
    print(f"  M={M} K={K} E={E}")
    print("="*55)
    tests=[
        ("B1 fp16 baseline glt=0", case_fp16),
        ("B2 bf16 baseline glt=0", case_bf16),
        ("I1 int8w bf16 2D +i32bias +pts glt=0", case_int8w_bf16scale_i32bias_pts_glt0),
        ("I2 int8w bf16 2D +i32bias +pts glt=1", case_int8w_bf16scale_i32bias_pts_glt1),
        ("I3 int8w bf16 2D nobias +pts glt=0", case_int8w_bf16scale_nobias_pts_glt0),
        ("I4 int8w bf16 2D nobias +pts glt=1", case_int8w_bf16scale_nobias_pts_glt1),
        # f32 bias — matching actual sglang code
        ("I5 int8w bf16 2D +f32bias +pts glt=0", case_int8w_bf16scale_f32bias_pts_glt0),
        ("I6 int8w bf16 2D +f32bias +pts glt=1", case_int8w_bf16scale_f32bias_pts_glt1),
        # bf16 bias — alternative
        ("I7 int8w bf16 2D +bf16bias +pts glt=1", case_int8w_bf16scale_bf16bias_pts_glt1),
        ("P1 A3 REPRO int32w i64 3D glt=1", case_int32w_i64_3dscale_f32bias_pts_glt1),
    ]
    results={}
    for name,fn in tests: results[name]=run_one(name,fn,a.verbose)
    print(f"\n{'='*55}\n  SUMMARY\n{'='*55}")
    passed=sum(1 for v in results.values() if v)
    for n,o in results.items(): print(f"  [{green('PASS') if o else red('FAIL')}] {n}")
    print(f"\n  {passed}/{len(results)} passed")
    print("\n  Key for A5 adaptation: I6 (f32 bias + glt=1) matches actual sglang code params")
    print("  P1 should pass on A3, fail on A5 (int32 w not supported)")
    return 0 if passed==len(results) else 1

if __name__=="__main__": sys.exit(main())
