#!/usr/bin/env python3
"""
npu_grouped_matmul — updated to match actual runtime parameters from debug output.

Actual params (A3 debug):
  x:         int8   (M, K)
  weight:    int32  (E, K, N_packed)    N_packed = N_original_int8 // 4
  scale:     int64  (E, 1, N_logical)   N_logical = N_packed * 8  ← 3D, NOT 2D!
  bias:      f32    (E, N_logical)
  pertoken:  f32    (M,)
  group_list: int64, per-expert token counts, len=E
  glt=1, gt=0, si=2, out=bf16

Key diff vs earlier tests:
  - scale is 3D (E, 1, N_logical), not 2D (E, N_logical)
  - glt=1 with per-expert counts (len=E), not glt=0

Usage: python3 test_npu_grouped_matmul.py [-v]
"""

import argparse
import sys
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401


def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def blue(s):  return f"\033[94m{s}\033[0m"


# Test shapes
M, K = 48, 256
E = 2
# For int8 weight: N=256 directly
# For int32 weight: N_original=128, N_packed=32, N_logical=32*8=256
N_INT8 = 256        # for non-packed int8 weight
N_ORIG = 128        # original int8 N before pack
N_LOGICAL = 256     # = N_ORIG//4 * 8 = 32 * 8

# Per-expert token counts (len=E, glt=1 — matching actual code)
GL = torch.tensor([M//3, M - M//3], device='npu', dtype=torch.int64)  # [16, 32]


def run_one(name, fn, verbose):
    print(f"\n{blue('─'*55)}")
    print(f"  [{name}]")
    try:
        ok, msg = fn(verbose)
        print(f"  {green('PASS')}  {msg}")
        return True
    except Exception as e:
        print(f"  {red('FAIL')}")
        if verbose:
            traceback.print_exc()
        else:
            s = str(e).split('\n')[0][:350]
            print(f"  {type(e).__name__}: {s}")
        return False


def f32_to_i64_scale(t: torch.Tensor) -> torch.Tensor:
    arr = t.cpu().numpy().view(np.uint32)
    return torch.from_numpy(arr.astype(np.int64)).npu()


# =========================================================================
# Baselines
# =========================================================================
def case_baseline_fp16(verbose):
    x = [torch.randn(M, K, device='npu', dtype=torch.float16)]
    w = [torch.randn(E, K, N_INT8, device='npu', dtype=torch.float16)]
    out = torch.ops.npu.npu_grouped_matmul(
        x, w, group_list=GL, split_item=2,
        group_type=0, group_list_type=1,
        output_dtype=torch.float16,
    )
    return True, f"fp16 baseline glt=1, out={out[0].shape}"


def case_baseline_bf16(verbose):
    x = [torch.randn(M, K, device='npu', dtype=torch.bfloat16)]
    w = [torch.randn(E, K, N_INT8, device='npu', dtype=torch.bfloat16)]
    out = torch.ops.npu.npu_grouped_matmul(
        x, w, group_list=GL, split_item=2,
        group_type=0, group_list_type=1,
        output_dtype=torch.bfloat16,
    )
    return True, f"bf16 baseline glt=1, out={out[0].shape}"


# =========================================================================
# int8 weight (not packed) + 3D scale + glt=1
# =========================================================================
def case_int8w_scale3d_bf16_bias_i32_pts(verbose):
    """int8 w, scale=bf16 3D(E,1,N), bias=int32, +pts, glt=1."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w = [torch.randint(-128, 127, (E, K, N_INT8), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.randn(E, 1, N_INT8, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "bias": [torch.zeros(E, N_INT8, device='npu', dtype=torch.int32)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "group_list": GL, "split_item": 2,
        "group_type": 0, "group_list_type": 1,
        "output_dtype": torch.bfloat16,
    }
    if verbose:
        print(f"    x={x[0].dtype}{list(x[0].shape)} w={w[0].dtype}{list(w[0].shape)}")
        print(f"    scale={kw['scale'][0].dtype}{list(kw['scale'][0].shape)} bias={kw['bias'][0].dtype}{list(kw['bias'][0].shape)}")
    out = torch.ops.npu.npu_grouped_matmul(x=x, weight=w, **kw)
    return True, f"int8×int8, scale=bf16 3D, bias=i32, +pts, glt=1"


def case_int8w_scale3d_bf16_pts_nobias(verbose):
    """int8 w, scale=bf16 3D, +pts, no bias, glt=1."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w = [torch.randint(-128, 127, (E, K, N_INT8), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.randn(E, 1, N_INT8, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "group_list": GL, "split_item": 2,
        "group_type": 0, "group_list_type": 1,
        "output_dtype": torch.bfloat16,
    }
    out = torch.ops.npu.npu_grouped_matmul(x=x, weight=w, **kw)
    return True, f"int8×int8, scale=bf16 3D, +pts no bias, glt=1"


# =========================================================================
# int32 weight (int4-packed) — EXACT debug-output match
#   weight: int32 (E, K, N_ORIG//4)
#   scale:  int64 3D (E, 1, N_LOGICAL)
#   bias:   f32 (E, N_LOGICAL)
#   glt=1
# =========================================================================
def case_exact_debug_repro(verbose):
    """EXACT match of A3 debug output:
    int8 x, int32 w, scale=i64 3D(E,1,N_logical), bias=f32, +pts, glt=1."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, N_ORIG), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]  # (E, K, N_ORIG//4)

    scale_f32 = torch.randn(E, 1, N_LOGICAL, device='npu', dtype=torch.float32).abs() + 0.01
    scale_i64 = [f32_to_i64_scale(scale_f32)]  # 3D (E, 1, N_LOGICAL)

    bias = [torch.randn(E, N_LOGICAL, device='npu', dtype=torch.float32)]
    pts = [torch.ones(M, device='npu', dtype=torch.float32)]

    if verbose:
        print(f"    x={x[0].dtype}{list(x[0].shape)}")
        print(f"    w={w[0].dtype}{list(w[0].shape)}")
        print(f"    scale={scale_i64[0].dtype}{list(scale_i64[0].shape)}")
        print(f"    bias={bias[0].dtype}{list(bias[0].shape)}")
        print(f"    pts={pts[0].dtype}{list(pts[0].shape)}")
        print(f"    gl={GL.tolist()} glt=1 gt=0 si=2 out=bf16")

    out = torch.ops.npu.npu_grouped_matmul(
        x=x, weight=w, scale=scale_i64, bias=bias,
        per_token_scale=pts, group_list=GL,
        split_item=2, group_type=0, group_list_type=1,
        output_dtype=torch.bfloat16,
    )
    return True, f"EXACT DEBUG REPRO, out={out[0].shape} {out[0].dtype}"


def case_int32w_scale3d_bf16_bias_bf16_pts(verbose):
    """int32 w, scale=bf16 3D, bias=bf16, +pts, glt=1."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, N_ORIG), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]
    kw = {
        "scale": [torch.randn(E, 1, N_LOGICAL, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "bias": [torch.zeros(E, N_LOGICAL, device='npu', dtype=torch.bfloat16)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "group_list": GL, "split_item": 2,
        "group_type": 0, "group_list_type": 1,
        "output_dtype": torch.bfloat16,
    }
    out = torch.ops.npu.npu_grouped_matmul(x=x, weight=w, **kw)
    return True, f"int32w, scale=bf16 3D, bias=bf16, +pts, glt=1"


def case_int32w_scale3d_bf16_nobias_pts(verbose):
    """int32 w, scale=bf16 3D, no bias, +pts, glt=1."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, N_ORIG), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]
    kw = {
        "scale": [torch.randn(E, 1, N_LOGICAL, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "group_list": GL, "split_item": 2,
        "group_type": 0, "group_list_type": 1,
        "output_dtype": torch.bfloat16,
    }
    out = torch.ops.npu.npu_grouped_matmul(x=x, weight=w, **kw)
    return True, f"int32w, scale=bf16 3D, no bias, +pts, glt=1"


def case_int32w_antiquant_bf16_pts(verbose):
    """int32 w, antiquant_scale+offset 3D(bf16), +pts, glt=1."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, N_ORIG), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]
    kw = {
        "antiquant_scale": [torch.randn(E, 1, N_LOGICAL, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "antiquant_offset": [torch.zeros(E, 1, N_LOGICAL, device='npu', dtype=torch.bfloat16)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "group_list": GL, "split_item": 2,
        "group_type": 0, "group_list_type": 1,
        "output_dtype": torch.bfloat16,
    }
    out = torch.ops.npu.npu_grouped_matmul(x=x, weight=w, **kw)
    return True, f"int32w, antiquant 3D, +pts, glt=1"


# =========================================================================
# Compare: 2D scale (old pattern) vs 3D scale (actual) — same test as before for reference
# =========================================================================
def case_int32w_scale2d_i64_bias_f32_pts_glt0(verbose):
    """OLD pattern: 2D scale, glt=0 — previously passed on A3."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, N_ORIG), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]
    scale_f32 = torch.randn(E, N_LOGICAL, device='npu', dtype=torch.float32).abs() + 0.01
    kw = {
        "scale": [f32_to_i64_scale(scale_f32)],  # 2D (E, N_LOGICAL)
        "bias": [torch.randn(E, N_LOGICAL, device='npu', dtype=torch.float32)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "group_list": GL, "split_item": 2,
        "group_type": 0, "group_list_type": 0,  # glt=0
        "output_dtype": torch.bfloat16,
    }
    out = torch.ops.npu.npu_grouped_matmul(x=x, weight=w, **kw)
    return True, f"REF: 2D scale i64, glt=0 (old PASS)"


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  NPU Grouped Matmul — matching A3 debug output")
    print(f"  torch: {torch.__version__}, torch_npu: {torch_npu.__version__}")
    print(f"  device: {torch.npu.get_device_name(0)}")
    print(f"  M={M} K={K} E={E} (N_int8={N_INT8}, N_orig={N_ORIG}, N_logical={N_LOGICAL})")
    print(f"  gl per-expert={GL.tolist()} glt=1")
    print("=" * 55)

    tests = [
        ("B1 fp16 baseline glt=1",              case_baseline_fp16),
        ("B2 bf16 baseline glt=1",              case_baseline_bf16),
        ("I1 int8w scale=bf16 3D bias=i32 +pts", case_int8w_scale3d_bf16_bias_i32_pts),
        ("I2 int8w scale=bf16 3D +pts nobias",   case_int8w_scale3d_bf16_pts_nobias),
        ("P1 EXACT DEBUG REPRO (scale=i64 3D, glt=1)", case_exact_debug_repro),
        ("P2 int32w scale=bf16 3D bias=bf16 +pts", case_int32w_scale3d_bf16_bias_bf16_pts),
        ("P3 int32w scale=bf16 3D nobias +pts",   case_int32w_scale3d_bf16_nobias_pts),
        ("P4 int32w antiquant 3D +pts",          case_int32w_antiquant_bf16_pts),
        ("REF 2D scale i64 glt=0 (old PASS)",    case_int32w_scale2d_i64_bias_f32_pts_glt0),
    ]

    results = {}
    for name, fn in tests:
        results[name] = run_one(name, fn, args.verbose)

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  [{green('PASS') if ok else red('FAIL')}] {name}")
    print(f"\n  {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
