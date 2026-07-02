#!/usr/bin/env python3
"""
npu_grouped_matmul probe — w4a8 INT8 MoE on NPU V5.

Key constraints discovered:
  - group_list len == weight.dim(0) == E  (glt=0, gt=0)
  - K must be >= 128 and divisible by 128 (quantGroupNum tiling)
  - bias N must match weight's LOGICAL N (int32 weight interpreted as int4: N_logical = N_int32 * 8)
  - Per-token quant + bf16 output → scale must be bf16, bias=int32

Probe strategy: try different dtype combos with K=256, N=256.
"""

import sys
import argparse
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401


def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def blue(s):  return f"\033[94m{s}\033[0m"


M, K, N = 48, 256, 256   # K must be divisible by 128
E = 2
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


def gmm(x, weight, **kw):
    return torch.ops.npu.npu_grouped_matmul(
        x=x, weight=weight,
        group_list=GL, split_item=2,
        group_type=0, group_list_type=0,
        **kw,
    )


def info(x, w, kw):
    """Print key dtypes/shapes."""
    parts = []
    for k, v in kw.items():
        if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
            parts.append(f"{k}={v[0].dtype}{list(v[0].shape)}")
        elif isinstance(v, torch.dtype):
            parts.append(f"{k}={v}")
    print(f"    x={x[0].dtype}{list(x[0].shape)} w={w[0].dtype}{list(w[0].shape)}")
    print(f"    " + " ".join(parts))


# =========================================================================
def case_fp16_baseline(verbose):
    """fp16 × fp16 3D, K=256."""
    x = [torch.randn(M, K, device='npu', dtype=torch.float16)]
    w = [torch.randn(E, K, N, device='npu', dtype=torch.float16)]
    kw = {"output_dtype": torch.float16}
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"fp16 K=256, out={out[0].shape}"


def case_bf16_baseline(verbose):
    """bf16 × bf16 3D, K=256."""
    x = [torch.randn(M, K, device='npu', dtype=torch.bfloat16)]
    w = [torch.randn(E, K, N, device='npu', dtype=torch.bfloat16)]
    kw = {"output_dtype": torch.bfloat16}
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"bf16 K=256, out={out[0].shape}"


# --- int8 weight (not packed) ---
def case_int8w_scale_bf16_bias_int32_pts(verbose):
    """int8 x, int8 w, scale=bf16, bias=int32, +pts."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w = [torch.randint(-128, 127, (E, K, N), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.randn(E, N, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "bias": [torch.zeros(E, N, device='npu', dtype=torch.int32)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"int8×int8, scale=bf16, bias=int32, +pts"


def case_int8w_scale_bf16_pts_nobias(verbose):
    """int8 x, int8 w, scale=bf16, +pts, no bias."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w = [torch.randint(-128, 127, (E, K, N), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.randn(E, N, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"int8×int8, scale=bf16, +pts, no bias"


def case_int8w_scale_i64_bias_i32(verbose):
    """int8 x, int8 w, scale=int64, bias=int32, NO pts (pure quant)."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w = [torch.randint(-128, 127, (E, K, N), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.ones(E, N, device='npu', dtype=torch.int64)],
        "bias": [torch.zeros(E, N, device='npu', dtype=torch.int32)],
        "output_dtype": torch.bfloat16,
        # no per_token_scale
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"int8×int8, scale=int64, bias=int32, no pts"


def case_int8w_scale_i64_bias_i32_pts(verbose):
    """int8 x, int8 w, scale=int64, bias=int32, +pts."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w = [torch.randint(-128, 127, (E, K, N), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.ones(E, N, device='npu', dtype=torch.int64)],
        "bias": [torch.zeros(E, N, device='npu', dtype=torch.int32)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"int8×int8, scale=int64, bias=int32, +pts"


# --- int32 packed weight ---
# WARNING: operator interprets int32 weight as int4-packed (8 int4 per int32).
# So logical N = (int8_N // 4) * 8 = int8_N * 2.
# bias/scale N must match this LOGICAL N.
# Use smaller int8_N so logical N fits.

INT8_N_SMALL = 128  # smaller weight N for int32 tests
INT32_N_LOGICAL = INT8_N_SMALL * 2  # = 256 (operator's logical N from int4 interpretation)


def case_int32w_scale_bf16_bias_bf16_pts(verbose):
    """int8 x, int32 w (int4-packed), scale=bf16, bias=bf16, +pts.
    N_bias = 2*N_weight to match int4 logical N."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, INT8_N_SMALL), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]  # (E, K, 32)
    kw = {
        "scale": [torch.randn(E, INT32_N_LOGICAL, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "bias": [torch.zeros(E, INT32_N_LOGICAL, device='npu', dtype=torch.bfloat16)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"int8×int32(int4), scale/bias=bf16(N={INT32_N_LOGICAL}), +pts"


def case_int32w_scale_i64_bias_f32_pts_exact(verbose):
    """int8 x, int32 w (int4-packed), scale=int64, bias=f32, +pts.
    N_bias = 2*N_weight to match int4 logical N."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, INT8_N_SMALL), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]
    w_scale_f32 = torch.randn(E, INT32_N_LOGICAL, device='npu', dtype=torch.float32).abs() + 0.01
    scale_i64 = [torch.from_numpy(
        w_scale_f32.cpu().numpy().view(np.uint32).astype(np.int64)
    ).npu()]
    kw = {
        "scale": scale_i64,
        "bias": [torch.randn(E, INT32_N_LOGICAL, device='npu', dtype=torch.float32)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"EXACT REPRO K=256, int32w, scale=i64 bias=f32(N={INT32_N_LOGICAL})"


# --- bf16 act + int8 weight ---
def case_bf16act_int8w_scale_bf16_pts(verbose):
    """bf16 act, int8 w, scale=bf16, +pts."""
    x = [torch.randn(M, K, device='npu', dtype=torch.bfloat16)]
    w = [torch.randint(-128, 127, (E, K, N), device='npu', dtype=torch.int8)]
    kw = {
        "scale": [torch.randn(E, N, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "bias": [torch.zeros(E, N, device='npu', dtype=torch.bfloat16)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"bf16×int8, scale=bf16, bias=bf16, +pts"


# --- int8 x, int32 w, antiquant ---
def case_int8_int32w_antiquant_pts(verbose):
    """int8 x, int32 w (int4-packed), antiquant_scale+offset (bf16), +pts.
    N_antiquant = 2*N_weight to match int4 logical N."""
    x = [torch.randint(-128, 127, (M, K), device='npu', dtype=torch.int8)]
    w_int8 = torch.randint(-128, 127, (E, K, INT8_N_SMALL), device='npu', dtype=torch.int8)
    w = [w_int8.view(torch.int32).contiguous()]
    kw = {
        "antiquant_scale": [torch.randn(E, INT32_N_LOGICAL, device='npu', dtype=torch.bfloat16).abs() + 0.01],
        "antiquant_offset": [torch.zeros(E, INT32_N_LOGICAL, device='npu', dtype=torch.bfloat16)],
        "per_token_scale": [torch.ones(M, device='npu', dtype=torch.float32)],
        "output_dtype": torch.bfloat16,
    }
    if verbose: info(x, w, kw)
    out = gmm(x, w, **kw)
    return True, f"int8×int32(int4), antiquant bf16(N={INT32_N_LOGICAL}), +pts"


# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  NPU Grouped Matmul Probe — K=256 N=256")
    print(f"  torch: {torch.__version__}, torch_npu: {torch_npu.__version__}")
    print(f"  device: {torch.npu.get_device_name(0)}")
    print(f"  M={M} K={K} N={N} E={E} gl={GL.tolist()}")
    print("=" * 55)

    tests = [
        ("B1 fp16 baseline K=256",               case_fp16_baseline),
        ("B2 bf16 baseline K=256",               case_bf16_baseline),
        ("I1 int8w scale=bf16 bias=int32 +pts",  case_int8w_scale_bf16_bias_int32_pts),
        ("I2 int8w scale=bf16 +pts no bias",     case_int8w_scale_bf16_pts_nobias),
        ("I3 int8w scale=int64 bias=int32 no pts", case_int8w_scale_i64_bias_i32),
        ("I4 int8w scale=int64 bias=int32 +pts", case_int8w_scale_i64_bias_i32_pts),
        ("P1 int32w scale=bf16 bias=bf16 +pts",  case_int32w_scale_bf16_bias_bf16_pts),
        ("P2 int32w EXACT scale=i64 bias=f32 +pts", case_int32w_scale_i64_bias_f32_pts_exact),
        ("A1 bf16act int8w scale=bf16 bias=bf16 +pts", case_bf16act_int8w_scale_bf16_pts),
        ("A2 int8 int32w antiquant bf16 +pts",   case_int8_int32w_antiquant_pts),
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
