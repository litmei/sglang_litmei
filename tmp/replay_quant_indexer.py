#!/usr/bin/env python3
"""
从 dump 文件加载实际运行的 tensor，重新执行 npu_quant_lightning_indexer 算子，
用于离线复现 segfault。

使用方式：
    # 1. 在 A5 上启动服务时设置 dump 目录
    export SGLANG_DSA_INDEXER_DUMP_DIR=/home/litmei/workspace/code/sglang_litmei/tmp/dump
    # 2. 触发 segfault
    # 3. 运行此脚本重放
    python3 tmp/replay_quant_indexer.py --dump tmp/dump/quant_inputs.pt

可选：
    --bf16-compare: 同时加载 bf16 路径的 dump，对比输出
    --contig: 强制 .contiguous() 后再执行（验证是否是 stride/storage_offset 问题）
    --device: 指定 NPU 设备，默认 npu:0
"""

import argparse
import os
import sys

import torch


def load_dump(dump_path: str):
    """加载 dump 文件，返回 (meta, tensors)。"""
    print(f"[load] loading {dump_path}")
    data = torch.load(dump_path, map_location="cpu", weights_only=False)
    meta = data["meta"]

    tensors = {
        "q_int8": data["q_int8"],
        "k_int8": data["k_int8"],
        "weights_fp16": data["weights_fp16"],
        "q_scale": data["q_scale"],
        "k_scale": data["k_scale"],
        "block_table": data["block_table"],
        "actual_seq_q_i32": data["actual_seq_q_i32"],
        "actual_seq_kv_i32": data["actual_seq_kv_i32"],
    }
    return meta, tensors


def print_meta_summary(meta: dict, label: str = ""):
    """打印 dump 的元信息。"""
    if label:
        print(f"\n=== {label} ===")
    print(f"scalar_args: {meta['scalar_args']}")
    for name in [
        "q_int8",
        "k_int8",
        "weights_fp16",
        "q_scale",
        "k_scale",
        "block_table",
        "actual_seq_q_i32",
        "actual_seq_kv_i32",
    ]:
        m = meta[name]
        print(
            f"  {name}: shape={m['shape']} dtype={m['dtype']} "
            f"stride={m['stride']} storage_offset={m['storage_offset']} "
            f"contig={m['is_contiguous']}"
        )
        if "val" in m:
            print(f"    val={m['val']}")


def replay(
    meta: dict,
    tensors: dict,
    device: str = "npu:0",
    force_contiguous: bool = False,
):
    """在 NPU 上重放算子调用。"""
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)

    # 移到 NPU
    t_npu = {}
    for name, t in tensors.items():
        t_npu[name] = t.to(device)
        if force_contiguous:
            t_npu[name] = t_npu[name].contiguous()

    args = meta["scalar_args"]
    print(f"\n[replay] calling npu_quant_lightning_indexer on {device}")
    print(f"  sparse_count={args['sparse_count']} sparse_mode={args['sparse_mode']}")
    print(f"  layout_query={args['layout_query']} layout_key={args['layout_key']}")
    print(
        f"  query_quant_mode={args['query_quant_mode']} key_quant_mode={args['key_quant_mode']}"
    )

    # 同步确保错误尽早暴露
    torch.npu.synchronize()
    print("[replay] tensors on NPU, calling operator...")

    try:
        indices = torch_npu.npu_quant_lightning_indexer(
            query=t_npu["q_int8"],
            key=t_npu["k_int8"],
            weights=t_npu["weights_fp16"],
            query_dequant_scale=t_npu["q_scale"],
            key_dequant_scale=t_npu["k_scale"],
            actual_seq_lengths_query=t_npu["actual_seq_q_i32"],
            actual_seq_lengths_key=t_npu["actual_seq_kv_i32"],
            block_table=t_npu["block_table"],
            layout_query=args["layout_query"],
            layout_key=args["layout_key"],
            sparse_count=args["sparse_count"],
            sparse_mode=args["sparse_mode"],
            query_quant_mode=args["query_quant_mode"],
            key_quant_mode=args["key_quant_mode"],
        )
        torch.npu.synchronize()
        print(f"[replay] SUCCESS")
        print(f"  indices shape={tuple(indices.shape)} dtype={indices.dtype}")
        print(f"  indices[0,0,:5]={indices[0,0,:5].cpu().tolist()}")
        return indices
    except Exception as e:
        print(f"[replay] FAILED: {type(e).__name__}: {e}")
        raise


def replay_bf16(
    meta: dict,
    tensors: dict,
    device: str = "npu:0",
    force_contiguous: bool = False,
):
    """对照：用 BF16 算子重放（需要 bf16 的 q/k/w）。

    注意：dump 的 q_int8/k_int8 是量化后的，无法直接用于 BF16 算子。
    此函数仅用于说明 BF16 算子是否能在相同 block_table/seq_len 下工作。
    需要单独 dump BF16 路径的输入。
    """
    print("[replay_bf16] 需要单独 dump BF16 路径的输入（未实现）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump",
        default="tmp/dump/quant_inputs.pt",
        help="dump 文件路径",
    )
    parser.add_argument(
        "--device", default="npu:0", help="NPU 设备"
    )
    parser.add_argument(
        "--contig",
        action="store_true",
        help="强制 .contiguous() 后再执行",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="只打印元信息，不执行算子",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dump):
        print(f"[error] dump file not found: {args.dump}")
        sys.exit(1)

    meta, tensors = load_dump(args.dump)
    print_meta_summary(meta, label="dumped meta")

    # 检查数据范围
    print("\n=== tensor stats ===")
    for name in ["q_int8", "k_int8", "weights_fp16", "q_scale", "k_scale"]:
        t = tensors[name]
        print(
            f"  {name}: min={t.float().min().item():.4f} "
            f"max={t.float().max().item():.4f} "
            f"mean={t.float().mean().item():.4f}"
        )

    if args.print_only:
        print("\n[--print-only] 不执行算子")
        return

    replay(meta, tensors, device=args.device, force_contiguous=args.contig)


if __name__ == "__main__":
    main()
