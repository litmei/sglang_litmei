"""最小测试用例：复现 npu_quant_lightning_indexer 在 spawn 子进程中报错的问题。

现象：
- 主进程（_launch_subprocesses 的 wait_for_ready 前面）调用 foo() 能通过
- 子进程（run_scheduler_process 开头）调用 foo() 立马报错

假设：scheduler.py 的顶层 import 在 spawn 子进程中有副作用，导致算子报错。
本测试逐步缩小范围：
  Step 1: 主进程 foo()                           （基线，应通过）
  Step 2: 子进程 foo()，只 import torch/torch_npu  （最小依赖）
  Step 3: 子进程 foo()，先 import sglang.srt.managers.scheduler （加 scheduler import）
  Step 4: 子进程 foo()，import scheduler 后再初始化 NPU

运行：
    python3 test/registered/ascend/basic_function/kernels/test_repro_subprocess_npu_quant.py
"""

import multiprocessing as mp
import os
import sys
import traceback


def foo():
    """复现的算子调用（与 scheduler.py 中 foo 相同）。"""

    print(f"------- start foo (pid={os.getpid()}) ------", flush=True)

    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 0
    sparse_count = 2048
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype=torch.int32).reshape(b, -1)
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
    key_dequant_scale = key_dequant_scale.to(torch.float16)
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
        if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
        if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)

    npu_out = torch_npu.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                    key_dequant_scale.npu(),
                                                    actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                    actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                    block_table=block_table.npu(),
                                                    query_quant_mode=query_quant_mode,
                                                    key_quant_mode=key_quant_mode,
                                                    layout_query=layout_query,
                                                    layout_key=layout_key, sparse_count=sparse_count,
                                                    sparse_mode=sparse_mode)
    print(f"------- finish foo (pid={os.getpid()}) out={npu_out.shape} ------", flush=True)


# ----------------------------------------------------------------------
# 子进程入口变体
# ----------------------------------------------------------------------

def child_minimal():
    """Step 2: 只 import torch/torch_npu，不 import scheduler。"""
    try:
        print(f"[child-minimal] start (pid={os.getpid()})", flush=True)
        foo()
        print(f"[child-minimal] SUCCESS (pid={os.getpid()})", flush=True)
    except Exception as e:
        print(f"[child-minimal] FAILED (pid={os.getpid()}): {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def child_with_scheduler_import():
    """Step 3: 先 import sglang.srt.managers.scheduler，再调用 foo()。

    模拟真实 run_scheduler_process 的执行环境：
    spawn 子进程会重新 import 模块，scheduler.py 的顶层 import 会全部执行。
    如果这一步失败而 Step 2 成功，说明是 scheduler.py 的 import 有副作用。
    """
    try:
        print(f"[child-sched-import] start (pid={os.getpid()})", flush=True)
        print("[child-sched-import] importing sglang.srt.managers.scheduler ...", flush=True)
        import sglang.srt.managers.scheduler  # noqa: F401
        print("[child-sched-import] import done", flush=True)
        foo()
        print(f"[child-sched-import] SUCCESS (pid={os.getpid()})", flush=True)
    except Exception as e:
        print(f"[child-sched-import] FAILED (pid={os.getpid()}): {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def child_with_scheduler_import_and_npu_init():
    """Step 4: import scheduler + 显式 NPU 初始化，再调用 foo()。"""
    try:
        print(f"[child-sched-init] start (pid={os.getpid()})", flush=True)
        import torch
        import torch_npu
        import sglang.srt.managers.scheduler  # noqa: F401

        torch.npu.set_device(0)
        _ = torch.zeros(1, device="npu:0")
        torch.npu.synchronize()
        print(f"[child-sched-init] NPU initialized (pid={os.getpid()})", flush=True)

        foo()
        print(f"[child-sched-init] SUCCESS (pid={os.getpid()})", flush=True)
    except Exception as e:
        print(f"[child-sched-init] FAILED (pid={os.getpid()}): {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


# ----------------------------------------------------------------------
# 工具：启动子进程并报告结果
# ----------------------------------------------------------------------

def run_subprocess(name: str, target):
    mp.set_start_method("spawn", force=True)
    proc = mp.Process(target=target)
    proc.start()
    proc.join()
    status = "SUCCESS" if proc.exitcode == 0 else f"FAILED (exitcode={proc.exitcode})"
    print(f"[main] {name}: {status}", flush=True)
    return proc.exitcode == 0


def main():
    # ===== Step 1: 主进程调用 foo() =====
    print("=" * 60, flush=True)
    print("[main] Step 1: 主进程调用 foo()（基线）", flush=True)
    print("=" * 60, flush=True)
    try:
        foo()
        print("[main] Step 1: SUCCESS", flush=True)
    except Exception as e:
        print(f"[main] Step 1: FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        print("[main] 主进程都跑不过，先修主进程的问题", flush=True)
        return

    # ===== Step 2: spawn 子进程，只 import torch/torch_npu =====
    print("", flush=True)
    print("=" * 60, flush=True)
    print("[main] Step 2: spawn 子进程，最小依赖（无 scheduler import）", flush=True)
    print("=" * 60, flush=True)
    s2_ok = run_subprocess("Step 2 (minimal)", child_minimal)

    # ===== Step 3: spawn 子进程，先 import scheduler =====
    print("", flush=True)
    print("=" * 60, flush=True)
    print("[main] Step 3: spawn 子进程，import scheduler 后调用 foo()", flush=True)
    print("=" * 60, flush=True)
    s3_ok = run_subprocess("Step 3 (with scheduler import)", child_with_scheduler_import)

    # ===== Step 4: spawn 子进程，import scheduler + NPU 初始化 =====
    print("", flush=True)
    print("=" * 60, flush=True)
    print("[main] Step 4: spawn 子进程，import scheduler + NPU init + foo()", flush=True)
    print("=" * 60, flush=True)
    s4_ok = run_subprocess("Step 4 (scheduler + npu init)", child_with_scheduler_import_and_npu_init)

    # ===== 总结 =====
    print("", flush=True)
    print("=" * 60, flush=True)
    print("[main] 总结", flush=True)
    print("=" * 60, flush=True)
    print(f"  Step 1 (main process):              SUCCESS", flush=True)
    print(f"  Step 2 (minimal spawn):             {'SUCCESS' if s2_ok else 'FAILED'}", flush=True)
    print(f"  Step 3 (with scheduler import):     {'SUCCESS' if s3_ok else 'FAILED'}", flush=True)
    print(f"  Step 4 (scheduler + npu init):      {'SUCCESS' if s4_ok else 'FAILED'}", flush=True)
    print("", flush=True)
    if not s2_ok and not s3_ok:
        print("[main] 结论：spawn 子进程本身就跑不过，与 scheduler import 无关", flush=True)
    elif s2_ok and not s3_ok:
        print("[main] 结论：scheduler.py 的 import 导致了问题！", flush=True)
        print("[main] 下一步：用二分法排查 scheduler.py 的 import 列表", flush=True)
    elif s2_ok and s3_ok and not s4_ok:
        print("[main] 结论：NPU 初始化后反而失败（可能 device 占用问题）", flush=True)
    elif s2_ok and s3_ok and s4_ok:
        print("[main] 结论：都没复现，可能需要更接近真实启动流程", flush=True)


if __name__ == "__main__":
    main()
