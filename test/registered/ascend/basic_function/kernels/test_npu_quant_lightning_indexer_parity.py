#!/usr/bin/env python3
"""
npu_lightning_indexer (bf16) 与 npu_quant_lightning_indexer (int8) 输出一致性测试。

验证思路：
  1. 生成相同的 bf16 query/key/weights
  2. 调用 npu_lightning_indexer 得到参考 indices（bf16 路径）
  3. 对 query/key 做 per-token-head 对称量化 -> int8 + dequant_scale
  4. 调用 npu_quant_lightning_indexer 得到量化 indices（int8 路径）
  5. 对比两者 topk_indices 的重合率
  6. 附带 CPU 参考实现，验证算子调用参数正确

运行方式（需 NPU 环境）：
    python3 -m pytest test_npu_quant_lightning_indexer_parity.py -s
或：
    python3 test_npu_quant_lightning_indexer_parity.py

环境变量：
    SGLANG_TEST_NPU_DEVICE  指定 NPU 设备号，默认 "0"
"""

import unittest

import numpy as np
import torch

# ---------------------------------------------------------------------------
# NPU 可用性检查
# ---------------------------------------------------------------------------
try:
    import torch_npu  # noqa: F401

    NPU_AVAILABLE = torch.npu.is_available() if hasattr(torch, "npu") else False
except ImportError:
    torch_npu = None
    NPU_AVAILABLE = False

NPU_DEVICE = "npu:0"


# ---------------------------------------------------------------------------
# 量化工具：per-token-head 对称量化
# ---------------------------------------------------------------------------
def per_token_head_symmetric_quantize(
    x: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对最后一维 (D) 做 per-token-head 对称量化。

    输入:  x shape [..., D],  dtype = bf16/fp16/fp32
    输出:  q_int8 shape [..., D], dtype = int8
           scale  shape [...],    dtype = fp16   (对应 npu_quant_lightning_indexer 的 dequant_scale)

    反量化: x ≈ q_int8 * scale
    """
    amax = x.abs().amax(dim=-1)  # [...]
    scale = (amax / 127.0 + eps).to(torch.float16)  # [...]
    # 广播 scale 到 x 的 shape
    q_int8 = torch.round(x / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return q_int8, scale


# ---------------------------------------------------------------------------
# FP8 量化工具：模拟 A5 环境 KV cache 以 FP8 (E4M3FN) 存储
# ---------------------------------------------------------------------------
def bf16_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """bf16/fp16/fp32 -> float8_e4m3fn。

    E4M3FN: 1 sign + 4 exponent + 3 mantissa bits
    范围: [-448, 448], 最小正常值: 2^-6 = 0.015625
    适合存储 KV cache（值域相对集中）。
    """
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("当前 PyTorch 版本不支持 float8_e4m3fn")
    return x.to(torch.float8_e4m3fn)


def fp8_to_bf16(x_fp8: torch.Tensor) -> torch.Tensor:
    """float8_e4m3fn -> bf16。"""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("当前 PyTorch 版本不支持 float8_e4m3fn")
    return x_fp8.to(torch.bfloat16)


def fp8_kv_cache_to_int8(
    k_fp8: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """模拟实际场景：KV cache 以 FP8 存储，需先反量化到 bf16，再量化到 int8。

    输入: k_fp8 shape [..., D], dtype = float8_e4m3fn
    输出: k_int8 shape [..., D], dtype = int8
          k_scale shape [...], dtype = fp16

    流程: FP8 -> BF16 -> per-token-head int8 量化
    """
    k_bf16 = fp8_to_bf16(k_fp8)  # FP8 -> BF16
    k_int8, k_scale = per_token_head_symmetric_quantize(k_bf16, eps=eps)
    return k_int8, k_scale


# ---------------------------------------------------------------------------
# CPU 参考实现（仅用于小规模验证算子调用参数正确性）
# ---------------------------------------------------------------------------
def ref_lightning_indexer_bf16(
    query: torch.Tensor,  # [T1, N1, D]  bf16
    key: torch.Tensor,  # [T2, N2, D]   bf16 (N2=1, 已展开为连续)
    weights: torch.Tensor,  # [T1, N1]   bf16
    seq_lens_q: list[int],  # 每个 batch 的 query 长度
    seq_lens_k: list[int],  # 每个 batch 的 key 长度
    sparse_count: int,
    sparse_mode: int,
) -> torch.Tensor:
    """
    CPU 参考实现: Top-k{[1]@[(W@[1])⊙ReLU(Q@K^T)]}
    返回 [T1, N2, sparse_count] 的 int32 indices。
    仅支持 TND 布局 + 非分页 key（key 已展开为连续 [T2, N2, D]）。
    """
    T1, N1, D = query.shape
    N2 = key.shape[1]
    device = query.device
    dtype = torch.float32

    q = query.to(dtype)  # [T1, N1, D]
    k = key.to(dtype)  # [T2, N2, D]
    w = weights.to(dtype)  # [T1, N1]

    # 按 batch 切分计算（因为不同 batch 的 key 长度不同）
    out_indices = torch.full((T1, N2, sparse_count), -1, dtype=torch.int32, device=device)

    q_offset = 0
    k_offset = 0
    for b, (sq, sk) in enumerate(zip(seq_lens_q, seq_lens_k)):
        q_b = q[q_offset : q_offset + sq]  # [sq, N1, D]
        k_b = k[k_offset : k_offset + sk]  # [sk, N2, D]
        w_b = w[q_offset : q_offset + sq]  # [sq, N1]

        # Q @ K^T: [sq, N1, D] x [sk, N2, D] -> [sq, N1, sk]
        logits = torch.einsum("qnd,skd->qns", q_b, k_b)  # [sq, N1, sk]

        relu_logits = torch.relu(logits)  # [sq, N1, sk]

        # W @ [1]: 把 weights 从 [sq, N1] 广播到 [sq, N1, sk]
        w_expanded = w_b.unsqueeze(-1)  # [sq, N1, 1]
        weighted = w_expanded * relu_logits  # [sq, N1, sk]

        # [1] @ weighted: 对 N1 维求和 -> [sq, sk]
        scores = weighted.sum(dim=1)  # [sq, sk]

        # causal mask 在 scores 上应用（不能在 logits 上 mask，因为 relu(-inf)=0 会抹掉 mask）
        if sparse_mode == 3:
            # rightDownCausal: Q[i] 能看到 K[j] 当 j <= i + (sk - sq)
            sq_i = torch.arange(sq, device=device)
            sk_j = torch.arange(sk, device=device)
            mask = sk_j.unsqueeze(0) <= (sq_i.unsqueeze(1) + (sk - sq))  # [sq, sk]
            scores = scores.masked_fill(~mask, float("-inf"))

        # top-k（全局 token index）
        actual_k = min(sparse_count, sk)
        topk_vals, topk_local = torch.topk(scores, actual_k, dim=-1)  # [sq, actual_k]

        # 与 NPU 一致：mask 范围不足 sparse_count 的位置用 -1 padding
        topk_global = topk_local + k_offset  # [sq, actual_k]
        invalid = torch.isinf(topk_vals) | torch.isnan(topk_vals)  # [sq, actual_k]
        topk_global = topk_global.masked_fill(invalid, -1)

        out_indices[q_offset : q_offset + sq, 0, :actual_k] = topk_global.to(torch.int32)
        q_offset += sq
        k_offset += sk

    return out_indices


def ref_lightning_indexer_quant(
    q_int8: torch.Tensor,  # [T1, N1, D]  int8
    k_int8: torch.Tensor,  # [T2, N2, D]  int8
    q_scale: torch.Tensor,  # [T1, N1]    fp16
    k_scale: torch.Tensor,  # [T2, N2]    fp16
    weights: torch.Tensor,  # [T1, N1]    fp16
    seq_lens_q: list[int],
    seq_lens_k: list[int],
    sparse_count: int,
    sparse_mode: int,
) -> torch.Tensor:
    """
    CPU 参考实现(量化版): Top-k{[1]@[(W@[1])⊙ReLU(Scale_Q@Scale_K^T⊙Q_int8@K_int8^T)]}
    返回 [T1, N2, sparse_count] 的 int32 indices。
    """
    T1, N1, D = q_int8.shape
    N2 = k_int8.shape[1]
    device = q_int8.device

    q = q_int8.to(torch.float32)  # [T1, N1, D]
    k = k_int8.to(torch.float32)  # [T2, N2, D]
    sq_scale = q_scale.to(torch.float32)  # [T1, N1]
    sk_scale = k_scale.to(torch.float32)  # [T2, N2]
    w = weights.to(torch.float32)  # [T1, N1]

    out_indices = torch.full((T1, N2, sparse_count), -1, dtype=torch.int32, device=device)

    q_offset = 0
    k_offset = 0
    for b, (sq, sk) in enumerate(zip(seq_lens_q, seq_lens_k)):
        q_b = q[q_offset : q_offset + sq]  # [sq, N1, D]
        k_b = k[k_offset : k_offset + sk]  # [sk, N2, D]
        w_b = w[q_offset : q_offset + sq]  # [sq, N1]
        sq_s = sq_scale[q_offset : q_offset + sq]  # [sq, N1]
        sk_s = sk_scale[k_offset : k_offset + sk]  # [sk, N2]

        # Q_int8 @ K_int8^T: [sq, N1, sk]
        int_logits = torch.einsum("qnd,skd->qns", q_b, k_b)

        # Scale_Q @ Scale_K^T: [sq, N1, sk]（外积）
        scale_logits = torch.einsum("qn,sn->qns", sq_s, sk_s)

        # 反量化后的 logits
        logits = int_logits * scale_logits

        relu_logits = torch.relu(logits)
        w_expanded = w_b.unsqueeze(-1)
        weighted = w_expanded * relu_logits
        scores = weighted.sum(dim=1)

        # causal mask 在 scores 上应用（relu(-inf)=0 会抹掉 logits 上的 mask）
        if sparse_mode == 3:
            sq_i = torch.arange(sq, device=device)
            sk_j = torch.arange(sk, device=device)
            mask = sk_j.unsqueeze(0) <= (sq_i.unsqueeze(1) + (sk - sq))
            scores = scores.masked_fill(~mask, float("-inf"))

        actual_k = min(sparse_count, sk)
        topk_vals, topk_local = torch.topk(scores, actual_k, dim=-1)
        topk_global = topk_local + k_offset
        invalid = torch.isinf(topk_vals) | torch.isnan(topk_vals)
        topk_global = topk_global.masked_fill(invalid, -1)
        out_indices[q_offset : q_offset + sq, 0, :actual_k] = topk_global.to(torch.int32)
        q_offset += sq
        k_offset += sk

    return out_indices


# ---------------------------------------------------------------------------
# NPU 算子调用封装
# ---------------------------------------------------------------------------
def expand_paged_key_to_tnd(
    key_paged: torch.Tensor,  # [block_count, block_size, N2, D]
    block_table: torch.Tensor,  # [B, max_blocks_per_seq]
    seq_lens_k: list[int],
    block_size: int,
) -> torch.Tensor:
    """将 PA_BSND 布局的 key 展开为连续 TND [T2, N2, D]，供 CPU 参考实现使用。"""
    blocks = []
    for b, sk in enumerate(seq_lens_k):
        num_blocks = (sk + block_size - 1) // block_size
        bt = block_table[b, :num_blocks]  # [num_blocks]
        for blk_idx in bt.tolist():
            blocks.append(key_paged[blk_idx])  # [block_size, N2, D]
    key_tnd = torch.cat(blocks, dim=0)  # [total_blocks * block_size, N2, D]
    total_tokens = sum(seq_lens_k)
    return key_tnd[:total_tokens]  # 截断到实际 token 数


def npu_lightning_indexer_call(
    query: torch.Tensor,  # [T1, N1, D] bf16
    key: torch.Tensor,  # [block_count, block_size, N2, D] bf16
    weights: torch.Tensor,  # [T1, N1] bf16
    seq_lens_q: list[int],
    seq_lens_k: list[int],
    block_table: torch.Tensor,  # [B, max_blocks]
    sparse_count: int,
    sparse_mode: int,
) -> torch.Tensor:
    """调用非量化 lightning_indexer，返回 [T1, N2, sparse_count] indices。"""
    B = len(seq_lens_q)
    # TND 布局: query 用前缀和
    actual_seq_q = torch.tensor(
        np.cumsum(seq_lens_q), dtype=torch.int32, device=query.device
    )
    # PA_BSND 布局: key 用实际长度（非前缀和），与 dsa_indexer.py 一致
    actual_seq_k = torch.tensor(
        seq_lens_k, dtype=torch.int32, device=query.device
    )

    indices, _ = torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k,
        block_table=block_table.to(torch.int32).to(query.device),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )
    return indices  # [T1, N2, sparse_count]


def npu_quant_lightning_indexer_call(
    q_int8: torch.Tensor,  # [T1, N1, D] int8
    k_int8: torch.Tensor,  # [block_count, block_size, N2, D] int8
    weights: torch.Tensor,  # [T1, N1] fp16
    q_scale: torch.Tensor,  # [T1, N1] fp16
    k_scale: torch.Tensor,  # [block_count, block_size, N2] fp16
    seq_lens_q: list[int],
    seq_lens_k: list[int],
    block_table: torch.Tensor,
    sparse_count: int,
    sparse_mode: int,
) -> torch.Tensor:
    """调用量化 lightning_indexer，返回 [T1, N2, sparse_count] indices。"""
    device = q_int8.device
    # TND query: 前缀和; PA_BSND key: 实际长度
    actual_seq_q = torch.tensor(np.cumsum(seq_lens_q), dtype=torch.int32, device=device)
    actual_seq_k = torch.tensor(seq_lens_k, dtype=torch.int32, device=device)

    indices = torch_npu.npu_quant_lightning_indexer(
        query=q_int8,
        key=k_int8,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=k_scale,
        query_quant_mode=0,
        key_quant_mode=0,
        actual_seq_lengths_query=actual_seq_q,
        actual_seq_lengths_key=actual_seq_k,
        block_table=block_table.to(torch.int32).to(device),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
    )
    return indices  # [T1, N2, sparse_count]


# ---------------------------------------------------------------------------
# 对比工具
# ---------------------------------------------------------------------------
def compute_overlap_ratio(
    indices_a: torch.Tensor,  # [T1, N2, k]
    indices_b: torch.Tensor,  # [T1, N2, k]
) -> float:
    """计算每行 topk indices 的平均重合率（集合交集 / k）。"""
    T1, N2, k = indices_a.shape
    total_overlap = 0.0
    count = 0
    for t in range(T1):
        for n in range(N2):
            set_a = set(indices_a[t, n].tolist())
            set_a.discard(-1)
            set_b = set(indices_b[t, n].tolist())
            set_b.discard(-1)
            if len(set_a) == 0 and len(set_b) == 0:
                continue
            union = set_a | set_b
            if len(union) == 0:
                continue
            total_overlap += len(set_a & set_b) / max(len(set_a), len(set_b))
            count += 1
    return total_overlap / count if count > 0 else 0.0


def sort_indices_rowwise(indices: torch.Tensor) -> torch.Tensor:
    """对每行的 topk indices 排序，用于稳定比较（忽略 topk 内部顺序）。"""
    sorted_vals, _ = torch.sort(indices, dim=-1)
    return sorted_vals


# ---------------------------------------------------------------------------
# 测试数据生成
# ---------------------------------------------------------------------------
def gen_test_data(
    seq_lens_q: list[int],
    seq_lens_k: list[int],
    n1: int = 64,
    n2: int = 1,
    d: int = 128,
    block_size: int = 128,
    sparse_count: int = 64,
    sparse_mode: int = 3,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """生成完整的测试数据（CPU 或 NPU）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    B = len(seq_lens_q)
    T1 = sum(seq_lens_q)
    total_k_tokens = sum(seq_lens_k)
    total_blocks = sum((sk + block_size - 1) // block_size for sk in seq_lens_k)
    max_blocks_per_seq = max((sk + block_size - 1) // block_size for sk in seq_lens_k)

    # query: [T1, N1, D] — 用较大幅值确保 Q@K^T 有区分度，避免 topk 退化为噪声
    query = torch.tensor(
        np.random.uniform(-5, 5, (T1, n1, d)), dtype=dtype, device=device
    )
    # key (paged): [total_blocks, block_size, N2, D]
    key = torch.tensor(
        np.random.uniform(-5, 5, (total_blocks, block_size, n2, d)),
        dtype=dtype,
        device=device,
    )
    # weights: [T1, N1] — 用较大幅值让不同 head 的贡献有区分度
    weights = torch.tensor(
        np.random.uniform(-0.5, 0.5, (T1, n1)), dtype=dtype, device=device
    )

    # block_table: [B, max_blocks_per_seq]
    block_table = torch.zeros((B, max_blocks_per_seq), dtype=torch.int32, device=device)
    blk_offset = 0
    for b, sk in enumerate(seq_lens_k):
        nb = (sk + block_size - 1) // block_size
        block_table[b, :nb] = torch.arange(
            blk_offset, blk_offset + nb, dtype=torch.int32, device=device
        )
        blk_offset += nb

    return {
        "query": query,
        "key": key,
        "weights": weights,
        "block_table": block_table,
        "seq_lens_q": seq_lens_q,
        "seq_lens_k": seq_lens_k,
        "n1": n1,
        "n2": n2,
        "d": d,
        "block_size": block_size,
        "sparse_count": sparse_count,
        "sparse_mode": sparse_mode,
        "total_blocks": total_blocks,
    }


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------
@unittest.skipUnless(NPU_AVAILABLE, "NPU 不可用，跳过")
class TestNpuQuantLightningIndexerParity(unittest.TestCase):
    """npu_lightning_indexer vs npu_quant_lightning_indexer 一致性测试。"""

    OVERLAP_THRESHOLD = 0.90  # 量化引入的误差允许 10% 的 topk 差异

    def setUp(self):
        """每个测试前清理 NPU 状态。"""
        torch.npu.set_device(0)
        torch.npu.empty_cache()

    def tearDown(self):
        """每个测试后同步并清理，避免异步错误影响后续测试。"""
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def _run_parity_test(self, data: dict, label: str):
        """核心：对比 bf16 算子与 int8 算子的 topk indices 重合率。"""
        device = NPU_DEVICE
        q = data["query"].to(device)
        k = data["key"].to(device)
        w = data["weights"].to(device)
        bt = data["block_table"]
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]

        # --- 非量化路径 (bf16) ---
        indices_bf16 = npu_lightning_indexer_call(q, k, w, sq, sk, bt, sc, sm)
        torch.npu.synchronize()  # 同步以尽早发现异步错误
        # [T1, N2, k]

        # --- 量化路径 (int8) ---
        # weights 需要 fp16
        w_fp16 = w.to(torch.float16)
        q_int8, q_scale = per_token_head_symmetric_quantize(q)
        # key: [total_blocks, block_size, N2, D] -> scale [total_blocks, block_size, N2]
        k_int8, k_scale = per_token_head_symmetric_quantize(k)
        q_int8 = q_int8.to(device)
        q_scale = q_scale.to(device)
        k_int8 = k_int8.to(device)
        k_scale = k_scale.to(device)

        indices_int8 = npu_quant_lightning_indexer_call(
            q_int8, k_int8, w_fp16, q_scale, k_scale, sq, sk, bt, sc, sm
        )
        torch.npu.synchronize()  # 同步以尽早发现异步错误

        # --- 对比 ---
        overlap = compute_overlap_ratio(indices_bf16.cpu(), indices_int8.cpu())
        print(f"\n[{label}] bf16 vs int8 topk 重合率: {overlap:.4f}")
        print(f"  bf16 indices[0,0,:10]: {indices_bf16[0,0,:10].cpu().tolist()}")
        print(f"  int8 indices[0,0,:10]: {indices_int8[0,0,:10].cpu().tolist()}")
        print(f"  bf16 shape: {indices_bf16.shape}, int8 shape: {indices_int8.shape}")

        self.assertGreaterEqual(
            overlap,
            self.OVERLAP_THRESHOLD,
            f"{label}: bf16 vs int8 重合率 {overlap:.4f} < 阈值 {self.OVERLAP_THRESHOLD}",
        )

    def test_single_batch_prefill_spmode3(self):
        """单 batch prefill（sq==sk），sparse_mode=3 (causal)。"""
        data = gen_test_data(
            seq_lens_q=[512],
            seq_lens_k=[512],
            sparse_count=64,
            sparse_mode=3,
            seed=42,
        )
        self._run_parity_test(data, "single_batch_prefill_spmode3")

    def test_multi_batch_prefill_spmode3(self):
        """多 batch prefill（不同长度，sq==sk），sparse_mode=3。"""
        data = gen_test_data(
            seq_lens_q=[512, 1024],
            seq_lens_k=[512, 1024],
            sparse_count=64,
            sparse_mode=3,
            seed=100,
        )
        self._run_parity_test(data, "multi_batch_prefill_spmode3")

    def test_decode_spmode3(self):
        """decode 场景（query 长度=1），sparse_mode=3。"""
        data = gen_test_data(
            seq_lens_q=[1, 1, 1],
            seq_lens_k=[256, 512, 1024],
            sparse_count=64,
            sparse_mode=3,
            seed=200,
        )
        self._run_parity_test(data, "decode_spmode3")

    def test_large_sparse_count(self):
        """大 sparse_count=2048，验证长上下文。"""
        data = gen_test_data(
            seq_lens_q=[4096],
            seq_lens_k=[4096],
            sparse_count=2048,
            sparse_mode=3,
            seed=300,
        )
        self._run_parity_test(data, "large_sparse_count")

    def test_spmode0_no_mask(self):
        """sparse_mode=0 (无 mask)，验证基础等价性。"""
        data = gen_test_data(
            seq_lens_q=[512],
            seq_lens_k=[512],
            sparse_count=64,
            sparse_mode=0,
            seed=400,
        )
        self._run_parity_test(data, "spmode0_no_mask")


@unittest.skipUnless(NPU_AVAILABLE, "NPU 不可用，跳过")
class TestNpuQuantRealisticScenario(unittest.TestCase):
    """模拟实际运行场景的测试，匹配 dsa_indexer.py 中的真实参数。

    关键差异（vs 其他测试）：
    - n1=32（GLM-5.2 实际值），pad 到 64
    - 用 npu_dynamic_quant（而非手动量化）
    - block_count 较大（模拟真实 KV cache pool）
    - sparse_count=2048（实际配置）
    """

    def setUp(self):
        torch.npu.set_device(0)
        torch.npu.empty_cache()

    def tearDown(self):
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def _gen_realistic_data(
        self,
        seq_lens_q: list[int],
        seq_lens_k: list[int],
        n1: int = 32,
        block_count: int = 500,
        seed: int = 42,
    ) -> dict:
        """生成匹配实际运行场景的测试数据。

        Args:
            block_count: KV cache pool 中的总 block 数（远大于实际使用的 block）
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        n2 = 1
        d = 128
        block_size = 128
        B = len(seq_lens_q)
        T1 = sum(seq_lens_q)

        # query: [T1, N1, D] bf16
        query = torch.randn(T1, n1, d, dtype=torch.bfloat16) * 3.0

        # key (paged): [block_count, block_size, N2, D] bf16
        # block_count 远大于实际使用的 block，模拟真实 KV cache pool
        key = torch.randn(block_count, block_size, n2, d, dtype=torch.bfloat16) * 3.0

        # weights: [T1, N1] bf16
        weights = torch.randn(T1, n1, dtype=torch.bfloat16) * 0.3

        # block_table: [B, max_blocks_per_seq]
        max_blocks_per_seq = max(
            (sk + block_size - 1) // block_size for sk in seq_lens_k
        )
        block_table = torch.zeros(
            (B, max_blocks_per_seq), dtype=torch.int32
        )
        blk_offset = 0
        for b, sk in enumerate(seq_lens_k):
            nb = (sk + block_size - 1) // block_size
            block_table[b, :nb] = torch.arange(
                blk_offset, blk_offset + nb, dtype=torch.int32
            )
            blk_offset += nb

        return {
            "query": query,
            "key": key,
            "weights": weights,
            "block_table": block_table,
            "seq_lens_q": seq_lens_q,
            "seq_lens_k": seq_lens_k,
            "n1": n1,
            "n2": n2,
            "d": d,
            "block_size": block_size,
            "sparse_count": 2048,
            "sparse_mode": 3,
            "block_count": block_count,
        }

    def _run_realistic_test(self, data: dict, label: str):
        """用 npu_dynamic_quant + N1 pad 到 64，匹配实际运行路径。"""
        device = NPU_DEVICE
        q = data["query"].to(device)
        k = data["key"].to(device)
        w = data["weights"].to(device)
        bt = data["block_table"].clone()
        # 如果有 block_table 偏移（模拟 KV cache pool 切片），应用到 block_table
        if "_block_table_offset" in data:
            offset = data["_block_table_offset"]
            # 只对非 0 位置加偏移（0 是 padding）
            mask = bt != 0
            bt[mask] += offset
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]
        n1_orig = data["n1"]

        # actual_seq_lengths: TND query 用前缀和, PA_BSND key 用实际长度
        actual_seq_q = torch.tensor(
            np.cumsum(sq), dtype=torch.int32, device=device
        )
        actual_seq_k = torch.tensor(sk, dtype=torch.int32, device=device)

        print(f"\n[{label}] n1_orig={n1_orig}, block_count={data['block_count']}")
        print(f"  q: shape={tuple(q.shape)} dtype={q.dtype}")
        print(f"  k: shape={tuple(k.shape)} dtype={k.dtype}")
        print(f"  w: shape={tuple(w.shape)} dtype={w.dtype}")
        print(f"  actual_seq_q={actual_seq_q.tolist()}")
        print(f"  actual_seq_k={actual_seq_k.tolist()}")
        print(f"  block_table: shape={tuple(bt.shape)} dtype={bt.dtype}")

        # --- bf16 路径（对照组）---
        indices_bf16, _ = torch_npu.npu_lightning_indexer(
            query=q,
            key=k,
            weights=w,
            actual_seq_lengths_query=actual_seq_q,
            actual_seq_lengths_key=actual_seq_k,
            block_table=bt.to(device),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sc,
            sparse_mode=sm,
        )
        torch.npu.synchronize()
        print(f"  [bf16] indices shape={tuple(indices_bf16.shape)}")
        print(f"  [bf16] indices[0,0,:5]={indices_bf16[0,0,:5].cpu().tolist()}")

        # --- int8 路径（用 npu_dynamic_quant，匹配实际源码）---
        # 1) 量化 query: [T1, N1, D] -> int8 [T1, N1, D] + fp16 scale [T1, N1]
        q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
        q_scale = q_scale.to(torch.float16)

        # 2) 量化 key: [block_count, block_size, N2, D] -> int8 + fp16 scale
        num_pages, page_size, n2, d = k.shape
        k_2d = k.view(num_pages * page_size, d)
        k_int8, k_scale = torch_npu.npu_dynamic_quant(k_2d)
        k_int8 = k_int8.view(num_pages, page_size, n2, d)
        k_scale = k_scale.to(torch.float16).view(num_pages, page_size, n2)

        # 3) N1 pad 到 64（npu_quant_lightning_indexer 要求 N1=64）
        weights_fp16 = w.to(torch.float16)
        if n1_orig != 64:
            pad_n = 64 - n1_orig
            q_int8 = torch.nn.functional.pad(q_int8, (0, 0, 0, pad_n))
            q_scale = torch.nn.functional.pad(q_scale, (0, pad_n))
            weights_fp16 = torch.nn.functional.pad(weights_fp16, (0, pad_n))

        print(f"  [int8] q_int8: shape={tuple(q_int8.shape)} dtype={q_int8.dtype} stride={q_int8.stride()} contig={q_int8.is_contiguous()}")
        print(f"  [int8] q_scale: shape={tuple(q_scale.shape)} dtype={q_scale.dtype} stride={q_scale.stride()} contig={q_scale.is_contiguous()}")
        print(f"  [int8] k_int8: shape={tuple(k_int8.shape)} dtype={k_int8.dtype} stride={k_int8.stride()} contig={k_int8.is_contiguous()}")
        print(f"  [int8] k_scale: shape={tuple(k_scale.shape)} dtype={k_scale.dtype} stride={k_scale.stride()} contig={k_scale.is_contiguous()}")
        print(f"  [int8] weights_fp16: shape={tuple(weights_fp16.shape)} dtype={weights_fp16.dtype} stride={weights_fp16.stride()} contig={weights_fp16.is_contiguous()}")
        # 打印 storage_offset，用于对比实际运行
        print(f"  [int8] storage_offset: q_int8={q_int8.storage_offset()} k_int8={k_int8.storage_offset()} q_scale={q_scale.storage_offset()} k_scale={k_scale.storage_offset()} w={weights_fp16.storage_offset()}")

        # 4) 调用量化算子
        indices_int8 = torch_npu.npu_quant_lightning_indexer(
            query=q_int8,
            key=k_int8,
            weights=weights_fp16,
            query_dequant_scale=q_scale,
            key_dequant_scale=k_scale,
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=actual_seq_q,
            actual_seq_lengths_key=actual_seq_k,
            block_table=bt.to(device),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sc,
            sparse_mode=sm,
        )
        torch.npu.synchronize()
        print(f"  [int8] indices shape={tuple(indices_int8.shape)}")
        print(f"  [int8] indices[0,0,:5]={indices_int8[0,0,:5].cpu().tolist()}")

        # --- 对比 ---
        overlap = compute_overlap_ratio(indices_bf16.cpu(), indices_int8.cpu())
        print(f"  [{label}] bf16 vs int8 topk 重合率: {overlap:.4f}")
        self.assertGreaterEqual(overlap, 0.85, f"{label}: 重合率 {overlap:.4f} < 0.85")

    def test_realistic_decode_small(self):
        """模拟实际 decode 场景：B=1, sq=6, sk=6, n1=32, block_count=500."""
        data = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=32,
            block_count=500,
            seed=42,
        )
        self._run_realistic_test(data, "realistic_decode_small")

    def test_realistic_decode_large_pool(self):
        """模拟大 KV cache pool：B=1, sq=6, sk=6, n1=32, block_count=26857."""
        data = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=32,
            block_count=26857,
            seed=42,
        )
        self._run_realistic_test(data, "realistic_decode_large_pool")

    def test_realistic_prefill(self):
        """模拟实际 prefill 场景：B=1, sq=512, sk=512, n1=32, block_count=500."""
        data = self._gen_realistic_data(
            seq_lens_q=[512],
            seq_lens_k=[512],
            n1=32,
            block_count=500,
            seed=100,
        )
        self._run_realistic_test(data, "realistic_prefill")

    def test_realistic_n1_64_nodynamic(self):
        """对照：n1=64（原生），用 npu_dynamic_quant，验证 n1=32+pad 是否等价."""
        data = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=64,
            block_count=500,
            seed=42,
        )
        self._run_realistic_test(data, "realistic_n1_64")

    def test_realistic_kv_pool_slice(self):
        """模拟 KV cache pool 切片：k_paged 从更大的 pool 切片得到，storage_offset 非 0。

        实际运行中 past_key_states 是 KV cache pool 的一部分，storage_offset 可能非 0。
        虽然 npu_dynamic_quant 返回新 tensor（offset=0），但此测试验证切片输入不会影响。
        """
        data = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=32,
            block_count=500,
            seed=42,
        )
        # 从更大的 pool 切片，模拟 KV cache pool
        pool_extra = 100
        k_orig = data["key"]
        pool = torch.randn(
            k_orig.shape[0] + pool_extra,
            k_orig.shape[1],
            k_orig.shape[2],
            k_orig.shape[3],
            dtype=k_orig.dtype,
        )
        # 切片，使 data["key"] 的 storage_offset 非 0
        data["key"] = pool[pool_extra:]
        # block_table 中的 index 需要偏移
        data["_block_table_offset"] = pool_extra
        self._run_realistic_test(data, "realistic_kv_pool_slice")

    def test_realistic_nonzero_block_table(self):
        """模拟 block_table 指向非 0 block：实际运行中 block_table[0,0] 可能非 0。"""
        data = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=32,
            block_count=500,
            seed=42,
        )
        # 修改 block_table 指向非 0 block
        data["block_table"][0, 0] = 100
        self._run_realistic_test(data, "realistic_nonzero_block_table")

    def test_realistic_multi_call(self):
        """模拟连续多次调用：实际运行中可能在同一 stream 上连续调用多次。"""
        data = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=32,
            block_count=500,
            seed=42,
        )
        # 第一次调用
        self._run_realistic_test(data, "realistic_multi_call_1")
        # 第二次调用（不同 seed）
        data2 = self._gen_realistic_data(
            seq_lens_q=[6],
            seq_lens_k=[6],
            n1=32,
            block_count=500,
            seed=100,
        )
        self._run_realistic_test(data2, "realistic_multi_call_2")


@unittest.skipUnless(NPU_AVAILABLE, "NPU 不可用，跳过")
class TestNpuQuantFP8Scenario(unittest.TestCase):
    """FP8 (E4M3FN) 直接输入场景测试，模拟 A5 环境。

    A5 环境中 KV cache 以 FP8 存储。此测试直接将 FP8 tensor 传给
    npu_quant_lightning_indexer 算子（不转 int8），验证算子是否原生支持 FP8。

    两种传法：
    1. view(torch.int8): 共享底层存储，FP8 字节直接当 int8 解释
    2. 直接传 FP8 dtype: 让算子自己处理（可能报错）
    """

    OVERLAP_THRESHOLD = 0.85

    def setUp(self):
        torch.npu.set_device(0)
        torch.npu.empty_cache()

    def tearDown(self):
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def _gen_fp8_data(
        self,
        seq_lens_q: list[int],
        seq_lens_k: list[int],
        n1: int = 32,
        block_count: int = 500,
        seed: int = 42,
    ) -> dict:
        """生成 FP8 场景的测试数据。query 和 key 均为 FP8。"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        n2 = 1
        d = 128
        block_size = 128
        B = len(seq_lens_q)
        T1 = sum(seq_lens_q)

        # query: [T1, N1, D] bf16 -> FP8
        q_bf16 = torch.randn(T1, n1, d, dtype=torch.bfloat16) * 3.0
        query_fp8 = bf16_to_fp8(q_bf16)
        # key (paged): [block_count, block_size, N2, D] bf16 -> FP8
        k_bf16 = torch.randn(
            block_count, block_size, n2, d, dtype=torch.bfloat16
        ) * 3.0
        key_fp8 = bf16_to_fp8(k_bf16)
        # weights: [T1, N1] bf16
        weights = torch.randn(T1, n1, dtype=torch.bfloat16) * 0.3

        # block_table
        max_blocks_per_seq = max(
            (sk + block_size - 1) // block_size for sk in seq_lens_k
        )
        block_table = torch.zeros(
            (B, max_blocks_per_seq), dtype=torch.int32
        )
        blk_offset = 0
        for b, sk in enumerate(seq_lens_k):
            nb = (sk + block_size - 1) // block_size
            block_table[b, :nb] = torch.arange(
                blk_offset, blk_offset + nb, dtype=torch.int32
            )
            blk_offset += nb

        return {
            "query_bf16": q_bf16,
            "query_fp8": query_fp8,
            "key_bf16": k_bf16,
            "key_fp8": key_fp8,
            "weights": weights,
            "block_table": block_table,
            "seq_lens_q": seq_lens_q,
            "seq_lens_k": seq_lens_k,
            "n1": n1,
            "n2": n2,
            "d": d,
            "block_size": block_size,
            "sparse_count": 2048,
            "sparse_mode": 3,
            "block_count": block_count,
        }

    def _run_fp8_direct_view_test(self, data: dict, label: str):
        """方式1: FP8 -> view(int8) 后传给算子。

        FP8 与 int8 都是 1 字节，view(int8) 共享底层存储。
        scale 设为 1.0，让算子把 FP8 字节直接当 int8 用。
        这会产生错误结果（位模式不同），但验证算子是否不 segfault。
        """
        device = NPU_DEVICE
        q_fp8 = data["query_fp8"].to(device)
        k_fp8 = data["key_fp8"].to(device)
        w = data["weights"].to(device)
        bt = data["block_table"]
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]
        n1_orig = data["n1"]

        actual_seq_q = torch.tensor(
            np.cumsum(sq), dtype=torch.int32, device=device
        )
        actual_seq_k = torch.tensor(sk, dtype=torch.int32, device=device)

        print(f"\n[{label}] n1_orig={n1_orig}, block_count={data['block_count']}")
        print(f"  q_fp8: shape={tuple(q_fp8.shape)} dtype={q_fp8.dtype}")
        print(f"  k_fp8: shape={tuple(k_fp8.shape)} dtype={k_fp8.dtype}")

        # view(int8)
        num_pages, page_size, n2, d = k_fp8.shape
        q_view = q_fp8.view(torch.int8)
        k_view = k_fp8.view(torch.int8)

        # scale 用 1.0
        q_scale = torch.ones(
            q_view.shape[0], q_view.shape[1],
            dtype=torch.float16, device=device
        )
        k_scale = torch.ones(
            num_pages, page_size, n2, dtype=torch.float16, device=device
        )
        weights_fp16 = w.to(torch.float16)

        # N1 pad 到 64
        if n1_orig != 64:
            pad_n = 64 - n1_orig
            q_view = torch.nn.functional.pad(q_view, (0, 0, 0, pad_n))
            q_scale = torch.nn.functional.pad(q_scale, (0, pad_n))
            weights_fp16 = torch.nn.functional.pad(weights_fp16, (0, pad_n))

        print(f"  q_view: shape={tuple(q_view.shape)} dtype={q_view.dtype}")
        print(f"  k_view: shape={tuple(k_view.shape)} dtype={k_view.dtype}")

        try:
            indices = torch_npu.npu_quant_lightning_indexer(
                query=q_view,
                key=k_view,
                weights=weights_fp16,
                query_dequant_scale=q_scale,
                key_dequant_scale=k_scale,
                query_quant_mode=0,
                key_quant_mode=0,
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_key=actual_seq_k,
                block_table=bt.to(device),
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=sc,
                sparse_mode=sm,
            )
            torch.npu.synchronize()
            print(f"  [fp8_view] SUCCESS: indices shape={tuple(indices.shape)}")
            print(f"  [fp8_view] indices[0,0,:5]={indices[0,0,:5].cpu().tolist()}")
        except Exception as e:
            print(f"  [fp8_view] FAILED: {type(e).__name__}: {e}")
            raise

    def _run_fp8_direct_dtype_test(self, data: dict, label: str):
        """方式2: 直接传 FP8 dtype 给算子（不 view int8）。

        算子文档说仅支持 int8，此测试验证算子对 FP8 dtype 的行为。
        如果报错，说明算子不支持 FP8；如果不报错，验证结果。
        """
        device = NPU_DEVICE
        q_fp8 = data["query_fp8"].to(device)
        k_fp8 = data["key_fp8"].to(device)
        w = data["weights"].to(device)
        bt = data["block_table"]
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]
        n1_orig = data["n1"]

        actual_seq_q = torch.tensor(
            np.cumsum(sq), dtype=torch.int32, device=device
        )
        actual_seq_k = torch.tensor(sk, dtype=torch.int32, device=device)

        print(f"\n[{label}] n1_orig={n1_orig}, block_count={data['block_count']}")
        print(f"  q_fp8: shape={tuple(q_fp8.shape)} dtype={q_fp8.dtype}")
        print(f"  k_fp8: shape={tuple(k_fp8.shape)} dtype={k_fp8.dtype}")

        # scale 用 1.0
        num_pages, page_size, n2, d = k_fp8.shape
        q_scale = torch.ones(
            q_fp8.shape[0], q_fp8.shape[1],
            dtype=torch.float16, device=device
        )
        k_scale = torch.ones(
            num_pages, page_size, n2, dtype=torch.float16, device=device
        )
        weights_fp16 = w.to(torch.float16)

        # N1 pad 到 64
        if n1_orig != 64:
            pad_n = 64 - n1_orig
            q_fp8 = torch.nn.functional.pad(q_fp8, (0, 0, 0, pad_n))
            q_scale = torch.nn.functional.pad(q_scale, (0, pad_n))
            weights_fp16 = torch.nn.functional.pad(weights_fp16, (0, pad_n))

        print(f"  q_fp8 (padded): shape={tuple(q_fp8.shape)} dtype={q_fp8.dtype}")

        try:
            indices = torch_npu.npu_quant_lightning_indexer(
                query=q_fp8,
                key=k_fp8,
                weights=weights_fp16,
                query_dequant_scale=q_scale,
                key_dequant_scale=k_scale,
                query_quant_mode=0,
                key_quant_mode=0,
                actual_seq_lengths_query=actual_seq_q,
                actual_seq_lengths_key=actual_seq_k,
                block_table=bt.to(device),
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=sc,
                sparse_mode=sm,
            )
            torch.npu.synchronize()
            print(f"  [fp8_dtype] SUCCESS: indices shape={tuple(indices.shape)}")
            print(f"  [fp8_dtype] indices[0,0,:5]={indices[0,0,:5].cpu().tolist()}")
        except Exception as e:
            print(f"  [fp8_dtype] FAILED: {type(e).__name__}: {e}")
            raise

    def _run_fp8_parity_test(self, data: dict, label: str):
        """FP8 parity 测试：FP8 直接输入 vs BF16 参考路径。

        对比 FP8 直接传给算子的输出与 BF16 算子输出的重合率。
        用于验证 FP8 路径的正确性（如果算子支持 FP8）。
        """
        device = NPU_DEVICE
        q_bf16 = data["query_bf16"].to(device)
        k_bf16 = data["key_bf16"].to(device)
        q_fp8 = data["query_fp8"].to(device)
        k_fp8 = data["key_fp8"].to(device)
        w = data["weights"].to(device)
        bt = data["block_table"]
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]
        n1_orig = data["n1"]

        actual_seq_q = torch.tensor(
            np.cumsum(sq), dtype=torch.int32, device=device
        )
        actual_seq_k = torch.tensor(sk, dtype=torch.int32, device=device)

        print(f"\n[{label}] n1_orig={n1_orig}, block_count={data['block_count']}")

        # --- BF16 参考路径 ---
        indices_bf16, _ = torch_npu.npu_lightning_indexer(
            query=q_bf16,
            key=k_bf16,
            weights=w,
            actual_seq_lengths_query=actual_seq_q,
            actual_seq_lengths_key=actual_seq_k,
            block_table=bt.to(device),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sc,
            sparse_mode=sm,
        )
        torch.npu.synchronize()
        print(f"  [bf16] indices[0,0,:5]={indices_bf16[0,0,:5].cpu().tolist()}")

        # --- FP8 路径（view int8）---
        num_pages, page_size, n2, d = k_fp8.shape
        q_view = q_fp8.view(torch.int8)
        k_view = k_fp8.view(torch.int8)
        q_scale = torch.ones(
            q_view.shape[0], q_view.shape[1],
            dtype=torch.float16, device=device
        )
        k_scale = torch.ones(
            num_pages, page_size, n2, dtype=torch.float16, device=device
        )
        weights_fp16 = w.to(torch.float16)

        if n1_orig != 64:
            pad_n = 64 - n1_orig
            q_view = torch.nn.functional.pad(q_view, (0, 0, 0, pad_n))
            q_scale = torch.nn.functional.pad(q_scale, (0, pad_n))
            weights_fp16 = torch.nn.functional.pad(weights_fp16, (0, pad_n))

        indices_fp8 = torch_npu.npu_quant_lightning_indexer(
            query=q_view,
            key=k_view,
            weights=weights_fp16,
            query_dequant_scale=q_scale,
            key_dequant_scale=k_scale,
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=actual_seq_q,
            actual_seq_lengths_key=actual_seq_k,
            block_table=bt.to(device),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=sc,
            sparse_mode=sm,
        )
        torch.npu.synchronize()
        print(f"  [fp8] indices[0,0,:5]={indices_fp8[0,0,:5].cpu().tolist()}")

        overlap = compute_overlap_ratio(indices_bf16.cpu(), indices_fp8.cpu())
        print(f"  [{label}] bf16 vs fp8 topk 重合率: {overlap:.4f}")
        # FP8 view int8 会产生错误结果，重合率低是预期的
        # 这里不强制阈值，只打印对比结果

    # ===== view(int8) 方式测试 =====
    def test_fp8_view_decode_small(self):
        """FP8 view int8 - decode 场景：B=1, sq=6, sk=6, n1=32."""
        data = self._gen_fp8_data(
            seq_lens_q=[6], seq_lens_k=[6], n1=32, block_count=500, seed=42
        )
        self._run_fp8_direct_view_test(data, "fp8_view_decode_small")

    def test_fp8_view_decode_large_pool(self):
        """FP8 view int8 - 大 pool：B=1, sq=6, sk=6, n1=32, block_count=26857."""
        data = self._gen_fp8_data(
            seq_lens_q=[6], seq_lens_k=[6], n1=32, block_count=26857, seed=42
        )
        self._run_fp8_direct_view_test(data, "fp8_view_decode_large_pool")

    def test_fp8_view_prefill(self):
        """FP8 view int8 - prefill 场景：B=1, sq=512, sk=512, n1=32."""
        data = self._gen_fp8_data(
            seq_lens_q=[512], seq_lens_k=[512], n1=32, block_count=500, seed=100
        )
        self._run_fp8_direct_view_test(data, "fp8_view_prefill")

    # ===== 直接传 FP8 dtype 测试 =====
    def test_fp8_dtype_decode_small(self):
        """FP8 直接 dtype - decode 场景：B=1, sq=6, sk=6, n1=32."""
        data = self._gen_fp8_data(
            seq_lens_q=[6], seq_lens_k=[6], n1=32, block_count=500, seed=42
        )
        self._run_fp8_direct_dtype_test(data, "fp8_dtype_decode_small")

    def test_fp8_dtype_decode_large_pool(self):
        """FP8 直接 dtype - 大 pool：B=1, sq=6, sk=6, n1=32, block_count=26857."""
        data = self._gen_fp8_data(
            seq_lens_q=[6], seq_lens_k=[6], n1=32, block_count=26857, seed=42
        )
        self._run_fp8_direct_dtype_test(data, "fp8_dtype_decode_large_pool")

    # ===== parity 对比测试 =====
    def test_fp8_parity_decode_small(self):
        """FP8 vs BF16 parity - decode 场景。"""
        data = self._gen_fp8_data(
            seq_lens_q=[6], seq_lens_k=[6], n1=32, block_count=500, seed=42
        )
        self._run_fp8_parity_test(data, "fp8_parity_decode_small")

    def test_fp8_parity_prefill(self):
        """FP8 vs BF16 parity - prefill 场景。"""
        data = self._gen_fp8_data(
            seq_lens_q=[512], seq_lens_k=[512], n1=32, block_count=500, seed=100
        )
        self._run_fp8_parity_test(data, "fp8_parity_prefill")


@unittest.skipUnless(NPU_AVAILABLE, "NPU 不可用，跳过")
class TestNpuLightningIndexerAgainstRef(unittest.TestCase):
    """NPU 算子 vs CPU 参考实现，验证算子调用参数正确。"""

    MATCH_THRESHOLD = 0.85  # NPU 算子内部实现可能与参考实现略有差异

    def setUp(self):
        torch.npu.set_device(0)
        torch.npu.empty_cache()

    def tearDown(self):
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def _run_ref_comparison(self, data: dict, label: str):
        """对比 NPU 算子与 CPU 参考实现。"""
        device = NPU_DEVICE
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]

        # --- CPU 参考实现 (bf16) ---
        q_cpu = data["query"]
        k_cpu = data["key"]
        w_cpu = data["weights"]
        # 展开分页 key 为连续 TND
        k_tnd = expand_paged_key_to_tnd(
            k_cpu, data["block_table"], sk, data["block_size"]
        )
        ref_indices = ref_lightning_indexer_bf16(
            q_cpu, k_tnd, w_cpu, sq, sk, sc, sm
        )

        # --- NPU 算子 (bf16) ---
        q_npu = q_cpu.to(device)
        k_npu = k_cpu.to(device)
        w_npu = w_cpu.to(device)
        npu_indices = npu_lightning_indexer_call(
            q_npu, k_npu, w_npu, sq, sk, data["block_table"], sc, sm
        ).cpu()

        overlap = compute_overlap_ratio(ref_indices, npu_indices)
        print(f"\n[{label}] CPU-ref vs NPU-bf16 重合率: {overlap:.4f}")
        print(f"  ref  indices[0,0,:10]: {ref_indices[0,0,:10].tolist()}")
        print(f"  npu  indices[0,0,:10]: {npu_indices[0,0,:10].tolist()}")

        self.assertGreaterEqual(
            overlap,
            self.MATCH_THRESHOLD,
            f"{label}: CPU-ref vs NPU 重合率 {overlap:.4f} < 阈值 {self.MATCH_THRESHOLD}",
        )

    def test_npu_bf16_matches_ref(self):
        """验证 NPU 非量化算子与 CPU 参考实现一致。"""
        data = gen_test_data(
            seq_lens_q=[256],
            seq_lens_k=[256],
            sparse_count=32,
            sparse_mode=3,
            seed=42,
        )
        self._run_ref_comparison(data, "npu_bf16_vs_ref")

    def test_npu_int8_matches_ref(self):
        """验证 NPU 量化算子与 CPU 量化参考实现一致。"""
        device = NPU_DEVICE
        data = gen_test_data(
            seq_lens_q=[256],
            seq_lens_k=[256],
            sparse_count=32,
            sparse_mode=3,
            seed=42,
        )
        sq = data["seq_lens_q"]
        sk = data["seq_lens_k"]
        sc = data["sparse_count"]
        sm = data["sparse_mode"]

        # CPU 量化参考
        q_cpu = data["query"]
        k_cpu = data["key"]
        w_cpu = data["weights"]
        k_tnd = expand_paged_key_to_tnd(
            k_cpu, data["block_table"], sk, data["block_size"]
        )
        q_int8_cpu, q_scale_cpu = per_token_head_symmetric_quantize(q_cpu)
        k_int8_cpu, k_scale_cpu = per_token_head_symmetric_quantize(k_tnd)
        ref_indices = ref_lightning_indexer_quant(
            q_int8_cpu, k_int8_cpu, q_scale_cpu, k_scale_cpu,
            w_cpu.to(torch.float16), sq, sk, sc, sm,
        )

        # NPU 量化算子
        q_npu = q_cpu.to(device)
        k_npu = k_cpu.to(device)
        w_npu = w_cpu.to(torch.float16).to(device)
        q_int8, q_scale = per_token_head_symmetric_quantize(q_npu)
        k_int8, k_scale = per_token_head_symmetric_quantize(k_npu)
        npu_indices = npu_quant_lightning_indexer_call(
            q_int8, k_int8, w_npu, q_scale, k_scale, sq, sk,
            data["block_table"], sc, sm,
        ).cpu()

        overlap = compute_overlap_ratio(ref_indices, npu_indices)
        print(f"\n[npu_int8_vs_ref] CPU-ref-quant vs NPU-int8 重合率: {overlap:.4f}")
        print(f"  ref  indices[0,0,:10]: {ref_indices[0,0,:10].tolist()}")
        print(f"  npu  indices[0,0,:10]: {npu_indices[0,0,:10].tolist()}")

        self.assertGreaterEqual(
            overlap,
            self.MATCH_THRESHOLD,
            f"CPU-ref-quant vs NPU-int8 重合率 {overlap:.4f} < 阈值 {self.MATCH_THRESHOLD}",
        )


class TestQuantizationLogic(unittest.TestCase):
    """纯 CPU 测试：验证量化逻辑本身的正确性（无需 NPU）。"""

    def test_quantize_dequantize_error(self):
        """验证 per-token-head 对称量化的反量化误差在可接受范围。"""
        torch.manual_seed(0)
        x = torch.randn(64, 128) * 3.0  # [T=64, D=128]
        q_int8, scale = per_token_head_symmetric_quantize(x)
        # 反量化
        x_deq = q_int8.to(torch.float32) * scale.unsqueeze(-1).to(torch.float32)
        max_err = (x.to(torch.float32) - x_deq).abs().max().item()
        print(f"\n[quantize] max abs error: {max_err:.6f}")
        self.assertLess(max_err, 0.05, f"量化误差 {max_err} 过大")

    def test_bf16_vs_quant_scores_close(self):
        """验证 bf16 路径与 int8 路径的相关性分数接近。"""
        torch.manual_seed(1)
        T, N1, D = 4, 64, 128
        S2 = 64
        q = torch.randn(T, N1, D, dtype=torch.bfloat16) * 2
        k = torch.randn(S2, 1, D, dtype=torch.bfloat16) * 2
        w = torch.randn(T, N1, dtype=torch.bfloat16) * 0.01

        # bf16 分数
        logits_bf16 = torch.einsum("qnd,skd->qns", q.float(), k.float())  # [T, N1, S2]
        relu_bf16 = torch.relu(logits_bf16)
        scores_bf16 = (w.float().unsqueeze(-1) * relu_bf16).sum(dim=1)  # [T, S2]

        # int8 分数
        q_int8, q_scale = per_token_head_symmetric_quantize(q)
        k_int8, k_scale = per_token_head_symmetric_quantize(k)
        int_logits = torch.einsum(
            "qnd,skd->qns", q_int8.float(), k_int8.float()
        )
        scale_logits = torch.einsum(
            "qn,sn->qns", q_scale.float(), k_scale.float()
        )
        logits_int8 = int_logits * scale_logits
        relu_int8 = torch.relu(logits_int8)
        scores_int8 = (w.float().unsqueeze(-1) * relu_int8).sum(dim=1)

        # 相对误差
        rel_err = (
            (scores_bf16 - scores_int8).abs()
            / (scores_bf16.abs() + 1e-6)
        ).mean().item()
        print(f"\n[scores] bf16 vs int8 平均相对误差: {rel_err:.6f}")
        self.assertLess(rel_err, 0.1, f"分数相对误差 {rel_err} 过大")

        # topk 重合率
        k_top = 32
        _, topk_bf16 = torch.topk(scores_bf16, k_top, dim=-1)
        _, topk_int8 = torch.topk(scores_int8, k_top, dim=-1)
        overlap = 0.0
        for t in range(T):
            s_bf16 = set(topk_bf16[t].tolist())
            s_int8 = set(topk_int8[t].tolist())
            overlap += len(s_bf16 & s_int8) / k_top
        overlap /= T
        print(f"[scores] bf16 vs int8 topk 重合率: {overlap:.4f}")
        self.assertGreater(overlap, 0.90, f"topk 重合率 {overlap} 过低")

    def test_fp8_roundtrip_error(self):
        """验证 BF16 -> FP8 -> BF16 的反量化误差（A5 环境 KV cache 存储精度）。"""
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("当前 PyTorch 版本不支持 float8_e4m3fn")
        torch.manual_seed(0)
        x = torch.randn(64, 128, dtype=torch.bfloat16) * 3.0  # [T=64, D=128]
        x_fp8 = bf16_to_fp8(x)
        x_back = fp8_to_bf16(x_fp8)
        max_err = (x.float() - x_back.float()).abs().max().item()
        rel_err = ((x.float() - x_back.float()).abs() / (x.float().abs() + 1e-6)).mean().item()
        print(f"\n[fp8_roundtrip] max abs error: {max_err:.6f}, mean rel error: {rel_err:.6f}")
        # E4M3FN 的精度较低，允许较大的误差
        self.assertLess(max_err, 0.5, f"FP8 roundtrip 误差 {max_err} 过大")

    def test_fp8_to_int8_pipeline_error(self):
        """验证 FP8 -> BF16 -> int8 完整 pipeline 的反量化误差。"""
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("当前 PyTorch 版本不支持 float8_e4m3fn")
        torch.manual_seed(1)
        x = torch.randn(64, 128, dtype=torch.bfloat16) * 3.0

        # 直接 int8 量化
        q_int8_direct, scale_direct = per_token_head_symmetric_quantize(x)
        x_deq_direct = q_int8_direct.float() * scale_direct.float().unsqueeze(-1)

        # FP8 -> BF16 -> int8 量化（A5 实际路径）
        x_fp8 = bf16_to_fp8(x)
        x_back = fp8_to_bf16(x_fp8)
        q_int8_fp8, scale_fp8 = per_token_head_symmetric_quantize(x_back)
        x_deq_fp8 = q_int8_fp8.float() * scale_fp8.float().unsqueeze(-1)

        # 对比：直接 int8 vs FP8->int8
        max_err = (x_deq_direct - x_deq_fp8).abs().max().item()
        # 对比：原始 BF16 vs FP8->int8
        max_err_to_orig = (x.float() - x_deq_fp8).abs().max().item()
        print(f"\n[fp8_to_int8] direct-int8 vs fp8-int8 max err: {max_err:.6f}")
        print(f"[fp8_to_int8] orig-bf16 vs fp8-int8 max err: {max_err_to_orig:.6f}")
        # FP8 引入额外误差，但应小于 0.5
        self.assertLess(max_err_to_orig, 0.5, f"FP8->int8 pipeline 误差 {max_err_to_orig} 过大")


if __name__ == "__main__":
    unittest.main(verbosity=2)
