from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _get_num_vectorcore() -> int:
    try:
        import triton.backends.ascend.runtime  # noqa: F401
    except ImportError:
        pass

    try:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, num_vectorcore = get_device_properties()
        return int(num_vectorcore)
    except Exception:
        device = torch.npu.current_device()
        device_properties = triton.runtime.driver.active.utils.get_device_properties(
            device
        )
        num_vectorcore = int(device_properties.get("num_vectorcore", -1))
        if num_vectorcore <= 0:
            raise RuntimeError("Failed to detect Ascend vector core count.")
        return num_vectorcore


@lru_cache(maxsize=1)
def _get_npu_aicore_num() -> int:
    try:
        import triton.backends.ascend.runtime  # noqa: F401
    except ImportError:
        pass

    device = torch.npu.current_device()
    props = triton.runtime.driver.active.utils.get_device_properties(device)
    aicore_num = int(props.get("num_aicore", 0))
    if aicore_num <= 0:
        raise RuntimeError(f"Failed to query num_aicore from device properties: {props}")
    return aicore_num


NUM_CUBE_CORES = _get_npu_aicore_num()


@triton.jit(do_not_specialize=["batch_size", "topk", "parent_stride"])
def _build_tree_efficient_kernel(
    parent_list_ptr,
    selected_index_ptr,
    verified_seq_len_ptr,
    tree_mask_ptr,
    positions_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    batch_size,
    topk,
    parent_stride,
    TREE_MASK_MODE: tl.constexpr,
    DRAFT_TOKEN_NUM: tl.constexpr,
    BLOCK_DRAFT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    offsets = tl.arange(0, BLOCK_DRAFT)
    offsets_i64 = offsets.to(tl.int64)
    row_mask = offsets < DRAFT_TOKEN_NUM
    selected_stride = DRAFT_TOKEN_NUM - 1

    for bs in tl.range(pid, batch_size, num_programs):
        bs_i64 = bs.to(tl.int64)
        batch_token_base = bs_i64 * DRAFT_TOKEN_NUM
        batch_parent_base = bs_i64 * parent_stride
        batch_selected_base = bs_i64 * selected_stride
        seq_len = tl.load(verified_seq_len_ptr + bs_i64)

        seq_tree_idx = batch_token_base * DRAFT_TOKEN_NUM
        for prev_bs in tl.range(0, bs, 1):
            prev_bs_i64 = prev_bs.to(tl.int64)
            prev_seq_len = tl.load(verified_seq_len_ptr + prev_bs_i64)
            seq_tree_idx += prev_seq_len * DRAFT_TOKEN_NUM

        tl.store(
            retrieve_index_ptr + batch_token_base + offsets_i64,
            batch_token_base + offsets_i64,
            mask=row_mask,
        )
        tl.store(
            retrieve_next_token_ptr + batch_token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(
            retrieve_next_sibling_ptr + batch_token_base + offsets_i64,
            -1,
            mask=row_mask,
        )
        tl.store(positions_ptr + batch_token_base, seq_len)

        for i in range(DRAFT_TOKEN_NUM - 1, 0, -1):
            current_selected_offset = batch_selected_base + (i - 1)
            parent_tb_idx = tl.load(selected_index_ptr + current_selected_offset) // topk
            parent_position = 0

            if parent_tb_idx > 0:
                parent_token_idx = tl.load(
                    parent_list_ptr + batch_parent_base + parent_tb_idx
                )
                parent_position = DRAFT_TOKEN_NUM
                for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                    candidate_token_idx = tl.load(
                        selected_index_ptr + batch_selected_base + candidate_pos
                    )
                    if (
                        parent_position == DRAFT_TOKEN_NUM
                        and candidate_token_idx == parent_token_idx
                    ):
                        parent_position = candidate_pos + 1

            if parent_position != DRAFT_TOKEN_NUM:
                existing_next = tl.load(
                    retrieve_next_token_ptr + batch_token_base + parent_position
                )
                if existing_next == -1:
                    tl.store(
                        retrieve_next_token_ptr + batch_token_base + parent_position, i
                    )
                else:
                    tl.store(
                        retrieve_next_token_ptr + batch_token_base + parent_position, i
                    )
                    tl.store(
                        retrieve_next_sibling_ptr + batch_token_base + i, existing_next
                    )

        for tid in range(DRAFT_TOKEN_NUM):
            if TREE_MASK_MODE == 0:
                token_tree_idx = (
                    seq_tree_idx + (seq_len + DRAFT_TOKEN_NUM) * tid + seq_len
                )
            else:
                token_tree_idx = batch_token_base * DRAFT_TOKEN_NUM + batch_token_base

            tl.store(
                tree_mask_ptr + token_tree_idx + offsets_i64,
                offsets == 0,
                mask=row_mask,
            )

            if tid > 0:
                position = 0
                cur_position = tid - 1
                active = 1

                for _ in range(DRAFT_TOKEN_NUM):
                    if active == 1:
                        position += 1
                        tl.store(
                            tree_mask_ptr + token_tree_idx + 1 + cur_position,
                            True,
                        )
                        parent_tb_idx = (
                            tl.load(selected_index_ptr + batch_selected_base + cur_position)
                            // topk
                        )
                        if parent_tb_idx == 0:
                            active = 0
                        else:
                            token_idx = tl.load(
                                parent_list_ptr + batch_parent_base + parent_tb_idx
                            )
                            next_position = DRAFT_TOKEN_NUM - 1
                            for candidate_pos in range(DRAFT_TOKEN_NUM - 1):
                                candidate_token_idx = tl.load(
                                    selected_index_ptr
                                    + batch_selected_base
                                    + candidate_pos
                                )
                                if (
                                    next_position == DRAFT_TOKEN_NUM - 1
                                    and candidate_token_idx == token_idx
                                ):
                                    next_position = candidate_pos
                            cur_position = next_position

                tl.store(positions_ptr + batch_token_base + tid, seq_len + position)


def build_tree_kernel_efficient_triton(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    batch_size = int(verified_seq_len.numel())
    parent_stride = topk * (depth - 1) + 1
    num_cores = _get_num_vectorcore()
    block_draft = triton.next_power_of_2(draft_token_num)

    _build_tree_efficient_kernel[(num_cores,)](
        parent_list_ptr=parent_list.reshape(-1),
        selected_index_ptr=selected_index.reshape(-1),
        verified_seq_len_ptr=verified_seq_len.reshape(-1),
        tree_mask_ptr=tree_mask.reshape(-1),
        positions_ptr=positions.reshape(-1),
        retrieve_index_ptr=retrieve_index.reshape(-1),
        retrieve_next_token_ptr=retrieve_next_token.reshape(-1),
        retrieve_next_sibling_ptr=retrieve_next_sibling.reshape(-1),
        batch_size=batch_size,
        topk=topk,
        parent_stride=parent_stride,
        TREE_MASK_MODE=int(tree_mask_mode),
        DRAFT_TOKEN_NUM=draft_token_num,
        BLOCK_DRAFT=block_draft,
    )


RETRIEVE_FROM_POOL = 1
MAX_STEP = 6


@triton.jit(do_not_specialize=["batch_size", "pool_len"])
def _cache_location_assigns_kernel(
    req_pool_indices_ptr,
    token_pool_ptr,
    start_offset_ptr,
    end_offset_ptr,
    out_cache_loc_ptr,
    batch_size,
    pool_len,
    ASSIGN_MODE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BS_UPPER: tl.constexpr,
    MAX_STEP_CONST: tl.constexpr,
):
    pid = tl.program_id(0)
    for row_idx in tl.range(pid, batch_size, NUM_CORES):
        req_idx = tl.load(req_pool_indices_ptr + row_idx)
        kv_start = tl.load(start_offset_ptr + row_idx)
        kv_end = tl.load(end_offset_ptr + row_idx)
        step = kv_end - kv_start

        prefix_idx = tl.arange(0, BS_UPPER)
        prefix_start = tl.load(
            start_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0
        )
        prefix_end = tl.load(
            end_offset_ptr + prefix_idx, mask=prefix_idx < row_idx, other=0
        )
        cache_idx_start = tl.sum(prefix_end - prefix_start, axis=0)

        token_ptr = token_pool_ptr + req_idx * pool_len + kv_start
        cache_ptr = out_cache_loc_ptr + cache_idx_start
        elem = tl.arange(0, MAX_STEP_CONST)
        mask = elem < step

        if ASSIGN_MODE == 1:
            data = tl.load(token_ptr + elem, mask=mask, other=0)
            tl.store(cache_ptr + elem, data, mask=mask)


def cache_loc_update(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc_copy: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(req_pool_indices.shape[0])
    if batch_size == 0:
        return out_cache_loc_copy
    num_cores = _get_num_vectorcore()
    bs_upper = int(triton.next_power_of_2(batch_size))
    _cache_location_assigns_kernel[(num_cores,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc_copy,
        batch_size,
        int(req_to_token.shape[1]),
        ASSIGN_MODE=RETRIEVE_FROM_POOL,
        NUM_CORES=int(max(1, num_cores)),
        BS_UPPER=bs_upper,
        MAX_STEP_CONST=MAX_STEP,
    )
    return out_cache_loc_copy


@triton.jit
def verify_tree_greedy_kernel(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    accept_index_stride,
    num_draft_tokens: tl.constexpr,
):
    req_idx = tl.program_id(0)
    base = req_idx * num_draft_tokens

    last_accepted_idx = tl.load(retrieve_index + base).to(tl.int32)
    tl.store(accept_index + req_idx * accept_index_stride, last_accepted_idx)

    num_accepted = 0
    rejected = False

    for i in range(1, num_draft_tokens):
        if not rejected:
            draft_token = tl.load(candidates + base + i).to(tl.int32)
            target_token = tl.load(target_predict + base + i - 1).to(tl.int32)

            if draft_token == target_token:
                draft_idx = tl.load(retrieve_index + base + i).to(tl.int32)
                tl.store(predicts + last_accepted_idx, target_token)
                num_accepted += 1
                tl.store(
                    accept_index + req_idx * accept_index_stride + num_accepted,
                    draft_idx,
                )
                last_accepted_idx = draft_idx
            else:
                rejected = True

    final_pos = last_accepted_idx - base
    final_token = tl.load(target_predict + base + final_pos).to(tl.int32)

    tl.store(accept_token_num + req_idx, num_accepted)
    tl.store(predicts + last_accepted_idx, final_token)


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    bs, num_draft_tokens = candidates.shape

    verify_tree_greedy_kernel[(bs,)](
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        target_predict=target_predict,
        accept_index_stride=accept_index.shape[1],
        num_draft_tokens=num_draft_tokens,
    )


_GENERIC_CONFIGS = [
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 32}),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 128, "BLOCK_K": 64}),
]

_FAST_CONFIGS = [
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_B": 8, "BLOCK_N": 128, "BLOCK_K": 128}),
]

_B1_FAST_CONFIGS = [
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}),
]


@triton.autotune(configs=_GENERIC_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_generic_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_b_tiles = tl.cdiv(B, BLOCK_B)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_m_groups = tl.cdiv(M, GROUP_M)
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // mn_tiles
        rem = block_idx % mn_tiles
        b_tile_idx = rem // num_n_tiles
        n_tile_idx = rem % num_n_tiles

        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            if m_idx < M:
                acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

                for k0 in range(0, K, BLOCK_K):
                    offs_k = k0 + tl.arange(0, BLOCK_K)

                    a_ptrs = (
                        a_ptr
                        + offs_b[:, None] * stride_ab
                        + m_idx * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    w_ptrs = (
                        w_ptr
                        + m_idx * stride_wm
                        + offs_k[:, None] * stride_wk
                        + offs_n[None, :] * stride_wn
                    )

                    mask_a = (offs_b[:, None] < B) & (offs_k[None, :] < K)
                    mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

                    a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                    w = tl.load(w_ptrs, mask=mask_w, other=0.0)

                    acc += tl.dot(a, w)

                c_ptrs = (
                    c_ptr
                    + offs_b[:, None] * stride_cb
                    + m_idx * stride_cm
                    + offs_n[None, :] * stride_cn
                )
                mask_c = (offs_b[:, None] < B) & (offs_n[None, :] < N)
                out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
                tl.store(c_ptrs, out, mask=mask_c)


@triton.autotune(configs=_FAST_CONFIGS, key=["B", "M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    B,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_b_tiles = B // BLOCK_B
    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_b_tiles * num_n_tiles
    mn_tiles = num_b_tiles * num_n_tiles

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // mn_tiles
        rem = block_idx % mn_tiles
        b_tile_idx = rem // num_n_tiles
        n_tile_idx = rem % num_n_tiles

        offs_b = b_tile_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = n_tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)

                a_ptrs = (
                    a_ptr
                    + offs_b[:, None] * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w_ptrs = (
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + offs_n[None, :] * stride_wn
                )

                a = tl.load(a_ptrs)
                w = tl.load(w_ptrs)
                acc += tl.dot(a, w)

            c_ptrs = (
                c_ptr
                + offs_b[:, None] * stride_cb
                + m_idx * stride_cm
                + offs_n[None, :] * stride_cn
            )
            out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
            tl.store(c_ptrs, out)


@triton.autotune(configs=_B1_FAST_CONFIGS, key=["M", "K", "N"])
@triton.jit
def _batch_matmul_transpose_b1_fast_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    stride_ab,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_wn,
    stride_cb,
    stride_cm,
    stride_cn,
    M,
    K,
    N,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    num_n_tiles = N // BLOCK_N
    num_m_groups = M // GROUP_M
    num_blocks = num_m_groups * num_n_tiles

    offs_n = tl.arange(0, BLOCK_N)

    for block_idx in range(pid, num_blocks, num_core):
        m_group_idx = block_idx // num_n_tiles
        n_tile_idx = block_idx % num_n_tiles
        n_base = n_tile_idx * BLOCK_N
        m_base = m_group_idx * GROUP_M

        for g in range(0, GROUP_M):
            m_idx = m_base + g
            acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, K, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                a_ptrs = (
                    a_ptr
                    + 0 * stride_ab
                    + m_idx * stride_am
                    + offs_k[None, :] * stride_ak
                )
                w_ptrs = (
                    w_ptr
                    + m_idx * stride_wm
                    + offs_k[:, None] * stride_wk
                    + (n_base + offs_n)[None, :] * stride_wn
                )
                a = tl.load(a_ptrs)
                w = tl.load(w_ptrs)
                acc += tl.dot(a, w)

            c_ptrs = (
                c_ptr
                + 0 * stride_cb
                + m_idx * stride_cm
                + (n_base + offs_n)[None, :] * stride_cn
            )
            out = acc.to(tl.bfloat16) if IS_BF16 else acc.to(tl.float16)
            tl.store(c_ptrs, out)


def _to_nd_from_nz(tensor_b: torch.Tensor) -> torch.Tensor:
    if tensor_b.ndim != 4:
        raise ValueError("NZ format expects tensor_b with 4 dimensions [M, N/16, K, 16].")
    if tensor_b.shape[-1] != 16:
        raise ValueError("NZ format expects tensor_b.shape[-1] == 16.")
    m, n16, k, inner = tensor_b.shape
    n = n16 * inner
    return tensor_b.permute(0, 2, 1, 3).contiguous().view(m, k, n)


def batch_matmul_transpose(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    tensor_c: torch.Tensor,
    format_mode: str | None = None,
    quant_mode: str | None = None,
) -> None:
    del quant_mode

    mode = "ND" if format_mode is None else str(format_mode).upper()
    if mode != "ND" and mode != "NZ":
        mode = "ND"

    if mode == "NZ":
        tensor_b_nd = _to_nd_from_nz(tensor_b)
    else:
        tensor_b_nd = tensor_b

    b, m, k = tensor_a.shape
    n = tensor_b_nd.shape[2]

    a = tensor_a.contiguous()
    w = tensor_b_nd.contiguous()

    group_m = 1 if m < 16 else (2 if (m % 2 == 0) else 1)
    use_fast = (
        (b % 8 == 0) and (n % 128 == 0) and (k % 128 == 0) and (m % group_m == 0)
    )
    use_b1_fast = (
        (b == 1) and (n % 128 == 0) and (k % 64 == 0) and (m % group_m == 0)
    )

    grid = (NUM_CUBE_CORES,)
    if use_b1_fast:
        _batch_matmul_transpose_b1_fast_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
    elif use_fast:
        _batch_matmul_transpose_fast_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            b,
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
    else:
        _batch_matmul_transpose_generic_kernel[grid](
            a,
            w,
            tensor_c,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            tensor_c.stride(0),
            tensor_c.stride(1),
            tensor_c.stride(2),
            b,
            m,
            k,
            n,
            IS_BF16=(a.dtype == torch.bfloat16),
            GROUP_M=group_m,
        )
