"""
Minimal parity test for build_tree_kernel_efficient across three branches in
eagle_utils.py#L204-252:

  1. torch.ops.npu.build_tree_kernel_efficient   (reference, A3 only, user runs)
  2. sglang.srt.hardware_backend.npu.kernels.build_tree_kernel_efficient_triton
       - NPU triton kernel (kernels.py:184)
       - Computes seq_len_prefix_sum internally via loop.
       - Uses vectorized loads with BLOCK_DRAFT.
  3. sglang.srt.speculative.eagle_utils.sgl_build_tree_kernel_triton
       - XPU triton kernel (spec_tree.py:19)
       - Takes seq_len_prefix_sum as external input.
       - Per-batch grid program.

Expected behavior:
  - Simple chain drafts: all three branches should agree.
  - Tree drafts with siblings: results should still agree (both triton kernels
    implement the same tree-building algorithm), but edge cases in index
    handling may cause divergence.

Usage:
  # Run only the two accessible triton branches (no A3 needed):
  python test_build_tree_kernel_parity.py

  # Also run the torch.ops.npu reference (A3 machine only):
  python test_build_tree_kernel_parity.py --with-reference

  # Use CPU tensors for debugging shape/dtype issues (kernels may fail):
  python test_build_tree_kernel_parity.py --device cpu

  # Only run specific cases:
  python test_build_tree_kernel_parity.py --cases chain_bs1 tree_with_siblings
"""
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

# Branch 2: NPU triton kernel wrapper
from sglang.srt.hardware_backend.npu.kernels import (
    build_tree_kernel_efficient_triton as npu_triton_build,
)

# Branch 3: XPU triton kernel wrapper (defined in eagle_utils.py)
from sglang.srt.speculative.eagle_utils import (
    sgl_build_tree_kernel_triton as xpu_triton_build,
)


# ---------------------------------------------------------------------------
# Input container
# ---------------------------------------------------------------------------
@dataclass
class BuildTreeInputs:
    """All tensors required by build_tree_kernel_efficient.

    Shapes (for batch_size B, draft_token_num N, depth D, topk K):
      parent_list:    (B, K*(D-1)+1)   int64   [col 0 = bonus, rest = candidates]
      selected_index: (B, N-1)         int64   top_scores_index per draft token
      verified_seq_len: (B,)           int64   seq_len per batch (no draft tokens)
      tree_mask:      (S*N + N*N*B,)  bool    FULL_MASK: True-init buffer
                       (N*B,)         bool    QLEN_ONLY: True-init buffer
      positions:      (B*N,)          int64   output: position per draft token
      retrieve_index:    (B, N)       int64   output, init -1
      retrieve_next_token:    (B, N)   int64   output, init -1
      retrieve_next_sibling: (B, N)   int64   output, init -1

    Scalar args:
      topk, depth (spec_steps), draft_token_num (num_verify_tokens),
      tree_mask_mode (0=FULL_MASK, 1=QLEN_ONLY, 2=QLEN_ONLY_BITPACKING)
    """

    parent_list: torch.Tensor
    selected_index: torch.Tensor
    verified_seq_len: torch.Tensor
    tree_mask: torch.Tensor
    positions: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    topk: int
    depth: int
    draft_token_num: int
    tree_mask_mode: int

    def clone_with_fresh_outputs(self) -> "BuildTreeInputs":
        """Clone inputs but with fresh -1-initialized retrieve buffers and
        zero positions, so each branch starts from the same initial state."""
        device = self.parent_list.device
        bs = self.parent_list.shape[0]
        N = self.draft_token_num
        # Fresh tree_mask buffer (re-init to True for FULL_MASK, matching the
        # caller's behavior in eagle_utils.py)
        if self.tree_mask_mode == 0:  # FULL_MASK
            tm = torch.full_like(self.tree_mask, True)
        elif self.tree_mask_mode == 1:  # QLEN_ONLY
            tm = torch.full_like(self.tree_mask, True)
        else:
            tm = torch.zeros_like(self.tree_mask)
        return BuildTreeInputs(
            parent_list=self.parent_list.clone(),
            selected_index=self.selected_index.clone(),
            verified_seq_len=self.verified_seq_len.clone(),
            tree_mask=tm,
            positions=torch.empty((bs * N,), dtype=torch.long, device=device),
            retrieve_index=torch.full((bs, N), -1, dtype=torch.long, device=device),
            retrieve_next_token=torch.full((bs, N), -1, dtype=torch.long, device=device),
            retrieve_next_sibling=torch.full((bs, N), -1, dtype=torch.long, device=device),
            topk=self.topk,
            depth=self.depth,
            draft_token_num=self.draft_token_num,
            tree_mask_mode=self.tree_mask_mode,
        )

    def describe(self) -> str:
        lines = []
        B, N = self.parent_list.shape[0], self.draft_token_num
        lines.append(
            f"  bs={B}, draft_token_num={N}, topk={self.topk}, "
            f"depth={self.depth}, tree_mask_mode={self.tree_mask_mode}"
        )
        lines.append(f"  verified_seq_len={self.verified_seq_len.tolist()}")
        for b in range(B):
            lines.append(f"  [batch {b}]")
            lines.append(f"    parent_list={self.parent_list[b].tolist()}")
            lines.append(f"    selected_index={self.selected_index[b].tolist()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Branch runners
# ---------------------------------------------------------------------------
def run_npu_triton(inp: BuildTreeInputs) -> Dict[str, torch.Tensor]:
    """Branch 2: NPU triton kernel."""
    x = inp.clone_with_fresh_outputs()
    npu_triton_build(
        parent_list=x.parent_list,
        selected_index=x.selected_index,
        verified_seq_len=x.verified_seq_len,
        tree_mask=x.tree_mask,
        positions=x.positions,
        retrieve_index=x.retrieve_index,
        retrieve_next_token=x.retrieve_next_token,
        retrieve_next_sibling=x.retrieve_next_sibling,
        topk=x.topk,
        depth=x.depth,
        draft_token_num=x.draft_token_num,
        tree_mask_mode=x.tree_mask_mode,
    )
    return {
        "positions": x.positions.clone(),
        "retrieve_index": x.retrieve_index.clone(),
        "retrieve_next_token": x.retrieve_next_token.clone(),
        "retrieve_next_sibling": x.retrieve_next_sibling.clone(),
        "tree_mask": x.tree_mask.clone(),
    }


def run_xpu_triton(inp: BuildTreeInputs) -> Dict[str, torch.Tensor]:
    """Branch 3: XPU triton kernel."""
    x = inp.clone_with_fresh_outputs()
    xpu_triton_build(
        parent_list=x.parent_list,
        selected_index=x.selected_index,
        verified_seq_len=x.verified_seq_len,
        tree_mask=x.tree_mask,
        positions=x.positions,
        retrieve_index=x.retrieve_index,
        retrieve_next_token=x.retrieve_next_token,
        retrieve_next_sibling=x.retrieve_next_sibling,
        topk=x.topk,
        depth=x.depth,
        draft_token_num=x.draft_token_num,
        tree_mask_mode=x.tree_mask_mode,
    )
    return {
        "positions": x.positions.clone(),
        "retrieve_index": x.retrieve_index.clone(),
        "retrieve_next_token": x.retrieve_next_token.clone(),
        "retrieve_next_sibling": x.retrieve_next_sibling.clone(),
        "tree_mask": x.tree_mask.clone(),
    }


def run_reference(inp: BuildTreeInputs) -> Dict[str, torch.Tensor]:
    """Branch 1: torch.ops.npu.build_tree_kernel_efficient (A3 only)."""
    x = inp.clone_with_fresh_outputs()
    torch.ops.npu.build_tree_kernel_efficient(
        x.parent_list,
        x.selected_index,
        x.verified_seq_len,
        x.tree_mask,
        x.positions,
        x.retrieve_index,
        x.retrieve_next_token,
        x.retrieve_next_sibling,
        x.topk,
        x.depth,
        x.draft_token_num,
        x.tree_mask_mode,
    )
    return {
        "positions": x.positions.clone(),
        "retrieve_index": x.retrieve_index.clone(),
        "retrieve_next_token": x.retrieve_next_token.clone(),
        "retrieve_next_sibling": x.retrieve_next_sibling.clone(),
        "tree_mask": x.tree_mask.clone(),
    }


REFERENCE_BRANCH = "reference"


# ---------------------------------------------------------------------------
# Helpers for building inputs
# ---------------------------------------------------------------------------
def _make_inputs(
    *,
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int = 0,
) -> BuildTreeInputs:
    """Build BuildTreeInputs with properly initialized output buffers.

    For FULL_MASK mode (tree_mask_mode=0), the tree_mask buffer size is:
        seq_lens_sum * N + N * N * B
    where seq_lens_sum = sum(verified_seq_len).
    """
    device = parent_list.device
    bs = parent_list.shape[0]
    N = draft_token_num
    seq_lens_sum = int(verified_seq_len.sum().item())

    if tree_mask_mode == 0:  # FULL_MASK
        tm_size = seq_lens_sum * N + N * N * bs
        tree_mask = torch.full((tm_size,), True, dtype=torch.bool, device=device)
    elif tree_mask_mode == 1:  # QLEN_ONLY
        tree_mask = torch.full((N * bs * N,), True, dtype=torch.bool, device=device)
    else:  # QLEN_ONLY_BITPACKING
        packed_dtype = torch.uint8
        tree_mask = torch.zeros((N * bs,), dtype=packed_dtype, device=device)

    return BuildTreeInputs(
        parent_list=parent_list.to(torch.long),
        selected_index=selected_index.to(torch.long),
        verified_seq_len=verified_seq_len.to(torch.long),
        tree_mask=tree_mask,
        positions=torch.empty((bs * N,), dtype=torch.long, device=device),
        retrieve_index=torch.full((bs, N), -1, dtype=torch.long, device=device),
        retrieve_next_token=torch.full((bs, N), -1, dtype=torch.long, device=device),
        retrieve_next_sibling=torch.full((bs, N), -1, dtype=torch.long, device=device),
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=tree_mask_mode,
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def case_chain_bs1(device: str) -> BuildTreeInputs:
    """Single chain: 0->1->2->3. topk=1, depth=3, draft_token_num=4.

    parent_list stride = topk*(depth-1)+1 = 1*2+1 = 3.
    selected_index shape = (1, 3).

    Tree structure:
      pos0 (bonus) -> pos1 -> pos2 -> pos3

    All three branches should produce:
      retrieve_index = [[0, 1, 2, 3]]
      retrieve_next_token = [[1, 2, 3, -1]]
      retrieve_next_sibling = [[-1, -1, -1, -1]]
      positions = [[seq_len, seq_len+1, seq_len+2, seq_len+3]]
    """
    return _make_inputs(
        # parent_list[0, 0] = bonus token (value irrelevant for tree structure)
        # parent_list[0, 1] = top_scores_index value of draft 1 (= 0) for parent lookup
        # parent_list[0, 2] = top_scores_index value of draft 2 (= 1) for parent lookup
        parent_list=torch.tensor([[100, 0, 1]], device=device),
        # top_scores_index[0, i] = i (chain: each draft's parent_tb_idx = i // topk = i)
        selected_index=torch.tensor([[0, 1, 2]], device=device),
        verified_seq_len=torch.tensor([7], device=device),
        topk=1,
        depth=3,
        draft_token_num=4,
    )


def case_tree_with_siblings(device: str) -> BuildTreeInputs:
    """Tree with siblings: root has 2 children, child 1 has 1 child.

    topk=2, depth=2, draft_token_num=4.
    parent_list stride = 2*1+1 = 3.
    selected_index shape = (1, 3).

    Tree structure:
      pos0 (bonus)
       |- pos1 (depth-1 child)
       |   |- pos3 (depth-2 child of pos1)
       |- pos2 (depth-1 child, sibling of pos1)

    top_scores_index layout (topk=2):
      values [0, 1] -> depth-1 children (parent is root)
      values [2, 3] -> depth-2 children (parent is a depth-1 token)

    All three branches should produce:
      retrieve_index = [[0, 1, 2, 3]]
      retrieve_next_token = [[1, 3, -1, -1]]
      retrieve_next_sibling = [[-1, 2, -1, -1]]
      positions = [[seq_len, seq_len+1, seq_len+1, seq_len+2]]
    """
    return _make_inputs(
        # parent_list[0, 0] = bonus token
        # parent_list[0, 1] = top_scores_index value of draft 1 (= 0) for depth-2 parent lookup
        # parent_list[0, 2] = unused (only 1 depth-2 candidate)
        parent_list=torch.tensor([[100, 0, -1]], device=device),
        # Draft 1: index 0 (depth-1, parent=root)
        # Draft 2: index 1 (depth-1, parent=root)
        # Draft 3: index 2 (depth-2, parent=draft 1 since parent_list[0, 1]=0 matches draft 1's index)
        selected_index=torch.tensor([[0, 1, 2]], device=device),
        verified_seq_len=torch.tensor([5], device=device),
        topk=2,
        depth=2,
        draft_token_num=4,
    )


def case_tree_deep_siblings(device: str) -> BuildTreeInputs:
    """Deeper tree with multiple sibling chains.

    topk=2, depth=3, draft_token_num=6.
    parent_list stride = 2*2+1 = 5.
    selected_index shape = (1, 5).

    Tree structure:
      pos0 (bonus)
       |- pos1 (depth-1)         |- pos3 (depth-1, sibling)
       |   |- pos5 (depth-2)     |   |- pos4 (depth-2)
       |- pos2 (depth-1, sibling of pos1)

    Wait, with topk=2, depth=3, draft_token_num = 1 + 2*2 = 5, not 6.
    Let me use draft_token_num=5.

    Actually, num_verify_tokens is independent. Let me use draft_token_num=5
    with topk=2, depth=3. parent_stride = 2*2+1 = 5.

    Tree:
      pos0 (bonus)
       |- pos1 (depth-1, idx 0)        |- pos2 (depth-1, idx 1, sibling)
       |   |- pos3 (depth-2, idx 2)   |   |- pos4 (depth-2, idx 3)

    All branches should produce:
      retrieve_index = [[0, 1, 2, 3, 4]]
      retrieve_next_token = [[1, 3, -1, 4, -1]]
      retrieve_next_sibling = [[-1, 2, -1, -1, -1]]
      positions = [[seq_len, seq_len+1, seq_len+1, seq_len+2, seq_len+2]]
    """
    return _make_inputs(
        # parent_list[0, 0] = bonus
        # parent_list[0, 1] = idx of draft 1 (= 0) for depth-2 parent lookup
        # parent_list[0, 2] = idx of draft 2 (= 1) for depth-2 parent lookup
        # parent_list[0, 3..4] = unused (no depth-3 candidates in this test)
        parent_list=torch.tensor([[100, 0, 1, -1, -1]], device=device),
        # Draft 1: idx 0 (depth-1, parent=root)
        # Draft 2: idx 1 (depth-1, parent=root, sibling of draft 1)
        # Draft 3: idx 2 (depth-2, parent=draft 1)
        # Draft 4: idx 3 (depth-2, parent=draft 2)
        selected_index=torch.tensor([[0, 1, 2, 3]], device=device),
        verified_seq_len=torch.tensor([10], device=device),
        topk=2,
        depth=3,
        draft_token_num=5,
    )


def case_batched_chain(device: str) -> BuildTreeInputs:
    """Batched chains: 2 batches, each a simple chain.

    topk=1, depth=2, draft_token_num=3.
    parent_list stride = 1*1+1 = 2.

    Tree per batch:
      pos0 -> pos1 -> pos2

    All branches should produce:
      retrieve_index = [[0,1,2], [3,4,5]]
      retrieve_next_token = [[1,2,-1], [4,5,-1]]
      retrieve_next_sibling = [[-1,-1,-1], [-1,-1,-1]]
      positions = [[seq_len0, seq_len0+1, seq_len0+2],
                   [seq_len1, seq_len1+1, seq_len1+2]]
    """
    return _make_inputs(
        parent_list=torch.tensor(
            [
                [100, 0],   # batch 0: bonus=100, draft 1's idx=0
                [200, 0],   # batch 1: bonus=200, draft 1's idx=0
            ],
            device=device,
        ),
        selected_index=torch.tensor(
            [
                [0, 1],  # batch 0: draft 1 idx=0, draft 2 idx=1
                [0, 1],  # batch 1: same structure
            ],
            device=device,
        ),
        verified_seq_len=torch.tensor([3, 5], device=device),
        topk=1,
        depth=2,
        draft_token_num=3,
    )


def case_batched_mixed(device: str) -> BuildTreeInputs:
    """Batched: batch 0 is a chain, batch 1 is a tree with siblings.

    topk=2, depth=2, draft_token_num=4.

    Batch 0 (chain):
      pos0 -> pos1 -> pos2 -> pos3
      (But with topk=2, depth=2, only depth-1 candidates exist.
       A chain of 4 needs depth=3. Let me adjust: chain uses only depth-1
       children, all parented to root, but that's not a chain...)

    Actually, with topk=2, depth=2, max draft_token_num = 1 + 2 = 3.
    For draft_token_num=4, we need depth=3 (1 + 2*1 = 3, still not 4) or
    topk=3 (1 + 3*1 = 4). Let me use topk=3, depth=2, draft_token_num=4.

    Hmm, this is getting complicated. Let me just use 2 separate test cases
    instead of mixing in a batch.
    """
    # Use topk=2, depth=2, draft_token_num=3 for both batches
    # Batch 0: chain pos0->pos1->pos2 (but depth=2 means only depth-1 children)
    # Actually with depth=2, all non-root tokens are at depth-1, so no chain.
    # Let me use depth=3, topk=1, draft_token_num=3 (chain of 3)
    # But then topk=1 means no siblings possible.
    #
    # Simplest batched mixed: topk=2, depth=2, draft_token_num=3
    # Batch 0: 2 depth-1 children (siblings), no depth-2
    # Batch 1: 2 depth-1 children (siblings), no depth-2
    # This is not really "mixed". Let me skip the mixed case and just do
    # a batched version of the siblings case.
    return _make_inputs(
        parent_list=torch.tensor(
            [
                [100, 0, -1],   # batch 0
                [200, 0, -1],   # batch 1
            ],
            device=device,
        ),
        selected_index=torch.tensor(
            [
                [0, 1, 2],  # batch 0: 2 depth-1 + 1 depth-2
                [0, 1, 2],  # batch 1: same
            ],
            device=device,
        ),
        verified_seq_len=torch.tensor([4, 8], device=device),
        topk=2,
        depth=2,
        draft_token_num=4,
    )


def case_single_token(device: str) -> BuildTreeInputs:
    """Edge case: only bonus token, no drafts.

    topk=1, depth=1, draft_token_num=1.
    parent_list stride = 1*0+1 = 1.
    selected_index shape = (1, 0) — empty.

    All branches should produce:
      retrieve_index = [[0]]
      retrieve_next_token = [[-1]]
      retrieve_next_sibling = [[-1]]
      positions = [[seq_len]]
    """
    return _make_inputs(
        parent_list=torch.tensor([[100]], device=device),
        selected_index=torch.zeros((1, 0), dtype=torch.long, device=device),
        verified_seq_len=torch.tensor([3], device=device),
        topk=1,
        depth=1,
        draft_token_num=1,
    )


def case_qlen_only_mode(device: str) -> BuildTreeInputs:
    """QLEN_ONLY tree_mask_mode: smaller tree_mask buffer.

    Same tree as case_tree_with_siblings but with tree_mask_mode=1.
    The retrieve_* and positions outputs should be identical to FULL_MASK;
    only the tree_mask buffer layout differs.
    """
    return _make_inputs(
        parent_list=torch.tensor([[100, 0, -1]], device=device),
        selected_index=torch.tensor([[0, 1, 2]], device=device),
        verified_seq_len=torch.tensor([5], device=device),
        topk=2,
        depth=2,
        draft_token_num=4,
        tree_mask_mode=1,  # QLEN_ONLY
    )


ALL_CASES: Dict[str, Callable[[str], BuildTreeInputs]] = {
    "chain_bs1": case_chain_bs1,
    "tree_with_siblings": case_tree_with_siblings,
    "tree_deep_siblings": case_tree_deep_siblings,
    "batched_chain": case_batched_chain,
    "batched_mixed": case_batched_mixed,
    "single_token": case_single_token,
    "qlen_only_mode": case_qlen_only_mode,
}


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------
def _fmt(t: torch.Tensor) -> str:
    if t.numel() == 0:
        return "[]"
    if t.dim() <= 2:
        return str(t.tolist())
    return str(t.flatten().tolist()[:32]) + ("..." if t.numel() > 32 else "")


def compare_and_report(
    case_name: str,
    inputs: BuildTreeInputs,
    results: Dict[str, Dict[str, torch.Tensor]],
    reference_name: Optional[str],
) -> None:
    """Print results and compare branches against the reference."""
    print(f"\n{'=' * 72}")
    print(f"Test case: {case_name}")
    print(f"{'=' * 72}")
    print("Inputs:")
    print(inputs.describe())

    print("\nOutputs:")
    keys = ["retrieve_index", "retrieve_next_token", "retrieve_next_sibling", "positions"]
    for branch, res in results.items():
        print(f"  [{branch}]")
        for k in keys:
            print(f"    {k:24s}: {_fmt(res[k])}")
        # tree_mask is large; only show summary
        tm = res["tree_mask"]
        print(f"    {'tree_mask':24s}: shape={tuple(tm.shape)}, "
              f"dtype={tm.dtype}, sum={int(tm.sum())}, numel={tm.numel()}")

    if reference_name and reference_name in results:
        ref = results[reference_name]
        print(f"\nComparison against reference ({reference_name}):")
        for branch, res in results.items():
            if branch == reference_name:
                continue
            matches = all(torch.equal(res[k], ref[k]) for k in keys)
            status = "MATCH" if matches else "DIFFER"
            print(f"  {branch:14s} vs {reference_name}: {status}")
            if not matches:
                for k in keys:
                    if not torch.equal(res[k], ref[k]):
                        print(f"    {k}:")
                        print(f"      {branch:14s}: {_fmt(res[k])}")
                        print(f"      {reference_name:14s}: {_fmt(ref[k])}")
            # Also compare tree_mask
            if not torch.equal(res["tree_mask"], ref["tree_mask"]):
                print(f"    tree_mask differs: "
                      f"sum {branch}={int(res['tree_mask'].sum())} vs "
                      f"sum {reference_name}={int(ref['tree_mask'].sum())}")
    else:
        # No reference; compare NPU triton vs XPU triton
        if "npu_triton" in results and "xpu_triton" in results:
            a, b = results["npu_triton"], results["xpu_triton"]
            matches = all(torch.equal(a[k], b[k]) for k in keys)
            status = "MATCH" if matches else "DIFFER"
            print(f"\nNPU triton vs XPU triton: {status}")
            if not matches:
                for k in keys:
                    if not torch.equal(a[k], b[k]):
                        print(f"  {k}:")
                        print(f"    npu_triton: {_fmt(a[k])}")
                        print(f"    xpu_triton: {_fmt(b[k])}")
            if not torch.equal(a["tree_mask"], b["tree_mask"]):
                print(f"  tree_mask differs: "
                      f"sum npu={int(a['tree_mask'].sum())} vs "
                      f"sum xpu={int(b['tree_mask'].sum())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="npu",
        choices=["npu", "cuda", "cpu"],
        help="Device to place tensors on (default: npu).",
    )
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Also run torch.ops.npu.build_tree_kernel_efficient (A3 only).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Subset of case names to run. Default: all.",
    )
    args = parser.parse_args()

    # Build branch runners
    runners: Dict[str, Callable[[BuildTreeInputs], Dict[str, torch.Tensor]]] = {
        "npu_triton": run_npu_triton,
        "xpu_triton": run_xpu_triton,
    }
    reference_name: Optional[str] = None
    if args.with_reference:
        runners[REFERENCE_BRANCH] = run_reference
        reference_name = REFERENCE_BRANCH

    # Select cases
    case_names = list(ALL_CASES.keys()) if args.cases is None else args.cases
    for name in case_names:
        if name not in ALL_CASES:
            print(f"Unknown case: {name}. Available: {list(ALL_CASES.keys())}")

    # Run each case
    for case_name in case_names:
        if case_name not in ALL_CASES:
            continue
        try:
            inputs = ALL_CASES[case_name](args.device)
        except Exception as e:
            print(f"\n[SKIP] {case_name}: failed to build inputs: {e}")
            continue

        results: Dict[str, Dict[str, torch.Tensor]] = {}
        for branch_name, runner in runners.items():
            try:
                results[branch_name] = runner(inputs)
            except Exception as e:
                print(f"\n[ERROR] {case_name} / {branch_name}: {type(e).__name__}: {e}")
                results[branch_name] = {
                    "positions": torch.empty(0, dtype=torch.long),
                    "retrieve_index": torch.empty(0, dtype=torch.long),
                    "retrieve_next_token": torch.empty(0, dtype=torch.long),
                    "retrieve_next_sibling": torch.empty(0, dtype=torch.long),
                    "tree_mask": torch.empty(0, dtype=torch.bool),
                }
        compare_and_report(case_name, inputs, results, reference_name)

    print(f"\n{'=' * 72}")
    print("Done.")


if __name__ == "__main__":
    main()
