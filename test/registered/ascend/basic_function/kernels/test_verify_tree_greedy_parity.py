"""
Minimal parity test for verify_tree_greedy across three branches in
eagle_utils.py#L386-426:

  1. sgl_kernel_npu.sample.verify_tree_greedy   (reference, A5 only, user runs)
  2. sglang.srt.hardware_backend.npu.kernels.verify_tree_greedy_triton
       - NPU triton kernel (kernels.py:542)
       - Linear positional traversal: i = 1, 2, ... compares
         candidates[base+i] with target_predict[base+i-1].
       - Does NOT use retrieve_next_token / retrieve_next_sibling.
  3. sglang.srt.speculative.eagle_utils.verify_tree_greedy_triton
       - XPU triton kernel (spec_tree.py:177)
       - True tree traversal using retrieve_next_token (children) and
         retrieve_next_sibling (siblings).
       - Compares candidates[cur_index] with target_predict[last_accept_retrieve_idx].

Expected behavior:
  - Linear chain drafts: all three branches should agree.
  - Tree drafts with siblings: NPU triton (#2) will likely differ from the
    reference (#1) and XPU triton (#3), because #2 ignores sibling traversal.

Usage:
  # Run only the two accessible triton branches (no A5 needed):
  python test_verify_tree_greedy_parity.py

  # Also run the sgl_kernel_npu reference (A5 machine only):
  python test_verify_tree_greedy_parity.py --with-reference

  # Use CPU tensors for debugging shape/dtype issues (kernels may fail):
  python test_verify_tree_greedy_parity.py --device cpu
"""
import argparse
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

# Branch 2: NPU triton kernel wrapper
from sglang.srt.hardware_backend.npu.kernels import (
    verify_tree_greedy_triton as npu_triton_verify,
)

# Branch 3: XPU triton kernel wrapper (local def in eagle_utils.py, wraps
# verify_tree_greedy_kernel_triton from spec_tree.py)
from sglang.srt.speculative.eagle_utils import (
    verify_tree_greedy_triton as xpu_triton_verify,
)


# ---------------------------------------------------------------------------
# Input container
# ---------------------------------------------------------------------------
@dataclass
class VerifyInputs:
    """All tensors required by verify_tree_greedy.

    Shapes (for batch_size B and num_draft_tokens N):
      predicts:          (B*N,)            int32   mutable output
      accept_index:      (B, N)            int32   mutable output, init -1
      accept_token_num:  (B,)              int32   mutable output, init 0
      candidates:        (B, N)            int32   draft tokens (col 0 = bonus)
      retrieve_index:    (B, N)            int64   flat idx of each tree node
      retrieve_next_token:    (B, N)       int64   first child idx (-1 = leaf)
      retrieve_next_sibling: (B, N)       int64   next sibling idx (-1 = none)
      target_predict:    (B, N)            int32   target model pred per node
    """

    predicts: torch.Tensor
    accept_index: torch.Tensor
    accept_token_num: torch.Tensor
    candidates: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    target_predict: torch.Tensor

    def clone(self) -> "VerifyInputs":
        return VerifyInputs(
            predicts=self.predicts.clone(),
            accept_index=self.accept_index.clone(),
            accept_token_num=self.accept_token_num.clone(),
            candidates=self.candidates.clone(),
            retrieve_index=self.retrieve_index.clone(),
            retrieve_next_token=self.retrieve_next_token.clone(),
            retrieve_next_sibling=self.retrieve_next_sibling.clone(),
            target_predict=self.target_predict.clone(),
        )

    def describe(self) -> str:
        """Human-readable description of the tree structure for debugging."""
        lines = []
        B, N = self.candidates.shape
        for b in range(B):
            lines.append(f"  [batch {b}] num_draft_tokens={N}")
            for i in range(N):
                lines.append(
                    f"    pos {i}: candidate={int(self.candidates[b, i])}, "
                    f"target_pred={int(self.target_predict[b, i])}, "
                    f"retrieve_idx={int(self.retrieve_index[b, i])}, "
                    f"next_token={int(self.retrieve_next_token[b, i])}, "
                    f"next_sibling={int(self.retrieve_next_sibling[b, i])}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Branch runners
# ---------------------------------------------------------------------------
def run_npu_triton(inp: VerifyInputs) -> Dict[str, torch.Tensor]:
    """Branch 2: NPU triton kernel."""
    x = inp.clone()
    npu_triton_verify(
        predicts=x.predicts,
        accept_index=x.accept_index,
        accept_token_num=x.accept_token_num,
        candidates=x.candidates,
        retrieve_index=x.retrieve_index,
        retrieve_next_token=x.retrieve_next_token,
        retrieve_next_sibling=x.retrieve_next_sibling,
        target_predict=x.target_predict,
    )
    return {
        "predicts": x.predicts.clone(),
        "accept_index": x.accept_index.clone(),
        "accept_token_num": x.accept_token_num.clone(),
    }


def run_xpu_triton(inp: VerifyInputs) -> Dict[str, torch.Tensor]:
    """Branch 3: XPU triton kernel (true tree traversal)."""
    x = inp.clone()
    xpu_triton_verify(
        predicts=x.predicts,
        accept_index=x.accept_index,
        accept_token_num=x.accept_token_num,
        candidates=x.candidates,
        retrieve_index=x.retrieve_index,
        retrieve_next_token=x.retrieve_next_token,
        retrieve_next_sibling=x.retrieve_next_sibling,
        target_predict=x.target_predict,
    )
    return {
        "predicts": x.predicts.clone(),
        "accept_index": x.accept_index.clone(),
        "accept_token_num": x.accept_token_num.clone(),
    }


def run_reference(inp: VerifyInputs) -> Dict[str, torch.Tensor]:
    """Branch 1: sgl_kernel_npu.sample.verify_tree_greedy (A5 only)."""
    from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

    x = inp.clone()
    # NOTE: sgl_kernel_npu uses `retrive_*` (sic) kwarg names.
    verify_tree_greedy(
        predicts=x.predicts,
        accept_index=x.accept_index,
        accept_token_num=x.accept_token_num,
        candidates=x.candidates,
        retrive_index=x.retrieve_index,
        retrive_next_token=x.retrieve_next_token,
        retrive_next_sibling=x.retrieve_next_sibling,
        target_predict=x.target_predict,
    )
    return {
        "predicts": x.predicts.clone(),
        "accept_index": x.accept_index.clone(),
        "accept_token_num": x.accept_token_num.clone(),
    }


BRANCHES: Dict[str, Callable[[VerifyInputs], Dict[str, torch.Tensor]]] = {
    "npu_triton": run_npu_triton,
    "xpu_triton": run_xpu_triton,
}
REFERENCE_BRANCH = "reference"


# ---------------------------------------------------------------------------
# Helpers for building tree structure
# ---------------------------------------------------------------------------
def _make_inputs(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
) -> VerifyInputs:
    """Build VerifyInputs with properly initialized mutable buffers."""
    B, N = candidates.shape
    device = candidates.device
    return VerifyInputs(
        predicts=torch.zeros(B * N, dtype=torch.int32, device=device),
        accept_index=torch.full((B, N), -1, dtype=torch.int32, device=device),
        accept_token_num=torch.zeros(B, dtype=torch.int32, device=device),
        candidates=candidates.to(torch.int32),
        target_predict=target_predict.to(torch.int32),
        retrieve_index=retrieve_index.to(torch.int64),
        retrieve_next_token=retrieve_next_token.to(torch.int64),
        retrieve_next_sibling=retrieve_next_sibling.to(torch.int64),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def case_linear_all_match(device: str) -> VerifyInputs:
    """Chain 0->1->2->3. Target matches every draft.

    Expected: accept_token_num=3, accept_index=[0,1,2,3],
              predicts[0..2]=target_pred[0..2], predicts[3]=target_pred[3].
    All three branches should agree.
    """
    return _make_inputs(
        candidates=torch.tensor([[100, 200, 300, 400]], device=device),
        target_predict=torch.tensor([[200, 300, 400, 999]], device=device),
        retrieve_index=torch.tensor([[0, 1, 2, 3]], device=device),
        retrieve_next_token=torch.tensor([[1, 2, 3, -1]], device=device),
        retrieve_next_sibling=torch.tensor([[-1, -1, -1, -1]], device=device),
    )


def case_linear_reject_first(device: str) -> VerifyInputs:
    """Chain 0->1->2->3. First draft rejected (target_token != draft_token).

    Expected: accept_token_num=0, accept_index=[0,-1,-1,-1],
              predicts[0]=target_pred[0] (final fallback).
    All three branches should agree.
    """
    return _make_inputs(
        candidates=torch.tensor([[100, 200, 300, 400]], device=device),
        target_predict=torch.tensor([[999, 999, 999, 999]], device=device),
        retrieve_index=torch.tensor([[0, 1, 2, 3]], device=device),
        retrieve_next_token=torch.tensor([[1, 2, 3, -1]], device=device),
        retrieve_next_sibling=torch.tensor([[-1, -1, -1, -1]], device=device),
    )


def case_linear_partial_accept(device: str) -> VerifyInputs:
    """Chain 0->1->2->3. Accept 1 token then reject.

    Expected: accept_token_num=1, accept_index=[0,1,-1,-1],
              predicts[0]=target_pred[0], predicts[1]=target_pred[1] (final).
    All three branches should agree.
    """
    return _make_inputs(
        candidates=torch.tensor([[100, 200, 300, 400]], device=device),
        target_predict=torch.tensor([[200, 999, 999, 999]], device=device),
        retrieve_index=torch.tensor([[0, 1, 2, 3]], device=device),
        retrieve_next_token=torch.tensor([[1, 2, 3, -1]], device=device),
        retrieve_next_sibling=torch.tensor([[-1, -1, -1, -1]], device=device),
    )


def case_tree_match_at_sibling(device: str) -> VerifyInputs:
    """Tree: root->(pos1, pos2), pos1->pos3. Target matches pos2 (sibling), not pos1.

      pos0 (root, candidate=100)
       |- pos1 (candidate=200)  -- sibling chain -> pos2
       |   |- pos3 (candidate=400)
       |- pos2 (candidate=300)  -- MATCHES target_pred[0]=300

    NPU triton (linear): only checks candidates[1]=200 vs target_pred[0]=300,
      rejects immediately, no acceptance.
    XPU triton (tree): tries pos1 (200!=300), follows sibling to pos2 (300==300),
      accepts pos2, then tries child of pos2 (-1), stops.

    Expected divergence: NPU triton gives accept_token_num=0, XPU triton gives
    accept_token_num=1, accept_index=[0, 2, -1, -1].
    """
    return _make_inputs(
        candidates=torch.tensor([[100, 200, 300, 400]], device=device),
        target_predict=torch.tensor([[300, 999, 999, 999]], device=device),
        retrieve_index=torch.tensor([[0, 1, 2, 3]], device=device),
        # root's first child = pos1; pos1's first child = pos3; pos2 is a leaf
        retrieve_next_token=torch.tensor([[1, 3, -1, -1]], device=device),
        # pos1's sibling = pos2; no other siblings
        retrieve_next_sibling=torch.tensor([[-1, 2, -1, -1]], device=device),
    )


def case_tree_deep_traversal(device: str) -> VerifyInputs:
    """Tree with sibling match at level 1, then chain match at level 2.

      pos0 (root, candidate=100)
       |- pos1 (candidate=200) -- sibling -> pos2
       |- pos2 (candidate=300) -- MATCHES target_pred[0]=300
          |- pos3 (candidate=400) -- MATCHES target_pred[2]=400

    NPU triton (linear): rejects at pos1 (200!=300), no acceptance.
    XPU triton (tree): accepts pos2 (level 1), then accepts pos3 (level 2).

    Expected divergence: NPU triton accept_token_num=0, XPU triton
    accept_token_num=2, accept_index=[0, 2, 3, -1].
    """
    return _make_inputs(
        candidates=torch.tensor([[100, 200, 300, 400]], device=device),
        # target_pred[0]=300 matches pos2; target_pred[2]=400 matches pos3
        target_predict=torch.tensor([[300, 999, 400, 999]], device=device),
        retrieve_index=torch.tensor([[0, 1, 2, 3]], device=device),
        # root->pos1; pos1->none; pos2->pos3; pos3->none
        retrieve_next_token=torch.tensor([[1, -1, 3, -1]], device=device),
        # pos1's sibling = pos2; no other siblings
        retrieve_next_sibling=torch.tensor([[-1, 2, -1, -1]], device=device),
    )


def case_batched_mixed(device: str) -> VerifyInputs:
    """Batch of 2: batch 0 is a linear chain (all match), batch 1 is a tree
    with sibling match. Tests that batching doesn't mix up states.

    Expected: batch 0 accepts 3 tokens; batch 1 accepts 1 token at sibling.
    NPU triton will match XPU for batch 0 but diverge for batch 1.
    """
    return _make_inputs(
        # batch 0: linear chain; batch 1: tree with siblings
        candidates=torch.tensor(
            [
                [100, 200, 300, 400],  # chain
                [500, 600, 700, 800],  # tree: pos1=600, pos2=700 (sibling)
            ],
            device=device,
        ),
        target_predict=torch.tensor(
            [
                [200, 300, 400, 999],  # chain: all match
                [700, 999, 999, 999],  # tree: target matches pos2
            ],
            device=device,
        ),
        retrieve_index=torch.tensor(
            [
                [0, 1, 2, 3],  # chain
                [4, 5, 6, 7],  # tree (flat indices)
            ],
            device=device,
        ),
        # batch 0: linear chain (pos i's child = pos i+1)
        # batch 1: root->pos1; pos1->pos3; pos2 is leaf; pos1's sibling=pos2
        retrieve_next_token=torch.tensor(
            [
                [1, 2, 3, -1],  # chain
                [5, 7, -1, -1],  # tree: root(batch1)->pos1, pos1->pos3
            ],
            device=device,
        ),
        retrieve_next_sibling=torch.tensor(
            [
                [-1, -1, -1, -1],  # chain: no siblings
                [-1, 6, -1, -1],  # tree: pos1's sibling = pos2
            ],
            device=device,
        ),
    )


ALL_CASES: Dict[str, Callable[[str], VerifyInputs]] = {
    "linear_all_match": case_linear_all_match,
    "linear_reject_first": case_linear_reject_first,
    "linear_partial_accept": case_linear_partial_accept,
    "tree_match_at_sibling": case_tree_match_at_sibling,
    "tree_deep_traversal": case_tree_deep_traversal,
    "batched_mixed": case_batched_mixed,
}


# ---------------------------------------------------------------------------
# Comparison & reporting
# ---------------------------------------------------------------------------
def _fmt(t: torch.Tensor) -> str:
    return str(t.tolist())


def compare_and_report(
    case_name: str,
    inputs: VerifyInputs,
    results: Dict[str, Dict[str, torch.Tensor]],
    reference_name: Optional[str],
) -> None:
    """Print results and compare branches against the reference."""
    print(f"\n{'=' * 72}")
    print(f"Test case: {case_name}")
    print(f"{'=' * 72}")
    print("Tree structure (inputs):")
    print(inputs.describe())

    print("\nOutputs:")
    for branch, res in results.items():
        print(f"  [{branch}]")
        print(f"    predicts:         {_fmt(res['predicts'])}")
        print(f"    accept_index:      {_fmt(res['accept_index'])}")
        print(f"    accept_token_num:  {_fmt(res['accept_token_num'])}")

    if reference_name and reference_name in results:
        ref = results[reference_name]
        print(f"\nComparison against reference ({reference_name}):")
        for branch, res in results.items():
            if branch == reference_name:
                continue
            matches = all(
                torch.equal(res[k], ref[k]) for k in ("predicts", "accept_index", "accept_token_num")
            )
            status = "MATCH" if matches else "DIFFER"
            print(f"  {branch:14s} vs {reference_name}: {status}")
            if not matches:
                for k in ("predicts", "accept_index", "accept_token_num"):
                    if not torch.equal(res[k], ref[k]):
                        print(f"    {k}:")
                        print(f"      {branch:14s}: {_fmt(res[k])}")
                        print(f"      {reference_name:14s}: {_fmt(ref[k])}")
    else:
        # No reference; compare NPU triton vs XPU triton
        if "npu_triton" in results and "xpu_triton" in results:
            a, b = results["npu_triton"], results["xpu_triton"]
            matches = all(
                torch.equal(a[k], b[k]) for k in ("predicts", "accept_index", "accept_token_num")
            )
            status = "MATCH" if matches else "DIFFER"
            print(f"\nNPU triton vs XPU triton: {status}")
            if not matches:
                for k in ("predicts", "accept_index", "accept_token_num"):
                    if not torch.equal(a[k], b[k]):
                        print(f"  {k}:")
                        print(f"    npu_triton: {_fmt(a[k])}")
                        print(f"    xpu_triton: {_fmt(b[k])}")


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
        help="Also run sgl_kernel_npu.sample.verify_tree_greedy (A5 only).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Subset of case names to run. Default: all.",
    )
    args = parser.parse_args()

    # Build branch runners
    runners: Dict[str, Callable[[VerifyInputs], Dict[str, torch.Tensor]]] = {
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
            continue

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
                    "predicts": torch.empty(0, dtype=torch.int32),
                    "accept_index": torch.empty(0, dtype=torch.int32),
                    "accept_token_num": torch.empty(0, dtype=torch.int32),
                }
        compare_and_report(case_name, inputs, results, reference_name)

    print(f"\n{'=' * 72}")
    print("Done.")


if __name__ == "__main__":
    main()
