from __future__ import annotations

import logging
from typing import NamedTuple


import numpy as np
import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, async_all_to_all
from sglang.srt.utils.common import get_bool_env_var, is_npu

logger = logging.getLogger(__name__)

if is_npu():
    import torch_npu


class FuseEPDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_state: torch.Tensor

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


class FuseEPCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_state: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


class NpuFuseEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.LOW_LATENCY,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        hidden_states, _ = self._get_buffer().fused_deep_moe(
            hidden_states,
            topk_idx=topk_output.topk_ids,
            topk_weights=topk_output.topk_weights,
            gmm1_permuted_weight=kwargs["gmm1_permuted_weight"],
            gmm1_permuted_weight_scale=kwargs["gmm1_permuted_weight_scale"],
            gmm2_weight=kwargs["gmm2_weight"],
            gmm2_weight_scale=kwargs["gmm2_weight_scale"],
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            num_experts=self.num_experts,
            fuse_mode=envs.SGLANG_NPU_FUSED_MOE_MODE.get(),
        )
        return FuseEPDispatchOutput(hidden_states)

    def combine(self, combine_input: CombineInput, **kwargs) -> torch.Tensor:
        pass

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class NpuDispatcherWithAllToAllOutput(NamedTuple):
    """AllToAllV dispatch output."""

    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    combine_metadata: MoEAllToAllCombineInput
    dynamic_scale: torch.Tensor | None = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_ALLTOALL


class MoEAllToAllCombineInput(NamedTuple):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    expanded_row_idx: torch.Tensor
    original_shape: torch.Size

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_ALLTOALL


class NpuDispatcherWithAllToAll(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.ALLTOALL,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode
        self.ep_rank = get_moe_ep_group().rank_in_group
        self.ep_size = get_moe_ep_group().world_size
        self.ep_group = get_moe_ep_group()

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs
    ) -> DispatchOutput:
        input_quant = get_bool_env_var("DEEP_NORMAL_MODE_USE_INT8_QUANT")
        if input_quant:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        top_k = topk_ids.shape[1]
        original_shape = hidden_states.shape
        topk_weights = topk_weights

        num_tokens = hidden_states.shape[:-1].numel()

        first_expert_idx = 0
        last_expert_idx = self.num_experts
        global_num_experts = self.num_experts

        sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                scale=pertoken_scale,
                active_num=num_tokens * top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=1 if input_quant else 0,
            )
        )

        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1

        return NpuDispatcherWithAllToAllOutput(
            hidden_states=sorted_hidden_states,
            dynamic_scale=pertoken_scale if input_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            combine_metadata=MoEAllToAllCombineInput(
                hidden_states=sorted_hidden_states,
                topk_weights=topk_weights,
                expanded_row_idx=expanded_row_idx,
                original_shape=original_shape,
            ),
        )

    def combine(self, combine_input) -> torch.Tensor:
        assert combine_input.original_shape is not None
        final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=combine_input.hidden_states,
            sorted_indices=torch.abs(combine_input.expanded_row_idx),
            probs=combine_input.topk_weights,
        )
        if len(combine_input.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(combine_input.original_shape)

        return final_hidden_states
