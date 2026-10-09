from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import msgspec
import torch

from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_exec,
    get_spec,
)
from sglang.srt.utils import is_cuda, is_hip, is_npu

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def decide_needs_cpu_seq_lens(
    attn_backends: Sequence[AttentionBackend],
) -> bool:
    """Whether FutureMap must publish seq_lens_cpu / sum.

    OR over per-backend needs_cpu_seq_lens; force True under TBO (it reads the
    CPU mirror outside the backend layer to split the batch) or ngram (its
    USE_FULL_MASK verify path reads the host mirror regardless of backend).
    """
    # Local import: keep overlap_utils' module-level deps leaf-only so it stays
    # importable everywhere; spec_info pulls in the spec/schedule_batch graph.
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if get_exec().overlap.enable_two_batch_overlap:
        # FIXME: support TBO without seq lens cpu value
        return True
    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if algo.is_ngram():
        # ngram's USE_FULL_MASK verify path reads seq_lens_cpu per req to size
        # the tree mask, regardless of the attn backend (e.g. Triton opts out).
        return True
    # Skip unset slots (e.g. draft_extend_attn_backend on some spec configs);
    # missing flag -> True so undeclared backends stay on the legacy path.
    return any(
        getattr(b, "needs_cpu_seq_lens", True) for b in attn_backends if b is not None
    )


def decide_needs_confidence_relay() -> bool:
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    algo = SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm)
    if not algo.is_dspark():
        return False
    return read_ragged_verify_mode() is not RaggedVerifyMode.STATIC


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
_DEBUG_ASSERT = envs.SGLANG_IS_IN_CI.get()


@torch.compile(dynamic=True, disable=_is_npu)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Two sources:

    - Prefill: H2D copy from pinned CPU staging (prefill_input_ids_cpu).
    - Decode/spec_v2: gather from FutureMap (last iter's sampled token).
    """
    if batch.prefill_input_ids_cpu is not None:
        prefill_gpu = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            if batch.enable_overlap and not batch.spec_algorithm.is_none():
                future_map.resolve_mixed_spec_tails(batch)
            decode_gpu = future_map.output_tokens_buf[batch.mix_running_indices]
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    decode_gpu,
                    future_map.output_tokens_buf,
                    batch.mix_running_indices,
                )
            batch.input_ids = torch.cat([prefill_gpu, decode_gpu])
        else:
            batch.input_ids = prefill_gpu
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                batch.input_ids, future_map.output_tokens_buf, batch.req_pool_indices
            )

    # Only the overlap path relays spec extras through the future_map; the
    # synchronous (non-overlap) V2 path installs next_draft_input directly.
    if batch.enable_overlap and not batch.spec_algorithm.is_none():
        future_map._resolve_spec_extras(batch)


CONFIDENCE_RELAY_RING_LAG: int = 2
CONFIDENCE_RELAY_RING_DEPTH: int = CONFIDENCE_RELAY_RING_LAG + 1


class ResolvedConfidence(msgspec.Struct):
    confidence: torch.Tensor
    generation: torch.Tensor


@dataclass
class RelayPayload:
    """Per-iteration stash payload for the FutureMap bufs. Non-spec fills only
    `bonus_tokens`; which spec extras get relayed is decided by
    `FutureMap.spec_algo`, not by this payload's shape."""

    bonus_tokens: Optional[torch.Tensor]
    topk_p: Optional[torch.Tensor] = None
    topk_index: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    draft_probs: Optional[torch.Tensor] = None
    dsa_topk_indices: Optional[torch.Tensor] = None
    # ngram delays the draft extend (ngram update)
    accept_tokens: Optional[torch.Tensor] = None
    accept_lens: Optional[torch.Tensor] = None

    @classmethod
    def from_ngram(cls, draft_input: NgramVerifyInput) -> RelayPayload:
        return cls(
            bonus_tokens=None,
            accept_tokens=draft_input.accept_tokens.reshape(
                -1, draft_input.draft_token_num
            ),
            accept_lens=draft_input.accept_lens,
        )

    @classmethod
    def from_draft_input(cls, draft_input: EagleDraftInput) -> RelayPayload:
        return cls(
            bonus_tokens=draft_input.bonus_tokens,
            topk_p=draft_input.topk_p,
            topk_index=draft_input.topk_index,
            hidden_states=draft_input.hidden_states,
            draft_probs=getattr(draft_input, "draft_probs", None),
            dsa_topk_indices=draft_input.dsa_topk_indices,
        )


class ConfidenceRelay(msgspec.Struct):
    device: torch.device
    req_pool_size: int
    pool: Any
    confidence_buf: Optional[torch.Tensor] = None
    conf_ring: Optional[torch.Tensor] = None
    gen_ring: Optional[torch.Tensor] = None
    copy_done: Optional[list] = None
    ring_pos: int = 0
    initialized: bool = False

    def _lazy_init(self, confidence: torch.Tensor) -> None:
        self.initialized = True
        gamma = confidence.shape[-1]
        self.confidence_buf = torch.empty(
            (self.req_pool_size, gamma), dtype=torch.float32, device=self.device
        )
        if _is_cuda:
            depth = CONFIDENCE_RELAY_RING_DEPTH
            self.conf_ring = torch.empty(
                (depth, self.req_pool_size, gamma),
                dtype=torch.float32,
                pin_memory=True,
            )
            self.gen_ring = torch.zeros((depth, self.req_pool_size), dtype=torch.int64)
            self.copy_done = [
                torch.get_device_module(self.device).Event() for _ in range(depth)
            ]

    def scatter(self, indices: torch.Tensor, confidence: torch.Tensor) -> None:
        if not self.initialized:
            self._lazy_init(confidence)
        self.confidence_buf[indices] = confidence.to(self.confidence_buf.dtype)

    def issue_ring_copy(self, *, stream, publish_ready) -> None:
        if not self.initialized or stream is None or publish_ready is None:
            return
        slot = self.ring_pos % CONFIDENCE_RELAY_RING_DEPTH
        stream.wait_event(publish_ready)
        with torch.get_device_module(self.device).stream(stream):
            self.conf_ring[slot].copy_(self.confidence_buf, non_blocking=True)
            self.copy_done[slot].record()
        self.gen_ring[slot].copy_(self.pool.req_generation)
        self.ring_pos += 1

    def resolve(
        self, batch: ScheduleBatch, *, stream, publish_ready
    ) -> Optional[ResolvedConfidence]:
        if not self.initialized:
            return None
        draft_input = batch.spec_info
        if draft_input is None:
            return None
        fi = draft_input.future_indices
        if fi is None or fi.shape[0] == 0:
            return None

        if stream is None or publish_ready is None:
            idx = batch.req_pool_indices
            idx_cpu = batch.req_pool_indices_cpu
            return ResolvedConfidence(
                confidence=self.confidence_buf[idx].cpu(),
                generation=self.pool.req_generation[idx_cpu].clone(),
            )

        if self.ring_pos < CONFIDENCE_RELAY_RING_LAG:
            return None
        slot = (self.ring_pos - CONFIDENCE_RELAY_RING_LAG) % CONFIDENCE_RELAY_RING_DEPTH
        if not self.copy_done[slot].query():
            return None

        idx_cpu = batch.req_pool_indices_cpu
        return ResolvedConfidence(
            confidence=self.conf_ring[slot][idx_cpu],
            generation=self.gen_ring[slot][idx_cpu],
        )


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.
    """

    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        self.device = device
        self.spec_algo = spec_algo
        # Computed by decide_needs_cpu_seq_lens(); see that helper for the
        # full decision (per-backend flag + TBO / piecewise CG overrides).
        self.needs_cpu_seq_lens = needs_cpu_seq_lens
        self.needs_confidence_relay = needs_confidence_relay
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]
        # Kept for the mixed-tail late binding (reserved-slot gather).
        self.req_to_token = req_to_token_pool.req_to_token
        # Host-side per-slot request ownership token (bumped on alloc_rows);
        # used to detect slot reuse for seq_lens_cpu_last (below).
        self.req_generation = req_to_token_pool.req_generation
        # SGLANG_NPU_USE_SEQ_LENS_CPU_LAST: the GPU-only path
        # (needs_cpu_seq_lens=False) skips the per-round .cpu() D2H, leaving the
        # backend branches without a host width so they copy the FULL-width
        # block table every round -- expensive in the later steps of long
        # sequences. This mirror hands out a one-publish-round-stale snapshot
        # instead (never blocking the host), so those branches can truncate to
        # an analytic upper bound again; see _resolve_seq_lens_cpu_last.
        self.use_seq_lens_cpu_last = envs.SGLANG_NPU_USE_SEQ_LENS_CPU_LAST.get()

        if _DEBUG_ASSERT:
            # Poisoned init: every row must be written before its first gather.
            self.output_tokens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.full(
                (self.req_pool_size,), -1, dtype=torch.int64, device=self.device
            )
        else:
            self.output_tokens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
            self.new_seq_lens_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        # Pinned host copy of new_seq_lens_buf + private stream for fwd-prepare
        # D2H pulls (gated only on publish, off the schedule stream). CUDA-only:
        # recovers occupancy lost to the WAR barrier (also CUDA-only); other
        # platforms have no barrier and use the plain .cpu() bootstrap path.
        if _is_cuda or self.use_seq_lens_cpu_last:
            self.new_seq_lens_cpu_pinned = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, pin_memory=True
            )
            self.fwd_prepare_d2h_stream = torch.get_device_module(self.device).Stream()
        else:
            self.new_seq_lens_cpu_pinned = None
            self.fwd_prepare_d2h_stream = None
        if self.use_seq_lens_cpu_last:
            # Double-buffered host mirror of new_seq_lens_buf ("seq_lens_cpu_last").
            # Each resolve kicks a private-stream D2H of the current publish into
            # bufs[cur] and hands out bufs[prev], which holds the PREVIOUS
            # publish's snapshot (one publish round stale, content otherwise
            # exact). Hand-out never blocks the host: the snapshot's event has
            # fired by the next resolve, because the scheduler's own result
            # processing waits on the forward-before-last's copy_done, which is
            # ordered after that publish. Each buffer carries the req_generation
            # snapshot taken at kick time; a slot reallocated to another request
            # fails the generation check and forces the exact-sync fallback, so
            # a reused slot can never be served a previous request's lengths.
            # Consumers of the mirror must add an analytic slack for the one
            # publish-round lag (see AscendAttnBackend).
            device_module = torch.get_device_module(self.device)
            self.seq_lens_cpu_last_bufs = [
                torch.empty(
                    (self.req_pool_size,), dtype=torch.int64, pin_memory=True
                )
                for _ in range(2)
            ]
            self.seq_lens_cpu_last_events = [
                device_module.Event() for _ in range(2)
            ]
            self.seq_lens_cpu_last_gens = [
                torch.zeros(self.req_pool_size, dtype=torch.int64) for _ in range(2)
            ]
            self.seq_lens_cpu_last_kicked = [False, False]
            # Provenance per buffer: only decode-family rounds (decode / idle /
            # target_verify) may feed the mirror chain. Their per-round growth
            # is analytic (+1 per draft step, draft-width-bounded for verify)
            # and downstream consumers compensate for the one-round lag. An
            # extend/prefill round's publish has data-dependent growth, so its
            # snapshot is marked invalid and the next rounds take the exact
            # fallback until a decode snapshot flows through.
            self.seq_lens_cpu_last_dec = [True, True]
            self._seq_lens_cpu_last_cur = 0
            self._seq_lens_cpu_last_ids: Optional[torch.Tensor] = None
        else:
            self.seq_lens_cpu_last_bufs = None
            self.seq_lens_cpu_last_events = None
            self.seq_lens_cpu_last_gens = None
            self.seq_lens_cpu_last_kicked = None
            self.seq_lens_cpu_last_dec = None
            self._seq_lens_cpu_last_cur = 0
            self._seq_lens_cpu_last_ids = None
        self.need_topk = False
        self.need_hidden_states = False
        self.topk_p_buf = None
        self.topk_index_buf = None
        self.hidden_states_buf = None
        self.draft_probs_buf = None
        self.dsa_topk_indices_buf = None

        # ngram-only relay bufs
        self.accept_tokens_buf: Optional[torch.Tensor] = None
        self.accept_lens_buf: Optional[torch.Tensor] = None

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it
        # Debug consume-once state: armed by a recording publish, consumed by
        # resolve; arm/consume strictly alternate across all batch interleavings.
        self._publish_fresh = False

        self.confidence_relay = ConfidenceRelay(
            device=self.device,
            req_pool_size=self.req_pool_size,
            pool=req_to_token_pool,
        )

    def _maybe_init_forward_bufs(self, payload: RelayPayload) -> None:
        # Local import (see decide_needs_cpu_seq_lens): keep module-level deps leaf.
        from sglang.srt.speculative.spec_utils import spec_need_hidden_states

        # Prefill can omit spec extras; initialize each buffer when decode first
        # carries it instead of fixing the layout from the first payload.
        if not self.need_topk and (
            self.spec_algo.is_some()
            and self.spec_algo.need_topk()
            and payload.topk_p is not None
        ):
            self.need_topk = True
            topk_p0 = payload.topk_p[0]
            topk_index0 = payload.topk_index[0]
            self.topk_p_buf = torch.empty(
                (self.req_pool_size, *topk_p0.shape),
                dtype=topk_p0.dtype,
                device=self.device,
            )
            self.topk_index_buf = torch.empty(
                (self.req_pool_size, *topk_index0.shape),
                dtype=topk_index0.dtype,
                device=self.device,
            )

        if not self.need_hidden_states and (
            self.spec_algo.is_some()
            and spec_need_hidden_states()
            and payload.hidden_states is not None
        ):
            self.need_hidden_states = True
            hidden_states0 = payload.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        if self.draft_probs_buf is None and payload.draft_probs is not None:
            draft_probs0 = payload.draft_probs[0]
            self.draft_probs_buf = torch.empty(
                (self.req_pool_size, *draft_probs0.shape),
                dtype=draft_probs0.dtype,
                device=self.device,
            )

    def _maybe_init_dsa_topk_indices_buf(self, payload: RelayPayload) -> None:
        if self.dsa_topk_indices_buf is not None or payload.dsa_topk_indices is None:
            return
        seed0 = payload.dsa_topk_indices[0]
        self.dsa_topk_indices_buf = torch.empty(
            (self.req_pool_size, *seed0.shape),
            dtype=payload.dsa_topk_indices.dtype,
            device=self.device,
        )

    def _maybe_init_ngram_bufs(self, payload: RelayPayload) -> None:
        if self.accept_tokens_buf is not None:
            return
        # zeros, not empty: an unstashed row resolves to accept_len 0 (empty
        # splice at draft prep) instead of a garbage length.
        self.accept_tokens_buf = torch.zeros(
            (self.req_pool_size, payload.accept_tokens.shape[1]),
            dtype=payload.accept_tokens.dtype,
            device=self.device,
        )
        self.accept_lens_buf = torch.zeros(
            (self.req_pool_size,),
            dtype=payload.accept_lens.dtype,
            device=self.device,
        )

    def resolve_confidence_cpu(
        self, batch: ScheduleBatch
    ) -> Optional[ResolvedConfidence]:
        if not self.needs_confidence_relay:
            return None
        return self.confidence_relay.resolve(
            batch,
            stream=self.fwd_prepare_d2h_stream,
            publish_ready=self.publish_ready,
        )

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        if self.spec_algo.is_ngram():
            draft_input = batch.spec_info
            if draft_input is None or draft_input.future_indices is None:
                return
            indices = draft_input.future_indices
            if indices.shape[0] == 0:
                return
            draft_input.accept_tokens = self.accept_tokens_buf[indices].flatten()
            draft_input.accept_lens = self.accept_lens_buf[indices]
            return
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            return
        indices = draft_input.future_indices
        if indices.shape[0] == 0:
            return
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        if self.need_topk:
            hidden_states_buf = (
                self.hidden_states_buf if self.need_hidden_states else None
            )
            (
                draft_input.topk_p,
                draft_input.topk_index,
                bonus_tokens,
                hidden_states,
            ) = gather_spec_extras(
                indices,
                self.topk_p_buf,
                self.topk_index_buf,
                self.output_tokens_buf,
                hidden_states_buf,
            )
            draft_input.bonus_tokens = bonus_tokens
            if hidden_states is not None:
                draft_input.hidden_states = hidden_states
            if self.draft_probs_buf is not None and draft_input.draft_probs is not None:
                draft_input.draft_probs = self.draft_probs_buf[indices]
        else:
            draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if self.need_hidden_states and not self.need_topk:
            draft_input.hidden_states = self.hidden_states_buf[indices]
        if draft_input.future_dsa_topk_indices_available:
            assert self.dsa_topk_indices_buf is not None
            draft_input.dsa_topk_indices = self.dsa_topk_indices_buf[indices]
        else:
            draft_input.dsa_topk_indices = None
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )

    def stash_bonus_tokens(
        self, indices: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> None:
        """Write only output_tokens_buf rows; for relays carrying no draft
        extras (stash() would lazy-init the spec bufs from the payload)."""
        self.output_tokens_buf[indices] = bonus_tokens.to(self.output_tokens_buf.dtype)

    def resolve_mixed_spec_tails(self, batch: ScheduleBatch) -> None:
        """Late-bind a spec mixed batch's decode tails (overlap): schedule-time
        lengths lag the in-flight step's accept count, so rebuild the tail rows
        from the published committed lengths behind the publish fence."""
        idx = batch.mix_running_indices
        n = int(idx.shape[0])
        if n == 0:
            return
        if self.publish_ready is not None:
            if _is_hip:
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        fresh = self.new_seq_lens_buf[idx]
        seq_lens = batch.seq_lens.clone()
        seq_lens[-n:] = fresh + 1
        batch.seq_lens = seq_lens
        out_cache_loc = batch.out_cache_loc.clone()
        out_cache_loc[-n:] = self.req_to_token[idx.long(), fresh.long()].to(
            out_cache_loc.dtype
        )
        batch.out_cache_loc = out_cache_loc

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            fresh_cpu = fresh.cpu()  # bootstrap / non-CUDA
        else:
            self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
            with torch.get_device_module(self.device).stream(
                self.fwd_prepare_d2h_stream
            ):
                self.new_seq_lens_cpu_pinned.copy_(
                    self.new_seq_lens_buf, non_blocking=True
                )
            self.fwd_prepare_d2h_stream.synchronize()
            fresh_cpu = self.new_seq_lens_cpu_pinned[batch.mix_running_indices_cpu]
        if batch.seq_lens_cpu is not None:
            seq_lens_cpu = batch.seq_lens_cpu.clone()
            seq_lens_cpu[-n:] = fresh_cpu + 1
            batch.seq_lens_cpu = seq_lens_cpu
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        batch.prefix_lens = batch.prefix_lens[:-n] + [
            int(x) for x in fresh_cpu.tolist()
        ]

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # schedule). The CPU mirror is gated by needs_cpu_seq_lens; backends that
        # opt out take the GPU-only path below. A private D2H stream overlaps the copy.
        draft_input = batch.spec_info
        if draft_input is None:
            return

        fi = draft_input.future_indices
        if fi is None:
            return
        if self.publish_ready is not None:
            if _DEBUG_ASSERT:
                # Consume-once: every event wait must be re-armed by a fresh
                # forward publish; a stale consume means a publish went missing.
                assert self._publish_fresh, "resolve without a fresh forward publish"
                self._publish_fresh = False
            if _is_hip:
                # Temporary workaround: Event.wait() regresses TPOT on AMD MI355.
                self.publish_ready.synchronize()
            else:
                self.publish_ready.wait()
        batch.seq_lens = self.new_seq_lens_buf[fi]

        if self.use_seq_lens_cpu_last:
            self._resolve_seq_lens_cpu_last(batch)
            return

        if not self.needs_cpu_seq_lens:
            # GPU gather above is kept (SB.seq_lens must advance each verify);
            # skip the .cpu() D2H. Downstream takes the GPU-only path.
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            if _DEBUG_ASSERT:
                # Poison consumed rows: each row must be re-published/seeded
                # before the next resolve gathers it (safe here: the forward's
                # re-publish is fenced behind this stream via wait_stream).
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        if self.fwd_prepare_d2h_stream is None or self.publish_ready is None:
            batch.seq_lens_cpu = batch.seq_lens.cpu()  # bootstrap / non-CUDA
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)
            return

        # Mechanism: don't sync the schedule stream; gate a private stream on the
        # publish event and copy into the static pinned buffer.
        self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
        with torch.get_device_module(self.device).stream(self.fwd_prepare_d2h_stream):
            self.new_seq_lens_cpu_pinned.copy_(self.new_seq_lens_buf, non_blocking=True)
        self.fwd_prepare_d2h_stream.synchronize()

        # FIXME: fi == batch.req_pool_indices; unify future_indices and req_pool_indices.
        batch.seq_lens_cpu = self.new_seq_lens_cpu_pinned[batch.req_pool_indices_cpu]
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        if _DEBUG_ASSERT:
            # After the D2H copy completed (synchronize above), so the pinned
            # mirror is not poisoned.
            _assert_nonneg_and_invalidate(batch.seq_lens, self.new_seq_lens_buf, fi)

    def _resolve_seq_lens_cpu_last(self, batch: ScheduleBatch) -> None:
        """Serve the GPU-only path (SGLANG_NPU_USE_SEQ_LENS_CPU_LAST) without
        ever blocking the host on a D2H before run_batch.

        The snapshot rides ``batch.seq_lens_cpu_last``: in the GPU-only world
        (needs_cpu_seq_lens=False) ``seq_lens_cpu`` stays None by contract and
        exact-value consumers keep binding the device tensor. Only width-bound
        consumers (block-table truncation) read the snapshot and add an
        analytic slack for the one-publish lag.

        Each round:
        1. Consume: hand out the snapshot published by the PREVIOUS forward
           when it is complete, decode-family and the batch composition is
           unchanged; otherwise fall back to the exact `.cpu()` sync (first
           round of a batch, composition change, slot reuse, extend round in
           the chain, or a not-yet-complete copy).
        2. Save: kick an async private-stream D2H of this round's publish into
           the other pinned buffer; it becomes the next round's snapshot. The
           copy is gated on the current publish event, so it is ordered after
           this forward's seq_lens commit.

        Note: in this mode the CI-only consumed-rows poisoning is replaced by a
        plain device-side assert. Scattering -1 into new_seq_lens_buf here
        would race the in-flight async snapshot copies on the private stream.
        """
        cur = self._seq_lens_cpu_last_cur
        prev = cur ^ 1
        ids = batch.req_pool_indices_cpu
        last_ids = self._seq_lens_cpu_last_ids
        ready = (
            self.seq_lens_cpu_last_kicked[prev]
            and self.seq_lens_cpu_last_dec[prev]
            and self.seq_lens_cpu_last_events[prev].query()
            and last_ids is not None
            and last_ids.shape == ids.shape
            and bool(torch.equal(last_ids, ids))
            and bool(
                torch.equal(
                    self.seq_lens_cpu_last_gens[prev][ids], self.req_generation[ids]
                )
            )
        )
        if _DEBUG_ASSERT:
            # Gather validity check only; no consumed-rows poisoning (see doc).
            torch._assert_async((batch.seq_lens >= 0).all())
        if ready:
            # seq_lens_cpu_last := the one-publish-round-stale snapshot.
            batch.seq_lens_cpu_last = self.seq_lens_cpu_last_bufs[prev][ids]
        else:
            # Exact fallback: one blocking D2H, same as the bootstrap path.
            batch.seq_lens_cpu_last = batch.seq_lens.cpu()
        # Preserve the GPU-only contract: exact-value consumers read the
        # device seq_lens, not a host mirror.
        batch.seq_lens_cpu = None
        batch.seq_lens_sum = None
        # Save this round's publish as the next round's seq_lens_cpu_last.
        if self.publish_ready is not None:
            self.fwd_prepare_d2h_stream.wait_event(self.publish_ready)
            with torch.get_device_module(self.device).stream(
                self.fwd_prepare_d2h_stream
            ):
                self.seq_lens_cpu_last_bufs[cur].copy_(
                    self.new_seq_lens_buf, non_blocking=True
                )
            self.seq_lens_cpu_last_events[cur].record(self.fwd_prepare_d2h_stream)
            # Ownership snapshot for the snapshot being kicked; a later alloc
            # bumps req_generation and fails the consume check above.
            self.seq_lens_cpu_last_gens[cur].copy_(self.req_generation)
            self.seq_lens_cpu_last_kicked[cur] = True
            self.seq_lens_cpu_last_dec[cur] = batch.forward_mode.is_decode_or_idle() or (
                batch.forward_mode.is_target_verify()
            )
        self._seq_lens_cpu_last_cur = prev
        self._seq_lens_cpu_last_ids = ids.clone()

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            return  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        publish_confidence = self.needs_confidence_relay and confidence is not None
        if publish_confidence:
            self.confidence_relay.scatter(indices, confidence)
        # Only spec_v2 needs the event; it gates the seq_lens D2H on the private stream.
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()
            self._publish_fresh = True
        if publish_confidence:
            self.confidence_relay.issue_ring_copy(
                stream=self.fwd_prepare_d2h_stream,
                publish_ready=self.publish_ready,
            )

    def stash(self, future_indices: torch.Tensor, payload: RelayPayload) -> None:
        indices = future_indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            return
        if self.spec_algo.is_ngram():
            self._maybe_init_ngram_bufs(payload)
            self.accept_tokens_buf[indices] = payload.accept_tokens
            self.accept_lens_buf[indices] = payload.accept_lens
            return
        self._maybe_init_forward_bufs(payload)
        self._maybe_init_dsa_topk_indices_buf(payload)
        self.output_tokens_buf[indices] = payload.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )
        if self.need_topk:
            self.topk_p_buf[indices] = payload.topk_p.to(self.topk_p_buf.dtype)
            self.topk_index_buf[indices] = payload.topk_index.to(
                self.topk_index_buf.dtype
            )
        if self.need_hidden_states:
            self.hidden_states_buf[indices] = payload.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        if self.draft_probs_buf is not None and payload.draft_probs is not None:
            self.draft_probs_buf[indices] = payload.draft_probs
        if (
            self.dsa_topk_indices_buf is not None
            and payload.dsa_topk_indices is not None
        ):
            self.dsa_topk_indices_buf[indices] = payload.dsa_topk_indices.to(
                self.dsa_topk_indices_buf.dtype
            )
