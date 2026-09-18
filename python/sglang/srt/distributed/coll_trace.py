"""Host-side enqueue tracer for distributed collectives.

Debug aid for hangs whose stuck collective is invisible in a Python stack dump:
a sampling dump only shows which host call is *waiting*, never which collective
the peer failed to issue. This wraps the ``torch.distributed`` collectives and
records each enqueue with a per-process sequence number plus a
shape/dtype/device signature, so the tail of a rank's trace is the last
collective that rank issued. Comparing two ranks' tails pinpoints the missing or
mismatched collective, and the device in the signature tells a CPU (gloo)
collective from an accelerator (HCCL) one.

Records go to an in-memory ring, not the log: emitting one log line per
collective would slow the scheduler host down and could mask the very
run-ahead the hang depends on. The scheduler watchdog prints the tail from
recent_collectives() when it times out.

Gated by SGLANG_NPU_COLL_TRACE.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from collections import deque
from typing import Any, Callable, List, Optional, Tuple

import torch.distributed as dist

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Deep enough to keep a few whole steps: a step issues ~50 collectives, and the
# cross-stream EDGE entries have to survive alongside them.
_TRACE_DEPTH = 256

_installed = False
_seq = 0
_recent: deque = deque(maxlen=_TRACE_DEPTH)
# Signature -> issuing call site, computed once per unique signature so the hot
# path never pays for frame introspection.
_seen_sigs: dict = {}


def _caller() -> str:
    # _record <- wrapper <- the code that issued the collective.
    try:
        frame = sys._getframe(3)
        return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    except Exception:
        return "?"


def _rank_str() -> str:
    try:
        if dist.is_initialized():
            return str(dist.get_rank())
    except Exception:
        pass
    return "?"


def _sig(t: Any) -> str:
    try:
        return f"{tuple(t.shape)}/{str(t.dtype).replace('torch.', '')}/{t.device.type}"
    except Exception:
        return "?"


def _sig_list(ts: Any) -> str:
    try:
        return f"n={len(ts)} {_sig(ts[0])}"
    except Exception:
        return "?"


def _find_group(args: tuple, kwargs: dict) -> Any:
    group = kwargs.get("group")
    if group is not None:
        return group
    for arg in args:
        if isinstance(arg, dist.ProcessGroup):
            return arg
    return None


def _record(op: str, group: Any, in_sig: str, out_sig: Optional[str]) -> None:
    global _seq
    _seq += 1
    try:
        group_size = group.size()
    except Exception:
        group_size = "?"
    key = f"{op} {in_sig} -> {out_sig}"
    origin = _seen_sigs.get(key)
    if origin is None:
        origin = _caller()
        _seen_sigs[key] = origin
    _recent.append(
        f"seq={_seq} op={op} group_size={group_size} in={in_sig} out={out_sig}"
        f" @{origin}"
    )


_graph_buffers: Any = None


def register_graph_replay(buffers: Any) -> None:
    """Remember the buffers bound by the last replayed graph.

    The watchdog dump reads them back from the (still alive) device, so the two
    ranks' values can be compared at hang time without adding a device sync to
    the hot path.
    """
    global _graph_buffers
    _graph_buffers = buffers


def graph_replay_buffer_state() -> str:
    """Contents of the DP token-count buffers the captured graph reads.

    Capture writes a uniform [num_tokens] * dp_size into these, and the replayed
    graph derives its dp-gather / combine segment sizes from them. If the two
    ranks disagree here, the segments disagree and the coupled HCCL op can never
    complete -- the failure mode the DFlash replay path documents when it
    refreshes exactly these buffers before replaying.
    """
    if _graph_buffers is None:
        return "graph buffers: none registered"
    parts = []
    for name in (
        "global_num_tokens_gpu",
        "global_num_tokens_for_logprob_gpu",
        "global_num_token_non_padded",
        "num_token_non_padded",
    ):
        buf = getattr(_graph_buffers, name, None)
        if buf is None:
            continue
        try:
            parts.append(f"{name}={buf.flatten().tolist()}")
        except Exception as exc:
            parts.append(f"{name}=<{type(exc).__name__}>")
    return "graph buffers: " + " ".join(parts)


def recent_collectives() -> List[str]:
    """Last few collective enqueues, oldest first."""
    return list(_recent)


def record_pin_keepalive(
    current_ct: int, evicted_iter: Optional[int], forward_done: Any
) -> None:
    """Detect a batch_record_buf pin released while its forward is in flight.

    batch_record_buf is a 2-slot ring, so a batch's tensors are pinned for
    exactly two iterations. If the scheduler host runs further ahead than that
    -- which needs no per-step host<->device sync: with overlap the decode path
    takes no D2H at all, and the gloo scheduler barrier only synchronises the
    two hosts, not host against device -- the pin is dropped while the forward
    stream still reads those tensors and the allocator is free to recycle them
    under the in-flight graph replay. A False query here is that failure.
    """
    if not envs.SGLANG_NPU_COLL_TRACE.get():
        return
    if forward_done is None:
        return
    try:
        done = bool(forward_done.query())
    except Exception:
        return
    _recent.append(
        f"PIN ct={current_ct} evicted_iter={evicted_iter} forward_done={done}"
    )


def record_dp_geometry(tag: str, **fields: Any) -> None:
    """Record the DP communication geometry decided for one forward.

    Interleaved with the collective entries, so the watchdog dump shows per step
    which geometry each rank used. The CUDA-graph replay bucket and the eager
    MAX_LEN padding must agree: the DP gather/combine split sizes are derived
    from them, and all_gather_into_tensor / reduce_scatter require every rank to
    split identically.
    """
    if not envs.SGLANG_NPU_COLL_TRACE.get():
        return
    _recent.append(f"GEOM {tag} " + " ".join(f"{k}={v}" for k, v in fields.items()))


def _record_edge(kind: str, waiter: Any, waited: Any) -> None:
    """Record a cross-stream synchronisation edge.

    Every edge is logged by object id; the watchdog dump prints the ids of the
    scheduler's named streams so a dumped tail can be decoded into "which stream
    waited on which". With overlap on there are two streams and a hang is often
    one unsatisfiable edge rather than a missing collective.
    """
    if not envs.SGLANG_NPU_COLL_TRACE.get():
        return
    global _seq
    _seq += 1
    _recent.append(
        f"seq={_seq} EDGE {kind} waiter={id(waiter):#x} waited={id(waited):#x}"
    )


def _patch_stream_syncs(device_module: Any) -> bool:
    """Hook Stream.wait_stream / wait_event on every instance.

    Patching the class beats instrumenting ~8 call sites: it also covers edges
    added later, and the Python-level methods are only ever called from sglang's
    own overlap machinery (the allocator's cross-stream waits are C++-level).
    """
    cls = getattr(device_module, "Stream", None)
    if cls is None or getattr(cls, "_sglang_coll_trace_patched", False):
        return False
    orig_wait_stream = cls.wait_stream
    orig_wait_event = cls.wait_event

    def wait_stream(self, other):
        _record_edge("wait_stream", self, other)
        return orig_wait_stream(self, other)

    def wait_event(self, event):
        _record_edge("wait_event", self, event)
        return orig_wait_event(self, event)

    cls.wait_stream = wait_stream
    cls.wait_event = wait_event
    cls._sglang_coll_trace_patched = True
    return True


def _make_wrapper(name: str, fn: Callable, sig_builder: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            in_sig, out_sig = sig_builder(args, kwargs)
            _record(name, _find_group(args, kwargs), in_sig, out_sig)
        except Exception:
            # Never let diagnostics break the collective itself.
            pass
        return fn(*args, **kwargs)

    return wrapper


def _sig_all_gather_into_tensor(args, kwargs) -> Tuple[str, Optional[str]]:
    return _sig(args[1]), _sig(args[0])


def _sig_all_gather(args, kwargs) -> Tuple[str, Optional[str]]:
    return _sig(args[1]), _sig_list(args[0])


def _sig_all_reduce(args, kwargs) -> Tuple[str, Optional[str]]:
    return _sig(args[0]), None


def _sig_reduce_scatter(args, kwargs) -> Tuple[str, Optional[str]]:
    return _sig_list(args[1]), _sig(args[0])


def _sig_reduce_scatter_tensor(args, kwargs) -> Tuple[str, Optional[str]]:
    return _sig(args[1]), _sig(args[0])


def _sig_broadcast(args, kwargs) -> Tuple[str, Optional[str]]:
    return _sig(args[0]), None


def _sig_barrier(args, kwargs) -> Tuple[str, Optional[str]]:
    return "-", None


_PATCH_TARGETS = (
    ("all_gather_into_tensor", _sig_all_gather_into_tensor),
    ("all_gather", _sig_all_gather),
    ("all_reduce", _sig_all_reduce),
    ("reduce_scatter", _sig_reduce_scatter),
    ("reduce_scatter_tensor", _sig_reduce_scatter_tensor),
    ("broadcast", _sig_broadcast),
    ("barrier", _sig_barrier),
)


def install_coll_trace() -> None:
    """Patch the torch.distributed collectives once per process."""
    global _installed
    if _installed or not envs.SGLANG_NPU_COLL_TRACE.get():
        return
    _installed = True
    patched = []
    for name, sig_builder in _PATCH_TARGETS:
        orig = getattr(dist, name, None)
        if orig is None:
            continue
        setattr(dist, name, _make_wrapper(name, orig, sig_builder))
        patched.append(name)
    try:
        import torch

        if _patch_stream_syncs(torch.get_device_module()):
            patched.append("Stream.wait_stream/wait_event")
    except Exception as exc:
        logger.warning("[coll-trace] stream sync hook failed: %r", exc)
    logger.warning(
        "[coll-trace] installed pid=%d depth=%d patched=%s",
        os.getpid(),
        _TRACE_DEPTH,
        ",".join(patched),
    )