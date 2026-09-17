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

_TRACE_DEPTH = 96

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


def recent_collectives() -> List[str]:
    """Last few collective enqueues, oldest first."""
    return list(_recent)


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
    logger.warning(
        "[coll-trace] installed pid=%d depth=%d patched=%s",
        os.getpid(),
        _TRACE_DEPTH,
        ",".join(patched),
    )