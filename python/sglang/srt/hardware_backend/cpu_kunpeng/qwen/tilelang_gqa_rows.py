"""TileLang (kunpeng_pto) implementation of ``qwen_gqa_rows``.

Experimental replacement for the C++ row scheduler behind
``SGLANG_KUNPENG_GQA_ROWS_TILELANG=1``. The kernel is copied from the delivery in
``test/srt/cpu_kunpeng/test_qwen_ops_tilelang.py`` (``gqa_rows_kernel``).

The kernel is compiled per concrete shape, because tilelang bakes every tensor
shape (and the scale) into the generated code. Its ``extern "C" void call(...)``
entry is registered into ``common_ops`` by shape so ``qwen_gqa_rows_tilelang`` can
be replayed by the common graph through
``qwen_graph_gqa_rows_tilelang_dispatch``
(``sgl-kernel/csrc/cpu/cpu_kunpeng/qwen/qwen_tilelang_gqa_rows.cpp``).

Supported geometry is the Qwen3.5/3.8 dense full-attention shape after TP
partition: ``heads in {2, 6}``, one KV head, ``head_dim == 256`` and the paged KV
cache (``page_size == 64``) used by the 920F deployment. Anything else keeps the
reference C++ kernel.
"""

import contextlib
import ctypes
import fcntl
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import torch
from tqdm import tqdm

import tilelang
import tilelang.language as T

from .capabilities import NATIVE_DECODE_MAX_TOKENS, get_capabilities

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}

NEG_INF = -1e30

# One task per chunk of rows (the kernel's ``chunk`` default) and the KV page
# size the 920F deployment allocates (``--page-size 64``).
_CHUNK = 32
_PAGE_SIZE = 64

_bridge = None
_register_call = None
_registered = set()

# Set while precompile_batches() runs so a shape-by-shape log line is not
# emitted for the whole sweep; the progress bar carries that information.
_precompiling = False

# Startup precompilation (SGLANG_KUNPENG_TL_OP_PRECOMPILE=1): rank 0 of each node
# compiles every row count ahead of the first request. The pool size is only an
# upper bound, since one shape per worker is submitted.
_PRECOMPILE_JOBS = 608


def enabled() -> bool:
    return os.environ.get("SGLANG_KUNPENG_GQA_ROWS_TILELANG", "0") == "1"


def _parse_fallback_batches(value):
    """Parse ``"1"`` / ``"1,3,4"`` into ``{1, 3, 4}``; empty means none."""
    batches = set()
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            batch = int(item)
        except ValueError:
            batch = 0
        if batch < 1:
            raise ValueError(
                "SGLANG_KUNPENG_GQA_ROWS_TILELANG_FALLBACK_BATCH_SIZE must "
                f"be a comma-separated list of positive ints, got {value!r}"
            )
        batches.add(batch)
    return batches


# Row counts that always use the reference C++ kernel, even though the TileLang
# kernel is available and compiled.
_CONFIGURED_FALLBACK_BATCHES = _parse_fallback_batches(
    os.environ.get(
        "SGLANG_KUNPENG_GQA_ROWS_TILELANG_FALLBACK_BATCH_SIZE", ""
    )
)

# Row counts whose TileLang compilation failed. They are served by the
# reference C++ kernel instead of crashing the worker.
_fallback_batches: set[int] = set()

if enabled():
    # The validated kernel builds its matmuls through unigemm; set the switch
    # once so both PrimFunc construction and tilelang.compile see it.
    os.environ.setdefault("TL_UNIGEMM", "1")
    print(
        "[QwenGQA] SGLANG_KUNPENG_GQA_ROWS_TILELANG=1: GQA rows use the "
        "TileLang kernel.",
        flush=True,
    )


def _gqa_rows_kernel(rows, slots, heads, head_dim, cache_tokens, capacity,
                     scale, chunk=_CHUNK, page_size=_PAGE_SIZE):
    """One task per chunk of rows: dense QK per chunk, then per-row softmax/PV."""
    n_chunk = (rows + chunk - 1) // chunk
    tile = 1024
    n_tiles = (capacity + tile - 1) // tile
    sp2_len = n_tiles * tile
    z_blocks = (cache_tokens + tile - 1) // tile
    num_tasks = n_chunk

    @T.prim_func
    def main(
        Q: T.Tensor((rows, heads, head_dim), 'bfloat16'),
        K: T.Tensor((cache_tokens, 1, head_dim), 'bfloat16'),
        V: T.Tensor((cache_tokens, 1, head_dim), 'bfloat16'),
        Table: T.Tensor((slots, capacity), 'int32'),
        Requests: T.Tensor((rows,), 'int64'),
        Lengths: T.Tensor((rows,), 'int64'),
        O: T.Tensor((rows, heads, head_dim), 'bfloat16'),
    ):
        with T.Kernel(num_tasks, is_cpu=True) as tid:
            r0 = tid * chunk
            r_len = T.min(r0 + chunk, rows) - r0

            Qb = T.alloc_buffer([chunk * heads, head_dim], dtype='bfloat16', scope='shared')
            Dense = T.alloc_buffer([chunk * heads, cache_tokens], dtype='float', scope='shared')
            SSc = T.alloc_buffer([heads, sp2_len], dtype='float', scope='shared')
            PScPool = T.alloc_buffer([chunk * heads, cache_tokens], dtype='bfloat16', scope='shared')
            Acc = T.alloc_buffer([chunk * heads, head_dim], dtype='float', scope='shared')
            MB = T.alloc_buffer([2], dtype='float', scope='shared')

            # zero PScPool once per task: non-live probs must contribute 0
            # in the chunk-level PV GEMM.  Row 0 is zeroed with scalar stores
            # and broadcast to the other rows with 1024-wide sve copies.
            for j in T.serial(cache_tokens):
                PScPool[0, j] = 0.0
            for i in T.serial(chunk * heads - 1):
                for t in T.serial(z_blocks):
                    zlen = T.min(cache_tokens - t * tile, tile)
                    z_raw = T.sve.load(PScPool[0, t * tile], zlen,
                                       dtype='bfloat16')
                    T.sve.store(z_raw, PScPool[i + 1, t * tile])

            # dense = einsum("rhd,td->rht", q[begin:stop].float(), cache_k)
            for r in T.serial(r_len):
                for h in T.serial(heads):
                    q_raw = T.sve.load(Q[r0 + r, h, 0], head_dim, dtype='bfloat16')
                    T.sve.store(q_raw, Qb[r * heads + h, 0])
            block_q = T.tile_mapping(Qb[0, 0], chunk * heads, head_dim)
            block_k = T.tile_mapping(K[0, 0, 0], cache_tokens, head_dim)
            block_s = T.tile_mapping(Dense[0, 0], chunk * heads, cache_tokens)
            T.matmul(False, True, block_q, block_k, block_s, init=True)

            # for offset, row in enumerate(range(begin, stop)):
            for r in T.serial(r_len):
                row = r0 + r
                req = Requests[row].astype('int32')
                length = Lengths[row].astype('int32')
                n_row_blocks = (length + page_size - 1) // page_size

                # scores = dense[offset].index_select(-1, index) * scale:
                # the row's live entries are page-contiguous, so gather one
                # 64-fp32 page per table read (first element of the group)
                for h in T.serial(heads):
                    for blk in T.serial(n_row_blocks):
                        base = Table[req, blk * page_size]
                        sv = T.sve.load(Dense[r * heads + h, base], page_size,
                                        dtype='float')
                        T.sve.mul(sv, scale)
                        T.sve.store(sv, SSc[h, blk * page_size])
                    for c in T.serial(sp2_len - length):
                        SSc[h, length + c] = NEG_INF

                # probs = softmax(scores, dim=-1), fp32, normalized in place
                for h in T.serial(heads):
                    MB[0] = NEG_INF
                    for t in T.serial(n_tiles):
                        sv = T.sve.load(SSc[h, t * tile], tile, dtype='float')
                        tm = T.sve.reduce_max(sv)
                        MB[0] = T.max(MB[0], tm)
                    MB[1] = 0.0
                    for t in T.serial(n_tiles):
                        sv = T.sve.load(SSc[h, t * tile], tile, dtype='float')
                        T.sve.sub(sv, MB[0])
                        T.sve.exp(sv)
                        T.sve.store(sv, SSc[h, t * tile])
                        ts = T.sve.reduce_sum(sv)
                        MB[1] = MB[1] + ts
                    inv = 1.0 / MB[1]
                    for t in T.serial(n_tiles):
                        sv = T.sve.load(SSc[h, t * tile], tile, dtype='float')
                        T.sve.mul(sv, inv)
                        T.sve.store(sv, SSc[h, t * tile])

                # scatter the normalized probs pool-shaped
                for blk in T.serial(n_row_blocks):
                    base = Table[req, blk * page_size]
                    for h in T.serial(heads):
                        sv = T.sve.load(SSc[h, blk * page_size], page_size,
                                        dtype='float')
                        sv16 = T.sve.convert(sv, 'bfloat16')
                        T.sve.store(sv16, PScPool[r * heads + h, base])

            # output = probs @ cache_v: ONE chunk-level GEMM over the pool
            block_p = T.tile_mapping(PScPool[0, 0], chunk * heads, cache_tokens)
            block_v = T.tile_mapping(V[0, 0, 0], cache_tokens, head_dim)
            block_o = T.tile_mapping(Acc[0, 0], chunk * heads, head_dim)
            T.matmul(False, False, block_p, block_v, block_o, init=True)

            # .to(q.dtype) per row: Acc[r*heads+h] -> bf16 -> O[row, h]
            for r in T.serial(r_len):
                row = r0 + r
                for h in T.serial(heads):
                    av = T.sve.load(Acc[r * heads + h, 0], head_dim,
                                    dtype='float')
                    o16 = T.sve.convert(av, 'bfloat16')
                    T.sve.store(o16, O[row, h, 0])

    return main


def _disk_cache_has_entry(prim):
    """Whether the disk cache already holds this PrimFunc's built kernel."""
    try:
        from tilelang.cache.kernel_cache import (
            KERNEL_LIB_PATH,
            PARAMS_PATH,
            WRAPPED_KERNEL_PATH,
            KernelCache,
        )
        from tilelang.env import is_cache_enabled
        from tilelang.utils.target import determine_platform

        if not is_cache_enabled():
            return False
        cache = KernelCache()
        key = cache._generate_key(
            func=prim,
            out_idx=[6],
            workspace_idx=None,
            execution_backend="cython",
            args=(),
            target="pto",
            target_host=None,
            platform=determine_platform("auto"),
            pass_configs=_PASS_CONFIGS,
        )
        cache_path = os.path.join(cache.cache_dir, key)
        return all(
            os.path.exists(os.path.join(cache_path, name))
            for name in (WRAPPED_KERNEL_PATH, PARAMS_PATH, KERNEL_LIB_PATH)
        )
    except Exception:
        return False


@contextlib.contextmanager
def _key_compile_lock(key):
    """Serialize one shape's build across processes sharing the cache dir."""
    try:
        from tilelang.env import TILELANG_CACHE_DIR

        lock_dir = os.path.join(TILELANG_CACHE_DIR, "locks")
        os.makedirs(lock_dir, exist_ok=True)
        path = os.path.join(
            lock_dir, "gqa_rows_%s.lock" % "_".join(str(v) for v in key)
        )
        fp = open(path, "w")
    except Exception:
        yield
        return
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        finally:
            fp.close()


@lru_cache(maxsize=None)
def _compiled(rows, slots, heads, head_dim, cache_tokens, capacity, scale):
    key = (rows, slots, heads, head_dim, cache_tokens, capacity, scale)
    prim = _gqa_rows_kernel(rows, slots, heads, head_dim, cache_tokens,
                            capacity, scale)
    with _key_compile_lock(key):
        cache_hit = _disk_cache_has_entry(prim)
        t0 = time.perf_counter()
        kernel = tilelang.compile(prim, out_idx=[6], target='pto',
                                  pass_configs=_PASS_CONFIGS)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        # Keep the ctypes handle alive next to the kernel; replay calls its
        # raw `call` entry directly, without the cython wrapper.
        lib = getattr(kernel.adapter, "lib", None)
        if lib is None:
            lib = ctypes.CDLL(kernel.adapter.lib_generator.libpath)
        if not hasattr(lib, "call"):
            raise RuntimeError("TileLang GQA rows library has no 'call' entry")
    if not cache_hit and not _precompiling:
        print(
            "[QwenGQA] rows=%d slots=%d heads=%d cache=%d capacity=%d "
            "compiled in %.0f ms" % (
                rows, slots, heads, cache_tokens, capacity, elapsed_ms
            ),
            flush=True,
        )
    return kernel, lib, cache_hit, elapsed_ms


def _shape_key(q, k, v, table, requests, lengths, scale):
    return (
        int(q.shape[0]),
        int(table.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
        int(k.shape[0]),
        int(table.shape[1]),
        float(scale),
    )


def supported(q, k, v, table, requests, lengths, scale) -> bool:
    """Whether the delivered TileLang kernel matches this call's geometry."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or table.ndim != 2:
        return False
    rows, heads, head_dim = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    if not (1 <= rows <= NATIVE_DECODE_MAX_TOKENS and heads in (2, 6)
            and head_dim == 256):
        return False
    if tuple(k.shape) != tuple(v.shape) or int(k.shape[1]) != 1 or int(k.shape[2]) != head_dim:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if table.dtype != torch.int32:
        return False
    if requests.numel() != rows or lengths.numel() != rows:
        return False
    if requests.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (
        torch.int32, torch.int64
    ):
        return False
    try:
        return math.isfinite(float(scale))
    except (TypeError, ValueError):
        return False


def _check(q, k, v, table, requests, lengths):
    assert all(t.dtype == torch.bfloat16 for t in (q, k, v))
    assert table.dtype == torch.int32
    assert requests.dtype in (torch.int32, torch.int64)
    assert lengths.dtype in (torch.int32, torch.int64)
    assert q.shape[0] == requests.numel() == lengths.numel()


def prepare(q, k, v, table, requests, lengths, scale):
    """Compile this shape and publish its entry to the graph dispatcher."""
    key = _shape_key(q, k, v, table, requests, lengths, scale)
    kernel, lib, cache_hit, elapsed_ms = _compiled(*key)
    if key not in _registered:
        if _bridge is None:
            raise RuntimeError(
                "TileLang GQA rows graph dispatch is not registered; "
                "SGLANG_KUNPENG_GQA_ROWS_TILELANG requires a rebuilt sgl-kernel"
            )
        _register_call(*key, ctypes.cast(lib.call, ctypes.c_void_p))
        _registered.add(key)
    return kernel, cache_hit, elapsed_ms


def gqa_rows(q, k, v, table, requests, lengths, scale):
    """Eager entry: compile/register this shape and run the TileLang kernel."""
    _check(q, k, v, table, requests, lengths)
    kernel, _, _ = prepare(q, k, v, table, requests, lengths, scale)
    return kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        table.contiguous(),
        requests.to(torch.int64).contiguous(),
        lengths.to(torch.int64).contiguous(),
    )


def _shape(q, k, v, table, requests, lengths, scale):
    if not supported(q, k, v, table, requests, lengths, scale):
        raise ValueError("unsupported TileLang GQA rows geometry")
    return [(tuple(q.shape), q.dtype)]


def fallback_batches() -> set:
    """Row counts whose TileLang compile failed in this process."""
    return _fallback_batches


def configured_fallback_batches() -> set:
    """Row counts pinned to the C++ kernel by env configuration."""
    return _CONFIGURED_FALLBACK_BATCHES


def tilelang_disabled(rows) -> bool:
    """True when this row count must use the reference C++ kernel."""
    return rows in _CONFIGURED_FALLBACK_BATCHES or rows in _fallback_batches


def mark_failed(rows, exc, q, k, v, table, requests, lengths):
    """Log a failed shape and degrade it to the reference C++ kernel."""
    if rows in _fallback_batches:
        return
    _fallback_batches.add(rows)
    print(
        "[QwenGQA] rows=%d TileLang fallback to C++: %s: %s "
        "Q=%s K=%s V=%s table=%s requests=%s lengths=%s"
        % (
            rows, type(exc).__name__, exc,
            (tuple(q.shape), q.dtype),
            (tuple(k.shape), k.dtype),
            (tuple(v.shape), v.dtype),
            (tuple(table.shape), table.dtype),
            (tuple(requests.shape), requests.dtype),
            (tuple(lengths.shape), lengths.dtype),
        ),
        flush=True,
    )


def precompile_batches(*, slots, capacity, heads, head_dim, cache_tokens,
                       scale, max_rows, jobs=_PRECOMPILE_JOBS, skip=()):
    """Compile and register ``rows = 1..max_rows`` except ``skip``.

    Each shape is one thread pinned to a distinct logical CPU: a single compile
    is single threaded, so the speedup comes from compiling different shapes
    concurrently. Returns the sorted list of row counts that failed.
    """
    wanted = [r for r in range(1, max_rows + 1) if r not in set(skip)]
    if not wanted:
        return []
    dtype = torch.bfloat16
    # Only shapes and dtypes reach the compiler; contents are never read.
    q = torch.empty((max_rows, heads, head_dim), dtype=dtype)
    k = torch.empty((cache_tokens, 1, head_dim), dtype=dtype)
    v = torch.empty((cache_tokens, 1, head_dim), dtype=dtype)
    table = torch.zeros((slots, capacity), dtype=torch.int32)
    requests = torch.zeros(max_rows, dtype=torch.int64)
    lengths = torch.zeros(max_rows, dtype=torch.int64)

    total_cpus = os.cpu_count() or 1
    workers = max(1, min(int(jobs), len(wanted), total_cpus))
    started = time.perf_counter()
    failures = []
    bind_error = []

    def _compile_one(rows, cpu):
        try:
            os.sched_setaffinity(0, {cpu})
        except OSError as exc:
            if not bind_error:
                bind_error.append(exc)
                print(
                    "[QwenGQA] precompile cannot bind cpu %d: %s"
                    % (cpu, exc),
                    flush=True,
                )
        t0 = time.perf_counter()
        try:
            _, cache_hit, _ = prepare(
                q[:rows], k, v, table, requests[:rows], lengths[:rows], scale,
            )
        except Exception as exc:  # reported by mark_failed, degraded at runtime
            mark_failed(rows, exc, q[:rows], k, v, table, requests[:rows],
                        lengths[:rows])
            return rows, None, (time.perf_counter() - t0) * 1e3
        return rows, cache_hit, (time.perf_counter() - t0) * 1e3

    global _precompiling
    hits = 0
    compiles = 0
    _precompiling = True
    bar = tqdm(
        total=len(wanted),
        desc="[QwenGQA] TL gqa_rows",
        unit="shape",
        dynamic_ncols=True,
        mininterval=1.0,
        disable=False,
        file=sys.stdout,
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_compile_one, rows, (rows - 1) % total_cpus)
                for rows in wanted
            ]
            for future in as_completed(futures):
                rows, cache_hit, elapsed_ms = future.result()
                if cache_hit is None:
                    failures.append(rows)
                elif cache_hit:
                    hits += 1
                else:
                    compiles += 1
                bar.set_postfix(
                    hit=hits, compiled=compiles,
                    last="%.0fms" % elapsed_ms, refresh=False,
                )
                bar.update(1)
    finally:
        _precompiling = False
        bar.close()

    failures.sort()
    print(
        "[QwenGQA] precompile done: %d shapes (%d compiled, %d hit) in "
        "%.1fs, workers=%d, failed=%s"
        % (
            len(wanted), compiles, hits, time.perf_counter() - started,
            workers, failures,
        ),
        flush=True,
    )
    return failures


def _first_gqa_layer(model):
    """Return a full-attention layer, whose attributes describe the shape."""
    from sglang.srt.layers.radix_attention import RadixAttention

    for module in model.modules():
        attn = getattr(module, "attn", None)
        if isinstance(attn, RadixAttention) and hasattr(module, "num_heads"):
            return module
    return None


def precompile_gqa_rows(runner) -> None:
    """Compile and register this shape family for rows=1..M at startup.

    Called once per scheduler process after the memory pool exists. The
    TileLang cache lives on local disk, so within a node only the rank whose
    in-node slot is 0 compiles; every other rank waits on the TP-group barrier.
    Peers register a shape lazily on first use, which is a local cache hit.
    """
    from sglang.srt.hardware_backend.cpu_kunpeng.distributed import rank_layout

    pp_size = int(runner.server_args.pp_size)
    rin_in_node = rank_layout.compute_rin_in_node(
        int(getattr(runner, "pp_rank", 0)),
        int(getattr(runner, "tp_rank", 0)),
        pp_size,
    )
    verbose = rin_in_node == 0  # one reporter per node

    def skip(reason):
        if verbose:
            print(f"[QwenGQA] precompile skipped: {reason}", flush=True)

    if not enabled():
        skip("SGLANG_KUNPENG_GQA_ROWS_TILELANG != 1")
        return
    if pp_size != 1 or runner.server_args.dp_size != 1:
        skip(
            f"pp_size={pp_size} "
            f"dp_size={runner.server_args.dp_size} (only 1/1 supported)"
        )
        return
    if runner.server_args.disaggregation_mode == "prefill":
        # The row-wise GQA attention runs on the decode instance.
        skip("prefill instance")
        return
    if runner.is_draft_worker:
        return  # the target runner in the same process precompiles
    layer = _first_gqa_layer(runner.model)
    if layer is None:
        skip("no full-attention (GQA) layer found")
        return
    heads = int(layer.num_heads)
    head_dim = int(layer.head_dim)
    kv_heads = int(getattr(layer, "num_kv_heads", 1))
    scale = float(layer.scaling)
    if kv_heads != 1 or head_dim != 256 or heads not in (2, 6):
        skip(
            f"unsupported GQA geometry hq={heads} hkv={kv_heads} "
            f"hd={head_dim} (needs hq in 2/6, hkv=1, hd=256)"
        )
        return
    table = getattr(getattr(runner, "req_to_token_pool", None), "req_to_token", None)
    kv_pool = getattr(runner, "token_to_kv_pool", None)
    if table is None or kv_pool is None:
        skip("request-to-token / KV pool unavailable")
        return
    slots, capacity = int(table.shape[0]), int(table.shape[1])
    cache_tokens = int(kv_pool.get_key_buffer(int(layer.layer_id)).shape[0])
    if slots <= 0 or capacity <= 0 or cache_tokens <= 0:
        skip(
            f"slots={slots} capacity={capacity} cache_tokens={cache_tokens}"
        )
        return
    max_rows = min(
        NATIVE_DECODE_MAX_TOKENS,
        int(get_capabilities().decode_max_tokens),
        max(int(runner.max_running_requests), int(get_capabilities().decode_max_requests)),
    )
    if max_rows <= 0:
        skip(f"max_rows={max_rows}")
        return
    if verbose:
        print(
            "[QwenGQA] precompile start: tp_rank=%d rin_in_node=%d "
            "rows=1..%d heads=%d head_dim=%d cache_tokens=%d slots=%d "
            "capacity=%d scale=%.6f cpp_fallback=%s"
            % (
                int(getattr(runner, "tp_rank", 0)), rin_in_node, max_rows,
                heads, head_dim, cache_tokens, slots, capacity, scale,
                sorted(_CONFIGURED_FALLBACK_BATCHES),
            ),
            flush=True,
        )

    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    failure = None
    if rin_in_node == 0:
        try:
            failures = precompile_batches(
                slots=slots,
                capacity=capacity,
                heads=heads,
                head_dim=head_dim,
                cache_tokens=cache_tokens,
                scale=scale,
                max_rows=max_rows,
                skip=_CONFIGURED_FALLBACK_BATCHES,
            )
            if failures:
                failure = (
                    "TileLang gqa_rows precompile failed for rows="
                    f"{failures}"
                )
        except BaseException as exc:
            failure = f"TileLang gqa_rows precompile crashed: {exc!r}"
    # Barrier first: a failed node warmer must not strand its peers on a later
    # collective before the assertion takes the process down.
    dist.barrier(group=tp_group.cpu_group)
    assert failure is None, failure


def register_tilelang_gqa_rows(bridge, register_dispatch):
    """Register the graph op and its C++ replay dispatch into common_ops."""
    global _bridge, _register_call
    from sglang.srt.graph import register_op

    try:
        register_call = bridge.qwen_tilelang_register_gqa_rows
    except AttributeError as exc:
        raise RuntimeError(
            "sgl-kernel is missing the TileLang GQA rows graph dispatch; "
            "rebuild it from sources that include qwen_tilelang_gqa_rows.cpp"
        ) from exc
    register_call.argtypes = [ctypes.c_int64] * 6 + [
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    register_call.restype = None
    _register_call = register_call
    _bridge = bridge
    register_op("qwen_gqa_rows_tilelang", _shape, gqa_rows)
    register_dispatch(bridge, "qwen_gqa_rows_tilelang",
                      "qwen_graph_gqa_rows_tilelang_dispatch")
