"""Test for the pybind11 binding of init_cpu_threads_env.

This test loads the standalone pybind11 module built from
``sgl-kernel/csrc/cpu/numa_utils_pybind/`` and verifies that
``init_cpu_threads_env`` pins the Python process's threads to the requested
CPU set.

Why a standalone pybind11 module?
---------------------------------
The top-level ``sgl-kernel/CMakeLists.txt`` builds the CUDA variant of
``common_ops`` and its SOURCES list does *not* include
``csrc/cpu/numa_utils.cpp``. So in a GPU install
``torch.ops.sgl_kernel.init_cpu_threads_env`` is NOT registered. When CUDA is
unavailable locally we instead expose the function through a small pybind11
module (no torch op registry involved).

Build the module (run once):
    pip install pybind11
    apt-get install -y libnuma numactl   # or: conda install libnuma numactl
    bash sgl-kernel/csrc/cpu/numa_utils_pybind/build.sh
    # Produced artifact: sgl-kernel/csrc/cpu/numa_utils_pybind/build/_numa_utils.*.so

Run as a pytest case:
    pytest sgl-kernel/tests/test_cpu_threads_env.py -s

Or as a standalone script:
    python sgl-kernel/tests/test_cpu_threads_env.py
"""

import glob
import os
import re
import sys

import pytest
import torch

# Candidate locations of the pybind11 _numa_utils.so. Add your own path here
# (or set the NUMA_UTILS_SO env var) if you built it elsewhere.
_DEFAULT_CANDIDATES = [
    # 1. Explicit override via env var.
    os.environ.get("NUMA_UTILS_SO", ""),
    # 2. Default build dir of build.sh.
    os.path.join(
        os.path.dirname(__file__), "..", "csrc", "cpu", "numa_utils_pybind", "build", "_numa_utils*.so"
    ),
]

_NUMA_UTILS_MODULE = None


def _load_module():
    """Import (or dlopen) the pybind11 _numa_utils module and cache it."""
    global _NUMA_UTILS_MODULE
    if _NUMA_UTILS_MODULE is not None:
        return _NUMA_UTILS_MODULE

    # Fast path: already importable (e.g. installed on PYTHONPATH).
    try:
        import _numa_utils  # noqa: F401

        _NUMA_UTILS_MODULE = _numa_utils
        return _NUMA_UTILS_MODULE
    except ImportError:
        pass

    # Otherwise, locate the .so via candidate patterns and load it.
    import importlib.util

    for pattern in _DEFAULT_CANDIDATES:
        if not pattern:
            continue
        for so_path in sorted(glob.glob(pattern)):
            try:
                spec = importlib.util.spec_from_file_location("_numa_utils", so_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                print(f"[cpu_threads_env] loaded pybind11 module from: {so_path}")
                _NUMA_UTILS_MODULE = mod
                return mod
            except Exception as e:
                print(f"[cpu_threads_env] failed to load {so_path}: {e}")

    return None


def _parse_cpu_ids(cpu_ids_str: str) -> set:
    """Parse a numactl-style cpu string (e.g. '0-3,7') into a set of ints."""
    result = set()
    for part in cpu_ids_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def _pick_cpu_ids(available: set, count: int = 4) -> str:
    sorted_cpus = sorted(available)
    chosen = sorted_cpus[:count]
    return ",".join(str(c) for c in chosen)


def test_init_cpu_threads_env_binding():
    mod = _load_module()
    if mod is None:
        pytest.skip(
            "_numa_utils pybind11 module not found. Build it with "
            "`bash sgl-kernel/csrc/cpu/numa_utils_pybind/build.sh` or set "
            "NUMA_UTILS_SO=/path/to/_numa_utils.so."
        )

    available = set(os.sched_getaffinity(0))
    assert len(available) >= 2, "need at least 2 available CPUs for the test"

    n = min(4, len(available))
    cpu_ids_str = _pick_cpu_ids(available, n)
    expected = _parse_cpu_ids(cpu_ids_str)

    orig_num_threads = torch.get_num_threads()
    orig_affinity = os.sched_getaffinity(0)

    try:
        result = mod.init_cpu_threads_env(cpu_ids_str)

        # 1) Return value is a non-empty description string.
        assert isinstance(result, str)
        assert "OMP threads binding" in result

        # 2) PyTorch intra-op thread count matches the requested CPU count.
        assert torch.get_num_threads() == len(expected)

        # 3) Main (calling) thread affinity is within the requested CPU set.
        #    The OpenMP master thread (= Python main thread) pins itself to
        #    omp_cpu_ids[0], so the affinity is a subset of `expected`.
        affinity = os.sched_getaffinity(0)
        assert affinity <= expected, f"main thread affinity {affinity} not in {expected}"
        assert len(affinity) >= 1

        # 4) Every requested core is mentioned in the returned mapping, and
        #    the number of mapped threads equals the number of requested CPUs.
        mentioned_cores = {int(m) for m in re.findall(r"core\s+(\d+)", result)}
        assert mentioned_cores == expected
        tid_core_pairs = re.findall(r"OMP tid:\s*(\d+),\s*core\s+(\d+)", result)
        assert len(tid_core_pairs) == len(expected)

        print("\n" + result)
    finally:
        # Restore what we can. (NUMA memory binding set by numa_set_membind
        # is process-wide and not easily reverted from Python, so we only
        # restore thread count and main-thread CPU affinity here.)
        torch.set_num_threads(orig_num_threads)
        os.sched_setaffinity(0, orig_affinity)


if __name__ == "__main__":
    mod = _load_module()
    if mod is None:
        print(
            "_numa_utils pybind11 module not found.\n"
            "Build it with `bash sgl-kernel/csrc/cpu/numa_utils_pybind/build.sh`\n"
            "or set NUMA_UTILS_SO=/path/to/_numa_utils.so."
        )
        sys.exit(1)
    test_init_cpu_threads_env_binding()
    print("OK: init_cpu_threads_env binding test passed.")