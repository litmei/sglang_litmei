#!/usr/bin/env bash
# Build the standalone pybind11 module for init_cpu_threads_env.
#
# Requirements:
#   - Python with torch + pybind11 installed
#   - libnuma (apt-get install libnuma numactl  OR  conda install libnuma numactl)
#
# Usage:
#   ./build.sh                # Release build, default jobs
#   ./build.sh Debug 16       # Debug build, 16 parallel jobs
set -e

BUILD_TYPE="${1:-Release}"
JOBS="${2:-$(nproc)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "==> Configuring in ${BUILD_DIR} (BUILD_TYPE=${BUILD_TYPE})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -Dpybind11_DIR=$(python -m pybind11 --cmakedir)

echo "==> Building with -j${JOBS}"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo
echo "==> Done. Module .so is at:"
ls -1 "${BUILD_DIR}"/_numa_utils*.so 2>/dev/null || true
echo
echo "Use it from Python:"
echo "  import sys; sys.path.insert(0, '${BUILD_DIR}')"
echo "  import _numa_utils"
echo "  print(_numa_utils.init_cpu_threads_env('0-3'))"