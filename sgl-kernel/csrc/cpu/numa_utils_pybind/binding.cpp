// Pybind11 binding for sgl-kernel/csrc/cpu/numa_utils.cpp.
//
// Why a standalone pybind11 module?
//   - The CUDA-built common_ops.so (from the top-level CMakeLists.txt) does not
//     include numa_utils.cpp, so torch.ops.sgl_kernel.init_cpu_threads_env is
//     unavailable in GPU installs.
//   - This minimal module exposes init_cpu_threads_env directly via pybind11
//     without going through the torch op registry, so it works even when CUDA
//     is unavailable.
//
// Build (see CMakeLists.txt and build.sh next to this file):
//   pip install pybind11            # if not yet installed
//   apt-get install -y libnuma numactl
//   cmake -S sgl-kernel/csrc/cpu/numa_utils_pybind -B build
//   cmake --build build -j

#include <c10/util/Exception.h>  // c10::Error thrown by TORCH_CHECK
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

// Forward declaration: defined in ../numa_utils.cpp (compiled into this module).
std::string init_cpu_threads_env(const std::string& cpu_ids);

namespace py = pybind11;

PYBIND11_MODULE(_numa_utils, m) {
    m.doc() = "Python binding for sgl-kernel/csrc/cpu/numa_utils.cpp";

    m.def(
        "init_cpu_threads_env",
        [](const std::string& cpu_ids) -> std::string {
            // TORCH_CHECK throws c10::Error; translate it to a plain
            // std::runtime_error so callers get a clean Python RuntimeError
            // instead of a pybind11 error_already_set wrapping an unknown
            // C++ exception type.
            try {
                return init_cpu_threads_env(cpu_ids);
            } catch (const c10::Error& e) {
                throw std::runtime_error(e.what());
            }
        },
        py::arg("cpu_ids"),
        "Pin the current process's OpenMP threads to the given CPU set,\n"
        "migrate existing memory pages to the corresponding NUMA node,\n"
        "restrict future memory allocation to that node, and set the\n"
        "PyTorch intra-op thread count.\n\n"
        "Args:\n"
        "    cpu_ids: numactl-style CPU spec, e.g. \"0-3\" or \"0,2,4-6\".\n\n"
        "Returns:\n"
        "    A human-readable string describing the OMP tid -> core mapping.\n");
}