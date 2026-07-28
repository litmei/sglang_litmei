cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(_numa_utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Python + pybind11 -----------------------------------------------------
# `pip install pybind11` provides the CMake config file.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ---- Torch (CPU only, no CUDA needed) --------------------------------------
# Resolve torch's cmake prefix path from the active Python env.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PY_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)

# ---- libnuma ---------------------------------------------------------------
# Honor conda/venv include dirs first (mirrors sgl-kernel/csrc/cpu/CMakeLists.txt).
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
    link_directories(${CONDA_LIB_DIR})
    set(CONDA_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    include_directories(${CONDA_INCLUDE_DIR})
    find_library(NUMA_LIB numa HINTS "${CONDA_LIB_DIR}")
elseif(DEFINED ENV{VIRTUAL_ENV})
    set(VENV_LIB_DIR "$ENV{VIRTUAL_ENV}/lib")
    link_directories(${VENV_LIB_DIR})
    set(VENV_INCLUDE_DIR "$ENV{VIRTUAL_ENV}/include")
    include_directories(${VENV_INCLUDE_DIR})
    find_library(NUMA_LIB numa)
else()
    find_library(NUMA_LIB numa)
endif()
if(NOT NUMA_LIB)
    message(FATAL_ERROR
        "libnuma not found.\n"
        "Install it with one of:\n"
        "  apt-get install libnuma numactl\n"
        "  conda install libnuma numactl")
endif()

# ---- Match the C++ ABI flag used by the installed torch --------------------
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(TORCH_CXX11_ABI STREQUAL "0")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
else()
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
endif()

# ---- Module sources --------------------------------------------------------
# numa_utils.cpp lives one directory up; it pulls in common.h from there.
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../numa_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/binding.cpp
)

pybind11_add_module(_numa_utils ${SOURCES})

target_include_directories(_numa_utils PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INSTALL_PREFIX}/include
    ${Python_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # so "common.h" resolves
)

target_link_libraries(_numa_utils PRIVATE
    ${TORCH_LIBRARIES}
    ${NUMA_LIB}
)

# numa_utils.cpp only uses OpenMP + libnuma + sched_setaffinity; it does NOT
# need AMX/AVX512. Use -march=native so the build runs on whatever CPU the
# host actually has (the parent sgl-kernel/csrc/cpu/CMakeLists.txt pins AMX
# flags for the GEMM/MoE kernels, which we don't compile here).
target_compile_options(_numa_utils PRIVATE
    -O3
    -Wno-unknown-pragmas
    -march=native
    -fopenmp
)

# Drop the .so next to the CMakeLists so the test can find it easily.
set_target_properties(_numa_utils PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS _numa_utils LIBRARY DESTINATION .)