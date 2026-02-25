# FindPipCUDAToolkit.cmake
#
# Locate CUDA toolkit — first trying the host system, then falling back
# to pip-installed packages (nvidia-cuda-nvcc, nvidia-cuda-cccl).
#
# This module should be included BEFORE project() to set CMAKE_CUDA_COMPILER
# when pip CUDA is used.
#
# Detection order:
#   1. Try find_package(CUDAToolkit QUIET) — succeeds if a host CUDA
#      installation is available; skip pip detection.
#   2. If env var WITH_PIP_CUDA_TOOLCHAIN is set to a path (e.g., .../cu13),
#      use that directory directly as the CUDA toolkit root.
#   3. Otherwise, try auto-detecting from the current Python environment's
#      site-packages (works with --no-build-isolation).

# --- Try host CUDA first ---
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  return()
endif()

find_program(_PIP_CUDA_PYTHON_EXE NAMES python3 python)
if(NOT _PIP_CUDA_PYTHON_EXE)
  return()
endif()

# --- Strategy 1: explicit path via env var ---
if(DEFINED ENV{WITH_PIP_CUDA_TOOLCHAIN})
  set(_PIP_CUDA_ROOT "$ENV{WITH_PIP_CUDA_TOOLCHAIN}")
  if(NOT EXISTS "${_PIP_CUDA_ROOT}/bin/nvcc")
    message(FATAL_ERROR
      "FindPipCUDAToolkit: WITH_PIP_CUDA_TOOLCHAIN is set to '${_PIP_CUDA_ROOT}' "
      "but nvcc was not found at '${_PIP_CUDA_ROOT}/bin/nvcc'")
  endif()
  # Prepare the directory (create lib64 symlink, unversioned .so symlinks,
  # libcuda.so stub) that CMake / nvcc expect but pip packages omit.
  execute_process(
    COMMAND "${_PIP_CUDA_PYTHON_EXE}" "${CMAKE_CURRENT_LIST_DIR}/find_pip_cuda.py"
            "${_PIP_CUDA_ROOT}"
    OUTPUT_QUIET
  )
  message(STATUS "FindPipCUDAToolkit: using env WITH_PIP_CUDA_TOOLCHAIN=${_PIP_CUDA_ROOT}")
else()
  # --- Strategy 2: auto-detect from current Python env ---
  execute_process(
    COMMAND "${_PIP_CUDA_PYTHON_EXE}" "${CMAKE_CURRENT_LIST_DIR}/find_pip_cuda.py"
    OUTPUT_VARIABLE _PIP_CUDA_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _PIP_CUDA_RESULT
  )

  if(NOT _PIP_CUDA_RESULT EQUAL 0)
    message(STATUS "FindPipCUDAToolkit: pip-installed CUDA toolkit not found")
    return()
  endif()

  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: auto-detected from Python environment")
endif()

# --- Common pip-CUDA setup ---
set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler (from pip)" FORCE)
set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "CUDA toolkit root (from pip)" FORCE)

list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_ROOT}/lib/stubs" "${_PIP_CUDA_ROOT}/lib")

message(STATUS "FindPipCUDAToolkit: using pip-installed CUDA toolkit")
message(STATUS "  nvcc: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  root: ${CUDAToolkit_ROOT}")
