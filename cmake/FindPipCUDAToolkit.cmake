# CUDA SDK discovery runs after project(), independently of the host toolchain.
# Explicitly disabled backends must not inspect or materialize a CUDA SDK.
if(DEFINED USE_CUDA AND NOT USE_CUDA)
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "disabled" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
elseif(NOT DEFINED USE_CUDA AND DEFINED ENV{USE_CUDA} AND NOT "$ENV{USE_CUDA}")
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "disabled" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
endif()

# --- Try host CUDA first ---
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE ON CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "host" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
endif()

set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
set(TILELANG_CUDA_TOOLKIT_SOURCE "none" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)

include("${CMAKE_CURRENT_LIST_DIR}/PythonToolchain.cmake")
set(_tilelang_cuda_probe "${CMAKE_CURRENT_LIST_DIR}/find_pip_cuda.py")

# --- Strategy 1: explicit path via env var ---
if(DEFINED ENV{WITH_PIP_CUDA_TOOLCHAIN})
  tilelang_probe_python_sdk("${_tilelang_cuda_probe}" _PIP_CUDA_OUTPUT _PIP_CUDA_PYTHON_EXE "$ENV{WITH_PIP_CUDA_TOOLCHAIN}")
  if(NOT _PIP_CUDA_OUTPUT)
    message(FATAL_ERROR
      "FindPipCUDAToolkit: WITH_PIP_CUDA_TOOLCHAIN is set to '$ENV{WITH_PIP_CUDA_TOOLCHAIN}' "
      "but no pip-installed CUDA toolkit could be resolved from that path")
  endif()
  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: using env WITH_PIP_CUDA_TOOLCHAIN=${_PIP_CUDA_ROOT}")
else()
  # --- Strategy 2: auto-detect from current Python env ---
  tilelang_probe_python_sdk("${_tilelang_cuda_probe}" _PIP_CUDA_OUTPUT _PIP_CUDA_PYTHON_EXE)
  if(NOT _PIP_CUDA_OUTPUT)
    message(STATUS "FindPipCUDAToolkit: pip-installed CUDA toolkit not found")
    return()
  endif()

  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: auto-detected from Python environment via ${_PIP_CUDA_PYTHON_EXE}")
endif()

# --- Common pip-CUDA setup ---
string(JSON _PIP_CUDA_NVCC GET "${_PIP_CUDA_OUTPUT}" "nvcc")
string(JSON _PIP_CUDA_LIBRARY_DIR GET "${_PIP_CUDA_OUTPUT}" "library_dir")

if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_NVCC}" CACHE FILEPATH "CUDA compiler (from pip)" FORCE)
endif()
if(NOT CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "CUDA toolkit root (from pip)" FORCE)
endif()
set(TILELANG_CUDA_TOOLKIT_AVAILABLE ON CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
set(TILELANG_CUDA_TOOLKIT_SOURCE "pip" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)

list(APPEND CMAKE_PROGRAM_PATH "${_PIP_CUDA_ROOT}/bin")
if(WIN32)
  list(APPEND CMAKE_PROGRAM_PATH "${_PIP_CUDA_ROOT}/bin/x86_64" "${_PIP_CUDA_ROOT}/nvvm/bin")
  list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_LIBRARY_DIR}")
  set(ENV{PATH} "${_PIP_CUDA_ROOT}/bin;${_PIP_CUDA_ROOT}/bin/x86_64;${_PIP_CUDA_ROOT}/nvvm/bin;$ENV{PATH}")
else()
  list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_ROOT}/lib/stubs" "${_PIP_CUDA_LIBRARY_DIR}")
endif()

message(STATUS "FindPipCUDAToolkit: using pip-installed CUDA toolkit")
message(STATUS "  nvcc: ${_PIP_CUDA_NVCC}")
message(STATUS "  root: ${CUDAToolkit_ROOT}")
