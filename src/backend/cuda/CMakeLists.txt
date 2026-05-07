# CUDA backend: toolchain, stub libraries, source files, and build configuration.
if(NOT USE_CUDA)
  return()
endif()

set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_NVCC_EXECUTABLE)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "CUDA compiler" FORCE)
elseif(WIN32)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
else()
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
endif()
add_compile_definitions("CUDA_MAJOR_VERSION=${CUDAToolkit_VERSION_MAJOR}")
# CUDA stubs derive runtime DLL/SONAME candidates from the same toolkit
# version parsed by find_package(CUDAToolkit).
set(TILELANG_CUDA_TOOLKIT_VERSION_DEFINITIONS
  "TILELANG_CUDA_TOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}"
  "TILELANG_CUDA_TOOLKIT_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}"
)

# Set `USE_CUDA=/usr/local/cuda-x.y`
cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH USE_CUDA)

if(WIN32 AND NOT TILELANG_USE_CUDA_STUBS)
  file(GLOB _tilelang_nvrtc_candidates
    "${CUDAToolkit_ROOT}/bin/x86_64/nvrtc64_*.dll"
    "${CUDAToolkit_ROOT}/bin/nvrtc64_*.dll")
  list(SORT _tilelang_nvrtc_candidates COMPARE NATURAL ORDER DESCENDING)
  if(_tilelang_nvrtc_candidates)
    list(GET _tilelang_nvrtc_candidates 0 _tilelang_nvrtc_library)
    tilelang_generate_windows_import_library("${_tilelang_nvrtc_library}" _tilelang_nvrtc_import_lib "nvrtc")
    set(CUDA_NVRTC_LIBRARY "${_tilelang_nvrtc_import_lib}" CACHE FILEPATH
      "NVRTC runtime library to link against" FORCE)
  endif()
endif()

if(TILELANG_USE_CUDA_STUBS)
  # ============================================================================
  # CUDA Driver Stub Library (libcuda_stub.so / cuda_stub.dll)
  # ============================================================================
  # This library provides drop-in replacements for CUDA driver API functions.
  # Instead of linking directly against libcuda.so (which would fail on
  # CPU-only machines), we link against this stub which loads libcuda.so
  # lazily at runtime on first API call.
  #
  # The stub exports global C functions matching the CUDA driver API:
  #   - cuModuleLoadData, cuLaunchKernel, cuMemsetD32_v2, etc.
  # These can be called directly without any wrapper macros.
  # ============================================================================
  add_library(cuda_stub SHARED src/backend/cuda/stubs/cuda.cc)
  target_include_directories(cuda_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  # Export symbols with visibility="default" when building
  target_compile_definitions(cuda_stub PRIVATE TILELANG_CUDA_STUB_EXPORTS)
  if(WIN32)
    target_link_libraries(cuda_stub PRIVATE psapi)
  else()
    target_link_libraries(cuda_stub PRIVATE ${CMAKE_DL_LIBS})
  endif()
  set_target_properties(cuda_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    # Use consistent naming
    OUTPUT_NAME "cuda_stub"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

  # ============================================================================
  # CUDA Runtime Stub Library (libcudart_stub.so / cudart_stub.dll)
  # ============================================================================
  # libcudart's SONAME includes its major version (e.g. libcudart.so.11.0 / .12 / .13).
  # Link against this stub instead of the real libcudart so a single wheel can
  # run in environments that provide different libcudart major versions.
  #
  # The stub exports a minimal set of CUDA Runtime API entrypoints used by TVM
  # and lazily loads libcudart at runtime on first API call.
  # ============================================================================
  add_library(cudart_stub SHARED src/backend/cuda/stubs/cudart.cc)
  target_include_directories(cudart_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(cudart_stub PRIVATE TILELANG_CUDART_STUB_EXPORTS
                                                 ${TILELANG_CUDA_TOOLKIT_VERSION_DEFINITIONS})
  if(WIN32)
    target_link_libraries(cudart_stub PRIVATE psapi)
  else()
    target_link_libraries(cudart_stub PRIVATE ${CMAKE_DL_LIBS})
  endif()
  set_target_properties(cudart_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "cudart_stub"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

  # Make TVM link against our CUDA Runtime stub instead of the real libcudart.
  #
  # NOTE: TVM's `find_cuda()` calls `find_library(CUDA_CUDART_LIBRARY cudart ...)`.
  # `find_library()` will not override an already-cached variable, so setting it
  # here ensures TVM doesn't record a DT_NEEDED on `libcudart.so.<major>`.
  set(CUDA_CUDART_LIBRARY cudart_stub CACHE STRING "CUDART library to link against" FORCE)

  # ============================================================================
  # NVRTC Stub Library (libnvrtc_stub.so / nvrtc_stub.dll)
  # ============================================================================
  # NVRTC's SONAME includes its major version (e.g. libnvrtc.so.11.2 / .12 / .13).
  # Link against this stub instead of the real NVRTC library so a single wheel
  # can run in environments that provide different NVRTC major versions.
  #
  # The stub exports a minimal set of NVRTC C API entrypoints used by TVM and
  # lazily loads libnvrtc at runtime on first API call.
  # ============================================================================
  add_library(nvrtc_stub SHARED src/backend/cuda/stubs/nvrtc.cc)
  target_include_directories(nvrtc_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(nvrtc_stub PRIVATE TILELANG_NVRTC_STUB_EXPORTS
                                                  ${TILELANG_CUDA_TOOLKIT_VERSION_DEFINITIONS})
  if(WIN32)
    target_link_libraries(nvrtc_stub PRIVATE psapi)
  else()
    target_link_libraries(nvrtc_stub PRIVATE ${CMAKE_DL_LIBS})
  endif()
  set_target_properties(nvrtc_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "nvrtc_stub"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

  # Make TVM link against our NVRTC stub instead of the real libnvrtc.
  #
  # NOTE: TVM's `find_cuda()` calls `find_library(CUDA_NVRTC_LIBRARY nvrtc ...)`.
  # `find_library()` will not override an already-cached variable, so setting it
  # here ensures TVM doesn't record a DT_NEEDED on `libnvrtc.so.<major>`.
  set(CUDA_NVRTC_LIBRARY nvrtc_stub CACHE STRING "NVRTC library to link against" FORCE)
endif()

file(GLOB TILE_LANG_CUDA_SRCS
  src/runtime/runtime.cc
  src/backend/cuda/codegen/ptx.cc
  src/backend/cuda/codegen/codegen_cuda.cc
  src/backend/cuda/codegen/codegen_py.cc
  src/target/codegen_utils.cc
  src/backend/cuda/codegen/codegen_cutedsl.cc
  src/backend/cuda/codegen/rt_mod_cuda.cc
  src/backend/cuda/codegen/rt_mod_cutedsl.cc
  src/backend/cuda/op/*.cc
)
list(REMOVE_ITEM TILE_LANG_CUDA_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/op/copy_analysis.cc")
list(APPEND TILE_LANG_SRCS ${TILE_LANG_CUDA_SRCS})

list(APPEND TILE_LANG_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)

# Register stubs for linking and install
if(TILELANG_USE_CUDA_STUBS)
  set(TILELANG_ACTIVE_BACKEND_STUB_LINK cuda_stub)
  set(TILELANG_ACTIVE_BACKEND_STUB_TARGETS cuda_stub cudart_stub nvrtc_stub)
endif()

# Register additional RPATH for CUDA toolkit lib directory
if(UNIX)
  set(TILELANG_ACTIVE_BACKEND_RPATH_EXTRA ":\$ORIGIN/../../nvidia/cu${CUDAToolkit_VERSION_MAJOR}/lib")
endif()

# Register patchelf removals (SONAMEs to strip for portable wheels)
set(TILELANG_ACTIVE_BACKEND_PATCHELF_REMOVE "libcuda.so.1;libcuda.so")
