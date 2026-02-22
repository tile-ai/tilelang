# Learn a lot from the MLC - LLM Project
# https://github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.26)

# Detect CUDA toolkit: tries host installation first, then falls back to
# pip-installed packages (env WITH_PIP_CUDA_TOOLCHAIN or auto-detect).
# Must be included before project() so CMAKE_CUDA_COMPILER is set.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindPipCUDAToolkit.cmake)

project(TILE_LANG C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND "$ENV{CIBUILDWHEEL}")
  # Warning came from tvm submodule
  string(APPEND CMAKE_CXX_FLAGS " -Wno-dangling-reference")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE TILELANG_GIT_SUBMODULE_RESULT
    )
    if(NOT TILELANG_GIT_SUBMODULE_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          "Failed to initialize git submodules. Please run "
          "`git submodule update --init --recursive` and re-run CMake."
      )
    endif()
  else()
    message(
      FATAL_ERROR
        "Git is required to initialize TileLang submodules. "
        "Please install git or fetch the submodules manually."
    )
  endif()
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM} with base_dir=${CMAKE_SOURCE_DIR}")
  if(APPLE)
    # Passing configs like `ccache base_dir=/xxx cc ...` is supported
    # (likely) since ccache 4.x, which has been provided by homebrew.
    # Our Linux builder image (manylinux2014 & manylinux_2_28) still
    # provides ccache 3.x and do not support this form.
    # `cibuildwheel` uses fixed folder on Linux (`/project`) as working directory,
    # so cache would work without setting `base_dir`.
    set(CCACHE_PROGRAM "${CCACHE_PROGRAM};base_dir=${CMAKE_SOURCE_DIR}")
  endif()
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
else()
  find_program(SCCACHE_PROGRAM sccache)
  if(SCCACHE_PROGRAM)
    message(STATUS "Using sccache: ${SCCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
  endif()
endif()

# Configs
set(TILELANG_BACKENDS CUDA ROCM METAL)

set(TILELANG_BACKEND_DOC_CUDA "Enable CUDA backend (ON/OFF/or CUDA SDK path)")
set(TILELANG_BACKEND_DOC_ROCM "Enable ROCm backend (ON/OFF/or ROCm SDK path)")
set(TILELANG_BACKEND_DOC_METAL "Enable Metal backend")

# TVM's config.cmake redefines USE_* options later, so we cache the user's choice
# (including explicit -DUSE_XXX arguments) before we include TVM and restore it
# afterwards.

macro(tilelang_define_backend_option BACKEND)
  set(_backend_var "USE_${BACKEND}")
  set(_doc "${TILELANG_BACKEND_DOC_${BACKEND}}")
  set(_user_override_var "TILELANG_USER_OVERRIDE_${_backend_var}")

  set(_user_override OFF)
  if(DEFINED ${_user_override_var})
    set(_user_override "${${_user_override_var}}")
  endif()

  if(DEFINED CACHE{${_backend_var}})
    get_property(_cache_type CACHE ${_backend_var} PROPERTY TYPE)
    if(_cache_type STREQUAL "UNINITIALIZED")
      set(_user_override ON)
    endif()
  endif()

  set(_default OFF)
  if(DEFINED ${_backend_var})
    set(_default "${${_backend_var}}")
  endif()

  option(${_backend_var} "${_doc}" "${_default}")
  # Remember if the user explicitly set this option so that later logic
  # won't auto-toggle backends they configured on the command line.
  set(${_user_override_var} ${_user_override} CACHE INTERNAL
    "User explicitly set ${_backend_var} during configuration" FORCE)
  set(TILELANG_OPTION_${_backend_var} "${${_backend_var}}")
endmacro()

foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  tilelang_define_backend_option(${BACKEND})
endforeach()

set(PREBUILD_CYTHON ON)

# CUDA stub libraries (cuda/cudart/nvrtc) are used to build wheels that can run
# across different CUDA Toolkit major versions and/or on CPU-only machines by
# avoiding hard DT_NEEDED dependencies on versioned CUDA SONAMEs.
#
# These stubs are currently POSIX-only (dlopen/dlsym via <dlfcn.h>).
if(WIN32 AND NOT CYGWIN)
  set(_TILELANG_USE_CUDA_STUBS_DEFAULT OFF)
else()
  set(_TILELANG_USE_CUDA_STUBS_DEFAULT ON)
endif()
option(TILELANG_USE_CUDA_STUBS
       "Use POSIX dlopen-based CUDA stub libraries (cuda/cudart/nvrtc) for portable wheels"
       ${_TILELANG_USE_CUDA_STUBS_DEFAULT})
unset(_TILELANG_USE_CUDA_STUBS_DEFAULT)
# Configs end

include(cmake/load_tvm.cmake)

if(EXISTS ${TVM_SOURCE}/cmake/config.cmake)
  include(${TVM_SOURCE}/cmake/config.cmake)
else()
  message(FATAL_ERROR "Nor tvm provided or submodule checkout-ed.")
endif()
# Re-apply TileLang's preferred backend settings after TVM's config may have
# overridden the USE_* cache entries.
foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  set(_backend_var "USE_${BACKEND}")
  set(_doc "${TILELANG_BACKEND_DOC_${BACKEND}}")
  set(${_backend_var} ${TILELANG_OPTION_${_backend_var}} CACHE STRING "${_doc}" FORCE)
  set(${_backend_var} ${TILELANG_OPTION_${_backend_var}})
endforeach()

# Include directories for TileLang
set(TILE_LANG_INCLUDES ${TVM_INCLUDES})

# Collect source files
file(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/transform/common/*.cc
  src/op/*.cc
  src/target/utils.cc
  src/target/codegen_c_host.cc
  src/target/codegen_cpp.cc
  src/target/rt_mod_cpp.cc
  # intrin_rule doesn't have system dependency
  src/target/intrin_rule*.cc
)

# Always include CPU-safe runtime helpers
list(APPEND TILE_LANG_SRCS
  src/runtime/error_helpers.cc
)

# Track if the user explicitly selected a backend via cache options.
set(TILELANG_BACKEND_USER_SELECTED OFF)
foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  set(_backend_var "USE_${BACKEND}")
  set(_override_var "TILELANG_USER_OVERRIDE_${_backend_var}")
  if(${_backend_var} OR ${_override_var})
    set(TILELANG_BACKEND_USER_SELECTED ON)
  endif()
endforeach()

# Only auto-select a backend when the user didn't specify one explicitly.
if(NOT TILELANG_BACKEND_USER_SELECTED)
  if($ENV{USE_METAL})
    set(USE_METAL ON)
  elseif(APPLE)
    message(STATUS "Enable Metal support by default.")
    set(USE_METAL ON)
  elseif($ENV{USE_ROCM})
    set(USE_ROCM ON)
  else()
    if($ENV{USE_CUDA})
      set(USE_CUDA ON)
    elseif(DEFINED ENV{USE_CUDA} AND NOT $ENV{USE_CUDA})
      # Build CPU-only when we explicitly disable CUDA
      set(USE_CUDA OFF)
    else()
      message(STATUS "Enable CUDA support by default.")
      set(USE_CUDA ON)
    endif()
  endif()
endif()

if(USE_METAL)
  if(NOT APPLE)
    # On non-Apple platforms USE_METAL=ON enables only codegen (Metal source
    # generation) without requiring the Metal/Foundation frameworks.
    message(STATUS "Metal backend on non-Apple: enabling codegen-only mode (no Metal runtime)")
    set(USE_METAL OFF)
  endif()
  file(GLOB TILE_LANG_METAL_SRCS
    src/target/rt_mod_metal.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_METAL_SRCS})
  # FIXME: CIBW failed with backtrace, why???
  set(TVM_FFI_USE_LIBBACKTRACE OFF)
elseif(USE_ROCM)
  set(CMAKE_HIP_STANDARD 17)
  include(${TVM_SOURCE}/cmake/utils/FindROCM.cmake)
  find_rocm(${USE_ROCM})
  add_compile_definitions(__HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__=1)

  file(GLOB TILE_LANG_HIP_SRCS
    src/target/codegen_hip.cc
    src/target/rt_mod_hip.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
  list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})
elseif(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
  add_compile_definitions("CUDA_MAJOR_VERSION=${CUDAToolkit_VERSION_MAJOR}")

  # Set `USE_CUDA=/usr/local/cuda-x.y`
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH USE_CUDA)

  if(TILELANG_USE_CUDA_STUBS)
    if(WIN32 AND NOT CYGWIN)
      message(FATAL_ERROR "TILELANG_USE_CUDA_STUBS=ON is not supported on Windows. "
                          "Please configure with -DTILELANG_USE_CUDA_STUBS=OFF.")
    endif()

    # ============================================================================
    # CUDA Driver Stub Library (libcuda_stub.so)
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
    add_library(cuda_stub SHARED src/target/stubs/cuda.cc)
    target_include_directories(cuda_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # Export symbols with visibility="default" when building
    target_compile_definitions(cuda_stub PRIVATE TILELANG_CUDA_STUB_EXPORTS)
    # Use dlopen/dlsym for runtime library loading
    target_link_libraries(cuda_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(cuda_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      # Use consistent naming
      OUTPUT_NAME "cuda_stub"
    )

    # ============================================================================
    # CUDA Runtime Stub Library (libcudart_stub.so)
    # ============================================================================
    # libcudart's SONAME includes its major version (e.g. libcudart.so.11.0 / .12 / .13).
    # Link against this stub instead of the real libcudart so a single wheel can
    # run in environments that provide different libcudart major versions.
    #
    # The stub exports a minimal set of CUDA Runtime API entrypoints used by TVM
    # and lazily loads libcudart at runtime on first API call.
    # ============================================================================
    add_library(cudart_stub SHARED src/target/stubs/cudart.cc)
    target_include_directories(cudart_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(cudart_stub PRIVATE TILELANG_CUDART_STUB_EXPORTS)
    target_link_libraries(cudart_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(cudart_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "cudart_stub"
    )

    # Make TVM link against our CUDA Runtime stub instead of the real libcudart.
    #
    # NOTE: TVM's `find_cuda()` calls `find_library(CUDA_CUDART_LIBRARY cudart ...)`.
    # `find_library()` will not override an already-cached variable, so setting it
    # here ensures TVM doesn't record a DT_NEEDED on `libcudart.so.<major>`.
    set(CUDA_CUDART_LIBRARY cudart_stub CACHE STRING "CUDART library to link against" FORCE)

    # ============================================================================
    # NVRTC Stub Library (libnvrtc_stub.so)
    # ============================================================================
    # NVRTC's SONAME includes its major version (e.g. libnvrtc.so.11.2 / .12 / .13).
    # Link against this stub instead of the real NVRTC library so a single wheel
    # can run in environments that provide different NVRTC major versions.
    #
    # The stub exports a minimal set of NVRTC C API entrypoints used by TVM and
    # lazily loads libnvrtc at runtime on first API call.
    # ============================================================================
    add_library(nvrtc_stub SHARED src/target/stubs/nvrtc.cc)
    target_include_directories(nvrtc_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(nvrtc_stub PRIVATE TILELANG_NVRTC_STUB_EXPORTS)
    target_link_libraries(nvrtc_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(nvrtc_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "nvrtc_stub"
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
    src/target/ptx.cc
    src/target/codegen_cuda.cc
    src/target/codegen_py.cc
    src/target/codegen_utils.cc
    src/target/codegen_cutedsl.cc
    src/target/rt_mod_cuda.cc
    src/target/rt_mod_cutedsl.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_CUDA_SRCS})

  list(APPEND TILE_LANG_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
  link_directories(${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)
endif()

set(USE_Z3      ON CACHE STRING "Use Z3 SMT solver for TileLang optimizations")
set(USE_PYPI_Z3 ON CACHE BOOL   "Use Z3 provided by PyPI z3-solver package")

if(USE_Z3 AND USE_PYPI_Z3)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/pypi-z3")
  find_package(Z3 REQUIRED)
endif()

# Include tvm after configs have been populated
add_subdirectory(${TVM_SOURCE} tvm EXCLUDE_FROM_ALL)

# Resolve compile warnings in tvm
add_compile_definitions(DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

# Set debug mode compile definitions
# Enable the TVM debug option, i.e., TVM_LOG_DEBUG
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building TileLang with DEBUG mode")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
endif()

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})

add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm)

# Place dev build outputs under build/lib for consistency
set_target_properties(tilelang PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
# Build cython extension
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp"
  COMMENT
    "Cythoning tilelang/jit/adapter/cython/cython_wrapper.pyx"
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
          --module-name tilelang_cython_wrapper
          --cplus --output-file "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
  VERBATIM)

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

python_add_library(tilelang_cython_wrapper MODULE "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp" ${USE_SABI} WITH_SOABI)

# Ensure dev builds drop the extension into build/lib alongside other shared libs
set_target_properties(tilelang_cython_wrapper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Install the extension into tilelang/lib inside the wheel
install(TARGETS tilelang_cython_wrapper
        LIBRARY DESTINATION tilelang/lib
        RUNTIME DESTINATION tilelang/lib
        ARCHIVE DESTINATION tilelang/lib)

# Copy libz3.so to build folder to workaround isolated build env issue
if(USE_Z3 AND USE_PYPI_Z3)
  get_target_property(Z3_LIBRARY_PATH z3::libz3 IMPORTED_LOCATION)
  install(FILES "${Z3_LIBRARY_PATH}" DESTINATION "${CMAKE_BINARY_DIR}/lib")
  if(APPLE)
    set_target_properties(tvm PROPERTIES BUILD_RPATH "@loader_path")
  else()
    set_target_properties(tvm PROPERTIES BUILD_RPATH "\$ORIGIN")
  endif()
endif()

set(TILELANG_OUTPUT_TARGETS tilelang tvm)

if(USE_CUDA AND TILELANG_USE_CUDA_STUBS)
  # Link against CUDA stub library instead of libcuda.so
  # This enables lazy loading of libcuda.so at runtime, allowing
  # `import tilelang` to succeed on CPU-only machines.
  foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
    target_link_libraries(${target} PUBLIC cuda_stub)
  endforeach()
  # Include CUDA stubs in output targets for RPATH configuration
  list(APPEND TILELANG_OUTPUT_TARGETS cuda_stub cudart_stub nvrtc_stub)
endif()

unset(PATCHELF_EXECUTABLE CACHE)

if(APPLE)
  set(TILELANG_INSTALL_RPATH "@loader_path;@loader_path/../../tvm_ffi/lib")
  if(USE_Z3 AND USE_PYPI_Z3)
    # Some z3 is placed in lib/ and some in bin/, we add both in rpath
    string(APPEND TILELANG_INSTALL_RPATH ";@loader_path/../../z3/lib;@loader_path/../../z3/bin")
  endif()
elseif(UNIX)
  set(TILELANG_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../tvm_ffi/lib")
  if(USE_Z3 AND USE_PYPI_Z3)
    string(APPEND TILELANG_INSTALL_RPATH ":\$ORIGIN/../../z3/lib")
  endif()
  if(USE_CUDA)
    string(APPEND TILELANG_INSTALL_RPATH ":\$ORIGIN/../../nvidia/cu${CUDAToolkit_VERSION_MAJOR}/lib")
  endif()
  find_program(PATCHELF_EXECUTABLE patchelf)
  if (NOT PATCHELF_EXECUTABLE)
    message(STATUS "`patchelf` not found.")
  endif()
endif()

# Let libtilelang search for tvm in the same directory
foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
  set_target_properties(${target} PROPERTIES INSTALL_RPATH "${TILELANG_INSTALL_RPATH}")
  set_target_properties(${target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
endforeach()

# Exclude libcuda.so to allow importing on a CPU-only machine
if(USE_CUDA AND TILELANG_USE_CUDA_STUBS AND PATCHELF_EXECUTABLE)
  # Run `patchelf` on built libraries to remove libcuda.so dependency.
  # Use `install(CODE ...)` instead of `add_custom_command(... POST_BUILD ...)`
  # to avoid race conditions during linking.
  foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
    install(CODE "
      execute_process(
        COMMAND ${PATCHELF_EXECUTABLE}
          --remove-needed libcuda.so.1
          --remove-needed libcuda.so
          \"$<TARGET_FILE:${target}>\"
        WORKING_DIRECTORY \"${CMAKE_INSTALL_PREFIX}\"
        RESULT_VARIABLE patchelf_result
      )
      if(patchelf_result EQUAL 0)
        message(STATUS \"`patchelf` successfully removed dependency `libcuda.so` from $<TARGET_FILE:${target}>\")
      else()
        message(WARNING \"`patchelf` failed to remove dependency `libcuda.so` from $<TARGET_FILE:${target}>\")
      endif()
    ")
  endforeach()
endif()

install(
  TARGETS ${TILELANG_OUTPUT_TARGETS}
  LIBRARY DESTINATION tilelang/lib
  RUNTIME DESTINATION tilelang/lib
  ARCHIVE DESTINATION tilelang/lib
)
