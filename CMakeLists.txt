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

# HIP stub libraries (hip/hiprtc) are used to build wheels that can be imported
# on machines without ROCm installed by avoiding hard DT_NEEDED dependencies on
# libamdhip64.so / libhiprtc.so.
#
# These stubs are currently POSIX-only (dlopen/dlsym via <dlfcn.h>).
if(WIN32 AND NOT CYGWIN)
  set(_TILELANG_USE_HIP_STUBS_DEFAULT OFF)
else()
  # Only meaningful when USE_ROCM is enabled.
  set(_TILELANG_USE_HIP_STUBS_DEFAULT ON)
endif()
option(TILELANG_USE_HIP_STUBS
       "Use POSIX dlopen-based HIP stub libraries (hip/hiprtc) for portable wheels"
       ${_TILELANG_USE_HIP_STUBS_DEFAULT})
unset(_TILELANG_USE_HIP_STUBS_DEFAULT)
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
# tvm tries to detect gtest by default, but may fail if its header is not installed.
set(USE_GTEST OFF)

# Include directories for TileLang
set(TILE_LANG_INCLUDES ${TVM_INCLUDES})
list(APPEND TILE_LANG_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Collect source files
file(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/transform/common/*.cc
  src/op/*.cc
  src/backend/cpu/op/*.cc
  src/backend/cuda/op/copy_analysis.cc
  src/backend/metal/op/*.cc
  src/backend/webgpu/op/*.cc
  src/target/utils.cc
  src/target/codegen_c_host.cc
  src/target/codegen_c.cc
  src/target/rt_mod_c.cc
  # intrin_rule doesn't have system dependency
  src/target/intrin_rule*.cc
)

# Always include CPU-safe runtime helpers
list(APPEND TILE_LANG_SRCS
  src/runtime/error_helpers.cc
)

set(TILELANG_OUTPUT_TARGETS tilelang tvm)

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

# Backend-local CMake files own native source lists, stubs, include paths, and
# compile definitions. Top-level CMake only selects and delegates.
include("${CMAKE_CURRENT_SOURCE_DIR}/src/backend/cuda/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/src/backend/rocm/CMakeLists.txt")
include("${CMAKE_CURRENT_SOURCE_DIR}/src/backend/metal/CMakeLists.txt")

set(USE_Z3      ON CACHE STRING "Use Z3 SMT solver for TileLang optimizations")
set(USE_PYPI_Z3 ON CACHE BOOL   "Use Z3 provided by PyPI z3-solver package")

if(USE_Z3 AND USE_PYPI_Z3)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/pypi-z3")
  find_package(Z3 REQUIRED)
endif()

# Enable custom logging so we control the output format (e.g. strip build paths
# from __FILE__ so wheel users don't see CI machine paths in warnings).
set(USE_CUSTOM_LOGGING ON CACHE BOOL "Use custom logging implementation" FORCE)

# Detect release (wheel) builds: in CI (cibuildwheel) or scikit-build-core wheel builds,
# we strip source paths from LOG(WARNING) etc. for a cleaner user experience.
# Local dev builds keep full paths for debugging.
if(DEFINED ENV{CIBUILDWHEEL} OR "$ENV{SKBUILD_STATE}" STREQUAL "wheel")
  set(TILELANG_RELEASE_BUILD_DEFAULT ON)
else()
  set(TILELANG_RELEASE_BUILD_DEFAULT OFF)
endif()
option(TILELANG_RELEASE_BUILD "Strip source paths from log messages (for wheel releases)" ${TILELANG_RELEASE_BUILD_DEFAULT})

# Include tvm after configs have been populated
add_subdirectory(${TVM_SOURCE} tvm EXCLUDE_FROM_ALL)

# Provide the custom LogMessageImpl / LogFatalImpl implementation to TVM,
# since TVM_LOG_CUSTOMIZE=1 requires them to be supplied by the user.
target_sources(tvm_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/logging.cc")
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tvm_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()

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
target_compile_definitions(tilelang_objs PRIVATE TVM_LOG_CUSTOMIZE=1)
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tilelang_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()

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

# Disable Cython's PEP-489 multi-phase init for the wrapper. The generated
# C++ depends on CPython's private `_xxsubinterpreters` module at import
# time, which is stripped from some distributor-built Python 3.12 builds
# (notably Red Hat's RHEL 9 system Python). Single-phase init avoids that
# dependency and matches Cython's own suggested workaround. See #2125.
target_compile_definitions(tilelang_cython_wrapper PRIVATE CYTHON_PEP489_MULTI_PHASE_INIT=0)

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

if(DEFINED TILELANG_ACTIVE_BACKEND_STUB_LINK)
  foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
    target_link_libraries(${target} PUBLIC ${TILELANG_ACTIVE_BACKEND_STUB_LINK})
  endforeach()
endif()

# Append stub targets after the linking loop so they don't link to themselves
if(DEFINED TILELANG_ACTIVE_BACKEND_STUB_TARGETS)
  list(APPEND TILELANG_OUTPUT_TARGETS ${TILELANG_ACTIVE_BACKEND_STUB_TARGETS})
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
  if(DEFINED TILELANG_ACTIVE_BACKEND_RPATH_EXTRA)
    string(APPEND TILELANG_INSTALL_RPATH "${TILELANG_ACTIVE_BACKEND_RPATH_EXTRA}")
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

# Strip backend runtime dependencies for portable wheels
if(DEFINED TILELANG_ACTIVE_BACKEND_PATCHELF_REMOVE AND PATCHELF_EXECUTABLE)
  foreach(_needed IN LISTS TILELANG_ACTIVE_BACKEND_PATCHELF_REMOVE)
    set(_patchelf_remove_args "${_patchelf_remove_args} --remove-needed ${_needed}")
  endforeach()
  foreach(target IN LISTS TILELANG_OUTPUT_TARGETS)
    install(CODE "
      execute_process(
        COMMAND ${PATCHELF_EXECUTABLE}${_patchelf_remove_args}
          \"$<TARGET_FILE:${target}>\"
        WORKING_DIRECTORY \"${CMAKE_INSTALL_PREFIX}\"
        RESULT_VARIABLE patchelf_result
      )
      if(patchelf_result EQUAL 0)
        message(STATUS \"patchelf: removed dependencies from $<TARGET_FILE:${target}>\")
      else()
        message(WARNING \"patchelf failed for $<TARGET_FILE:${target}>\")
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
