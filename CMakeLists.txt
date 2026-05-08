# Learn a lot from the MLC - LLM Project
# https://github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.26)

# Detect CUDA toolkit: tries host installation first, then falls back to
# pip-installed packages (env WITH_PIP_CUDA_TOOLCHAIN or auto-detect).
# Must be included before project() so CMAKE_CUDA_COMPILER is set.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/FindPipCUDAToolkit.cmake)

# Capture this CMakeLists.txt's directory once so `function()` bodies below
# can reference files shipped with the project. CMake 4.x resolves
# CMAKE_CURRENT_LIST_DIR inside a function() to the caller's listfile, which
# breaks resource paths when helpers are invoked from included Find modules
# (e.g. cmake/pypi-z3/FindZ3.cmake).
set(TILELANG_CMAKELISTS_DIR "${CMAKE_CURRENT_LIST_DIR}")

project(TILE_LANG C CXX)

# MSVC-specific build configuration. Must run after project() so MSVC is
# defined, but before add_subdirectory(${TVM_SOURCE}) below so subprojects
# inherit these settings. CMake also sets MSVC=1 when the compiler is
# clang-cl (it simulates MSVC), so the same block applies to both toolchains.
if(MSVC)
  # Force CMP0141 NEW for every subdirectory regardless of its own
  # cmake_minimum_required. The bundled TVM declares 3.18; without this,
  # CMake injects /Zi into the legacy CMAKE_*_FLAGS_DEBUG default there
  # rather than via the generator expression, polluting cached flags and
  # also overriding the Embedded format we set just below.
  set(CMAKE_POLICY_DEFAULT_CMP0141 NEW)

  # Use /Z7 (Embedded) so debug info lives inside each .obj instead of a
  # side-channel vc140.pdb -- eliminates the parallel-write race (MSVC C1041)
  # and gives compiler caches a single artifact to hash. /Zi + /FS would
  # normally serialize PDB writes via mspdbsrv, but sccache wraps each cl.exe
  # in its own subprocess and observably breaks that coordination -- the build
  # still hits C1041 even with /FS on the command line. /FS is kept here as a
  # best-effort fallback for any consumer that re-enables /Zi locally; sccache
  # in front of cl.exe may neutralize it. clang-cl recognizes /FS as a no-op,
  # so it is safe to keep in the shared MSVC-compatible flag block.
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded"
      CACHE STRING "MSVC debug info format" FORCE)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/FS>)

  # Exceptions: cl.exe enables C++ exceptions by default (with a C4530
  # warning), but clang-cl disables them unless /EHsc is passed explicitly.
  # The bundled TVM adds /EHsc inside its own subdirectory, but TileLang's
  # native targets (tilelang_objs, cuda_stub, ...) live above that scope and
  # never inherited the flag, so clang-cl rejected `throw` in those TUs.
  # Apply /EHsc and /bigobj globally so every MSVC-compatible target picks
  # them up regardless of subdirectory layering.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>)

  # clang-cl-specific compatibility shims. clang-cl is invoked with
  # MSVC-like flags but needs a couple of extra knobs to compile TVM/TileLang
  # cleanly:
  #   * -Wno-unused-command-line-argument: silence noise from MSVC-only
  #     flags injected by CMake/TVM (e.g. /FS, /Zc:preprocessor, /MP).
  #   * -fdelayed-template-parsing: TVM headers declare std::vector<T>
  #     members where T is only forward-declared at that point (e.g.
  #     std::vector<tvm::tir::Stmt> in src/transform/arg_binder.h). cl.exe
  #     defers template parsing by default so the late stmt.h include in
  #     the .cc file resolves T before instantiation; modern clang parses
  #     templates eagerly and emits "incomplete type" errors. Re-enabling
  #     delayed parsing matches MSVC semantics without touching the headers.
  #   * -Wno-unknown-attributes: TVM's tvm-ffi headers spam
  #     [[msvc::forceinline]] which clang-cl recognizes only in some
  #     contexts; the warning floods build logs without changing codegen.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-command-line-argument>
      $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unknown-attributes>
      $<$<COMPILE_LANGUAGE:CXX>:-fdelayed-template-parsing>)
  endif()
endif()

function(tilelang_get_python_interpreter OUTPUT_VAR)
  if(DEFINED Python_EXECUTABLE AND NOT "${Python_EXECUTABLE}" STREQUAL "" AND EXISTS "${Python_EXECUTABLE}")
    set(${OUTPUT_VAR} "${Python_EXECUTABLE}" PARENT_SCOPE)
    return()
  endif()

  if(DEFINED Python3_EXECUTABLE AND NOT "${Python3_EXECUTABLE}" STREQUAL "" AND EXISTS "${Python3_EXECUTABLE}")
    set(${OUTPUT_VAR} "${Python3_EXECUTABLE}" PARENT_SCOPE)
    return()
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(${OUTPUT_VAR} "${Python3_EXECUTABLE}" PARENT_SCOPE)
endfunction()

function(tilelang_generate_windows_import_library DLL_PATH OUTPUT_VAR STEM)
  if(NOT WIN32)
    set(${OUTPUT_VAR} "${DLL_PATH}" PARENT_SCOPE)
    return()
  endif()

  if(NOT EXISTS "${DLL_PATH}")
    message(FATAL_ERROR "Cannot generate an import library from missing DLL: ${DLL_PATH}")
  endif()

  get_filename_component(_tilelang_dll_name "${DLL_PATH}" NAME_WE)
  set(_tilelang_import_dir "${CMAKE_BINARY_DIR}/windows-import-libs/${STEM}/${_tilelang_dll_name}")
  file(MAKE_DIRECTORY "${_tilelang_import_dir}")
  set(_tilelang_import_lib "${_tilelang_import_dir}/${STEM}.lib")
  if(NOT EXISTS "${_tilelang_import_lib}")
    tilelang_get_python_interpreter(_tilelang_python)
    find_program(_tilelang_dumpbin NAMES dumpbin dumpbin.exe)
    find_program(_tilelang_libexe NAMES lib lib.exe)
    if(NOT _tilelang_python OR NOT _tilelang_dumpbin OR NOT _tilelang_libexe)
      message(FATAL_ERROR "Could not find Python/dumpbin/lib.exe to generate an import library for ${DLL_PATH}")
    endif()

    set(_tilelang_def_file "${_tilelang_import_dir}/${STEM}.def")
    execute_process(
      COMMAND "${_tilelang_python}" "${TILELANG_CMAKELISTS_DIR}/cmake/generate_windows_import_lib.py"
              --dumpbin "${_tilelang_dumpbin}"
              --dll "${DLL_PATH}"
              --def "${_tilelang_def_file}"
      RESULT_VARIABLE _tilelang_def_result
    )
    if(NOT _tilelang_def_result EQUAL 0 OR NOT EXISTS "${_tilelang_def_file}")
      message(FATAL_ERROR "Failed to extract exports from ${DLL_PATH}")
    endif()

    set(_tilelang_machine "X64")
    if(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
      set(_tilelang_machine "X86")
    elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
      set(_tilelang_machine "ARM64")
    endif()
    execute_process(
      COMMAND "${_tilelang_libexe}" "/def:${_tilelang_def_file}" "/machine:${_tilelang_machine}" "/out:${_tilelang_import_lib}"
      RESULT_VARIABLE _tilelang_lib_result
    )
    if(NOT _tilelang_lib_result EQUAL 0 OR NOT EXISTS "${_tilelang_import_lib}")
      message(FATAL_ERROR "Failed to generate import library ${_tilelang_import_lib} from ${DLL_PATH}")
    endif()
  endif()

  set(${OUTPUT_VAR} "${_tilelang_import_lib}" PARENT_SCOPE)
endfunction()

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
  if(APPLE OR WIN32)
    # Passing configs like `ccache base_dir=/xxx cc ...` is supported
    # (likely) since ccache 4.x, which has been provided by homebrew.
    # Our Linux builder image (manylinux2014 & manylinux_2_28) still
    # provides ccache 3.x and do not support this form.
    # `cibuildwheel` uses fixed folder on Linux (`/project`) as working directory,
    # so cache would work without setting `base_dir`.
    set(CCACHE_PROGRAM "${CCACHE_PROGRAM};base_dir=${CMAKE_SOURCE_DIR};hash_dir=false")
  endif()

  # Make ccache hits survive uv's PEP 517 build isolation on Windows.
  #
  # uv materializes each build into a freshly-named tempdir under
  #   %LOCALAPPDATA%\Temp\.tmpXXXXXXXX\builds-v0\.tmpYYYYYYYY
  # The path appears (a) as `-external:I` flags pointing at NVIDIA cu13 / z3
  # and (b) inside cl.exe's preprocessed `#line` directives. Both feed into
  # ccache's hash, so 3rdparty targets miss on every rebuild.
  #
  # ccache `-I*` / `-D*` options are already excluded from preprocessor-mode
  # hashing because they reflect in the preprocessed output. But `-external:I`
  # is an MSVC-specific extension ccache treats as opaque, so we tell it
  # explicitly to ignore that whole class of flags. Combined with
  # `prefix_command_cpp` below to rewrite #line paths in the preprocessed
  # text, the resulting result-key is stable across .tmpXXX rotations.
  if(WIN32)
    set(CCACHE_PROGRAM "${CCACHE_PROGRAM};ignore_options=-external:I*")
  endif()

  # Plug a wrapper around cl.exe -P so .tmpXXX path components in the
  # preprocessor output are normalized before ccache hashes them. Pick a
  # stable Python interpreter: the wrapper script's path is itself baked
  # into ccache's hashed config, so a path containing .tmpXXX would defeat
  # caching. Prefer the project's persistent .venv; skip the wrapper if no
  # stable interpreter is available (caching still works for tilelang_objs
  # but 3rdparty cross-rebuild hits will be reduced).
  if(WIN32)
    set(_tilelang_pep517_python "")
    foreach(_candidate
        "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe"
        "${CMAKE_SOURCE_DIR}/venv/Scripts/python.exe")
      if(EXISTS "${_candidate}")
        set(_tilelang_pep517_python "${_candidate}")
        break()
      endif()
    endforeach()
    if(_tilelang_pep517_python)
      set(_tilelang_pep517_wrapper
          "${_tilelang_pep517_python} ${CMAKE_SOURCE_DIR}/cmake/ccache_strip_pep517.py")
      set(CCACHE_PROGRAM
          "${CCACHE_PROGRAM};prefix_command_cpp=${_tilelang_pep517_wrapper}")
      message(STATUS "  ccache: stripping PEP 517 .tmpXXX paths via ${_tilelang_pep517_wrapper}")
    else()
      message(STATUS "  ccache: no stable .venv python found; PEP 517 .tmpXXX preprocessor leakage may reduce hits")
    endif()
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
# lazy-loading the real libraries at runtime (dlopen/dlsym on POSIX,
# LoadLibrary/GetProcAddress on Windows).
set(_TILELANG_USE_CUDA_STUBS_DEFAULT ON)
option(TILELANG_USE_CUDA_STUBS
       "Use stub libraries (cuda/cudart/nvrtc) for portable wheels"
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
# Add TileLang's own src/ to include path so cross-directory includes
# can use paths relative to src/ (e.g. "target/utils.h", "op/builtin.h").
list(INSERT TILE_LANG_INCLUDES 0 "${CMAKE_CURRENT_SOURCE_DIR}/src")

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
  # intrin_rule doesn't have system dependency; always compiled regardless of backend
  src/backend/cuda/codegen/intrin_rule_cuda.cc
  src/backend/rocm/codegen/intrin_rule_hip.cc
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
    elseif(TILELANG_CUDA_TOOLKIT_AVAILABLE)
      message(STATUS "Enable CUDA support by default.")
      set(USE_CUDA ON)
    else()
      message(STATUS "CUDA toolkit not found; building without CUDA support by default.")
      set(USE_CUDA OFF)
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

# Strip the hardcoded /Zi PRIVATE option that tvm-ffi attaches in its
# tvm_ffi_add_msvc_flags() helper. It overrides our Embedded debug info
# format and forces a side-channel vc140.pdb that races under parallel
# builds (C1041) and breaks compiler caches. Done via target property
# rewrite so 3rdparty/tvm-ffi remains pristine.
if(MSVC)
  foreach(_tilelang_ffi_tgt IN ITEMS
      tvm_ffi_objs tvm_ffi_shared tvm_ffi_static tvm_ffi_testing)
    if(TARGET ${_tilelang_ffi_tgt})
      get_target_property(_tilelang_ffi_opts ${_tilelang_ffi_tgt} COMPILE_OPTIONS)
      if(_tilelang_ffi_opts)
        list(REMOVE_ITEM _tilelang_ffi_opts "/Zi" "-Zi")
        set_target_properties(${_tilelang_ffi_tgt}
          PROPERTIES COMPILE_OPTIONS "${_tilelang_ffi_opts}")
      endif()
    endif()
  endforeach()
endif()

# Provide the custom LogMessageImpl / LogFatalImpl implementation to TVM,
# since TVM_LOG_CUSTOMIZE=1 requires them to be supplied by the user.
target_sources(tvm_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/logging.cc")
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tvm_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()

# Resolve compile warnings in tvm
add_compile_definitions(DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})
if(MSVC)
  # Keep the wider project on its current baseline, but let TileLang's C++
  # sources use C++20 so designated initializers compile under MSVC.
  target_compile_features(tilelang_objs PRIVATE cxx_std_20)
endif()

# Set debug mode compile definitions
# Enable the TVM debug option, i.e., TVM_LOG_DEBUG
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Building TileLang with DEBUG mode")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
endif()

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_definitions(tilelang_objs PRIVATE TVM_LOG_CUSTOMIZE=1)
if(WIN32)
  # Some TVM headers define inline/template members behind TVM_DLL. MSVC
  # rejects those definitions when TVM_DLL expands to dllimport in TileLang's
  # consumer translation units, so consume them as ordinary declarations here.
  target_compile_definitions(tilelang_objs PRIVATE TVM_DLL=)
endif()
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tilelang_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()

set(TILELANG_OUTPUT_TARGETS tvm)
if(WIN32)
  # On Windows, TileLang subclasses a large number of non-exported TVM C++
  # classes. Building those objects into tvm.dll avoids an otherwise fragile
  # cross-DLL ABI boundary while keeping Python-side registration semantics.
  #
  # MSVC can still discard translation-unit-local static init triggers from
  # object-library inputs when those TUs are otherwise only referenced for
  # side effects (for example TVM_FFI_STATIC_INIT_BLOCK() registrations).
  # Archive the TileLang objects first, then force-include that archive into
  # tvm.dll so tl.* object/global registrations survive the final link.
  add_library(tilelang_archive STATIC $<TARGET_OBJECTS:tilelang_objs>)
  set_target_properties(tilelang_archive PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LINKER_LANGUAGE CXX
  )
  target_link_libraries(tvm PRIVATE tilelang_archive)
  target_link_options(tvm PRIVATE
    "/WHOLEARCHIVE:$<TARGET_FILE:tilelang_archive>"
  )
  # TVM is added with EXCLUDE_FROM_ALL above, but on Windows we install
  # tvm.dll as the only native artifact that ships TileLang symbols, so make
  # sure `ninja` (and `cmake --build`) actually builds it by default.
  set_property(TARGET tvm PROPERTY EXCLUDE_FROM_ALL FALSE)
else()
  add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
  target_link_libraries(tilelang PUBLIC tvm)

  # Place dev build outputs under build/lib for consistency
  set_target_properties(tilelang PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  )
  list(PREPEND TILELANG_OUTPUT_TARGETS tilelang)
endif()
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
if(WIN32)
  # Python headers on Windows auto-link pythonXY_d.lib in Debug builds, but
  # some managed CPython distributions only ship the stable import library we
  # already link below.
  target_compile_definitions(tilelang_cython_wrapper PRIVATE Py_NO_LINK_LIB=1)
  if(DEFINED Python_VERSION_MAJOR AND DEFINED Python_VERSION_MINOR)
    target_link_options(tilelang_cython_wrapper PRIVATE
      "/NODEFAULTLIB:python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}_d.lib")
  endif()
endif()

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
