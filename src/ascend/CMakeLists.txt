# Ascend backend: source files, stubs, and build configuration.
#
# Ascend target helpers, layout registrations and frontend frames are pure C++
# registrations used by common code (TargetIsAscend is called from common
# transforms via backend/common/target_utils.h; tilelang.layout registers the
# tl.AscendFractalLayout object type at import time). Compile them regardless
# of USE_ASCEND.
#
# src/ascend/ir.cc belongs here for the same reason: tilelang.ascend.language
# binds tl.CubeFrame / tl.VectorFrame / tl.SimtVFFrame / tl.SimdVFFrame with
# @register_object at import time, and tilelang/__init__.py imports
# tilelang.ascend unconditionally, so those object types must exist in every
# build -- a missing type key raises "Cannot find object type index" and breaks
# "import tilelang" outright. The frames carry no CANN dependency.
file(GLOB TILE_LANG_ASCEND_ALWAYS_SRCS
  src/ascend/target_utils.cc
  src/ascend/ir.cc
  src/ascend/layout/ascend_layouts.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_ASCEND_ALWAYS_SRCS})

if(NOT USE_ASCEND)
  return()
endif()

message(STATUS "Ascend Backend is enabled")

# Native profiling is available when the CANN profiling headers are installed.
# Detect them on each configure without adding a user-facing profiling build
# option; builds without CANN simply skip the reporting path.
set(_tilelang_msprof_suffixes
  include/experiment/msprof
  include/profiling
  include/experiment/profiling
  pkg_inc/profiling
  include/experiment
  include
)

set(_tilelang_msprof_arch_suffixes)

foreach(_suffix IN LISTS _tilelang_msprof_suffixes)
  list(APPEND _tilelang_msprof_arch_suffixes
    "${CMAKE_SYSTEM_PROCESSOR}-linux/${_suffix}"
  )
endforeach()

find_path(
  _tilelang_msprof_include_dir
  aprof_pub.h
  HINTS
    "${ASCEND_HOME_PATH}"
    "${ASCEND_TOOLKIT_HOME}"
    "${ASCEND_HOME}"
    "$ENV{ASCEND_HOME_PATH}"
    "$ENV{ASCEND_TOOLKIT_HOME}"
    "$ENV{ASCEND_HOME}"
    /usr/local/Ascend/ascend-toolkit/latest
  PATH_SUFFIXES
    ${_tilelang_msprof_suffixes}
    ${_tilelang_msprof_arch_suffixes}
  NO_CACHE
)

set(_tilelang_ascend_profiler 0)

if(_tilelang_msprof_include_dir)
  set(_tilelang_ascend_profiler 1)

  set_property(
    SOURCE src/ascend/runtime/ascend_module.cc
    APPEND PROPERTY
    INCLUDE_DIRECTORIES "${_tilelang_msprof_include_dir}"
  )

  message(STATUS "Ascend CANN profiling enabled")
else()
  message(STATUS
    "CANN aprof_pub.h was not found; building Ascend without native "
    "profiling. Set ASCEND_HOME_PATH to the CANN SDK root and reconfigure "
    "to enable CANN Node reporting."
  )
endif()

set_property(
  SOURCE src/ascend/runtime/ascend_module.cc
  APPEND PROPERTY
  COMPILE_DEFINITIONS TILELANG_ASCEND_PROFILER=${_tilelang_ascend_profiler}
)

unset(_tilelang_ascend_profiler)
unset(_tilelang_msprof_include_dir)
unset(_tilelang_msprof_suffixes)
unset(_tilelang_msprof_arch_suffixes)
unset(_suffix)

# The codegen emits AscendC source text and carries no CANN build-time
# dependency; the runtime module links against the ascendcl stub below (or the
# real libascendcl when stubs are disabled), so USE_ASCEND=ON is safe on any
# non-Windows host.
file(GLOB TILE_LANG_ASCEND_ACTIVE_SRCS
  src/ascend/*.cc
  src/ascend/codegen/*.cc
  src/ascend/layout/*.cc
  src/ascend/op/*.cc
  src/ascend/runtime/*.cc
  src/ascend/transform/*.cc
  src/ascend/transform/auto_schedule/*.cc
)
list(REMOVE_ITEM TILE_LANG_ASCEND_ACTIVE_SRCS ${TILE_LANG_ASCEND_ALWAYS_SRCS})
list(APPEND TILE_LANG_SRCS ${TILE_LANG_ASCEND_ACTIVE_SRCS})

if(TILELANG_USE_ASCEND_STUBS)
  # ==========================================================================
  # ascendcl Stub Library (libstub_ascendcl.so)
  # ==========================================================================
  # Provides drop-in replacements for the CANN runtime entrypoints used by the
  # Ascend runtime module by lazily loading libascendcl.so at runtime
  # (preferring a copy already loaded by torch_npu). This keeps wheels free of
  # a hard DT_NEEDED on libascendcl.so so they can be imported without CANN.
  # ==========================================================================
  add_library(ascendcl_stub SHARED src/ascend/stubs/ascendcl.cc)
  target_compile_definitions(ascendcl_stub PRIVATE TILELANG_ASCENDCL_STUB_EXPORTS)
  target_link_libraries(ascendcl_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(ascendcl_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    # Keep the stub filename distinct from the real CANN library, matching the
    # CUDA/ROCm stub naming convention.
    OUTPUT_NAME "stub_ascendcl"
  )

  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK ascendcl_stub)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_TARGETS ascendcl_stub)
elseif(NOT WIN32)
  # Stubs disabled: link the real CANN runtime directly (on-hardware builds).
  find_library(TILELANG_ASCENDCL_LIBRARY ascendcl
    HINTS $ENV{ASCEND_HOME_PATH} $ENV{ASCEND_TOOLKIT_HOME}
          /usr/local/Ascend/ascend-toolkit/latest
    PATH_SUFFIXES lib64 runtime/lib64)
  if(NOT TILELANG_ASCENDCL_LIBRARY)
    message(FATAL_ERROR
      "TILELANG_USE_ASCEND_STUBS=OFF requires libascendcl, but it was not "
      "found. Install the CANN toolkit and set ASCEND_HOME_PATH (or "
      "ASCEND_TOOLKIT_HOME), or keep TILELANG_USE_ASCEND_STUBS=ON.")
  endif()
  message(STATUS "Linking Ascend runtime against ${TILELANG_ASCENDCL_LIBRARY}")
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK ${TILELANG_ASCENDCL_LIBRARY})
endif()
