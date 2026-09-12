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
