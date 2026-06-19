# ROCm backend: source files, toolchain, stubs, and build configuration.
#
# ROCm target helpers and intrinsic rules are pure C++ registrations used by
# common code and source generation. Compile them regardless of USE_ROCM.
file(GLOB TILE_LANG_ROCM_ALWAYS_SRCS
  src/rocm/codegen/intrin_rule_hip.cc
  src/rocm/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_ROCM_ALWAYS_SRCS})

if(NOT USE_ROCM)
  return()
endif()

message(STATUS "ROCM Backend is enabled")

set(CMAKE_HIP_STANDARD 17)
include(${TVM_SOURCE}/cmake/utils/FindROCM.cmake)
find_rocm(${USE_ROCM})
add_compile_definitions(__HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__=1)

# Fallback for build hosts without a ROCm runtime install (e.g. an NV-only
# machine producing a cross-target wheel). TVM's find_rocm() only succeeds
# when libamdhip64 is present, but with TILELANG_USE_HIP_STUBS=ON we don't
# need the runtime library at build time -- only the public HIP headers.
#
# Resolution order (first match wins):
#   1. -DTILELANG_HIP_INCLUDE_DIR=<dir> (explicit override)
#   2. $ENV{TILELANG_HIP_INCLUDE_DIR}
#   3. /opt/rocm/include (if a system ROCm install exists)
#   4. 3rdparty/hip-headers/include (vendored, default fallback)
#
# A minimal hsa/hsa.h is vendored alongside the HIP headers because TVM's
# src/runtime/rocm/rocm_device_api.cc unconditionally #includes <hsa/hsa.h>.
# Only two HSA symbols (hsa_init / hsa_shut_down) are actually referenced;
# both are exported by hip_stub and lazy-loaded from libhsa-runtime64 at run
# time, so no HSA library is linked.
if(NOT ROCM_FOUND AND TILELANG_USE_HIP_STUBS)
  set(_TL_VENDORED_HIP_INC "${CMAKE_SOURCE_DIR}/3rdparty/hip-headers/include")
  set(_TL_HIP_INC "")
  if(TILELANG_HIP_INCLUDE_DIR)
    set(_TL_HIP_INC "${TILELANG_HIP_INCLUDE_DIR}")
  elseif(DEFINED ENV{TILELANG_HIP_INCLUDE_DIR})
    set(_TL_HIP_INC "$ENV{TILELANG_HIP_INCLUDE_DIR}")
  elseif(IS_DIRECTORY "/opt/rocm/include/hip")
    set(_TL_HIP_INC "/opt/rocm/include")
  elseif(IS_DIRECTORY "${_TL_VENDORED_HIP_INC}/hip")
    set(_TL_HIP_INC "${_TL_VENDORED_HIP_INC}")
  endif()

  if(_TL_HIP_INC AND IS_DIRECTORY "${_TL_HIP_INC}/hip")
    message(STATUS
      "ROCm runtime library not found on host; using HIP headers from "
      "${_TL_HIP_INC} with hip_stub for linking. The resulting wheel still "
      "requires a real ROCm runtime to be present at execution time.")
    set(ROCM_FOUND TRUE)
    set(ROCM_INCLUDE_DIRS "${_TL_HIP_INC}")
    set(ROCM_HIPHCC_LIBRARY hip_stub CACHE STRING
        "HIP runtime library to link against" FORCE)
    set(ROCM_HSA_LIBRARY ROCM_HSA_LIBRARY-NOTFOUND CACHE STRING
        "HSA runtime library to link against" FORCE)
  else()
    message(STATUS
      "ROCm runtime library not found on host and no HIP headers could be "
      "located. Set -DTILELANG_HIP_INCLUDE_DIR=<dir> to a directory "
      "containing a `hip/` subtree, or restore the vendored headers under "
      "3rdparty/hip-headers/include/hip.")
  endif()
  unset(_TL_HIP_INC)
  unset(_TL_VENDORED_HIP_INC)
endif()

if(TILELANG_USE_HIP_STUBS)
  if(WIN32 AND NOT CYGWIN)
    message(FATAL_ERROR "TILELANG_USE_HIP_STUBS=ON is not supported on Windows. "
                        "Please configure with -DTILELANG_USE_HIP_STUBS=OFF.")
  endif()

  # ============================================================================
  # HIP Stub Library (libhip_stub.so)
  # ============================================================================
  # This library provides drop-in replacements for HIP runtime/module APIs by
  # lazily loading libamdhip64.so at runtime.
  #
  # It also provides minimal HSA wrappers (hsa_init / hsa_shut_down) to avoid a
  # hard DT_NEEDED dependency on libhsa-runtime64 in ROCm-enabled wheels.
  # ============================================================================
  add_library(hip_stub SHARED src/rocm/stubs/hip.cc)
  target_include_directories(hip_stub PRIVATE ${ROCM_INCLUDE_DIRS})
  target_compile_definitions(hip_stub PRIVATE TILELANG_HIP_STUB_EXPORTS)
  target_link_libraries(hip_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(hip_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "hip_stub"
  )

  # ============================================================================
  # HIPRTC Stub Library (libhiprtc_stub.so)
  # ============================================================================
  # This library provides a minimal HIPRTC API surface and lazily loads
  # libhiprtc.so at runtime.
  # ============================================================================
  add_library(hiprtc_stub SHARED src/rocm/stubs/hiprtc.cc)
  target_include_directories(hiprtc_stub PRIVATE ${ROCM_INCLUDE_DIRS})
  target_compile_definitions(hiprtc_stub PRIVATE TILELANG_HIPRTC_STUB_EXPORTS)
  target_link_libraries(hiprtc_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(hiprtc_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "hiprtc_stub"
  )

  # Make TVM link against our HIP stub instead of the real libamdhip64.so.
  #
  # NOTE: TVM's `find_rocm()` calls `find_library(ROCM_HIPHCC_LIBRARY amdhip64 ...)`.
  # `find_library()` will not override an already-cached variable, so setting it
  # here ensures TVM doesn't record a DT_NEEDED on libamdhip64.
  set(ROCM_HIPHCC_LIBRARY hip_stub CACHE STRING "HIP runtime library to link against" FORCE)

  # Prevent TVM from recording a DT_NEEDED on libhsa-runtime64.
  # The few HSA entrypoints used by TVM are stubbed by hip_stub and resolved
  # lazily when available.
  set(ROCM_HSA_LIBRARY ROCM_HSA_LIBRARY-NOTFOUND CACHE STRING
      "HSA runtime library to link against" FORCE)
endif()

file(GLOB TILE_LANG_ROCM_ACTIVE_SRCS
  src/rocm/codegen/codegen_hip.cc
  src/rocm/codegen/rt_mod_hip.cc
  src/rocm/op/*.cc
  src/rocm/transform/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_ROCM_ACTIVE_SRCS})
list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})

# Register stubs for linking and install
if(TILELANG_USE_HIP_STUBS)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK hip_stub)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_TARGETS hip_stub hiprtc_stub)
endif()
