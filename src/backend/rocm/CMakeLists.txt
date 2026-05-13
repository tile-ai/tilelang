# ROCm backend: toolchain, stub libraries, source files, and build configuration.
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
# need the runtime library at build time -- only the public HIP/HSA headers
# (e.g. from `apt install hip-runtime-amd-dev hsa-rocr-dev`, which are
# header packages and do not require an AMD GPU or driver).
#
# Set TILELANG_HIP_INCLUDE_DIR to a directory that contains hip/ and hsa/
# subtrees, or rely on the default /opt/rocm/include if those headers are
# installed there. Linking is satisfied by tilelang's hip_stub target.
if(NOT ROCM_FOUND AND TILELANG_USE_HIP_STUBS)
  set(_TL_HIP_INC "")
  if(TILELANG_HIP_INCLUDE_DIR)
    set(_TL_HIP_INC "${TILELANG_HIP_INCLUDE_DIR}")
  elseif(DEFINED ENV{TILELANG_HIP_INCLUDE_DIR})
    set(_TL_HIP_INC "$ENV{TILELANG_HIP_INCLUDE_DIR}")
  elseif(IS_DIRECTORY "/opt/rocm/include")
    set(_TL_HIP_INC "/opt/rocm/include")
  endif()

  if(_TL_HIP_INC AND IS_DIRECTORY "${_TL_HIP_INC}/hip" AND IS_DIRECTORY "${_TL_HIP_INC}/hsa")
    message(STATUS
      "ROCm runtime library not found on host; using HIP/HSA headers from "
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
      "ROCm runtime library not found on host and HIP/HSA headers were not "
      "found. Install hip-runtime-amd-dev and hsa-rocr-dev, or set "
      "TILELANG_HIP_INCLUDE_DIR to a directory containing hip/ and hsa/.")
  endif()
  unset(_TL_HIP_INC)
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
  add_library(hip_stub SHARED src/backend/rocm/stubs/hip.cc)
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
  add_library(hiprtc_stub SHARED src/backend/rocm/stubs/hiprtc.cc)
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

file(GLOB TILE_LANG_HIP_SRCS
  src/backend/rocm/codegen/codegen_hip.cc
  src/backend/rocm/codegen/rt_mod_hip.cc
  src/backend/rocm/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})

# Register stubs for linking and install
if(TILELANG_USE_HIP_STUBS)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK hip_stub)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_TARGETS hip_stub hiprtc_stub)
endif()
