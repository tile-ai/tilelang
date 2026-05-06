# ROCm backend: toolchain, stub libraries, source files, and build configuration.
if(NOT USE_ROCM)
  return()
endif()

set(CMAKE_HIP_STANDARD 17)
include(${TVM_SOURCE}/cmake/utils/FindROCM.cmake)
find_rocm(${USE_ROCM})
add_compile_definitions(__HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__=1)

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
  add_library(hip_stub SHARED src/target/stubs/hip.cc)
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
  add_library(hiprtc_stub SHARED src/target/stubs/hiprtc.cc)
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
  src/target/codegen_hip.cc
  src/target/rt_mod_hip.cc
  src/backend/rocm/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})

# Register stubs for linking and install
if(TILELANG_USE_HIP_STUBS)
  set(TILELANG_ACTIVE_BACKEND_STUB_LINK hip_stub)
  set(TILELANG_ACTIVE_BACKEND_STUB_TARGETS hip_stub hiprtc_stub)
endif()
