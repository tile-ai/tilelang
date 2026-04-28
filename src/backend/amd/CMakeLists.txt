function(tilelang_configure_amd_backend OUT_SRCS OUT_INCLUDES)
  set(CMAKE_HIP_STANDARD 17 PARENT_SCOPE)
  include("${TVM_SOURCE}/cmake/utils/FindROCM.cmake")
  find_rocm(${USE_ROCM})
  add_compile_definitions(__HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__=1)

  if(TILELANG_USE_HIP_STUBS)
    if(WIN32 AND NOT CYGWIN)
      message(FATAL_ERROR "TILELANG_USE_HIP_STUBS=ON is not supported on Windows. "
                          "Please configure with -DTILELANG_USE_HIP_STUBS=OFF.")
    endif()

    add_library(hip_stub SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/target/stubs/hip.cc")
    target_include_directories(hip_stub PRIVATE ${ROCM_INCLUDE_DIRS})
    target_compile_definitions(hip_stub PRIVATE TILELANG_HIP_STUB_EXPORTS)
    target_link_libraries(hip_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(hip_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "hip_stub"
    )

    add_library(hiprtc_stub SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/target/stubs/hiprtc.cc")
    target_include_directories(hiprtc_stub PRIVATE ${ROCM_INCLUDE_DIRS})
    target_compile_definitions(hiprtc_stub PRIVATE TILELANG_HIPRTC_STUB_EXPORTS)
    target_link_libraries(hiprtc_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(hiprtc_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "hiprtc_stub"
    )

    set(ROCM_HIPHCC_LIBRARY hip_stub CACHE STRING "HIP runtime library to link against" FORCE)
    set(ROCM_HSA_LIBRARY ROCM_HSA_LIBRARY-NOTFOUND CACHE STRING
        "HSA runtime library to link against" FORCE)
  endif()

  set(_amd_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_hip.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/rt_mod_hip.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/intrin_rule_hip.cc"
  )

  set(${OUT_SRCS} ${_amd_srcs} PARENT_SCOPE)
  set(${OUT_INCLUDES} ${ROCM_INCLUDE_DIRS} PARENT_SCOPE)
endfunction()
