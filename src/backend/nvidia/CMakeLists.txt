function(tilelang_configure_nvidia_backend OUT_SRCS OUT_INCLUDES)
  set(CMAKE_CUDA_STANDARD 17 PARENT_SCOPE)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc" PARENT_SCOPE)
  add_compile_definitions("CUDA_MAJOR_VERSION=${CUDAToolkit_VERSION_MAJOR}")

  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH _tilelang_use_cuda)
  set(USE_CUDA "${_tilelang_use_cuda}" PARENT_SCOPE)

  if(TILELANG_USE_CUDA_STUBS)
    if(WIN32 AND NOT CYGWIN)
      message(FATAL_ERROR "TILELANG_USE_CUDA_STUBS=ON is not supported on Windows. "
                          "Please configure with -DTILELANG_USE_CUDA_STUBS=OFF.")
    endif()

    add_library(cuda_stub SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/target/stubs/cuda.cc")
    target_include_directories(cuda_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(cuda_stub PRIVATE TILELANG_CUDA_STUB_EXPORTS)
    target_link_libraries(cuda_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(cuda_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "cuda_stub"
    )

    add_library(cudart_stub SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/target/stubs/cudart.cc")
    target_include_directories(cudart_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(cudart_stub PRIVATE TILELANG_CUDART_STUB_EXPORTS)
    target_link_libraries(cudart_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(cudart_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "cudart_stub"
    )
    set(CUDA_CUDART_LIBRARY cudart_stub CACHE STRING "CUDART library to link against" FORCE)

    add_library(nvrtc_stub SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/target/stubs/nvrtc.cc")
    target_include_directories(nvrtc_stub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(nvrtc_stub PRIVATE TILELANG_NVRTC_STUB_EXPORTS)
    target_link_libraries(nvrtc_stub PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(nvrtc_stub PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      OUTPUT_NAME "nvrtc_stub"
    )
    set(CUDA_NVRTC_LIBRARY nvrtc_stub CACHE STRING "NVRTC library to link against" FORCE)
  endif()

  set(_nvidia_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/backend/nvidia/runtime/runtime.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/ptx.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_cuda.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_py.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_cutedsl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/rt_mod_cuda.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/rt_mod_cutedsl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/intrin_rule_cuda.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/annotate_warp_group_reg_alloc.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/fuse_mbarrier_arrive_expect_tx.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/inject_fence_proxy.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/inject_tcgen05_fence.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_blackwell_2sm.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_hopper_intrin.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_l2_persistent_annotation.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_ldg_stg.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_pdl.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_ptx_async_copy.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_shared_barrier.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/lower_shared_tmem.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/persist_threadblock.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/producer_consumer_ws.cc"
  )

  set(${OUT_SRCS} ${_nvidia_srcs} PARENT_SCOPE)
  set(${OUT_INCLUDES} ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
  link_directories(${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_DIR}/stubs)
endfunction()
