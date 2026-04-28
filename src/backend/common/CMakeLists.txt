function(tilelang_collect_common_backend OUT_SRCS OUT_INCLUDES)
  file(GLOB _common_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/layout/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transform/common/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/op/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_c_host.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/codegen_c.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/rt_mod_c.cc"
  )

  set(_nvidia_transform_srcs
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
  list(REMOVE_ITEM _common_srcs ${_nvidia_transform_srcs})

  list(APPEND _common_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/error_helpers.cc"
  )

  set(${OUT_SRCS} ${_common_srcs} PARENT_SCOPE)
  set(${OUT_INCLUDES} "" PARENT_SCOPE)
endfunction()
