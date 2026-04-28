function(tilelang_configure_apple_backend OUT_SRCS OUT_INCLUDES)
  if(NOT APPLE)
    message(STATUS "Metal backend on non-Apple: enabling codegen-only mode (no Metal runtime)")
    set(USE_METAL OFF PARENT_SCOPE)
    set(${OUT_SRCS} "" PARENT_SCOPE)
    set(${OUT_INCLUDES} "" PARENT_SCOPE)
    return()
  endif()

  set(_apple_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/target/rt_mod_metal.cc"
  )

  set(TVM_FFI_USE_LIBBACKTRACE OFF PARENT_SCOPE)
  set(${OUT_SRCS} ${_apple_srcs} PARENT_SCOPE)
  set(${OUT_INCLUDES} "" PARENT_SCOPE)
endfunction()
