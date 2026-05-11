# Metal backend: source files and build configuration.
if(NOT USE_METAL)
  return()
endif()

if(NOT APPLE)
  # On non-Apple platforms USE_METAL=ON enables only codegen (Metal source
  # generation) without requiring the Metal/Foundation frameworks.
  message(STATUS "Metal backend on non-Apple: enabling codegen-only mode (no Metal runtime)")
  set(USE_METAL OFF)
endif()

file(GLOB TILE_LANG_METAL_SRCS
  src/backend/metal/codegen/rt_mod_metal.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_METAL_SRCS})
# FIXME: CIBW failed with backtrace, why???
set(TVM_FFI_USE_LIBBACKTRACE OFF)
