# Metal backend: source files and build configuration.
#
# Metal source generation and operator lowering are pure C++ and can be used
# for cross-compilation on non-Apple hosts. Only optional runtime integration
# is gated by USE_METAL / APPLE below.
file(GLOB TILE_LANG_METAL_ALWAYS_SRCS
  src/metal/codegen/codegen_metal.cc
  src/metal/op/*.cc
  src/metal/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_METAL_ALWAYS_SRCS})

if(NOT USE_METAL)
  return()
endif()

message(STATUS "METAL Backend is enabled")

# FIXME: CIBW failed with backtrace, why???
set(TVM_FFI_USE_LIBBACKTRACE OFF)

if(NOT APPLE)
  # On non-Apple platforms USE_METAL=ON enables only codegen (Metal source
  # generation) without requiring the Metal/Foundation frameworks.
  message(STATUS "Metal backend on non-Apple: enabling codegen-only mode (no Metal runtime)")
  set(USE_METAL OFF)
  return()
endif()

file(GLOB TILE_LANG_METAL_ACTIVE_SRCS
  src/metal/codegen/rt_mod_metal.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_METAL_ACTIVE_SRCS})
