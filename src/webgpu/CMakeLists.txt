# WebGPU backend: pure C++ operator lowering registrations.
#
# WebGPU currently has no optional native toolchain integration in TileLang, so
# its sources are always safe to compile.

file(GLOB TILE_LANG_WEBGPU_SRCS
  src/webgpu/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_WEBGPU_SRCS})
