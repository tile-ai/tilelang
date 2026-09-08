# Ascend backend: source files and build configuration.
#
# NOTE: Unlike the CUDA/ROCm/Metal backends, Ascend has no `USE_ASCEND` guard
# yet, so its sources are compiled unconditionally. The codegen emits AscendC
# source text and carries no CANN build-time dependency, so it is safe to build
# on any host. A `USE_ASCEND` toggle can be added later without moving sources.

file(GLOB TILE_LANG_ASCEND_SRCS
  src/ascend/*.cc
  src/ascend/codegen/*.cc
  src/ascend/layout/*.cc
  src/ascend/op/*.cc
  src/ascend/runtime/*.cc
  src/ascend/transform/*.cc
  src/ascend/transform/auto_schedule/*.cc
)
# PTO is an Ascend codegen backend and reuses CodeGenTileLangPY. Keep this
# source compiled even when the CUDA backend itself is disabled.
list(APPEND TILE_LANG_ASCEND_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/codegen/codegen_py.cc"
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_ASCEND_SRCS})
