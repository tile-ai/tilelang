# CPU backend: source files that are always safe to compile.
#
# CPU target helpers, operator lowering, and C source codegen have no optional
# system dependencies. Keep them owned by the CPU backend instead of listing
# them in the top-level CMake source block.

file(GLOB TILE_LANG_CPU_SRCS
  src/cpu/codegen/codegen_c.cc
  src/cpu/codegen/rt_mod_c.cc
  src/cpu/op/*.cc
  src/cpu/target_utils.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_CPU_SRCS})
