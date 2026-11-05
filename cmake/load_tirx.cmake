# Select the compiler source explicitly.  An installed legacy TVM is never a
# fallback: all TileLang objects use the same TIRx and FFI object registry.
set(TVM_TIRX_SOURCE "${CMAKE_CURRENT_LIST_DIR}/../3rdparty/tvm-tirx"
    CACHE PATH "Source directory of the matching TIRx provider")
get_filename_component(TVM_TIRX_SOURCE "${TVM_TIRX_SOURCE}" ABSOLUTE)
if(NOT EXISTS "${TVM_TIRX_SOURCE}/include/tvm/ir/type.h" OR
   NOT EXISTS "${TVM_TIRX_SOURCE}/python/tvm_tirx/__init__.py")
  message(FATAL_ERROR "Initialize 3rdparty/tvm-tirx or set TVM_TIRX_SOURCE to the matching source tree")
endif()

set(TILELANG_FFI_TARGET_REF "3e906bc685ea45624bc80e8d7b7d86497d492af3")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${TVM_TIRX_SOURCE}/3rdparty/tvm-ffi/.git")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C
    "${TVM_TIRX_SOURCE}/3rdparty/tvm-ffi" rev-parse HEAD
    OUTPUT_VARIABLE _tilelang_ffi_sha OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _tilelang_ffi_result)
  if(NOT _tilelang_ffi_result EQUAL 0 OR
     NOT _tilelang_ffi_sha STREQUAL TILELANG_FFI_TARGET_REF)
    message(FATAL_ERROR "TileLang requires the fixed TIRx FFI ${TILELANG_FFI_TARGET_REF}; found ${_tilelang_ffi_sha}")
  endif()
endif()

message(STATUS "TileLang compiler source: ${TVM_TIRX_SOURCE}")
set(TILE_LANG_INCLUDES
  "${CMAKE_CURRENT_LIST_DIR}/../src"
  "${TVM_TIRX_SOURCE}/include"
  "${TVM_TIRX_SOURCE}/src")
