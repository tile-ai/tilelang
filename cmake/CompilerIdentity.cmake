# The frontend and provider record the same source-derived identity. A frontend
# sdist carries the frozen projection; a provider sdist verifies all source bytes.
find_package(Python REQUIRED COMPONENTS Interpreter)
set(TILELANG_IDENTITY_DIR "${CMAKE_CURRENT_BINARY_DIR}/tilelang-identity")
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "${TILELANG_SOURCE_ROOT}/tilelang_packaging/compiler_manifest.py"
          --root "${TILELANG_SOURCE_ROOT}" --output "${TILELANG_IDENTITY_DIR}"
  RESULT_VARIABLE _tilelang_identity_result
  OUTPUT_VARIABLE _tilelang_identity_output
  ERROR_VARIABLE _tilelang_identity_error)
if(NOT _tilelang_identity_result EQUAL 0)
  message(FATAL_ERROR "Cannot identify the paired compiler sources: ${_tilelang_identity_error}")
endif()
include("${TILELANG_IDENTITY_DIR}/source-info.cmake")
message(STATUS "TileLang compiler provider: ${TILELANG_PROVIDER_VERSION}")
