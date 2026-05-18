if(NOT DEFINED TILELANG_DYLIB_PATH)
  message(FATAL_ERROR "TILELANG_DYLIB_PATH is required")
endif()

if(NOT DEFINED TILELANG_DYLIB_FROM)
  message(FATAL_ERROR "TILELANG_DYLIB_FROM is required")
endif()

if(NOT DEFINED TILELANG_DYLIB_TO)
  message(FATAL_ERROR "TILELANG_DYLIB_TO is required")
endif()

if(NOT EXISTS "${TILELANG_DYLIB_PATH}")
  message(FATAL_ERROR "dylib does not exist: ${TILELANG_DYLIB_PATH}")
endif()

execute_process(
  COMMAND otool -L "${TILELANG_DYLIB_PATH}"
  OUTPUT_VARIABLE _tilelang_otool_output
  RESULT_VARIABLE _tilelang_otool_result
)

if(NOT _tilelang_otool_result EQUAL 0)
  message(FATAL_ERROR "Failed to inspect ${TILELANG_DYLIB_PATH}")
endif()

string(REPLACE "." "\\." _tilelang_dylib_from_regex "${TILELANG_DYLIB_FROM}")
if(_tilelang_otool_output MATCHES "\n[ \t]*${_tilelang_dylib_from_regex} \\(")
  execute_process(
    COMMAND /usr/bin/install_name_tool
      -change "${TILELANG_DYLIB_FROM}" "${TILELANG_DYLIB_TO}" "${TILELANG_DYLIB_PATH}"
    RESULT_VARIABLE _tilelang_install_name_tool_result
  )
  if(NOT _tilelang_install_name_tool_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to rewrite ${TILELANG_DYLIB_FROM} to ${TILELANG_DYLIB_TO} in ${TILELANG_DYLIB_PATH}")
  endif()
endif()
