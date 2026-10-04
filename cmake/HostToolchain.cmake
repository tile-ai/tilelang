# Host compiler setup is independent of the selected device SDK. Visual Studio
# generators manage their environment; Ninja needs it during configure and build.
include_guard(GLOBAL)

function(tilelang_get_python_interpreter OUTPUT_VAR)
  foreach(_var IN ITEMS Python_EXECUTABLE Python3_EXECUTABLE PYTHON_EXECUTABLE)
    if(DEFINED ${_var} AND EXISTS "${${_var}}")
      set(${OUTPUT_VAR} "${${_var}}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(${OUTPUT_VAR} "${Python3_EXECUTABLE}" PARENT_SCOPE)
endfunction()

if(NOT WIN32 OR NOT CMAKE_GENERATOR MATCHES "Ninja")
  return()
endif()

tilelang_get_python_interpreter(_tilelang_host_python)
if(NOT CMAKE_MAKE_PROGRAM)
  get_filename_component(_tilelang_python_bin "${_tilelang_host_python}" DIRECTORY)
  find_program(CMAKE_MAKE_PROGRAM NAMES ninja ninja.exe HINTS "${_tilelang_python_bin}")
endif()

# A caller-supplied toolchain or GNU-style compiler owns its setup. Never replace
# explicit CMake compilers, CC/CXX, launchers, or a developer prompt's environment.
if(CMAKE_TOOLCHAIN_FILE)
  return()
endif()
foreach(_compiler IN ITEMS "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}" "$ENV{CC}" "$ENV{CXX}")
  if(_compiler AND NOT _compiler MATCHES "(^|[/\\ ;])(cl|clang-cl)(\\.exe)?(\"|$)")
    return()
  endif()
endforeach()

set(_tilelang_host_environment "${CMAKE_BINARY_DIR}/tilelang-host-environment.json")
if(DEFINED ENV{VSCMD_VER} AND DEFINED ENV{VCINSTALLDIR} AND
   DEFINED ENV{INCLUDE} AND DEFINED ENV{LIB})
  set(_tilelang_developer_shell ON)
endif()
execute_process(
  COMMAND "${_tilelang_host_python}" "${CMAKE_CURRENT_LIST_DIR}/../tilelang/_host_toolchain.py"
          --environment "${_tilelang_host_environment}"
  OUTPUT_VARIABLE _tilelang_host_json
  ERROR_VARIABLE _tilelang_host_error
  RESULT_VARIABLE _tilelang_host_result
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _tilelang_host_result EQUAL 0)
  message(FATAL_ERROR "Windows host toolchain setup failed: ${_tilelang_host_error}")
endif()
string(JSON _tilelang_host_compiler GET "${_tilelang_host_json}" compiler)
string(JSON _tilelang_env_count LENGTH "${_tilelang_host_json}" environment)
math(EXPR _tilelang_env_last "${_tilelang_env_count} - 1")
foreach(_index RANGE ${_tilelang_env_last})
  string(JSON _name MEMBER "${_tilelang_host_json}" environment ${_index})
  string(JSON _value GET "${_tilelang_host_json}" environment "${_name}")
  set(ENV{${_name}} "${_value}")
endforeach()
foreach(_lang IN ITEMS C CXX)
  if(NOT CMAKE_${_lang}_COMPILER AND NOT _tilelang_developer_shell)
    if((_lang STREQUAL "C" AND NOT DEFINED ENV{CC}) OR
       (_lang STREQUAL "CXX" AND NOT DEFINED ENV{CXX}))
      set(CMAKE_${_lang}_COMPILER "${_tilelang_host_compiler}" CACHE FILEPATH "Host ${_lang} compiler")
    endif()
  endif()
endforeach()

# Configure's environment does not survive a later `cmake --build`. Restore the
# captured SDK paths for each compile/link command without modifying cached flags.
# This also composes with a user launcher or the project's compiler cache.
function(tilelang_enable_host_launchers)
  foreach(_lang IN ITEMS C CXX)
    foreach(_step IN ITEMS COMPILER LINKER)
      set(_launcher "${_tilelang_host_python};${CMAKE_CURRENT_FUNCTION_LIST_DIR}/host_toolchain_launcher.py;${_tilelang_host_environment}")
      if(CMAKE_${_lang}_${_step}_LAUNCHER)
        list(APPEND _launcher ${CMAKE_${_lang}_${_step}_LAUNCHER})
      endif()
      set(CMAKE_${_lang}_${_step}_LAUNCHER
          "${_launcher}"
          PARENT_SCOPE)
    endforeach()
  endforeach()
endfunction()
