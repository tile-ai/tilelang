# CUDA SDK discovery runs after project(), independently of the host toolchain.
# Explicitly disabled backends must not inspect or materialize a CUDA SDK.
if(DEFINED USE_CUDA AND NOT USE_CUDA)
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "disabled" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
elseif(NOT DEFINED USE_CUDA AND DEFINED ENV{USE_CUDA} AND NOT "$ENV{USE_CUDA}")
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "disabled" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
endif()

# --- Try host CUDA first ---
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  set(TILELANG_CUDA_TOOLKIT_AVAILABLE ON CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
  set(TILELANG_CUDA_TOOLKIT_SOURCE "host" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)
  return()
endif()

set(TILELANG_CUDA_TOOLKIT_AVAILABLE OFF CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
set(TILELANG_CUDA_TOOLKIT_SOURCE "none" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)

set(_PIP_CUDA_PYTHON_CANDIDATES "")

macro(_tilelang_append_python_candidate _candidate)
  if(DEFINED ${_candidate})
    if(NOT "${${_candidate}}" STREQUAL "" AND EXISTS "${${_candidate}}")
      list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${${_candidate}}")
    endif()
  endif()
endmacro()

foreach(_candidate_var IN ITEMS Python3_EXECUTABLE Python_EXECUTABLE PYTHON_EXECUTABLE)
  if(DEFINED ${_candidate_var})
    _tilelang_append_python_candidate(${_candidate_var})
  endif()
endforeach()

if(DEFINED ENV{VIRTUAL_ENV})
  if(WIN32)
    set(_tilelang_virtualenv_python "$ENV{VIRTUAL_ENV}/Scripts/python.exe")
  else()
    set(_tilelang_virtualenv_python "$ENV{VIRTUAL_ENV}/bin/python")
  endif()
  if(EXISTS "${_tilelang_virtualenv_python}")
    list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_tilelang_virtualenv_python}")
  endif()
endif()

if(DEFINED ENV{UV_PROJECT_ENVIRONMENT})
  if(WIN32)
    set(_tilelang_uv_python "$ENV{UV_PROJECT_ENVIRONMENT}/Scripts/python.exe")
  else()
    set(_tilelang_uv_python "$ENV{UV_PROJECT_ENVIRONMENT}/bin/python")
  endif()
  if(EXISTS "${_tilelang_uv_python}")
    list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_tilelang_uv_python}")
  endif()
endif()

foreach(_venv_dir IN ITEMS ".venv" "venv")
  if(WIN32)
    set(_tilelang_local_python "${CMAKE_SOURCE_DIR}/${_venv_dir}/Scripts/python.exe")
  else()
    set(_tilelang_local_python "${CMAKE_SOURCE_DIR}/${_venv_dir}/bin/python")
  endif()
  if(EXISTS "${_tilelang_local_python}")
    list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_tilelang_local_python}")
  endif()
endforeach()

find_program(_PIP_CUDA_PYTHON_FALLBACK NAMES python3 python)
if(_PIP_CUDA_PYTHON_FALLBACK)
  list(APPEND _PIP_CUDA_PYTHON_CANDIDATES "${_PIP_CUDA_PYTHON_FALLBACK}")
endif()

list(REMOVE_DUPLICATES _PIP_CUDA_PYTHON_CANDIDATES)
if(NOT _PIP_CUDA_PYTHON_CANDIDATES)
  return()
endif()

function(_tilelang_run_find_pip_cuda _out_json _out_python)
  set(_result_json "")
  set(_result_python "")

  foreach(_python IN LISTS _PIP_CUDA_PYTHON_CANDIDATES)
    if(ARGC GREATER 2)
      execute_process(
        COMMAND "${_python}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/find_pip_cuda.py" "${ARGV2}"
        OUTPUT_VARIABLE _candidate_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _candidate_result
      )
    else()
      execute_process(
        COMMAND "${_python}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/find_pip_cuda.py"
        OUTPUT_VARIABLE _candidate_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _candidate_result
      )
    endif()

    if(_candidate_result EQUAL 0)
      set(_result_json "${_candidate_output}")
      set(_result_python "${_python}")
      break()
    endif()
  endforeach()

  set(${_out_json} "${_result_json}" PARENT_SCOPE)
  set(${_out_python} "${_result_python}" PARENT_SCOPE)
endfunction()

# --- Strategy 1: explicit path via env var ---
if(DEFINED ENV{WITH_PIP_CUDA_TOOLCHAIN})
  _tilelang_run_find_pip_cuda(_PIP_CUDA_OUTPUT _PIP_CUDA_PYTHON_EXE "$ENV{WITH_PIP_CUDA_TOOLCHAIN}")
  if(NOT _PIP_CUDA_OUTPUT)
    message(FATAL_ERROR
      "FindPipCUDAToolkit: WITH_PIP_CUDA_TOOLCHAIN is set to '$ENV{WITH_PIP_CUDA_TOOLCHAIN}' "
      "but no pip-installed CUDA toolkit could be resolved from that path")
  endif()
  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: using env WITH_PIP_CUDA_TOOLCHAIN=${_PIP_CUDA_ROOT}")
else()
  # --- Strategy 2: auto-detect from current Python env ---
  _tilelang_run_find_pip_cuda(_PIP_CUDA_OUTPUT _PIP_CUDA_PYTHON_EXE)
  if(NOT _PIP_CUDA_OUTPUT)
    message(STATUS "FindPipCUDAToolkit: pip-installed CUDA toolkit not found")
    return()
  endif()

  string(JSON _PIP_CUDA_ROOT GET "${_PIP_CUDA_OUTPUT}" "root")
  message(STATUS "FindPipCUDAToolkit: auto-detected from Python environment via ${_PIP_CUDA_PYTHON_EXE}")
endif()

# --- Common pip-CUDA setup ---
string(JSON _PIP_CUDA_NVCC GET "${_PIP_CUDA_OUTPUT}" "nvcc")
string(JSON _PIP_CUDA_LIBRARY_DIR GET "${_PIP_CUDA_OUTPUT}" "library_dir")

if(NOT CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER "${_PIP_CUDA_NVCC}" CACHE FILEPATH "CUDA compiler (from pip)" FORCE)
endif()
if(NOT CUDAToolkit_ROOT)
  set(CUDAToolkit_ROOT "${_PIP_CUDA_ROOT}" CACHE PATH "CUDA toolkit root (from pip)" FORCE)
endif()
set(TILELANG_CUDA_TOOLKIT_AVAILABLE ON CACHE INTERNAL "Whether a CUDA toolkit is available" FORCE)
set(TILELANG_CUDA_TOOLKIT_SOURCE "pip" CACHE INTERNAL "How TileLang discovered CUDA" FORCE)

list(APPEND CMAKE_PROGRAM_PATH "${_PIP_CUDA_ROOT}/bin")
if(WIN32)
  list(APPEND CMAKE_PROGRAM_PATH "${_PIP_CUDA_ROOT}/bin/x86_64" "${_PIP_CUDA_ROOT}/nvvm/bin")
  list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_LIBRARY_DIR}")
  set(ENV{PATH} "${_PIP_CUDA_ROOT}/bin;${_PIP_CUDA_ROOT}/bin/x86_64;${_PIP_CUDA_ROOT}/nvvm/bin;$ENV{PATH}")
else()
  list(APPEND CMAKE_LIBRARY_PATH "${_PIP_CUDA_ROOT}/lib/stubs" "${_PIP_CUDA_LIBRARY_DIR}")
endif()

message(STATUS "FindPipCUDAToolkit: using pip-installed CUDA toolkit")
message(STATUS "  nvcc: ${_PIP_CUDA_NVCC}")
message(STATUS "  root: ${CUDAToolkit_ROOT}")
