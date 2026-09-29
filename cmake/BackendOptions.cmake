# Resolve each backend independently. Linux builds include ROCm and Ascend by
# default: vendored/local headers and runtime stubs avoid a build-time dependency
# on either SDK. CUDA still needs a toolkit, while macOS defaults to Metal.
set(TILELANG_BACKENDS CUDA ROCM METAL LLVM ASCEND)

set(TILELANG_BACKEND_DOC_CUDA "Enable CUDA backend (ON/OFF/or CUDA SDK path)")
set(TILELANG_BACKEND_DOC_ROCM "Enable ROCm backend (ON/OFF/or ROCm SDK path)")
set(TILELANG_BACKEND_DOC_METAL "Enable Metal backend")
set(TILELANG_BACKEND_DOC_LLVM "Enable LLVM backend")
set(TILELANG_BACKEND_DOC_ASCEND "Enable Ascend backend")

foreach(BACKEND IN LISTS TILELANG_BACKENDS)
  set(_backend_var "USE_${BACKEND}")
  set(_doc "${TILELANG_BACKEND_DOC_${BACKEND}}")
  set(_default OFF)
  if(BACKEND STREQUAL "CUDA" AND NOT APPLE AND TILELANG_CUDA_TOOLKIT_AVAILABLE)
    set(_default ON)
  elseif(BACKEND STREQUAL "METAL" AND APPLE)
    set(_default ON)
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
         (BACKEND STREQUAL "ROCM" OR BACKEND STREQUAL "ASCEND"))
    set(_default ON)
  endif()

  # CMake variables (including cached values and -D arguments) take precedence
  # over environment variables. Environment variables initialize a fresh cache.
  if(DEFINED ${_backend_var})
    set(_default "${${_backend_var}}")
  elseif(DEFINED ENV{${_backend_var}})
    set(_default "$ENV{${_backend_var}}")
  endif()
  # STRING preserves SDK paths as well as ON/OFF values.
  set(${_backend_var} "${_default}" CACHE STRING "${_doc}")

  # TVM's config.cmake redefines USE_* options later. Save the resolved value
  # so the caller can restore it after including that configuration.
  set(TILELANG_OPTION_${_backend_var} "${${_backend_var}}")
endforeach()
