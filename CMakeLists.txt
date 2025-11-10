# Learn a lot from the MLC - LLM Project
# https://github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.26)
project(TILE_LANG C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND "$ENV{CIBUILDWHEEL}")
  # Warning came from tvm submodule
  string(APPEND CMAKE_CXX_FLAGS " -Wno-dangling-reference")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE TILELANG_GIT_SUBMODULE_RESULT
    )
    if(NOT TILELANG_GIT_SUBMODULE_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          "Failed to initialize git submodules. Please run "
          "`git submodule update --init --recursive` and re-run CMake."
      )
    endif()
  else()
    message(
      FATAL_ERROR
        "Git is required to initialize TileLang submodules. "
        "Please install git or fetch the submodules manually."
    )
  endif()
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM} with base_dir=${CMAKE_SOURCE_DIR}")
  if(APPLE)
    # Passing configs like `ccache base_dir=/xxx cc ...` is supported
    # (likely) since ccache 4.x, which has been provided by homebrew.
    # Our Linux builder image (manylinux2014 & manylinux_2_28) still
    # provides ccache 3.x and do not support this form.
    # `cibuildwheel` uses fixed folder on Linux (`/project`) as working directory,
    # so cache would work without setting `base_dir`.
    set(CCACHE_PROGRAM "${CCACHE_PROGRAM};base_dir=${CMAKE_SOURCE_DIR}")
  endif()
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
else()
  find_program(SCCACHE_PROGRAM sccache)
  if(SCCACHE_PROGRAM)
    message(STATUS "Using sccache: ${SCCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "CUDA compiler launcher")
  endif()
endif()

# Configs
set(USE_CUDA OFF)
set(USE_ROCM OFF)
set(USE_METAL OFF)
set(PREBUILD_CYTHON ON)
# Configs end

include(cmake/load_tvm.cmake)

if(EXISTS ${TVM_SOURCE}/cmake/config.cmake)
  include(${TVM_SOURCE}/cmake/config.cmake)
else()
  message(FATAL_ERROR "Nor tvm provided or submodule checkout-ed.")
endif()

# Include directories for TileLang
set(TILE_LANG_INCLUDES ${TVM_INCLUDES})

# Collect source files
file(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/op/*.cc
  src/target/utils.cc
  src/target/codegen_cpp.cc
  src/target/rt_mod_cpp.cc
  # intrin_rule doesn't have system dependency
  src/target/intrin_rule*.cc
)

# Backend-specific checks and configs
if($ENV{USE_METAL})
  set(USE_METAL ON)
elseif(APPLE)
  message(STATUS "Enable Metal support by default.")
  set(USE_METAL ON)
elseif($ENV{USE_ROCM})
  set(USE_ROCM ON)
else()
  if($ENV{USE_CUDA})
    set(USE_CUDA ON)
  elseif(DEFINED ENV{USE_CUDA} AND NOT $ENV{USE_CUDA})
    # Build CPU-only when we explicitly disable CUDA
    set(USE_CUDA OFF)
  else()
    message(STATUS "Enable CUDA support by default.")
    set(USE_CUDA ON)
  endif()
endif()

if(USE_METAL)
  file(GLOB TILE_LANG_METAL_SRCS
    src/target/rt_mod_metal.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_METAL_SRCS})
  # FIXME: CIBW failed with backtrace, why???
  set(TVM_FFI_USE_LIBBACKTRACE OFF)
elseif(USE_ROCM)
  set(CMAKE_HIP_STANDARD 17)
  include(${TVM_SOURCE}/cmake/utils/FindROCM.cmake)
  find_rocm($ENV{USE_ROCM})
  add_compile_definitions(__HIP_PLATFORM_AMD__ __HIP_PLATFORM_HCC__=1)

  file(GLOB TILE_LANG_HIP_SRCS
    src/target/codegen_hip.cc
    src/target/rt_mod_hip.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
  list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})
elseif(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
  add_compile_definitions("CUDA_MAJOR_VERSION=${CUDAToolkit_VERSION_MAJOR}")

  # Set `USE_CUDA=/usr/local/cuda-x.y`
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH USE_CUDA)

  file(GLOB TILE_LANG_CUDA_SRCS
    src/runtime/*.cc
    src/target/ptx.cc
    src/target/codegen_cuda.cc
    src/target/rt_mod_cuda.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_CUDA_SRCS})

  list(APPEND TILE_LANG_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Include tvm after configs have been populated
add_subdirectory(${TVM_SOURCE} tvm EXCLUDE_FROM_ALL)

# Resolve compile warnings in tvm
add_compile_definitions(DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)

add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
endif()

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})

add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
add_library(tilelang_module SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm_runtime tvm)
target_link_libraries(tilelang_module PUBLIC tvm)
# Build cython extension
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp"
  COMMENT
    "Cythoning tilelang/jit/adapter/cython/cython_wrapper.pyx"
  COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
          --module-name tilelang_cython_wrapper
          --cplus --output-file "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
  VERBATIM)

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(USE_SABI USE_SABI ${SKBUILD_SABI_VERSION})
endif()

python_add_library(tilelang_cython_wrapper MODULE "${CMAKE_BINARY_DIR}/tilelang_cython_wrapper.cpp" ${USE_SABI} WITH_SOABI)
# Install extension into the tilelang package directory
install(TARGETS tilelang_cython_wrapper
        LIBRARY DESTINATION tilelang
        RUNTIME DESTINATION tilelang
        ARCHIVE DESTINATION tilelang)

# let libtilelang to search tvm/tvm_runtime in same dir
if(APPLE)
  set_target_properties(tilelang PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../../tvm_ffi/lib")
  set_target_properties(tilelang_module PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../../tvm_ffi/lib")
  set_target_properties(tvm PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../../tvm_ffi/lib")
  set_target_properties(tvm_runtime PROPERTIES INSTALL_RPATH "@loader_path;@loader_path/../../tvm_ffi/lib")
elseif(UNIX)
  set_target_properties(tilelang PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../tvm_ffi/lib")
  set_target_properties(tilelang_module PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../tvm_ffi/lib")
  set_target_properties(tvm PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../tvm_ffi/lib")
  set_target_properties(tvm_runtime PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/../../tvm_ffi/lib")
endif()

install(
  TARGETS tvm tvm_runtime tilelang_module tilelang
  LIBRARY DESTINATION tilelang/lib
)
