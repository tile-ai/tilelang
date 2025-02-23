# Copyright(c) Microsoft Corporation.
# Licensed under the MIT License.
# Learn a lot from the MLC - LLM Project
# https: // github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(TILE_LANG C CXX)

# Set default build type to Release if not provided
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Define a custom macro for globbing files with conditional CONFIGURE_DEPENDS
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()

# Handle TVM prebuild path or use default configuration
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "TVM_PREBUILD_PATH: ${TVM_PREBUILD_PATH}")

  if(EXISTS ${TVM_PREBUILD_PATH}/config.cmake)
    include(${TVM_PREBUILD_PATH}/config.cmake)
  endif()
else()
  if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
    include(${CMAKE_BINARY_DIR}/config.cmake)
  elseif(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()

  # Set default build type to RelWithDebInfo if not provided
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    message(STATUS "Setting default build type to ${CMAKE_BUILD_TYPE}")
  endif()
endif()

# include cmake modules
include(CheckCXXCompilerFlag)

# Enable static runtime build if required
if(TILE_LANG_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

# Enforce CUDA standard
if(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

# Enforce HIP standard
if(USE_ROCM)
  set(CMAKE_HIP_STANDARD 17)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__ ${CMAKE_CXX_FLAGS}")
endif()

# Enforce C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate TVM prebuild path
if(NOT DEFINED TVM_PREBUILD_PATH)
  if(DEFINED ENV{TVM_PREBUILD_PATH})
    set(TVM_PREBUILD_PATH "$ENV{TVM_PREBUILD_PATH}")
  endif()
endif()

# Locate TVM source directory
if(NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
  elseif(DEFINED TVM_PREBUILD_PATH)
    set(TVM_SOURCE_DIR "${TVM_PREBUILD_PATH}/..")
  else()
    set(TVM_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/tvm)
  endif()
endif()

# Handle TVM prebuild or build TVM from source
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "Using prebuilt TVM from ${TVM_PREBUILD_PATH}")
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
else()
  message(STATUS "Building TVM from source at ${TVM_SOURCE_DIR}")
  add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
endif()

# Collect source files
tilelang_file_glob(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/op/*.cc
  src/target/utils.cc
  src/target/codegen_cpp.cc
  src/target/rt_mod_cpp.cc
  # webgpu doesn't have system dependency
  src/target/codegen_webgpu.cc
)

# Include CUDA source files if CUDA is enabled
if(USE_CUDA)
  tilelang_file_glob(GLOB TILE_LANG_CUDA_SRCS
    src/runtime/*.cc
    src/target/codegen_cuda.cc
    src/target/rt_mod_cuda.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_CUDA_SRCS})
endif()

# Include ROCm source files if ROCm is enabled
if(USE_ROCM)
  tilelang_file_glob(GLOB TILE_LANG_HIP_SRCS
    src/target/codegen_hip.cc
    src/target/rt_mod_hip.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
endif()

message(STATUS "Collected source files: ${TILE_LANG_SRCS}")

# Add TileLang object library
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
# Include directories for TileLang
set(TILE_LANG_INCLUDES
  ${TVM_SOURCE_DIR}/include
  ${TVM_SOURCE_DIR}/src
  ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
  ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)

# Find CUDA Toolkit
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please set CUDAToolkit_ROOT.")
  endif()

  message(STATUS "CUDA Toolkit includes: ${CUDAToolkit_INCLUDE_DIRS}")
  list(APPEND TILE_LANG_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif(USE_CUDA)

# Find ROCM Toolkit
if(USE_ROCM)
  find_rocm(${USE_ROCM})
  message(STATUS "USE_ROCM: ${USE_ROCM}")

  if(ROCM_FOUND)
    # always set the includedir
    # avoid global retrigger of cmake
    include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
    add_definitions(-D__HIP_PLATFORM_HCC__=1)
  else()
    message(FATAL_ERROR "ROCM Toolkit not found. Please set HIP_ROOT.")
  endif(ROCM_FOUND)

  message(STATUS "ROCM Toolkit includes: ${ROCM_INCLUDE_DIRS}")
  list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})
endif(USE_ROCM)

# Define compile-time macros
set(TILE_LANG_COMPILE_DEFS
  DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
  __STDC_FORMAT_MACROS=1
  PICOJSON_USE_INT64
)

# Set target properties for object library
target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_definitions(tilelang_objs PRIVATE ${TILE_LANG_COMPILE_DEFS})
target_compile_definitions(tilelang_objs PRIVATE -DTILE_LANG_EXPORTS)

# Shared library
add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm_runtime)

# Static library
add_library(tilelang_static STATIC $<TARGET_OBJECTS:tilelang_objs>)
add_dependencies(tilelang_static tvm_runtime)
set_target_properties(tilelang_static PROPERTIES OUTPUT_NAME tilelang)

# Debug build type-specific definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(tilelang PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tilelang_static PRIVATE "TVM_LOG_DEBUG")
endif()

# Module shared library
add_library(tilelang_module SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang_module PUBLIC tvm)

# Install targets
if(TILE_LANG_INSTALL_STATIC_LIB)
  install(TARGETS tilelang_static tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  )
else()
  if(DEFINED TVM_PREBUILD_PATH)
    install(TARGETS tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  else()
    install(TARGETS tvm_runtime tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  endif()
endif()
