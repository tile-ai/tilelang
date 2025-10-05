cmake_minimum_required(VERSION 3.26)
project(TILE_LANG C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.SABIModule REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(TVM REQUIRED)

if(NOT TVM_FOUND)
  if(EXISTS ${TVM_SOURCE}/cmake/config.cmake)
    include(${TVM_SOURCE}/cmake/config.cmake)
    add_subdirectory(${TVM_SOURCE} tvm EXCLUDE_FROM_ALL)
  else()
    message(FATAL_ERROR "Nor prebuilt tvm provided or submodule checkout-ed.")
  endif()
endif()

if(DEFINED ENV{USE_METAL})
  set(USE_METAL ON)
endif()


# Collect source files
file(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/op/*.cc
  src/target/utils.cc
  src/target/codegen_cpp.cc
  src/target/rt_mod_cpp.cc
  # webgpu doesn't have system dependency
  src/target/codegen_webgpu.cc
  # intrin_rule doesn't have system dependency
  src/target/intrin_rule*.cc
)

# Add TileLang object library
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

# Include directories for TileLang
set(TILE_LANG_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/ffi/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
  ${TVM_SOURCE}/3rdparty/dmlc-core/include
)

target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})

add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm_runtime)
target_link_libraries(tilelang PUBLIC tvm)

set(PREBUILD_CYTHON ON)

if(PREBUILD_CYTHON)
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/tl_cython_wrapper.cpp"
    COMMENT
      "Cythoning tilelang/jit/adapter/cython/cython_wrapper.pyx"
    COMMAND Python::Interpreter -m cython
            "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
            --cplus --output-file "${CMAKE_BINARY_DIR}/tl_cython_wrapper.cpp"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tilelang/jit/adapter/cython/cython_wrapper.pyx"
    VERBATIM)

  python_add_library(tilelang_cython MODULE "${CMAKE_BINARY_DIR}/tl_cython_wrapper.cpp" USE_SABI 3.8)
  install(TARGETS tilelang_cython LIBRARY DESTINATION tilelang/lib)
endif()

install(TARGETS tvm tvm_runtime tilelang LIBRARY DESTINATION tilelang/lib)
