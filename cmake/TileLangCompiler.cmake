# Add TileLang to an existing TIRx compiler.  Both the embedding CMake build and
# the companion provider wheel include this file, so registration and private
# C++ interfaces always live in one shared library.
if(TARGET tilelang_objs)
  return()
endif()
get_filename_component(TILELANG_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
if(NOT TARGET tvm_tirx_compiler)
  message(FATAL_ERROR "TileLangCompiler.cmake requires the TIRx compiler target")
endif()

if(NOT DEFINED TILE_LANG_SRCS)
  message(FATAL_ERROR "The embedding build must supply the complete TileLang source list")
endif()
set(TILELANG_NATIVE_SOURCE_OVERRIDE "" CACHE STRING
    "Development-only source subset; partial builds cannot be installed")
if(TILELANG_NATIVE_SOURCE_OVERRIDE)
  set(TILE_LANG_SRCS "")
  foreach(_source IN LISTS TILELANG_NATIVE_SOURCE_OVERRIDE)
    get_filename_component(_source "${_source}" ABSOLUTE BASE_DIR "${TILELANG_SOURCE_DIR}")
    list(APPEND TILE_LANG_SRCS "${_source}")
  endforeach()
  install(CODE "message(FATAL_ERROR \"A partial TileLang compiler build cannot be installed\")")
endif()

option(TILELANG_BUILD_NATIVE_TESTS
       "Build the native regression-test bridges into the compiler" OFF)
if(TILELANG_BUILD_NATIVE_TESTS)
  # Keep private C++ test interfaces in this same compiler on every platform.
  # Release providers leave this option off and expose no test-only registry.
  list(APPEND TILE_LANG_SRCS
    "${TILELANG_SOURCE_DIR}/testing/cpp/native_operator_probe.cc"
    "${TILELANG_SOURCE_DIR}/testing/cpp/native_constraint_probe.cc"
    "${TILELANG_SOURCE_DIR}/testing/cpp/native_pipeline_access_probe.cc")
  if(NOT USE_ROCM)
    # This injector is pure IR analysis. CPU/CUDA tests exercise its contracts
    # without requiring an AMD runtime or adding the ROCm code generator.
    list(APPEND TILE_LANG_SRCS
      "${TILELANG_SOURCE_DIR}/src/rocm/transform/async_copy_injector.cc")
  endif()
  if(USE_ASCEND)
    list(APPEND TILE_LANG_SRCS
      "${TILELANG_SOURCE_DIR}/testing/cpp/native_ascend_schedule_probe.cc"
      "${TILELANG_SOURCE_DIR}/testing/cpp/native_ascend_device_api_probe.cc")
  endif()
endif()

add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})
target_link_libraries(tilelang_objs PRIVATE tvm_tirx::header)
target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_features(tilelang_objs PRIVATE cxx_std_20)
set_target_properties(tilelang_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(TARGET tvm_llvm_header)
  target_link_libraries(tilelang_objs PRIVATE tvm_llvm_header)
endif()
if(MSVC)
  target_compile_definitions(tilelang_objs PRIVATE TVM_TIRX_EXPORTS)
endif()
if(TVM_TIRX_USE_CUSTOM_LOGGING)
  target_compile_definitions(tilelang_objs PRIVATE TVM_LOG_CUSTOMIZE=1)
endif()
if(TILELANG_RELEASE_BUILD)
  target_compile_definitions(tilelang_objs PRIVATE TILELANG_RELEASE_BUILD=1)
endif()
target_compile_definitions(tilelang_objs PRIVATE
  TILELANG_FFI_TARGET_REF="${TILELANG_FFI_TARGET_REF}")
# Only this translation unit embeds the digest. A source change must not force
# recompilation of every native object solely to update a build-identity string.
set_property(SOURCE "${TILELANG_SOURCE_DIR}/src/ir/launch_region.cc" APPEND PROPERTY
  COMPILE_DEFINITIONS
  TILELANG_COMPILER_SOURCE_DIGEST="${TILELANG_COMPILER_SOURCE_DIGEST}"
  TILELANG_PROVIDER_VERSION="${TILELANG_PROVIDER_VERSION}")
target_sources(tvm_tirx_compiler PRIVATE $<TARGET_OBJECTS:tilelang_objs>)
add_custom_target(tilelang DEPENDS tvm_tirx_compiler)
