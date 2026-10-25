set(_tilelang_installed_tirx_default OFF)
if(DEFINED ENV{TILELANG_USE_INSTALLED_TIRX})
  set(_tilelang_installed_tirx_default "$ENV{TILELANG_USE_INSTALLED_TIRX}")
endif()
option(TILELANG_USE_INSTALLED_TIRX "Link an independently installed standalone TIRx" ${_tilelang_installed_tirx_default})
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/TVM_TIRX_VERSION" TILELANG_TIRX_VERSION LIMIT_COUNT 1)
set(TVM_TIRX_SOURCE "$ENV{TVM_TIRX_SOURCE}" CACHE PATH "Standalone TVM TIRx source directory")

function(tilelang_require_tirx_scope include_directory)
  file(READ "${include_directory}/tvm/codegen/codegen.h" _codegen_declarations)
  string(FIND "${_codegen_declarations}" "kSourceOnly" _native_source_codegen)
  string(FIND "${_codegen_declarations}" "kNonRestrictParamIndices" _native_parameter_qualifiers)
  if(_native_source_codegen EQUAL -1 OR _native_parameter_qualifiers EQUAL -1)
    message(FATAL_ERROR
      "This migration requires native source-only code generation and parameter qualifiers. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  file(READ "${include_directory}/tvm/tirx/transform.h" _storage_transform_declarations)
  string(FIND "${_storage_transform_declarations}" "StorageRewrite(bool enable_reuse" _storage_options)
  string(FIND "${_storage_transform_declarations}" "bool allow_local_inplace" _local_inplace)
  if(_storage_options EQUAL -1 OR _local_inplace EQUAL -1)
    message(FATAL_ERROR
      "This migration requires configurable native storage reuse. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  file(READ "${include_directory}/tvm/tirx/transform.h" _transform_declarations)
  string(FIND "${_transform_declarations}" "Pass OutlineDeviceRegions()" _device_outlining)
  if(_device_outlining EQUAL -1)
    message(FATAL_ERROR
      "This migration requires separate native device outlining and launch lowering. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  if(NOT EXISTS "${include_directory}/tvm/tirx/packed_api.h")
    message(FATAL_ERROR
      "This migration requires the public native packed-argument decoder. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  string(FIND "${_transform_declarations}" "HoistIf(bool allow_nontrivial_else" _two_sided_hoist)
  file(READ "${include_directory}/tvm/tirx/op/abi.h" _abi_declarations)
  string(FIND "${_abi_declarations}" "call_packed_with_result_op()" _explicit_result)
  if(_two_sided_hoist EQUAL -1 OR _explicit_result EQUAL -1)
    message(FATAL_ERROR
      "This migration requires configurable native loop hoisting and explicit packed results. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  set(_shared_op_header "${include_directory}/tvm/ir/op.h")
  if(NOT EXISTS "${_shared_op_header}")
    message(FATAL_ERROR "Standalone TIRx shared operator headers are required.")
  endif()
  file(READ "${_shared_op_header}" _shared_op_declarations)
  string(FIND "${_shared_op_declarations}" "kAttrsMayContainExpr" _expression_attributes)
  if(_expression_attributes EQUAL -1)
    message(FATAL_ERROR
      "This migration requires binding-aware native call attributes. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  string(FIND "${_shared_op_declarations}" "TVM_TIRX_DLL ArgumentInfo(" _argument_info_factory)
  if(_argument_info_factory EQUAL -1)
    message(FATAL_ERROR
      "This migration requires the exported native operator-signature descriptor factory. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  if(NOT EXISTS "${include_directory}/tvm/tirx/data_type_rewriter.h")
    message(FATAL_ERROR
      "This migration requires the public native data-type rewriter extension API. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
  set(_region_header "${include_directory}/tvm/tirx/op/region.h")
  if(NOT EXISTS "${_region_header}")
    message(FATAL_ERROR "Standalone TIRx region headers are required: ${_region_header}")
  endif()
  file(READ "${_region_header}" _region_declarations)
  string(FIND "${_region_declarations}" "const Op& scope_op()" _scope_declaration)
  if(_scope_declaration EQUAL -1)
    message(FATAL_ERROR
      "This migration requires the native lexical-scope support in the workspace's tvm-tirx checkout. "
      "Build/install that checkout rather than an unpatched 0.1.0a2 package; see MIGRATION_TIRX.md.")
  endif()
  set(_op_attributes_header "${include_directory}/tvm/tirx/op/attr.h")
  if(NOT EXISTS "${_op_attributes_header}")
    message(FATAL_ERROR "Standalone TIRx operator attribute headers are required.")
  endif()
  file(READ "${_op_attributes_header}" _op_attributes)
  string(FIND "${_op_attributes}" "kVectorizeCall" _vectorize_extension)
  if(_vectorize_extension EQUAL -1)
    message(FATAL_ERROR
      "This migration requires native dialect-call vectorization support. "
      "Build/install the patched workspace's tvm-tirx checkout; see MIGRATION_TIRX.md.")
  endif()
endfunction()

if(TILELANG_USE_INSTALLED_TIRX)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
  find_package(tvm_tirx CONFIG QUIET)
  if(NOT tvm_tirx_FOUND)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_tirx.config --cmakedir
      OUTPUT_VARIABLE _tilelang_tirx_cmake OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _tilelang_tirx_config_result)
    if(NOT _tilelang_tirx_config_result EQUAL 0)
      message(FATAL_ERROR
        "Install standalone apache-tvm-tirx or set tvm_tirx_DIR to its CMake package directory.")
    endif()
    find_package(tvm_tirx CONFIG REQUIRED PATHS "${_tilelang_tirx_cmake}" NO_DEFAULT_PATH)
  endif()
  if(NOT TARGET tvm_ffi::header)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
      OUTPUT_VARIABLE _tilelang_ffi_cmake OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    find_package(tvm_ffi CONFIG REQUIRED PATHS "${_tilelang_ffi_cmake}" NO_DEFAULT_PATH)
  endif()
  if(NOT tvm_tirx_VERSION STREQUAL TILELANG_TIRX_VERSION)
    message(FATAL_ERROR "Unsupported standalone TIRx version ${tvm_tirx_VERSION}; expected ${TILELANG_TIRX_VERSION}.")
  endif()
  tilelang_require_tirx_scope("${tvm_tirx_INCLUDE_DIR}")
  get_target_property(_tilelang_tirx_runtime tvm_runtime::shared IMPORTED_LOCATION)
  get_filename_component(_tilelang_tirx_libdir "${_tilelang_tirx_runtime}" DIRECTORY)
  foreach(BACKEND IN ITEMS CUDA ROCM METAL)
    if(USE_${BACKEND})
      string(TOLOWER "${BACKEND}" _tilelang_backend)
      set(_tilelang_sidecar "${_tilelang_tirx_libdir}/${CMAKE_SHARED_LIBRARY_PREFIX}tvm_runtime_${_tilelang_backend}${CMAKE_SHARED_LIBRARY_SUFFIX}")
      if(NOT EXISTS "${_tilelang_sidecar}")
        message(FATAL_ERROR
          "USE_${BACKEND} requires the matching standalone TIRx runtime: ${_tilelang_sidecar}. "
          "Build/install a paired dependency with TVM_TIRX_USE_${BACKEND}=ON, or explicitly disable USE_${BACKEND}.")
      endif()
    endif()
  endforeach()
  set(TILELANG_TIRX_COMPILER tvm_tirx_compiler::shared)
  set(TVM_TIRX_INCLUDES "${tvm_tirx_INCLUDE_DIR}" "${tvm_ffi_INCLUDE_DIR}" "${tvm_ffi_DLPACK_INCLUDE_DIR}")
  message(STATUS "Using installed standalone TIRx: ${tvm_tirx_DIR}")
  return()
endif()

if(NOT TVM_TIRX_SOURCE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tvm-tirx/CMakeLists.txt")
  get_filename_component(TVM_TIRX_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../tvm-tirx" ABSOLUTE)
endif()

if(TVM_TIRX_SOURCE)
  get_filename_component(_tilelang_requested_tirx "${TVM_TIRX_SOURCE}" ABSOLUTE)
  file(REAL_PATH "${TVM_TIRX_SOURCE}" _tilelang_resolved_tirx)
  file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty" _tilelang_resolved_thirdparty)
  string(FIND "${_tilelang_requested_tirx}/" "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/" _tilelang_vendored_path)
  string(FIND "${_tilelang_resolved_tirx}/" "${_tilelang_resolved_thirdparty}/" _tilelang_vendored_source)
  if(_tilelang_vendored_path EQUAL 0 OR _tilelang_vendored_source EQUAL 0)
    message(FATAL_ERROR "Standalone TIRx must remain outside TileLang's 3rdparty directory, including symlinks.")
  endif()
endif()

if(NOT EXISTS "${TVM_TIRX_SOURCE}/include/tvm/tirx/type.h" OR
   NOT EXISTS "${TVM_TIRX_SOURCE}/python/tvm_tirx/__init__.py")
  message(FATAL_ERROR
    "TileLang requires standalone apache/tvm-tirx. Set TVM_TIRX_SOURCE to its checkout; "
    "the TVM monorepo and the former TileLang/tvm fork are not supported.")
endif()

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/TVM_TIRX_COMMIT" TILELANG_TIRX_COMMIT LIMIT_COUNT 1)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${TVM_TIRX_SOURCE}/.git")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${TVM_TIRX_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE _tilelang_tirx_commit OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  if(NOT _tilelang_tirx_commit STREQUAL TILELANG_TIRX_COMMIT)
    message(FATAL_ERROR
      "Unsupported TIRx revision ${_tilelang_tirx_commit}; expected ${TILELANG_TIRX_COMMIT}.")
  endif()
endif()

set(TVM_TIRX_INCLUDES
  "${TVM_TIRX_SOURCE}/include"
  "${TVM_TIRX_SOURCE}/src"
  "${TVM_TIRX_SOURCE}/3rdparty/tvm-ffi/include"
  "${TVM_TIRX_SOURCE}/3rdparty/tvm-ffi/3rdparty/dlpack/include"
)
set(TVM_TIRX_CMAKE_DIR "${TVM_TIRX_SOURCE}/cmake")
set(TILELANG_TIRX_COMPILER tvm_tirx_compiler)
tilelang_require_tirx_scope("${TVM_TIRX_SOURCE}/include")
message(STATUS "Using standalone TIRx: ${TVM_TIRX_SOURCE} (${TILELANG_TIRX_COMMIT})")
