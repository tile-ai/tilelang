// tilelang target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef __OBJC__
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/core/TensorBase.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_error_ndim_mismatch_packed = NULL;
static void* __tvm_error_dtype_mismatch_packed = NULL;
static void* __tvm_error_expect_eq_packed = NULL;
static void* __tvm_error_byte_offset_mismatch_packed = NULL;
static void* __tvm_error_device_type_mismatch_packed = NULL;
static void* __tvm_error_null_ptr_packed = NULL;
static void* __tvm_set_device_packed = NULL;
static void* gemm_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t gemm(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t gemm(void* self_handle, void* args, int32_t num_args, void* result) {
  TVMFFIAny stack[9];
  void* stack_ffi_any = stack;
  if (!((num_args == 3))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "gemm: num_args should be 3", (long long)(num_args), (long long)(3));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "gemm: args pointer is NULL");
    return -1;
  }
  int32_t A_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((A_handle_type_index == 0) || (A_handle_type_index == 4)) || (A_handle_type_index == 7)) || (64 <= A_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel gemm input A expected pointer or tensor handle");
    return -1;
  }
  int32_t B_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((B_handle_type_index == 0) || (B_handle_type_index == 4)) || (B_handle_type_index == 7)) || (64 <= B_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel gemm input B expected pointer or tensor handle");
    return -1;
  }
  int32_t C_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((C_handle_type_index == 0) || (C_handle_type_index == 4)) || (C_handle_type_index == 7)) || (64 <= C_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel gemm input C expected pointer or tensor handle");
    return -1;
  }
  void* A_handle = ((A_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* B_handle = ((B_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* C_handle = ((C_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  bool gemm_A_is_null = (A_handle == NULL);
  if (!(!gemm_A_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "gemm.A is expected to have non-NULL pointer");
    return -1;
  }
  bool gemm_B_is_null = (B_handle == NULL);
  if (!(!gemm_B_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "gemm.B is expected to have non-NULL pointer");
    return -1;
  }
  bool gemm_C_is_null = (C_handle == NULL);
  if (!(!gemm_C_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "gemm.C is expected to have non-NULL pointer");
    return -1;
  }
  void* gemm_A_shape = (((DLTensor*)A_handle)[0].shape);
  void* gemm_B_shape = (((DLTensor*)B_handle)[0].shape);
  void* gemm_C_shape = (((DLTensor*)C_handle)[0].shape);
  if ((((DLTensor*)A_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_1;
    result_1.type_index = kTVMFFINone;
    result_1.zero_padding = 0;
    result_1.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_1) != 0) {
      return -1;
    }
  }
  void* gemm_A_strides = (((DLTensor*)A_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)A_handle)[0].device.device_id);
  void* A = (((DLTensor*)A_handle)[0].data);
  if ((((DLTensor*)B_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_2;
    result_2.type_index = kTVMFFINone;
    result_2.zero_padding = 0;
    result_2.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_2) != 0) {
      return -1;
    }
  }
  void* gemm_B_strides = (((DLTensor*)B_handle)[0].strides);
  void* B = (((DLTensor*)B_handle)[0].data);
  if ((((DLTensor*)C_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_3;
    result_3.type_index = kTVMFFINone;
    result_3.zero_padding = 0;
    result_3.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_3) != 0) {
      return -1;
    }
  }
  void* gemm_C_strides = (((DLTensor*)C_handle)[0].strides);
  void* C = (((DLTensor*)C_handle)[0].data);
  if ((((((DLTensor*)A_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)A_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)A_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_4;
    result_4.type_index = kTVMFFINone;
    result_4.zero_padding = 0;
    result_4.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_4) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)gemm_A_shape)[0]) != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)gemm_A_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_5;
    result_5.type_index = kTVMFFINone;
    result_5.zero_padding = 0;
    result_5.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_5) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)gemm_A_shape)[1]) != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)gemm_A_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_6;
    result_6.type_index = kTVMFFINone;
    result_6.zero_padding = 0;
    result_6.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_6) != 0) {
      return -1;
    }
  }
  int32_t condval;
  if ((gemm_A_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)gemm_A_strides)[1]);
  }
  if (condval != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_1;
    if ((gemm_A_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)gemm_A_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_1);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_7;
    result_7.type_index = kTVMFFINone;
    result_7.zero_padding = 0;
    result_7.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_7) != 0) {
      return -1;
    }
  }
  int32_t condval_2;
  if ((gemm_A_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)gemm_A_strides)[0]);
  }
  if (condval_2 != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_3;
    if ((gemm_A_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)gemm_A_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_3);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_8;
    result_8.type_index = kTVMFFINone;
    result_8.zero_padding = 0;
    result_8.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_8) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)A_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_9;
    result_9.type_index = kTVMFFINone;
    result_9.zero_padding = 0;
    result_9.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_9) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)A_handle)[0].device.device_type) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_10;
    result_10.type_index = kTVMFFINone;
    result_10.zero_padding = 0;
    result_10.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_10) != 0) {
      return -1;
    }
  }
  if (A == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"A";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_11;
    result_11.type_index = kTVMFFINone;
    result_11.zero_padding = 0;
    result_11.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_11) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)B_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)B_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)B_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_12;
    result_12.type_index = kTVMFFINone;
    result_12.zero_padding = 0;
    result_12.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_12) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)gemm_B_shape)[0]) != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)gemm_B_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_13;
    result_13.type_index = kTVMFFINone;
    result_13.zero_padding = 0;
    result_13.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_13) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)gemm_B_shape)[1]) != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)gemm_B_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_14;
    result_14.type_index = kTVMFFINone;
    result_14.zero_padding = 0;
    result_14.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_14) != 0) {
      return -1;
    }
  }
  int32_t condval_4;
  if ((gemm_B_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)gemm_B_strides)[1]);
  }
  if (condval_4 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_5;
    if ((gemm_B_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)gemm_B_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_5);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_15;
    result_15.type_index = kTVMFFINone;
    result_15.zero_padding = 0;
    result_15.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_15) != 0) {
      return -1;
    }
  }
  int32_t condval_6;
  if ((gemm_B_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)gemm_B_strides)[0]);
  }
  if (condval_6 != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_7;
    if ((gemm_B_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)gemm_B_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_7);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_16;
    result_16.type_index = kTVMFFINone;
    result_16.zero_padding = 0;
    result_16.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_16) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)B_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_17;
    result_17.type_index = kTVMFFINone;
    result_17.zero_padding = 0;
    result_17.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_17) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)B_handle)[0].device.device_id) != (((DLTensor*)A_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_18;
    result_18.type_index = kTVMFFINone;
    result_18.zero_padding = 0;
    result_18.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_18) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)B_handle)[0].device.device_type) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)B_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_19;
    result_19.type_index = kTVMFFINone;
    result_19.zero_padding = 0;
    result_19.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_19) != 0) {
      return -1;
    }
  }
  if (B == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"B";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_20;
    result_20.type_index = kTVMFFINone;
    result_20.zero_padding = 0;
    result_20.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_20) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)C_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)C_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)C_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_21;
    result_21.type_index = kTVMFFINone;
    result_21.zero_padding = 0;
    result_21.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_21) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)gemm_C_shape)[0]) != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)gemm_C_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_22;
    result_22.type_index = kTVMFFINone;
    result_22.zero_padding = 0;
    result_22.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_22) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)gemm_C_shape)[1]) != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)gemm_C_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_23;
    result_23.type_index = kTVMFFINone;
    result_23.zero_padding = 0;
    result_23.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_23) != 0) {
      return -1;
    }
  }
  int32_t condval_8;
  if ((gemm_C_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)gemm_C_strides)[1]);
  }
  if (condval_8 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_9;
    if ((gemm_C_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)gemm_C_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_9);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_24;
    result_24.type_index = kTVMFFINone;
    result_24.zero_padding = 0;
    result_24.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_24) != 0) {
      return -1;
    }
  }
  int32_t condval_10;
  if ((gemm_C_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)gemm_C_strides)[0]);
  }
  if (condval_10 != 1024) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1024;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_11;
    if ((gemm_C_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)gemm_C_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_11);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_25;
    result_25.type_index = kTVMFFINone;
    result_25.zero_padding = 0;
    result_25.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_25) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)C_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_26;
    result_26.type_index = kTVMFFINone;
    result_26.zero_padding = 0;
    result_26.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_26) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)C_handle)[0].device.device_id) != (((DLTensor*)A_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)A_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_27;
    result_27.type_index = kTVMFFINone;
    result_27.zero_padding = 0;
    result_27.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_27) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)C_handle)[0].device.device_type) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)C_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_28;
    result_28.type_index = kTVMFFINone;
    result_28.zero_padding = 0;
    result_28.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_28) != 0) {
      return -1;
    }
  }
  if (C == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"gemm";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"C";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_29;
    result_29.type_index = kTVMFFINone;
    result_29.zero_padding = 0;
    result_29.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_29) != 0) {
      return -1;
    }
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)8);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_30;
  result_30.type_index = kTVMFFINone;
  result_30.zero_padding = 0;
  result_30.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_30) != 0) {
    return -1;
  }
  if (A == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = A;
  if (B == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = B;
  if (C == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = C;
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
  if (gemm_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "gemm_kernel", &gemm_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_31;
  result_31.type_index = kTVMFFINone;
  result_31.zero_padding = 0;
  result_31.v_int64 = 0;
  if (TVMFFIFunctionCall(gemm_kernel_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_31) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return gemm(self, args, num_args, result);
}
