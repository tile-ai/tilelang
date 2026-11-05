#pragma once

#include <tvm/ffi/c_api.h>

// Storage is an array of initialized, owned TVMFFIAny values supplied by the
// caller. Its capacity remains the caller's contract; the generated boundary
// rejects null pointers and indices outside the historical int32 domain.
static inline int TLAnyListCheck(void *storage, int64_t index) {
  if (storage == NULL) {
    TVMFFIErrorSetRaisedFromCStr("ValueError",
                                 "Any-list storage must not be null");
    return -1;
  }
  if (index < 0 || index > INT32_MAX) {
    TVMFFIErrorSetRaisedFromCStr("ValueError",
                                 "Any-list index must fit nonnegative int32");
    return -1;
  }
  return 0;
}

static inline int TLAnyListResetItem(void *storage, int64_t index) {
  if (TLAnyListCheck(storage, index) != 0)
    return -1;
  TVMFFIAny *slot = ((TVMFFIAny *)storage) + index;
  TVMFFIAny previous = *slot;
  // Clear before running a destructor, so reentrant code sees an empty slot.
  slot->type_index = kTVMFFINone;
  slot->zero_padding = 0;
  slot->v_int64 = 0;
  if (previous.type_index >= kTVMFFIStaticObjectBegin) {
    return TVMFFIObjectDecRef(previous.v_obj);
  }
  return 0;
}

static inline int TLAnyListBorrow(void *storage, int64_t index,
                                  TVMFFIAny *argument) {
  if (TLAnyListCheck(storage, index) != 0)
    return -1;
  *argument = ((const TVMFFIAny *)storage)[index];
  return 0;
}

// Keep a getter argument alive across evaluation of later operands and the
// callback. The invocation releases this temporary owner on every exit path.
static inline int TLAnyListAcquire(void *storage, int64_t index,
                                   TVMFFIAny *argument) {
  if (TLAnyListCheck(storage, index) != 0)
    return -1;
  return TVMFFIAnyViewToOwnedAny(((const TVMFFIAny *)storage) + index,
                                 argument);
}

static inline void TLAnyListClearArguments(TVMFFIAny *arguments,
                                           int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    TLAnyListResetItem(arguments, i);
  }
}

// Match public FFI TypeTraits conversions before reading the union. An owned
// Object/String/Tensor cannot become a void* whose owner is then released.
static inline int TLAnyListCheckResult(TVMFFIAny *result, int32_t expected) {
  if (expected == kTVMFFIOpaquePtr) {
    if (result->type_index == kTVMFFINone) {
      result->v_ptr = NULL;
      return 0;
    }
    if (result->type_index == kTVMFFIOpaquePtr)
      return 0;
    TVMFFIErrorSetRaisedFromCStr(
        "TypeError",
        "Any-list packed result must be an opaque pointer or None");
    return -1;
  }
  if (expected == kTVMFFIFloat) {
    if (result->type_index == kTVMFFIFloat)
      return 0;
    if (result->type_index == kTVMFFIInt || result->type_index == kTVMFFIBool) {
      result->v_float64 = (double)result->v_int64;
      result->type_index = kTVMFFIFloat;
      return 0;
    }
    TVMFFIErrorSetRaisedFromCStr(
        "TypeError",
        "Any-list packed result must be a float, integer, or bool");
    return -1;
  }
  if (result->type_index == kTVMFFIInt || result->type_index == kTVMFFIBool)
    return 0;
  TVMFFIErrorSetRaisedFromCStr(
      "TypeError", "Any-list packed result must be an integer or bool");
  return -1;
}

static inline int TLAnyListMoveResult(void *storage, int64_t index,
                                      TVMFFIAny *result) {
  if (TLAnyListCheck(storage, index) != 0)
    return -1;
  // The callback owns result separately even when it returned an input alias.
  // Only replace the destination after the callback succeeded.
  if (TLAnyListResetItem(storage, index) != 0)
    return -1;
  ((TVMFFIAny *)storage)[index] = *result;
  result->type_index = kTVMFFINone;
  result->zero_padding = 0;
  result->v_int64 = 0;
  return 0;
}
