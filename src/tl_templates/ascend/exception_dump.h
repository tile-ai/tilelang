/*!
 * \file exception_dump.h
 * \brief Host/device ABI for opt-in Ascend AI Core exception dumps.
 */
#ifndef TILELANG_TL_TEMPLATES_ASCEND_EXCEPTION_DUMP_H_
#define TILELANG_TL_TEMPLATES_ASCEND_EXCEPTION_DUMP_H_

#include <cstddef>
#include <cstdint>

namespace tl {
namespace ascend {

constexpr uint64_t kExceptionDumpMagic = 0x474e414c454c4954ULL;
constexpr uint32_t kExceptionDumpVersion = 1;
constexpr uint32_t kMaxExceptionDumpTensors = 16;

// ACL scalar storage sizes admitted by the frontend. Keep this explicit so
// malformed serialized metadata or a false MAGIC match cannot invent a dtype.
constexpr uint32_t ExceptionDumpElementBytes(int32_t dtype) {
  switch (dtype) {
  case 2:  // int8
  case 4:  // uint8
  case 12: // bool
    return 1;
  case 1:  // float16
  case 6:  // int16
  case 7:  // uint16
  case 27: // bfloat16
    return 2;
  case 0: // float32
  case 3: // int32
  case 8: // uint32
    return 4;
  case 9:  // int64
  case 10: // uint64
  case 11: // float64
    return 8;
  default:
    return 0;
  }
}

// Fixed-width fields keep the generated device signature and the TVM-FFI
// runtime independent of CANN headers. `size` is bytes; `elements` is the
// flattened element count used for the dump's ND shape. The frontend currently
// admits only supported scalar element types, so no lane expansion is needed.
struct alignas(8) ExceptionDumpTensor {
  uint64_t address;
  uint64_t size;
  uint64_t elements;
  int32_t dtype;
  int32_t reserved;
};

struct alignas(8) ExceptionDumpInfo {
  // Embedded by value in the launch arguments: the callback cannot dereference
  // a pointer to a host-side descriptor after an asynchronous device failure.
  uint64_t magic;
  uint32_t version;
  // Only the first `count` slots are active; unused slots/reserved fields are
  // zero-initialized by the TVM-FFI packer. kernel_name must be NUL-terminated.
  uint32_t count;
  char kernel_name[128];
  ExceptionDumpTensor tensors[kMaxExceptionDumpTensors];
};

// Changing this layout requires coordinated codegen, runtime and decoder
// changes, including the version marker used in serialized launch metadata.
static_assert(sizeof(ExceptionDumpTensor) == 32, "Exception dump tensor ABI");
static_assert(offsetof(ExceptionDumpInfo, tensors) == 144,
              "Exception dump metadata ABI");
static_assert(sizeof(ExceptionDumpInfo) == 656, "Exception dump metadata ABI");

} // namespace ascend
} // namespace tl

#endif // TILELANG_TL_TEMPLATES_ASCEND_EXCEPTION_DUMP_H_
