/*!
 * \file ascend_exception_dump.h
 * \brief Ascend exception callback registration and host-argument decoding.
 */
#ifndef TILELANG_ASCEND_RUNTIME_ASCEND_EXCEPTION_DUMP_H_
#define TILELANG_ASCEND_RUNTIME_ASCEND_EXCEPTION_DUMP_H_

#include "tl_templates/ascend/exception_dump.h"

#include <cstddef>
#include <cstdint>

namespace tvm {
namespace ascend {

// Called only for kernels compiled with tl.ascend_exception_dump. Registration
// is thread-safe; failure is reported to the caller and can be retried.
void EnsureExceptionDumpCallback();

// These helpers also accept unaligned host memory. A malformed/foreign argument
// buffer is ignored, and no exception escapes into CANN's callback thread.
bool FindExceptionDumpInfo(const void *args, size_t size,
                           ::tl::ascend::ExceptionDumpInfo *info) noexcept;
// Returns the CANN save result (0 on success), or -1 for an unrecognized
// record, -2 for no dumpable tensors, and -3 for an unavailable API/internal
// failure.
int32_t DumpExceptionHostArgs(const void *args, size_t size) noexcept;

} // namespace ascend
} // namespace tvm

#endif // TILELANG_ASCEND_RUNTIME_ASCEND_EXCEPTION_DUMP_H_
