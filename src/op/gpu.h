#ifndef TILELANG_OP_GPU_H_
#define TILELANG_OP_GPU_H_

#include <tvm/tirx/op/gpu.h>

namespace tvm::tl {

const Op &StorageSyncOp();

inline bool IsStorageSyncOp(const Expr &operation) {
  return operation.same_as(StorageSyncOp()) ||
         operation.same_as(tirx::gpu_storage_sync_op());
}

} // namespace tvm::tl

#endif
