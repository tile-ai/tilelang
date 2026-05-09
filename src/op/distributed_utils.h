/*!
 * \file tl/op/distributed_utils.h
 * \brief Compiler-side helpers for distributed address lowering.
 */

#ifndef TVM_TL_OP_DISTRIBUTED_UTILS_H_
#define TVM_TL_OP_DISTRIBUTED_UTILS_H_

#include <tvm/tirx/builtin.h>

#include "distributed.h"
#include "operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

inline PrimExpr RemotePESentinel() {
  return IntImm(DataType::Int(64), -1);
}

inline bool IsRemotePE(const PrimExpr &pe) {
  if (const auto *int_pe = pe.as<IntImmNode>()) {
    return int_pe->value != -1;
  }
  return true;
}

inline PrimExpr MakeAddress(const Buffer &buffer,
                            const Array<PrimExpr> &indices) {
  return Call(DataType::Handle(), builtin::address_of(),
              {BufferLoad(buffer, indices)});
}

inline PrimExpr MakeRemappedAddress(const LowerArgs &T, const Buffer &buffer,
                                    const Array<PrimExpr> &indices) {
  Buffer remapped = buffer;
  if (T.buffer_remap.count(buffer)) {
    remapped = T.buffer_remap[buffer];
  }
  return MakeAddress(remapped, indices);
}

inline PrimExpr GetOffsetFromLocalBase(PrimExpr local_addr) {
  PrimExpr local_rank = Call(DataType::Int(64), tl::get_rank(), {});
  PrimExpr local_base_ptr =
      Call(DataType::Handle(), tl::get_remote_base_ptr(), {local_rank});
  return Sub(Call(DataType::Handle(), tl::get_uintptr_t(), {local_addr}),
             local_base_ptr);
}

inline PrimExpr RemapRemoteAddress(PrimExpr local_addr, PrimExpr remote_pe) {
  if (!IsRemotePE(remote_pe)) {
    return local_addr;
  }
  return Call(DataType::Handle(), tl::get_remote_base_ptr(), {remote_pe}) +
         GetOffsetFromLocalBase(std::move(local_addr));
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_DISTRIBUTED_UTILS_H_
