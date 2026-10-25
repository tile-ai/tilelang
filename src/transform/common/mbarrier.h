#ifndef TVM_TL_TRANSFORM_COMMON_MBARRIER_H_
#define TVM_TL_TRANSFORM_COMMON_MBARRIER_H_

#include "../../op/builtin.h"
#include <tvm/ir/expr.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;

/*!
 * \brief Create an mbarrier buffer with shared.barrier storage scope.
 *
 * \param name The name of the buffer.
 * \param num_barriers The number of barriers in the buffer.
 * \return A Buffer object for mbarrier with shared.barrier scope.
 */
inline TensorVar CreateMBarrierBuffer(const std::string &name,
                                      int num_barriers) {
  auto scope = runtime::MemoryScope::IsDefined("shared.barrier")
                   ? runtime::MemoryScope::Get("shared.barrier")
                   : runtime::MemoryScope::Register("shared.barrier");
  return TensorVar(
      name, TensorType({PrimExpr(num_barriers)}, PrimType::UInt(64), scope, 8));
}

/*!
 * \brief Create a BufferLoad reference to a specific barrier slot.
 *
 * \param barrier_buf The shared.barrier scope buffer.
 * \param barrier_id  The index expression for the barrier slot.
 * \return A BufferLoad expression referencing the barrier.
 */
inline PrimExpr MakeBarrierRef(const TensorVar &barrier_buf,
                               PrimExpr barrier_id) {
  return TensorLoad(barrier_buf, {std::move(barrier_id)});
}

const std::string injected_mbarrier_name_ =
    "mbarrier"; // todo: avoid conflict with user-defined mbarriers

} // namespace tl
} // namespace tvm
#endif // TVM_TL_TRANSFORM_COMMON_MBARRIER_H_
