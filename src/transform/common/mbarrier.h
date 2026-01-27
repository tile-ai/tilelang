#ifndef TVM_TL_TRANSFORM_COMMON_MBARRIER_H_
#define TVM_TL_TRANSFORM_COMMON_MBARRIER_H_

#include "../../op/builtin.h"
#include <tvm/ir/expr.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Create an mbarrier buffer with shared.barrier storage scope.
 *
 * \param name The name of the buffer.
 * \param num_barriers The number of barriers in the buffer.
 * \return A Buffer object for mbarrier with shared.barrier scope.
 */
inline Buffer CreateMBarrierBuffer(const std::string &name, int num_barriers) {
  Var data(name, PointerType(PrimType(DataType::UInt(64)), "shared.barrier"));
  return Buffer(data, DataType::UInt(64),
                {IntImm(DataType::Int(32), num_barriers)}, {}, PrimExpr(), name,
                0, 0, kDefault);
}

const std::string injected_mbarrier_name_ =
    "mbarrier"; // todo: avoid conflict with user-defined mbarriers

} // namespace tl
} // namespace tvm
#endif // TVM_TL_TRANSFORM_COMMON_MBARRIER_H_
