/*!
 * \file intrin_rule_c.cc
 * \brief Portable int32 intrinsic lowering for C hosts.
 */
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

TVM_REGISTER_OP("tirx.clz")
    .set_attr<FLowerIntrinsic>("c.FLowerIntrinsic", [](PrimExpr expr) {
      Call call = Downcast<Call>(expr);
      DataType dtype = call->args[0].dtype();
      if (!dtype.is_scalar() || dtype.bits() != 32 ||
          (!dtype.is_int() && !dtype.is_uint())) {
        return expr;
      }
      PrimExpr value = cast(DataType::UInt(dtype.bits()), call->args[0]);
      PrimExpr count = make_const(call.dtype(), 0);
      for (int bit = 0; bit < dtype.bits(); ++bit) {
        count =
            count + cast(call.dtype(),
                         value < make_const(value.dtype(), uint64_t{1} << bit));
      }
      return count;
    });
} // namespace tl
} // namespace tvm
