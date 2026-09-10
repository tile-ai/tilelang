/*!
 * \file tl/metal/op/math.cc
 * \brief Metal implementations for tl.* math intrinsics that need a
 *        target-specific lowering (registered via FLowerIntrinsic).
 *
 * Only intrinsics whose Metal fold is safe are registered here:
 *   - tl.infinity folds to a constant FloatImm (target-independent).
 * Intrinsics whose CUDA fold emits a device-template extern call
 * (tl.pow_of_int -> tl::pow_of_int<...>, round_ties_away_from_zero ->
 * tl::RoundTiesAwayFromZero) are NOT registered until MSL template
 * equivalents exist in the Metal codegen preamble.
 */

#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;

PrimExpr infinity_op(PrimExpr args);

namespace {

TVM_REGISTER_OP("tl.infinity")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", infinity_op);

} // namespace

} // namespace tl
} // namespace tvm
