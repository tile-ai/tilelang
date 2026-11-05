/*!
 * \file intrin_rule_metal.cc
 * \brief Metal intrinsic lowering rules that TVM's own table lacks.
 *
 * TVM registers the common math intrinsics for "metal.FLowerIntrinsic" in
 * 3rdparty/tvm/src/target/metal/intrin_rule_metal.cc. rsqrt is missing
 * there, so it fell through to the default legalizer and became
 * 1 / sqrt(x): an extra division, and a different rounding from the
 * native instruction (RMSNorm-style kernels compared against hand-written
 * Metal that uses rsqrt() differ by an ULP). Metal Shading Language has
 * rsqrt() for float and half.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

TVM_REGISTER_OP("tirx.rsqrt")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                               DispatchPureExtern<Direct>);

} // namespace intrin
} // namespace codegen
} // namespace tvm
