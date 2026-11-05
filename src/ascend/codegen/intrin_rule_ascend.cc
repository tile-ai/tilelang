/*!
 * \file intrin_rule_ascend.cc
 * \brief Ascend intrinsic rules for math functions.
 *
 * The default FloatSuffix lowering uses exact dtype matching
 * (t == DataType::Float(32)) which fails for vectorized types.
 * AscendMath uses t.MatchesCode(kDLFloat) like CUDAMath to support any lane
 * count.
 *
 * Registered with "ascend.FLowerIntrinsic" for the split Ascend backend.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/attr.h>

#include "backend/common/codegen/intrin_rule.h"

namespace tvm::tl {
namespace {

using tirx::FLowerIntrinsic;

struct AscendMath {
  std::string operator()(const PrimType &t, const std::string &name) const {
    if (t.MatchesCode(kDLFloat)) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16:
        return "h" + name;
      default:
        return "";
      }
    }
    return "";
  }
};

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name :
       {"sqrt",  "rsqrt", "exp",   "exp2", "exp10", "log",   "log2",
        "log10", "log1p", "tanh",  "tan",  "sin",   "cos",   "atan",
        "atan2", "erf",   "floor", "ceil", "fabs",  "round", "trunc"}) {
    OpDef(ffi::String("prim.") + name)
        .set_attr<tirx::FLowerIntrinsic>(
            "tl.ascend.FLowerIntrinsic",
            intrin::DispatchPureExtern<AscendMath>);
  }
}
} // namespace tvm::tl
