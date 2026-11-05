/*!
 * \file cpu/codegen/intrin_rule_c.cc
 * \brief Native scalar C math calls required by TileLang's CPU emitter.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/attr.h>

#include "backend/common/codegen/intrin_rule.h"

namespace tvm::tl {
namespace {

struct CMath {
  std::string operator()(const PrimType &type, const std::string &name) const {
    if (!type.IsScalar() || !type.MatchesCode(kDLFloat))
      return "";
    if (type.bits() == 32)
      return name + "f";
    if (type.bits() == 64)
      return name;
    return "";
  }
};

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  // The target's default table covers most scalar math, but not these calls.
  // Keep the additions on TileLang's C path and do not change device rules.
  for (const char *name : {"fabs", "exp2", "fmod"}) {
    OpDef(ffi::String("prim.") + name)
        .set_attr<tirx::FLowerIntrinsic>("tl.c.FLowerIntrinsic",
                                         intrin::DispatchPureExtern<CMath>);
  }
}

} // namespace tvm::tl
