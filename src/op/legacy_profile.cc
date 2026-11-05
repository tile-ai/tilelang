/*!
 * \file tl/op/legacy_profile.cc
 * \brief Retain the public legacy profile marker on the native IR boundary.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl {
namespace {

PrimExpr IgnoreLegacyProfile(PrimExpr call) {
  // Old TVM's ordinary LLVM backend ignores these Hexagon profiling markers
  // without evaluating the id operand.  Retain that rule, including an id
  // expression that would otherwise have effects.  A void Call can only be
  // used as a statement; after legalization its Evaluate contains this zero.
  return IntImm(PrimType::Int(32), 0, call->loc);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name :
       {"tl.start_profile_intrinsic", "tl.end_profile_intrinsic"}) {
    OpDef(name)
        .signature(sig::arg<IntExpr>("id", "The legacy profile marker id."))
        .set_attr(op_attr::kFixedReturnType, PrimType::Void())
        .set_attr(op_attr::kCallEffectKind, CallEffectKind::kPure)
        .set_attr<tirx::FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                         IgnoreLegacyProfile);
  }
}

} // namespace
} // namespace tvm::tl
