/*!
 * \file tl/ascend/target_utils.cc
 * \brief Ascend target attribute helpers.
 */

#include "ascend/target_utils.h"

#include <tvm/arith/pattern.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>

#include "op/builtin.h"

namespace tvm {
namespace tl {

using ffi::String;

bool TargetIsAscend(Target target) {
  return target.defined() && target->kind->name == "ascend";
}

bool TargetIsPTO(Target target) {
  // PTO shares the "ascend" target kind with AscendC; "pto" in target keys
  // selects the PTO codegen (see tilelang/ascend/target.py). Keep this the
  // single C++ decision point so LayoutInference, Verify/Vectorize passes
  // and any later consumer cannot drift apart on the key spelling.
  if (!TargetIsAscend(target)) {
    return false;
  }
  for (const String &key : target->keys) {
    if (key == "pto") {
      return true;
    }
  }
  return false;
}

ffi::Optional<String> MatchAscendVectorRng(const PrimExpr &value,
                                           int64_t lanes) {
  const auto *call = value.as<tirx::CallNode>();
  if (!call || !call->op.same_as(rng_rand_float()) ||
      call->dtype != DataType::Float(32) || (lanes != 2 && lanes != 4)) {
    return std::nullopt;
  }
  ICHECK_EQ(call->args.size(), 1);
  const auto *dist = call->args[0].as<tirx::StringImmNode>();
  ICHECK(dist) << "Ascend RNG distribution must be constant";
  if (dist->value == "uniform" || dist->value == "normal") {
    return dist->value;
  }
  return std::nullopt;
}

// AscendC emits one Philox vector draw for a contiguous store. Keeping this
// case in lane order requires a single draw and an effect-free address.
bool CanVectorizeAscendRngStore(const Target &target, const tirx::For &loop) {
  using namespace tirx;
  if (!TargetIsAscend(target) || TargetIsPTO(target)) {
    return false;
  }
  const auto *store = loop->body.as<BufferStoreNode>();
  const auto *lanes = as_const_int(loop->extent);
  if (!store || !lanes || store->predicate || store->indices.size() != 1 ||
      !store->value.dtype().is_scalar() ||
      SideEffect(store->indices[0]) != CallEffectKind::kPure) {
    return false;
  }
  PrimExpr value = store->value;
  if (const auto *cast = value.as<CastNode>()) {
    value = cast->value;
  }
  if (!MatchAscendVectorRng(value, *lanes)) {
    return false;
  }
  auto coeff = arith::DetectLinearEquation(store->indices[0], {loop->loop_var});
  return !coeff.empty() && is_one(coeff[0]);
}

bool IsAscendVectorizableFP8(DataType dtype) {
  // NOTE: E8M0 is a special type of FP8 which is not handled here.
  // We only handle FP8 types which can be represented with
  // __asc_fp8_interpretation_t here.
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TargetIsAscend",
                        [](Target target) { return TargetIsAscend(target); });
}

} // namespace tl
} // namespace tvm
