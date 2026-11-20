/* Direct PlanValueCast context probe.
 *
 * Calls PlanValueCast directly with CastContext combinations that the
 * Verify/Vectorize passes cannot produce end-to-end, and checks both the
 * rejection reasons and the emitted vector/scalar plans.
 *
 * The two passes can only produce the context combinations that their
 * classification yields (scalar lanes=1, VMI lanes=64/128/256).  The planner
 * guards for {kVectorVmi, 1} and {kScalar, 128} are therefore unreachable
 * end-to-end; this local probe calls the planner directly so those contracts
 * are tested without adding a product FFI.
 *
 * Build (from the repository root; the library lives in build/lib and the
 * TVM DSOs must be linked explicitly):
 *   c++ -std=c++17 -O0 testing/ascend/language/pto/cast_plan_context_probe.cc \
 *       -Isrc -I3rdparty/tvm/include -I3rdparty/tvm/src \
 *       -I3rdparty/tvm/3rdparty/dlpack/include \
 *       -I3rdparty/tvm/3rdparty/tvm-ffi/include \
 *       -I3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include \
 *       -Lbuild/lib -ltilelang -ltvm_compiler -ltvm_ffi -ltvm_runtime \
 *       -Wl,-rpath,$PWD/build/lib -o /tmp/cast_plan_probe
 *   LD_LIBRARY_PATH=$PWD/build/lib /tmp/cast_plan_probe
 */
#include <iostream>
#include <string>
#include <variant>

#include "ascend/transform/parallel_to_pto_utils.h"
#include <tvm/tirx/expr.h>

namespace {

using tvm::DataType;
using tvm::tirx::Cast;
using tvm::tirx::Var;
using tvm::tl::pto::CastContext;
using tvm::tl::pto::CastValueKind;
using tvm::tl::pto::PlanValueCast;
using tvm::tl::pto::ScalarCastAction;
using tvm::tl::pto::ScalarCastPlan;
using tvm::tl::pto::VectorCastOp;
using tvm::tl::pto::VectorCastPlan;

int failures = 0;

void Check(bool condition, const std::string &what) {
  if (!condition) {
    std::cout << "FAIL " << what << "\n";
    ++failures;
  } else {
    std::cout << "PASS " << what << "\n";
  }
}

const VectorCastPlan *
AsVector(const tvm::ffi::Optional<tvm::tl::pto::CastPlan> &plan,
         const std::string &what) {
  if (!plan.has_value()) {
    std::cout << "FAIL " << what << ": no plan\n";
    ++failures;
    return nullptr;
  }
  const auto *vector = std::get_if<VectorCastPlan>(&plan.value());
  if (vector == nullptr) {
    std::cout << "FAIL " << what << ": not a vector plan\n";
    ++failures;
  }
  return vector;
}

const ScalarCastPlan *
AsScalar(const tvm::ffi::Optional<tvm::tl::pto::CastPlan> &plan,
         const std::string &what) {
  if (!plan.has_value()) {
    std::cout << "FAIL " << what << ": no plan\n";
    ++failures;
    return nullptr;
  }
  const auto *scalar = std::get_if<ScalarCastPlan>(&plan.value());
  if (scalar == nullptr) {
    std::cout << "FAIL " << what << ": not a scalar plan\n";
    ++failures;
  }
  return scalar;
}

} // namespace

int main() {
  Var x("x", DataType::Int(8));
  Cast widen(DataType::Int(16), x);
  Cast cross(DataType::UInt(16), x);
  Cast same_width(DataType::UInt(8), x);

  std::string why;
  CastContext vector_lane1{CastValueKind::kVectorVmi, 1};
  auto plan = PlanValueCast(widen.get(), vector_lane1, &why);
  Check(!plan.has_value() &&
            why.find("requires logical lanes 64/128/256") != std::string::npos,
        "kVectorVmi lanes=1 is rejected as VMI, never scalar: " + why);

  why.clear();
  CastContext scalar_lanes{CastValueKind::kScalar, 128};
  plan = PlanValueCast(widen.get(), scalar_lanes, &why);
  Check(!plan.has_value() &&
            why.find("must have exactly one logical lane") != std::string::npos,
        "kScalar lanes=128 is rejected, never VMI: " + why);

  why.clear();
  CastContext vector64{CastValueKind::kVectorVmi, 64};
  plan = PlanValueCast(cross.get(), vector64, &why);
  if (const VectorCastPlan *steps = AsVector(plan, "int8->uint16 VMI plan")) {
    Check(steps->steps.size() == 2, "int8->uint16 has 2 steps");
    if (steps->steps.size() == 2) {
      Check(steps->steps[0].op == VectorCastOp::kIntegerExtend &&
                steps->steps[0].from == DataType::Int(8) &&
                steps->steps[0].to == DataType::Int(16),
            "step 1 extends with the source signedness to the target width");
      Check(steps->steps[1].op == VectorCastOp::kIntegerReinterpret &&
                steps->steps[1].from == DataType::Int(16) &&
                steps->steps[1].to == DataType::UInt(16),
            "step 2 reinterprets at the same width");
    }
  }

  why.clear();
  CastContext scalar1{CastValueKind::kScalar, 1};
  plan = PlanValueCast(cross.get(), scalar1, &why);
  if (const ScalarCastPlan *scalar =
          AsScalar(plan, "int8->uint16 scalar plan")) {
    Check(scalar->action == ScalarCastAction::kRebuildCast,
          "int8->uint16 scalar keeps a rebuilt ordinary Cast");
    Check(scalar->annotations.empty(),
          "scalar annotations are validated/empty");
  }

  why.clear();
  CastContext vector128{CastValueKind::kVectorVmi, 128};
  plan = PlanValueCast(same_width.get(), vector128, &why);
  if (const VectorCastPlan *steps =
          AsVector(plan, "int8->uint8 VMI same-width plan")) {
    Check(steps->steps.size() == 1 &&
              steps->steps[0].op == VectorCastOp::kIntegerReinterpret,
          "same-width cross-sign is one vinterpret_cast");
  }

  return failures == 0 ? 0 : 1;
}
