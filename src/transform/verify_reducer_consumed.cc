/*!
 * \file verify_reducer_consumed.cc
 * \brief Assert that no reducer v2 construct survives past materialization.
 *
 * Runs after LowerTileOp. If any `local.reducer` allocation, first-class
 * reducer op, unconsumed multiplicity marker, or reducer_info_v2 annotation
 * is still present, some pass failed to consume it — that must be a compile
 * error, never silently-wrong generated code.
 */

#include "support/check.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/reducer.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

void CheckFunc(const PrimFunc &f) {
  PostOrderVisit(f->body, [&](const ObjectRef &obj) {
    if (const auto *block = obj.as<SBlockNode>()) {
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(!IsReducerV2Buffer(buffer))
            << "internal error: reducer `" << buffer
            << "` (scope local.reducer) survived past "
               "ReducerPlanAndMaterialize; it was never materialized.";
      }
      ICHECK(!block->annotations.count(attr::kReducerInfoV2))
          << "internal error: reducer_info_v2 annotation survived past "
             "ReducerPlanAndMaterialize.";
    } else if (const auto *call = obj.as<CallNode>()) {
      ICHECK(!call->op.same_as(ReducerInitOp::Get()) &&
             !call->op.same_as(ReducerUpdateOp::Get()) &&
             !call->op.same_as(FinalizeReducerV2Op::Get()))
          << "internal error: first-class reducer op `" << call->op
          << "` survived past ReducerPlanAndMaterialize.";
    } else if (const auto *attr_stmt = obj.as<AttrStmtNode>()) {
      ICHECK(attr_stmt->attr_key != attr::kParallelMultiplicity)
          << "internal error: unconsumed tl.parallel_multiplicity marker "
             "survived past PartitionLoop; the marked statement's execution "
             "count is undefined.";
    }
  });
}

} // namespace

using namespace tirx::transform;

tvm::transform::Pass VerifyReducerConsumed() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    CheckFunc(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VerifyReducerConsumed", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.VerifyReducerConsumed",
                        VerifyReducerConsumed);
}

} // namespace tl
} // namespace tvm
