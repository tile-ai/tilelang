#include "transform/common/loop_aware_visitor.h"
#include "compat/analyzer.h"

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>

namespace tvm::tl {
using namespace tvm::tirx;
using namespace tvm::prim;

ffi::Optional<VisitInterrupt> LoopAwareVisitor::Visit_(const ForNode *op) {
  return constraint_scope_.WithNewScope([&]() -> ffi::Optional<VisitInterrupt> {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->begin));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->end));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->step));
    bool pure_bounds = SideEffect(op->begin) == CallEffectKind::kPure &&
                       SideEffect(op->end) == CallEffectKind::kPure &&
                       SideEffect(op->step) == CallEffectKind::kPure;
    if (pure_bounds && tl::arith::CanProve(analyzer_, op->step > 0))
      tl::arith::Bind(analyzer_, op->var, Range(op->begin, op->end));
    return constraint_scope_.WithNewScope([&]()
                                              -> ffi::Optional<VisitInterrupt> {
      if (pure_bounds) {
        if (tl::arith::CanProve(analyzer_, op->step > 0)) {
          constraint_scope_.Current().Emplace(analyzer_, op->end > op->begin);
          constraint_scope_.Current().Emplace(analyzer_,
                                              op->end - op->begin > 0);
        } else if (tl::arith::CanProve(analyzer_, op->step < 0)) {
          constraint_scope_.Current().Emplace(analyzer_, op->begin > op->end);
          constraint_scope_.Current().Emplace(analyzer_,
                                              op->begin - op->end > 0);
        }
        PrimExpr inside =
            (op->step > 0 && op->var >= op->begin && op->var < op->end) ||
            (op->step < 0 && op->var <= op->begin && op->var > op->end);
        constraint_scope_.Current().Emplace(analyzer_, inside);
      }
      return this->Visit(op->body);
    });
  });
}

} // namespace tvm::tl
