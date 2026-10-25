#ifndef TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_
#define TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_

#include "op/parallel.h"
#include <tvm/tirx/stmt_functor.h>

#include <functional>

namespace tvm::tl {

inline const ForNode *SingleLoop(const SeqStmt &body) {
  return body->seq.size() == 1 ? body->seq[0].as<ForNode>() : nullptr;
}

class ParallelLoopLayoutValidator : public tirx::StmtExprVisitor {
public:
  explicit ParallelLoopLayoutValidator(
      std::function<bool(const RegionStmtNode *)> opaque_region = {})
      : opaque_region_(std::move(opaque_region)) {}

  static void
  Validate(const Stmt &statement,
           std::function<bool(const RegionStmtNode *)> opaque_region = {}) {
    ffi::make_object<ParallelLoopLayoutValidator>(std::move(opaque_region))
        ->Visit(statement);
  }

private:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    if (opaque_region_ && opaque_region_(node))
      return std::nullopt;
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *node) final {
    if (node->kind != ForKind::kParallel)
      return StmtExprVisitor::Visit_(node);
    auto annotation = node->annotations.Get(attr::kParallelLoopLayout);
    TVM_FFI_CHECK(annotation.has_value(), ValueError)
        << "Parallel loop missing layout annotation: "
        << ffi::GetRef<For>(node);
    Fragment layout = annotation.value().as_or_throw<Fragment>();
    const ForNode *inner = node;
    int dimensions = 1;
    while (const ForNode *child = SingleLoop(inner->body)) {
      if (child->kind != ForKind::kParallel)
        break;
      TVM_FFI_CHECK(!child->annotations.count(attr::kParallelLoopLayout),
                    ValueError)
          << "Only the outermost parallel loop may own a layout";
      inner = child;
      ++dimensions;
    }
    TVM_FFI_CHECK_EQ(layout->InputDim(), dimensions, ValueError)
        << "Layout InputDim mismatch for parallel loop";
    return Visit(inner->body);
  }

  std::function<bool(const RegionStmtNode *)> opaque_region_;
};

} // namespace tvm::tl

#endif
