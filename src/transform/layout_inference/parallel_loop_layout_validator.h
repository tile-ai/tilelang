/*!
 * \file parallel_loop_layout_validator.h
 * \brief Validate explicit native metadata for complete parallel loop nests.
 */
#ifndef TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_
#define TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_

#include "../common/native_layout_utils.h"
#include "native_policy.h"

namespace tvm::tl {

inline int CountNestedParallelLoops(const ForObj *loop) {
  int count = 0;
  while (loop && loop->kind == ForKind::kParallel) {
    ++count;
    loop = SingleNestedFor(loop->body);
  }
  return count;
}

class ParallelLoopLayoutValidator : public tirx::StmtExprVisitor {
public:
  explicit ParallelLoopLayoutValidator(
      NativeLayoutInferenceOptions options = {})
      : options_(std::move(options)) {}
  static void Validate(const Stmt &stmt,
                       NativeLayoutInferenceOptions options = {}) {
    ffi::make_object<ParallelLoopLayoutValidator>(std::move(options))
        ->Visit(stmt);
  }
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    if (options_.IsOpaque(op))
      return std::nullopt;
    auto previous = hints_;
    if (op->op->name == "tl.attr") {
      auto annotation = ReadNativeAnnotations(op);
      if (annotation.at("key").cast<ffi::String>() == "tl.loop.annotations")
        hints_ = annotation.at("value").cast<ffi::Map<ffi::String, ffi::Any>>();
    }
    auto result = StmtExprVisitor::Visit_(op);
    hints_ = std::move(previous);
    return result;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForObj *op) final {
    if (op->kind != ForKind::kParallel)
      return StmtExprVisitor::Visit_(op);
    auto annotations = ReadLoopAnnotations(op);
    for (const auto &[key, value] : hints_)
      annotations.Set(key, value);
    auto layout = annotations.Get(attr::kParallelLoopLayout);
    TVM_FFI_CHECK(layout.has_value(), ValueError)
        << "Parallel loop is missing inferred layout metadata: "
        << ffi::GetRef<For>(op);
    auto fragment = layout->cast<Fragment>();
    int count = CountNestedParallelLoops(op);
    TVM_FFI_CHECK_EQ(static_cast<int>(fragment->InputDim()), count, ValueError)
        << "Parallel layout InputDim must match the complete loop nest";
    const ForObj *inner = op;
    for (int i = 1; i < count; ++i) {
      inner = SingleNestedFor(inner->body);
      TVM_FFI_CHECK(!inner->annotations.count(attr::kParallelLoopLayout),
                    ValueError)
          << "Only the outermost parallel loop may carry a layout";
    }
    auto previous = hints_;
    hints_ = {};
    auto result = Visit(inner->body);
    hints_ = std::move(previous);
    return result;
  }

private:
  NativeLayoutInferenceOptions options_;
  ffi::Map<ffi::String, ffi::Any> hints_;
};

} // namespace tvm::tl
#endif // TVM_TL_TRANSFORM_PARALLEL_LOOP_LAYOUT_VALIDATOR_H_
