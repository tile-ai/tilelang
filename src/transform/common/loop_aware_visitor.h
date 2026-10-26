#include "compat/metadata.h"
#ifndef TL_TRANSFORM_COMMON_LOOP_AWARE_VISITOR_H_
#define TL_TRANSFORM_COMMON_LOOP_AWARE_VISITOR_H_

#include <tvm/tirx/stmt_functor_with_analyzer.h>

namespace tvm::tl {

class LoopAwareVisitor : public tvm::tl::StmtExprVisitorWithAnalyzer {
public:
  using tvm::tl::StmtExprVisitorWithAnalyzer::Visit_;
  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *loop) override;
};

} // namespace tvm::tl
#endif
