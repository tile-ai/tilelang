#include "compat/metadata.h"
#ifndef TL_TRANSFORM_COMMON_LOOP_AWARE_MUTATOR_H_
#define TL_TRANSFORM_COMMON_LOOP_AWARE_MUTATOR_H_

#include <tvm/tirx/stmt_functor_with_analyzer.h>

namespace tvm::tl {

class LoopAwareMutator : public tvm::tl::StmtExprMutatorWithAnalyzer {
public:
  using tvm::tl::StmtExprMutatorWithAnalyzer::Mutate;
  using tvm::tl::StmtExprMutatorWithAnalyzer::Mutate_;
  using tvm::tl::StmtExprMutatorWithAnalyzer::StmtExprMutatorWithAnalyzer;
  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtNode *sequence,
                                 ffi::InplaceMode mode) override;
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *loop,
                                 ffi::InplaceMode mode) override;
};

} // namespace tvm::tl
#endif
