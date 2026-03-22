#ifndef TVM_TL_TRANSFORM_COMMON_TMA_COPY_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_TMA_COPY_UTILS_H_

#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tir;

inline Stmt StripTmaCopyWriteBufferAttr(Stmt stmt) {
  class TmaCopyWriteBufferAttrStripper : public StmtExprMutator {
  public:
    Stmt VisitStmt_(const AttrStmtNode *op) final {
      if (op->attr_key == "tl.tma_copy_write_buffer") {
        return VisitStmt(op->body);
      }
      return StmtExprMutator::VisitStmt_(op);
    }
  };

  return TmaCopyWriteBufferAttrStripper()(std::move(stmt));
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_TMA_COPY_UTILS_H_
