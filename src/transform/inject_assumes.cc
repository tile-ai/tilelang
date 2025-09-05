
#include "tvm/arith/analyzer.h"
#include "tvm/ir/expr.h"
#include "tvm/ir/transform.h"
#include "tvm/node/structural_hash.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"

namespace tvm::tl {
using namespace tir;

class AssumeInjector : public tvm::tir::StmtExprMutator {
  using Base = tvm::tir::StmtExprMutator;
public:
  AssumeInjector(PrimFunc f): f(f) {}
  static PrimFunc Substitute(PrimFunc f) {
    auto injector = AssumeInjector(f);
    f.CopyOnWrite()->body = injector(f->body);
    return f;
  }
private:
  struct AssertCreator {
    tvm::StructuralHash sh;
    tvm::StructuralEqual se;
    std::unordered_map<size_t, std::vector<tvm::PrimExpr>> buckets;
    std::vector<PrimExpr> exprs;
    void addExpr(PrimExpr e) {
      size_t h = sh(e);
      auto bucket = buckets[h];
      auto it = std::find_if(bucket.begin(), bucket.end(), [&](auto y) {
        return se(e, y, true);
      });
      if(it == bucket.end()) {
        exprs.push_back(e);
        buckets[h].push_back(e);
      }
    }
    void addBuffer(Buffer buf) {
      for(auto shape: buf->shape) {
        if(shape->IsInstance<IntImmNode>()) continue;
        addExpr(shape);
      }
    }
    Stmt build(Stmt body) {
      if(exprs.empty()) return body;
      PrimExpr red = GT(exprs[0], 0);
      for(size_t i = 1; i < exprs.size(); ++i) {
        red = And(GT(exprs[i], 0), red);
      }
      auto simplified = arith::Analyzer{}.Simplify(red, 10);
      auto msg = StringImm("Invalid Buffer Shape: buffer shape should be greater than 0");
      return AssertStmt(simplified, msg, body);
    }
  };
  Stmt VisitStmt_(const DeclBufferNode* op) final {
    auto body = VisitStmt(op->body);
    AssertCreator c;
    c.addBuffer(op->buffer);
    return DeclBuffer(op->buffer, c.build(body), op->span);
  }
  Stmt VisitStmt_(const BlockNode* op) final {
    auto body = VisitStmt(op->body);
    AssertCreator c;
    if(root_node) {
      for(auto item: f->buffer_map) {
        c.addBuffer(item.second);
      }
    }
    for(auto item: op->alloc_buffers) {
      c.addBuffer(item);
    }
    for(auto item: op->match_buffers) {
      c.addBuffer(item->buffer);
    }
    return Block(
      op->iter_vars,
      op->reads,
      op->writes,
      op->name_hint,
      c.build(body),
      op->init,
      op->alloc_buffers,
      op->match_buffers,
      op->annotations,
      op->span
    );
  }
  PrimFunc f;
  bool root_node { true };
};

using namespace tir::transform;

tvm::transform::Pass InjectAssumes() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AssumeInjector::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAssumes", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectAssumes", InjectAssumes);
});

} // namespace tvm::tl
