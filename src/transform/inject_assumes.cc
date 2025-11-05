
#include "tvm/arith/analyzer.h"
#include "tvm/ffi/optional.h"
#include "tvm/ir/expr.h"
#include "tvm/ir/transform.h"
#include "tvm/node/structural_hash.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"
#include <sstream>

namespace tvm::tl {
using namespace tir;

class AssumeInjector : public tvm::tir::StmtExprMutator {
  using Base = tvm::tir::StmtExprMutator;

public:
  AssumeInjector(PrimFunc f) : f(f) {}
  static PrimFunc Substitute(PrimFunc f) {
    auto injector = AssumeInjector(f);
    f.CopyOnWrite()->body = injector(f->body);
    return f;
  }

private:
  struct AssertCreator {
    struct Item {
      PrimExpr expr;
      std::vector<Buffer> buffers;
    };
    tvm::StructuralHash sh;
    tvm::StructuralEqual se;
    // grouped by expr, since the amount of variadic shape symbols is usually
    // much smaller than buffer
    std::vector<Item> items;
    // hash => index in items
    std::unordered_map<size_t, std::vector<size_t>> buckets;
    void addExpr(PrimExpr e, Buffer buffer) {
      size_t h = sh(e);
      auto &bucket = buckets[h];
      auto it = std::find_if(bucket.begin(), bucket.end(), [&](size_t y) {
        return se(e, items[y].expr, true);
      });
      if (it == bucket.end()) {
        auto index = items.size();
        items.push_back({e, {buffer}});
        bucket.push_back(index);
      } else {
        items[*it].buffers.push_back(buffer);
      }
    }
    void addBuffer(Buffer buf) {
      for (auto shape : buf->shape) {
        if (shape->IsInstance<IntImmNode>())
          continue;
        addExpr(shape, buf);
      }
    }
    Stmt build(Stmt body) {
      auto analyzer = arith::Analyzer{};
      for (const auto &e : items) {
        auto simplified = analyzer.Simplify(GT(e.expr, 0));
        std::stringstream ss;
        ss << "Buffer shape should be greater than 0: shape `" << e.expr
           << "` from buffer ";
        for (size_t i = 0; i < e.buffers.size(); i++) {
          if (i)
            ss << ", ";
          ss << "`" << e.buffers[i]->name << "`";
        }
        body = AttrStmt(simplified, tir::attr::tilelang_assume,
                        StringImm(ss.str()), body);
      }
      return body;
    }
  };
  Stmt VisitStmt_(const DeclBufferNode *op) final {
    auto body = VisitStmt(op->body);
    AssertCreator c;
    c.addBuffer(op->buffer);
    return DeclBuffer(op->buffer, c.build(body), op->span);
  }
  std::optional<PrimExpr> getAssumeExpr(Stmt stmt) {
    auto eval = stmt.as<EvaluateNode>();
    if (!eval)
      return std::nullopt;
    auto call = eval->value.as<CallNode>();
    if (!call)
      return std::nullopt;
    if (!call->op.same_as(builtin::assume()))
      return std::nullopt;
    return call->args[0];
  }
  Stmt VisitStmt_(const SeqStmtNode *op) final {
    struct AssumeGroup {
      std::optional<PrimExpr> e;
      std::vector<Stmt> stmts;
    };
    std::vector<AssumeGroup> groups = {AssumeGroup{std::nullopt, {}}};
    for (auto i = 0; i < op->seq.size(); i++) {
      auto stmt = VisitStmt(op->seq[i]);
      if (auto e = getAssumeExpr(stmt)) {
        groups.push_back(AssumeGroup{*e, {}});
      } else {
        groups.back().stmts.push_back(stmt);
      }
    }
    for (size_t i = groups.size(); i--;) {
      auto &g = groups[i];
      if (g.e) {
        Stmt body = g.stmts.size() == 1 ? g.stmts[0] : SeqStmt(g.stmts);
        std::stringstream ss;
        ss << "Assume: " << *(g.e);
        AttrStmt attr = AttrStmt(*g.e, tir::attr::tilelang_assume,
                                 StringImm(ss.str()), body);
        groups[i - 1].stmts.push_back(attr);
      } else {
        ICHECK(i == 0) << "only the first group can have no assume";
      }
    }
    return groups[0].stmts.size() == 1 ? groups[0].stmts[0]
                                       : SeqStmt(groups[0].stmts);
    // return SeqStmt(groups[0].stmts);
  }
  Stmt VisitStmt_(const BlockNode *op) final {
    auto body = VisitStmt(op->body);
    AssertCreator c;
    if (root_node) {
      for (auto item : f->buffer_map) {
        c.addBuffer(item.second);
      }
    }
    for (auto item : op->alloc_buffers) {
      c.addBuffer(item);
    }
    for (auto item : op->match_buffers) {
      c.addBuffer(item->buffer);
    }
    return Block(op->iter_vars, op->reads, op->writes, op->name_hint,
                 c.build(body), op->init, op->alloc_buffers, op->match_buffers,
                 op->annotations, op->span);
  }
  PrimFunc f;
  bool root_node{true};
};

using namespace tir::transform;

tvm::transform::Pass InjectAssumes() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AssumeInjector::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAssumes", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectAssumes", InjectAssumes);
}

} // namespace tvm::tl
