/*!
 * \file common.h
 * \brief Common utilities for TL transforms
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt.h>

#include <tvm/tir/builtin.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"
#include <queue>

#include "../../op/utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

class ParallelLoopTransformer : public IRMutatorWithAnalyzer {
public:
  static Stmt Substitute(const Stmt &stmt, bool skip_thread_partition = false) {
    arith::Analyzer analyzer;
    ParallelLoopTransformer transformer(&analyzer);
    return transformer.VisitStmt(stmt);
  }

  ParallelLoopTransformer(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  Stmt VisitStmt_(const ForNode *op) final {

    if (op->kind != ForKind::kParallel)
      return StmtMutator::VisitStmt_(op);

    // Collect loop variables and ranges
    auto for_node = tvm::ffi::GetRef<For>(op);
    Array<Var> loop_vars;
    Array<PrimExpr> loop_extents;
    Stmt body = op->body;

    // Bind the range of outer loop variables
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(0, op->extent));
    loop_vars.push_back(op->loop_var);
    loop_extents.push_back(op->extent);

    // If there are inner loops, bind their ranges as well
    while (const ForNode *inner = body.as<ForNode>()) {
      analyzer_->Bind(inner->loop_var, Range::FromMinExtent(0, inner->extent));
      loop_vars.push_back(inner->loop_var);
      loop_extents.push_back(inner->extent);
      body = inner->body;
    }

    ICHECK(loop_vars.size() == loop_extents.size())
        << "loop_vars and loop_extents size mismatch";

    // Collect buffer access information
    BufferAccessCollector collector;
    collector(op->body);

    PrimExpr condition;

    for (const auto &[buffer, indices] : collector.buffer_indices) {
      ICHECK(indices.size() == buffer->shape.size())
          << "indices size mismatch with buffer shape";

      for (size_t i = 0; i < indices.size(); ++i) {
        auto index = indices[i];
        auto bound = analyzer_->const_int_bound(index);

        // Collect the variables that used in the index
        std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> used_vars;
        // post order visit the index
        PostOrderVisit(index, [&](const ObjectRef &obj) {
          if (const VarNode *v = obj.as<VarNode>()) {
            used_vars.insert(tvm::ffi::GetRef<Var>(v));
          }
        });
        if (used_vars.empty()) {
          continue;
        }

        // find related loop vars
        Array<Var> related_loop_vars;
        for (size_t j = 0; j < loop_vars.size(); ++j) {
          auto loop_var = loop_vars[j];
          // if find related, pop the loop_vars and loop_extents
          if (used_vars.count(loop_var)) {
            related_loop_vars.push_back(loop_var);
          }
          if (related_loop_vars.size() > 1) {
            // Only one related loop var is supported transformation currently.
            return for_node;
          }

          auto bound = analyzer_->const_int_bound(index);
          int64_t upper_bound = bound->max_value + 1;
          int64_t shape = Downcast<IntImm>(buffer->shape[i])->value;
          if (upper_bound < shape) {
            PrimExpr predicate = LT(index, IntImm(index.dtype(), upper_bound));
            condition =
                condition.defined() ? And(condition, predicate) : predicate;
          }
        }
      }
    }

    if (condition.defined()) {
      body = IfThenElse(condition, body);

      for (int j = loop_vars.size() - 1; j >= 0; --j) {
        auto loop_var = loop_vars[j];
        auto loop_extent = loop_extents[j];
        body = For(loop_var, 0, loop_extent, ForKind::kParallel, body);
      }

      return Downcast<For>(body);
    }

    // Only traverse the outer loop
    return for_node;
  }

  // Helper class for collecting buffer access information, only counts fragment
  // buffer access
  class BufferAccessCollector : public StmtExprVisitor {
  public:
    void VisitExpr_(const BufferLoadNode *op) final {
      if (IsFragmentBuffer(op->buffer)) {
        if (buffer_indices.find(op->buffer) == buffer_indices.end()) {
          buffer_indices[op->buffer] = op->indices;
        } else {
          // check equal
          ICHECK(StructuralEqual()(buffer_indices[op->buffer], op->indices))
              << "indices mismatch for buffer: " << op->buffer;
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    void VisitStmt_(const BufferStoreNode *op) final {
      if (IsFragmentBuffer(op->buffer)) {
        if (buffer_indices.find(op->buffer) == buffer_indices.end()) {
          buffer_indices[op->buffer] = op->indices;
        } else {
          // check equal
          ICHECK(StructuralEqual()(buffer_indices[op->buffer], op->indices))
              << "indices mismatch for buffer: " << op->buffer;
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    std::unordered_map<Buffer, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>
        buffer_indices;
  };
};

} // namespace tl
} // namespace tvm
