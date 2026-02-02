/*!
 * \file buffer_analysis.h
 * \brief Utilities for analyzing buffer accesses in TIR
 */

#pragma once

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tir;

/*! \brief Collect buffer variables that are written within a statement */
class WrittenBufferCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> written_buffers;

  static std::unordered_set<const VarNode *> Collect(const Stmt &stmt) {
    WrittenBufferCollector collector;
    collector(stmt);
    return std::move(collector.written_buffers);
  }

protected:
  void VisitStmt_(const BufferStoreNode *op) final {
    written_buffers.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK_GE(op->args.size(), 5U);
      if (const auto *buf = op->args[1].as<VarNode>()) {
        const auto *flag = op->args[4].as<IntImmNode>();
        if (!flag || (flag->value & 2)) { // Write flag
          written_buffers.insert(buf);
        }
      }
    } else if (op->op.same_as(builtin::address_of())) {
      if (const auto *load = op->args[0].as<BufferLoadNode>()) {
        written_buffers.insert(load->buffer->data.get());
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

/*! \brief Check if an expression reads any of the specified buffers */
class BufferReadChecker : public ExprVisitor {
public:
  bool reads_written_buffer = false;

  static bool Check(const PrimExpr &expr,
                    const std::unordered_set<const VarNode *> &written) {
    BufferReadChecker checker(written);
    checker(expr);
    return checker.reads_written_buffer;
  }

private:
  const std::unordered_set<const VarNode *> &written_buffers_;

  explicit BufferReadChecker(const std::unordered_set<const VarNode *> &written)
      : written_buffers_(written) {}

  void VisitExpr_(const BufferLoadNode *op) final {
    if (written_buffers_.count(op->buffer->data.get())) {
      reads_written_buffer = true;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      if (const auto *buf = op->args[1].as<VarNode>()) {
        const auto *flag = op->args[4].as<IntImmNode>();
        if ((!flag || (flag->value & 1)) && written_buffers_.count(buf)) {
          reads_written_buffer = true;
        }
      }
    }
    ExprVisitor::VisitExpr_(op);
  }
};

} // namespace tl
} // namespace tvm
