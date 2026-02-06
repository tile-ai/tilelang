/*!
 * \file legalize_safe_memory_access.cc
 * \brief legalize safe memory access
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <utility>

#include "../op/builtin.h"
#include "../op/parallel.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

// GlobalMemChecker for a BufferLoad/BufferStore node:
// 1. Identify BufferLoad and BufferStore nodes.
// 2. Check if the buffer is in global scope.
// 3. For each index, compare against the buffer's shape.
//    If the index might exceed the shape (upper bound too large),
//    log a warning or handle accordingly.
struct GlobalMemChecker : public StmtExprVisitor {

  GlobalMemChecker(arith::Analyzer *analyzer, bool recursively_collect_conds)
      : analyzer_(analyzer),
        recursively_collect_conds_(recursively_collect_conds) {}
  void VisitExpr_(const BufferLoadNode *op) final {
    // Check if the buffer is in global scope
    // This is because we are writing TilePrograms, where out of bounds
    // accesses only happen in the global buffer.
    if (IsGlobalBuffer(op->buffer)) {
      CheckBufferIndices(op->buffer, op->indices, /*is_load=*/true);
    }
    if (recursively_collect_conds_) {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // Check if the buffer is in global scope
    if (IsGlobalBuffer(op->buffer)) {
      CheckBufferIndices(op->buffer, op->indices, /*is_load=*/false);
    }
    if (recursively_collect_conds_) {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  // Helper function to determine if a buffer is global
  bool IsGlobalBuffer(const Buffer &buffer) {
    // The storage scope is often encoded in the buffer->data var name or
    // associated attributes. In typical TVM IR, global buffers have scope
    // "global". Here we assume a helper function GetPtrStorageScope is
    // available. If not, you might need to parse buffer->data->name_hint or
    // associated attributes.
    String scope = buffer.scope();
    return scope == "global";
  }

  // Check each index against the buffer shape dimensions
  void CheckBufferIndices(const Buffer &buffer, const Array<PrimExpr> &indices,
                          bool is_load) {
    // Ensure indices count matches buffer dimension
    if (indices.size() != buffer->shape.size()) {
      LOG(WARNING) << "Buffer access dimension mismatch: indices size ("
                   << indices.size() << ") vs. shape size ("
                   << buffer->shape.size() << ")";
      return;
    }

    for (size_t i = 0; i < indices.size(); i++) {
      PrimExpr index = indices[i];
      PrimExpr shape_dim = buffer->shape[i];

      bool is_index_constant = true;
      PostOrderVisit(index, [&](const ObjectRef &obj) {
        if (const VarNode *v = obj.as<VarNode>()) {
          is_index_constant = false;
        }
        if (const BufferLoadNode *v = obj.as<BufferLoadNode>()) {
          is_index_constant = false;
        }
      });
      if (is_index_constant) {
        // If index is a constant, we can skip the check
        continue;
      }

      // We want to check if index < shape_dim can be proven.
      // If analyzer->CanProve(index < shape_dim) returns false,
      // it means we cannot prove the access is within bounds.
      PrimExpr upper_bound_cond = index < shape_dim;
      if (!analyzer_->CanProve(upper_bound_cond,
                               arith::ProofStrength::kSymbolicBound)) {
        _conditions.push_back(upper_bound_cond);
      }
      // Check if index >= 0 can be proven.
      PrimExpr lower_bound_cond = index >= 0;
      if (!analyzer_->CanProve(lower_bound_cond,
                               arith::ProofStrength::kSymbolicBound)) {
        _conditions.push_back(lower_bound_cond);
      }
    }
  }

  Array<PrimExpr> GetConditions() { return _conditions; }

private:
  Array<PrimExpr> _conditions;
  arith::Analyzer *analyzer_;
  bool recursively_collect_conds_;
};

class SafeMemorysRewriter : public IRMutatorWithAnalyzer {
public:
  // Static method to substitute and transform the given PrimFunc
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    // Create an instance of the legalizer with the analyzer
    SafeMemorysRewriter substituter(&analyzer);
    // Get a mutable copy of the function node
    PrimFuncNode *fptr = f.CopyOnWrite();
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    // Apply the legalizer to the function body
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  // Constructor initializing the base class with the analyzer
  SafeMemorysRewriter(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}
  // Constructor initializing the base class with the analyzer

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));

    // For Load/Store, we only check the current node, not its children.
    // Since rewriter will recursively visit children.
    GlobalMemChecker checker(analyzer_, /*recursively_collect_conds=*/false);
    checker(load);
    Array<PrimExpr> conditions = checker.GetConditions();

    if (conditions.empty()) {
      return load;
    }

    // For loading, we can always use safe value if the access is out of
    // bounds
    PrimExpr value = load;
    for (auto cond : conditions) {
      ICHECK(cond.dtype() == DataType::Bool(1))
          << "condition is not a boolean: " << cond;
      value = if_then_else(cond, value, GetSafeValue(load->buffer));
    }
    return value;
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    // Check if the buffer is in global scope
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));

    GlobalMemChecker checker(analyzer_, /*recursively_collect_conds=*/false);
    checker(store);
    Array<PrimExpr> conditions = checker.GetConditions();

    // Skip boundary check if the store value is an IfThenElse
    if (const IfThenElseNode *if_node = store->value.as<IfThenElseNode>()) {
      if (!conditions.empty()) {
        LOG(WARNING)
            << "Skipping boundary check for store with IfThenElse value: "
            << store->value
            << "\nAs manual boundary check detected, potential out-of-bounds "
               "access may occur."
            << "\nAuto detect boundaries are " << conditions;
        return store;
      }
      return store;
    }

    if (conditions.empty()) {
      return store;
    }

    // If a store is out of bounds, we skip the corresponding stmt directly.
    Stmt store_with_conditions = store;
    for (auto cond : conditions) {
      store_with_conditions = IfThenElse(cond, store_with_conditions);
    }
    return store_with_conditions;
  }

  // Recursively check Load/Store in the call arguments.
  // For example
  // T.call_extern("handle", "atomicAddx2", T.address_of(C),
  // T.address_of(C_shared))

  // NOTE(chaofan): This is currently not the most rigorous solution.
  // The check here is primarily intended to handle extern functions like
  // atomicAdd, which may involve memory access. Due to their special nature,
  // the BufferLoad in their parameters might be used for boundary checks of the
  // current statement. The current solution adopts a simplified approach:
  // directly applying the boundary constraints of all parameters to the
  // statement. While not entirely precise, it addresses most common scenarios.
  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto evaluate = Downcast<Evaluate>(op);

    if (const CallNode *call_op = op->value.as<CallNode>()) {
      auto call = Downcast<Call>(op->value);
      if (call->op == builtin::call_extern()) {
        // For CallExtern, we recursively collect conditions from all children.
        // Since we cannot rewrite any BufferLoad in its children (Rewrite will
        // cause potential Nullptr exception).
        GlobalMemChecker checker(analyzer_, /*recursively_collect_conds=*/true);
        checker(call);
        Array<PrimExpr> conditions = checker.GetConditions();

        if (conditions.empty()) {
          return evaluate;
        }

        Stmt evaluate_with_conditions = evaluate;
        for (auto cond : conditions) {
          evaluate_with_conditions = IfThenElse(cond, evaluate_with_conditions);
        }
        return evaluate_with_conditions;
      }
    }

    return evaluate;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    if (op->annotations.count(attr::kSafeValueMap)) {
      auto map = op->annotations.Get(attr::kSafeValueMap)
                     ->as<Map<Var, PrimExpr>>()
                     .value();
      for (const auto &[var, safe_value] : map) {
        ICHECK(buffer_data_to_buffer_.count(var))
            << "buffer " << var << " is not found in the block "
            << buffer_data_to_buffer_;
        auto buffer = buffer_data_to_buffer_[var];
        annotated_safe_value_map_.Set(buffer, safe_value);
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  // Get the safe value of the buffer
  PrimExpr GetSafeValue(const Buffer &buffer) {
    if (annotated_safe_value_map_.count(buffer)) {
      return annotated_safe_value_map_[buffer];
    }
    return make_zero(buffer->dtype);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, PrimExpr> annotated_safe_value_map_;
};

// Create a pass that legalizes vectorized loops in the IRModule
tvm::transform::Pass LegalizeSafeMemoryAccess() {
  using namespace tir::transform;
  // Define the transformation function to be applied
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_safe_memory_legalize =
        ctx->GetConfig<Bool>(kDisableSafeMemoryLegalize, Bool(false)).value();
    if (disable_safe_memory_legalize) {
      return f;
    }
    return SafeMemorysRewriter::Substitute(std::move(f));
  };
  // Create and return a PrimFunc pass with the transformation function
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeSafeMemoryAccess", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSafeMemoryAccess",
                        LegalizeSafeMemoryAccess);
}

} // namespace tl
} // namespace tvm
