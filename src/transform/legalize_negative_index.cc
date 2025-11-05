/*!
 * \file legalize_negative_index.cc
 * \brief Legalize negative indices in buffer load expressions.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRVisitorWithAnalyzer;

enum class IndexSignState { kNonNegative, kNegative, kUnknown };

class NegativeIndexAnalyzer : public IRVisitorWithAnalyzer {
public:
  explicit NegativeIndexAnalyzer(
      std::unordered_map<const BufferLoadNode *, std::vector<IndexSignState>>
          *result)
      : result_(result) {}

  void VisitExpr_(const BufferLoadNode *op) final {
    auto load = tvm::ffi::GetRef<BufferLoad>(op);
    std::vector<IndexSignState> states;
    states.reserve(op->indices.size());
    bool needs_record = false;

    for (size_t i = 0; i < op->indices.size(); ++i) {
      PrimExpr simplified = analyzer_.Simplify(op->indices[i]);
      if (analyzer_.CanProve(simplified >= 0)) {
        states.push_back(IndexSignState::kNonNegative);
        continue;
      }

      if (analyzer_.CanProve(simplified < 0)) {
        states.push_back(IndexSignState::kNegative);
        needs_record = true;
        continue;
      }

      states.push_back(IndexSignState::kUnknown);
      needs_record = true;
      LOG(WARNING) << "LegalizeNegativeIndex: cannot prove non-negative index "
                   << simplified << " for buffer " << load->buffer->name
                   << " (axis " << i << ").";
    }

    if (needs_record) {
      (*result_)[op] = std::move(states);
    }

    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

private:
  std::unordered_map<const BufferLoadNode *, std::vector<IndexSignState>>
      *result_;
};

class NegativeIndexRewriter : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc
  Apply(PrimFunc func,
        const std::unordered_map<const BufferLoadNode *,
                                 std::vector<IndexSignState>> &states) {
    arith::Analyzer analyzer;
    NegativeIndexRewriter rewriter(&analyzer, states);
    if (!func->body.defined()) {
      return func;
    }
    PrimFuncNode *func_node = func.CopyOnWrite();
    func_node->body = rewriter.VisitStmt(func_node->body);
    return func;
  }

private:
  NegativeIndexRewriter(
      arith::Analyzer *analyzer,
      const std::unordered_map<const BufferLoadNode *,
                               std::vector<IndexSignState>> &states)
      : arith::IRMutatorWithAnalyzer(analyzer), states_(states) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load =
        Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));

    auto it = states_.find(op);
    if (it == states_.end()) {
      return load;
    }

    auto indices = load->indices;
    bool changed = false;

    const auto &state_vector = it->second;
    ICHECK_EQ(state_vector.size(), indices.size())
        << "State vector size mismatch for buffer load " << load->buffer->name;

    for (size_t i = 0; i < indices.size(); ++i) {
      if (state_vector[i] != IndexSignState::kNegative) {
        continue;
      }
      PrimExpr extent = load->buffer->shape[i];
      indices.Set(i, analyzer_->Simplify(extent + indices[i]));
      changed = true;
    }

    if (!changed) {
      return load;
    }

    return BufferLoad(load->buffer, indices);
  }

  const std::unordered_map<const BufferLoadNode *, std::vector<IndexSignState>>
      &states_;
};

PrimFunc LegalizeNegativeIndex(PrimFunc func) {
  if (!func->body.defined()) {
    return func;
  }

  std::unordered_map<const BufferLoadNode *, std::vector<IndexSignState>>
      states;
  NegativeIndexAnalyzer analyzer(&states);
  analyzer(func->body);
  if (states.empty()) {
    return func;
  }

  return NegativeIndexRewriter::Apply(std::move(func), states);
}

tvm::transform::Pass LegalizeNegativeIndexPass() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, PassContext) {
    return LegalizeNegativeIndex(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeNegativeIndex", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeNegativeIndex",
                        LegalizeNegativeIndexPass);
}

} // namespace tl
} // namespace tvm
