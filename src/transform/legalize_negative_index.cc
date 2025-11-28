/*!
 * \file legalize_negative_index.cc
 * \brief Legalize negative indices in buffer load/store expressions.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <variant>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRVisitorWithAnalyzer;

enum class IndexSignState { kNonNegative, kNegative, kUnknown };

using BufferAccessVariant =
    std::variant<const BufferLoadNode *, const BufferStoreNode *>;
using LoadStore2StateMap =
    std::unordered_map<BufferAccessVariant, std::vector<IndexSignState>>;

class NegativeIndexAnalyzer : public IRVisitorWithAnalyzer {
public:
  explicit NegativeIndexAnalyzer(LoadStore2StateMap *result)
      : result_(result) {}

private:
  std::vector<IndexSignState> ProcessIdx(const ffi::Array<PrimExpr> &indices,
                                         ffi::String buffer_name) {
    std::vector<IndexSignState> states;
    states.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
      PrimExpr simplified = analyzer_.Simplify(indices[i]);
      IndexSignState state = IndexSignState::kUnknown;

      // Handle vector patterns first to avoid querying lanes() on
      // scalable vectors (which is not allowed at compile-time).
      if (const auto *ramp = simplified.as<RampNode>()) {
        // For scalable vectors, we cannot rely on a constant lane count.
        // Use sufficient (but not necessary) conditions:
        // - If base >= 0 and stride >= 0, all lanes are non-negative.
        // - If base < 0 and stride <= 0, all lanes are negative.
        bool base_nonneg = analyzer_.CanProve(ramp->base >= 0);
        bool base_neg = analyzer_.CanProve(ramp->base < 0);
        bool stride_nonneg = analyzer_.CanProve(ramp->stride >= 0);
        bool stride_nonpos = analyzer_.CanProve(ramp->stride <= 0);

        if (base_nonneg && stride_nonneg) {
          state = IndexSignState::kNonNegative;
        } else if (base_neg && stride_nonpos) {
          state = IndexSignState::kNegative;
        } else {
          DLOG(WARNING)
              << "LegalizeNegativeIndex: cannot prove non-negative index "
              << simplified << " for buffer " << buffer_name << " (axis " << i
              << ", index " + indices[i]->Script() + ").";
        }
      } else if (const auto *broadcast = simplified.as<BroadcastNode>()) {
        auto v = analyzer_.Simplify(broadcast->value);
        if (analyzer_.CanProve(v >= 0))
          state = IndexSignState::kNonNegative;
        else if (analyzer_.CanProve(v < 0))
          state = IndexSignState::kNegative;
        else {
          // Try const bound if proof unavailable
          auto vb = analyzer_.const_int_bound(v);
          if (vb->min_value >= 0)
            state = IndexSignState::kNonNegative;
          else if (vb->max_value < 0)
            state = IndexSignState::kNegative;
          else
            DLOG(WARNING)
                << "LegalizeNegativeIndex: cannot prove non-negative index "
                << simplified << " for buffer " << buffer_name << " (axis " << i
                << ", index " + indices[i]->Script() + ").";
        }
      } else {
        // Assume scalar (or non-Ramp/Broadcast) index; avoid querying lanes().
        // Fall back to scalar reasoning. If this expression is actually a
        // vector-but-not-Ramp/Broadcast, treat as unknown to be safe.
        // Try to prove scalar first; if proof fails, leave as unknown.
        if (analyzer_.CanProve(simplified >= 0))
          state = IndexSignState::kNonNegative;
        else if (analyzer_.CanProve(simplified < 0))
          state = IndexSignState::kNegative;
        else
          DLOG(WARNING)
              << "LegalizeNegativeIndex: cannot prove non-negative index "
              << simplified << " for buffer " << buffer_name << " (axis " << i
              << ", index " + indices[i]->Script() + ").";
      }
      states.push_back(state);
    }

    return std::move(states);
  }

  bool NeedRecord(const std::vector<IndexSignState> &states) {
    return std::any_of(states.begin(), states.end(),
                       [](const IndexSignState &state) {
                         return state == IndexSignState::kUnknown ||
                                state == IndexSignState::kNegative;
                       });
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    std::vector<IndexSignState> states =
        ProcessIdx(op->indices, op->buffer->name);

    if (NeedRecord(states))
      (*result_)[op] = std::move(states);

    IRVisitorWithAnalyzer::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    std::vector<IndexSignState> states =
        ProcessIdx(op->indices, op->buffer->name);

    if (NeedRecord(states))
      (*result_)[op] = std::move(states);

    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

private:
  LoadStore2StateMap *result_;
};

class NegativeIndexRewriter : public arith::IRMutatorWithAnalyzer {
public:
  static PrimFunc Apply(PrimFunc func, const LoadStore2StateMap &states) {
    arith::Analyzer analyzer;
    NegativeIndexRewriter rewriter(&analyzer, states);
    PrimFuncNode *func_node = func.CopyOnWrite();
    func_node->body = rewriter.VisitStmt(func_node->body);
    return func;
  }

private:
  NegativeIndexRewriter(arith::Analyzer *analyzer,
                        const LoadStore2StateMap &states)
      : arith::IRMutatorWithAnalyzer(analyzer), states_(states) {}

  ffi::Array<PrimExpr> UpdateIdx(const ffi::Array<PrimExpr> &indices,
                                 const ffi::Array<PrimExpr> &buffer_shape,
                                 const std::vector<IndexSignState> &state_vec) {
    ICHECK_EQ(state_vec.size(), indices.size())
        << "State vector size mismatch for buffer load/store indices ("
        << indices << ")";
    ffi::Array<PrimExpr> new_indices = indices;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (state_vec[i] != IndexSignState::kNegative)
        continue;
      new_indices.Set(i, analyzer_->Simplify(buffer_shape[i] + indices[i]));
    }
    return new_indices;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load =
        Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));

    auto it = states_.find(op);
    if (it == states_.end())
      return load;

    auto indices = UpdateIdx(load->indices, load->buffer->shape, it->second);
    return BufferLoad(load->buffer, indices, load->predicate);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store =
        Downcast<BufferStore>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));

    auto it = states_.find(op);
    if (it == states_.end())
      return store;

    auto indices = UpdateIdx(store->indices, store->buffer->shape, it->second);
    return BufferStore(store->buffer, store->value, indices, store->predicate);
  }

private:
  const LoadStore2StateMap &states_;
};

PrimFunc LegalizeNegativeIndex(PrimFunc func) {
  if (!func->body.defined()) {
    return func;
  }

  LoadStore2StateMap states;
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
