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

      // Handle scalar indices with the standard analyzer
      if (simplified.dtype().lanes() == 1) {
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
      // Vector indices: try to reason about non-negativity/negativity
      // Common patterns are Ramp(base, stride, lanes) and Broadcast(value,
      // lanes).
      else if (const auto *ramp = simplified.as<RampNode>()) {
        // Compute a safe lower/upper bound for the vector lanes
        // lower_bound = base_min + min(0, stride_min) * (lanes - 1)
        // upper_bound = base_max + max(0, stride_max) * (lanes - 1)
        auto base_bound = analyzer_.const_int_bound(ramp->base);
        auto stride_bound = analyzer_.const_int_bound(ramp->stride);
        int lanes = *as_const_int(ramp->lanes);

        int64_t base_min = base_bound->min_value;
        int64_t base_max = base_bound->max_value;
        int64_t s_min = stride_bound->min_value;
        int64_t s_max = stride_bound->max_value;

        // Guard against overflow is not strictly necessary here because
        // bounds may be +/-inf represented by sentinel values.
        int64_t lower = base_min;
        if (s_min < 0)
          lower += s_min * (lanes - 1);
        int64_t upper = base_max;
        if (s_max > 0)
          upper += s_max * (lanes - 1);

        if (lower >= 0)
          state = IndexSignState::kNonNegative;
        else if (upper < 0)
          state = IndexSignState::kNegative;
        else
          DLOG(WARNING)
              << "LegalizeNegativeIndex: cannot prove non-negative index "
              << simplified << " for buffer " << buffer_name << " (axis " << i
              << ", index " + indices[i]->Script() + ").";
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
