/*!
 * \file legalize_negative_index.cc
 * \brief Legalize negative indices in buffer load expressions.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/op.h>
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

      // Handle scalar indices with the standard analyzer
      if (simplified.dtype().lanes() == 1) {
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
        LOG(WARNING)
            << "LegalizeNegativeIndex: cannot prove non-negative index "
            << simplified << " for buffer " << load->buffer->name << " (axis "
            << i << ").";
        continue;
      }

      // Vector indices: try to reason about non-negativity/negativity
      // Common patterns are Ramp(base, stride, lanes) and Broadcast(value,
      // lanes).
      IndexSignState vec_state = IndexSignState::kUnknown;
      if (const auto *ramp = simplified.as<RampNode>()) {
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

        if (lower >= 0) {
          vec_state = IndexSignState::kNonNegative;
        } else if (upper < 0) {
          vec_state = IndexSignState::kNegative;
        } else {
          vec_state = IndexSignState::kUnknown;
        }
      } else if (const auto *bc = simplified.as<BroadcastNode>()) {
        auto v = analyzer_.Simplify(bc->value);
        if (analyzer_.CanProve(v >= 0)) {
          vec_state = IndexSignState::kNonNegative;
        } else if (analyzer_.CanProve(v < 0)) {
          vec_state = IndexSignState::kNegative;
        } else {
          // Try const bound if proof unavailable
          auto vb = analyzer_.const_int_bound(v);
          if (vb->min_value >= 0) {
            vec_state = IndexSignState::kNonNegative;
          } else if (vb->max_value < 0) {
            vec_state = IndexSignState::kNegative;
          } else {
            vec_state = IndexSignState::kUnknown;
          }
        }
      }

      if (vec_state == IndexSignState::kNonNegative) {
        states.push_back(IndexSignState::kNonNegative);
        continue;
      }
      if (vec_state == IndexSignState::kNegative) {
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
