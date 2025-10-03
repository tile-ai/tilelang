/*!
 * \file atomicadd_vectorize.cc
 * \brief A tool to automatically vectorize atomic add
 */

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../transform/loop_partition.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include <numeric>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

struct AtomicAddVectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

class BufferIndiceSimplify : public StmtExprMutator {
public:
  BufferIndiceSimplify(arith::Analyzer *analyzer) : analyzer_(analyzer) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *node) final {
    auto visited = StmtExprMutator::VisitExpr_(node);
    auto n = Downcast<BufferLoad>(visited);
    auto nptr = n.CopyOnWrite();
    nptr->indices = nptr->indices.Map(
        [&](const auto &e) { return analyzer_->Simplify(e); });
    return n;
  }
  Stmt VisitStmt_(const BufferStoreNode *node) final {
    auto visited = StmtExprMutator::VisitStmt_(node);
    auto n = Downcast<BufferStore>(visited);
    auto nptr = n.CopyOnWrite();
    nptr->indices = nptr->indices.Map(
        [&](const auto &e) { return analyzer_->Simplify(e); });
    return n;
  }
  arith::Analyzer *analyzer_;
};

class AtomicAddVectorizePlanner : public arith::IRVisitorWithAnalyzer {
public:
  AtomicAddVectorizePlanner() = default;
  int max_vector_size = 1;
  AtomicAddVectorizePlanResult Plan(const For &node, Var thread_var,
                                    Range thread_bounds, int vectorize_hint) {
    this->max_vector_size = vectorize_hint;
    this->thread_var = std::move(thread_var);
    this->thread_bounds = std::move(thread_bounds);
    this->operator()(node);
    return {vector_size_, dynamic_, condition_};
  }

private:
  void VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    iter_map_.Set(node->loop_var, Range(node->min, node->extent));

    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::call_extern() && node->args.size() >= 2) {
      if (const auto *func_name = node->args[0].as<StringImmNode>()) {
        if (func_name->value == "AtomicAdd") {
          const BufferLoadNode *buffer_load_dst =
              node->args[1].as<BufferLoadNode>();
          const BufferLoadNode *buffer_load_src =
              node->args[2].as<BufferLoadNode>();
          if (buffer_load_src && buffer_load_src->buffer.defined() &&
              buffer_load_dst && buffer_load_dst->buffer.defined()) {

            Buffer dst_buffer = buffer_load_dst->buffer;
            Array<PrimExpr> indices_dst = buffer_load_dst->indices;
            UpdateVectorSize(indices_dst, dst_buffer);
            Buffer src_buffer = buffer_load_src->buffer;
            Array<PrimExpr> indices_src = buffer_load_src->indices;
            UpdateVectorSize(indices_src, src_buffer);
          }
        }
      }
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void UpdateVectorSize(const Array<PrimExpr> &indices, const Buffer &buffer) {
    if (!inner_for_)
      return;
    auto extent_ptr = inner_for_->extent.as<IntImmNode>();
    if (!extent_ptr)
      return;

    const DataType &access_type = buffer->dtype;
    // i // 2, i % 8 can also be vectorized as factor 16
    // so we should disable this GCD optimization

    max_vector_size = arith::ZeroAwareGCD(max_vector_size, extent_ptr->value);

    auto last_dim = buffer->shape.back();
    auto mod_set = analyzer_.modular_set(last_dim);
    // when dynamic shape like [m, k]: coeff=1, base=0, GCD will block
    // conditionally tail vectorize
    if (buffer->shape.back().as<IntImmNode>()) {

      max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->coeff);

      auto gcd_base = arith::ZeroAwareGCD(max_vector_size, mod_set->base);
      // If gcd_base is equal to the last dimension,
      // we should analyze the second-to-last dimension
      // in relation to the last dimension.
      if (gcd_base < Downcast<IntImm>(last_dim)->value) {
        max_vector_size = gcd_base;
      }

      vector_size_ = arith::ZeroAwareGCD(max_vector_size, vector_size_);

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
        elem_offset = elem_offset + indices[i] * stride;
        stride = stride * buffer->shape[i];
      }
      PrimExpr thread_extent = thread_bounds->extent;
      while (!IndiceCanVectorize(elem_offset, thread_var, thread_extent,
                                 vector_size_, &analyzer_)) {
        vector_size_ /= 2;
      }
    } else if (vector_size_ <= 4) {
      // dynamic shape load: get the vectorization condition
      dynamic_ = true;
      PrimExpr offset = buffer.OffsetOf(indices).back();
      condition_ = (truncmod(offset, vector_size_) == 0);
    }
  }

  const ForNode *inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 4;
  Var thread_var;
  Range thread_bounds;
  bool dynamic_ = false;
  PrimExpr condition_;
};

class AtomicAddVectorizeRewriter : public StmtExprMutator {
public:
  AtomicAddVectorizeRewriter(const AtomicAddVectorizePlanResult &plan,
                             Var thread_var, const Range &thread_bounds)
      : vector_size_(plan.vector_size), condition_(plan.condition),
        dynamic_(plan.dynamic), tx_var_(std::move(thread_var)) {
    const int64_t *tx_ext = as_const_int(thread_bounds->extent);
    ICHECK(tx_ext)
        << "thread_bounds->extent must be a constant for vectorization.";
    extent_tx_ = static_cast<int>(*tx_ext);
  }

  For run(For for_node, const Fragment &loop_layout,
          arith::Analyzer *analyzer) {
    int old_loop_depth = loop_layout->InputDim();
    int new_loop_depth = loop_layout->OutputDim();

    Array<Var> vars;
    for (int i = 0; i < new_loop_depth; i++) {
      Var var = Var(std::string{char('i' + i)});
      vars.push_back(var);
    }
    vars.push_back(tx_var_);
    Map<Var, PrimExpr> vmap;
    Stmt body = std::move(for_node);
    auto inv_loop = loop_layout->Inverse();
    auto indices = inv_loop->Forward(Array<PrimExpr>(vars.begin(), vars.end()));
    // the innerest iter_var need expand because of vectorize

    const ForNode *loop = body.as<ForNode>();
    ICHECK(loop != nullptr);
    vmap.Set(loop->loop_var, indices[0] * vector_size_);
    body = loop->body;
    for (int i = 1; i < old_loop_depth; i++) {
      const ForNode *loop = body.as<ForNode>();
      ICHECK(loop != nullptr);
      vmap.Set(loop->loop_var, indices[i]);
      body = loop->body;
    }
    body = Substitute(body, vmap);

    // innerest iter_var extent need to be shorter because of vectorize

    body = For(vars[new_loop_depth - 1],
               make_zero(vars[new_loop_depth - 1]->dtype),
               div(inv_loop->InputShape()[new_loop_depth - 1], vector_size_),
               ForKind::kSerial, body);
    analyzer->Bind(vars[new_loop_depth - 1],
                   Range(0, div(inv_loop->InputShape()[new_loop_depth - 1],
                                vector_size_)));

    for (int i = new_loop_depth - 2; i >= 0; i--) {
      body = For(vars[i], make_zero(vars[i]->dtype), inv_loop->InputShape()[i],
                 ForKind::kSerial, body);
      analyzer->Bind(vars[i], Range(0, inv_loop->InputShape()[i]));
    }

    body = BufferIndiceSimplify(analyzer)(body);

    auto node = LoopPragmaUnroll(Downcast<For>(body));
    if (loop_layout->ThreadRange().defined()) {
      auto range = loop_layout->ThreadRange();
      auto thread_var_with_offset = tx_var_ - range->min;
      node.CopyOnWrite()->body =
          Substitute(node->body, {{tx_var_, thread_var_with_offset}});
    }
    auto new_stmt = this->VisitStmt(node);
    return Downcast<For>(new_stmt);
  }

private:
  PrimExpr VisitExpr_(const CallNode *node) final {
    if (dynamic_) {
      return StmtExprMutator::VisitExpr_(node);
    }
    if (vector_size_ == 2 || vector_size_ == 4) {
      if (node->op == builtin::call_extern() && node->args.size() >= 2) {
        if (const auto *func_name = node->args[0].as<StringImmNode>()) {
          if (func_name->value == "AtomicAdd") {
            const BufferLoadNode *temp_dst_node =
                node->args[1].as<BufferLoadNode>();
            const BufferLoadNode *temp_value_node =
                node->args[2].as<BufferLoadNode>();
            if (!temp_dst_node || !temp_value_node) {
              return StmtExprMutator::VisitExpr_(node);
            }
            const BufferLoad dst_node =
                Downcast<BufferLoad>(node->args[1].as<BufferLoadNode>());
            const BufferLoad value_node =
                Downcast<BufferLoad>(node->args[2].as<BufferLoadNode>());

            Call address_of_dst =
                Call(DataType::Handle(), builtin::address_of(), {dst_node});
            Call address_of_value =
                Call(DataType::Handle(), builtin::address_of(), {value_node});
            Array<PrimExpr> new_args;
            if (vector_size_ == 2) {
              new_args.push_back(StringImm("AtomicAddx2"));
            } else {
              new_args.push_back(StringImm("AtomicAddx4"));
            }
            new_args.push_back(address_of_dst);
            new_args.push_back(address_of_value);

            Call new_call =
                tvm::tir::Call(node->dtype, builtin::call_extern(), new_args);

            return new_call;
          }
        }
      }
    }
    return StmtExprMutator::VisitExpr_(node);
  }

  const ForNode *inner_for_;
  const int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
  const Var tx_var_;
  int extent_tx_;
};

static int GetVectorizeSizeMax(int compute_capability, DataType dtype) {

  if (dtype == DataType::Float(16)) {
    return 2;
  }
  if (dtype == DataType::BFloat(16)) {
    if (compute_capability > 75) {
      return 2;
    } else {
      return 1;
    }
  }
  if (dtype == DataType::Float(32)) {
    if (compute_capability >= 90) {
      return 4;
    } else {
      return 1;
    }
  }
  return 1;
}

For VectorizeAtomicAdd(const For &for_node, const Var &thread_var,
                       const Range &thread_bounds, int compute_capability,
                       arith::Analyzer *analyzer, const Fragment &loop_layout) {

  int vectorize_size_max = 1;

  PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
    if (const auto *call = obj.as<CallNode>()) {
      if (call->op == builtin::call_extern() && call->args.size() >= 2) {
        const auto *func_name = call->args[0].as<StringImmNode>();
        if (func_name->value == "AtomicAdd") {
          DataType dtype = call->args[1].as<BufferLoadNode>()->dtype;
          vectorize_size_max = GetVectorizeSizeMax(compute_capability, dtype);
        }
      }
    }
  });

  if (vectorize_size_max != 1) {
    int vectorize_hint = vectorize_size_max;
    AtomicAddVectorizePlanResult res = {1, false, 0};
    AtomicAddVectorizePlanner planner;
    For simplified_for_node =
        PartitionLoop(for_node, thread_var, analyzer, loop_layout);
    res = planner.Plan(simplified_for_node, thread_var, thread_bounds,
                       vectorize_hint);
    vectorize_hint = res.vector_size;

    if (vectorize_hint == 1)
      return for_node;
    auto rewriter = AtomicAddVectorizeRewriter(res, thread_var, thread_bounds);
    return rewriter.run(for_node, loop_layout, analyzer);
  } else {
    return for_node;
  }
}

} // namespace tl
} // namespace tvm