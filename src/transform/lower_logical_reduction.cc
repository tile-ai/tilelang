/*!
 * \file lower_logical_reduction.cc
 * \brief Materialize logical reduction expressions before layout inference.
 *
 * Producers stay at the dynamic evaluation point, including lazy branches
 * and while conditions. Uniform shared/global reductions use reducer v2;
 * other cases use thread-local loops. Both retain logical BufferLoads.
 */
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/reducer.h"
#include "../op/utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;

TVM_REGISTER_PASS_CONFIG_OPTION("tl.disable_parallel_logical_reduction", Bool);

namespace {
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

const Op &AnyOf() { return Op::Get("tl.any_of"); }
const Op &AllOf() { return Op::Get("tl.all_of"); }

bool ContainsReduction(const PrimExpr &expr) {
  bool found = false;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      found |= call->op.same_as(AnyOf()) || call->op.same_as(AllOf());
    }
  });
  return found;
}

PrimExpr FullRegion(const Buffer &buffer, int mask) {
  return Call(DataType::Handle(), region(),
              {BufferLoad(buffer, {0}), IntImm(DataType::Int(32), mask),
               buffer->shape[0]});
}

class LogicalReductionLowerer : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func, bool enable_parallel) {
    LogicalReductionLowerer lowerer;
    lowerer.enable_parallel_ = enable_parallel;
    for (const Var &param : func->params) {
      lowerer.uniform_vars_.insert(param);
    }
    func.CopyOnWrite()->body = lowerer.VisitStmt(func->body);
    return func;
  }

private:
  struct BlockState {
    Array<Buffer> allocations;
    Map<Var, Map<String, Any>> reducers;
  };

  bool IsUniform_(const PrimExpr &expr) const {
    bool uniform = true;
    PostOrderVisit(expr, [&](const ObjectRef &node) {
      if (const auto *var = node.as<VarNode>()) {
        uniform &= uniform_vars_.count(GetRef<Var>(var)) != 0;
      } else if (const auto *load = node.as<BufferLoadNode>()) {
        uniform &= IsSharedBuffer(load->buffer) ||
                   IsGlobalBuffer(load->buffer) ||
                   uniform_buffers_.count(load->buffer);
      } else if (const auto *call = node.as<CallNode>()) {
        if (!call->op.same_as(AnyOf()) && !call->op.same_as(AllOf()) &&
            SideEffect(GetRef<Call>(call)) > CallEffectKind::kReadState) {
          uniform = false;
        }
      }
    });
    return uniform;
  }

  Buffer Allocate_(DataType dtype, const String &scope) {
    ICHECK(!blocks_.empty())
        << "logical reductions must appear inside a kernel block";
    Buffer buffer = decl_buffer(
        {1}, dtype, "logical_reduce_" + std::to_string(next_id_++), scope);
    blocks_.back().allocations.push_back(buffer);
    return buffer;
  }

  std::pair<PrimExpr, Array<Stmt>> Capture_(const PrimExpr &expr,
                                            bool uniform_context) {
    Array<Stmt> saved = std::move(prefix_);
    prefix_ = {};
    bool old_context = uniform_context_;
    uniform_context_ &= uniform_context;
    PrimExpr value = VisitExpr(expr);
    Array<Stmt> emitted = std::move(prefix_);
    prefix_ = std::move(saved);
    uniform_context_ = old_context;
    return {value, emitted};
  }

  Stmt VisitStmt(const Stmt &stmt) final {
    Array<Stmt> saved = std::move(prefix_);
    prefix_ = {};
    Stmt result = StmtExprMutator::VisitStmt(stmt);
    prefix_.push_back(result);
    result = SeqStmt::Flatten(prefix_);
    prefix_ = std::move(saved);
    return result;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    blocks_.push_back({});
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    BlockState state = std::move(blocks_.back());
    blocks_.pop_back();
    if (state.allocations.empty())
      return block;
    auto *node = block.CopyOnWrite();
    for (const Buffer &buffer : state.allocations) {
      node->alloc_buffers.push_back(buffer);
    }
    Map<Var, Map<String, Any>> info;
    if (auto existing = node->annotations.Get(attr::kReducerInfoV2)) {
      info = existing.value().cast<Map<Var, Map<String, Any>>>();
    }
    for (const auto &[var, value] : state.reducers)
      info.Set(var, value);
    if (!info.empty())
      node->annotations.Set(attr::kReducerInfoV2, info);
    return block;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    bool old_threads = has_threads_;
    bool old_context = uniform_context_;
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      std::string tag = iv->thread_tag;
      if (tag == "threadIdx.x") {
        const auto *extent = op->value.as<IntImmNode>();
        has_threads_ = extent && extent->value > 0 &&
                       (extent->value & (extent->value - 1)) == 0;
      } else if (tag == "threadIdx.y" || tag == "threadIdx.z") {
        uniform_context_ &= is_one(op->value);
      } else if (tag.find("blockIdx.") == 0) {
        uniform_vars_.insert(iv->var);
      }
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    has_threads_ = old_threads;
    uniform_context_ = old_context;
    return result;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (IsUniform_(value))
      uniform_vars_.insert(op->var);
    Bind result = GetRef<Bind>(op);
    result.CopyOnWrite()->value = value;
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = GetRef<For>(op);
    PrimExpr min = VisitExpr(op->min);
    PrimExpr extent = VisitExpr(op->extent);
    bool old_context = uniform_context_;
    bool parallel =
        op->kind == ForKind::kParallel || op->kind == ForKind::kThreadBinding;
    uniform_context_ &= !parallel && IsUniform_(min) && IsUniform_(extent);
    if (uniform_context_)
      uniform_vars_.insert(op->loop_var);
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(min, extent));
    // A non-uniform break/continue can affect later iterations. Until loop
    // exit uniformity is proven, conservatively avoid collectives in the loop.
    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (const auto *branch = node.as<IfThenElseNode>()) {
        uniform_context_ &= IsUniform_(branch->condition);
      }
    });
    Stmt body = VisitStmt(op->body);
    uniform_context_ = old_context;
    auto *node = loop.CopyOnWrite();
    node->min = min;
    node->extent = extent;
    node->body = body;
    return loop;
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    bool old_context = uniform_context_;
    uniform_context_ &= IsUniform_(condition);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined())
      else_case = VisitStmt(op->else_case.value());
    uniform_context_ = old_context;
    return IfThenElse(condition, then_case, else_case, op->span);
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    // Divergent exits can change collective execution counts. Conservatively
    // keep loops containing non-uniform branches thread-local.
    bool old_context = uniform_context_;
    PostOrderVisit(op->body, [&](const ObjectRef &node) {
      if (const auto *branch = node.as<IfThenElseNode>()) {
        uniform_context_ &= IsUniform_(branch->condition);
      }
    });
    uniform_context_ &= IsUniform_(op->condition);
    auto [condition, condition_stmts] = Capture_(op->condition, true);
    Stmt body = VisitStmt(op->body);
    uniform_context_ = old_context;
    if (condition_stmts.empty())
      return While(condition, body, op->span);
    condition_stmts.push_back(IfThenElse(
        Not(condition),
        Evaluate(Call(DataType::Void(), builtin::break_loop(), {}))));
    condition_stmts.push_back(body);
    return While(Bool(true), SeqStmt::Flatten(condition_stmts), op->span);
  }

  PrimExpr RewriteConditional_(const PrimExpr &condition,
                               const PrimExpr &then_value,
                               const PrimExpr &else_value) {
    PrimExpr cond = VisitExpr(condition);
    auto [lhs, lhs_stmts] = Capture_(then_value, IsUniform_(cond));
    auto [rhs, rhs_stmts] = Capture_(else_value, IsUniform_(cond));
    if (lhs_stmts.empty() && rhs_stmts.empty())
      return if_then_else(cond, lhs, rhs);
    Buffer result = Allocate_(lhs.dtype(), "local");
    lhs_stmts.push_back(BufferStore(result, lhs, {0}));
    rhs_stmts.push_back(BufferStore(result, rhs, {0}));
    prefix_.push_back(IfThenElse(cond, SeqStmt::Flatten(lhs_stmts),
                                 SeqStmt::Flatten(rhs_stmts)));
    if (IsUniform_(cond) && IsUniform_(lhs) && IsUniform_(rhs)) {
      uniform_buffers_.insert(result);
    }
    return BufferLoad(result, {0});
  }

  PrimExpr VisitExpr_(const AndNode *op) final {
    if (ContainsReduction(op->b))
      return RewriteConditional_(op->a, op->b, Bool(false));
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const OrNode *op) final {
    if (ContainsReduction(op->b))
      return RewriteConditional_(op->a, Bool(true), op->b);
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const LetNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    if (IsUniform_(value))
      uniform_vars_.insert(op->var);
    auto [body, stmts] = Capture_(op->body, true);
    if (stmts.empty())
      return Let(op->var, value, body, op->span);
    prefix_.push_back(Bind(op->var, value));
    for (const Stmt &stmt : stmts)
      prefix_.push_back(stmt);
    return body;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else()) &&
        (ContainsReduction(op->args[1]) || ContainsReduction(op->args[2]))) {
      return RewriteConditional_(op->args[0], op->args[1], op->args[2]);
    }
    bool is_any = op->op.same_as(AnyOf());
    if (!is_any && !op->op.same_as(AllOf()))
      return StmtExprMutator::VisitExpr_(op);
    ICHECK_EQ(op->args.size(), 2U);
    Call ptr = Downcast<Call>(op->args[0]);
    ICHECK(ptr->op.same_as(access_ptr()))
        << "logical reductions require T.access_ptr with a logical BufferLoad";
    BufferLoad base = Downcast<BufferLoad>(ptr->args[0]);
    Array<PrimExpr> indices;
    bool uniform = true;
    for (const PrimExpr &index : base->indices) {
      indices.push_back(VisitExpr(index));
      uniform &= IsUniform_(indices.back());
    }
    PrimExpr extent = analyzer_.Simplify(VisitExpr(op->args[1]));
    uniform &= IsUniform_(extent);
    const auto *constant_extent = extent.as<IntImmNode>();
    if (constant_extent) {
      ICHECK_GE(constant_extent->value, 0);
      if (constant_extent->value == 0)
        return Bool(!is_any);
    }
    Buffer source = base->buffer;
    bool fragment = IsFragmentBuffer(source);
    bool collective =
        enable_parallel_ && has_threads_ && uniform_context_ && uniform &&
        constant_extent && (constant_extent->value >= 32 || fragment) &&
        (IsSharedBuffer(source) || IsGlobalBuffer(source) || fragment);
    ICHECK(!fragment || collective)
        << "logical reductions over distributed fragments require a uniform "
           "collective context; copy to a thread-local buffer for local scans";

    Var offset("logical_reduce_i", extent.dtype());
    analyzer_.Bind(offset,
                   Range::FromMinExtent(make_zero(offset.dtype()), extent));
    PrimExpr flat = make_zero(extent.dtype());
    for (size_t i = 0; i < indices.size(); ++i) {
      flat = flat * source->shape[i] + indices[i];
    }
    flat = flat + offset;
    Array<PrimExpr> load_indices;
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
      load_indices.insert(
          load_indices.begin(),
          analyzer_.Simplify(i == 0 ? flat : floormod(flat, source->shape[i])));
      flat = floordiv(flat, source->shape[i]);
    }
    PrimExpr value = BufferLoad(source, load_indices);
    PrimExpr predicate;
    if (value.dtype().is_scalar()) {
      predicate = value != make_zero(value.dtype());
    } else {
      ICHECK(value.dtype().is_fixed_length_vector());
      predicate = Bool(!is_any);
      for (int lane = 0; lane < value.dtype().lanes(); ++lane) {
        PrimExpr element = Shuffle::ExtractElement(value, lane);
        PrimExpr nonzero = element != make_zero(element.dtype());
        if (is_any) {
          predicate = Or(predicate, nonzero);
        } else {
          predicate = And(predicate, nonzero);
        }
      }
    }

    if (collective) {
      Buffer acc = Allocate_(DataType::Int(32), "local.reducer");
      Buffer result = Allocate_(DataType::Int(32), "local.fragment");
      blocks_.back().reducers.Set(
          acc->data, {{"op", String(is_any ? "bitor" : "bitand")}});
      prefix_.push_back(Evaluate(Call(DataType::Handle(), ReducerInitOp::Get(),
                                      {FullRegion(acc, 2)})));
      Stmt update = Evaluate(
          Call(DataType::Handle(), reducer_update(),
               {BufferLoad(acc, {0}), Cast(DataType::Int(32), predicate)}));
      prefix_.push_back(For(offset, make_zero(offset.dtype()), extent,
                            ForKind::kParallel, update));
      prefix_.push_back(
          Evaluate(Call(DataType::Handle(), FinalizeReducerV2Op::Get(),
                        {FullRegion(acc, 3), FullRegion(result, 2)})));
      uniform_buffers_.insert(result);
      return BufferLoad(result, {0}) != 0;
    }

    Buffer result = Allocate_(DataType::Bool(), "local");
    prefix_.push_back(BufferStore(result, Bool(!is_any), {0}));
    Stmt found = SeqStmt::Flatten(
        BufferStore(result, Bool(is_any), {0}),
        Evaluate(Call(DataType::Void(), builtin::break_loop(), {})));
    prefix_.push_back(
        For(offset, make_zero(offset.dtype()), extent, ForKind::kSerial,
            IfThenElse(is_any ? predicate : Not(predicate), found)));
    if (uniform && (IsSharedBuffer(source) || IsGlobalBuffer(source))) {
      uniform_buffers_.insert(result);
    }
    return BufferLoad(result, {0});
  }

  arith::Analyzer analyzer_;
  VarSet uniform_vars_;
  BufferSet uniform_buffers_;
  std::vector<BlockState> blocks_;
  Array<Stmt> prefix_;
  int next_id_{0};
  bool enable_parallel_{true};
  bool uniform_context_{true};
  bool has_threads_{false};
};
} // namespace

tvm::transform::Pass LowerLogicalReduction() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      tvm::transform::PassContext ctx) {
    bool disabled = ctx->GetConfig<Bool>(
                           "tl.disable_parallel_logical_reduction", Bool(false))
                        .value();
    return LogicalReductionLowerer::Rewrite(std::move(func), !disabled);
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0,
                                             "tl.LowerLogicalReduction", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerLogicalReduction",
                                   LowerLogicalReduction);
}
} // namespace tvm::tl
