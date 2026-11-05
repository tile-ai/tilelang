/*!
 * \file lower_cpu_atomics.cc
 * \brief Lower CPU atomics to ordered serial native tensor accesses.
 *
 * MarkCPUAtomics runs before tile-op lowering and keeps the CPU grid serial.
 * This pass preserves call-argument evaluation, return_prev, conditional
 * expressions and repeated while conditions while replacing scalar atomics
 * with native Bind/TensorLoad/TensorStore. Memory-order flags have no further
 * effect on this serial execution route.
 */

#include "op/builtin.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <utility>

namespace tvm::tl {
namespace {
using ffi::Array;
using ffi::Optional;
using ffi::UnchangedOr;
using namespace prim;

enum class AtomicKind { kAdd, kMax, kMin, kOr, kStore, kAddVector };
struct AtomicOp {
  AtomicKind kind;
  int lanes;
  bool returns_previous;
};

std::optional<AtomicOp> MatchAtomic(const Expr &op) {
  if (op.same_as(atomic_add_elem_op()))
    return AtomicOp{AtomicKind::kAdd, 1, false};
  if (op.same_as(atomic_add_ret_elem_op()))
    return AtomicOp{AtomicKind::kAdd, 1, true};
  if (op.same_as(atomic_max_elem_op()))
    return AtomicOp{AtomicKind::kMax, 1, false};
  if (op.same_as(atomic_max_ret_elem_op()))
    return AtomicOp{AtomicKind::kMax, 1, true};
  if (op.same_as(atomic_min_elem_op()))
    return AtomicOp{AtomicKind::kMin, 1, false};
  if (op.same_as(atomic_min_ret_elem_op()))
    return AtomicOp{AtomicKind::kMin, 1, true};
  if (op.same_as(atomic_or_elem_op()))
    return AtomicOp{AtomicKind::kOr, 1, false};
  if (op.same_as(atomic_store_elem_op()))
    return AtomicOp{AtomicKind::kStore, 1, false};
  if (op.same_as(atomic_addx2_elem_op()))
    return AtomicOp{AtomicKind::kAddVector, 2, false};
  if (op.same_as(atomic_addx4_elem_op()))
    return AtomicOp{AtomicKind::kAddVector, 4, false};
  if (op.same_as(atomic_addx2_ret_elem_op()))
    return AtomicOp{AtomicKind::kAddVector, 2, true};
  if (op.same_as(atomic_addx4_ret_elem_op()))
    return AtomicOp{AtomicKind::kAddVector, 4, true};
  return std::nullopt;
}

PrimExpr Combine(AtomicKind kind, const PrimExpr &previous,
                 const PrimExpr &value) {
  switch (kind) {
  case AtomicKind::kAdd:
  case AtomicKind::kAddVector:
    return previous + value;
  case AtomicKind::kMax:
    return max(previous, value);
  case AtomicKind::kMin:
    return min(previous, value);
  case AtomicKind::kOr:
    return previous | value;
  case AtomicKind::kStore:
    return value;
  }
  TVM_FFI_THROW(InternalError) << "Unreachable CPU atomic kind";
}

class AtomicFinder : public tirx::StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  bool found = false;
  Optional<ffi::VisitInterrupt> Visit_(const CallObj *call) final {
    if (call->op.same_as(atomic_load_elem_op()) || MatchAtomic(call->op)) {
      found = true;
      return ffi::VisitInterrupt();
    }
    if (const auto *op = call->op.as<OpObj>();
        op && op->name == "tl.annotation")
      return std::nullopt;
    return StmtExprVisitor::Visit_(call);
  }
};

struct AtomicAccess {
  tirx::TensorVar tensor;
  Array<PrimExpr> indices;
};

class CPUAtomicRewriter : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  // Each statement owns its own evaluation prefix. Children cannot leak
  // prefixes across a branch or into the enclosing statement's scope.
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode mode) final {
    const auto *statement = value.as<StmtObj>();
    if (statement == nullptr)
      return StmtExprMutator::Mutate(value, mode);
    auto outer = std::move(prefix_);
    prefix_ = {};
    auto result = StmtExprMutator::Mutate(value, mode);
    Stmt rewritten = std::move(result).ValueOrUnchanged(value).cast<Stmt>();
    if (!prefix_.empty()) {
      prefix_.push_back(rewritten);
      rewritten = SeqStmt(std::move(prefix_), statement->loc);
    }
    prefix_ = std::move(outer);
    return ffi::Any(rewritten);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    if (const auto *call = op->value.as<CallObj>()) {
      if (auto atomic = MatchAtomic(call->op)) {
        LowerAtomic(*call, *atomic, false);
        return Evaluate(0, op->loc);
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    // Freeze each read at its original evaluation point. Otherwise a later
    // return_prev prefix could update memory before an earlier sibling read.
    auto indices = MutateIndices(op->indices, mode);
    return Capture(TensorLoad(op->source, indices, op->loc), "cpu_load")
        .as_or_throw<PrimExpr>();
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LetObj *op,
                                InplaceMode mode) final {
    auto value = Mutate(op->value, mode).ValueOrUnchanged(op->value);
    prefix_.push_back(Bind(op->var, value, op->loc));
    return Mutate(op->body, mode).ValueOrUnchanged(op->body);
  }

  UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    auto outer = std::move(prefix_);
    prefix_ = {};
    PrimExpr condition =
        Mutate(op->condition, mode).ValueOrUnchanged(op->condition);
    auto condition_prefix = std::move(prefix_);
    prefix_ = {};
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    prefix_ = std::move(outer);
    if (condition_prefix.empty())
      return While(condition, body, op->loc);
    condition_prefix.push_back(
        If(!condition, Break(op->loc), std::nullopt, op->loc));
    condition_prefix.push_back(body);
    return While(MakeConst(PrimType::Bool(), true),
                 SeqStmt(std::move(condition_prefix), op->loc), op->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (op->op.same_as(atomic_load_elem_op())) {
      auto access = ParseAccess(op->args[0], mode);
      return Capture(TensorLoad(access.tensor, access.indices, op->loc),
                     "atomic_load");
    }
    if (auto atomic = MatchAtomic(op->op)) {
      TVM_FFI_CHECK(atomic->returns_previous, ValueError)
          << "CPU atomic lowering: a non-returning atomic must be a standalone "
             "statement";
      TVM_FFI_CHECK_EQ(atomic->lanes, 1, ValueError)
          << "CPU atomic lowering: atomic_addx" << atomic->lanes
          << " with return_prev=True is not supported on CPU";
      return LowerAtomic(*op, *atomic, true).value();
    }
    if (op->op.same_as(Op::Get("prim.if_then_else"))) {
      PrimExpr condition = Mutate(op->args[0], mode)
                               .ValueOrUnchanged(op->args[0])
                               .as_or_throw<PrimExpr>();
      auto true_branch = CaptureBranch(op->args[1], mode);
      auto false_branch = CaptureBranch(op->args[2], mode);
      if (true_branch.first.empty() && false_branch.first.empty()) {
        return Call(op->ty, op->op,
                    {condition, true_branch.second, false_branch.second},
                    op->attrs, op->ty_args, op->loc);
      }
      tirx::TensorType slot_type({}, op->ty.as_or_throw<PrimType>(),
                                 runtime::MemoryScope("local"));
      tirx::TensorVar slot("atomic_branch_" + std::to_string(counter_++),
                           slot_type, op->loc);
      prefix_.push_back(Bind(slot,
                             Call(slot_type, tirx::alloc_tensor_op(), {}, {},
                                  {slot_type}, op->loc),
                             op->loc));
      true_branch.first.push_back(TensorStore(
          slot, {}, true_branch.second.as_or_throw<PrimExpr>(), op->loc));
      false_branch.first.push_back(TensorStore(
          slot, {}, false_branch.second.as_or_throw<PrimExpr>(), op->loc));
      prefix_.push_back(
          If(condition, SeqStmt(std::move(true_branch.first), op->loc),
             SeqStmt(std::move(false_branch.first), op->loc), op->loc));
      return TensorLoad(slot, {}, op->loc);
    }
    if (const auto *primitive = op->op.as<OpObj>()) {
      if (primitive->name == "tl.annotation")
        return ffi::Unchanged();
      if (primitive->name == "tl.access_ptr" ||
          primitive->name == "tirx.address_of") {
        // A TensorLoad inside an address operator denotes an address, not a
        // read.
        if (const auto *load = op->args[0].as<TensorLoadObj>()) {
          Array<Expr> arguments;
          arguments.push_back(TensorLoad(
              load->source, MutateIndices(load->indices, mode), load->loc));
          for (size_t i = 1; i < op->args.size(); ++i)
            arguments.push_back(
                Mutate(op->args[i], mode).ValueOrUnchanged(op->args[i]));
          return Call(op->ty, op->op, arguments, op->attrs, op->ty_args,
                      op->loc);
        }
      }
    }
    Expr rewritten = StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
        ffi::GetRef<Expr>(op));
    const auto *call = rewritten.as<CallObj>();
    if (call && call->ty.as<PrimTypeObj>() &&
        !call->ty.as_or_throw<PrimType>().IsVoid()) {
      auto primitive = call->op.as<Op>();
      static const auto effects = Op::GetAttrMap<int>("TCallEffectKind");
      if (!primitive.has_value() || !effects.count(primitive.value()) ||
          effects[primitive.value()] > static_cast<int>(CallEffectKind::kPure))
        return Capture(rewritten, "cpu_call");
    }
    return rewritten;
  }

private:
  Expr Capture(Expr value, const std::string &name) {
    Var variable(name + "_" + std::to_string(counter_++), value->ty,
                 value->loc);
    prefix_.push_back(Bind(variable, std::move(value), variable->loc));
    return variable;
  }

  std::pair<Array<Stmt>, Expr> CaptureBranch(const Expr &expression,
                                             InplaceMode mode) {
    auto outer = std::move(prefix_);
    prefix_ = {};
    Expr result = Mutate(expression, mode).ValueOrUnchanged(expression);
    auto branch = std::move(prefix_);
    prefix_ = std::move(outer);
    return {std::move(branch), std::move(result)};
  }

  Array<PrimExpr> MutateIndices(const Array<PrimExpr> &indices,
                                InplaceMode mode) {
    Array<PrimExpr> result;
    for (const auto &index : indices)
      result.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    return result;
  }

  AtomicAccess ParseAccess(const Expr &pointer, InplaceMode mode) {
    const auto *call = pointer.as<CallObj>();
    TVM_FFI_CHECK(call && call->op.same_as(access_ptr()) && !call->args.empty(),
                  ValueError)
        << "CPU atomic lowering expects a typed tl.access_ptr";
    const auto *load = call->args[0].as<TensorLoadObj>();
    TVM_FFI_CHECK(load != nullptr, ValueError)
        << "CPU atomic lowering expects a TensorLoad address operand";
    return {load->source.as_or_throw<tirx::TensorVar>(),
            MutateIndices(load->indices, mode)};
  }

  Array<PrimExpr> ElementIndices(Array<PrimExpr> indices, int lane) {
    TVM_FFI_CHECK(!indices.empty(), ValueError)
        << "A vector atomic needs an element axis";
    auto last = indices.back();
    if (const auto *ramp = last.as<prim::RampObj>())
      indices.Set(indices.size() - 1,
                  ramp->base + MakeConst(ramp->base.ty(), lane) * ramp->stride);
    else if (lane != 0)
      indices.Set(indices.size() - 1, last + MakeConst(last.ty(), lane));
    return indices;
  }

  Optional<Expr> LowerAtomic(const CallObj &call, const AtomicOp &op,
                             bool result_needed) {
    AtomicAccess destination =
        ParseAccess(call.args[0], InplaceMode::kDisallow);
    if (op.kind == AtomicKind::kAddVector) {
      AtomicAccess source = ParseAccess(call.args[1], InplaceMode::kDisallow);
      for (int lane = 0; lane < op.lanes; ++lane) {
        auto dst_indices = ElementIndices(destination.indices, lane);
        auto src_indices = ElementIndices(source.indices, lane);
        PrimExpr value = cast(TensorLoad(source.tensor, src_indices),
                              destination.tensor.dtype());
        prefix_.push_back(TensorStore(
            destination.tensor, dst_indices,
            TensorLoad(destination.tensor, dst_indices) + value, call.loc));
      }
      return std::nullopt;
    }
    Expr argument = Mutate(call.args[1]).ValueOrUnchanged(call.args[1]);
    PrimExpr value =
        cast(argument.as_or_throw<PrimExpr>(), destination.tensor.dtype());
    if (op.kind == AtomicKind::kStore) {
      prefix_.push_back(TensorStore(destination.tensor, destination.indices,
                                    value, call.loc));
      return std::nullopt;
    }
    PrimExpr previous =
        Capture(TensorLoad(destination.tensor, destination.indices, call.loc),
                "atomic_old")
            .as_or_throw<PrimExpr>();
    prefix_.push_back(TensorStore(destination.tensor, destination.indices,
                                  Combine(op.kind, previous, value), call.loc));
    return result_needed ? Optional<Expr>(previous) : std::nullopt;
  }

  Array<Stmt> prefix_;
  int counter_ = 0;
};
} // namespace

tvm::transform::Pass LowerCPUAtomics() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    auto finder = ffi::make_object<AtomicFinder>();
    finder->Visit(function->body);
    if (!finder->found)
      return function;
    auto rewriter = ffi::make_object<CPUAtomicRewriter>();
    function.CopyOnWrite()->body =
        rewriter->Mutate(function->body).ValueOrUnchanged(function->body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerCPUAtomics");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cpu.transform.LowerCPUAtomics",
                                   LowerCPUAtomics);
}
} // namespace tvm::tl
