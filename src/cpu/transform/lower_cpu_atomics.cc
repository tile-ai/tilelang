#include "compat/expr.h"
#include "compat/visit.h"
#include "cpu/op/builtin.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <string>
#include <vector>

namespace tvm::tl {
namespace {

enum class AtomicKind { kAdd, kMax, kMin, kOr, kStore, kLoad };

struct AtomicOperation {
  AtomicKind kind;
  int lanes;
  bool returns_previous;
};

std::optional<AtomicOperation> MatchAtomic(const Expr &operation) {
  const auto *node = operation.as<OpNode>();
  if (!node)
    return std::nullopt;
  std::string name = node->name;
  bool returns = name.find("_ret_") != std::string::npos;
  if (operation.same_as(atomic_load_elem_op()))
    return AtomicOperation{AtomicKind::kLoad, 1, true};
  if (operation.same_as(atomic_store_elem_op()))
    return AtomicOperation{AtomicKind::kStore, 1, false};
  if (operation.same_as(atomic_or_elem_op()))
    return AtomicOperation{AtomicKind::kOr, 1, false};
  if (operation.same_as(atomic_add_elem_op()) ||
      operation.same_as(atomic_add_ret_elem_op())) {
    return AtomicOperation{AtomicKind::kAdd, 1, returns};
  }
  if (operation.same_as(atomic_max_elem_op()) ||
      operation.same_as(atomic_max_ret_elem_op())) {
    return AtomicOperation{AtomicKind::kMax, 1, returns};
  }
  if (operation.same_as(atomic_min_elem_op()) ||
      operation.same_as(atomic_min_ret_elem_op())) {
    return AtomicOperation{AtomicKind::kMin, 1, returns};
  }
  if (operation.same_as(atomic_addx2_elem_op()) ||
      operation.same_as(atomic_addx2_ret_elem_op())) {
    return AtomicOperation{AtomicKind::kAdd, 2, returns};
  }
  if (operation.same_as(atomic_addx4_elem_op()) ||
      operation.same_as(atomic_addx4_ret_elem_op())) {
    return AtomicOperation{AtomicKind::kAdd, 4, returns};
  }
  return std::nullopt;
}

bool ContainsAtomic(const ffi::ObjectRef &value) {
  bool found = false;
  PostOrderVisit(value, [&](const ffi::ObjectRef &object) {
    if (const auto *call = object.as<CallNode>())
      found |= MatchAtomic(call->op).has_value();
  });
  return found;
}

PrimExpr Combine(AtomicKind kind, const PrimExpr &previous,
                 const PrimExpr &value) {
  switch (kind) {
  case AtomicKind::kAdd:
    return previous + value;
  case AtomicKind::kMax:
    return Max(previous, value);
  case AtomicKind::kMin:
    return Min(previous, value);
  case AtomicKind::kOr:
    return previous | value;
  case AtomicKind::kStore:
    return value;
  case AtomicKind::kLoad:
    return previous;
  }
  TVM_FFI_UNREACHABLE();
}

struct AtomicAccess {
  tirx::TensorVar tensor;
  ffi::Array<PrimExpr> indices;
};

class CPUAtomicLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;

  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                                    ffi::InplaceMode mode) final {
    if (auto statement = value.as<Stmt>()) {
      auto previous = std::move(prefix_);
      prefix_.clear();
      auto result = StmtExprMutator::Mutate(value, mode)
                        .ValueOrUnchanged(ffi::Any(value))
                        .cast<Stmt>();
      prefix_.push_back(result);
      result = SeqStmt(prefix_, statement.value()->loc);
      prefix_ = std::move(previous);
      return ffi::Any(result);
    }
    auto result =
        StmtExprMutator::Mutate(value, mode).ValueOrUnchanged(ffi::Any(value));
    if (auto expression = result.as<PrimExpr>()) {
      if (!expression.value().ty().IsVoid() &&
          SideEffect(expression.value()) > CallEffectKind::kPure) {
        return ffi::Any(Snapshot(expression.value()));
      }
    }
    return result;
  }

private:
  PrimExpr Snapshot(const PrimExpr &expression) {
    PrimVar variable("atomic_value_" + std::to_string(counter_++),
                     expression.ty());
    prefix_.push_back(Bind(variable, expression));
    return variable;
  }

  AtomicAccess ParseAccess(const Expr &expression) {
    const auto *call = expression.as<CallNode>();
    TVM_FFI_CHECK(call && (call->op.same_as(access_ptr()) ||
                           call->op.same_as(tirx::address_of_op())),
                  ValueError)
        << "CPU atomics require a typed tensor access pointer";
    TensorRegion region = NormalizeToBufferRegion(call->args[0]);
    ffi::Array<PrimExpr> indices;
    for (const Range &range : region->region) {
      indices.push_back(Mutate(range->begin).ValueOrUnchanged(range->begin));
    }
    return {region->source.as_or_throw<tirx::TensorVar>(), indices};
  }

  ffi::Array<PrimExpr> ElementIndices(ffi::Array<PrimExpr> indices, int lane) {
    TVM_FFI_CHECK(lane == 0 || !indices.empty(), ValueError)
        << "Vector atomics require a non-scalar tensor";
    if (lane)
      indices.Set(indices.size() - 1,
                  indices.back() + MakeConst(indices.back().ty(), lane));
    return indices;
  }

  PrimExpr LowerAtomic(const CallNode *node, const AtomicOperation &operation,
                       bool keep_result) {
    TVM_FFI_CHECK(operation.lanes == 1 || !operation.returns_previous,
                  ValueError)
        << "CPU vector atomics do not support return_prev";
    size_t operands = operation.kind == AtomicKind::kLoad ? 1 : 2;
    TVM_FFI_CHECK(node->args.size() == operands ||
                      node->args.size() == operands + 1,
                  ValueError)
        << "CPU atomic has an invalid operand count";
    auto order = node->args.size() == operands ? std::optional<int64_t>(0)
                                               : GetConstInt(node->args.back());
    TVM_FFI_CHECK(order && *order >= 0 && *order <= 5, ValueError)
        << "CPU atomic memory order must be a constant in [0, 5]";
    TVM_FFI_CHECK(!keep_result || operation.returns_previous, ValueError)
        << "A non-returning CPU atomic cannot appear in an expression";
    AtomicAccess destination = ParseAccess(node->args[0]);
    if (operation.kind == AtomicKind::kLoad) {
      return Snapshot(TensorLoad(destination.tensor, destination.indices));
    }
    ffi::Array<PrimExpr> values;
    if (operation.lanes == 1) {
      auto value = Mutate(node->args[1])
                       .ValueOrUnchanged(node->args[1])
                       .as_or_throw<PrimExpr>();
      values.push_back(prim::cast(value, destination.tensor.dtype()));
    } else if (node->args[1]->ty.as<PtrTypeNode>()) {
      auto source = ParseAccess(node->args[1]);
      for (int lane = 0; lane < operation.lanes; ++lane) {
        auto value =
            TensorLoad(source.tensor, ElementIndices(source.indices, lane));
        values.push_back(
            Snapshot(prim::cast(value, destination.tensor.dtype())));
      }
    } else {
      PrimExpr vector = Mutate(node->args[1])
                            .ValueOrUnchanged(node->args[1])
                            .as_or_throw<PrimExpr>();
      TVM_FFI_CHECK_EQ(vector.ty().lanes(), operation.lanes, ValueError);
      for (int lane = 0; lane < operation.lanes; ++lane) {
        values.push_back(prim::cast(Shuffle({vector}, {IntImm::Int32(lane)}),
                                    destination.tensor.dtype()));
      }
    }
    ffi::Array<PrimExpr> previous;
    for (int lane = 0; lane < operation.lanes; ++lane) {
      auto indices = ElementIndices(destination.indices, lane);
      PrimExpr old_value =
          operation.kind == AtomicKind::kStore
              ? MakeConst(destination.tensor.dtype(), 0)
              : Snapshot(TensorLoad(destination.tensor, indices));
      prefix_.push_back(TensorStore(
          destination.tensor, indices,
          Combine(operation.kind, old_value, values[lane]), node->loc));
      previous.push_back(old_value);
    }
    if (!keep_result)
      return PrimExpr(0);
    if (operation.lanes == 1)
      return previous[0];
    ffi::Array<PrimExpr> lanes;
    for (int lane = 0; lane < operation.lanes; ++lane)
      lanes.push_back(IntImm::Int32(lane));
    return Shuffle(previous, lanes);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    if (auto operation = MatchAtomic(node->op))
      return LowerAtomic(node, *operation, true);
    if (node->op.same_as(prim::if_then_else_op()) &&
        SideEffect(ffi::GetRef<Expr>(node)) > CallEffectKind::kPure) {
      PrimExpr condition = Mutate(node->args[0])
                               .ValueOrUnchanged(node->args[0])
                               .as_or_throw<PrimExpr>();
      auto parent = std::move(prefix_);
      prefix_.clear();
      PrimExpr then_value = Mutate(node->args[1])
                                .ValueOrUnchanged(node->args[1])
                                .as_or_throw<PrimExpr>();
      auto then_prefix = std::move(prefix_);
      prefix_.clear();
      PrimExpr else_value = Mutate(node->args[2])
                                .ValueOrUnchanged(node->args[2])
                                .as_or_throw<PrimExpr>();
      auto else_prefix = std::move(prefix_);
      prefix_ = std::move(parent);
      tirx::TensorVar result(
          "atomic_select_" + std::to_string(counter_++),
          tirx::TensorType({}, then_value.ty(), runtime::MemoryScope("local")));
      prefix_.push_back(Bind(result, Call(result.ty(), tirx::alloc_tensor_op(),
                                          {}, std::nullopt, {result.ty()})));
      then_prefix.push_back(TensorStore(result, {}, then_value));
      else_prefix.push_back(TensorStore(result, {}, else_value));
      prefix_.push_back(
          If(condition, SeqStmt(then_prefix), SeqStmt(else_prefix), node->loc));
      return TensorLoad(result, {});
    }
    if (node->op.same_as(access_ptr()) ||
        node->op.same_as(tirx::address_of_op())) {
      auto access = ParseAccess(ffi::GetRef<Expr>(node));
      ffi::Array<Range> ranges;
      for (const auto &index : access.indices)
        ranges.push_back(Range::FromMinExtent(index, MakeConst(index.ty(), 1)));
      auto arguments = node->args;
      arguments.Set(0, TensorRegion(access.tensor, ranges, TensorRegionType()));
      return Call(node->ty, node->op, arguments, node->attrs, node->ty_args,
                  node->loc);
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LetNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    PrimVar variable("atomic_let_" + std::to_string(counter_++), value.ty(),
                     node->loc);
    prefix_.push_back(Bind(variable, value, node->loc));
    PrimExpr body = tirx::SubstituteWithDataTypeLegalization(
        node->body, [&](const Var &candidate) -> ffi::Optional<PrimExpr> {
          return candidate.same_as(node->var)
                     ? ffi::Optional<PrimExpr>(variable)
                     : std::nullopt;
        });
    return Mutate(body, mode).ValueOrUnchanged(body);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateNode *node,
                                 ffi::InplaceMode mode) final {
    if (const auto *call = node->value.as<CallNode>()) {
      if (auto operation = MatchAtomic(call->op)) {
        LowerAtomic(call, *operation, false);
        return SeqStmt({}, node->loc);
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *node,
                                 ffi::InplaceMode mode) final {
    auto parent = std::move(prefix_);
    prefix_.clear();
    PrimExpr condition =
        Mutate(node->condition, mode).ValueOrUnchanged(node->condition);
    auto checks = std::move(prefix_);
    prefix_ = std::move(parent);
    SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    if (checks.empty())
      return While(condition, body, node->loc);
    checks.push_back(
        If(!condition, SeqStmt(Break(node->loc)), std::nullopt, node->loc));
    checks.push_back(body);
    return While(IntImm::Bool(true), SeqStmt(checks), node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    TVM_FFI_CHECK(!ContainsAtomic(node->end) && !ContainsAtomic(node->step),
                  ValueError)
        << "CPU atomic operations in a For end or step require an explicit "
           "while loop";
    TVM_FFI_CHECK(node->kind != ForKind::kParallel ||
                      !ContainsAtomic(node->body),
                  ValueError)
        << "Serial CPU atomic lowering cannot be applied to a concurrent loop";
    PrimExpr begin = Mutate(node->begin, mode).ValueOrUnchanged(node->begin);
    SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    return For(node->var, begin, node->end, node->step, node->kind, body,
               node->annotations, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    ffi::Array<PrimExpr> indices;
    for (const auto &index : node->indices)
      indices.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    return TensorStore(node->dest, indices, value, node->loc);
  }

  ffi::Array<Stmt> prefix_;
  int counter_{0};
};

} // namespace

namespace transform {

tvm::transform::Pass LowerCPUAtomics() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body && ContainsAtomic(function->body.value())) {
      auto lowerer = ffi::make_object<CPUAtomicLowerer>();
      SeqStmt body = function->body.value();
      function.CopyOnWrite()->body =
          lowerer->Mutate(body).ValueOrUnchanged(body);
      function = WithAttr(std::move(function), attr::kCPUHadAtomics, true);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerCPUAtomics");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cpu.transform.LowerCPUAtomics",
                                   LowerCPUAtomics);
}

} // namespace transform
} // namespace tvm::tl
