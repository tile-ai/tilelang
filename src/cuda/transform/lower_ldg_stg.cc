#include "backend/common/target_utils.h"
#include "compat/expr.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <cmath>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct GlobalAccess {
  TensorVar tensor;
  PrimExpr base;
  PrimType type;
  int bits;
};

Optional<GlobalAccess>
MatchAccess(const Expr &tensor, const Array<PrimExpr> &indices, PrimType type) {
  auto source = tensor.as<TensorVar>();
  if (!source || !IsGlobalBuffer(*source) || indices.size() != 1 ||
      !source.value().ty()->HasDefaultLayout())
    return std::nullopt;
  PrimExpr base = sym::Analyzer()->Simplify(indices[0]);
  if (const auto *ramp = base.as<RampNode>()) {
    if (!IsOne(ramp->stride))
      return std::nullopt;
    base = ramp->base;
  } else if (!base.ty().IsScalar()) {
    return std::nullopt;
  }
  int bits = type.bits() * type.lanes();
  if (bits != 32 && bits != 64 && bits != 128 && bits != 256)
    return std::nullopt;
  return GlobalAccess{*source, base, type, bits};
}

PrimExpr BitCast(PrimExpr value, PrimType type) {
  if (value.ty() == type)
    return value;
  return Call(type, reinterpret_op(), {value}).as_or_throw<PrimExpr>();
}

class GlobalMemoryLowerer : public StmtExprMutator {
public:
  GlobalMemoryLowerer(bool unguarded, bool predicated)
      : unguarded_(unguarded), predicated_(predicated) {}

private:
  bool CanSpeculate(const PrimExpr &value, bool allow_global_loads) {
    if (SideEffect(value) > (allow_global_loads ? CallEffectKind::kReadState
                                                : CallEffectKind::kPure))
      return false;
    bool safe = true;
    PostOrderVisit(value, [&](const ObjectRef &object) {
      if (const auto *load = object.as<TensorLoadNode>()) {
        safe &= allow_global_loads && MatchAccess(load->source, load->indices,
                                                  GetRef<TensorLoad>(load).ty())
                                          .has_value();
        for (const PrimExpr &index : load->indices)
          safe &= CanSpeculate(index, false);
      }
      Optional<PrimExpr> divisor;
      if (const auto *division = object.as<DivNode>())
        divisor = division->b;
      if (const auto *division = object.as<FloorDivNode>())
        divisor = division->b;
      if (const auto *division = object.as<ModNode>())
        divisor = division->b;
      if (const auto *division = object.as<FloorModNode>())
        divisor = division->b;
      if (divisor && divisor.value().ty().MatchesCode(kDLInt, kDLUInt)) {
        safe &= analyzer_->CanProve(*divisor != 0);
      }
      if (const auto *call = object.as<CallNode>();
          call && call->op.same_as(prim::if_then_else_op())) {
        safe &= CanSpeculate(call->args[0].as_or_throw<PrimExpr>(), false);
      }
    });
    return safe;
  }

  Expr Pointer(const GlobalAccess &access, InplaceMode mode) {
    PrimExpr base = Mutate(access.base, mode).ValueOrUnchanged(access.base);
    return Call(std::nullopt, address_of_op(),
                {TensorLoad(access.tensor, {base})});
  }

  PrimExpr Load(const GlobalAccess &access, Optional<PrimExpr> predicate,
                InplaceMode mode) {
    Array<Expr> arguments{Pointer(access, mode)};
    if (predicate)
      arguments.push_back(*predicate);
    PrimType packed = PrimType::UInt(32, access.bits / 32);
    PrimExpr value =
        Call(packed, Op::Get("tl.ldg" + std::to_string(access.bits)), arguments)
            .as_or_throw<PrimExpr>();
    return BitCast(value, access.type);
  }

  Stmt Store(const TensorStoreNode *node, const GlobalAccess &access,
             Optional<PrimExpr> predicate, InplaceMode mode) {
    auto previous = predicate_;
    predicate_ = predicate;
    PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    predicate_ = previous;
    Array<Expr> arguments{Pointer(access, mode),
                          BitCast(value, PrimType::UInt(32, access.bits / 32))};
    if (predicate)
      arguments.push_back(*predicate);
    return Evaluate(Call(PrimType::Void(),
                         Op::Get("tl.stg" + std::to_string(access.bits)),
                         arguments, std::nullopt, {}, node->loc));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    if (unguarded_ || predicate_) {
      if (auto access = MatchAccess(node->source, node->indices,
                                    GetRef<TensorLoad>(node).ty())) {
        return Load(*access, predicate_, mode);
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    if (unguarded_) {
      if (auto access =
              MatchAccess(node->dest, node->indices, node->value.ty())) {
        return Store(node, *access, std::nullopt, mode);
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *node, InplaceMode mode) final {
    if (predicated_ && !node->else_case && node->then_case->seq.size() == 1) {
      if (const auto *store = node->then_case->seq[0].as<TensorStoreNode>()) {
        auto access =
            MatchAccess(store->dest, store->indices, store->value.ty());
        if (access && CanSpeculate(store->value, true) &&
            CanSpeculate(access.value().base, false)) {
          PrimExpr condition =
              Mutate(node->condition, mode).ValueOrUnchanged(node->condition);
          PrimVar predicate("global_store_predicate", condition.ty());
          return SeqStmt({Bind(predicate, condition),
                          Store(store, *access, predicate, mode)});
        }
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    if (node->op.same_as(address_of_op())) {
      const auto *load = node->args[0].as<TensorLoadNode>();
      if (load) {
        Array<PrimExpr> indices = load->indices.Map([&](const PrimExpr &index) {
          return Mutate(index, mode).ValueOrUnchanged(index);
        });
        return Call(node->ty, node->op,
                    {TensorLoad(load->source, indices, load->loc)}, node->attrs,
                    node->ty_args, node->loc);
      }
    }
    if (predicated_ && node->op.same_as(prim::if_then_else_op())) {
      const auto *load = node->args[1].as<TensorLoadNode>();
      PrimExpr otherwise = node->args[2].as_or_throw<PrimExpr>();
      if (const auto *broadcast = otherwise.as<BroadcastNode>())
        otherwise = broadcast->value;
      PrimExpr condition = node->args[0].as_or_throw<PrimExpr>();
      bool zero_fill = IsZero(otherwise);
      if (const auto *constant = otherwise.as<FloatImmNode>()) {
        zero_fill = constant->value == 0.0 && !std::signbit(constant->value);
      }
      if (load && zero_fill && CanSpeculate(condition, false)) {
        auto access = MatchAccess(load->source, load->indices,
                                  GetRef<TensorLoad>(load).ty());
        if (access && CanSpeculate(access.value().base, false)) {
          condition = Mutate(condition, mode).ValueOrUnchanged(condition);
          if (predicate_)
            condition = And(*predicate_, condition);
          return Load(*access, condition, mode);
        }
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  bool unguarded_;
  bool predicated_;
  Optional<PrimExpr> predicate_;
  sym::Analyzer analyzer_;
};
} // namespace

tvm::transform::Pass LowerLDGSTG() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> Optional<tirx::Function> {
        auto target = function->GetAttr<Target>(tvm::attr::kTarget);
        if (!function->body || !target || !TargetIsCuda(*target) ||
            TargetIsCuTeDSL(*target)) {
          return function;
        }
        bool unguarded =
            context->GetConfig<bool>(kEnableLowerLDGSTG).value_or(false);
        bool predicated = context->GetConfig<bool>(kEnableLowerLDGSTGPredicated)
                              .value_or(false);
        if (!unguarded && !predicated)
          return function;
        auto lowerer = make_object<GlobalMemoryLowerer>(unguarded, predicated);
        SeqStmt body = *function->body;
        function.CopyOnWrite()->body =
            lowerer->Mutate(body).ValueOrUnchanged(body);
        return function;
      },
      0, "tl.cuda.LowerLDGSTG");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.cuda.transform.LowerLDGSTG", LowerLDGSTG);
}
} // namespace tvm::tl
