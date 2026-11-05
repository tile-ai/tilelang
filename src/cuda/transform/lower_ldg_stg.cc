/*!
 * \file lower_ldg_stg.cc
 * \brief Lower contiguous native global accesses to CUDA ldg/stg intrinsics.
 *
 * Both ordinary and predicated lowering are opt-in. Native tensors keep their
 * address/layout contract until tl.access_ptr is lowered; an index Ramp alone
 * is not proof that a strided or swizzled tensor is physically contiguous.
 */

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cmath>
#include <numeric>
#include <optional>

#include "backend/common/target_utils.h"
#include "cuda/op/builtin.h"
#include "ir/analyzer.h"
#include "ir/tensor_utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

class LowerLDGSTGRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  LowerLDGSTGRewriter(bool ordinary, bool predicated)
      : ordinary_(ordinary), predicated_(predicated) {}

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *store,
                            InplaceMode mode) final {
    if (ordinary_) {
      if (auto access =
              GetAccess(store->dest, store->indices, store->value->ty))
        return LowerStore(store, access.value(), std::nullopt, mode);
    }
    return StmtExprMutator::Mutate_(store, mode);
  }

  UnchangedOr<Stmt> Mutate_(const IfObj *branch, InplaceMode mode) final {
    // Native branch bodies are always SeqStmt, including singleton bodies.
    if (predicated_ && !branch->else_case &&
        branch->then_case->seq.size() == 1 &&
        SafeScalarExpression(branch->condition)) {
      if (auto store = branch->then_case->seq[0].as<TensorStoreObj>()) {
        if (auto access =
                GetAccess(store->dest, store->indices, store->value->ty)) {
          if (SafeStoreValue(store->value)) {
            PrimExpr predicate = Mutate(branch->condition, mode)
                                     .ValueOrUnchanged(branch->condition);
            return LowerStore(store, access.value(), predicate, mode);
          }
        }
      }
    }
    return StmtExprMutator::Mutate_(branch, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *load,
                                InplaceMode mode) final {
    if (ordinary_ || predicate_) {
      if (auto access = GetAccess(load->source, load->indices, load->ty))
        return LowerLoad(load, access.value(), predicate_);
    }
    return StmtExprMutator::Mutate_(load, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    // TensorLoad under these operators denotes an addressable element. It is
    // not a memory read, including on a second invocation of this pass.
    if (call->op.same_as(address_of_op()) ||
        call->op.same_as(Op::Get("tl.access_ptr")))
      return ffi::Unchanged();
    if (predicated_ && call->op.same_as(if_then_else_op()) &&
        call->args.size() == 3 && PositiveZero(call->args[2])) {
      auto condition = call->args[0].as<PrimExpr>();
      auto load = call->args[1].as<TensorLoadObj>();
      if (condition && load && SafeScalarExpression(condition.value())) {
        if (auto access = GetAccess(load->source, load->indices, load->ty)) {
          PrimExpr guard = Mutate(condition.value(), mode)
                               .ValueOrUnchanged(condition.value());
          if (predicate_)
            guard = And(predicate_.value(), guard);
          return LowerLoad(load, access.value(), guard);
        }
      }
    }
    return StmtExprMutator::Mutate_(call, mode);
  }

private:
  struct Access {
    TensorVar tensor;
    PrimExpr base;
    PrimType value_type;
    int bits;
    int index_lanes;
  };

  bool ordinary_;
  bool predicated_;
  Optional<PrimExpr> predicate_;
  sym::Analyzer analyzer_{CreateTileLangAnalyzer()};

  static bool PositiveZero(const Expr &value) {
    if (auto broadcast = value.as<BroadcastObj>())
      return PositiveZero(broadcast->value);
    if (auto integer = value.as<IntImmObj>())
      return integer->value == 0;
    if (auto floating = value.as<FloatImmObj>())
      return floating->value == 0.0 && !std::signbit(floating->value);
    return false;
  }

  // Predicated instructions still evaluate their address/value arguments.
  // Keep memory reads and potentially trapping arithmetic inside the original
  // branch unless the load itself can be replaced by an equivalent masked ldg.
  static bool HasTrappingArithmetic(const ffi::ObjectRef &node) {
    return node.as<DivObj>() || node.as<ModObj>() || node.as<FloorDivObj>() ||
           node.as<FloorModObj>();
  }

  static bool SafeScalarExpression(const Expr &expression) {
    if (SideEffect(expression) > CallEffectKind::kPure)
      return false;
    bool safe = true;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        expression,
        [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          safe = safe && !HasTrappingArithmetic(node);
          return ffi::WalkResult::Advance();
        });
    return safe;
  }

  std::optional<Access> GetAccess(const Expr &source,
                                  const Array<PrimExpr> &indices,
                                  const Type &value_type) {
    auto tensor = source.as<TensorVar>();
    if (!tensor || tensor.value().scope() != runtime::MemoryScope::Global())
      return std::nullopt;
    TVM_FFI_CHECK_EQ(indices.size(), 1U, ValueError)
        << "LowerLDGSTG requires a flattened tensor with one index: "
        << tensor.value()->name;
    auto primitive = value_type.as<PrimType>();
    if (!primitive || primitive.value().IsScalableVector())
      return std::nullopt;
    PrimType type = primitive.value();
    int64_t bits = static_cast<int64_t>(type.bits()) * type.lanes();
    if (bits != 32 && bits != 64 && bits != 128 && bits != 256)
      return std::nullopt;
    PrimExpr base = indices[0];
    int lanes = 1;
    if (auto ramp = base.as<RampObj>()) {
      if (!IsOne(ramp->stride))
        return std::nullopt;
      lanes = ramp->ty.as_or_throw<PrimType>().lanes();
      base = ramp->base;
    } else if (base->ty.as_or_throw<PrimType>().lanes() != 1) {
      return std::nullopt;
    }
    if (!SafeScalarExpression(base))
      return std::nullopt;
    // These explicit PTX transactions require their natural byte alignment.
    // A Ramp only proves lane spacing. Keep an unaligned vector access in IR
    // for the owning emitter's guarded/byte-copy implementation instead of
    // replacing it with an unconditionally aligned ldg/stg instruction.
    const int bytes = static_cast<int>(bits) / 8;
    const auto tensor_type = tensor.value().ty();
    const PrimType element_type = tensor.value().dtype();
    if (element_type.WithLanes(1) == PrimType::Bool() ||
        tensor_type->data_alignment <= 0 ||
        tensor_type->data_alignment % bytes != 0)
      return std::nullopt;
    Array<PrimExpr> offsets = tensor_type->ElemOffset({base});
    if (offsets.size() != 1 || !SafeScalarExpression(offsets[0]))
      return std::nullopt;
    const int element_bits = element_type.bits() * element_type.lanes();
    const int divisor =
        static_cast<int>(bits) / std::gcd(static_cast<int>(bits), element_bits);
    if (!analyzer_->CanProveEqual(
            floormod(offsets[0], IntImm(offsets[0].ty(), divisor)), 0))
      return std::nullopt;
    // Prove physical contiguity for every lane, retaining type-carried offset
    // and strides. This rejects e.g. a stride-2 view even with Ramp(base,1,4).
    if (lanes > 1) {
      for (int lane = 1; lane < lanes; ++lane) {
        PrimExpr delta = IntImm(base->ty.as_or_throw<PrimType>(), lane);
        Array<PrimExpr> next = tensor.value().ty()->ElemOffset({base + delta});
        if (next.size() != 1 ||
            !analyzer_->CanProveEqual(next[0] - offsets[0], delta))
          return std::nullopt;
      }
    }
    return Access{tensor.value(), base, type, static_cast<int>(bits), lanes};
  }

  bool SafeStoreValue(const Expr &value) {
    bool safe = true;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        value,
        [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (!safe)
            return ffi::WalkResult::Skip();
          if (auto load = node.as<TensorLoadObj>()) {
            safe = GetAccess(load->source, load->indices, load->ty).has_value();
          } else if (auto call = node.as<CallObj>()) {
            if (!call->op.same_as(if_then_else_op()))
              safe = SideEffect(GetRef<Call>(call)) <= CallEffectKind::kPure;
          } else if (HasTrappingArithmetic(node)) {
            safe = false;
          }
          return ffi::WalkResult::Advance();
        });
    return safe;
  }

  static Op LoadOp(int bits) {
    switch (bits) {
    case 32:
      return ldg32();
    case 64:
      return ldg64();
    case 128:
      return ldg128();
    default:
      return ldg256();
    }
  }

  static Op StoreOp(int bits) {
    switch (bits) {
    case 32:
      return stg32();
    case 64:
      return stg64();
    case 128:
      return stg128();
    default:
      return stg256();
    }
  }

  static Expr Pointer(const Access &access, int mask, const Location &loc) {
    return MakeTensorAccessPtr(
        TensorLoad(access.tensor, {access.base}, loc),
        IntImm(access.base->ty.as_or_throw<PrimType>(), access.index_lanes),
        mask);
  }

  static PrimExpr LowerLoad(const TensorLoadObj *load, const Access &access,
                            const Optional<PrimExpr> &predicate) {
    PrimType raw_type = PrimType::UInt(32, access.bits / 32);
    Array<Expr> arguments{Pointer(access, 1, load->loc)};
    if (predicate)
      arguments.push_back(predicate.value());
    Expr raw =
        Call(raw_type, LoadOp(access.bits), arguments, {}, {}, load->loc);
    return tirx::reinterpret(Type(access.value_type), raw, load->loc)
        .as_or_throw<PrimExpr>();
  }

  Stmt LowerStore(const TensorStoreObj *store, const Access &access,
                  const Optional<PrimExpr> &predicate, InplaceMode mode) {
    Optional<PrimExpr> previous = predicate_;
    predicate_ = predicate;
    Expr value = Mutate(store->value, mode).ValueOrUnchanged(store->value);
    predicate_ = previous;
    PrimType raw_type = PrimType::UInt(32, access.bits / 32);
    value = tirx::reinterpret(Type(raw_type), value, store->loc);
    Array<Expr> arguments{Pointer(access, 2, store->loc), value};
    if (predicate)
      arguments.push_back(predicate.value());
    return Evaluate(Call(PrimType::Void(), StoreOp(access.bits), arguments, {},
                         {}, store->loc),
                    store->loc);
  }
};

tvm::transform::Pass LowerLDGSTG() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext context) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (!target || target.value()->kind->name != "cuda" ||
        TargetIsCuTeDSL(target.value()))
      return function;
    bool ordinary =
        context->GetConfig<bool>(kEnableLowerLDGSTG).value_or(false);
    bool predicated =
        context->GetConfig<bool>(kEnableLowerLDGSTGPredicated).value_or(false);
    if (!ordinary && !predicated)
      return function;
    auto rewriter = ffi::make_object<LowerLDGSTGRewriter>(ordinary, predicated);
    auto *node = function.CopyOnWrite();
    node->body = rewriter->Mutate(node->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(node->body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerLDGSTG");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerLDGSTG",
                                   LowerLDGSTG);
}
} // namespace tvm::tl
