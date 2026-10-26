#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
#include "transform/common/data_type_rewriter.h"
#include <tvm/backend/opencl/op/memory.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/vector.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <vector>

#include "tvm/ffi/dtype.h"
#include "tvm/tirx/expr.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

using namespace tvm::prim;

namespace {
int GetLanesOrVScaleFactor(const PrimType &ty) {
  if (ty.IsScalableVector()) {
    return ty.VScaleFactor();
  }
  return ty.lanes();
}

bool IsVScaleCall(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    return call->op.same_as(prim::vscale_op());
  }
  return false;
}

bool TargetHasRVV(ffi::Optional<Target> target) {
  if (!target.has_value())
    return false;
  static auto target_has_feature_fn =
      tvm::ffi::Function::GetGlobal("codegen.target_has_feature");
  return target_has_feature_fn.has_value() &&
         (*target_has_feature_fn)("v", target).cast<bool>();
}

bool TargetHasVLA(ffi::Optional<Target> target) {
  if (!target.has_value())
    return false;
  bool has_vla =
      target.value()->GetAttr<bool>("feature.has_sve").value_or(false);
  if (!has_vla) {
    if (auto mattr =
            target.value()->GetAttr<ffi::Array<ffi::String>>("mattr")) {
      for (const ffi::String &attr : mattr.value()) {
        if (attr == "+sve") {
          has_vla = true;
          break;
        }
      }
    }
  }
  has_vla |= TargetHasRVV(target);
  return has_vla;
}

bool ContainsCallNode(const Stmt &stmt) {
  struct CallFinder : tvm::tl::StmtExprVisitor {
    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
      if (auto expr = value.as<PrimExpr>(); expr && expr->as<CallNode>()) {
        return VisitInterrupt();
      }
      return tvm::tl::StmtExprVisitor::Visit(value);
    }
  };
  return ffi::make_object<CallFinder>()->Visit(stmt).has_value();
}

PrimType GetTextureElementType(const Expr &texture) {
  const auto *pointer_type = texture->ty.as<PtrTypeNode>();
  TVM_FFI_ICHECK(pointer_type) << "Texture arguments must have PtrType";
  return pointer_type->element_type.as_or_throw<PrimType>();
}
} // namespace

inline PrimExpr CreateNewLanes(bool is_scalable, int lanes_or_vscale_factor) {
  if (is_scalable) {
    return prim::Mul(
        Call(PrimType::Int(32), prim::vscale_op(), {}).as_or_throw<PrimExpr>(),
        lanes_or_vscale_factor);
  } else {
    return lanes_or_vscale_factor;
  }
}

inline PrimExpr BroadcastTo(PrimExpr e, int lanes, bool is_scalable) {

  if (GetLanesOrVScaleFactor(e.ty()) == lanes &&
      e.ty().IsScalableVector() == is_scalable)
    return e;

  if (const prim::BroadcastNode *op = e.as<prim::BroadcastNode>()) {
    TVM_FFI_ICHECK(op->ty.as_or_throw<PrimType>().IsScalableVector() ==
                   is_scalable)
        << "Can't broadcast between scalable and fixed length vectors.";
    int e_lanes = GetLanesOrVScaleFactor(op->ty.as_or_throw<PrimType>());

    if (lanes % e_lanes == 0) {
      return prim::Broadcast(op->value, CreateNewLanes(is_scalable, lanes));
    }
  }

  TVM_FFI_ICHECK(e.ty().IsScalar())
      << "Cannot broadcast lanes=" << GetLanesOrVScaleFactor(e.ty())
      << " is_scalable=" << e.ty().IsScalableVector() << " to " << lanes;

  return prim::Broadcast(e, CreateNewLanes(is_scalable, lanes));
}

bool EnableTensorLevelPredication(ffi::Optional<Target> target) {
  tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
  ffi::Optional<bool> enable_tensor_predication =
      pass_ctx->GetConfig<bool>("tirx.enable_tensor_level_predication");
  if (enable_tensor_predication.has_value()) {
    return enable_tensor_predication.value();
  }

  return TargetHasVLA(target);
}

class TryPredicateTensorAccesses : public tvm::tl::StmtExprMutator {
public:
  explicit TryPredicateTensorAccesses(bool allow_offset_predication)
      : allow_offset_predication_(allow_offset_predication) {}

  std::pair<bool, Stmt> Run(Stmt stmt, PrimExpr condition) {

    if (!condition->IsInstance<prim::LTNode>()) {
      return {false, stmt};
    }

    prim::LT lt = condition.as_or_throw<prim::LT>();

    if (!lt->a->IsInstance<prim::RampNode>() ||
        !lt->b->IsInstance<prim::BroadcastNode>()) {
      return {false, stmt};
    }

    prim::Ramp pred_ramp = lt->a.as_or_throw<prim::Ramp>();
    base_ = pred_ramp->base;
    stride_ = pred_ramp->stride;
    lanes_ = pred_ramp->lanes;
    limit_ = lt->b.as_or_throw<prim::Broadcast>()->value;

    Stmt predicated_stmt =
        Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    if (num_accesses_analyzed_ > 0 &&
        num_accesses_analyzed_ == num_accesses_rewritten_) {
      return {true, predicated_stmt};
    }
    return {false, stmt};
  }

private:
  using tvm::tl::StmtExprMutator::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *op,
                                InplaceMode inplace_mode) final {
    auto load = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                    .as_or_throw<TensorLoad>();
    return TryPredicateTensorAccess(load);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) final {
    auto store = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                     .as_or_throw<TensorStore>();
    return TryPredicateTensorAccess(store);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    Call call = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();
    if (!call->op.same_as(tirx::masked_load_op()) &&
        !call->op.same_as(tirx::masked_store_op())) {
      return call;
    }

    bool is_load = call->op.same_as(tirx::masked_load_op());
    ffi::Array<PrimExpr> indices;
    for (size_t i = is_load ? 1 : 2; i + 1 < call->args.size(); ++i) {
      indices.push_back(call->args[i].as_or_throw<PrimExpr>());
    }
    if (auto lane_mask = GetLaneMask(indices)) {
      PrimExpr predicate = call->args.back().as_or_throw<PrimExpr>();
      predicate = allow_offset_predication_ ? predicate & lane_mask.value()
                                            : lane_mask.value();
      ffi::Array<Expr> args = call->args;
      args.Set(args.size() - 1, predicate);
      return Call(call->ty, call->op, args, call->attrs, call->ty_args,
                  call->loc);
    }
    return call;
  }

  ffi::Optional<PrimExpr> GetLaneMask(const ffi::Array<PrimExpr> &indices) {
    num_accesses_analyzed_ += 1;

    if (!indices.size() || !indices[0]->IsInstance<prim::RampNode>()) {
      return std::nullopt;
    }
    prim::Ramp ramp = indices[0].as_or_throw<prim::Ramp>();

    if (!tl::StructuralEqual()(ramp->stride, stride_) ||
        !tl::StructuralEqual()(ramp->lanes, lanes_)) {
      return std::nullopt;
    }

    bool same_base = tl::StructuralEqual()(ramp->base, base_);
    if (!same_base) {

      if (!allow_offset_predication_) {
        return std::nullopt;
      }
    }

    PrimType buf_predicate_dtype =
        ramp->ty.as_or_throw<PrimType>().IsScalableVector()
            ? PrimType::ScalableVector(
                  DLDataTypeCode::kDLUInt, 1,
                  GetLanesOrVScaleFactor(ramp->ty.as_or_throw<PrimType>()))
            : PrimType::UInt(
                  1, GetLanesOrVScaleFactor(ramp->ty.as_or_throw<PrimType>()));
    PrimExpr lane_mask = Call(buf_predicate_dtype,
                              tirx::get_active_lane_mask_op(), {base_, limit_})
                             .as_or_throw<PrimExpr>();

    num_accesses_rewritten_ += 1;
    return lane_mask;
  }

  PrimExpr TryPredicateTensorAccess(TensorLoad load) {
    if (auto mask = GetLaneMask(load->indices)) {
      ffi::Array<Expr> args{load->source.as_or_throw<tvm::tirx::TensorVar>()};
      for (const PrimExpr &index : load->indices)
        args.push_back(index);
      args.push_back(mask.value());
      return Call(load->ty, tirx::masked_load_op(), args, {}, {}, load->loc)
          .as_or_throw<PrimExpr>();
    }
    return load;
  }

  Stmt TryPredicateTensorAccess(TensorStore store) {
    if (auto mask = GetLaneMask(store->indices)) {
      ffi::Array<Expr> args{store->dest.as_or_throw<TensorVar>(), store->value};
      for (const PrimExpr &index : store->indices)
        args.push_back(index);
      args.push_back(mask.value());
      return Evaluate(Call(PrimType::Void(), tirx::masked_store_op(), args, {},
                           {}, store->loc),
                      store->loc);
    }
    return store;
  }

  PrimExpr base_{ffi::UnsafeInit{}};

  PrimExpr stride_{ffi::UnsafeInit{}};

  PrimExpr lanes_{ffi::UnsafeInit{}};

  PrimExpr limit_{ffi::UnsafeInit{}};

  bool allow_offset_predication_;

  size_t num_accesses_analyzed_ = 0;

  size_t num_accesses_rewritten_ = 0;
};

class VecAllocAccess : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate_;

  VecAllocAccess(const VarNode *buf, Var var, PrimExpr var_lanes)
      : buf_(buf), var_(var), var_lanes_(var_lanes) {}

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *op,
                                InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad load =
        indices.UnchangedOrSameAs(op->indices)
            ? ffi::GetRef<TensorLoad>(op)
            : TensorLoad(op->source.as_or_throw<TensorVar>(),
                         std::move(indices).ValueUnchecked(), op->loc);
    return UpdateTensorAccess(load);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) final {
    auto value = Mutate(op->value, inplace_mode);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore store =
        value.UnchangedOrSameAs(op->value) &&
                indices.UnchangedOrSameAs(op->indices)
            ? ffi::GetRef<TensorStore>(op)
            : TensorStore(op->dest.as_or_throw<TensorVar>(),
                          std::move(indices).ValueOrUnchanged(op->indices),
                          std::move(value).ValueOrUnchanged(op->value),
                          op->loc);
    return UpdateTensorAccess(store);
  }

private:
  template <typename Node> Node UpdateTensorAccess(Node node) {

    if (node->dest.template as_or_throw<TensorVar>().get() != buf_) {
      return node;
    }

    TensorVar buf{ffi::UnsafeInit{}};
    ffi::Any mapped = VarRemapGet(node->dest.template as_or_throw<TensorVar>());
    if (mapped != nullptr) {
      buf = mapped.as_or_throw<TensorVar>();
    } else {

      ffi::Array<PrimExpr> shape =
          node->dest.template as_or_throw<TensorVar>().shape();
      shape.Set(
          shape.size() - 1,
          tl::arith::Simplify(analyzer_, shape[shape.size() - 1] * var_lanes_));

      auto type = ffi::make_object<TensorTypeNode>(
          *node->dest.template as_or_throw<TensorVar>().ty().get());
      WidenLayout(type.get(), shape);
      type->shape = shape;
      buf = TensorVar(node->dest.template as_or_throw<TensorVar>()->name,
                      TensorType(std::move(type)),
                      node->dest.template as_or_throw<TensorVar>()->loc);
      VarRemapSet(node->dest.template as_or_throw<TensorVar>(), buf);
    }

    ffi::Array<PrimExpr> indices = node->indices;
    indices.Set(indices.size() - 1,
                tl::arith::Simplify(analyzer_,
                                    indices[indices.size() - 1] * var_lanes_ +
                                        var_.as_or_throw<PrimExpr>()));

    auto writer = node.CopyOnWrite();
    writer->dest = buf;
    writer->indices = indices;
    return node;
  }

  TensorLoad UpdateTensorAccess(TensorLoad node) {
    TensorVar tensor = node->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor.get() != buf_)
      return node;
    TensorVar buf{ffi::UnsafeInit{}};
    auto mapped = VarRemapGet(tensor);
    if (mapped != nullptr) {
      buf = mapped.as_or_throw<TensorVar>();
    } else {
      ffi::Array<PrimExpr> shape = tensor.shape();
      shape.Set(shape.size() - 1,
                tl::arith::Simplify(analyzer_, shape.back() * var_lanes_));
      auto type = ffi::make_object<TensorTypeNode>(*tensor.ty().get());
      WidenLayout(type.get(), shape);
      type->shape = shape;
      buf = TensorVar(tensor->name, TensorType(std::move(type)), tensor->loc);
      VarRemapSet(tensor, buf);
    }
    ffi::Array<PrimExpr> indices = node->indices;
    indices.Set(
        indices.size() - 1,
        tl::arith::Simplify(analyzer_, indices.back() * var_lanes_ +
                                           var_.as_or_throw<PrimExpr>()));
    return TensorLoad(buf, indices, node->loc);
  }

  void WidenLayout(TensorTypeNode *type, const ffi::Array<PrimExpr> &shape) {
    if (!type->layout.has_value())
      return;
    auto tile = type->layout.value().as_or_throw<TileLayout>();
    TVM_FFI_CHECK(MatchShapeStridePattern(
                      TensorType(ffi::make_object<TensorTypeNode>(*type)))
                      .has_value(),
                  ValueError)
        << "Vectorized allocation requires a regular memory layout";
    ffi::Array<Iter> shard;
    for (size_t i = 0; i < tile->shard.size(); ++i) {
      PrimExpr stride = tile->shard[i]->stride;
      if (i + 1 != tile->shard.size())
        stride *= var_lanes_;
      shard.push_back(Iter(shape[i], tl::arith::Simplify(analyzer_, stride),
                           Axis::Get("m")));
    }
    tile.CopyOnWrite()->shard = std::move(shard);
    type->layout = tile;
  }

  const VarNode *buf_;

  Var var_;

  PrimExpr var_lanes_;

  sym::Analyzer analyzer_;
};

class Vectorizer : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;

  Vectorizer(Var var, PrimExpr var_lanes, ffi::Optional<Target> target)
      : var_(var), var_lanes_(var_lanes), target_(target) {
    PrimType var_ty = var->ty.as_or_throw<PrimType>();
    ramp_ = prim::Ramp(IntImm(var_ty, 0), IntImm(var_ty, 1), var_lanes);
    VarRemapSet(var_, ramp_);
  }

  Stmt Run(const Stmt &body, InplaceMode inplace_mode) {
    Stmt result = Mutate(body, inplace_mode).ValueOrUnchanged(body);
    return scalarize_binding_scope_ ? Scalarize(body) : result;
  }

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                               InplaceMode inplace_mode) final {
    if (const auto *stmt = value.as<StmtNode>()) {
      TVM_FFI_ICHECK(!need_scalarize_);
      auto result =
          tvm::tl::StmtExprMutator::Mutate(value, InplaceMode::kDisallow);
      if (need_scalarize_) {
        need_scalarize_ = false;
        if (stmt->IsInstance<BindNode>()) {
          scalarize_binding_scope_ = true;
          return ffi::Unchanged();
        }
        return ffi::Any(Scalarize(ffi::GetRef<Stmt>(stmt)));
      }
      return result;
    }
    return tvm::tl::StmtExprMutator::Mutate(value, InplaceMode::kDisallow);
  }

  UnchangedOr<Expr> Mutate_(const OpaqueExprNode *op,
                            InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const TupleNode *op,
                            InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const TupleGetItemNode *op,
                            InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *op,
                            InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const GlobalVarNode *op,
                            InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Expr> Mutate_(const OpNode *op, InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::AddNode *op,
                                InplaceMode inplace_mode) final {
    return AddSubVec(
        op, [](PrimExpr a, PrimExpr b) { return a + b; }, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SubNode *op,
                                InplaceMode inplace_mode) final {
    return AddSubVec(
        op, [](PrimExpr a, PrimExpr b) { return a - b; }, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::MulNode *op,
                                InplaceMode inplace_mode) final {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::Unchanged();
    } else {
      bool is_vec_a = a.ty().IsScalableVector() || a.ty().IsFixedLengthVector();
      bool is_vec_b = b.ty().IsScalableVector() || b.ty().IsFixedLengthVector();
      if (is_vec_a && is_vec_b) {

        TVM_FFI_ICHECK(a.ty().IsScalableVector() == b.ty().IsScalableVector())
            << "Fixed length and scalable vectors can't be mixed in "
               "multiplication.";
      }
      if (is_vec_a || is_vec_b) {
        const prim::RampNode *b_ramp = b.as<prim::RampNode>();
        const prim::RampNode *a_ramp = a.as<prim::RampNode>();
        if (a_ramp && b.ty().IsScalar() &&
            tl::arith::CanProve(analyzer_, b > 0)) {
          PrimExpr lanes = a_ramp->lanes;
          return prim::Ramp(a_ramp->base * b, a_ramp->stride * b, lanes);
        }
        if (b_ramp && a.ty().IsScalar() &&
            tl::arith::CanProve(analyzer_, a > 0)) {
          PrimExpr lanes = b_ramp->lanes;
          return prim::Ramp(b_ramp->base * a, b_ramp->stride * a, lanes);
        }
        int a_lanes = GetLanesOrVScaleFactor(a.ty());
        int b_lanes = GetLanesOrVScaleFactor(b.ty());
        int max_lanes = std::max(a_lanes, b_lanes);
        bool is_scalable =
            a.ty().IsScalableVector() || b.ty().IsScalableVector();
        return prim::Mul(BroadcastTo(a, max_lanes, is_scalable),
                         BroadcastTo(b, max_lanes, is_scalable));
      }
    }
    return BinaryVec<prim::Mul>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::DivNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::Div>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::ModNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::Mod>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::FloorDiv>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::FloorMod>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::MinNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::Min>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::Max>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::EQ>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::NENode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::NE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LTNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::LT>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LENode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::LE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GTNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::GT>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GENode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::GE>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::AndNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::And>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::OrNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::Or>(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LShiftNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::LShift>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::RShiftNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::RShift>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseAndNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseAnd>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseOrNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseOr>(op, inplace_mode);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseXorNode *op,
                                InplaceMode inplace_mode) final {
    return BinaryVec<prim::BitwiseXor>(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BitwiseNotNode *op,
                                InplaceMode inplace_mode) final {
    auto a = this->Mutate(op->a, inplace_mode);
    if (a.UnchangedOrSameAs(op->a))
      return ffi::Unchanged();
    return prim::BitwiseNot(std::move(a).ValueOrUnchanged(op->a), op->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NotNode *op,
                                InplaceMode inplace_mode) final {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    if (a_unchanged) {
      return ffi::Unchanged();
    } else {
      return !(a);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::RampNode *op,
                                InplaceMode inplace_mode) final {
    PrimExpr base =
        this->Mutate(op->base, inplace_mode).ValueOrUnchanged(op->base);
    PrimExpr stride =
        this->Mutate(op->stride, inplace_mode).ValueOrUnchanged(op->stride);
    TVM_FFI_ICHECK(!base.ty().IsScalableVector())
        << "Creating scalable vectors from existing vectors is not supported.";
    TVM_FFI_ICHECK(!stride.ty().IsScalableVector())
        << "Ramp stride with scalable dtype is not supported";
    if (base.ty().IsFixedLengthVector() && stride.ty().IsScalar()) {
      TVM_FFI_ICHECK(op->lanes->IsInstance<IntImmNode>())
          << "Vectorizing over existing scalable vectors is not supported.";
      const prim::RampNode *base_ramp = base.as<prim::RampNode>();
      int op_lanes = op->lanes.as_or_throw<IntImm>()->value.as<int>().value();
      int base_ramp_lanes =
          base_ramp->lanes.as_or_throw<IntImm>()->value.as<int>().value();
      if (tl::arith::CanProve(
              analyzer_,
              base_ramp->stride ==
                  stride * tl::MakeConst(stride.ty(), base_ramp_lanes))) {
        return prim::Ramp(base_ramp->base, stride, op_lanes * base_ramp_lanes);
      }
    }
    int lanes = std::max(base.ty().lanes(), stride.ty().lanes());
    base = BroadcastTo(base, lanes, false);
    stride = BroadcastTo(stride, lanes, false);
    ffi::Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(prim::Ramp(prim::Shuffle::ExtractElement(base, i),
                                 prim::Shuffle::ExtractElement(stride, i),
                                 op->lanes));
    }
    return prim::Shuffle::Concat(elems);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode *op,
                                InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);
    if (value.ty().IsScalableVector() || value.ty().IsFixedLengthVector()) {
      need_scalarize_ = true;
      return ffi::Unchanged();
    }
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      return prim::Broadcast(op->value, op->lanes);
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode *op,
                                InplaceMode inplace_mode) final {
    auto cond_update = this->Mutate(op->condition, inplace_mode);
    bool cond_unchanged = cond_update.UnchangedOrSameAs(op->condition);
    PrimExpr cond = std::move(cond_update).ValueOrUnchanged(op->condition);
    auto t_update = this->Mutate(op->true_value, inplace_mode);
    bool t_unchanged = t_update.UnchangedOrSameAs(op->true_value);
    PrimExpr t = std::move(t_update).ValueOrUnchanged(op->true_value);
    auto f_update = this->Mutate(op->false_value, inplace_mode);
    bool f_unchanged = f_update.UnchangedOrSameAs(op->false_value);
    PrimExpr f = std::move(f_update).ValueOrUnchanged(op->false_value);
    if (cond_unchanged && t_unchanged && f_unchanged) {
      return ffi::Unchanged();
    } else {
      int cond_lanes = GetLanesOrVScaleFactor(cond.ty());
      int t_lanes = GetLanesOrVScaleFactor(t.ty());
      int f_lanes = GetLanesOrVScaleFactor(f.ty());
      int lanes = std::max(std::max(cond_lanes, t_lanes), f_lanes);
      bool is_scalable = cond.ty().IsScalableVector() ||
                         t.ty().IsScalableVector() || f.ty().IsScalableVector();
      return prim::Select(BroadcastTo(cond, lanes, is_scalable),
                          BroadcastTo(t, lanes, is_scalable),
                          BroadcastTo(f, lanes, is_scalable));
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode *op,
                                InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      if (value.ty().IsScalableVector()) {
        return prim::Cast(
            PrimType::ScalableVector(op->ty.as_or_throw<PrimType>().code(),
                                     op->ty.as_or_throw<PrimType>().bits(),
                                     value.ty().VScaleFactor()),
            value);
      } else {
        return prim::Cast(
            op->ty.as_or_throw<PrimType>().WithLanes(value.ty().lanes()),
            value);
      }
    }
  }

  PrimExpr MutateIfThenElseExpr_(const CallNode *op, InplaceMode inplace_mode) {
    PrimExpr cond = this->Mutate(op->args[0].as_or_throw<PrimExpr>())
                        .ValueOrUnchanged(op->args[0].as_or_throw<PrimExpr>());
    if (cond.ty().IsScalableVector() || cond.ty().IsFixedLengthVector()) {
      need_scalarize_ = true;
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    }
    PrimExpr t = this->Mutate(op->args[1].as_or_throw<PrimExpr>())
                     .ValueOrUnchanged(op->args[1].as_or_throw<PrimExpr>());
    PrimExpr f = this->Mutate(op->args[2].as_or_throw<PrimExpr>())
                     .ValueOrUnchanged(op->args[2].as_or_throw<PrimExpr>());
    if (cond.same_as(op->args[0]) && t.same_as(op->args[1]) &&
        f.same_as(op->args[2])) {
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    } else {
      int t_lanes = GetLanesOrVScaleFactor(t.ty());
      int f_lanes = GetLanesOrVScaleFactor(f.ty());
      int lanes = std::max(t_lanes, f_lanes);
      bool is_scalable = t.ty().IsScalableVector() || f.ty().IsScalableVector();
      t = BroadcastTo(t, lanes, is_scalable);
      f = BroadcastTo(f, lanes, is_scalable);
      PrimType op_ty = op->ty.as_or_throw<PrimType>();
      if (is_scalable) {
        return Call(PrimType::ScalableVector(op_ty.code(), op_ty.bits(), lanes),
                    op->op, {cond, t, f}, op->attrs, {}, op->loc)
            .as_or_throw<PrimExpr>();
      } else {
        return Call(op_ty.WithLanes(lanes), op->op, {cond, t, f}, op->attrs, {},
                    op->loc)
            .as_or_throw<PrimExpr>();
      }
    }
  }

  PrimExpr MutateReinterpretExpr_(const CallNode *op,
                                  InplaceMode inplace_mode) {
    TVM_FFI_ICHECK(op->op.same_as(tirx::reinterpret_op()));
    PrimExpr input = op->args[0].as_or_throw<PrimExpr>();
    PrimExpr value = this->Mutate(input).ValueOrUnchanged(input);
    if (value.same_as(op->args[0])) {
      return ffi::GetRef<Call>(op).as_or_throw<PrimExpr>();
    } else {
      int lanes = GetLanesOrVScaleFactor(value.ty());
      PrimType op_ty = op->ty.as_or_throw<PrimType>();
      if (value.ty().IsScalableVector()) {
        return Call(PrimType::ScalableVector(op_ty.code(), op_ty.bits(), lanes),
                    op->op, {value}, op->attrs, {}, op->loc)
            .as_or_throw<PrimExpr>();
      } else {
        int new_lanes =
            (op_ty.code() != DLDataTypeCode::kDLFloat4_e2m1fn &&
             input.ty().code() != DLDataTypeCode::kDLFloat4_e2m1fn)
                ? (value.ty().bits() * value.ty().lanes()) / op_ty.bits()
                : value.ty().lanes();
        return Call(op_ty.WithLanes(new_lanes), op->op, {value}, op->attrs, {},
                    op->loc)
            .as_or_throw<PrimExpr>();
      }
    }
  }

  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    if (op->op.same_as(prim::assume_op())) {
      PrimExpr predicate = op->args[0].as_or_throw<PrimExpr>();
      if (SideEffect(predicate) > CallEffectKind::kPure) {
        need_scalarize_ = true;
        return ffi::Unchanged();
      }
      predicate = Mutate(predicate, inplace_mode).ValueOrUnchanged(predicate);
      if (predicate.ty().IsScalableVector()) {
        need_scalarize_ = true;
        return ffi::Unchanged();
      }
      if (!predicate.ty().IsScalar()) {
        PrimExpr conjunction = prim::Shuffle::ExtractElement(predicate, 0);
        for (int lane = 1; lane < predicate.ty().lanes(); ++lane) {
          conjunction =
              conjunction && prim::Shuffle::ExtractElement(predicate, lane);
        }
        predicate = conjunction;
      }
      return Call(op->ty, op->op, {predicate}, op->attrs, op->ty_args, op->loc);
    }
    auto rule = Op::HasAttrMap(kVectorizeCall)
                    ? Op::GetAttrMap(kVectorizeCall).get(op->op, nullptr)
                    : nullptr;
    if (rule != nullptr) {
      ffi::TypedFunction<Expr(Expr)> rewrite(
          [self = ffi::GetObjectPtr<Vectorizer>(this)](Expr operand) {
            return self->Mutate(operand).ValueOrUnchanged(operand);
          });
      auto result = rule(ffi::GetRef<Call>(op), var_.as_or_throw<PrimVar>(),
                         var_lanes_, rewrite, target_);
      if (!result) {
        need_scalarize_ = true;
        return ffi::Unchanged();
      }
      return result.value();
    }
    auto optional_ret_ty = op->ty.as<PrimType>();
    if (!optional_ret_ty) {
      if (Op::GetAttrMap(tvm::op_attr::kCallEffectKind)
              .get(op->op, CallEffectKind::kOpaque) > CallEffectKind::kPure) {
        need_scalarize_ = true;
        return ffi::Unchanged();
      }

      ffi::Array<Expr> new_args;
      for (const Expr &arg : op->args) {
        Expr new_arg = this->Mutate(arg).ValueOrUnchanged(arg);
        if (auto prim_arg = new_arg.as<PrimExpr>();
            prim_arg && (prim_arg.value().ty().IsScalableVector() ||
                         prim_arg.value().ty().IsFixedLengthVector())) {
          need_scalarize_ = true;
          return ffi::Unchanged();
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return ffi::Unchanged();
      }
      return Call(op->ty, op->op, new_args, op->attrs, op->ty_args, op->loc);
    }
    PrimType ret_ty = optional_ret_ty.value();
    if (op->op.same_as(prim::if_then_else_op())) {
      return MutateIfThenElseExpr_(op, inplace_mode);
    } else if (op->op.same_as(backend::opencl::texture2d_load_op())) {
      int lane = 0;
      ffi::Array<PrimExpr> fcd = MutateArray(
          {op->args.back().as_or_throw<PrimExpr>()}, &lane, inplace_mode);
      PrimType dtype = GetTextureElementType(op->args[0]);
      TVM_FFI_ICHECK(lane * dtype.bits() <= op->args[4].as<IntImmNode>()->value)
          << "Expected Data to be Read is lesser than or equal to Texture Load "
             "length";

      auto new_args = op->args;
      new_args.pop_back();
      new_args.push_back(fcd[0]);
      return Call(ret_ty.WithLanes(lane), op->op, new_args, op->attrs, {},
                  op->loc);
    } else if (op->op.same_as(backend::opencl::texture2d_store_op())) {
      int lane = 0;

      ffi::Array<PrimExpr> value{op->args.back().as_or_throw<PrimExpr>()};
      ffi::Array<PrimExpr> mutated_value =
          MutateArray(value, &lane, inplace_mode);
      PrimType dtype = GetTextureElementType(op->args[0]);
      TVM_FFI_ICHECK(lane * dtype.bits() == op->args[4].as<IntImmNode>()->value)
          << "Expected Data to be Written equal to Texture Store length";
      ffi::Array<Expr> new_args = op->args;
      new_args.Set(new_args.size() - 1, mutated_value[0]);
      return Call(ret_ty, op->op, new_args, op->attrs, {}, op->loc);
    } else if (op->op.same_as(tirx::reinterpret_op())) {
      return MutateReinterpretExpr_(op, inplace_mode);
    }
    auto optional_op = op->op.as<Op>();
    bool vectorizable = optional_op &&
                        op_vectorizable_.get(optional_op.value(), false) &&
                        !ret_ty.IsScalableVector();

    if (!vectorizable) {
      if (Op::GetAttrMap(tvm::op_attr::kCallEffectKind)
              .get(op->op, CallEffectKind::kOpaque) > CallEffectKind::kPure) {
        need_scalarize_ = true;
        return ffi::Unchanged();
      }

      ffi::Array<Expr> new_args;
      for (const Expr &arg : op->args) {
        Expr new_arg = this->Mutate(arg).ValueOrUnchanged(arg);
        if (auto prim_arg = new_arg.as<PrimExpr>();
            prim_arg && (prim_arg.value().ty().IsScalableVector() ||
                         prim_arg.value().ty().IsFixedLengthVector())) {
          need_scalarize_ = true;
          return ffi::Unchanged();
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return ffi::Unchanged();
      } else {
        return Call(ret_ty, op->op, new_args, op->attrs, {}, op->loc);
      }
    } else {
      int lane = 0;
      ffi::Array<Expr> new_args;
      if (op->op.same_as(tirx::call_llvm_pure_intrin_op())) {

        ffi::Array<Expr> op_expr_args;
        for (size_t i = 1; i < op->args.size(); ++i) {

          op_expr_args.push_back(op->args[i]);
        }

        ffi::Array<Expr> updated_args =
            MutateCallArgs(op_expr_args, &lane, inplace_mode);
        new_args.push_back(op->args[0]);

        for (size_t i = 0; i < updated_args.size(); ++i) {
          new_args.push_back(updated_args[i]);
        }
      } else {
        new_args = MutateCallArgs(op->args, &lane, inplace_mode);
      }

      if (op->args.same_as(new_args)) {
        return ffi::Unchanged();
      } else {
        return Call(ret_ty.WithLanes(lane), op->op, new_args, op->attrs, {},
                    op->loc);
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *op,
                                InplaceMode inplace_mode) final {
    auto load = ffi::GetRef<TensorLoad>(op);

    auto fmutate = [this](const PrimExpr &index) {
      return this->Mutate(index).ValueOrUnchanged(index);
    };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    if (!indices.same_as(op->indices)) {
      return TensorLoad(op->source.as_or_throw<tvm::tirx::TensorVar>(), indices,
                        op->loc);
    }

    return load;
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode *op,
                                InplaceMode inplace_mode) final {
    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);

    auto mapped = VarRemapGet(op->var);
    if (mapped != nullptr) {
      TVM_FFI_ICHECK(deep_equal_(mapped.as_or_throw<PrimExpr>(), value))
          << "Let cannot bind the same var to two different values";
    }
    if (GetLanesOrVScaleFactor(value.ty()) !=
        GetLanesOrVScaleFactor(op->value.ty())) {
      Var new_var(op->var->name, value.ty());
      VarRemapSet(op->var, new_var);
      return prim::Let(
          new_var, value,
          this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body));
    } else {
      VarRemapSet(op->var, op->var);
      auto body_update = this->Mutate(op->body, inplace_mode);
      bool body_unchanged = body_update.UnchangedOrSameAs(op->body);
      PrimExpr body = std::move(body_update).ValueOrUnchanged(op->body);
      if (value_unchanged && body_unchanged) {
        return ffi::Unchanged();
      } else {
        return prim::Let(op->var, value, body);
      }
    }
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode *op,
                                InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(op->vectors.size() == 1 && op->indices.size() == 1)
        << "Cannot vectorize ShuffleNode with multiple vectors or indices: the "
           "vector size is "
        << op->vectors.size() << " and the index size is "
        << op->indices.size();
    int lane_vectors = 0;
    int lane_indices = 0;
    ffi::Array<PrimExpr> vectors =
        MutateArray(op->vectors, &lane_vectors, inplace_mode);
    ffi::Array<PrimExpr> indices =
        MutateArray(op->indices, &lane_indices, inplace_mode);
    if (vectors.same_as(op->vectors) && indices.same_as(op->indices)) {
      return ffi::Unchanged();
    }

    int new_vec_length =
        var_lanes_.as_or_throw<IntImm>()->value.as<int>().value() /
        op->vectors[0].ty().lanes();
    PrimExpr updated_index = indices[0];

    auto f_check_index = [this, op](const PrimExpr &index) {
      if (const auto *ramp = index.as<prim::RampNode>()) {
        if (ramp->base->IsInstance<IntImmNode>() &&
            ramp->base.as_or_throw<IntImm>()->value == 0 &&
            ramp->stride->IsInstance<IntImmNode>() &&
            ramp->stride.as_or_throw<IntImm>()->value == 1 &&
            ramp->lanes->IsInstance<IntImmNode>() &&
            ramp->lanes.as_or_throw<IntImm>()->value ==
                var_lanes_.as_or_throw<IntImm>()->value) {
          return true;
        }
      }

      if (const auto *floordiv = index.as<prim::FloorModNode>()) {
        if (const auto *ramp = floordiv->a.as<prim::RampNode>()) {
          if (const auto *broadcast = floordiv->b.as<prim::BroadcastNode>()) {
            if (ramp->base->IsInstance<IntImmNode>() &&
                ramp->base.as_or_throw<IntImm>()->value == 0 &&
                ramp->stride->IsInstance<IntImmNode>() &&
                ramp->stride.as_or_throw<IntImm>()->value == 1 &&
                ramp->lanes->IsInstance<IntImmNode>() &&
                ramp->lanes.as_or_throw<IntImm>()->value ==
                    var_lanes_.as_or_throw<IntImm>()->value &&
                broadcast->value->IsInstance<IntImmNode>() &&
                broadcast->value.as_or_throw<IntImm>()->value ==
                    op->vectors[0].ty().lanes() &&
                broadcast->lanes->IsInstance<IntImmNode>() &&
                broadcast->lanes.as_or_throw<IntImm>()->value ==
                    var_lanes_.as_or_throw<IntImm>()->value) {
              return true;
            }
          }
        }
      }

      return false;
    };
    TVM_FFI_ICHECK(f_check_index(updated_index));

    if (new_vec_length == 1) {
      PrimType var_ty = var_->ty.as_or_throw<PrimType>();
      auto f_substitute =
          [old_var = var_, replacement = tvm::IntImm(var_ty, 0)](
              const Var &var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (var.same_as(old_var))
          return ffi::Any(replacement);
        return ffi::Unchanged();
      };
      return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(op->vectors[0],
                                                           f_substitute)
          .as_or_throw<PrimExpr>();
    } else {
      PrimExpr prev_ramp = ramp_;
      PrimExpr prev_var_lanes = var_lanes_;
      PrimType var_ty = var_->ty.as_or_throw<PrimType>();
      ramp_ = prim::Ramp(IntImm(var_ty, 0), IntImm(var_ty, 2), new_vec_length);
      var_lanes_ = tvm::IntImm(var_lanes_.ty(), new_vec_length);
      lane_vectors = 0;
      vectors = MutateArray(op->vectors, &lane_vectors, inplace_mode);
      ramp_ = prev_ramp;
      var_lanes_ = prev_var_lanes;
      return vectors[0];
    }
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) final {
    auto store = ffi::GetRef<TensorStore>(op);

    auto fmutate = [this](const PrimExpr &index) {
      return this->Mutate(index).ValueOrUnchanged(index);
    };
    ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);

    auto value_update = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_update.UnchangedOrSameAs(op->value);
    PrimExpr value = std::move(value_update).ValueOrUnchanged(op->value);

    if (!indices.same_as(op->indices) || !value_unchanged) {
      TVM_FFI_ICHECK(
          !op->dest.as_or_throw<TensorVar>().dtype().IsScalableVector())
          << "Vectorizing over scalable tensor elements is not supported in "
             "vectorizer.";

      int other_index_lanes = op->dest.as_or_throw<TensorVar>().dtype().lanes();
      for (size_t i = 0; i < indices.size() - 1; i++) {
        other_index_lanes *= indices[i].ty().lanes();

        TVM_FFI_ICHECK(!indices[i].ty().IsScalableVector())
            << "Only the last index can be scalable.";
      }

      PrimType last_index_dtype = indices[indices.size() - 1].ty();
      int lanes_in_last_index = GetLanesOrVScaleFactor(last_index_dtype);
      int index_lanes = other_index_lanes * lanes_in_last_index;

      int value_dtype_lanes = GetLanesOrVScaleFactor(value.ty());
      bool is_last_index_scalable = last_index_dtype.IsScalableVector();
      int total_lanes = std::max(index_lanes, value_dtype_lanes);

      TVM_FFI_ICHECK_EQ(total_lanes % other_index_lanes, 0)
          << "When storing to tensor "
          << op->dest.as_or_throw<TensorVar>()->name << ", cannot produce "
          << total_lanes
          << " lanes of storage location by changing the last index.";
      int last_index_lanes = total_lanes / other_index_lanes;

      indices.Set(indices.size() - 1,
                  BroadcastTo(indices[indices.size() - 1], last_index_lanes,
                              is_last_index_scalable));

      auto writer = store.CopyOnWrite();
      writer->indices = indices;
      writer->value = BroadcastTo(value, total_lanes, is_last_index_scalable);
    }

    return store;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    if (op->kind == ForKind::kVectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    TVM_FFI_ICHECK(IsZero(op->begin));
    TVM_FFI_ICHECK(!op->end.ty().IsScalableVector() &&
                   !op->end.ty().IsFixedLengthVector());
    auto end_update = this->Mutate(op->end, inplace_mode);
    bool end_unchanged = end_update.UnchangedOrSameAs(op->end);
    PrimExpr end = std::move(end_update).ValueOrUnchanged(op->end);
    PrimExpr step =
        this->Mutate(op->step, inplace_mode).ValueOrUnchanged(op->step);
    if (end.ty().IsScalableVector() || end.ty().IsFixedLengthVector() ||
        step.ty().IsScalableVector() || step.ty().IsFixedLengthVector()) {
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    auto body_update = this->Mutate(op->body, inplace_mode);
    bool body_unchanged = body_update.UnchangedOrSameAs(op->body);
    Stmt body = std::move(body_update).ValueOrUnchanged(op->body);
    if (end_unchanged && step.same_as(op->step) && body_unchanged) {
      return ffi::Unchanged();
    } else {
      auto n = ffi::make_object<ForNode>(*op);
      n->end = end;
      n->step = step;
      n->body = body;
      return For(n);
    }
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(!op->condition.ty().IsScalableVector() &&
                   !op->condition.ty().IsFixedLengthVector());
    auto condition_update = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged =
        condition_update.UnchangedOrSameAs(op->condition);
    PrimExpr condition =
        std::move(condition_update).ValueOrUnchanged(op->condition);

    bool cond_need_scalarize = false;
    std::swap(cond_need_scalarize, need_scalarize_);

    auto then_case_update = this->Mutate(op->then_case, inplace_mode);
    bool then_case_unchanged =
        then_case_update.UnchangedOrSameAs(op->then_case);
    Stmt then_case =
        std::move(then_case_update).ValueOrUnchanged(op->then_case);
    ffi::Optional<SeqStmt> else_case = std::nullopt;
    if (op->else_case) {
      else_case = this->Mutate(op->else_case.value(), inplace_mode)
                      .ValueOrUnchanged(op->else_case.value());
    }

    if (EnableTensorLevelPredication(target_) &&
        (condition.ty().IsScalableVector() ||
         condition.ty().IsFixedLengthVector()) &&
        !else_case.has_value()) {
      std::pair<bool, Stmt> success_stmt_pair =
          ffi::make_object<TryPredicateTensorAccesses>(TargetHasRVV(target_))
              ->Run(then_case, condition);
      bool can_remove_if_then_else = success_stmt_pair.first;
      if (can_remove_if_then_else) {
        return success_stmt_pair.second;
      }
    }

    if (cond_need_scalarize || condition.ty().IsScalableVector() ||
        condition.ty().IsFixedLengthVector()) {
      return Scalarize(ffi::GetRef<Stmt>(op));
    }
    if (condition_unchanged && then_case_unchanged &&
        else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      return If(condition, then_case, else_case);
    }
  }

  UnchangedOr<Stmt> Mutate_(const WhileNode *op,
                            InplaceMode inplace_mode) final {
    TVM_FFI_THROW(InternalError)
        << "A while loop inside a vectorized loop not supported.";
    TVM_FFI_UNREACHABLE();
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) final {
    auto prim_value = op->value.as<PrimExpr>();
    if (!prim_value) {
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }
    PrimExpr value = this->Mutate(prim_value.value(), inplace_mode)
                         .ValueOrUnchanged(prim_value.value());

    if (need_scalarize_) {
      need_scalarize_ = false;
      scalarize_binding_scope_ = true;
      return ffi::Unchanged();
    }
    TVM_FFI_ICHECK(VarRemapGet(op->var) == nullptr)
        << "SSA violation, a single var is binded twice";

    if (GetLanesOrVScaleFactor(value.ty()) !=
        GetLanesOrVScaleFactor(prim_value.value().ty())) {
      Var new_var(op->var->name, value.ty());
      VarRemapSet(op->var, new_var);
      return Bind(new_var, value);
    } else {
      VarRemapSet(op->var, op->var);
      if (value.same_as(op->value)) {
        return ffi::Unchanged();
      } else {
        return Bind(op->var, value, op->loc);
      }
    }
  }

  Stmt Scalarize(Stmt stmt) {
    PrimType var_ty = var_->ty.as_or_throw<PrimType>();
    PrimVar scalar_index(var_->name + ".s", var_ty);
    stmt = tl::SubstituteWithDataTypeLegalization(
        stmt, [&](const Var &variable) -> ffi::Optional<PrimExpr> {
          if (variable.same_as(var_))
            return scalar_index;
          return std::nullopt;
        });
    return For(scalar_index, IntImm(var_ty, 0), var_lanes_, IntImm(var_ty, 1),
               ForKind::kDefault, stmt);
  }

private:
  sym::Analyzer analyzer_;
  bool scalarize_binding_scope_{false};

  prim::ExprDeepEqual deep_equal_;

  Var var_;

  PrimExpr var_lanes_;

  PrimExpr ramp_{ffi::UnsafeInit{}};

  bool need_scalarize_{false};

  decltype(Op::GetAttrMap(tvm::tirx::op_attr::kVectorizable)) op_vectorizable_ =
      Op::GetAttrMap(tvm::tirx::op_attr::kVectorizable);

  ffi::Optional<Target> target_;

  ffi::Array<PrimExpr> MutateArray(const ffi::Array<PrimExpr> &arr,
                                   int *p_lanes, InplaceMode inplace_mode) {
    if (arr.size() == 0)
      return arr;
    int &lanes = *p_lanes;
    bool changed = false;
    std::vector<PrimExpr> new_arr;
    new_arr.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      PrimExpr old_elem = arr[i];
      auto new_elem_update = this->Mutate(old_elem);
      bool new_elem_unchanged = new_elem_update.UnchangedOrSameAs(old_elem);
      PrimExpr new_elem = std::move(new_elem_update).ValueOrUnchanged(old_elem);
      if (!new_elem_unchanged)
        changed = true;
      new_arr.push_back(new_elem);
      lanes = std::max(lanes, new_elem.ty().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].ty().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes, false);
        changed = true;
      }
    }
    if (!changed)
      return arr;
    return ffi::Array<PrimExpr>(new_arr);
  }

  ffi::Array<Expr> MutateCallArgs(const ffi::Array<Expr> &arr, int *p_lanes,
                                  InplaceMode inplace_mode) {
    if (arr.empty())
      return arr;
    int &lanes = *p_lanes;
    bool changed = false;
    std::vector<Expr> new_arr;
    new_arr.reserve(arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
      const Expr &old_elem = arr[i];
      auto new_elem_update = this->Mutate(old_elem);
      bool new_elem_unchanged = new_elem_update.UnchangedOrSameAs(old_elem);
      Expr new_elem = std::move(new_elem_update).ValueOrUnchanged(old_elem);
      changed = changed || !new_elem_unchanged;
      new_arr.push_back(new_elem);
      if (auto prim_elem = new_elem.as<PrimExpr>()) {
        lanes = std::max(lanes, prim_elem.value().ty().lanes());
      }
    }
    for (size_t i = 0; i < new_arr.size(); ++i) {
      if (auto prim_elem = new_arr[i].as<PrimExpr>();
          prim_elem && prim_elem.value().ty().lanes() != lanes) {
        new_arr[i] = BroadcastTo(prim_elem.value(), lanes, false);
        changed = true;
      }
    }
    if (!changed)
      return arr;
    return ffi::Array<Expr>(new_arr);
  }
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T *op, InplaceMode inplace_mode) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value,
                  "constraint");
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return TOp(BroadcastTo(a, lanes, is_scalable),
                 BroadcastTo(b, lanes, is_scalable), op->loc);
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T *op, FCompute fcompute, InplaceMode inplace_mode) {
    auto a_update = this->Mutate(op->a, inplace_mode);
    bool a_unchanged = a_update.UnchangedOrSameAs(op->a);
    PrimExpr a = std::move(a_update).ValueOrUnchanged(op->a);
    auto b_update = this->Mutate(op->b, inplace_mode);
    bool b_unchanged = b_update.UnchangedOrSameAs(op->b);
    PrimExpr b = std::move(b_update).ValueOrUnchanged(op->b);
    if (a_unchanged && b_unchanged) {
      return ffi::GetRef<PrimExpr>(op);
    } else {
      int a_lanes = GetLanesOrVScaleFactor(a.ty());
      int b_lanes = GetLanesOrVScaleFactor(b.ty());
      int lanes = std::max(a_lanes, b_lanes);
      if (lanes != 1) {
        const prim::RampNode *b_ramp = b.as<prim::RampNode>();
        const prim::RampNode *a_ramp = a.as<prim::RampNode>();
        if (a.ty().IsScalar() && b_ramp) {
          return prim::Ramp(
              fcompute(a, b_ramp->base),
              fcompute(IntImm(b_ramp->stride.ty(), 0), b_ramp->stride),
              b_ramp->lanes);
        }
        if (b.ty().IsScalar() && a_ramp) {
          return prim::Ramp(fcompute(a_ramp->base, b), a_ramp->stride,
                            a_ramp->lanes);
        }
      }
      bool is_scalable = a.ty().IsScalableVector() || b.ty().IsScalableVector();
      return fcompute(BroadcastTo(a, lanes, is_scalable),
                      BroadcastTo(b, lanes, is_scalable));
    }
  }
};

Stmt PreserveUnrollPolicy(Stmt body,
                          const ffi::Map<ffi::String, ffi::Any> &annotations) {
  struct PolicyTransfer : tvm::tl::StmtExprMutator {
    using tvm::tl::StmtExprMutator::Mutate;
    ffi::Map<ffi::String, ffi::Any> policy;

    UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                                 InplaceMode inplace_mode) final {
      if (value.as<ExprNode>())
        return ffi::Unchanged();
      return tvm::tl::StmtExprMutator::Mutate(value, inplace_mode);
    }

    UnchangedOr<Stmt> Mutate_(const ForNode *op,
                              InplaceMode inplace_mode) final {
      auto annotations = op->annotations;
      for (const auto &[key, value] : policy) {
        auto existing = annotations.Get(key);
        if (!existing.has_value() || existing.value() == nullptr) {
          annotations.Set(key, value);
        }
      }
      if (annotations.same_as(op->annotations))
        return ffi::Unchanged();
      auto copy = ffi::make_object<ForNode>(*op);
      copy->annotations = std::move(annotations);
      return For(copy);
    }
  };
  auto transfer = ffi::make_object<PolicyTransfer>();
  for (const char *key : {tirx::attr::kAutoUnrollMaxStep.name,
                          tirx::attr::kUnrollExplicit.name}) {
    if (auto value = annotations.Get(key);
        value.has_value() && value.value() != nullptr) {
      transfer->policy.Set(key, value.value());
    }
  }
  if (transfer->policy.empty())
    return body;
  return transfer->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
}

class LoopVectorizer : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                               InplaceMode inplace_mode) final {
    if (value.as<ExprNode>())
      return ffi::Unchanged();
    return tvm::tl::StmtExprMutator::Mutate(value, inplace_mode);
  }

  explicit LoopVectorizer(DictAttrs attrs)
      : target_(attrs.GetAttr(tvm::attr::kTarget)) {}

  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    if (op->kind == ForKind::kVectorized) {
      PrimExpr extent = Range::GetExtentFromBeginEnd(op->begin, op->end);
      auto *extent_as_int = extent.as<IntImmNode>();

      TVM_FFI_ICHECK(IsZero(op->begin));

      if (extent_as_int && extent_as_int->value > 1 && TargetHasRVV(target_) &&
          !ContainsCallNode(op->body)) {
        return VectorizeFixedLoopForRVV(op,
                                        extent_as_int->value.as<int>().value());
      }

      if (!extent_as_int || extent_as_int->value < 1) {
        struct VScaleFinder : tvm::tl::StmtExprVisitor {
          ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
            if (auto expr = value.as<PrimExpr>(); expr && IsVScaleCall(*expr)) {
              return VisitInterrupt();
            }
            return tvm::tl::StmtExprVisitor::Visit(value);
          }
        };
        bool is_scalable_expr =
            ffi::make_object<VScaleFinder>()->Visit(extent).has_value();
        TVM_FFI_ICHECK(is_scalable_expr && TargetHasVLA(target_))
            << "Failed to vectorize loop with extent " << extent
            << " for target " << target_;
      }
      Stmt body = PreserveUnrollPolicy(op->body, op->annotations);
      return ffi::make_object<Vectorizer>(op->var, extent, target_)
          ->Run(body, inplace_mode);
    } else {
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

private:
  Stmt VectorizeFixedLoopForRVV(const ForNode *op, int64_t extent) {

    static constexpr int kDefaultVScaleFactor = 4;
    PrimType index_dtype = op->var.ty();
    PrimExpr zero = IntImm(index_dtype, 0);
    PrimExpr fixed_extent = IntImm(index_dtype, extent);
    PrimExpr scalable_lanes = CreateNewLanes(true, kDefaultVScaleFactor);
    PrimType lane_dtype = scalable_lanes.ty();
    PrimExpr scalable_lanes_index = scalable_lanes;
    if (scalable_lanes_index.ty() != index_dtype) {
      scalable_lanes_index = prim::Cast(index_dtype, scalable_lanes_index);
    }
    PrimExpr num_chunks = ceildiv(fixed_extent, scalable_lanes_index);

    PrimVar outer(op->var->name + ".vla.o", index_dtype);
    PrimVar inner(op->var->name + ".vla.i", lane_dtype);
    PrimExpr inner_index = inner;
    if (inner_index.ty() != index_dtype) {
      inner_index = prim::Cast(index_dtype, inner_index);
    }
    PrimExpr index = outer * scalable_lanes_index + inner_index;
    auto substituter = ffi::make_object<tvm::tl::StmtExprMutator>();
    substituter->VarRemapSet(op->var, index);
    Stmt body = substituter->Mutate(op->body).ValueOrUnchanged(op->body);
    Stmt guarded_body = If(index < fixed_extent, body, std::nullopt, op->loc);
    Stmt vector_loop =
        For(inner, IntImm(lane_dtype, 0), scalable_lanes, IntImm(lane_dtype, 1),
            ForKind::kVectorized, guarded_body, op->annotations, op->loc);
    Stmt loop = For(outer, zero, num_chunks, IntImm(index_dtype, 1),
                    ForKind::kDefault, vector_loop, {}, op->loc);

    return this->Mutate(loop, InplaceMode::kDisallow).ValueOrUnchanged(loop);
  }

  ffi::Optional<Target> target_;
};

class VectorizeSkipper : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                               InplaceMode inplace_mode) final {
    if (value.as<ExprNode>())
      return ffi::Unchanged();
    return tvm::tl::StmtExprMutator::Mutate(value, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    Stmt stmt = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<ForNode>();
    if (op->kind == ForKind::kVectorized) {
      auto copy = ffi::make_object<ForNode>(*op);
      copy->kind = ForKind::kDefault;
      return For(copy);
    } else {
      return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) {
  return ffi::make_object<VectorizeSkipper>()
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(std::move(stmt));
}

} // namespace

Pass VectorizeLoop(bool enable_vectorize) {
  if (!enable_vectorize)
    return tirx::transform::VectorizeLoop(false);
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    auto *n = f.CopyOnWrite();
    if (enable_vectorize) {
      n->body = ffi::make_object<LoopVectorizer>(n->attrs)
                    ->Mutate(n->body, InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body);
    } else {
      n->body = ffi::make_object<VectorizeSkipper>()
                    ->Mutate(n->body, InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body);
    }
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tl.VectorizeLoop");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.VectorizeLoop",
                                   VectorizeLoop);
}

} // namespace tvm::tl::tirx_passes
