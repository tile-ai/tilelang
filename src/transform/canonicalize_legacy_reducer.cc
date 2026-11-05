/*!
 * \file canonicalize_legacy_reducer.cc
 * \brief Rewrite legacy (v1) reducer syntax into first-class reducer v2 ops.
 *
 * Legacy form (deprecated):
 *   acc = T.alloc_reducer(shape, dtype, op, replication="all"/"none")
 *   T.clear(acc)                      # or T.fill(acc, value)
 *   acc[i] += v                       # sum; T.max/T.min RMW for max/min
 *   T.finalize_reducer(acc)           # in-place
 *   ... reads of acc ...
 *
 * The same legacy syntax is also accepted on a v2 allocation
 * (T.alloc_reducer without `replication`, the pre-v2 default): a
 * `local.reducer` buffer that is never touched by any first-class v2 op
 * is canonicalized under the same whitelist, keeping code written against
 * the old default compiling. Mixing v2 ops and legacy syntax on one buffer
 * is not canonicalized and fails in VerifyReducerEpoch.
 *
 * Canonical v2 form produced by this pass:
 *   local.reducer allocation + reducer_info_v2 annotation
 *   tl.reducer_init(acc)
 *   tl.reducer_update(acc[i], v)
 *   tl.finalize_reducer_v2(acc, dst)  # fresh dst fragment
 *   ... reads redirected to dst ...
 *
 * Mapping rules (strict whitelist — anything else is a compile error, never
 * silently accepted):
 *   - the fill value must be the combine identity (plain init), except for
 *     idempotent combines (max/min) where a non-identity fill is equivalent
 *     to a one-time seed and is forwarded as such. A non-zero sum fill is
 *     rejected: its v1 behavior was participant-count-dependent.
 *   - an update store must be exactly `combine(acc[idx], value)` with the
 *     declared combine op and identical load/store indices.
 *   - `T.finalize_reducer(acc)`'s batch annotation is forwarded.
 *   - accesses to the reducer after finalize read the new destination
 *     fragment; writes after finalize are rejected.
 *
 * This pass runs before VerifyReducerEpoch, so converted programs are then
 * verified under the same rules as native v2 code. It is a deprecation shim:
 * once legacy syntax is removed, this file is deleted with it.
 */

#include "common/native_reducer_utils.h"
#include "op/fill.h"

#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace prim;
using namespace ffi;
using namespace reducer_native;

std::optional<double> TryConstValue(PrimExpr value) {
  while (const auto *cast = value.as<CastObj>())
    value = cast->value;
  if (const auto *integer = value.as<IntImmObj>()) {
    if (auto number = integer->value.as<int64_t>())
      return static_cast<double>(*number);
  }
  if (const auto *real = value.as<FloatImmObj>())
    return real->value;
  return std::nullopt;
}

class LegacyReducerCanonicalizer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  tirx::Function Run(tirx::Function function) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto metadata = MetadataIndex::Collect(function);
    std::unordered_set<const VarObj *> used_v2;
    VisitNativePostOrder(function->body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallObj>()) {
        if (call->op.same_as(ReducerInitOp::Get()) ||
            call->op.same_as(FinalizeReducerV2Op::Get())) {
          auto args = DecodeTileOpArguments(call).operands;
          TVM_FFI_CHECK(!args.empty(), ValueError);
          used_v2.insert(reducer_native::RegionTensor(args[0]).get());
        } else if (call->op.same_as(reducer_update())) {
          used_v2.insert(ParseReducerUpdate(call).reducer.get());
        }
      }
    });
    for (const auto &[tensor, allocation] : metadata.allocations) {
      auto info = metadata.legacy.Get(tensor);
      const bool already_v2 = !info.has_value();
      if (already_v2) {
        if (used_v2.count(tensor.get()))
          continue;
        info = metadata.v2.Get(tensor);
      }
      if (!info)
        continue;
      auto op = info->Get("op");
      TVM_FFI_CHECK(op.has_value(), ValueError)
          << "reducer metadata for " << tensor << " is missing the combine op";
      LegacyReducer entry;
      entry.old_tensor = tensor;
      entry.acc = already_v2 ? tensor : WithScope(tensor, "local.reducer");
      entry.dst = WithScope(tensor, "local.fragment", "_result");
      entry.op_name = op->as_or_throw<String>();
      entry.op = ParseReducerV2OpType(entry.op_name);
      TVM_FFI_CHECK(entry.op == ReducerV2OpType::kSum ||
                        entry.op == ReducerV2OpType::kMax ||
                        entry.op == ReducerV2OpType::kMin,
                    ValueError)
          << "legacy reducers only support sum/max/min; use the first-class v2 "
             "API";
      reducers_.emplace(tensor.get(), std::move(entry));
    }
    if (reducers_.empty())
      return function;
    SeqStmt body = Mutate(function->body).ValueOrUnchanged(function->body);
    for (const auto &[var, entry] : reducers_)
      TVM_FFI_CHECK(entry.phase == Phase::kFinalized, ValueError)
          << "legacy reducer " << entry.old_tensor
          << " has no T.finalize_reducer; cannot canonicalize";
    auto rewrite = make_object<MetadataRewriter>([this](Metadata value) {
      InfoMap v2;
      if (auto old = value.Get(attr::kReducerInfoV2))
        v2 = ReadInfo(*old);
      if (auto old = value.Get(attr::kReducerInfo)) {
        for (const auto &[tensor, info] : ReadInfo(*old)) {
          auto found = reducers_.find(tensor.get());
          TVM_FFI_CHECK(found != reducers_.end(), ValueError)
              << "reducer_info must refer to an allocated tensor";
          v2.Set(found->second.acc, {{"op", found->second.op_name}});
        }
        value.erase(attr::kReducerInfo);
      }
      if (!v2.empty())
        value.Set(attr::kReducerInfoV2, v2);
      if (auto old = value.Get(attr::kLayoutMap)) {
        Map<TensorVar, Layout> layouts;
        for (const auto &[tensor, layout] : ReadLayouts(*old)) {
          auto found = reducers_.find(tensor.get());
          layouts.Set(found == reducers_.end() ? tensor : found->second.acc,
                      layout);
        }
        value.Set(attr::kLayoutMap, layouts);
      }
      return value;
    });
    body = rewrite->Mutate(body).ValueOrUnchanged(body);
    function.CopyOnWrite()->body = body;
    return function;
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto found = reducers_.find(op->var.get());
    if (found == reducers_.end())
      return StmtExprMutator::Mutate_(op, mode);
    const auto *call = op->value.as<CallObj>();
    TVM_FFI_CHECK(call && call->op.same_as(alloc_tensor_op()), ValueError)
        << "legacy reducer handle must come from an allocation";
    const auto &entry = found->second;
    return SeqStmt({Allocate(entry.acc, op->loc, call->attrs),
                    Allocate(entry.dst, op->loc, call->attrs)},
                   op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!call)
      return StmtExprMutator::Mutate_(op, mode);
    // Metadata keys are remapped after statement processing, not interpreted
    // as executable reads of an opaque reducer handle.
    if (IsMetadataCall(call))
      return Unchanged();
    if (call->op.same_as(Fill::Get())) {
      auto decoded = DecodeTileOpArguments(call);
      TVM_FFI_CHECK_EQ(decoded.operands.size(), 2U, ValueError);
      if (auto *entry = FindReducer(decoded.operands[0])) {
        TVM_FFI_CHECK(entry->phase == Phase::kAllocated, ValueError)
            << "second T.clear/T.fill on legacy reducer " << entry->old_tensor
            << ": one allocation supports exactly one reduction epoch";
        PrimType dtype = entry->old_tensor.dtype();
        PrimExpr value = decoded.operands[1].as_or_throw<PrimExpr>();
        auto fill = TryConstValue(value);
        auto identity = TryConstValue(ReducerV2Identity(entry->op, dtype));
        const bool is_identity = fill && identity && *fill == *identity;
        Array<Expr> args{FullRegion(entry->acc)};
        if (!is_identity) {
          TVM_FFI_CHECK(entry->op != ReducerV2OpType::kSum, ValueError)
              << "legacy sum reducer is filled with a non-zero value; its v1 "
                 "behavior depended on the physical thread count. Use "
                 "T.reducer_init(acc, value)";
          value = Mutate(value, mode).ValueOrUnchanged(value);
          args.push_back(value.ty() == dtype ? value
                                             : prim::cast(value, dtype));
        }
        entry->phase = Phase::kActive;
        return TileCall(ReducerInitOp::Get(), args, {}, op->loc);
      }
    }
    if (call->op.same_as(FinalizeReducerOp::Get())) {
      auto decoded = DecodeTileOpArguments(call);
      if (decoded.operands.size() == 1) {
        if (auto *entry = FindReducer(decoded.operands[0])) {
          TVM_FFI_CHECK(entry->phase == Phase::kActive, ValueError)
              << "T.finalize_reducer on legacy reducer without a preceding "
                 "T.clear/T.fill";
          entry->phase = Phase::kFinalized;
          return TileCall(FinalizeReducerV2Op::Get(),
                          {FullRegion(entry->acc), FullRegion(entry->dst)},
                          decoded.annotations, op->loc);
        }
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (op->op->name == "tl.attr") {
      auto fields = ReadNativeAnnotations(op);
      const auto key = fields.at("key").as_or_throw<String>();
      if (key == "tl.kernel" || IsReducerMetadataKey(key)) {
        auto copy = make_object<RegionStmtObj>(*op);
        copy->body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
        return Stmt(copy);
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    auto tensor = op->dest.as_or_throw<TensorVar>();
    auto found = reducers_.find(tensor.get());
    if (found == reducers_.end())
      return StmtExprMutator::Mutate_(op, mode);
    auto &entry = found->second;
    TVM_FFI_CHECK(entry.phase == Phase::kActive, ValueError)
        << "store to legacy reducer " << tensor
        << (entry.phase == Phase::kFinalized ? " after T.finalize_reducer"
                                             : " before T.clear/T.fill");
    auto contribution = MatchUpdate(entry, op);
    TVM_FFI_CHECK(contribution.has_value(), ValueError)
        << "unsupported store to legacy reducer: expected acc[i] += v or a "
           "matching T.max/T.min update";
    PrimExpr value =
        Mutate(*contribution, mode).ValueOrUnchanged(*contribution);
    Array<PrimExpr> indices;
    for (const auto &index : op->indices)
      indices.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    // reducer_update is a two-operand intrinsic, not a tile operator with a
    // trailing annotation tuple. Keep the update descriptor as a TensorLoad.
    return Evaluate(Call(PrimType::Void(), reducer_update(),
                         {TensorLoad(entry.acc, indices, op->loc), value},
                         std::nullopt, {}, op->loc),
                    op->loc);
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    auto found = reducers_.find(tensor.get());
    if (found == reducers_.end())
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_CHECK(found->second.phase == Phase::kFinalized, ValueError)
        << "read of legacy reducer before T.finalize_reducer; partial values "
           "are not readable";
    Array<PrimExpr> indices;
    for (const auto &index : op->indices)
      indices.push_back(Mutate(index, mode).ValueOrUnchanged(index));
    return TensorLoad(found->second.dst, indices, op->loc);
  }

  UnchangedOr<Expr> Mutate_(const VarObj *op, InplaceMode mode) final {
    auto found = reducers_.find(op);
    if (found == reducers_.end())
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_CHECK(found->second.phase == Phase::kFinalized, ValueError)
        << "unsupported use of legacy reducer handle " << op->name
        << " inside its reduction epoch";
    return found->second.dst;
  }

private:
  enum class Phase { kAllocated, kActive, kFinalized };
  struct LegacyReducer {
    TensorVar old_tensor{UnsafeInit{}};
    TensorVar acc{UnsafeInit{}};
    TensorVar dst{UnsafeInit{}};
    ReducerV2OpType op;
    String op_name;
    Phase phase{Phase::kAllocated};
  };
  LegacyReducer *FindReducer(const Expr &region) {
    auto tensor = reducer_native::RegionTensor(region);
    auto found = reducers_.find(tensor.get());
    return found == reducers_.end() ? nullptr : &found->second;
  }
  Optional<PrimExpr> MatchUpdate(const LegacyReducer &entry,
                                 const TensorStoreObj *op) {
    auto self = [&](const PrimExpr &value) {
      const auto *load = value.as<TensorLoadObj>();
      return load && load->source.same_as(entry.old_tensor) &&
             StructuralEqual()(load->indices, op->indices);
    };
    auto binary = [&](PrimExpr a, PrimExpr b) -> Optional<PrimExpr> {
      if (self(a))
        return b;
      if (self(b))
        return a;
      return std::nullopt;
    };
    if (entry.op == ReducerV2OpType::kSum)
      if (auto node = op->value.as<AddObj>())
        return binary(node->a, node->b);
    if (entry.op == ReducerV2OpType::kMax)
      if (auto node = op->value.as<MaxObj>())
        return binary(node->a, node->b);
    if (entry.op == ReducerV2OpType::kMin)
      if (auto node = op->value.as<MinObj>())
        return binary(node->a, node->b);
    return std::nullopt;
  }
  std::unordered_map<const VarObj *, LegacyReducer> reducers_;
};
} // namespace

tvm::transform::Pass CanonicalizeLegacyReducer() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    return make_object<LegacyReducerCanonicalizer>()->Run(std::move(function));
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.CanonicalizeLegacyReducer");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.CanonicalizeLegacyReducer",
                                   CanonicalizeLegacyReducer);
}
} // namespace tvm::tl
