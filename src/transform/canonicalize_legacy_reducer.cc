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

#include "support/check.h"
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>

#include "../op/fill.h"
#include "../op/reducer.h"
#include "../op/region.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*! \brief Build a tl.region(...) call for `buffer[mins]` with unit extents
 *  (point region) or full extents. */
PrimExpr MakeRegionCall(const Buffer &buffer, const Array<PrimExpr> &mins,
                        const Array<PrimExpr> &extents, int access_mask) {
  Array<PrimExpr> args;
  args.push_back(BufferLoad(buffer, mins));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const auto &extent : extents) {
    args.push_back(extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

PrimExpr MakeFullRegionCall(const Buffer &buffer, int access_mask) {
  Array<PrimExpr> zeros;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    zeros.push_back(make_zero(DataType::Int(32)));
  }
  return MakeRegionCall(buffer, zeros, buffer->shape, access_mask);
}

/*! \brief Evaluate a (possibly cast-wrapped) constant expression as double. */
std::optional<double> TryConstValue(PrimExpr e) {
  while (const auto *cast = e.as<CastNode>()) {
    e = cast->value;
  }
  if (const auto *i = e.as<IntImmNode>()) {
    return static_cast<double>(i->value);
  }
  if (const auto *f = e.as<FloatImmNode>()) {
    return f->value;
  }
  return std::nullopt;
}

class LegacyReducerCanonicalizer : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    LegacyReducerCanonicalizer canonicalizer;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = canonicalizer.VisitStmt(f->body);
    return f;
  }

private:
  enum class Phase : int { kAllocated, kActive, kFinalized };

  struct LegacyReducer {
    Buffer old_buffer;
    Buffer acc; // new local.reducer handle
    Buffer dst; // out-of-place finalize destination
    ReducerV2OpType op;
    Optional<PrimExpr> seed;
    Phase phase{Phase::kAllocated};
  };

  // ---- allocation & annotation rewrite ------------------------------------

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (auto anno = op->annotations.Get(tl::attr::kReducerInfo)) {
      auto map = anno.value().as<Map<Var, Map<String, String>>>();
      ICHECK(map) << "malformed reducer_info annotation";
      for (const auto &[var, info] : map.value()) {
        legacy_op_.emplace(var.get(),
                           ParseReducerV2OpType(info.Get("op").value()));
      }
    }
    for (const auto &buffer : op->alloc_buffers) {
      auto it = legacy_op_.find(buffer->data.get());
      if (it == legacy_op_.end()) {
        continue;
      }
      LegacyReducer entry;
      entry.old_buffer = buffer;
      entry.op = it->second;
      Var acc_var(buffer->data->name_hint,
                  PointerType(PrimType(buffer->dtype), "local.reducer"));
      entry.acc =
          Buffer(acc_var, buffer->dtype, buffer->shape, buffer->strides,
                 buffer->elem_offset, buffer->name, buffer->data_alignment,
                 buffer->offset_factor, buffer->buffer_type);
      Var dst_var(buffer->data->name_hint + "_result",
                  PointerType(PrimType(buffer->dtype), "local.fragment"));
      entry.dst = Buffer(dst_var, buffer->dtype, buffer->shape, buffer->strides,
                         buffer->elem_offset, buffer->name + "_result",
                         buffer->data_alignment, buffer->offset_factor,
                         buffer->buffer_type);
      reducers_.emplace(buffer->data.get(), std::move(entry));
    }

    auto result = StmtExprMutator::VisitStmt_(op).as<SBlock>().value();
    auto *p_result = result.CopyOnWrite();

    bool changed = false;
    Array<Buffer> new_allocs;
    Map<Var, Map<String, Any>> v2_info;
    if (auto anno = p_result->annotations.Get(attr::kReducerInfoV2)) {
      if (auto as_map = anno.value().as<Map<Var, Map<String, Any>>>()) {
        v2_info = as_map.value();
      }
    }
    for (const auto &buffer : p_result->alloc_buffers) {
      auto it = reducers_.find(buffer->data.get());
      if (it == reducers_.end()) {
        new_allocs.push_back(buffer);
        continue;
      }
      const LegacyReducer &entry = it->second;
      ICHECK(entry.phase == Phase::kFinalized)
          << "legacy reducer `" << buffer
          << "` has no T.finalize_reducer; cannot canonicalize.";
      new_allocs.push_back(entry.acc);
      new_allocs.push_back(entry.dst);
      Map<String, Any> info;
      switch (entry.op) {
      case ReducerV2OpType::kSum:
        info.Set("op", String("sum"));
        break;
      case ReducerV2OpType::kMax:
        info.Set("op", String("max"));
        break;
      case ReducerV2OpType::kMin:
        info.Set("op", String("min"));
        break;
      default:
        LOG(FATAL) << "legacy (v1) reducers only support sum/max/min";
      }
      if (entry.seed.defined()) {
        info.Set("seed", entry.seed.value());
      }
      v2_info.Set(entry.acc->data, info);
      changed = true;
    }
    if (changed) {
      p_result->alloc_buffers = new_allocs;
      p_result->annotations.Set(attr::kReducerInfoV2, v2_info);
      p_result->annotations.erase(tl::attr::kReducerInfo);
    }
    return result;
  }

  // ---- epoch statement rewrites --------------------------------------------

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // T.clear / T.fill on a legacy reducer opens the epoch.
    if (call->op.same_as(Fill::Get())) {
      if (LegacyReducer *entry = FindReducerInRegionArg(call->args[0])) {
        ICHECK(entry->phase == Phase::kAllocated)
            << "second T.clear/T.fill on legacy reducer `" << entry->old_buffer
            << "`: one allocation supports exactly one reduction epoch.";
        DataType dtype = entry->old_buffer->dtype;
        PrimExpr identity = ReducerV2Identity(entry->op, dtype);
        auto fill_const = TryConstValue(call->args[1]);
        auto identity_const = TryConstValue(identity);
        bool is_identity = fill_const.has_value() &&
                           identity_const.has_value() &&
                           *fill_const == *identity_const;
        if (!is_identity) {
          if (entry->op == ReducerV2OpType::kSum) {
            LOG(FATAL) << "legacy reducer `" << entry->old_buffer
                       << "` (op=sum) is filled with a non-zero value; its v1 "
                          "behavior depended on the physical thread count. Use "
                          "T.alloc_reducer(..., seed=value) with the v2 API "
                          "instead.";
          }
          // max/min are idempotent: a per-partial clamp equals a one-time
          // seed, so the v1 fill value is forwarded as the epoch seed
          // (normalized to the reducer dtype).
          if (fill_const.has_value()) {
            entry->seed = make_const(dtype, *fill_const);
          } else if (call->args[1].dtype() == dtype) {
            entry->seed = call->args[1];
          } else {
            entry->seed = Cast(dtype, call->args[1]);
          }
        }
        entry->phase = Phase::kActive;
        return Evaluate(Call(DataType::Handle(), ReducerInitOp::Get(),
                             {MakeFullRegionCall(entry->acc, kAccessWrite)}));
      }
    }
    // In-place T.finalize_reducer(acc) becomes out-of-place v2 finalize.
    if (call->op.same_as(FinalizeReducerOp::Get()) && call->args.size() == 1) {
      if (LegacyReducer *entry = FindReducerInRegionArg(call->args[0])) {
        ICHECK(entry->phase == Phase::kActive)
            << "T.finalize_reducer on legacy reducer `" << entry->old_buffer
            << "` without a preceding T.clear/T.fill.";
        entry->phase = Phase::kFinalized;
        return Evaluate(Call(DataType::Handle(), FinalizeReducerV2Op::Get(),
                             {MakeFullRegionCall(entry->acc, kAccessReadWrite),
                              MakeFullRegionCall(entry->dst, kAccessWrite)},
                             call->annotations)); // forwards batch
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto it = reducers_.find(op->buffer->data.get());
    if (it == reducers_.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    LegacyReducer &entry = it->second;
    ICHECK(entry.phase == Phase::kActive)
        << "store to legacy reducer `" << entry.old_buffer
        << (entry.phase == Phase::kFinalized ? "` after T.finalize_reducer."
                                             : "` before T.clear/T.fill.");
    PrimExpr contribution = MatchUpdate(entry, op);
    ICHECK(contribution.defined())
        << "unsupported store to legacy reducer `" << entry.old_buffer
        << "`: expected `acc[i] += v` (sum) or "
           "`acc[i] = T.max/T.min(acc[i], v)` matching the declared op, got "
        << GetRef<Stmt>(op)
        << ". Rewrite with the v2 API (T.reducer_update) if this is a "
           "different access pattern.";
    contribution = VisitExpr(contribution);
    return Evaluate(
        Call(DataType::Handle(), ReducerUpdateOp::Get(),
             {MakeRegionCall(entry.acc, op->indices,
                             Array<PrimExpr>(op->indices.size(),
                                             IntImm(DataType::Int(32), 1)),
                             kAccessReadWrite),
              contribution}));
  }

  // ---- post-finalize redirection & opaqueness ------------------------------

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto it = reducers_.find(op->buffer->data.get());
    if (it == reducers_.end()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    const LegacyReducer &entry = it->second;
    ICHECK(entry.phase == Phase::kFinalized)
        << "read of legacy reducer `" << entry.old_buffer
        << "` before T.finalize_reducer; partial values are not readable.";
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    return BufferLoad(entry.dst, load->indices);
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = reducers_.find(op);
    if (it != reducers_.end()) {
      const LegacyReducer &entry = it->second;
      ICHECK(entry.phase == Phase::kFinalized)
          << "unsupported use of legacy reducer handle `" << op->name_hint
          << "` inside its reduction epoch.";
      return entry.dst->data;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  // ---- helpers --------------------------------------------------------------

  LegacyReducer *FindReducerInRegionArg(const PrimExpr &arg) {
    if (auto call = arg.as<CallNode>()) {
      if (call->op.same_as(RegionOp::Get())) {
        if (auto load = call->args[0].as<BufferLoadNode>()) {
          auto it = reducers_.find(load->buffer->data.get());
          if (it != reducers_.end()) {
            return &it->second;
          }
        }
      }
    } else if (auto load = arg.as<BufferLoadNode>()) {
      auto it = reducers_.find(load->buffer->data.get());
      if (it != reducers_.end()) {
        return &it->second;
      }
    }
    return nullptr;
  }

  /*! \brief Match `combine(acc[idx], v)` against the declared combine op and
   *  return the contribution `v`, or an undefined PrimExpr on mismatch. */
  PrimExpr MatchUpdate(const LegacyReducer &entry, const BufferStoreNode *op) {
    auto is_self_load = [&](const PrimExpr &expr) {
      const auto *load = expr.as<BufferLoadNode>();
      return load != nullptr &&
             load->buffer->data.get() == entry.old_buffer->data.get() &&
             StructuralEqual()(load->indices, op->indices);
    };
    auto match_binary = [&](const PrimExpr &a, const PrimExpr &b) -> PrimExpr {
      if (is_self_load(a)) {
        return b;
      }
      if (is_self_load(b)) {
        return a;
      }
      return PrimExpr();
    };
    switch (entry.op) {
    case ReducerV2OpType::kSum:
      if (const auto *add = op->value.as<AddNode>()) {
        return match_binary(add->a, add->b);
      }
      break;
    case ReducerV2OpType::kMax:
      if (const auto *max = op->value.as<MaxNode>()) {
        return match_binary(max->a, max->b);
      }
      break;
    case ReducerV2OpType::kMin:
      if (const auto *min = op->value.as<MinNode>()) {
        return match_binary(min->a, min->b);
      }
      break;
    default:
      break; // legacy (v1) reducers only support sum/max/min
    }
    return PrimExpr();
  }

  std::unordered_map<const VarNode *, ReducerV2OpType> legacy_op_;
  std::unordered_map<const VarNode *, LegacyReducer> reducers_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass CanonicalizeLegacyReducer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LegacyReducerCanonicalizer::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CanonicalizeLegacyReducer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.CanonicalizeLegacyReducer",
                        CanonicalizeLegacyReducer);
}

} // namespace tl
} // namespace tvm
