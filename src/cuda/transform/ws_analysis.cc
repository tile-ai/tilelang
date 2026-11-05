/*!
 * \file ws_analysis.cc
 * \brief Statement analysis shared by the warp-specialization passes.
 */

#include "./ws_analysis.h"
#include "ir/lexical_scope.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "cuda/target_utils.h"
#include "layout/cute_layout.h"
#include "op/gemm.h"
#include "op/gemm_sp.h"
#include "op/native_utils.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/native_layout_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;

namespace {

/// Detect if a statement is a SIMT global-to-shared memory copy.
/// Matches any statement that writes to shared memory and reads from global
/// memory, without reading shared or local buffers (which would indicate
/// consumer-side compute).  This is intentionally broader than "pure direct
/// copy" so that T.Parallel with complex indexing / if_then_else (later
/// lowered to cp.async) is also captured.
class SimtProducerDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    auto detector = ffi::make_object<SimtProducerDetector>();
    detector->Visit(stmt);
    return detector->writes_shared_ && detector->reads_global_ &&
           !detector->reads_shared_local_;
  }

private:
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    writes_shared_ |= IsSharedBuffer(op->dest.as_or_throw<TensorVar>());
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    reads_global_ |= IsGlobalBuffer(tensor);
    reads_shared_local_ |=
        IsSharedBuffer(tensor) || IsLocalBuffer(tensor, true);
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (const auto *name = op->op.as<OpObj>();
        name && name->name == "tl.annotation")
      return std::nullopt;
    // Address-of TensorLoad denotes no memory read. Its index expressions can
    // themselves load memory, so visit those, but never the addressed element.
    if (op->op.same_as(tirx::address_of_op()) ||
        op->op.same_as(tl::access_ptr())) {
      if (const auto *load = op->args[0].as<TensorLoadObj>()) {
        for (const PrimExpr &index : load->indices)
          Visit(index);
        for (size_t i = 1; i < op->args.size(); ++i)
          Visit(op->args[i]);
        return std::nullopt;
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  bool writes_shared_{false}, reads_global_{false}, reads_shared_local_{false};
};

Optional<Call> EvaluateCall(const Stmt &stmt) {
  if (const auto *seq = stmt.as<SeqStmtObj>()) {
    return seq->seq.size() == 1 ? EvaluateCall(seq->seq[0]) : Optional<Call>{};
  }
  if (const auto *evaluate = stmt.as<EvaluateObj>())
    return evaluate->value.as<Call>();
  if (const auto *branch = stmt.as<IfObj>()) {
    return !branch->else_case ? EvaluateCall(branch->then_case)
                              : Optional<Call>{};
  }
  if (const auto *region = stmt.as<RegionStmtObj>()) {
    // These wrappers impose a lexical/metadata scope, not new control flow.
    if (region->op->name == "tl.attr" ||
        region->op.same_as(tl::lexical_scope_op()))
      return EvaluateCall(region->body);
  }
  return std::nullopt;
}

class BufferRemapper : public StmtExprMutator {
public:
  explicit BufferRemapper(const BufferRemap &remap) {
    for (const auto &[old_tensor, new_tensor] : remap)
      VarRemapSet(old_tensor, new_tensor);
  }
};

} // namespace

Optional<Call> GetEvaluateCallInSimpleWrapper(const Stmt &stmt) {
  return EvaluateCall(stmt);
}

static bool ContainsPtxCpAsync(const Stmt &stmt) {
  bool found = false;
  VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *call = node.as<CallObj>()) {
      if (call->op.same_as(tl::ptx_cp_async_bytes()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        found = true;
      }
    }
  });
  return found;
}

static bool IsPtxCommitGroup(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  return call.has_value() && call.value()->op.same_as(tl::ptx_commit_group());
}

static bool IsPtxWaitGroup(const Stmt &stmt) {
  Optional<Call> call = GetEvaluateCallInSimpleWrapper(stmt);
  return call.has_value() && call.value()->op.same_as(tl::ptx_wait_group());
}

bool IsBarrierOrTmaControlCall(const CallObj *call) {
  // TODO: support warp specialization for cluster kernels and for TMA
  // gather4 / scatter4 (their raw intrinsics and hand protocols decline).
  return call->op.same_as(mbarrier_wait_parity()) ||
         call->op.same_as(mbarrier_expect_tx()) ||
         call->op.same_as(tl::ptx_arrive_barrier()) ||
         call->op.same_as(tl::ptx_arrive_cluster_barrier()) ||
         call->op.same_as(tl::ptx_arrive_barrier_expect_tx()) ||
         call->op.same_as(tl::named_barrier_arrive()) ||
         call->op.same_as(tl::ptx_fence_barrier_init()) ||
         call->op.same_as(tl::ptx_cp_async_barrier()) ||
         call->op.same_as(tl::ptx_cp_async_barrier_noinc()) ||
         call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col()) ||
         call->op.same_as(tma_load_multicast()) ||
         call->op.same_as(tma_load_gather4()) ||
         call->op.same_as(tma_store()) ||
         call->op.same_as(tma_store_scatter4()) ||
         call->op.same_as(tma_store_cluster()) ||
         call->op.same_as(tma_store_arrive()) ||
         call->op.same_as(tma_store_wait()) ||
         call->op.same_as(Op::Get("tirx.gpu_storage_sync")) ||
         call->op.same_as(Op::Get("tl.storage_sync")) ||
         call->op.same_as(tl::sync_grid()) ||
         call->op.same_as(tl::syncthreads_count()) ||
         call->op.same_as(tl::syncthreads_and()) ||
         call->op.same_as(tl::syncthreads_or());
}

std::optional<GemmInfo> GetGemmInfo(const TileOperator &op) {
  if (const auto *gemm = op.as<GemmNode>())
    return GemmInfo{RegionTensor(gemm->cRegion_), gemm->wgWait_};
  if (const auto *gemm = op.as<GemmSPNode>())
    return GemmInfo{RegionTensor(gemm->cRegion_), gemm->wg_wait};
  return std::nullopt;
}

TileStmtKind ClassifyCopy(const CopyNode *copy, Target target) {
  if (copy == nullptr) {
    return TileStmtKind::kConsumer;
  }
  cuda::CopyInstSelection result =
      cuda::ClassifyWarpSpecializedCopy(*copy, target);
  if (!result.supported) {
    return TileStmtKind::kConsumer;
  }
  if (cuda::CopyInstIsTMAStore(result.inst)) {
    return TileStmtKind::kTmaStore;
  }
  if (cuda::CopyInstIsTMALoad(result.inst)) {
    return TileStmtKind::kTmaProducer;
  }
  if (cuda::CopyInstIsCPAsync(result.inst)) {
    return TileStmtKind::kCpAsyncProducer;
  }
  return TileStmtKind::kConsumer;
}

TileStmtKind ClassifyStmt(const Stmt &stmt, Target target) {
  // Tile-op Calls: classify directly via CopyNode checks.
  if (auto *eval = stmt.as<EvaluateObj>()) {
    if (auto *call = eval->value.as<CallObj>()) {
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (tile_op.defined()) {
        if (auto *copy = tile_op.as<CopyNode>()) {
          return ClassifyCopy(copy, target);
        }
        // Im2Col lowers to tma_load_im2col on Hopper — treat as TMA
        // producer so it goes to the producer warp group.
        if (tile_op.as<Im2ColOpNode>()) {
          if (TargetIsHopper(target)) {
            return TileStmtKind::kTmaProducer;
          }
        }
        if (auto gemm = GetGemmInfo(tile_op)) {
          if (IsTmemBuffer(gemm->accumulator)) {
            return TileStmtKind::kTcgen05Mma;
          }
        }
        return TileStmtKind::kConsumer; // non-copy tile-op
      }
    }
  }
  // Explicit cp.async producer-side statements are already low-level builtins.
  if (ContainsPtxCpAsync(stmt) || IsPtxCommitGroup(stmt) ||
      IsPtxWaitGroup(stmt)) {
    return TileStmtKind::kCpAsyncRaw;
  }
  // Non-tile-op: check for SIMT global-to-shared copy.
  if (SimtProducerDetector::Detect(stmt)) {
    return TileStmtKind::kSimtProducer;
  }
  return TileStmtKind::kConsumer;
}

bool IsProducer(TileStmtKind kind) {
  return kind == TileStmtKind::kTmaProducer ||
         kind == TileStmtKind::kCpAsyncProducer ||
         kind == TileStmtKind::kCpAsyncRaw ||
         kind == TileStmtKind::kSimtProducer;
}

bool HasManualWarpSpecialization(const Stmt &stmt) {
  bool found = false;
  VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
    if (const auto *region = node.as<RegionStmtObj>();
        region && region->op->name == "tl.attr") {
      const auto data = ReadNativeAnnotations(region);
      const auto key = data.at("key").cast<String>();
      found |=
          key == "warp_specialize" || key == attr::kWarpSpecializationScope;
    }
  });
  return found;
}

Stmt RemapBuffers(Stmt stmt, const BufferRemap &remap) {
  if (remap.empty())
    return stmt;
  auto mutator = ffi::make_object<BufferRemapper>(remap);
  return mutator->Mutate(stmt).ValueOrUnchanged(stmt);
}

bool IsTmaCompatibleLayout(const Layout &layout, const TensorVar &buffer) {
  Optional<cute::ComposedLayout> composed =
      cute::ComposedLayoutFromTileLang(layout);
  if (!composed.has_value())
    return false;
  // Recast to byte space (the swizzle atom is defined on byte addresses).
  cute::ComposedLayout composed_bytes =
      composed.value().Recast(buffer.dtype().bits(), /*new_bits=*/8);
  return composed_bytes->swizzle->IsTMACompatible();
}

void CollectAnnotatedLayouts(const Stmt &body, BufferLayoutMap &layouts) {
  auto collect = [&](const Map<String, Any> &annotations) {
    auto encoded = annotations.Get("layout_map");
    if (!encoded)
      return;
    for (const auto &[key, value] : encoded->cast<Map<Any, Any>>()) {
      auto tensor = key.cast<TensorVar>();
      Layout layout;
      if (auto candidate = value.as<Layout>())
        layout = *candidate;
      layouts.insert_or_assign(tensor, std::make_pair(tensor, layout));
    }
  };
  VisitNativePostOrder(body, [&](const ObjectRef &node) {
    if (const auto *call = node.as<CallObj>()) {
      if (const auto *name = call->op.as<OpObj>();
          name && name->name == "tl.annotation")
        collect(ReadNativeAnnotations(call));
    } else if (const auto *region = node.as<RegionStmtObj>();
               region && region->op->name == "tl.attr") {
      auto fields = ReadNativeAnnotations(region);
      if (fields.at("key").cast<String>() == "tl.kernel")
        collect(fields.at("value").cast<Map<String, Any>>());
    }
  });
}

TileOpArguments ReadWSCall(const CallObj *call) {
  if (call->attrs) {
    auto attributes = call->attrs.value().as<DictAttrs>();
    if (attributes &&
        attributes.value().GetAttr<Array<Any>>("annotation_schema"))
      return DecodeTileOpArguments(call);
    if (attributes)
      return TileOpArguments{call->args, attributes.value()->dict};
  }
  return TileOpArguments{call->args, {}};
}

Call RewriteWSCall(const CallObj *call, const Map<String, Any> &annotations,
                   Optional<Op> replacement_op) {
  TileOpArguments decoded = ReadWSCall(call);
  auto op = replacement_op.value_or(call->op.as_or_throw<Op>());
  auto attrs = call->attrs ? call->attrs.value().as<DictAttrs>() : std::nullopt;
  if (!std::string(op->name).starts_with("tl.tileop.") &&
      !(attrs && attrs.value().GetAttr<Array<Any>>("annotation_schema")))
    return Call(call->ty, op, decoded.operands, DictAttrs(annotations),
                call->ty_args, call->loc);
  return MakeAnnotatedCall(call->ty, op, decoded.operands, annotations,
                           call->loc);
}

} // namespace tl
} // namespace tvm
