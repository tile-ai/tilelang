#include "ws_analysis.h"

#include "cuda/op/builtin.h"
#include "cuda/op/copy.h"
#include "cuda/target_utils.h"
#include "layout/cute_layout.h"
#include "op/gemm.h"
#include "op/gemm_sp.h"
#include "transform/common/memory_access.h"

#include <tvm/tirx/op/region.h>

#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class TensorRemapper : public StmtExprMutator {
public:
  explicit TensorRemapper(BufferRemap remap) : remap_(std::move(remap)) {
    for (const auto &[source, replacement] : remap_)
      TVM_FFI_CHECK(source.dtype() == replacement.dtype(), TypeError)
          << "Warp-specialization tensor remaps must preserve element types";
  }

private:
  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>()) {
      if (auto found = remap_.find(*tensor); found != remap_.end())
        return Expr(found->second);
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    auto tensor = node->var.as<TensorVar>();
    auto found = tensor ? remap_.find(*tensor) : remap_.end();
    if (found == remap_.end())
      return StmtExprMutator::Mutate_(node, mode);
    Expr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    if (auto call = value.as<Call>();
        call && (call.value()->op.same_as(alloc_tensor_op()) ||
                 call.value()->op.same_as(decl_tensor_op()))) {
      value =
          Call(found->second.ty(), call.value()->op, call.value()->args,
               call.value()->attrs, {found->second.ty()}, call.value()->loc);
    }
    TVM_FFI_CHECK(ffi::StructuralEqual()(value->ty, found->second.ty()),
                  TypeError)
        << "A remapped tensor definition must produce its replacement type";
    return Bind(found->second, value, node->loc);
  }

  BufferRemap remap_;
};
} // namespace

ffi::Optional<Call> GetEvaluateCallInSimpleWrapper(const Stmt &statement) {
  if (const auto *evaluation = statement.as<EvaluateNode>())
    return evaluation->value.as<Call>();
  if (const auto *sequence = statement.as<SeqStmtNode>();
      sequence && sequence->seq.size() == 1)
    return GetEvaluateCallInSimpleWrapper(sequence->seq[0]);
  if (const auto *conditional = statement.as<IfNode>();
      conditional && !conditional->else_case)
    return GetEvaluateCallInSimpleWrapper(conditional->then_case);
  if (const auto *region = statement.as<RegionStmtNode>();
      region && region->body_params.empty() && region->result_vars.empty() &&
      (region->op.same_as(tirx::scope_op()) || region->op->name == "tl.scope"))
    return GetEvaluateCallInSimpleWrapper(region->body);
  return std::nullopt;
}

bool IsBarrierOrTmaControlCall(const CallNode *call) {
  auto operation = call->op.as<Op>();
  if (!operation)
    return false;
  std::string name = operation.value()->name;
  static const std::unordered_set<std::string> barriers{
      "tirx.gpu_storage_sync",
      "tl.sync_grid",
      "tl.cluster_sync",
      "tl.syncthreads_count",
      "tl.syncthreads_and",
      "tl.syncthreads_or",
      "tl.named_barrier_arrive",
      "tl.ptx_fence_barrier_init",
      "tl.ptx_cp_async_barrier_noinc",
      "tl.ptx_arrive_cluster_barrier"};
  return barriers.count(name) || name.starts_with("tl.tma_") ||
         name.starts_with("tl.mbarrier_") ||
         name.starts_with("tirx.ptx.mbarrier") ||
         name == "tirx.ptx.cp_async_mbarrier_arrive";
}

std::optional<GemmInfo> GetGemmInfo(const TileOperator &operation) {
  if (const auto *gemm = operation.as<GemmNode>())
    return GemmInfo{gemm->cRegion_->source.as_or_throw<tirx::TensorVar>(),
                    gemm->wgWait_};
  if (const auto *gemm = operation.as<GemmSPNode>())
    return GemmInfo{gemm->cRegion_->source.as_or_throw<tirx::TensorVar>(),
                    gemm->wg_wait};
  return std::nullopt;
}

TileStmtKind ClassifyCopy(const CopyNode *copy, Target target) {
  if (!copy)
    return TileStmtKind::kConsumer;
  auto result = cuda::ClassifyWarpSpecializedCopy(*copy, target);
  TVM_FFI_CHECK(result.supported, ValueError) << result.reason;
  if (cuda::CopyInstIsTMAStore(result.inst))
    return TileStmtKind::kTmaStore;
  if (cuda::CopyInstIsTMALoad(result.inst))
    return TileStmtKind::kTmaProducer;
  if (cuda::CopyInstIsCPAsync(result.inst))
    return TileStmtKind::kCpAsyncProducer;
  return TileStmtKind::kConsumer;
}

TileStmtKind ClassifyStmt(const Stmt &statement, Target target) {
  if (const auto *evaluation = statement.as<EvaluateNode>()) {
    if (auto call = evaluation->value.as<Call>()) {
      if (call.value()->op.same_as(ptx_cp_async()) && call.value()->attrs &&
          call.value()
              ->attrs.value()
              .as_or_throw<DictAttrs>()
              .GetAttr<bool>("tl.ws_managed_cp_async")
              .value_or(false))
        return TileStmtKind::kCpAsyncProducer;
      auto operation = ParseOperator(*call);
      if (operation.defined()) {
        if (const auto *copy = operation.as<CopyNode>())
          return ClassifyCopy(copy, target);
        if (operation.as<Im2ColOpNode>() && TargetIsHopper(target))
          return TileStmtKind::kTmaProducer;
        if (auto gemm = GetGemmInfo(operation);
            gemm && IsTmemBuffer(gemm->accumulator))
          return TileStmtKind::kTcgen05Mma;
        return TileStmtKind::kConsumer;
      }
    }
  }
  bool raw_copy = false;
  PostOrderVisit(statement, [&](const ffi::ObjectRef &object) {
    const auto *call = object.as<CallNode>();
    auto operation = call ? call->op.as<Op>() : std::nullopt;
    if (operation) {
      std::string name = operation.value()->name;
      raw_copy |=
          name == "tl.ptx_cp_async" || name.starts_with("tirx.ptx.cp_async");
    }
  });
  if (raw_copy)
    return TileStmtKind::kCpAsyncRaw;
  auto collector = ffi::make_object<MemoryAccessCollector>();
  collector->Visit(statement);
  bool reads_global = false, writes_shared = false, reads_shared_local = false;
  for (const MemoryAccess &access : collector->accesses) {
    const ffi::String &scope = access.location.scope;
    bool shared = scope == "shared" || scope == "shared.dyn";
    reads_global |= (access.mask & kAccessRead) && scope == "global";
    writes_shared |= (access.mask & kAccessWrite) && shared;
    reads_shared_local |=
        (access.mask & kAccessRead) && (shared || scope.find("local") == 0);
  }
  return reads_global && writes_shared && !reads_shared_local
             ? TileStmtKind::kSimtProducer
             : TileStmtKind::kConsumer;
}

bool IsProducer(TileStmtKind kind) {
  return kind == TileStmtKind::kTmaProducer ||
         kind == TileStmtKind::kCpAsyncProducer ||
         kind == TileStmtKind::kCpAsyncRaw ||
         kind == TileStmtKind::kSimtProducer;
}

bool HasManualWarpSpecialization(const Stmt &statement) {
  bool found = false;
  PostOrderVisit(statement, [&](const ffi::ObjectRef &object) {
    if (const auto *region = object.as<RegionStmtNode>())
      found |= region->attrs->dict.count("warp_specialize") ||
               region->attrs->dict.count(attr::kWarpSpecializationScope);
  });
  return found;
}

Stmt RemapBuffers(Stmt statement, const BufferRemap &remap) {
  auto rewriter = ffi::make_object<TensorRemapper>(remap);
  return rewriter->Mutate(statement).ValueOrUnchanged(statement);
}

bool IsTmaCompatibleLayout(const Layout &layout,
                           const tirx::TensorVar &tensor) {
  auto composed = cute::ComposedLayoutFromTileLang(layout);
  return composed && composed.value()
                         .Recast(tensor.dtype().bits(), 8)
                         ->swizzle->IsTMACompatible();
}

void CollectAnnotatedLayouts(const RegionStmt &region,
                             BufferLayoutMap &layouts) {
  auto annotations =
      region->attrs.GetAttr<ffi::Map<tirx::TensorVar, Layout>>("layout_map");
  if (annotations)
    for (const auto &[tensor, layout] : *annotations)
      layouts.insert_or_assign(tensor, std::make_pair(tensor, layout));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.cuda.analysis.StorageAliases",
           [](Stmt statement) {
             MemoryAliases aliases;
             ffi::Map<tirx::TensorVar, tirx::TensorVar> result;
             PostOrderVisit(statement, [&](const ffi::ObjectRef &object) {
               if (const auto *binding = object.as<BindNode>()) {
                 aliases.Bind(binding);
                 auto tensor = binding->var.as<tirx::TensorVar>();
                 MemoryLocation location = aliases.Resolve(binding->var);
                 if (tensor && location.allocation)
                   if (auto owner =
                           location.allocation.value().as<tirx::TensorVar>())
                     result.Set(*tensor, *owner);
               }
             });
             return result;
           })
      .def("tl.cuda.analysis.SideEffect",
           [](Expr expression) {
             return static_cast<int>(SideEffect(expression));
           })
      .def("tl.cuda.analysis.AccessSummary",
           [](Stmt statement) {
             auto collector = ffi::make_object<MemoryAccessCollector>();
             collector->Visit(statement);
             ffi::Map<Var, int64_t> memory;
             for (const MemoryAccess &access : collector->accesses)
               if (access.location.allocation) {
                 Var storage = *access.location.allocation;
                 memory.Set(storage,
                            memory.Get(storage).value_or(0) | access.mask);
               }
             ffi::Array<Var> definitions;
             PostOrderVisit(statement, [&](const ffi::ObjectRef &object) {
               if (const auto *binding = object.as<BindNode>())
                 definitions.push_back(binding->var);
             });
             return ffi::Array<ffi::Any>{memory, tvm::UndefinedVars(statement),
                                         definitions, collector->exits};
           })
      .def("tl.cuda.analysis.ClassifyStmt",
           [](Stmt statement, Target target) {
             return static_cast<int>(ClassifyStmt(statement, target));
           })
      .def("tl.cuda.analysis.HasManualWarpSpecialization",
           HasManualWarpSpecialization)
      .def("tl.cuda.analysis.IsTmaCompatibleLayout", IsTmaCompatibleLayout)
      .def("tl.cuda.analysis.RemapBuffers",
           [](Stmt statement,
              ffi::Map<tirx::TensorVar, tirx::TensorVar> replacements) {
             BufferRemap remap;
             for (const auto &[source, replacement] : replacements)
               remap.emplace(source, replacement);
             return RemapBuffers(statement, remap);
           });
}
} // namespace tvm::tl
