/*!
 * \file tl/backend/cuda/op/copy_analysis.cc
 * \brief CUDA copy instruction classification helpers.
 */

#include "backend/cuda/op/copy.h"

#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/tir/transform.h>

#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace cuda {

using namespace tir;

namespace {

PrimExpr TMABytesFromElements(PrimExpr elements, DataType dtype) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  int bits = dtype.bits();
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

PrimExpr TMABitsFromElements(PrimExpr elements, DataType dtype) {
  return cast(DataType::Int(64), elements) *
         IntImm(DataType::Int(64), dtype.bits());
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetDisableTMA(const CopyNode &op) {
  return GetBoolAnnotation(op, "disable_tma");
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

bool CheckGlobalStrides(const Buffer &buffer, arith::Analyzer *analyzer,
                        bool emit_diagnostics) {
  Array<PrimExpr> strides = buffer->strides;
  if (strides.empty()) {
    PrimExpr stride = 1;
    strides.resize(buffer->shape.size());
    for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
      strides.Set(i, stride);
      stride *= buffer->shape[i];
    }
  }

  if (!strides.empty() &&
      analyzer->CanProve(strides[strides.size() - 1] != 1,
                         arith::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "TMA bulk copy requires contiguous innermost global stride"
          << ", but got " << strides[strides.size() - 1] << " for buffer "
          << buffer->name << ", fallback to normal copy.";
    }
    return false;
  }

  for (size_t i = 0; i + 1 < strides.size(); ++i) {
    PrimExpr stride_bytes = TMABytesFromElements(strides[i], buffer->dtype);
    if (analyzer->CanProve(
            FloorMod(stride_bytes, IntImm(DataType::Int(64), 16)) != 0,
            arith::ProofStrength::kSymbolicBound)) {
      if (emit_diagnostics) {
        DLOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                      << stride_bytes << " for buffer " << buffer->name
                      << ", fallback to normal copy.";
      }
      return false;
    }
    if (const int64_t *stride =
            as_const_int(analyzer->Simplify(stride_bytes))) {
      if (*stride >= (int64_t{1} << 40)) {
        if (emit_diagnostics) {
          DLOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                        << stride_bytes << " for buffer " << buffer->name
                        << ", fallback to normal copy.";
        }
        return false;
      }
    }
  }
  return true;
}

bool CheckBulkLoad(const CopyNode &op, Target target, arith::Analyzer *analyzer,
                   bool check_last_dim, bool emit_diagnostics) {
  if (!TargetHasBulkCopy(target)) {
    return false;
  }
  if (op.src.scope() != "global" ||
      (op.dst.scope() != "shared.dyn" && op.dst.scope() != "shared")) {
    return false;
  }
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(
              TMABitsFromElements(op.src_range[op.src_range.size() - 1]->extent,
                                  op.src->dtype),
              IntImm(DataType::Int(64), 128)) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "src range must have last dim multiple of 16 for tma bulk load "
          << op.src->name << " range "
          << op.src_range[op.src_range.size() - 1]->extent << " * "
          << op.src->dtype.bits() << " bits % 128 != 0";
    }
    return false;
  }

  if (op.src->dtype != op.dst->dtype) {
    if (emit_diagnostics) {
      DLOG(WARNING) << "src and dst must have the same dtype for tma load "
                    << op.src->name << " vs. " << op.dst->name << " dtype "
                    << op.src->dtype << " vs. " << op.dst->dtype
                    << " will be fallback to normal copy";
    }
    return false;
  }
  return CheckGlobalStrides(op.src, analyzer, emit_diagnostics);
}

bool CheckBulkStore(const CopyNode &op, Target target,
                    arith::Analyzer *analyzer, bool check_last_dim,
                    bool emit_diagnostics) {
  if (!TargetHasBulkCopy(target)) {
    return false;
  }
  if ((op.src.scope() != "shared.dyn" && op.src.scope() != "shared") ||
      op.dst.scope() != "global") {
    return false;
  }
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(
              TMABitsFromElements(op.dst_range[op.dst_range.size() - 1]->extent,
                                  op.dst->dtype),
              IntImm(DataType::Int(64), 128)) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "dst range must have last dim multiple of 16 for tma bulk store "
          << op.dst->name << " range "
          << op.dst_range[op.dst_range.size() - 1]->extent << " * "
          << op.dst->dtype.bits() << " bits % 128 != 0";
    }
    return false;
  }
  if (op.src->dtype != op.dst->dtype) {
    if (emit_diagnostics) {
      DLOG(WARNING) << "src and dst must have the same dtype for tma store "
                    << op.src->name << " vs. " << op.dst->name << " dtype "
                    << op.src->dtype << " vs. " << op.dst->dtype
                    << " will be fallback to normal copy";
    }
    return false;
  }
  return CheckGlobalStrides(op.dst, analyzer, emit_diagnostics);
}

bool CheckBulkCopy1D(const Buffer &global_tensor, const Buffer &shared_tensor,
                     const Array<Range> &global_range,
                     const Array<Range> &shared_range,
                     const LayoutMap &layout_map, arith::Analyzer *analyzer) {
  bool shared_is_contiguous = true;
  if (layout_map.count(shared_tensor)) {
    Layout existing =
        layout_map.Get(shared_tensor).value().as<Layout>().value();
    Layout linear_layout = makeLinearLayout(shared_tensor->shape);
    shared_is_contiguous = StructuralEqual()(existing, linear_layout);
  }

  bool global_is_contiguous = true;
  bool global_not_full_dim_encounter = false;
  for (int i = global_range.size() - 1; i >= 0; i--) {
    if (!global_not_full_dim_encounter) {
      if (!analyzer->CanProve(global_range[i]->extent ==
                                      global_tensor->shape[i] &&
                                  global_range[i]->min == 0,
                              arith::ProofStrength::kSymbolicBound)) {
        global_not_full_dim_encounter = true;
      }
    } else {
      if (!analyzer->CanProve(global_range[i]->extent == 1,
                              arith::ProofStrength::kSymbolicBound)) {
        global_is_contiguous = false;
        break;
      }
    }
  }

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }
  PrimExpr global_elements = 1;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_elements *= global_range[i]->extent;
  }
  bool element_match =
      analyzer->CanProveEqual(shared_elements, global_elements);
  return shared_is_contiguous && global_is_contiguous && element_match;
}

bool CheckBulkLoad1D(const CopyNode &op, Target target,
                     const LayoutMap &layout_map, arith::Analyzer *analyzer,
                     bool emit_diagnostics) {
  if (!CheckBulkLoad(op, target, analyzer, false, emit_diagnostics)) {
    return false;
  }
  return CheckBulkCopy1D(op.src, op.dst, op.src_range, op.dst_range, layout_map,
                         analyzer);
}

bool CheckBulkStore1D(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, arith::Analyzer *analyzer,
                      bool emit_diagnostics) {
  if (!CheckBulkStore(op, target, analyzer, false, emit_diagnostics)) {
    return false;
  }
  return CheckBulkCopy1D(op.dst, op.src, op.dst_range, op.src_range, layout_map,
                         analyzer);
}

bool CheckLDSMCopy(const CopyNode &op, Target target) {
  return TargetHasLdmatrix(target) && IsSharedBuffer(op.src) &&
         IsFragmentBuffer(op.dst);
}

bool CheckSTSMCopy(const CopyNode &op, Target target) {
  return TargetHasStmatrix(target) && IsFragmentBuffer(op.src) &&
         IsSharedBuffer(op.dst);
}

bool CheckTMemLoad(const CopyNode &op, Target target) {
  return TargetHasTmem(target) && op.src.scope() == "shared.tmem" &&
         IsFragmentBuffer(op.dst);
}

bool CheckTMemStore(const CopyNode &op, Target target) {
  return TargetHasTmem(target) && IsFragmentBuffer(op.src) &&
         op.dst.scope() == "shared.tmem";
}

bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
  return IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src->dtype == op.dst->dtype;
}

bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, arith::Analyzer *analyzer) {
  if (!TargetHasAsyncCopy(target)) {
    return false;
  }
  if (!CheckCPAsyncCopyPreconditions(op)) {
    return false;
  }
  // Skip vectorize size checks here because the layout is not stable during
  // layout inference and transform classification.
  return true;
}

} // namespace

const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "Normal";
  case CopyInst::kLDSM:
    return "LDSM";
  case CopyInst::kSTSM:
    return "STSM";
  case CopyInst::kBulkLoad:
    return "BulkLoad";
  case CopyInst::kBulkStore:
    return "BulkStore";
  case CopyInst::kCPAsync:
    return "CPAsync";
  case CopyInst::kBulkLoad1D:
    return "BulkLoad1D";
  case CopyInst::kBulkStore1D:
    return "BulkStore1D";
  case CopyInst::kTMemLoad:
    return "TMemLoad";
  case CopyInst::kTMemStore:
    return "TMemStore";
  case CopyInst::kInvalid:
    return "Invalid";
  default:
    return "Unknown";
  }
}

bool CopyInstIsTMA(CopyInst inst) {
  return inst == CopyInst::kBulkLoad || inst == CopyInst::kBulkStore ||
         inst == CopyInst::kBulkLoad1D || inst == CopyInst::kBulkStore1D;
}

bool CopyInstIsCPAsync(CopyInst inst) { return inst == CopyInst::kCPAsync; }

namespace {

struct CopyFacts {
  bool cuda_like_target = false;
  bool has_layout_map = false;
  bool layout_dependent_tma_available = false;
  bool pass_context_disables_tma = false;
  bool explicit_tma = false;
  bool explicit_cp_async = false;
  bool no_implicit_async_commit_wait = false;
  bool disable_tma = false;
  bool can_bulk_load_1d = false;
  bool can_bulk_store_1d = false;
  bool can_bulk_load = false;
  bool can_bulk_store = false;
  bool can_bulk_load_ignore_last_dim = false;
  bool can_bulk_store_ignore_last_dim = false;
  bool can_cp_async = false;
  bool can_ldsm = false;
  bool can_stsm = false;
  bool can_tmem_load = false;
  bool can_tmem_store = false;
  std::string tma_unavailable_reason;
  std::string async_unavailable_reason;
};

bool IsCudaLikeTarget(Target target) {
  return target.defined() && (TargetIsCuda(target) || TargetIsCuTeDSL(target));
}

CopyInstSelection Supported(CopyInst inst) {
  return CopyInstSelection{inst, true, ""};
}

CopyInstSelection Unsupported(std::string reason) {
  return CopyInstSelection{CopyInst::kInvalid, false, std::move(reason)};
}

std::string MakeTmaUnavailableReason(const CopyNode &op) {
  std::ostringstream oss;
  oss << "T.tma_copy() requires TMA-capable target and global<->shared copy "
         "pattern, but TMA is not available for src="
      << op.src->name << ", dst=" << op.dst->name;
  return oss.str();
}

std::string MakeAsyncUnavailableReason(const CopyNode &op, Target target) {
  std::ostringstream oss;
  if (!target.defined()) {
    oss << "T.async_copy requires a defined target.";
  } else if (!TargetHasAsyncCopy(target)) {
    oss << "T.async_copy is only supported on targets with cp.async support "
           "(SM80+). Got target="
        << target;
  } else if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    oss << "T.async_copy only supports global->shared/shared.dyn copies. "
           "Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
  } else if (op.src->dtype != op.dst->dtype) {
    oss << "T.async_copy requires equal byte-addressable dtypes. Got src "
           "dtype="
        << op.src->dtype << ", dst dtype=" << op.dst->dtype << ".";
  } else {
    oss << "Explicit async copy semantics require cp.async lowering, but "
           "constraints were not satisfied. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype << ").";
  }
  return oss.str();
}

bool IsAutoAsyncCopyEnabled(bool default_enabled) {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  return pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(default_enabled))
      .value();
}

CopyInst SelectTmaInst(const CopyFacts &facts, bool allow_load,
                       bool allow_store, bool check_last_dim) {
  if (allow_load && facts.can_bulk_load_1d) {
    return CopyInst::kBulkLoad1D;
  }
  if (allow_store && facts.can_bulk_store_1d) {
    return CopyInst::kBulkStore1D;
  }
  if (allow_load && (check_last_dim ? facts.can_bulk_load
                                    : facts.can_bulk_load_ignore_last_dim)) {
    return CopyInst::kBulkLoad;
  }
  if (allow_store && (check_last_dim ? facts.can_bulk_store
                                     : facts.can_bulk_store_ignore_last_dim)) {
    return CopyInst::kBulkStore;
  }
  return CopyInst::kInvalid;
}

CopyInst SelectSyncLikeInst(const CopyFacts &facts) {
  if (facts.can_ldsm) {
    return CopyInst::kLDSM;
  }
  if (facts.can_stsm) {
    return CopyInst::kSTSM;
  }
  if (facts.can_tmem_load) {
    return CopyInst::kTMemLoad;
  }
  if (facts.can_tmem_store) {
    return CopyInst::kTMemStore;
  }
  return CopyInst::kNormal;
}

CopyFacts AnalyzeCopyFacts(const CopyNode &op, const CopyAnalysisContext &ctx) {
  CopyFacts facts;
  facts.cuda_like_target = IsCudaLikeTarget(ctx.target);
  facts.has_layout_map = ctx.layout_map != nullptr;
  facts.explicit_tma = GetIsTmaCopy(op);
  facts.explicit_cp_async = GetIsAsyncCopy(op);
  facts.no_implicit_async_commit_wait = GetNoImplicitAsyncCommitWait(op);
  facts.disable_tma = GetDisableTMA(op);
  facts.tma_unavailable_reason = MakeTmaUnavailableReason(op);
  facts.async_unavailable_reason = MakeAsyncUnavailableReason(op, ctx.target);
  facts.pass_context_disables_tma =
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>(kDisableTMALower, Bool(false))
          .value();

  if (!facts.cuda_like_target) {
    return facts;
  }

  arith::Analyzer local_analyzer;
  arith::Analyzer *analyzer =
      ctx.analyzer != nullptr ? ctx.analyzer : &local_analyzer;
  static const LayoutMap empty_layout_map;
  const LayoutMap &layout_map =
      ctx.layout_map != nullptr ? *ctx.layout_map : empty_layout_map;
  bool is_cutedsl = TargetIsCuTeDSL(ctx.target);
  facts.layout_dependent_tma_available =
      facts.has_layout_map && !is_cutedsl && !ctx.buffer_oob;

  if (facts.layout_dependent_tma_available) {
    facts.can_bulk_load_1d =
        CheckBulkLoad1D(op, ctx.target, layout_map, analyzer,
                        /*emit_diagnostics=*/false);
    facts.can_bulk_store_1d =
        CheckBulkStore1D(op, ctx.target, layout_map, analyzer,
                         /*emit_diagnostics=*/false);
  }

  if (facts.can_bulk_load_1d) {
    facts.can_bulk_load_ignore_last_dim = true;
  } else {
    facts.can_bulk_load_ignore_last_dim =
        CheckBulkLoad(op, ctx.target, analyzer, /*check_last_dim=*/false,
                      ctx.emit_diagnostics);
    facts.can_bulk_load =
        CheckBulkLoad(op, ctx.target, analyzer, /*check_last_dim=*/true,
                      ctx.emit_diagnostics);
  }

  if (facts.can_bulk_store_1d) {
    facts.can_bulk_store_ignore_last_dim = true;
  } else {
    facts.can_bulk_store_ignore_last_dim =
        CheckBulkStore(op, ctx.target, analyzer, /*check_last_dim=*/false,
                       ctx.emit_diagnostics);
    facts.can_bulk_store =
        CheckBulkStore(op, ctx.target, analyzer, /*check_last_dim=*/true,
                       ctx.emit_diagnostics);
  }

  facts.can_cp_async = CheckCPAsyncCopy(op, ctx.target, layout_map, analyzer);
  facts.can_ldsm = CheckLDSMCopy(op, ctx.target);
  facts.can_stsm = CheckSTSMCopy(op, ctx.target);
  facts.can_tmem_load = CheckTMemLoad(op, ctx.target);
  facts.can_tmem_store = CheckTMemStore(op, ctx.target);
  return facts;
}

} // namespace

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (facts.explicit_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/true,
                      /*check_last_dim=*/true);
    return inst == CopyInst::kInvalid
               ? Unsupported(facts.tma_unavailable_reason)
               : Supported(inst);
  }

  if (facts.explicit_cp_async || facts.no_implicit_async_commit_wait) {
    return facts.can_cp_async ? Supported(CopyInst::kCPAsync)
                              : Unsupported(facts.async_unavailable_reason);
  }

  if (!facts.disable_tma && !facts.pass_context_disables_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/false, /*allow_store=*/true,
                      /*check_last_dim=*/true);
    if (inst != CopyInst::kInvalid) {
      return Supported(inst);
    }
  }

  return Supported(SelectSyncLikeInst(facts));
}

std::string ClassifyCopyForInstructionAnnotation(const CopyNode &op,
                                                 Target target,
                                                 bool in_pipeline) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target) {
    return "sync";
  }

  if (facts.explicit_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/true,
                      /*check_last_dim=*/false);
    return CopyInstIsTMA(inst) ? "tma" : "sync";
  }

  if (facts.explicit_cp_async || facts.no_implicit_async_commit_wait) {
    return facts.can_cp_async ? "cp_async" : "sync";
  }

  if (in_pipeline && IsAutoAsyncCopyEnabled(/*default_enabled=*/false) &&
      facts.can_cp_async) {
    return "cp_async";
  }

  return "sync";
}

CopyInstSelection ClassifyWarpSpecializedProducerCopy(const CopyNode &op,
                                                      Target target) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target) {
    return Supported(CopyInst::kNormal);
  }

  if (facts.explicit_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/false,
                      /*check_last_dim=*/false);
    return inst == CopyInst::kInvalid
               ? Unsupported(facts.tma_unavailable_reason)
               : Supported(inst);
  }

  if (facts.explicit_cp_async || facts.no_implicit_async_commit_wait) {
    return facts.can_cp_async ? Supported(CopyInst::kCPAsync)
                              : Unsupported(facts.async_unavailable_reason);
  }

  if (!facts.disable_tma) {
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/false,
                      /*check_last_dim=*/true);
    if (inst != CopyInst::kInvalid) {
      return Supported(inst);
    }
  }

  return Supported(SelectSyncLikeInst(facts));
}

bool IsPipelineManagedCPAsyncCopy(const CopyNode &op, Target target) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target || facts.explicit_tma ||
      facts.explicit_cp_async) {
    return false;
  }
  return facts.can_cp_async;
}

} // namespace cuda
} // namespace tl
} // namespace tvm
