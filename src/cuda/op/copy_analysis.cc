#include "compat/analyzer.h"
/*!
 * \file tl/cuda/op/copy_analysis.cc
 * \brief CUDA copy instruction classification helpers.
 */

#include "cuda/op/copy.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "op/utils.h"

#include <tvm/ir/transform.h>

#include <optional>
#include <sstream>
#include <utility>

namespace tvm {
namespace tl {
namespace cuda {

using namespace tirx;
using namespace ffi;

namespace {

int TMAPayloadElementBits(PrimType dtype) {
  if (IsFP4UnpackedType(dtype)) {
    return 4;
  }
  return dtype.bits();
}

PrimExpr TMABytesFromElements(PrimExpr elements, int bits) {
  PrimExpr elements_i64 = Cast(PrimType::Int(64), elements);
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(PrimType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(PrimType::Int(64), bits) +
                      IntImm(PrimType::Int(64), 7),
                  IntImm(PrimType::Int(64), 8));
}

PrimExpr TMAGlobalBytesFromElements(PrimExpr elements, PrimType dtype) {
  return TMABytesFromElements(elements, TMAPayloadElementBits(dtype));
}

PrimExpr TMAGlobalBitsFromElements(PrimExpr elements, PrimType dtype) {
  return Cast(PrimType::Int(64), elements) *
         IntImm(PrimType::Int(64), TMAPayloadElementBits(dtype));
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    auto enabled = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(enabled.has_value(), ValueError)
        << "T.copy " << key << " must be an integer or boolean";
    return *enabled != 0;
  }
  return false;
}

bool GetDisableTMA(const CopyNode &op) {
  return GetBoolAnnotation(op, "disable_tma");
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

int64_t GetClusterMask(const CopyNode &op) {
  if (auto val = op.annotations.Get("cluster_mask")) {
    auto mask = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(mask && *mask >= 0 && *mask <= 0xffff, ValueError)
        << "T.copy cluster_mask must be a constant 16-bit mask";
    return *mask;
  }
  return 0;
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

bool GetTmaDescriptorBaseIsDeviceBound(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kTmaDescriptorBaseIsDeviceBound);
}

enum class PreferredCopyInstruction {
  kAuto,
  kTMA,
  kCPAsync,
  kSync,
};

constexpr const char *kPreferInstruction = "prefer_instruction";

constexpr std::pair<const char *, PreferredCopyInstruction>
    kPreferredCopyInstructions[] = {
        {"tma", PreferredCopyInstruction::kTMA},
        {"cp_async", PreferredCopyInstruction::kCPAsync},
        {"sync", PreferredCopyInstruction::kSync},
};

std::optional<std::string> GetStringAnnotation(const CopyNode &op,
                                               const char *key) {
  auto val = op.annotations.Get(key);
  if (!val) {
    return std::nullopt;
  }
  if (auto string = val->as<ffi::String>()) {
    return std::string(string.value());
  }
  if (auto string = val->as<StringImm>()) {
    return std::string(string.value()->value);
  }
  TVM_FFI_THROW(ValueError)
      << "T.copy " << key << " annotation must be a string";
  TVM_FFI_UNREACHABLE();
}

PreferredCopyInstruction
ParsePreferredCopyInstruction(const std::string &prefer) {
  for (const auto &[name, inst] : kPreferredCopyInstructions) {
    if (prefer == name) {
      return inst;
    }
  }
  LOG(FATAL) << "Unsupported T.copy prefer_instruction=\"" << prefer
             << "\". Expected one of: \"tma\", \"cp_async\", \"sync\".";
  return PreferredCopyInstruction::kAuto;
}

PreferredCopyInstruction GetPreferredInstruction(const CopyNode &op) {
  if (auto prefer = GetStringAnnotation(op, kPreferInstruction)) {
    return ParsePreferredCopyInstruction(prefer.value());
  }
  return PreferredCopyInstruction::kAuto;
}

bool CheckGlobalStrides(const TensorVar &buffer, sym::Analyzer analyzer,
                        bool emit_diagnostics) {
  auto pattern = tirx::MatchShapeStridePattern(buffer.ty());
  if (!pattern.has_value()) {
    return false;
  }
  Array<PrimExpr> strides = pattern.value().get<1>();

  if (!strides.empty() &&
      tl::arith::CanProve(analyzer, strides[strides.size() - 1] != 1,
                          sym::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "TMA bulk copy requires contiguous innermost global stride"
          << ", but got " << strides[strides.size() - 1] << " for buffer "
          << buffer->name << ", fallback to normal copy.";
    }
    return false;
  }

  for (size_t i = 0; i + 1 < strides.size(); ++i) {
    PrimExpr stride_bytes =
        TMAGlobalBytesFromElements(strides[i], buffer.dtype());
    if (tl::arith::CanProve(
            analyzer,
            FloorMod(stride_bytes, IntImm(PrimType::Int(64), 16)) != 0,
            sym::ProofStrength::kSymbolicBound)) {
      if (emit_diagnostics) {
        DLOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                      << stride_bytes << " for buffer " << buffer->name
                      << ", fallback to normal copy.";
      }
      return false;
    }
    if (auto stride =
            GetConstInt(tl::arith::Simplify(analyzer, stride_bytes))) {
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

bool CheckBulkLoad(const CopyNode &op, Target target, sym::Analyzer analyzer,
                   bool check_last_dim, bool emit_diagnostics) {
  if (!TargetHasBulkCopy(target) || op.src_range.empty() ||
      op.dst_range.empty()) {
    return false;
  }
  if (op.src.scope().name() != "global" ||
      (op.dst.scope().name() != "shared.dyn" &&
       op.dst.scope().name() != "shared")) {
    return false;
  }
  if (check_last_dim &&
      tl::arith::CanProve(
          analyzer,
          FloorMod(TMAGlobalBitsFromElements(
                       op.src_range[op.src_range.size() - 1]->extent,
                       op.src.dtype()),
                   IntImm(PrimType::Int(64), 128)) != 0,
          sym::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "src range must have last dim multiple of 16 for tma bulk load "
          << op.src->name << " range "
          << op.src_range[op.src_range.size() - 1]->extent << " * "
          << op.src.dtype().bits() << " bits % 128 != 0";
    }
    return false;
  }

  if (!IsValidTMALoadDtypePair(op.src.dtype(), op.dst.dtype())) {
    if (emit_diagnostics) {
      DLOG(WARNING) << "src and dst must have compatible dtypes for tma load "
                    << op.src->name << " vs. " << op.dst->name << " dtype "
                    << op.src.dtype() << " vs. " << op.dst.dtype()
                    << " will be fallback to normal copy";
    }
    return false;
  }
  return CheckGlobalStrides(op.src, analyzer, emit_diagnostics);
}

bool CheckBulkStore(const CopyNode &op, Target target, sym::Analyzer analyzer,
                    bool check_last_dim, bool emit_diagnostics) {
  if (!TargetHasBulkCopy(target) || op.src_range.empty() ||
      op.dst_range.empty()) {
    return false;
  }
  if ((op.src.scope().name() != "shared.dyn" &&
       op.src.scope().name() != "shared") ||
      op.dst.scope().name() != "global") {
    return false;
  }
  if (check_last_dim &&
      tl::arith::CanProve(
          analyzer,
          FloorMod(TMAGlobalBitsFromElements(
                       op.dst_range[op.dst_range.size() - 1]->extent,
                       op.dst.dtype()),
                   IntImm(PrimType::Int(64), 128)) != 0,
          sym::ProofStrength::kSymbolicBound)) {
    if (emit_diagnostics) {
      DLOG(WARNING)
          << "dst range must have last dim multiple of 16 for tma bulk store "
          << op.dst->name << " range "
          << op.dst_range[op.dst_range.size() - 1]->extent << " * "
          << op.dst.dtype().bits() << " bits % 128 != 0";
    }
    return false;
  }
  if (!IsValidTMAStoreDtypePair(op.dst.dtype(), op.src.dtype())) {
    if (emit_diagnostics) {
      DLOG(WARNING) << "src and dst must have compatible dtypes for tma store "
                    << op.src->name << " vs. " << op.dst->name << " dtype "
                    << op.src.dtype() << " vs. " << op.dst.dtype()
                    << " will be fallback to normal copy";
    }
    return false;
  }
  return CheckGlobalStrides(op.dst, analyzer, emit_diagnostics);
}

bool IsSemanticallyLinearLayout(const Layout &layout,
                                const Array<PrimExpr> &shape,
                                sym::Analyzer analyzer) {
  Array<PrimExpr> input_shape = layout->InputShape();
  if (input_shape.size() != shape.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (!tl::arith::CanProveEqual(analyzer, input_shape[i], shape[i])) {
      return false;
    }
  }

  Layout linear_layout = MakeLinearLayout(shape);
  return tl::arith::CanProveEqual(analyzer, layout->GetLinearizedForwardIndex(),
                                  linear_layout->GetLinearizedForwardIndex());
}

bool IsContiguousGlobalRegion(const TensorVar &global_tensor,
                              const Array<Range> &global_range,
                              sym::Analyzer analyzer) {
  ICHECK_EQ(global_range.size(), global_tensor.shape().size());
  auto pattern = tirx::MatchShapeStridePattern(global_tensor.ty());
  if (!pattern.has_value()) {
    return false;
  }
  Array<PrimExpr> strides = pattern.value().get<1>();

  // Pointer-based 1D TMA can coalesce multiple logical dimensions only when
  // every varying dimension is packed directly after its contiguous suffix.
  bool non_full_dim_encountered = false;
  PrimExpr expected_stride = 1;
  for (int i = static_cast<int>(global_range.size()) - 1; i >= 0; --i) {
    bool is_singleton =
        tl::arith::CanProve(analyzer, global_range[i]->extent == 1,
                            sym::ProofStrength::kSymbolicBound);
    if (!non_full_dim_encountered) {
      if (!is_singleton &&
          !tl::arith::CanProveEqual(analyzer, strides[i], expected_stride)) {
        return false;
      }

      bool is_full_dim = tl::arith::CanProve(
          analyzer,
          global_range[i]->extent == global_tensor.shape()[i] &&
              global_range[i]->begin == 0,
          sym::ProofStrength::kSymbolicBound);
      if (is_full_dim) {
        expected_stride *= global_tensor.shape()[i];
      } else {
        non_full_dim_encountered = true;
      }
    } else if (!is_singleton) {
      return false;
    }
  }
  return true;
}

bool CheckBulkCopy1D(const TensorVar &global_tensor,
                     const TensorVar &shared_tensor,
                     const Array<Range> &global_range,
                     const Array<Range> &shared_range,
                     const LayoutMap &layout_map, sym::Analyzer analyzer) {
  bool shared_is_contiguous =
      IsContiguousGlobalRegion(shared_tensor, shared_range, analyzer);
  if (layout_map.count(shared_tensor)) {
    Layout existing =
        layout_map.Get(shared_tensor).value().as<Layout>().value();
    shared_is_contiguous =
        shared_is_contiguous &&
        IsSemanticallyLinearLayout(existing, shared_tensor.shape(), analyzer);
  }

  bool global_is_contiguous =
      IsContiguousGlobalRegion(global_tensor, global_range, analyzer);

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }

  PrimExpr global_elements = 1;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_elements *= global_range[i]->extent;
  }

  bool element_match =
      tl::arith::CanProveEqual(analyzer, shared_elements, global_elements);

  PrimExpr total_bits = shared_elements * shared_tensor.dtype().bits();

  // Reject only when the total transfer size is provably not 16-byte
  // aligned (e.g. constant shapes like 63 x fp32). Symbolic shapes stay on
  // the 1D TMA path, consistent with the last-dim check in CheckBulkLoad.
  bool total_16b_aligned =
      !tl::arith::CanProve(analyzer, FloorMod(total_bits, 128) != 0,
                           sym::ProofStrength::kSymbolicBound);

  return shared_is_contiguous && global_is_contiguous && element_match &&
         total_16b_aligned;
}

bool CanProveRegionInBounds(const TensorVar &buffer, const Array<Range> &region,
                            sym::Analyzer analyzer) {
  if (buffer.shape().size() != region.size()) {
    return false;
  }
  for (size_t i = 0; i < region.size(); ++i) {
    PrimExpr min = region[i]->begin;
    PrimExpr extent = region[i]->extent;
    if (!tl::arith::CanProve(analyzer,
                             min >= 0 && min + extent <= buffer.shape()[i],
                             sym::ProofStrength::kSymbolicBound)) {
      return false;
    }
  }
  return true;
}

bool CanProveCopyInBounds(const CopyNode &op, sym::Analyzer analyzer) {
  return CanProveRegionInBounds(op.src, op.src_range, analyzer) &&
         CanProveRegionInBounds(op.dst, op.dst_range, analyzer);
}

bool CheckBulkLoad1D(const CopyNode &op, Target target,
                     const LayoutMap &layout_map, sym::Analyzer analyzer,
                     bool emit_diagnostics) {
  if (!CheckBulkLoad(op, target, analyzer, false, emit_diagnostics)) {
    return false;
  }
  return CheckBulkCopy1D(op.src, op.dst, op.src_range, op.dst_range, layout_map,
                         analyzer);
}

bool CheckBulkStore1D(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, sym::Analyzer analyzer,
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
  return TargetHasTmem(target) && IsTmemBuffer(op.src) &&
         IsFragmentBuffer(op.dst);
}

bool CheckTMemStore(const CopyNode &op, Target target) {
  return TargetHasTmem(target) && IsFragmentBuffer(op.src) &&
         IsTmemBuffer(op.dst);
}

bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
  return IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src.dtype() == op.dst.dtype();
}

bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, sym::Analyzer analyzer) {
  if (!TargetCudaHasAsyncCopy(target)) {
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
  case CopyInst::kBulkLoadGather4:
    return "BulkLoadGather4";
  case CopyInst::kBulkStoreScatter4:
    return "BulkStoreScatter4";
  case CopyInst::kInvalid:
    return "Invalid";
  default:
    return "Unknown";
  }
}

bool CopyInstIsTMALoad(CopyInst inst) {
  return inst == CopyInst::kBulkLoad || inst == CopyInst::kBulkLoad1D;
}

bool CopyInstIsTMAStore(CopyInst inst) {
  return inst == CopyInst::kBulkStore || inst == CopyInst::kBulkStore1D;
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
  bool tma_descriptor_base_is_device_bound = false;
  PreferredCopyInstruction prefer_instruction = PreferredCopyInstruction::kAuto;
  bool disable_tma = false;
  int64_t cluster_mask = 0;
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
  if (GetTmaDescriptorBaseIsDeviceBound(op)) {
    const TensorVar &global_buffer = IsGlobalBuffer(op.src) ? op.src : op.dst;
    oss << "Descriptor-based TMA cannot use global base pointer `"
        << global_buffer->name
        << "` because it is bound inside the device function body. "
           "TensorMap descriptors are encoded on the host; use plain T.copy "
           "to allow a descriptorless or synchronous fallback.";
    return oss.str();
  }
  if (IsFP4UnpackedType(op.src.dtype()) || IsFP4UnpackedType(op.dst.dtype())) {
    oss << "T.tma_copy() only supports float4_e2m1_unpacked as an FP4 unpack "
           "load from packed global float4_e2m1fn to unpacked shared memory. "
           "The reverse unpacked shared -> packed global TMA store is not "
           "supported. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src.dtype() << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst.dtype() << ").";
    return oss.str();
  }
  oss << "T.tma_copy() requires TMA-capable target and global<->shared copy "
         "pattern, but TMA is not available for src="
      << op.src->name << ", dst=" << op.dst->name;
  return oss.str();
}

std::string MakeAsyncUnavailableReason(const CopyNode &op, Target target) {
  std::ostringstream oss;
  if (!target.defined()) {
    oss << "T.async_copy requires a defined target.";
  } else if (!TargetCudaHasAsyncCopy(target)) {
    oss << "T.async_copy is only supported on targets with cp.async support "
           "(SM80+). Got target="
        << target;
  } else if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    oss << "T.async_copy only supports global->shared/shared.dyn copies. "
           "Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
  } else if (op.src.dtype() != op.dst.dtype()) {
    oss << "T.async_copy requires equal byte-addressable dtypes. Got src "
           "dtype="
        << op.src.dtype() << ", dst dtype=" << op.dst.dtype() << ".";
  } else {
    oss << "Explicit async copy semantics require cp.async lowering, but "
           "constraints were not satisfied. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src.dtype() << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst.dtype() << ").";
  }
  return oss.str();
}

bool IsAutoAsyncCopyEnabled(bool default_enabled) {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  return pass_ctx->GetConfig<bool>(kEnableAsyncCopy).value_or(default_enabled);
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
  facts.tma_descriptor_base_is_device_bound =
      GetTmaDescriptorBaseIsDeviceBound(op);
  facts.prefer_instruction = GetPreferredInstruction(op);
  facts.disable_tma = GetDisableTMA(op);
  facts.cluster_mask = GetClusterMask(op);
  facts.tma_unavailable_reason = MakeTmaUnavailableReason(op);
  facts.async_unavailable_reason = MakeAsyncUnavailableReason(op, ctx.target);
  facts.pass_context_disables_tma = tvm::transform::PassContext::Current()
                                        ->GetConfig<bool>(kDisableTMALower)
                                        .value_or(false);

  if (!facts.cuda_like_target) {
    return facts;
  }

  sym::Analyzer analyzer = ctx.analyzer.value_or(sym::Analyzer());
  static const LayoutMap empty_layout_map;
  const LayoutMap &layout_map =
      ctx.layout_map != nullptr ? *ctx.layout_map : empty_layout_map;
  bool is_cutedsl = TargetIsCuTeDSL(ctx.target);
  facts.layout_dependent_tma_available =
      facts.has_layout_map && !is_cutedsl && CanProveCopyInBounds(op, analyzer);

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
    facts.can_bulk_load =
        CheckBulkLoad(op, ctx.target, analyzer, /*check_last_dim=*/true,
                      ctx.emit_diagnostics);
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
    facts.can_bulk_store =
        CheckBulkStore(op, ctx.target, analyzer,
                       /*check_last_dim=*/true, ctx.emit_diagnostics);
  } else {
    facts.can_bulk_store_ignore_last_dim =
        CheckBulkStore(op, ctx.target, analyzer,
                       /*check_last_dim=*/false, ctx.emit_diagnostics);
    facts.can_bulk_store =
        CheckBulkStore(op, ctx.target, analyzer,
                       /*check_last_dim=*/true, ctx.emit_diagnostics);
  }

  facts.can_cp_async = CheckCPAsyncCopy(op, ctx.target, layout_map, analyzer);
  facts.can_ldsm = CheckLDSMCopy(op, ctx.target);
  facts.can_stsm = CheckSTSMCopy(op, ctx.target);
  facts.can_tmem_load = CheckTMemLoad(op, ctx.target);
  facts.can_tmem_store = CheckTMemStore(op, ctx.target);
  if (facts.tma_descriptor_base_is_device_bound) {
    // BulkLoad1D/BulkStore1D pass the device-computed address directly and do
    // not create a host-side TensorMap descriptor, so keep those facts intact.
    facts.can_bulk_load = false;
    facts.can_bulk_store = false;
    facts.can_bulk_load_ignore_last_dim = false;
    facts.can_bulk_store_ignore_last_dim = false;
  }
  return facts;
}

} // namespace

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  // tile::gather4 / scatter4 markers take precedence over generic TMA paths.
  // The IR carries explicit row indices via annotations and must always be
  // lowered through LowerBulkCopyGather4 (no fallback path makes sense).
  if (GetBoolAnnotation(op, "is_gather4")) {
    if (GetTmaDescriptorBaseIsDeviceBound(op)) {
      return Unsupported(MakeTmaUnavailableReason(op));
    }
    return Supported(CopyInst::kBulkLoadGather4);
  }
  if (GetBoolAnnotation(op, "is_scatter4")) {
    if (GetTmaDescriptorBaseIsDeviceBound(op)) {
      return Unsupported(MakeTmaUnavailableReason(op));
    }
    return Supported(CopyInst::kBulkStoreScatter4);
  }

  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (facts.cluster_mask != 0) {
    if (facts.can_bulk_load) {
      return Supported(CopyInst::kBulkLoad);
    }
    std::ostringstream oss;
    oss << "cluster_mask=0x" << std::hex << facts.cluster_mask
        << " requires descriptor-based TMA (kBulkLoad), but the copy does not "
           "meet TMA bulk-load constraints. src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
    return Unsupported(oss.str());
  }

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

  if (facts.prefer_instruction == PreferredCopyInstruction::kTMA) {
    if (facts.disable_tma) {
      return Unsupported("T.copy prefer_instruction=\"tma\" conflicts with "
                         "disable_tma=true.");
    }
    if (facts.pass_context_disables_tma) {
      return Unsupported("T.copy prefer_instruction=\"tma\" conflicts with "
                         "pass config tl.disable_tma_lower=true.");
    }
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/true,
                      /*check_last_dim=*/true);
    return inst == CopyInst::kInvalid
               ? Unsupported("T.copy prefer_instruction=\"tma\" could not be "
                             "honored: " +
                             facts.tma_unavailable_reason)
               : Supported(inst);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kCPAsync) {
    if (!IsAutoAsyncCopyEnabled(/*default_enabled=*/true)) {
      return Unsupported(
          "T.copy prefer_instruction=\"cp_async\" conflicts with "
          "pass config tl.enable_async_copy=false.");
    }
    return facts.can_cp_async
               ? Supported(CopyInst::kCPAsync)
               : Unsupported("T.copy prefer_instruction="
                             "\"cp_async\" could not be honored: " +
                             facts.async_unavailable_reason);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kSync) {
    return Supported(SelectSyncLikeInst(facts));
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

CopyInstSelection ClassifyWarpSpecializedCopy(const CopyNode &op,
                                              Target target) {
  CopyAnalysisContext ctx(target);

  if (IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst)) {
    return SelectCopyInstForLowering(op, ctx);
  }

  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target) {
    return Supported(CopyInst::kNormal);
  }

  if (facts.cluster_mask != 0) {
    return facts.can_bulk_load ? Supported(CopyInst::kBulkLoad)
                               : Unsupported(facts.tma_unavailable_reason);
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

  // Honor prefer_instruction like SelectCopyInstForLowering, restricted to
  // producer-side loads.
  if (facts.prefer_instruction == PreferredCopyInstruction::kTMA) {
    if (facts.disable_tma) {
      return Unsupported("T.copy prefer_instruction=\"tma\" conflicts with "
                         "disable_tma=true.");
    }
    if (facts.pass_context_disables_tma) {
      return Unsupported("T.copy prefer_instruction=\"tma\" conflicts with "
                         "pass config tl.disable_tma_lower=true.");
    }
    CopyInst inst =
        SelectTmaInst(facts, /*allow_load=*/true, /*allow_store=*/false,
                      /*check_last_dim=*/true);
    return inst == CopyInst::kInvalid
               ? Unsupported("T.copy prefer_instruction=\"tma\" could not be "
                             "honored: " +
                             facts.tma_unavailable_reason)
               : Supported(inst);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kCPAsync) {
    if (!IsAutoAsyncCopyEnabled(/*default_enabled=*/true)) {
      return Unsupported(
          "T.copy prefer_instruction=\"cp_async\" conflicts with "
          "pass config tl.enable_async_copy=false.");
    }
    return facts.can_cp_async
               ? Supported(CopyInst::kCPAsync)
               : Unsupported("T.copy prefer_instruction="
                             "\"cp_async\" could not be honored: " +
                             facts.async_unavailable_reason);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kSync) {
    return Supported(SelectSyncLikeInst(facts));
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
  CopyAnalysisContext ctx(target);
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  if (!facts.cuda_like_target || facts.explicit_tma || facts.cluster_mask ||
      facts.prefer_instruction == PreferredCopyInstruction::kTMA ||
      facts.prefer_instruction == PreferredCopyInstruction::kSync) {
    return false;
  }
  if (!facts.explicit_cp_async && !facts.no_implicit_async_commit_wait &&
      !IsAutoAsyncCopyEnabled(true))
    return false;
  return facts.can_cp_async;
}

} // namespace cuda
} // namespace tl
} // namespace tvm
