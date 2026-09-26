/*!
 * \file tl/cuda/op/gemm_blockscaled.cc
 * \brief CUDA instruction selection for block-scaled GEMM.
 */

#include "op/gemm_blockscaled.h"

#include <cstdint>
#include <utility>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op_attr_types.h>

#include "cuda/target_utils.h"
#include "op/tcgen5_meta.h"
#include "op/utils.h"
#include "span_utils.h"

namespace tvm {
namespace tl {
namespace cuda {

// Instruction keys resolved by the Python registry
// (tilelang/cuda/op/gemm/__init__.py) to block-scaled implementation classes.
constexpr const char *kCudaTCGEN05BlockScaled = "cuda.tcgen05.blockscaled";
constexpr const char *kCudaMMABlockScaled = "cuda.mma.blockscaled";

TVM_REGISTER_OP("tl.tileop.tcgen05_gemm_blockscaled")
    .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",
                                        "tcgen05_gemm_blockscaled")
    .set_attr<OpBuilderFunc>(
        "TLOpBuilder",
        [](ffi::Array<PrimExpr> args,
           ffi::Map<ffi::String, ffi::ObjectRef> annotations) {
          annotations.Set("is_tcgen05", IntImm(DataType::Int(32), 1));
          return GemmBlockScaled(std::move(args), std::move(annotations));
        })
    .set_num_inputs(-1)
    .set_attr<tirx::TCallEffectKind>("TCallEffectKind",
                                     Integer(tirx::CallEffectKind::kOpaque));

namespace {

ffi::String SelectBlockScaledGemmInst(const GemmBlockScaled &op,
                                      int /*block_size*/,
                                      const Target &target) {
  if (op->isWgmma_) {
    LOG(FATAL) << "Block-scaled GEMM does not support WGMMA lowering."
               << SpanHintSuffix({op->a_->span, op->b_->span, op->c_->span});
  }

  bool use_2cta = false;
  if (auto value = op->annotations_.Get("use_2cta")) {
    const auto *imm = value.value().as<tirx::IntImmNode>();
    ICHECK(imm) << "use_2cta annotation must be an IntImmNode";
    use_2cta = imm->value != 0;
  }

  // The block-scaled TCGEN05 emitter implements shared/shared operands only;
  // dense TCGEN05 also accepts TMEM A and therefore has a different contract.
  bool shared_operands = IsSharedBuffer(op->a_) && IsSharedBuffer(op->b_);
  if (TargetIsSm100(target) && shared_operands && IsTmemBuffer(op->c_) &&
      GetTCGEN5MMAMeta(op->m_, op->n_, op->k_, op->a_->dtype, op->c_->dtype)
          .first) {
    return kCudaTCGEN05BlockScaled;
  }

  bool requires_tcgen05 = op->isTcgen05_ || use_2cta;
  bool matches_sm120_mma =
      TargetIsSM120(target) &&
      (IsSharedBuffer(op->a_) || IsFragmentBuffer(op->a_)) &&
      IsSharedBuffer(op->b_) && IsFragmentBuffer(op->c_);
  if (!requires_tcgen05 && matches_sm120_mma) {
    return kCudaMMABlockScaled;
  }

  const char *requirement =
      requires_tcgen05
          ? "Blackwell SM100 TCGEN5MMA (A/B in shared memory, C in tensor "
            "memory) when is_tcgen05 or use_2cta is set"
          : "either Blackwell SM100 TCGEN5MMA (A/B in shared memory, C in "
            "tensor memory) or SM120 mma.sync (A in shared memory or a "
            "fragment, "
            "B in shared memory, C in a fragment)";
  LOG(FATAL) << "Block-scaled GEMM requires " << requirement
             << ", but got target=" << target << ", A scope=" << op->a_.scope()
             << ", B scope=" << op->b_.scope() << ", C scope=" << op->c_.scope()
             << ", M=" << op->m_ << ", N=" << op->n_ << ", K=" << op->k_ << "."
             << SpanHintSuffix({op->a_->span, op->b_->span, op->c_->span});
  return {};
}

bool MatchCudaBlockScaledGemmTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaGemmBlockScaled() {
  RegisterGemmBlockScaledImpl(GemmBlockScaledImpl{
      "cuda.GemmBlockScaled",
      MatchCudaBlockScaledGemmTarget,
      SelectBlockScaledGemmInst,
  });
  return true;
}

const bool cuda_gemm_blockscaled_registered = RegisterCudaGemmBlockScaled();

} // namespace

} // namespace cuda

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.get_tcgen5_blockscaled_instr_desc",
      [](int atom_m, int atom_n, DataType a_dtype, DataType b_dtype,
         bool a_is_k_major, bool b_is_k_major, int scale_in_a, int scale_in_b,
         int a_sf_id, int b_sf_id) {
        uint32_t desc = GetTCGEN5BlockScaledInstrDesc(
            atom_m, atom_n, a_dtype, b_dtype, a_is_k_major, b_is_k_major,
            scale_in_a, scale_in_b, a_sf_id, b_sf_id);
        return Integer(static_cast<int64_t>(desc));
      });
}

} // namespace tl
} // namespace tvm
