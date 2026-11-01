/*!
 * \file tl/rocm/op/gemm.cc
 * \brief ROCm implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "op/utils.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "rocm/target_utils.h"

#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

namespace {

constexpr const char *kROCmMFMA = "rocm.mfma";
constexpr const char *kROCmWMMA = "rocm.wmma";

bool IsFullSharedTile(const BufferRegion &region, int rows, int cols) {
  const Buffer &buffer = region->buffer;
  if (!IsSharedBuffer(buffer) || buffer->shape.size() != 2 ||
      !buffer->strides.empty() || buffer->data_alignment < 16 ||
      buffer->data_alignment % 16 != 0) {
    return false;
  }
  return is_const_int(buffer->shape[0], rows) &&
         is_const_int(buffer->shape[1], cols) && is_zero(buffer->elem_offset) &&
         is_zero(region->region[0]->min) && is_zero(region->region[1]->min) &&
         is_const_int(region->region[0]->extent, rows) &&
         is_const_int(region->region[1]->extent, cols);
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 16;

  // Guard the callers that do not go through ComputeWarpPartition: a zero
  // warp count makes the modulo below divide by zero and kills the process.
  ICHECK_GT(num_warps, 0) << "num_warps must be positive, but got "
                          << num_warps;
  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (policy.IsFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.IsFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      int max_n_warps = N / kNPerWarp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.IsSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static ffi::String SelectInst(const GemmNode &op, int block_size,
                                Target target) {
    (void)op;
    (void)block_size;

    if (TargetIsCDNA(target)) {
      return kROCmMFMA;
    }
    if (TargetIsRDNA(target)) {
      return kROCmWMMA;
    }
    LOG(FATAL) << "Unsupported ROCm target for gemm: " << target->str();
    return kROCmMFMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, ffi::String gemm_inst) {
    (void)gemm_inst;
    const int warp_size = TargetRocmGetWarpSize(target);
    // A block narrower than one wavefront yields zero warps, which used to
    // reach ComputeDefaultWarpPartition and abort the process with SIGFPE on
    // its `M % (m_warp * kMPerWarp)`. Thread counts tuned for NVIDIA's 32-lane
    // warp hit this on CDNA, where the wavefront is 64 wide.
    ICHECK_GE(block_size, warp_size)
        << "T.gemm needs at least one full wavefront, but this block has only "
        << block_size << " threads while the wavefront size for "
        << target->str() << " is " << warp_size
        << ". Raise the kernel's thread count to a multiple of " << warp_size
        << ".";
    int num_warps = block_size / warp_size;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps);
  }

  static bool ReuseExistingSharedLayout(ffi::String gemm_inst,
                                        const tvm::tl::Gemm &op,
                                        const Buffer &buffer,
                                        const Layout &layout,
                                        const Target &target) {
    if (gemm_inst != kROCmWMMA || !TargetIsGfx1200(target) || op->transA_ ||
        op->transB_ || op->kPack_ != 1 ||
        op->a_->dtype != DataType::BFloat(16) ||
        op->b_->dtype != DataType::BFloat(16) ||
        op->c_->dtype != DataType::Float(32) || op->m_ % 16 != 0 ||
        op->n_ % 16 != 0 || op->k_ % 16 != 0 ||
        !IsFullSharedTile(op->aRegion_, op->m_, op->k_) ||
        !IsFullSharedTile(op->bRegion_, op->k_, op->n_) ||
        layout.as<FragmentNode>()) {
      return false;
    }
    // Limit reuse to gfx1200, where these operand loads have been validated.
    // gfx1200 loads eight BF16 values per lane. With NN row-major storage,
    // A's K slice is contiguous and 16-byte aligned, while B's K slice is
    // strided and the existing emitter scalarizes its loads. Require the
    // complete physical mapping, not just matching logical tile dimensions.
    // Transposes, mixed shared/register operands and leading stage axes keep
    // their existing inferred-layout requirements.
    return layout->IsEqual(MakeLinearLayout(buffer->shape).operator->());
  }
};

} // namespace rocm

namespace {

bool MatchROCmGemmTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmGemm() {
  RegisterGemmImpl(GemmImpl{
      "rocm.Gemm",
      MatchROCmGemmTarget,
      rocm::Gemm::SelectInst,
      rocm::Gemm::ComputeWarpPartition,
      rocm::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool rocm_gemm_registered = RegisterROCmGemm();

} // namespace

} // namespace tl
} // namespace tvm
