/*!
 * \file tl/cpu/op/gemm.cc
 * \brief CPU implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "support/check.h"

#include "backend/common/target_utils.h"
#include "cpu/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace cpu {

namespace {

constexpr const char *kCPUScalar = "cpu.scalar";
constexpr const char *kCPUAMXBF16 = "cpu.amx.bf16";
constexpr const char *kCPUAMXInt8 = "cpu.amx.int8";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)block_size;
    if (TargetHasAMX(target)) {
      if (op.a_->dtype == DataType::BFloat(16) &&
          op.b_->dtype == DataType::BFloat(16) &&
          op.c_->dtype == DataType::Float(32)) {
        return kCPUAMXBF16;
      }
      const bool a_int8 =
          op.a_->dtype == DataType::Int(8) || op.a_->dtype == DataType::UInt(8);
      const bool b_int8 =
          op.b_->dtype == DataType::Int(8) || op.b_->dtype == DataType::UInt(8);
      if (a_int8 && b_int8 && op.c_->dtype == DataType::Int(32)) {
        return kCPUAMXInt8;
      }
    }
    return kCPUScalar;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)M;
    (void)N;
    (void)block_size;
    (void)target;
    (void)gemm_inst;
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }
};

} // namespace cpu

namespace {

bool MatchCPUGemmTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUGemm() {
  RegisterGemmImpl(GemmImpl{
      "cpu.Gemm",
      MatchCPUGemmTarget,
      cpu::Gemm::SelectInst,
      cpu::Gemm::ComputeWarpPartition,
      cpu::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool cpu_gemm_registered = RegisterCPUGemm();

} // namespace

} // namespace tl
} // namespace tvm
