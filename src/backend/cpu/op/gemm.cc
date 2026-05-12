/*!
 * \file tl/backend/cpu/op/gemm.cc
 * \brief CPU implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace cpu {

namespace {

constexpr const char *kCPUScalar = "cpu.scalar";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
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

  static String InstructionKind(String gemm_inst) {
    (void)gemm_inst;
    return "scalar";
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
      cpu::Gemm::InstructionKind,
  });
  return true;
}

const bool cpu_gemm_registered = RegisterCPUGemm();

} // namespace

} // namespace tl
} // namespace tvm
