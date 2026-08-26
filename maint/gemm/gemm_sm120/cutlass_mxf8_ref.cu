// CUTLASS SM120 MXFP8 (kind::mxf8f6f4) block-scaled GEMM reference used by
// correctness_evaluation_mxf8_vs_cutlass.py. All four {e4m3, e5m2} A/B
// pairings are instantiated; the tile is 128x128x128 with UE8M0 scales at
// 32-element granularity (Sm1xxBlockScaledConfig<32>).
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace {

using namespace cute;

using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

template <typename ElemA, typename ElemB> struct MxF8Gemm {
  using ElementPairA = cutlass::mx_float8_t<ElemA>;
  using ElementPairB = cutlass::mx_float8_t<ElemB>;
  static constexpr int AlignmentA = 16;
  static constexpr int AlignmentB = 16;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
          ElementPairA, LayoutA, AlignmentA, ElementPairB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

void check_tensor(torch::Tensor const &tensor, char const *name) {
  TORCH_CHECK(tensor.device().type() == torch::kCUDA, name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <typename ElemA, typename ElemB>
void run_mxf8_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor SFA,
                   torch::Tensor SFB, torch::Tensor C, torch::Tensor D,
                   int64_t m, int64_t n, int64_t k) {
  using Config = MxF8Gemm<ElemA, ElemB>;
  using Gemm = typename Config::Gemm;
  using GemmKernel = typename Config::GemmKernel;

  const int M = static_cast<int>(m);
  const int N = static_cast<int>(n);
  const int K = static_cast<int>(k);
  const int64_t m_pad = (m + 127) / 128 * 128;
  const int64_t n_pad = (n + 127) / 128 * 128;

  TORCH_CHECK(K % 128 == 0, "K must be a multiple of the CUTLASS tile K=128");

  check_tensor(A, "A");
  check_tensor(B, "B");
  check_tensor(SFA, "SFA");
  check_tensor(SFB, "SFB");
  check_tensor(C, "C");
  check_tensor(D, "D");

  TORCH_CHECK(A.numel() == m * k, "A must contain FP8 bytes");
  TORCH_CHECK(B.numel() == n * k, "B must contain FP8 bytes");
  TORCH_CHECK(SFA.numel() == m_pad * (k / 32),
              "SFA must be CUTLASS-layout UE8M0 bytes (128-row padded)");
  TORCH_CHECK(SFB.numel() == n_pad * (k / 32),
              "SFB must be CUTLASS-layout UE8M0 bytes (128-row padded)");
  TORCH_CHECK(C.numel() == m * n, "C must be MxN f32");
  TORCH_CHECK(D.numel() == m * n, "D must be MxN f32");

  auto problem = cute::make_shape(M, N, K, 1);
  auto stride_A = cutlass::make_cute_packed_stride(
      typename GemmKernel::StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(
      typename GemmKernel::StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(
      typename GemmKernel::StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(
      typename GemmKernel::StrideD{}, {M, N, 1});

  using Sm1xxBlkScaledConfig =
      typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<ElemA *>(A.data_ptr()), stride_A,
       reinterpret_cast<ElemB *>(B.data_ptr()), stride_B,
       reinterpret_cast<cutlass::float_ue8m0_t *>(SFA.data_ptr()), layout_SFA,
       reinterpret_cast<cutlass::float_ue8m0_t *>(SFB.data_ptr()), layout_SFB},
      {{1.0f, 0.0f},
       reinterpret_cast<ElementC *>(C.data_ptr()),
       stride_C,
       reinterpret_cast<ElementD *>(D.data_ptr()),
       stride_D}};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS can_implement failed: ", cutlassGetStatusString(status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  status = gemm.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS initialize failed: ", cutlassGetStatusString(status));

  status = gemm.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS run failed: ", cutlassGetStatusString(status));
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUTLASS kernel failed");
}

} // namespace

void cutlass_mxf8_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor SFA,
                       torch::Tensor SFB, torch::Tensor C, torch::Tensor D,
                       int64_t m, int64_t n, int64_t k, bool a_is_e4m3,
                       bool b_is_e4m3) {
#if !(defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||                             \
      defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))
  TORCH_CHECK(
      false,
      "CUTLASS was not compiled with SM120/SM121 block-scale MMA support");
#else
  if (a_is_e4m3 && b_is_e4m3) {
    run_mxf8_gemm<cutlass::float_e4m3_t, cutlass::float_e4m3_t>(A, B, SFA, SFB,
                                                                C, D, m, n, k);
  } else if (a_is_e4m3 && !b_is_e4m3) {
    run_mxf8_gemm<cutlass::float_e4m3_t, cutlass::float_e5m2_t>(A, B, SFA, SFB,
                                                                C, D, m, n, k);
  } else if (!a_is_e4m3 && b_is_e4m3) {
    run_mxf8_gemm<cutlass::float_e5m2_t, cutlass::float_e4m3_t>(A, B, SFA, SFB,
                                                                C, D, m, n, k);
  } else {
    run_mxf8_gemm<cutlass::float_e5m2_t, cutlass::float_e5m2_t>(A, B, SFA, SFB,
                                                                C, D, m, n, k);
  }
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_mxf8_gemm", &cutlass_mxf8_gemm,
        "CUTLASS SM120 MXFP8 block-scaled GEMM (all {e4m3,e5m2} pairings)");
}
