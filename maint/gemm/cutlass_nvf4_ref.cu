#include <torch/extension.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace {

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using ElementPairA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPairB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

static constexpr int AlignmentA = 16 * 8 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using TileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

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
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

template <typename T>
struct Dummy {
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using GemmKernel = typename Dummy<void>::GemmKernel;
using Gemm = typename Dummy<void>::Gemm;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

void check_tensor(torch::Tensor const& tensor, char const* name, torch::DeviceType device) {
  TORCH_CHECK(tensor.device().type() == device, name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void cutlass_nvf4_gemm_128x128x256(torch::Tensor A, torch::Tensor B,
                                   torch::Tensor SFA, torch::Tensor SFB,
                                   torch::Tensor C, torch::Tensor D) {
#if !(defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))
  TORCH_CHECK(false, "CUTLASS was not compiled with SM120/SM121 block-scale MMA support");
#else
  constexpr int M = 128;
  constexpr int N = 128;
  constexpr int K = 256;

  check_tensor(A, "A", torch::kCUDA);
  check_tensor(B, "B", torch::kCUDA);
  check_tensor(SFA, "SFA", torch::kCUDA);
  check_tensor(SFB, "SFB", torch::kCUDA);
  check_tensor(C, "C", torch::kCUDA);
  check_tensor(D, "D", torch::kCUDA);

  TORCH_CHECK(A.numel() == M * K / 2, "A must contain packed NVF4 bytes for 128x256");
  TORCH_CHECK(B.numel() == N * K / 2, "B must contain packed NVF4 bytes for 128x256");
  TORCH_CHECK(SFA.numel() == M * (K / 16), "SFA must be CUTLASS-layout UE4M3 bytes");
  TORCH_CHECK(SFB.numel() == N * (K / 16), "SFB must be CUTLASS-layout UE4M3 bytes");
  TORCH_CHECK(C.numel() == M * N, "C must be 128x128 f32");
  TORCH_CHECK(D.numel() == M * N, "D must be 128x128 f32");

  auto problem = cute::make_shape(M, N, K, 1);
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<ElementA*>(A.data_ptr()), stride_A,
       reinterpret_cast<ElementB*>(B.data_ptr()), stride_B,
       reinterpret_cast<ElementPairA::ScaleFactorType*>(SFA.data_ptr()), layout_SFA,
       reinterpret_cast<ElementPairB::ScaleFactorType*>(SFB.data_ptr()), layout_SFB},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC*>(C.data_ptr()), stride_C,
       reinterpret_cast<ElementD*>(D.data_ptr()), stride_D}};

  Gemm gemm;
  auto status = gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS can_implement failed: ",
              cutlassGetStatusString(status));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

  status = gemm.initialize(arguments, workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS initialize failed: ",
              cutlassGetStatusString(status));

  status = gemm.run();
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS run failed: ",
              cutlassGetStatusString(status));
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess, "CUTLASS kernel failed");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cutlass_nvf4_gemm_128x128x256", &cutlass_nvf4_gemm_128x128x256,
        "CUTLASS SM120 NVF4 block-scaled GEMM 128x128x256");
}
