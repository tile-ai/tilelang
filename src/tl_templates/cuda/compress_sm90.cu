#include <torch/extension.h>

#include <iostream>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

#define CUTLASS_CHECK(status)                                             \
  {                                                                       \
    cutlass::Status error = status;                                       \
    if (error != cutlass::Status::kSuccess) {                             \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                << " at: " << __LINE__ << std::endl;                      \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  }

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

using ElementA = cutlass::half_t;
using ElementE = unsigned char;
using LayoutTagA = cutlass::layout::RowMajor;

using ProblemShape = Shape<int, int, int, int>;

using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
using StrideE = StrideA;

using SparseConfig = cutlass::Sm90GemmSparseConfig<
    cute::sparse_elem<2, ElementA>, cute::SM90::GMMA::Major::K,
    cute::sparse_elem<8, ElementE>, cute::C<128> >;

using CompressorUtility =
    cutlass::transform::kernel::StructuredSparseCompressorUtility<
        ProblemShape, ElementA, LayoutTagA, SparseConfig>;

using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    ProblemShape, ElementA, LayoutTagA, SparseConfig, cutlass::arch::Sm90>;

using Compressor =
    cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

std::tuple<torch::Tensor, torch::Tensor> compress_sm90(torch::Tensor A) {
  assert(A.dim() == 2);
  int M = A.size(0);
  int N = -1;  // not used
  int K = A.size(1);
  int L = 1;
  ProblemShape problem_shape = make_tuple(M, N, K, L);
  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));

  CompressorUtility compressor_utility(problem_shape, stride_A);
  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  StrideE stride_E =
      cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(ME, KE, L));

  torch::Tensor A_compressed = torch::zeros(
      {M, KC}, torch::TensorOptions().dtype(torch::kHalf).device(A.device()));

  torch::Tensor E = torch::zeros(
      {ME, KE}, torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename Compressor::Arguments arguments{problem_shape,
                                           {
                                               A.data_ptr(),
                                               stride_A,
                                               A_compressed.data_ptr(),
                                               E.data_ptr(),
                                           },
                                           {hw_info}};

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  return std::make_tuple(A_compressed, E);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress_sm90", torch::wrap_pybind_function(compress_sm90),
        "compress_sm90");
}
