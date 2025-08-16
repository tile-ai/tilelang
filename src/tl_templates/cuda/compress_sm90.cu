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
template<typename T, int BlockK, bool transposed>
std::tuple<torch::Tensor, torch::Tensor> compress_impl(torch::Tensor A) {
  using ElementA = T;
  using ElementE = uint8_t;
  using LayoutTagA = conditional_t<transposed, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using ProblemShape = cute::Shape<int, int, int, int>;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  using StrideE = StrideA;

  // NOTE: this is derived from sparse sm90 mma atoms
  // Ref: https://github.com/NVIDIA/cutlass/blob/dc4817921edda44a549197ff3a9dcf5df0636e7b/include/cute/atom/mma_traits_sm90_gmma_sparse.hpp
  using SparseE = conditional_t<(sizeof_bits_v<ElementA> == 32), cute::sparse_elem<4, ElementE>, cute::sparse_elem<8, ElementE>>;
  static constexpr GMMA::Major GmmaMajorA = transposed ? cute::SM90::GMMA::Major::MN : cute::SM90::GMMA::Major::K;
  using SparseConfig = cutlass::Sm90GemmSparseConfig<
      cute::sparse_elem<2, ElementA>, GmmaMajorA,
      SparseE, cute::C<BlockK>>;

  using CompressorUtility =
      cutlass::transform::kernel::StructuredSparseCompressorUtility<
          ProblemShape, ElementA, LayoutTagA, SparseConfig>;

  using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
      ProblemShape, ElementA, LayoutTagA, SparseConfig, cutlass::arch::Sm90>;

  using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  TORCH_CHECK(A.is_contiguous(), "A need to be contiguous");
  TORCH_CHECK(A.dim() == 2, "Might support batch dim in the future ");

  int M = -1;
  int K = -1;
  int N = -1;  // not used, but required for config
  int L = 1;
  if constexpr(transposed) {
    M = A.size(1);
    K = A.size(0);
  } else {
    M = A.size(0);
    K = A.size(1);
  }

  ProblemShape problem_shape = make_tuple(M, N, K, L);
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));

  CompressorUtility compressor_utility(problem_shape, stride_A);
  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  StrideE stride_E = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(ME, KE, L));
  auto dtype = A.dtype().toScalarType();
  torch::Tensor A_compressed = torch::zeros(KC * M,
        torch::TensorOptions().dtype(dtype).device(A.device()));
  torch::Tensor E = torch::zeros({ME, KE},
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = A.device().index();
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

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

  if constexpr (transposed) {
    return std::make_tuple(A_compressed.view({KC, M}), E);
  } else {
    return std::make_tuple(A_compressed.view({M, KC}), E);
  }
}

// block <= 128
// Ref https://github.com/NVIDIA/cutlass/blob/c2ad7c5b20f131c4ba33601860f1da3f9c9df0f3/include/cutlass/gemm/collective/builders/sm90_sparse_gmma_builder.inl#L145-L146
#define DISPATCH_BLOCK_K(TYPE, BLOCK_K, FACTOR, TENSOR, TRANSPOSED)                                        \
  [&]() -> std::tuple<torch::Tensor, torch::Tensor> {                                                      \
    switch (BLOCK_K) {                                                                                     \
      case int(32 * FACTOR): return compress_impl<TYPE, int(32 * FACTOR), TRANSPOSED>(TENSOR);             \
      case int(64 * FACTOR): return compress_impl<TYPE, int(64 * FACTOR), TRANSPOSED>(TENSOR);             \
      case int(128 * FACTOR): return compress_impl<TYPE, int(128 * FACTOR), TRANSPOSED>(TENSOR);           \
      default:                                                                                             \
        TORCH_CHECK(false, "Unsupported block_k: ", BLOCK_K);                                              \
    }                                                                                                      \
  }()

#define DISPATCH_CONTIGUOUS(TRANSPOSED)                                                                    \
  [&]() -> std::tuple<torch::Tensor, torch::Tensor> {                                                      \
    switch (dtype) {                                                                                       \
      case torch::kFloat32:                                                                                \
        return DISPATCH_BLOCK_K(float, block_k, 0.5, A, TRANSPOSED);                                       \
      case torch::kFloat16:                                                                                \
      case torch::kBFloat16:                                                                               \
        return DISPATCH_BLOCK_K(cute::half_t, block_k, 1, A, TRANSPOSED);                                  \
      case torch::kFloat8_e4m3fn:                                                                          \
        return DISPATCH_BLOCK_K(cute::float_e4m3_t, block_k, 2, A, TRANSPOSED);                            \
      case torch::kFloat8_e5m2:                                                                            \
        return DISPATCH_BLOCK_K(cute::float_e5m2_t, block_k, 2, A, TRANSPOSED);                            \
      case torch::kChar:                                                                                   \
        return DISPATCH_BLOCK_K(int8_t, block_k, 2, A, TRANSPOSED);                                        \
      case torch::kByte:                                                                                   \
        return DISPATCH_BLOCK_K(uint8_t, block_k, 2, A, TRANSPOSED);                                       \
      default:                                                                                             \
        TORCH_CHECK(false, "Unsupported dtype");                                                           \
    }                                                                                                      \
  }()

std::tuple<torch::Tensor, torch::Tensor> compress_sm90(torch::Tensor A, int64_t block_k, bool transposed) {
  auto dtype = A.dtype().toScalarType();
  return transposed ? DISPATCH_CONTIGUOUS(true) : DISPATCH_CONTIGUOUS(false);
}

#undef DISPATCH_BLOCK_K
#undef DISPATCH_CONTIGUOUS

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress_sm90", torch::wrap_pybind_function(compress_sm90),
        "compress_sm90");
}
