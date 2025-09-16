#include <cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h>
#include <stdio.h>

namespace tl {

static int const kSparse = 2;
template <typename T, typename Shape> struct ShapeCheck {
  static constexpr bool value = false;
};

template <typename Shape> struct ShapeCheck<cutlass::half_t, Shape> {
  static constexpr bool value =
      (Shape::kM % 32 == 0) && (Shape::kN % 32 == 0) && (Shape::kK % 32 == 0);
};

template <typename Shape> struct ShapeCheck<cutlass::bfloat16_t, Shape> {
  static constexpr bool value =
      ShapeCheck<cutlass::half_t, Shape>::value; // Same as half
};

template <typename Shape> struct ShapeCheck<int8_t, Shape> {
  static constexpr bool value =
      (Shape::kM % 16 == 0) && (Shape::kN % 16 == 0) && (Shape::kK % 64 == 0);
};

template <typename Shape> struct ShapeCheck<uint8_t, Shape> {
  static constexpr bool value =
      (Shape::kM % 16 == 0) && (Shape::kN % 16 == 0) && (Shape::kK % 64 == 0);
};

// ref:
// https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h
template <typename T> struct DispatchInstructionShape {
  static_assert(!std::is_same_v<T, T>,
                "Unsupported type for DispatchInstructionShape");
};

template <> struct DispatchInstructionShape<cutlass::half_t> {
  using Shape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

template <> struct DispatchInstructionShape<cutlass::bfloat16_t> {
  using Shape = cutlass::gemm::GemmShape<16, 8, 32>;
  using Operator = cutlass::arch::OpMultiplyAdd;
};

// TODO: Not supported for now
// template<>
// struct DispatchInstructionShape<cutlass::tfloat32_t> {
//   using Shape = cutlass::gemm::GemmShape<16, 8, 16>;
//   using Operator = cutlass::arch::OpMultiplyAdd;
// };

template <> struct DispatchInstructionShape<int8_t> {
  using Shape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

template <> struct DispatchInstructionShape<uint8_t> {
  using Shape = cutlass::gemm::GemmShape<16, 8, 64>;
  using Operator = cutlass::arch::OpMultiplyAddSaturate;
};

// TODO: Not supported for now
// template<>
// struct DispatchInstructionShape<cutlass::int4b_t> {
//   using Shape = cutlass::gemm::GemmShape<16, 8, 128>;
//   using Operator = cutlass::arch::OpMultiplyAddSaturate;
// };

template <typename T, bool transpose, int M, int K>
struct DispatchSharedMemoryLayoutA;

template <typename T, int M, int K>
struct DispatchSharedMemoryLayoutA<T, false, M, K> {
  using SmemLayoutA = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<T>::value, K / kSparse>;
};

template <typename T, int M, int K>
struct DispatchSharedMemoryLayoutA<T, true, M, K> {
  static int const Crosswise_A =
      cutlass::platform::min(int(128 / sizeof(T)), M);
  using SmemLayoutA = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<T>::value, Crosswise_A>;
};

template <typename T, bool transpose, int N, int K>
struct DispatchSharedMemoryLayoutB;

template <typename T, int N, int K>
struct DispatchSharedMemoryLayoutB<T, false, N, K> {
  static_assert(
      cutlass::sizeof_bits<T>::value != 8,
      "int8, uint8, float8 only support column major layout for matrix B");
  static int const Crosswise_B =
      cutlass::platform::min(int(128 / sizeof(T)), N);
  using SmemLayoutB = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      cutlass::sizeof_bits<T>::value, Crosswise_B>;
};

template <typename T, int N, int K>
struct DispatchSharedMemoryLayoutB<T, true, N, K> {
  static int const kCrosswiseB = (K > (1024 / cutlass::sizeof_bits<T>::value))
                                     ? (1024 / cutlass::sizeof_bits<T>::value)
                                     : K;
  using SmemLayoutB = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      cutlass::sizeof_bits<T>::value, kCrosswiseB>;
};

template <typename T> struct DispatchType {
  static_assert(std::is_same<T, void>::value, "Unsupported dtype");
};

template <> struct DispatchType<cutlass::half_t> {
  using Type = cutlass::half_t;
};

template <> struct DispatchType<cutlass::bfloat16_t> {
  using Type = cutlass::bfloat16_t;
};

template <> struct DispatchType<unsigned char> {
  using Type = uint8_t;
};

template <> struct DispatchType<signed char> {
  using Type = int8_t;
};

template <typename Shape, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  static_assert(Shape::kM % num_warp_m == 0);
  static_assert(Shape::kN % num_warp_n == 0);
  using ElementA = typename DispatchType<A_type_raw>::Type;
  using ElementB = typename DispatchType<B_type_raw>::Type;
  using ElementC = C_type_raw;

  static_assert(std::is_same_v<ElementA, ElementB>,
                "A and B are not the same type");
  static_assert(ShapeCheck<ElementA, Shape>::value,
                "Invalid shape for ElementA");

  using LayoutA =
      typename std::conditional_t<trans_A, cutlass::layout::ColumnMajor,
                                  cutlass::layout::RowMajor>;
  using LayoutB =
      typename std::conditional_t<trans_B, cutlass::layout::ColumnMajor,
                                  cutlass::layout::RowMajor>;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = Shape;
  using SmemLayoutA =
      typename DispatchSharedMemoryLayoutA<ElementA, trans_A,
                                           ThreadblockShape::kM,
                                           ThreadblockShape::kK>::SmemLayoutA;
  using SmemLayoutB =
      typename DispatchSharedMemoryLayoutB<ElementB, trans_B,
                                           ThreadblockShape::kN,
                                           ThreadblockShape::kK>::SmemLayoutB;

  using WarpShape = cutlass::gemm::GemmShape<ThreadblockShape::kM / num_warp_m,
                                             ThreadblockShape::kN / num_warp_n,
                                             ThreadblockShape::kK>;
  using InstructionShape = typename DispatchInstructionShape<ElementA>::Shape;
  using Operator = typename DispatchInstructionShape<ElementA>::Operator;
  static_assert(WarpShape::kK % InstructionShape::kK == 0,
                "K dimension must be divisible by instruction shape K.");

  // instruction/warp config
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
      cutlass::arch::SparseMma<InstructionShape, 32, ElementA,
                               cutlass::layout::RowMajor, ElementB,
                               cutlass::layout::ColumnMajor, ElementC,
                               cutlass::layout::RowMajor, Operator>,
      cutlass::MatrixShape<1, 1>>;
  using MmaWarp =
      cutlass::gemm::warp::SparseMmaTensorOp<WarpShape, ElementA, SmemLayoutA,
                                             ElementB, SmemLayoutB, ElementC,
                                             LayoutC, Policy>;
  static_assert(kSparse == MmaWarp::kSparse, "not 2:4 structured sparse");

  using SmemLayoutE = typename MmaWarp::LayoutE;
  static_assert(std::is_same_v<SmemLayoutE, cutlass::layout::ColumnMajor>,
                "Meta data layout must be ColumnMajor for sparse mma.");

  // other traits
  using FragmentA = typename MmaWarp::FragmentA;
  using FragmentB = typename MmaWarp::FragmentB;
  using FragmentC = typename MmaWarp::FragmentC;
  using FragmentE = typename MmaWarp::FragmentE;

  using IteratorA = typename MmaWarp::IteratorA;
  using IteratorB = typename MmaWarp::IteratorB;
  using IteratorE = typename MmaWarp::IteratorE;

  using TensorRefA = typename IteratorA::TensorRef;
  using TensorRefB = typename IteratorB::TensorRef;
  using TensorRefE = typename IteratorE::TensorRef;
  using ElementE = typename TensorRefE::Element;

  static int const kElementsPerElementE = MmaWarp::kElementsPerElementE;
  static_assert(kSparse == MmaWarp::kSparse, "not 2:4 structured sparse");

  using ShapeA = cutlass::MatrixShape<Shape::kM, Shape::kK / kSparse>;
  using ShapeB = cutlass::MatrixShape<Shape::kK, Shape::kN>;
  using ShapeE =
      cutlass::MatrixShape<Shape::kM * 2,
                           Shape::kK / kSparse / kElementsPerElementE / 2>;

  static int constexpr kKgroups = WarpShape::kK / InstructionShape::kK;

  template <typename E_type_raw>
  static CUTLASS_DEVICE void
  body(A_type_raw *pA, E_type_raw *pE, B_type_raw *pB, FragmentC &accum,
       const int warp_idx_m, const int warp_idx_n, const int lane_id) {
    MmaWarp mma_op;
    FragmentA frag_a;
    FragmentB frag_b;
    FragmentE frag_e;
    const TensorRefA ref_A(
        (ElementA *)pA,
        MmaWarp::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn}));
    const TensorRefE ref_E(
        (ElementE *)pE,
        MmaWarp::LayoutE::packed({ShapeE::kRow, ShapeE::kColumn}));
    const TensorRefB ref_B(
        (ElementB *)pB,
        MmaWarp::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn}));
    IteratorA iter_A(ref_A, lane_id);
    IteratorE iter_E(ref_E, lane_id);
    IteratorB iter_B(ref_B, lane_id);
    iter_A.add_tile_offset({warp_idx_m, 0});
    iter_E.add_tile_offset({warp_idx_m, 0});
    iter_B.add_tile_offset({0, warp_idx_n});
    if constexpr (clear_accum) {
      accum.clear();
    }
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups; ++k) {
      iter_A.load(frag_a);
      iter_E.load(frag_e);
      iter_B.load(frag_b);
      ++iter_A;
      ++iter_E;
      ++iter_B;
      mma_op(accum, frag_a, frag_b, accum, frag_e);
    }
  }
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, typename A_type,
          typename B_type, typename C_type, typename E_type>
TL_DEVICE void gemm_sp_ss(A_type *pA, B_type *pB, C_type *accum, E_type *pE) {
  using MMA =
      GemmTensorOp<cutlass::gemm::GemmShape<M, N, K>, num_warp_m, num_warp_n,
                   trans_A, trans_B, clear_accum, A_type, B_type, C_type>;
  using FragmentC = typename MMA::FragmentC;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  MMA::body(pA, pE, pB, *(FragmentC *)(accum), warp_id % num_warp_m,
            warp_id / num_warp_m, lane_id);
}

} // namespace tl
