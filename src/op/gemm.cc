/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Compute the prime factorization of an integer.
 *
 * Returns the prime factors of x in non-decreasing order by repeatedly dividing
 * out the smallest possible factor.
 *
 * @param x Integer to factorize. If x <= 1, an empty vector is returned.
 * @return std::vector<int> Prime factors of x (with multiplicity), in
 * non-decreasing order.
 */
static std::vector<int> toPrimeFactors(int x) {
  int i = 2;
  std::vector<int> result;
  while (x > 1) {
    if (x % i == 0) {
      x /= i;
      result.push_back(i);
    } else {
      i++;
    }
  }
  return result;
}

/**
 * @brief Construct a Gemm operator from serialized TL arguments and a buffer
 * map.
 *
 * This constructor deserializes operator parameters from `args` and resolves
 * buffer references via `vmap`, populating an internal GemmNode with:
 * - device pointers for A, B, C and their corresponding Buffer objects,
 * - transpose flags for A and B,
 * - matrix dimensions M, N, K,
 * - warp allocation policy and clear_accum flag,
 * - strides and memory offsets for A and B,
 * - optional kPack (must be 1 or 2) and optional wg_wait.
 *
 * The populated GemmNode is stored into the wrapper's internal `data_`.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
 * @param vmap Mapping from access pointer vars to Buffer objects used to
 *   resolve the Buffer corresponding to each pointer argument.
 *
 * @note If `kPack` is provided it must be 1 or 2; otherwise the constructor
 *       fails with an ICHECK (runtime assertion). No other validation is
 *       performed here.
 */
Gemm::Gemm(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<GemmNode> node = make_object<GemmNode>();

  node->Aptr = args[0];
  node->Bptr = args[1];
  node->Cptr = args[2];
  node->A = vmap[GetVarFromAccessPtr(node->Aptr)];
  node->B = vmap[GetVarFromAccessPtr(node->Bptr)];
  node->C = vmap[GetVarFromAccessPtr(node->Cptr)];
  node->trans_A = args[3].as<Bool>().value();
  node->trans_B = args[4].as<Bool>().value();
  node->M = args[5].as<IntImm>().value()->value;
  node->N = args[6].as<IntImm>().value()->value;
  node->K = args[7].as<IntImm>().value()->value;
  node->policy = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clear_accum = args[9].as<Bool>().value();
  node->stride_A = args[10].as<IntImm>().value()->value;
  node->stride_B = args[11].as<IntImm>().value()->value;
  node->offset_A = args[12].as<IntImm>().value()->value;
  node->offset_B = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack = args[14].as<IntImm>().value()->value;
    if (node->kPack != 1 && node->kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 15) {
    node->wg_wait = args[15].as<IntImm>().value()->value;
  }
  data_ = std::move(node);
}

/**
 * @brief Create a copy of this GemmNode as a TileOperator.
 *
 * Constructs a new GemmNode by copying the current node state and returns it
 * wrapped in a Gemm TileOperator.
 *
 * @return TileOperator A Gemm operator that owns a copy of this node.
 */
TileOperator GemmNode::Clone() const {
  auto op = make_object<GemmNode>(*this);
  return Gemm(op);
}

GemmNode::GemmInst GemmNode::GetGemmInst(int block_size, Target target) const {
  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  bool allow_wgmma = TargetIsHopper(target) && (this->M >= 64) &&
                     (num_warps % 4 == 0) && CheckWGMMA();
  if (allow_wgmma) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target->str();
  }
}

std::pair<int, int>
GemmWarpPolicyNode::ComputeWarpPartition(int M, int N, int block_size,
                                         Target target, bool use_wgmma) const {
  int num_warps = block_size / TargetGetWarpSize(target);
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16; // Rows processed by a single warp
  constexpr int kNPerWarp = 8;  // Columns processed by a single warp

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (use_wgmma) {
    ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128×k threads.";

    constexpr int kGroup = 4; // Number of warps in a warp-group

    m_warp = kGroup; // Initially, only one warp-group on M dimension
    n_warp = num_warps / m_warp; // Rest all on N dimension

    if (this->isFullRow()) {
      // Try to put as many warp-groups as possible on M dimension
      // (decreasing multiples of 4, ensuring divisibility by M)
      for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
        if (M % (cand * kMPerWarp) == 0) {
          m_warp = cand;
          n_warp = num_warps / m_warp;
          break;
        }
      }
    } else if (this->isFullCol()) {
      // Try to use warps on N dimension; if N is not divisible, split excess
      // groups to M
      int cand_n = n_warp;                 // Initially assume all on N
      if (N % (cand_n * kNPerWarp) != 0) { // N direction division fails
        int max_n = N / kNPerWarp;
        // Find a feasible n_warp from max possible downwards, ensuring
        // num_warps/n_warp is multiple of 4
        for (int n = std::min(cand_n, max_n); n >= 1; --n) {
          if (num_warps % n == 0 && (num_warps / n) % kGroup == 0) {
            n_warp = n;
            m_warp = num_warps / n_warp;
            break;
          }
        }
      }
    } else if (this->isSquare()) {
      // Exhaustive search, but m must be multiple of 4
      int max_m = M / kMPerWarp;
      int max_n = N / kNPerWarp;

      float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;

      float best_score = std::numeric_limits<float>::max();
      int best_m = kGroup, best_n = n_warp;

      for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
        if (num_warps % m)
          continue;
        int n = num_warps / m;
        if (n > max_n)
          continue;

        float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
        float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
        float score = std::abs(m_per_warp / n_per_warp - ideal);

        if (score < best_score) {
          best_score = score;
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
        << "m_warp * n_warp must equal num_warps";

    // Store the computed values in the object's member variables
    this->m_warp = m_warp;
    this->n_warp = n_warp;

    return {m_warp, n_warp};
  }

  if (this->isFullRow()) {
    // Try to partition M first
    m_warp = num_warps;
    n_warp = 1;

    // If M cannot be evenly divided by m_warp*16, try to split remaining warps
    // to N
    if (M % (m_warp * kMPerWarp) != 0) {
      // Calculate how many warps we can use for M
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      // Use remaining warps for N
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (this->isFullCol()) {
    // Try to partition N first
    m_warp = 1;
    n_warp = num_warps;

    // If N cannot be evenly divided by n_warp*8, try to split remaining warps
    // to M
    if (N % (n_warp * kNPerWarp) != 0) {
      // Calculate how many warps we can use for N
      int max_n_warps = N / kNPerWarp;
      n_warp = max_n_warps;
      // Use remaining warps for M
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (this->isSquare()) {
    // First calculate the maximum possible warps for each dimension
    int max_m_warps =
        M / kMPerWarp; // Each warp needs at least 16 elements in M

    // Calculate the ideal ratio of M/N warps based on the matrix dimensions
    float ideal_ratio = 1.0f;
    if (N > 0) {
      ideal_ratio = static_cast<float>(M) / N;
    }

    // Try to find the best balanced partition
    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();

    // Try all possible combinations that satisfy the constraints
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      // Calculate how balanced this partition is
      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
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

  // Store the computed values in the object's member variables
  this->m_warp = m_warp;
  this->n_warp = n_warp;

  return {m_warp, n_warp};
}

/**
 * @brief Checks whether WGMMA (warp-group MMA) can be used for this GEMM.
 *
 * Evaluates device-memory placement, data-type combinations, transpose flags,
 * and K divisibility constraints required for the Hopper WGMMA code path.
 *
 * The check returns true only when:
 * - B resides in shared memory ("shared" or "shared.dyn"); and
 * - (C, A, B) dtypes match one of the supported combinations below and K
 *   satisfies the required alignment; and
 * - for combinations that require specific orientations, A is not transposed
 *   and B is transposed.
 *
 * Supported combinations and constraints:
 * - C=float16:
 *   - A=float16, B=float16: K % 16 == 0
 *   - Various float8 mixes (e4m3/e5m2): require (!trans_A && trans_B) and K %
 * 32 == 0
 * - C=float32:
 *   - A=float16, B=float16: K % 16 == 0
 *   - A=bfloat16, B=bfloat16: K % 16 == 0
 *   - A=float32, B=float32: require (!trans_A && trans_B) and K % 8 == 0
 *   - Various float8 mixes: require (!trans_A && trans_B) and K % 32 == 0
 * - C=int32:
 *   - 8-bit integer combinations (Int8/UInt8): require (!trans_A && trans_B)
 * and K % 32 == 0
 *
 * @return true if WGMMA is supported for the current buffers, dtypes, and
 *         transpose/shape constraints; false otherwise.
 */
bool GemmNode::CheckWGMMA() const {
  if (B.scope() != "shared.dyn" && B.scope() != "shared") {
    return false;
  }

  if (C->dtype == DataType::Float(16)) {
    if (A->dtype == DataType::Float(16) && B->dtype == DataType::Float(16))
      return K % 16 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else
      return false;
  } else if (C->dtype == DataType::Float(32)) {
    if (A->dtype == DataType::Float(16) && B->dtype == DataType::Float(16))
      return K % 16 == 0;
    else if (A->dtype == DataType::BFloat(16) &&
             B->dtype == DataType::BFloat(16))
      return K % 16 == 0;
    else if (A->dtype == DataType::Float(32) && B->dtype == DataType::Float(32))
      return (!trans_A) && trans_B && K % 8 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else
      return false;
  } else if (C->dtype == DataType::Int(32)) {
    if (A->dtype == DataType::Int(8) && B->dtype == DataType::Int(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype == DataType::Int(8) && B->dtype == DataType::UInt(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype == DataType::UInt(8) && B->dtype == DataType::Int(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype == DataType::UInt(8) && B->dtype == DataType::UInt(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else
      return false;
  } else {
    return false;
  }
}

/**
 * @brief Parse and return the numeric GPU architecture from a Target's "arch"
 * attribute.
 *
 * Examines the target's "arch" string and, if it matches the pattern
 * "sm_<num>", returns <num> as an int. If the attribute is present but does not
 * match that pattern, returns 0.
 *
 * Preconditions: the target must have an "arch" attribute (this is checked via
 * ICHECK).
 *
 * @return int The parsed architecture number (e.g., 80 for "sm_80"), or 0 if
 * the arch string does not match "sm_<num>".
 */
static int GetArchInt(Target target) {
  int arch_int = 0;
  auto s = target->GetAttr<String>("arch");
  ICHECK(s.defined());
  std::string arch = s.value();
  if (arch.rfind("sm_", 0) == 0) {
    arch_int = std::stoi(arch.substr(3));
  } else {
    arch_int = 0;
  }
  return arch_int;
}

/**
 * @brief Lower the GEMM operator to a TL TIR call expression.
 *
 * Constructs a tl::gemm call string parameterized by M, N, K, warp partition,
 * transpose flags, accumulation clearing, target-specific stride/offset/kPack
 * and optional workgroup wait value, then returns an Evaluate(call) node
 * invoking tl::tl_gemm with the composed string and the A/B/C buffer handles.
 *
 * @param T Contains lowering context including thread bounds and target.
 * @param analyzer Optional arithmetic analyzer used by lowering (may be
 * nullptr).
 * @return Stmt A TIR statement representing the evaluated TL GEMM call.
 */
Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto block_size = *as_const_int(T.thread_bounds->extent);
  GemmInst gemm_inst = GetGemmInst(block_size, T.target);
  auto [warp_m, warp_n] = policy->ComputeWarpPartition(
      M, N, block_size, T.target, gemm_inst == GemmInst::kWGMMA);

  std::stringstream ss;
  std::string op_name = "tl::gemm_ss";
  if (A.scope() == "local.fragment") {
    ICHECK(B.scope() != "local.fragment");
    op_name = "tl::gemm_rs";
  } else if (B.scope() == "local.fragment") {
    op_name = "tl::gemm_sr";
  }
  ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << trans_A << ", " << trans_B;
  ss << ", " << clear_accum;
  if (TargetIsCuda(T.target) && (GetArchInt(T.target) >= 75)) {
    ss << ", " << stride_A << ", " << stride_B;
    ss << ", " << offset_A << ", " << offset_B;
  }
  if (TargetIsCDNA(T.target)) {
    // for cdna gemm, we need to specify kPack
    ss << ", " << kPack;
  } else if (TargetIsHopper(T.target)) {
    ss << ", " << (gemm_inst == GemmInst::kWGMMA ? "true" : "false");
  }
  if (wg_wait != 0) {
    ss << ", " << wg_wait;
  }
  ss << ">";

  auto new_call = Call(DataType::Handle(), tl::tl_gemm(),
                       Array<PrimExpr>{StringImm(ss.str()), Aptr, Bptr, Cptr});
  return Evaluate(new_call);
}

/**
 * @brief Infer and bind target-specific memory/layout mappings for A, B, and C.
 *
 * Infers per-buffer layouts (fragment or shared-memory layouts) for this GEMM
 * operator according to the target architecture, thread bounds, warp
 * partitioning, data types, and transpose flags, then binds fragment layouts
 * to the thread range when required.
 *
 * Preconditions:
 * - C.scope() == "local.fragment"
 *
 * Side effects:
 * - Marks layout inference as completed (sets completed_ = true).
 * - May abort via ICHECK on unsupported targets, invalid buffer scopes, or
 *   incompatible shape constraints.
 *
 * @param T Input layout-inference context (provides thread bounds and target).
 * @return LayoutMap mapping A, B, and C to their inferred layouts.
 */
LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  ICHECK(C.scope() == "local.fragment");
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  GemmInst gemm_inst = GetGemmInst(block_size, T.target);
  auto [warp_m, warp_n] = policy->ComputeWarpPartition(
      M, N, block_size, T.target, gemm_inst == GemmInst::kWGMMA);

  if (TargetIsVolta(T.target)) {
    auto fragment =
        makeGemmVoltaFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      results.Set(A, makeGemmVoltaABLayout(*as_const_int(A->shape[dim_A - 2]),
                                           *as_const_int(A->shape[dim_A - 1]),
                                           true, trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      ICHECK(trans_A == false);
      auto fragment = makeGemmVoltaFragmentA(M, N, K, M / warp_m, N / warp_n);
      results.Set(A, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }

    ICHECK(B.scope() == "shared" || B.scope() == "shared.dyn");
    int dim_B = B->shape.size();
    results.Set(B, makeGemmVoltaABLayout(*as_const_int(B->shape[dim_B - 2]),
                                         *as_const_int(B->shape[dim_B - 1]),
                                         false, trans_B ? 2 : 1));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target) ||
             TargetIsSM120(T.target)) {
    auto fragment =
        makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      results.Set(A,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   A->dtype.bits(), trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      auto fragment = makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
                                        A->dtype.bits(), trans_A);
      results.Set(A, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      results.Set(B,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   B->dtype.bits(), trans_B ? 2 : 1));
    } else if (B.scope() == "local.fragment") {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    auto fragment =
        gemm_inst == GemmInst::kWGMMA
            ? makeGemmFragmentCHopper(M, N, M / warp_m, N / warp_n,
                                      C->dtype.bits())
            : makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      const int64_t continuity =
          trans_A ? 4 * mat_continuous / warp_m : mat_continuous;
      auto ABLayout =
          gemm_inst == GemmInst::kWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       A->dtype.bits(), trans_A ? 1 : 2)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 A->dtype.bits(), trans_A ? 1 : 2);
      results.Set(A, ABLayout);
    } else {
      auto fragment = makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
                                        A->dtype.bits(), trans_A);
      results.Set(A, fragment->BindThreadRange(thread_range));
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      const int64_t continuity =
          trans_B ? mat_continuous : mat_continuous / warp_n;
      auto ABLayout =
          gemm_inst == GemmInst::kWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       B->dtype.bits(), trans_B ? 2 : 1)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 B->dtype.bits(), trans_B ? 2 : 1);
      results.Set(B, ABLayout);
    } else {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    }
  } else if (TargetIsCDNA(T.target)) {
    auto fragment =
        makeGemmFragmentCCDNA(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(A->shape[dim_A - 2]),
          *as_const_int(A->shape[dim_A - 1]), A->dtype.bits(), kPack);
      results.Set(A, shared_layout);
    } else if (A.scope() == "local.fragment") {
      auto fragment = makeGemmFragmentACDNA(M, N, K, M / warp_m, N / warp_n,
                                            A->dtype.bits(), trans_A);
      results.Set(A, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(B->shape[dim_B - 2]),
          *as_const_int(B->shape[dim_B - 1]), B->dtype.bits(), kPack);

      results.Set(B, shared_layout);
    } else if (B.scope() == "local.fragment") {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
