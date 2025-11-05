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
#include "tcgen5_meta.h"

namespace tvm {
namespace tl {

using namespace tir;

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
 * @note If `kPack` is provided it must be 1; otherwise the constructor
 *       fails with an ICHECK (runtime assertion). No other validation is
 *       performed here.
 */
Gemm::Gemm(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

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
  node->clear_accum = args[9].as<PrimExpr>().value();
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
  node->mbarptr = args[16];
  if (node->mbarptr.as<CallNode>()) {
    node->mbar = vmap[GetVarFromAccessPtr(node->mbarptr)];
  } else {
    node->mbar = std::nullopt;
  }
  node->C_coords = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
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
  auto op = tvm::ffi::make_object<GemmNode>(*this);
  return Gemm(op);
}

bool GemmNode::AllowTCGEN5MMA(Target target) const {
  return TargetIsSm100(target) &&
         ((A.scope() == "shared.dyn" || A.scope() == "shared" ||
           A.scope() == "shared.tmem") &&
          (B.scope() == "shared.dyn" || B.scope() == "shared") &&
          C.scope() == "shared.tmem") &&
         GetTCGEN5MMAMeta(M, N, K, A->dtype, C->dtype).first;
}

bool GemmNode::AllowWGMMA(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && (this->M >= 64) && (num_warps % 4 == 0) &&
         CheckWGMMA();
}

GemmInst GemmNode::GetGemmInst(int block_size, Target target) const {
  bool allow_tcgen5mma = AllowTCGEN5MMA(target);
  bool allow_wgmma = AllowWGMMA(block_size, target);
  if (allow_tcgen5mma) {
    return GemmInst::kTCGEN5MMA;
  } else if (allow_wgmma) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target;
  }
}

std::pair<int, int> GemmWarpPolicyNode::ComputeWarpPartition(
    int M, int N, int block_size, Target target, GemmInst gemm_inst) const {
  int num_warps = block_size / TargetGetWarpSize(target);
  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    return {1, num_warps}; // TCGEN5MMA doesn't care about warp partitioning
  }

  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16; // Rows processed by a single warp
  int kNPerWarp = 8;            // Columns processed by a single warp
  if (TargetIsVolta(target)) {
    kNPerWarp = 16;
  }
  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (gemm_inst == GemmInst::kWGMMA) {
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
        << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
        << ", n_warp: " << n_warp << ", num_warps: " << num_warps;

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
      // m_per_warp and n_per_warp must be greater than 1
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      // m * n must equal num_warps
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
  auto s = target->GetAttr<tvm::ffi::String>("arch");
  ICHECK(s.has_value());
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
  auto [warp_m, warp_n] =
      policy->ComputeWarpPartition(M, N, block_size, T.target, gemm_inst);

  std::stringstream ss;
  std::string op_name;

  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    auto [can_use_tcgen5mma, meta] =
        GetTCGEN5MMAMeta(M, N, K, A->dtype, C->dtype);
    ICHECK(can_use_tcgen5mma);
    ICHECK(B.scope() == "shared.dyn" || B.scope() == "shared");
    ICHECK(C.scope() == "shared.tmem");
    ICHECK(mbar.has_value()) << "mbar must be provided for TCGEN5MMA";
    if (A.scope() == "shared.tmem") {
      op_name = "tl::tcgen5mma_gemm_ts";
    } else if (A.scope() == "shared.dyn" || A.scope() == "shared") {
      op_name = "tl::tcgen5mma_gemm_ss";
    } else {
      ICHECK(0)
          << "Unsupported A scope for TCGEN5MMA: "
          << A.scope(); // If this is triggered, it means Tilelang has bugs.
    }
    ICHECK(wg_wait == -1)
        << "Currently only wg_wait == -1 is supported for TCGEN5MMA. Please "
           "use "
           "wg_wait = -1 and manually synchronize with mbarrier.";

    std::string accum_dtype = "";
    if (C->dtype.is_float()) {
      if (C->dtype.bits() == 32) {
        accum_dtype = "float";
      }
    }
    ICHECK(!accum_dtype.empty())
        << "Unsupported C dtype for TCGEN5MMA: " << C->dtype;
    ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
    ss << meta.atom_m << ", " << meta.atom_n << ", " << meta.atom_k << ", ";
    ss << trans_A << ", " << trans_B << ", ";
    ss << accum_dtype;
    ss << ">";

    auto C_buffer = T.buffer_remap.count(C) ? T.buffer_remap[C] : C;
    Array<PrimExpr> new_args;
    new_args.push_back(StringImm(ss.str()));
    new_args.push_back(Aptr);
    new_args.push_back(Bptr);
    new_args.push_back(BufferLoad(C_buffer, C_coords));
    new_args.push_back(mbarptr);
    new_args.push_back(clear_accum);
    auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);

    // Since TCGEN5MMA atoms provided by CUTLASS always have an internal
    // `elect_one_sync()`, we check if we are calling it using full warps
    constexpr int warp_size = 32;
    ICHECK(
        analyzer->CanProveEqual(FloorMod(T.thread_bounds->min, warp_size), 0) &&
        analyzer->CanProveEqual(FloorMod(T.thread_bounds->extent, warp_size),
                                0))
        << "TCGEN5MMA requires thread bounds to be multiples of warp size (32) "
           "and aligned to warps.";
    if (analyzer->CanProveEqual(T.thread_bounds->extent, warp_size)) {
      // If the thread bounds is exactly one warp, we can use the original call
      return Evaluate(new_call);
    } else {
      // Add an if-else clause
      auto tcgen5mma_call =
          IfThenElse(EQ(FloorDiv(T.thread_var, warp_size),
                        FloorDiv(T.thread_bounds->min, warp_size)),
                     Evaluate(new_call));
      return tcgen5mma_call;
    }
  }

  if (A.scope() == "local.fragment") {
    ICHECK(B.scope() != "local.fragment");
    ICHECK(!trans_A)
        << "gemm_rs requires the A operand to be in non-transposed layout.";
    op_name = "tl::gemm_rs";
  } else if (B.scope() == "local.fragment") {
    op_name = "tl::gemm_sr";
  } else {
    op_name = "tl::gemm_ss";
  }
  ICHECK(C.scope() == "local.fragment");

  ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << trans_A << ", " << trans_B;
  auto clear_accum_bool = clear_accum.as<Bool>();
  ICHECK(clear_accum_bool.has_value())
      << "clear_accum must be a constant Bool type, got " << clear_accum;
  ss << ", " << bool(clear_accum_bool.value());
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

  // Emit wg_wait if necessary
  if (TargetIsHopper(T.target)) {
    if (wg_wait != 0) {
      ss << ", " << wg_wait;
    }
  } else if (TargetIsSm100(T.target)) {
    // NOTE On sm100, only the leading thread issues the TCGEN5MMA instruction
    // but all threads need to wait, so we emit another statement for cases
    // where wg_wait == 0.
    ICHECK(wg_wait == 0 || wg_wait == -1)
        << "wg_wait must be 0 or -1 for Sm100";
  } else {
    ICHECK(wg_wait == 0)
        << "wg_wait must be 0 for non-Hopper and non-Sm100 targets";
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
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  GemmInst gemm_inst = GetGemmInst(block_size, T.target);
  auto [warp_m, warp_n] =
      policy->ComputeWarpPartition(M, N, block_size, T.target, gemm_inst);
  if (TargetIsVolta(T.target)) {
    ICHECK(C.scope() == "local.fragment")
        << "Volta gemm only supports C in local.fragment scope, got "
        << C.scope();
    auto fragment =
        makeGemmVoltaFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      results.Set(A, makeGemmVoltaABLayout(*as_const_int(A->shape[dim_A - 2]),
                                           *as_const_int(A->shape[dim_A - 1]),
                                           true, !trans_A));
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
                                         false, trans_B));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target) ||
             TargetIsSM120(T.target) ||
             (TargetIsSm100(T.target) && gemm_inst == GemmInst::kMMA)) {
    ICHECK(C.scope() == "local.fragment")
        << "MMA only supports C in local.fragment scope, got " << C.scope();

    auto fragment =
        makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      results.Set(A,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   A->dtype.bits(), !trans_A));
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
                                   B->dtype.bits(), trans_B));
    } else if (B.scope() == "local.fragment") {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    ICHECK(C.scope() == "local.fragment")
        << (gemm_inst == GemmInst::kWGMMA ? "WGMMA " : "MMA ")
        << "only supports C in local.fragment scope, got " << C.scope();
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
                                       A->dtype.bits(), !trans_A)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 A->dtype.bits(), !trans_A);
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
                                       B->dtype.bits(), trans_B)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 B->dtype.bits(), trans_B);
      results.Set(B, ABLayout);
    } else {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    }
  } else if (gemm_inst == GemmInst::kTCGEN5MMA) {
    ICHECK(C.scope() == "shared.tmem")
        << "TCGEN5MMA only supports C in shared.tmem scope, got " << C.scope();
    ICHECK(A.scope() == "shared.dyn" || A.scope() == "shared")
        << "Current TCGEN5MMA only supports A in shared.dyn scope";
    auto [can_use_tcgen5mma, meta] =
        GetTCGEN5MMAMeta(M, N, K, A->dtype, C->dtype);
    ICHECK(can_use_tcgen5mma);
    {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      results.Set(A, makeGemmABLayoutSm100(mat_stride, mat_continuous,
                                           mat_continuous, A->dtype.bits(),
                                           trans_A ? 1 : 2));
    }
    {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      const int64_t continuity = mat_continuous;
      results.Set(B,
                  makeGemmABLayoutSm100(mat_stride, mat_continuous, continuity,
                                        B->dtype.bits(), trans_B ? 2 : 1));
    }
    {
      Layout res;
      IterVar i = make_itervar("i", M);
      IterVar j = make_itervar("j", N);
      ICHECK(M % meta.atom_m == 0);
      PrimExpr atom_idx = FloorDiv(i, meta.atom_m) +
                          FloorDiv(j, meta.atom_n) * (M / meta.atom_m);
      PrimExpr ai = FloorMod(i, meta.atom_m); // "ai" means "atom_i"
      PrimExpr aj = FloorMod(j, meta.atom_n);
      if (meta.atom_m == 128) {
        // Layout D
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d)
        res = Layout(Array{i, j}, {ai, aj + atom_idx * meta.atom_n});
      } else if (meta.atom_m == 64) {
        // Layout E
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e)
        // since .ws variant is used About why we use .ws variant here, please
        // refer to gemm_sm100.h
        res = Layout(Array{i, j}, {FloorDiv(ai, 32) * 32 + FloorMod(ai, 32) +
                                       FloorDiv(aj, meta.atom_n / 2) * 64,
                                   FloorMod(aj, meta.atom_n / 2) +
                                       atom_idx * (meta.atom_n / 2)});
      } else if (meta.atom_m == 32) {
        // Layout G
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-g)
        res = Layout(
            Array{i, j},
            {FloorMod(ai, 32) + FloorDiv(aj, meta.atom_n / 4) * 32,
             FloorMod(aj, meta.atom_n / 4) + atom_idx * (meta.atom_n / 4)});
      } else {
        ICHECK(0);
      }
      results.Set(C, res);
    }
  } else if (TargetIsCDNA(T.target)) {
    ICHECK(C.scope() == "local.fragment")
        << "CDNA gemm (FMMA) only supports C in local.fragment scope, got "
        << C.scope();
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
                                            A->dtype.bits(), kPack, trans_A);
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

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, GemmInst gemm_inst) {
                          policy->ComputeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
}

} // namespace tl
} // namespace tvm
