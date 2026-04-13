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
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a Gemm operator from serialized TL arguments.
 *
 * Deserializes operator parameters from `args` and resolves buffer references,
 * populating an internal GemmNode with buffers, transpose flags, M/N/K,
 * warp policy, clear_accum, strides, offsets, optional kPack/wg_wait, and
 * optional mbarrier.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) internal wg_wait (Int),
 *      (optional) mbar (BufferLoad), cCoord_y (PrimExpr), cCoord_x (PrimExpr)]
 */
Gemm::Gemm(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[2], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, b_access, c_access});

  node->a_ = node->aRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[3].as<Bool>().value();
  node->transB_ = args[4].as<Bool>().value();
  node->m_ = args[5].as<IntImm>().value()->value;
  node->n_ = args[6].as<IntImm>().value()->value;
  node->k_ = args[7].as<IntImm>().value()->value;
  node->policy_ = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  node->strideA_ = args[10].as<IntImm>().value()->value;
  node->strideB_ = args[11].as<IntImm>().value()->value;
  node->offsetA_ = args[12].as<IntImm>().value()->value;
  node->offsetB_ = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack_ = args[14].as<IntImm>().value()->value;
    if (node->kPack_ != 1 && node->kPack_ != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 15) {
    node->wgWait_ = args[15].as<IntImm>().value()->value;
  }
  if (auto val = annotations.Get("is_wgmma")) {
    const auto *int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "is_wgmma annotation must be IntImmNode";
    node->isWgmma_ = int_val->value != 0;
  }
  if (auto val = annotations.Get("is_tcgen05")) {
    const auto *int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "is_tcgen05 annotation must be IntImmNode";
    node->isTcgen05_ = int_val->value != 0;
  }
  if (args.size() > 16 && args[16]->IsInstance<BufferLoadNode>()) {
    node->mbar_ = Downcast<BufferLoad>(args[16]);
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
  node->annotations_ = annotations;
  data_ = std::move(node);
}

AccessRegions GemmNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(bRegion_);
  if (!is_one(clearAccum_)) {
    result.reads.push_back(cRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
}

TileOperator GemmNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmNode>(*this);
  return Gemm(op);
}

bool GemmNode::allowTcgen5Mma(Target target) const {
  bool scope_ok = (IsSharedBuffer(a_) || a_.scope() == "shared.tmem") &&
                  IsSharedBuffer(b_) && c_.scope() == "shared.tmem";
  if (!TargetIsSm100(target) || !scope_ok)
    return false;
  // For TS variant (A from TMEM), use B's dtype as the input dtype
  DataType ab_dtype = (a_.scope() == "shared.tmem") ? b_->dtype : a_->dtype;
  return GetTCGEN5MMAMeta(m_, n_, k_, ab_dtype, c_->dtype).first;
}

bool GemmNode::allowWgmma(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && (this->m_ >= 64) && (num_warps % 4 == 0) &&
         checkWgmma();
}

GemmInst GemmNode::getGemmInst(int block_size, Target target) const {
  if (isWgmma_) {
    if (!allowWgmma(block_size, target)) {
      LOG(FATAL) << "T.wgmma_gemm() requires Hopper WGMMA lowering, but "
                    "constraints were not satisfied. Got target="
                 << target << ", A(scope=" << a_.scope()
                 << ", dtype=" << a_->dtype << "), B(scope=" << b_.scope()
                 << ", dtype=" << b_->dtype << "), C(scope=" << c_.scope()
                 << ", dtype=" << c_->dtype << "), M=" << m_ << ", N=" << n_
                 << ", K=" << k_ << ".";
    }
    return GemmInst::kWGMMA;
  }
  if (isTcgen05_) {
    if (!allowTcgen5Mma(target)) {
      LOG(FATAL) << "T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering, "
                    "but constraints were not satisfied. Got target="
                 << target << ", A(scope=" << a_.scope()
                 << ", dtype=" << a_->dtype << "), B(scope=" << b_.scope()
                 << ", dtype=" << b_->dtype << "), C(scope=" << c_.scope()
                 << ", dtype=" << c_->dtype << "), M=" << m_ << ", N=" << n_
                 << ", K=" << k_ << ".";
    }
    return GemmInst::kTCGEN5MMA;
  }
  bool allow_tcgen5mma = allowTcgen5Mma(target);
  bool allow_wgmma = allowWgmma(block_size, target);
  if (allow_tcgen5mma) {
    return GemmInst::kTCGEN5MMA;
  } else if (allow_wgmma) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsRDNA(target)) {
    return GemmInst::kWMMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else if (TargetIsCPU(target)) {
    return GemmInst::kScalar;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target->str();
    return GemmInst::kMMA;
  }
}

std::pair<int, int> GemmWarpPolicyNode::computeWarpPartition(
    int M, int N, int block_size, Target target, GemmInst gemm_inst) const {
  int num_warps = block_size / TargetGetWarpSize(target);
  if (gemm_inst == GemmInst::kTCGEN5MMA) {
    this->m_warp = 1;
    this->n_warp = num_warps;
    return {1, num_warps}; // TCGEN5MMA doesn't care about warp partitioning
  }

  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16; // Rows processed by a single warp
  int kNPerWarp = 8;            // Columns processed by a single warp
  if (TargetIsVolta(target)) {
    kNPerWarp = 16;
  } else if (TargetIsCDNA(target)) {
    kNPerWarp = 16;
  } else if (TargetIsRDNA(target)) {
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
 * Returns true only when B resides in shared memory and the (C, A, B) dtype
 * combination plus K alignment matches one of the supported WGMMA variants.
 */
bool GemmNode::checkWgmma() const {
  if (b_.scope() != "shared.dyn" && b_.scope() != "shared") {
    return false;
  }

  if (c_->dtype == DataType::Float(16)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else if (c_->dtype == DataType::Float(32)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::BFloat(16) &&
             b_->dtype == DataType::BFloat(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::Float(32) &&
             b_->dtype == DataType::Float(32))
      return (!transA_) && transB_ && k_ % 8 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else if (c_->dtype == DataType::Int(32)) {
    if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::UInt(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::UInt(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::UInt(8) && b_->dtype == DataType::UInt(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else {
    return false;
  }
}

Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.lower")) {
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    // NOTE(wt): Decide GemmInst and compute warp partition on Python side
    auto prim_func = Downcast<PrimFunc>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.layout_map, T.target,
             T.thread_bounds, T.thread_var, mbar_phase));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol =
        prim_func->attrs.GetAttr<tvm::ffi::String>("global_symbol");
    ICHECK(global_symbol.has_value());
    if (prim_func->body.as<BlockRealizeNode>()) {
      BlockRealize block_realize = Downcast<BlockRealize>(prim_func->body);
      auto block = block_realize->block;
      {
        BlockNode *n = block.CopyOnWrite();
        n->name_hint = global_symbol.value();
        n->annotations.Set(tl::attr::kLexicalAllocScope,
                           IntImm(DataType::Int(32), 1));
      }
      return BlockRealize(block_realize->iter_values, block_realize->predicate,
                          block);
    }
    // wrap with block realize node
    Map<String, ObjectRef> block_annotations;
    block_annotations.Set(tl::attr::kLexicalAllocScope,
                          IntImm(DataType::Int(32), 1));
    return BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/global_symbol.value(), prim_func->body,
              /*init=*/Optional<Stmt>(), /*alloc_buffers=*/{},
              /*match_buffers=*/{}, /*annotations=*/block_annotations));
  } else {
    LOG(FATAL) << "No lower function found for gemm";
    return Stmt();
  }
}

LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;

  if (const auto f = ffi::Function::GetGlobal("tl.gemm.infer_layout")) {
    results = Downcast<LayoutMap>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.target, T.thread_bounds));
    // Bind all fragment layouts with the provided thread range
    for (auto kv : results) {
      const Buffer &buf = kv.first;
      const Layout &layout = kv.second;
      if (auto frag = layout.as<Fragment>()) {
        results.Set(buf, frag.value()->BindThreadRange(T.thread_bounds));
      }
    }
  } else {
    LOG(FATAL) << "No infer layout function found for gemm";
  }

  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.wgmma_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "wgmma_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_wgmma",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.tcgen05_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tcgen05_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tcgen05",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, GemmInst gemm_inst) {
                          policy->computeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
  refl::GlobalDef().def("tl.GemmGetGemmInst",
                        [](Gemm gemm, int block_size, Target target) {
                          return gemm->getGemmInst(block_size, target);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.get_tcgen5_mma_meta", [](int M, int N, int K, DataType ab_dtype,
                                   DataType c_dtype, bool disable_2cta) {
        auto [success, meta] =
            GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype, disable_2cta);
        Array<Integer> result;
        if (success) {
          result.push_back(Integer(meta.atom_m));
          result.push_back(Integer(meta.atom_n));
          result.push_back(Integer(meta.atom_k));
          result.push_back(Integer(meta.enable_ws));
          result.push_back(Integer(meta.enable_2cta));
        }
        return result;
      });
  refl::GlobalDef().def(
      "tl.get_tcgen5_instr_desc",
      [](int atom_m, int atom_n, int atom_k, DataType ab_dtype,
         DataType c_dtype, bool a_is_k_major, bool b_is_k_major, int scale_in_a,
         int scale_in_b) {
        uint32_t desc = GetTCGEN5InstrDesc(atom_m, atom_n, atom_k, ab_dtype,
                                           c_dtype, a_is_k_major, b_is_k_major,
                                           scale_in_a, scale_in_b);
        return Integer(static_cast<int64_t>(desc));
      });
}

} // namespace tl
} // namespace tvm
