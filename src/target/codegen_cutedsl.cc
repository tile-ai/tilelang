/*!
 * \file target/codegen_cutedsl.cc
 */

#include "codegen_cutedsl.h"
#include "codegen_utils.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "arith/pattern_match.h"

namespace tvm {
namespace codegen {
namespace {

// The threshold of the loop extent to use cutlass.range_constexpr
// Higher values would lead to DSLOptimizationWarning:
// This static loop has 128 iterations, which may be very slow to compile,
//  consider using `cutlass.range(..., unroll_full=True)` instead.
const int64_t LOOP_UNROLL_THRESHOLD = 64;

void ReplaceAll(std::string &str, const std::string &from,
                const std::string &to) {
  ICHECK(!from.empty()) << "ReplaceAll(): `from` must be non-empty";
  auto pos = str.find(from);
  while (pos != std::string::npos) {
    str.replace(pos, from.size(), to);
    pos = str.find(from, pos + to.size());
  }
}

} // namespace

CodeGenTileLangCuTeDSL::CodeGenTileLangCuTeDSL() {
  // Read fastmath configuration from current PassContext
  auto pass_ctx = tvm::transform::PassContext::Current();

  // Read tl.enable_fast_math config, default to false
  enable_fastmath_ =
      pass_ctx->GetConfig<Bool>(tl::kEnableFastMath, Bool(false)).value();
}

std::string CodeGenTileLangCuTeDSL::CanonicalizeFastmathFunctionName_(
    const std::string &func_name) const {
  static const std::unordered_map<std::string, std::string> kFastMathMap = {
      {"divf", "tl.divf"},   {"exp", "tl.exp"},    {"expf", "tl.exp"},
      {"exp2", "tl.exp2"},   {"exp2f", "tl.exp2"}, {"log", "tl.log"},
      {"logf", "tl.log"},    {"log2", "tl.log2"},  {"log2f", "tl.log2"},
      {"log10", "tl.log10"}, {"tan", "tl.tan"},    {"cos", "tl.cos"},
      {"sin", "tl.sin"},     {"sqrt", "tl.sqrt"},  {"sqrtf", "tl.sqrt"},
  };

  auto it = kFastMathMap.find(func_name);
  if (it != kFastMathMap.end()) {
    return it->second;
  }
  return "";
}

void CodeGenTileLangCuTeDSL::PrintFuncDecorator_(
    std::ostream &os) { // NOLINT(*)
  os << "@cute.kernel\n";
}

void CodeGenTileLangCuTeDSL::PreFunctionBody_(const PrimFunc &f) {
  PrintIndent();
  stream << "threadIdx = tl.ThreadIdx()" << "\n";
  PrintIndent();
  stream << "blockIdx = tl.BlockIdx()" << "\n";
}

namespace {
std::string DTypeToString(DataType t) {
  ICHECK(t.is_scalar()) << "unsupported type " << t;

  if (t.is_void()) {
    return "void";
  }
  if (t == tl::cuTensorMapType()) {
    return "CUtensorMap";
  }

  int bits = t.bits();
  std::string elem_type;
  if (t.is_float()) {
    if (bits == 16 || bits == 32 || bits == 64) {
      elem_type = "Float" + std::to_string(bits);
    }
  } else if (t.is_bfloat16()) {
    elem_type = "BFloat16";
  } else if (t.is_float8()) {
    if (t.is_float8_e3m4()) {
      // unsupported
    } else if (t.is_float8_e4m3()) {
      elem_type =
          "Float8E4M3FN"; // Only Float8E4M3FN is supported at the moment
    } else if (t.is_float8_e4m3b11fnuz()) {
      // unsupported
    } else if (t.is_float8_e4m3fn()) {
      elem_type = "Float8E4M3FN";
    } else if (t.is_float8_e4m3fnuz()) {
      // unsupported
    } else if (t.is_float8_e5m2()) {
      elem_type = "Float8E5M2";
    } else if (t.is_float8_e5m2fnuz()) {
      // unsupported
    } else if (t.is_float8_e8m0fnu()) {
      elem_type = "Float8E8M0FNU";
    }
  } else if (t.is_float6()) {
    if (t.is_float6_e3m2fn()) {
      elem_type = "Float6E3M2FN";
    } else if (t.is_float6_e2m3fn()) {
      elem_type = "Float6E2M3FN";
    }
  } else if (t.is_float4()) {
    if (t.is_float4_e2m1fn()) {
      elem_type = "Float4E2M1FN";
    }
  } else if (t.is_bool()) {
    elem_type = "Boolean";
  } else if (t.is_uint()) {
    if (bits == 8 || bits == 16 || bits == 32 || bits == 64 || bits == 128) {
      elem_type = "Uint" + std::to_string(bits);
    }
  } else if (t.is_int()) {
    if (bits == 4 || bits == 8 || bits == 16 || bits == 32 || bits == 64 ||
        bits == 128) {
      elem_type = "Int" + std::to_string(bits);
    }
  }

  if (elem_type.empty()) {
    LOG(FATAL) << "Cannot convert type " << t << " to CuTeDSL type!";
  }

  return "cutlass." + elem_type;
}
} // namespace

void CodeGenTileLangCuTeDSL::PrintType(DataType t,
                                       std::ostream &os) { // NOLINT(*)
  CHECK(t.is_scalar()) << "Should not print a non-scalar type in CuTeDSL: "
                       << t;
  os << DTypeToString(t);
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const BroadcastNode *op,
                                        std::ostream &os) { // NOLINT(*)
  os << "tl.make_filled_tensor((" << PrintExpr_(op->lanes) << ",), "
     << PrintExpr_(op->value) << ").load()";
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const FloatImmNode *op,
                                        std::ostream &os) { // NOLINT(*)
  switch (op->dtype.bits()) {
  case 64:
  case 32:
  case 16:
  case 8:
  case 4: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      // For CuTeDSL, use Python's float('inf') instead of CUDA macros
      PrintType(op->dtype, temp);
      temp << "(";
      if (op->value < 0) {
        temp << "float('-inf')";
      } else {
        temp << "float('inf')";
      }
      temp << ")";
    } else if (std::isnan(op->value)) {
      // For CuTeDSL, use Python's float('nan')
      PrintType(op->dtype, temp);
      temp << "(float('nan'))";
    } else {
      // For CuTeDSL, use Python's float.fromhex() with hexfloat for full
      // precision
      PrintType(op->dtype, temp);
      temp << "(float.fromhex('" << std::hexfloat << op->value << "'))";
    }
    MarkConst(temp.str());
    os << temp.str();
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const CastNode *op,
                                        std::ostream &os) { // NOLINT(*)
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  if (from_ty.is_scalar())
    return CodeGenTileLangPY::VisitExpr_(op, os);

  // Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  PrintIndent();
  stream << sret << " = tl.make_rmem_tensor((" << target_ty.lanes() << ",), ";
  PrintType(target_ty.element_of(), stream);
  stream << ")\n";

  std::string src = SSAGetID(PrintExpr_(op->value), from_ty);

  PrintIndent();
  stream << sret << ".store(" << src << ".to(";
  PrintType(target_ty.element_of(), stream);
  stream << "))\n";
  os << sret << ".load()";
  return;
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const DivNode *op,
                                        std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinaryExpr_("//", op->dtype, op->a, op->b, os);
  } else {
    if (enable_fastmath_) {
      os << "tl.divf(" << PrintExpr_(op->a) << ", " << PrintExpr_(op->b)
         << ", fastmath=True)";
    } else {
      PrintBinaryExpr_("tl.divf", op->dtype, op->a, op->b, os);
    }
  }
}
void CodeGenTileLangCuTeDSL::VisitExpr_(const MinNode *op,
                                        std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("tl.min", op->dtype, op->a, op->b, os);
}
void CodeGenTileLangCuTeDSL::VisitExpr_(const MaxNode *op,
                                        std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("tl.max", op->dtype, op->a, op->b, os);
}

/**
 * @brief Emit CuTeDSL-specific code for a call expression.
 *
 * This visitor handles CallNode intrinsics and builtins that require emitting
 * CuTeDSL-specific code (inline PTX/ASM sequences, TensorLanguage runtime
 * calls, WMMA/TMA helpers, barriers, cp.async primitives, index-map based
 * stores, reinterpret/packing helpers, and various mma/ldmatrix patterns). The
 * function writes the generated code to the provided output stream and falls
 * back to the Python codegen for unrecognized calls.
 *
 * The method recognizes and emits code for (non-exhaustive): cp.async and its
 * commit/wait variants, tma_load/store and im2col variants, ptX
 * ldmatrix/stmatrix helpers, mbarrier APIs, cooperative grid sync, WMMA/legacy
 * MMA intrinsics (fill/load/store/mma/bmma/ptx_mma/ptx_mma_sp), low-level PTX
 * asm helpers (ldg32, cp_async bulk/init/arrive/wait barriers), reinterpret
 * paths for special small-float encodings (e.g., float4 e2m1fn), tl::tl_gemm
 * and related external calls, and other TL runtime calls.
 *
 * Side effects:
 * - Emits to `os` and the internal codegen output stream.
 * - May set internal feature flags (e.g., need_cooperative_groups_).
 * - May open/close SSA scopes and mutate internal variable mappings.
 * - May call LOG(FATAL) / CHECK / ICHECK on invalid or unsupported argument
 *   patterns.
 *
 * @param op The call node to generate code for; the function inspects op->op
 *           and op->args to determine the appropriate emission.
 * @param os  Output stream to receive expression-level output when the caller
 *            expects an expression result (some paths write directly to the
 *            member stream instead).
 */
void CodeGenTileLangCuTeDSL::VisitExpr_(const CallNode *op,
                                        std::ostream &os) { // NOLINT(*)
  auto print_extern_call_stmt = [&](std::string name, size_t start = 0,
                                    size_t end = 0) {
    // Cache context into a private ss, otherwise the let node may generate
    // within the function call arguments.
    std::ostringstream ss;
    for (size_t i = start; i < op->args.size() - end; i++) {
      if (i > start)
        ss << ", ";
      ss << PrintExpr_(op->args[i]);
    }

    PrintIndent();
    stream << name << "(";
    stream << ss.str();
    stream << ")\n";
  };

  auto print_mbarrier_obj = [&](PrimExpr barrier_id) {
    std::ostringstream ss;
    if (barrier_id.as<IntImmNode>()) {
      // incase the barrier_id is an integer, we need to print the barrier_id as
      // an integer
      ss << "(" << mbarrier_name_ << "+" << barrier_id << ")";
    } else {
      // otherwise may be a T.get_mbarrier() call or BufferLoad Node
      // we need to print the barrier_id as a string
      ss << PrintExpr_(barrier_id);
    }
    return ss.str();
  };

  if (op->op.same_as(builtin::ptx_cp_async())) {
    std::string dst = PrintExpr_(op->args[0]);
    std::string dst_offset = PrintExpr_(op->args[1]);
    std::string src = PrintExpr_(op->args[2]);
    std::string src_offset = PrintExpr_(op->args[3]);
    std::string size = PrintExpr_(op->args[4]);
    // use size of argument list to indicate whether or not to use predicated
    // cp.async
    if (op->args.size() == 5) {
      PrintIndent();
      stream << "tl.cp_async_gs(" << size << ", " << dst << ", " << dst_offset
             << ", " << src << ", " << src_offset << ")\n";
    } else {
      std::string condition = PrintExpr_(op->args[5]);
      PrintIndent();
      stream << "tl.cp_async_gs_conditional(" << size << ", " << dst << ", "
             << dst_offset << ", " << src << ", " << src_offset << ", "
             << condition << ")\n";
    }
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    print_extern_call_stmt("tl.cp_async_commit");
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    print_extern_call_stmt("tl.cp_async_wait");
  } else if (op->op.same_as(builtin::create_barriers())) {
    PrintIndent();
    int barrier_count = Downcast<IntImm>(op->args[0])->value;
    stream << mbarrier_name_
           << " = tl.alloc_smem(cutlass.Uint64, size_in_elems=" << barrier_count
           << ")\n";
  } else if (op->op.same_as(tl::get_mbarrier())) {
    ICHECK_EQ(op->args.size(), 1);
    std::string barrier_id = PrintExpr_(op->args[0]);
    os << "(" << mbarrier_name_ << "+" << barrier_id << ")";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
    if (op->args.size() == 1) {
      PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      stream << "tl.mbarrier_arrive(" << mbarrier_obj << ")\n";
    } else if (op->args.size() == 3) {
      PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto cta_id = PrintExpr_(op->args[1]);
      auto pred = PrintExpr_(op->args[2]);
      stream << "tl.mbarrier_arrive(" << mbarrier_obj << ", " << cta_id << ", "
             << pred << ")\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier "
                 << op->args.size();
    }
  } else if (op->op.same_as(builtin::ptx_init_barrier_thread_count())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto arrive_count = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_init(" << mbarrier_obj << ", " << arrive_count
           << ")\n";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    if (op->args.size() == 2) {
      PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto transaction_bytes = PrintExpr_(op->args[1]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ")\n";
    } else if (op->args.size() == 4) {
      PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto transaction_bytes = PrintExpr_(op->args[1]);
      auto cta_id = PrintExpr_(op->args[2]);
      auto pred = PrintExpr_(op->args[3]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ", " << cta_id << ", " << pred << ")\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier_expect_tx "
                 << op->args.size();
    }
  } else if (op->op.same_as(builtin::ptx_cp_async_barrier())) {
    print_extern_call_stmt("tl.mbarrier_cp_async_arrive");
  } else if (op->op.same_as(tl::ptx_fence_barrier_init())) {
    print_extern_call_stmt("tl.fence_barrier_init");
  } else if (op->op.same_as(tl::ptx_cp_async_barrier_noinc())) {
    print_extern_call_stmt("tl.mbarrier_cp_async_arrive_noinc");
  } else if (op->op.same_as(tl::mbarrier_expect_tx())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto transaction_bytes = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_expect_tx(" << mbarrier_obj << ", "
           << transaction_bytes << ")\n";
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto phase = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_wait(" << mbarrier_obj << ", " << phase << ")\n";
  } else if (op->op.same_as(tl::ptx_init_tensor_memory())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ptx_deallocate_tensor_memory())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::no_set_max_nreg())) {
    // do nothing
  } else if (op->op.same_as(tl::tma_load())) {
    std::ostringstream ss;
    ICHECK_GE(op->args.size(), 2);
    auto pol = op->args[op->args.size() - 1].as<IntImmNode>();
    ICHECK(pol) << "Eviction policy must be IntImm";
    ICHECK_GE(pol->value, 0);
    ICHECK_LT(static_cast<size_t>(pol->value), eviction_policy_names_.size());
    auto eviction_policy = eviction_policy_names_[pol->value];
    // Simplify the code by using the default eviction policy
    if (eviction_policy != "EVICT_NORMAL") {
      LOG(FATAL) << "Eviction policy " << eviction_policy
                 << " is not supported currently";
    } else {
      ss << "tl.tma_load(";
    }
    auto desc = op->args[0];
    ss << PrintExpr_(desc) << ", ";
    ss << print_mbarrier_obj(op->args[1]) << ", ";
    ss << PrintExpr_(op->args[2]) << ", (";
    for (size_t i = 3; i < op->args.size() - 1; i++) {
      if (i > 3)
        ss << ", ";
      ss << PrintExpr_(op->args[i]);
    }
    ss << "))\n";
    PrintIndent();
    stream << ss.str();
  } else if (op->op.same_as(tl::tma_load_im2col())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::tma_store())) {
    std::stringstream ss;
    // Check minimum argument count (desc, data, at least one coord,
    // need_reduce, eviction)
    ICHECK_GE(op->args.size(), 4) << "tma_store requires at least 4 arguments "
                                     "(desc, data, coords..., need_reduce, "
                                     "eviction_policy), got "
                                  << op->args.size();

    // Safely extract need_reduce flag
    auto need_reduce_ptr = op->args[op->args.size() - 2].as<IntImmNode>();
    ICHECK(need_reduce_ptr)
        << "tma_store need_reduce flag (args[-2]) must be IntImm, got "
        << op->args[op->args.size() - 2]->GetTypeKey();
    auto need_reduce = need_reduce_ptr->value;
    if (need_reduce) {
      LOG(FATAL) << "Currently unsupported op: " << op->op;
    }

    // Safely extract and validate eviction policy index
    auto eviction_idx_ptr = op->args[op->args.size() - 1].as<IntImmNode>();
    ICHECK(eviction_idx_ptr)
        << "tma_store eviction policy (args[-1]) must be IntImm, got "
        << op->args[op->args.size() - 1]->GetTypeKey();
    ICHECK_GE(eviction_idx_ptr->value, 0)
        << "tma_store eviction policy index must be >= 0, got "
        << eviction_idx_ptr->value;
    ICHECK_LT(static_cast<size_t>(eviction_idx_ptr->value),
              eviction_policy_names_.size())
        << "tma_store eviction policy index " << eviction_idx_ptr->value
        << " out of bounds (max " << eviction_policy_names_.size() - 1 << ")";
    auto eviction_policy = eviction_policy_names_[eviction_idx_ptr->value];

    ss << "tl.tma_store(";
    auto desc = op->args[0];
    ss << PrintExpr_(desc) << ", ";
    ss << PrintExpr_(op->args[1]) << ", (";
    for (size_t i = 2; i < op->args.size() - 2; i++) {
      if (i > 2)
        ss << ", ";
      ss << PrintExpr_(op->args[i]);
    }
    ss << ")";
    if (eviction_policy != "EVICT_NORMAL") {
      ss << ", eviction_kind = nvvm.EvictKind." << eviction_policy.substr(6);
    }
    ss << ")\n";
    PrintIndent();
    stream << ss.str();
  } else if (op->op.same_as(tl::ptx_ldmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl.ptx_ldmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::ptx_stmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl.ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::fence_proxy_async())) {
    print_extern_call_stmt("tl.fence_proxy_async");
  } else if (op->op.same_as(tl::tma_store_arrive())) {
    print_extern_call_stmt("tl.tma_store_arrive");
  } else if (op->op.same_as(tl::tma_store_wait())) {
    PrintIndent();
    stream << "tl.tma_store_wait(0)\n";
  } else if (op->op.same_as(tl::warpgroup_arrive())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warpgroup_commit_batch())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warpgroup_wait())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warpgroup_fence_operand())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::set_max_nreg())) {
    PrintIndent();
    int nreg = Downcast<IntImm>(op->args[0])->value;
    int is_inc = Downcast<IntImm>(op->args[1])->value;
    std::string func_name =
        is_inc ? "tl.warpgroup_reg_alloc" : "tl.warpgroup_reg_dealloc";
    stream << func_name << "(" << nreg << ")\n";
  } else if (op->op.same_as(tl::wait_wgmma())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "tl.pack_half2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::sync_grid())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::loop_break())) {
    PrintIndent();
    stream << "break\n";
  } else if (op->op.same_as(builtin::ptx_mma())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ptx_mma_sm70())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::ptx_mma_sp())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ptx_wgmma_ss())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ptx_wgmma_rs())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ss())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ts())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::tcgen05_mma_arrive())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::ptx_ldmatrix())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::mma_store())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::mma_fill())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::ptx_cp_async_bulk())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::ptx_wait_barrier())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::ptx_ldg32())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::reinterpret())) {
    DataType tgt_dtype = op->dtype;
    DataType src_dtype = op->args[0]->dtype;
    ICHECK_EQ(tgt_dtype.lanes() * tgt_dtype.bits(),
              src_dtype.lanes() * src_dtype.bits())
        << "reinterpret expects source and target to have the same number of "
           "bits";

    const BufferLoadNode *load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1)
        << "CodeGenTileLangCuTeDSL only supports flat memory";

    PrimExpr index = load->indices[0];
    if (const RampNode *node = index.as<RampNode>(); node) {
      auto *p_stride = as_const_int(node->stride);
      CHECK(p_stride);
      ICHECK_EQ(*p_stride, 1) << "reinterpret expects contiguous elements";
      index = node->base;
    }

    auto ptr_str = GetBufferPtr_(load->buffer.get(), index);
    os << "tl.make_tensor(tl.recast_ptr(" << ptr_str << ", dtype=";
    PrintType(tgt_dtype.element_of(), os);
    os << "), (" << tgt_dtype.lanes() << ",)).load()";
  } else if (op->op.same_as(builtin::thread_return())) {
    os << "return";
  } else if (op->op.same_as(tl::tl_gemm())) {
    ICHECK(op->args.size() == 4) << "tl_gemm expects 4 arguments <op_instance, "
                                    "A_ptr, B_ptr, C_ptr>, but got "
                                 << op->args.size();

    auto op_instance = Downcast<StringImm>(op->args[0]);
    PrintCallExtern_(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                     op_instance->value, op->args, true, os);
  } else if (op->op.same_as(tl::tl_gemm_sp())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::get_lane_idx())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::get_warp_idx_sync())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::get_warp_idx())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::get_warp_group_idx())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::tl_shuffle_elect())) {
    os << "tl.shuffle_elect(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::initialize_wgmma_descriptor())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::initialize_tcgen05_descriptor())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::increase_descriptor_offset())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::__exp())) {
    os << "tl.exp2(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__exp10())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::__log())) {
    os << "tl.log(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__log2())) {
    os << "tl.log2(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__log10())) {
    os << "tl.log10(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__tan())) {
    os << "tl.tan(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__cos())) {
    os << "tl.cos(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__sin())) {
    os << "tl.sin(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::ieee_add())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_sub())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_mul())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_fmaf())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_frcp())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_fsqrt())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_frsqrt())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::ieee_fdiv())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warp_reduce_sum())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warp_reduce_max())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warp_reduce_min())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warp_reduce_bitand())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(tl::warp_reduce_bitor())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::address_of())) {
    const BufferLoadNode *load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1)
        << "CodeGenTileLangCuTeDSL only supports flat memory";
    os << GetBufferPtr_(load->buffer.get(), load->indices[0]);
  } else {
    CodeGenTileLangPY::VisitExpr_(op, os);
  }
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const BufferLoadNode *op,
                                        std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1)
      << "Load from non-flat memory not supported.";
  ICHECK(!op->predicate.defined())
      << "Predicated buffer load is not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;

  const int value_lanes = value_dtype.lanes();
  if (value_lanes == element_dtype.lanes()) {
    std::string ref = GetBufferRef_(value_dtype, op->buffer.get(), index);
    if (ref.back() == ')') {
      ref += ".load()";
    }
    os << ref;
  } else {
    ICHECK_GE(value_lanes, element_dtype.lanes())
        << "Unsupported load/store: value lanes < buffer element lanes";
    bool is_contiguous = false;
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, value_lanes / element_dtype.lanes())
            .Match(index)) {
      is_contiguous = true;
    }

    if (is_contiguous) {
      std::string ref =
          GetBufferRef_(value_dtype, op->buffer.get(), base.Eval());
      if (ref.back() == ')') {
        ref += ".load()";
      }
      os << ref;
    } else {
      ICHECK(element_dtype.is_scalar())
          << "buffer element type for non-contiguous load must be scalar "
             "currently";

      std::string sret = name_supply_->FreshName("_");
      PrintIndent();
      stream << sret << " = tl.make_rmem_tensor((" << value_lanes << ",), ";
      PrintType(element_dtype, stream);
      stream << ")\n";

      std::string vid = GetVarID(buffer_var.get());
      const RampNode *ramp = index.as<RampNode>();
      ICHECK(ramp)
          << "Expected Ramp index for vectorized non-contiguous access";
      for (int i = 0; i < value_lanes; ++i) {
        auto idx_expr =
            arith::Analyzer().Simplify(ramp->base + ramp->stride * i);

        PrintIndent();
        stream << sret << "[" << i << "] = "
               << GetBufferRef_(element_dtype, op->buffer.get(), idx_expr)
               << "\n";
      }
      os << sret << ".load()";
    }
  }
}

void CodeGenTileLangCuTeDSL::VisitStmt_(const BufferStoreNode *op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";
  ICHECK(!op->predicate.defined())
      << "Predicated buffer store is not supported.";

  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];
  Var buffer_var = op->buffer->data;
  std::string value_str = PrintExpr_(op->value);

  int value_lanes = value_dtype.lanes();
  if (value_lanes == element_dtype.lanes()) {
    std::string ref = GetBufferRef_(value_dtype, op->buffer.get(), index_expr);
    PrintIndent();

    if (ref.back() != ')') {
      stream << ref << " = " << RemoveOutermostParentheses(value_str) << "\n";
    } else {
      stream << ref << ".store(" << RemoveOutermostParentheses(value_str)
             << ")\n";
    }
  } else {
    bool is_contiguous = false;
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, value_lanes / element_dtype.lanes())
            .Match(index_expr)) {
      is_contiguous = true;
    }

    if (is_contiguous) {
      PrintVecStore_(op->buffer.get(), value_dtype, base.Eval(), value_str);
    } else {
      ICHECK(element_dtype.is_scalar())
          << "buffer element type for non-contiguous store must be scalar "
             "currently";

      // store elements separately
      value_str = SSAGetID(value_str, element_dtype);
      for (int i = 0; i < value_lanes; ++i) {
        const RampNode *ramp = index_expr.as<RampNode>();
        ICHECK(ramp);
        auto idx_expr =
            arith::Analyzer().Simplify(ramp->base + ramp->stride * i);

        PrintIndent();
        stream << GetBufferRef_(element_dtype, op->buffer.get(), idx_expr)
               << " = ";
        PrintVecElemLoad_(value_str, value_dtype, i, stream);
        stream << "\n";
      }
    }
  }
}

void CodeGenTileLangCuTeDSL::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;

  if (scope == "local.descriptor.wgmma") {
    stream << vid << " = tl.GmmaDescriptor()\n";
  } else if (scope == "local.descriptor.tcgen05_smem") {
    LOG(FATAL) << "Currently unsupported scope: " << scope;
  } else if (scope == "local.descriptor.tcgen05_instr") {
    LOG(FATAL) << "Currently unsupported scope: " << scope;
  } else if (scope == "shared.dyn") {
    stream << vid << " = tl.make_tensor(tl.get_dyn_smem(";
    PrintType(op->dtype, stream);
    // there is no bound check for Tensor access, so just set shape to 1
    stream << ", alignment=1024), (1,))\n";
  } else {
    size_t constant_size = op->ConstantAllocationSize();
    ICHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now, but get "
        << constant_size << " for " << op->buffer_var->name_hint;

    if (scope == "shared") {
      stream << vid << " = tl.make_tensor(tl.alloc_smem(";
      PrintType(op->dtype, stream);
      stream << ", " << constant_size << "), (" << constant_size << ",))\n";
    } else if (scope == "shared.barrier") {
      ICHECK(false) << "Unsupported scope: " << scope;
    } else if (scope == "local") {
      stream << vid << " = tl.make_rmem_tensor((" << constant_size << "),";
      PrintType(op->dtype, stream);
      stream << ")\n";
    } else if (scope == "local.var") {
      PrimExpr init = tir::make_const(op->dtype, 0);
      auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
      if (init_it != op->annotations.end()) {
        PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
        if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
          user_init = tir::Cast(op->dtype, user_init);
        }
        init = user_init;
      }
      stream << vid << " = " << PrintExpr_(init) << "\n";
    } else {
      ICHECK(false) << "Unsupported scope: " << scope;
    }
  }

  RegisterHandleType_(op->buffer_var.get(), op->dtype);
  PrintStmt_(op->body);
}

void CodeGenTileLangCuTeDSL::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (!iv->thread_tag.empty()) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex_(iv);
      }
    }
    VisitStmt(op->body);
  } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode *queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    VisitExpr(commit_group, stream);
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group =
        Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    VisitExpr(wait_group, stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    VisitStmt(inner->body);
  } else if (op->attr_key == "threadblock_swizzle_pattern") {
    this->PrintIndent();
    const StringImmNode *pattern = op->value.as<StringImmNode>();
    ICHECK(pattern);
    std::string call_str = pattern->value;
    // replace :: with . and replace < with ( and replace > with )
    ReplaceAll(call_str, "::", ".");
    ReplaceAll(call_str, "<", "(");
    ReplaceAll(call_str, ">", ")");
    this->stream << "blockIdx = " << call_str << "\n";
    this->VisitStmt(op->body);
  } else if (op->attr_key == "pragma_unroll_factor") {
    const IntImmNode *factor = op->value.as<IntImmNode>();
    ICHECK(factor);
    unroll_factor_[op->node.as<VarNode>()] = Downcast<IntImm>(factor);
    CodeGenTileLangPY::VisitStmt_(op);
  } else {
    CodeGenTileLangPY::VisitStmt_(op);
  }
}

void CodeGenTileLangCuTeDSL::VisitStmt_(const ForNode *op) {
  if (op->kind != tir::ForKind::kUnrolled) {
    CodeGenTileLangPY::VisitStmt_(op);
    return;
  }

  auto start_expr = arith::Analyzer().Simplify(op->min);
  auto stop_expr = arith::Analyzer().Simplify(op->extent + op->min);
  std::string unroll_factor;
  if (auto it = unroll_factor_.find(op->loop_var.get());
      it != unroll_factor_.end()) {
    unroll_factor = PrintExpr_(it->second);
  }
  bool use_range_constexpr = unroll_factor.empty() &&
                             as_const_int(op->extent) != nullptr &&
                             *as_const_int(op->extent) <= LOOP_UNROLL_THRESHOLD;
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "for " << vid << " in cutlass.range";
  if (use_range_constexpr) {
    stream << "_constexpr";
  }
  stream << "(";
  if (!is_zero(start_expr)) {
    PrintExpr_(start_expr, stream);
    stream << ", ";
  }
  PrintExpr_(stop_expr, stream);
  if (!unroll_factor.empty()) {
    stream << ", unroll=" << unroll_factor;
  } else if (!use_range_constexpr) {
    stream << ", unroll_full=True";
  }
  stream << "):\n";
  int for_scope = BeginScope();
  PrintStmt_(op->body);
  EndScope(for_scope);
}

void CodeGenTileLangCuTeDSL::VisitStmt_(const IfThenElseNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(cond) << ":\n";
  int then_scope = BeginScope();
  if (const CallNode *call = op->condition.as<CallNode>();
      call && call->op.same_as(tl::tl_shuffle_elect())) {
    PrintIndent();
    stream << "with cute.arch.elect_one():\n";
    int with_scope = BeginScope();
    PrintStmt_(op->then_case);
    EndScope(with_scope);
  } else {
    PrintStmt_(op->then_case);
  }
  EndScope(then_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt_(op->else_case.value());
    EndScope(else_scope);
  }
}

void CodeGenTileLangCuTeDSL::VisitStmt_(const EvaluateNode *op) {
  if (is_const_int(op->value))
    return;
  const CallNode *call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    LOG(FATAL) << "Currently unsupported op: " << call->op;
  }
  if (call && (call->op.same_as(tvm::tl::device_assert()))) {
    std::string cond = RemoveOutermostParentheses(PrintExpr_(call->args[0]));
    PrintIndent();
    stream << "assert " << cond << "\n";
  } else if (call && call->op.same_as(tvm::tl::device_assert_with_msg())) {
    std::string cond = RemoveOutermostParentheses(PrintExpr_(call->args[0]));
    std::string msg_expr = PrintExpr_(call->args[1]);
    PrintIndent();
    stream << "assert " << cond << ", " << msg_expr << "\n";
  } else if (call && call->op.same_as(builtin::tvm_storage_sync())) {
    PrintStorageSync_(call);
  } else {
    CodeGenTileLangPY::VisitStmt_(op);
  }
}

void CodeGenTileLangCuTeDSL::PrintVecElemLoad_(const std::string &vec,
                                               DataType t, int i,
                                               std::ostream &os) { // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }
  os << vec << "[" << i << "]";
}

void CodeGenTileLangCuTeDSL::PrintVecElemStore_(const std::string &vec,
                                                DataType t, int i,
                                                const std::string &value) {
  PrintIndent();
  stream << vec << "[" << i << "] = " << value << "\n";
}

void CodeGenTileLangCuTeDSL::PrintVecStore_(const BufferNode *buffer,
                                            DataType t, PrimExpr base,
                                            const std::string &value) {
  ICHECK(!t.is_scalar()) << "PrintVecStore_() should not be used for scalar";

  std::string ref = GetBufferRef_(t, buffer, base);
  PrintIndent();
  stream << ref << ".store(" << value << ")\n";
}

void CodeGenTileLangCuTeDSL::PrintVecBinaryOp_(const std::string &opstr,
                                               DataType dtype, PrimExpr lhs,
                                               PrimExpr rhs,
                                               std::ostream &os) { // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  PrintIndent();
  stream << sret << " = tl.make_rmem_tensor((" << dtype.lanes() << ",), ";
  PrintType(dtype.element_of(), stream);
  stream << ")\n";

  std::string vlhs = SSAGetID(PrintExpr_(lhs), lhs.dtype());
  std::string vrhs = SSAGetID(PrintExpr_(rhs), rhs.dtype());

  const std::string one_char_op{"+-*%<>^|&"};
  const std::string two_char_op{"// == != <= >="};
  if ((opstr.size() == 1 && one_char_op.find(opstr) != std::string::npos) ||
      (opstr.size() == 2 && two_char_op.find(opstr) != std::string::npos)) {
    PrintIndent();
    stream << sret << ".store(" << vlhs << " " << opstr << " " << vrhs << ")\n";
  } else {
    // Unpack into individual ops.
    for (int i = 0, lanes = dtype.lanes(); i < lanes; ++i) {
      std::ostringstream value_temp;
      if (isalpha(opstr[0])) {
        value_temp << opstr << "(";
        PrintVecElemLoad_(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad_(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad_(vlhs, lhs.dtype(), i, value_temp);
        value_temp << opstr;
        PrintVecElemLoad_(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore_(sret, dtype, i, value_temp.str());
    }
  }
  os << sret << ".load()";
}

void CodeGenTileLangCuTeDSL::PrintBinaryExpr_(const std::string &opstr,
                                              DataType dtype, PrimExpr lhs,
                                              PrimExpr rhs,
                                              std::ostream &os) { // NOLINT(*)
  if (dtype.is_scalar()) {
    CodeGenTileLangPY::PrintBinaryExpr_(opstr, dtype, lhs, rhs, os);
  } else {
    PrintVecBinaryOp_(opstr, dtype, lhs, rhs, os);
  }
}

void CodeGenTileLangCuTeDSL::PrintBinaryIntrinsic_(
    const CallNode *op, const char *opstr,
    std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_scalar()) {
    CodeGenTileLangPY::PrintBinaryIntrinsic_(op, opstr, os);
  } else {
    PrintVecBinaryOp_(opstr, op->dtype, op->args[0], op->args[1], os);
  }
}

void CodeGenTileLangCuTeDSL::PrintCallExtern_(Type ret_type,
                                              ffi::String global_symbol,
                                              const ffi::Array<PrimExpr> &args,
                                              bool skip_first_arg,
                                              std::ostream &os) { // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);

  std::string global_symbol_str = global_symbol;
  ReplaceAll(global_symbol_str, "::", ".");

  std::vector<std::string> sargs;
  // when the template arguments occurs at the end, merge them with function
  // arguments
  if (global_symbol_str.back() == '>') {
    auto pos = global_symbol_str.rfind('<');
    ICHECK(pos != std::string::npos);
    std::string template_args =
        global_symbol_str.substr(pos + 1, global_symbol_str.size() - pos - 2);
    ReplaceAll(template_args, "true", "True");
    ReplaceAll(template_args, "false", "False");
    sargs.push_back(template_args);

    global_symbol_str.resize(pos);
  }
  const size_t arg_begin = static_cast<size_t>(skip_first_arg);
  for (size_t i = arg_begin; i < args.size(); ++i) {
    std::string sarg = PrintExpr_(args[i]);
    if (ret_dtype.is_fixed_length_vector()) {
      std::string val = SSAGetID(sarg, args[i].dtype());
      sargs.push_back(std::move(val));
    } else {
      sargs.push_back(sarg);
    }
  }

  // Replace "<...>" with "(...)". Nested "<" is not supported
  {
    auto pos_left = global_symbol_str.find('<');
    while (pos_left != std::string::npos) {
      auto pos_right = global_symbol_str.find('>', pos_left + 1);
      if (pos_right != std::string::npos) {
        auto args =
            global_symbol_str.substr(pos_left + 1, pos_right - pos_left - 1);
        ReplaceAll(args, "true", "True");
        ReplaceAll(args, "false", "False");
        global_symbol_str.replace(pos_left, args.size() + 2, "(" + args + ")");
      }
      pos_left = global_symbol_str.find('<');
    }
  }

  // Special cases:
  // Map C math functions to Python/cutedsl equivalents
  const auto canonicalized_global_symbol_str =
      CanonicalizeFastmathFunctionName_(global_symbol_str);
  const bool canonicalized = !canonicalized_global_symbol_str.empty();
  if (canonicalized) {
    global_symbol_str = canonicalized_global_symbol_str;
  }

  // Atomic Functions
  if (global_symbol_str.substr(0, 6) == "Atomic") {
    global_symbol_str = "tl." + global_symbol_str;
    // Convert first argument (Buffer) to pointer for atomic operations
    if (const BufferLoadNode *load = args[arg_begin].as<BufferLoadNode>()) {
      ICHECK_EQ(load->indices.size(), 1)
          << "CodeGenTileLangCuTeDSL only supports flat memory";
      sargs[0] = GetBufferPtr_(load->buffer.get(), load->indices[0]);
    }
  }
  // some optional template arguments might be ommited, so add names explicitly
  // for remain arguments
  if (global_symbol_str == "tl.gemm_ss" || global_symbol_str == "tl.gemm_rs" ||
      global_symbol_str == "tl.gemm_sr" || global_symbol_str == "tl.gemm_rr") {
    ICHECK(sargs.size() >= 3);
    sargs[sargs.size() - 3] = "A_ptr=" + sargs[sargs.size() - 3];
    sargs[sargs.size() - 2] = "B_ptr=" + sargs[sargs.size() - 2];
    sargs[sargs.size() - 1] = "C_ptr=" + sargs[sargs.size() - 1];
  }

  if (ret_dtype.is_fixed_length_vector()) {
    // maybe simplify this if TensorSSA suppports this OP
    std::string sret = name_supply_->FreshName("_");
    PrintIndent();
    stream << sret << " = tl.make_rmem_tensor((" << ret_dtype.lanes() << ",), ";
    PrintType(ret_dtype.element_of(), stream);
    stream << ")\n";

    // Emit a scalar call for each lane.
    bool has_template_arg = (sargs.size() > args.size() - arg_begin);
    for (int i = 0; i < ret_dtype.lanes(); ++i) {
      std::ostringstream scall;
      scall << global_symbol_str << "(";
      for (size_t j = 0; j < sargs.size(); ++j) {
        if (j != 0) {
          scall << ", ";
        }

        if (j == 0 && has_template_arg) {
          scall << sargs[j];
        } else {
          PrintVecElemLoad_(
              sargs[j],
              args[arg_begin + j - static_cast<size_t>(has_template_arg)]
                  .dtype(),
              i, scall);
        }
      }
      if (canonicalized && enable_fastmath_) {
        if (!sargs.empty()) {
          scall << ", ";
        }
        scall << "fastmath=True";
      }
      scall << ")";
      PrintVecElemStore_(sret, ret_dtype, i, scall.str());
    }
    os << sret << ".load()";
  } else {
    os << global_symbol_str << "(";
    for (size_t i = 0; i < sargs.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << sargs[i];
    }
    if (canonicalized && enable_fastmath_) {
      if (!sargs.empty()) {
        os << ", ";
      }
      os << "fastmath=True";
    }
    os << ")";
  }
}

std::string CodeGenTileLangCuTeDSL::GetBufferPtr_(const BufferNode *buffer,
                                                  PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  const std::string vid = GetVarID(buffer_var);

  DataType buffer_element_dtype = buffer->dtype;
  bool is_handle_type_match =
      HandleTypeMatch_(buffer_var, buffer_element_dtype);
  std::string ptr_str;
  if (is_handle_type_match) {
    ptr_str = vid + ".iterator";
  } else {
    ptr_str = "tl.recast_ptr(" + vid +
              ".iterator, dtype=" + DTypeToString(buffer_element_dtype) + ")";
  }

  std::string index_str = PrintExpr_(index);
  return "(" + ptr_str + " + " + index_str + ")";
}

// The following forms can be returned:
// (1) vid
// (2) vid[i]
// (3) tl.make_tensor_at_offset(...)[0]
// (4) tl.make_tensor_at_offset(...)
//
// Form (4) is needed when the whole tensor is loaded or stored.
// It's the only form that ends with ")". Using this fact, BufferLoadNode will
// add ".load()" and BufferStoreNode will add ".store()".
std::string CodeGenTileLangCuTeDSL::GetBufferRef_(DataType t,
                                                  const BufferNode *buffer,
                                                  PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }
  if (scope == "local.var" || scope.find("local.descriptor") == 0) {
    return vid;
  }

  DataType buffer_element_dtype = buffer->dtype;
  bool is_handle_type_match =
      HandleTypeMatch_(buffer_var, buffer_element_dtype);
  std::string ptr_str;
  if (is_handle_type_match) {
    ptr_str = vid + ".iterator";
  } else {
    ptr_str = "tl.recast_ptr(" + vid +
              ".iterator, dtype=" + DTypeToString(buffer_element_dtype) + ")";
  }

  const std::string index_str = PrintExpr_(index);

  if (t == buffer_element_dtype) {
    if (is_handle_type_match && buffer_element_dtype.is_scalar() &&
        (scope == "local" || scope == "shared" || scope == "shared.dyn" ||
         scope == "shared.barrier")) {
      // Tensors in these scopes are allocated as one-dimensional, so can be
      // assessed via "[]" correctly. Other tensors may be multi-dimensional,
      // and must be assessed via ptr, otherwise CuTeDSL will interpret "[]"
      // access using its visiting order and layout.
      return vid + "[" + index_str + "]";
    } else {
      std::ostringstream os;
      os << "tl.make_tensor_at_offset(" << ptr_str << ", " << index_str
         << ", (1,), div_by=" << buffer_element_dtype.lanes() << ")";
      // for vector data types, ".load()" (added by BufferLoadNode) is neeed
      // instead of "[0]"
      if (buffer_element_dtype.is_scalar()) {
        os << "[0]";
      }
      return os.str();
    }
  } else {
    const int num = t.bits() * t.lanes();
    const int den = buffer_element_dtype.bits() * buffer_element_dtype.lanes();
    ICHECK_EQ(num % den, 0) << "Cannot form view: bitwidth not divisible";
    int buffer_size = num / den;

    std::ostringstream os;
    os << "tl.make_tensor_at_offset(" << ptr_str << ", " << index_str << ", ("
       << buffer_size << ",), div_by=" << buffer_size << ")";
    return os.str();
  }
}

void CodeGenTileLangCuTeDSL::BindThreadIndex_(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));

  auto &thread_tag = iv->thread_tag;
  ICHECK(thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
         thread_tag == "threadIdx.z" || thread_tag == "blockIdx.x" ||
         thread_tag == "blockIdx.y" || thread_tag == "blockIdx.z");

  // cute.arch.thread_idx() and block_idx() are Int32
  DataType from_dtype = DataType::Int(32);
  var_idmap_[iv->var.get()] =
      CastFromTo_(thread_tag, from_dtype, iv->var.dtype());
}

void CodeGenTileLangCuTeDSL::PrintStorageSync_(const CallNode *op) {
  auto args = op->args;
  const std::string &sync = args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // do nothing
  } else if (sync == "shared" || sync == "shared.dyn") {
    PrintIndent();
    if (args.size() == 1) {
      stream << "tl.sync_threads()\n";
    } else if (args.size() == 2) {
      auto barrier_id_ptr = args[1].as<IntImmNode>();
      ICHECK(barrier_id_ptr)
          << "storage_sync barrier_id (args[1]) must be IntImm, got "
          << args[1]->GetTypeKey();
      auto barrier_id = barrier_id_ptr->value;
      stream << "tl.sync_thread_partial(" << barrier_id << ")\n";
    } else if (args.size() == 3) {
      auto barrier_id_ptr = args[1].as<IntImmNode>();
      ICHECK(barrier_id_ptr)
          << "storage_sync barrier_id (args[1]) must be IntImm, got "
          << args[1]->GetTypeKey();
      auto thread_count_ptr = args[2].as<IntImmNode>();
      ICHECK(thread_count_ptr)
          << "storage_sync thread_count (args[2]) must be IntImm, got "
          << args[2]->GetTypeKey();
      auto barrier_id = barrier_id_ptr->value;
      auto thread_count = thread_count_ptr->value;
      stream << "tl.sync_thread_partial(" << barrier_id << ", " << thread_count
             << ")\n";
    } else {
      LOG(FATAL) << "Invalid number of arguments for storage sync: "
                 << args.size();
    }
  } else if (sync == "global") {
    LOG(FATAL) << "PrintStorageSync_ for global is not supported for now";
  } else {
    LOG(FATAL) << "Unknown storage sync scope: " << sync;
  }
}

} // namespace codegen
} // namespace tvm
