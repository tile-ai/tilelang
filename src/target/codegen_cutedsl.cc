/*!
 * \file target/codegen_cutedsl.cc
 */

#include "codegen_cutedsl.h"
#include "codegen_utils.h"
#include "ptx.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/builtin.h>
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

// Helper to check if a statement subtree contains loop break ops
// (either tl::loop_break() or builtin::break_loop())
class LoopBreakDetector : public tir::StmtExprVisitor {
public:
  bool found = false;
  void VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(tl::loop_break()) ||
        op->op.same_as(builtin::break_loop())) {
      found = true;
    }
    if (!found)
      StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const ForNode *op) override {
    // Don't recurse into nested for loops — their breaks are their own
  }
};

static bool ContainsLoopBreak(const Stmt &stmt) {
  LoopBreakDetector det;
  det(stmt);
  return det.found;
}

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
      {"divf", "tl.divf"},    {"exp", "tl.exp"},    {"expf", "tl.exp"},
      {"exp2", "tl.exp2"},    {"exp2f", "tl.exp2"}, {"log", "tl.log"},
      {"logf", "tl.log"},     {"log2", "tl.log2"},  {"log2f", "tl.log2"},
      {"log10", "tl.log10"},  {"tan", "tl.tan"},    {"cos", "tl.cos"},
      {"sin", "tl.sin"},      {"sqrt", "tl.sqrt"},  {"sqrtf", "tl.sqrt"},
      {"tanh", "tl.tanh"},    {"tanhf", "tl.tanh"}, {"rsqrt", "tl.rsqrt"},
      {"rsqrtf", "tl.rsqrt"}, {"fabs", "tl.fabsf"}, {"fabsf", "tl.fabsf"},
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
  // Note: We need to pass the dtype to make_filled_tensor so it can create
  // the correct CuTeDSL type (e.g., cutlass.Int32 instead of Python int)
  std::ostringstream dtype_str;
  DataType dt = op->value.dtype();
  // CuTeDSL/MLIR normalizes unsigned integer tensor loads to signed types
  // (e.g., Uint8 pointer -> i8 tensor elements). Use signed type here to
  // match, avoiding type mismatch in tl.where() operations.
  if (dt.is_uint()) {
    PrintType(DataType::Int(dt.bits()), dtype_str);
  } else {
    PrintType(dt, dtype_str);
  }
  os << "tl.make_filled_tensor((" << PrintExpr_(op->lanes) << ",), "
     << dtype_str.str() << "(" << PrintExpr_(op->value) << ")).load()";
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

void CodeGenTileLangCuTeDSL::VisitExpr_(const IntImmNode *op,
                                        std::ostream &os) { // NOLINT(*)
  // CuTeDSL's tensor __setitem__ uses as_numeric() which converts bare
  // Python ints to Int32.  For non-int32 integer literals (e.g. int16, uint8),
  // wrap with the CuTeDSL type constructor so the value has the correct width.
  if (op->dtype == DataType::Bool()) {
    os << (op->value ? "True" : "False");
  } else if (op->dtype != DataType::Int(32)) {
    std::ostringstream temp;
    PrintType(op->dtype, temp);
    temp << "(" << op->value << ")";
    MarkConst(temp.str());
    os << temp.str();
  } else {
    std::ostringstream temp;
    temp << op->value;
    MarkConst(temp.str());
    os << temp.str();
  }
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const CastNode *op,
                                        std::ostream &os) { // NOLINT(*)
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  if (from_ty.is_scalar())
    return CodeGenTileLangPY::VisitExpr_(op, os);

  int lanes = target_ty.lanes();

  // CuTeDSL requires narrow precision (e.g. FP8) vector .to() casts to
  // operate on 32-bit aligned vectors.  For unaligned widths (e.g.
  // float8x2), pad the source to aligned width, cast, then return
  // the aligned rmem tensor.  The store path is responsible for
  // extracting only value_lanes elements when writing back.
  bool is_narrow_unaligned = target_ty.bits() < 32 && lanes > 1 &&
                             (target_ty.bits() * lanes) % 32 != 0;
  int aligned_lanes = is_narrow_unaligned ? (32 / target_ty.bits()) : lanes;

  std::string src = SSAGetID(PrintExpr_(op->value), from_ty);

  // If unaligned, pad source to aligned width
  std::string cast_src = src;
  if (is_narrow_unaligned) {
    cast_src = name_supply_->FreshName("_pad_src");
    PrintIndent();
    stream << cast_src << " = tl.make_rmem_tensor((" << aligned_lanes << ",), ";
    PrintType(from_ty.element_of(), stream);
    stream << ")\n";
    for (int i = 0; i < aligned_lanes; ++i) {
      PrintIndent();
      if (i < lanes) {
        stream << cast_src << "[" << i << "] = " << src << "[" << i << "]\n";
      } else {
        stream << cast_src << "[" << i << "] = ";
        PrintType(from_ty.element_of(), stream);
        stream << "(0)\n";
      }
    }
  }

  // Cast (always aligned now)
  std::string cast_dst = name_supply_->FreshName("_cast");
  PrintIndent();
  stream << cast_dst << " = tl.make_rmem_tensor((" << aligned_lanes << ",), ";
  PrintType(target_ty.element_of(), stream);
  stream << ")\n";
  PrintIndent();
  if (is_narrow_unaligned) {
    stream << cast_dst << ".store(" << cast_src << ".load().to(";
  } else {
    stream << cast_dst << ".store(" << src << ".to(";
  }
  PrintType(target_ty.element_of(), stream);
  stream << "))\n";

  if (is_narrow_unaligned) {
    // Return the aligned rmem tensor (not .load()) so downstream code
    // uses rmem element access (e.g. cast_dst[i]) instead of MLIR
    // vector extractelement, which fails for FP8 types due to
    // unrealized_conversion_cast in LLVM translation.
    os << cast_dst;
  } else {
    os << cast_dst << ".load()";
  }
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

  // NOTE: builtin::if_then_else is handled by the base class
  // (CodeGenTileLangPY) as a Python ternary: (true_val if cond else false_val).
  // This is correct for expression contexts (range(), arithmetic, etc.). When
  // the result is used in a BufferStore that needs TensorSSA, the store handler
  // wraps it with tl.where().

  if (op->op.same_as(builtin::ptx_cp_async())) {
    // args[0] = dst_access_ptr, args[1] = src_access_ptr, args[2] = bytes,
    // args[3] = predicate (optional)
    ICHECK(op->args.size() == 3 || op->args.size() == 4)
        << "ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
           "src_access_ptr, bytes, [predicate])";

    std::string dst = PrintExpr_(op->args[0]);
    std::string src = PrintExpr_(op->args[1]);
    std::string size = PrintExpr_(op->args[2]);

    if (op->args.size() == 3) {
      this->PrintIndent();
      stream << "tl.cp_async_gs(" << size << ", " << dst << ", " << src
             << ")\n";
    } else {
      std::string condition = PrintExpr_(op->args[3]);
      this->PrintIndent();
      stream << "tl.cp_async_gs_conditional(" << size << ", " << dst << ", "
             << src << ", " << condition << ")\n";
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
      auto mbarrier_obj = PrintExpr_(op->args[0]);
      stream << "tl.mbarrier_arrive(" << mbarrier_obj << ")\n";
    } else if (op->args.size() == 3) {
      PrintIndent();
      auto mbarrier_obj = PrintExpr_(op->args[0]);
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
    auto mbarrier_obj = PrintExpr_(op->args[0]);
    auto arrive_count = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_init(" << mbarrier_obj << ", " << arrive_count
           << ")\n";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    if (op->args.size() == 2) {
      PrintIndent();
      auto mbarrier_obj = PrintExpr_(op->args[0]);
      auto transaction_bytes = PrintExpr_(op->args[1]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ")\n";
    } else if (op->args.size() == 4) {
      PrintIndent();
      auto mbarrier_obj = PrintExpr_(op->args[0]);
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
    auto mbarrier_obj = PrintExpr_(op->args[0]);
    auto transaction_bytes = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_expect_tx(" << mbarrier_obj << ", "
           << transaction_bytes << ")\n";
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = PrintExpr_(op->args[0]);
    auto phase = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_wait(" << mbarrier_obj << ", " << phase << ")\n";
  } else if (op->op.same_as(tl::ptx_init_tensor_memory())) {
    ICHECK_EQ(op->args.size(), 2U);
    PrintIndent();
    auto tmem_buffer = PrintExpr_(op->args[0]);
    auto num_cols = PrintExpr_(op->args[1]);
    stream << "tl.tmem_allocate(" << tmem_buffer << ", " << num_cols << ")\n";
  } else if (op->op.same_as(tl::ptx_deallocate_tensor_memory())) {
    ICHECK_EQ(op->args.size(), 2U);
    PrintIndent();
    auto tmem_buffer = PrintExpr_(op->args[0]);
    auto num_cols = PrintExpr_(op->args[1]);
    stream << "tl.tmem_deallocate(" << tmem_buffer << ", " << num_cols << ")\n";
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
    ss << PrintExpr_(op->args[1]) << ", ";
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
      // Use tma_reduce for reduce mode
      ss << "tl.tma_reduce(";
      auto desc = op->args[0];
      ss << PrintExpr_(desc) << ", ";
      ss << PrintExpr_(op->args[1]) << ", (";
      for (size_t i = 2; i < op->args.size() - 2; i++) {
        if (i > 2)
          ss << ", ";
        ss << PrintExpr_(op->args[i]);
      }
      ss << "))\n";
      PrintIndent();
      stream << ss.str();
      return;
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
    PrintIndent();
    stream << "tl.warpgroup_arrive()\n";
  } else if (op->op.same_as(tl::warpgroup_commit_batch())) {
    PrintIndent();
    stream << "tl.warpgroup_commit_batch()\n";
  } else if (op->op.same_as(tl::warpgroup_wait())) {
    PrintIndent();
    int num_mma = Downcast<IntImm>(op->args[0])->value;
    stream << "tl.warpgroup_wait(" << num_mma << ")\n";
  } else if (op->op.same_as(tl::warpgroup_fence_operand())) {
    // no-op: warpgroup_fence_operand is not needed in CuTeDSL
  } else if (op->op.same_as(tl::set_max_nreg())) {
    PrintIndent();
    int nreg = Downcast<IntImm>(op->args[0])->value;
    int is_inc = Downcast<IntImm>(op->args[1])->value;
    std::string func_name =
        is_inc ? "tl.warpgroup_reg_alloc" : "tl.warpgroup_reg_dealloc";
    stream << func_name << "(" << nreg << ")\n";
  } else if (op->op.same_as(tl::wait_wgmma())) {
    PrintIndent();
    int num_mma = Downcast<IntImm>(op->args[0])->value;
    stream << "tl.wgmma_wait_group(" << num_mma << ")\n";
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "tl.pack_half2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::sync_grid())) {
    PrintIndent();
    stream << "tl.sync_grid()\n";
  } else if (op->op.same_as(tl::loop_break()) ||
             op->op.same_as(builtin::break_loop())) {
    if (in_break_loop_) {
      PrintIndent();
      stream << "_loop_break_" << current_break_id_ << " = 1\n";
      break_emitted_in_seq_ = true;
    } else {
      PrintIndent();
      stream << "break\n";
    }
  } else if (op->op.same_as(builtin::ptx_mma())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp64, ...
    // arg 4: B precision: fp16, fp64, ...
    // arg 5: C precision: fp32, fp64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index
    // arg 12: saturate
    // arg 13: (optional) 1-bit operator (xor or and)
    ICHECK(op->args.size() == 13U || op->args.size() == 14U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = GetVarPtr_(op->args[6]);
    std::string a_bias = PrintExpr_(op->args[7]);
    std::string b_ref = GetVarPtr_(op->args[8]);
    std::string b_bias = PrintExpr_(op->args[9]);
    std::string c_ref = GetVarPtr_(op->args[10]);
    std::string c_bias = PrintExpr_(op->args[11]);

    // Generate call to tl.ptx_mma dispatcher
    PrintIndent();
    stream << "tl.ptx_mma(\"" << shape << "\", \"" << A_layout << "\", \""
           << B_layout << "\", \"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << a_ref << ", " << a_bias << ", " << b_ref
           << ", " << b_bias << ", " << c_ref << ", " << c_bias << ")\n";
  } else if (op->op.same_as(tl::ptx_mma_sm70())) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (op->op.same_as(builtin::ptx_mma_sp())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16, fp32, ...
    // arg 4: B precision: fp16, fp32, ...
    // arg 5: C precision: fp16, fp32, ...
    // arg 6: A multiplicand pointer
    // arg 7: A multiplicand index
    // arg 8: B multiplicand pointer
    // arg 9: B multiplicand index
    // arg 10: C accumulator pointer
    // arg 11: C accumulator index
    // arg 12: metadata pointer
    // arg 13: metadata index
    // arg 14: sparse_selector
    // arg 15: saturate
    ICHECK_EQ(op->args.size(), 16U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = GetVarPtr_(op->args[6]);
    std::string a_bias = PrintExpr_(op->args[7]);
    std::string b_ref = GetVarPtr_(op->args[8]);
    std::string b_bias = PrintExpr_(op->args[9]);
    std::string c_ref = GetVarPtr_(op->args[10]);
    std::string c_bias = PrintExpr_(op->args[11]);
    std::string meta_ref = GetVarPtr_(op->args[12]);
    std::string meta_bias = PrintExpr_(op->args[13]);
    std::string sparse_selector = PrintExpr_(op->args[14]);
    std::string saturate =
        Downcast<Bool>(op->args[15])->value ? "True" : "False";

    PrintIndent();
    stream << "tl.ptx_mma_sp(\"" << shape << "\", \"" << A_layout << "\", \""
           << B_layout << "\", \"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << a_ref << ", " << a_bias << ", " << b_ref
           << ", " << b_bias << ", " << c_ref << ", " << c_bias << ", "
           << meta_ref << ", " << meta_bias << ", " << sparse_selector << ", "
           << saturate << ")\n";
  } else if (op->op.same_as(tl::ptx_wgmma_ss())) {
    // arg 0: shape (StringImm, e.g. "m64n128k16")
    // arg 1: a_is_k_major (Bool)
    // arg 2: b_is_k_major (Bool)
    // arg 3: A_dtype (StringImm)
    // arg 4: B_dtype (StringImm)
    // arg 5: C_dtype (StringImm)
    // arg 6: A descriptor (Var)
    // arg 7: A offset (PrimExpr)
    // arg 8: B descriptor (Var)
    // arg 9: B offset (PrimExpr)
    // arg 10: C accumulator (Var)
    // arg 11: C offset (PrimExpr)
    // arg 12: scale_out (PrimExpr)
    // arg 13: scale_in_a (Bool)
    // arg 14: scale_in_b (Bool)
    ICHECK_EQ(op->args.size(), 15U) << "ptx_wgmma_ss expects 15 args";
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    bool a_is_k_major = Downcast<Bool>(op->args[1])->value;
    bool b_is_k_major = Downcast<Bool>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_desc = PrintExpr_(op->args[6]);
    std::string A_offset = PrintExpr_(op->args[7]);
    std::string b_desc = PrintExpr_(op->args[8]);
    std::string B_offset = PrintExpr_(op->args[9]);
    std::string c_ref = GetVarPtr_(op->args[10]);
    std::string c_offset = PrintExpr_(op->args[11]);
    std::string scale_out = PrintExpr_(op->args[12]);
    bool scale_in_a = Downcast<Bool>(op->args[13])->value;
    bool scale_in_b = Downcast<Bool>(op->args[14])->value;
    // tnspA = !a_is_k_major, tnspB = !b_is_k_major
    std::string tnspA = a_is_k_major ? "False" : "True";
    std::string tnspB = b_is_k_major ? "False" : "True";
    // scaleA/scaleB: True (scale_in) -> 1, False -> -1
    int scaleA = scale_in_a ? 1 : -1;
    int scaleB = scale_in_b ? 1 : -1;
    PrintIndent();
    stream << "tl.wgmma_ss(\"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << m << ", " << n << ", " << k << ", " << tnspA
           << ", " << tnspB << ", " << scaleA << ", " << scaleB << ", ("
           << a_desc << " + " << A_offset << "), (" << b_desc << " + "
           << B_offset << "), " << c_ref << " + " << c_offset << ", "
           << scale_out << ")\n";
  } else if (op->op.same_as(tl::ptx_wgmma_rs())) {
    // arg 0: shape (StringImm, e.g. "m64n128k16")
    // arg 1: b_is_k_major (Bool)
    // arg 2: A_dtype (StringImm)
    // arg 3: B_dtype (StringImm)
    // arg 4: C_dtype (StringImm)
    // arg 5: A register buffer (Var)
    // arg 6: A offset (PrimExpr)
    // arg 7: B descriptor (Var)
    // arg 8: B offset (PrimExpr)
    // arg 9: C accumulator (Var)
    // arg 10: C offset (PrimExpr)
    // arg 11: scale_out (PrimExpr)
    // arg 12: scale_in_a (Bool)
    // arg 13: scale_in_b (Bool)
    ICHECK_EQ(op->args.size(), 14U) << "ptx_wgmma_rs expects 14 args";
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    bool b_is_k_major = Downcast<Bool>(op->args[1])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[2])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string a_ref = GetVarPtr_(op->args[5]);
    std::string A_offset = PrintExpr_(op->args[6]);
    std::string b_desc = PrintExpr_(op->args[7]);
    std::string B_offset = PrintExpr_(op->args[8]);
    std::string c_ref = GetVarPtr_(op->args[9]);
    std::string c_offset = PrintExpr_(op->args[10]);
    std::string scale_out = PrintExpr_(op->args[11]);
    bool scale_in_a = Downcast<Bool>(op->args[12])->value;
    bool scale_in_b = Downcast<Bool>(op->args[13])->value;
    // tnspB = !b_is_k_major (A is always K-major in RS)
    std::string tnspB = b_is_k_major ? "False" : "True";
    int scaleA = scale_in_a ? 1 : -1;
    int scaleB = scale_in_b ? 1 : -1;
    PrintIndent();
    stream << "tl.wgmma_rs(\"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << m << ", " << n << ", " << k << ", " << tnspB
           << ", " << scaleA << ", " << scaleB << ", " << a_ref << " + "
           << A_offset << ", (" << b_desc << " + " << B_offset << "), " << c_ref
           << " + " << c_offset << ", " << scale_out << ")\n";
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ss())) {
    ICHECK_EQ(op->args.size(), 14U)
        << "ptx_tcgen05_mma_ss expects 14 arguments";
    std::string kind_dtype = Downcast<StringImm>(op->args[0])->value;
    std::string a_desc = PrintExpr_(op->args[1]);
    std::string A_offset = PrintExpr_(op->args[2]);
    std::string b_desc = PrintExpr_(op->args[3]);
    std::string B_offset = PrintExpr_(op->args[4]);
    std::string c_ref = PrintExpr_(op->args[5]);
    std::string c_offset = PrintExpr_(op->args[6]);
    std::string desc_val = PrintExpr_(op->args[7]);
    std::string scale_out = PrintExpr_(op->args[8]);
    std::string mask0 = PrintExpr_(op->args[9]);
    std::string mask1 = PrintExpr_(op->args[10]);
    std::string mask2 = PrintExpr_(op->args[11]);
    std::string mask3 = PrintExpr_(op->args[12]);
    bool enable_ws = Downcast<Bool>(op->args[13])->value;
    PrintIndent();
    if (enable_ws) {
      stream << "tl.tcgen05mma_ws_ss(\"" << kind_dtype << "\", (" << a_desc
             << " + " << A_offset << "), (" << b_desc << " + " << B_offset
             << "), " << c_ref << "[0] + " << c_offset << ", " << desc_val
             << ", " << scale_out << ")\n";
    } else {
      stream << "tl.tcgen05mma_ss(\"" << kind_dtype << "\", (" << a_desc
             << " + " << A_offset << "), (" << b_desc << " + " << B_offset
             << "), " << c_ref << "[0] + " << c_offset << ", " << desc_val
             << ", " << scale_out << ", " << mask0 << ", " << mask1 << ", "
             << mask2 << ", " << mask3 << ")\n";
    }
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ts())) {
    ICHECK_EQ(op->args.size(), 13U)
        << "ptx_tcgen05_mma_ts expects 13 arguments";
    std::string kind_dtype = Downcast<StringImm>(op->args[0])->value;
    std::string a_ref = PrintExpr_(op->args[1]);
    std::string A_offset = PrintExpr_(op->args[2]);
    std::string b_desc = PrintExpr_(op->args[3]);
    std::string B_offset = PrintExpr_(op->args[4]);
    std::string c_ref = PrintExpr_(op->args[5]);
    std::string c_offset = PrintExpr_(op->args[6]);
    std::string desc_val = PrintExpr_(op->args[7]);
    std::string scale_out = PrintExpr_(op->args[8]);
    std::string mask0 = PrintExpr_(op->args[9]);
    std::string mask1 = PrintExpr_(op->args[10]);
    std::string mask2 = PrintExpr_(op->args[11]);
    std::string mask3 = PrintExpr_(op->args[12]);
    PrintIndent();
    stream << "tl.tcgen05mma_ts(\"" << kind_dtype << "\", " << a_ref << "[0] + "
           << A_offset << ", (" << b_desc << " + " << B_offset << "), " << c_ref
           << "[0] + " << c_offset << ", " << desc_val << ", " << scale_out
           << ", " << mask0 << ", " << mask1 << ", " << mask2 << ", " << mask3
           << ")\n";
  } else if (op->op.same_as(tl::tcgen05_mma_arrive())) {
    ICHECK_EQ(op->args.size(), 1U) << "tcgen05_mma_arrive expects 1 argument";
    PrintIndent();
    stream << "tl.tcgen05_mma_arrive(" << PrintExpr_(op->args[0]) << ")\n";
  } else if (op->op.same_as(builtin::ptx_ldmatrix())) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared
    // memory.
    ICHECK_EQ(op->args.size(), 7U);
    bool trans = Downcast<Bool>(op->args[0])->value;
    int num = Downcast<Integer>(op->args[1])->value;
    std::string local_ptr = GetVarPtr_(op->args[3]);
    std::string local_elem_offset = PrintExpr_(op->args[4]);
    std::string smem_ptr = PrintExpr_(op->args[5]);
    std::string smem_elem_offset = PrintExpr_(op->args[6]);

    std::string func_name = "tl.ptx_ldmatrix_x" + std::to_string(num);
    if (trans)
      func_name += "_trans";
    PrintIndent();
    stream << func_name << "(" << smem_ptr << " + " << smem_elem_offset << ", "
           << local_ptr << " + " << local_elem_offset << ")\n";
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
    if (load) {
      // Path 1: BufferLoad - use recast_ptr for memory access
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
    } else {
      // Path 2: General expression - use arith.bitcast
      std::string expr_str = PrintExpr_(op->args[0]);
      os << "tl.bitcast(" << expr_str << ", ";
      PrintType(tgt_dtype.element_of(), os);
      os << ")";
    }
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
    // get_lane_idx(warp_size?) -> threadIdx.x % warp_size
    ICHECK_LE(op->args.size(), 1U)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    std::string warp_size = op->args.empty() ? "32" : PrintExpr_(op->args[0]);
    os << "(tl.thread_idx() % " << warp_size << ")";
  } else if (op->op.same_as(tl::get_warp_idx_sync())) {
    // get_warp_idx_sync(warp_size?) -> threadIdx.x // warp_size
    ICHECK_LE(op->args.size(), 1U)
        << "tl.get_warp_idx_sync expects at most one argument <warp_size>.";
    std::string warp_size = op->args.empty() ? "32" : PrintExpr_(op->args[0]);
    os << "(tl.thread_idx() // " << warp_size << ")";
  } else if (op->op.same_as(tl::get_warp_idx())) {
    // get_warp_idx(warp_size?) -> threadIdx.x // warp_size
    ICHECK_LE(op->args.size(), 1U)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    std::string warp_size = op->args.empty() ? "32" : PrintExpr_(op->args[0]);
    os << "(tl.thread_idx() // " << warp_size << ")";
  } else if (op->op.same_as(tl::get_warp_group_idx())) {
    // get_warp_group_idx(warp_size?, warps_per_group?) ->
    //   threadIdx.x // (warp_size * warps_per_group)
    ICHECK_LE(op->args.size(), 2U)
        << "tl.get_warp_group_idx expects <warp_size, warps_per_group>.";
    std::string warp_size = !op->args.empty() ? PrintExpr_(op->args[0]) : "32";
    std::string warps_per_group =
        op->args.size() >= 2 ? PrintExpr_(op->args[1]) : "4";
    os << "(tl.thread_idx() // (" << warp_size << " * " << warps_per_group
       << "))";
  } else if (op->op.same_as(tl::tl_shuffle_elect())) {
    os << "tl.shuffle_elect(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::initialize_wgmma_descriptor())) {
    // TIR args: (descriptor, start_address, layout_type, leading, stride)
    // Python args: (layout_type, leading, stride, desc, start_address)
    ICHECK_EQ(op->args.size(), 5U)
        << "initialize_wgmma_descriptor expects 5 arguments";
    std::string descriptor = PrintExpr_(op->args[0]);
    std::string start_address = PrintExpr_(op->args[1]);
    std::string layout_type = PrintExpr_(op->args[2]);
    std::string leading = PrintExpr_(op->args[3]);
    std::string stride = PrintExpr_(op->args[4]);
    os << "tl.initialize_wgmma_descriptor(" << layout_type << ", " << leading
       << ", " << stride << ", " << descriptor << ", " << start_address << ")";
  } else if (op->op.same_as(tl::initialize_tcgen05_descriptor())) {
    ICHECK_EQ(op->args.size(), 7U)
        << "initialize_tcgen05_descriptor expects 7 arguments";
    std::string descriptor = PrintExpr_(op->args[0]);
    std::string start_address = PrintExpr_(op->args[1]);
    std::string leading = PrintExpr_(op->args[2]);
    std::string stride = PrintExpr_(op->args[3]);
    std::string base_offset = PrintExpr_(op->args[4]);
    std::string leading_abs = PrintExpr_(op->args[5]);
    std::string swizzle_mode = PrintExpr_(op->args[6]);
    os << "tl.initialize_tcgen05_descriptor(" << descriptor << ", "
       << start_address << ", " << leading << ", " << stride << ", "
       << base_offset << ", " << leading_abs << ", " << swizzle_mode << ")";
  } else if (op->op.same_as(tl::increase_descriptor_offset())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "increase_descriptor_offset expects 2 arguments";
    std::string descriptor = PrintExpr_(op->args[0]);
    std::string offset = PrintExpr_(op->args[1]);
    os << "tl.increase_descriptor_offset(" << descriptor << ", " << offset
       << ")";
  } else if (op->op.same_as(tl::__exp())) {
    os << "tl.exp2(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (op->op.same_as(tl::__exp10())) {
    os << "tl.exp10(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
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
    // ieee_add(a, b, rounding_mode)
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    os << "tl.ieee_fadd(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (op->op.same_as(tl::ieee_sub())) {
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    os << "tl.ieee_fsub(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (op->op.same_as(tl::ieee_mul())) {
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    os << "tl.ieee_fmul(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (op->op.same_as(tl::ieee_fmaf())) {
    // ieee_fmaf(a, b, c, rounding_mode)
    std::string rounding_mode = Downcast<StringImm>(op->args[3])->value;
    os << "tl.ieee_fmaf(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
       << ", rounding=\"" << rounding_mode << "\")";
  } else if (op->op.same_as(tl::ieee_frcp())) {
    std::string rounding_mode = Downcast<StringImm>(op->args[1])->value;
    os << "tl.ieee_frcp(" << PrintExpr_(op->args[0]) << ", rounding=\""
       << rounding_mode << "\")";
  } else if (op->op.same_as(tl::ieee_fsqrt())) {
    std::string rounding_mode = Downcast<StringImm>(op->args[1])->value;
    os << "tl.ieee_fsqrt(" << PrintExpr_(op->args[0]) << ", rounding=\""
       << rounding_mode << "\")";
  } else if (op->op.same_as(tl::ieee_frsqrt())) {
    os << "tl.rsqrt(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_fdiv())) {
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    os << "tl.ieee_fdiv(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (op->op.same_as(tl::warp_reduce_sum())) {
    os << "tl.warp_reduce_sum(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_max())) {
    os << "tl.warp_reduce_max(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_min())) {
    os << "tl.warp_reduce_min(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitand())) {
    os << "tl.warp_reduce_bitand(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitor())) {
    os << "tl.warp_reduce_bitor(" << PrintExpr_(op->args[0]) << ")";
  } else if (op->op.same_as(builtin::address_of())) {
    const BufferLoadNode *load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    ICHECK_EQ(load->indices.size(), 1)
        << "CodeGenTileLangCuTeDSL only supports flat memory";
    os << GetBufferPtr_(load->buffer.get(), load->indices[0]);
  } else if (op->op.same_as(tl::atomic_add_elem_op())) {
    // atomic_add_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicAdd(" << dst_ptr << ", " << src_value << ")\n";
  } else if (op->op.same_as(tl::atomic_add_ret_elem_op())) {
    // atomic_add_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicAdd(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_addx2_elem_op())) {
    // atomic_addx2_elem_op(dst_ptr, src_ptr[, memory_order]) -> may return prev
    // value
    os << "tl.AtomicAddx2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_addx4_elem_op())) {
    // atomic_addx4_elem_op(dst_ptr, src_ptr[, memory_order]) -> may return prev
    // value
    os << "tl.AtomicAddx4(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_load_elem_op())) {
    // atomic_load_elem_op(src_ptr, memory_order) -> returns loaded value
    os << "tl.AtomicLoad(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_store_elem_op())) {
    // atomic_store_elem_op(dst_ptr, value, memory_order)
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string value = PrintExpr_(op->args[1]);
    std::string memory_order = PrintExpr_(op->args[2]);
    this->PrintIndent();
    this->stream << "tl.AtomicStore(" << dst_ptr << ", " << value << ", "
                 << memory_order << ")\n";
  } else if (op->op.same_as(tl::atomic_max_elem_op())) {
    // atomic_max_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicMax(" << dst_ptr << ", " << src_value << ")\n";
  } else if (op->op.same_as(tl::atomic_max_ret_elem_op())) {
    // atomic_max_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicMaxRet(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_min_elem_op())) {
    // atomic_min_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicMin(" << dst_ptr << ", " << src_value << ")\n";
  } else if (op->op.same_as(tl::atomic_min_ret_elem_op())) {
    // atomic_min_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicMinRet(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (op->op.same_as(builtin::shift_right())) {
    // CuTeDSL type promotion fix: Int8 >> 4 returns Int32 in CuTeDSL,
    // but TIR expects result type to match operand type. Wrap in explicit
    // type conversion to match CUDA behavior.
    ICHECK_EQ(op->args.size(), 2U);
    DataType result_dtype = op->dtype;
    std::string lhs = PrintExpr_(op->args[0]);
    std::string rhs = PrintExpr_(op->args[1]);
    PrintType(result_dtype, os);
    os << "((" << lhs << " >> " << rhs << "))";
  } else if (op->op.same_as(builtin::shift_left())) {
    // Same fix for shift_left
    ICHECK_EQ(op->args.size(), 2U);
    DataType result_dtype = op->dtype;
    std::string lhs = PrintExpr_(op->args[0]);
    std::string rhs = PrintExpr_(op->args[1]);
    PrintType(result_dtype, os);
    os << "((" << lhs << " << " << rhs << "))";
  } else {
    CodeGenTileLangPY::VisitExpr_(op, os);
  }
}

void CodeGenTileLangCuTeDSL::VisitExpr_(const SelectNode *op,
                                        std::ostream &os) { // NOLINT(*)
  // Emit Python ternary: (true_val if cond else false_val).
  // This yields ArithValue which is fine in expression contexts (range(),
  // etc.). When used as a BufferStore value that requires TensorSSA, the store
  // handler will wrap it with tl.where() at statement level.
  std::string cond = PrintExpr_(op->condition);
  std::string t = PrintExpr_(op->true_value);
  std::string f = PrintExpr_(op->false_value);
  os << "(" << t << " if " << cond << " else " << f << ")";
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

  // CuTeDSL requires narrow precision vector loads to be 32-bit aligned.
  // For unaligned widths (e.g. float8x2), load an aligned-width vector
  // and extract the needed elements via scalar copy.
  bool is_narrow_unaligned = value_dtype.bits() < 32 && value_lanes > 1 &&
                             (value_dtype.bits() * value_lanes) % 32 != 0;

  if (is_narrow_unaligned) {
    int aligned_lanes = 32 / value_dtype.bits();
    std::string vid = GetVarID(buffer_var.get());
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var.get())) {
      scope = alloc_storage_scope_.at(buffer_var.get());
    }

    // Compute scalar base offset
    PrimExpr scalar_base;
    if (value_lanes == element_dtype.lanes()) {
      scalar_base = index * value_lanes;
    } else {
      // Contiguous vectorized load: extract base from ramp index
      arith::PVar<PrimExpr> ramp_base;
      ICHECK(arith::ramp(ramp_base, 1, value_lanes / element_dtype.lanes())
                 .Match(index))
          << "Non-contiguous narrow-precision load not supported";
      scalar_base = ramp_base.Eval() * element_dtype.lanes();
    }

    // Load aligned vector into an rmem tensor (not a raw MLIR vector).
    // This ensures downstream element access uses rmem tensor indexing
    // (which CuTeDSL handles correctly for FP8) instead of MLIR
    // extractelement (which fails due to unrealized_conversion_cast).
    std::string aligned_rmem = name_supply_->FreshName("_aload");

    if (scope == "local") {
      // For rmem: load aligned_lanes elements without cute.assume.
      // cute.assume(offset, divby=N) silently truncates offsets that
      // are not exact multiples of N; for rmem there is no hardware
      // alignment constraint, so we skip assume (use default div_by=1).
      // Shape uses aligned_lanes so .load()/.store() satisfy CuTeDSL's
      // 32-bit alignment requirement for narrow types (FP8 etc.).
      PrintIndent();
      stream << aligned_rmem << " = tl.make_rmem_tensor((" << aligned_lanes
             << ",), ";
      PrintType(value_dtype.element_of(), stream);
      stream << ")\n";
      PrintIndent();
      stream << aligned_rmem << ".store(tl.make_tensor_at_offset(" << vid
             << ".iterator, " << PrintExpr_(scalar_base) << ", ("
             << aligned_lanes << ",)).load())\n";
    } else {
      // For shared/global memory: load exactly value_lanes elements with
      // no alignment assumption (div_by=1).  This avoids:
      //  - MisalignedAddress from div_by=aligned_lanes on non-aligned offsets
      //  - OOB from loading aligned_lanes elements near the buffer boundary
      bool is_handle_match = HandleTypeMatch_(buffer_var.get(), element_dtype);
      std::string ptr_str;
      if (is_handle_match) {
        ptr_str = vid + ".iterator";
      } else {
        std::ostringstream ptr_os;
        ptr_os << "tl.recast_ptr(" << vid << ".iterator, dtype=";
        PrintType(value_dtype.element_of(), ptr_os);
        ptr_os << ")";
        ptr_str = ptr_os.str();
      }
      PrintIndent();
      stream << aligned_rmem << " = tl.make_rmem_tensor((" << value_lanes
             << ",), ";
      PrintType(value_dtype.element_of(), stream);
      stream << ")\n";
      PrintIndent();
      stream << aligned_rmem << ".store(tl.make_tensor_at_offset(" << ptr_str
             << ", " << PrintExpr_(scalar_base) << ", (" << value_lanes
             << ",)).load())\n";
    }

    // Return the rmem tensor (not .load()) so downstream code uses
    // rmem element access instead of MLIR vector extractelement.
    os << aligned_rmem;
  } else if (value_lanes == element_dtype.lanes()) {
    std::string ref = GetBufferRef_(value_dtype, op->buffer.get(), index);
    // Check if this is a barrier buffer - barrier pointers don't need .load()
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var.get())) {
      scope = alloc_storage_scope_.at(buffer_var.get());
    }
    if (ref.back() == ')' && scope != "shared.barrier") {
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

  // Pre-compute Select/if_then_else as tl.where() at statement level.
  // Python ternary (ArithValue) is rejected by CuTeDSL .store(); tl.where()
  // produces TensorSSA which works in all store paths (.store(), vec store,
  // etc.). Also handles Cast(Select/if_then_else) — the cast is applied via
  // .to() on the tl.where result.
  bool value_is_conditional = false;
  PrimExpr cond_expr, true_expr, false_expr;
  DataType cast_to_dtype; // non-void if a Cast wraps the conditional
  // Helper to detect Select/if_then_else in a PrimExpr.
  auto detect_conditional = [&](const PrimExpr &expr) {
    if (auto sel = expr.as<SelectNode>()) {
      value_is_conditional = true;
      cond_expr = sel->condition;
      true_expr = sel->true_value;
      false_expr = sel->false_value;
    } else if (auto call = expr.as<CallNode>();
               call && call->op.same_as(tir::builtin::if_then_else())) {
      value_is_conditional = true;
      cond_expr = call->args[0];
      true_expr = call->args[1];
      false_expr = call->args[2];
    }
  };
  detect_conditional(op->value);
  if (!value_is_conditional) {
    // Check for Cast(Select/if_then_else)
    if (auto cast_node = op->value.as<CastNode>()) {
      detect_conditional(cast_node->value);
      if (value_is_conditional) {
        cast_to_dtype = cast_node->dtype;
      }
    }
  }

  std::string value_str;
  if (value_is_conditional) {
    int lanes = value_dtype.lanes();
    if (lanes == 0)
      lanes = 1;
    // Helper: wrap a sub-expression as TensorSSA via make_rmem_tensor.
    auto as_tsa = [this](const PrimExpr &e, int n,
                         DataType elem_dt) -> std::string {
      std::string s = PrintExpr_(e);
      if (n == 0)
        n = 1;
      if (s.size() >= 7 && s.compare(s.size() - 7, 7, ".load()") == 0)
        return s;
      std::string var = name_supply_->FreshName("_tsa");
      PrintIndent();
      if (elem_dt.is_bool()) {
        stream << var << " = tl.make_rmem_tensor((" << n
               << ",), cutlass.Boolean)\n";
      } else {
        stream << var << " = tl.make_rmem_tensor((" << n << ",), "
               << DTypeToString(elem_dt) << ")\n";
      }
      for (int i = 0; i < n; ++i) {
        PrintIndent();
        if (n == 1) {
          stream << var << "[0] = " << s << "\n";
        } else {
          stream << var << "[" << i << "] = " << s << "[" << i << "]\n";
        }
      }
      return var + ".load()";
    };
    DataType true_ty = true_expr.dtype().element_of();
    DataType false_ty = false_expr.dtype().element_of();
    std::string cond_tsa = as_tsa(cond_expr, 1, DataType::Bool());
    std::string then_tsa = as_tsa(true_expr, lanes, true_ty);
    std::string else_tsa = as_tsa(false_expr, lanes, false_ty);
    if (true_ty != false_ty) {
      DataType common =
          (true_ty.bits() >= false_ty.bits()) ? true_ty : false_ty;
      if (common.bits() < 32 && true_ty.is_float() && false_ty.is_float())
        common = DataType::Float(32);
      std::string common_str = DTypeToString(common);
      then_tsa += ".to(" + common_str + ")";
      else_tsa += ".to(" + common_str + ")";
    }
    std::string result = name_supply_->FreshName("_where");
    PrintIndent();
    stream << result << " = tl.where(" << cond_tsa << ", " << then_tsa << ", "
           << else_tsa << ")\n";
    // If the conditional was wrapped in a Cast, apply .to() on the result.
    // tl.where() returns TensorSSA, and .to() on TensorSSA returns TensorSSA.
    if (cast_to_dtype.bits() > 0) {
      std::string cast_result = name_supply_->FreshName("_wcast");
      PrintIndent();
      stream << cast_result << " = " << result << ".to("
             << DTypeToString(cast_to_dtype.element_of()) << ")\n";
      value_str = cast_result;
    } else {
      value_str = result;
    }
  } else {
    value_str = PrintExpr_(op->value);
  }

  // CuTeDSL does not support implicit narrowing assignments (e.g. storing
  // Int32 to an Int16 tensor).  When the value's scalar width exceeds the
  // buffer element width, wrap with an explicit cast so that CuTeDSL's
  // Integer constructor emits arith.trunci (or the appropriate float
  // truncation).  This mirrors C/CUDA's implicit narrowing conversions.
  DataType value_elem = value_dtype.element_of();
  DataType buf_elem = element_dtype.element_of();
  if (value_elem.bits() > buf_elem.bits() && value_elem.lanes() == 1 &&
      buf_elem.lanes() == 1 &&
      (value_elem.is_int() || value_elem.is_uint()) ==
          (buf_elem.is_int() || buf_elem.is_uint())) {
    value_str = CastFromTo_(value_str, value_elem, buf_elem);
    value_dtype = buf_elem.with_lanes(value_dtype.lanes());
  }

  int value_lanes = value_dtype.lanes();

  // CuTeDSL requires narrow precision (e.g. FP8) vector stores to be 32-bit
  // aligned. For unaligned widths (e.g. float8x2), pad to aligned width via
  // scalar element copy, then store as aligned vector.
  bool is_narrow_unaligned = value_dtype.bits() < 32 && value_lanes > 1 &&
                             (value_dtype.bits() * value_lanes) % 32 != 0;

  if (is_narrow_unaligned) {
    int aligned_lanes = 32 / value_dtype.bits(); // e.g. 4 for FP8
    // value_str may be an rmem tensor name (from Cast/BufferLoad narrow path)
    // or a .load() expression. SSAGetID will alias it if already a name.
    value_str = SSAGetID(value_str, value_dtype);

    // Determine store target scope
    std::string vid = GetVarID(buffer_var.get());
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var.get())) {
      scope = alloc_storage_scope_.at(buffer_var.get());
    }

    if (scope == "local") {
      // For local rmem: use element-by-element assignment.
      // A padded (aligned_lanes,) .store() writes beyond value_lanes
      // elements, causing overlapping writes and OOB at the end of the
      // rmem tensor.  And cute.assume with div_by=aligned_lanes silently
      // truncates non-aligned offsets.  Element assignment (vid[i]=val)
      // avoids both issues and does not trigger CuTeDSL's 32-bit
      // alignment check that .store() enforces for FP8 types.
      PrimExpr scalar_base;
      if (value_lanes == element_dtype.lanes()) {
        scalar_base = index_expr * value_lanes;
      } else {
        arith::PVar<PrimExpr> ramp_base;
        ICHECK(arith::ramp(ramp_base, 1, value_lanes / element_dtype.lanes())
                   .Match(index_expr))
            << "Non-contiguous narrow-precision store not supported";
        scalar_base = ramp_base.Eval() * element_dtype.lanes();
      }
      for (int i = 0; i < value_lanes; ++i) {
        PrintIndent();
        stream << vid << "[" << PrintExpr_(scalar_base) << " + " << i
               << "] = " << value_str << "[" << i << "]\n";
      }
    } else {
      // For global/shared: use scalar element stores (works at any alignment).
      // CuTeDSL supports scalar FP8 stores via rmem_tensor -> global_tensor[i].
      PrimExpr scalar_base;
      if (value_lanes == element_dtype.lanes()) {
        scalar_base = index_expr * value_lanes;
      } else {
        arith::PVar<PrimExpr> ramp_base;
        ICHECK(arith::ramp(ramp_base, 1, value_lanes / element_dtype.lanes())
                   .Match(index_expr))
            << "Non-contiguous narrow-precision store not supported";
        scalar_base = ramp_base.Eval() * element_dtype.lanes();
      }

      bool is_handle_match = HandleTypeMatch_(buffer_var.get(), element_dtype);
      std::string ptr_str;
      if (is_handle_match) {
        ptr_str = vid + ".iterator";
      } else {
        std::ostringstream ptr_os;
        ptr_os << "tl.recast_ptr(" << vid << ".iterator, dtype=";
        PrintType(value_dtype.element_of(), ptr_os);
        ptr_os << ")";
        ptr_str = ptr_os.str();
      }

      // Create a tensor view and store each element individually
      std::string view_var = name_supply_->FreshName("_sview");
      PrintIndent();
      stream << view_var << " = tl.make_tensor(" << ptr_str << " + "
             << PrintExpr_(scalar_base) << ", (" << value_lanes << ",))\n";
      for (int i = 0; i < value_lanes; ++i) {
        PrintIndent();
        stream << view_var << "[" << i << "] = " << value_str << "[" << i
               << "]\n";
      }
    }
  } else if (value_lanes == element_dtype.lanes()) {
    std::string ref = GetBufferRef_(value_dtype, op->buffer.get(), index_expr);
    PrintIndent();

    if (ref.back() != ')') {
      // Direct element assignment (e.g. vid[i] = value).
      // For conditionals (pre-computed as tl.where), extract scalar via [0].
      if (value_is_conditional && value_lanes == 1) {
        stream << ref << " = " << value_str << "[0]\n";
      } else {
        stream << ref << " = " << RemoveOutermostParentheses(value_str) << "\n";
      }
    } else {
      // CuTeDSL Tensor.store() expects TensorSSA; scalar expressions yield
      // ArithValue. Conditionals are already converted to tl.where()
      // (TensorSSA) at the top. For other scalar expressions, wrap in
      // make_filled_tensor.
      std::string store_rhs = value_str;
      if (!value_is_conditional && value_lanes == 1 &&
          !op->value.as<BufferLoadNode>()) {
        if (store_rhs.size() < 7 ||
            store_rhs.compare(store_rhs.size() - 7, 7, ".load()") != 0) {
          store_rhs = "tl.make_filled_tensor((1,), " + store_rhs + ").load()";
        }
      }
      stream << ref << ".store(" << RemoveOutermostParentheses(store_rhs)
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
    stream << vid << " = tl.Tcgen05SmemDescriptor()\n";
  } else if (scope == "local.descriptor.tcgen05_instr") {
    stream << vid << " = 0\n";
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
      stream << vid << " = tl.alloc_smem(cutlass.Uint64, size_in_elems="
             << constant_size << ")\n";
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
  bool has_break = ContainsLoopBreak(op->body);

  if (has_break) {
    // Emit guard variable before the loop
    int break_id = loop_break_counter_++;
    PrintIndent();
    stream << "_loop_break_" << break_id << " = 0\n";

    // Save and set break loop state
    bool old_in_break = in_break_loop_;
    int old_break_id = current_break_id_;
    in_break_loop_ = true;
    current_break_id_ = break_id;
    break_emitted_in_seq_ = false;

    // Emit the for header (same logic as below, but always non-unrolled path
    // since break loops use runtime conditions)
    PrintIndent();
    std::string vid = AllocVarID(op->loop_var.get());
    stream << "for " << vid << " in range(";
    if (is_zero(op->min)) {
      PrintExpr_(op->extent, stream);
    } else {
      PrintExpr_(op->min, stream);
      stream << ", ";
      PrimExpr upper_bound = arith::Analyzer().Simplify(op->extent + op->min);
      PrintExpr_(upper_bound, stream);
    }
    stream << "):\n";

    // Emit the body wrapped in guard check
    int for_scope = BeginScope();
    PrintIndent();
    stream << "if _loop_break_" << break_id << " == 0:\n";
    int guard_scope = BeginScope();
    PrintStmt_(op->body);
    EndScope(guard_scope);
    EndScope(for_scope);

    // Restore state
    in_break_loop_ = old_in_break;
    current_break_id_ = old_break_id;
    return;
  }

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

void CodeGenTileLangCuTeDSL::VisitStmt_(const SeqStmtNode *op) {
  if (!in_break_loop_) {
    // Normal path: delegate to base class
    CodeGenTileLangPY::VisitStmt_(op);
    return;
  }

  // In a break loop: after visiting a statement that contains loop_break(),
  // wrap the remaining statements in an `if _loop_break_N == 0:` guard
  // so they don't execute in the same iteration after the break.
  int guard_scope = -1;
  for (size_t i = 0; i < op->seq.size(); ++i) {
    break_emitted_in_seq_ = false;
    PrintStmt_(op->seq[i]);
    if (break_emitted_in_seq_ && guard_scope < 0 && i + 1 < op->seq.size()) {
      // Insert guard for remaining statements
      PrintIndent();
      stream << "if _loop_break_" << current_break_id_ << " == 0:\n";
      guard_scope = BeginScope();
      break_emitted_in_seq_ = false;
    }
  }
  if (guard_scope >= 0) {
    EndScope(guard_scope);
  }
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
  // Quantization Functions (decode_i4u_to_f16, decode_i4s_to_f16, etc.)
  if (global_symbol_str.substr(0, 7) == "decode_") {
    global_symbol_str = "tl." + global_symbol_str;
  }
  // Warp-level primitives (__activemask, __shfl_down_sync, __shfl_sync)
  if (global_symbol_str.substr(0, 2) == "__") {
    global_symbol_str = "tl." + global_symbol_str;
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
  // CuTeDSL only supports i1 (Boolean) in rmem; use Uint8 for gmem pointers.
  DataType effective_dtype = buffer_element_dtype;
  if (buffer_element_dtype.is_bool()) {
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var)) {
      scope = alloc_storage_scope_.at(buffer_var);
    }
    if (scope.empty())
      scope = GetPtrStorageScope(buffer->data);
    if (scope != "local" && scope != "local.var") {
      effective_dtype = DataType::UInt(8);
    }
  }
  // shared.barrier is allocated via tl.alloc_smem() which returns _Pointer
  // (not _Tensor), so it doesn't have .iterator — use vid directly.
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty())
    scope = GetPtrStorageScope(buffer->data);

  std::string ptr_str;
  if (scope == "shared.barrier") {
    ptr_str = vid;
  } else {
    bool is_handle_type_match = HandleTypeMatch_(buffer_var, effective_dtype);
    if (is_handle_type_match) {
      ptr_str = vid + ".iterator";
    } else {
      ptr_str = "tl.recast_ptr(" + vid +
                ".iterator, dtype=" + DTypeToString(effective_dtype) + ")";
    }
  }

  std::string index_str = PrintExpr_(index);
  return "(" + ptr_str + " + " + index_str + ")";
}

std::string CodeGenTileLangCuTeDSL::GetVarPtr_(const PrimExpr &expr) {
  // For local buffers (rmem tensors), we need to use .iterator to get the
  // pointer since local buffers in CuTeDSL are tensors, not raw pointers
  if (const VarNode *var = expr.as<VarNode>()) {
    return GetVarID(var) + ".iterator";
  }
  return PrintExpr_(expr);
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
  // CuTeDSL only supports i1 (Boolean) in rmem. For gmem/shared bool buffers,
  // use Uint8 instead (matches PyTorch's torch.bool memory layout).
  DataType effective_dtype = buffer_element_dtype;
  if (buffer_element_dtype.is_bool() && scope != "local" &&
      scope != "local.var") {
    effective_dtype = DataType::UInt(8);
  }
  bool is_handle_type_match = HandleTypeMatch_(buffer_var, effective_dtype);
  std::string ptr_str;
  if (is_handle_type_match) {
    ptr_str = vid + ".iterator";
  } else {
    ptr_str = "tl.recast_ptr(" + vid +
              ".iterator, dtype=" + DTypeToString(effective_dtype) + ")";
  }

  // CuTeDSL make_tensor_at_offset(ptr, offset, shape, div_by) expects a
  // single scalar offset. For a Ramp index, pass only the base to avoid
  // emitting a tuple (base+0, base+1, ...) that the runtime rejects.
  PrimExpr offset_expr = index;
  if (const RampNode *ramp = index.as<RampNode>()) {
    ICHECK(is_one(ramp->stride))
        << "GetBufferRef_: non-unit Ramp stride not supported, got "
        << ramp->stride;
    offset_expr = ramp->base;
  } else {
    arith::PVar<PrimExpr> ramp_base;
    int lanes = t.lanes() > 0 ? t.lanes() : 1;
    if (arith::ramp(ramp_base, 1, lanes).Match(index)) {
      offset_expr = ramp_base.Eval();
    }
  }
  const std::string index_str = PrintExpr_(offset_expr);

  if (t == buffer_element_dtype) {
    if (scope == "shared.barrier") {
      // shared.barrier is allocated via tl.alloc_smem() which returns _Pointer.
      // _Pointer does not support subscript access [i], but supports pointer
      // arithmetic (ptr + i). Use pointer addition instead of subscript.
      return "(" + vid + " + " + index_str + ")";
    } else if (is_handle_type_match && buffer_element_dtype.is_scalar() &&
               (scope == "local" || scope == "shared")) {
      // Tensors in these scopes are allocated as one-dimensional, so can be
      // assessed via "[]" correctly. Other tensors may be multi-dimensional,
      // and must be assessed via ptr, otherwise CuTeDSL will interpret "[]"
      // access using its visiting order and layout.
      // Note: shared.dyn is excluded because its shape is set to (1,) and
      // direct indexing would cause out-of-bounds access.
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
