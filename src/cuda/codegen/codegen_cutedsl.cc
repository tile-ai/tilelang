/*!
 * \file target/codegen_cutedsl.cc
 */

#include "codegen_cutedsl.h"
#include "backend/common/codegen/codegen_utils.h"
#include "cuda/codegen/ptx.h"
#include "support/check.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cuda/op/builtin.h"
#include "op/builtin.h"
#include <tvm/ir/prim/pattern_match.h>
#include <tvm/tirx/op/abi.h>

namespace tvm {
namespace codegen {

using namespace ffi;
namespace {

// Typed TensorMap operations carry coordinates, including a one-dimensional
// tuple. Raw bulk copies use the same TileLang op with a scalar byte count.
bool IsTensorMapPointer(const Expr &value) {
  const auto *pointer = value->ty.as<PtrTypeObj>();
  return pointer != nullptr &&
         pointer->element_type.as<tirx::TensorMapTypeObj>() != nullptr;
}

// Helper to check if a statement subtree contains loop break ops
// (either tl::loop_break() or builtin::break_loop())
class LoopBreakDetector : public tirx::StmtExprVisitor {
public:
  bool found = false;
  ffi::Optional<VisitInterrupt> Visit_(const CallObj *op) override {
    if (IsCodegenOp(op, "tl.loop_break") || IsCodegenOp(op, "tl.loop_break")) {
      found = true;
    }
    if (!found)
      return StmtExprVisitor::Visit_(op);
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const ForObj *op) override {
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const WhileObj *op) override {
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const BreakObj *op) override {
    found = true;
    return std::nullopt;
  }
};

static bool ContainsLoopBreak(const Stmt &stmt) {
  auto det = ffi::make_object<LoopBreakDetector>();
  det->Visit(stmt);
  return det->found;
}

static bool IsThreadReturnEvaluate(const Stmt &stmt) {
  const auto *eval = stmt.as<EvaluateObj>();
  if (eval == nullptr) {
    return false;
  }
  const auto *call = eval->value.as<CallObj>();
  return call != nullptr && IsCodegenOp(call, "tirx.gpu_thread_return");
}

struct ConditionalThreadReturn {
  PrimExpr condition;
  std::optional<Stmt> pre_return;
};

std::optional<ConditionalThreadReturn>
GetConditionalThreadReturn(const Stmt &stmt) {
  const auto *if_node = stmt.as<IfObj>();
  if (if_node == nullptr || if_node->else_case) {
    return std::nullopt;
  }
  if (IsThreadReturnEvaluate(if_node->then_case)) {
    return ConditionalThreadReturn{if_node->condition, std::nullopt};
  }
  if (const auto *seq = if_node->then_case.as<SeqStmtObj>();
      seq != nullptr && !seq->seq.empty() &&
      IsThreadReturnEvaluate(seq->seq[seq->seq.size() - 1])) {
    if (seq->seq.size() == 1) {
      return ConditionalThreadReturn{if_node->condition, std::nullopt};
    }
    Array<Stmt> pre_return;
    for (size_t i = 0; i + 1 < seq->seq.size(); ++i) {
      pre_return.push_back(seq->seq[i]);
    }
    Stmt pre_return_stmt =
        pre_return.size() == 1 ? pre_return[0] : SeqStmt(pre_return);
    return ConditionalThreadReturn{if_node->condition, pre_return_stmt};
  }
  return std::nullopt;
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

bool IsValidCPAsyncTransferBytes(int64_t bytes) {
  return bytes == 4 || bytes == 8 || bytes == 16;
}

std::optional<PrimType> GetAccessPtrElementType(const Expr &expr) {
  if (const auto *ptr = expr->ty.as<PtrTypeObj>()) {
    if (auto type = ptr->element_type.as<PrimType>())
      return type.value();
  }
  return std::nullopt;
}

int GetTileLangCPAsyncTransferBytes(const CallObj *op) {
  ICHECK(op->args.size() == 3 || op->args.size() == 4)
      << "tl::ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
         "src_access_ptr, num_elems, [predicate])";
  const auto *num_elems_imm = op->args[2].as<IntImmObj>();
  ICHECK(num_elems_imm) << "tl::ptx_cp_async num_elems must be IntImm, but got "
                        << op->args[2];
  int64_t num_elems = CodegenInt(num_elems_imm);
  ICHECK_GT(num_elems, 0);

  auto dst_elem_type = GetAccessPtrElementType(op->args[0]);
  auto src_elem_type = GetAccessPtrElementType(op->args[1]);
  ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
      << "tl::ptx_cp_async expects address_of, tl.access_ptr, or "
         "tvm_access_ptr operands";

  int64_t dst_total_bits =
      num_elems * dst_elem_type.value().bits() * dst_elem_type.value().lanes();
  int64_t src_total_bits =
      num_elems * src_elem_type.value().bits() * src_elem_type.value().lanes();
  ICHECK_EQ(dst_total_bits, src_total_bits)
      << "tl::ptx_cp_async requires src/dst transfer widths to match, but got "
      << dst_total_bits << " vs " << src_total_bits << " bits";
  ICHECK_EQ(dst_total_bits % 8, 0)
      << "tl::ptx_cp_async requires byte-aligned transfers, but got "
      << dst_total_bits << " bits";

  int64_t total_bytes = dst_total_bits / 8;
  ICHECK(IsValidCPAsyncTransferBytes(total_bytes))
      << "tl::ptx_cp_async requires a final PTX byte width in {4, 8, 16}, but "
         "got "
      << total_bytes;
  return static_cast<int>(total_bytes);
}

class ClusterInfoExtractor {
public:
  std::optional<std::tuple<int64_t, int64_t, int64_t>>
  extract(const tirx::Function &f) {
    auto dims = f->GetAttr<Array<PrimExpr>>("cluster_dims");
    if (!dims) {
      if (auto native = f->GetAttr<Array<PrimExpr>>("cuda.launch_dimensions")) {
        TVM_FFI_CHECK_EQ(native.value().size(), 6, ValueError)
            << "cuda.launch_dimensions requires block.xyz and cluster.xyz";
        dims = Array<PrimExpr>{native.value()[3], native.value()[4],
                               native.value()[5]};
      }
    }
    if (!dims)
      return std::nullopt;
    TVM_FFI_CHECK_EQ(dims.value().size(), 3, ValueError);
    auto x = CodegenInt(dims.value()[0]);
    auto y = CodegenInt(dims.value()[1]);
    auto z = CodegenInt(dims.value()[2]);
    TVM_FFI_CHECK(x > 0 && y > 0 && z > 0, ValueError)
        << "CuTeDSL rasterization requires static positive cluster dimensions";
    if (x == 1 && y == 1 && z == 1)
      return std::nullopt;
    return std::make_tuple(x, y, z);
  }
};

} // namespace

CodeGenTileLangCuTeDSL::CodeGenTileLangCuTeDSL() {
  // Read fastmath configuration from current PassContext
  auto pass_ctx = tvm::transform::PassContext::Current();

  // Read tl.enable_fast_math config, default to false
  enable_fastmath_ =
      pass_ctx->GetConfig<bool>(tl::kEnableFastMath).value_or(false);
}

void CodeGenTileLangCuTeDSL::InitFuncState_(const tirx::Function &f) {
  raw_pointer_vars_.clear();
  zero_like_vars_.clear();
  cluster_dims_ = ClusterInfoExtractor().extract(f);
  CodeGenTileLangPY::InitFuncState_(f);
}

std::string CodeGenTileLangCuTeDSL::CanonicalizeFastmathFunctionName_(
    const std::string &func_name) const {
  static const std::unordered_map<std::string, std::string> kFastMathMap = {
      {"divf", "tl.divf"},
      {"exp", "tl.exp"},
      {"expf", "tl.exp"},
      {"exp2", "tl.exp2"},
      {"exp2f", "tl.exp2"},
      {"log", "tl.log"},
      {"logf", "tl.log"},
      {"log1p", "tl.log1p"},
      {"log1pf", "tl.log1p"},
      {"log2", "tl.log2"},
      {"log2f", "tl.log2"},
      {"log10", "tl.log10"},
      {"tan", "tl.tan"},
      {"cos", "tl.cos"},
      {"sin", "tl.sin"},
      {"sqrt", "tl.sqrt"},
      {"sqrtf", "tl.sqrt"},
      {"tanh", "tl.tanh"},
      {"tanhf", "tl.tanh"},
      {"rsqrt", "tl.rsqrt"},
      {"rsqrtf", "tl.rsqrt"},
      {"fabs", "tl.fabsf"},
      {"fabsf", "tl.fabsf"},
      {"copysign", "tl.copysignf"},
      {"copysignf", "tl.copysignf"},
      {"isfinite", "tl.isfinite"},
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

void CodeGenTileLangCuTeDSL::PreFunctionBody_(const tirx::Function &f) {
  PrintIndent();
  stream << "threadIdx = tl.ThreadIdx()" << "\n";
  PrintIndent();
  stream << "blockIdx = tl.BlockIdx()" << "\n";
}

namespace {
std::string DTypeToString(PrimType t) {
  ICHECK(t.IsScalar()) << "unsupported type " << t;

  if (t.IsVoid()) {
    return "void";
  }
  if (t == tl::CuTensorMapType()) {
    return "CUtensorMap";
  }

  int bits = t.bits();
  std::string elem_type;
  if (t.MatchesCode(kDLFloat)) {
    if (bits == 16 || bits == 32 || bits == 64) {
      elem_type = "Float" + std::to_string(bits);
    }
  } else if (tl::IsTensorFloat32(t)) {
    // CuTeDSL TF32 scalar arithmetic is only valid in MMA paths; use FP32 for
    // storage and elementwise arithmetic since TF32 has the same memory layout.
    elem_type = "Float32";
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    elem_type = "BFloat16";
  } else if (CodegenIsFloat8(t)) {
    if (t.MatchesElementType(kDLFloat8_e3m4, 8)) {
      // unsupported
    } else if (t.MatchesElementType(kDLFloat8_e4m3, 8)) {
      elem_type =
          "Float8E4M3FN"; // Only Float8E4M3FN is supported at the moment
    } else if (t.MatchesElementType(kDLFloat8_e4m3b11fnuz, 8)) {
      // unsupported
    } else if (t.MatchesElementType(kDLFloat8_e4m3fn, 8)) {
      elem_type = "Float8E4M3FN";
    } else if (t.MatchesElementType(kDLFloat8_e4m3fnuz, 8)) {
      // unsupported
    } else if (t.MatchesElementType(kDLFloat8_e5m2, 8)) {
      elem_type = "Float8E5M2";
    } else if (t.MatchesElementType(kDLFloat8_e5m2fnuz, 8)) {
      // unsupported
    } else if (t.MatchesElementType(kDLFloat8_e8m0fnu, 8)) {
      elem_type = "Float8E8M0FNU";
    }
  } else if (CodegenIsFloat6(t)) {
    if (t.MatchesElementType(kDLFloat6_e3m2fn, 6)) {
      elem_type = "Float6E3M2FN";
    } else if (t.MatchesElementType(kDLFloat6_e2m3fn, 6)) {
      elem_type = "Float6E2M3FN";
    }
  } else if (CodegenIsFloat4(t)) {
    if (tl::IsFloat4E2M1Unpacked(t)) {
      elem_type = "Float4E2M1FN_unpack";
    } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
      elem_type = "Float4E2M1FN";
    }
  } else if (t.MatchesCode(kDLBool)) {
    elem_type = "Boolean";
  } else if (t.MatchesCode(kDLUInt)) {
    if (bits == 8 || bits == 16 || bits == 32 || bits == 64 || bits == 128) {
      elem_type = "Uint" + std::to_string(bits);
    }
  } else if (t.MatchesCode(kDLInt)) {
    if (bits == 4 || bits == 8 || bits == 16 || bits == 32 || bits == 64 ||
        bits == 128) {
      elem_type = "Int" + std::to_string(bits);
    }
  }

  if (elem_type.empty()) {
    TVM_FFI_THROW(InternalError)
        << "Cannot convert type " << t << " to CuTeDSL type!";
  }

  if (elem_type == "Float4E2M1FN_unpack") {
    return "tl." + elem_type;
  }
  return "cutlass." + elem_type;
}
} // namespace

void CodeGenTileLangCuTeDSL::PrintType(const PrimType &t,
                                       std::ostream &os) { // NOLINT(*)
  ICHECK(t.IsScalar()) << "Should not print a non-scalar type in CuTeDSL: "
                       << t;
  os << DTypeToString(t);
}

static bool IsZeroLike(const Expr &expr) {
  Expr value = expr;
  if (const prim::CastObj *cast = value.as<prim::CastObj>()) {
    value = cast->value;
  }
  if (const IntImmObj *imm = value.as<IntImmObj>()) {
    return imm->value == 0;
  }
  if (const FloatImmObj *imm = value.as<FloatImmObj>()) {
    return imm->value == 0.0;
  }
  return false;
}

static bool EndsWithLoad(const std::string &s) {
  return s.size() >= 7 && s.compare(s.size() - 7, 7, ".load()") == 0;
}

static bool IsZeroTensorLike(
    const Expr &expr,
    const std::unordered_set<const VarObj *> *zero_like_vars = nullptr) {
  Expr value = expr;
  while (true) {
    if (const prim::CastObj *cast = value.as<prim::CastObj>()) {
      value = cast->value;
      continue;
    }
    if (const prim::BroadcastObj *broadcast = value.as<prim::BroadcastObj>()) {
      value = broadcast->value;
      continue;
    }
    break;
  }
  if (const VarObj *var = value.as<VarObj>()) {
    return zero_like_vars != nullptr && zero_like_vars->count(var);
  }
  return IsZeroLike(value);
}

class TensorLoadDetector : public tirx::StmtExprVisitor {
public:
  bool found = false;

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) override {
    found = true;
    return std::nullopt;
  }
};

static bool ContainsTensorLoad(const Expr &expr) {
  auto det = ffi::make_object<TensorLoadDetector>();
  det->Visit(expr);
  return det->found;
}

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::BroadcastObj *op,
                                       std::ostream &os) { // NOLINT(*)
  if (CodegenDType(op).MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    bool is_zero = IsZeroTensorLike(op->value, &zero_like_vars_);
    ICHECK(is_zero)
        << "CuTeDSL cannot materialize scalar Float4E2M1FN values; only zero "
           "broadcast is supported via packed storage";
    os << "tl.make_filled_tensor((" << PrintExpr_(op->lanes)
       << ",), 0, dtype=cutlass.Float4E2M1FN).load()";
    return;
  }

  // Note: We need to pass the dtype to make_filled_tensor so it can create
  // the correct CuTeDSL type (e.g., cutlass.Int32 instead of Python int)
  std::ostringstream dtype_str;
  PrimType dt = op->value.ty();
  // CuTeDSL/MLIR normalizes unsigned integer tensor loads to signed types
  // (e.g., Uint8 pointer -> i8 tensor elements). Use signed type here to
  // match, avoiding type mismatch in tl.where() operations.
  if (dt.MatchesCode(kDLUInt)) {
    PrintType(PrimType::Int(dt.bits()), dtype_str);
  } else {
    PrintType(dt, dtype_str);
  }
  os << "tl.make_filled_tensor((" << PrintExpr_(op->lanes) << ",), "
     << dtype_str.str() << "(" << PrintExpr_(op->value) << ")).load()";
}

void CodeGenTileLangCuTeDSL::Dispatch_(const FloatImmObj *op,
                                       std::ostream &os) { // NOLINT(*)
  switch (CodegenDType(op).bits()) {
  case 64:
  case 32:
  case 16:
  case 8:
  case 4: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      // For CuTeDSL, use Python's float('inf') instead of CUDA macros
      PrintType(CodegenDType(op), temp);
      temp << "(";
      if (op->value < 0) {
        temp << "float('-inf')";
      } else {
        temp << "float('inf')";
      }
      temp << ")";
    } else if (std::isnan(op->value)) {
      // For CuTeDSL, use Python's float('nan')
      PrintType(CodegenDType(op), temp);
      temp << "(float('nan'))";
    } else {
      // For CuTeDSL, use Python's float.fromhex() with hexfloat for full
      // precision
      PrintType(CodegenDType(op), temp);
      temp << "(float.fromhex('" << FlexibleHexFormat(op->value) << "'))";
    }
    MarkConst(temp.str());
    os << temp.str();
    break;
  }
  default:
    TVM_FFI_THROW(InternalError)
        << "Bad bit-width for float: " << CodegenDType(op) << "\n";
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const IntImmObj *op,
                                       std::ostream &os) { // NOLINT(*)
  // CuTeDSL's tensor __setitem__ uses as_numeric() which converts bare
  // Python ints to Int32.  For non-int32 integer literals (e.g. int16, uint8),
  // wrap with the CuTeDSL type constructor so the value has the correct width.
  if (CodegenDType(op) == PrimType::Bool()) {
    os << (op->value ? "True" : "False");
  } else if (CodegenDType(op) != PrimType::Int(32)) {
    std::ostringstream temp;
    PrintType(CodegenDType(op), temp);
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

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::CastObj *op,
                                       std::ostream &os) { // NOLINT(*)
  PrimType from_ty = op->value.ty();
  PrimType target_ty = CodegenDType(op);
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  const bool from_is_cutedsl_fp8 =
      from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
      from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
      from_ty.MatchesElementType(kDLFloat8_e5m2, 8);
  const bool target_is_fp8_widening =
      target_ty == PrimType::Float(16) || target_ty == PrimType::Float(32) ||
      target_ty.MatchesElementType(kDLBfloat, 16);
  const bool target_is_cutedsl_fp8 =
      target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
      target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
      target_ty.MatchesElementType(kDLFloat8_e5m2, 8);
  const bool from_is_fp8_narrowing_source =
      from_ty == PrimType::Float(16) || from_ty == PrimType::Float(32) ||
      from_ty.MatchesElementType(kDLBfloat, 16);
  if (from_ty.IsScalar() && target_ty.IsScalar() &&
      (from_ty.MatchesCode(kDLInt) || from_ty.MatchesCode(kDLUInt)) &&
      (target_ty.MatchesCode(kDLInt) || target_ty.MatchesCode(kDLUInt)) &&
      target_ty.bits() > from_ty.bits()) {
    ffi::Optional<PrimExpr> shift_lhs;
    ffi::Optional<PrimExpr> shift_rhs;
    bool is_left_shift = false;
    if (const auto *shift = op->value.as<prim::LShiftObj>()) {
      shift_lhs = shift->a;
      shift_rhs = shift->b;
      is_left_shift = true;
    } else if (const auto *shift = op->value.as<prim::RShiftObj>()) {
      shift_lhs = shift->a;
      shift_rhs = shift->b;
    } else if (const auto *call = op->value.as<CallObj>();
               call && (IsCodegenOp(call, "prim.lshift") ||
                        IsCodegenOp(call, "prim.rshift"))) {
      ICHECK_EQ(call->args.size(), 2U);
      shift_lhs = call->args[0].as_or_throw<PrimExpr>();
      shift_rhs = call->args[1].as_or_throw<PrimExpr>();
      is_left_shift = IsCodegenOp(call, "prim.lshift");
    }
    if (shift_lhs.has_value()) {
      const PrimType lhs_ty = shift_lhs.value().ty();
      const bool same_signedness =
          (lhs_ty.MatchesCode(kDLInt) && target_ty.MatchesCode(kDLInt)) ||
          (lhs_ty.MatchesCode(kDLUInt) && target_ty.MatchesCode(kDLUInt));
      if (same_signedness && lhs_ty.bits() < target_ty.bits()) {
        const std::string lhs = PrintExpr_(shift_lhs.value());
        const std::string rhs = PrintExpr_(shift_rhs.value());
        PrintType(target_ty, os);
        os << "((";
        PrintType(target_ty, os);
        os << "(" << lhs << ") " << (is_left_shift ? "<<" : ">>") << " " << rhs
           << "))";
        return;
      }
    }
  }

  if (from_ty.IsScalar() && from_is_cutedsl_fp8 && target_is_fp8_widening) {
    // CuTeDSL/NVGPU widens FP8 through nvgpu.cvt_fpext, whose operand must be
    // a 32-bit aligned vector.  Pack the scalar into a four-lane FP8 vector,
    // reuse the tensor cast path, then read back lane zero.
    std::string src =
        SSAGetID(PrintExpr_(op->value), from_ty,
                 tvm::SideEffect(op->value) == CallEffectKind::kPure);
    std::string src_vec = name_supply_->FreshName("_scalar_fp8_src");
    PrintIndent();
    stream << src_vec << " = tl.make_rmem_tensor((4,), ";
    PrintType(from_ty, stream);
    stream << ")\n";
    PrintIndent();
    stream << src_vec << "[0] = " << src << "\n";
    for (int i = 1; i < 4; ++i) {
      PrintIndent();
      stream << src_vec << "[" << i << "] = ";
      PrintType(from_ty, stream);
      stream << "(0)\n";
    }

    std::string f16_vec = name_supply_->FreshName("_scalar_fp8_f16");
    PrintIndent();
    stream << f16_vec << " = tl.make_rmem_tensor((4,), cutlass.Float16)\n";
    PrintIndent();
    stream << f16_vec << ".store(tl.cast_tensor(" << src_vec
           << ".load(), cutlass.Float16))\n";
    if (target_ty == PrimType::Float(16)) {
      os << f16_vec << "[0]";
    } else {
      PrintType(target_ty, os);
      os << "(" << f16_vec << "[0])";
    }
    return;
  }

  if (from_ty.IsScalar() && target_is_cutedsl_fp8 &&
      from_is_fp8_narrowing_source) {
    // CuTeDSL lowers FP8 narrowing through vector tensor casts.  Pack a scalar
    // source into an aligned four-lane tensor, cast it, then extract lane zero.
    std::string src =
        SSAGetID(PrintExpr_(op->value), from_ty,
                 tvm::SideEffect(op->value) == CallEffectKind::kPure);
    std::string src_vec = name_supply_->FreshName("_scalar_fp8_src");
    PrintIndent();
    stream << src_vec << " = tl.make_rmem_tensor((4,), ";
    PrintType(from_ty, stream);
    stream << ")\n";
    PrintIndent();
    stream << src_vec << "[0] = " << src << "\n";
    for (int i = 1; i < 4; ++i) {
      PrintIndent();
      stream << src_vec << "[" << i << "] = ";
      PrintType(from_ty, stream);
      stream << "(0)\n";
    }

    std::string dst_vec = name_supply_->FreshName("_scalar_fp8_dst");
    PrintIndent();
    stream << dst_vec << " = tl.make_rmem_tensor((4,), ";
    PrintType(target_ty, stream);
    stream << ")\n";
    PrintIndent();
    stream << dst_vec << ".store(tl.cast_tensor(" << src_vec << ".load(), ";
    PrintType(target_ty, stream);
    stream << "))\n";
    os << dst_vec << "[0]";
    return;
  }

  if (from_ty.IsScalar())
    return CodeGenTileLangPY::Dispatch_(op, os);

  int lanes = target_ty.lanes();

  // CuTeDSL requires narrow precision (e.g. FP8) vector .to() casts to
  // operate on 32-bit aligned vectors.  For unaligned widths (e.g.
  // float8x2), pad the source to aligned width, cast, then return
  // the aligned rmem tensor.  The store path is responsible for
  // extracting only value_lanes elements when writing back.
  bool is_narrow_unaligned = target_ty.bits() < 32 && lanes > 1 &&
                             (target_ty.bits() * lanes) % 32 != 0;
  bool source_is_narrow_unaligned =
      from_ty.bits() < 32 && from_ty.lanes() > 1 &&
      (from_ty.bits() * from_ty.lanes()) % 32 != 0;
  int aligned_lanes = is_narrow_unaligned ? (32 / target_ty.bits()) : lanes;

  std::string src =
      SSAGetID(PrintExpr_(op->value), from_ty,
               tvm::SideEffect(op->value) == CallEffectKind::kPure);

  if (source_is_narrow_unaligned && !is_narrow_unaligned) {
    int source_aligned_lanes = 32 / from_ty.bits();
    std::string aligned_dst = name_supply_->FreshName("_aligned_cast");
    PrintIndent();
    stream << aligned_dst << " = tl.make_rmem_tensor((" << source_aligned_lanes
           << ",), ";
    PrintType(target_ty.WithLanes(1), stream);
    stream << ")\n";
    PrintIndent();
    stream << aligned_dst << ".store(tl.cast_tensor(tl.as_tensor_ssa(" << src
           << "), ";
    PrintType(target_ty.WithLanes(1), stream);
    stream << "))\n";

    std::string cast_dst = name_supply_->FreshName("_cast");
    PrintIndent();
    stream << cast_dst << " = tl.make_rmem_tensor((" << lanes << ",), ";
    PrintType(target_ty.WithLanes(1), stream);
    stream << ")\n";
    for (int i = 0; i < lanes; ++i) {
      PrintIndent();
      stream << cast_dst << "[" << i << "] = " << aligned_dst << "[" << i
             << "]\n";
    }
    os << cast_dst << ".load()";
    return;
  }

  // If unaligned, pad source to aligned width
  std::string cast_src = src;
  bool cast_src_is_tensor = false;
  if (is_narrow_unaligned) {
    cast_src = name_supply_->FreshName("_pad_src");
    cast_src_is_tensor = true;
    PrintIndent();
    stream << cast_src << " = tl.make_rmem_tensor((" << aligned_lanes << ",), ";
    PrintType(from_ty.WithLanes(1), stream);
    stream << ")\n";
    for (int i = 0; i < aligned_lanes; ++i) {
      PrintIndent();
      if (i < lanes) {
        stream << cast_src << "[" << i << "] = " << src << "[" << i << "]\n";
      } else {
        stream << cast_src << "[" << i << "] = ";
        PrintType(from_ty.WithLanes(1), stream);
        stream << "(0)\n";
      }
    }
  }

  // Cast (always aligned now)
  std::string cast_dst = name_supply_->FreshName("_cast");
  PrintIndent();
  stream << cast_dst << " = tl.make_rmem_tensor((" << aligned_lanes << ",), ";
  PrintType(target_ty.WithLanes(1), stream);
  stream << ")\n";
  PrintIndent();
  stream << cast_dst << ".store(tl.cast_tensor(" << cast_src;
  if (cast_src_is_tensor) {
    stream << ".load()";
  }
  stream << ", ";
  PrintType(target_ty.WithLanes(1), stream);
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

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::DivObj *op,
                                       std::ostream &os) { // NOLINT(*)
  if (CodegenDType(op).MatchesCode(kDLInt) ||
      CodegenDType(op).MatchesCode(kDLUInt)) {
    PrintIntegerDivMod_("truncdiv", CodegenDType(op), op->a, op->b, os);
  } else {
    if (enable_fastmath_) {
      os << "tl.divf(" << PrintExpr_(op->a) << ", " << PrintExpr_(op->b)
         << ", fastmath=True)";
    } else {
      PrintBinaryExpr_("tl.divf", CodegenDType(op), op->a, op->b, os);
    }
  }
}
void CodeGenTileLangCuTeDSL::PrintIntegerDivMod_(const char *mode,
                                                 PrimType dtype, PrimExpr lhs,
                                                 PrimExpr rhs,
                                                 std::ostream &os) {
  TVM_FFI_CHECK(dtype.MatchesCode(kDLInt, kDLUInt), TypeError);
  const auto a = PrintExpr_(lhs);
  const auto b = PrintExpr_(rhs);
  os << "tl.integer_divmod(" << a << ", " << b << ", ";
  PrintType(dtype.WithLanes(1), os);
  os << ", \"" << mode << "\")";
}
#define TL_CUTEDSL_DIVMOD(Node, Mode)                                          \
  void CodeGenTileLangCuTeDSL::Dispatch_(const Node *op, std::ostream &os) {   \
    PrintIntegerDivMod_(Mode, CodegenDType(op), op->a, op->b, os);             \
  }
TL_CUTEDSL_DIVMOD(prim::ModObj, "truncmod")
TL_CUTEDSL_DIVMOD(prim::FloorDivObj, "floordiv")
TL_CUTEDSL_DIVMOD(prim::FloorModObj, "floormod")
#undef TL_CUTEDSL_DIVMOD
#define TL_CUTEDSL_SHIFT(Node, Symbol)                                         \
  void CodeGenTileLangCuTeDSL::Dispatch_(const Node *op, std::ostream &os) {   \
    std::ostringstream value;                                                  \
    PrintBinaryExpr_(Symbol, CodegenDType(op), op->a, op->b, value);           \
    os << "tl.cast_tensor(" << value.str() << ", ";                            \
    PrintType(CodegenDType(op).WithLanes(1), os);                              \
    os << ")";                                                                 \
  }
TL_CUTEDSL_SHIFT(prim::LShiftObj, "<<")
TL_CUTEDSL_SHIFT(prim::RShiftObj, ">>")
#undef TL_CUTEDSL_SHIFT

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::MinObj *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("tl.min", CodegenDType(op), op->a, op->b, os);
}
void CodeGenTileLangCuTeDSL::Dispatch_(const prim::MaxObj *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("tl.max", CodegenDType(op), op->a, op->b, os);
}

/**
 * @brief Emit CuTeDSL-specific code for a call expression.
 *
 * This visitor handles CallObj intrinsics and builtins that require emitting
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
 * paths for special small-float encodings (e.g., float4 e2m1fn)
 * and related external calls, and other TL runtime calls.
 *
 * Side effects:
 * - Emits to `os` and the internal codegen output stream.
 * - May set internal feature flags (e.g., need_cooperative_groups_).
 * - May open/close SSA scopes and mutate internal variable mappings.
 * - May call LOG(FATAL) / ICHECK on invalid or unsupported argument
 *   patterns.
 *
 * @param op The call node to generate code for; the function inspects op->op
 *           and op->args to determine the appropriate emission.
 * @param os  Output stream to receive expression-level output when the caller
 *            expects an expression result (some paths write directly to the
 *            member stream instead).
 */
void CodeGenTileLangCuTeDSL::Dispatch_(const CallObj *op, std::ostream &os) {
  const auto args = CodegenOperands(op);
  const auto annotations = CodegenCallAnnotations(op);
  if (IsCodegenOp(op, "tirx.tensor_data_ptr")) {
    TVM_FFI_CHECK_EQ(args.size(), 1, ValueError);
    os << GetVarPtr_(args[0]);
    return;
  }
  if (IsCodegenOp(op, "tl.ptr_access")) {
    TVM_FFI_CHECK_EQ(args.size(), 4, ValueError);
    TVM_FFI_CHECK(args[0]->ty.as<PtrTypeObj>(), TypeError);
    os << "(" << GetVarPtr_(args[0]) << " + " << PrintExpr_(args[1]) << ")";
    return;
  }
  if (IsCodegenOp(op, "tirx.ptr_byte_offset") ||
      IsCodegenOp(op, "tirx.handle_add_byte_offset")) {
    TVM_FFI_CHECK_EQ(args.size(), 2, ValueError);
    const auto ptr = op->ty.as_or_throw<PtrType>();
    const auto dtype =
        ptr->element_type.as<PrimType>().value_or(PrimType::UInt(8));
    os << "tl.recast_ptr((tl.recast_ptr(" << GetVarPtr_(args[0])
       << ", dtype=cutlass.Uint8) + " << PrintExpr_(args[1]) << "), dtype=";
    PrintType(dtype, os);
    os << ")";
    return;
  }
  if (IsCodegenOp(op, "tirx.reinterpret") && op->ty.as<PtrTypeObj>()) {
    TVM_FFI_CHECK_EQ(args.size(), 1, ValueError);
    const auto dtype =
        op->ty.as_or_throw<PtrType>()->element_type.as_or_throw<PrimType>();
    os << "tl.recast_ptr(" << GetVarPtr_(args[0]) << ", dtype=";
    PrintType(dtype, os);
    os << ")";
    return;
  }
  if (IsCodegenOp(op, "tl.ptx_cp_async_bytes")) {
    TVM_FFI_CHECK(args.size() == 3 || args.size() == 4, ValueError);
    const int64_t bytes = CodegenInt(args[2]);
    TVM_FFI_CHECK(IsValidCPAsyncTransferBytes(bytes), ValueError);
    const auto dst = PrintExpr_(args[0]);
    const auto src = PrintExpr_(args[1]);
    PrintIndent();
    stream << (args.size() == 3 ? "tl.cp_async_gs("
                                : "tl.cp_async_gs_conditional(")
           << bytes << ", " << dst << ", " << src;
    if (args.size() == 4)
      stream << ", " << PrintExpr_(args[3]);
    stream << ")\n";
    return;
  }
  // NOLINT(*)
  if (IsCodegenOp(op, "tl.clamp")) {
    TVM_FFI_CHECK_EQ(args.size(), 3, ValueError);
    // The shared lowering binds x/lo/hi once and preserves their NaN
    // priority. A bare min(max(x, lo), hi) drops that public contract.
    os << PrintExpr_(tl::LowerClamp(GetRef<Call>(op).as_or_throw<PrimExpr>()));
    return;
  }
  auto print_extern_call_stmt = [&](std::string name, size_t start = 0,
                                    size_t end = 0) {
    // Cache context into a private ss, otherwise the let node may generate
    // within the function call arguments.
    std::ostringstream ss;
    for (size_t i = start; i < args.size() - end; i++) {
      if (i > start)
        ss << ", ";
      ss << PrintExpr_(args[i]);
    }

    PrintIndent();
    stream << name << "(";
    stream << ss.str();
    stream << ")\n";
  };

  if (IsCodegenOp(op, "prim.if_then_else") ||
      IsCodegenOp(op, "prim.if_then_else")) {
    ICHECK_EQ(args.size(), 3U)
        << "if_then_else expects <condition, true_value, false_value>";
    auto cast_branch = [&](std::string value, PrimType branch_dtype) {
      PrimType result_dtype = CodegenDType(op);
      if (result_dtype.MatchesCode(kDLUInt) && result_dtype.bits() == 8 &&
          result_dtype.IsScalar()) {
        std::ostringstream casted;
        PrintType(result_dtype, casted);
        casted << "(" << value << ")";
        return casted.str();
      }
      if (branch_dtype == result_dtype || !branch_dtype.IsScalar() ||
          !result_dtype.IsScalar()) {
        return value;
      }
      const bool integer_like = (branch_dtype.MatchesCode(kDLInt) ||
                                 branch_dtype.MatchesCode(kDLUInt)) &&
                                (result_dtype.MatchesCode(kDLInt) ||
                                 result_dtype.MatchesCode(kDLUInt));
      const bool float_like = branch_dtype.MatchesCode(kDLFloat) &&
                              result_dtype.MatchesCode(kDLFloat);
      if (!integer_like && !float_like) {
        return value;
      }
      std::ostringstream casted;
      PrintType(result_dtype, casted);
      casted << "(" << value << ")";
      return casted.str();
    };
    std::string cond = PrintExpr_(args[0]);
    std::string true_val =
        cast_branch(PrintExpr_(args[1]), CodegenDType(args[1]));
    std::string false_val =
        cast_branch(PrintExpr_(args[2]), CodegenDType(args[2]));
    os << "(" << true_val << " if " << cond << " else " << false_val << ")";
  } else if (IsCodegenOp(op, "tl.cp_async_bytes")) {
    // args[0] = dst_access_ptr, args[1] = src_access_ptr, args[2] = bytes,
    // args[3] = predicate (optional)
    ICHECK(args.size() == 3 || args.size() == 4)
        << "ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
           "src_access_ptr, bytes, [predicate])";

    std::string dst = PrintExpr_(args[0]);
    std::string src = PrintExpr_(args[1]);
    std::string size = PrintExpr_(args[2]);

    if (args.size() == 3) {
      this->PrintIndent();
      stream << "tl.cp_async_gs(" << size << ", " << dst << ", " << src
             << ")\n";
    } else {
      std::string condition = PrintExpr_(args[3]);
      this->PrintIndent();
      stream << "tl.cp_async_gs_conditional(" << size << ", " << dst << ", "
             << src << ", " << condition << ")\n";
    }
  } else if (IsCodegenOp(op, "tl.ptx_cp_async")) {
    // TileLang version: args[0] = dst_access_ptr, args[1] = src_access_ptr,
    // args[2] = num_elems, args[3] = predicate (optional)
    int total_bytes = GetTileLangCPAsyncTransferBytes(op);

    std::string dst = PrintExpr_(args[0]);
    std::string src = PrintExpr_(args[1]);
    std::string size = std::to_string(total_bytes);

    if (args.size() == 3) {
      this->PrintIndent();
      stream << "tl.cp_async_gs(" << size << ", " << dst << ", " << src
             << ")\n";
    } else {
      std::string condition = PrintExpr_(args[3]);
      this->PrintIndent();
      stream << "tl.cp_async_gs_conditional(" << size << ", " << dst << ", "
             << src << ", " << condition << ")\n";
    }
  } else if (IsCodegenOp(op, "tl.ptx_commit_group")) {
    print_extern_call_stmt("tl.cp_async_commit");
  } else if (IsCodegenOp(op, "tl.ptx_wait_group")) {
    print_extern_call_stmt("tl.cp_async_wait");
  } else if (IsCodegenOp(op, "tl.create_barriers")) {
    PrintIndent();
    int barrier_count = CodegenInt(args[0]);
    stream << mbarrier_name_
           << " = tl.alloc_smem(cutlass.Uint64, size_in_elems=" << barrier_count
           << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_arrive_barrier")) {
    ICHECK_EQ(args.size(), 1);
    PrintIndent();
    auto mbarrier_obj = PrintExpr_(args[0]);
    stream << "tl.mbarrier_arrive(" << mbarrier_obj << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_arrive_cluster_barrier")) {
    ICHECK_EQ(args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = PrintExpr_(args[0]);
    auto cta_id = PrintExpr_(args[1]);
    stream << "tl.mbarrier_arrive(" << mbarrier_obj << ", " << cta_id << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_init_barrier_thread_count")) {
    ICHECK_EQ(args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = PrintExpr_(args[0]);
    auto arrive_count = PrintExpr_(args[1]);
    stream << "tl.mbarrier_init(" << mbarrier_obj << ", " << arrive_count
           << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_arrive_barrier_expect_tx")) {
    if (args.size() == 2) {
      PrintIndent();
      auto mbarrier_obj = PrintExpr_(args[0]);
      auto transaction_bytes = PrintExpr_(args[1]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ")\n";
    } else if (args.size() == 4) {
      PrintIndent();
      auto mbarrier_obj = PrintExpr_(args[0]);
      auto transaction_bytes = PrintExpr_(args[1]);
      auto cta_id = PrintExpr_(args[2]);
      auto pred = PrintExpr_(args[3]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ", " << cta_id << ", " << pred << ")\n";
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid parameter  for tl::arrive_barrier_expect_tx "
          << args.size();
    }
  } else if (IsCodegenOp(op, "tl.ptx_cp_async_barrier")) {
    print_extern_call_stmt("tl.mbarrier_cp_async_arrive");
  } else if (IsCodegenOp(op, "tl.ptx_fence_barrier_init")) {
    print_extern_call_stmt("tl.fence_barrier_init");
  } else if (IsCodegenOp(op, "tl.ptx_cp_async_barrier_noinc")) {
    print_extern_call_stmt("tl.mbarrier_cp_async_arrive_noinc");
  } else if (IsCodegenOp(op, "tl.mbarrier_expect_tx")) {
    ICHECK_EQ(args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = PrintExpr_(args[0]);
    auto transaction_bytes = PrintExpr_(args[1]);
    stream << "tl.mbarrier_expect_tx(" << mbarrier_obj << ", "
           << transaction_bytes << ")\n";
  } else if (IsCodegenOp(op, "tl.mbarrier_wait_parity")) {
    ICHECK_EQ(args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = PrintExpr_(args[0]);
    auto phase = PrintExpr_(args[1]);
    stream << "tl.mbarrier_wait(" << mbarrier_obj << ", " << phase << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_init_tensor_memory")) {
    ICHECK_EQ(args.size(), 2U);
    PrintIndent();
    auto tmem_buffer = PrintExpr_(args[0]);
    auto num_cols = PrintExpr_(args[1]);
    bool use_2cta = annotations.find("use_2cta") != annotations.end() &&
                    CodegenBool(annotations["use_2cta"]);
    stream << "tl.tmem_allocate(" << tmem_buffer << ", " << num_cols
           << ", use_2cta=" << (use_2cta ? "True" : "False") << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_deallocate_tensor_memory")) {
    ICHECK_EQ(args.size(), 2U);
    PrintIndent();
    auto tmem_buffer = PrintExpr_(args[0]);
    auto num_cols = PrintExpr_(args[1]);
    bool use_2cta = annotations.find("use_2cta") != annotations.end() &&
                    CodegenBool(annotations["use_2cta"]);
    stream << "tl.tmem_deallocate(" << tmem_buffer << ", " << num_cols
           << ", use_2cta=" << (use_2cta ? "True" : "False") << ")\n";
  } else if (IsCodegenOp(op, "tl.no_set_max_nreg")) {
    // do nothing
  } else if (IsCodegenOp(op, "tl.prefetch_tma_descriptor")) {
    print_extern_call_stmt("tl.prefetch_tma_descriptor");
  } else if (IsCodegenOp(op, "tl.tma_load")) {
    std::ostringstream ss;
    ICHECK_GE(args.size(), 2);
    auto pol = args[args.size() - 1].as<IntImmObj>();
    ICHECK(pol) << "Eviction policy must be IntImm";
    ICHECK_GE(CodegenInt(pol), 0);
    ICHECK_LT(static_cast<size_t>(CodegenInt(pol)),
              eviction_policy_names_.size());
    auto eviction_policy = eviction_policy_names_[CodegenInt(pol)];
    // Simplify the code by using the default eviction policy
    if (eviction_policy != "EVICT_NORMAL") {
      TVM_FFI_THROW(InternalError) << "Eviction policy " << eviction_policy
                                   << " is not supported currently";
    } else {
      ss << "tl.tma_load(";
    }
    auto desc = args[0];
    ss << PrintExpr_(desc) << ", ";
    ss << PrintExpr_(args[1]) << ", ";
    ss << PrintBytePointerExpr_(args[2]) << ", (";
    for (size_t i = 3; i < args.size() - 1; i++) {
      if (i > 3)
        ss << ", ";
      ss << PrintExpr_(args[i]);
    }
    if (args.size() == 5 && IsTensorMapPointer(desc))
      ss << ",";
    ss << ")";
    if (annotations.find("use_2cta") != annotations.end() &&
        CodegenBool(annotations["use_2cta"])) {
      ss << ", use_2cta=True";
    }
    ss << ")\n";
    PrintIndent();
    stream << ss.str();
  } else if (IsCodegenOp(op, "tl.tma_load_im2col")) {
    std::ostringstream ss;
    ICHECK_GE(args.size(), 6)
        << "tma_load_im2col requires desc, mbar, smem, coordinates, "
           "im2col offsets, and eviction policy";
    auto pol = args[args.size() - 1].as<IntImmObj>();
    ICHECK(pol) << "Eviction policy must be IntImm";
    ICHECK_GE(CodegenInt(pol), 0);
    ICHECK_LT(static_cast<size_t>(CodegenInt(pol)),
              eviction_policy_names_.size());
    auto eviction_policy = eviction_policy_names_[CodegenInt(pol)];
    if (eviction_policy != "EVICT_NORMAL") {
      TVM_FFI_THROW(InternalError) << "Eviction policy " << eviction_policy
                                   << " is not supported currently";
    }

    auto desc = args[0];
    ss << "tl.tma_load(";
    ss << PrintExpr_(desc) << ", ";
    ss << PrintExpr_(args[1]) << ", ";
    ss << PrintBytePointerExpr_(args[2]) << ", (";
    for (size_t i = 3; i < args.size() - 3; i++) {
      if (i > 3)
        ss << ", ";
      ss << PrintExpr_(args[i]);
    }
    if (args.size() == 7 && IsTensorMapPointer(desc))
      ss << ",";
    ss << "), im2col_offsets=(";
    for (size_t i = args.size() - 3; i < args.size() - 1; i++) {
      if (i > args.size() - 3)
        ss << ", ";
      ss << PrintExpr_(args[i]);
    }
    ss << "))\n";
    PrintIndent();
    stream << ss.str();
  } else if (IsCodegenOp(op, "tl.tma_store")) {
    std::stringstream ss;
    // Check minimum argument count (desc, data, at least one coord,
    // need_reduce, eviction)
    ICHECK_GE(args.size(), 4) << "tma_store requires at least 4 arguments "
                                 "(desc, data, coords..., need_reduce, "
                                 "eviction_policy), got "
                              << args.size();

    // Safely extract need_reduce flag
    auto need_reduce_ptr = args[args.size() - 2].as<IntImmObj>();
    ICHECK(need_reduce_ptr)
        << "tma_store need_reduce flag (args[-2]) must be IntImm, got "
        << args[args.size() - 2]->GetTypeKey();
    auto need_reduce = CodegenInt(need_reduce_ptr);
    if (need_reduce) {
      // Use tma_reduce for reduce mode
      ss << "tl.tma_reduce(";
      auto desc = args[0];
      ss << PrintExpr_(desc) << ", ";
      ss << PrintExpr_(args[1]) << ", (";
      for (size_t i = 2; i < args.size() - 2; i++) {
        if (i > 2)
          ss << ", ";
        ss << PrintExpr_(args[i]);
      }
      if (args.size() == 5 && IsTensorMapPointer(desc))
        ss << ",";
      ss << "))\n";
      PrintIndent();
      stream << ss.str();
      return;
    }

    // Safely extract and validate eviction policy index
    auto eviction_idx_ptr = args[args.size() - 1].as<IntImmObj>();
    ICHECK(eviction_idx_ptr)
        << "tma_store eviction policy (args[-1]) must be IntImm, got "
        << args[args.size() - 1]->GetTypeKey();
    ICHECK_GE(CodegenInt(eviction_idx_ptr), 0)
        << "tma_store eviction policy index must be >= 0, got "
        << CodegenInt(eviction_idx_ptr);
    ICHECK_LT(static_cast<size_t>(CodegenInt(eviction_idx_ptr)),
              eviction_policy_names_.size())
        << "tma_store eviction policy index " << CodegenInt(eviction_idx_ptr)
        << " out of bounds (max " << eviction_policy_names_.size() - 1 << ")";
    auto eviction_policy = eviction_policy_names_[CodegenInt(eviction_idx_ptr)];

    ss << "tl.tma_store(";
    auto desc = args[0];
    ss << PrintExpr_(desc) << ", ";
    ss << PrintExpr_(args[1]) << ", (";
    for (size_t i = 2; i < args.size() - 2; i++) {
      if (i > 2)
        ss << ", ";
      ss << PrintExpr_(args[i]);
    }
    if (args.size() == 5 && IsTensorMapPointer(desc))
      ss << ",";
    ss << ")";
    if (eviction_policy != "EVICT_NORMAL") {
      ss << ", eviction_kind = nvvm.EvictKind." << eviction_policy.substr(6);
    }
    ss << ")\n";
    PrintIndent();
    stream << ss.str();
  } else if (IsCodegenOp(op, "tl.ptx_ldmatrix")) {
    int trans = CodegenInt(args[0]);
    int num = CodegenInt(args[1]);
    std::string func_name = "tl.ptx_ldmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (IsCodegenOp(op, "tl.ptx_stmatrix")) {
    int trans = CodegenInt(args[0]);
    int num = CodegenInt(args[1]);
    bool is_shape_encoded = args.size() >= 4 && args.back().as<StringImmObj>();
    if (is_shape_encoded) {
      ICHECK_EQ(args.back().as_or_throw<StringImm>()->value, "m8n8")
          << "CuTeDSL stmatrix codegen only supports m8n8";
    }
    std::string func_name = "tl.ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2, is_shape_encoded ? 1 : 0);
  } else if (IsCodegenOp(op, "tl.fence_proxy_async")) {
    print_extern_call_stmt("tl.fence_proxy_async");
  } else if (IsCodegenOp(op, "tl.tma_store_arrive")) {
    print_extern_call_stmt("tl.tma_store_arrive");
  } else if (IsCodegenOp(op, "tl.tma_store_wait")) {
    int count = CodegenInt(args[0]);
    bool read = args.size() == 1 || CodegenInt(args[1]) != 0;
    PrintIndent();
    stream << "tl.tma_store_wait(" << count
           << ", read=" << (read ? "True" : "False") << ")\n";
  } else if (IsCodegenOp(op, "tl.warpgroup_arrive")) {
    PrintIndent();
    stream << "tl.warpgroup_arrive()\n";
  } else if (IsCodegenOp(op, "tl.warpgroup_commit_batch")) {
    PrintIndent();
    stream << "tl.warpgroup_commit_batch()\n";
  } else if (IsCodegenOp(op, "tl.warpgroup_wait")) {
    PrintIndent();
    int num_mma = CodegenInt(args[0]);
    stream << "tl.warpgroup_wait(" << num_mma << ")\n";
  } else if (IsCodegenOp(op, "tl.warpgroup_fence_operand")) {
    // no-op: warpgroup_fence_operand is not needed in CuTeDSL
  } else if (IsCodegenOp(op, "tl.set_max_nreg")) {
    PrintIndent();
    int nreg = CodegenInt(args[0]);
    int is_inc = CodegenInt(args[1]);
    std::string func_name =
        is_inc ? "tl.warpgroup_reg_alloc" : "tl.warpgroup_reg_dealloc";
    stream << func_name << "(" << nreg << ")\n";
  } else if (IsCodegenOp(op, "tl.annotate_producer_reg_dealloc") ||
             IsCodegenOp(op, "tl.annotate_consumer_reg_alloc")) {
    return;
  } else if (IsCodegenOp(op, "tl.wait_wgmma")) {
    PrintIndent();
    int num_mma = CodegenInt(args[0]);
    stream << "tl.wgmma_wait_group(" << num_mma << ")\n";
  } else if (IsCodegenOp(op, "tl.pack_b16")) {
    os << "tl.pack_half2(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ")";
  } else if (IsCodegenOp(op, "tl.sync_grid")) {
    PrintIndent();
    stream << "tl.sync_grid()\n";
  } else if (IsCodegenOp(op, "tl.pdl_trigger")) {
    PrintIndent();
    stream << "tl.griddepcontrol_launch_dependents()\n";
  } else if (IsCodegenOp(op, "tl.pdl_sync")) {
    PrintIndent();
    stream << "tl.griddepcontrol_wait()\n";
  } else if (IsCodegenOp(op, "tl.cluster_arrive_relaxed")) {
    PrintIndent();
    stream << "tl.cluster_arrive_relaxed()\n";
  } else if (IsCodegenOp(op, "tl.cluster_arrive")) {
    PrintIndent();
    stream << "tl.cluster_arrive()\n";
  } else if (IsCodegenOp(op, "tl.cluster_wait")) {
    PrintIndent();
    stream << "tl.cluster_wait()\n";
  } else if (IsCodegenOp(op, "tl.cluster_sync")) {
    PrintIndent();
    stream << "tl.cluster_sync()\n";
  } else if (IsCodegenOp(op, "tl.block_rank_in_cluster")) {
    os << "tl.block_rank_in_cluster()";
  } else if (IsCodegenOp(op, "tl.clc_try_cancel")) {
    print_extern_call_stmt("tl.clc_try_cancel");
  } else if (IsCodegenOp(op, "tl.clc_try_cancel_multicast")) {
    print_extern_call_stmt("tl.clc_try_cancel_multicast");
  } else if (IsCodegenOp(op, "tl.clc_is_canceled")) {
    os << "tl.clc_is_canceled(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.clc_get_first_ctaid_x")) {
    os << "tl.clc_get_first_ctaid_x(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.clc_get_first_ctaid_y")) {
    os << "tl.clc_get_first_ctaid_y(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.clc_get_first_ctaid_z")) {
    os << "tl.clc_get_first_ctaid_z(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.loop_break") ||
             IsCodegenOp(op, "tl.loop_break")) {
    EmitLoopBreak_();
  } else if (IsCodegenOp(op, "tirx.ptx_legacy.mma")) {
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
    ICHECK(args.size() == 13U || args.size() == 14U);
    std::string shape = args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = GetVarPtr_(args[6]);
    std::string a_bias = PrintExpr_(args[7]);
    std::string b_ref = GetVarPtr_(args[8]);
    std::string b_bias = PrintExpr_(args[9]);
    std::string c_ref = GetVarPtr_(args[10]);
    std::string c_bias = PrintExpr_(args[11]);

    // Generate call to tl.ptx_mma dispatcher
    PrintIndent();
    stream << "tl.ptx_mma(\"" << shape << "\", \"" << A_layout << "\", \""
           << B_layout << "\", \"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << a_ref << ", " << a_bias << ", " << b_ref
           << ", " << b_bias << ", " << c_ref << ", " << c_bias << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_mma_sm70")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tl.ptx_mma_sp")) {
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
    ICHECK_EQ(args.size(), 16U);
    std::string shape = args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = GetVarPtr_(args[6]);
    std::string a_bias = PrintExpr_(args[7]);
    std::string b_ref = GetVarPtr_(args[8]);
    std::string b_bias = PrintExpr_(args[9]);
    std::string c_ref = GetVarPtr_(args[10]);
    std::string c_bias = PrintExpr_(args[11]);
    std::string meta_ref = GetVarPtr_(args[12]);
    std::string meta_bias = PrintExpr_(args[13]);
    std::string sparse_selector = PrintExpr_(args[14]);
    std::string saturate = CodegenBool(args[15]) ? "True" : "False";

    PrintIndent();
    stream << "tl.ptx_mma_sp(\"" << shape << "\", \"" << A_layout << "\", \""
           << B_layout << "\", \"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << a_ref << ", " << a_bias << ", " << b_ref
           << ", " << b_bias << ", " << c_ref << ", " << c_bias << ", "
           << meta_ref << ", " << meta_bias << ", " << sparse_selector << ", "
           << saturate << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_wgmma_ss")) {
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
    ICHECK_EQ(args.size(), 15U) << "ptx_wgmma_ss expects 15 args";
    std::string shape = args[0].as_or_throw<StringImm>()->value;
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    bool a_is_k_major = CodegenBool(args[1]);
    bool b_is_k_major = CodegenBool(args[2]);
    std::string A_dtype = args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = args[5].as_or_throw<StringImm>()->value;
    std::string a_desc = PrintExpr_(args[6]);
    std::string A_offset = PrintExpr_(args[7]);
    std::string b_desc = PrintExpr_(args[8]);
    std::string B_offset = PrintExpr_(args[9]);
    std::string c_ref = GetVarPtr_(args[10]);
    std::string c_offset = PrintExpr_(args[11]);
    std::string scale_out = PrintExpr_(args[12]);
    bool scale_in_a = CodegenBool(args[13]);
    bool scale_in_b = CodegenBool(args[14]);
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
  } else if (IsCodegenOp(op, "tl.ptx_wgmma_rs")) {
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
    ICHECK_EQ(args.size(), 14U) << "ptx_wgmma_rs expects 14 args";
    std::string shape = args[0].as_or_throw<StringImm>()->value;
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    bool b_is_k_major = CodegenBool(args[1]);
    std::string A_dtype = args[2].as_or_throw<StringImm>()->value;
    std::string B_dtype = args[3].as_or_throw<StringImm>()->value;
    std::string C_dtype = args[4].as_or_throw<StringImm>()->value;
    std::string a_ref = GetVarPtr_(args[5]);
    std::string A_offset = PrintExpr_(args[6]);
    std::string b_desc = PrintExpr_(args[7]);
    std::string B_offset = PrintExpr_(args[8]);
    std::string c_ref = GetVarPtr_(args[9]);
    std::string c_offset = PrintExpr_(args[10]);
    std::string scale_out = PrintExpr_(args[11]);
    bool scale_in_a = CodegenBool(args[12]);
    bool scale_in_b = CodegenBool(args[13]);
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
  } else if (IsCodegenOp(op, "tl.ptx_tcgen05_mma_ss")) {
    ICHECK_EQ(args.size(), 15U) << "ptx_tcgen05_mma_ss expects 15 arguments";
    std::string kind_dtype = args[0].as_or_throw<StringImm>()->value;
    std::string a_desc = PrintExpr_(args[1]);
    std::string A_offset = PrintExpr_(args[2]);
    std::string b_desc = PrintExpr_(args[3]);
    std::string B_offset = PrintExpr_(args[4]);
    std::string c_ref = PrintExpr_(args[5]);
    std::string c_offset = PrintExpr_(args[6]);
    std::string desc_val = PrintExpr_(args[7]);
    std::string scale_out = PrintExpr_(args[8]);
    std::string mask0 = PrintExpr_(args[9]);
    std::string mask1 = PrintExpr_(args[10]);
    std::string mask2 = PrintExpr_(args[11]);
    std::string mask3 = PrintExpr_(args[12]);
    bool enable_ws = CodegenBool(args[13]);
    bool enable_2cta = CodegenBool(args[14]);
    PrintIndent();
    if (enable_ws) {
      ICHECK(!enable_2cta)
          << "enable_ws and enable_2cta cannot be true at the same time";
      stream << "tl.tcgen05mma_ws_ss(\"" << kind_dtype << "\", (" << a_desc
             << " + " << A_offset << "), (" << b_desc << " + " << B_offset
             << "), " << c_ref << "[0] + " << c_offset << ", " << desc_val
             << ", " << scale_out << ")\n";
    } else {
      stream << "tl.tcgen05mma_ss(\"" << kind_dtype << "\", (" << a_desc
             << " + " << A_offset << "), (" << b_desc << " + " << B_offset
             << "), " << c_ref << "[0] + " << c_offset << ", " << desc_val
             << ", " << scale_out << ", " << mask0 << ", " << mask1 << ", "
             << mask2 << ", " << mask3
             << ", use_2cta=" << (enable_2cta ? "True" : "False") << ")\n";
    }
  } else if (IsCodegenOp(op, "tl.ptx_tcgen05_mma_ts")) {
    ICHECK_EQ(args.size(), 14U) << "ptx_tcgen05_mma_ts expects 14 arguments";
    std::string kind_dtype = args[0].as_or_throw<StringImm>()->value;
    std::string a_ref = PrintExpr_(args[1]);
    std::string A_offset = PrintExpr_(args[2]);
    std::string b_desc = PrintExpr_(args[3]);
    std::string B_offset = PrintExpr_(args[4]);
    std::string c_ref = PrintExpr_(args[5]);
    std::string c_offset = PrintExpr_(args[6]);
    std::string desc_val = PrintExpr_(args[7]);
    std::string scale_out = PrintExpr_(args[8]);
    std::string mask0 = PrintExpr_(args[9]);
    std::string mask1 = PrintExpr_(args[10]);
    std::string mask2 = PrintExpr_(args[11]);
    std::string mask3 = PrintExpr_(args[12]);
    bool enable_2cta = CodegenBool(args[13]);
    PrintIndent();
    stream << "tl.tcgen05mma_ts(\"" << kind_dtype << "\", " << a_ref << "[0] + "
           << A_offset << ", (" << b_desc << " + " << B_offset << "), " << c_ref
           << "[0] + " << c_offset << ", " << desc_val << ", " << scale_out
           << ", " << mask0 << ", " << mask1 << ", " << mask2 << ", " << mask3
           << ", use_2cta=" << (enable_2cta ? "True" : "False") << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_tcgen05_mma_blockscaled_ss")) {
    ICHECK_EQ(args.size(), 16U)
        << "ptx_tcgen05_mma_blockscaled_ss expects 16 arguments";
    std::string kind_dtype = args[0].as_or_throw<StringImm>()->value;
    std::string a_desc = PrintExpr_(args[1]);
    std::string A_offset = PrintExpr_(args[2]);
    std::string b_desc = PrintExpr_(args[3]);
    std::string B_offset = PrintExpr_(args[4]);
    std::string c_ref = PrintExpr_(args[5]);
    std::string c_offset = PrintExpr_(args[6]);
    std::string desc_val = PrintExpr_(args[7]);
    std::string scale_out = PrintExpr_(args[8]);
    std::string sfa_ref = PrintExpr_(args[9]);
    std::string sfa_offset = PrintExpr_(args[10]);
    std::string sfb_ref = PrintExpr_(args[11]);
    std::string sfb_offset = PrintExpr_(args[12]);
    bool enable_2cta = CodegenBool(args[15]);
    PrintIndent();
    stream << "tl.tcgen05mma_blockscaled_ss(\"" << kind_dtype << "\", ("
           << a_desc << " + " << A_offset << "), (" << b_desc << " + "
           << B_offset << "), " << c_ref << "[0] + " << c_offset << ", "
           << desc_val << ", " << scale_out << ", " << sfa_ref << "[0] + "
           << sfa_offset << ", " << sfb_ref << "[0] + " << sfb_offset
           << ", use_2cta=" << (enable_2cta ? "True" : "False") << ")\n";
  } else if (IsCodegenOp(op, "tl.tcgen05_ld")) {
    ICHECK(args.size() == 6U || args.size() == 7U)
        << "tcgen05_ld expects 6 or 7 arguments";
    int inst_bits = CodegenInt(args[0]);
    int chunks = CodegenInt(args[1]);
    bool pack16 = CodegenBool(args[2]);
    std::string tmem_start_col = PrintExpr_(args[3]);
    std::string col_offset = PrintExpr_(args[4]);
    // The optional datapath count selects the half-subpartition (16dp)
    // wrappers; a six-argument call from before it existed means the
    // sub-partition-filling 32dp wrappers.
    bool has_datapaths = args.size() == 7U;
    int datapaths = has_datapaths ? CodegenInt(args[5]) : 32;
    std::string dst_ptr = PrintExpr_(args[has_datapaths ? 6 : 5]);
    PrintIndent();
    stream << "tl.tcgen05_ld_" << datapaths << "dp" << inst_bits << "bNx("
           << chunks << ", " << (pack16 ? "True" : "False") << ", "
           << tmem_start_col << ", " << col_offset << ", " << dst_ptr << ")\n";
  } else if (IsCodegenOp(op, "tl.tcgen05_st")) {
    ICHECK(args.size() == 6U || args.size() == 7U)
        << "tcgen05_st expects 6 or 7 arguments";
    int inst_bits = CodegenInt(args[0]);
    int chunks = CodegenInt(args[1]);
    bool unpack16 = CodegenBool(args[2]);
    std::string tmem_start_col = PrintExpr_(args[3]);
    std::string col_offset = PrintExpr_(args[4]);
    // The optional datapath count selects the half-subpartition (16dp)
    // wrappers; a six-argument call from before it existed means the
    // sub-partition-filling 32dp wrappers.
    bool has_datapaths = args.size() == 7U;
    int datapaths = has_datapaths ? CodegenInt(args[5]) : 32;
    std::string src_ptr = PrintExpr_(args[has_datapaths ? 6 : 5]);
    PrintIndent();
    stream << "tl.tcgen05_st_" << datapaths << "dp" << inst_bits << "bNx("
           << chunks << ", " << (unpack16 ? "True" : "False") << ", "
           << tmem_start_col << ", " << col_offset << ", " << src_ptr << ")\n";
  } else if (IsCodegenOp(op, "tl.tcgen05_mma_arrive")) {
    ICHECK_EQ(args.size(), 1U) << "tcgen05_mma_arrive expects 1 argument";
    PrintIndent();
    bool use_2cta = annotations.find("use_2cta") != annotations.end() &&
                    CodegenBool(annotations["use_2cta"]);
    stream << "tl.tcgen05_mma_arrive(" << PrintExpr_(args[0])
           << ", use_2cta=" << (use_2cta ? "True" : "False") << ")\n";
  } else if (IsCodegenOp(op, "tl.tcgen05_before_thread_sync")) {
    ICHECK_EQ(args.size(), 0U)
        << "tcgen05_before_thread_sync expects no arguments";
    PrintIndent();
    stream << "tl.tcgen05_before_thread_sync()\n";
  } else if (IsCodegenOp(op, "tl.tcgen05_after_thread_sync")) {
    ICHECK_EQ(args.size(), 0U)
        << "tcgen05_after_thread_sync expects no arguments";
    PrintIndent();
    stream << "tl.tcgen05_after_thread_sync()\n";
  } else if (IsCodegenOp(op, "tl.ptx_tcgen05_cp_warpx4")) {
    ICHECK_EQ(args.size(), 3U) << "ptx_tcgen05_cp_warpx4 expects 3 arguments";
    std::string smem_ptr = PrintExpr_(args[0]);
    std::string tmem_ptr = PrintExpr_(args[1]);
    std::string tmem_col_offset = PrintExpr_(args[2]);
    bool use_2cta = false;
    if (annotations.find("use_2cta") != annotations.end()) {
      use_2cta = CodegenBool(annotations["use_2cta"]);
    }
    PrintIndent();
    stream << "tl.tcgen05_cp_warpx4(" << smem_ptr << ", " << tmem_ptr << ", "
           << tmem_col_offset << ", use_2cta=" << (use_2cta ? "True" : "False")
           << ")\n";
  } else if (IsCodegenOp(op, "tl.ptx_tcgen05_sf_warp_transpose")) {
    ICHECK_EQ(args.size(), 1U)
        << "ptx_tcgen05_sf_warp_transpose expects 1 argument";
    PrintIndent();
    stream << "tl.tcgen05_sf_warp_transpose(" << PrintExpr_(args[0]) << ")\n";
  } else if (IsCodegenOp(op, "tirx.ptx_legacy.ldmatrix")) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared
    // memory.
    ICHECK_EQ(args.size(), 7U);
    bool trans = CodegenBool(args[0]);
    int num = CodegenInt(args[1]);
    std::string local_ptr = GetVarPtr_(args[3]);
    std::string local_elem_offset = PrintExpr_(args[4]);
    std::string smem_ptr = PrintExpr_(args[5]);
    std::string smem_elem_offset = PrintExpr_(args[6]);

    std::string func_name = "tl.ptx_ldmatrix_x" + std::to_string(num);
    if (trans)
      func_name += "_trans";
    PrintIndent();
    stream << func_name << "(" << smem_ptr << " + " << smem_elem_offset << ", "
           << local_ptr << " + " << local_elem_offset << ")\n";
  } else if (IsCodegenOp(op, "tirx.cuda.mma_store")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tirx.cuda.mma_fill")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tl.ptx_cp_async_bulk")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tl.ptx_st_bulk_shared")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tl.ptx_wait_barrier")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tl.ptx_ldg32")) {
    TVM_FFI_THROW(InternalError) << "Currently unsupported op: " << op->op;
  } else if (IsCodegenOp(op, "tirx.reinterpret")) {
    PrimType tgt_dtype = CodegenDType(op);
    PrimType src_dtype = CodegenDType(args[0]);
    ICHECK_EQ(tgt_dtype.lanes() * tgt_dtype.bits(),
              src_dtype.lanes() * src_dtype.bits())
        << "reinterpret expects source and target to have the same number of "
           "bits";

    const TensorLoadObj *load = args[0].as<TensorLoadObj>();
    if (load) {
      std::string scope;
      const VarObj *buffer_var = load->source.as<VarObj>();
      if (alloc_storage_scope_.count(buffer_var)) {
        scope = alloc_storage_scope_.at(buffer_var);
      }
      if (scope.empty()) {
        scope = CodegenStorageScope(load->source.as_or_throw<Var>());
      }
      if (scope == "local.var") {
        std::string expr_str = GetBufferRef_(
            src_dtype, load->source.as<VarObj>(),
            LinearizeBufferIndices_(load->source.as<VarObj>(), load->indices));
        os << "tl.bitcast(" << expr_str << ", ";
        PrintType(tgt_dtype.WithLanes(1), os);
        os << ")";
        return;
      }

      // Path 1: BufferLoad - use recast_ptr for memory access
      PrimExpr index =
          LinearizeBufferIndices_(load->source.as<VarObj>(), load->indices);
      if (const prim::RampObj *node = index.as<prim::RampObj>(); node) {
        ICHECK(prim::IsOne(node->stride))
            << "reinterpret expects contiguous elements";
        index = node->base;
      } else {
        tvm::PVar<PrimExpr> ramp_base;
        if (tvm::ramp(ramp_base, 1, tgt_dtype.lanes()).Match(index)) {
          index = ramp_base.Eval();
        }
      }

      auto ptr_str = GetBufferPtr_(load->source.as<VarObj>(), index);
      os << "tl.make_tensor(tl.recast_ptr(" << ptr_str << ", dtype=";
      PrintType(tgt_dtype.WithLanes(1), os);
      os << "), (" << tgt_dtype.lanes() << ",))";
      if (tgt_dtype.IsScalar()) {
        os << "[0]";
      } else {
        os << ".load()";
      }
    } else {
      // Path 2: General expression - use arith.bitcast
      std::string expr_str = PrintExpr_(args[0]);
      os << "tl.bitcast(" << expr_str << ", ";
      PrintType(tgt_dtype.WithLanes(1), os);
      os << ")";
    }
  } else if (IsCodegenOp(op, "tirx.gpu_thread_return")) {
    os << "return";
  } else if (IsCodegenOp(op, "tl.shfl_sync")) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_sync expects <mask, value, src_lane, width>.";
    os << "tl.__shfl_sync(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ", " << PrintExpr_(args[2]) << ", "
       << PrintExpr_(args[3]) << ")";
  } else if (IsCodegenOp(op, "tl.shfl_xor_sync")) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "tl.__shfl_xor_sync(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ", " << PrintExpr_(args[2]) << ", "
       << PrintExpr_(args[3]) << ")";
  } else if (IsCodegenOp(op, "tl.shfl_down_sync")) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_down_sync expects <mask, value, delta, width>.";
    os << "tl.__shfl_down_sync(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ", " << PrintExpr_(args[2]) << ", "
       << PrintExpr_(args[3]) << ")";
  } else if (IsCodegenOp(op, "tl.shfl_up_sync")) {
    ICHECK_EQ(args.size(), 4U)
        << "tl.shfl_up_sync expects <mask, value, delta, width>.";
    os << "tl.__shfl_up_sync(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ", " << PrintExpr_(args[2]) << ", "
       << PrintExpr_(args[3]) << ")";
  } else if (IsCodegenOp(op, "tl.match_any_sync")) {
    ICHECK_EQ(args.size(), 2U) << "tl.match_any_sync expects <mask, value>.";
    os << "tl.__match_any_sync(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.get_lane_idx")) {
    // get_lane_idx(warp_size?) -> threadIdx.x % warp_size
    ICHECK_LE(args.size(), 1U)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    std::string warp_size = args.empty() ? "32" : PrintExpr_(args[0]);
    os << "(tl.thread_idx() % " << warp_size << ")";
  } else if (IsCodegenOp(op, "tl.get_warp_idx_sync")) {
    // get_warp_idx_sync(warp_size?) -> threadIdx.x // warp_size
    ICHECK_LE(args.size(), 1U)
        << "tl.get_warp_idx_sync expects at most one argument <warp_size>.";
    std::string warp_size = args.empty() ? "32" : PrintExpr_(args[0]);
    os << "(tl.thread_idx() // " << warp_size << ")";
  } else if (IsCodegenOp(op, "tl.get_warp_idx")) {
    // get_warp_idx(warp_size?) -> threadIdx.x // warp_size
    ICHECK_LE(args.size(), 1U)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    std::string warp_size = args.empty() ? "32" : PrintExpr_(args[0]);
    os << "(tl.thread_idx() // " << warp_size << ")";
  } else if (IsCodegenOp(op, "tl.get_warp_group_idx")) {
    // get_warp_group_idx(warp_size?, warps_per_group?) ->
    //   threadIdx.x // (warp_size * warps_per_group)
    ICHECK_LE(args.size(), 2U)
        << "tl.get_warp_group_idx expects <warp_size, warps_per_group>.";
    std::string warp_size = !args.empty() ? PrintExpr_(args[0]) : "32";
    std::string warps_per_group = args.size() >= 2 ? PrintExpr_(args[1]) : "4";
    os << "(tl.thread_idx() // (" << warp_size << " * " << warps_per_group
       << "))";
  } else if (IsCodegenOp(op, "tl.tl_shuffle_elect")) {
    os << "tl.shuffle_elect(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.initialize_wgmma_descriptor")) {
    // TIR args: (descriptor, start_address, layout_type, leading, stride)
    // Python args: (layout_type, leading, stride, desc, start_address)
    ICHECK_EQ(args.size(), 5U)
        << "initialize_wgmma_descriptor expects 5 arguments";
    std::string descriptor = PrintExpr_(args[0]);
    std::string start_address = PrintExpr_(args[1]);
    std::string layout_type = PrintExpr_(args[2]);
    std::string leading = PrintExpr_(args[3]);
    std::string stride = PrintExpr_(args[4]);
    os << "tl.initialize_wgmma_descriptor(" << layout_type << ", " << leading
       << ", " << stride << ", " << descriptor << ", " << start_address << ")";
  } else if (IsCodegenOp(op, "tl.initialize_tcgen05_descriptor")) {
    ICHECK_EQ(args.size(), 7U)
        << "initialize_tcgen05_descriptor expects 7 arguments";
    std::string descriptor = PrintExpr_(args[0]);
    std::string start_address = PrintBytePointerExpr_(args[1]);
    std::string leading = PrintExpr_(args[2]);
    std::string stride = PrintExpr_(args[3]);
    std::string base_offset = PrintExpr_(args[4]);
    std::string leading_abs = PrintExpr_(args[5]);
    std::string swizzle_mode = PrintExpr_(args[6]);
    os << "tl.initialize_tcgen05_descriptor(" << descriptor << ", "
       << start_address << ", " << leading << ", " << stride << ", "
       << base_offset << ", " << leading_abs << ", " << swizzle_mode << ")";
  } else if (IsCodegenOp(op, "tl.increase_descriptor_offset")) {
    ICHECK_EQ(args.size(), 2U)
        << "increase_descriptor_offset expects 2 arguments";
    std::string descriptor = PrintExpr_(args[0]);
    std::string offset = PrintExpr_(args[1]);
    os << "tl.increase_descriptor_offset(" << descriptor << ", " << offset
       << ")";
  } else if (IsCodegenOp(op, "tl.__exp")) {
    os << "tl.exp(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__exp10")) {
    os << "tl.exp10(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__log")) {
    os << "tl.log(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__log2")) {
    os << "tl.log2(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__log10")) {
    os << "tl.log10(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__tan")) {
    os << "tl.tan(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__cos")) {
    os << "tl.cos(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.__sin")) {
    os << "tl.sin(" << PrintExpr_(args[0]) << ", fastmath=True)";
  } else if (IsCodegenOp(op, "tl.fast_rcp")) {
    os << "(1.0 / " << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.ieee_add")) {
    // ieee_add(a, b, rounding_mode)
    std::string rounding_mode = args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fadd(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ", rounding=\"" << rounding_mode << "\")";
  } else if (IsCodegenOp(op, "tl.ieee_sub")) {
    std::string rounding_mode = args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fsub(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ", rounding=\"" << rounding_mode << "\")";
  } else if (IsCodegenOp(op, "tl.ieee_mul")) {
    std::string rounding_mode = args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fmul(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ", rounding=\"" << rounding_mode << "\")";
  } else if (IsCodegenOp(op, "tl.ieee_fmaf")) {
    // ieee_fmaf(a, b, c, rounding_mode)
    std::string rounding_mode = args[3].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fmaf(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ", " << PrintExpr_(args[2]) << ", rounding=\"" << rounding_mode
       << "\")";
  } else if (IsCodegenOp(op, "tl.ieee_frcp")) {
    std::string rounding_mode = args[1].as_or_throw<StringImm>()->value;
    os << "tl.ieee_frcp(" << PrintExpr_(args[0]) << ", rounding=\""
       << rounding_mode << "\")";
  } else if (IsCodegenOp(op, "tl.ieee_fsqrt")) {
    std::string rounding_mode = args[1].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fsqrt(" << PrintExpr_(args[0]) << ", rounding=\""
       << rounding_mode << "\")";
  } else if (IsCodegenOp(op, "tl.ieee_frsqrt")) {
    os << "tl.rsqrt(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.ieee_fdiv")) {
    std::string rounding_mode = args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fdiv(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ", rounding=\"" << rounding_mode << "\")";
  } else if (IsCodegenOp(op, "tl.fma") || IsCodegenOp(op, "tl.fmul")) {
    // Round-to-nearest fma/mul map onto the IEEE helpers with the fixed
    // "rn" mode. The CuTeDSL helpers are scalar-only, so vector calls are
    // expanded per lane through a register tensor.
    bool is_fma = IsCodegenOp(op, "tl.fma");
    const char *py_func = is_fma ? "tl.ieee_fmaf" : "tl.ieee_fmul";
    if (CodegenDType(op).IsScalar()) {
      os << py_func << "(" << PrintExpr_(args[0]) << ", "
         << PrintExpr_(args[1]);
      if (is_fma) {
        os << ", " << PrintExpr_(args[2]);
      }
      os << ", rounding=\"rn\")";
    } else {
      int lanes = CodegenDType(op).lanes();
      std::vector<std::string> vec_ids;
      vec_ids.reserve(args.size());
      for (const Expr &arg : args) {
        ICHECK_EQ(CodegenDType(arg).lanes(), lanes)
            << "tl." << (is_fma ? "fma" : "fmul")
            << " expects vectorized arguments to share the call lanes";
        vec_ids.push_back(
            SSAGetID(PrintExpr_(arg), arg->ty,
                     tvm::SideEffect(arg) == CallEffectKind::kPure));
      }
      std::string sret = name_supply_->FreshName(is_fma ? "_fma" : "_fmul");
      PrintIndent();
      stream << sret << " = tl.make_rmem_tensor((" << lanes << ",), ";
      PrintType(CodegenDType(op).WithLanes(1), stream);
      stream << ")\n";
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value;
        value << py_func << "(";
        for (size_t j = 0; j < vec_ids.size(); ++j) {
          if (j != 0) {
            value << ", ";
          }
          PrintVecElemLoad_(vec_ids[j], CodegenDType(args[j]), i, value);
        }
        value << ", rounding=\"rn\")";
        PrintVecElemStore_(sret, CodegenDType(op), i, value.str());
      }
      os << sret << ".load()";
    }
  } else if (IsCodegenOp(op, "tl.warp_reduce_sum")) {
    os << "tl.warp_reduce_sum(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.warp_reduce_max")) {
    os << "tl.warp_reduce_max(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.warp_reduce_min")) {
    os << "tl.warp_reduce_min(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.warp_reduce_bitand")) {
    os << "tl.warp_reduce_bitand(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.warp_reduce_bitor")) {
    os << "tl.warp_reduce_bitor(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tl.add2")) {
    ICHECK_EQ(args.size(), 2U);
    os << "(" << PrintExpr_(args[0]) << " + " << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.sub2")) {
    ICHECK_EQ(args.size(), 2U);
    os << "(" << PrintExpr_(args[0]) << " - " << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.mul2")) {
    ICHECK_EQ(args.size(), 2U);
    os << "(" << PrintExpr_(args[0]) << " * " << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.fma2")) {
    ICHECK_EQ(args.size(), 3U);
    os << "((" << PrintExpr_(args[0]) << " * " << PrintExpr_(args[1]) << ") + "
       << PrintExpr_(args[2]) << ")";
  } else if (IsCodegenOp(op, "tl.max2") || IsCodegenOp(op, "tl.max2_nan")) {
    ICHECK_EQ(args.size(), 2U);
    os << "tl.max2(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ")";
  } else if (IsCodegenOp(op, "tl.min2") || IsCodegenOp(op, "tl.min2_nan")) {
    ICHECK_EQ(args.size(), 2U);
    os << "tl.min2(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ")";
  } else if (IsCodegenOp(op, "tl.abs2")) {
    ICHECK_EQ(args.size(), 1U);
    os << "tl.abs2(" << PrintExpr_(args[0]) << ")";
  } else if (IsCodegenOp(op, "tirx.address_of")) {
    const TensorLoadObj *load = args[0].as<TensorLoadObj>();
    ICHECK(args.size() == 1 && load);
    os << GetBufferPtr_(
        load->source.as<VarObj>(),
        LinearizeBufferIndices_(load->source.as<VarObj>(), load->indices));
  } else if (IsCodegenOp(op, "tirx.handle_add_byte_offset")) {
    ICHECK_EQ(args.size(), 2U);
    os << "tl.handle_add_byte_offset(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.atomic_add_elem_op")) {
    // atomic_add_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(args[0]);
    std::string src_value = PrintExpr_(args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicAdd(" << dst_ptr << ", " << src_value << ")\n";
  } else if (IsCodegenOp(op, "tl.atomic_add_ret_elem_op")) {
    // atomic_add_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicAdd(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ")";
  } else if (IsCodegenOp(op, "tl.atomic_addx2_elem_op")) {
    // atomic_addx2_elem_op(dst_ptr, src_ptr[, memory_order]) -> may return prev
    // value
    os << "tl.AtomicAddx2(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.atomic_addx4_elem_op")) {
    // atomic_addx4_elem_op(dst_ptr, src_ptr[, memory_order]) -> may return prev
    // value
    os << "tl.AtomicAddx4(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.atomic_load_elem_op")) {
    // atomic_load_elem_op(src_ptr, memory_order) -> returns loaded value
    os << "tl.AtomicLoad(" << PrintExpr_(args[0]) << ", " << PrintExpr_(args[1])
       << ")";
  } else if (IsCodegenOp(op, "tl.atomic_store_elem_op")) {
    // atomic_store_elem_op(dst_ptr, value, memory_order)
    std::string dst_ptr = PrintExpr_(args[0]);
    std::string value = PrintExpr_(args[1]);
    std::string memory_order = PrintExpr_(args[2]);
    this->PrintIndent();
    this->stream << "tl.AtomicStore(" << dst_ptr << ", " << value << ", "
                 << memory_order << ")\n";
  } else if (IsCodegenOp(op, "tl.atomic_or_elem_op")) {
    // atomic_or_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(args[0]);
    std::string src_value = PrintExpr_(args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicOr(" << dst_ptr << ", " << src_value;
    if (args.size() > 2) {
      this->stream << ", " << PrintExpr_(args[2]);
    }
    this->stream << ")\n";
  } else if (IsCodegenOp(op, "tl.atomic_max_elem_op")) {
    // atomic_max_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(args[0]);
    std::string src_value = PrintExpr_(args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicMax(" << dst_ptr << ", " << src_value << ")\n";
  } else if (IsCodegenOp(op, "tl.atomic_max_ret_elem_op")) {
    // atomic_max_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicMaxRet(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "tl.atomic_min_elem_op")) {
    // atomic_min_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(args[0]);
    std::string src_value = PrintExpr_(args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicMin(" << dst_ptr << ", " << src_value << ")\n";
  } else if (IsCodegenOp(op, "tl.atomic_min_ret_elem_op")) {
    // atomic_min_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicMinRet(" << PrintExpr_(args[0]) << ", "
       << PrintExpr_(args[1]) << ")";
  } else if (IsCodegenOp(op, "prim.rshift")) {
    // CuTeDSL type promotion fix: Int8 >> 4 returns Int32 in CuTeDSL,
    // but TIR expects result type to match operand type. Wrap in explicit
    // type conversion to match CUDA behavior.
    ICHECK_EQ(args.size(), 2U);
    PrimType result_dtype = CodegenDType(op);
    std::string lhs = PrintExpr_(args[0]);
    std::string rhs = PrintExpr_(args[1]);
    PrintType(result_dtype, os);
    os << "((" << lhs << " >> " << rhs << "))";
  } else if (IsCodegenOp(op, "prim.lshift")) {
    // Same fix for shift_left
    ICHECK_EQ(args.size(), 2U);
    PrimType result_dtype = CodegenDType(op);
    std::string lhs = PrintExpr_(args[0]);
    std::string rhs = PrintExpr_(args[1]);
    PrintType(result_dtype, os);
    os << "((" << lhs << " << " << rhs << "))";
  } else {
    CodeGenTileLangPY::Dispatch_(op, os);
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::SelectObj *op,
                                       std::ostream &os) { // NOLINT(*)
  // Emit Python ternary: (true_val if cond else false_val).
  // This yields ArithValue which is fine in expression contexts (range(),
  // etc.). When used as a BufferStore value that requires TensorSSA, the store
  // handler will wrap it with tl.where() at statement level.
  std::string cond = PrintExpr_(op->condition);
  std::string t = PrintExpr_(op->true_value);
  std::string f = PrintExpr_(op->false_value);
  auto cast_branch = [&](std::string value, PrimType branch_dtype) {
    PrimType result_dtype = CodegenDType(op);
    if (result_dtype.MatchesCode(kDLUInt) && result_dtype.bits() == 8 &&
        result_dtype.IsScalar()) {
      std::ostringstream casted;
      PrintType(result_dtype, casted);
      casted << "(" << value << ")";
      return casted.str();
    }
    if (branch_dtype == result_dtype || !branch_dtype.IsScalar() ||
        !result_dtype.IsScalar()) {
      return value;
    }
    const bool integer_like =
        (branch_dtype.MatchesCode(kDLInt) ||
         branch_dtype.MatchesCode(kDLUInt)) &&
        (result_dtype.MatchesCode(kDLInt) || result_dtype.MatchesCode(kDLUInt));
    const bool float_like = branch_dtype.MatchesCode(kDLFloat) &&
                            result_dtype.MatchesCode(kDLFloat);
    if (!integer_like && !float_like) {
      return value;
    }
    std::ostringstream casted;
    PrintType(result_dtype, casted);
    casted << "(" << value << ")";
    return casted.str();
  };
  t = cast_branch(t, op->true_value.ty());
  f = cast_branch(f, op->false_value.ty());
  os << "(" << t << " if " << cond << " else " << f << ")";
}

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::LetObj *op,
                                       std::ostream &os) { // NOLINT(*)
  std::string value = PrintExpr_(op->value);
  ICHECK(!var_idmap_.count(op->var.get()));
  const bool is_zero_like = IsZeroTensorLike(op->value, &zero_like_vars_);
  PrintIndent();
  stream << AllocVarID(op->var.get()) << " = " << value << "\n";
  if (is_zero_like) {
    zero_like_vars_.insert(op->var.get());
  }
  os << PrintExpr_(op->body);
  if (is_zero_like) {
    zero_like_vars_.erase(op->var.get());
  }
  bool removed = var_idmap_.erase(op->var.get());
  ICHECK(removed);
}

void CodeGenTileLangCuTeDSL::Dispatch_(const prim::ShuffleObj *op,
                                       std::ostream &os) { // NOLINT(*)
  std::vector<std::string> concat_vec;

  for (const PrimExpr &vec : op->vectors) {
    std::string vec_value = PrintExpr_(vec);
    if (vec.ty().IsScalar()) {
      concat_vec.push_back(vec_value);
    } else {
      std::string vec_id = SSAGetID(
          vec_value, vec.ty(), tvm::SideEffect(vec) == CallEffectKind::kPure);
      for (int i = 0; i < vec.ty().lanes(); ++i) {
        std::ostringstream elem;
        PrintVecElemLoad_(vec_id, vec.ty(), i, elem);
        concat_vec.push_back(elem.str());
      }
    }
  }

  if (op->indices.size() == 1) {
    ICHECK(op->indices[0]->IsInstance<IntImmObj>())
        << "ShuffleObj indices must be constants at codegen time, got "
        << op->indices[0];
    int64_t idx = CodegenInt(op->indices[0]);
    ICHECK_GE(idx, 0);
    ICHECK_LT(idx, static_cast<int64_t>(concat_vec.size()));
    os << concat_vec[idx];
    return;
  }

  std::string sret = name_supply_->FreshName("_shuffle");
  PrintIndent();
  stream << sret << " = tl.make_rmem_tensor((" << op->indices.size() << ",), ";
  PrintType(CodegenDType(op).WithLanes(1), stream);
  stream << ")\n";

  for (size_t i = 0; i < op->indices.size(); ++i) {
    ICHECK(op->indices[i]->IsInstance<IntImmObj>())
        << "ShuffleObj indices must be constants at codegen time, got "
        << op->indices[i];
    int64_t idx = CodegenInt(op->indices[i]);
    ICHECK_GE(idx, 0);
    ICHECK_LT(idx, static_cast<int64_t>(concat_vec.size()));
    PrintVecElemStore_(sret, CodegenDType(op), static_cast<int>(i),
                       concat_vec[idx]);
  }
  os << sret << ".load()";
}

void CodeGenTileLangCuTeDSL::Dispatch_(const TensorLoadObj *op,
                                       std::ostream &os) { // NOLINT(*)
  ICHECK_GE(op->indices.size(), 1U)
      << "Buffer load must have at least one index.";

  PrimType value_dtype = CodegenDType(op);
  PrimExpr index =
      LinearizeBufferIndices_(op->source.as<VarObj>(), op->indices);
  Var buffer_var = op->source.as_or_throw<Var>();
  PrimType element_dtype = op->source.as_or_throw<TensorVar>().dtype();

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
    PrimExpr scalar_base(ffi::UnsafeInit{});
    if (value_lanes == element_dtype.lanes()) {
      scalar_base = index * value_lanes;
    } else {
      // Contiguous vectorized load: extract base from ramp index
      tvm::PVar<PrimExpr> ramp_base;
      ICHECK(tvm::ramp(ramp_base, 1, value_lanes / element_dtype.lanes())
                 .Match(index))
          << "Non-contiguous narrow-precision load not supported";
      scalar_base = ramp_base.Eval() * element_dtype.lanes();
    }

    // Load aligned vector into an rmem tensor (not a raw MLIR vector).
    // This ensures downstream element access uses rmem tensor indexing
    // (which CuTeDSL handles correctly for FP8) instead of MLIR
    // extractelement (which fails due to unrealized_conversion_cast).
    std::string aligned_rmem = name_supply_->FreshName("_aload");

    if (value_dtype.bits() < 8) {
      const int packed_bits = value_lanes * value_dtype.bits();
      ICHECK_EQ(packed_bits % 8, 0)
          << "Sub-byte vector loads must cover whole bytes";
      const int packed_bytes = packed_bits / 8;
      const int aligned_bytes = aligned_lanes * value_dtype.bits() / 8;
      const int byte_lanes = 8 / value_dtype.bits();

      std::string ptr_str;
      if (scope == "local" || scope == "shared" || scope == "shared.dyn") {
        ptr_str = vid + ".iterator";
      } else {
        bool is_handle_match =
            HandleTypeMatch_(buffer_var.get(), element_dtype);
        if (is_handle_match) {
          ptr_str = vid + ".iterator";
        } else {
          std::ostringstream ptr_os;
          ptr_os << "tl.recast_ptr(" << vid << ".iterator, dtype=";
          PrintType(value_dtype.WithLanes(1), ptr_os);
          ptr_os << ")";
          ptr_str = ptr_os.str();
        }
      }

      std::string dst_bytes = name_supply_->FreshName("_pload_dst");
      std::string src_view = name_supply_->FreshName("_pload_view");
      std::string src_bytes = name_supply_->FreshName("_pload_src");
      PrintIndent();
      stream << aligned_rmem << " = tl.make_rmem_tensor((" << aligned_lanes
             << ",), ";
      PrintType(value_dtype.WithLanes(1), stream);
      stream << ")\n";
      PrintIndent();
      stream << dst_bytes << " = cute.recast_tensor(" << aligned_rmem
             << ", cutlass.Uint8)\n";
      for (int i = 0; i < aligned_bytes; ++i) {
        PrintIndent();
        stream << dst_bytes << "[" << i << "] = cutlass.Uint8(0)\n";
      }
      PrintIndent();
      stream << src_view << " = tl.make_tensor_at_offset(" << ptr_str << ", "
             << PrintExpr_(scalar_base) << ", (" << value_lanes
             << ",), div_by=" << byte_lanes << ")\n";
      PrintIndent();
      stream << src_bytes << " = cute.recast_tensor(" << src_view
             << ", cutlass.Uint8)\n";
      for (int i = 0; i < packed_bytes; ++i) {
        PrintIndent();
        stream << dst_bytes << "[" << i << "] = " << src_bytes << "[" << i
               << "]\n";
      }

      os << aligned_rmem;
      return;
    }

    if (scope == "local" || scope == "shared" || scope == "shared.dyn") {
      // For rmem/smem: load aligned_lanes elements without cute.assume.
      // cute.assume(offset, divby=N) silently truncates offsets that
      // are not exact multiples of N; for rmem there is no hardware
      // alignment constraint, so we skip assume (use default div_by=1).
      // Shape uses aligned_lanes so .load()/.store() satisfy CuTeDSL's
      // 32-bit alignment requirement for narrow types (FP8 etc.).
      PrintIndent();
      stream << aligned_rmem << " = tl.make_rmem_tensor((" << aligned_lanes
             << ",), ";
      PrintType(value_dtype.WithLanes(1), stream);
      stream << ")\n";
      PrintIndent();
      stream << aligned_rmem
             << ".store(tl.load_tensor(tl.make_tensor_at_offset(" << vid
             << ".iterator, " << PrintExpr_(scalar_base) << ", ("
             << aligned_lanes << ",))))\n";
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
        PrintType(value_dtype.WithLanes(1), ptr_os);
        ptr_os << ")";
        ptr_str = ptr_os.str();
      }
      PrintIndent();
      stream << aligned_rmem << " = tl.make_rmem_tensor((" << value_lanes
             << ",), ";
      PrintType(value_dtype.WithLanes(1), stream);
      stream << ")\n";
      PrintIndent();
      stream << aligned_rmem
             << ".store(tl.load_tensor(tl.make_tensor_at_offset(" << ptr_str
             << ", " << PrintExpr_(scalar_base) << ", (" << value_lanes
             << ",))))\n";
    }

    // Return the rmem tensor (not .load()) so downstream code uses
    // rmem element access instead of MLIR vector extractelement.
    os << aligned_rmem;
  } else if (value_lanes == element_dtype.lanes()) {
    std::string ref =
        GetBufferRef_(value_dtype, op->source.as<VarObj>(), index);
    // Check if this is a barrier buffer - barrier pointers don't need .load()
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var.get())) {
      scope = alloc_storage_scope_.at(buffer_var.get());
    }
    if (ref.back() == ')' && scope != "shared.barrier" &&
        scope != "shared.cluster_barrier") {
      ref = "tl.load_tensor(" + ref + ")";
    }
    os << ref;
  } else {
    ICHECK_GE(value_lanes, element_dtype.lanes())
        << "Unsupported load/store: value lanes < buffer element lanes";
    bool is_contiguous = false;
    tvm::PVar<PrimExpr> base;
    if (tvm::ramp(base, 1, value_lanes / element_dtype.lanes()).Match(index)) {
      is_contiguous = true;
    }

    if (is_contiguous) {
      std::string ref =
          GetBufferRef_(value_dtype, op->source.as<VarObj>(), base.Eval());
      if (ref.back() == ')') {
        ref = "tl.load_tensor(" + ref + ")";
      }
      os << ref;
    } else {
      ICHECK(element_dtype.IsScalar())
          << "buffer element type for non-contiguous load must be scalar "
             "currently";

      std::string sret = name_supply_->FreshName("_");
      PrintIndent();
      stream << sret << " = tl.make_rmem_tensor((" << value_lanes << ",), ";
      PrintType(element_dtype, stream);
      stream << ")\n";

      std::string vid = GetVarID(buffer_var.get());
      const prim::RampObj *ramp = index.as<prim::RampObj>();
      ICHECK(ramp)
          << "Expected Ramp index for vectorized non-contiguous access";
      for (int i = 0; i < value_lanes; ++i) {
        auto idx_expr =
            sym::Analyzer()->Simplify(ramp->base + ramp->stride * i);

        PrintIndent();
        stream << sret << "[" << i << "] = "
               << GetBufferRef_(element_dtype, op->source.as<VarObj>(),
                                idx_expr)
               << "\n";
      }
      os << sret << ".load()";
    }
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const TensorStoreObj *op) {
  ICHECK_GE(op->indices.size(), 1U)
      << "Buffer store must have at least one index.";

  PrimType value_dtype = op->value.ty();
  PrimType element_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  PrimExpr index_expr =
      LinearizeBufferIndices_(op->dest.as<VarObj>(), op->indices);
  Var buffer_var = op->dest.as_or_throw<Var>();

  // Pre-compute Select/if_then_else as tl.where() at statement level.
  // Python ternary (ArithValue) is rejected by CuTeDSL .store(); tl.where()
  // produces TensorSSA which works in all store paths (.store(), vec store,
  // etc.). Also handles Cast(Select/if_then_else) — the cast is applied via
  // .to() on the tl.where result.
  bool value_is_conditional = false;
  PrimExpr cond_expr(ffi::UnsafeInit{}), true_expr(ffi::UnsafeInit{}),
      false_expr(ffi::UnsafeInit{});
  PrimType cast_to_dtype =
      PrimType::Void(); // non-void if a Cast wraps the conditional
  // Helper to detect Select/if_then_else in a PrimExpr.
  auto detect_conditional = [&](const Expr &expr) {
    if (auto sel = expr.as<prim::SelectObj>()) {
      value_is_conditional = true;
      cond_expr = sel->condition;
      true_expr = sel->true_value;
      false_expr = sel->false_value;
    } else if (auto call = expr.as<CallObj>();
               call && IsCodegenOp(call, "prim.if_then_else")) {
      value_is_conditional = true;
      cond_expr = call->args[0].as_or_throw<PrimExpr>();
      true_expr = call->args[1].as_or_throw<PrimExpr>();
      false_expr = call->args[2].as_or_throw<PrimExpr>();
    }
  };
  detect_conditional(op->value);
  if (!value_is_conditional) {
    // Check for Cast(Select/if_then_else)
    if (auto cast_node = op->value.as<prim::CastObj>()) {
      detect_conditional(cast_node->value);
      if (value_is_conditional) {
        cast_to_dtype = CodegenDType(cast_node);
      }
    }
  }

  std::string value_str;
  if (value_is_conditional) {
    int lanes = value_dtype.lanes();
    if (lanes == 0)
      lanes = 1;
    PrimType conditional_dtype =
        cast_to_dtype.bits() > 0 ? cast_to_dtype : value_dtype;
    int conditional_lanes = conditional_dtype.lanes();
    if (conditional_lanes == 0)
      conditional_lanes = 1;
    const bool true_is_zero_ast = IsZeroTensorLike(true_expr, &zero_like_vars_);
    const bool false_is_zero_ast =
        IsZeroTensorLike(false_expr, &zero_like_vars_);
    const bool true_has_buffer_load = ContainsTensorLoad(true_expr);
    const bool false_has_buffer_load = ContainsTensorLoad(false_expr);
    const bool is_subbyte_vector_conditional =
        conditional_dtype.WithLanes(1).bits() < 8 &&
        conditional_dtype.lanes() > 1;
    if ((true_is_zero_ast && false_has_buffer_load) ||
        (false_is_zero_ast && true_has_buffer_load)) {
      const bool data_is_true_branch = false_is_zero_ast;
      PrimExpr data_expr = data_is_true_branch ? true_expr : false_expr;
      PrimType elem_dtype = conditional_dtype.WithLanes(1);

      std::string result_tensor = name_supply_->FreshName("_if_tensor");
      PrintIndent();
      stream << result_tensor << " = tl.make_rmem_tensor((" << conditional_lanes
             << ",), " << DTypeToString(elem_dtype) << ")\n";

      if (elem_dtype.bits() < 8 && conditional_lanes > 1) {
        const int packed_bits = conditional_lanes * elem_dtype.bits();
        ICHECK_EQ(packed_bits % 8, 0)
            << "Sub-byte conditional vectors must cover whole bytes";
        const int packed_bytes = packed_bits / 8;
        std::string result_bytes = name_supply_->FreshName("_if_dst");
        PrintIndent();
        stream << result_bytes << " = cute.recast_tensor(" << result_tensor
               << ", cutlass.Uint8)\n";
        for (int i = 0; i < packed_bytes; ++i) {
          PrintIndent();
          stream << result_bytes << "[" << i << "] = cutlass.Uint8(0)\n";
        }
      } else {
        for (int i = 0; i < conditional_lanes; ++i) {
          PrintIndent();
          stream << result_tensor << "[" << i << "] = ";
          if (elem_dtype.MatchesCode(kDLBool)) {
            stream << "False\n";
          } else {
            stream << DTypeToString(elem_dtype) << "(0)\n";
          }
        }
      }

      std::string cond = PrintExpr_(cond_expr);
      PrintIndent();
      if (data_is_true_branch) {
        stream << "if " << cond << ":\n";
      } else {
        stream << "if not (" << cond << "):\n";
      }
      int scope = BeginScope();
      PrimType data_dtype = data_expr.ty();
      std::string data;
      if (cast_to_dtype.bits() > 0 && data_dtype != conditional_dtype) {
        if (conditional_dtype.IsScalar()) {
          data = PrintExpr_(prim::Cast(conditional_dtype, data_expr));
        } else {
          data = PrintExpr_(data_expr);
          data = "tl.as_tensor_ssa(" + data + ").to(" +
                 DTypeToString(elem_dtype) + ")";
        }
      } else {
        data = PrintExpr_(data_expr);
      }

      if (elem_dtype.bits() < 8 && conditional_lanes > 1) {
        const int packed_bits = conditional_lanes * elem_dtype.bits();
        ICHECK_EQ(packed_bits % 8, 0)
            << "Sub-byte conditional vectors must cover whole bytes";
        const int packed_bytes = packed_bits / 8;
        std::string data_tensor = name_supply_->FreshName("_if_src_tensor");
        PrintIndent();
        stream << data_tensor << " = tl.as_rmem_tensor(" << data << ", ("
               << conditional_lanes << ",), " << DTypeToString(elem_dtype)
               << ")\n";

        std::string src_bytes = name_supply_->FreshName("_if_src");
        std::string dst_bytes = name_supply_->FreshName("_if_dst");
        PrintIndent();
        stream << src_bytes << " = cute.recast_tensor(" << data_tensor
               << ", cutlass.Uint8)\n";
        PrintIndent();
        stream << dst_bytes << " = cute.recast_tensor(" << result_tensor
               << ", cutlass.Uint8)\n";
        for (int i = 0; i < packed_bytes; ++i) {
          PrintIndent();
          stream << dst_bytes << "[" << i << "] = " << src_bytes << "[" << i
                 << "]\n";
        }
      } else if (conditional_lanes == 1) {
        PrintIndent();
        stream << result_tensor << "[0] = " << data << "\n";
      } else {
        PrintIndent();
        stream << result_tensor << ".store(tl.as_tensor_ssa(" << data << "))\n";
      }
      EndScope(scope);

      value_str = result_tensor;
    } else if (is_subbyte_vector_conditional) {
      std::string cond = PrintExpr_(cond_expr);
      std::string true_data = PrintExpr_(true_expr);
      std::string false_data = PrintExpr_(false_expr);
      auto is_zero_tensor_string = [](const std::string &s) {
        return s.find("tl.make_filled_tensor") != std::string::npos &&
               s.find(", 0, dtype=") != std::string::npos;
      };
      const bool true_is_zero =
          true_is_zero_ast || is_zero_tensor_string(true_data);
      const bool false_is_zero =
          false_is_zero_ast || is_zero_tensor_string(false_data);
      ICHECK(true_is_zero || false_is_zero)
          << "Sub-byte conditional vectors currently require one zero branch";
      std::string data = true_is_zero ? false_data : true_data;
      const PrimType elem_dtype = conditional_dtype.WithLanes(1);
      const int packed_bits = conditional_dtype.lanes() * elem_dtype.bits();
      ICHECK_EQ(packed_bits % 8, 0)
          << "Sub-byte conditional vectors must cover whole bytes";
      const int packed_bytes = packed_bits / 8;

      std::string result_tensor = name_supply_->FreshName("_where_tensor");
      std::string data_tensor = name_supply_->FreshName("_where_data");
      std::string data_bytes = name_supply_->FreshName("_where_src");
      std::string result_bytes = name_supply_->FreshName("_where_dst");
      PrintIndent();
      stream << result_tensor << " = tl.make_rmem_tensor(("
             << conditional_dtype.lanes() << ",), " << DTypeToString(elem_dtype)
             << ")\n";
      PrintIndent();
      stream << data_tensor << " = tl.as_rmem_tensor(" << data << ", ("
             << conditional_dtype.lanes() << ",), " << DTypeToString(elem_dtype)
             << ")\n";
      PrintIndent();
      stream << data_bytes << " = cute.recast_tensor(" << data_tensor
             << ", cutlass.Uint8)\n";
      PrintIndent();
      stream << result_bytes << " = cute.recast_tensor(" << result_tensor
             << ", cutlass.Uint8)\n";
      for (int i = 0; i < packed_bytes; ++i) {
        PrintIndent();
        stream << result_bytes << "[" << i << "] = ";
        if (true_is_zero) {
          stream << "(cutlass.Uint8(0) if " << cond << " else " << data_bytes
                 << "[" << i << "])\n";
        } else {
          stream << "(" << data_bytes << "[" << i << "] if " << cond
                 << " else cutlass.Uint8(0))\n";
        }
      }
      value_str = result_tensor;
    } else {
      // Helper: wrap a sub-expression as TensorSSA via make_rmem_tensor.
      auto as_tsa = [this](const PrimExpr &e, int n,
                           PrimType elem_dt) -> std::string {
        std::string s = PrintExpr_(e);
        if (n == 0)
          n = 1;
        if (EndsWithLoad(s))
          return s;
        std::string var = name_supply_->FreshName("_tsa");
        PrintIndent();
        if (elem_dt.MatchesCode(kDLBool)) {
          stream << var << " = tl.make_rmem_tensor((" << n
                 << ",), cutlass.Boolean)\n";
        } else {
          stream << var << " = tl.make_rmem_tensor((" << n << ",), "
                 << DTypeToString(elem_dt) << ")\n";
        }
        if (elem_dt.bits() < 8 && n > 1) {
          const int packed_bits = n * elem_dt.bits();
          ICHECK_EQ(packed_bits % 8, 0)
              << "Sub-byte conditional vectors must cover whole bytes";
          const int packed_bytes = packed_bits / 8;
          std::string src_bytes = name_supply_->FreshName("_where_src");
          std::string dst_bytes = name_supply_->FreshName("_where_dst");
          PrintIndent();
          stream << src_bytes << " = cute.recast_tensor(" << s
                 << ", cutlass.Uint8)\n";
          PrintIndent();
          stream << dst_bytes << " = cute.recast_tensor(" << var
                 << ", cutlass.Uint8)\n";
          for (int i = 0; i < packed_bytes; ++i) {
            PrintIndent();
            stream << dst_bytes << "[" << i << "] = " << src_bytes << "[" << i
                   << "]\n";
          }
        } else {
          for (int i = 0; i < n; ++i) {
            PrintIndent();
            if (n == 1) {
              stream << var << "[0] = " << s << "\n";
            } else {
              stream << var << "[" << i << "] = " << s << "[" << i << "]\n";
            }
          }
        }
        return var + ".load()";
      };
      PrimType true_ty = true_expr.ty().WithLanes(1);
      PrimType false_ty = false_expr.ty().WithLanes(1);
      std::string cond_tsa = as_tsa(cond_expr, 1, PrimType::Bool());
      std::string then_tsa = as_tsa(true_expr, lanes, true_ty);
      std::string else_tsa = as_tsa(false_expr, lanes, false_ty);
      if (true_ty != false_ty) {
        PrimType common =
            (true_ty.bits() >= false_ty.bits()) ? true_ty : false_ty;
        if (common.bits() < 32 && true_ty.MatchesCode(kDLFloat) &&
            false_ty.MatchesCode(kDLFloat))
          common = PrimType::Float(32);
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
               << DTypeToString(cast_to_dtype.WithLanes(1)) << ")\n";
        value_str = cast_result;
      } else {
        value_str = result;
      }
      if (conditional_dtype.WithLanes(1).bits() < 8 &&
          conditional_dtype.lanes() > 1) {
        std::string tensor_result = name_supply_->FreshName("_where_tensor");
        PrintIndent();
        stream << tensor_result << " = tl.make_rmem_tensor(("
               << conditional_dtype.lanes() << ",), "
               << DTypeToString(conditional_dtype.WithLanes(1)) << ")\n";
        PrintIndent();
        stream << tensor_result << ".store(" << value_str << ")\n";
        value_str = tensor_result;
      }
    }
  } else {
    value_str = PrintExpr_(op->value);
  }

  // CuTeDSL does not support implicit narrowing assignments (e.g. storing
  // Int32 to an Int16 tensor).  When the value's scalar width exceeds the
  // buffer element width, wrap with an explicit cast so that CuTeDSL's
  // Integer constructor emits arith.trunci (or the appropriate float
  // truncation).  This mirrors C/CUDA's implicit narrowing conversions.
  PrimType value_elem = value_dtype.WithLanes(1);
  PrimType buf_elem = element_dtype.WithLanes(1);
  if (value_elem.bits() > buf_elem.bits() && value_elem.lanes() == 1 &&
      buf_elem.lanes() == 1 &&
      (value_elem.MatchesCode(kDLInt) || value_elem.MatchesCode(kDLUInt)) ==
          (buf_elem.MatchesCode(kDLInt) || buf_elem.MatchesCode(kDLUInt))) {
    value_str = CastFromTo_(value_str, value_elem, buf_elem);
    value_dtype = buf_elem.WithLanes(value_dtype.lanes());
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
    value_str = SSAGetID(value_str, value_dtype,
                         tvm::SideEffect(op->value) == CallEffectKind::kPure);

    // Determine store target scope
    std::string vid = GetVarID(buffer_var.get());
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var.get())) {
      scope = alloc_storage_scope_.at(buffer_var.get());
    }

    if (scope == "local") {
      // For local rmem: use element-by-element assignment when the scalar
      // element can be dereferenced.  Sub-byte types such as FP4 cannot be
      // scalar-dereferenced by CuTeDSL, so move their packed byte storage
      // through a Uint8 recast instead.
      PrimExpr scalar_base(ffi::UnsafeInit{});
      if (value_lanes == element_dtype.lanes()) {
        scalar_base = index_expr * value_lanes;
      } else {
        tvm::PVar<PrimExpr> ramp_base;
        ICHECK(tvm::ramp(ramp_base, 1, value_lanes / element_dtype.lanes())
                   .Match(index_expr))
            << "Non-contiguous narrow-precision store not supported";
        scalar_base = ramp_base.Eval() * element_dtype.lanes();
      }
      if (value_dtype.bits() < 8) {
        const int packed_bits = value_lanes * value_dtype.bits();
        ICHECK_EQ(packed_bits % 8, 0)
            << "Sub-byte local vector stores must cover whole bytes";
        const int packed_bytes = packed_bits / 8;
        const int byte_lanes = 8 / value_dtype.bits();
        std::string src_bytes = name_supply_->FreshName("_pstore_src");
        std::string dst_view = name_supply_->FreshName("_pstore_view");
        std::string dst_bytes = name_supply_->FreshName("_pstore_dst");
        PrintIndent();
        stream << src_bytes << " = cute.recast_tensor(" << value_str
               << ", cutlass.Uint8)\n";
        PrintIndent();
        stream << dst_view << " = tl.make_tensor_at_offset(" << vid
               << ".iterator, " << PrintExpr_(scalar_base) << ", ("
               << value_lanes << ",), div_by=" << byte_lanes << ")\n";
        PrintIndent();
        stream << dst_bytes << " = cute.recast_tensor(" << dst_view
               << ", cutlass.Uint8)\n";
        for (int i = 0; i < packed_bytes; ++i) {
          PrintIndent();
          stream << dst_bytes << "[" << i << "] = " << src_bytes << "[" << i
                 << "]\n";
        }
      } else {
        for (int i = 0; i < value_lanes; ++i) {
          PrintIndent();
          stream << vid << "[" << PrintExpr_(scalar_base) << " + " << i
                 << "] = " << value_str << "[" << i << "]\n";
        }
      }
    } else {
      // For global/shared: use scalar element stores (works at any alignment).
      // CuTeDSL supports scalar FP8 stores via rmem_tensor -> global_tensor[i].
      PrimExpr scalar_base(ffi::UnsafeInit{});
      if (value_lanes == element_dtype.lanes()) {
        scalar_base = index_expr * value_lanes;
      } else {
        tvm::PVar<PrimExpr> ramp_base;
        ICHECK(tvm::ramp(ramp_base, 1, value_lanes / element_dtype.lanes())
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
        PrintType(value_dtype.WithLanes(1), ptr_os);
        ptr_os << ")";
        ptr_str = ptr_os.str();
      }

      // Create a tensor view and store each element individually.  For
      // sub-byte types, use a packed Uint8 view because CuTeDSL rejects scalar
      // dereference of FP4/FP6 storage.
      std::string view_var = name_supply_->FreshName("_sview");
      PrintIndent();
      if (value_dtype.bits() < 8) {
        stream << view_var << " = tl.make_tensor_at_offset(" << ptr_str << ", "
               << PrintExpr_(scalar_base) << ", (" << value_lanes
               << ",), div_by=" << (8 / value_dtype.bits()) << ")\n";
      } else {
        stream << view_var << " = tl.make_tensor(" << ptr_str << " + "
               << PrintExpr_(scalar_base) << ", (" << value_lanes << ",))\n";
      }
      if (value_dtype.bits() < 8) {
        const int packed_bits = value_lanes * value_dtype.bits();
        ICHECK_EQ(packed_bits % 8, 0)
            << "Sub-byte vector stores must cover whole bytes";
        const int packed_bytes = packed_bits / 8;
        const int byte_lanes = 8 / value_dtype.bits();
        std::string src_bytes = name_supply_->FreshName("_pstore_src");
        std::string dst_bytes = name_supply_->FreshName("_pstore_dst");
        PrintIndent();
        stream << src_bytes << " = cute.recast_tensor(" << value_str
               << ", cutlass.Uint8)\n";
        PrintIndent();
        stream << dst_bytes << " = cute.recast_tensor(" << view_var
               << ", cutlass.Uint8)\n";
        for (int i = 0; i < packed_bytes; ++i) {
          PrintIndent();
          stream << dst_bytes << "[" << i << "] = " << src_bytes << "[" << i
                 << "]\n";
        }
      } else {
        for (int i = 0; i < value_lanes; ++i) {
          PrintIndent();
          stream << view_var << "[" << i << "] = " << value_str << "[" << i
                 << "]\n";
        }
      }
    }
  } else if (value_lanes == element_dtype.lanes()) {
    std::string ref =
        GetBufferRef_(value_dtype, op->dest.as<VarObj>(), index_expr);

    if (ref.back() != ')') {
      PrintIndent();
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
      if (!value_is_conditional && value_lanes > 1 &&
          !op->value.as<TensorLoadObj>() &&
          (store_rhs.size() < 7 ||
           store_rhs.compare(store_rhs.size() - 7, 7, ".load()") != 0)) {
        std::string store_tensor = name_supply_->FreshName("_store_tensor");
        PrintIndent();
        stream << store_tensor << " = tl.make_rmem_tensor((" << value_lanes
               << ",), " << DTypeToString(value_dtype.WithLanes(1)) << ")\n";
        for (int i = 0; i < value_lanes; ++i) {
          PrintIndent();
          stream << store_tensor << "[" << i << "] = ";
          PrintVecElemLoad_(store_rhs, value_dtype, i, stream);
          stream << "\n";
        }
        store_rhs = store_tensor + ".load()";
      } else if (!value_is_conditional && value_lanes == 1 &&
                 !op->value.as<TensorLoadObj>()) {
        if (store_rhs.size() < 7 ||
            store_rhs.compare(store_rhs.size() - 7, 7, ".load()") != 0) {
          store_rhs = "tl.make_filled_tensor((1,), " + store_rhs + ").load()";
        }
      }
      PrintIndent();
      stream << "tl.store_tensor(" << ref << ", "
             << RemoveOutermostParentheses(store_rhs) << ")\n";
    }
  } else {
    bool is_contiguous = false;
    tvm::PVar<PrimExpr> base;
    if (tvm::ramp(base, 1, value_lanes / element_dtype.lanes())
            .Match(index_expr)) {
      is_contiguous = true;
    }

    if (is_contiguous) {
      PrintVecStore_(op->dest.as<VarObj>(), value_dtype, base.Eval(),
                     value_str);
    } else {
      ICHECK(element_dtype.IsScalar())
          << "buffer element type for non-contiguous store must be scalar "
             "currently";

      // store elements separately
      value_str = SSAGetID(value_str, element_dtype,
                           tvm::SideEffect(op->value) == CallEffectKind::kPure);
      for (int i = 0; i < value_lanes; ++i) {
        const prim::RampObj *ramp = index_expr.as<prim::RampObj>();
        ICHECK(ramp);
        auto idx_expr =
            sym::Analyzer()->Simplify(ramp->base + ramp->stride * i);

        PrintIndent();
        stream << GetBufferRef_(element_dtype, op->dest.as<VarObj>(), idx_expr)
               << " = ";
        PrintVecElemLoad_(value_str, value_dtype, i, stream);
        stream << "\n";
      }
    }
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const BindObj *op) {
  if (auto pointer_type = op->var->ty.as<PtrType>()) {
    const std::string value = GetVarPtr_(op->value);
    if (auto dtype = pointer_type.value()->element_type.as<PrimType>())
      RegisterHandleType_(op->var.get(), dtype.value());
    alloc_storage_scope_[op->var.get()] = pointer_type.value()->scope.name();
    raw_pointer_vars_.insert(op->var.get());
    PrintIndent();
    stream << AllocVarID(op->var.get()) << " = " << value << "\n";
    return;
  }
  if (IsZeroTensorLike(op->value, &zero_like_vars_))
    zero_like_vars_.insert(op->var.get());
  CodeGenTileLangPY::Dispatch_(op);
}

void CodeGenTileLangCuTeDSL::EmitTensorAllocation_(const BindObj *op,
                                                   const CallObj *allocation) {
  TensorVar tensor = op->var.as_or_throw<TensorVar>();
  const auto annotations = CodegenCallAnnotations(allocation);
  PrimType alloc_dtype = tensor.dtype();
  std::string vid = AllocVarID(tensor.get());
  PrintIndent();
  std::string scope = tensor.scope().name();
  alloc_storage_scope_[tensor.get()] = scope;

  if (scope == "local.descriptor.wgmma") {
    stream << vid << " = tl.GmmaDescriptor()\n";
  } else if (scope == "local.descriptor.tcgen05_smem") {
    stream << vid << " = tl.Tcgen05SmemDescriptor()\n";
  } else if (scope == "local.descriptor.tcgen05_instr") {
    stream << vid << " = 0\n";
  } else if (scope == "shared.dyn") {
    stream << vid << " = tl.make_tensor(tl.get_dyn_smem(";
    PrintType(alloc_dtype, stream);
    // there is no bound check for Tensor access, so just set shape to 1
    stream << ", alignment=1024), (1,))\n";
  } else {
    std::optional<int64_t> opt_size = std::optional<int64_t>(
        CodegenAllocationElements(op->var.as_or_throw<TensorVar>()));
    ICHECK(opt_size.has_value() && opt_size.value() > 0)
        << "Can only handle constant size stack allocation for now, but get "
        << (opt_size.has_value() ? opt_size.value() : 0) << " for "
        << tensor->name;
    size_t constant_size = static_cast<size_t>(opt_size.value());
    const bool scalarized_vector_alloc =
        (scope == "shared" || scope == "local") &&
        alloc_dtype.IsFixedLengthVector();
    PrimType storage_dtype =
        scalarized_vector_alloc ? alloc_dtype.WithLanes(1) : alloc_dtype;
    size_t storage_size = constant_size;
    if (scalarized_vector_alloc) {
      storage_size *= static_cast<size_t>(alloc_dtype.lanes());
    }
    if ((scope == "shared" || scope == "local") && storage_dtype.bits() < 8) {
      storage_size += static_cast<size_t>(32 / storage_dtype.bits() - 1);
    }

    if (scope == "shared") {
      stream << vid << " = tl.make_tensor(tl.alloc_smem(";
      PrintType(storage_dtype, stream);
      stream << ", " << storage_size << "), (" << storage_size << ",))\n";
    } else if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
      stream << vid << " = tl.alloc_smem(cutlass.Uint64, size_in_elems="
             << constant_size << ")\n";
    } else if (scope == "local") {
      stream << vid << " = tl.make_rmem_tensor((" << storage_size << "),";
      PrintType(storage_dtype, stream);
      stream << ")\n";
    } else if (scope == "local.var") {
      PrimExpr init = prim::MakeConst(alloc_dtype, 0);
      auto init_it = annotations.find("tl.local_var_init");
      if (init_it != annotations.end()) {
        PrimExpr user_init = (*init_it).second.cast<PrimExpr>();
        if (!user_init.ty().IsVoid() && user_init.ty() != alloc_dtype) {
          user_init = prim::Cast(alloc_dtype, user_init);
        }
        init = user_init;
      }
      stream << vid << " = " << PrintExpr_(init) << "\n";
    } else {
      ICHECK(false) << "Unsupported scope: " << scope;
    }
  }

  PrimType registered_dtype = ((scope == "shared" || scope == "local") &&
                               alloc_dtype.IsFixedLengthVector())
                                  ? alloc_dtype.WithLanes(1)
                                  : alloc_dtype;
  RegisterHandleType_(tensor.get(), registered_dtype);
}

void CodeGenTileLangCuTeDSL::EmitTensorDeclaration_(const BindObj *op,
                                                    const CallObj *call) {
  TensorVar tensor = op->var.as_or_throw<TensorVar>();
  TVM_FFI_CHECK_EQ(call->args.size(), 1, ValueError);
  const std::string pointer = GetVarPtr_(call->args[0]);
  const auto dtype = tensor.dtype();
  const auto scope = tensor.scope().name();
  RegisterHandleType_(tensor.get(), dtype);
  alloc_storage_scope_[tensor.get()] = scope;
  const std::string name = AllocVarID(tensor.get());
  PrintIndent();
  stream << name << " = tl.make_tensor(" << pointer << ", (";
  for (const auto &dim : tensor.shape())
    stream << PrintExpr_(dim) << ", ";
  stream << "))\n";
}

void CodeGenTileLangCuTeDSL::Dispatch_(const RegionStmtObj *op) {
  if (op->op.same_as(tirx::launch_thread_op())) {
    TVM_FFI_CHECK(op->args.size() == 2 && op->body_params.size() == 1 &&
                      op->result_vars.empty(),
                  ValueError);
    const auto &var = op->body_params[0].as_or_throw<PrimVar>();
    BindThreadIndex_(var, op->args[0].as_or_throw<StringImm>()->value);
    PrintSuite_(op->body);
    var_idmap_.erase(var.get());
    return;
  }
  if (op->op->name != "tl.attr")
    return CodeGenTileLangPY::Dispatch_(op);
  const auto attr = DecodeTileLangAttribute(op);
  if (attr.key == "lexical_alloc_scope")
    return CodeGenTileLangPY::Dispatch_(op);
  if (attr.key == "threadblock_swizzle_pattern") {
    std::string func_name;
    int64_t panel_size;
    if (auto tuple = attr.value.as<tvm::Tuple>()) {
      TVM_FFI_CHECK_EQ(tuple.value()->fields.size(), 2, ValueError);
      func_name = tuple.value()->fields[0].as_or_throw<StringImm>()->value;
      panel_size = CodegenInt(tuple.value()->fields[1]);
    } else {
      const auto values = attr.value.cast<Array<Any>>();
      TVM_FFI_CHECK_EQ(values.size(), 2, ValueError);
      func_name = CodegenAnnotationString(values[0]);
      panel_size = values[1].as<IntImmObj>()
                       ? CodegenInt(values[1].as<IntImmObj>())
                       : values[1].cast<int64_t>();
    }
    TVM_FFI_CHECK(panel_size > 0 && (func_name == "rasterization2DRow" ||
                                     func_name == "rasterization2DColumn"),
                  ValueError);
    PrintIndent();
    if (cluster_dims_) {
      auto [x, y, z] = *cluster_dims_;
      TVM_FFI_CHECK(y == 1 && z == 1 && panel_size % x == 0, ValueError);
      stream << "blockIdx = tl." << func_name << "WithCluster("
             << panel_size / x << ", " << x << ")\n";
    } else {
      stream << "blockIdx = tl." << func_name << "(" << panel_size << ")\n";
    }
  } else if (attr.key == "pragma_unroll_factor") {
    const auto *variable = attr.node.as<VarObj>();
    TVM_FFI_CHECK(variable, TypeError);
    unroll_factor_.insert_or_assign(variable, attr.value.cast<IntImm>());
  }
  PrintSuite_(op->body);
}

void CodeGenTileLangCuTeDSL::EmitLoopBreak_() {
  PrintIndent();
  if (in_break_loop_) {
    stream << "_loop_break_" << current_break_id_ << " = 1\n";
    break_emitted_in_seq_ = true;
  } else {
    stream << "break\n";
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const BreakObj *) {
  // The native frontend constructs Break, rather than tl.loop_break. CuTe
  // rejects a literal break in a dynamic loop; retain the existing per-loop
  // guard and current-Seq suffix suppression used by the intrinsic form.
  EmitLoopBreak_();
}

void CodeGenTileLangCuTeDSL::Dispatch_(const ForObj *op) {
  TVM_FFI_CHECK(op->HasTrivialStep(), ValueError)
      << "CuTeDSL loop lowering currently requires unit step";
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
    std::string vid = AllocVarID(op->var.get());
    stream << "for " << vid << " in range(";
    if (IsZero(op->begin)) {
      PrintExpr_((op->end - op->begin), stream);
    } else {
      PrintExpr_(op->begin, stream);
      stream << ", ";
      PrimExpr upper_bound = sym::Analyzer()->Simplify(op->end);
      PrintExpr_(upper_bound, stream);
    }
    stream << "):\n";

    // Emit the body wrapped in guard check
    int for_scope = BeginScope();
    PrintIndent();
    stream << "if _loop_break_" << break_id << " == 0:\n";
    int guard_scope = BeginScope();
    PrintSuite_(op->body);
    EndScope(guard_scope);
    EndScope(for_scope);

    // Restore state
    in_break_loop_ = old_in_break;
    current_break_id_ = old_break_id;
    return;
  }

  if (op->kind != ForKind::kUnrolled) {
    CodeGenTileLangPY::Dispatch_(op);
    return;
  }

  auto start_expr = sym::Analyzer()->Simplify(op->begin);
  auto stop_expr = sym::Analyzer()->Simplify(op->end);
  std::string unroll_factor;
  // Native For retains loop-local hints instead of creating an AttrStmt.
  // Preserve the attribute carrier as a fallback for explicitly lowered IR.
  if (auto factor = op->annotations.Get("pragma_unroll_factor")) {
    const auto *literal = factor->as<IntImmObj>();
    unroll_factor =
        std::to_string(literal ? CodegenInt(literal) : factor->cast<int64_t>());
  } else if (auto it = unroll_factor_.find(op->var.get());
             it != unroll_factor_.end()) {
    unroll_factor = PrintExpr_(it->second);
  }
  const auto extent = sym::Analyzer()->Simplify(op->end - op->begin);
  const auto *constant_extent = extent.as<IntImmObj>();
  bool use_range_constexpr =
      unroll_factor.empty() && constant_extent &&
      CodegenInt(constant_extent) <= LOOP_UNROLL_THRESHOLD;
  PrintIndent();
  std::string vid = AllocVarID(op->var.get());
  stream << "for " << vid << " in cutlass.range";
  if (use_range_constexpr) {
    stream << "_constexpr";
  }
  stream << "(";
  if (!IsZero(start_expr)) {
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
  PrintSuite_(op->body);
  EndScope(for_scope);
}

void CodeGenTileLangCuTeDSL::Dispatch_(const SeqStmtObj *op) {
  // CuTeDSL does not allow dynamic early returns. Convert
  // `if cond: thread_return(); rest` into `if not cond: rest`.
  std::vector<int> thread_return_guard_scopes;

  // In a break loop, after visiting a statement that contains loop_break(),
  // wrap the remaining statements in an `if _loop_break_N == 0:` guard so they
  // don't execute in the same iteration after the break.
  int guard_scope = -1;
  for (size_t i = 0; i < op->seq.size(); ++i) {
    if (auto guarded_return = GetConditionalThreadReturn(op->seq[i])) {
      if (guarded_return->pre_return) {
        PrintIndent();
        stream << "if "
               << RemoveOutermostParentheses(
                      PrintExpr_(guarded_return->condition))
               << ":\n";
        int pre_return_scope = BeginScope();
        PrintStmt_(guarded_return->pre_return.value());
        EndScope(pre_return_scope);
      }
      if (i + 1 < op->seq.size()) {
        PrintIndent();
        stream << "if not ("
               << RemoveOutermostParentheses(
                      PrintExpr_(guarded_return->condition))
               << "):\n";
        thread_return_guard_scopes.push_back(BeginScope());
      }
      continue;
    }
    if (IsThreadReturnEvaluate(op->seq[i])) {
      break;
    }

    break_emitted_in_seq_ = false;
    PrintStmt_(op->seq[i]);
    if (in_break_loop_ && break_emitted_in_seq_ && guard_scope < 0 &&
        i + 1 < op->seq.size()) {
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
  for (auto it = thread_return_guard_scopes.rbegin();
       it != thread_return_guard_scopes.rend(); ++it) {
    EndScope(*it);
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const IfObj *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(cond) << ":\n";
  int then_scope = BeginScope();
  if (const CallObj *call = op->condition.as<CallObj>();
      call && IsCodegenOp(call, "tl.tl_shuffle_elect")) {
    PrintIndent();
    stream << "with cute.arch.elect_one():\n";
    int with_scope = BeginScope();
    PrintSuite_(op->then_case);
    EndScope(with_scope);
  } else {
    PrintSuite_(op->then_case);
  }
  EndScope(then_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintSuite_(op->else_case.value());
    EndScope(else_scope);
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const EvaluateObj *op) {
  if (op->value.as<IntImmObj>())
    return;
  const CallObj *call = op->value.as<CallObj>();
  if (call && (IsCodegenOp(call, "tl.device_assert"))) {
    std::string cond = RemoveOutermostParentheses(PrintExpr_(call->args[0]));
    PrintIndent();
    stream << "tl.runtime_assert(" << cond << ")\n";
  } else if (call && IsCodegenOp(call, "tl.device_assert_with_msg")) {
    std::string cond = RemoveOutermostParentheses(PrintExpr_(call->args[0]));
    std::string msg_expr = PrintExpr_(call->args[1]);
    PrintIndent();
    stream << "tl.runtime_assert(" << cond << ", " << msg_expr << ")\n";
  } else if (call && IsCodegenOp(call, "tl.sync_warp")) {
    PrintIndent();
    stream << "tl.sync_warp(";
    if (!call->args.empty()) {
      stream << PrintExpr_(call->args[0]);
    }
    stream << ")\n";
  } else if (call && (IsCodegenOp(call, "tl.storage_sync") ||
                      IsCodegenOp(call, "tirx.gpu_storage_sync"))) {
    PrintStorageSync_(call);
  } else {
    CodeGenTileLangPY::Dispatch_(op);
  }
}

void CodeGenTileLangCuTeDSL::PrintVecElemLoad_(const std::string &vec,
                                               PrimType t, int i,
                                               std::ostream &os) { // NOLINT(*)
  if (t.IsScalar()) {
    os << vec;
    return;
  }
  os << vec << "[" << i << "]";
}

void CodeGenTileLangCuTeDSL::PrintVecElemStore_(const std::string &vec,
                                                PrimType t, int i,
                                                const std::string &value) {
  PrintIndent();
  stream << vec << "[" << i << "] = " << value << "\n";
}

void CodeGenTileLangCuTeDSL::PrintVecStore_(const VarObj *buffer, PrimType t,
                                            PrimExpr base,
                                            const std::string &value) {
  ICHECK(!t.IsScalar()) << "PrintVecStore_() should not be used for scalar";

  std::string ref = GetBufferRef_(t, buffer, base);
  std::string store_rhs = value;
  if (t.WithLanes(1).bits() < 8) {
    if (store_rhs.size() < 7 ||
        store_rhs.compare(store_rhs.size() - 7, 7, ".load()") != 0) {
      store_rhs += ".load()";
    }
    PrintIndent();
    stream << "tl.store_tensor(" << ref << ", " << store_rhs << ")\n";
    return;
  }

  if (store_rhs.size() < 7 ||
      store_rhs.compare(store_rhs.size() - 7, 7, ".load()") != 0) {
    std::string store_tensor = name_supply_->FreshName("_store_tensor");
    PrintIndent();
    stream << store_tensor << " = tl.make_rmem_tensor((" << t.lanes() << ",), "
           << DTypeToString(t.WithLanes(1)) << ")\n";
    for (int i = 0; i < t.lanes(); ++i) {
      PrintIndent();
      stream << store_tensor << "[" << i << "] = ";
      PrintVecElemLoad_(store_rhs, t, i, stream);
      stream << "\n";
    }
    store_rhs = store_tensor + ".load()";
  }
  PrintIndent();
  stream << "tl.store_tensor(" << ref << ", " << store_rhs << ")\n";
}

void CodeGenTileLangCuTeDSL::PrintVecBinaryOp_(const std::string &opstr,
                                               PrimType dtype, PrimExpr lhs,
                                               PrimExpr rhs,
                                               std::ostream &os) { // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  PrintIndent();
  stream << sret << " = tl.make_rmem_tensor((" << dtype.lanes() << ",), ";
  PrintType(dtype.WithLanes(1), stream);
  stream << ")\n";

  auto print_operand = [this](const Expr &expr) {
    std::string value = PrintExpr_(expr);
    return ContainsTensorLoad(expr)
               ? value
               : SSAGetID(value, expr->ty,
                          tvm::SideEffect(expr) == CallEffectKind::kPure);
  };
  std::string vlhs = print_operand(lhs);
  std::string vrhs = print_operand(rhs);

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
        PrintVecElemLoad_(vlhs, lhs.ty(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad_(vrhs, rhs.ty(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad_(vlhs, lhs.ty(), i, value_temp);
        value_temp << opstr;
        PrintVecElemLoad_(vrhs, rhs.ty(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore_(sret, dtype, i, value_temp.str());
    }
  }
  os << sret << ".load()";
}

void CodeGenTileLangCuTeDSL::PrintBinaryExpr_(const std::string &opstr,
                                              PrimType dtype, PrimExpr lhs,
                                              PrimExpr rhs,
                                              std::ostream &os) { // NOLINT(*)
  if (dtype.IsScalar()) {
    CodeGenTileLangPY::PrintBinaryExpr_(opstr, dtype, lhs, rhs, os);
  } else {
    PrintVecBinaryOp_(opstr, dtype, lhs, rhs, os);
  }
}

void CodeGenTileLangCuTeDSL::PrintBinaryIntrinsic_(
    const CallObj *op, const char *opstr,
    std::ostream &os) { // NOLINT(*)
  if (CodegenDType(op).IsScalar()) {
    CodeGenTileLangPY::PrintBinaryIntrinsic_(op, opstr, os);
  } else {
    PrintVecBinaryOp_(opstr, CodegenDType(op),
                      op->args[0].as_or_throw<PrimExpr>(),
                      op->args[1].as_or_throw<PrimExpr>(), os);
  }
}

void CodeGenTileLangCuTeDSL::PrintCallExtern_(Type ret_type,
                                              String global_symbol,
                                              const Array<Expr> &args,
                                              bool skip_first_arg,
                                              std::ostream &os) { // NOLINT(*)
  PrimType ret_dtype = ret_type.as<PrimType>().value_or(PrimType::Void());

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
    if (ret_dtype.IsFixedLengthVector()) {
      std::string val =
          SSAGetID(sarg, CodegenDType(args[i]),
                   tvm::SideEffect(args[i]) == CallEffectKind::kPure);
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
    if (const TensorLoadObj *load = args[arg_begin].as<TensorLoadObj>()) {
      sargs[0] = GetBufferPtr_(
          load->source.as<VarObj>(),
          LinearizeBufferIndices_(load->source.as<VarObj>(), load->indices));
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
  // some optional template arguments might be omitted, so add names explicitly
  // for remain arguments
  if (global_symbol_str == "tl.gemm_ss" || global_symbol_str == "tl.gemm_rs" ||
      global_symbol_str == "tl.gemm_sr" || global_symbol_str == "tl.gemm_rr") {
    ICHECK(sargs.size() >= 3);
    sargs[sargs.size() - 3] = "A_ptr=" + sargs[sargs.size() - 3];
    sargs[sargs.size() - 2] = "B_ptr=" + sargs[sargs.size() - 2];
    sargs[sargs.size() - 1] = "C_ptr=" + sargs[sargs.size() - 1];
  }

  if (ret_dtype.IsFixedLengthVector()) {
    // maybe simplify this if TensorSSA supports this OP
    std::string sret = name_supply_->FreshName("_");
    PrintIndent();
    stream << sret << " = tl.make_rmem_tensor((" << ret_dtype.lanes() << ",), ";
    PrintType(ret_dtype.WithLanes(1), stream);
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
                  ->ty.as_or_throw<PrimType>(),
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

PrimExpr CodeGenTileLangCuTeDSL::LinearizeBufferIndices_(
    const VarObj *buffer, const Array<PrimExpr> &indices) {
  const TensorVar tensor = CodegenTensor(buffer);
  TVM_FFI_CHECK(!indices.empty(), ValueError);
  if (indices.size() == 1 &&
      (tensor.shape().size() != 1 || !indices[0].ty().IsScalar()))
    return indices[0];
  TVM_FFI_CHECK_EQ(indices.size(), tensor.shape().size(), ValueError);
  auto layout = tensor.ty()->layout;
  TVM_FFI_CHECK(
      !layout.has_value() ||
          ffi::StructuralEqual()(
              layout.value(), tirx::TileLayout::DefaultLayout(tensor.shape())),
      ValueError)
      << "CuTeDSL requires layout lowering before linearization";
  PrimType index_type = indices[0].ty();
  PrimExpr offset = prim::MakeConst(index_type, 0);
  PrimExpr stride = prim::MakeConst(index_type, 1);
  for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
    offset = offset + prim::cast(indices[i], index_type) * stride;
    stride = stride * prim::cast(tensor.shape()[i], index_type);
  }
  return sym::Analyzer()->Simplify(offset);
}

std::string CodeGenTileLangCuTeDSL::GetBufferPtr_(const VarObj *buffer,
                                                  PrimExpr index) {
  const VarObj *buffer_var = buffer;
  const std::string vid = GetVarID(buffer_var);

  PrimType buffer_element_dtype = CodegenTensor(buffer).dtype();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty())
    scope = CodegenStorageScope(GetRef<Var>(buffer));
  const bool scalarized_local_vector_buffer =
      (scope == "local" || scope == "shared") &&
      buffer_element_dtype.IsFixedLengthVector();
  // CuTeDSL only supports i1 (Boolean) in rmem; use Uint8 for gmem pointers.
  PrimType effective_dtype = scalarized_local_vector_buffer
                                 ? buffer_element_dtype.WithLanes(1)
                                 : buffer_element_dtype;
  if (buffer_element_dtype.MatchesCode(kDLBool)) {
    if (scope != "local" && scope != "local.var") {
      effective_dtype = PrimType::UInt(8);
    }
  }
  // shared.barrier and shared.cluster_barrier are allocated via tl.alloc_smem()
  // which returns _Pointer (not _Tensor), so it doesn't have .iterator — use
  // vid directly.

  std::string ptr_str;
  if (raw_pointer_vars_.count(buffer_var)) {
    bool is_handle_type_match = HandleTypeMatch_(buffer_var, effective_dtype);
    if (is_handle_type_match) {
      ptr_str = vid;
    } else {
      ptr_str = "tl.recast_ptr(" + vid +
                ", dtype=" + DTypeToString(effective_dtype) + ")";
    }
  } else if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
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
  if (scalarized_local_vector_buffer) {
    index_str = "(" + index_str + " * " +
                std::to_string(buffer_element_dtype.lanes()) + ")";
  }
  return "(" + ptr_str + " + " + index_str + ")";
}

std::string
CodeGenTileLangCuTeDSL::GetSubByteBufferPtrAsByte_(const VarObj *buffer,
                                                   PrimExpr index) {
  PrimType buffer_element_dtype = CodegenTensor(buffer).dtype();
  int elems_per_byte = 1;
  if (tl::IsFloat4E2M1Unpacked(buffer_element_dtype)) {
    // This dtype is a shared-memory storage/layout tag for tcgen05 blockscale
    // operands.  CUDA codegen treats its buffer index as a byte offset after
    // casting to uint8_t*, not as packed FP4 nibbles.
    elems_per_byte = 1;
  } else {
    ICHECK_LT(buffer_element_dtype.bits(), 8)
        << "GetSubByteBufferPtrAsByte_ expects a sub-byte buffer, got "
        << buffer_element_dtype;
    ICHECK_EQ(8 % buffer_element_dtype.bits(), 0)
        << "Unsupported sub-byte width: " << buffer_element_dtype;
    elems_per_byte = 8 / buffer_element_dtype.bits();
  }

  const VarObj *buffer_var = buffer;
  const std::string vid = GetVarID(buffer_var);

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = CodegenStorageScope(GetRef<Var>(buffer));
  }

  std::string ptr_str;
  PrimType byte_dtype = PrimType::UInt(8);
  if (raw_pointer_vars_.count(buffer_var)) {
    bool is_handle_type_match = HandleTypeMatch_(buffer_var, byte_dtype);
    if (is_handle_type_match) {
      ptr_str = vid;
    } else {
      ptr_str =
          "tl.recast_ptr(" + vid + ", dtype=" + DTypeToString(byte_dtype) + ")";
    }
  } else if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
    ptr_str = vid;
  } else {
    bool is_handle_type_match = HandleTypeMatch_(buffer_var, byte_dtype);
    if (is_handle_type_match) {
      ptr_str = vid + ".iterator";
    } else {
      ptr_str = "tl.recast_ptr(" + vid +
                ".iterator, dtype=" + DTypeToString(byte_dtype) + ")";
    }
  }

  std::string index_str = PrintExpr_(index);
  if (elems_per_byte != 1) {
    index_str = "(" + index_str + " // " + std::to_string(elems_per_byte) + ")";
  }
  return "(" + ptr_str + " + " + index_str + ")";
}

std::string CodeGenTileLangCuTeDSL::PrintBytePointerExpr_(const Expr &expr) {
  const CallObj *call = expr.as<CallObj>();
  if (call && IsCodegenOp(call, "tirx.address_of")) {
    ICHECK_EQ(call->args.size(), 1U);
    const TensorLoadObj *load = call->args[0].as<TensorLoadObj>();
    if (load && tl::IsFloat4E2M1Unpacked(
                    load->source.as_or_throw<TensorVar>().dtype())) {
      ICHECK_EQ(load->indices.size(), 1U)
          << "CodeGenTileLangCuTeDSL only supports flat memory";
      return GetSubByteBufferPtrAsByte_(load->source.as<VarObj>(),
                                        load->indices[0]);
    }
  }
  return PrintExpr_(expr);
}

std::string CodeGenTileLangCuTeDSL::GetVarPtr_(const Expr &expr) {
  if (const VarObj *var = expr.as<VarObj>()) {
    if (raw_pointer_vars_.count(var)) {
      return GetVarID(var);
    }
    // Descriptor allocations are Python descriptor objects, not CuTe tensors.
    // Preserve the object at tensor_data_ptr/alias boundaries so the MMA
    // consumer uses its descriptor-offset operator (bytes for TCGEN05).
    if (const auto *tensor = var->ty.as<TensorTypeObj>()) {
      const std::string scope = tensor->scope.name();
      if (scope == "local.descriptor.wgmma" ||
          scope == "local.descriptor.tcgen05_smem") {
        return GetVarID(var);
      }
    }
    // Ordinary rmem/smem tensors expose their pointer through .iterator.
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
std::string CodeGenTileLangCuTeDSL::GetBufferRef_(PrimType t,
                                                  const VarObj *buffer,
                                                  PrimExpr index) {
  const VarObj *buffer_var = buffer;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = CodegenStorageScope(GetRef<Var>(buffer));
  }
  if (scope == "local.var" || scope.find("local.descriptor") == 0) {
    return vid;
  }

  PrimType buffer_element_dtype = CodegenTensor(buffer).dtype();
  const bool scalarized_local_vector_buffer =
      (scope == "local" || scope == "shared") &&
      buffer_element_dtype.IsFixedLengthVector();
  // CuTeDSL only supports i1 (Boolean) in rmem. For gmem/shared bool buffers,
  // use Uint8 instead (matches PyTorch's torch.bool memory layout).
  PrimType effective_dtype = scalarized_local_vector_buffer
                                 ? buffer_element_dtype.WithLanes(1)
                                 : buffer_element_dtype;
  if (buffer_element_dtype.MatchesCode(kDLBool) && scope != "local" &&
      scope != "local.var") {
    effective_dtype = PrimType::UInt(8);
  }
  bool is_handle_type_match = HandleTypeMatch_(buffer_var, effective_dtype);
  std::string ptr_str;
  if (raw_pointer_vars_.count(buffer_var)) {
    if (is_handle_type_match) {
      ptr_str = vid;
    } else {
      ptr_str = "tl.recast_ptr(" + vid +
                ", dtype=" + DTypeToString(effective_dtype) + ")";
    }
  } else if (is_handle_type_match) {
    ptr_str = vid + ".iterator";
  } else {
    ptr_str = "tl.recast_ptr(" + vid +
              ".iterator, dtype=" + DTypeToString(effective_dtype) + ")";
  }

  // CuTeDSL make_tensor_at_offset(ptr, offset, shape, div_by) expects a
  // single scalar offset. For a Ramp index, pass only the base to avoid
  // emitting a tuple (base+0, base+1, ...) that the runtime rejects.
  PrimExpr offset_expr = index;
  if (const prim::RampObj *ramp = index.as<prim::RampObj>()) {
    ICHECK(IsOne(ramp->stride))
        << "GetBufferRef_: non-unit Ramp stride not supported, got "
        << ramp->stride;
    offset_expr = ramp->base;
  } else {
    tvm::PVar<PrimExpr> ramp_base;
    int lanes = t.lanes() > 0 ? t.lanes() : 1;
    if (tvm::ramp(ramp_base, 1, lanes).Match(index)) {
      offset_expr = ramp_base.Eval();
    }
  }
  const std::string index_str = PrintExpr_(offset_expr);
  const std::string scalarized_index_str =
      scalarized_local_vector_buffer
          ? "(" + index_str + " * " +
                std::to_string(buffer_element_dtype.lanes()) + ")"
          : index_str;
  const int scalarized_div_by =
      scalarized_local_vector_buffer &&
              buffer_element_dtype.WithLanes(1).bits() < 8
          ? 8 / buffer_element_dtype.WithLanes(1).bits()
          : buffer_element_dtype.lanes();

  if (t == buffer_element_dtype) {
    if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
      // shared.barrier and shared.cluster_barrier are allocated via
      // tl.alloc_smem() which returns _Pointer. _Pointer does not support
      // subscript access [i], but supports pointer arithmetic (ptr + i). Use
      // pointer addition instead of subscript.
      return "(" + vid + " + " + index_str + ")";
    } else if (scalarized_local_vector_buffer) {
      std::ostringstream os;
      os << "tl.make_tensor_at_offset(" << vid << ".iterator, "
         << scalarized_index_str << ", (" << buffer_element_dtype.lanes()
         << ",), div_by=" << scalarized_div_by << ")";
      return os.str();
    } else if (is_handle_type_match && buffer_element_dtype.IsScalar() &&
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
         << ", (1,), div_by=" << scalarized_div_by << ")";
      // for vector data types, ".load()" (added by BufferLoadNode) is needed
      // instead of "[0]"
      if (buffer_element_dtype.IsScalar()) {
        os << "[0]";
      }
      return os.str();
    }
  } else if (scalarized_local_vector_buffer &&
             t == buffer_element_dtype.WithLanes(1)) {
    return vid + "[" + index_str + "]";
  } else {
    const int num = t.bits() * t.lanes();
    PrimType storage_element_dtype = scalarized_local_vector_buffer
                                         ? buffer_element_dtype.WithLanes(1)
                                         : buffer_element_dtype;
    const int den =
        storage_element_dtype.bits() * storage_element_dtype.lanes();
    ICHECK_EQ(num % den, 0) << "Cannot form view: bitwidth not divisible";
    int buffer_size = num / den;

    std::ostringstream os;
    os << "tl.make_tensor_at_offset(" << ptr_str << ", "
       << (scalarized_local_vector_buffer ? scalarized_index_str : index_str)
       << ", (" << buffer_size << ",), div_by=" << buffer_size << ")";
    return os.str();
  }
}

void CodeGenTileLangCuTeDSL::BindThreadIndex_(const PrimVar &var,
                                              const String &thread_tag) {
  TVM_FFI_CHECK(!var_idmap_.count(var.get()), ValueError);
  TVM_FFI_CHECK(thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
                    thread_tag == "threadIdx.z" || thread_tag == "blockIdx.x" ||
                    thread_tag == "blockIdx.y" || thread_tag == "blockIdx.z",
                ValueError);
  var_idmap_[var.get()] = CastFromTo_(thread_tag, PrimType::Int(32), var.ty());
}

void CodeGenTileLangCuTeDSL::PrintStorageSync_(const CallObj *op) {
  const auto args = CodegenOperands(op);
  const std::string &sync = args[0].as<StringImmObj>()->value;
  if (sync == "warp") {
    // do nothing
  } else if (sync == "shared" || sync == "shared.dyn") {
    PrintIndent();
    if (args.size() == 1) {
      stream << "tl.sync_threads()\n";
    } else if (args.size() == 2) {
      ICHECK(CodegenDType(args[1]).MatchesCode(kDLInt))
          << "storage_sync barrier_id must be integer type, got "
          << CodegenDType(args[1]);
      stream << "tl.sync_thread_partial(" << PrintExpr_(args[1]) << ")\n";
    } else if (args.size() == 3) {
      ICHECK(CodegenDType(args[1]).MatchesCode(kDLInt))
          << "storage_sync barrier_id must be integer type, got "
          << CodegenDType(args[1]);
      ICHECK(CodegenDType(args[2]).MatchesCode(kDLInt))
          << "storage_sync thread_count must be integer type, got "
          << CodegenDType(args[2]);
      stream << "tl.sync_thread_partial(" << PrintExpr_(args[1]) << ", "
             << PrintExpr_(args[2]) << ")\n";
    } else {
      TVM_FFI_THROW(InternalError)
          << "Invalid number of arguments for storage sync: " << args.size();
    }
  } else if (sync == "cluster") {
    PrintIndent();
    stream << "tl.cluster_sync()\n";
  } else if (sync == "global") {
    TVM_FFI_THROW(InternalError)
        << "PrintStorageSync_ for global is not supported for now";
  } else {
    TVM_FFI_THROW(InternalError) << "Unknown storage sync scope: " << sync;
  }
}

} // namespace codegen
} // namespace tvm
