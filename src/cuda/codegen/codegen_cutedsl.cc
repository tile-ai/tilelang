/*!
 * \file target/codegen_cutedsl.cc
 */

#include "codegen_cutedsl.h"
#include "backend/common/codegen/codegen_utils.h"
#include "cuda/codegen/ptx.h"
#include "op/dtype.h"
#include "support/check.h"
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op/memory.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cuda/op/builtin.h"
#include <tvm/ir/prim/pattern_match.h>

namespace tvm {
namespace codegen {

using namespace ffi;
namespace {

// Helper to check if a statement subtree contains loop break ops
// (either tl::loop_break() or builtin::break_loop())
class LoopBreakDetector : public tirx::StmtExprVisitor {
public:
  bool found = false;
  ffi::Optional<ffi::VisitInterrupt> Visit_(const BreakNode *op) override {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *op) override {
    return std::nullopt;
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileNode *op) override {
    return std::nullopt;
  }
};

static bool ContainsLoopBreak(const Stmt &stmt) {
  auto detector = ffi::make_object<LoopBreakDetector>();
  detector->Visit(stmt);
  return detector->found;
}

static bool IsThreadReturnEvaluate(const Stmt &stmt) {
  if (const auto *returned = stmt.as<ReturnNode>()) {
    const auto *value = returned->value.as<TupleNode>();
    TVM_FFI_CHECK(value && value->fields.empty(), ValueError)
        << "CuTeDSL device kernels cannot return a value";
    return true;
  }
  const auto *eval = stmt.as<EvaluateNode>();
  if (eval == nullptr) {
    return false;
  }
  const auto *call = eval->value.as<CallNode>();
  return call != nullptr && PythonIsOp(call->op, "thread_return");
}

struct ConditionalThreadReturn {
  PrimExpr condition;
  std::optional<Stmt> pre_return;
};

std::optional<ConditionalThreadReturn>
GetConditionalThreadReturn(const Stmt &stmt) {
  const auto *if_node = stmt.as<IfNode>();
  if (if_node == nullptr || if_node->else_case) {
    return std::nullopt;
  }
  if (IsThreadReturnEvaluate(if_node->then_case)) {
    return ConditionalThreadReturn{if_node->condition, std::nullopt};
  }
  if (const auto *seq = if_node->then_case.as<SeqStmtNode>();
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
  const auto *ptr_call = expr.as<CallNode>();
  if (ptr_call == nullptr) {
    return std::nullopt;
  }
  if (PythonIsOp(ptr_call->op, "address_of")) {
    const auto *buffer_load = ptr_call->args[0].as<TensorLoadNode>();
    ICHECK(buffer_load) << "address_of arg must be BufferLoad";
    return buffer_load->source.as_or_throw<TensorVar>().dtype();
  }
  if (PythonIsOp(ptr_call->op, "tvm_access_ptr")) {
    ICHECK(!ptr_call->args.empty());
    return PythonDType(ptr_call->args[0]);
  }
  if (PythonIsOp(ptr_call->op, "access_ptr")) {
    ICHECK_EQ(ptr_call->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *buffer_load = ptr_call->args[0].as<TensorLoadNode>();
    ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
    return buffer_load->source.as_or_throw<TensorVar>().dtype();
  }
  return std::nullopt;
}

int GetTileLangCPAsyncTransferBytes(const CallNode *op) {
  ICHECK(op->args.size() == 3 || op->args.size() == 4)
      << "tl::ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
         "src_access_ptr, num_elems, [predicate])";
  const auto *num_elems_imm = op->args[2].as<IntImmNode>();
  ICHECK(num_elems_imm) << "tl::ptx_cp_async num_elems must be IntImm, but got "
                        << op->args[2];
  int64_t num_elems = num_elems_imm->value.as<int64_t>().value();
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
private:
  void VisitStmt(const tirx::Function &f) {
    if (f->GetAttr<Array<PrimExpr>>("cluster_dims").has_value()) {
      launch_with_cluster = true;
      auto cluster_dims = f->GetAttr<Array<PrimExpr>>("cluster_dims").value();
      cluster_grid_x_ext =
          cluster_dims[0].as<IntImmNode>()->value.as<int64_t>().value();
      cluster_grid_y_ext =
          cluster_dims[1].as<IntImmNode>()->value.as<int64_t>().value();
      cluster_grid_z_ext =
          cluster_dims[2].as<IntImmNode>()->value.as<int64_t>().value();
      ICHECK(cluster_grid_x_ext > 0 && cluster_grid_y_ext > 0 &&
             cluster_grid_z_ext > 0);
    }
  }

  bool launch_with_cluster = false;
  int64_t cluster_grid_x_ext = 1;
  int64_t cluster_grid_y_ext = 1;
  int64_t cluster_grid_z_ext = 1;

public:
  std::optional<std::tuple<int64_t, int64_t, int64_t>>
  extract(const tirx::Function &f) {
    this->VisitStmt(f);
    if (launch_with_cluster) {
      return std::make_tuple(cluster_grid_x_ext, cluster_grid_y_ext,
                             cluster_grid_z_ext);
    }
    return std::nullopt;
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
    elem_type = "TFloat32";
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    elem_type = "BFloat16";
  } else if (t.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3,
                           kDLFloat8_e4m3b11fnuz, kDLFloat8_e4m3fn,
                           kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                           kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu)) {
    if (t.MatchesCode(kDLFloat8_e3m4)) {
      // unsupported
    } else if (t.MatchesCode(kDLFloat8_e4m3)) {
      elem_type =
          "Float8E4M3FN"; // Only Float8E4M3FN is supported at the moment
    } else if (t.MatchesCode(kDLFloat8_e4m3b11fnuz)) {
      // unsupported
    } else if (t.MatchesCode(kDLFloat8_e4m3fn)) {
      elem_type = "Float8E4M3FN";
    } else if (t.MatchesCode(kDLFloat8_e4m3fnuz)) {
      // unsupported
    } else if (t.MatchesCode(kDLFloat8_e5m2)) {
      elem_type = "Float8E5M2";
    } else if (t.MatchesCode(kDLFloat8_e5m2fnuz)) {
      // unsupported
    } else if (t.MatchesCode(kDLFloat8_e8m0fnu)) {
      elem_type = "Float8E8M0FNU";
    }
  } else if (t.MatchesCode(kDLFloat6_e2m3fn, kDLFloat6_e3m2fn)) {
    if (t.MatchesCode(kDLFloat6_e3m2fn)) {
      elem_type = "Float6E3M2FN";
    } else if (t.MatchesCode(kDLFloat6_e2m3fn)) {
      elem_type = "Float6E2M3FN";
    }
  } else if (t.MatchesCode(kDLFloat4_e2m1fn)) {
    if (tl::IsFP4UnpackedType(t)) {
      elem_type = "Float4E2M1FN_unpack";
    } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
      elem_type = "Float4E2M1FN";
    }
  } else if (t.MatchesElementType(kDLBool, 1)) {
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
    LOG(FATAL) << "Cannot convert type " << t << " to CuTeDSL type!";
  }

  if (elem_type == "Float4E2M1FN_unpack") {
    return "tl." + elem_type;
  }
  return "cutlass." + elem_type;
}
} // namespace

void CodeGenTileLangCuTeDSL::PrintType(PrimType t,
                                       std::ostream &os) { // NOLINT(*)
  ICHECK(t.IsScalar()) << "Should not print a non-scalar type in CuTeDSL: "
                       << t;
  os << DTypeToString(t);
}

static bool IsZeroLike(const Expr &expr) {
  Expr value = expr;
  if (const CastNode *cast = value.as<CastNode>()) {
    value = cast->value;
  }
  if (const IntImmNode *imm = value.as<IntImmNode>()) {
    return imm->value.as<int64_t>().value() == 0;
  }
  if (const FloatImmNode *imm = value.as<FloatImmNode>()) {
    return imm->value == 0.0;
  }
  return false;
}

static bool EndsWithLoad(const std::string &s) {
  return s.size() >= 7 && s.compare(s.size() - 7, 7, ".load()") == 0;
}

static bool IsZeroTensorLike(
    const Expr &expr,
    const std::unordered_set<const VarNode *> *zero_like_vars = nullptr) {
  Expr value = expr;
  while (true) {
    if (const CastNode *cast = value.as<CastNode>()) {
      value = cast->value;
      continue;
    }
    if (const BroadcastNode *broadcast = value.as<BroadcastNode>()) {
      value = broadcast->value;
      continue;
    }
    break;
  }
  if (const VarNode *var = value.as<VarNode>()) {
    return zero_like_vars != nullptr && zero_like_vars->count(var);
  }
  return IsZeroLike(value);
}

class BufferLoadDetector : public tirx::StmtExprVisitor {
public:
  bool found = false;
  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadNode *op) override {
    found = true;
    return std::nullopt;
  }
};

static bool ContainsBufferLoad(const Expr &expr) {
  auto detector = ffi::make_object<BufferLoadDetector>();
  detector->Visit(expr);
  return detector->found;
}

void CodeGenTileLangCuTeDSL::Dispatch_(const BroadcastNode *op,
                                       std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLFloat4_e2m1fn, 4)) {
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
  PrimType dt = PythonDType(op->value);
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

void CodeGenTileLangCuTeDSL::Dispatch_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  switch (PythonDType(GetRef<Expr>(op)).bits()) {
  case 64:
  case 32:
  case 16:
  case 8:
  case 4: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      // For CuTeDSL, use Python's float('inf') instead of CUDA macros
      PrintType(PythonDType(GetRef<Expr>(op)), temp);
      temp << "(";
      if (op->value < 0) {
        temp << "float('-inf')";
      } else {
        temp << "float('inf')";
      }
      temp << ")";
    } else if (std::isnan(op->value)) {
      // For CuTeDSL, use Python's float('nan')
      PrintType(PythonDType(GetRef<Expr>(op)), temp);
      temp << "(float('nan'))";
    } else {
      // For CuTeDSL, use Python's float.fromhex() with hexfloat for full
      // precision
      PrintType(PythonDType(GetRef<Expr>(op)), temp);
      temp << "(float.fromhex('" << FlexibleHexFormat(op->value) << "'))";
    }
    MarkConst(temp.str());
    os << temp.str();
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << PythonDType(GetRef<Expr>(op))
               << "\n";
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const IntImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  // CuTeDSL's tensor __setitem__ uses as_numeric() which converts bare
  // Python ints to Int32.  For non-int32 integer literals (e.g. int16, uint8),
  // wrap with the CuTeDSL type constructor so the value has the correct width.
  if (PythonDType(GetRef<Expr>(op)) == PrimType::Bool()) {
    os << (op->value ? "True" : "False");
  } else if (PythonDType(GetRef<Expr>(op)) != PrimType::Int(32)) {
    std::ostringstream temp;
    PrintType(PythonDType(GetRef<Expr>(op)), temp);
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

void CodeGenTileLangCuTeDSL::Dispatch_(const CastNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrimType from_ty = PythonDType(op->value);
  PrimType target_ty = PythonDType(GetRef<Expr>(op));
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  const bool from_is_cutedsl_fp8 = from_ty.MatchesCode(kDLFloat8_e4m3) ||
                                   from_ty.MatchesCode(kDLFloat8_e4m3fn) ||
                                   from_ty.MatchesCode(kDLFloat8_e5m2);
  const bool target_is_fp8_widening =
      target_ty == PrimType::Float(16) || target_ty == PrimType::Float(32) ||
      target_ty.MatchesElementType(kDLBfloat, 16);
  const bool target_is_cutedsl_fp8 = target_ty.MatchesCode(kDLFloat8_e4m3) ||
                                     target_ty.MatchesCode(kDLFloat8_e4m3fn) ||
                                     target_ty.MatchesCode(kDLFloat8_e5m2);
  const bool from_is_fp8_narrowing_source =
      from_ty == PrimType::Float(16) || from_ty == PrimType::Float(32) ||
      from_ty.MatchesElementType(kDLBfloat, 16);
  if (from_ty.IsScalar() && target_ty.IsScalar() &&
      (from_ty.MatchesCode(kDLInt) || from_ty.MatchesCode(kDLUInt)) &&
      (target_ty.MatchesCode(kDLInt) || target_ty.MatchesCode(kDLUInt)) &&
      target_ty.bits() > from_ty.bits()) {
    const auto *left_shift = op->value.as<LShiftNode>();
    const auto *right_shift = op->value.as<RShiftNode>();
    if (left_shift || right_shift) {
      PrimExpr left = left_shift ? left_shift->a : right_shift->a;
      PrimExpr right = left_shift ? left_shift->b : right_shift->b;
      PrimType left_type = left.ty();
      if (left_type.code() == target_ty.code() &&
          left_type.bits() < target_ty.bits()) {
        std::string left_source = PrintExpr_(left);
        std::string right_source = PrintExpr_(right);
        PrintType(target_ty, os);
        os << "((";
        PrintType(target_ty, os);
        os << "(" << left_source << ") " << (left_shift ? "<<" : ">>") << " "
           << right_source << "))";
        return;
      }
    }
    if (const CallNode *call = op->value.as<CallNode>();
        call && (PythonIsOp(call->op, "shift_left") ||
                 PythonIsOp(call->op, "shift_right"))) {
      ICHECK_EQ(call->args.size(), 2U);
      const PrimType lhs_ty = PythonDType(call->args[0]);
      const bool same_signedness =
          (lhs_ty.MatchesCode(kDLInt) && target_ty.MatchesCode(kDLInt)) ||
          (lhs_ty.MatchesCode(kDLUInt) && target_ty.MatchesCode(kDLUInt));
      if (same_signedness && lhs_ty.bits() < target_ty.bits()) {
        const std::string lhs = PrintExpr_(call->args[0]);
        const std::string rhs = PrintExpr_(call->args[1]);
        PrintType(target_ty, os);
        os << "((";
        PrintType(target_ty, os);
        os << "(" << lhs << ") "
           << (PythonIsOp(call->op, "shift_left") ? "<<" : ">>") << " " << rhs
           << "))";
        return;
      }
    }
  }

  if (from_ty.IsScalar() && from_is_cutedsl_fp8 && target_is_fp8_widening) {
    // CuTeDSL/NVGPU widens FP8 through nvgpu.cvt_fpext, whose operand must be
    // a 32-bit aligned vector.  Pack the scalar into a four-lane FP8 vector,
    // reuse the tensor cast path, then read back lane zero.
    std::string src = SSAGetID(PrintExpr_(op->value), from_ty);
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
    std::string src = SSAGetID(PrintExpr_(op->value), from_ty);
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

  std::string src = SSAGetID(PrintExpr_(op->value), from_ty);

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

void CodeGenTileLangCuTeDSL::Dispatch_(const DivNode *op,
                                       std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
    PrintBinaryExpr_("//", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
  } else {
    if (enable_fastmath_) {
      os << "tl.divf(" << PrintExpr_(op->a) << ", " << PrintExpr_(op->b)
         << ", fastmath=True)";
    } else {
      PrintBinaryExpr_("tl.divf", PythonDType(GetRef<Expr>(op)), op->a, op->b,
                       os);
    }
  }
}
void CodeGenTileLangCuTeDSL::Dispatch_(const MinNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("tl.min", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}
void CodeGenTileLangCuTeDSL::Dispatch_(const MaxNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("tl.max", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
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
void CodeGenTileLangCuTeDSL::Dispatch_(const CallNode *op,
                                       std::ostream &os) { // NOLINT(*)
  auto annotations = op->attrs.value_or(DictAttrs(Map<String, Any>{}))
                         .as_or_throw<DictAttrs>()
                         ->dict;
  if (PythonIsOp(op->op, "clamp")) {
    os << PrintExpr_(tl::LowerClamp(GetRef<Expr>(op).as_or_throw<PrimExpr>()));
    return;
  }
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

  if (PythonIsOp(op->op, "if_then_else") ||
      PythonIsOp(op->op, "if_then_else")) {
    ICHECK_EQ(op->args.size(), 3U)
        << "if_then_else expects <condition, true_value, false_value>";
    auto cast_branch = [&](std::string value, PrimType branch_dtype) {
      PrimType result_dtype = PythonDType(GetRef<Expr>(op));
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
    std::string cond = PrintExpr_(op->args[0]);
    std::string true_val =
        cast_branch(PrintExpr_(op->args[1]), PythonDType(op->args[1]));
    std::string false_val =
        cast_branch(PrintExpr_(op->args[2]), PythonDType(op->args[2]));
    os << "(" << true_val << " if " << cond << " else " << false_val << ")";
  } else if (PythonIsOp(op->op, "ptx_cp_async")) {
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
  } else if (PythonIsOp(op->op, "ptx_cp_async")) {
    // TileLang version: args[0] = dst_access_ptr, args[1] = src_access_ptr,
    // args[2] = num_elems, args[3] = predicate (optional)
    int total_bytes = GetTileLangCPAsyncTransferBytes(op);

    std::string dst = PrintExpr_(op->args[0]);
    std::string src = PrintExpr_(op->args[1]);
    std::string size = std::to_string(total_bytes);

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
  } else if (PythonIsOp(op->op, "ptx_commit_group")) {
    print_extern_call_stmt("tl.cp_async_commit");
  } else if (PythonIsOp(op->op, "ptx_wait_group")) {
    print_extern_call_stmt("tl.cp_async_wait");
  } else if (PythonIsOp(op->op, "create_barriers")) {
    PrintIndent();
    int barrier_count = tl::GetIntAnnotation(op->args[0]).value();
    stream << mbarrier_name_
           << " = tl.alloc_smem(cutlass.Uint64, size_in_elems=" << barrier_count
           << ")\n";
  } else if (PythonIsOp(op->op, "ptx_arrive_barrier") ||
             PythonIsOp(op->op, "mbarrier_arrive")) {
    ICHECK_EQ(op->args.size(), 1);
    PrintIndent();
    auto mbarrier_obj = GetVarPtr_(op->args[0]);
    stream << "tl.mbarrier_arrive(" << mbarrier_obj << ")\n";
  } else if (PythonIsOp(op->op, "ptx_arrive_cluster_barrier")) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = GetVarPtr_(op->args[0]);
    auto cta_id = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_arrive(" << mbarrier_obj << ", " << cta_id << ")\n";
  } else if (PythonIsOp(op->op, "ptx_init_barrier_thread_count") ||
             PythonIsOp(op->op, "mbarrier_init")) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = GetVarPtr_(op->args[0]);
    auto arrive_count = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_init(" << mbarrier_obj << ", " << arrive_count
           << ")\n";
  } else if (PythonIsOp(op->op, "ptx_arrive_barrier_expect_tx") ||
             PythonIsOp(op->op, "mbarrier_arrive_expect_tx")) {
    if (op->args.size() == 2) {
      PrintIndent();
      auto mbarrier_obj = GetVarPtr_(op->args[0]);
      auto transaction_bytes = PrintExpr_(op->args[1]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ")\n";
    } else if (op->args.size() == 4) {
      PrintIndent();
      auto mbarrier_obj = GetVarPtr_(op->args[0]);
      auto transaction_bytes = PrintExpr_(op->args[1]);
      auto cta_id = PrintExpr_(op->args[2]);
      auto pred = PrintExpr_(op->args[3]);
      stream << "tl.arrive_and_expect_tx(" << mbarrier_obj << ", "
             << transaction_bytes << ", " << cta_id << ", " << pred << ")\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier_expect_tx "
                 << op->args.size();
    }
  } else if (PythonIsOp(op->op, "ptx_cp_async_barrier")) {
    auto pointer = GetVarPtr_(op->args[0]);
    PrintIndent();
    stream << "tl.mbarrier_cp_async_arrive(" << pointer << ")\n";
  } else if (PythonIsOp(op->op, "ptx_fence_barrier_init")) {
    print_extern_call_stmt("tl.fence_barrier_init");
  } else if (PythonIsOp(op->op, "ptx_cp_async_barrier_noinc")) {
    auto pointer = GetVarPtr_(op->args[0]);
    PrintIndent();
    stream << "tl.mbarrier_cp_async_arrive_noinc(" << pointer << ")\n";
  } else if (PythonIsOp(op->op, "mbarrier_expect_tx")) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = GetVarPtr_(op->args[0]);
    auto transaction_bytes = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_expect_tx(" << mbarrier_obj << ", "
           << transaction_bytes << ")\n";
  } else if (PythonIsOp(op->op, "mbarrier_wait_parity")) {
    ICHECK_EQ(op->args.size(), 2);
    PrintIndent();
    auto mbarrier_obj = GetVarPtr_(op->args[0]);
    auto phase = PrintExpr_(op->args[1]);
    stream << "tl.mbarrier_wait(" << mbarrier_obj << ", " << phase << ")\n";
  } else if (PythonIsOp(op->op, "ptx_init_tensor_memory")) {
    ICHECK_EQ(op->args.size(), 2U);
    PrintIndent();
    auto tmem_buffer = PrintExpr_(op->args[0]);
    auto num_cols = PrintExpr_(op->args[1]);
    bool use_2cta = annotations.count("use_2cta") &&
                    tl::GetIntAnnotation(annotations["use_2cta"]).value();
    stream << "tl.tmem_allocate(" << tmem_buffer << ", " << num_cols
           << ", use_2cta=" << (use_2cta ? "True" : "False") << ")\n";
  } else if (PythonIsOp(op->op, "ptx_deallocate_tensor_memory")) {
    ICHECK_EQ(op->args.size(), 2U);
    PrintIndent();
    auto tmem_buffer = PrintExpr_(op->args[0]);
    auto num_cols = PrintExpr_(op->args[1]);
    bool use_2cta = annotations.count("use_2cta") &&
                    tl::GetIntAnnotation(annotations["use_2cta"]).value();
    stream << "tl.tmem_deallocate(" << tmem_buffer << ", " << num_cols
           << ", use_2cta=" << (use_2cta ? "True" : "False") << ")\n";
  } else if (PythonIsOp(op->op, "no_set_max_nreg")) {
    // do nothing
  } else if (PythonIsOp(op->op, "prefetch_tma_descriptor")) {
    print_extern_call_stmt("tl.prefetch_tma_descriptor");
  } else if (PythonIsOp(op->op, "tma_load")) {
    std::ostringstream ss;
    ICHECK_GE(op->args.size(), 2);
    auto pol = op->args[op->args.size() - 1].as<IntImmNode>();
    ICHECK(pol) << "Eviction policy must be IntImm";
    ICHECK_GE(pol->value.as<int64_t>().value(), 0);
    ICHECK_LT(static_cast<size_t>(pol->value.as<int64_t>().value()),
              eviction_policy_names_.size());
    auto eviction_policy =
        eviction_policy_names_[pol->value.as<int64_t>().value()];
    // Simplify the code by using the default eviction policy
    if (eviction_policy != "EVICT_NORMAL") {
      LOG(FATAL) << "Eviction policy " << eviction_policy
                 << " is not supported currently";
    } else {
      ss << "tl.tma_load(";
    }
    auto desc = op->args[0];
    ss << PrintExpr_(desc) << ", ";
    ss << GetVarPtr_(op->args[1]) << ", ";
    ss << PrintBytePointerExpr_(op->args[2]) << ", (";
    for (size_t i = 3; i < op->args.size() - 1; i++) {
      if (i > 3)
        ss << ", ";
      ss << PrintExpr_(op->args[i]);
    }
    ss << ")";
    if (annotations.count("use_2cta") &&
        tl::GetIntAnnotation(annotations["use_2cta"]).value()) {
      ss << ", use_2cta=True";
    }
    ss << ")\n";
    PrintIndent();
    stream << ss.str();
  } else if (PythonIsOp(op->op, "tma_load_im2col")) {
    std::ostringstream ss;
    ICHECK_GE(op->args.size(), 6)
        << "tma_load_im2col requires desc, mbar, smem, coordinates, "
           "im2col offsets, and eviction policy";
    auto pol = op->args[op->args.size() - 1].as<IntImmNode>();
    ICHECK(pol) << "Eviction policy must be IntImm";
    ICHECK_GE(pol->value.as<int64_t>().value(), 0);
    ICHECK_LT(static_cast<size_t>(pol->value.as<int64_t>().value()),
              eviction_policy_names_.size());
    auto eviction_policy =
        eviction_policy_names_[pol->value.as<int64_t>().value()];
    if (eviction_policy != "EVICT_NORMAL") {
      LOG(FATAL) << "Eviction policy " << eviction_policy
                 << " is not supported currently";
    }

    auto desc = op->args[0];
    ss << "tl.tma_load(";
    ss << PrintExpr_(desc) << ", ";
    ss << GetVarPtr_(op->args[1]) << ", ";
    ss << PrintBytePointerExpr_(op->args[2]) << ", (";
    for (size_t i = 3; i < op->args.size() - 3; i++) {
      if (i > 3)
        ss << ", ";
      ss << PrintExpr_(op->args[i]);
    }
    ss << "), im2col_offsets=(";
    for (size_t i = op->args.size() - 3; i < op->args.size() - 1; i++) {
      if (i > op->args.size() - 3)
        ss << ", ";
      ss << PrintExpr_(op->args[i]);
    }
    ss << "))\n";
    PrintIndent();
    stream << ss.str();
  } else if (PythonIsOp(op->op, "tma_store")) {
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
    auto need_reduce = need_reduce_ptr->value.as<int64_t>().value();
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
    ICHECK_GE(eviction_idx_ptr->value.as<int64_t>().value(), 0)
        << "tma_store eviction policy index must be >= 0, got "
        << eviction_idx_ptr->value.as<int64_t>().value();
    ICHECK_LT(
        static_cast<size_t>(eviction_idx_ptr->value.as<int64_t>().value()),
        eviction_policy_names_.size())
        << "tma_store eviction policy index "
        << eviction_idx_ptr->value.as<int64_t>().value()
        << " out of bounds (max " << eviction_policy_names_.size() - 1 << ")";
    auto eviction_policy =
        eviction_policy_names_[eviction_idx_ptr->value.as<int64_t>().value()];

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
  } else if (PythonIsOp(op->op, "ptx_ldmatrix")) {
    int trans = tl::GetIntAnnotation(op->args[0]).value();
    int num = tl::GetIntAnnotation(op->args[1]).value();
    std::string func_name = "tl.ptx_ldmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (PythonIsOp(op->op, "ptx_stmatrix")) {
    int trans = tl::GetIntAnnotation(op->args[0]).value();
    int num = tl::GetIntAnnotation(op->args[1]).value();
    bool is_shape_encoded =
        op->args.size() >= 4 && op->args.back().as<StringImmNode>();
    if (is_shape_encoded) {
      ICHECK_EQ(op->args.back().as_or_throw<StringImm>()->value, "m8n8")
          << "CuTeDSL stmatrix codegen only supports m8n8";
    }
    std::string func_name = "tl.ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2, is_shape_encoded ? 1 : 0);
  } else if (PythonIsOp(op->op, "fence_proxy_async")) {
    print_extern_call_stmt("tl.fence_proxy_async");
  } else if (PythonIsOp(op->op, "tma_store_arrive")) {
    print_extern_call_stmt("tl.tma_store_arrive");
  } else if (PythonIsOp(op->op, "tma_store_wait")) {
    int count = tl::GetIntAnnotation(op->args[0]).value();
    bool read =
        op->args.size() == 1 || tl::GetIntAnnotation(op->args[1]).value() != 0;
    PrintIndent();
    stream << "tl.tma_store_wait(" << count
           << ", read=" << (read ? "True" : "False") << ")\n";
  } else if (PythonIsOp(op->op, "warpgroup_arrive")) {
    PrintIndent();
    stream << "tl.warpgroup_arrive()\n";
  } else if (PythonIsOp(op->op, "warpgroup_commit_batch")) {
    PrintIndent();
    stream << "tl.warpgroup_commit_batch()\n";
  } else if (PythonIsOp(op->op, "warpgroup_wait")) {
    PrintIndent();
    int num_mma = tl::GetIntAnnotation(op->args[0]).value();
    stream << "tl.warpgroup_wait(" << num_mma << ")\n";
  } else if (PythonIsOp(op->op, "warpgroup_fence_operand")) {
    // no-op: warpgroup_fence_operand is not needed in CuTeDSL
  } else if (PythonIsOp(op->op, "set_max_nreg")) {
    PrintIndent();
    int nreg = tl::GetIntAnnotation(op->args[0]).value();
    int is_inc = tl::GetIntAnnotation(op->args[1]).value();
    std::string func_name =
        is_inc ? "tl.warpgroup_reg_alloc" : "tl.warpgroup_reg_dealloc";
    stream << func_name << "(" << nreg << ")\n";
  } else if (PythonIsOp(op->op, "annotate_producer_reg_dealloc") ||
             PythonIsOp(op->op, "annotate_consumer_reg_alloc")) {
    return;
  } else if (PythonIsOp(op->op, "wait_wgmma")) {
    PrintIndent();
    int num_mma = tl::GetIntAnnotation(op->args[0]).value();
    stream << "tl.wgmma_wait_group(" << num_mma << ")\n";
  } else if (PythonIsOp(op->op, "pack_b16")) {
    os << "tl.pack_half2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "sync_grid")) {
    PrintIndent();
    stream << "tl.sync_grid()\n";
  } else if (PythonIsOp(op->op, "pdl_trigger")) {
    PrintIndent();
    stream << "tl.griddepcontrol_launch_dependents()\n";
  } else if (PythonIsOp(op->op, "pdl_sync")) {
    PrintIndent();
    stream << "tl.griddepcontrol_wait()\n";
  } else if (PythonIsOp(op->op, "cluster_arrive_relaxed")) {
    PrintIndent();
    stream << "tl.cluster_arrive_relaxed()\n";
  } else if (PythonIsOp(op->op, "cluster_arrive")) {
    PrintIndent();
    stream << "tl.cluster_arrive()\n";
  } else if (PythonIsOp(op->op, "cluster_wait")) {
    PrintIndent();
    stream << "tl.cluster_wait()\n";
  } else if (PythonIsOp(op->op, "cluster_sync")) {
    PrintIndent();
    stream << "tl.cluster_sync()\n";
  } else if (PythonIsOp(op->op, "block_rank_in_cluster")) {
    os << "tl.block_rank_in_cluster()";
  } else if (PythonIsOp(op->op, "clc_try_cancel")) {
    print_extern_call_stmt("tl.clc_try_cancel");
  } else if (PythonIsOp(op->op, "clc_try_cancel_multicast")) {
    print_extern_call_stmt("tl.clc_try_cancel_multicast");
  } else if (PythonIsOp(op->op, "clc_is_canceled")) {
    os << "tl.clc_is_canceled(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "clc_get_first_ctaid_x")) {
    os << "tl.clc_get_first_ctaid_x(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "clc_get_first_ctaid_y")) {
    os << "tl.clc_get_first_ctaid_y(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "clc_get_first_ctaid_z")) {
    os << "tl.clc_get_first_ctaid_z(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "loop_break") ||
             PythonIsOp(op->op, "break_loop")) {
    if (in_break_loop_) {
      PrintIndent();
      stream << "_loop_break_" << current_break_id_ << " = 1\n";
      break_emitted_in_seq_ = true;
    } else {
      PrintIndent();
      stream << "break\n";
    }
  } else if (PythonIsOp(op->op, "ptx_mma") ||
             PythonIsOp(op->op, "tirx.ptx_legacy.mma")) {
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
    TVM_FFI_CHECK(op->args.size() == 13U, ValueError)
        << "CuTeDSL MMA does not support bitwise MMA operators";
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = GetVarPtr_(op->args[6]);
    std::string a_bias = PrintExpr_(op->args[7]);
    std::string b_ref = GetVarPtr_(op->args[8]);
    std::string b_bias = PrintExpr_(op->args[9]);
    std::string c_ref = GetVarPtr_(op->args[10]);
    std::string c_bias = PrintExpr_(op->args[11]);
    std::string saturate = PrintExpr_(op->args[12]);

    // Generate call to tl.ptx_mma dispatcher
    PrintIndent();
    stream << "tl.ptx_mma(\"" << shape << "\", \"" << A_layout << "\", \""
           << B_layout << "\", \"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << a_ref << ", " << a_bias << ", " << b_ref
           << ", " << b_bias << ", " << c_ref << ", " << c_bias << ", "
           << saturate << ")\n";
  } else if (PythonIsOp(op->op, "ptx_mma_sm70")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "ptx_mma_sp")) {
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
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
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
        tl::GetIntAnnotation(op->args[15]).value() ? "True" : "False";

    PrintIndent();
    stream << "tl.ptx_mma_sp(\"" << shape << "\", \"" << A_layout << "\", \""
           << B_layout << "\", \"" << A_dtype << "\", \"" << B_dtype << "\", \""
           << C_dtype << "\", " << a_ref << ", " << a_bias << ", " << b_ref
           << ", " << b_bias << ", " << c_ref << ", " << c_bias << ", "
           << meta_ref << ", " << meta_bias << ", " << sparse_selector << ", "
           << saturate << ")\n";
  } else if (PythonIsOp(op->op, "ptx_wgmma_ss")) {
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
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    bool a_is_k_major = tl::GetIntAnnotation(op->args[1]).value();
    bool b_is_k_major = tl::GetIntAnnotation(op->args[2]).value();
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_desc = PrintExpr_(op->args[6]);
    std::string A_offset = PrintExpr_(op->args[7]);
    std::string b_desc = PrintExpr_(op->args[8]);
    std::string B_offset = PrintExpr_(op->args[9]);
    std::string c_ref = GetVarPtr_(op->args[10]);
    std::string c_offset = PrintExpr_(op->args[11]);
    std::string scale_out = PrintExpr_(op->args[12]);
    bool scale_in_a = tl::GetIntAnnotation(op->args[13]).value();
    bool scale_in_b = tl::GetIntAnnotation(op->args[14]).value();
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
  } else if (PythonIsOp(op->op, "ptx_wgmma_rs")) {
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
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    bool b_is_k_major = tl::GetIntAnnotation(op->args[1]).value();
    std::string A_dtype = op->args[2].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string a_ref = GetVarPtr_(op->args[5]);
    std::string A_offset = PrintExpr_(op->args[6]);
    std::string b_desc = PrintExpr_(op->args[7]);
    std::string B_offset = PrintExpr_(op->args[8]);
    std::string c_ref = GetVarPtr_(op->args[9]);
    std::string c_offset = PrintExpr_(op->args[10]);
    std::string scale_out = PrintExpr_(op->args[11]);
    bool scale_in_a = tl::GetIntAnnotation(op->args[12]).value();
    bool scale_in_b = tl::GetIntAnnotation(op->args[13]).value();
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
  } else if (PythonIsOp(op->op, "ptx_tcgen05_mma_ss")) {
    ICHECK_EQ(op->args.size(), 15U)
        << "ptx_tcgen05_mma_ss expects 15 arguments";
    std::string kind_dtype = op->args[0].as_or_throw<StringImm>()->value;
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
    bool enable_ws = tl::GetIntAnnotation(op->args[13]).value();
    bool enable_2cta = tl::GetIntAnnotation(op->args[14]).value();
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
  } else if (PythonIsOp(op->op, "ptx_tcgen05_mma_ts")) {
    ICHECK_EQ(op->args.size(), 14U)
        << "ptx_tcgen05_mma_ts expects 14 arguments";
    std::string kind_dtype = op->args[0].as_or_throw<StringImm>()->value;
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
    bool enable_2cta = tl::GetIntAnnotation(op->args[13]).value();
    PrintIndent();
    stream << "tl.tcgen05mma_ts(\"" << kind_dtype << "\", " << a_ref << "[0] + "
           << A_offset << ", (" << b_desc << " + " << B_offset << "), " << c_ref
           << "[0] + " << c_offset << ", " << desc_val << ", " << scale_out
           << ", " << mask0 << ", " << mask1 << ", " << mask2 << ", " << mask3
           << ", use_2cta=" << (enable_2cta ? "True" : "False") << ")\n";
  } else if (PythonIsOp(op->op, "ptx_tcgen05_mma_blockscaled_ss")) {
    ICHECK_EQ(op->args.size(), 16U)
        << "ptx_tcgen05_mma_blockscaled_ss expects 16 arguments";
    std::string kind_dtype = op->args[0].as_or_throw<StringImm>()->value;
    std::string a_desc = PrintExpr_(op->args[1]);
    std::string A_offset = PrintExpr_(op->args[2]);
    std::string b_desc = PrintExpr_(op->args[3]);
    std::string B_offset = PrintExpr_(op->args[4]);
    std::string c_ref = PrintExpr_(op->args[5]);
    std::string c_offset = PrintExpr_(op->args[6]);
    std::string desc_val = PrintExpr_(op->args[7]);
    std::string scale_out = PrintExpr_(op->args[8]);
    std::string sfa_ref = PrintExpr_(op->args[9]);
    std::string sfa_offset = PrintExpr_(op->args[10]);
    std::string sfb_ref = PrintExpr_(op->args[11]);
    std::string sfb_offset = PrintExpr_(op->args[12]);
    bool enable_2cta = tl::GetIntAnnotation(op->args[15]).value();
    PrintIndent();
    stream << "tl.tcgen05mma_blockscaled_ss(\"" << kind_dtype << "\", ("
           << a_desc << " + " << A_offset << "), (" << b_desc << " + "
           << B_offset << "), " << c_ref << "[0] + " << c_offset << ", "
           << desc_val << ", " << scale_out << ", " << sfa_ref << "[0] + "
           << sfa_offset << ", " << sfb_ref << "[0] + " << sfb_offset
           << ", use_2cta=" << (enable_2cta ? "True" : "False") << ")\n";
  } else if (PythonIsOp(op->op, "tcgen05_ld")) {
    ICHECK(op->args.size() == 6U || op->args.size() == 7U)
        << "tcgen05_ld expects 6 or 7 arguments";
    int inst_bits = tl::GetIntAnnotation(op->args[0]).value();
    int chunks = tl::GetIntAnnotation(op->args[1]).value();
    bool pack16 = tl::GetIntAnnotation(op->args[2]).value();
    std::string tmem_start_col = PrintExpr_(op->args[3]);
    std::string col_offset = PrintExpr_(op->args[4]);
    // The optional datapath count selects the half-subpartition (16dp)
    // wrappers; a six-argument call from before it existed means the
    // sub-partition-filling 32dp wrappers.
    bool has_datapaths = op->args.size() == 7U;
    int datapaths =
        has_datapaths ? tl::GetIntAnnotation(op->args[5]).value() : 32;
    std::string dst_ptr = PrintExpr_(op->args[has_datapaths ? 6 : 5]);
    PrintIndent();
    stream << "tl.tcgen05_ld_" << datapaths << "dp" << inst_bits << "bNx("
           << chunks << ", " << (pack16 ? "True" : "False") << ", "
           << tmem_start_col << ", " << col_offset << ", " << dst_ptr << ")\n";
  } else if (PythonIsOp(op->op, "tcgen05_st")) {
    ICHECK(op->args.size() == 6U || op->args.size() == 7U)
        << "tcgen05_st expects 6 or 7 arguments";
    int inst_bits = tl::GetIntAnnotation(op->args[0]).value();
    int chunks = tl::GetIntAnnotation(op->args[1]).value();
    bool unpack16 = tl::GetIntAnnotation(op->args[2]).value();
    std::string tmem_start_col = PrintExpr_(op->args[3]);
    std::string col_offset = PrintExpr_(op->args[4]);
    // The optional datapath count selects the half-subpartition (16dp)
    // wrappers; a six-argument call from before it existed means the
    // sub-partition-filling 32dp wrappers.
    bool has_datapaths = op->args.size() == 7U;
    int datapaths =
        has_datapaths ? tl::GetIntAnnotation(op->args[5]).value() : 32;
    std::string src_ptr = PrintExpr_(op->args[has_datapaths ? 6 : 5]);
    PrintIndent();
    stream << "tl.tcgen05_st_" << datapaths << "dp" << inst_bits << "bNx("
           << chunks << ", " << (unpack16 ? "True" : "False") << ", "
           << tmem_start_col << ", " << col_offset << ", " << src_ptr << ")\n";
  } else if (PythonIsOp(op->op, "tcgen05_mma_arrive")) {
    ICHECK_EQ(op->args.size(), 1U) << "tcgen05_mma_arrive expects 1 argument";
    PrintIndent();
    bool use_2cta = annotations.count("use_2cta") &&
                    tl::GetIntAnnotation(annotations["use_2cta"]).value();
    stream << "tl.tcgen05_mma_arrive(" << PrintExpr_(op->args[0])
           << ", use_2cta=" << (use_2cta ? "True" : "False") << ")\n";
  } else if (PythonIsOp(op->op, "tcgen05_before_thread_sync")) {
    ICHECK_EQ(op->args.size(), 0U)
        << "tcgen05_before_thread_sync expects no arguments";
    PrintIndent();
    stream << "tl.tcgen05_before_thread_sync()\n";
  } else if (PythonIsOp(op->op, "tcgen05_after_thread_sync")) {
    ICHECK_EQ(op->args.size(), 0U)
        << "tcgen05_after_thread_sync expects no arguments";
    PrintIndent();
    stream << "tl.tcgen05_after_thread_sync()\n";
  } else if (PythonIsOp(op->op, "ptx_tcgen05_cp_warpx4")) {
    ICHECK_EQ(op->args.size(), 3U)
        << "ptx_tcgen05_cp_warpx4 expects 3 arguments";
    std::string smem_ptr = PrintExpr_(op->args[0]);
    std::string tmem_ptr = PrintExpr_(op->args[1]);
    std::string tmem_col_offset = PrintExpr_(op->args[2]);
    bool use_2cta = false;
    if (annotations.count("use_2cta")) {
      use_2cta = tl::GetIntAnnotation(annotations["use_2cta"]).value();
    }
    PrintIndent();
    stream << "tl.tcgen05_cp_warpx4(" << smem_ptr << ", " << tmem_ptr << ", "
           << tmem_col_offset << ", use_2cta=" << (use_2cta ? "True" : "False")
           << ")\n";
  } else if (PythonIsOp(op->op, "ptx_tcgen05_sf_warp_transpose")) {
    ICHECK_EQ(op->args.size(), 1U)
        << "ptx_tcgen05_sf_warp_transpose expects 1 argument";
    PrintIndent();
    stream << "tl.tcgen05_sf_warp_transpose(" << PrintExpr_(op->args[0])
           << ")\n";
  } else if (PythonIsOp(op->op, "tirx.ptx_legacy.ldmatrix")) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared
    // memory.
    ICHECK_EQ(op->args.size(), 7U);
    bool trans = tl::GetIntAnnotation(op->args[0]).value();
    int num = tl::GetIntAnnotation(op->args[1]).value();
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
  } else if (PythonIsOp(op->op, "mma_store")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "mma_fill")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "ptx_cp_async_bulk")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "ptx_st_bulk_shared")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "ptx_wait_barrier")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "ptx_ldg32")) {
    LOG(FATAL) << "Currently unsupported op: " << op->op;
  } else if (PythonIsOp(op->op, "reinterpret")) {
    PrimType tgt_dtype = PythonDType(GetRef<Expr>(op));
    PrimType src_dtype = PythonDType(op->args[0]);
    ICHECK_EQ(tgt_dtype.lanes() * tgt_dtype.bits(),
              src_dtype.lanes() * src_dtype.bits())
        << "reinterpret expects source and target to have the same number of "
           "bits";

    const TensorLoadNode *load = op->args[0].as<TensorLoadNode>();
    if (load) {
      std::string scope;
      const VarNode *buffer_var = load->source.as_or_throw<TensorVar>().get();
      if (alloc_storage_scope_.count(buffer_var)) {
        scope = alloc_storage_scope_.at(buffer_var);
      }
      if (scope.empty()) {
        scope = PythonStorageScope(load->source.as_or_throw<TensorVar>());
      }
      if (scope == "local.var") {
        std::string expr_str = GetBufferRef_(
            src_dtype, load->source.as_or_throw<TensorVar>(),
            LinearizeBufferIndices_(load->source.as_or_throw<TensorVar>(),
                                    load->indices));
        os << "tl.bitcast(" << expr_str << ", ";
        PrintType(tgt_dtype.WithLanes(1), os);
        os << ")";
        return;
      }

      // Path 1: BufferLoad - use recast_ptr for memory access
      PrimExpr index = LinearizeBufferIndices_(
          load->source.as_or_throw<TensorVar>(), load->indices);
      if (const RampNode *node = index.as<RampNode>(); node) {
        auto p_stride = tl::GetConstInt(node->stride);
        ICHECK(p_stride);
        ICHECK_EQ(*p_stride, 1) << "reinterpret expects contiguous elements";
        index = node->base;
      } else {
        tvm::PVar<PrimExpr> ramp_base;
        if (tvm::ramp(ramp_base, 1, tgt_dtype.lanes()).Match(index)) {
          index = ramp_base.Eval();
        }
      }

      auto ptr_str =
          GetBufferPtr_(load->source.as_or_throw<TensorVar>(), index);
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
      std::string expr_str = PrintExpr_(op->args[0]);
      os << "tl.bitcast(" << expr_str << ", ";
      PrintType(tgt_dtype.WithLanes(1), os);
      os << ")";
    }
  } else if (PythonIsOp(op->op, "thread_return")) {
    os << "return";
  } else if (PythonIsOp(op->op, "shfl_sync")) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_sync expects <mask, value, src_lane, width>.";
    os << "tl.__shfl_sync(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ", "
       << PrintExpr_(op->args[3]) << ")";
  } else if (PythonIsOp(op->op, "shfl_xor_sync")) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "tl.__shfl_xor_sync(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ", "
       << PrintExpr_(op->args[3]) << ")";
  } else if (PythonIsOp(op->op, "shfl_down_sync")) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_down_sync expects <mask, value, delta, width>.";
    os << "tl.__shfl_down_sync(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ", "
       << PrintExpr_(op->args[3]) << ")";
  } else if (PythonIsOp(op->op, "shfl_up_sync")) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_up_sync expects <mask, value, delta, width>.";
    os << "tl.__shfl_up_sync(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ", "
       << PrintExpr_(op->args[3]) << ")";
  } else if (PythonIsOp(op->op, "match_any_sync")) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.match_any_sync expects <mask, value>.";
    os << "tl.__match_any_sync(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "get_lane_idx")) {
    // get_lane_idx(warp_size?) -> threadIdx.x % warp_size
    ICHECK_LE(op->args.size(), 1U)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    std::string warp_size = op->args.empty() ? "32" : PrintExpr_(op->args[0]);
    os << "(tl.thread_idx() % " << warp_size << ")";
  } else if (PythonIsOp(op->op, "get_warp_idx_sync")) {
    // get_warp_idx_sync(warp_size?) -> threadIdx.x // warp_size
    ICHECK_LE(op->args.size(), 1U)
        << "tl.get_warp_idx_sync expects at most one argument <warp_size>.";
    std::string warp_size = op->args.empty() ? "32" : PrintExpr_(op->args[0]);
    os << "(tl.thread_idx() // " << warp_size << ")";
  } else if (PythonIsOp(op->op, "get_warp_idx")) {
    // get_warp_idx(warp_size?) -> threadIdx.x // warp_size
    ICHECK_LE(op->args.size(), 1U)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    std::string warp_size = op->args.empty() ? "32" : PrintExpr_(op->args[0]);
    os << "(tl.thread_idx() // " << warp_size << ")";
  } else if (PythonIsOp(op->op, "get_warp_group_idx")) {
    // get_warp_group_idx(warp_size?, warps_per_group?) ->
    //   threadIdx.x // (warp_size * warps_per_group)
    ICHECK_LE(op->args.size(), 2U)
        << "tl.get_warp_group_idx expects <warp_size, warps_per_group>.";
    std::string warp_size = !op->args.empty() ? PrintExpr_(op->args[0]) : "32";
    std::string warps_per_group =
        op->args.size() >= 2 ? PrintExpr_(op->args[1]) : "4";
    os << "(tl.thread_idx() // (" << warp_size << " * " << warps_per_group
       << "))";
  } else if (PythonIsOp(op->op, "tl_shuffle_elect")) {
    os << "tl.shuffle_elect(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "initialize_wgmma_descriptor")) {
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
  } else if (PythonIsOp(op->op, "initialize_tcgen05_descriptor")) {
    ICHECK_EQ(op->args.size(), 7U)
        << "initialize_tcgen05_descriptor expects 7 arguments";
    std::string descriptor = PrintExpr_(op->args[0]);
    std::string start_address = PrintBytePointerExpr_(op->args[1]);
    std::string leading = PrintExpr_(op->args[2]);
    std::string stride = PrintExpr_(op->args[3]);
    std::string base_offset = PrintExpr_(op->args[4]);
    std::string leading_abs = PrintExpr_(op->args[5]);
    std::string swizzle_mode = PrintExpr_(op->args[6]);
    os << "tl.initialize_tcgen05_descriptor(" << descriptor << ", "
       << start_address << ", " << leading << ", " << stride << ", "
       << base_offset << ", " << leading_abs << ", " << swizzle_mode << ")";
  } else if (PythonIsOp(op->op, "increase_descriptor_offset")) {
    ICHECK_EQ(op->args.size(), 2U)
        << "increase_descriptor_offset expects 2 arguments";
    std::string descriptor = PrintExpr_(op->args[0]);
    std::string offset = PrintExpr_(op->args[1]);
    os << "tl.increase_descriptor_offset(" << descriptor << ", " << offset
       << ")";
  } else if (PythonIsOp(op->op, "__exp")) {
    os << "tl.exp(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__exp10")) {
    os << "tl.exp10(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__log")) {
    os << "tl.log(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__log2")) {
    os << "tl.log2(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__log10")) {
    os << "tl.log10(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__tan")) {
    os << "tl.tan(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__cos")) {
    os << "tl.cos(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "__sin")) {
    os << "tl.sin(" << PrintExpr_(op->args[0]) << ", fastmath=True)";
  } else if (PythonIsOp(op->op, "fast_rcp")) {
    os << "(1.0 / " << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "ieee_add")) {
    // ieee_add(a, b, rounding_mode)
    std::string rounding_mode = op->args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fadd(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "ieee_sub")) {
    std::string rounding_mode = op->args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fsub(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "ieee_mul")) {
    std::string rounding_mode = op->args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fmul(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "ieee_fmaf")) {
    // ieee_fmaf(a, b, c, rounding_mode)
    std::string rounding_mode = op->args[3].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fmaf(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
       << ", rounding=\"" << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "ieee_frcp")) {
    std::string rounding_mode = op->args[1].as_or_throw<StringImm>()->value;
    os << "tl.ieee_frcp(" << PrintExpr_(op->args[0]) << ", rounding=\""
       << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "ieee_fsqrt")) {
    std::string rounding_mode = op->args[1].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fsqrt(" << PrintExpr_(op->args[0]) << ", rounding=\""
       << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "ieee_frsqrt")) {
    os << "tl.rsqrt(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "ieee_fdiv")) {
    std::string rounding_mode = op->args[2].as_or_throw<StringImm>()->value;
    os << "tl.ieee_fdiv(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", rounding=\"" << rounding_mode << "\")";
  } else if (PythonIsOp(op->op, "fma") || PythonIsOp(op->op, "fmul")) {
    // Round-to-nearest fma/mul map onto the IEEE helpers with the fixed
    // "rn" mode. The CuTeDSL helpers are scalar-only, so vector calls are
    // expanded per lane through a register tensor.
    bool is_fma = PythonIsOp(op->op, "fma");
    const char *py_func = is_fma ? "tl.ieee_fmaf" : "tl.ieee_fmul";
    if (PythonDType(GetRef<Expr>(op)).IsScalar()) {
      os << py_func << "(" << PrintExpr_(op->args[0]) << ", "
         << PrintExpr_(op->args[1]);
      if (is_fma) {
        os << ", " << PrintExpr_(op->args[2]);
      }
      os << ", rounding=\"rn\")";
    } else {
      int lanes = PythonDType(GetRef<Expr>(op)).lanes();
      std::vector<std::string> vec_ids;
      vec_ids.reserve(op->args.size());
      for (const Expr &arg : op->args) {
        ICHECK_EQ(PythonDType(arg).lanes(), lanes)
            << "tl." << (is_fma ? "fma" : "fmul")
            << " expects vectorized arguments to share the call lanes";
        vec_ids.push_back(SSAGetID(PrintExpr_(arg), PythonDType(arg)));
      }
      std::string sret = name_supply_->FreshName(is_fma ? "_fma" : "_fmul");
      PrintIndent();
      stream << sret << " = tl.make_rmem_tensor((" << lanes << ",), ";
      PrintType(PythonDType(GetRef<Expr>(op)).WithLanes(1), stream);
      stream << ")\n";
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value;
        value << py_func << "(";
        for (size_t j = 0; j < vec_ids.size(); ++j) {
          if (j != 0) {
            value << ", ";
          }
          PrintVecElemLoad_(vec_ids[j], PythonDType(op->args[j]), i, value);
        }
        value << ", rounding=\"rn\")";
        PrintVecElemStore_(sret, PythonDType(GetRef<Expr>(op)), i, value.str());
      }
      os << sret << ".load()";
    }
  } else if (PythonIsOp(op->op, "warp_reduce_sum")) {
    os << "tl.warp_reduce_sum(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "warp_reduce_max")) {
    os << "tl.warp_reduce_max(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "warp_reduce_min")) {
    os << "tl.warp_reduce_min(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "warp_reduce_bitand")) {
    os << "tl.warp_reduce_bitand(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "warp_reduce_bitor")) {
    os << "tl.warp_reduce_bitor(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "add2")) {
    ICHECK_EQ(op->args.size(), 2U);
    os << "(" << PrintExpr_(op->args[0]) << " + " << PrintExpr_(op->args[1])
       << ")";
  } else if (PythonIsOp(op->op, "sub2")) {
    ICHECK_EQ(op->args.size(), 2U);
    os << "(" << PrintExpr_(op->args[0]) << " - " << PrintExpr_(op->args[1])
       << ")";
  } else if (PythonIsOp(op->op, "mul2")) {
    ICHECK_EQ(op->args.size(), 2U);
    os << "(" << PrintExpr_(op->args[0]) << " * " << PrintExpr_(op->args[1])
       << ")";
  } else if (PythonIsOp(op->op, "fma2")) {
    ICHECK_EQ(op->args.size(), 3U);
    os << "((" << PrintExpr_(op->args[0]) << " * " << PrintExpr_(op->args[1])
       << ") + " << PrintExpr_(op->args[2]) << ")";
  } else if (PythonIsOp(op->op, "max2") || PythonIsOp(op->op, "max2_nan")) {
    ICHECK_EQ(op->args.size(), 2U);
    os << "tl.max2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "min2") || PythonIsOp(op->op, "min2_nan")) {
    ICHECK_EQ(op->args.size(), 2U);
    os << "tl.min2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "abs2")) {
    ICHECK_EQ(op->args.size(), 1U);
    os << "tl.abs2(" << PrintExpr_(op->args[0]) << ")";
  } else if (PythonIsOp(op->op, "address_of")) {
    const TensorLoadNode *load = op->args[0].as<TensorLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    os << GetBufferPtr_(
        load->source.as_or_throw<TensorVar>(),
        LinearizeBufferIndices_(load->source.as_or_throw<TensorVar>(),
                                load->indices));
  } else if (PythonIsOp(op->op, "tensor_data_ptr")) {
    os << GetBufferPtr_(op->args[0].as_or_throw<TensorVar>(), PrimExpr(0));
  } else if (PythonIsOp(op->op, "ptr_byte_offset")) {
    auto pointer_type = op->ty.as_or_throw<PtrType>();
    auto element_type =
        pointer_type->element_type.as_or_throw<PrimType>().WithLanes(1);
    if (element_type.IsVoid())
      element_type = PrimType::UInt(8);
    os << "tl.recast_ptr(tl.recast_ptr(" << GetVarPtr_(op->args[0])
       << ", dtype=cutlass.Uint8) + ";
    if (auto offset = tl::GetConstInt(op->args[1])) {
      os << *offset;
    } else {
      os << PrintExpr_(op->args[1]);
    }
    os << ", dtype=" << DTypeToString(element_type) << ")";
  } else if (PythonIsOp(op->op, "handle_add_byte_offset")) {
    ICHECK_EQ(op->args.size(), 2U);
    os << "tl.handle_add_byte_offset(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "atomic_add_elem_op")) {
    // atomic_add_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicAdd(" << dst_ptr << ", " << src_value << ")\n";
  } else if (PythonIsOp(op->op, "atomic_add_ret_elem_op")) {
    // atomic_add_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicAdd(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "atomic_addx2_elem_op")) {
    // atomic_addx2_elem_op(dst_ptr, src_ptr[, memory_order]) -> may return prev
    // value
    os << "tl.AtomicAddx2(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "atomic_addx4_elem_op")) {
    // atomic_addx4_elem_op(dst_ptr, src_ptr[, memory_order]) -> may return prev
    // value
    os << "tl.AtomicAddx4(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "atomic_load_elem_op")) {
    // atomic_load_elem_op(src_ptr, memory_order) -> returns loaded value
    os << "tl.AtomicLoad(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "atomic_store_elem_op")) {
    // atomic_store_elem_op(dst_ptr, value, memory_order)
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string value = PrintExpr_(op->args[1]);
    std::string memory_order = PrintExpr_(op->args[2]);
    this->PrintIndent();
    this->stream << "tl.AtomicStore(" << dst_ptr << ", " << value << ", "
                 << memory_order << ")\n";
  } else if (PythonIsOp(op->op, "atomic_or_elem_op")) {
    // atomic_or_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicOr(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr_(op->args[2]);
    }
    this->stream << ")\n";
  } else if (PythonIsOp(op->op, "atomic_max_elem_op")) {
    // atomic_max_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicMax(" << dst_ptr << ", " << src_value << ")\n";
  } else if (PythonIsOp(op->op, "atomic_max_ret_elem_op")) {
    // atomic_max_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicMaxRet(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "atomic_min_elem_op")) {
    // atomic_min_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr_(op->args[0]);
    std::string src_value = PrintExpr_(op->args[1]);
    this->PrintIndent();
    this->stream << "tl.AtomicMin(" << dst_ptr << ", " << src_value << ")\n";
  } else if (PythonIsOp(op->op, "atomic_min_ret_elem_op")) {
    // atomic_min_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "tl.AtomicMinRet(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
  } else if (PythonIsOp(op->op, "shift_right")) {
    // CuTeDSL type promotion fix: Int8 >> 4 returns Int32 in CuTeDSL,
    // but TIR expects result type to match operand type. Wrap in explicit
    // type conversion to match CUDA behavior.
    ICHECK_EQ(op->args.size(), 2U);
    PrimType result_dtype = PythonDType(GetRef<Expr>(op));
    std::string lhs = PrintExpr_(op->args[0]);
    std::string rhs = PrintExpr_(op->args[1]);
    PrintType(result_dtype, os);
    os << "((" << lhs << " >> " << rhs << "))";
  } else if (PythonIsOp(op->op, "shift_left")) {
    // Same fix for shift_left
    ICHECK_EQ(op->args.size(), 2U);
    PrimType result_dtype = PythonDType(GetRef<Expr>(op));
    std::string lhs = PrintExpr_(op->args[0]);
    std::string rhs = PrintExpr_(op->args[1]);
    PrintType(result_dtype, os);
    os << "((" << lhs << " << " << rhs << "))";
  } else {
    CodeGenTileLangPY::Dispatch_(op, os);
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const SelectNode *op,
                                       std::ostream &os) { // NOLINT(*)
  // Emit Python ternary: (true_val if cond else false_val).
  // This yields ArithValue which is fine in expression contexts (range(),
  // etc.). When used as a BufferStore value that requires TensorSSA, the store
  // handler will wrap it with tl.where() at statement level.
  std::string cond = PrintExpr_(op->condition);
  std::string t = PrintExpr_(op->true_value);
  std::string f = PrintExpr_(op->false_value);
  auto cast_branch = [&](std::string value, PrimType branch_dtype) {
    PrimType result_dtype = PythonDType(GetRef<Expr>(op));
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
  t = cast_branch(t, PythonDType(op->true_value));
  f = cast_branch(f, PythonDType(op->false_value));
  os << "(" << t << " if " << cond << " else " << f << ")";
}

void CodeGenTileLangCuTeDSL::Dispatch_(const ShuffleNode *op,
                                       std::ostream &os) { // NOLINT(*)
  std::vector<std::string> concat_vec;

  for (const PrimExpr &vec : op->vectors) {
    std::string vec_value = PrintExpr_(vec);
    if (PythonDType(vec).IsScalar()) {
      concat_vec.push_back(vec_value);
    } else {
      std::string vec_id = SSAGetID(vec_value, PythonDType(vec));
      for (int i = 0; i < PythonDType(vec).lanes(); ++i) {
        std::ostringstream elem;
        PrintVecElemLoad_(vec_id, PythonDType(vec), i, elem);
        concat_vec.push_back(elem.str());
      }
    }
  }

  if (op->indices.size() == 1) {
    ICHECK(op->indices[0]->IsInstance<IntImmNode>())
        << "ShuffleNode indices must be constants at codegen time, got "
        << op->indices[0];
    int64_t idx = tl::GetIntAnnotation(op->indices[0]).value();
    ICHECK_GE(idx, 0);
    ICHECK_LT(idx, static_cast<int64_t>(concat_vec.size()));
    os << concat_vec[idx];
    return;
  }

  std::string sret = name_supply_->FreshName("_shuffle");
  PrintIndent();
  stream << sret << " = tl.make_rmem_tensor((" << op->indices.size() << ",), ";
  PrintType(PythonDType(GetRef<Expr>(op)).WithLanes(1), stream);
  stream << ")\n";

  for (size_t i = 0; i < op->indices.size(); ++i) {
    ICHECK(op->indices[i]->IsInstance<IntImmNode>())
        << "ShuffleNode indices must be constants at codegen time, got "
        << op->indices[i];
    int64_t idx = tl::GetIntAnnotation(op->indices[i]).value();
    ICHECK_GE(idx, 0);
    ICHECK_LT(idx, static_cast<int64_t>(concat_vec.size()));
    PrintVecElemStore_(sret, PythonDType(GetRef<Expr>(op)), static_cast<int>(i),
                       concat_vec[idx]);
  }
  os << sret << ".load()";
}

void CodeGenTileLangCuTeDSL::Dispatch_(const TensorLoadNode *op,
                                       std::ostream &os) { // NOLINT(*)
  ICHECK_GE(op->indices.size(), 1U)
      << "Buffer load must have at least one index.";

  PrimType value_dtype = PythonDType(GetRef<Expr>(op));
  PrimExpr index =
      LinearizeBufferIndices_(op->source.as_or_throw<TensorVar>(), op->indices);
  Var buffer_var = op->source.as_or_throw<TensorVar>();
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
    PrimExpr scalar_base(0);
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
      stream << aligned_rmem << ".store(tl.make_tensor_at_offset(" << ptr_str
             << ", " << PrintExpr_(scalar_base) << ", (" << value_lanes
             << ",)).load())\n";
    }

    // Return the rmem tensor (not .load()) so downstream code uses
    // rmem element access instead of MLIR vector extractelement.
    os << aligned_rmem;
  } else if (value_lanes == element_dtype.lanes()) {
    std::string ref =
        GetBufferRef_(value_dtype, op->source.as_or_throw<TensorVar>(), index);
    // Check if this is a barrier buffer - barrier pointers don't need .load()
    std::string scope;
    if (alloc_storage_scope_.count(buffer_var.get())) {
      scope = alloc_storage_scope_.at(buffer_var.get());
    }
    if (ref.back() == ')' && scope != "shared.barrier" &&
        scope != "shared.cluster_barrier") {
      ref += ".load()";
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
      std::string ref = GetBufferRef_(
          value_dtype, op->source.as_or_throw<TensorVar>(), base.Eval());
      if (ref.back() == ')') {
        ref += ".load()";
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
      const RampNode *ramp = index.as<RampNode>();
      ICHECK(ramp)
          << "Expected Ramp index for vectorized non-contiguous access";
      for (int i = 0; i < value_lanes; ++i) {
        auto idx_expr =
            sym::Analyzer()->Simplify(ramp->base + ramp->stride * i);

        PrintIndent();
        stream << sret << "[" << i << "] = "
               << GetBufferRef_(element_dtype,
                                op->source.as_or_throw<TensorVar>(), idx_expr)
               << "\n";
      }
      os << sret << ".load()";
    }
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const TensorStoreNode *op) {
  ICHECK_GE(op->indices.size(), 1U)
      << "Buffer store must have at least one index.";

  PrimType value_dtype = PythonDType(op->value);
  PrimType element_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  PrimExpr index_expr =
      LinearizeBufferIndices_(op->dest.as_or_throw<TensorVar>(), op->indices);
  Var buffer_var = op->dest.as_or_throw<TensorVar>();

  // Pre-compute Select/if_then_else as tl.where() at statement level.
  // Python ternary (ArithValue) is rejected by CuTeDSL .store(); tl.where()
  // produces TensorSSA which works in all store paths (.store(), vec store,
  // etc.). Also handles Cast(Select/if_then_else) — the cast is applied via
  // .to() on the tl.where result.
  bool value_is_conditional = false;
  Expr cond_expr = PrimExpr(0), true_expr = PrimExpr(0),
       false_expr = PrimExpr(0);
  PrimType cast_to_dtype =
      PrimType::Void(); // non-void if a Cast wraps the conditional
  // Helper to detect Select/if_then_else in a PrimExpr.
  auto detect_conditional = [&](const Expr &expr) {
    if (auto sel = expr.as<SelectNode>()) {
      value_is_conditional = true;
      cond_expr = sel->condition;
      true_expr = sel->true_value;
      false_expr = sel->false_value;
    } else if (auto call = expr.as<CallNode>();
               call && PythonIsOp(call->op, "if_then_else")) {
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
        cast_to_dtype = PythonDType(GetRef<Expr>(cast_node));
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
    const bool true_has_buffer_load = ContainsBufferLoad(true_expr);
    const bool false_has_buffer_load = ContainsBufferLoad(false_expr);
    const bool is_subbyte_vector_conditional =
        conditional_dtype.WithLanes(1).bits() < 8 &&
        conditional_dtype.lanes() > 1;
    if ((true_is_zero_ast && false_has_buffer_load) ||
        (false_is_zero_ast && true_has_buffer_load)) {
      const bool data_is_true_branch = false_is_zero_ast;
      Expr data_expr = data_is_true_branch ? true_expr : false_expr;
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
          if (elem_dtype.MatchesElementType(kDLBool, 1)) {
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
      PrimType data_dtype = PythonDType(data_expr);
      std::string data;
      if (cast_to_dtype.bits() > 0 && data_dtype != conditional_dtype) {
        if (conditional_dtype.IsScalar()) {
          data = PrintExpr_(
              prim::Cast(conditional_dtype, data_expr.as_or_throw<PrimExpr>()));
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
      auto as_tsa = [this](const Expr &e, int n,
                           PrimType elem_dt) -> std::string {
        std::string s = PrintExpr_(e);
        if (n == 0)
          n = 1;
        if (EndsWithLoad(s))
          return s;
        std::string var = name_supply_->FreshName("_tsa");
        PrintIndent();
        if (elem_dt.MatchesElementType(kDLBool, 1)) {
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
      PrimType true_ty = PythonDType(true_expr).WithLanes(1);
      PrimType false_ty = PythonDType(false_expr).WithLanes(1);
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
    value_str = SSAGetID(value_str, value_dtype);

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
      PrimExpr scalar_base(0);
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
      PrimExpr scalar_base(0);
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
    std::string ref = GetBufferRef_(
        value_dtype, op->dest.as_or_throw<TensorVar>(), index_expr);

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
          !op->value.as<TensorLoadNode>() &&
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
                 !op->value.as<TensorLoadNode>()) {
        if (store_rhs.size() < 7 ||
            store_rhs.compare(store_rhs.size() - 7, 7, ".load()") != 0) {
          store_rhs = "tl.make_filled_tensor((1,), " + store_rhs + ").load()";
        }
      }
      PrintIndent();
      stream << ref << ".store(" << RemoveOutermostParentheses(store_rhs)
             << ")\n";
    }
  } else {
    bool is_contiguous = false;
    tvm::PVar<PrimExpr> base;
    if (tvm::ramp(base, 1, value_lanes / element_dtype.lanes())
            .Match(index_expr)) {
      is_contiguous = true;
    }

    if (is_contiguous) {
      PrintVecStore_(op->dest.as_or_throw<TensorVar>(), value_dtype,
                     base.Eval(), value_str);
    } else {
      ICHECK(element_dtype.IsScalar())
          << "buffer element type for non-contiguous store must be scalar "
             "currently";

      // store elements separately
      value_str = SSAGetID(value_str, element_dtype);
      for (int i = 0; i < value_lanes; ++i) {
        const RampNode *ramp = index_expr.as<RampNode>();
        ICHECK(ramp);
        auto idx_expr =
            sym::Analyzer()->Simplify(ramp->base + ramp->stride * i);

        PrintIndent();
        stream << GetBufferRef_(element_dtype,
                                op->dest.as_or_throw<TensorVar>(), idx_expr)
               << " = ";
        PrintVecElemLoad_(value_str, value_dtype, i, stream);
        stream << "\n";
      }
    }
  }
}

void CodeGenTileLangCuTeDSL::Dispatch_(const BindNode *op) {
  if (const auto *call = op->value.as<CallNode>();
      call && call->op.same_as(tirx::decl_tensor_op())) {
    auto tensor = op->var.as_or_throw<TensorVar>();
    PrimType element_type = tensor.dtype().WithLanes(1);
    std::string pointer = GetVarPtr_(call->args[0]);
    std::string scope = tensor.scope().name();
    RegisterHandleType_(tensor.get(), element_type);
    alloc_storage_scope_[tensor.get()] = scope;
    PrintIndent();
    stream << AllocVarID(tensor.get()) << " = ";
    if (scope == "shared.barrier" || scope == "shared.cluster_barrier") {
      raw_pointer_vars_.insert(tensor.get());
      stream << "tl.recast_ptr(" << pointer
             << ", dtype=" << DTypeToString(element_type) << ")\n";
    } else {
      stream << "tl.make_tensor_at_offset(tl.recast_ptr(" << pointer
             << ", dtype=" << DTypeToString(element_type) << "), 0, ("
             << PrintExpr_(PythonTensorSize(tensor) * tensor.dtype().lanes())
             << ",))\n";
    }
    return;
  }
  if (auto *ptr_type = op->var->ty.as<PtrTypeNode>()) {
    if (auto *prim_type = ptr_type->element_type.as<PrimTypeNode>()) {
      if (const CallNode *call = op->value.as<CallNode>();
          call && PythonIsOp(call->op, "reinterpret") &&
          call->args.size() == 1U && PythonIsPointer(op->value)) {
        RegisterHandleType_(op->var.get(), GetRef<PrimType>(prim_type));
        raw_pointer_vars_.insert(op->var.get());

        std::string address_space = "tl.cute.AddressSpace.generic";
        const std::string scope = ptr_type->scope.name();
        if (scope == "global" || scope.empty()) {
          address_space = "tl.cute.AddressSpace.gmem";
        } else if (scope == "shared" || scope.rfind("shared.", 0) == 0) {
          address_space = "tl.cute.AddressSpace.smem";
        } else if (scope == "local" || scope.rfind("local.", 0) == 0) {
          address_space = "tl.cute.AddressSpace.rmem";
        }

        PrintIndent();
        stream << AllocVarID(op->var.get()) << " = tl.cute.make_ptr(";
        PrintType(GetRef<PrimType>(prim_type), stream);
        stream << ", " << PrintExpr_(call->args[0]) << ", " << address_space
               << ")\n";
        return;
      }
      std::string pointer = GetVarPtr_(op->value);
      RegisterHandleType_(op->var.get(), GetRef<PrimType>(prim_type));
      raw_pointer_vars_.insert(op->var.get());
      PrintIndent();
      stream << AllocVarID(op->var.get()) << " = " << pointer << "\n";
      return;
    }
  }

  if (IsZeroTensorLike(op->value, &zero_like_vars_)) {
    zero_like_vars_.insert(op->var.get());
  }
  CodeGenTileLangPY::Dispatch_(op);
}

void CodeGenTileLangCuTeDSL::PrintAllocTensor_(const BindNode *op,
                                               const CallNode *allocation) {
  auto tensor = op->var.as_or_throw<TensorVar>();
  auto annotations = allocation->attrs.value_or(DictAttrs(Map<String, Any>{}))
                         .as_or_throw<DictAttrs>()
                         ->dict;
  PrimType alloc_dtype = tensor.dtype();
  std::string vid = AllocVarID(tensor.get());
  PrintIndent();
  std::string scope = PythonStorageScope(tensor);
  alloc_storage_scope_[tensor.get()] = scope;

  if (scope == "local.descriptor.wgmma") {
    stream << vid << " = tl.GmmaDescriptor()\n";
  } else if (scope == "local.descriptor.tcgen05_smem") {
    stream << vid << " = tl.Tcgen05SmemDescriptor()\n";
  } else if (scope == "local.descriptor.tcgen05_instr") {
    stream << vid << " = 0\n";
  } else if (scope == "shared.dyn") {
    stream << vid << " = tl.make_tensor(tl.get_dyn_smem(";
    PrintType(alloc_dtype.WithLanes(1), stream);
    stream << ", alignment=1024), ("
           << PrintExpr_(PythonTensorSize(tensor) * alloc_dtype.lanes())
           << ",))\n";
  } else {
    std::optional<int64_t> opt_size = tl::GetConstInt(PythonTensorSize(tensor));
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
      auto init_it = annotations.find(tl::attr::kLocalVarInit);
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

  PrimType registered_dtype =
      ((scope == "shared" || scope == "shared.dyn" || scope == "local") &&
       alloc_dtype.IsFixedLengthVector())
          ? alloc_dtype.WithLanes(1)
          : alloc_dtype;
  RegisterHandleType_(tensor.get(), registered_dtype);
}

void CodeGenTileLangCuTeDSL::Dispatch_(const RegionStmtNode *op) {
  if (op->op.same_as(tirx::launch_thread_op())) {
    BindThreadIndex_(op->body_params[0].as_or_throw<PrimVar>(),
                     op->args[0].as_or_throw<StringImm>()->value);
    PrintStmt_(op->body);
    return;
  }
  if (PythonIsOp(op->op, "thread_binding")) {
    auto attributes = op->attrs.as_or_throw<DictAttrs>()->dict;
    auto tag = attributes.at("thread_tag").cast<String>();
    BindThreadIndex_(op->body_params[0].as_or_throw<PrimVar>(), tag);
    PrintStmt_(op->body);
    return;
  }
  if (auto swizzle = op->attrs.as_or_throw<DictAttrs>()->dict.Get(
          "threadblock_swizzle_pattern")) {
    auto fields = swizzle.value().cast<tvm::Tuple>()->fields;
    auto name = fields[0].as_or_throw<StringImm>()->value;
    auto panel = tl::GetConstInt(fields[1]);
    TVM_FFI_CHECK(panel && *panel > 0, ValueError)
        << "Invalid threadblock swizzle panel";
    PrintIndent();
    stream << "blockIdx = tl." << name << "(" << *panel << ")\n";
  }
  CodeGenTileLangPY::Dispatch_(op);
}

void CodeGenTileLangCuTeDSL::Dispatch_(const ForNode *op) {
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
    PrintExpr_(op->begin, stream);
    stream << ", ";
    PrintExpr_(op->end, stream);
    stream << ", ";
    PrintExpr_(op->step, stream);
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

  if (op->kind != ForKind::kUnrolled) {
    CodeGenTileLangPY::Dispatch_(op);
    return;
  }

  auto start_expr = sym::Analyzer()->Simplify(op->begin);
  auto stop_expr = sym::Analyzer()->Simplify((op->end - op->begin) + op->begin);
  std::string unroll_factor;
  if (auto it = unroll_factor_.find(op->var.get());
      it != unroll_factor_.end()) {
    unroll_factor = PrintExpr_(it->second);
  }
  bool use_range_constexpr =
      unroll_factor.empty() &&
      tl::GetConstInt((op->end - op->begin)).has_value() &&
      *tl::GetConstInt((op->end - op->begin)) <= LOOP_UNROLL_THRESHOLD;
  PrintIndent();
  std::string vid = AllocVarID(op->var.get());
  stream << "for " << vid << " in cutlass.range";
  if (use_range_constexpr) {
    stream << "_constexpr";
  }
  stream << "(";
  PrintExpr_(start_expr, stream);
  stream << ", ";
  PrintExpr_(stop_expr, stream);
  stream << ", ";
  PrintExpr_(op->step, stream);
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

void CodeGenTileLangCuTeDSL::Dispatch_(const BreakNode *op) {
  TVM_FFI_CHECK(in_break_loop_, ValueError)
      << "CuTeDSL break requires a normalized for-loop";
  PrintIndent();
  stream << "_loop_break_" << current_break_id_ << " = 1\n";
  break_emitted_in_seq_ = true;
}

void CodeGenTileLangCuTeDSL::Dispatch_(const SeqStmtNode *op) {
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

void CodeGenTileLangCuTeDSL::Dispatch_(const IfNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(cond) << ":\n";
  int then_scope = BeginScope();
  if (const CallNode *call = op->condition.as<CallNode>();
      call && PythonIsOp(call->op, "tl_shuffle_elect")) {
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

void CodeGenTileLangCuTeDSL::Dispatch_(const EvaluateNode *op) {
  if (tl::GetConstInt(op->value))
    return;
  const CallNode *call = op->value.as<CallNode>();
  if (call && (call->op.same_as(tvm::tl::device_assert()))) {
    std::string cond = RemoveOutermostParentheses(PrintExpr_(call->args[0]));
    PrintIndent();
    stream << "tl.runtime_assert(" << cond << ")\n";
  } else if (call && call->op.same_as(tvm::tl::device_assert_with_msg())) {
    std::string cond = RemoveOutermostParentheses(PrintExpr_(call->args[0]));
    std::string msg_expr = PrintExpr_(call->args[1]);
    PrintIndent();
    stream << "tl.runtime_assert(" << cond << ", " << msg_expr << ")\n";
  } else if (call && call->op.same_as(tvm::tl::sync_warp())) {
    PrintIndent();
    stream << "tl.sync_warp(";
    if (!call->args.empty()) {
      stream << PrintExpr_(call->args[0]);
    }
    stream << ")\n";
  } else if (call && PythonIsOp(call->op, "gpu_storage_sync")) {
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

void CodeGenTileLangCuTeDSL::PrintVecStore_(const TensorVar &buffer, PrimType t,
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
    stream << ref << ".store(" << store_rhs << ")\n";
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
  stream << ref << ".store(" << store_rhs << ")\n";
}

void CodeGenTileLangCuTeDSL::PrintVecBinaryOp_(const std::string &opstr,
                                               PrimType dtype, Expr lhs,
                                               Expr rhs,
                                               std::ostream &os) { // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  PrintIndent();
  stream << sret << " = tl.make_rmem_tensor((" << dtype.lanes() << ",), ";
  PrintType(dtype.WithLanes(1), stream);
  stream << ")\n";

  auto print_operand = [this](const Expr &expr) {
    std::string value = PrintExpr_(expr);
    return ContainsBufferLoad(expr) ? value
                                    : SSAGetID(value, PythonDType(expr));
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
        PrintVecElemLoad_(vlhs, PythonDType(lhs), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad_(vrhs, PythonDType(rhs), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad_(vlhs, PythonDType(lhs), i, value_temp);
        value_temp << opstr;
        PrintVecElemLoad_(vrhs, PythonDType(rhs), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore_(sret, dtype, i, value_temp.str());
    }
  }
  os << sret << ".load()";
}

void CodeGenTileLangCuTeDSL::PrintBinaryExpr_(const std::string &opstr,
                                              PrimType dtype, Expr lhs,
                                              Expr rhs,
                                              std::ostream &os) { // NOLINT(*)
  if (dtype.IsScalar()) {
    CodeGenTileLangPY::PrintBinaryExpr_(opstr, dtype, lhs, rhs, os);
  } else {
    PrintVecBinaryOp_(opstr, dtype, lhs, rhs, os);
  }
}

void CodeGenTileLangCuTeDSL::PrintBinaryIntrinsic_(
    const CallNode *op, const char *opstr,
    std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)).IsScalar()) {
    CodeGenTileLangPY::PrintBinaryIntrinsic_(op, opstr, os);
  } else {
    PrintVecBinaryOp_(opstr, PythonDType(GetRef<Expr>(op)), op->args[0],
                      op->args[1], os);
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
      std::string val = SSAGetID(sarg, PythonDType(args[i]));
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
    if (const TensorLoadNode *load = args[arg_begin].as<TensorLoadNode>()) {
      sargs[0] = GetBufferPtr_(
          load->source.as_or_throw<TensorVar>(),
          LinearizeBufferIndices_(load->source.as_or_throw<TensorVar>(),
                                  load->indices));
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
              PythonDType(
                  args[arg_begin + j - static_cast<size_t>(has_template_arg)]),
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
    const TensorVar &buffer, const ffi::Array<PrimExpr> &indices) {
  return PythonTensorOffset(buffer, indices);
}

std::string CodeGenTileLangCuTeDSL::GetBufferPtr_(const TensorVar &buffer,
                                                  PrimExpr index) {
  const VarNode *buffer_var = buffer.get();
  const std::string vid = GetVarID(buffer_var);

  PrimType buffer_element_dtype = buffer.dtype();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty())
    scope = PythonStorageScope(buffer);
  const bool scalarized_local_vector_buffer =
      (scope == "local" || scope == "shared" || scope == "shared.dyn") &&
      buffer_element_dtype.IsFixedLengthVector();
  // CuTeDSL only supports i1 (Boolean) in rmem; use Uint8 for gmem pointers.
  PrimType effective_dtype = scalarized_local_vector_buffer
                                 ? buffer_element_dtype.WithLanes(1)
                                 : buffer_element_dtype;
  if (buffer_element_dtype.MatchesElementType(kDLBool, 1)) {
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
  } else if (scope.starts_with("local.descriptor")) {
    ICHECK_EQ(tl::GetConstInt(index).value_or(-1), 0);
    return vid;
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
CodeGenTileLangCuTeDSL::GetSubByteBufferPtrAsByte_(const TensorVar &buffer,
                                                   PrimExpr index) {
  PrimType buffer_element_dtype = buffer.dtype();
  int elems_per_byte = 1;
  if (tl::IsFP4UnpackedType(buffer_element_dtype)) {
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

  const VarNode *buffer_var = buffer.get();
  const std::string vid = GetVarID(buffer_var);

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = PythonStorageScope(buffer);
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
  const CallNode *call = expr.as<CallNode>();
  if (call && PythonIsOp(call->op, "address_of")) {
    ICHECK_EQ(call->args.size(), 1U);
    const TensorLoadNode *load = call->args[0].as<TensorLoadNode>();
    if (load &&
        tl::IsFP4UnpackedType(load->source.as_or_throw<TensorVar>().dtype())) {
      ICHECK_EQ(load->indices.size(), 1U)
          << "CodeGenTileLangCuTeDSL only supports flat memory";
      return GetSubByteBufferPtrAsByte_(load->source.as_or_throw<TensorVar>(),
                                        load->indices[0]);
    }
  }
  return PrintExpr_(expr);
}

std::string CodeGenTileLangCuTeDSL::GetVarPtr_(const Expr &expr) {
  if (const auto *load = expr.as<TensorLoadNode>()) {
    auto buffer = load->source.as_or_throw<TensorVar>();
    return GetBufferPtr_(buffer,
                         LinearizeBufferIndices_(buffer, load->indices));
  }
  // For local buffers (rmem tensors), we need to use .iterator to get the
  // pointer since local buffers in CuTeDSL are tensors, not raw pointers
  if (const VarNode *var = expr.as<VarNode>()) {
    if (raw_pointer_vars_.count(var)) {
      return GetVarID(var);
    }
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
// It's the only form that ends with ")". Using this fact, TensorLoadNode will
// add ".load()" and TensorStoreNode will add ".store()".
std::string CodeGenTileLangCuTeDSL::GetBufferRef_(PrimType t,
                                                  const TensorVar &buffer,
                                                  PrimExpr index) {
  const VarNode *buffer_var = buffer.get();
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = PythonStorageScope(buffer);
  }
  if (scope == "local.var" || scope.find("local.descriptor") == 0) {
    return vid;
  }

  PrimType buffer_element_dtype = buffer.dtype();
  const bool scalarized_local_vector_buffer =
      (scope == "local" || scope == "shared" || scope == "shared.dyn") &&
      buffer_element_dtype.IsFixedLengthVector();
  // CuTeDSL only supports i1 (Boolean) in rmem. For gmem/shared bool buffers,
  // use Uint8 instead (matches PyTorch's torch.bool memory layout).
  PrimType effective_dtype = scalarized_local_vector_buffer
                                 ? buffer_element_dtype.WithLanes(1)
                                 : buffer_element_dtype;
  if (buffer_element_dtype.MatchesElementType(kDLBool, 1) && scope != "local" &&
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
  if (const RampNode *ramp = index.as<RampNode>()) {
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
      // for vector data types, ".load()" (added by TensorLoadNode) is needed
      // instead of "[0]"
      if (buffer_element_dtype.IsScalar()) {
        os << "[0]";
      }
      return os.str();
    }
  } else if (scalarized_local_vector_buffer &&
             t == buffer_element_dtype.WithLanes(1)) {
    if (scope == "shared.dyn") {
      return "tl.make_tensor_at_offset(" + ptr_str + ", " + index_str +
             ", (1,), div_by=1)[0]";
    }
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

void CodeGenTileLangCuTeDSL::BindThreadIndex_(const PrimVar &variable,
                                              const std::string &tag) {
  TVM_FFI_CHECK(tag == "threadIdx.x" || tag == "threadIdx.y" ||
                    tag == "threadIdx.z" || tag == "blockIdx.x" ||
                    tag == "blockIdx.y" || tag == "blockIdx.z",
                ValueError)
      << "Unsupported thread binding: " << tag;
  var_idmap_[variable.get()] =
      CastFromTo_(tag, PrimType::Int(32), variable.ty());
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
      ICHECK(PythonDType(args[1]).MatchesCode(kDLInt))
          << "storage_sync barrier_id must be integer type, got "
          << PythonDType(args[1]);
      stream << "tl.sync_thread_partial(" << PrintExpr_(args[1]) << ")\n";
    } else if (args.size() == 3) {
      ICHECK(PythonDType(args[1]).MatchesCode(kDLInt))
          << "storage_sync barrier_id must be integer type, got "
          << PythonDType(args[1]);
      ICHECK(PythonDType(args[2]).MatchesCode(kDLInt))
          << "storage_sync thread_count must be integer type, got "
          << PythonDType(args[2]);
      stream << "tl.sync_thread_partial(" << PrintExpr_(args[1]) << ", "
             << PrintExpr_(args[2]) << ")\n";
    } else {
      LOG(FATAL) << "Invalid number of arguments for storage sync: "
                 << args.size();
    }
  } else if (sync == "cluster") {
    PrintIndent();
    stream << "tl.cluster_sync()\n";
  } else if (sync == "global") {
    LOG(FATAL) << "PrintStorageSync_ for global is not supported for now";
  } else {
    LOG(FATAL) << "Unknown storage sync scope: " << sync;
  }
}

} // namespace codegen
} // namespace tvm
