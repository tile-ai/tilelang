#include "codegen_ascend.h"
#include "ascend/op/nd2nz.h"
#include "ir/analyzer.h"
#include "native_ascend_ir.h"
#include "support/check.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include <tvm/ir/prim/pattern_match.h>

#include "ascend/op/builtin.h"

namespace tvm {
namespace codegen {
using ffi::GetRef;
using prim::IsOne;
using prim::IsZero;
using prim::MakeConst;

namespace {

constexpr const char *kModeMerging = "MODE_MERGING";
constexpr const char *kSimdOpPrefix = "tl.simd.";

bool IsSimdOp(const CallObj *op, std::string *op_name = nullptr) {
  if (auto call_op = op->op.as<Op>()) {
    const std::string &name = call_op.value()->name;
    if (name.rfind(kSimdOpPrefix, 0) == 0) {
      if (op_name != nullptr) {
        *op_name = name;
      }
      return true;
    }
  }
  return false;
}

bool IsSimdMergingCall(const CallObj *op,
                       std::string *intrinsic_name = nullptr) {
  const auto result = op->ty.as<PrimType>();
  if (!result || !result.value().IsVoid() || CodegenOperands(op).empty()) {
    return false;
  }
  const auto *mode = CodegenOperands(op).back().as<StringImmObj>();
  if (mode == nullptr || mode->value != kModeMerging) {
    return false;
  }
  std::string op_name;
  if (!IsSimdOp(op, &op_name)) {
    return false;
  }
  if (intrinsic_name != nullptr) {
    *intrinsic_name = op_name.substr(std::string(kSimdOpPrefix).size());
  }
  return true;
}

// CCE merging overloads validated on dav-3510. Keep the existing C API path
// for other element types, including FP8 and 64-bit vector representations.
bool SupportsNativeSimdMerging(const std::string &name, PrimType dtype) {
  const bool is_integer =
      (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32);
  const bool is_float =
      dtype.MatchesCode(kDLFloat) && (dtype.bits() == 16 || dtype.bits() == 32);
  if (!is_integer && !is_float && !dtype.MatchesElementType(kDLBfloat, 16)) {
    return false;
  }
  if (name == "vdup") {
    // CANN 9.2's scalar BF16 overload merges into an uninitialized temporary
    // instead of the old destination. Vector broadcast uses a different
    // overload.
    return !dtype.MatchesElementType(kDLBfloat, 16);
  }
  if (name == "vabs" || name == "vneg") {
    return (is_integer && dtype.MatchesCode(kDLInt)) || is_float;
  }
  if (name == "vabsdif" || name == "vexp" || name == "vln" || name == "vsqrt" ||
      name == "vaxpy") {
    return is_float;
  }
  if (name == "vrelu") {
    return is_float || (dtype.MatchesCode(kDLInt) && dtype.bits() == 32);
  }
  if (name == "vshl" || name == "vshr" || name == "vshls" || name == "vshrs") {
    return is_integer;
  }
  if (name == "vdiv" || name == "vcadd" || name == "vcmax" || name == "vcmin") {
    return (is_integer && dtype.bits() >= 16) || is_float;
  }
  if (name == "vmul" || name == "vmuls") {
    return !is_integer || dtype.bits() >= 16;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Per-op SFU precision resolution.
//
// Priority (high -> low):
//   1. per-op "precision" annotation (Python `precision=` kwarg)
//   2. `fallback` (legacy: fast_math for vdiv; bare SFU otherwise)
//
// The Python side (tilelang/ascend/language/simd.py) is the single source
// of truth for alias resolution: string aliases are normalized to integer
// codes (0=hw, 1=exact, 2=ftz_false) before they reach codegen, so no
// alias table lives here (mirrors the l2_cache_ctrl pattern).
// ---------------------------------------------------------------------------
enum class SfuPrecision { kHw, kExact, kKeepSub };

static SfuPrecision PrecisionFromCode(int code) {
  if (code == 1)
    return SfuPrecision::kExact;
  if (code == 2)
    return SfuPrecision::kKeepSub;
  return SfuPrecision::kHw;
}

// MODE_MERGING calls are legalized to void calls, so use the explicit result
// dtype supplied by the caller rather than reading op->dtype here.
SfuPrecision ResolveSfuPrecision(const Call &op, PrimType result_dtype,
                                 SfuPrecision fallback) {
  // Precise paths (vdiv_0ulp_ftz_true, *_ftz_false wrappers) are float32-only:
  // non-fp32 keeps the hardware instruction regardless of annotations
  // (historical UsePreciseVdiv contract: "non-fp32 always uses hardware").
  if (!result_dtype.MatchesCode(kDLFloat) || result_dtype.bits() != 32) {
    return fallback;
  }
  // Per-op "precision" annotation (int code from the Python side,
  // normalized from the string aliases in tilelang/ascend/language/simd.py
  // -- no alias table here, mirroring the l2_cache_ctrl pattern).
  if (auto p = CodegenCallAnnotations(op.get()).Get("precision")) {
    return PrecisionFromCode(AscendAnnotationInt(p.value()));
  }
  return fallback;
}

bool IsAscendWarpReduceDType(PrimType dtype) {
  return dtype.IsScalar() &&
         ((dtype.MatchesCode(kDLFloat) &&
           (dtype.bits() == 16 || dtype.bits() == 32)) ||
          ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
           dtype.bits() == 32));
}

static std::string GetAscendScopeQualifier(const std::string &scope) {
  // Keep scalar accesses volatile for the vendor compiler. SS accesses in
  // TIR follow ordinary Buffer optimization and synchronization semantics.
  if (scope == "shared.ss" || scope == "shared.ss.dyn")
    return "volatile __ssbuf__";
  if (scope == "shared" || scope == "shared.dyn")
    return "__ubuf__";
  if (scope == "shared.l0a" || scope == "shared.l0a.dyn")
    return "__ca__";
  if (scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return "__cb__";
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return "__cc__";
  if (scope == "shared.l1" || scope == "shared.l1.dyn")
    return "__cbuf__";
  return "";
}

std::string FixIndent(const std::string &body) {
  size_t min_indent = std::string::npos;
  std::istringstream iss(body);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.empty()) {
      continue;
    }
    size_t pos = line.find_first_not_of(' ');
    if (pos != std::string::npos && pos < min_indent) {
      min_indent = pos;
    }
  }
  if (min_indent == std::string::npos || min_indent <= 2) {
    return body;
  }
  size_t strip = min_indent - 2;
  iss.clear();
  iss.seekg(0);
  std::ostringstream oss;
  while (std::getline(iss, line)) {
    if (line.size() >= strip &&
        line.find_first_not_of(' ') != std::string::npos) {
      oss << line.substr(strip) << "\n";
    } else {
      oss << line << "\n";
    }
  }
  return oss.str();
}

// Map a TVM DataType to the Ascend CCE vector type suffix (e.g. Float32 ->
// "f32").
std::string CCEVectorSuffix(PrimType dtype) {
  if (dtype.MatchesElementType(kDLFloat8_e4m3, 8) ||
      dtype.MatchesElementType(kDLFloat8_e4m3fn, 8))
    return "f8e4m3";
  if (dtype.MatchesElementType(kDLFloat8_e5m2, 8))
    return "f8e5m2";
  if (dtype.MatchesElementType(kDLFloat8_e8m0fnu, 8))
    return "f8e8m0";
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4))
    return "f4e2m1x2";
  if (dtype.MatchesCode(kDLFloat)) {
    if (dtype.bits() == 16)
      return "f16";
    if (dtype.bits() == 32)
      return "f32";
    TVM_FFI_THROW(InternalError)
        << "Unsupported Ascend SIMD floating-point element type " << dtype;
  }
  if (dtype.MatchesElementType(kDLBfloat, 16))
    return "bf16";
  if (dtype.MatchesCode(kDLInt) && (dtype.bits() == 8 || dtype.bits() == 16 ||
                                    dtype.bits() == 32 || dtype.bits() == 64))
    return "s" + std::to_string(dtype.bits());
  if (dtype.MatchesCode(kDLUInt) && (dtype.bits() == 8 || dtype.bits() == 16 ||
                                     dtype.bits() == 32 || dtype.bits() == 64))
    return "u" + std::to_string(dtype.bits());
  TVM_FFI_THROW(InternalError)
      << "Unsupported Ascend SIMD vector element type " << dtype;
  return "";
}

// Get element width in bits for choosing b8/b16/b32 predicate/load/store
// variants.
int CCEElemWidthBits(PrimType dtype) {
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4))
    return 8;
  return dtype.bits();
}

// Map DataType to the C++ type name used in __ubuf__ pointer casts.
std::string CCEUBufType(PrimType dtype) {
  if (dtype.MatchesElementType(kDLFloat8_e4m3, 8) ||
      dtype.MatchesElementType(kDLFloat8_e4m3fn, 8))
    return "float8_e4m3_t";
  if (dtype.MatchesElementType(kDLFloat8_e5m2, 8))
    return "float8_e5m2_t";
  if (dtype.MatchesElementType(kDLFloat8_e8m0fnu, 8))
    return "float8_e8m0_t";
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4))
    return "float4_e2m1x2_t";
  if (dtype.MatchesCode(kDLFloat))
    return (dtype.bits() == 16) ? "half" : "float";
  if (dtype.MatchesElementType(kDLBfloat, 16))
    return "bfloat16_t";
  if (dtype.MatchesCode(kDLInt))
    return "int" + std::to_string(dtype.bits()) + "_t";
  if (dtype.MatchesCode(kDLUInt))
    return "uint" + std::to_string(dtype.bits()) + "_t";
  return "float";
}

struct PipePair {
  std::string source;
  std::string destination;
};

PipePair GetPipePair(const std::string &hard_event) {
  static const std::unordered_set<std::string> kValidPipes = {
      "MTE1", "MTE2", "MTE3", "M", "V", "S", "FIX"};
  auto sep = hard_event.find('_');
  ICHECK(sep != std::string::npos) << "HardEvent must use _ split.";
  std::string src = hard_event.substr(0, sep);
  std::string dst = hard_event.substr(sep + 1);
  ICHECK(kValidPipes.count(src) && kValidPipes.count(dst))
      << "HardEvent: " << hard_event << " is not supported.";
  return {"PIPE_" + src, "PIPE_" + dst};
}

const char *GetCrossCoreSyncScope(int64_t mode_id) {
  switch (mode_id) {
  case 0:
    return "inter";
  case 1:
    return "subblock";
  case 2:
    return "block";
  case 4:
    return "intra";
  default:
    TVM_FFI_THROW(InternalError)
        << "Ascend cross-core sync supports only modes 0, 1, 2, and 4, "
        << "but got " << mode_id;
    return nullptr;
  }
}

int64_t GetIntImmArg(const CallObj *op, size_t index, const char *intrinsic,
                     const char *argument_name) {
  ICHECK_LT(index, CodegenOperands(op).size())
      << intrinsic << " is missing argument " << index << " (" << argument_name
      << ")";
  const auto *value = CodegenOperands(op)[index].as<IntImmObj>();
  ICHECK(value) << intrinsic << " argument " << index << " (" << argument_name
                << ") must be an IntImm, but got "
                << CodegenOperands(op)[index];
  return CodegenInt(value);
}

std::string GetStringImmArg(const CallObj *op, size_t index,
                            const char *intrinsic, const char *argument_name) {
  ICHECK_LT(index, CodegenOperands(op).size())
      << intrinsic << " is missing argument " << index << " (" << argument_name
      << ")";
  const auto *value = CodegenOperands(op)[index].as<StringImmObj>();
  ICHECK(value) << intrinsic << " argument " << index << " (" << argument_name
                << ") must be a StringImm, but got "
                << CodegenOperands(op)[index];
  return value->value;
}
} // namespace

CodeGenTileLangAscend::VFModeScope::VFModeScope(CodeGenTileLangAscend *codegen,
                                                VFMode mode)
    : codegen_(codegen), previous_mode_(codegen->vf_mode_) {
  if (previous_mode_ != VFMode::kNone) {
    TVM_FFI_THROW(InternalError)
        << "Nested Ascend VF blocks are not supported: cannot enter "
        << codegen_->VFModeName(mode) << " while emitting "
        << codegen_->VFModeName(previous_mode_);
  }
  codegen_->vf_mode_ = mode;
}

CodeGenTileLangAscend::VFModeScope::~VFModeScope() {
  codegen_->vf_mode_ = previous_mode_;
}

void CodeGenTileLangAscend::Init(bool output_ssa) {
  CodeGenC::Init(output_ssa);
  auto pass_ctx = tvm::transform::PassContext::Current();
  enable_fast_math_ =
      pass_ctx->GetConfig<bool>("tl.enable_fast_math").value_or(false);
  preamble_stream_ << "#include <tl_templates/ascend/common.h>\n";
  preamble_stream_ << "#include <tl_templates/ascend/debug.h>\n";
}

void CodeGenTileLangAscend::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" ";
  switch (kernel_mode_) {
  case AscendKernelMode::kMix:
    // dav-3510's direct AIC-to-AIV data paths require the physical 1:2 group.
    // mix_aiv_count_ controls how many of those AIVs execute the vector body.
    os << "__global__ __mix__(1, 2) ";
    break;
  case AscendKernelMode::kVector:
    os << "__global__ __vector__ ";
    break;
  case AscendKernelMode::kCube:
    os << "__global__ __cube__ ";
    break;
  }
}

void CodeGenTileLangAscend::PrintCCEVectorSuffix(const PrimType &t,
                                                 std::ostream &os) {
  os << CCEVectorSuffix(t);
}

static std::string GetAscendFP8Type(PrimType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.IsScalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "_2";
  } else if (lanes == 4) {
    vec = "_4";
  } else if (lanes == 8) {
    vec = "_8";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2, 4, 8) for "
           "Ascend fp8";
  }
  if (type.MatchesElementType(kDLFloat8_e4m3, 8) ||
      type.MatchesElementType(kDLFloat8_e4m3fn, 8)) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.MatchesElementType(kDLFloat8_e5m2, 8)) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.MatchesElementType(kDLFloat8_e8m0fnu, 8)) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported Ascend fp8 type " << type;
  }
  return stream.str();
}

static std::string GetAscendFP8ScalarValueType(PrimType type) {
  if (type.MatchesElementType(kDLFloat8_e4m3, 8) ||
      type.MatchesElementType(kDLFloat8_e4m3fn, 8)) {
    return "tl::float_e4m3_t";
  }
  if (type.MatchesElementType(kDLFloat8_e5m2, 8)) {
    return "tl::float_e5m2_t";
  }
  TVM_FFI_THROW(InternalError)
      << "Unsupported Ascend scalar fp8 conversion type " << type;
  return "";
}

const char *CodeGenTileLangAscend::VFModeName(VFMode mode) const {
  switch (mode) {
  case VFMode::kNone:
    return "outside VF blocks";
  case VFMode::kSimt:
    return "SimtVF";
  case VFMode::kSimd:
    return "SimdVF";
  }
  return "an unknown VF mode";
}

void CodeGenTileLangAscend::ValidateSimdVectorType(
    PrimType dtype, const std::string &context) const {
  if (dtype.lanes() <= 1) {
    TVM_FFI_THROW(InternalError)
        << context << " expects an Ascend SIMD vector register, got " << dtype;
  }
  if (dtype.MatchesCode(kDLBool)) {
    if (dtype.lanes() != 256) {
      TVM_FFI_THROW(InternalError)
          << context << " has invalid SIMD predicate type " << dtype
          << ": SimdVF predicates must be boolx256";
    }
    return;
  }

  PrimType element_dtype = dtype.WithLanes(1);
  CCEVectorSuffix(element_dtype);
  int64_t register_bits =
      static_cast<int64_t>(element_dtype.bits()) * dtype.lanes();
  if (register_bits != 2048) {
    int expected_lanes = 2048 / element_dtype.bits();
    TVM_FFI_THROW(InternalError)
        << context << " has invalid SIMD register type " << dtype << ": got "
        << register_bits
        << " bits, but SimdVF requires one full 2048-bit register ("
        << element_dtype << "x" << expected_lanes << ")";
  }
}

void CodeGenTileLangAscend::RejectGenericVectorInSimd(const char *op_name,
                                                      PrimType dtype) const {
  if (IsInsideSimdVF() && dtype.lanes() > 1) {
    TVM_FFI_THROW(InternalError)
        << "Generic vector " << op_name << " with dtype " << dtype
        << " cannot be emitted inside SimdVF; use an explicit "
           "T.simd.* operation or fix the preceding SIMD lowering";
  }
}

void CodeGenTileLangAscend::ValidateGenericVectorType(const char *op_name,
                                                      PrimType dtype) {
  RejectGenericVectorInSimd(op_name, dtype);
  std::ostringstream type;
  PrintType(dtype, type);
}

void CodeGenTileLangAscend::PrintType(const PrimType &t, std::ostream &os) {
  if (t.IsVoid()) {
    os << "void";
    return;
  }
  int lanes = t.lanes();
  if (t.MatchesCode(kDLOpaqueHandle)) {
    ICHECK_EQ(lanes, 1) << "does not support vector types for handles";
    os << "void*";
    return;
  }

  if (IsInsideSimdVF() && lanes > 1) {
    ValidateSimdVectorType(t, "SimdVF type");
    os << (t.MatchesCode(kDLBool)
               ? "vector_bool"
               : "vector_" + CCEVectorSuffix(t.WithLanes(1)));
    return;
  }
  if (lanes > 1) {
    // SIMT vector comparisons use one ushort per predicate lane. Wider
    // predicates and 16-bit values use uint carriers.
    if (t.MatchesCode(kDLBool) && lanes <= 8) {
      ICHECK(lanes <= 4 || lanes % 2 == 0)
          << "Ascend wide vector bool type requires an even lane count, got "
          << t;
      os << (lanes <= 4 ? "ushort" + std::to_string(lanes)
                        : "uint" + std::to_string(lanes / 2));
      return;
    }
    if ((t.MatchesElementType(kDLFloat, 16) ||
         t.MatchesElementType(kDLBfloat, 16)) &&
        lanes <= 8 && lanes % 2 == 0) {
      os << "uint" << lanes / 2;
      return;
    }
    if ((t.MatchesCode(kDLFloat) && t.bits() == 32) ||
        ((t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) && t.bits() == 32)) {
      if (lanes == 2 || lanes == 4) {
        os << (t.MatchesCode(kDLFloat) ? "float"
               : t.MatchesCode(kDLInt) ? "int"
                                       : "uint")
           << lanes;
        return;
      }
    }
    if (CodegenIsFloat8(t) && (lanes == 2 || lanes == 4 || lanes == 8)) {
      os << GetAscendFP8Type(t);
      return;
    }
    if (t.MatchesElementType(kDLFloat4_e2m1fn, 4) && lanes == 2) {
      os << "float4_e2m1x2_t";
      return;
    }
    TVM_FFI_THROW(InternalError)
        << VFModeName(vf_mode_) << " cannot convert vector type " << t
        << " to an Ascend type";
  }

  if (t.IsVoid()) {
    os << "void";
    return;
  }
  if (t == PrimType::Bool()) {
    os << "bool";
    return;
  }

  if (t.MatchesCode(kDLFloat)) {
    if (t.bits() == 16) {
      os << "half";
    } else if (t.bits() == 32) {
      os << "float";
    } else if (t.bits() == 64) {
      os << "double";
    } else {
      TVM_FFI_THROW(InternalError)
          << "Cannot convert scalar type " << t << " to Ascend type";
    }
    return;
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    os << "bfloat16_t";
    return;
  } else if (CodegenIsFloat8(t)) {
    os << (IsInsideSimtVF() ? GetAscendFP8ScalarValueType(t)
                            : GetAscendFP8Type(t));
    return;
  } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    os << "float4_e2m1x2_t";
    return;
  } else if (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) {
    if (t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64) {
      os << (t.MatchesCode(kDLInt) ? "int" : "uint") << t.bits() << "_t";
      return;
    }
  }
  TVM_FFI_THROW(InternalError)
      << "Cannot convert scalar type " << t << " to Ascend type";
}

void CodeGenTileLangAscend::AddFunction(const GlobalVar &gvar,
                                        const tirx::Function &f) {
  ICHECK(IsOutsideVF())
      << "Ascend VF mode leaked across PrimFunc code generation";
  has_gemm_l0_ = false;
  has_gemm_l1_ = false;
  has_philox_rng_ = false;
  ascend_rng_state_var_.clear();
  unroll_factor_.clear();

  // Pre-scan body to determine kernel mode (Mix / Vector / Cube)
  bool has_cube = false, has_vector = false;
  bool has_vector_subcores = false;
  bool has_nd2nz_post_copy = false;
  bool has_cooperative_groups = false;
  int vector_count = 2;
  VisitAscendIR(f->body, [&](const ffi::ObjectRef &n) {
    if (auto block = DecodeAscendBlock(n.as<RegionStmtObj>())) {
      if (block->name == "CUBE")
        has_cube = true;
      else if (block->name == "VECTOR") {
        has_vector = true;
        if (auto opt = block->annotations.Get("vector_count")) {
          has_vector_subcores = true;
          const auto count = AscendAnnotationInt(opt.value());
          ICHECK((count == 1 || count == 2))
              << "Mixed-kernel vector_count must be the constant integer 1 or "
                 "2, got "
              << opt.value();
          vector_count = count;
        }
      }
    }
    if (const auto *call = n.as<CallObj>()) {
      if (IsCodegenOp(call, "tl.ascend_gemm_l1") ||
          IsCodegenOp(call, "tl.ascend_blockscaled_gemm_l1")) {
        has_gemm_l1_ = true;
      }
      if (IsCodegenOp(call, "tl.ascend_mad") ||
          IsCodegenOp(call, "tl.ascend_mad_mx")) {
        has_gemm_l0_ = true;
      }
      if (IsCodegenOp(call, "tl.ascend_nd2nz_scatter") ||
          IsCodegenOp(call, "tl.ascend_nd2nz_post_copy")) {
        has_nd2nz_copy_ = true;
      }
      if (IsCodegenOp(call, "tl.ascend_nd2nz_post_copy")) {
        has_nd2nz_post_copy = true;
      }
      if (IsCodegenOp(call, "tl.ascend_read_gm_bypass_dcache") ||
          IsCodegenOp(call, "tl.ascend_write_gm_bypass_dcache")) {
        has_gm_bypass_dcache_ = true;
      }
      if (IsCodegenOp(call, "tl.rng_init")) {
        has_philox_rng_ = true;
      }
      if (IsCodegenOp(call, "tl.sync_warp")) {
        has_cooperative_groups = true;
      }
    }
  });

  // Conditionally include gemm.h when ascend_gemm_l1 is used
  if (has_gemm_l1_ && !has_gemm_l1_included_) {
    preamble_stream_ << "#include <tl_templates/ascend/gemm.h>\n";
    has_gemm_l1_included_ = true;
  }

  // Conditionally include nd2nz_copy.h when ascend_nd2nz_copy is used
  if (has_nd2nz_copy_ && !has_nd2nz_copy_included_) {
    preamble_stream_ << "#include <tl_templates/ascend/nd2nz_copy.h>\n";
    has_nd2nz_copy_included_ = true;
  }

  // Conditionally include dcache_bypass.h when scalar global GM access emits
  // tl::read/write_gm_bypass_dcache
  if (has_gm_bypass_dcache_ && !has_gm_bypass_dcache_included_) {
    preamble_stream_ << "#include <tl_templates/ascend/dcache_bypass.h>\n";
    has_gm_bypass_dcache_included_ = true;
  }

  // Conditionally include philox_rng.h when tl.rng_init is used
  if (has_philox_rng_ && !has_philox_rng_included_) {
    preamble_stream_ << "#include <tl_templates/ascend/philox_rng.h>\n";
    has_philox_rng_included_ = true;
  }

  // sync_warp lowers to cooperative_groups::coalesced_threads().sync().
  if (has_cooperative_groups && !has_cooperative_groups_included_) {
    preamble_stream_ << "#include <simt_api/cooperative_groups.h>\n";
    has_cooperative_groups_included_ = true;
  }

  if (has_vector && (has_cube || has_vector_subcores)) {
    // An explicit subcore group needs the 1C2V launch geometry even when
    // its Cube stream is empty. A pure Vector launch exposes only sid 0.
    kernel_mode_ = AscendKernelMode::kMix;
    mix_aiv_count_ = vector_count;
  } else if (has_cube) {
    kernel_mode_ = AscendKernelMode::kCube;
  } else if (has_vector) {
    kernel_mode_ = AscendKernelMode::kVector;
  } else if (has_gemm_l0_ || has_gemm_l1_ || has_nd2nz_post_copy) {
    // ND2NZ scatter executes on the Vector core and must not make a
    // scatter-only kernel Cube. Preserve the existing post-copy classification.
    kernel_mode_ = AscendKernelMode::kCube;
  } else {
    kernel_mode_ = AscendKernelMode::kVector;
  }

  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol) << "CodeGenTileLangAscend: Expect PrimFunc to have the "
                           "global_symbol attribute";
  current_function_name_ = global_symbol.value();
  CodeGenC::AddFunction(gvar, f);
  ICHECK(IsOutsideVF())
      << "Ascend VF mode leaked after PrimFunc code generation";
}

void CodeGenTileLangAscend::PreFunctionBody(const tirx::Function &f) {
  this->stream << "  asc_init();\n";
  // Emit the function-entry #line anchor after the compiler prologue, so the
  // directive's line mapping is not consumed by generated code.
  CodeGenTileLangCBase::PreFunctionBody(f);
}

ffi::Array<Var>
CodeGenTileLangAscend::CollectVFCaptures(const RegionStmtObj *op) const {
  const auto block = DecodeAscendBlock(op).value();
  ffi::Array<Var> captures;
  for (const Var &var : tvm::UndefinedVars(op->body)) {
    auto it = var_idmap_.find(var.get());
    if (block.name == "SIMT_VF" && it != var_idmap_.end() &&
        it->second == "block_idx")
      continue;
    const bool pointer =
        var->ty.as<PtrTypeObj>() || var->ty.as<TensorTypeObj>();
    const auto primitive = var->ty.as<PrimType>();
    TVM_FFI_CHECK(pointer || (primitive && primitive.value().IsScalar() &&
                              !primitive.value().IsVoid()),
                  TypeError)
        << block.name << " cannot capture " << var->ty;
    captures.push_back(var);
  }
  return captures;
}

void CodeGenTileLangAscend::EmitVFFunction(const RegionStmtObj *op,
                                           const ffi::Array<Var> &captures,
                                           const std::string &helper_name,
                                           const std::string &func_attrs,
                                           VFMode vf_mode) {
  for (const auto &v : captures) {
    if ((v->ty.as<PtrTypeObj>() || v->ty.as<TensorTypeObj>())) {
      alloc_storage_scope_[v.get()] =
          runtime::MemoryScope(CodegenStorageScope(v));
    }
  }

  std::unordered_map<const VarObj *, std::string> saved_var_idmap;
  saved_var_idmap.swap(var_idmap_);
  UniqueNameSupply saved_name_supply = name_supply_;
  name_supply_ = UniqueNameSupply();
  for (const auto &kv : saved_var_idmap) {
    var_idmap_[kv.first] = kv.second;
  }
  // Captures whose printed name is not a valid C identifier (which is only
  // accessible at kernel scope and must be passed in as a parameter) get a
  // sanitized parameter name. The sanitized name is written back into the
  // body-local var_idmap_ so body references print the parameter, and recorded
  // in vf_param_name for use in the function signature.
  std::unordered_map<const VarObj *, std::string> vf_param_name;
  for (const auto &v : captures) {
    auto it = var_idmap_.find(v.get());
    std::string name =
        (it != var_idmap_.end()) ? it->second : std::string(v->name);
    if (name.find('.') != std::string::npos) {
      std::replace(name.begin(), name.end(), '.', '_');
      var_idmap_[v.get()] = name;
    }
    vf_param_name[v.get()] = name;
  }
  // Reserve all capture parameter names in the fresh UniqueNameSupply so that
  // newly generated variables inside the VF body do not shadow them.
  for (const auto &v : captures) {
    auto it = var_idmap_.find(v.get());
    std::string cap_name =
        (it != var_idmap_.end()) ? it->second : std::string(v->name);
    name_supply_->ReserveName(cap_name);
  }
  // Render body to temporary stream
  std::ostringstream saved_main_stream;
  stream.swap(saved_main_stream);
  std::string helper_body;
  {
    // VF mode governs only helper-body expression lowering. Capture setup and
    // the helper signature use the surrounding kernel ABI, so keeping them in
    // VF mode can change scalar capture types (for example fp8_e4_t).
    VFModeScope vf_mode_scope(this, vf_mode);
    this->Dispatch(op->body);
    helper_body = FixIndent(stream.str());
  }
  stream.str("");
  stream.clear();
  stream.swap(saved_main_stream);

  var_idmap_.swap(saved_var_idmap);
  name_supply_ = saved_name_supply;

  auto resolve_var_name = [this](const Var &v) -> std::string {
    auto it = var_idmap_.find(v.get());
    if (it != var_idmap_.end()) {
      return it->second;
    }
    return v->name;
  };

  decl_stream << func_attrs << " " << helper_name << "(";
  for (size_t i = 0; i < captures.size(); ++i) {
    if (i != 0) {
      decl_stream << ", ";
    }
    const Var &v = captures[i];
    auto pit = vf_param_name.find(v.get());
    std::string vname =
        (pit != vf_param_name.end()) ? pit->second : resolve_var_name(v);
    if (!(v->ty.as<PtrTypeObj>() || v->ty.as<TensorTypeObj>())) {
      CodeGenC::PrintType(v->ty, decl_stream);
      decl_stream << " " << vname;
    } else {
      auto scope_it = alloc_storage_scope_.find(v.get());
      if (scope_it != alloc_storage_scope_.end() &&
          scope_it->second.name() == "local.var") {
        auto it = handle_data_type_.find(v.get());
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
        } else {
          CodeGenC::PrintType(v->ty, decl_stream);
        }
        decl_stream << " " << vname;
      } else {
        auto it = handle_data_type_.find(v.get());

        PrintStorageScope(runtime::MemoryScope(CodegenStorageScope(v)),
                          decl_stream);
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
          decl_stream << "* " << vname;
        } else {
          decl_stream << "void* " << vname;
        }
      }
    }
  }
  decl_stream << ") {\n" << helper_body << "}\n\n";
}

void CodeGenTileLangAscend::EmitBlockRegion_(const RegionStmtObj *op) {
  const auto block = DecodeAscendBlock(op).value();
  if (block.name == "CUBE") {
    if (kernel_mode_ == AscendKernelMode::kMix) {
      PrintIndent();
      stream << "if ASC_IS_AIC {\n";
      int scope = BeginScope();
      PrintStmt(op->body);
      EndScope(scope);
      PrintIndent();
      stream << "}\n";
    } else {
      PrintStmt(op->body);
    }
    return;
  }
  if (block.name == "VECTOR") {
    if (kernel_mode_ == AscendKernelMode::kMix) {
      PrintIndent();
      stream << "if ASC_IS_AIV {\n";
      int scope = BeginScope();
      if (mix_aiv_count_ == 1) {
        PrintIndent();
        stream << "if (asc_get_sub_block_id() == 0) {\n";
        int active_aiv_scope = BeginScope();
        PrintStmt(op->body);
        EndScope(active_aiv_scope);
        PrintIndent();
        stream << "}\n";
      } else {
        PrintStmt(op->body);
      }
      EndScope(scope);
      PrintIndent();
      stream << "}\n";
    } else {
      PrintStmt(op->body);
    }
    return;
  }
  if (block.name == "SIMD_VF") {
    if (!has_simd_inst_included_) {
      preamble_stream_ << "#include <tl_templates/ascend/simd_inst.h>\n";
      has_simd_inst_included_ = true;
    }
    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simdvf_helper_counter_++;
    {
      auto it = block.annotations.find("tl.vf_source_index");
      if (it != block.annotations.end()) {
        if (auto *imm = (*it).second.as<IntImmObj>()) {
          vf_idx = CodegenInt(imm);
        }
      }
    }
    // Unrolling can clone a VF region while preserving its source index.
    std::string helper_name = name_supply_->FreshName(
        current_function_name_ + "_simd_vf_" + std::to_string(vf_idx));
    EmitVFFunction(op, captures, helper_name, "__simd_vf__ inline void",
                   VFMode::kSimd);

    PrintIndent();
    stream << helper_name << "(";
    for (size_t i = 0; i < captures.size(); ++i) {
      if (i != 0)
        stream << ", ";
      auto it = var_idmap_.find(captures[i].get());
      stream << (it != var_idmap_.end() ? it->second
                                        : std::string(captures[i]->name));
    }
    stream << ");\n";
    return;
  }
  if (block.name == "SIMT_VF") {
    int64_t thread_x = 1, thread_y = 1, thread_z = 1;
    // Detect the threadIdx.{x,y,z} extents with a visitor rather than assuming
    // the thread_extent AttrStmts lead the block body. Earlier passes (e.g.
    // buffer allocation planning) may emit AllocBuffer/DeclBuffer statements
    // before the thread bindings; a positional scan would then miss them and
    // wrongly fall back to a single thread.
    VisitAscendIR(op->body, [&](const ffi::ObjectRef &node) {
      const auto *region = node.as<RegionStmtObj>();
      if (!region || !region->op.same_as(tirx::launch_thread_op()))
        return;
      TVM_FFI_CHECK(region->args.size() == 2 && region->body_params.size() == 1,
                    ValueError);
      const auto tag = region->args[0].as_or_throw<StringImm>()->value;
      const int64_t extent = CodegenInt(region->args[1]);
      TVM_FFI_CHECK_GT(extent, 0, ValueError);
      if (tag == "threadIdx.x")
        thread_x = extent;
      else if (tag == "threadIdx.y")
        thread_y = extent;
      else if (tag == "threadIdx.z")
        thread_z = extent;
    });

    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simtvf_helper_counter_++;
    {
      auto it = block.annotations.find("tl.vf_source_index");
      if (it != block.annotations.end()) {
        if (auto *imm = (*it).second.as<IntImmObj>()) {
          vf_idx = CodegenInt(imm);
        }
      }
    }
    // Unrolling can clone a VF region while preserving its source index.
    std::string helper_name = name_supply_->FreshName(
        current_function_name_ + "_simt_vf_" + std::to_string(vf_idx));

    int64_t total_threads = thread_x * thread_y * thread_z;
    std::string func_attrs = "__simt_vf__ ";
    if (total_threads > 0) {
      func_attrs += "__launch_bounds__(" + std::to_string(total_threads) + ") ";
    }
    func_attrs += "inline void";
    EmitVFFunction(op, captures, helper_name, func_attrs, VFMode::kSimt);

    PrintIndent();
    stream << "asc_vf_call<" << helper_name << ">(cce::dim3(" << thread_x;
    if (thread_y > 1 || thread_z > 1) {
      stream << ", " << thread_y;
      if (thread_z > 1)
        stream << ", " << thread_z;
    }
    stream << ")";
    for (size_t i = 0; i < captures.size(); ++i) {
      auto it = var_idmap_.find(captures[i].get());
      stream << ", "
             << (it != var_idmap_.end() ? it->second
                                        : std::string(captures[i]->name));
    }
    stream << ");\n";
    return;
  }
  PrintStmt(op->body);
}

void CodeGenTileLangAscend::Dispatch_(const prim::BroadcastObj *op,
                                      std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(CodegenDType(op).lanes());
  RejectGenericVectorInSimd("Broadcast", CodegenDType(op));

  // Keep casts outside broadcasts so the vector value is converted as a
  // whole. Besides avoiding target-specific Broadcast(Cast(Call)) patterns,
  // this lets vector casts use packed conversion intrinsics such as
  // float2 -> bfloat16x2.
  if (const auto *cast = op->value.as<prim::CastObj>()) {
    PrimExpr source = prim::Broadcast(cast->value, op->lanes);
    PrintExpr(prim::Cast(CodegenDType(op), source), os);
    return;
  }

  // Match CUDA codegen's stateful RNG vectorization: the loop vectorizer
  // leaves an opaque scalar RNG call under Broadcast, and target codegen
  // replaces the vector with one backend runtime call. Ascend's default 64-bit
  // vector width makes the FP32 case a float2.
  const CallObj *rng_call = op->value.as<CallObj>();
  if (rng_call != nullptr) {
    const CallObj *call = rng_call;
    if (IsCodegenOp(call, "tl.rng_rand_float") && (lanes == 2 || lanes == 4) &&
        CodegenDType(call).MatchesCode(kDLFloat) &&
        CodegenDType(call).bits() == 32) {
      ICHECK_EQ(CodegenOperands(call).size(), 1);
      const auto *dist = CodegenOperands(call)[0].as<StringImmObj>();
      ICHECK(dist != nullptr) << "Ascend RNG distribution must be constant";
      if (dist->value == "uniform" || dist->value == "normal") {
        EmitPhiloxVectorFloat(os, dist->value, lanes);
        return;
      }
    }
  }

  if ((CodegenDType(op).MatchesCode(kDLInt) ||
       CodegenDType(op).MatchesCode(kDLUInt)) &&
      CodegenDType(op).bits() == 8) {
    const auto *p = op->value.as<IntImmObj>();
    if (p) {
      if (lanes == 4) {
        // make_int8x4
        ICHECK(p);
        int64_t v = CodegenInt(p) & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (CodegenDType(op).MatchesCode(kDLUInt)) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
        return;
      } else if (lanes == 32 && IsInsideSimtVF()) {
        // make_int8x32
        const auto *p = op->value.as<IntImmObj>();
        ICHECK(p);
        int64_t v = CodegenInt(p) & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (CodegenDType(op).MatchesCode(kDLUInt)) {
          os << "make_ulonglong4(" << v << ", " << v << ", " << v << ", " << v
             << ")";
        } else {
          os << "make_longlong4(" << v << ", " << v << ", " << v << ", " << v
             << ")";
        }
        return;
      }
    }
  }

  if (CodegenDType(op).MatchesCode(kDLFloat) && CodegenDType(op).bits() == 32 &&
      CodegenDType(op).lanes() == 8) {
    std::string v = PrintExpr(op->value);
    os << "make_ulonglong4(";
    for (int i = 0; i < 4; ++i) {
      if (i != 0)
        os << ", ";
      os << "*(unsigned long long*)&make_float2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if ((CodegenDType(op).MatchesCode(kDLInt) ||
       CodegenDType(op).MatchesCode(kDLUInt)) &&
      CodegenDType(op).bits() == 4) {
    bool fail = false;
    const auto *p = op->value.as<IntImmObj>();
    ICHECK(p) << "BroadcastObj " << op << " value: " << op->value
              << " is not a constant";
    int64_t v = CodegenInt(p) & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (CodegenDType(op).MatchesCode(kDLUInt)) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) |
          (v << 4) | v;
      if (lanes == 8) {
        if (CodegenDType(op).MatchesCode(kDLUInt)) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(CodegenDType(op), os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0)
            os << ", ";
          if (CodegenDType(op).MatchesCode(kDLUInt)) {
            os << "(uint)" << v;
          } else {
            os << "(int)" << v;
          }
        }
        os << ')';
      } else {
        fail = true;
      }
    }

    if (!fail) {
      return;
    }
  }

  // Legacy packed broadcast carriers do not denote addressable SIMT vectors,
  // so materialize them lane by lane.
  const bool is_packed_16 =
      (CodegenDType(op).MatchesElementType(kDLFloat, 16) ||
       CodegenDType(op).MatchesElementType(kDLBfloat, 16)) &&
      lanes <= 8 && lanes % 2 == 0;
  const bool is_packed_predicate = CodegenDType(op).MatchesCode(kDLBool) &&
                                   lanes > 4 && lanes <= 8 && lanes % 2 == 0;
  if (is_packed_16 || is_packed_predicate) {
    // Small FP16/BF16 vectors and wide predicates use uint carriers in
    // generated Ascend C. Materialize the carrier lane by lane instead of
    // relying on constructors whose argument count follows the carrier rather
    // than the logical lane count. Larger vectors are CCE SIMD registers and
    // must be produced by SIMD intrinsics such as vdup.
    TVM_FFI_ICHECK_LE(lanes, 8)
        << "Ascend packed Broadcast supports at most 8 lanes outside SIMD "
           "intrinsics, got "
        << CodegenDType(op);
    std::string result = name_supply_->FreshName("_");
    PrintIndent();
    PrintType(CodegenDType(op), stream);
    stream << ' ' << result << ";\n";
    SSAOperationScope ssa_scope(this);
    std::string value =
        SSAGetID(PrintExpr(op->value), op->value.ty(),
                 tvm::SideEffect(op->value) == CallEffectKind::kPure);
    for (int i = 0; i < lanes; ++i) {
      PrintVecElemStore(result, CodegenDType(op), i, value);
    }
    os << result;
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(CodegenDType(op), os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenTileLangAscend::Dispatch_(const prim::RampObj *op,
                                      std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(CodegenDType(op).lanes());
  ValidateGenericVectorType("Ramp", CodegenDType(op));
  os << "make_";
  PrintType(CodegenDType(op), os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << '(' << PrintExpr(op->base) << ")+(" << PrintExpr(op->stride) << '*'
       << i << ')';
  }
  os << ')';
}

std::string
CodeGenTileLangAscend::ResolveSimdIntrinsicName_(const Call &call,
                                                 PrimType result_dtype) const {
  const std::string op_name = (call->op).as_or_throw<Op>()->name;
  const std::string intrinsic_name =
      op_name.substr(std::string(kSimdOpPrefix).size());

  // Both zeroing and merging calls use the same precision wrapper. Merging
  // calls are void, so their actual result dtype is supplied by the caller.
  if (intrinsic_name == "vdiv") {
    const bool is_f32 =
        result_dtype.MatchesCode(kDLFloat) && result_dtype.bits() == 32;
    const SfuPrecision prec = ResolveSfuPrecision(call, result_dtype,
                                                  (is_f32 && !enable_fast_math_)
                                                      ? SfuPrecision::kExact
                                                      : SfuPrecision::kHw);
    if (is_f32 && prec == SfuPrecision::kExact) {
      return "vdiv_0ulp_ftz_true";
    }
  } else if (intrinsic_name == "vexp" || intrinsic_name == "vln" ||
             intrinsic_name == "vsqrt") {
    const SfuPrecision prec =
        ResolveSfuPrecision(call, result_dtype, SfuPrecision::kHw);
    if (prec == SfuPrecision::kKeepSub) {
      const char *suffix =
          intrinsic_name == "vsqrt" ? "_0ulp_ftz_false" : "_1ulp_ftz_false";
      return intrinsic_name + suffix;
    }
  }
  return intrinsic_name;
}

void CodeGenTileLangAscend::PrintSimdCall_(const Call &call,
                                           size_t num_expr_args,
                                           std::ostream &os) {
  os << "simd_inst::" << ResolveSimdIntrinsicName_(call, CodegenDType(call))
     << "(";
  for (size_t i = 0; i < CodegenOperands(call).size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    if (i < num_expr_args) {
      PrintExpr(CodegenOperands(call)[i], os);
    } else {
      // Trailing arguments name control enums and must stay unquoted.
      os << (CodegenOperands(call)[i]).as_or_throw<StringImm>()->value;
    }
  }
  os << ")";
}

bool CodeGenTileLangAscend::EmitSimdMergingCall(const CallObj *op,
                                                std::ostream &os) {
  std::string intrinsic_name;
  if (!IsSimdMergingCall(op, &intrinsic_name)) {
    return false;
  }

  ICHECK_GE(CodegenOperands(op).size(), 3)
      << "Legalized MODE_MERGING call must contain a destination, operands, "
         "and mode";
  ICHECK(IsAscendPointer(CodegenOperands(op)[0]))
      << "Legalized MODE_MERGING destination must be a pointer";
  std::string resolved_name = ResolveSimdIntrinsicName_(
      GetRef<Call>(op), CodegenDType(CodegenOperands(op)[1]));
  PrimType dtype = CodegenDType(CodegenOperands(op)[1]);
  if (intrinsic_name == "vdup") {
    // Scalar broadcast overloads are selected by the destination vector type.
    const auto *destination = CodegenOperands(op)[0].as<CallObj>();
    ICHECK(destination && IsCodegenOp(destination, "tirx.address_of") &&
           destination->args.size() == 1);
    dtype = CodegenDType(destination->args[0]);
  }

  // Use CCE merging for the validated overloads of these C API wrappers.
  // Explicit zeroing + select can cause predicate spills in masked updates.
  // Precision-specific SFU algorithms retain their software merging wrappers.
  static const std::unordered_set<std::string> kCapiMergingOps = {
      "vadd",  "vsub",  "vmul",  "vdiv",  "vabsdif", "vmax",  "vmin",  "vand",
      "vor",   "vxor",  "vshl",  "vshr",  "vln",     "vsqrt", "vabs",  "vneg",
      "vrelu", "vnot",  "vexp",  "vdup",  "vdupv",   "vmaxs", "vmins", "vmuls",
      "vshls", "vshrs", "vaxpy", "vcadd", "vcmax",   "vcmin"};
  const bool use_wrapper = resolved_name != intrinsic_name ||
                           (kCapiMergingOps.count(intrinsic_name) &&
                            !SupportsNativeSimdMerging(intrinsic_name, dtype));
  if (!use_wrapper && intrinsic_name == "vdupv") {
    resolved_name = "vdup";
  }
  os << (use_wrapper ? "simd_inst::" : "::") << resolved_name << "(*(";
  PrintExpr(CodegenOperands(op)[0], os);
  os << ")";
  for (size_t i = 1; i < CodegenOperands(op).size(); ++i) {
    os << ", ";
    if (const auto *string_arg = CodegenOperands(op)[i].as<StringImmObj>()) {
      os << string_arg->value;
    } else {
      PrintExpr(CodegenOperands(op)[i], os);
    }
  }
  os << ")";
  return true;
}

void CodeGenTileLangAscend::EmitPhiloxVectorFloat(
    std::ostream &os, const std::string &distribution, int lanes) {
  ICHECK(IsInsideSimtVF())
      << "Ascend Philox vector RNG must be used inside T.SimtVF()";
  ICHECK(!ascend_rng_state_var_.empty())
      << "Ascend Philox vector RNG requires a preceding T.rng_init call";
  ICHECK(distribution == "uniform" || distribution == "normal")
      << "Unsupported Ascend vector RNG distribution: " << distribution;
  ICHECK(lanes == 2 || lanes == 4)
      << "Unsupported Ascend vector RNG width: " << lanes;
  os << "tl::philox_rand_" << distribution << lanes;
  os << "(&" << ascend_rng_state_var_ << ")";
}

void CodeGenTileLangAscend::EmitCApiCall_(
    const std::string &function_name,
    std::initializer_list<CApiArgument> arguments) {
  std::vector<std::string> rendered_arguments;
  rendered_arguments.reserve(arguments.size());
  for (const CApiArgument &argument : arguments) {
    rendered_arguments.push_back(argument.prefix + PrintExpr(argument.value) +
                                 argument.suffix);
  }

  PrintIndent();
  stream << function_name << "(";
  for (size_t i = 0; i < rendered_arguments.size(); ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << rendered_arguments[i];
  }
  stream << ");\n";
}

void CodeGenTileLangAscend::EmitPipeBarrier_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 1)
      << "tl.ascend_pipe_barrier expects exactly 1 argument (pipe_t string)";
  const std::string pipe =
      GetStringImmArg(op, 0, "tl.ascend_pipe_barrier", "pipe");
  static const std::unordered_set<std::string> kValidPipes = {
      "PIPE_ALL",  "PIPE_V",    "PIPE_M", "PIPE_MTE1",
      "PIPE_MTE2", "PIPE_MTE3", "PIPE_S", "PIPE_FIX"};
  ICHECK(kValidPipes.count(pipe))
      << "Unsupported Ascend pipeline barrier: " << pipe;

  if (pipe == "PIPE_V") {
    return;
  }
  PrintIndent();
  if (pipe == "PIPE_ALL" || pipe == "PIPE_S") {
    stream << "asc_sync();\n";
  } else {
    stream << "asc_sync_pipe(" << pipe << ");\n";
  }
}

void CodeGenTileLangAscend::EmitHardEventSync_(const CallObj *op, bool notify) {
  const char *intrinsic = notify ? "tl.ascend_set_flag" : "tl.ascend_wait_flag";
  ICHECK_EQ(CodegenOperands(op).size(), 2)
      << intrinsic
      << " expects exactly 2 arguments (hard_event string, event_id)";
  const std::string hard_event =
      GetStringImmArg(op, 0, intrinsic, "hard_event");
  const PipePair pipes = GetPipePair(hard_event);
  const std::string event_id = PrintExpr(CodegenOperands(op)[1]);

  PrintIndent();
  stream << (notify ? "asc_sync_notify(" : "asc_sync_wait(") << pipes.source
         << ", " << pipes.destination << ", static_cast<event_t>(" << event_id
         << "));\n";
}

void CodeGenTileLangAscend::EmitCrossCoreSync_(const CallObj *op, bool arrive) {
  const char *intrinsic = arrive ? "tl.ascend_cross_core_set_flag"
                                 : "tl.ascend_cross_core_wait_flag";
  ICHECK_EQ(CodegenOperands(op).size(), 3)
      << intrinsic
      << " expects exactly 3 arguments (mode_id int, pipe string, flag_id)";
  const int64_t mode_id = GetIntImmArg(op, 0, intrinsic, "mode_id");
  const char *scope = GetCrossCoreSyncScope(mode_id);
  const std::string pipe = GetStringImmArg(op, 1, intrinsic, "pipe");
  const std::string flag_id = PrintExpr(CodegenOperands(op)[2]);

  PrintIndent();
  stream << "asc_sync_" << scope << (arrive ? "_arrive(" : "_wait(") << pipe
         << ", " << flag_id << ");\n";
}

void CodeGenTileLangAscend::EmitGmToUbufCopy_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 11)
      << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";

  // burstLen and stride use bytes; right_pad uses elements of the buffer dtype.
  std::string pointer_type = "uint8_t";
  const auto *right_pad = CodegenOperands(op)[6].as<IntImmObj>();
  if (right_pad && right_pad->value != 0) {
    const PrimType element_type =
        AscendPointerElementType(CodegenOperands(op)[0]);
    const int bits = element_type.bits();
    ICHECK(bits == 8 || bits == 16 || bits == 32)
        << "Ascend padded GM->UB copy supports 8/16/32-bit elements, got "
        << bits;
    pointer_type = "uint" + std::to_string(bits) + "_t";
  }

  // IR arg 2 is sid, which asc_copy_gm2ub_align does not take.
  EmitCApiCall_(
      "asc_copy_gm2ub_align",
      {{CodegenOperands(op)[0], "(__ubuf__ " + pointer_type + "*)(", ")"},
       {CodegenOperands(op)[1], "(__gm__ " + pointer_type + "*)(", ")"},
       {CodegenOperands(op)[3]},
       {CodegenOperands(op)[4]},
       {CodegenOperands(op)[5]},
       {CodegenOperands(op)[6]},
       {CodegenOperands(op)[7]},
       {CodegenOperands(op)[8], "static_cast<asc_load_l2_cache_mode>(", ")"},
       {CodegenOperands(op)[9]},
       {CodegenOperands(op)[10]}});
}

void CodeGenTileLangAscend::EmitUbufToGmCopy_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 8)
      << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";

  // IR arg 2 is sid, which asc_copy_ub2gm_align does not take.
  EmitCApiCall_(
      "asc_copy_ub2gm_align",
      {{CodegenOperands(op)[0], "(__gm__ uint8_t*)(", ")"},
       {CodegenOperands(op)[1], "(__ubuf__ uint8_t*)(", ")"},
       {CodegenOperands(op)[3]},
       {CodegenOperands(op)[4]},
       {CodegenOperands(op)[5], "static_cast<asc_store_l2_cache_mode>(", ")"},
       {CodegenOperands(op)[6]},
       {CodegenOperands(op)[7]}});
}

void CodeGenTileLangAscend::EmitGmToL1Copy_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 12)
      << "tl.ascend_copy_gm_to_cbuf expects exactly 12 arguments";

  const bool use_dn2nz =
      GetIntImmArg(op, 9, "tl.ascend_copy_gm_to_cbuf", "transpose") != 0;
  const std::string nz_c0_stride = PrintExpr(CodegenOperands(op)[10]);
  PrintIndent();
  stream << "asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(" << nz_c0_stride
         << "), 0);\n";

  std::string destination_prefix;
  std::string source_prefix;
  std::string pointer_suffix;
  const std::string physical_dtype =
      GetStringImmArg(op, 11, "tl.ascend_copy_gm_to_cbuf", "physical_dtype");
  if (!physical_dtype.empty()) {
    destination_prefix = "(__cbuf__ " + physical_dtype + "*)(";
    source_prefix = "(__gm__ " + physical_dtype + "*)(";
    pointer_suffix = ")";
  }

  // IR args 2, 9, 10, and 11 configure lowering and are not DMA arguments.
  EmitCApiCall_(
      use_dn2nz ? "asc_copy_gm2l1_dn2nz" : "asc_copy_gm2l1_nd2nz",
      {{CodegenOperands(op)[0], destination_prefix, pointer_suffix},
       {CodegenOperands(op)[1], source_prefix, pointer_suffix},
       {CodegenOperands(op)[3]},
       {CodegenOperands(op)[4], "static_cast<asc_load_l2_cache_mode>(", ")"},
       {CodegenOperands(op)[5]},
       {CodegenOperands(op)[6]},
       {CodegenOperands(op)[7]},
       {CodegenOperands(op)[8]}});
}

void CodeGenTileLangAscend::EmitL0cToUbufCopy_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 26)
      << "tl.ascend_copy_matrix_cc_to_ub expects exactly 26 arguments";

  PrintIndent();
  stream << "asc_set_l0c_copy_nz_para(1, 0, 0);\n";
  EmitCApiCall_(
      "asc_copy_l0c2ub",
      {{CodegenOperands(op)[0]},
       {CodegenOperands(op)[1]},
       {CodegenOperands(op)[3]},
       {CodegenOperands(op)[4]},
       {CodegenOperands(op)[5]},
       {CodegenOperands(op)[6]},
       {CodegenOperands(op)[8]},
       {CodegenOperands(op)[7], "static_cast<asc_dual_dst_mode>(", ")"},
       {CodegenOperands(op)[10], "static_cast<asc_unit_flag_mode>(", ")"},
       {CodegenOperands(op)[11], "static_cast<asc_quant_mode>(", ")"},
       {CodegenOperands(op)[12], "static_cast<asc_relu_pre_mode>(", ")"},
       {CodegenOperands(op)[13]},
       {CodegenOperands(op)[14]},
       {CodegenOperands(op)[25]},
       {CodegenOperands(op)[9]}});
}

void CodeGenTileLangAscend::EmitL0cToGmCopy_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 25)
      << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";

  PrintIndent();
  stream << "asc_set_l0c_copy_nz_para(1, 0, 0);\n";
  EmitCApiCall_(
      "asc_copy_l0c2gm",
      {{CodegenOperands(op)[0]},
       {CodegenOperands(op)[1]},
       {CodegenOperands(op)[3]},
       {CodegenOperands(op)[4]},
       {CodegenOperands(op)[5]},
       {CodegenOperands(op)[6]},
       {CodegenOperands(op)[7], "static_cast<asc_store_l2_cache_mode>(", ")"},
       {CodegenOperands(op)[9], "static_cast<asc_unit_flag_mode>(", ")"},
       {CodegenOperands(op)[10], "static_cast<asc_quant_mode>(", ")"},
       {CodegenOperands(op)[11], "static_cast<asc_relu_pre_mode>(", ")"},
       {CodegenOperands(op)[12]},
       {CodegenOperands(op)[13]},
       {CodegenOperands(op)[24]},
       {CodegenOperands(op)[8]}});
}

void CodeGenTileLangAscend::EmitSetCopyPadValue_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 1)
      << "tl.ascend_set_copy_pad_value expects exactly 1 argument";
  const int bit_width = CodegenDType(CodegenOperands(op)[0]).bits();
  ICHECK(bit_width == 8 || bit_width == 16 || bit_width == 32)
      << "Data type: " << CodegenDType(CodegenOperands(op)[0])
      << " is not supported in ascend";

  const std::string value = PrintExpr(CodegenOperands(op)[0]);
  const std::string target_value = name_supply_->FreshName("__asc_pad_val");
  PrintIndent();
  PrintType(CodegenDType(CodegenOperands(op)[0]), stream);
  stream << " " << target_value << " = " << value << ";\n";

  PrintIndent();
  stream << "asc_set_copy_pad_val(*reinterpret_cast<uint" << bit_width
         << "_t*>(&" << target_value << "));\n";
}

void CodeGenTileLangAscend::EmitFillL1_(const CallObj *op) {
  constexpr const char *kIntrinsic = "tl.ascend_fill_l1";
  ICHECK_EQ(CodegenOperands(op).size(), 7)
      << kIntrinsic << " expects exactly 7 arguments";
  const int64_t fill_word_bits =
      GetIntImmArg(op, 6, kIntrinsic, "fill_word_bits");
  ICHECK(fill_word_bits == 16 || fill_word_bits == 32)
      << kIntrinsic << " fill_word_bits must be 16 or 32";

  std::vector<std::string> arguments;
  arguments.reserve(6);
  for (size_t i = 0; i < 6; ++i) {
    arguments.push_back(PrintExpr(CodegenOperands(op)[i]));
  }

  PrintIndent();
  stream << "asc_fill_l1((__cbuf__ uint" << fill_word_bits
         << "_t*)((__cbuf__ uint8_t*)" << arguments[0] << " + " << arguments[1]
         << "), (uint32_t)(" << arguments[2]
         << "), { .repeat = static_cast<uint64_t>(" << arguments[3]
         << "), .blk_num = static_cast<uint64_t>(" << arguments[4]
         << "), .dst_gap = static_cast<uint64_t>(" << arguments[5] << ")});\n";
}

void CodeGenTileLangAscend::EmitL1ToL0Copy_(const CallObj *op, bool is_l0a) {
  const char *intrinsic =
      is_l0a ? "tl.ascend_load_cbuf_to_ca" : "tl.ascend_load_cbuf_to_cb";
  ICHECK_EQ(CodegenOperands(op).size(), 9)
      << intrinsic << " expects exactly 9 arguments, got "
      << CodegenOperands(op).size();

  const bool is_transpose = GetIntImmArg(op, 8, intrinsic, "transpose") == 1;
  const char *function_name =
      is_transpose
          ? (is_l0a ? "asc_copy_l12l0a_transpose" : "asc_copy_l12l0b_transpose")
          : (is_l0a ? "asc_copy_l12l0a" : "asc_copy_l12l0b");
  EmitCApiCall_(function_name, {{CodegenOperands(op)[0]},
                                {CodegenOperands(op)[1]},
                                {CodegenOperands(op)[2]},
                                {CodegenOperands(op)[3]},
                                {CodegenOperands(op)[4]},
                                {CodegenOperands(op)[5]},
                                {CodegenOperands(op)[6]},
                                {CodegenOperands(op)[7]}});
}

void CodeGenTileLangAscend::EmitMxSfLoad_(const CallObj *op, bool is_l0a) {
  const char *intrinsic =
      is_l0a ? "tl.ascend_load_ca_sf" : "tl.ascend_load_cb_sf";
  ICHECK_EQ(CodegenOperands(op).size(), 8)
      << intrinsic << " expects exactly 8 arguments, got "
      << CodegenOperands(op).size();

  // The MX destination is the DATA tile's address in 16-byte units; the
  // scale source must be viewed in its physical E8M0 representation.
  EmitCApiCall_(is_l0a ? "asc_copy_l12l0a_mx" : "asc_copy_l12l0b_mx",
                {{CodegenOperands(op)[0], "(uint64_t)(uintptr_t)(", ") / 16"},
                 {CodegenOperands(op)[1], "(__cbuf__ fp8_e8m0_t*)(", ")"},
                 {CodegenOperands(op)[2]},
                 {CodegenOperands(op)[3]},
                 {CodegenOperands(op)[4]},
                 {CodegenOperands(op)[5]},
                 {CodegenOperands(op)[6]},
                 {CodegenOperands(op)[7]}});
}

void CodeGenTileLangAscend::EmitUbufToL1Copy_(const CallObj *op) {
  ICHECK_EQ(CodegenOperands(op).size(), 7)
      << "tl.ascend_copy_ubuf_to_cbuf expects exactly 7 arguments";

  // IR arg 2 is sub_blockid, which asc_copy_ub2l1 does not take.
  EmitCApiCall_("asc_copy_ub2l1",
                {{CodegenOperands(op)[0], "(__cbuf__ void*)(", ")"},
                 {CodegenOperands(op)[1], "(__ubuf__ void*)(", ")"},
                 {CodegenOperands(op)[3]},
                 {CodegenOperands(op)[4]},
                 {CodegenOperands(op)[5]},
                 {CodegenOperands(op)[6]}});
}

void CodeGenTileLangAscend::EmitNd2NzScatter_(const CallObj *op) {
  auto geometry = tl::ParseNd2NzGeometry(op, true);
  if (geometry.Empty())
    return;
  int64_t rows = geometry.rows, cols = geometry.cols;
  const std::string src_dtype =
      GetStringImmArg(op, 5, "tl.ascend_nd2nz_scatter", "src_dtype");
  const std::string dst_dtype =
      GetStringImmArg(op, 6, "tl.ascend_nd2nz_scatter", "dst_dtype");

  std::ostringstream function_name;
  function_name << (IsInsideSimdVF() ? "ascend_nd2nz_scatter_callee"
                                     : "ascend_nd2nz_scatter")
                << "<" << rows << ", " << cols << ", " << geometry.dst_pitch
                << ", " << src_dtype << ", " << dst_dtype << ">";
  EmitCApiCall_(
      function_name.str(),
      {{CodegenOperands(op)[0], "(__ubuf__ " + src_dtype + "*)(", ")"},
       {CodegenOperands(op)[1], "(__ubuf__ " + dst_dtype + "*)(", ")"}});
}

void CodeGenTileLangAscend::EmitNd2NzPostCopy_(const CallObj *op) {
  auto geometry = tl::ParseNd2NzGeometry(op, false);
  if (geometry.Empty())
    return;
  const int64_t burst_num = geometry.Bursts();
  const int64_t burst_len = geometry.rows;
  const int64_t source_gap = geometry.src_pitch - geometry.rows;
  const int64_t destination_gap = geometry.dst_pitch - geometry.rows;
  const std::string destination = PrintExpr(CodegenOperands(op)[0]);
  const std::string source = PrintExpr(CodegenOperands(op)[1]);

  PrintIndent();
  stream << "asc_copy_ub2l1((__cbuf__ void*)(" << destination
         << "), (__ubuf__ void*)(" << source << "), " << burst_num << ", "
         << burst_len << ", " << source_gap << ", " << destination_gap
         << ");\n";
}

void CodeGenTileLangAscend::EmitMad_(const CallObj *op, bool is_mx) {
  const char *intrinsic = is_mx ? "tl.ascend_mad_mx" : "tl.ascend_mad";
  ICHECK_EQ(CodegenOperands(op).size(), 10)
      << intrinsic << " expects exactly 10 arguments";
  EmitCApiCall_(is_mx ? "asc_mmad_mx" : "asc_mmad", {{CodegenOperands(op)[0]},
                                                     {CodegenOperands(op)[1]},
                                                     {CodegenOperands(op)[2]},
                                                     {CodegenOperands(op)[3]},
                                                     {CodegenOperands(op)[4]},
                                                     {CodegenOperands(op)[5]},
                                                     {CodegenOperands(op)[6]},
                                                     {CodegenOperands(op)[7]},
                                                     {CodegenOperands(op)[8]},
                                                     {CodegenOperands(op)[9]}});
}

void CodeGenTileLangAscend::EmitGemmL1_(const CallObj *op) {
  constexpr const char *kIntrinsic = "tl.ascend_gemm_l1";
  ICHECK_EQ(CodegenOperands(op).size(), 12)
      << kIntrinsic << " expects exactly 12 arguments";
  const int64_t m = GetIntImmArg(op, 3, kIntrinsic, "M");
  const int64_t k = GetIntImmArg(op, 4, kIntrinsic, "K");
  const int64_t n = GetIntImmArg(op, 5, kIntrinsic, "N");
  const int64_t tile_k_sub = GetIntImmArg(op, 6, kIntrinsic, "tile_k_sub");
  const bool transpose_b = GetIntImmArg(op, 7, kIntrinsic, "transpose_b") != 0;
  const std::string dtype = GetStringImmArg(op, 9, kIntrinsic, "dtype");

  std::ostringstream function_name;
  function_name << "ascend_gemm_l1<" << m << ", " << k << ", " << n << ", "
                << tile_k_sub << ", " << (transpose_b ? "true" : "false")
                << ", " << dtype << ">";
  EmitCApiCall_(function_name.str(), {{CodegenOperands(op)[0]},
                                      {CodegenOperands(op)[1]},
                                      {CodegenOperands(op)[2]},
                                      {CodegenOperands(op)[8]},
                                      {CodegenOperands(op)[10]},
                                      {CodegenOperands(op)[11]}});
}

void CodeGenTileLangAscend::EmitBlockscaledGemmL1_(const CallObj *op) {
  constexpr const char *kIntrinsic = "tl.ascend_blockscaled_gemm_l1";
  ICHECK_EQ(CodegenOperands(op).size(), 18)
      << kIntrinsic << " expects exactly 18 arguments";
  const int64_t m = GetIntImmArg(op, 5, kIntrinsic, "M");
  const int64_t k = GetIntImmArg(op, 6, kIntrinsic, "K");
  const int64_t n = GetIntImmArg(op, 7, kIntrinsic, "N");
  const int64_t tile_k_sub = GetIntImmArg(op, 8, kIntrinsic, "tile_k_sub");
  const bool transpose_b = GetIntImmArg(op, 9, kIntrinsic, "transpose_b") != 0;
  const std::string input_dtype =
      GetStringImmArg(op, 11, kIntrinsic, "input_dtype");
  const std::string scale_dtype =
      GetStringImmArg(op, 12, kIntrinsic, "scale_dtype");
  const std::string accumulator_dtype =
      GetStringImmArg(op, 13, kIntrinsic, "accumulator_dtype");
  const int64_t scale_nz_stride =
      GetIntImmArg(op, 16, kIntrinsic, "scale_nz_stride");

  std::ostringstream function_name;
  function_name << "ascend_blockscaled_gemm_l1<" << m << ", " << k << ", " << n
                << ", " << tile_k_sub << ", "
                << (transpose_b ? "true" : "false") << ", " << scale_nz_stride
                << ", " << input_dtype << ", " << scale_dtype << ", "
                << accumulator_dtype << ">";
  EmitCApiCall_(
      function_name.str(),
      {{CodegenOperands(op)[0]},
       {CodegenOperands(op)[1]},
       {CodegenOperands(op)[2]},
       {CodegenOperands(op)[3], "(__cbuf__ " + scale_dtype + "*)(", ")"},
       {CodegenOperands(op)[4], "(__cbuf__ " + scale_dtype + "*)(", ")"},
       {CodegenOperands(op)[10]},
       {CodegenOperands(op)[14]},
       {CodegenOperands(op)[15]},
       {CodegenOperands(op)[17]}});
}

bool CodeGenTileLangAscend::EmitAscendMemoryCall_(const CallObj *op) {
  if (IsCodegenOp(op, "tl.ascend_copy_gm_to_ubuf")) {
    EmitGmToUbufCopy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_set_copy_pad_value")) {
    EmitSetCopyPadValue_(op);
  } else if (IsCodegenOp(op, "tl.ascend_copy_ubuf_to_gm")) {
    EmitUbufToGmCopy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_copy_gm_to_cbuf")) {
    EmitGmToL1Copy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_fill_l1")) {
    EmitFillL1_(op);
  } else if (IsCodegenOp(op, "tl.ascend_load_cbuf_to_ca") ||
             IsCodegenOp(op, "tl.ascend_load_cbuf_to_cb")) {
    EmitL1ToL0Copy_(op, IsCodegenOp(op, "tl.ascend_load_cbuf_to_ca"));
  } else if (IsCodegenOp(op, "tl.ascend_load_ca_sf") ||
             IsCodegenOp(op, "tl.ascend_load_cb_sf")) {
    EmitMxSfLoad_(op, IsCodegenOp(op, "tl.ascend_load_ca_sf"));
  } else if (IsCodegenOp(op, "tl.ascend_copy_matrix_cc_to_ub")) {
    EmitL0cToUbufCopy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_copy_matrix_cc_to_gm")) {
    EmitL0cToGmCopy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_copy_ubuf_to_cbuf")) {
    EmitUbufToL1Copy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_nd2nz_scatter")) {
    EmitNd2NzScatter_(op);
  } else if (IsCodegenOp(op, "tl.ascend_nd2nz_post_copy")) {
    EmitNd2NzPostCopy_(op);
  } else if (IsCodegenOp(op, "tl.ascend_mad") ||
             IsCodegenOp(op, "tl.ascend_mad_mx")) {
    EmitMad_(op, IsCodegenOp(op, "tl.ascend_mad_mx"));
  } else if (IsCodegenOp(op, "tl.ascend_gemm_l1")) {
    EmitGemmL1_(op);
  } else if (IsCodegenOp(op, "tl.ascend_blockscaled_gemm_l1")) {
    EmitBlockscaledGemmL1_(op);
  } else {
    return false;
  }
  return true;
}

void CodeGenTileLangAscend::Dispatch_(const CallObj *op, std::ostream &os) {
  std::string simd_op_name;
  if (IsSimdOp(op, &simd_op_name)) {
    if (!IsInsideSimdVF()) {
      TVM_FFI_THROW(InternalError)
          << simd_op_name << " can only be emitted inside SimdVF";
    }
  }
  if (IsCodegenOp(op, "prim.bitwise_not") && CodegenDType(op).lanes() > 1) {
    RejectGenericVectorInSimd("bitwise Not", CodegenDType(op));
    if (CodegenIsVectorBool(CodegenDType(op)) &&
        CodegenDType(op).lanes() <= 8) {
      ICHECK_EQ(CodegenOperands(op).size(), 1U);
      EmitSimtVectorPredicateNot(CodegenOperands(op)[0].as_or_throw<PrimExpr>(),
                                 CodegenDType(op), os);
      return;
    }
  }
  if (EmitSimdMergingCall(op, os)) {
    return;
  }
  if (EmitAscendMemoryCall_(op)) {
    return;
  }
  if (IsCodegenOp(op, "tirx.reinterpret")) {
    SSAOperationScope ssa_scope(this);
    CodeGenC::Dispatch_(op, os);
    return;
  }
  if (IsCodegenOp(op, "tl.device_assert")) {
    ICHECK_EQ(CodegenOperands(op).size(), 1)
        << "tl.device_assert expects exactly 1 argument (condition)";
    this->PrintIndent();
    this->stream << "device_assert(" << PrintExpr(CodegenOperands(op)[0])
                 << ");\n";
  } else if (IsCodegenOp(op, "tl.device_assert_with_msg")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2)
        << "tl.device_assert_with_msg expects exactly 2 arguments "
           "(condition, msg)";
    this->PrintIndent();
    this->stream << "device_assert_with_msg("
                 << PrintExpr(CodegenOperands(op)[0]) << ", "
                 << PrintExpr(CodegenOperands(op)[1]) << ");\n";
  } else if (IsCodegenOp(op, "tl.ascend_pipe_barrier")) {
    EmitPipeBarrier_(op);
  } else if (IsCodegenOp(op, "tl.ascend_set_flag") ||
             IsCodegenOp(op, "tl.ascend_wait_flag")) {
    EmitHardEventSync_(op, IsCodegenOp(op, "tl.ascend_set_flag"));
  } else if (IsCodegenOp(op, "tl.ascend_threadfence")) {
    ICHECK_EQ(CodegenOperands(op).size(), 0)
        << "tl.ascend_threadfence expects 0 arguments";
    this->PrintIndent();
    this->stream << "asc_threadfence();\n";
  } else if (IsCodegenOp(op, "tl.sync_warp")) {
    ICHECK_LE(CodegenOperands(op).size(), 1U)
        << "tl.sync_warp expects an optional lane mask";
    if (CodegenOperands(op).empty()) {
      this->PrintIndent();
      this->stream << "cooperative_groups::coalesced_threads().sync();\n";
    } else {
      // coalesced_threads() captures the active lanes at the call site. Enter
      // the branch only for named lanes so the resulting group matches mask.
      std::string mask = this->PrintExpr(CodegenOperands(op)[0]);
      this->PrintIndent();
      this->stream << "if (((static_cast<uint32_t>(" << mask
                   << ") >> laneid()) & 1U) != 0U) { "
                      "cooperative_groups::coalesced_threads().sync(); }\n";
    }
  } else if (IsCodegenOp(op, "tl.ballot_sync")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2U)
        << "tl.ballot_sync expects <mask, predicate>.";
    os << "((uint64_t)(asc_ballot(";
    PrintExpr(CodegenOperands(op)[1], os);
    os << ") & static_cast<uint32_t>(";
    PrintExpr(CodegenOperands(op)[0], os);
    os << ")))";
  } else if (IsCodegenOp(op, "tl.ballot")) {
    ICHECK_EQ(CodegenOperands(op).size(), 1U)
        << "tl.ballot expects <predicate>.";
    os << "((uint64_t)asc_ballot(";
    PrintExpr(CodegenOperands(op)[0], os);
    os << "))";
  } else if (IsCodegenOp(op, "tl.activemask")) {
    ICHECK(CodegenOperands(op).empty()) << "tl.activemask takes no arguments.";
    os << "((uint64_t)asc_activemask())";
  } else if (IsCodegenOp(op, "tl.ascend_get_buf")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3)
        << "tl.ascend_get_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str =
        (CodegenOperands(op)[0]).as_or_throw<StringImm>()->value;
    int64_t mode = (CodegenOperands(op)[2])
                       .as_or_throw<IntImm>()
                       ->value.as<int64_t>()
                       .value();
    this->PrintIndent();
    this->stream << "asc_lock(" << pipe_str << ", ";
    this->PrintExpr(CodegenOperands(op)[1], this->stream);
    // ASC_LOCK_BLOCK is the C API default; emit it explicitly for readability.
    this->stream << (mode ? ", ASC_LOCK_NON_BLOCK" : ", ASC_LOCK_BLOCK");
    this->stream << ");\n";
  } else if (IsCodegenOp(op, "tl.ascend_rls_buf")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3)
        << "tl.ascend_rls_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str =
        (CodegenOperands(op)[0]).as_or_throw<StringImm>()->value;
    int64_t mode = (CodegenOperands(op)[2])
                       .as_or_throw<IntImm>()
                       ->value.as<int64_t>()
                       .value();
    this->PrintIndent();
    this->stream << "asc_unlock(" << pipe_str << ", ";
    this->PrintExpr(CodegenOperands(op)[1], this->stream);

    // ASC_LOCK_BLOCK is the C API default; emit it explicitly for readability.
    this->stream << (mode ? ", ASC_LOCK_NON_BLOCK" : ", ASC_LOCK_BLOCK");
    this->stream << ");\n";
  } else if (IsCodegenOp(op, "tl.ascend_set_hf32_mode")) {
    ICHECK_EQ(CodegenOperands(op).size(), 1)
        << "tl.ascend_set_hf32_mode expects exactly 1 argument (mode_int)";
    int mode = (CodegenOperands(op)[0])
                   .as_or_throw<IntImm>()
                   ->value.as<int64_t>()
                   .value();
    this->PrintIndent();
    if (mode == 0) {
      this->stream << "asc_disable_hf32();\n";
    } else {
      this->stream << "asc_enable_hf32();\n";
      this->PrintIndent();
      this->stream << "asc_set_hf32_round_mode("
                   << (mode == 1 ? "asc_hf32_round_mode::NEAREST_AWAY"
                                 : "asc_hf32_round_mode::NEAREST_EVEN")
                   << ");\n";
    }
  } else if (IsCodegenOp(op, "tl.ascend_set_mmad_direction")) {
    ICHECK_EQ(CodegenOperands(op).size(), 1);
    const auto *direction = CodegenOperands(op)[0].as<StringImmObj>();
    ICHECK(direction && (direction->value == "m" || direction->value == "n"))
        << "tl.ascend_set_mmad_direction expects 'm' or 'n'";
    this->PrintIndent();
    this->stream << "asc_set_mmad_direction_" << direction->value << "();\n";
  } else if (IsCodegenOp(op, "tl.ascend_set_atomic")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2)
        << "tl.ascend_set_atomic expects 2 arguments (op_str, typed_zero)";
    std::string atomic_op =
        (CodegenOperands(op)[0]).as_or_throw<StringImm>()->value;
    const char *fn = atomic_op == "max"   ? "asc_set_atomic_max"
                     : atomic_op == "min" ? "asc_set_atomic_min"
                                          : "asc_set_atomic_add";
    ICHECK(atomic_op == "add" || atomic_op == "max" || atomic_op == "min")
        << "tl.ascend_set_atomic op must be add/max/min, got " << atomic_op;

    PrimType atomic_dtype = CodegenDType(CodegenOperands(op)[1]);
    bool supported_dtype =
        atomic_dtype.MatchesElementType(kDLBfloat, 16) ||
        (atomic_dtype.MatchesCode(kDLFloat) &&
         (atomic_dtype.bits() == 16 || atomic_dtype.bits() == 32)) ||
        (atomic_dtype.MatchesCode(kDLInt) &&
         (atomic_dtype.bits() == 8 || atomic_dtype.bits() == 16 ||
          atomic_dtype.bits() == 32));
    ICHECK(supported_dtype)
        << "Ascend atomic does not support data type: " << atomic_dtype;

    this->PrintIndent();
    this->stream << fn;
    if (atomic_dtype.MatchesElementType(kDLBfloat, 16)) {
      this->stream << "_bfloat";
    } else if (atomic_dtype.MatchesCode(kDLFloat) &&
               atomic_dtype.bits() == 16) {
      this->stream << "_float16";
    } else if (atomic_dtype.MatchesCode(kDLFloat) &&
               atomic_dtype.bits() == 32) {
      this->stream << "_float";
    } else if (atomic_dtype.MatchesCode(kDLInt) && atomic_dtype.bits() == 8) {
      this->stream << "_int8";
    } else if (atomic_dtype.MatchesCode(kDLInt) && atomic_dtype.bits() == 16) {
      this->stream << "_int16";
    } else if (atomic_dtype.MatchesCode(kDLInt) && atomic_dtype.bits() == 32) {
      this->stream << "_int32";
    }

    this->stream << "();\n";
  } else if (IsCodegenOp(op, "tl.ascend_set_atomic_none")) {
    ICHECK_EQ(CodegenOperands(op).size(), 0)
        << "tl.ascend_set_atomic_none expects no arguments";
    this->PrintIndent();
    this->stream << "asc_set_atomic_none();\n";
  } else if (IsCodegenOp(op, "tl.warp_reduce_sum") ||
             IsCodegenOp(op, "tl.warp_reduce_max") ||
             IsCodegenOp(op, "tl.warp_reduce_min")) {
    ICHECK_EQ(CodegenOperands(op).size(), 1U)
        << "Ascend hardware warp reduction expects exactly one argument";
    PrimType dtype = CodegenDType(CodegenOperands(op)[0]);
    ICHECK(IsAscendWarpReduceDType(dtype))
        << "Ascend hardware warp reduction supports only scalar float16, "
           "float32, int32, and uint32 values, but got "
        << dtype;
    const char *intrinsic =
        IsCodegenOp(op, "tl.warp_reduce_sum")
            ? "asc_reduce_add"
            : (IsCodegenOp(op, "tl.warp_reduce_max") ? "asc_reduce_max"
                                                     : "asc_reduce_min");
    os << intrinsic << "(" << PrintExpr(CodegenOperands(op)[0]) << ")";
  } else if (IsCodegenOp(op, "tl.ascend_cross_core_set_flag") ||
             IsCodegenOp(op, "tl.ascend_cross_core_wait_flag")) {
    EmitCrossCoreSync_(op, IsCodegenOp(op, "tl.ascend_cross_core_set_flag"));
  } else if (IsCodegenOp(op, "tirx.address_of")) {
    const auto *load = CodegenOperands(op)[0].as<TensorLoadObj>();
    ICHECK(CodegenOperands(op).size() == 1 && load);
    const auto *buf_var = load->source.as_or_throw<TensorVar>().get();
    std::string scope;
    if (alloc_storage_scope_.count(buf_var)) {
      scope = alloc_storage_scope_.at(buf_var).name();
    }
    bool needs_ptr_arith = (scope.find("l0") != std::string::npos ||
                            scope.find("l1") != std::string::npos);
    // SimdVF: `local` vector-register array (alloc_local). access_ptr lowering
    // expands the 64-lane element access into a Ramp index; recover the element
    // index (ramp base / lanes) and take the address of that register slot.
    bool is_local_vreg = IsInsideSimdVF() && scope == "local" &&
                         load->indices.size() == 1 &&
                         load->indices[0].ty().lanes() > 1;
    // Packed fp4 (float4_e2m1fnx2) held in GM/UBuf: emitted as the 1-byte
    // `float4_e2m1x2_t`. Handle only after the local-vreg / L0-L1 cases, since
    // a fp4 SIMD register slot must still use the register-index addressing
    // below.
    bool is_fp4_ptr =
        CodegenDType(load).MatchesElementType(kDLFloat4_e2m1fn, 4) &&
        load->indices.size() == 1 && !is_local_vreg && !needs_ptr_arith;
    if (is_local_vreg) {
      PrimExpr index = load->indices[0];
      ValidateSimdVectorType(CodegenDType(load),
                             "SimdVF local register array element");
      const auto *ramp = index.as<prim::RampObj>();
      PrimExpr base = ramp ? ramp->base : index;
      int lanes = index.ty().lanes();
      PrimExpr elem_idx =
          analyzer_->Simplify(floordiv(base, MakeConst(base.ty(), lanes)));
      os << "(&(" << GetVarID(buf_var) << "[";
      PrintExpr(elem_idx, os);
      os << "]))";
    } else if (needs_ptr_arith) {
      os << "((";
      PrintStorageScope(runtime::MemoryScope(scope), os);
      PrintType(CodegenDType(load), os);
      os << "*)" << GetVarID(buf_var);
      if (load->indices.size() == 1) {
        PrimExpr idx = load->indices[0];
        // LowerIntrin represents an fp4x2 access as a 2-lane Ramp and scales
        // its packed-byte offset by the lane count.  L0/L1 pointers need the
        // same scalar-byte recovery as GM/UB pointers below; otherwise this
        // prints invalid pointer arithmetic such as `ptr + int2(...)`.
        if (CodegenDType(load).MatchesElementType(kDLFloat4_e2m1fn, 4) &&
            idx.ty().lanes() > 1) {
          const auto *ramp = idx.as<prim::RampObj>();
          ICHECK(ramp) << "Expected a Ramp index for packed fp4 L0/L1 access, "
                       << "got " << idx;
          idx = analyzer_->Simplify(floordiv(
              ramp->base, MakeConst(ramp->base.ty(), idx.ty().lanes())));
        }
        if (!IsZero(idx)) {
          os << " + ";
          PrintExpr(idx, os);
        }
      }
      os << ")";
    } else if (is_fp4_ptr) {
      // Packed fp4 (float4_e2m1fnx2): the tvm_access_ptr lowering scaled the
      // offset by lanes and wrapped a Ramp (treating the packed pair as a
      // 2-wide vector). Recover the scalar byte index (ramp base / lanes) and
      // emit a plain 1-byte pointer `(&(((scope float4_e2m1x2_t*)var)[byte]))`.
      std::string fp4_scope =
          !scope.empty()
              ? scope
              : CodegenStorageScope(load->source.as_or_throw<TensorVar>());
      PrimExpr index = load->indices[0];
      const auto *ramp = index.as<prim::RampObj>();
      PrimExpr base = ramp ? ramp->base : index;
      int lanes = index.ty().lanes();
      PrimExpr byte_idx =
          lanes > 1
              ? analyzer_->Simplify(floordiv(base, MakeConst(base.ty(), lanes)))
              : base;
      os << "(&(((";
      PrintStorageScope(runtime::MemoryScope(fp4_scope), os);
      os << "float4_e2m1x2_t*)" << GetVarID(buf_var) << ")[";
      PrintExpr(byte_idx, os);
      os << "]))";
    } else {
      CodeGenC::Dispatch_(op, os);
    }
  }
  // --- Scalar GM dcache bypass intrinsics ---
  else if (IsCodegenOp(op, "tl.ascend_read_gm_bypass_dcache")) {
    // ascend_read_gm_bypass_dcache(address_of(BufferLoad))
    ICHECK_EQ(CodegenOperands(op).size(), 1);
    const auto *addr = CodegenOperands(op)[0].as<CallObj>();
    ICHECK(addr && IsCodegenOp(addr, "tirx.address_of"));
    const auto *load = addr->args[0].as<TensorLoadObj>();
    ICHECK(load && load->indices.size() == 1);
    os << "tl::read_gm_bypass_dcache("
       << GetGmBypassPtr(CodegenDType(load), load->source.as<VarObj>(),
                         load->indices[0])
       << ")";
  } else if (IsCodegenOp(op, "tl.ascend_write_gm_bypass_dcache")) {
    // ascend_write_gm_bypass_dcache(address_of(BufferLoad), value)
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    const auto *addr = CodegenOperands(op)[0].as<CallObj>();
    ICHECK(addr && IsCodegenOp(addr, "tirx.address_of"));
    const auto *load = addr->args[0].as<TensorLoadObj>();
    ICHECK(load && load->indices.size() == 1);
    // Evaluate ptr and value to strings first. The string PrintExpr overload
    // flushes any SSA statements (e.g. from if_then_else) to this->stream
    // before returning the expression text, so they land ahead of the call
    // line instead of being injected into the argument list.
    std::string ptr = GetGmBypassPtr(
        CodegenDType(load), load->source.as<VarObj>(), load->indices[0]);
    std::string value = this->PrintExpr(CodegenOperands(op)[1]);
    this->PrintIndent();
    this->stream << "tl::write_gm_bypass_dcache(" << ptr << ", " << value
                 << ");\n";
  }
  // --- SimdVF vector/scalar FMA: (dst, src0, src1, mask, mode) ---
  else if (IsCodegenOp(op, "tl.simd.vmula") ||
           IsCodegenOp(op, "tl.simd.vmadd") ||
           IsCodegenOp(op, "tl.simd.vaxpy")) {
    ICHECK_EQ(CodegenOperands(op).size(), 5);
    PrintSimdCall_(GetRef<Call>(op), 4, os);
  }
  // --- SimdVF histogram: (dst, src, mask, bin) ---
  else if (IsCodegenOp(op, "tl.simd.dhistv2") ||
           IsCodegenOp(op, "tl.simd.chistv2")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    const char *name =
        IsCodegenOp(op, "tl.simd.dhistv2") ? "dhistv2" : "chistv2";
    os << "simd_inst::" << name << "(";
    PrintExpr(CodegenOperands(op)[0], os); // dst
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // src
    os << ", ";
    PrintExpr(CodegenOperands(op)[2], os); // mask
    const auto *bin = CodegenOperands(op)[3].as<IntImmObj>();
    ICHECK(bin) << name << " requires bin to be the integer 0 or 1";
    ICHECK(bin->value == 0 || bin->value == 1)
        << name << " only supports bin=0 or bin=1, but got " << bin->value;
    os << ", Bin_N" << bin->value;
    os << ")";
  }
  // --- SimdVF carry arithmetic: (src0, src1, mask) -> (carry, result) ---
  else if (IsCodegenOp(op, "tl.simd.vaddc") ||
           IsCodegenOp(op, "tl.simd.vsubc")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF carry family: sub/add-with-carry-in -> (carry, result) ---
  else if (IsCodegenOp(op, "tl.simd.vaddcs") ||
           IsCodegenOp(op, "tl.simd.vsubcs")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 4, os);
  }
  // --- SimdVF widening multiply: (src0, src1, mask) -> (lo, hi) ---
  else if (IsCodegenOp(op, "tl.simd.vmull")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF leaky/parametric relu ---
  else if (IsCodegenOp(op, "tl.simd.vlrelu") ||
           IsCodegenOp(op, "tl.simd.vprelu")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF runtime tail predicate: (value, width) ---
  else if (IsCodegenOp(op, "tl.simd.update_mask")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    int width = (int)(CodegenOperands(op)[1])
                    .as_or_throw<IntImm>()
                    ->value.as<int64_t>()
                    .value();
    os << "simd_inst::update_mask_b" << width << "(";
    PrintExpr(CodegenOperands(op)[0], os); // value
    os << ")";
  }
  // --- SimdVF predicate pack/unpack: (src, part) ---
  else if (IsCodegenOp(op, "tl.simd.ppack") ||
           IsCodegenOp(op, "tl.simd.punpack")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 1, os);
  }
  // --- SimdVF predicate interleave/deinterleave: (src0, src1, width) -> pair
  // ---
  else if (IsCodegenOp(op, "tl.simd.pintlv") ||
           IsCodegenOp(op, "tl.simd.pdintlv")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    const char *name = IsCodegenOp(op, "tl.simd.pintlv") ? "pintlv" : "pdintlv";
    int width = (int)(CodegenOperands(op)[2])
                    .as_or_throw<IntImm>()
                    ->value.as<int64_t>()
                    .value();
    os << "simd_inst::" << name << "_b" << width << "(";
    PrintExpr(CodegenOperands(op)[0], os); // src0
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // src1
    os << ")";
  }
  // --- SimdVF widen unpack: (src, part) ---
  else if (IsCodegenOp(op, "tl.simd.vunpack")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 1, os);
  }
  // --- SimdVF unsqueeze: (mask) -> prefix count ---
  else if (IsCodegenOp(op, "tl.simd.vusqz")) {
    ICHECK_EQ(CodegenOperands(op).size(), 1);
    PrimType elem = CodegenDType(op).WithLanes(1);
    os << "simd_inst::vusqz<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(CodegenOperands(op)[0], os); // mask
    os << ")";
  }
  // --- SimdVF vector/scalar binary ops: (src0, src1, mask, mode) ---
  else if (IsCodegenOp(op, "tl.simd.vadd") || IsCodegenOp(op, "tl.simd.vsub") ||
           IsCodegenOp(op, "tl.simd.vmul") ||
           IsCodegenOp(op, "tl.simd.vabsdif") ||
           IsCodegenOp(op, "tl.simd.vmax") || IsCodegenOp(op, "tl.simd.vmin") ||
           IsCodegenOp(op, "tl.simd.vdiv") || IsCodegenOp(op, "tl.simd.vand") ||
           IsCodegenOp(op, "tl.simd.vor") || IsCodegenOp(op, "tl.simd.vxor") ||
           IsCodegenOp(op, "tl.simd.vshl") || IsCodegenOp(op, "tl.simd.vshr") ||
           IsCodegenOp(op, "tl.simd.vadds") ||
           IsCodegenOp(op, "tl.simd.vmuls") ||
           IsCodegenOp(op, "tl.simd.vmaxs") ||
           IsCodegenOp(op, "tl.simd.vmins") ||
           IsCodegenOp(op, "tl.simd.vshls") ||
           IsCodegenOp(op, "tl.simd.vshrs")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF predicate ops ---
  else if (IsCodegenOp(op, "tl.simd.pand") || IsCodegenOp(op, "tl.simd.por") ||
           IsCodegenOp(op, "tl.simd.pxor") || IsCodegenOp(op, "tl.simd.pnot") ||
           IsCodegenOp(op, "tl.simd.psel")) {
    ICHECK(CodegenOperands(op).size() ==
           (IsCodegenOp(op, "tl.simd.pnot") ? 2 : 3));
    PrintSimdCall_(GetRef<Call>(op), CodegenOperands(op).size(), os);
  }
  // --- SimdVF unary ops: (src, mask, mode) ---
  else if (IsCodegenOp(op, "tl.simd.vexp") || IsCodegenOp(op, "tl.simd.vln") ||
           IsCodegenOp(op, "tl.simd.vsqrt") ||
           IsCodegenOp(op, "tl.simd.vabs") || IsCodegenOp(op, "tl.simd.vneg") ||
           IsCodegenOp(op, "tl.simd.vrelu") ||
           IsCodegenOp(op, "tl.simd.vnot")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF cross-lane reduce/permutation ---
  else if (IsCodegenOp(op, "tl.simd.vcpadd") ||
           IsCodegenOp(op, "tl.simd.vcmax") ||
           IsCodegenOp(op, "tl.simd.vcmin") ||
           IsCodegenOp(op, "tl.simd.vcadd") ||
           IsCodegenOp(op, "tl.simd.vcgadd") ||
           IsCodegenOp(op, "tl.simd.vcgmax") ||
           IsCodegenOp(op, "tl.simd.vcgmin") ||
           IsCodegenOp(op, "tl.simd.vsqz")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF index ramp: (index, order) -> vci<T>(index, order) ---
  else if (IsCodegenOp(op, "tl.simd.vci")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrimType elem = CodegenDType(op).WithLanes(1);
    os << "simd_inst::vci<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(CodegenOperands(op)[0], os); // index
    os << ", " << (CodegenOperands(op)[1]).as_or_throw<StringImm>()->value
       << ")"; // order
  }
  // --- SimdVF compare: (src0, src1, mask, op) -> vcmp_<op>(...) ---
  else if (IsCodegenOp(op, "tl.simd.vcmp")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    os << "simd_inst::vcmp_"
       << (CodegenOperands(op)[3]).as_or_throw<StringImm>()->value << "(";
    PrintExpr(CodegenOperands(op)[0], os); // src0
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // src1
    os << ", ";
    PrintExpr(CodegenOperands(op)[2], os); // mask
    os << ")";
  }
  // --- SimdVF compare-scalar: (src, scalar, mask, op) -> vcmps_<op>(...) ---
  else if (IsCodegenOp(op, "tl.simd.vcmps")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    os << "simd_inst::vcmps_"
       << (CodegenOperands(op)[3]).as_or_throw<StringImm>()->value << "(";
    PrintExpr(CodegenOperands(op)[0], os); // src
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // scalar
    os << ", ";
    PrintExpr(CodegenOperands(op)[2], os); // mask
    os << ")";
  }
  // --- SimdVF register permutation: (src0, src1) -> pair ---
  else if (IsCodegenOp(op, "tl.simd.vintlv") ||
           IsCodegenOp(op, "tl.simd.vdintlv")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF pair element access: pair_get(pair, index) -> pair.vN ---
  else if (IsCodegenOp(op, "tl.simd.pair_get")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrintExpr(CodegenOperands(op)[0], os); // pair
    int idx = (int)(CodegenOperands(op)[1])
                  .as_or_throw<IntImm>()
                  ->value.as<int64_t>()
                  .value();
    os << (idx == 0 ? ".v0" : ".v1");
  }
  // --- SimdVF register pack: (src, part) ---
  else if (IsCodegenOp(op, "tl.simd.vpack")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrimType elem = CodegenDType(op).WithLanes(1);
    os << "simd_inst::vpack<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(CodegenOperands(op)[0], os); // src
    os << ", " << (CodegenOperands(op)[1]).as_or_throw<StringImm>()->value
       << ")";
  }
  // --- SimdVF vector broadcast vdupv: (src, mask, pos, mode) ---
  else if (IsCodegenOp(op, "tl.simd.vdupv")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF gather 32B blocks: (base, index [, mask]) ---
  else if (IsCodegenOp(op, "tl.simd.vgatherb")) {
    ICHECK(CodegenOperands(op).size() == 2 || CodegenOperands(op).size() == 3);
    PrimType elem = CodegenDType(op).WithLanes(1);
    os << "simd_inst::vgatherb<";
    PrintType(elem, os);
    os << ">((" << "__ubuf__" << " ";
    PrintType(elem, os);
    os << "*)";
    PrintExpr(CodegenOperands(op)[0], os); // base
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // index
    if (CodegenOperands(op).size() == 3) {
      os << ", ";
      PrintExpr(CodegenOperands(op)[2], os); // mask
    }
    os << ")";
  }
  // --- SimdVF gather elements: (base, index, mask) ---
  else if (IsCodegenOp(op, "tl.simd.vgather2")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    // No explicit template arg / pointer cast: the base already prints as a
    // typed `__ubuf__ ELEM*`, so overload resolution picks the widening
    // uint8_t/int8_t intrinsic (uint8 -> uint16) from the source pointer type.
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF predicate load: (addr, dist) ---
  else if (IsCodegenOp(op, "tl.simd.pld")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    std::string dist = (CodegenOperands(op)[1]).as_or_throw<StringImm>()->value;
    static const std::unordered_map<std::string, std::string> kPldFn = {
        {"NORM", "plds_norm"},
        {"US", "plds_upsample"},
        {"DS", "plds_downsample"},
    };
    auto it = kPldFn.find(dist);
    ICHECK(it != kPldFn.end())
        << "Unsupported SimdVF predicate load distribution: " << dist;
    os << "simd_inst::" << it->second << "((__ubuf__ uint32_t*)";
    PrintExpr(CodegenOperands(op)[0], os); // addr
    os << ", 0)";
  }
  // --- SimdVF predicate store: (addr, src, dist) ---
  else if (IsCodegenOp(op, "tl.simd.pst")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    std::string addr = this->PrintExpr(CodegenOperands(op)[0]);
    std::string src = this->PrintExpr(CodegenOperands(op)[1]);
    std::string dist = (CodegenOperands(op)[2]).as_or_throw<StringImm>()->value;
    static const std::unordered_map<std::string, std::string> kPstFn = {
        {"NORM", "psts_norm"},
        {"PK", "psts_pack"},
    };
    auto it = kPstFn.find(dist);
    ICHECK(it != kPstFn.end())
        << "Unsupported SimdVF predicate store distribution: " << dist;
    this->PrintIndent();
    this->stream << "simd_inst::" << it->second << "(" << src
                 << ", (__ubuf__ uint32_t*)" << addr << ", 0);\n";
  }
  // --- SimdVF load: (addr, dist [, post-update increment]) ---
  else if (IsCodegenOp(op, "tl.simd.vld")) {
    ICHECK(CodegenOperands(op).size() == 2 || CodegenOperands(op).size() == 3);
    bool post_update = CodegenOperands(op).size() == 3;
    PrimType elem = CodegenDType(op).WithLanes(1);
    std::string dist = (CodegenOperands(op)[1]).as_or_throw<StringImm>()->value;
    // Each load distribution maps to a dedicated single-purpose entry point;
    // the element width (B8/B16/B32) is carried by the type argument, so the
    // width-suffixed dist names collapse onto one function per family.
    static const std::unordered_map<std::string, std::string> kVldFn = {
        {"NORM", "vlds_norm"},
        {"NORM_B8", "vlds_norm"},
        {"NORM_B16", "vlds_norm"},
        {"NORM_B32", "vlds_norm"},
        {"BRC_B8", "vlds_brc_elem"},
        {"BRC_B16", "vlds_brc_elem"},
        {"BRC_B32", "vlds_brc_elem"},
        {"US_B8", "vlds_upsample"},
        {"US_B16", "vlds_upsample"},
        {"DS_B8", "vlds_downsample"},
        {"DS_B16", "vlds_downsample"},
        {"UNPK_B8", "vlds_unpack"},
        {"UNPK_B16", "vlds_unpack"},
        {"UNPK_B32", "vlds_unpack"},
        {"UNPK4_B8", "vlds_unpack4"},
        {"BLK", "vlds_brc_datablock"},
        {"E2B_B16", "vlds_brc_elem2datablock"},
        {"E2B_B32", "vlds_brc_elem2datablock"},
    };
    auto it = kVldFn.find(dist);
    ICHECK(it != kVldFn.end())
        << "Unsupported SimdVF load distribution: " << dist;
    os << "simd_inst::" << it->second << (post_update ? "_postupdate" : "")
       << "<" << CCEUBufType(elem) << ">((__ubuf__ " << CCEUBufType(elem)
       << "*)";
    PrintExpr(CodegenOperands(op)[0], os);
    os << ", ";
    if (post_update) {
      PrintExpr(CodegenOperands(op)[2], os);
    } else {
      os << "0";
    }
    os << ")";
  }
  // --- SimdVF dual-dest load: (addr, dist [, offset]) -> vec_pair ---
  else if (IsCodegenOp(op, "tl.simd.vld2")) {
    ICHECK(CodegenOperands(op).size() >= 2 && CodegenOperands(op).size() <= 3);
    PrimType elem = CodegenDType(op).WithLanes(1);
    std::string dist = (CodegenOperands(op)[1]).as_or_throw<StringImm>()->value;
    bool has_off = (CodegenOperands(op).size() == 3);
    os << "simd_inst::vld_x2<" << CCEUBufType(elem) << ">((" << "__ubuf__"
       << " " << CCEUBufType(elem) << "*)";
    PrintExpr(CodegenOperands(op)[0], os); // addr
    os << ", ";
    if (has_off) {
      os << "asc_update_addr_reg_b" << CCEElemWidthBits(elem) << "(";
      PrintExpr(CodegenOperands(op)[2], os);
      os << "), " << dist;
    } else {
      os << dist;
    }
    os << ")";
  }
  // --- SimdVF store: (addr, src, mask, dist) ---
  else if (IsCodegenOp(op, "tl.simd.vsts")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    PrimType dtype = CodegenDType(CodegenOperands(op)[1]).WithLanes(1);
    std::string dist = (CodegenOperands(op)[3]).as_or_throw<StringImm>()->value;
    // Each store distribution maps to a dedicated single-purpose entry point;
    // the element width (B8/B16/B32) is carried by the type argument, so the
    // width-suffixed dist names collapse onto one function per family.
    static const std::unordered_map<std::string, std::string> kVstFn = {
        {"NORM_B8", "vsts_norm"},         {"NORM_B16", "vsts_norm"},
        {"NORM_B32", "vsts_norm"},        {"ONEPT_B8", "vsts_1st"},
        {"ONEPT_B16", "vsts_1st"},        {"ONEPT_B32", "vsts_1st"},
        {"PK_B16", "vsts_pack_b16"},      {"PK_B32", "vsts_pack_b32"},
        {"PK4_B32", "vsts_pack_quarter"}, {"INTLV_B8", "vsts_intlv"},
        {"INTLV_B16", "vsts_intlv"},      {"INTLV_B32", "vsts_intlv"},
    };
    auto it = kVstFn.find(dist);
    ICHECK(it != kVstFn.end())
        << "Unsupported SimdVF store distribution: " << dist;
    // Capture operands first so SSA temporaries (e.g. from reinterpret) are
    // emitted on their own lines before the call statement.
    std::string src = this->PrintExpr(CodegenOperands(op)[1]);
    std::string addr = this->PrintExpr(CodegenOperands(op)[0]);
    std::string mask = this->PrintExpr(CodegenOperands(op)[2]);
    this->PrintIndent();
    this->stream << "simd_inst::" << it->second << "(" << src << ", (__ubuf__ "
                 << CCEUBufType(dtype) << "*)" << addr << ", 0, " << mask
                 << ");\n";
  }

  // --- SimdVF dual-source interleaved store: (addr, src0, src1) ---
  else if (IsCodegenOp(op, "tl.simd.vst2")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrimType dtype = CodegenDType(CodegenOperands(op)[1]).WithLanes(1);
    std::string src0 = this->PrintExpr(CodegenOperands(op)[1]);
    std::string src1 = this->PrintExpr(CodegenOperands(op)[2]);
    std::string addr = this->PrintExpr(CodegenOperands(op)[0]);
    this->PrintIndent();
    this->stream << "simd_inst::vst_x2(" << src0 << ", " << src1
                 << ", (__ubuf__ " << CCEUBufType(dtype) << "*)" << addr
                 << ");\n";
  }

  // --- SimdVF scatter-store blocks: (src, base, stride, mask [, post]) ---
  else if (IsCodegenOp(op, "tl.simd.vsstb")) {
    ICHECK(CodegenOperands(op).size() == 4 || CodegenOperands(op).size() == 5);
    bool post_update = CodegenOperands(op).size() == 5;
    ICHECK_EQ(op->ty.as<PtrTypeObj>() != nullptr, post_update);
    PrimType dtype = CodegenDType(CodegenOperands(op)[0]).WithLanes(1);
    // Capture operands first so SSA temporaries (e.g. from reinterpret) are
    // emitted on their own lines before the call statement.
    std::string src = this->PrintExpr(CodegenOperands(op)[0]);
    std::string base = this->PrintExpr(CodegenOperands(op)[1]);
    std::string stride = this->PrintExpr(CodegenOperands(op)[2]);
    std::string mask = this->PrintExpr(CodegenOperands(op)[3]);
    if (post_update) {
      // Keep POST_UPDATE as an expression so the frontend can write the
      // advanced pointer back into its mutable handle carrier.
      std::string post =
          (CodegenOperands(op)[4]).as_or_throw<StringImm>()->value;
      ICHECK_EQ(post, "POST_UPDATE");
      os << "simd_inst::vsstb(" << src << ", (__ubuf__ " << CCEUBufType(dtype)
         << "*)" << base << ", " << stride << ", " << mask << ", " << post
         << ")";
    } else {
      this->PrintIndent();
      this->stream << "simd_inst::vsstb(" << src << ", (__ubuf__ "
                   << CCEUBufType(dtype) << "*)" << base << ", " << stride
                   << ", " << mask << ");\n";
    }
  }
  // --- SimdVF scatter-store: (src, base, index, mask) ---
  else if (IsCodegenOp(op, "tl.simd.vscatter")) {
    ICHECK_EQ(CodegenOperands(op).size(), 4);
    PrimType dtype = CodegenDType(CodegenOperands(op)[0]).WithLanes(1);
    // Capture operands first so SSA temporaries (e.g. from reinterpret) are
    // emitted on their own lines before the call statement.
    std::string src = this->PrintExpr(CodegenOperands(op)[0]);
    std::string base = this->PrintExpr(CodegenOperands(op)[1]);
    std::string index = this->PrintExpr(CodegenOperands(op)[2]);
    std::string mask = this->PrintExpr(CodegenOperands(op)[3]);
    this->PrintIndent();
    this->stream << "simd_inst::vscatter(" << src << ", (__ubuf__ "
                 << CCEUBufType(dtype) << "*)" << base << ", " << index << ", "
                 << mask << ");\n";
  }
  // --- SimdVF dup: scalar broadcast ---
  else if (IsCodegenOp(op, "tl.simd.vdup")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrimType elem = CodegenDType(op).WithLanes(1);
    os << "simd_inst::vdup<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(CodegenOperands(op)[0], os); // scalar
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // mask
    os << ", " << (CodegenOperands(op)[2]).as_or_throw<StringImm>()->value
       << ")"; // mode
  }
  // --- SimdVF cast: (src, mask, ...extra) ---
  else if (IsCodegenOp(op, "tl.simd.vcvt")) {
    ICHECK(CodegenOperands(op).size() >= 4 && CodegenOperands(op).size() <= 6);
    PrimType elem = CodegenDType(op).WithLanes(1);
    os << "simd_inst::vcvt<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(CodegenOperands(op)[0], os); // src
    os << ", ";
    PrintExpr(CodegenOperands(op)[1], os); // mask
    for (int i = 2; i < static_cast<int>(CodegenOperands(op).size()); ++i) {
      os << ", " << (CodegenOperands(op)[i]).as_or_throw<StringImm>()->value;
    }
    os << ")";
  }
  // --- SimdVF vexpdif/select: (src0, src1, mask) ---
  else if (IsCodegenOp(op, "tl.simd.vexpdif") ||
           IsCodegenOp(op, "tl.simd.vsel")) {
    ICHECK_EQ(CodegenOperands(op).size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF select by lane index: (src, index) ---
  else if (IsCodegenOp(op, "tl.simd.vselr")) {
    ICHECK_EQ(CodegenOperands(op).size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  } else if (IsCodegenOp(op, "tl.simd.mem_bar")) {
    std::string mem_type_str =
        (CodegenOperands(op)[0]).as_or_throw<StringImm>()->value;
    this->PrintIndent();
    this->stream << "simd_inst::mem_bar(" << mem_type_str << ");\n";
  } else if (IsCodegenOp(op, "tl.loop_break")) {
    this->PrintIndent();
    this->stream << "break;\n";
    // --- Atomic ops: emit asc_atomic_{add,max,min}(ptr, val) ---
  } else if (IsCodegenOp(op, "tl.atomic_add_elem_op") ||
             IsCodegenOp(op, "tl.atomic_max_elem_op") ||
             IsCodegenOp(op, "tl.atomic_min_elem_op") ||
             IsCodegenOp(op, "tl.atomic_add_ret_elem_op") ||
             IsCodegenOp(op, "tl.atomic_max_ret_elem_op") ||
             IsCodegenOp(op, "tl.atomic_min_ret_elem_op")) {
    ICHECK(CodegenOperands(op).size() == 2 || CodegenOperands(op).size() == 3)
        << "Ascend scalar atomic ops expect (ptr, value[, memory_order])";
    const char *fn = nullptr;
    bool is_ret = false;
    if (IsCodegenOp(op, "tl.atomic_add_elem_op") ||
        IsCodegenOp(op, "tl.atomic_add_ret_elem_op"))
      fn = "add";
    else if (IsCodegenOp(op, "tl.atomic_max_elem_op") ||
             IsCodegenOp(op, "tl.atomic_max_ret_elem_op"))
      fn = "max";
    else
      fn = "min";
    is_ret = IsCodegenOp(op, "tl.atomic_add_ret_elem_op") ||
             IsCodegenOp(op, "tl.atomic_max_ret_elem_op") ||
             IsCodegenOp(op, "tl.atomic_min_ret_elem_op");
    auto emit_ptr_and_val = [&](std::ostream &out) {
      // args[0] is address_of(BufferLoad); its codegen already emits a
      // correctly scoped, element-typed, offset pointer.
      out << "asc_atomic_" << fn << "(" << PrintExpr(CodegenOperands(op)[0])
          << ", ";
      PrintExpr(CodegenOperands(op)[1], out);
      // Ascend SIMT atomic APIs only accept (ptr, value). The optional
      // cross-backend memory-order argument is intentionally ignored.
      out << ")";
    };
    if (is_ret) {
      emit_ptr_and_val(os);
    } else {
      this->PrintIndent();
      emit_ptr_and_val(this->stream);
      this->stream << ";\n";
    }
  } else if (IsCodegenOp(op, "tl.rng_init")) {
    // Ascend SIMT Philox RNG. Must be used inside a T.SimtVF(...) block so it
    // compiles into a __simt_vf__ helper. args: seed, seq, off, generator.
    ICHECK(IsInsideSimtVF())
        << "tl.rng_init on Ascend must be used inside a T.SimtVF(...) block";
    ascend_rng_state_var_ = name_supply_->FreshName("__asc_rng_state");
    this->PrintIndent();
    this->stream << "tl::AscendPhiloxState " << ascend_rng_state_var_ << ";\n";
    this->PrintIndent();
    this->stream << "tl::philox_init(&" << ascend_rng_state_var_ << ", "
                 << PrintExpr(CodegenOperands(op)[0]) << ", "
                 << PrintExpr(CodegenOperands(op)[1]) << ", "
                 << PrintExpr(CodegenOperands(op)[2]) << ");\n";
    // args[3] (curand generator string) is ignored on Ascend.
  } else if (IsCodegenOp(op, "tl.rng_rand")) {
    ICHECK(!ascend_rng_state_var_.empty())
        << "tl.rng_rand requires a preceding T.rng_init call in the same "
           "T.SimtVF() block";
    os << "tl::philox_rand(&" << ascend_rng_state_var_ << ")";
  } else if (IsCodegenOp(op, "tl.rng_rand_float")) {
    ICHECK(!ascend_rng_state_var_.empty())
        << "tl.rng_rand_float requires a preceding T.rng_init call in the "
           "same T.SimtVF() block";
    ICHECK_NE(CodegenDType(op).bits(), 64)
        << "float64 RNG (tl.rng_rand_float bit=64) is not supported on Ascend";
    const std::string dist = CodegenOperands(op)[0].as<StringImmObj>()->value;
    ICHECK(dist == "uniform" || dist == "normal")
        << "Unsupported RNG distribution on Ascend: " << dist;
    os << "tl::philox_rand_" << dist << "(&" << ascend_rng_state_var_ << ")";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenTileLangAscend::Dispatch_(const prim::NotObj *op,
                                      std::ostream &os) {
  if (CodegenDType(op).lanes() > 1) {
    RejectGenericVectorInSimd("Not", CodegenDType(op));
    if (CodegenIsVectorBool(CodegenDType(op)) &&
        CodegenDType(op).lanes() <= 8) {
      EmitSimtVectorPredicateNot(op->a, CodegenDType(op), os);
      return;
    }
  }
  CodeGenC::Dispatch_(op, os);
}

bool CodeGenTileLangAscend::IsGlobalGmBuffer(const VarObj *buffer) const {
  const VarObj *buf_var = buffer;
  auto it = alloc_storage_scope_.find(buf_var);
  std::string scope =
      (it != alloc_storage_scope_.end())
          ? std::string(it->second.name())
          : std::string(CodegenStorageScope(ffi::GetRef<Var>(buffer)));
  return scope == "global";
}

std::string CodeGenTileLangAscend::GetGmBypassPtr(PrimType dtype,
                                                  const VarObj *buffer,
                                                  PrimExpr index) {
  const VarObj *buf_var = buffer;
  std::ostringstream os;
  os << "((";
  PrintStorageScope(runtime::MemoryScope("global"), os);
  PrintType(dtype, os);
  os << "*)" << GetVarID(buf_var) << " + ";
  PrintExpr(index, os);
  os << ")";
  return os.str();
}

void CodeGenTileLangAscend::Dispatch_(const TensorLoadObj *op,
                                      std::ostream &os) {
  if (op->indices.size() == 1) {
    PrimExpr index = NormalizeAscendVectorIndex(op->indices[0], analyzer_);
    if (!index.same_as(op->indices[0])) {
      Dispatch_(TensorLoad(op->source, {index}, op->loc).get(), os);
      return;
    }
  }
  PrimType value_dtype = CodegenDType(op);
  const VarObj *buf_var = op->source.as<VarObj>();

  // For Ascend: scalar FP8 loads from shared/UBUF memory need conversion
  // through raw bytes, because the scalar FP8 type (tl::float_e4m3_t) is a
  // struct whose operator= does not support __ubuf__ to local assignment.
  if (value_dtype.IsScalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      const std::string scope = it->second.name();
      if (scope == "shared" || scope == "shared.dyn") {
        // __ubuf__ does not support custom struct pointer
        ICHECK_EQ(op->indices.size(), 1);
        PrimExpr index = op->indices[0];
        std::string buf_name = GetVarID(buf_var);

        // tl::float_e4m3_t::from_bits(*(__ubuf__ uint8_t*)(buf + off))
        os << GetAscendFP8ScalarValueType(value_dtype) << "::from_bits("
           << "*(";
        PrintStorageScope(runtime::MemoryScope(scope), os);
        os << "uint8_t*)(" << buf_name;
        if (!IsZero(index)) {
          os << " + ";
          PrintExpr(index, os);
        }
        os << "))";
        return;
      }
    }
  }

  // emit a real vector load when provably aligned, otherwise a brute-force
  // per-lane scalar gather.
  PrimType element_dtype = op->source.as_or_throw<TensorVar>().dtype();
  if (value_dtype.lanes() > 1 && value_dtype.lanes() != element_dtype.lanes() &&
      op->indices.size() == 1) {
    EmitScalarizedLoad(op, os);
    return;
  }

  CodeGenC::Dispatch_(op, os);
}

bool CodeGenTileLangAscend::VectorAccessIsAligned(const PrimExpr &index,
                                                  int lanes) {
  // Only a unit-stride ramp can become a reinterpret-cast vector access.
  tvm::PVar<PrimExpr> base;
  if (!tvm::ramp(base, 1, lanes).Match(index)) {
    return false;
  }
  // Prove the ramp base is divisible by the vector width.
  PrimExpr ramp_base = base.Eval();
  PrimExpr lanes_const = MakeConst(ramp_base.ty(), lanes);
  return analyzer_->CanProveEqual(floormod(ramp_base, lanes_const),
                                  prim::MakeConst(ramp_base.ty(), 0));
}

void CodeGenTileLangAscend::EmitScalarizedLoad(const TensorLoadObj *op,
                                               std::ostream &os) {
  PrimType value_dtype = CodegenDType(op);
  ValidateGenericVectorType("BufferLoad", value_dtype);
  int lanes = value_dtype.lanes();
  PrimExpr index = op->indices[0];

  // A vectorized GM access on the scalar core (outside any VF body) has no
  // valid lowering: the only scalar-pipe GM path is the per-element
  // read_gm_bypass_dcache emitted in VisitExpr_. Vector GM traffic must go
  // through an MTE copy (T.copy) or a SimtVF/SimdVF region.
  if (IsOutsideVF() && IsGlobalGmBuffer(op->source.as<VarObj>())) {
    TVM_FFI_THROW(InternalError)
        << "Ascend: cannot vectorize GM access of "
        << op->source.as_or_throw<TensorVar>()->name << " (lanes=" << lanes
        << ") on the scalar core. Scalar-core GM reads must be a single "
           "scalar element (lowered to read_gm_bypass_dcache); use "
           "T.copy or a T.SimtVF/T.SimdVF region for vector GM traffic.";
  }

  // Provably aligned: a single reinterpret-cast vector load (fast path).
  if (VectorAccessIsAligned(index, lanes)) {
    tvm::PVar<PrimExpr> base;
    ICHECK(tvm::ramp(base, 1, lanes).Match(index));
    std::string ref =
        GetVecLoad(value_dtype, op->source.as<VarObj>(), base.Eval());
    HandleVolatileLoads(ref, op, os);
    return;
  }

  // Otherwise scalarize: load each lane from its scalar element offset and pack
  // with a brace-initializer (which bisheng accepts, unlike `T(a, b)`).
  PrimType elem_dtype = value_dtype.WithLanes(1);
  const prim::RampObj *ramp = index.as<prim::RampObj>();
  std::ostringstream init;
  PrintType(value_dtype, init);
  init << "{";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) {
      init << ", ";
    }
    PrimExpr lane_index =
        ramp ? (ramp->base + ramp->stride * MakeConst(ramp->base.ty(), i))
             : index;
    std::string lane_ref = GetBufferRef(elem_dtype, op->source.as<VarObj>(),
                                        analyzer_->Simplify(lane_index));
    init << lane_ref;
  }
  init << "}";
  os << init.str();
}

void CodeGenTileLangAscend::Dispatch_(const TensorStoreObj *op) {
  if (op->indices.size() == 1) {
    PrimExpr index = NormalizeAscendVectorIndex(op->indices[0], analyzer_);
    if (!index.same_as(op->indices[0])) {
      Dispatch_(TensorStore(op->dest, {index}, op->value, op->loc).get());
      return;
    }
  }
  PrimType value_dtype = op->value.ty();
  PrimType element_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  const VarObj *buf_var = op->dest.as<VarObj>();
  PrimExpr index_expr = op->indices[0];

  // A vector store that stays on the scalar core (outside any VF body) cannot
  // use the simt-only vector constructors a broadcast needs (make_float2,
  // make_int2, ...); bisheng rejects them outside a VF body. Such a store is
  // element-wise scalar work, so emit one scalar store per lane.
  if (IsOutsideVF() && value_dtype.lanes() > 1 &&
      value_dtype.lanes() != element_dtype.lanes() && op->indices.size() == 1 &&
      !IsGlobalGmBuffer(op->dest.as<VarObj>()) &&
      EmitOutOfVFBroadcastStore(op)) {
    return;
  }

  if (value_dtype.lanes() > 1 && op->indices.size() == 1) {
    const auto *broadcast_index = index_expr.as<prim::BroadcastObj>();
    auto scope_it = alloc_storage_scope_.find(buf_var);
    std::string buffer_scope = op->dest.as_or_throw<TensorVar>().scope().name();
    bool is_local_scope = buffer_scope == "local" ||
                          buffer_scope == "local.var" ||
                          (scope_it != alloc_storage_scope_.end() &&
                           (scope_it->second.name() == "local" ||
                            scope_it->second.name() == "local.var"));
    if (broadcast_index && is_local_scope &&
        value_dtype.lanes() == CodegenDType(broadcast_index).lanes()) {
      std::string buf_name = GetVarID(buf_var);
      std::string value = PrintExpr(op->value);

      PrintIndent();
      stream << "*(";
      PrintType(value_dtype, stream);
      stream << "*)(" << buf_name;
      if (!IsZero(broadcast_index->value)) {
        stream << " + ";
        PrintExpr(broadcast_index->value, stream);
      }
      stream << ") = " << value << ";\n";
      return;
    }
  }

  // For Ascend: scalar FP8 stores to shared/UBUF memory need to write the raw
  // .data byte, because tl::float_e4m3_t::operator= does not accept a local
  // struct on a __ubuf__ lvalue target.
  if (value_dtype.IsScalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      const std::string scope = it->second.name();
      if (scope == "shared" || scope == "shared.dyn") {
        // __ubuf__ does not support custom struct pointer
        ICHECK_EQ(op->indices.size(), 1);
        std::string buf_name = GetVarID(buf_var);
        std::string value = PrintExpr(op->value);

        // ((__ubuf__ uint8_t*)buf)[off] = value.data;
        PrintIndent();
        stream << "((";
        PrintStorageScope(runtime::MemoryScope(scope), stream);
        stream << "uint8_t*)" << buf_name << ")[" << PrintExpr(index_expr)
               << "] = " << value << ".data;\n";
        return;
      }
    }
  }

  if (value_dtype.lanes() > 1 && value_dtype.lanes() != element_dtype.lanes() &&
      op->indices.size() == 1) {
    if (EmitScalarizedStore(op)) {
      return;
    }
  }

  CodeGenC::Dispatch_(op);
}

bool CodeGenTileLangAscend::EmitOutOfVFBroadcastStore(
    const TensorStoreObj *op) {
  const auto *broadcast = op->value.as<prim::BroadcastObj>();
  const auto *ramp = op->indices[0].as<prim::RampObj>();
  if (broadcast == nullptr || ramp == nullptr || !IsOne(ramp->stride) ||
      op->dest.as_or_throw<TensorVar>().dtype().lanes() != 1) {
    return false;
  }
  const int lanes = CodegenDType(broadcast).lanes();
  PrimType elem_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  SSAOperationScope ssa_scope(this);
  std::string value =
      SSAGetID(PrintExpr(broadcast->value), elem_dtype,
               tvm::SideEffect(broadcast->value) == CallEffectKind::kPure);
  for (int i = 0; i < lanes; ++i) {
    PrimExpr lane_index =
        analyzer_->Simplify(ramp->base + MakeConst(ramp->base.ty(), i));
    PrintIndent();
    stream << GetBufferRef(elem_dtype, op->dest.as<VarObj>(), lane_index)
           << " = " << value << ";\n";
  }
  return true;
}

bool CodeGenTileLangAscend::EmitScalarizedStore(const TensorStoreObj *op) {
  PrimType value_dtype = op->value.ty();
  ValidateGenericVectorType("BufferStore", value_dtype);
  int lanes = value_dtype.lanes();
  PrimExpr index = op->indices[0];

  // A vectorized GM access on the scalar core (outside any VF body) has no
  // valid lowering: the only scalar-pipe GM path is the per-element
  // write_gm_bypass_dcache emitted in VisitStmt_. Vector GM traffic must go
  // through an MTE copy (T.copy) or a SimtVF/SimdVF region.
  if (IsOutsideVF() && IsGlobalGmBuffer(op->dest.as<VarObj>())) {
    TVM_FFI_THROW(InternalError)
        << "Ascend: cannot vectorize GM access of "
        << op->dest.as_or_throw<TensorVar>()->name << " (lanes=" << lanes
        << ") on the scalar core. Scalar-core GM writes must be a single "
           "scalar element (lowered to write_gm_bypass_dcache); use "
           "T.copy or a T.SimtVF/T.SimdVF region for vector GM traffic.";
  }

  // Provably aligned: let the base class emit a single reinterpret vector
  // store.
  if (VectorAccessIsAligned(index, lanes)) {
    return false;
  }

  // Otherwise scalarize: evaluate the value once, then store each lane to its
  // scalar element offset. PrintVecElemLoad reads lane i from the value temp;
  // no vector index is constructed.
  const prim::RampObj *ramp = index.as<prim::RampObj>();
  ICHECK(ramp) << "scalarized store expects a ramp index, got " << index;
  PrimType elem_dtype = value_dtype.WithLanes(1);
  SSAOperationScope ssa_scope(this);
  std::string value =
      SSAGetID(PrintExpr(op->value), value_dtype,
               tvm::SideEffect(op->value) == CallEffectKind::kPure);
  for (int i = 0; i < lanes; ++i) {
    PrimExpr lane_index =
        ramp->base + ramp->stride * MakeConst(ramp->base.ty(), i);
    std::string lane_ref = GetBufferRef(elem_dtype, op->dest.as<VarObj>(),
                                        analyzer_->Simplify(lane_index));
    std::ostringstream lane_val;
    PrintVecElemLoad(value, value_dtype, i, lane_val);
    PrintIndent();
    stream << lane_ref << " = " << lane_val.str() << ";\n";
  }
  return true;
}

void CodeGenTileLangAscend::Dispatch_(const prim::SelectObj *op,
                                      std::ostream &os) { // NOLINT(*)
  if (!op->condition.ty().IsFixedLengthVector()) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  ValidateGenericVectorType("Select", op->condition.ty());
  ValidateGenericVectorType("Select", CodegenDType(op));

  TVM_FFI_ICHECK(CodegenDType(op->false_value) == CodegenDType(op) &&
                 CodegenDType(op->true_value) == CodegenDType(op) &&
                 CodegenDType(op).lanes() == op->condition.ty().lanes());

  std::string result = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(CodegenDType(op), stream);
  stream << ' ' << result << ";\n";
  SSAOperationScope ssa_scope(this);
  {
    std::string condition =
        SSAGetID(PrintExpr(op->condition), op->condition.ty(),
                 tvm::SideEffect(op->condition) == CallEffectKind::kPure);
    std::string true_value =
        SSAGetID(PrintExpr(op->true_value), CodegenDType(op),
                 tvm::SideEffect(op->true_value) == CallEffectKind::kPure);
    std::string false_value =
        SSAGetID(PrintExpr(op->false_value), CodegenDType(op),
                 tvm::SideEffect(op->false_value) == CallEffectKind::kPure);

    int lanes = CodegenDType(op).lanes();
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value;
      value << "(bool(";
      PrintVecElemLoad(condition, op->condition.ty(), i, value);
      value << ")?";
      PrintVecElemLoad(true_value, CodegenDType(op), i, value);
      value << ':';
      PrintVecElemLoad(false_value, CodegenDType(op), i, value);
      value << ')';
      PrintVecElemStore(result, CodegenDType(op), i, value.str());
    }
  }
  os << result;
}

void CodeGenTileLangAscend::Dispatch_(const prim::ShuffleObj *op,
                                      std::ostream &os) { // NOLINT(*)
  for (const PrimExpr &vector : op->vectors) {
    if (vector.ty().lanes() > 1) {
      ValidateGenericVectorType("Shuffle", vector.ty());
    }
  }
  if (CodegenDType(op).lanes() > 1) {
    ValidateGenericVectorType("Shuffle", CodegenDType(op));
  }
  SSAOperationScope ssa_scope(this);

  // Materialize the concatenated input lanes once before applying the shuffle
  // indices. Bisheng does not accept the generic C codegen's vector
  // constructors such as float2(a, b), and packed FP16/BF16 vectors require
  // element-wise access through their uint carrier.
  std::vector<std::string> input_lanes;
  for (const PrimExpr &vector : op->vectors) {
    std::string vector_value =
        SSAGetID(PrintExpr(vector), vector.ty(),
                 tvm::SideEffect(vector) == CallEffectKind::kPure);
    if (vector.ty().IsScalar()) {
      input_lanes.push_back(std::move(vector_value));
      continue;
    }
    for (int i = 0; i < vector.ty().lanes(); ++i) {
      std::ostringstream element;
      PrintVecElemLoad(vector_value, vector.ty(), i, element);
      input_lanes.push_back(element.str());
    }
  }

  std::vector<std::string> output_lanes;
  output_lanes.reserve(op->indices.size());
  for (const PrimExpr &index : op->indices) {
    const auto *constant_index = index.as<IntImmObj>();
    TVM_FFI_ICHECK(constant_index != nullptr)
        << "ShuffleObj indices must be constants at Ascend codegen time, got "
        << index;
    TVM_FFI_ICHECK_GE(constant_index->value, 0)
        << "ShuffleObj index must be non-negative, got "
        << constant_index->value;
    TVM_FFI_ICHECK_LT(static_cast<size_t>(CodegenInt(constant_index)),
                      input_lanes.size())
        << "ShuffleObj index " << constant_index->value
        << " is outside the concatenated input with " << input_lanes.size()
        << " lanes";
    output_lanes.push_back(input_lanes[CodegenInt(constant_index)]);
  }

  if (CodegenDType(op).IsScalar()) {
    TVM_FFI_ICHECK_EQ(output_lanes.size(), 1);
    os << output_lanes[0];
    return;
  }

  TVM_FFI_ICHECK_EQ(static_cast<int>(output_lanes.size()),
                    CodegenDType(op).lanes());
  std::string result = name_supply_->FreshName("_");
  PrintIndent();
  PrintType(CodegenDType(op), stream);
  stream << ' ' << result << ";\n";
  for (int i = 0; i < CodegenDType(op).lanes(); ++i) {
    PrintVecElemStore(result, CodegenDType(op), i, output_lanes[i]);
  }
  os << result;
}

void CodeGenTileLangAscend::Dispatch_(const ForObj *op) {
  if (op->kind == ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll";
    // Native loop-local annotations take precedence over an enclosing hint.
    if (auto factor = op->annotations.Get("pragma_unroll_factor")) {
      const auto *literal = factor->as<IntImmObj>();
      stream << " "
             << (literal ? CodegenInt(literal) : factor->cast<int64_t>());
    } else if (auto it = unroll_factor_.find(op->var);
               it != unroll_factor_.end()) {
      stream << " " << PrintExpr(it->second);
    }
    stream << "\n";
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangAscend::Dispatch_(const WhileObj *op) {
  if (IsOne(op->condition)) {
    PrintIndent();
    stream << "while (1) {\n";
    int while_scope = BeginScope();
    PrintStmt(op->body);
    this->EndScope(while_scope);
    PrintIndent();
    stream << "}\n";
    return;
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangAscend::Dispatch_(const RegionStmtObj *op) {
  if (op->op.same_as(tirx::launch_thread_op())) {
    TVM_FFI_CHECK(op->args.size() == 2 && op->body_params.size() == 1 &&
                      op->result_vars.empty(),
                  ValueError);
    const auto var = op->body_params[0].as_or_throw<PrimVar>();
    const auto tag = op->args[0].as_or_throw<StringImm>()->value;
    auto prior = var_idmap_.find(var.get());
    std::optional<std::string> saved;
    if (prior != var_idmap_.end())
      saved = prior->second;
    if (tag == "blockIdx.x")
      var_idmap_[var.get()] = "block_idx";
    else if (tag == "cthread") {
      // One native launch binder denotes one subcore index. Retain that
      // definition across all copies and VF captures in its lexical body.
      const std::string name = name_supply_->FreshName(var->name);
      var_idmap_[var.get()] = name;
      PrintIndent();
      PrintType(var.ty(), stream);
      stream << " " << name << " = asc_get_sub_block_id();\n";
    } else if (tag == "threadIdx.x" || tag == "threadIdx.y" ||
               tag == "threadIdx.z") {
      if (IsOutsideVF()) {
        // The outer hardware launch is materialized by asc_vf_call. A thread
        // used outside an actual VF cannot be replaced by a fictitious zero.
        for (const Var &free : tvm::UndefinedVars(op->body))
          TVM_FFI_CHECK(!free.same_as(var), ValueError)
              << "Ascend thread index used outside VF";
      } else
        var_idmap_[var.get()] = tag;
    } else
      TVM_FFI_THROW(ValueError) << "Unsupported Ascend thread tag: " << tag;
    PrintStmt(op->body);
    if (saved)
      var_idmap_[var.get()] = *saved;
    else
      var_idmap_.erase(var.get());
    return;
  }
  if (DecodeAscendBlock(op))
    return EmitBlockRegion_(op);
  if (op->op->name == "tl.attr") {
    const auto attr = DecodeTileLangAttribute(op);
    if (attr.key == "tilelang_assume" || attr.key == "tl.assume") {
      tl::ScopedAnalyzerConstraints context(analyzer_,
                                            attr.node.cast<PrimExpr>());
      PrintStmt(op->body);
      return;
    }
    if (attr.key == "tl.simtvf_scope" || attr.key == "tl.simdvf_scope") {
      PrintStmt(op->body);
      return;
    }
    if (attr.key == "pragma_unroll_factor") {
      const auto var = attr.node.cast<PrimVar>();
      const int64_t factor = AscendAnnotationInt(attr.value);
      TVM_FFI_CHECK_GT(factor, 0, ValueError);
      unroll_factor_.insert_or_assign(var, IntImm(PrimType::Int(32), factor));
      PrintStmt(op->body);
      unroll_factor_.erase(var);
      return;
    }
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangAscend::EmitTensorAllocation_(const BindObj *op) {
  const TensorVar tensor = op->var.as_or_throw<TensorVar>();
  const auto *call = op->value.as<CallObj>();
  std::string scope = CodegenStorageScope(tensor);

  if (scope == "shared") {
    TVM_FFI_THROW(InternalError)
        << "Static shared memory (scope='shared') is not supported on "
           "Ascend. "
        << "Use dynamic shared memory (scope='shared.dyn') with "
           "T.alloc_shared.";
  }

  // SimdVF: emit vector_<T> for fragment buffers
  if (IsInsideSimdVF() && scope == "local.fragment") {
    std::string vid = AllocVarID(tensor.get());
    this->PrintIndent();
    stream << "vector_";
    PrintCCEVectorSuffix(tensor.dtype(), stream);
    stream << " " << vid << ";\n";
    alloc_storage_scope_[tensor.get()] = runtime::MemoryScope(scope);
    RegisterHandleType(tensor.get(), tensor.dtype());
    return;
  }

  std::string qualifier = GetAscendScopeQualifier(scope);
  if (!qualifier.empty()) {
    std::string vid = AllocVarID(tensor.get());
    // Generate: __qualifier__ <type> *buf = (__qualifier__ <type> *)0;
    PrimType dtype = tensor.dtype();
    this->PrintIndent();
    stream << qualifier << " ";
    PrintType(dtype, stream);
    stream << " *" << vid << " = (" << qualifier << " ";
    PrintType(dtype, stream);
    if (IsAscendSS(tensor.get())) {
      ICHECK(kernel_mode_ == AscendKernelMode::kMix && mix_aiv_count_ == 2)
          << "SS buffers require a MIX kernel with two AIVs";
      auto elements = std::optional<int64_t>(CodegenAllocationElements(tensor));
      ICHECK(elements.has_value() && elements.value() > 0 &&
             static_cast<size_t>(elements.value()) <=
                 3072 / dtype.StorageBytes())
          << "SS allocations require a static total size of at most 3072 bytes";
    }
    // SS, like L1, starts at zero in its own address space. This is the
    // dav-3510 device value returned by AscendC::GetSsbufBaseAddr().
    stream << " *)0;\n";
    alloc_storage_scope_[tensor.get()] = runtime::MemoryScope(scope);
    RegisterHandleType(tensor.get(), dtype);
    return;
  }

  // SimdVF: emit vector_<T> for local.var vector register temps.
  // CCE vector register variables should be declared uninitialized
  // since they are always written before read via SIMD operations.
  // Without this, zero-init via Broadcast expands to make_vector_f32(...)
  // with one argument per lane, which is both verbose and unnecessary.
  if (IsInsideSimdVF() && scope == "local.var" && tensor.dtype().lanes() > 1) {
    ValidateSimdVectorType(tensor.dtype(), "SimdVF local.var allocation");
    std::string vid = AllocVarID(tensor.get());
    this->PrintIndent();
    PrintType(tensor.dtype(), stream);
    stream << " " << vid << ";\n";
    alloc_storage_scope_[tensor.get()] = runtime::MemoryScope(scope);
    RegisterHandleType(tensor.get(), tensor.dtype());
    return;
  }

  if (scope == "local.var") {
    ICHECK_EQ(CodegenAllocationElements(tensor), 1)
        << "local.var should be scalar";
    PrimType dtype = tensor.dtype();
    std::string vid = AllocVarID(tensor.get());
    ffi::Optional<PrimExpr> init;
    if (!dtype.MatchesCode(kDLOpaqueHandle)) {
      init = prim::MakeConst(dtype, 0);
    }
    if (auto initializer =
            CodegenCallAnnotations(call).Get("tl.local_var_init")) {
      PrimExpr user_init = initializer.value().cast<PrimExpr>();
      if (!user_init.ty().IsVoid() && user_init.ty() != dtype) {
        user_init = prim::Cast(dtype, user_init);
      }
      init = user_init;
    }
    std::string init_value =
        init.has_value() ? PrintExpr(init.value()) : "nullptr";
    PrintIndent();
    if (dtype.MatchesCode(kDLOpaqueHandle)) {
      // Handle local variables are exclusively mutable UB pointers in SimdVF.
      stream << "__ubuf__ void*";
    } else {
      PrintType(dtype, stream);
    }
    stream << " " << vid << " = " << init_value << ";\n";
    alloc_storage_scope_[tensor.get()] = runtime::MemoryScope(scope);
    RegisterHandleType(tensor.get(), dtype);
    return;
  }

  // Native alloc_tensor permits omitted attributes. Supply the empty terminal
  // dictionary expected by the shared C emitter without changing the input IR.
  if (!call->attrs.has_value()) {
    Call allocation = GetRef<Call>(call);
    allocation.CopyOnWrite()->attrs =
        DictAttrs(ffi::Map<ffi::String, ffi::Any>{});
    Bind binding = GetRef<Bind>(op);
    binding.CopyOnWrite()->value = std::move(allocation);
    CodeGenC::Dispatch_(binding.get());
    return;
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangAscend::Dispatch_(const BindObj *op) {
  BindAscendIndexConstant(op, analyzer_);
  if (const auto *call = op->value.as<CallObj>();
      call && call->op.same_as(tirx::alloc_tensor_op()))
    return EmitTensorAllocation_(op);
  // Inside SimdVF body, bindings for vector registers and masks need concrete
  // CCE-friendly types instead of generic C handle printing.
  if (IsInsideSimdVF() && op->var->ty.as<PrimTypeObj>()) {
    if (CodegenDType(op->value).lanes() > 1) {
      ValidateSimdVectorType(CodegenDType(op->value), "SimdVF binding");
    }
    std::string vid = AllocVarID(op->var.get());
    if (op->var->ty.as<PtrTypeObj>() && op->var->ty.as<PtrTypeObj>()) {
      const auto *ptr = op->var->ty.as<PtrTypeObj>();
      std::string scope = CodegenStorageScope(op->var);
      if (scope == "local.fragment") {
        const auto *prim = ptr->element_type.as<PrimTypeObj>();
        ICHECK(prim)
            << "SimdVF local.fragment bind expects primitive element type";
        PrintIndent();
        stream << "vector_";
        PrintCCEVectorSuffix(ffi::GetRef<PrimType>(prim), stream);
        stream << " " << vid << ";\n";
        alloc_storage_scope_[op->var.get()] = runtime::MemoryScope(scope);
        RegisterHandleType(op->var.get(), ffi::GetRef<PrimType>(prim));
        return;
      }
    }
    if (const auto *call = op->value.as<CallObj>()) {
      // simd_pset/pge(elem_width_bits, dist) -> vector_bool vid = p*_bXX(dist);
      if (IsCodegenOp(call, "tl.simd.pset") ||
          IsCodegenOp(call, "tl.simd.pge")) {
        int elem_bits = (int)(CodegenOperands(call)[0])
                            .as_or_throw<IntImm>()
                            ->value.as<int64_t>()
                            .value();
        std::string dist =
            (CodegenOperands(call).size() >= 2 &&
             CodegenOperands(call)[1].as<StringImmObj>())
                ? (CodegenOperands(call)[1]).as_or_throw<StringImm>()->value
                : "PAT_ALL";
        PrintIndent();
        stream << "vector_bool " << vid << " = asc_create_mask_b" << elem_bits
               << "(" << dist << ");\n";
        return;
      }
      // pair_get(pair, idx) -> inline as pair.vN, no auto decl
      if (IsCodegenOp(call, "tl.simd.pair_get")) {
        int idx = (int)(CodegenOperands(call)[1])
                      .as_or_throw<IntImm>()
                      ->value.as<int64_t>()
                      .value();
        std::ostringstream tmp;
        PrintExpr(CodegenOperands(call)[0], tmp);
        tmp << (idx == 0 ? ".v0" : ".v1");
        var_idmap_[op->var.get()] = tmp.str();
        return;
      }
    }
    // Materialize the RHS into a string first: a reinterpret (and other exprs)
    // emit SSA helper statements via SSAGetID directly into `stream`. Printing
    // straight after "auto vid = " would inject those declarations mid-line and
    // produce illegal C++. Building the string first flushes the helper decls
    // onto their own lines before the assignment is written.
    std::string rhs = PrintExpr(op->value);
    PrintIndent();
    stream << "auto " << vid << " = " << rhs << ";\n";
    return;
  }

  // A handle var bound by a LetStmt (e.g. the pointer materialized by
  // T.make_tensor: `src = reinterpret(handle, src_ptrs[0])`) carries its
  // element type and storage scope in `type_annotation`, but the base
  // CodeGenC::VisitStmt_ only inspects the scalar DataType (always "handle")
  // and emits a scope-less `void*`. That loses the `__gm__`/`__ubuf__`
  // qualifier and the element type, so later casts to `__gm__ T*` are rejected
  // by bisheng. Read the PointerType directly and emit the qualified pointer.
  if (op->var->ty.as<PtrTypeObj>()) {
    if (const auto *ptr = op->var->ty.as<PtrTypeObj>()) {
      if (const auto *prim = ptr->element_type.as<PrimTypeObj>()) {
        std::string scope = CodegenStorageScope(op->var);
        // Scoped pointer type, e.g. "__gm__ float *".
        std::ostringstream type_os;
        PrintStorageScope(runtime::MemoryScope(scope), type_os);
        PrintType(ffi::GetRef<PrimType>(prim), type_os);
        type_os << " *";
        std::string ptr_type = type_os.str();

        // A reinterpret from an integer address (the make_tensor case) is
        // emitted by the base codegen as a `void*` bit-cast, which bisheng
        // refuses to convert to a scoped pointer. Emit the bit-cast directly
        // against the scoped pointer type so no `void*` intermediate appears.
        const auto *call = op->value.as<CallObj>();
        if (call && IsCodegenOp(call, "tirx.reinterpret") &&
            CodegenOperands(call)[0]->ty.as<PrimTypeObj>() &&
            !CodegenDType(CodegenOperands(call)[0])
                 .MatchesCode(kDLOpaqueHandle)) {
          SSAOperationScope ssa_scope(this);
          std::string rhs =
              SSAGetID(PrintExpr(CodegenOperands(call)[0]),
                       CodegenDType(CodegenOperands(call)[0]),
                       tvm::SideEffect(CodegenOperands(call)[0]) ==
                           CallEffectKind::kPure);
          std::string vid = AllocVarID(op->var.get());
          PrintIndent();
          stream << ptr_type << vid << " = (*(" << ptr_type << " *)(&(" << rhs
                 << ")));\n";
        } else {
          // Emit the value first: PrintExpr may spill helper temporaries into
          // the stream, which must land before the declaration rather than
          // inside its initializer.
          std::string value = PrintExpr(op->value);
          std::string vid = AllocVarID(op->var.get());
          PrintIndent();
          stream << ptr_type << vid << " = (" << ptr_type << ")" << value
                 << ";\n";
        }
        alloc_storage_scope_[op->var.get()] = runtime::MemoryScope(scope);
        RegisterHandleType(op->var.get(), ffi::GetRef<PrimType>(prim));
        return;
      }
    }
  }
  CodeGenC::Dispatch_(op);
}

inline void PrintConst(const FloatImmObj *op, std::ostream &os,
                       CodeGenTileLangAscend *p) { // NOLINT(*)
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (CodegenDType(op).MatchesElementType(kDLBfloat, 16) ||
      CodegenDType(op).MatchesElementType(kDLFloat, 16) ||
      CodegenDType(op).MatchesCode(kDLFloat)) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      temp << "tl::limits::";
      if (CodegenDType(op).MatchesCode(kDLFloat)) {
        temp << (op->value < 0 ? "kFloatNInf" : "kFloatInf");
      } else if (CodegenDType(op).MatchesElementType(kDLFloat, 16)) {
        temp << (op->value < 0 ? "kHalfNInf" : "kHalfInf");
      } else {
        temp << (op->value < 0 ? "kBf16NInf" : "kBf16Inf");
      }
    } else if (std::isnan(op->value)) {
      temp << "tl::limits::";
      if (CodegenDType(op).MatchesCode(kDLFloat)) {
        temp << "kFloatNaN";
      } else if (CodegenDType(op).MatchesElementType(kDLFloat, 16)) {
        temp << "kHalfNaN";
      } else {
        temp << "kBf16NaN";
      }
    } else {
      p->PrintType(CodegenDType(op), temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (CodegenIsFloat8(CodegenDType(op)) || CodegenIsFloat4(CodegenDType(op))) {
    p->PrintType(CodegenDType(op), os);
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat64/kFloat32 (kFloat16 is handled above)
  switch (CodegenDType(op).bits()) {
  default:
    TVM_FFI_THROW(InternalError)
        << "Bad bit-width for float: " << CodegenDType(op) << "\n";
  }
}

void CodeGenTileLangAscend::Dispatch_(const FloatImmObj *op,
                                      std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscend::Dispatch_(const prim::CastObj *op,
                                      std::ostream &os) { // NOLINT(*)
  PrimType from_ty = op->value.ty();
  PrimType target_ty = CodegenDType(op);
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  if (from_ty.IsScalar()) {
    if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
        tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(";
      PrintExpr(op->value, os);
      os << ")";
      return;
    }

    if (from_ty.MatchesElementType(kDLFloat, 16) &&
        tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (from_ty.MatchesElementType(kDLBfloat, 16) &&
        tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(__bfloat162float(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) &&
        target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32) {
      os << "static_cast<float>(";
      PrintExpr(op->value, os);
      os << ")";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) &&
        target_ty.MatchesElementType(kDLFloat, 16)) {
      os << "static_cast<half>(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) &&
        target_ty.MatchesElementType(kDLBfloat, 16)) {
      os << "__float2bfloat16_rn(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    return CodeGenC::Dispatch_(op, os);
  }

  ValidateGenericVectorType("Cast", from_ty);
  ValidateGenericVectorType("Cast", target_ty);

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  SSAOperationScope ssa_scope(this);
  std::string src =
      SSAGetID(PrintExpr(op->value), from_ty,
               tvm::SideEffect(op->value) == CallEffectKind::kPure);

  int lanes = from_ty.lanes();

  auto PrintVectorizedCast =
      [&](const std::string &cast_func, const std::string &src_type,
          const std::string &dst_type, const std::string &extra_args = "",
          bool src_needs_reinterpret = false,
          bool dst_needs_reinterpret = false) {
        int num_chunks = lanes / 2;
        std::string src_cast = src_needs_reinterpret
                                   ? "reinterpret_cast<" + src_type + "*>"
                                   : "(" + src_type + "*)";
        std::string dst_cast = dst_needs_reinterpret
                                   ? "reinterpret_cast<" + dst_type + "*>"
                                   : "(" + dst_type + "*)";

        for (int i = 0; i < num_chunks; i++) {
          PrintIndent();
          stream << "(" << dst_cast << "(&" << sret << "))[" << i
                 << "] = " << cast_func << "((" << src_cast << "(&" << src
                 << "))[" << i << "]" << extra_args << ");\n";
        }
        os << sret;
      };

  // Handle conversion from float16 to float32.
  if (from_ty.MatchesElementType(kDLFloat, 16) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__half22float2", "half2", "float2");
      return;
    }
  }

  // Handle conversion from float32 to float16.
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesElementType(kDLFloat, 16)) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22half2_rn", "float2", "half2");
      return;
    }
  }

  // Handle conversion from bfloat16 to float32
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__bfloat1622float2", "bfloat16x2_t", "float2", "",
                          /*src_needs_reinterpret=*/true);
      return;
    }
  }

  // Handle conversion from float32 to bfloat16
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesElementType(kDLBfloat, 16)) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22bfloat162_rn", "float2", "bfloat16x2_t", "",
                          /*src_needs_reinterpret=*/false,
                          /*dst_needs_reinterpret=*/true);
      return;
    }
  }

  // Handle conversion from float32 to float8 (E4M3/E5M2)
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
        target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string type_suffix = target_type_is_e4m3 ? "__ASC_E4M3" : "__ASC_E5M2";

    // Use __asc_cvt_float2_to_fp8x2 for vectorized conversion (float2 -> fp8x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      std::string extra_args = std::string(", __ASC_SATFINITE, ") + type_suffix;
      PrintVectorizedCast("__asc_cvt_float2_to_fp8x2", "float2", "uint16_t",
                          extra_args, false, true);
      return;
    }
  }

  // Handle conversion from float16 to float8 (E4M3/E5M2)
  if (from_ty.MatchesElementType(kDLFloat, 16) &&
      tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
        target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string cvt_func = target_type_is_e4m3
                               ? "tl::detail::cast_half2_to_fp8_e4m3x2"
                               : "tl::detail::cast_half2_to_fp8_e5m2x2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, "half2", "uint16_t", "", false, true);
      return;
    }
  }

  // Handle conversion from bfloat16 to float8 (E4M3/E5M2)
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
        target_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string cvt_func = target_type_is_e4m3
                               ? "tl::detail::cast_bfloat162_to_fp8_e4m3x2"
                               : "tl::detail::cast_bfloat162_to_fp8_e5m2x2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, "bfloat16x2_t", "uint16_t", "", true, true);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float32
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.MatchesCode(kDLFloat) &&
      target_ty.bits() == 32) {
    bool from_type_is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                             from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string src_x2_type =
        from_type_is_e4m3 ? "float8_e4m3x2_t" : "float8_e5m2x2_t";
    std::string cvt_func =
        from_type_is_e4m3 ? "__e4m3x22float2" : "__e5m2x22float2";

    // Use Ascend x2 intrinsics for vectorized conversion (fp8x2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, src_x2_type, "float2", "", true, false);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float16
  if (tl::IsAscendVectorizableFP8(from_ty) &&
      target_ty.MatchesElementType(kDLFloat, 16)) {
    bool from_type_is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                             from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string src_x2_type =
        from_type_is_e4m3 ? "float8_e4m3x2_t" : "float8_e5m2x2_t";
    std::string cvt_func = from_type_is_e4m3
                               ? "tl::detail::cast_fp8_e4m3x2_to_half2"
                               : "tl::detail::cast_fp8_e5m2x2_to_half2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, src_x2_type, "half2", "", true, false);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to bfloat16
  if (tl::IsAscendVectorizableFP8(from_ty) &&
      target_ty.MatchesElementType(kDLBfloat, 16)) {
    bool from_type_is_e4m3 = from_ty.MatchesElementType(kDLFloat8_e4m3, 8) ||
                             from_ty.MatchesElementType(kDLFloat8_e4m3fn, 8);
    std::string src_x2_type =
        from_type_is_e4m3 ? "float8_e4m3x2_t" : "float8_e5m2x2_t";
    std::string cvt_func = from_type_is_e4m3
                               ? "tl::detail::cast_fp8_e4m3x2_to_bfloat162"
                               : "tl::detail::cast_fp8_e5m2x2_to_bfloat162";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, src_x2_type, "bfloat16x2_t", "", true,
                          true);
      return;
    }
  }

  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangAscend::Dispatch_(const VarObj *op,
                                      std::ostream &os) { // NOLINT(*)
  // block_idx / blockIdx.* / threadIdx.* map to ASC thread builtins,
  // which are unsigned (uint32). The IR binds them to a signed-int loop var
  // (IterVar forbids a uint var, see materialize_kernel_launch.cc). Without a
  // cast the generated C++ uses uint32 at the reference site, diverging from
  // the IR's int32 dtype and breaking e.g. template argument deduction in
  // tl::write_gm_bypass_dcache. Cast back to the var's IR dtype to keep types
  // consistent. The sanitized VF-parameter form ("blockIdx_x", no dot) is a
  // real int32 parameter and must not be touched.
  auto it = var_idmap_.find(op);
  auto primitive = op->ty.as<PrimType>();
  if (it != var_idmap_.end() && primitive && primitive->MatchesCode(kDLInt) &&
      (it->second == "block_idx" || it->second.rfind("blockIdx.", 0) == 0 ||
       it->second.rfind("threadIdx.", 0) == 0)) {
    os << "((";
    PrintType(CodegenDType(op), os);
    os << ")" << it->second << ")";
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangAscend::PrintStorageSync(const CallObj *op) {
  TVM_FFI_CHECK_EQ(CodegenOperands(op).size(), 1U, ValueError)
      << "Ascend code generation does not support named/partial barriers";
  PrintIndent();
  stream << "asc_syncthreads();\n";
}

void CodeGenTileLangAscend::PrintStorageScope(
    const runtime::MemoryScope &memory_scope, std::ostream &os) {
  const auto scope = memory_scope.name();
  if (scope == "global") {
    os << "__gm__ ";
  } else {
    std::string qualifier = GetAscendScopeQualifier(scope);
    if (!qualifier.empty()) {
      os << qualifier << " ";
    }
  }
}

bool CodeGenTileLangAscend::IsScopePartOfType() const {
  // Ascend requires memory scope qualifiers as part of the pointer type
  return true;
}

void CodeGenTileLangAscend::PrintVecElemLoad(const std::string &vec,
                                             const PrimType &t, int i,
                                             std::ostream &os) {
  if (t.IsScalar()) {
    os << vec;
    return;
  }
  ICHECK(!IsInsideSimdVF())
      << "Ascend SIMD register " << t
      << " is not addressable through SIMT per-lane codegen";
  ValidateGenericVectorType("per-lane load", t);
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < t.lanes() && i < 8)
      << "Ascend " << t << " small-vector lanes max 8" << ": i=" << i
      << " t.bits()=" << t.bits() << " t.lanes()=" << t.lanes();
  if (t.MatchesElementType(kDLFloat, 16)) {
    os << "((half2*)(&(" << vec << ")))[" << (i / 2) << "]." << access[i % 2];
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    os << "((bfloat16x2_t*)(&(" << vec << ")))[" << (i / 2) << "]."
       << access[i % 2];
  } else if (tl::IsAscendVectorizableFP8(t)) {
    // Lanes are packed one byte each, so byte i is lane i.
    os << GetAscendFP8ScalarValueType(t) << "::from_bits(((uint8_t*)(&(" << vec
       << ")))[" << i << "])";
  } else if (CodegenIsVectorBool(t) && t.lanes() > 4) {
    os << "((ushort2*)(&(" << vec << ")))[" << (i / 2) << "]." << access[i % 2];
  } else if (t.lanes() <= 4) {
    os << vec << "." << access[i];
  } else {
    os << "((" << vec << ")." << access[i / 2] << ")";
  }
}

void CodeGenTileLangAscend::PrintVecElemStore(const std::string &vec,
                                              const PrimType &t, int i,
                                              const std::string &value) {
  if (t.IsScalar()) {
    PrintIndent();
    stream << vec << " = " << value << ";\n";
    return;
  }
  ICHECK(!IsInsideSimdVF())
      << "Ascend SIMD register " << t
      << " is not addressable through SIMT per-lane codegen";
  ValidateGenericVectorType("per-lane store", t);
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < t.lanes() && i < 8)
      << "Ascend " << t << " small-vector lanes max 8" << ": i=" << i
      << " t.bits()=" << t.bits() << " t.lanes()=" << t.lanes();
  if (t.MatchesElementType(kDLFloat, 16)) {
    stream << "((half2*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    stream << "((bfloat16x2_t*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (tl::IsAscendVectorizableFP8(t)) {
    // Round arithmetic results to fp8; copying an fp8 value preserves its bits.
    stream << "((uint8_t*)(&(" << vec << ")))[" << i
           << "] = " << GetAscendFP8ScalarValueType(t) << "(" << value
           << ").data;\n";
  } else if (CodegenIsVectorBool(t) && t.lanes() > 4) {
    stream << "((ushort2*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.lanes() <= 4) {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  } else {
    stream << "((" << vec << ")." << access[i / 2] << ") = " << value << ";\n";
  }
}

std::string CodeGenTileLangAscend::GetBufferRef(const PrimType &t,
                                                const VarObj *buffer,
                                                PrimExpr index) {
  if (IsAscendSS(buffer)) {
    ICHECK(vf_mode_ == VFMode::kNone && t.MatchesCode(kDLUInt) &&
           (t.bits() == 32 || t.bits() == 64) && t.lanes() == 1)
        << "SS supports scalar uint32/uint64 accesses outside VF blocks only";
  }
  const VarObj *buffer_var = buffer;
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var).name();
  }
  if (scope == "local.var") {
    return GetVarID(buffer_var);
  }
  return CodeGenC::GetBufferRef(t, buffer, index);
}

void CodeGenTileLangAscend::PrintVecBinaryOp(const std::string &op,
                                             const PrimType &t, PrimExpr lhs,
                                             PrimExpr rhs, std::ostream &os) {
  ValidateGenericVectorType("binary operation", t);
  // Comparisons need per-lane predicates for Select. Scalarize operations
  // without a native vector form and FP16/BF16/FP8 values carried in integers.
  const bool lacks_vector_operation = op == "/" || op == "%" || op == "fmodf" ||
                                      op == "fmod" || op == "min" ||
                                      op == "max";
  bool needs_scalarization =
      CodegenIsVectorBool(t) || (lacks_vector_operation && t.lanes() > 1) ||
      t.MatchesElementType(kDLFloat, 16) ||
      t.MatchesElementType(kDLBfloat, 16) || tl::IsAscendVectorizableFP8(t);
  if (!needs_scalarization) {
    CodeGenC::PrintVecBinaryOp(op, t, lhs, rhs, os);
    return;
  }

  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  SSAOperationScope ssa_scope(this);
  {
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.ty(),
                                tvm::SideEffect(lhs) == CallEffectKind::kPure);
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.ty(),
                                tvm::SideEffect(rhs) == CallEffectKind::kPure);
    for (int i = 0; i < t.lanes(); ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.ty(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.ty(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.ty(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.ty(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  os << sret;
}

void CodeGenTileLangAscend::EmitSimtVectorPredicateNot(const PrimExpr &value,
                                                       PrimType dtype,
                                                       std::ostream &os) {
  ICHECK(CodegenIsVectorBool(dtype) && dtype.lanes() <= 8)
      << "Ascend SIMT predicate not expects a small predicate vector, got "
      << dtype;
  ICHECK_EQ(value.ty(), dtype)
      << "Ascend small-vector predicate not dtype mismatch";

  std::string result = name_supply_->FreshName("_");
  PrintIndent();
  PrintType(dtype, stream);
  stream << ' ' << result << ";\n";
  SSAOperationScope ssa_scope(this);
  std::string operand =
      SSAGetID(PrintExpr(value), value.ty(),
               tvm::SideEffect(value) == CallEffectKind::kPure);
  for (int i = 0; i < dtype.lanes(); ++i) {
    std::ostringstream lane_value;
    lane_value << "(!";
    PrintVecElemLoad(operand, value.ty(), i, lane_value);
    lane_value << ')';
    PrintVecElemStore(result, dtype, i, lane_value.str());
  }
  os << result;
}

void CodeGenTileLangAscend::PrintCallExtern(Type ret_type,
                                            ffi::String global_symbol,
                                            const ffi::Array<Expr> &args,
                                            bool skip_first_arg,
                                            std::ostream &os) {
  PrimType ret_dtype = ret_type.as<PrimType>().value_or(PrimType::Void());
  // LowerIntrin names float32 sqrt "sqrtf" in every execution domain.
  // Select the scalar-core implementation using the existing VF context.
  if (IsOutsideVF() && ret_dtype == PrimType::Float(32) &&
      global_symbol == "sqrtf") {
    global_symbol = "AscendC::Std::sqrt";
    if (!has_ascend_std_math_included_) {
      // The Std headers use AscendC annotations from kernel_operator.h.
      decl_stream << "#include <kernel_operator.h>\n";
      decl_stream << "#include <utils/std/cmath.h>\n";
      has_ascend_std_math_included_ = true;
    }
  }
  if (ret_dtype.IsFixedLengthVector()) {
    ValidateGenericVectorType("external call", ret_dtype);
    // Scalarize: for each lane, extract scalar args, call the scalar
    // function, store result into the result vector lane.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    SSAOperationScope ssa_scope(this);
    {
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val =
            SSAGetID(PrintExpr(args[i]), CodegenDType(args[i]),
                     tvm::SideEffect(args[i]) == CallEffectKind::kPure);
        sargs.push_back(std::move(val));
      }
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], CodegenDType(args[arg_begin + j]), i,
                           scall);
        }
        scall << ")";
        PrintVecElemStore(sret, ret_dtype, i, scall.str());
      }
    }
    os << sret;
    return;
  }

  CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
}

} // namespace codegen
} // namespace tvm
