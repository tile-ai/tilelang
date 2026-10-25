#include "codegen_ascend.h"

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
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include "compat/visit.h"
#include "transform/common/storage_size.h"
#include <tvm/ir/prim/pattern_match.h>

#include "ascend/op/builtin.h"
#include "ascend/op/utils.h"
#include "backend/common/target_utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace codegen {
using ffi::GetRef;
using namespace prim;

namespace {
PrimExpr MakeZero(PrimType type) { return MakeConst(type, 0); }
bool IsOne(const PrimExpr &value) { return tl::GetConstInt(value) == 1; }
bool is_zero(const PrimExpr &value) { return tl::GetConstInt(value) == 0; }
ffi::Map<ffi::String, ffi::Any> Attributes(const CallNode *node) {
  return node->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict;
}
ffi::Map<ffi::String, ffi::Any> Attributes(const Call &node) {
  return Attributes(node.get());
}
ffi::Map<ffi::String, ffi::Any> Attributes(const RegionStmtNode *node) {
  return node->attrs->dict;
}
String RegionName(const RegionStmtNode *node) {
  auto name = node->attrs->dict.Get("name_hint");
  return name ? name->cast<String>() : String("");
}
} // namespace

namespace {

constexpr const char *kModeMerging = "MODE_MERGING";
constexpr const char *kSimdOpPrefix = "tl.simd.";

bool IsSimdOp(const CallNode *op, std::string *op_name = nullptr) {
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

bool IsSimdMergingCall(const CallNode *op,
                       std::string *intrinsic_name = nullptr) {
  if (!IsSimdOp(op) || op->args.empty()) {
    return false;
  }
  auto primitive = op->ty.as<PrimType>();
  const auto *tuple = op->ty.as<TupleTypeNode>();
  if (!(primitive && primitive->IsVoid()) &&
      !(tuple && tuple->fields.empty())) {
    return false;
  }
  const auto *mode = op->args.back().as<StringImmNode>();
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
// dtype supplied by the caller rather than reading
// PythonDType(GetRef<Expr>(op)) here.
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
  if (auto p = Attributes(op).Get("precision")) {
    return PrecisionFromCode(tl::GetIntAnnotation(*p).value());
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

// Map a TVM PrimType to the Ascend CCE vector type suffix (e.g. Float32 ->
// "f32").
std::string CCEVectorSuffix(PrimType dtype) {
  if (dtype.MatchesCode(kDLFloat8_e4m3) || dtype.MatchesCode(kDLFloat8_e4m3fn))
    return "f8e4m3";
  if (dtype.MatchesCode(kDLFloat8_e5m2))
    return "f8e5m2";
  if (dtype.MatchesCode(kDLFloat8_e8m0fnu))
    return "f8e8m0";
  if (dtype.MatchesCode(kDLFloat4_e2m1fn))
    return "f4e2m1x2";
  if (dtype.MatchesCode(kDLFloat)) {
    if (dtype.bits() == 16)
      return "f16";
    if (dtype.bits() == 32)
      return "f32";
    LOG(FATAL) << "Unsupported Ascend SIMD floating-point element type "
               << dtype;
  }
  if (dtype.MatchesElementType(kDLBfloat, 16))
    return "bf16";
  if (dtype.MatchesCode(kDLInt) && (dtype.bits() == 8 || dtype.bits() == 16 ||
                                    dtype.bits() == 32 || dtype.bits() == 64))
    return "s" + std::to_string(dtype.bits());
  if (dtype.MatchesCode(kDLUInt) && (dtype.bits() == 8 || dtype.bits() == 16 ||
                                     dtype.bits() == 32 || dtype.bits() == 64))
    return "u" + std::to_string(dtype.bits());
  LOG(FATAL) << "Unsupported Ascend SIMD vector element type " << dtype;
  return "";
}

// Get element width in bits for choosing b8/b16/b32 predicate/load/store
// variants.
int CCEElemWidthBits(PrimType dtype) {
  if (dtype.MatchesCode(kDLFloat4_e2m1fn))
    return 8;
  return dtype.bits();
}

// Map PrimType to the C++ type name used in __ubuf__ pointer casts.
std::string CCEUBufType(PrimType dtype) {
  if (dtype.MatchesCode(kDLFloat8_e4m3) || dtype.MatchesCode(kDLFloat8_e4m3fn))
    return "float8_e4m3_t";
  if (dtype.MatchesCode(kDLFloat8_e5m2))
    return "float8_e5m2_t";
  if (dtype.MatchesCode(kDLFloat8_e8m0fnu))
    return "float8_e8m0_t";
  if (dtype.MatchesCode(kDLFloat4_e2m1fn))
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
    LOG(FATAL) << "Ascend cross-core sync supports only modes 0, 1, 2, and 4, "
               << "but got " << mode_id;
    return nullptr;
  }
}

int64_t GetIntImmArg(const CallNode *op, size_t index, const char *intrinsic,
                     const char *argument_name) {
  ICHECK_LT(index, op->args.size()) << intrinsic << " is missing argument "
                                    << index << " (" << argument_name << ")";
  const auto *value = op->args[index].as<IntImmNode>();
  ICHECK(value) << intrinsic << " argument " << index << " (" << argument_name
                << ") must be an IntImm, but got " << op->args[index];
  return value->value.as<int64_t>().value();
}

std::string GetStringImmArg(const CallNode *op, size_t index,
                            const char *intrinsic, const char *argument_name) {
  ICHECK_LT(index, op->args.size()) << intrinsic << " is missing argument "
                                    << index << " (" << argument_name << ")";
  const auto *value = op->args[index].as<StringImmNode>();
  ICHECK(value) << intrinsic << " argument " << index << " (" << argument_name
                << ") must be a StringImm, but got " << op->args[index];
  return value->value;
}
} // namespace

CodeGenTileLangAscend::VFModeScope::VFModeScope(CodeGenTileLangAscend *codegen,
                                                VFMode mode)
    : codegen_(codegen), previous_mode_(codegen->vf_mode_) {
  if (previous_mode_ != VFMode::kNone) {
    LOG(FATAL) << "Nested Ascend VF blocks are not supported: cannot enter "
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
      pass_ctx->GetConfig<bool>(tl::kEnableFastMath).value_or(false);
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

void CodeGenTileLangAscend::PrintCCEVectorSuffix(PrimType t, std::ostream &os) {
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
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8) for "
                  "Ascend fp8";
  }
  if (type.MatchesCode(kDLFloat8_e4m3) || type.MatchesCode(kDLFloat8_e4m3fn)) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.MatchesCode(kDLFloat8_e5m2)) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.MatchesCode(kDLFloat8_e8m0fnu)) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported Ascend fp8 type " << type;
  }
  return stream.str();
}

static std::string GetAscendFP8ScalarValueType(PrimType type) {
  if (type.MatchesCode(kDLFloat8_e4m3) || type.MatchesCode(kDLFloat8_e4m3fn)) {
    return "tl::float_e4m3_t";
  }
  if (type.MatchesCode(kDLFloat8_e5m2)) {
    return "tl::float_e5m2_t";
  }
  LOG(FATAL) << "Unsupported Ascend scalar fp8 conversion type " << type;
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
    LOG(FATAL) << context << " expects an Ascend SIMD vector register, got "
               << dtype;
  }
  if (dtype.MatchesCode(kDLBool)) {
    if (dtype.lanes() != 256) {
      LOG(FATAL) << context << " has invalid SIMD predicate type " << dtype
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
    LOG(FATAL) << context << " has invalid SIMD register type " << dtype
               << ": got " << register_bits
               << " bits, but SimdVF requires one full 2048-bit register ("
               << element_dtype << "x" << expected_lanes << ")";
  }
}

void CodeGenTileLangAscend::RejectGenericVectorInSimd(const char *op_name,
                                                      PrimType dtype) const {
  if (IsInsideSimdVF() && dtype.lanes() > 1) {
    LOG(FATAL) << "Generic vector " << op_name << " with dtype " << dtype
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
    if (t.MatchesCode(kDLFloat8_e4m3, kDLFloat8_e4m3fn, kDLFloat8_e5m2,
                      kDLFloat8_e8m0fnu) &&
        (lanes == 2 || lanes == 4 || lanes == 8)) {
      os << GetAscendFP8Type(t);
      return;
    }
    if (t.MatchesCode(kDLFloat4_e2m1fn) && lanes == 2) {
      os << "float4_e2m1x2_t";
      return;
    }
    LOG(FATAL) << VFModeName(vf_mode_) << " cannot convert vector type " << t
               << " to an Ascend type";
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
      LOG(FATAL) << "Cannot convert scalar type " << t << " to Ascend type";
    }
    return;
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    os << "bfloat16_t";
    return;
  } else if (t.MatchesCode(kDLFloat8_e4m3, kDLFloat8_e4m3fn, kDLFloat8_e5m2,
                           kDLFloat8_e8m0fnu)) {
    os << (IsInsideSimtVF() ? GetAscendFP8ScalarValueType(t)
                            : GetAscendFP8Type(t));
    return;
  } else if (t.MatchesCode(kDLFloat4_e2m1fn)) {
    os << "float4_e2m1x2_t";
    return;
  } else if (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) {
    if (t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64) {
      os << (t.MatchesCode(kDLInt) ? "int" : "uint") << t.bits() << "_t";
      return;
    }
  }
  LOG(FATAL) << "Cannot convert scalar type " << t << " to Ascend type";
}

void CodeGenTileLangAscend::AddFunction(const GlobalVar &global,
                                        const tirx::Function &f) {
  ICHECK(IsOutsideVF())
      << "Ascend VF mode leaked across PrimFunc code generation";
  this->InitFuncState(f);
  has_gemm_l0_ = false;
  has_gemm_l1_ = false;
  has_philox_rng_ = false;
  ascend_rng_state_var_.clear();
  sub_block_id_var_.clear();
  ReserveKeywordsAsUnique();
  name_supply_->ReserveName("block_idx");

  // Register function parameters with their storage scope for codegen
  for (const auto &param : f->params) {
    if (PythonIsPointer(param)) {
      auto scope = PythonStorageScope(param);
      alloc_storage_scope_[param.get()] = runtime::MemoryScope(scope);
    }
  }

  // Pre-scan body to determine kernel mode (Mix / Vector / Cube)
  bool has_cube = false, has_vector = false;
  bool has_vector_subcores = false;
  bool has_nd2nz_post_copy = false;
  bool has_cooperative_groups = false;
  bool has_sub_block_id = false;
  int vector_count = 2;
  tl::PostOrderVisit(f->body.value(), [&](const ffi::ObjectRef &n) {
    if (const auto *block = n.as<RegionStmtNode>()) {
      if (block->op.same_as(tirx::launch_thread_op()) &&
          block->args[0].as_or_throw<StringImm>()->value == "cthread") {
        has_sub_block_id = true;
      }
      if (RegionName(block) == "CUBE")
        has_cube = true;
      else if (RegionName(block) == "VECTOR") {
        has_vector = true;
        if (auto opt = block->attrs->dict.Get("vector_count")) {
          has_vector_subcores = true;
          auto count = tl::GetIntAnnotation(opt.value());
          ICHECK(count && (*count == 1 || *count == 2))
              << "Mixed-kernel vector_count must be the constant integer 1 or "
                 "2, got "
              << opt.value();
          vector_count = *count;
        }
      }
    }
    if (const auto *call = n.as<CallNode>()) {
      if (call->op.same_as(tl::ascend_gemm_l1()) ||
          call->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
        has_gemm_l1_ = true;
      }
      if (call->op.same_as(tl::ascend_mad()) ||
          call->op.same_as(tl::ascend_mad_mx())) {
        has_gemm_l0_ = true;
      }
      if (call->op.same_as(tl::ascend_nd2nz_scatter()) ||
          call->op.same_as(tl::ascend_nd2nz_post_copy())) {
        has_nd2nz_copy_ = true;
      }
      if (call->op.same_as(tl::ascend_nd2nz_post_copy())) {
        has_nd2nz_post_copy = true;
      }
      if (call->op.same_as(tl::ascend_read_gm_bypass_dcache()) ||
          call->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
        has_gm_bypass_dcache_ = true;
      }
      if (call->op.same_as(tl::rng_init())) {
        has_philox_rng_ = true;
      }
      if (call->op.same_as(tl::sync_warp())) {
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
  bool no_alias = f->GetAttr(tirx::attr::kNoAlias).value_or(false);

  if (has_sub_block_id ||
      (kernel_mode_ == AscendKernelMode::kMix && mix_aiv_count_ == 1)) {
    sub_block_id_var_ = name_supply_->FreshName("sub_block_id");
  }

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  CodeGenC::PrintExtraAttrs(f, stream);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) {
      stream << ", ";
    }
    if (PythonIsPointer(v)) {
      if (auto *ptr = v->ty.as<PtrTypeNode>()) {
        if (ptr->scope.name() == "grid_constant") {
          stream << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, stream);
          stream << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }

      if (auto tensor = v.as<TensorVar>()) {
        PrintType(tensor->dtype(), stream);
        stream << '*';
        RegisterHandleType(v.get(), tensor->dtype());
      } else {
        CodeGenC::PrintType(v->ty, stream);
      }
      if (auto *ptr = v->ty.as<PtrTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), GetRef<PrimType>(prim));
        }
      }

      if (no_alias) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(v->ty, stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body.value());
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
  ICHECK(IsOutsideVF())
      << "Ascend VF mode leaked after PrimFunc code generation";
}

void CodeGenTileLangAscend::PreFunctionBody(const tirx::Function &f) {
  this->stream << "  asc_init();\n";
  if (!sub_block_id_var_.empty()) {
    stream << "  int32_t " << sub_block_id_var_
           << " = asc_get_sub_block_id();\n";
  }
  // Emit the function-entry #line anchor after the compiler prologue, so the
  // directive's line mapping is not consumed by generated code.
  CodeGenTileLangCBase::PreFunctionBody(f);
}

ffi::Array<Var>
CodeGenTileLangAscend::CollectVFCaptures(const RegionStmtNode *op) const {
  ffi::Array<Var> predefined = op->body_params;

  ffi::Array<Var> undefined = tvm::UndefinedVars(op->body, predefined);
  ffi::Array<Var> captures;
  std::unordered_set<const VarNode *> seen;
  for (const Var &var : undefined) {
    if (seen.count(var.get())) {
      continue;
    }
    // SimtVF helpers are launched via asc_vf_call and retain kernel launch
    // context, so their block index must not be captured. A plain __simd_vf__
    // helper has no such context, so block_idx is passed as a parameter.
    auto it = var_idmap_.find(var.get());
    if (RegionName(op) == "SIMT_VF" && it != var_idmap_.end() &&
        it->second == "block_idx") {
      continue;
    }
    seen.insert(var.get());

    const PrimType dtype =
        PythonIsPointer(var) ? PrimType::Void() : PythonDType(var);
    ICHECK(PythonIsPointer(var) ||
           (dtype.lanes() == 1 &&
            (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt) ||
             dtype.MatchesCode(kDLFloat) || dtype.MatchesCode(kDLBool))))
        << RegionName(op) << " capture variable `" << var
        << "` has unsupported dtype `" << dtype
        << "`. Only scalar int/uint/float/bool/handle are supported.";
    captures.push_back(var);
  }
  return captures;
}

void CodeGenTileLangAscend::EmitVFFunction(const RegionStmtNode *op,
                                           const ffi::Array<Var> &captures,
                                           const std::string &helper_name,
                                           const std::string &func_attrs,
                                           VFMode vf_mode) {
  for (const auto &v : captures) {
    if (PythonIsPointer(v)) {
      alloc_storage_scope_[v.get()] =
          runtime::MemoryScope(PythonStorageScope(v));
      if (auto tensor = v.as<TensorVar>()) {
        RegisterHandleType(v.get(), tensor->dtype());
      } else if (auto pointer = v->ty.as<PtrTypeNode>()) {
        if (auto element_type = pointer->element_type.as<PrimType>()) {
          RegisterHandleType(v.get(), element_type.value());
        }
      }
    }
  }

  std::unordered_map<const VarNode *, std::string> saved_var_idmap;
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
  std::unordered_map<const VarNode *, std::string> vf_param_name;
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
    name_supply_->ReserveName(cap_name, false);
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
    this->PrintStmt(op->body);
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
    if (!PythonIsPointer(v)) {
      PrintType(PythonDType(v), decl_stream);
      decl_stream << " " << vname;
    } else {
      auto scope_it = alloc_storage_scope_.find(v.get());
      if (scope_it != alloc_storage_scope_.end() &&
          scope_it->second.name() == "local.var") {
        auto it = handle_data_type_.find(v.get());
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
        } else {
          PrintType(PythonDType(v), decl_stream);
        }
        decl_stream << " " << vname;
      } else {
        auto it = handle_data_type_.find(v.get());
        if (it == handle_data_type_.end()) {
          for (const auto &kv : handle_data_type_) {
            if (kv.first->name == v->name) {
              it = handle_data_type_.find(kv.first);
              break;
            }
          }
        }
        PrintStorageScope(runtime::MemoryScope(PythonStorageScope(v)),
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

void CodeGenTileLangAscend::Dispatch_(const RegionStmtNode *op) {
  if (op->op.same_as(tirx::launch_thread_op())) {
    auto tag = op->args[0].as_or_throw<StringImm>()->value;
    auto variable = op->body_params[0].as_or_throw<PrimVar>();
    if (tag == "cthread") {
      ICHECK(!sub_block_id_var_.empty());
      var_idmap_[variable.get()] = sub_block_id_var_;
    } else if (tag == "blockIdx.x") {
      var_idmap_[variable.get()] = "block_idx";
    } else if (tag.starts_with("threadIdx.")) {
      if (!IsOutsideVF())
        var_idmap_[variable.get()] = tag;
    } else {
      CodeGenC::BindThreadIndex(variable, tag);
    }
    PrintStmt(op->body);
    return;
  }
  if (RegionName(op) == "CUBE") {
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
  if (RegionName(op) == "VECTOR") {
    if (kernel_mode_ == AscendKernelMode::kMix) {
      PrintIndent();
      stream << "if ASC_IS_AIV {\n";
      int scope = BeginScope();
      if (mix_aiv_count_ == 1) {
        PrintIndent();
        stream << "if (" << sub_block_id_var_ << " == 0) {\n";
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
  if (RegionName(op) == "SIMD_VF") {
    if (!has_simd_inst_included_) {
      preamble_stream_ << "#include <tl_templates/ascend/simd_inst.h>\n";
      has_simd_inst_included_ = true;
    }
    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simdvf_helper_counter_++;
    {
      auto it = Attributes(op).find("tl.vf_source_index");
      if (it != Attributes(op).end()) {
        if (auto *imm = (*it).second.as<IntImmNode>()) {
          vf_idx = imm->value.as<int64_t>().value();
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
  if (RegionName(op) == "SIMT_VF") {
    int64_t thread_x = 1, thread_y = 1, thread_z = 1;
    // Detect the threadIdx.{x,y,z} extents with a visitor rather than assuming
    // the thread_extent AttrStmts lead the block body. Earlier passes (e.g.
    // buffer allocation planning) may emit AllocBuffer/DeclBuffer statements
    // before the thread bindings; a positional scan would then miss them and
    // wrongly fall back to a single thread.
    tl::PostOrderVisit(op->body, [&](const ffi::ObjectRef &node) {
      const auto *region = node.as<RegionStmtNode>();
      if (!region || !region->op.same_as(tirx::launch_thread_op()))
        return;
      auto tag = region->args[0].as_or_throw<StringImm>()->value;
      auto extent = tl::GetConstInt(region->args[1]);
      ICHECK(extent) << "SIMT VF thread extents must be constant";
      if (tag == "threadIdx.x")
        thread_x = *extent;
      else if (tag == "threadIdx.y")
        thread_y = *extent;
      else if (tag == "threadIdx.z")
        thread_z = *extent;
    });

    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simtvf_helper_counter_++;
    {
      auto it = Attributes(op).find("tl.vf_source_index");
      if (it != Attributes(op).end()) {
        if (auto *imm = (*it).second.as<IntImmNode>()) {
          vf_idx = imm->value.as<int64_t>().value();
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

void CodeGenTileLangAscend::Dispatch_(const BroadcastNode *op,
                                      std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(
      (op->lanes).as_or_throw<IntImm>()->value.as<int64_t>().value());
  RejectGenericVectorInSimd("Broadcast", PythonDType(GetRef<Expr>(op)));

  // Keep casts outside broadcasts so the vector value is converted as a
  // whole. Besides avoiding target-specific Broadcast(Cast(Call)) patterns,
  // this lets vector casts use packed conversion intrinsics such as
  // float2 -> bfloat16x2.
  if (const auto *cast = op->value.as<CastNode>()) {
    PrimExpr source = Broadcast(cast->value, op->lanes);
    PrintExpr(Cast(PythonDType(GetRef<Expr>(op)), source), os);
    return;
  }

  // Match CUDA codegen's stateful RNG vectorization: the loop vectorizer
  // leaves an opaque scalar RNG call under Broadcast, and target codegen
  // replaces the vector with one backend runtime call. Ascend's default 64-bit
  // vector width makes the FP32 case a float2.
  const CallNode *rng_call = op->value.as<CallNode>();
  if (rng_call != nullptr) {
    const CallNode *call = rng_call;
    if (call->op.same_as(tl::rng_rand_float()) && (lanes == 2 || lanes == 4) &&
        PythonDType(GetRef<Expr>(call)).MatchesCode(kDLFloat) &&
        PythonDType(GetRef<Expr>(call)).bits() == 32) {
      ICHECK_EQ(call->args.size(), 1);
      const auto *dist = call->args[0].as<StringImmNode>();
      ICHECK(dist != nullptr) << "Ascend RNG distribution must be constant";
      if (dist->value == "uniform" || dist->value == "normal") {
        EmitPhiloxVectorFloat(os, dist->value, lanes);
        return;
      }
    }
  }

  if ((PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
       PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) &&
      PythonDType(GetRef<Expr>(op)).bits() == 8) {
    auto p = tl::GetConstInt(op->value);
    if (p) {
      if (lanes == 4) {
        // make_int8x4
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
        return;
      } else if (lanes == 32 && IsInsideSimtVF()) {
        // make_int8x32
        auto p = tl::GetConstInt(op->value);
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
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

  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat) &&
      PythonDType(GetRef<Expr>(op)).bits() == 32 &&
      PythonDType(GetRef<Expr>(op)).lanes() == 8) {
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

  if ((PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
       PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) &&
      PythonDType(GetRef<Expr>(op)).bits() == 4) {
    bool fail = false;
    auto p = tl::GetConstInt(op->value);
    ICHECK(p) << "BroadcastNode " << op << " value: " << op->value
              << " is not a constant";
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) |
          (v << 4) | v;
      if (lanes == 8) {
        if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(PythonDType(GetRef<Expr>(op)), os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0)
            os << ", ";
          if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
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
      (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLFloat, 16) ||
       PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16)) &&
      lanes <= 8 && lanes % 2 == 0;
  const bool is_packed_predicate =
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLBool) && lanes > 4 &&
      lanes <= 8 && lanes % 2 == 0;
  if (is_packed_16 || is_packed_predicate) {
    // Small FP16/BF16 vectors and wide predicates use uint carriers in
    // generated Ascend C. Materialize the carrier lane by lane instead of
    // relying on constructors whose argument count follows the carrier rather
    // than the logical lane count. Larger vectors are CCE SIMD registers and
    // must be produced by SIMD intrinsics such as vdup.
    TVM_FFI_ICHECK_LE(lanes, 8)
        << "Ascend packed Broadcast supports at most 8 lanes outside SIMD "
           "intrinsics, got "
        << PythonDType(GetRef<Expr>(op));
    std::string result = name_supply_->FreshName("_");
    PrintIndent();
    PrintType(PythonDType(GetRef<Expr>(op)), stream);
    stream << ' ' << result << ";\n";
    SSAOperationScope ssa_scope(this);
    std::string value = SSAGetID(PrintExpr(op->value), PythonDType(op->value));
    for (int i = 0; i < lanes; ++i) {
      PrintVecElemStore(result, PythonDType(GetRef<Expr>(op)), i, value);
    }
    os << result;
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(PythonDType(GetRef<Expr>(op)), os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenTileLangAscend::Dispatch_(const RampNode *op,
                                      std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(
      (op->lanes).as_or_throw<IntImm>()->value.as<int64_t>().value());
  ValidateGenericVectorType("Ramp", PythonDType(GetRef<Expr>(op)));
  os << "make_";
  PrintType(PythonDType(GetRef<Expr>(op)), os);
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
  os << "simd_inst::" << ResolveSimdIntrinsicName_(call, PythonDType(call))
     << "(";
  for (size_t i = 0; i < call->args.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    if (i < num_expr_args) {
      PrintExpr(call->args[i], os);
    } else {
      // Trailing arguments name control enums and must stay unquoted.
      os << (call->args[i]).as_or_throw<StringImm>()->value;
    }
  }
  os << ")";
}

bool CodeGenTileLangAscend::EmitSimdMergingCall(const CallNode *op,
                                                std::ostream &os) {
  std::string intrinsic_name;
  if (!IsSimdMergingCall(op, &intrinsic_name)) {
    return false;
  }

  ICHECK_GE(op->args.size(), 3)
      << "Legalized MODE_MERGING call must contain a destination, operands, "
         "and mode";
  ICHECK(PythonIsPointer(op->args[0]))
      << "Legalized MODE_MERGING destination must be a pointer";
  std::string resolved_name =
      ResolveSimdIntrinsicName_(GetRef<Call>(op), PythonDType(op->args[1]));
  PrimType dtype = PythonDType(op->args[1]);
  if (intrinsic_name == "vdup") {
    // Scalar broadcast overloads are selected by the destination vector type.
    const auto *destination = op->args[0].as<CallNode>();
    ICHECK(destination && PythonIsOp(destination->op, "address_of") &&
           destination->args.size() == 1);
    dtype = PythonDType(destination->args[0]);
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
  PrintExpr(op->args[0], os);
  os << ")";
  for (size_t i = 1; i < op->args.size(); ++i) {
    os << ", ";
    if (const auto *string_arg = op->args[i].as<StringImmNode>()) {
      os << string_arg->value;
    } else {
      PrintExpr(op->args[i], os);
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

void CodeGenTileLangAscend::EmitPipeBarrier_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1)
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

void CodeGenTileLangAscend::EmitHardEventSync_(const CallNode *op,
                                               bool notify) {
  const char *intrinsic = notify ? "tl.ascend_set_flag" : "tl.ascend_wait_flag";
  ICHECK_EQ(op->args.size(), 2)
      << intrinsic
      << " expects exactly 2 arguments (hard_event string, event_id)";
  const std::string hard_event =
      GetStringImmArg(op, 0, intrinsic, "hard_event");
  const PipePair pipes = GetPipePair(hard_event);
  const std::string event_id = PrintExpr(op->args[1]);

  PrintIndent();
  stream << (notify ? "asc_sync_notify(" : "asc_sync_wait(") << pipes.source
         << ", " << pipes.destination << ", static_cast<event_t>(" << event_id
         << "));\n";
}

void CodeGenTileLangAscend::EmitCrossCoreSync_(const CallNode *op,
                                               bool arrive) {
  const char *intrinsic = arrive ? "tl.ascend_cross_core_set_flag"
                                 : "tl.ascend_cross_core_wait_flag";
  ICHECK_EQ(op->args.size(), 3)
      << intrinsic
      << " expects exactly 3 arguments (mode_id int, pipe string, flag_id)";
  const int64_t mode_id = GetIntImmArg(op, 0, intrinsic, "mode_id");
  const char *scope = GetCrossCoreSyncScope(mode_id);
  const std::string pipe = GetStringImmArg(op, 1, intrinsic, "pipe");
  const std::string flag_id = PrintExpr(op->args[2]);

  PrintIndent();
  stream << "asc_sync_" << scope << (arrive ? "_arrive(" : "_wait(") << pipe
         << ", " << flag_id << ");\n";
}

void CodeGenTileLangAscend::EmitGmToUbufCopy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 11)
      << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";

  // burstLen and stride use bytes; right_pad uses elements of the buffer dtype.
  std::string pointer_type = "uint8_t";
  const auto *right_pad = op->args[6].as<IntImmNode>();
  if (right_pad && right_pad->value != 0) {
    auto pointer = op->args[0]->ty.as<PtrTypeNode>();
    ICHECK(pointer) << "Padded DMA destination must be a typed pointer";
    PrimType element_type = pointer->element_type.as_or_throw<PrimType>();
    const int bits = element_type.bits();
    ICHECK(bits == 8 || bits == 16 || bits == 32)
        << "Ascend padded GM->UB copy supports 8/16/32-bit elements, got "
        << bits;
    pointer_type = "uint" + std::to_string(bits) + "_t";
  }

  // IR arg 2 is sid, which asc_copy_gm2ub_align does not take.
  EmitCApiCall_("asc_copy_gm2ub_align",
                {{op->args[0], "(__ubuf__ " + pointer_type + "*)(", ")"},
                 {op->args[1], "(__gm__ " + pointer_type + "*)(", ")"},
                 {op->args[3]},
                 {op->args[4]},
                 {op->args[5]},
                 {op->args[6]},
                 {op->args[7]},
                 {op->args[8], "static_cast<asc_load_l2_cache_mode>(", ")"},
                 {op->args[9]},
                 {op->args[10]}});
}

void CodeGenTileLangAscend::EmitUbufToGmCopy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 8)
      << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";

  // IR arg 2 is sid, which asc_copy_ub2gm_align does not take.
  EmitCApiCall_("asc_copy_ub2gm_align",
                {{op->args[0], "(__gm__ uint8_t*)(", ")"},
                 {op->args[1], "(__ubuf__ uint8_t*)(", ")"},
                 {op->args[3]},
                 {op->args[4]},
                 {op->args[5], "static_cast<asc_store_l2_cache_mode>(", ")"},
                 {op->args[6]},
                 {op->args[7]}});
}

void CodeGenTileLangAscend::EmitGmToL1Copy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 12)
      << "tl.ascend_copy_gm_to_cbuf expects exactly 12 arguments";

  const bool use_dn2nz =
      GetIntImmArg(op, 9, "tl.ascend_copy_gm_to_cbuf", "transpose") != 0;
  const std::string nz_c0_stride = PrintExpr(op->args[10]);
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
  EmitCApiCall_(use_dn2nz ? "asc_copy_gm2l1_dn2nz" : "asc_copy_gm2l1_nd2nz",
                {{op->args[0], destination_prefix, pointer_suffix},
                 {op->args[1], source_prefix, pointer_suffix},
                 {op->args[3]},
                 {op->args[4], "static_cast<asc_load_l2_cache_mode>(", ")"},
                 {op->args[5]},
                 {op->args[6]},
                 {op->args[7]},
                 {op->args[8]}});
}

void CodeGenTileLangAscend::EmitL0cToUbufCopy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 26)
      << "tl.ascend_copy_matrix_cc_to_ub expects exactly 26 arguments";

  PrintIndent();
  stream << "asc_set_l0c_copy_nz_para(1, 0, 0);\n";
  EmitCApiCall_("asc_copy_l0c2ub",
                {{op->args[0]},
                 {op->args[1]},
                 {op->args[3]},
                 {op->args[4]},
                 {op->args[5]},
                 {op->args[6]},
                 {op->args[8]},
                 {op->args[7], "static_cast<asc_dual_dst_mode>(", ")"},
                 {op->args[10], "static_cast<asc_unit_flag_mode>(", ")"},
                 {op->args[11], "static_cast<asc_quant_mode>(", ")"},
                 {op->args[12], "static_cast<asc_relu_pre_mode>(", ")"},
                 {op->args[13]},
                 {op->args[14]},
                 {op->args[25]},
                 {op->args[9]}});
}

void CodeGenTileLangAscend::EmitL0cToGmCopy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 25)
      << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";

  PrintIndent();
  stream << "asc_set_l0c_copy_nz_para(1, 0, 0);\n";
  EmitCApiCall_("asc_copy_l0c2gm",
                {{op->args[0]},
                 {op->args[1]},
                 {op->args[3]},
                 {op->args[4]},
                 {op->args[5]},
                 {op->args[6]},
                 {op->args[7], "static_cast<asc_store_l2_cache_mode>(", ")"},
                 {op->args[9], "static_cast<asc_unit_flag_mode>(", ")"},
                 {op->args[10], "static_cast<asc_quant_mode>(", ")"},
                 {op->args[11], "static_cast<asc_relu_pre_mode>(", ")"},
                 {op->args[12]},
                 {op->args[13]},
                 {op->args[24]},
                 {op->args[8]}});
}

void CodeGenTileLangAscend::EmitSetCopyPadValue_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1)
      << "tl.ascend_set_copy_pad_value expects exactly 1 argument";
  const int bit_width = PythonDType(op->args[0]).bits();
  ICHECK(bit_width == 8 || bit_width == 16 || bit_width == 32)
      << "Data type: " << PythonDType(op->args[0])
      << " is not supported in ascend";

  const std::string value = PrintExpr(op->args[0]);
  const std::string target_value = name_supply_->FreshName("__asc_pad_val");
  PrintIndent();
  PrintType(PythonDType(op->args[0]), stream);
  stream << " " << target_value << " = " << value << ";\n";

  PrintIndent();
  stream << "asc_set_copy_pad_val(*reinterpret_cast<uint" << bit_width
         << "_t*>(&" << target_value << "));\n";
}

void CodeGenTileLangAscend::EmitFillL1_(const CallNode *op) {
  constexpr const char *kIntrinsic = "tl.ascend_fill_l1";
  ICHECK_EQ(op->args.size(), 7) << kIntrinsic << " expects exactly 7 arguments";
  const int64_t fill_word_bits =
      GetIntImmArg(op, 6, kIntrinsic, "fill_word_bits");
  ICHECK(fill_word_bits == 16 || fill_word_bits == 32)
      << kIntrinsic << " fill_word_bits must be 16 or 32";

  std::vector<std::string> arguments;
  arguments.reserve(6);
  for (size_t i = 0; i < 6; ++i) {
    arguments.push_back(PrintExpr(op->args[i]));
  }

  PrintIndent();
  stream << "asc_fill_l1((__cbuf__ uint" << fill_word_bits
         << "_t*)((__cbuf__ uint8_t*)" << arguments[0] << " + " << arguments[1]
         << "), (uint32_t)(" << arguments[2]
         << "), { .repeat = static_cast<uint64_t>(" << arguments[3]
         << "), .blk_num = static_cast<uint64_t>(" << arguments[4]
         << "), .dst_gap = static_cast<uint64_t>(" << arguments[5] << ")});\n";
}

void CodeGenTileLangAscend::EmitL1ToL0Copy_(const CallNode *op, bool is_l0a) {
  const char *intrinsic =
      is_l0a ? "tl.ascend_load_cbuf_to_ca" : "tl.ascend_load_cbuf_to_cb";
  ICHECK_EQ(op->args.size(), 9)
      << intrinsic << " expects exactly 9 arguments, got " << op->args.size();

  const bool is_transpose = GetIntImmArg(op, 8, intrinsic, "transpose") == 1;
  const char *function_name =
      is_transpose
          ? (is_l0a ? "asc_copy_l12l0a_transpose" : "asc_copy_l12l0b_transpose")
          : (is_l0a ? "asc_copy_l12l0a" : "asc_copy_l12l0b");
  EmitCApiCall_(function_name, {{op->args[0]},
                                {op->args[1]},
                                {op->args[2]},
                                {op->args[3]},
                                {op->args[4]},
                                {op->args[5]},
                                {op->args[6]},
                                {op->args[7]}});
}

void CodeGenTileLangAscend::EmitMxSfLoad_(const CallNode *op, bool is_l0a) {
  const char *intrinsic =
      is_l0a ? "tl.ascend_load_ca_sf" : "tl.ascend_load_cb_sf";
  ICHECK_EQ(op->args.size(), 8)
      << intrinsic << " expects exactly 8 arguments, got " << op->args.size();

  // The MX destination is the DATA tile's address in 16-byte units; the
  // scale source must be viewed in its physical E8M0 representation.
  EmitCApiCall_(is_l0a ? "asc_copy_l12l0a_mx" : "asc_copy_l12l0b_mx",
                {{op->args[0], "(uint64_t)(uintptr_t)(", ") / 16"},
                 {op->args[1], "(__cbuf__ fp8_e8m0_t*)(", ")"},
                 {op->args[2]},
                 {op->args[3]},
                 {op->args[4]},
                 {op->args[5]},
                 {op->args[6]},
                 {op->args[7]}});
}

void CodeGenTileLangAscend::EmitUbufToL1Copy_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 7)
      << "tl.ascend_copy_ubuf_to_cbuf expects exactly 7 arguments";

  // IR arg 2 is sub_blockid, which asc_copy_ub2l1 does not take.
  EmitCApiCall_("asc_copy_ub2l1", {{op->args[0], "(__cbuf__ void*)(", ")"},
                                   {op->args[1], "(__ubuf__ void*)(", ")"},
                                   {op->args[3]},
                                   {op->args[4]},
                                   {op->args[5]},
                                   {op->args[6]}});
}

void CodeGenTileLangAscend::EmitNd2NzScatter_(const CallNode *op) {
  constexpr const char *kIntrinsic = "tl.ascend_nd2nz_scatter";
  ICHECK_EQ(op->args.size(), 6) << kIntrinsic << " expects exactly 6 arguments";
  const int64_t rows = GetIntImmArg(op, 2, kIntrinsic, "rows");
  const int64_t cols = GetIntImmArg(op, 3, kIntrinsic, "cols");
  const std::string dst_dtype = GetStringImmArg(op, 4, kIntrinsic, "dst_dtype");
  const std::string src_dtype = GetStringImmArg(op, 5, kIntrinsic, "src_dtype");

  std::ostringstream function_name;
  function_name << (IsInsideSimdVF() ? "ascend_nd2nz_scatter_callee"
                                     : "ascend_nd2nz_scatter")
                << "<" << rows << ", " << cols << ", " << src_dtype << ", "
                << dst_dtype << ">";
  EmitCApiCall_(function_name.str(),
                {{op->args[0], "(__ubuf__ " + src_dtype + "*)(", ")"},
                 {op->args[1], "(__ubuf__ " + dst_dtype + "*)(", ")"}});
}

void CodeGenTileLangAscend::EmitNd2NzPostCopy_(const CallNode *op) {
  constexpr const char *kIntrinsic = "tl.ascend_nd2nz_post_copy";
  ICHECK_EQ(op->args.size(), 6) << kIntrinsic << " expects exactly 6 arguments";
  const int64_t rows = GetIntImmArg(op, 2, kIntrinsic, "rows");
  const int64_t cols = GetIntImmArg(op, 3, kIntrinsic, "cols");
  const int64_t full_rows = GetIntImmArg(op, 4, kIntrinsic, "full_rows");
  const std::string dst_dtype = GetStringImmArg(op, 5, kIntrinsic, "dst_dtype");

  const int64_t element_bytes = dst_dtype == "float" ? 4 : 2;
  const int64_t elements_per_c0 = 32 / element_bytes;
  const int64_t burst_num = cols / elements_per_c0;
  const int64_t burst_len = rows;
  constexpr int64_t kSourceGap = 1;
  const int64_t destination_gap = full_rows - rows;
  const std::string destination = PrintExpr(op->args[0]);
  const std::string source = PrintExpr(op->args[1]);

  PrintIndent();
  stream << "asc_copy_ub2l1((__cbuf__ void*)(" << destination
         << "), (__ubuf__ void*)(" << source << "), " << burst_num << ", "
         << burst_len << ", " << kSourceGap << ", " << destination_gap
         << ");\n";
}

void CodeGenTileLangAscend::EmitMad_(const CallNode *op, bool is_mx) {
  const char *intrinsic = is_mx ? "tl.ascend_mad_mx" : "tl.ascend_mad";
  ICHECK_EQ(op->args.size(), 10)
      << intrinsic << " expects exactly 10 arguments";
  EmitCApiCall_(is_mx ? "asc_mmad_mx" : "asc_mmad", {{op->args[0]},
                                                     {op->args[1]},
                                                     {op->args[2]},
                                                     {op->args[3]},
                                                     {op->args[4]},
                                                     {op->args[5]},
                                                     {op->args[6]},
                                                     {op->args[7]},
                                                     {op->args[8]},
                                                     {op->args[9]}});
}

void CodeGenTileLangAscend::EmitGemmL1_(const CallNode *op) {
  constexpr const char *kIntrinsic = "tl.ascend_gemm_l1";
  ICHECK_EQ(op->args.size(), 12)
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
  EmitCApiCall_(function_name.str(), {{op->args[0]},
                                      {op->args[1]},
                                      {op->args[2]},
                                      {op->args[8]},
                                      {op->args[10]},
                                      {op->args[11]}});
}

void CodeGenTileLangAscend::EmitBlockscaledGemmL1_(const CallNode *op) {
  constexpr const char *kIntrinsic = "tl.ascend_blockscaled_gemm_l1";
  ICHECK_EQ(op->args.size(), 18)
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
  EmitCApiCall_(function_name.str(),
                {{op->args[0]},
                 {op->args[1]},
                 {op->args[2]},
                 {op->args[3], "(__cbuf__ " + scale_dtype + "*)(", ")"},
                 {op->args[4], "(__cbuf__ " + scale_dtype + "*)(", ")"},
                 {op->args[10]},
                 {op->args[14]},
                 {op->args[15]},
                 {op->args[17]}});
}

bool CodeGenTileLangAscend::EmitAscendMemoryCall_(const CallNode *op) {
  if (op->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
    EmitGmToUbufCopy_(op);
  } else if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
    EmitSetCopyPadValue_(op);
  } else if (op->op.same_as(tl::ascend_copy_ubuf_to_gm())) {
    EmitUbufToGmCopy_(op);
  } else if (op->op.same_as(tl::ascend_copy_gm_to_cbuf())) {
    EmitGmToL1Copy_(op);
  } else if (op->op.same_as(tl::ascend_fill_l1())) {
    EmitFillL1_(op);
  } else if (op->op.same_as(tl::ascend_load_cbuf_to_ca()) ||
             op->op.same_as(tl::ascend_load_cbuf_to_cb())) {
    EmitL1ToL0Copy_(op, op->op.same_as(tl::ascend_load_cbuf_to_ca()));
  } else if (op->op.same_as(tl::ascend_load_ca_sf()) ||
             op->op.same_as(tl::ascend_load_cb_sf())) {
    EmitMxSfLoad_(op, op->op.same_as(tl::ascend_load_ca_sf()));
  } else if (op->op.same_as(tl::ascend_copy_matrix_cc_to_ub())) {
    EmitL0cToUbufCopy_(op);
  } else if (op->op.same_as(tl::ascend_copy_matrix_cc_to_gm())) {
    EmitL0cToGmCopy_(op);
  } else if (op->op.same_as(tl::ascend_copy_ubuf_to_cbuf())) {
    EmitUbufToL1Copy_(op);
  } else if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
    EmitNd2NzScatter_(op);
  } else if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
    EmitNd2NzPostCopy_(op);
  } else if (op->op.same_as(tl::ascend_mad()) ||
             op->op.same_as(tl::ascend_mad_mx())) {
    EmitMad_(op, op->op.same_as(tl::ascend_mad_mx()));
  } else if (op->op.same_as(tl::ascend_gemm_l1())) {
    EmitGemmL1_(op);
  } else if (op->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
    EmitBlockscaledGemmL1_(op);
  } else {
    return false;
  }
  return true;
}

void CodeGenTileLangAscend::Dispatch_(const CallNode *op, std::ostream &os) {
  std::string simd_op_name;
  if (IsSimdOp(op, &simd_op_name)) {
    if (!IsInsideSimdVF()) {
      LOG(FATAL) << simd_op_name << " can only be emitted inside SimdVF";
    }
  }
  if (PythonIsOp(op->op, "bitwise_not") &&
      PythonDType(GetRef<Expr>(op)).lanes() > 1) {
    RejectGenericVectorInSimd("bitwise Not", PythonDType(GetRef<Expr>(op)));
    if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLBool) &&
        PythonDType(GetRef<Expr>(op)).lanes() <= 8) {
      ICHECK_EQ(op->args.size(), 1U);
      EmitSimtVectorPredicateNot(op->args[0].as_or_throw<PrimExpr>(),
                                 PythonDType(GetRef<Expr>(op)), os);
      return;
    }
  }
  if (EmitSimdMergingCall(op, os)) {
    return;
  }
  if (EmitAscendMemoryCall_(op)) {
    return;
  }
  if (PythonIsOp(op->op, "reinterpret")) {
    SSAOperationScope ssa_scope(this);
    CodeGenC::Dispatch_(op, os);
    return;
  }
  if (op->op.same_as(tl::device_assert())) {
    ICHECK_EQ(op->args.size(), 1)
        << "tl.device_assert expects exactly 1 argument (condition)";
    this->PrintIndent();
    this->stream << "device_assert(" << PrintExpr(op->args[0]) << ");\n";
  } else if (op->op.same_as(tl::device_assert_with_msg())) {
    ICHECK_EQ(op->args.size(), 2)
        << "tl.device_assert_with_msg expects exactly 2 arguments "
           "(condition, msg)";
    this->PrintIndent();
    this->stream << "device_assert_with_msg(" << PrintExpr(op->args[0]) << ", "
                 << PrintExpr(op->args[1]) << ");\n";
  } else if (op->op.same_as(tl::ascend_pipe_barrier())) {
    EmitPipeBarrier_(op);
  } else if (op->op.same_as(tl::ascend_set_flag()) ||
             op->op.same_as(tl::ascend_wait_flag())) {
    EmitHardEventSync_(op, op->op.same_as(tl::ascend_set_flag()));
  } else if (op->op.same_as(tl::ascend_threadfence())) {
    ICHECK_EQ(op->args.size(), 0)
        << "tl.ascend_threadfence expects 0 arguments";
    this->PrintIndent();
    this->stream << "asc_threadfence();\n";
  } else if (op->op.same_as(tl::sync_warp())) {
    ICHECK_LE(op->args.size(), 1U)
        << "tl.sync_warp expects an optional lane mask";
    if (op->args.empty()) {
      this->PrintIndent();
      this->stream << "cooperative_groups::coalesced_threads().sync();\n";
    } else {
      // coalesced_threads() captures the active lanes at the call site. Enter
      // the branch only for named lanes so the resulting group matches mask.
      std::string mask = this->PrintExpr(op->args[0]);
      this->PrintIndent();
      this->stream << "if (((static_cast<uint32_t>(" << mask
                   << ") >> laneid()) & 1U) != 0U) { "
                      "cooperative_groups::coalesced_threads().sync(); }\n";
    }
  } else if (op->op.same_as(tl::ballot_sync())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ballot_sync expects <mask, predicate>.";
    os << "((uint64_t)(asc_ballot(";
    PrintExpr(op->args[1], os);
    os << ") & static_cast<uint32_t>(";
    PrintExpr(op->args[0], os);
    os << ")))";
  } else if (op->op.same_as(tl::ballot())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.ballot expects <predicate>.";
    os << "((uint64_t)asc_ballot(";
    PrintExpr(op->args[0], os);
    os << "))";
  } else if (op->op.same_as(tl::activemask())) {
    ICHECK(op->args.empty()) << "tl.activemask takes no arguments.";
    os << "((uint64_t)asc_activemask())";
  } else if (op->op.same_as(tl::ascend_get_buf())) {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.ascend_get_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str = (op->args[0]).as_or_throw<StringImm>()->value;
    int64_t mode =
        (op->args[2]).as_or_throw<IntImm>()->value.as<int64_t>().value();
    this->PrintIndent();
    this->stream << "asc_lock(" << pipe_str << ", ";
    this->PrintExpr(op->args[1], this->stream);
    // ASC_LOCK_BLOCK is the C API default; emit it explicitly for readability.
    this->stream << (mode ? ", ASC_LOCK_NON_BLOCK" : ", ASC_LOCK_BLOCK");
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_rls_buf())) {
    ICHECK_EQ(op->args.size(), 3)
        << "tl.ascend_rls_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str = (op->args[0]).as_or_throw<StringImm>()->value;
    int64_t mode =
        (op->args[2]).as_or_throw<IntImm>()->value.as<int64_t>().value();
    this->PrintIndent();
    this->stream << "asc_unlock(" << pipe_str << ", ";
    this->PrintExpr(op->args[1], this->stream);

    // ASC_LOCK_BLOCK is the C API default; emit it explicitly for readability.
    this->stream << (mode ? ", ASC_LOCK_NON_BLOCK" : ", ASC_LOCK_BLOCK");
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
    ICHECK_EQ(op->args.size(), 1)
        << "tl.ascend_set_hf32_mode expects exactly 1 argument (mode_int)";
    int mode = (op->args[0]).as_or_throw<IntImm>()->value.as<int64_t>().value();
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
  } else if (op->op.same_as(tl::ascend_set_mmad_direction())) {
    ICHECK_EQ(op->args.size(), 1);
    const auto *direction = op->args[0].as<StringImmNode>();
    ICHECK(direction && (direction->value == "m" || direction->value == "n"))
        << "tl.ascend_set_mmad_direction expects 'm' or 'n'";
    this->PrintIndent();
    this->stream << "asc_set_mmad_direction_" << direction->value << "();\n";
  } else if (op->op.same_as(tl::ascend_set_atomic())) {
    ICHECK_EQ(op->args.size(), 2)
        << "tl.ascend_set_atomic expects 2 arguments (op_str, typed_zero)";
    std::string atomic_op = (op->args[0]).as_or_throw<StringImm>()->value;
    const char *fn = atomic_op == "max"   ? "asc_set_atomic_max"
                     : atomic_op == "min" ? "asc_set_atomic_min"
                                          : "asc_set_atomic_add";
    ICHECK(atomic_op == "add" || atomic_op == "max" || atomic_op == "min")
        << "tl.ascend_set_atomic op must be add/max/min, got " << atomic_op;

    PrimType atomic_dtype = PythonDType(op->args[1]);
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
  } else if (op->op.same_as(tl::ascend_set_atomic_none())) {
    ICHECK_EQ(op->args.size(), 0)
        << "tl.ascend_set_atomic_none expects no arguments";
    this->PrintIndent();
    this->stream << "asc_set_atomic_none();\n";
  } else if (op->op.same_as(tl::warp_reduce_sum()) ||
             op->op.same_as(tl::warp_reduce_max()) ||
             op->op.same_as(tl::warp_reduce_min())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "Ascend hardware warp reduction expects exactly one argument";
    PrimType dtype = PythonDType(op->args[0]);
    ICHECK(IsAscendWarpReduceDType(dtype))
        << "Ascend hardware warp reduction supports only scalar float16, "
           "float32, int32, and uint32 values, but got "
        << dtype;
    const char *intrinsic =
        op->op.same_as(tl::warp_reduce_sum())
            ? "asc_reduce_add"
            : (op->op.same_as(tl::warp_reduce_max()) ? "asc_reduce_max"
                                                     : "asc_reduce_min");
    os << intrinsic << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ascend_cross_core_set_flag()) ||
             op->op.same_as(tl::ascend_cross_core_wait_flag())) {
    EmitCrossCoreSync_(op, op->op.same_as(tl::ascend_cross_core_set_flag()));
  } else if (PythonIsOp(op->op, "address_of")) {
    const auto *load = op->args[0].as<TensorLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    const auto *buf_var = load->source.as_or_throw<Var>().get();
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
                         PythonDType(load->indices[0]).lanes() > 1;
    // Packed fp4 (float4_e2m1fnx2) held in GM/UBuf: emitted as the 1-byte
    // `float4_e2m1x2_t`. Handle only after the local-vreg / L0-L1 cases, since
    // a fp4 SIMD register slot must still use the register-index addressing
    // below.
    bool is_fp4_ptr =
        PythonDType(GetRef<Expr>(load)).MatchesCode(kDLFloat4_e2m1fn) &&
        load->indices.size() == 1 && !is_local_vreg && !needs_ptr_arith;
    if (is_local_vreg) {
      PrimExpr index = load->indices[0];
      ValidateSimdVectorType(PythonDType(GetRef<Expr>(load)),
                             "SimdVF local register array element");
      const auto *ramp = index.as<RampNode>();
      PrimExpr base = ramp ? ramp->base : index;
      int lanes = PythonDType(index).lanes();
      PrimExpr elem_idx = analyzer_->Simplify(
          floordiv(base, MakeConst(PythonDType(base), lanes)));
      os << "(&(" << GetVarID(buf_var) << "[";
      PrintExpr(elem_idx, os);
      os << "]))";
    } else if (needs_ptr_arith) {
      os << "((";
      PrintStorageScope(runtime::MemoryScope(scope), os);
      PrintType(PythonDType(GetRef<Expr>(load)), os);
      os << "*)" << GetVarID(buf_var);
      if (load->indices.size() == 1) {
        PrimExpr idx = load->indices[0];
        // LowerIntrin represents an fp4x2 access as a 2-lane Ramp and scales
        // its packed-byte offset by the lane count.  L0/L1 pointers need the
        // same scalar-byte recovery as GM/UB pointers below; otherwise this
        // prints invalid pointer arithmetic such as `ptr + int2(...)`.
        if (PythonDType(GetRef<Expr>(load)).MatchesCode(kDLFloat4_e2m1fn) &&
            PythonDType(idx).lanes() > 1) {
          const auto *ramp = idx.as<RampNode>();
          ICHECK(ramp) << "Expected a Ramp index for packed fp4 L0/L1 access, "
                       << "got " << idx;
          idx = analyzer_->Simplify(
              floordiv(ramp->base, MakeConst(PythonDType(ramp->base),
                                             PythonDType(idx).lanes())));
        }
        if (!is_zero(idx)) {
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
      std::string fp4_scope = !scope.empty()
                                  ? scope
                                  : std::string(PythonStorageScope(
                                        load->source.as_or_throw<Var>()));
      PrimExpr index = load->indices[0];
      const auto *ramp = index.as<RampNode>();
      PrimExpr base = ramp ? ramp->base : index;
      int lanes = PythonDType(index).lanes();
      PrimExpr byte_idx =
          lanes > 1 ? analyzer_->Simplify(
                          floordiv(base, MakeConst(PythonDType(base), lanes)))
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
  else if (op->op.same_as(tl::ascend_read_gm_bypass_dcache())) {
    // ascend_read_gm_bypass_dcache(address_of(BufferLoad))
    ICHECK_EQ(op->args.size(), 1);
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && PythonIsOp(addr->op, "address_of"));
    const auto *load = addr->args[0].as<TensorLoadNode>();
    ICHECK(load && load->indices.size() == 1);
    os << "tl::read_gm_bypass_dcache("
       << GetGmBypassPtr(PythonDType(GetRef<Expr>(load)),
                         load->source.as_or_throw<TensorVar>().get(),
                         load->indices[0])
       << ")";
  } else if (op->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
    // ascend_write_gm_bypass_dcache(address_of(BufferLoad), value)
    ICHECK_EQ(op->args.size(), 2);
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && PythonIsOp(addr->op, "address_of"));
    const auto *load = addr->args[0].as<TensorLoadNode>();
    ICHECK(load && load->indices.size() == 1);
    // Evaluate ptr and value to strings first. The string PrintExpr overload
    // flushes any SSA statements (e.g. from if_then_else) to this->stream
    // before returning the expression text, so they land ahead of the call
    // line instead of being injected into the argument list.
    std::string ptr = GetGmBypassPtr(
        PythonDType(GetRef<Expr>(load)),
        load->source.as_or_throw<TensorVar>().get(), load->indices[0]);
    std::string value = this->PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "tl::write_gm_bypass_dcache(" << ptr << ", " << value
                 << ");\n";
  }
  // --- SimdVF vector/scalar FMA: (dst, src0, src1, mask, mode) ---
  else if (op->op.same_as(tl::simd_vmula()) ||
           op->op.same_as(tl::simd_vmadd()) ||
           op->op.same_as(tl::simd_vaxpy())) {
    ICHECK_EQ(op->args.size(), 5);
    PrintSimdCall_(GetRef<Call>(op), 4, os);
  }
  // --- SimdVF histogram: (dst, src, mask, bin) ---
  else if (op->op.same_as(tl::simd_dhistv2()) ||
           op->op.same_as(tl::simd_chistv2())) {
    ICHECK_EQ(op->args.size(), 4);
    const char *name =
        op->op.same_as(tl::simd_dhistv2()) ? "dhistv2" : "chistv2";
    os << "simd_inst::" << name << "(";
    PrintExpr(op->args[0], os); // dst
    os << ", ";
    PrintExpr(op->args[1], os); // src
    os << ", ";
    PrintExpr(op->args[2], os); // mask
    const auto *bin = op->args[3].as<IntImmNode>();
    ICHECK(bin) << name << " requires bin to be the integer 0 or 1";
    ICHECK(bin->value == 0 || bin->value == 1)
        << name << " only supports bin=0 or bin=1, but got " << bin->value;
    os << ", Bin_N" << bin->value;
    os << ")";
  }
  // --- SimdVF carry arithmetic: (src0, src1, mask) -> (carry, result) ---
  else if (op->op.same_as(tl::simd_vaddc()) ||
           op->op.same_as(tl::simd_vsubc())) {
    ICHECK_EQ(op->args.size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF carry family: sub/add-with-carry-in -> (carry, result) ---
  else if (op->op.same_as(tl::simd_vaddcs()) ||
           op->op.same_as(tl::simd_vsubcs())) {
    ICHECK_EQ(op->args.size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 4, os);
  }
  // --- SimdVF widening multiply: (src0, src1, mask) -> (lo, hi) ---
  else if (op->op.same_as(tl::simd_vmull())) {
    ICHECK_EQ(op->args.size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF leaky/parametric relu ---
  else if (op->op.same_as(tl::simd_vlrelu()) ||
           op->op.same_as(tl::simd_vprelu())) {
    ICHECK_EQ(op->args.size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF runtime tail predicate: (value, width) ---
  else if (op->op.same_as(tl::simd_update_mask())) {
    ICHECK_EQ(op->args.size(), 2);
    int width =
        (int)(op->args[1]).as_or_throw<IntImm>()->value.as<int64_t>().value();
    os << "simd_inst::update_mask_b" << width << "(";
    PrintExpr(op->args[0], os); // value
    os << ")";
  }
  // --- SimdVF predicate pack/unpack: (src, part) ---
  else if (op->op.same_as(tl::simd_ppack()) ||
           op->op.same_as(tl::simd_punpack())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 1, os);
  }
  // --- SimdVF predicate interleave/deinterleave: (src0, src1, width) -> pair
  // ---
  else if (op->op.same_as(tl::simd_pintlv()) ||
           op->op.same_as(tl::simd_pdintlv())) {
    ICHECK_EQ(op->args.size(), 3);
    const char *name = op->op.same_as(tl::simd_pintlv()) ? "pintlv" : "pdintlv";
    int width =
        (int)(op->args[2]).as_or_throw<IntImm>()->value.as<int64_t>().value();
    os << "simd_inst::" << name << "_b" << width << "(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ")";
  }
  // --- SimdVF widen unpack: (src, part) ---
  else if (op->op.same_as(tl::simd_vunpack())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 1, os);
  }
  // --- SimdVF unsqueeze: (mask) -> prefix count ---
  else if (op->op.same_as(tl::simd_vusqz())) {
    ICHECK_EQ(op->args.size(), 1);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    os << "simd_inst::vusqz<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // mask
    os << ")";
  }
  // --- SimdVF vector/scalar binary ops: (src0, src1, mask, mode) ---
  else if (op->op.same_as(tl::simd_vadd()) || op->op.same_as(tl::simd_vsub()) ||
           op->op.same_as(tl::simd_vmul()) ||
           op->op.same_as(tl::simd_vabsdif()) ||
           op->op.same_as(tl::simd_vmax()) || op->op.same_as(tl::simd_vmin()) ||
           op->op.same_as(tl::simd_vdiv()) || op->op.same_as(tl::simd_vand()) ||
           op->op.same_as(tl::simd_vor()) || op->op.same_as(tl::simd_vxor()) ||
           op->op.same_as(tl::simd_vshl()) || op->op.same_as(tl::simd_vshr()) ||
           op->op.same_as(tl::simd_vadds()) ||
           op->op.same_as(tl::simd_vmuls()) ||
           op->op.same_as(tl::simd_vmaxs()) ||
           op->op.same_as(tl::simd_vmins()) ||
           op->op.same_as(tl::simd_vshls()) ||
           op->op.same_as(tl::simd_vshrs())) {
    ICHECK_EQ(op->args.size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF predicate ops ---
  else if (op->op.same_as(tl::simd_pand()) || op->op.same_as(tl::simd_por()) ||
           op->op.same_as(tl::simd_pxor()) || op->op.same_as(tl::simd_pnot()) ||
           op->op.same_as(tl::simd_psel())) {
    ICHECK(op->args.size() == (op->op.same_as(tl::simd_pnot()) ? 2 : 3));
    PrintSimdCall_(GetRef<Call>(op), op->args.size(), os);
  }
  // --- SimdVF unary ops: (src, mask, mode) ---
  else if (op->op.same_as(tl::simd_vexp()) || op->op.same_as(tl::simd_vln()) ||
           op->op.same_as(tl::simd_vsqrt()) ||
           op->op.same_as(tl::simd_vabs()) || op->op.same_as(tl::simd_vneg()) ||
           op->op.same_as(tl::simd_vrelu()) ||
           op->op.same_as(tl::simd_vnot())) {
    ICHECK_EQ(op->args.size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF cross-lane reduce/permutation ---
  else if (op->op.same_as(tl::simd_vcpadd()) ||
           op->op.same_as(tl::simd_vcmax()) ||
           op->op.same_as(tl::simd_vcmin()) ||
           op->op.same_as(tl::simd_vcadd()) ||
           op->op.same_as(tl::simd_vcgadd()) ||
           op->op.same_as(tl::simd_vcgmax()) ||
           op->op.same_as(tl::simd_vcgmin()) ||
           op->op.same_as(tl::simd_vsqz())) {
    ICHECK_EQ(op->args.size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF index ramp: (index, order) -> vci<T>(index, order) ---
  else if (op->op.same_as(tl::simd_vci())) {
    ICHECK_EQ(op->args.size(), 2);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    os << "simd_inst::vci<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os);                                         // index
    os << ", " << (op->args[1]).as_or_throw<StringImm>()->value << ")"; // order
  }
  // --- SimdVF compare: (src0, src1, mask, op) -> vcmp_<op>(...) ---
  else if (op->op.same_as(tl::simd_vcmp())) {
    ICHECK_EQ(op->args.size(), 4);
    os << "simd_inst::vcmp_" << (op->args[3]).as_or_throw<StringImm>()->value
       << "(";
    PrintExpr(op->args[0], os); // src0
    os << ", ";
    PrintExpr(op->args[1], os); // src1
    os << ", ";
    PrintExpr(op->args[2], os); // mask
    os << ")";
  }
  // --- SimdVF compare-scalar: (src, scalar, mask, op) -> vcmps_<op>(...) ---
  else if (op->op.same_as(tl::simd_vcmps())) {
    ICHECK_EQ(op->args.size(), 4);
    os << "simd_inst::vcmps_" << (op->args[3]).as_or_throw<StringImm>()->value
       << "(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os); // scalar
    os << ", ";
    PrintExpr(op->args[2], os); // mask
    os << ")";
  }
  // --- SimdVF register permutation: (src0, src1) -> pair ---
  else if (op->op.same_as(tl::simd_vintlv()) ||
           op->op.same_as(tl::simd_vdintlv())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF pair element access: pair_get(pair, index) -> pair.vN ---
  else if (op->op.same_as(tl::simd_pair_get())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintExpr(op->args[0], os); // pair
    int idx =
        (int)(op->args[1]).as_or_throw<IntImm>()->value.as<int64_t>().value();
    os << (idx == 0 ? ".v0" : ".v1");
  }
  // --- SimdVF register pack: (src, part) ---
  else if (op->op.same_as(tl::simd_vpack())) {
    ICHECK_EQ(op->args.size(), 2);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    os << "simd_inst::vpack<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // src
    os << ", " << (op->args[1]).as_or_throw<StringImm>()->value << ")";
  }
  // --- SimdVF vector broadcast vdupv: (src, mask, pos, mode) ---
  else if (op->op.same_as(tl::simd_vdupv())) {
    ICHECK_EQ(op->args.size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF gather 32B blocks: (base, index [, mask]) ---
  else if (op->op.same_as(tl::simd_vgatherb())) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    os << "simd_inst::vgatherb<";
    PrintType(elem, os);
    os << ">((" << "__ubuf__" << " ";
    PrintType(elem, os);
    os << "*)";
    PrintExpr(op->args[0], os); // base
    os << ", ";
    PrintExpr(op->args[1], os); // index
    if (op->args.size() == 3) {
      os << ", ";
      PrintExpr(op->args[2], os); // mask
    }
    os << ")";
  }
  // --- SimdVF gather elements: (base, index, mask) ---
  else if (op->op.same_as(tl::simd_vgather2())) {
    ICHECK_EQ(op->args.size(), 3);
    // No explicit template arg / pointer cast: the base already prints as a
    // typed `__ubuf__ ELEM*`, so overload resolution picks the widening
    // uint8_t/int8_t intrinsic (uint8 -> uint16) from the source pointer type.
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF predicate load: (addr, dist) ---
  else if (op->op.same_as(tl::simd_pld())) {
    ICHECK_EQ(op->args.size(), 2);
    std::string dist = (op->args[1]).as_or_throw<StringImm>()->value;
    static const std::unordered_map<std::string, std::string> kPldFn = {
        {"NORM", "plds_norm"},
        {"US", "plds_upsample"},
        {"DS", "plds_downsample"},
    };
    auto it = kPldFn.find(dist);
    ICHECK(it != kPldFn.end())
        << "Unsupported SimdVF predicate load distribution: " << dist;
    os << "simd_inst::" << it->second << "((__ubuf__ uint32_t*)";
    PrintExpr(op->args[0], os); // addr
    os << ", 0)";
  }
  // --- SimdVF predicate store: (addr, src, dist) ---
  else if (op->op.same_as(tl::simd_pst())) {
    ICHECK_EQ(op->args.size(), 3);
    std::string addr = this->PrintExpr(op->args[0]);
    std::string src = this->PrintExpr(op->args[1]);
    std::string dist = (op->args[2]).as_or_throw<StringImm>()->value;
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
  else if (op->op.same_as(tl::simd_vld())) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3);
    bool post_update = op->args.size() == 3;
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    std::string dist = (op->args[1]).as_or_throw<StringImm>()->value;
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
    PrintExpr(op->args[0], os);
    os << ", ";
    if (post_update) {
      PrintExpr(op->args[2], os);
    } else {
      os << "0";
    }
    os << ")";
  }
  // --- SimdVF dual-dest load: (addr, dist [, offset]) -> vec_pair ---
  else if (op->op.same_as(tl::simd_vld2())) {
    ICHECK(op->args.size() >= 2 && op->args.size() <= 3);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    std::string dist = (op->args[1]).as_or_throw<StringImm>()->value;
    bool has_off = (op->args.size() == 3);
    os << "simd_inst::vld_x2<" << CCEUBufType(elem) << ">((" << "__ubuf__"
       << " " << CCEUBufType(elem) << "*)";
    PrintExpr(op->args[0], os); // addr
    os << ", ";
    if (has_off) {
      os << "asc_update_addr_reg_b" << CCEElemWidthBits(elem) << "(";
      PrintExpr(op->args[2], os);
      os << "), " << dist;
    } else {
      os << dist;
    }
    os << ")";
  }
  // --- SimdVF store: (addr, src, mask, dist) ---
  else if (op->op.same_as(tl::simd_vsts())) {
    ICHECK_EQ(op->args.size(), 4);
    PrimType dtype = PythonDType(op->args[1]).WithLanes(1);
    std::string dist = (op->args[3]).as_or_throw<StringImm>()->value;
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
    std::string src = this->PrintExpr(op->args[1]);
    std::string addr = this->PrintExpr(op->args[0]);
    std::string mask = this->PrintExpr(op->args[2]);
    this->PrintIndent();
    this->stream << "simd_inst::" << it->second << "(" << src << ", (__ubuf__ "
                 << CCEUBufType(dtype) << "*)" << addr << ", 0, " << mask
                 << ");\n";
  }

  // --- SimdVF dual-source interleaved store: (addr, src0, src1) ---
  else if (op->op.same_as(tl::simd_vst2())) {
    ICHECK_EQ(op->args.size(), 3);
    PrimType dtype = PythonDType(op->args[1]).WithLanes(1);
    std::string src0 = this->PrintExpr(op->args[1]);
    std::string src1 = this->PrintExpr(op->args[2]);
    std::string addr = this->PrintExpr(op->args[0]);
    this->PrintIndent();
    this->stream << "simd_inst::vst_x2(" << src0 << ", " << src1
                 << ", (__ubuf__ " << CCEUBufType(dtype) << "*)" << addr
                 << ");\n";
  }

  // --- SimdVF scatter-store blocks: (src, base, stride, mask [, post]) ---
  else if (op->op.same_as(tl::simd_vsstb())) {
    ICHECK(op->args.size() == 4 || op->args.size() == 5);
    bool post_update = op->args.size() == 5;
    ICHECK_EQ(PythonIsPointer(GetRef<Expr>(op)), post_update);
    PrimType dtype = PythonDType(op->args[0]).WithLanes(1);
    // Capture operands first so SSA temporaries (e.g. from reinterpret) are
    // emitted on their own lines before the call statement.
    std::string src = this->PrintExpr(op->args[0]);
    std::string base = this->PrintExpr(op->args[1]);
    std::string stride = this->PrintExpr(op->args[2]);
    std::string mask = this->PrintExpr(op->args[3]);
    if (post_update) {
      // Keep POST_UPDATE as an expression so the frontend can write the
      // advanced pointer back into its mutable handle carrier.
      std::string post = (op->args[4]).as_or_throw<StringImm>()->value;
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
  else if (op->op.same_as(tl::simd_vscatter())) {
    ICHECK_EQ(op->args.size(), 4);
    PrimType dtype = PythonDType(op->args[0]).WithLanes(1);
    // Capture operands first so SSA temporaries (e.g. from reinterpret) are
    // emitted on their own lines before the call statement.
    std::string src = this->PrintExpr(op->args[0]);
    std::string base = this->PrintExpr(op->args[1]);
    std::string index = this->PrintExpr(op->args[2]);
    std::string mask = this->PrintExpr(op->args[3]);
    this->PrintIndent();
    this->stream << "simd_inst::vscatter(" << src << ", (__ubuf__ "
                 << CCEUBufType(dtype) << "*)" << base << ", " << index << ", "
                 << mask << ");\n";
  }
  // --- SimdVF dup: scalar broadcast ---
  else if (op->op.same_as(tl::simd_vdup())) {
    ICHECK_EQ(op->args.size(), 3);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    os << "simd_inst::vdup<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // scalar
    os << ", ";
    PrintExpr(op->args[1], os);                                         // mask
    os << ", " << (op->args[2]).as_or_throw<StringImm>()->value << ")"; // mode
  }
  // --- SimdVF cast: (src, mask, ...extra) ---
  else if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK(op->args.size() >= 4 && op->args.size() <= 6);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    os << "simd_inst::vcvt<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os); // mask
    for (int i = 2; i < static_cast<int>(op->args.size()); ++i) {
      os << ", " << (op->args[i]).as_or_throw<StringImm>()->value;
    }
    os << ")";
  }
  // --- SimdVF vexpdif/select: (src0, src1, mask) ---
  else if (op->op.same_as(tl::simd_vexpdif()) ||
           op->op.same_as(tl::simd_vsel())) {
    ICHECK_EQ(op->args.size(), 3);
    PrintSimdCall_(GetRef<Call>(op), 3, os);
  }
  // --- SimdVF select by lane index: (src, index) ---
  else if (op->op.same_as(tl::simd_vselr())) {
    ICHECK_EQ(op->args.size(), 2);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  } else if (op->op.same_as(tl::simd_mem_bar())) {
    std::string mem_type_str = (op->args[0]).as_or_throw<StringImm>()->value;
    this->PrintIndent();
    this->stream << "simd_inst::mem_bar(" << mem_type_str << ");\n";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
    // --- Atomic ops: emit asc_atomic_{add,max,min}(ptr, val) ---
  } else if (op->op.same_as(tl::atomic_add_elem_op()) ||
             op->op.same_as(tl::atomic_max_elem_op()) ||
             op->op.same_as(tl::atomic_min_elem_op()) ||
             op->op.same_as(tl::atomic_add_ret_elem_op()) ||
             op->op.same_as(tl::atomic_max_ret_elem_op()) ||
             op->op.same_as(tl::atomic_min_ret_elem_op())) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3)
        << "Ascend scalar atomic ops expect (ptr, value[, memory_order])";
    const char *fn = nullptr;
    bool is_ret = false;
    if (op->op.same_as(tl::atomic_add_elem_op()) ||
        op->op.same_as(tl::atomic_add_ret_elem_op()))
      fn = "add";
    else if (op->op.same_as(tl::atomic_max_elem_op()) ||
             op->op.same_as(tl::atomic_max_ret_elem_op()))
      fn = "max";
    else
      fn = "min";
    is_ret = op->op.same_as(tl::atomic_add_ret_elem_op()) ||
             op->op.same_as(tl::atomic_max_ret_elem_op()) ||
             op->op.same_as(tl::atomic_min_ret_elem_op());
    auto emit_ptr_and_val = [&](std::ostream &out) {
      // args[0] is address_of(BufferLoad); its codegen already emits a
      // correctly scoped, element-typed, offset pointer.
      out << "asc_atomic_" << fn << "(" << PrintExpr(op->args[0]) << ", ";
      PrintExpr(op->args[1], out);
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
  } else if (op->op.same_as(tl::rng_init())) {
    // Ascend SIMT Philox RNG. Must be used inside a T.SimtVF(...) block so it
    // compiles into a __simt_vf__ helper. args: seed, seq, off, generator.
    ICHECK(IsInsideSimtVF())
        << "tl.rng_init on Ascend must be used inside a T.SimtVF(...) block";
    ascend_rng_state_var_ = name_supply_->FreshName("__asc_rng_state");
    this->PrintIndent();
    this->stream << "tl::AscendPhiloxState " << ascend_rng_state_var_ << ";\n";
    this->PrintIndent();
    this->stream << "tl::philox_init(&" << ascend_rng_state_var_ << ", "
                 << PrintExpr(op->args[0]) << ", " << PrintExpr(op->args[1])
                 << ", " << PrintExpr(op->args[2]) << ");\n";
    // args[3] (curand generator string) is ignored on Ascend.
  } else if (op->op.same_as(tl::rng_rand())) {
    ICHECK(!ascend_rng_state_var_.empty())
        << "tl.rng_rand requires a preceding T.rng_init call in the same "
           "T.SimtVF() block";
    os << "tl::philox_rand(&" << ascend_rng_state_var_ << ")";
  } else if (op->op.same_as(tl::rng_rand_float())) {
    ICHECK(!ascend_rng_state_var_.empty())
        << "tl.rng_rand_float requires a preceding T.rng_init call in the "
           "same T.SimtVF() block";
    ICHECK_NE(PythonDType(GetRef<Expr>(op)).bits(), 64)
        << "float64 RNG (tl.rng_rand_float bit=64) is not supported on Ascend";
    const std::string dist = op->args[0].as<StringImmNode>()->value;
    ICHECK(dist == "uniform" || dist == "normal")
        << "Unsupported RNG distribution on Ascend: " << dist;
    os << "tl::philox_rand_" << dist << "(&" << ascend_rng_state_var_ << ")";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenTileLangAscend::Dispatch_(const NotNode *op, std::ostream &os) {
  if (PythonDType(GetRef<Expr>(op)).lanes() > 1) {
    RejectGenericVectorInSimd("Not", PythonDType(GetRef<Expr>(op)));
    if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLBool) &&
        PythonDType(GetRef<Expr>(op)).lanes() <= 8) {
      EmitSimtVectorPredicateNot(op->a, PythonDType(GetRef<Expr>(op)), os);
      return;
    }
  }
  CodeGenC::Dispatch_(op, os);
}

bool CodeGenTileLangAscend::IsGlobalGmBuffer(const VarNode *buffer) const {
  const VarNode *buf_var = GetRef<Var>(buffer).get();
  auto it = alloc_storage_scope_.find(buf_var);
  std::string scope =
      (it != alloc_storage_scope_.end())
          ? std::string(it->second.name())
          : std::string(PythonStorageScope(GetRef<Var>(buffer)));
  return scope == "global";
}

std::string CodeGenTileLangAscend::GetGmBypassPtr(PrimType dtype,
                                                  const VarNode *buffer,
                                                  PrimExpr index) {
  const VarNode *buf_var = GetRef<Var>(buffer).get();
  std::ostringstream os;
  os << "((";
  PrintStorageScope(runtime::MemoryScope("global"), os);
  PrintType(dtype, os);
  os << "*)" << GetVarID(buf_var) << " + ";
  PrintExpr(index, os);
  os << ")";
  return os.str();
}

void CodeGenTileLangAscend::Dispatch_(const TensorLoadNode *op,
                                      std::ostream &os) {
  PrimType value_dtype = PythonDType(GetRef<Expr>(op));
  const VarNode *buf_var = op->source.as_or_throw<TensorVar>().get();

  // For Ascend: scalar FP8 loads from shared/UBUF memory need conversion
  // through raw bytes, because the scalar FP8 type (tl::float_e4m3_t) is a
  // struct whose operator= does not support __ubuf__ to local assignment.
  if (value_dtype.IsScalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      std::string scope = it->second.name();
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
        if (!is_zero(index)) {
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
  PVar<PrimExpr> base;
  if (!tvm::ramp(base, 1, lanes).Match(index)) {
    return false;
  }
  // Prove the ramp base is divisible by the vector width.
  PrimExpr ramp_base = base.Eval();
  PrimExpr lanes_const = MakeConst(PythonDType(ramp_base), lanes);
  return analyzer_->CanProveEqual(floormod(ramp_base, lanes_const),
                                  MakeZero(PythonDType(ramp_base)));
}

void CodeGenTileLangAscend::EmitScalarizedLoad(const TensorLoadNode *op,
                                               std::ostream &os) {
  PrimType value_dtype = PythonDType(GetRef<Expr>(op));
  ValidateGenericVectorType("BufferLoad", value_dtype);
  int lanes = value_dtype.lanes();
  PrimExpr index = op->indices[0];

  // A vectorized GM access on the scalar core (outside any VF body) has no
  // valid lowering: the only scalar-pipe GM path is the per-element
  // read_gm_bypass_dcache emitted in VisitExpr_. Vector GM traffic must go
  // through an MTE copy (T.copy) or a SimtVF/SimdVF region.
  if (IsOutsideVF() &&
      IsGlobalGmBuffer(op->source.as_or_throw<TensorVar>().get())) {
    LOG(FATAL) << "Ascend: cannot vectorize GM access of "
               << op->source.as_or_throw<TensorVar>()->name
               << " (lanes=" << lanes
               << ") on the scalar core. Scalar-core GM reads must be a single "
                  "scalar element (lowered to read_gm_bypass_dcache); use "
                  "T.copy or a T.SimtVF/T.SimdVF region for vector GM traffic.";
  }

  // Provably aligned: a single reinterpret-cast vector load (fast path).
  if (VectorAccessIsAligned(index, lanes)) {
    PVar<PrimExpr> base;
    ICHECK(tvm::ramp(base, 1, lanes).Match(index));
    std::string ref = GetVecLoad(
        value_dtype, op->source.as_or_throw<TensorVar>().get(), base.Eval());
    HandleVolatileLoads(ref, op, os);
    return;
  }

  // Otherwise scalarize: load each lane from its scalar element offset and pack
  // with a brace-initializer (which bisheng accepts, unlike `T(a, b)`).
  PrimType elem_dtype = value_dtype.WithLanes(1);
  const RampNode *ramp = index.as<RampNode>();
  std::ostringstream init;
  PrintType(value_dtype, init);
  init << "{";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) {
      init << ", ";
    }
    PrimExpr lane_index =
        ramp ? (ramp->base +
                ramp->stride * MakeConst(PythonDType(ramp->base), i))
             : index;
    std::string lane_ref =
        GetBufferRef(elem_dtype, op->source.as_or_throw<TensorVar>().get(),
                     analyzer_->Simplify(lane_index));
    init << lane_ref;
  }
  init << "}";
  os << init.str();
}

void CodeGenTileLangAscend::Dispatch_(const TensorStoreNode *op) {
  PrimType value_dtype = PythonDType(op->value);
  PrimType element_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  const VarNode *buf_var = op->dest.as_or_throw<TensorVar>().get();
  PrimExpr index_expr = op->indices[0];

  // A vector store that stays on the scalar core (outside any VF body) cannot
  // use the simt-only vector constructors a broadcast needs (make_float2,
  // make_int2, ...); bisheng rejects them outside a VF body. Such a store is
  // element-wise scalar work, so emit one scalar store per lane.
  if (IsOutsideVF() && value_dtype.lanes() > 1 &&
      value_dtype.lanes() != element_dtype.lanes() && op->indices.size() == 1 &&
      !IsGlobalGmBuffer(op->dest.as_or_throw<TensorVar>().get()) &&
      EmitOutOfVFBroadcastStore(op)) {
    return;
  }

  if (value_dtype.lanes() > 1 && op->indices.size() == 1) {
    const auto *broadcast_index = index_expr.as<BroadcastNode>();
    auto scope_it = alloc_storage_scope_.find(buf_var);
    std::string buffer_scope = op->dest.as_or_throw<TensorVar>().scope().name();
    bool is_local_scope = buffer_scope == "local" ||
                          buffer_scope == "local.var" ||
                          (scope_it != alloc_storage_scope_.end() &&
                           (scope_it->second.name() == "local" ||
                            scope_it->second.name() == "local.var"));
    if (broadcast_index && is_local_scope &&
        value_dtype.lanes() == (broadcast_index->lanes)
                                   .as_or_throw<IntImm>()
                                   ->value.as<int64_t>()
                                   .value()) {
      std::string buf_name = GetVarID(buf_var);
      std::string value = PrintExpr(op->value);

      PrintIndent();
      stream << "*(";
      PrintType(value_dtype, stream);
      stream << "*)(" << buf_name;
      if (!is_zero(broadcast_index->value)) {
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
      std::string scope = it->second.name();
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
    const TensorStoreNode *op) {
  const auto *broadcast = op->value.as<BroadcastNode>();
  const auto *ramp = op->indices[0].as<RampNode>();
  if (broadcast == nullptr || ramp == nullptr || !IsOne(ramp->stride) ||
      op->dest.as_or_throw<TensorVar>().dtype().lanes() != 1) {
    return false;
  }
  const IntImmNode *lanes =
      (broadcast->lanes).as_or_throw<IntImm>().as<IntImmNode>();
  ICHECK(lanes != nullptr) << "Broadcast lanes must be constant";
  PrimType elem_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  SSAOperationScope ssa_scope(this);
  std::string value = SSAGetID(PrintExpr(broadcast->value), elem_dtype);
  for (int i = 0; i < lanes->value; ++i) {
    PrimExpr lane_index =
        analyzer_->Simplify(ramp->base + MakeConst(PythonDType(ramp->base), i));
    PrintIndent();
    stream << GetBufferRef(elem_dtype, op->dest.as_or_throw<TensorVar>().get(),
                           lane_index)
           << " = " << value << ";\n";
  }
  return true;
}

bool CodeGenTileLangAscend::EmitScalarizedStore(const TensorStoreNode *op) {
  PrimType value_dtype = PythonDType(op->value);
  ValidateGenericVectorType("BufferStore", value_dtype);
  int lanes = value_dtype.lanes();
  PrimExpr index = op->indices[0];

  // A vectorized GM access on the scalar core (outside any VF body) has no
  // valid lowering: the only scalar-pipe GM path is the per-element
  // write_gm_bypass_dcache emitted in VisitStmt_. Vector GM traffic must go
  // through an MTE copy (T.copy) or a SimtVF/SimdVF region.
  if (IsOutsideVF() &&
      IsGlobalGmBuffer(op->dest.as_or_throw<TensorVar>().get())) {
    LOG(FATAL)
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
  const RampNode *ramp = index.as<RampNode>();
  ICHECK(ramp) << "scalarized store expects a ramp index, got " << index;
  PrimType elem_dtype = value_dtype.WithLanes(1);
  SSAOperationScope ssa_scope(this);
  std::string value = SSAGetID(PrintExpr(op->value), value_dtype);
  for (int i = 0; i < lanes; ++i) {
    PrimExpr lane_index =
        ramp->base + ramp->stride * MakeConst(PythonDType(ramp->base), i);
    std::string lane_ref =
        GetBufferRef(elem_dtype, op->dest.as_or_throw<TensorVar>().get(),
                     analyzer_->Simplify(lane_index));
    std::ostringstream lane_val;
    PrintVecElemLoad(value, value_dtype, i, lane_val);
    PrintIndent();
    stream << lane_ref << " = " << lane_val.str() << ";\n";
  }
  return true;
}

void CodeGenTileLangAscend::Dispatch_(const SelectNode *op,
                                      std::ostream &os) { // NOLINT(*)
  if (!PythonDType(op->condition).IsFixedLengthVector()) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  ValidateGenericVectorType("Select", PythonDType(op->condition));
  ValidateGenericVectorType("Select", PythonDType(GetRef<Expr>(op)));

  if (const auto *broadcast = op->condition.as<BroadcastNode>()) {
    std::string condition = PrintExpr(broadcast->value);
    std::string true_value = PrintExpr(op->true_value);
    std::string false_value = PrintExpr(op->false_value);
    os << "((" << condition << ") ? (" << true_value << ") : (" << false_value
       << "))";
    return;
  }

  TVM_FFI_ICHECK(PythonDType(op->false_value) ==
                     PythonDType(GetRef<Expr>(op)) &&
                 PythonDType(op->true_value) == PythonDType(GetRef<Expr>(op)) &&
                 PythonDType(GetRef<Expr>(op)).lanes() ==
                     PythonDType(op->condition).lanes());

  std::string result = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(PythonDType(GetRef<Expr>(op)), stream);
  stream << ' ' << result << ";\n";
  SSAOperationScope ssa_scope(this);
  {
    std::string condition =
        SSAGetID(PrintExpr(op->condition), PythonDType(op->condition));
    std::string true_value =
        SSAGetID(PrintExpr(op->true_value), PythonDType(GetRef<Expr>(op)));
    std::string false_value =
        SSAGetID(PrintExpr(op->false_value), PythonDType(GetRef<Expr>(op)));

    int lanes = PythonDType(GetRef<Expr>(op)).lanes();
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value;
      value << "(bool(";
      PrintVecElemLoad(condition, PythonDType(op->condition), i, value);
      value << ")?";
      PrintVecElemLoad(true_value, PythonDType(GetRef<Expr>(op)), i, value);
      value << ':';
      PrintVecElemLoad(false_value, PythonDType(GetRef<Expr>(op)), i, value);
      value << ')';
      PrintVecElemStore(result, PythonDType(GetRef<Expr>(op)), i, value.str());
    }
  }
  os << result;
}

void CodeGenTileLangAscend::Dispatch_(const ShuffleNode *op,
                                      std::ostream &os) { // NOLINT(*)
  for (const PrimExpr &vector : op->vectors) {
    if (PythonDType(vector).lanes() > 1) {
      ValidateGenericVectorType("Shuffle", PythonDType(vector));
    }
  }
  if (PythonDType(GetRef<Expr>(op)).lanes() > 1) {
    ValidateGenericVectorType("Shuffle", PythonDType(GetRef<Expr>(op)));
  }
  SSAOperationScope ssa_scope(this);

  // Materialize the concatenated input lanes once before applying the shuffle
  // indices. Bisheng does not accept the generic C codegen's vector
  // constructors such as float2(a, b), and packed FP16/BF16 vectors require
  // element-wise access through their uint carrier.
  std::vector<std::string> input_lanes;
  for (const PrimExpr &vector : op->vectors) {
    std::string vector_value = SSAGetID(PrintExpr(vector), PythonDType(vector));
    if (PythonDType(vector).IsScalar()) {
      input_lanes.push_back(std::move(vector_value));
      continue;
    }
    for (int i = 0; i < PythonDType(vector).lanes(); ++i) {
      std::ostringstream element;
      PrintVecElemLoad(vector_value, PythonDType(vector), i, element);
      input_lanes.push_back(element.str());
    }
  }

  std::vector<std::string> output_lanes;
  output_lanes.reserve(op->indices.size());
  for (const PrimExpr &index : op->indices) {
    const auto *constant_index = index.as<IntImmNode>();
    TVM_FFI_ICHECK(constant_index != nullptr)
        << "ShuffleNode indices must be constants at Ascend codegen time, got "
        << index;
    TVM_FFI_ICHECK_GE(constant_index->value.as<int64_t>().value(), 0)
        << "ShuffleNode index must be non-negative, got "
        << constant_index->value.as<int64_t>().value();
    TVM_FFI_ICHECK_LT(
        static_cast<size_t>(constant_index->value.as<int64_t>().value()),
        input_lanes.size())
        << "ShuffleNode index " << constant_index->value.as<int64_t>().value()
        << " is outside the concatenated input with " << input_lanes.size()
        << " lanes";
    output_lanes.push_back(
        input_lanes[constant_index->value.as<int64_t>().value()]);
  }

  if (PythonDType(GetRef<Expr>(op)).IsScalar()) {
    TVM_FFI_ICHECK_EQ(output_lanes.size(), 1);
    os << output_lanes[0];
    return;
  }

  TVM_FFI_ICHECK_EQ(static_cast<int>(output_lanes.size()),
                    PythonDType(GetRef<Expr>(op)).lanes());
  std::string result = name_supply_->FreshName("_");
  PrintIndent();
  PrintType(PythonDType(GetRef<Expr>(op)), stream);
  stream << ' ' << result << ";\n";
  for (int i = 0; i < PythonDType(GetRef<Expr>(op)).lanes(); ++i) {
    PrintVecElemStore(result, PythonDType(GetRef<Expr>(op)), i,
                      output_lanes[i]);
  }
  os << result;
}

void CodeGenTileLangAscend::Dispatch_(const ForNode *op) {
  auto factor = op->annotations.Get(tirx::attr::kPragmaUnroll);
  if (op->annotations.count(tirx::attr::kDisableUnroll.name)) {
    PrintIndent();
    stream << "#pragma unroll 1\n";
  } else if (factor || op->kind == ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll";
    if (factor) {
      auto count = tl::GetIntAnnotation(*factor);
      TVM_FFI_CHECK(count && *count > 0, ValueError)
          << "Ascend unroll factor must be a positive integer";
      stream << " " << *count;
    }
    stream << "\n";
  }
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangAscend::Dispatch_(const WhileNode *op) {
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

void CodeGenTileLangAscend::PrintAllocTensor(const BindNode *op,
                                             const CallNode *allocation) {
  auto tensor = op->var.as_or_throw<TensorVar>();
  auto annotations =
      allocation->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict;
  std::string scope = PythonStorageScope(tensor);

  if (scope == "shared") {
    LOG(FATAL) << "Static shared memory (scope='shared') is not supported on "
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
    if (tl::IsSSBuffer(tensor)) {
      ICHECK(kernel_mode_ == AscendKernelMode::kMix && mix_aiv_count_ == 2)
          << "SS buffers require a MIX kernel with two AIVs";
      auto elements = tl::GetConstInt(sym::Analyzer()->Simplify(
          floordiv(tl::GetBufferStorageSizeBytes(tensor),
                   static_cast<int64_t>(tensor.dtype().StorageBytes()))));
      ICHECK(elements.has_value() && elements.value() > 0 &&
             elements.value() <= 3072 / dtype.StorageBytes())
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
    ICHECK_EQ(tl::GetConstInt(PythonTensorSize(tensor)).value_or(0), 1)
        << "local.var should be scalar";
    PrimType dtype = tensor.dtype();
    std::string vid = AllocVarID(tensor.get());
    ffi::Optional<PrimExpr> init;
    if (!dtype.MatchesCode(kDLOpaqueHandle)) {
      init = MakeConst(dtype, 0);
    }
    auto init_it = annotations.find(tl::attr::kLocalVarInit);
    if (init_it != annotations.end()) {
      PrimExpr user_init = (*init_it).second.cast<PrimExpr>();
      if (!PythonDType(user_init).IsVoid() && PythonDType(user_init) != dtype) {
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

  // For other scopes, use parent class implementation
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangAscend::Dispatch_(const BindNode *op) {
  if (const auto *allocation = op->value.as<CallNode>()) {
    if (allocation->op.same_as(tirx::alloc_tensor_op())) {
      PrintAllocTensor(op, allocation);
      return;
    }
    if (allocation->op.same_as(tirx::decl_tensor_op())) {
      CodeGenC::Dispatch_(op);
      return;
    }
  }
  // Inside SimdVF body, bindings for vector registers and masks need concrete
  // CCE-friendly types instead of generic C handle printing.
  if (IsInsideSimdVF()) {
    if (auto primitive = op->value->ty.as<PrimType>();
        primitive && primitive->lanes() > 1) {
      ValidateSimdVectorType(PythonDType(op->value), "SimdVF binding");
    }
    std::string vid = AllocVarID(op->var.get());
    if (PythonIsPointer(op->var) && op->var->ty.as<PtrTypeNode>()) {
      const auto *ptr = op->var->ty.as<PtrTypeNode>();
      std::string scope = PythonStorageScope(op->var);
      if (scope == "local.fragment") {
        const auto *prim = ptr->element_type.as<PrimTypeNode>();
        ICHECK(prim)
            << "SimdVF local.fragment bind expects primitive element type";
        PrintIndent();
        stream << "vector_";
        PrintCCEVectorSuffix(GetRef<PrimType>(prim), stream);
        stream << " " << vid << ";\n";
        alloc_storage_scope_[op->var.get()] = runtime::MemoryScope(scope);
        RegisterHandleType(op->var.get(), GetRef<PrimType>(prim));
        return;
      }
    }
    if (const auto *call = op->value.as<CallNode>()) {
      // simd_pset/pge(elem_width_bits, dist) -> vector_bool vid = p*_bXX(dist);
      if (call->op.same_as(tl::simd_pset()) ||
          call->op.same_as(tl::simd_pge())) {
        int elem_bits = (int)(call->args[0])
                            .as_or_throw<IntImm>()
                            ->value.as<int64_t>()
                            .value();
        std::string dist =
            (call->args.size() >= 2 && call->args[1].as<StringImmNode>())
                ? (call->args[1]).as_or_throw<StringImm>()->value
                : "PAT_ALL";
        PrintIndent();
        stream << "vector_bool " << vid << " = asc_create_mask_b" << elem_bits
               << "(" << dist << ");\n";
        return;
      }
      // pair_get(pair, idx) -> inline as pair.vN, no auto decl
      if (call->op.same_as(tl::simd_pair_get())) {
        int idx = (int)(call->args[1])
                      .as_or_throw<IntImm>()
                      ->value.as<int64_t>()
                      .value();
        std::ostringstream tmp;
        PrintExpr(call->args[0], tmp);
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
  // CodeGenC::VisitStmt_ only inspects the scalar PrimType (always "handle")
  // and emits a scope-less `void*`. That loses the `__gm__`/`__ubuf__`
  // qualifier and the element type, so later casts to `__gm__ T*` are rejected
  // by bisheng. Read the PointerType directly and emit the qualified pointer.
  if (PythonIsPointer(op->var)) {
    if (const auto *ptr = op->var->ty.as<PtrTypeNode>()) {
      if (const auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        std::string scope = PythonStorageScope(op->var);
        // Scoped pointer type, e.g. "__gm__ float *".
        std::ostringstream type_os;
        PrintStorageScope(runtime::MemoryScope(scope), type_os);
        PrintType(GetRef<PrimType>(prim), type_os);
        type_os << " *";
        std::string ptr_type = type_os.str();

        // A reinterpret from an integer address (the make_tensor case) is
        // emitted by the base codegen as a `void*` bit-cast, which bisheng
        // refuses to convert to a scoped pointer. Emit the bit-cast directly
        // against the scoped pointer type so no `void*` intermediate appears.
        const auto *call = op->value.as<CallNode>();
        if (call && PythonIsOp(call->op, "reinterpret") &&
            !PythonIsPointer(call->args[0])) {
          SSAOperationScope ssa_scope(this);
          std::string rhs =
              SSAGetID(PrintExpr(call->args[0]), PythonDType(call->args[0]));
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
        RegisterHandleType(op->var.get(), GetRef<PrimType>(prim));
        return;
      }
    }
  }
  CodeGenC::Dispatch_(op);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscend *p) { // NOLINT(*)
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16) ||
      PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLFloat, 16) ||
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat)) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      temp << "tl::limits::";
      if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat)) {
        temp << (op->value < 0 ? "kFloatNInf" : "kFloatInf");
      } else if (PythonDType(GetRef<Expr>(op))
                     .MatchesElementType(kDLFloat, 16)) {
        temp << (op->value < 0 ? "kHalfNInf" : "kHalfInf");
      } else {
        temp << (op->value < 0 ? "kBf16NInf" : "kBf16Inf");
      }
    } else if (std::isnan(op->value)) {
      temp << "tl::limits::";
      if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat)) {
        temp << "kFloatNaN";
      } else if (PythonDType(GetRef<Expr>(op))
                     .MatchesElementType(kDLFloat, 16)) {
        temp << "kHalfNaN";
      } else {
        temp << "kBf16NaN";
      }
    } else {
      p->PrintType(PythonDType(GetRef<Expr>(op)), temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (PythonDType(GetRef<Expr>(op))
          .MatchesCode(kDLFloat8_e4m3, kDLFloat8_e4m3fn, kDLFloat8_e5m2,
                       kDLFloat8_e8m0fnu) ||
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat4_e2m1fn)) {
    p->PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat64/kFloat32 (kFloat16 is handled above)
  switch (PythonDType(GetRef<Expr>(op)).bits()) {
  default:
    LOG(FATAL) << "Bad bit-width for float: " << PythonDType(GetRef<Expr>(op))
               << "\n";
  }
}

void CodeGenTileLangAscend::Dispatch_(const FloatImmNode *op,
                                      std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscend::Dispatch_(const CastNode *op,
                                      std::ostream &os) { // NOLINT(*)
  PrimType from_ty = PythonDType(op->value);
  PrimType target_ty = PythonDType(GetRef<Expr>(op));
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
  std::string src = SSAGetID(PrintExpr(op->value), from_ty);

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
    bool target_type_is_e4m3 = target_ty.MatchesCode(kDLFloat8_e4m3) ||
                               target_ty.MatchesCode(kDLFloat8_e4m3fn);
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
    bool target_type_is_e4m3 = target_ty.MatchesCode(kDLFloat8_e4m3) ||
                               target_ty.MatchesCode(kDLFloat8_e4m3fn);
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
    bool target_type_is_e4m3 = target_ty.MatchesCode(kDLFloat8_e4m3) ||
                               target_ty.MatchesCode(kDLFloat8_e4m3fn);
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
    bool from_type_is_e4m3 = from_ty.MatchesCode(kDLFloat8_e4m3) ||
                             from_ty.MatchesCode(kDLFloat8_e4m3fn);
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
    bool from_type_is_e4m3 = from_ty.MatchesCode(kDLFloat8_e4m3) ||
                             from_ty.MatchesCode(kDLFloat8_e4m3fn);
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
    bool from_type_is_e4m3 = from_ty.MatchesCode(kDLFloat8_e4m3) ||
                             from_ty.MatchesCode(kDLFloat8_e4m3fn);
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

void CodeGenTileLangAscend::Dispatch_(const VarNode *op,
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
    PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << ")" << it->second << ")";
    return;
  }
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangAscend::PrintStorageSync(const CallNode *op) {
  ICHECK_GE(op->args.size(), 1U);
  PrintIndent();
  stream << "asc_syncthreads();\n";
}

void CodeGenTileLangAscend::PrintStorageScope(
    const runtime::MemoryScope &memory_scope, std::ostream &os) {
  std::string scope = memory_scope.name();
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
  } else if (t.MatchesCode(kDLBool) && t.lanes() > 4) {
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
  } else if (t.MatchesCode(kDLBool) && t.lanes() > 4) {
    stream << "((ushort2*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.lanes() <= 4) {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  } else {
    stream << "((" << vec << ")." << access[i / 2] << ") = " << value << ";\n";
  }
}

std::string CodeGenTileLangAscend::GetBufferRef(const PrimType &t,
                                                const VarNode *buffer,
                                                PrimExpr index) {
  if (tl::IsSSBuffer(GetRef<Var>(buffer).as_or_throw<TensorVar>())) {
    ICHECK(vf_mode_ == VFMode::kNone && t.MatchesCode(kDLUInt) &&
           (t.bits() == 32 || t.bits() == 64) && t.lanes() == 1)
        << "SS supports scalar uint32/uint64 accesses outside VF blocks only";
  }
  const VarNode *buffer_var = GetRef<Var>(buffer).get();
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
      t.MatchesCode(kDLBool) || (lacks_vector_operation && t.lanes() > 1) ||
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
    std::string vlhs = SSAGetID(PrintExpr(lhs), PythonDType(lhs));
    std::string vrhs = SSAGetID(PrintExpr(rhs), PythonDType(rhs));
    for (int i = 0; i < t.lanes(); ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, PythonDType(lhs), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, PythonDType(rhs), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, PythonDType(lhs), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, PythonDType(rhs), i, value_temp);
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
  ICHECK(dtype.MatchesCode(kDLBool) && dtype.lanes() <= 8)
      << "Ascend SIMT predicate not expects a small predicate vector, got "
      << dtype;
  ICHECK_EQ(PythonDType(value), dtype)
      << "Ascend small-vector predicate not dtype mismatch";

  std::string result = name_supply_->FreshName("_");
  PrintIndent();
  PrintType(dtype, stream);
  stream << ' ' << result << ";\n";
  SSAOperationScope ssa_scope(this);
  std::string operand = SSAGetID(PrintExpr(value), PythonDType(value));
  for (int i = 0; i < dtype.lanes(); ++i) {
    std::ostringstream lane_value;
    lane_value << "(!";
    PrintVecElemLoad(operand, PythonDType(value), i, lane_value);
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
  PrimType ret_dtype =
      (IsVoidType(ret_type) ? PrimType::Void()
                            : ret_type.as_or_throw<PrimType>());
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
        std::string val = SSAGetID(PrintExpr(args[i]), PythonDType(args[i]));
        sargs.push_back(std::move(val));
      }
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], PythonDType(args[arg_begin + j]), i,
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
