#include "codegen_ascend.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include "arith/pattern_match.h"

#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace codegen {
using ffi::GetRef;

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

bool IsVmiOp(const CallNode *op, std::string *op_name = nullptr) {
  if (auto opt_call_op = op->op.as<Op>()) {
    const std::string &name = opt_call_op.value()->name;
    if (name.rfind("tl.vmi.", 0) == 0) {
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
  if (!op->dtype.is_void() || op->args.empty()) {
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
bool SupportsNativeSimdMerging(const std::string &name, DataType dtype) {
  const bool is_integer =
      (dtype.is_int() || dtype.is_uint()) &&
      (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32);
  const bool is_float =
      dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32);
  if (!is_integer && !is_float && !dtype.is_bfloat16()) {
    return false;
  }
  if (name == "vdup") {
    // CANN 9.2's scalar BF16 overload merges into an uninitialized temporary
    // instead of the old destination. Vector broadcast uses a different
    // overload.
    return !dtype.is_bfloat16();
  }
  if (name == "vabs" || name == "vneg") {
    return (is_integer && dtype.is_int()) || is_float;
  }
  if (name == "vabsdif" || name == "vexp" || name == "vln" || name == "vsqrt" ||
      name == "vaxpy") {
    return is_float;
  }
  if (name == "vrelu") {
    return is_float || (dtype.is_int() && dtype.bits() == 32);
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
SfuPrecision ResolveSfuPrecision(const Call &op, DataType result_dtype,
                                 SfuPrecision fallback) {
  // Precise paths (vdiv_0ulp_ftz_true, *_ftz_false wrappers) are float32-only:
  // non-fp32 keeps the hardware instruction regardless of annotations
  // (historical UsePreciseVdiv contract: "non-fp32 always uses hardware").
  if (!result_dtype.is_float() || result_dtype.bits() != 32) {
    return fallback;
  }
  // Per-op "precision" annotation (int code from the Python side,
  // normalized from the string aliases in tilelang/ascend/language/simd.py
  // -- no alias table here, mirroring the l2_cache_ctrl pattern).
  if (auto p = op->annotations.Get("precision")) {
    return PrecisionFromCode(Downcast<IntImm>(p.value())->value);
  }
  return fallback;
}

bool IsAscendWarpReduceDType(DataType dtype) {
  return dtype.is_scalar() &&
         ((dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
          ((dtype.is_int() || dtype.is_uint()) && dtype.bits() == 32));
}

static std::string GetAscendScopeQualifier(const std::string &scope) {
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
std::string CCEVectorSuffix(DataType dtype) {
  if (dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn())
    return "f8e4m3";
  if (dtype.is_float8_e5m2())
    return "f8e5m2";
  if (dtype.is_float8_e8m0fnu())
    return "f8e8m0";
  if (dtype.is_float4_e2m1fn())
    return "f4e2m1x2";
  if (dtype.is_float()) {
    if (dtype.bits() == 16)
      return "f16";
    if (dtype.bits() == 32)
      return "f32";
    LOG(FATAL) << "Unsupported Ascend SIMD floating-point element type "
               << dtype;
  }
  if (dtype.is_bfloat16())
    return "bf16";
  if (dtype.is_int() && (dtype.bits() == 8 || dtype.bits() == 16 ||
                         dtype.bits() == 32 || dtype.bits() == 64))
    return "s" + std::to_string(dtype.bits());
  if (dtype.is_uint() && (dtype.bits() == 8 || dtype.bits() == 16 ||
                          dtype.bits() == 32 || dtype.bits() == 64))
    return "u" + std::to_string(dtype.bits());
  LOG(FATAL) << "Unsupported Ascend SIMD vector element type " << dtype;
  return "";
}

// Get element width in bits for choosing b8/b16/b32 predicate/load/store
// variants.
int CCEElemWidthBits(DataType dtype) {
  if (dtype.is_float4_e2m1fn())
    return 8;
  return dtype.bits();
}

// Map DataType to the C++ type name used in __ubuf__ pointer casts.
std::string CCEUBufType(DataType dtype) {
  if (dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn())
    return "float8_e4m3_t";
  if (dtype.is_float8_e5m2())
    return "float8_e5m2_t";
  if (dtype.is_float8_e8m0fnu())
    return "float8_e8m0_t";
  if (dtype.is_float4_e2m1fn())
    return "float4_e2m1x2_t";
  if (dtype.is_float())
    return (dtype.bits() == 16) ? "half" : "float";
  if (dtype.is_bfloat16())
    return "bfloat16_t";
  if (dtype.is_int())
    return "int" + std::to_string(dtype.bits()) + "_t";
  if (dtype.is_uint())
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
  return value->value;
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
      pass_ctx->GetConfig<Bool>(tl::kEnableFastMath, Bool(false)).value();
  decl_stream << "#include <tl_templates/ascend/common.h>\n";
  decl_stream << "#include <tl_templates/ascend/debug.h>\n";
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

void CodeGenTileLangAscend::PrintCCEVectorSuffix(DataType t, std::ostream &os) {
  os << CCEVectorSuffix(t);
}

static std::string GetAscendFP8Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
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
  if (type.is_float8_e4m3() || type.is_float8_e4m3fn()) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2()) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.is_float8_e8m0fnu()) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported Ascend fp8 type " << type;
  }
  return stream.str();
}

static std::string GetAscendFP8ScalarValueType(DataType type) {
  if (type.is_float8_e4m3() || type.is_float8_e4m3fn()) {
    return "tl::float_e4m3_t";
  }
  if (type.is_float8_e5m2()) {
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
    DataType dtype, const std::string &context) const {
  if (dtype.lanes() <= 1) {
    LOG(FATAL) << context << " expects an Ascend SIMD vector register, got "
               << dtype;
  }
  if (dtype.is_bool()) {
    if (dtype.lanes() != 256) {
      LOG(FATAL) << context << " has invalid SIMD predicate type " << dtype
                 << ": SimdVF predicates must be boolx256";
    }
    return;
  }

  DataType element_dtype = dtype.element_of();
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
                                                      DataType dtype) const {
  if (IsInsideSimdVF() && dtype.lanes() > 1) {
    LOG(FATAL) << "Generic vector " << op_name << " with dtype " << dtype
               << " cannot be emitted inside SimdVF; use an explicit "
                  "T.simd.* operation or fix the preceding SIMD lowering";
  }
}

void CodeGenTileLangAscend::ValidateGenericVectorType(const char *op_name,
                                                      DataType dtype) {
  RejectGenericVectorInSimd(op_name, dtype);
  std::ostringstream type;
  PrintType(dtype, type);
}

void CodeGenTileLangAscend::PrintType(DataType t, std::ostream &os) {
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types for handles";
    os << "void*";
    return;
  }

  if (IsInsideSimdVF() && lanes > 1) {
    ValidateSimdVectorType(t, "SimdVF type");
    os << (t.is_bool() ? "vector_bool"
                       : "vector_" + CCEVectorSuffix(t.element_of()));
    return;
  }
  if (lanes > 1) {
    // SIMT vector comparisons use one ushort per predicate lane. Wider
    // predicates and 16-bit values use uint carriers.
    if (t.is_bool() && lanes <= 8) {
      ICHECK(lanes <= 4 || lanes % 2 == 0)
          << "Ascend wide vector bool type requires an even lane count, got "
          << t;
      os << (lanes <= 4 ? "ushort" + std::to_string(lanes)
                        : "uint" + std::to_string(lanes / 2));
      return;
    }
    if ((t.is_float16() || t.is_bfloat16()) && lanes <= 8 && lanes % 2 == 0) {
      os << "uint" << lanes / 2;
      return;
    }
    if ((t.is_float() && t.bits() == 32) ||
        ((t.is_int() || t.is_uint()) && t.bits() == 32)) {
      if (lanes == 2 || lanes == 4) {
        os << (t.is_float() ? "float" : t.is_int() ? "int" : "uint") << lanes;
        return;
      }
    }
    if (t.is_float8() && (lanes == 2 || lanes == 4 || lanes == 8)) {
      os << GetAscendFP8Type(t);
      return;
    }
    if (t.is_float4_e2m1fn() && lanes == 2) {
      os << "float4_e2m1x2_t";
      return;
    }
    LOG(FATAL) << VFModeName(vf_mode_) << " cannot convert vector type " << t
               << " to an Ascend type";
  }

  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }

  if (t.is_float()) {
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
  } else if (t.is_bfloat16()) {
    os << "bfloat16_t";
    return;
  } else if (t.is_float8()) {
    os << (IsInsideSimtVF() ? GetAscendFP8ScalarValueType(t)
                            : GetAscendFP8Type(t));
    return;
  } else if (t.is_float4_e2m1fn()) {
    os << "float4_e2m1x2_t";
    return;
  } else if (t.is_int() || t.is_uint()) {
    if (t.bits() == 8 || t.bits() == 16 || t.bits() == 32 || t.bits() == 64) {
      os << (t.is_int() ? "int" : "uint") << t.bits() << "_t";
      return;
    }
  }
  LOG(FATAL) << "Cannot convert scalar type " << t << " to Ascend type";
}

void CodeGenTileLangAscend::AddFunction(const PrimFunc &f) {
  ICHECK(IsOutsideVF())
      << "Ascend VF mode leaked across PrimFunc code generation";
  this->InitFuncState(f);
  has_gemm_l0_ = false;
  has_gemm_l1_ = false;
  unroll_factor_.clear();
  ReserveKeywordsAsUnique();
  name_supply_->ReserveName("block_idx");

  // Register function parameters with their storage scope for codegen
  for (const auto &param : f->params) {
    if (param->dtype.is_handle()) {
      auto scope = GetPtrStorageScope(param);
      alloc_storage_scope_[param.get()] = scope;
    }
  }

  // Pre-scan body to determine kernel mode (Mix / Vector / Cube)
  bool has_cube = false, has_vector = false;
  bool has_nd2nz_post_copy = false;
  bool has_cooperative_groups = false;
  int vector_count = 2;
  tirx::PostOrderVisit(f->body, [&](const ffi::ObjectRef &n) {
    if (const auto *block = n.as<SBlockNode>()) {
      if (block->name_hint == "CUBE")
        has_cube = true;
      else if (block->name_hint == "VECTOR") {
        has_vector = true;
        if (auto opt = block->annotations.Get("vector_count")) {
          const auto *count = opt.value().as<IntImmNode>();
          ICHECK(count != nullptr && (count->value == 1 || count->value == 2))
              << "Mixed-kernel vector_count must be the constant integer 1 or "
                 "2, got "
              << opt.value();
          vector_count = count->value;
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
    decl_stream << "#include <tl_templates/ascend/gemm.h>\n";
    has_gemm_l1_included_ = true;
  }

  // Conditionally include nd2nz_copy.h when ascend_nd2nz_copy is used
  if (has_nd2nz_copy_ && !has_nd2nz_copy_included_) {
    decl_stream << "#include <tl_templates/ascend/nd2nz_copy.h>\n";
    has_nd2nz_copy_included_ = true;
  }

  // Conditionally include dcache_bypass.h when scalar global GM access emits
  // tl::read/write_gm_bypass_dcache
  if (has_gm_bypass_dcache_ && !has_gm_bypass_dcache_included_) {
    decl_stream << "#include <tl_templates/ascend/dcache_bypass.h>\n";
    has_gm_bypass_dcache_included_ = true;
  }

  // Conditionally include philox_rng.h when tl.rng_init is used
  if (has_philox_rng_ && !has_philox_rng_included_) {
    decl_stream << "#include <tl_templates/ascend/philox_rng.h>\n";
    has_philox_rng_included_ = true;
  }

  // sync_warp lowers to cooperative_groups::coalesced_threads().sync().
  if (has_cooperative_groups && !has_cooperative_groups_included_) {
    decl_stream << "#include <simt_api/cooperative_groups.h>\n";
    has_cooperative_groups_included_ = true;
  }

  if (has_cube && has_vector) {
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
  bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  CodeGenC::PrintExtraAttrs(f, stream);
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tirx::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) {
      stream << ", ";
    }
    if (v.dtype().is_handle()) {
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
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

      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias) {
        PrintRestrict(v, stream);
      }
    } else {
      CodeGenC::PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";
  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
  ICHECK(IsOutsideVF())
      << "Ascend VF mode leaked after PrimFunc code generation";
}

void CodeGenTileLangAscend::PreFunctionBody(const PrimFunc &f) {
  this->stream << "  asc_init();\n";
}

ffi::Array<Var>
CodeGenTileLangAscend::CollectVFCaptures(const SBlockNode *op) const {
  ffi::Array<Var> predefined;
  for (const Buffer &buf : op->alloc_buffers) {
    predefined.push_back(buf->data);
  }

  ffi::Array<Var> undefined = UndefinedVars(op->body, predefined);
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
    if (op->name_hint == "SIMT_VF" && it != var_idmap_.end() &&
        it->second == "block_idx") {
      continue;
    }
    seen.insert(var.get());

    const DataType dtype = var->dtype;
    ICHECK(dtype.lanes() == 1 &&
           (dtype.is_int() || dtype.is_uint() || dtype.is_float() ||
            dtype.is_handle() || dtype.is_bool()))
        << op->name_hint << " capture variable `" << var
        << "` has unsupported dtype `" << dtype
        << "`. Only scalar int/uint/float/bool/handle are supported.";
    captures.push_back(var);
  }
  return captures;
}

void CodeGenTileLangAscend::EmitVFFunction(const SBlockNode *op,
                                           const ffi::Array<Var> &captures,
                                           const std::string &helper_name,
                                           const std::string &func_attrs,
                                           VFMode vf_mode) {
  for (const auto &v : captures) {
    if (v->type_annotation.as<PointerTypeNode>()) {
      alloc_storage_scope_[v.get()] = GetPtrStorageScope(v);
    }
  }

  std::unordered_map<const VarNode *, std::string> saved_var_idmap;
  saved_var_idmap.swap(var_idmap_);
  NameSupply saved_name_supply = name_supply_;
  name_supply_ = NameSupply();
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
        (it != var_idmap_.end()) ? it->second : std::string(v->name_hint);
    if (name.find('.') != std::string::npos) {
      std::replace(name.begin(), name.end(), '.', '_');
      var_idmap_[v.get()] = name;
    }
    vf_param_name[v.get()] = name;
  }
  // Reserve all capture parameter names in the fresh NameSupply so that
  // newly generated variables inside the VF body do not shadow them.
  for (const auto &v : captures) {
    auto it = var_idmap_.find(v.get());
    std::string cap_name =
        (it != var_idmap_.end()) ? it->second : std::string(v->name_hint);
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
    this->VisitStmt(op->body);
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
    return v->name_hint;
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
    if (!v->type_annotation.as<PointerTypeNode>()) {
      PrintType(v.dtype(), decl_stream);
      decl_stream << " " << vname;
    } else {
      auto scope_it = alloc_storage_scope_.find(v.get());
      if (scope_it != alloc_storage_scope_.end() &&
          scope_it->second == "local.var") {
        auto it = handle_data_type_.find(v.get());
        if (it != handle_data_type_.end()) {
          PrintType(it->second, decl_stream);
        } else {
          PrintType(v.dtype(), decl_stream);
        }
        decl_stream << " " << vname;
      } else {
        auto it = handle_data_type_.find(v.get());
        if (it == handle_data_type_.end()) {
          for (const auto &kv : handle_data_type_) {
            if (kv.first->name_hint == v->name_hint) {
              it = handle_data_type_.find(kv.first);
              break;
            }
          }
        }
        PrintStorageScope(GetPtrStorageScope(v), decl_stream);
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

void CodeGenTileLangAscend::VisitStmt_(const SBlockNode *op) {
  if (op->name_hint == "CUBE") {
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
  if (op->name_hint == "VECTOR") {
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
  if (op->name_hint == "SIMD_VF") {
    if (!has_simd_inst_included_) {
      decl_stream << "#include <tl_templates/ascend/simd_inst.h>\n";
      has_simd_inst_included_ = true;
    }
    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simdvf_helper_counter_++;
    {
      auto it = op->annotations.find("tl.vf_source_index");
      if (it != op->annotations.end()) {
        if (auto *imm = (*it).second.as<IntImmNode>()) {
          vf_idx = imm->value;
        }
      }
    }
    std::string helper_name =
        current_function_name_ + "_simd_vf_" + std::to_string(vf_idx);
    EmitVFFunction(op, captures, helper_name, "__simd_vf__ inline void",
                   VFMode::kSimd);

    PrintIndent();
    stream << helper_name << "(";
    for (size_t i = 0; i < captures.size(); ++i) {
      if (i != 0)
        stream << ", ";
      auto it = var_idmap_.find(captures[i].get());
      stream << (it != var_idmap_.end() ? it->second
                                        : std::string(captures[i]->name_hint));
    }
    stream << ");\n";
    return;
  }
  if (op->name_hint == "SIMT_VF") {
    int64_t thread_x = 1, thread_y = 1, thread_z = 1;
    // Detect the threadIdx.{x,y,z} extents with a visitor rather than assuming
    // the thread_extent AttrStmts lead the block body. Earlier passes (e.g.
    // buffer allocation planning) may emit AllocBuffer/DeclBuffer statements
    // before the thread bindings; a positional scan would then miss them and
    // wrongly fall back to a single thread.
    tirx::PostOrderVisit(op->body, [&](const ffi::ObjectRef &node) {
      const auto *attr = node.as<AttrStmtNode>();
      if (attr == nullptr || attr->attr_key != tirx::attr::thread_extent) {
        return;
      }
      const auto *iv = attr->node.as<IterVarNode>();
      const auto *imm = attr->value.as<IntImmNode>();
      if (iv == nullptr || imm == nullptr) {
        return;
      }
      if (iv->thread_tag == "threadIdx.x") {
        thread_x = imm->value;
      } else if (iv->thread_tag == "threadIdx.y") {
        thread_y = imm->value;
      } else if (iv->thread_tag == "threadIdx.z") {
        thread_z = imm->value;
      }
    });

    auto captures = CollectVFCaptures(op);
    int64_t vf_idx = simtvf_helper_counter_++;
    {
      auto it = op->annotations.find("tl.vf_source_index");
      if (it != op->annotations.end()) {
        if (auto *imm = (*it).second.as<IntImmNode>()) {
          vf_idx = imm->value;
        }
      }
    }
    std::string helper_name =
        current_function_name_ + "_simt_vf_" + std::to_string(vf_idx);

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
                                        : std::string(captures[i]->name_hint));
    }
    stream << ");\n";
    return;
  }
  PrintStmt(op->body);
}

void CodeGenTileLangAscend::VisitExpr_(const BroadcastNode *op,
                                       std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  RejectGenericVectorInSimd("Broadcast", op->dtype);

  // Keep casts outside broadcasts so the vector value is converted as a
  // whole. Besides avoiding target-specific Broadcast(Cast(Call)) patterns,
  // this lets vector casts use packed conversion intrinsics such as
  // float2 -> bfloat16x2.
  if (const auto *cast = op->value.as<CastNode>()) {
    PrimExpr source = Broadcast(cast->value, op->lanes);
    PrintExpr(Cast(op->dtype, source), os);
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
        call->dtype.is_float() && call->dtype.bits() == 32) {
      ICHECK_EQ(call->args.size(), 1);
      const auto *dist = call->args[0].as<StringImmNode>();
      ICHECK(dist != nullptr) << "Ascend RNG distribution must be constant";
      if (dist->value == "uniform" || dist->value == "normal") {
        EmitPhiloxVectorFloat(os, dist->value, lanes);
        return;
      }
    }
  }

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8) {
    const int64_t *p = as_const_int(op->value);
    if (p) {
      if (lanes == 4) {
        // make_int8x4
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
        return;
      } else if (lanes == 32 && IsInsideSimtVF()) {
        // make_int8x32
        const int64_t *p = as_const_int(op->value);
        ICHECK(p);
        int64_t v = *p & 0xFF;
        v = (v << 24) | (v << 16) | (v << 8) | v;
        if (op->dtype.is_uint()) {
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

  if (op->dtype.is_float() && op->dtype.bits() == 32 &&
      op->dtype.lanes() == 8) {
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

  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 4) {
    bool fail = false;
    const int64_t *p = as_const_int(op->value);
    ICHECK(p) << "BroadcastNode " << op << " value: " << op->value
              << " is not a constant";
    int64_t v = *p & 0xF;

    if (lanes == 4) {
      v = (v << 12) | (v << 8) | (v << 4) | v;
      if (op->dtype.is_uint()) {
        os << "(uint16_t)" << v;
      } else {
        os << "(int16_t)" << v;
      }
    } else {
      v = (v << 28) | (v << 24) | (v << 20) | (v << 16) | (v << 12) | (v << 8) |
          (v << 4) | v;
      if (lanes == 8) {
        if (op->dtype.is_uint()) {
          os << "(uint)" << v;
        } else {
          os << "(int)" << v;
        }
      } else if (lanes == 16 || lanes == 32) {
        os << "make_";
        PrintType(op->dtype, os);
        os << '(';
        for (int i = 0; i < lanes / 8; ++i) {
          if (i != 0)
            os << ", ";
          if (op->dtype.is_uint()) {
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
      (op->dtype.is_float16() || op->dtype.is_bfloat16()) && lanes <= 8 &&
      lanes % 2 == 0;
  const bool is_packed_predicate =
      op->dtype.is_bool() && lanes > 4 && lanes <= 8 && lanes % 2 == 0;
  if (is_packed_16 || is_packed_predicate) {
    // Small FP16/BF16 vectors and wide predicates use uint carriers in
    // generated Ascend C. Materialize the carrier lane by lane instead of
    // relying on constructors whose argument count follows the carrier rather
    // than the logical lane count. Larger vectors are CCE SIMD registers and
    // must be produced by SIMD intrinsics such as vdup.
    TVM_FFI_ICHECK_LE(lanes, 8)
        << "Ascend packed Broadcast supports at most 8 lanes outside SIMD "
           "intrinsics, got "
        << op->dtype;
    std::string result = name_supply_->FreshName("_");
    PrintIndent();
    PrintType(op->dtype, stream);
    stream << ' ' << result << ";\n";
    SSAOperationScope ssa_scope(this);
    std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
    for (int i = 0; i < lanes; ++i) {
      PrintVecElemStore(result, op->dtype, i, value);
    }
    os << result;
    return;
  }

  std::string v = PrintExpr(op->value);
  os << "make_";
  PrintType(op->dtype, os);
  os << '(';
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenTileLangAscend::VisitExpr_(const RampNode *op,
                                       std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  ValidateGenericVectorType("Ramp", op->dtype);
  os << "make_";
  PrintType(op->dtype, os);
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
                                                 DataType result_dtype) const {
  const std::string op_name = Downcast<Op>(call->op)->name;
  const std::string intrinsic_name =
      op_name.substr(std::string(kSimdOpPrefix).size());

  // Both zeroing and merging calls use the same precision wrapper. Merging
  // calls are void, so their actual result dtype is supplied by the caller.
  if (intrinsic_name == "vdiv") {
    const bool is_f32 = result_dtype.is_float() && result_dtype.bits() == 32;
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
  os << "simd_inst::" << ResolveSimdIntrinsicName_(call, call->dtype) << "(";
  for (size_t i = 0; i < call->args.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    if (i < num_expr_args) {
      PrintExpr(call->args[i], os);
    } else {
      // Trailing arguments name control enums and must stay unquoted.
      os << Downcast<StringImm>(call->args[i])->value;
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
  ICHECK(op->args[0].dtype().is_handle())
      << "Legalized MODE_MERGING destination must be a pointer";
  std::string resolved_name =
      ResolveSimdIntrinsicName_(GetRef<Call>(op), op->args[1].dtype());
  DataType dtype = op->args[1].dtype();
  if (intrinsic_name == "vdup") {
    // Scalar broadcast overloads are selected by the destination vector type.
    const auto *destination = op->args[0].as<CallNode>();
    ICHECK(destination &&
           destination->op.same_as(tirx::builtin::address_of()) &&
           destination->args.size() == 1);
    dtype = destination->args[0].dtype();
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
    DataType element_type = op->args[0].dtype();
    if (element_type.is_handle()) {
      if (const auto *call = op->args[0].as<CallNode>()) {
        if (!call->args.empty()) {
          element_type = call->args[0].dtype();
        }
      }
    }
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
  const int bit_width = op->args[0].dtype().bits();
  ICHECK(bit_width == 8 || bit_width == 16 || bit_width == 32)
      << "Data type: " << op->args[0].dtype() << " is not supported in ascend";

  const std::string value = PrintExpr(op->args[0]);
  const std::string target_value = name_supply_->FreshName("__asc_pad_val");
  PrintIndent();
  PrintType(op->args[0].dtype(), stream);
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
  ICHECK(op->args.size() == 9 || op->args.size() == 16)
      << intrinsic << " expects 9 or 16 arguments, got " << op->args.size();

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

  if (op->args.size() == 16) {
    // MX destinations use 16-byte address units, while the scale source must
    // be viewed in its physical E8M0 representation.
    const char *mx_function_name =
        is_l0a ? "asc_copy_l12l0a_mx" : "asc_copy_l12l0b_mx";
    EmitCApiCall_(mx_function_name,
                  {{op->args[0], "(uint64_t)(uintptr_t)(", ") / 16"},
                   {op->args[9], "(__cbuf__ fp8_e8m0_t*)(", ")"},
                   {op->args[10]},
                   {op->args[11]},
                   {op->args[12]},
                   {op->args[13]},
                   {op->args[14]},
                   {op->args[15]}});
  }
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

void CodeGenTileLangAscend::VisitExpr_(const CallNode *op, std::ostream &os) {
  std::string simd_op_name;
  if (IsSimdOp(op, &simd_op_name)) {
    if (!IsInsideSimdVF()) {
      LOG(FATAL) << simd_op_name << " can only be emitted inside SimdVF";
    }
  }
  if (op->op.same_as(builtin::bitwise_not()) && op->dtype.lanes() > 1) {
    RejectGenericVectorInSimd("bitwise Not", op->dtype);
    if (op->dtype.is_vector_bool() && op->dtype.lanes() <= 8) {
      ICHECK_EQ(op->args.size(), 1U);
      EmitSimtVectorPredicateNot(op->args[0], op->dtype, os);
      return;
    }
  }
  if (EmitSimdMergingCall(op, os)) {
    return;
  }
  if (EmitAscendMemoryCall_(op)) {
    return;
  }
  if (op->op.same_as(builtin::reinterpret())) {
    SSAOperationScope ssa_scope(this);
    CodeGenC::VisitExpr_(op, os);
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
    std::string pipe_str = Downcast<StringImm>(op->args[0])->value;
    int64_t mode = Downcast<IntImm>(op->args[2])->value;
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
    std::string pipe_str = Downcast<StringImm>(op->args[0])->value;
    int64_t mode = Downcast<IntImm>(op->args[2])->value;
    this->PrintIndent();
    this->stream << "asc_unlock(" << pipe_str << ", ";
    this->PrintExpr(op->args[1], this->stream);

    // ASC_LOCK_BLOCK is the C API default; emit it explicitly for readability.
    this->stream << (mode ? ", ASC_LOCK_NON_BLOCK" : ", ASC_LOCK_BLOCK");
    this->stream << ");\n";
  } else if (op->op.same_as(tl::ascend_set_hf32_mode())) {
    ICHECK_EQ(op->args.size(), 1)
        << "tl.ascend_set_hf32_mode expects exactly 1 argument (mode_int)";
    int mode = Downcast<IntImm>(op->args[0])->value;
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
  } else if (op->op.same_as(tl::ascend_set_atomic())) {
    ICHECK_EQ(op->args.size(), 2)
        << "tl.ascend_set_atomic expects 2 arguments (op_str, typed_zero)";
    std::string atomic_op = Downcast<StringImm>(op->args[0])->value;
    const char *fn = atomic_op == "max"   ? "asc_set_atomic_max"
                     : atomic_op == "min" ? "asc_set_atomic_min"
                                          : "asc_set_atomic_add";
    ICHECK(atomic_op == "add" || atomic_op == "max" || atomic_op == "min")
        << "tl.ascend_set_atomic op must be add/max/min, got " << atomic_op;

    DataType atomic_dtype = op->args[1].dtype();
    bool supported_dtype =
        atomic_dtype.is_bfloat16() ||
        (atomic_dtype.is_float() &&
         (atomic_dtype.bits() == 16 || atomic_dtype.bits() == 32)) ||
        (atomic_dtype.is_int() &&
         (atomic_dtype.bits() == 8 || atomic_dtype.bits() == 16 ||
          atomic_dtype.bits() == 32));
    ICHECK(supported_dtype)
        << "Ascend atomic does not support data type: " << atomic_dtype;

    this->PrintIndent();
    this->stream << fn;
    if (atomic_dtype.is_bfloat16()) {
      this->stream << "_bfloat";
    } else if (atomic_dtype.is_float() && atomic_dtype.bits() == 16) {
      this->stream << "_float16";
    } else if (atomic_dtype.is_float() && atomic_dtype.bits() == 32) {
      this->stream << "_float";
    } else if (atomic_dtype.is_int() && atomic_dtype.bits() == 8) {
      this->stream << "_int8";
    } else if (atomic_dtype.is_int() && atomic_dtype.bits() == 16) {
      this->stream << "_int16";
    } else if (atomic_dtype.is_int() && atomic_dtype.bits() == 32) {
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
    DataType dtype = op->args[0].dtype();
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
  } else if (op->op.same_as(tirx::builtin::address_of())) {
    const auto *load = op->args[0].as<BufferLoadNode>();
    ICHECK(op->args.size() == 1 && load);
    const auto *buf_var = load->buffer->data.get();
    std::string scope;
    if (alloc_storage_scope_.count(buf_var)) {
      scope = alloc_storage_scope_.at(buf_var);
    }
    bool needs_ptr_arith = (scope.find("l0") != std::string::npos ||
                            scope.find("l1") != std::string::npos);
    // SimdVF: `local` vector-register array (alloc_local). access_ptr lowering
    // expands the 64-lane element access into a Ramp index; recover the element
    // index (ramp base / lanes) and take the address of that register slot.
    bool is_local_vreg = IsInsideSimdVF() && scope == "local" &&
                         load->indices.size() == 1 &&
                         load->indices[0].dtype().lanes() > 1;
    // Packed fp4 (float4_e2m1fnx2) held in GM/UBuf: emitted as the 1-byte
    // `float4_e2m1x2_t`. Handle only after the local-vreg / L0-L1 cases, since
    // a fp4 SIMD register slot must still use the register-index addressing
    // below.
    bool is_fp4_ptr = load->dtype.is_float4_e2m1fn() &&
                      load->indices.size() == 1 && !is_local_vreg &&
                      !needs_ptr_arith;
    if (is_local_vreg) {
      PrimExpr index = load->indices[0];
      ValidateSimdVectorType(load->dtype,
                             "SimdVF local register array element");
      const auto *ramp = index.as<RampNode>();
      PrimExpr base = ramp ? ramp->base : index;
      int lanes = index.dtype().lanes();
      PrimExpr elem_idx =
          analyzer_.Simplify(floordiv(base, make_const(base.dtype(), lanes)));
      os << "(&(" << GetVarID(buf_var) << "[";
      PrintExpr(elem_idx, os);
      os << "]))";
    } else if (needs_ptr_arith) {
      os << "((";
      PrintStorageScope(scope, os);
      PrintType(load->dtype, os);
      os << "*)" << GetVarID(buf_var);
      if (load->indices.size() == 1) {
        PrimExpr idx = load->indices[0];
        // LowerIntrin represents an fp4x2 access as a 2-lane Ramp and scales
        // its packed-byte offset by the lane count.  L0/L1 pointers need the
        // same scalar-byte recovery as GM/UB pointers below; otherwise this
        // prints invalid pointer arithmetic such as `ptr + int2(...)`.
        if (load->dtype.is_float4_e2m1fn() && idx.dtype().lanes() > 1) {
          const auto *ramp = idx.as<RampNode>();
          ICHECK(ramp) << "Expected a Ramp index for packed fp4 L0/L1 access, "
                       << "got " << idx;
          idx = analyzer_.Simplify(floordiv(
              ramp->base, make_const(ramp->base.dtype(), idx.dtype().lanes())));
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
      std::string fp4_scope =
          !scope.empty() ? scope
                         : std::string(GetPtrStorageScope(load->buffer->data));
      PrimExpr index = load->indices[0];
      const auto *ramp = index.as<RampNode>();
      PrimExpr base = ramp ? ramp->base : index;
      int lanes = index.dtype().lanes();
      PrimExpr byte_idx =
          lanes > 1 ? analyzer_.Simplify(
                          floordiv(base, make_const(base.dtype(), lanes)))
                    : base;
      os << "(&(((";
      PrintStorageScope(fp4_scope, os);
      os << "float4_e2m1x2_t*)" << GetVarID(buf_var) << ")[";
      PrintExpr(byte_idx, os);
      os << "]))";
    } else {
      CodeGenC::VisitExpr_(op, os);
    }
  }
  // --- Scalar GM dcache bypass intrinsics ---
  else if (op->op.same_as(tl::ascend_read_gm_bypass_dcache())) {
    // ascend_read_gm_bypass_dcache(address_of(BufferLoad))
    ICHECK_EQ(op->args.size(), 1);
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && addr->op.same_as(tirx::builtin::address_of()));
    const auto *load = addr->args[0].as<BufferLoadNode>();
    ICHECK(load && load->indices.size() == 1);
    os << "tl::read_gm_bypass_dcache("
       << GetGmBypassPtr(load->dtype, load->buffer.get(), load->indices[0])
       << ")";
  } else if (op->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
    // ascend_write_gm_bypass_dcache(address_of(BufferLoad), value)
    ICHECK_EQ(op->args.size(), 2);
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && addr->op.same_as(tirx::builtin::address_of()));
    const auto *load = addr->args[0].as<BufferLoadNode>();
    ICHECK(load && load->indices.size() == 1);
    // Evaluate ptr and value to strings first. The string PrintExpr overload
    // flushes any SSA statements (e.g. from if_then_else) to this->stream
    // before returning the expression text, so they land ahead of the call
    // line instead of being injected into the argument list.
    std::string ptr =
        GetGmBypassPtr(load->dtype, load->buffer.get(), load->indices[0]);
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
    int width = (int)Downcast<IntImm>(op->args[1])->value;
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
    int width = (int)Downcast<IntImm>(op->args[2])->value;
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
    DataType elem = op->dtype.element_of();
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
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vci<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os);                                   // index
    os << ", " << Downcast<StringImm>(op->args[1])->value << ")"; // order
  }
  // --- SimdVF compare: (src0, src1, mask, op) -> vcmp_<op>(...) ---
  else if (op->op.same_as(tl::simd_vcmp())) {
    ICHECK_EQ(op->args.size(), 4);
    os << "simd_inst::vcmp_" << Downcast<StringImm>(op->args[3])->value << "(";
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
    os << "simd_inst::vcmps_" << Downcast<StringImm>(op->args[3])->value << "(";
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
    int idx = (int)Downcast<IntImm>(op->args[1])->value;
    os << (idx == 0 ? ".v0" : ".v1");
  }
  // --- SimdVF register pack: (src, part) ---
  else if (op->op.same_as(tl::simd_vpack())) {
    ICHECK_EQ(op->args.size(), 2);
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vpack<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // src
    os << ", " << Downcast<StringImm>(op->args[1])->value << ")";
  }
  // --- SimdVF vector broadcast vdupv: (src, mask, pos, mode) ---
  else if (op->op.same_as(tl::simd_vdupv())) {
    ICHECK_EQ(op->args.size(), 4);
    PrintSimdCall_(GetRef<Call>(op), 2, os);
  }
  // --- SimdVF gather 32B blocks: (base, index [, mask]) ---
  else if (op->op.same_as(tl::simd_vgatherb())) {
    ICHECK(op->args.size() == 2 || op->args.size() == 3);
    DataType elem = op->dtype.element_of();
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
    std::string dist = Downcast<StringImm>(op->args[1])->value;
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
    std::string dist = Downcast<StringImm>(op->args[2])->value;
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
  // --- SimdVF load: (addr, dist) ---
  else if (op->op.same_as(tl::simd_vld())) {
    ICHECK_EQ(op->args.size(), 2);
    DataType elem = op->dtype.element_of();
    std::string dist = Downcast<StringImm>(op->args[1])->value;
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
    // Zero scalar offset keeps parity with the previous vlds(addr, 0, dist).
    os << "simd_inst::" << it->second << "<" << CCEUBufType(elem) << ">(("
       << "__ubuf__" << " " << CCEUBufType(elem) << "*)";
    PrintExpr(op->args[0], os); // addr
    os << ", 0)";
  }
  // --- SimdVF dual-dest load: (addr, dist [, offset]) -> vec_pair ---
  else if (op->op.same_as(tl::simd_vld2())) {
    ICHECK(op->args.size() >= 2 && op->args.size() <= 3);
    DataType elem = op->dtype.element_of();
    std::string dist = Downcast<StringImm>(op->args[1])->value;
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
    DataType dtype = op->args[1].dtype().element_of();
    std::string dist = Downcast<StringImm>(op->args[3])->value;
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

  // --- SimdVF scatter-store blocks: (src, base, stride, mask [, post]) ---
  else if (op->op.same_as(tl::simd_vsstb())) {
    ICHECK(op->args.size() == 4 || op->args.size() == 5);
    bool post_update = op->args.size() == 5;
    ICHECK_EQ(op->dtype.is_handle(), post_update);
    DataType dtype = op->args[0].dtype().element_of();
    // Capture operands first so SSA temporaries (e.g. from reinterpret) are
    // emitted on their own lines before the call statement.
    std::string src = this->PrintExpr(op->args[0]);
    std::string base = this->PrintExpr(op->args[1]);
    std::string stride = this->PrintExpr(op->args[2]);
    std::string mask = this->PrintExpr(op->args[3]);
    if (post_update) {
      // Keep POST_UPDATE as an expression so the frontend can write the
      // advanced pointer back into its mutable handle carrier.
      std::string post = Downcast<StringImm>(op->args[4])->value;
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
    DataType dtype = op->args[0].dtype().element_of();
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
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vdup<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // scalar
    os << ", ";
    PrintExpr(op->args[1], os);                                   // mask
    os << ", " << Downcast<StringImm>(op->args[2])->value << ")"; // mode
  }
  // --- SimdVF cast: (src, mask, ...extra) ---
  else if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK(op->args.size() >= 4 && op->args.size() <= 6);
    DataType elem = op->dtype.element_of();
    os << "simd_inst::vcvt<";
    PrintType(elem, os);
    os << ">(";
    PrintExpr(op->args[0], os); // src
    os << ", ";
    PrintExpr(op->args[1], os); // mask
    for (int i = 2; i < static_cast<int>(op->args.size()); ++i) {
      os << ", " << Downcast<StringImm>(op->args[i])->value;
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
    std::string mem_type_str = Downcast<StringImm>(op->args[0])->value;
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
    ICHECK_NE(op->dtype.bits(), 64)
        << "float64 RNG (tl.rng_rand_float bit=64) is not supported on Ascend";
    const std::string dist = op->args[0].as<StringImmNode>()->value;
    ICHECK(dist == "uniform" || dist == "normal")
        << "Unsupported RNG distribution on Ascend: " << dist;
    os << "tl::philox_rand_" << dist << "(&" << ascend_rng_state_var_ << ")";
  } else {
    std::string op_name;
    // Evaluate statements print their value through this CallNode visitor, so
    // statement-form VMI calls such as vstore are rejected here as well.
    if (IsVmiOp(op, &op_name)) {
      LOG(FATAL) << "Ascend CCE codegen does not support " << op_name
                 << "; use target='pto' for T.vmi.*";
    }
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangAscend::VisitExpr_(const NotNode *op, std::ostream &os) {
  if (op->dtype.lanes() > 1) {
    RejectGenericVectorInSimd("Not", op->dtype);
    if (op->dtype.is_vector_bool() && op->dtype.lanes() <= 8) {
      EmitSimtVectorPredicateNot(op->a, op->dtype, os);
      return;
    }
  }
  CodeGenC::VisitExpr_(op, os);
}

bool CodeGenTileLangAscend::IsGlobalGmBuffer(const BufferNode *buffer) const {
  const VarNode *buf_var = buffer->data.get();
  auto it = alloc_storage_scope_.find(buf_var);
  std::string scope = (it != alloc_storage_scope_.end())
                          ? it->second
                          : std::string(GetPtrStorageScope(buffer->data));
  return scope == "global";
}

std::string CodeGenTileLangAscend::GetGmBypassPtr(DataType dtype,
                                                  const BufferNode *buffer,
                                                  PrimExpr index) {
  const VarNode *buf_var = buffer->data.get();
  std::ostringstream os;
  os << "((";
  PrintStorageScope("global", os);
  PrintType(dtype, os);
  os << "*)" << GetVarID(buf_var) << " + ";
  PrintExpr(index, os);
  os << ")";
  return os.str();
}

void CodeGenTileLangAscend::VisitExpr_(const BufferLoadNode *op,
                                       std::ostream &os) {
  DataType value_dtype = op->dtype;
  const VarNode *buf_var = op->buffer->data.get();

  // For Ascend: scalar FP8 loads from shared/UBUF memory need conversion
  // through raw bytes, because the scalar FP8 type (tl::float_e4m3_t) is a
  // struct whose operator= does not support __ubuf__ to local assignment.
  if (value_dtype.is_scalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      const std::string &scope = it->second;
      if (scope == "shared" || scope == "shared.dyn") {
        // __ubuf__ does not support custom struct pointer
        ICHECK_EQ(op->indices.size(), 1);
        PrimExpr index = op->indices[0];
        std::string buf_name = GetVarID(buf_var);

        // tl::float_e4m3_t::from_bits(*(__ubuf__ uint8_t*)(buf + off))
        os << GetAscendFP8ScalarValueType(value_dtype) << "::from_bits("
           << "*(";
        PrintStorageScope(scope, os);
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
  DataType element_dtype = op->buffer->dtype;
  if (value_dtype.lanes() > 1 && value_dtype.lanes() != element_dtype.lanes() &&
      op->indices.size() == 1) {
    EmitScalarizedLoad(op, os);
    return;
  }

  CodeGenC::VisitExpr_(op, os);
}

bool CodeGenTileLangAscend::VectorAccessIsAligned(const PrimExpr &index,
                                                  int lanes) {
  // Only a unit-stride ramp can become a reinterpret-cast vector access.
  arith::PVar<PrimExpr> base;
  if (!arith::ramp(base, 1, lanes).Match(index)) {
    return false;
  }
  // Prove the ramp base is divisible by the vector width.
  PrimExpr ramp_base = base.Eval();
  PrimExpr lanes_const = make_const(ramp_base.dtype(), lanes);
  return analyzer_.CanProveEqual(floormod(ramp_base, lanes_const),
                                 make_zero(ramp_base.dtype()));
}

void CodeGenTileLangAscend::EmitScalarizedLoad(const BufferLoadNode *op,
                                               std::ostream &os) {
  DataType value_dtype = op->dtype;
  ValidateGenericVectorType("BufferLoad", value_dtype);
  int lanes = value_dtype.lanes();
  PrimExpr index = op->indices[0];

  // A vectorized GM access on the scalar core (outside any VF body) has no
  // valid lowering: the only scalar-pipe GM path is the per-element
  // read_gm_bypass_dcache emitted in VisitExpr_. Vector GM traffic must go
  // through an MTE copy (T.copy) or a SimtVF/SimdVF region.
  if (IsOutsideVF() && IsGlobalGmBuffer(op->buffer.get())) {
    LOG(FATAL) << "Ascend: cannot vectorize GM access of " << op->buffer->name
               << " (lanes=" << lanes
               << ") on the scalar core. Scalar-core GM reads must be a single "
                  "scalar element (lowered to read_gm_bypass_dcache); use "
                  "T.copy or a T.SimtVF/T.SimdVF region for vector GM traffic.";
  }

  // Provably aligned: a single reinterpret-cast vector load (fast path).
  if (VectorAccessIsAligned(index, lanes)) {
    arith::PVar<PrimExpr> base;
    ICHECK(arith::ramp(base, 1, lanes).Match(index));
    std::string ref = GetVecLoad(value_dtype, op->buffer.get(), base.Eval());
    HandleVolatileLoads(ref, op, os);
    return;
  }

  // Otherwise scalarize: load each lane from its scalar element offset and pack
  // with a brace-initializer (which bisheng accepts, unlike `T(a, b)`).
  DataType elem_dtype = value_dtype.element_of();
  const RampNode *ramp = index.as<RampNode>();
  std::ostringstream init;
  PrintType(value_dtype, init);
  init << "{";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0) {
      init << ", ";
    }
    PrimExpr lane_index =
        ramp ? (ramp->base + ramp->stride * make_const(ramp->base.dtype(), i))
             : index;
    std::string lane_ref = GetBufferRef(elem_dtype, op->buffer.get(),
                                        analyzer_.Simplify(lane_index));
    init << lane_ref;
  }
  init << "}";
  os << init.str();
}

void CodeGenTileLangAscend::VisitStmt_(const BufferStoreNode *op) {
  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  const VarNode *buf_var = op->buffer->data.get();
  PrimExpr index_expr = op->indices[0];

  if (value_dtype.lanes() > 1 && op->indices.size() == 1) {
    const auto *broadcast_index = index_expr.as<BroadcastNode>();
    auto scope_it = alloc_storage_scope_.find(buf_var);
    std::string buffer_scope = op->buffer.scope();
    bool is_local_scope =
        buffer_scope == "local" || buffer_scope == "local.var" ||
        (scope_it != alloc_storage_scope_.end() &&
         (scope_it->second == "local" || scope_it->second == "local.var"));
    if (broadcast_index && is_local_scope &&
        value_dtype.lanes() ==
            Downcast<IntImm>(broadcast_index->lanes)->value) {
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
  if (value_dtype.is_scalar() && tl::IsAscendVectorizableFP8(value_dtype)) {
    auto it = alloc_storage_scope_.find(buf_var);
    if (it != alloc_storage_scope_.end()) {
      const std::string &scope = it->second;
      if (scope == "shared" || scope == "shared.dyn") {
        // __ubuf__ does not support custom struct pointer
        ICHECK_EQ(op->indices.size(), 1);
        std::string buf_name = GetVarID(buf_var);
        std::string value = PrintExpr(op->value);

        // ((__ubuf__ uint8_t*)buf)[off] = value.data;
        PrintIndent();
        stream << "((";
        PrintStorageScope(scope, stream);
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

  CodeGenC::VisitStmt_(op);
}

bool CodeGenTileLangAscend::EmitScalarizedStore(const BufferStoreNode *op) {
  DataType value_dtype = op->value.dtype();
  ValidateGenericVectorType("BufferStore", value_dtype);
  int lanes = value_dtype.lanes();
  PrimExpr index = op->indices[0];

  // A vectorized GM access on the scalar core (outside any VF body) has no
  // valid lowering: the only scalar-pipe GM path is the per-element
  // write_gm_bypass_dcache emitted in VisitStmt_. Vector GM traffic must go
  // through an MTE copy (T.copy) or a SimtVF/SimdVF region.
  if (IsOutsideVF() && IsGlobalGmBuffer(op->buffer.get())) {
    LOG(FATAL)
        << "Ascend: cannot vectorize GM access of " << op->buffer->name
        << " (lanes=" << lanes
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
  DataType elem_dtype = value_dtype.element_of();
  SSAOperationScope ssa_scope(this);
  std::string value = SSAGetID(PrintExpr(op->value), value_dtype);
  for (int i = 0; i < lanes; ++i) {
    PrimExpr lane_index =
        ramp->base + ramp->stride * make_const(ramp->base.dtype(), i);
    std::string lane_ref = GetBufferRef(elem_dtype, op->buffer.get(),
                                        analyzer_.Simplify(lane_index));
    std::ostringstream lane_val;
    PrintVecElemLoad(value, value_dtype, i, lane_val);
    PrintIndent();
    stream << lane_ref << " = " << lane_val.str() << ";\n";
  }
  return true;
}

void CodeGenTileLangAscend::VisitExpr_(const SelectNode *op,
                                       std::ostream &os) { // NOLINT(*)
  if (!op->condition.dtype().is_fixed_length_vector()) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }

  ValidateGenericVectorType("Select", op->condition.dtype());
  ValidateGenericVectorType("Select", op->dtype);

  TVM_FFI_ICHECK(op->false_value->dtype == op->dtype &&
                 op->true_value->dtype == op->dtype &&
                 op->dtype.lanes() == op->condition.dtype().lanes());

  std::string result = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(op->dtype, stream);
  stream << ' ' << result << ";\n";
  SSAOperationScope ssa_scope(this);
  {
    std::string condition =
        SSAGetID(PrintExpr(op->condition), op->condition.dtype());
    std::string true_value = SSAGetID(PrintExpr(op->true_value), op->dtype);
    std::string false_value = SSAGetID(PrintExpr(op->false_value), op->dtype);

    int lanes = op->dtype.lanes();
    for (int i = 0; i < lanes; ++i) {
      std::ostringstream value;
      value << "(bool(";
      PrintVecElemLoad(condition, op->condition.dtype(), i, value);
      value << ")?";
      PrintVecElemLoad(true_value, op->dtype, i, value);
      value << ':';
      PrintVecElemLoad(false_value, op->dtype, i, value);
      value << ')';
      PrintVecElemStore(result, op->dtype, i, value.str());
    }
  }
  os << result;
}

void CodeGenTileLangAscend::VisitExpr_(const ShuffleNode *op,
                                       std::ostream &os) { // NOLINT(*)
  for (const PrimExpr &vector : op->vectors) {
    if (vector.dtype().lanes() > 1) {
      ValidateGenericVectorType("Shuffle", vector.dtype());
    }
  }
  if (op->dtype.lanes() > 1) {
    ValidateGenericVectorType("Shuffle", op->dtype);
  }
  SSAOperationScope ssa_scope(this);

  // Materialize the concatenated input lanes once before applying the shuffle
  // indices. Bisheng does not accept the generic C codegen's vector
  // constructors such as float2(a, b), and packed FP16/BF16 vectors require
  // element-wise access through their uint carrier.
  std::vector<std::string> input_lanes;
  for (const PrimExpr &vector : op->vectors) {
    std::string vector_value = SSAGetID(PrintExpr(vector), vector.dtype());
    if (vector.dtype().is_scalar()) {
      input_lanes.push_back(std::move(vector_value));
      continue;
    }
    for (int i = 0; i < vector.dtype().lanes(); ++i) {
      std::ostringstream element;
      PrintVecElemLoad(vector_value, vector.dtype(), i, element);
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
    TVM_FFI_ICHECK_GE(constant_index->value, 0)
        << "ShuffleNode index must be non-negative, got "
        << constant_index->value;
    TVM_FFI_ICHECK_LT(static_cast<size_t>(constant_index->value),
                      input_lanes.size())
        << "ShuffleNode index " << constant_index->value
        << " is outside the concatenated input with " << input_lanes.size()
        << " lanes";
    output_lanes.push_back(input_lanes[constant_index->value]);
  }

  if (op->dtype.is_scalar()) {
    TVM_FFI_ICHECK_EQ(output_lanes.size(), 1);
    os << output_lanes[0];
    return;
  }

  TVM_FFI_ICHECK_EQ(static_cast<int>(output_lanes.size()), op->dtype.lanes());
  std::string result = name_supply_->FreshName("_");
  PrintIndent();
  PrintType(op->dtype, stream);
  stream << ' ' << result << ";\n";
  for (int i = 0; i < op->dtype.lanes(); ++i) {
    PrintVecElemStore(result, op->dtype, i, output_lanes[i]);
  }
  os << result;
}

void CodeGenTileLangAscend::VisitStmt_(const ForNode *op) {
  if (op->kind == tirx::ForKind::kUnrolled) {
    PrintIndent();
    auto it = unroll_factor_.find(op->loop_var);
    if (it != unroll_factor_.end()) {
      stream << "#pragma unroll " << PrintExpr(it->second) << "\n";
    } else {
      stream << "#pragma unroll\n";
    }
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const WhileNode *op) {
  if (is_one(op->condition)) {
    PrintIndent();
    stream << "while (1) {\n";
    int while_scope = BeginScope();
    PrintStmt(op->body);
    this->EndScope(while_scope);
    PrintIndent();
    stream << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const AttrStmtNode *op) {
  // Push tl.assume facts (e.g. `stride % N == 0`) into analyzer
  if (op->attr_key == tirx::attr::tilelang_assume) {
    With<arith::ConstraintContext> cctx(&analyzer_,
                                        Downcast<PrimExpr>(op->node));
    this->VisitStmt(op->body);
    return;
  }
  // Ascend text-only backend currently does not lower thread binding attrs into
  // executable launch semantics. Keep generating the enclosed body instead of
  // failing in the generic C backend.
  if (op->attr_key == tirx::attr::thread_extent ||
      op->attr_key == s_tir::attr::virtual_thread) {
    if (const auto *iv = op->node.as<IterVarNode>()) {
      // Materialize a placeholder symbol so later uses (e.g. simtvf_tx) are
      // printable in expression codegen.
      if ((iv->thread_tag == "threadIdx.x" || iv->thread_tag == "threadIdx.y" ||
           iv->thread_tag == "threadIdx.z") &&
          IsOutsideVF()) {
        this->VisitStmt(op->body);
        return;
      }
      if (std::string::npos == iv->thread_tag.find("threadIdx.") &&
          std::string::npos == iv->thread_tag.find("blockIdx.")) {
        PrintIndent();
        PrintType(iv->var.dtype(), stream);
        stream << " " << AllocVarID(iv->var.get()) << " = ";
        if (iv->thread_tag == "cthread") {
          stream << "asc_get_sub_block_id()";
        } else {
          stream << "0";
        }
        stream << ";\n";
      } else if (iv->thread_tag == "blockIdx.x") {
        var_idmap_[iv->var.get()] = "block_idx";
      } else {
        var_idmap_[iv->var.get()] = iv->thread_tag;
      }
    }
    this->VisitStmt(op->body);
    return;
  }
  // SimdVF scope marker: transparent in codegen, just visit body
  if (op->attr_key == "tl.simdvf_scope") {
    this->VisitStmt(op->body);
    return;
  }
  if (op->attr_key == "pragma_unroll_factor") {
    const auto *factor = op->value.as<IntImmNode>();
    ICHECK(factor) << "pragma_unroll_factor must be a constant integer";
    const auto *loop_var = op->node.as<VarNode>();
    ICHECK(loop_var) << "pragma_unroll_factor must annotate a loop variable";
    unroll_factor_[ffi::GetRef<Var>(loop_var)] = Downcast<IntImm>(op->value);
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const AllocBufferNode *op) {
  std::string scope = GetPtrStorageScope(op->buffer->data);

  if (scope == "shared") {
    LOG(FATAL) << "Static shared memory (scope='shared') is not supported on "
                  "Ascend. "
               << "Use dynamic shared memory (scope='shared.dyn') with "
                  "T.alloc_shared.";
  }

  // SimdVF: emit vector_<T> for fragment buffers
  if (IsInsideSimdVF() && scope == "local.fragment") {
    std::string vid = AllocVarID(op->buffer->data.get());
    this->PrintIndent();
    stream << "vector_";
    PrintCCEVectorSuffix(op->buffer->dtype, stream);
    stream << " " << vid << ";\n";
    alloc_storage_scope_[op->buffer->data.get()] = scope;
    RegisterHandleType(op->buffer->data.get(), op->buffer->dtype);
    return;
  }

  std::string qualifier = GetAscendScopeQualifier(scope);
  if (!qualifier.empty()) {
    std::string vid = AllocVarID(op->buffer->data.get());
    // Generate: __qualifier__ <type> *buf = (__qualifier__ <type> *)0;
    DataType dtype = op->buffer->dtype;
    this->PrintIndent();
    stream << qualifier << " ";
    PrintType(dtype, stream);
    stream << " *" << vid << " = (" << qualifier << " ";
    PrintType(dtype, stream);
    stream << " *)0;\n";
    alloc_storage_scope_[op->buffer->data.get()] = scope;
    RegisterHandleType(op->buffer->data.get(), dtype);
    return;
  }

  // SimdVF: emit vector_<T> for local.var vector register temps.
  // CCE vector register variables should be declared uninitialized
  // since they are always written before read via SIMD operations.
  // Without this, zero-init via Broadcast expands to make_vector_f32(...)
  // with one argument per lane, which is both verbose and unnecessary.
  if (IsInsideSimdVF() && scope == "local.var" &&
      op->buffer->dtype.lanes() > 1) {
    ValidateSimdVectorType(op->buffer->dtype, "SimdVF local.var allocation");
    std::string vid = AllocVarID(op->buffer->data.get());
    this->PrintIndent();
    PrintType(op->buffer->dtype, stream);
    stream << " " << vid << ";\n";
    alloc_storage_scope_[op->buffer->data.get()] = scope;
    RegisterHandleType(op->buffer->data.get(), op->buffer->dtype);
    return;
  }

  if (scope == "local.var") {
    auto alloc = AllocBuffer(ffi::GetRef<AllocBuffer>(op));
    ICHECK_EQ(alloc.ConstantAllocationSize().value_or(0), 1)
        << "local.var should be scalar";
    DataType dtype = op->buffer->dtype;
    std::string vid = AllocVarID(op->buffer->data.get());
    ffi::Optional<PrimExpr> init;
    if (!dtype.is_handle()) {
      init = tirx::make_const(dtype, 0);
    }
    auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
    if (init_it != op->annotations.end()) {
      PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
      if (!user_init.dtype().is_void() && user_init.dtype() != dtype) {
        user_init = tirx::Cast(dtype, user_init);
      }
      init = user_init;
    }
    std::string init_value =
        init.defined() ? PrintExpr(init.value()) : "nullptr";
    PrintIndent();
    if (dtype.is_handle()) {
      // Handle local variables are exclusively mutable UB pointers in SimdVF.
      stream << "__ubuf__ void*";
    } else {
      PrintType(dtype, stream);
    }
    stream << " " << vid << " = " << init_value << ";\n";
    alloc_storage_scope_[op->buffer->data.get()] = scope;
    RegisterHandleType(op->buffer->data.get(), dtype);
    return;
  }

  // For other scopes, use parent class implementation
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangAscend::VisitStmt_(const BindNode *op) {
  // Inside SimdVF body, bindings for vector registers and masks need concrete
  // CCE-friendly types instead of generic C handle printing.
  if (IsInsideSimdVF()) {
    if (op->value.dtype().lanes() > 1) {
      ValidateSimdVectorType(op->value.dtype(), "SimdVF binding");
    }
    std::string vid = AllocVarID(op->var.get());
    if (op->var.dtype().is_handle() &&
        op->var->type_annotation.as<PointerTypeNode>()) {
      const auto *ptr = op->var->type_annotation.as<PointerTypeNode>();
      std::string scope = GetPtrStorageScope(op->var);
      if (scope == "local.fragment") {
        const auto *prim = ptr->element_type.as<PrimTypeNode>();
        ICHECK(prim)
            << "SimdVF local.fragment bind expects primitive element type";
        PrintIndent();
        stream << "vector_";
        PrintCCEVectorSuffix(prim->dtype, stream);
        stream << " " << vid << ";\n";
        alloc_storage_scope_[op->var.get()] = scope;
        RegisterHandleType(op->var.get(), prim->dtype);
        return;
      }
    }
    if (const auto *call = op->value.as<CallNode>()) {
      // simd_pset/pge(elem_width_bits, dist) -> vector_bool vid = p*_bXX(dist);
      if (call->op.same_as(tl::simd_pset()) ||
          call->op.same_as(tl::simd_pge())) {
        int elem_bits = (int)Downcast<IntImm>(call->args[0])->value;
        std::string dist =
            (call->args.size() >= 2 && call->args[1].as<StringImmNode>())
                ? Downcast<StringImm>(call->args[1])->value
                : "PAT_ALL";
        PrintIndent();
        stream << "vector_bool " << vid << " = asc_create_mask_b" << elem_bits
               << "(" << dist << ");\n";
        return;
      }
      // pair_get(pair, idx) -> inline as pair.vN, no auto decl
      if (call->op.same_as(tl::simd_pair_get())) {
        int idx = (int)Downcast<IntImm>(call->args[1])->value;
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
  // CodeGenC::VisitStmt_ only inspects the scalar DataType (always "handle")
  // and emits a scope-less `void*`. That loses the `__gm__`/`__ubuf__`
  // qualifier and the element type, so later casts to `__gm__ T*` are rejected
  // by bisheng. Read the PointerType directly and emit the qualified pointer.
  if (op->var.dtype().is_handle()) {
    if (const auto *ptr = op->var->type_annotation.as<PointerTypeNode>()) {
      if (const auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        std::string scope = GetPtrStorageScope(op->var);
        // Scoped pointer type, e.g. "__gm__ float *".
        std::ostringstream type_os;
        PrintStorageScope(scope, type_os);
        PrintType(prim->dtype, type_os);
        type_os << " *";
        std::string ptr_type = type_os.str();

        // A reinterpret from an integer address (the make_tensor case) is
        // emitted by the base codegen as a `void*` bit-cast, which bisheng
        // refuses to convert to a scoped pointer. Emit the bit-cast directly
        // against the scoped pointer type so no `void*` intermediate appears.
        const auto *call = op->value.as<CallNode>();
        if (call && call->op.same_as(builtin::reinterpret()) &&
            !call->args[0]->dtype.is_handle()) {
          SSAOperationScope ssa_scope(this);
          std::string rhs =
              SSAGetID(PrintExpr(call->args[0]), call->args[0]->dtype);
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
        alloc_storage_scope_[op->var.get()] = scope;
        RegisterHandleType(op->var.get(), prim->dtype);
        return;
      }
    }
  }
  CodeGenC::VisitStmt_(op);
}

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangAscend *p) { // NOLINT(*)
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (op->dtype.is_bfloat16() || op->dtype.is_float16() ||
      op->dtype.is_float()) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      temp << "tl::limits::";
      if (op->dtype.is_float()) {
        temp << (op->value < 0 ? "kFloatNInf" : "kFloatInf");
      } else if (op->dtype.is_float16()) {
        temp << (op->value < 0 ? "kHalfNInf" : "kHalfInf");
      } else {
        temp << (op->value < 0 ? "kBf16NInf" : "kBf16Inf");
      }
    } else if (std::isnan(op->value)) {
      temp << "tl::limits::";
      if (op->dtype.is_float()) {
        temp << "kFloatNaN";
      } else if (op->dtype.is_float16()) {
        temp << "kHalfNaN";
      } else {
        temp << "kBf16NaN";
      }
    } else {
      p->PrintType(op->dtype, temp);
      temp << '(' << std::hexfloat << op->value << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
      temp << ')';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    return;
  }
  // Type code is kFloat8_e5m2 or kE4M4Float
  if (op->dtype.is_float8() || op->dtype.is_float4()) {
    p->PrintType(op->dtype, os);
    os << '(' << std::hexfloat << op->value << 'f';
    os << "/*" << std::scientific << op->value << "*/";
    os << ')';
    return;
  }
  // Type code is kFloat64/kFloat32 (kFloat16 is handled above)
  switch (op->dtype.bits()) {
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangAscend::VisitExpr_(const FloatImmNode *op,
                                       std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangAscend::VisitExpr_(const CastNode *op,
                                       std::ostream &os) { // NOLINT(*)
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  if (from_ty.is_scalar()) {
    if (from_ty.is_float() && from_ty.bits() == 32 &&
        tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(";
      PrintExpr(op->value, os);
      os << ")";
      return;
    }

    if (from_ty.is_float16() && tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (from_ty.is_bfloat16() && tl::IsAscendVectorizableFP8(target_ty)) {
      os << GetAscendFP8ScalarValueType(target_ty) << "(__bfloat162float(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float() &&
        target_ty.bits() == 32) {
      os << "static_cast<float>(";
      PrintExpr(op->value, os);
      os << ")";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float16()) {
      os << "static_cast<half>(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_bfloat16()) {
      os << "__float2bfloat16_rn(static_cast<float>(";
      PrintExpr(op->value, os);
      os << "))";
      return;
    }

    return CodeGenC::VisitExpr_(op, os);
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
  if (from_ty.is_float16() && target_ty.is_float() && target_ty.bits() == 32) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__half22float2", "half2", "float2");
      return;
    }
  }

  // Handle conversion from float32 to float16.
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_float16()) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22half2_rn", "float2", "half2");
      return;
    }
  }

  // Handle conversion from bfloat16 to float32
  if (from_ty.is_bfloat16() && target_ty.is_float() && target_ty.bits() == 32) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__bfloat1622float2", "bfloat16x2_t", "float2", "",
                          /*src_needs_reinterpret=*/true);
      return;
    }
  }

  // Handle conversion from float32 to bfloat16
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_bfloat16()) {
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22bfloat162_rn", "float2", "bfloat16x2_t", "",
                          /*src_needs_reinterpret=*/false,
                          /*dst_needs_reinterpret=*/true);
      return;
    }
  }

  // Handle conversion from float32 to float8 (E4M3/E5M2)
  if (from_ty.is_float() && from_ty.bits() == 32 &&
      tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
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
  if (from_ty.is_float16() && tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
    std::string cvt_func = target_type_is_e4m3
                               ? "tl::detail::cast_half2_to_fp8_e4m3x2"
                               : "tl::detail::cast_half2_to_fp8_e5m2x2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, "half2", "uint16_t", "", false, true);
      return;
    }
  }

  // Handle conversion from bfloat16 to float8 (E4M3/E5M2)
  if (from_ty.is_bfloat16() && tl::IsAscendVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
    std::string cvt_func = target_type_is_e4m3
                               ? "tl::detail::cast_bfloat162_to_fp8_e4m3x2"
                               : "tl::detail::cast_bfloat162_to_fp8_e5m2x2";

    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast(cvt_func, "bfloat16x2_t", "uint16_t", "", true, true);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float32
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float() &&
      target_ty.bits() == 32) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
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
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_float16()) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
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
  if (tl::IsAscendVectorizableFP8(from_ty) && target_ty.is_bfloat16()) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
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

  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangAscend::VisitExpr_(const VarNode *op,
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
  if (it != var_idmap_.end() && op->dtype.is_int() &&
      (it->second == "block_idx" || it->second.rfind("blockIdx.", 0) == 0 ||
       it->second.rfind("threadIdx.", 0) == 0)) {
    os << "((";
    PrintType(op->dtype, os);
    os << ")" << it->second << ")";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangAscend::PrintStorageSync(const CallNode *op) {
  ICHECK_GE(op->args.size(), 1U);
  PrintIndent();
  stream << "asc_syncthreads();\n";
}

void CodeGenTileLangAscend::PrintStorageScope(const std::string &scope,
                                              std::ostream &os) {
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

void CodeGenTileLangAscend::PrintVecElemLoad(const std::string &vec, DataType t,
                                             int i, std::ostream &os) {
  if (t.is_scalar()) {
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
  if (t.is_float16()) {
    os << "((half2*)(&(" << vec << ")))[" << (i / 2) << "]." << access[i % 2];
  } else if (t.is_bfloat16()) {
    os << "((bfloat16x2_t*)(&(" << vec << ")))[" << (i / 2) << "]."
       << access[i % 2];
  } else if (t.is_vector_bool() && t.lanes() > 4) {
    os << "((ushort2*)(&(" << vec << ")))[" << (i / 2) << "]." << access[i % 2];
  } else if (t.lanes() <= 4) {
    os << vec << "." << access[i];
  } else {
    os << "((" << vec << ")." << access[i / 2] << ")";
  }
}

void CodeGenTileLangAscend::PrintVecElemStore(const std::string &vec,
                                              DataType t, int i,
                                              const std::string &value) {
  if (t.is_scalar()) {
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
  if (t.is_float16()) {
    stream << "((half2*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.is_bfloat16()) {
    stream << "((bfloat16x2_t*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.is_vector_bool() && t.lanes() > 4) {
    stream << "((ushort2*)(&(" << vec << ")))[" << (i / 2) << "]."
           << access[i % 2] << " = " << value << ";\n";
  } else if (t.lanes() <= 4) {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  } else {
    stream << "((" << vec << ")." << access[i / 2] << ") = " << value << ";\n";
  }
}

std::string CodeGenTileLangAscend::GetBufferRef(DataType t,
                                                const BufferNode *buffer,
                                                PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope == "local.var") {
    return GetVarID(buffer_var);
  }
  return CodeGenC::GetBufferRef(t, buffer, index);
}

void CodeGenTileLangAscend::PrintVecBinaryOp(const std::string &op, DataType t,
                                             PrimExpr lhs, PrimExpr rhs,
                                             std::ostream &os) {
  ValidateGenericVectorType("binary operation", t);
  // Vector comparisons need per-lane predicate values for vector Select.
  // Division, remainder, min/max, and packed 16-bit arithmetic also require
  // scalarization because Bisheng has no matching small-vector operation.
  const bool lacks_vector_operation = op == "/" || op == "%" || op == "fmodf" ||
                                      op == "fmod" || op == "min" ||
                                      op == "max";
  bool needs_scalarization = t.is_vector_bool() ||
                             (lacks_vector_operation && t.lanes() > 1) ||
                             t.is_float16() || t.is_bfloat16();
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
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());
    for (int i = 0; i < t.lanes(); ++i) {
      std::ostringstream value_temp;
      if (isalpha(op[0])) {
        value_temp << op << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << ", ";
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      } else {
        value_temp << "(";
        PrintVecElemLoad(vlhs, lhs.dtype(), i, value_temp);
        value_temp << op;
        PrintVecElemLoad(vrhs, rhs.dtype(), i, value_temp);
        value_temp << ")";
      }
      PrintVecElemStore(sret, t, i, value_temp.str());
    }
  }
  os << sret;
}

void CodeGenTileLangAscend::EmitSimtVectorPredicateNot(const PrimExpr &value,
                                                       DataType dtype,
                                                       std::ostream &os) {
  ICHECK(dtype.is_vector_bool() && dtype.lanes() <= 8)
      << "Ascend SIMT predicate not expects a small predicate vector, got "
      << dtype;
  ICHECK_EQ(value.dtype(), dtype)
      << "Ascend small-vector predicate not dtype mismatch";

  std::string result = name_supply_->FreshName("_");
  PrintIndent();
  PrintType(dtype, stream);
  stream << ' ' << result << ";\n";
  SSAOperationScope ssa_scope(this);
  std::string operand = SSAGetID(PrintExpr(value), value.dtype());
  for (int i = 0; i < dtype.lanes(); ++i) {
    std::ostringstream lane_value;
    lane_value << "(!";
    PrintVecElemLoad(operand, value.dtype(), i, lane_value);
    lane_value << ')';
    PrintVecElemStore(result, dtype, i, lane_value.str());
  }
  os << result;
}

void CodeGenTileLangAscend::PrintCallExtern(Type ret_type,
                                            ffi::String global_symbol,
                                            const ffi::Array<PrimExpr> &args,
                                            bool skip_first_arg,
                                            std::ostream &os) {
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
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
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }
      for (int i = 0; i < ret_dtype.lanes(); ++i) {
        std::ostringstream scall;
        scall << global_symbol << "(";
        for (size_t j = 0; j < sargs.size(); ++j) {
          if (j > 0)
            scall << ", ";
          PrintVecElemLoad(sargs[j], args[arg_begin + j].dtype(), i, scall);
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
