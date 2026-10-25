/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"
#include "ascend/codegen/sfu_precision.h"

#include "ascend/op/builtin.h"
#include "backend/common/codegen/codegen_utils.h"
#include "backend/common/target_utils.h"
#include "support/check.h"

#include "ascend/transform/native_visitor.h"
#include "compat/visit.h"
#include <algorithm>
#include <array>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/ir/with_context.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace tvm {
namespace codegen {

using namespace tirx;
using ffi::Map;
using ffi::ObjectRef;
using ffi::StructuralEqual;
using tl::PostOrderVisit;
using tl::Substitute;

namespace {
PrimExpr MakeZero(PrimType type) { return MakeConst(type, 0); }
bool UsesVar(const ffi::ObjectRef &root,
             const std::function<bool(const VarNode *)> &predicate) {
  bool found = false;
  tl::PostOrderVisit(root, [&](const ffi::ObjectRef &node) {
    if (auto variable = node.as<VarNode>())
      found = found || predicate(variable);
  });
  return found;
}
TensorVar TensorTypeOf(const VarNode *variable) {
  return GetRef<Var>(variable).as_or_throw<TensorVar>();
}
TensorVar TensorTypeOf(const TensorVar &variable) { return variable; }
ffi::Map<ffi::String, ffi::Any> Attributes(const CallNode *node) {
  return node->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict;
}
ffi::Map<ffi::String, ffi::Any> Attributes(const RegionStmtNode *node) {
  return node->attrs->dict;
}
ffi::Map<ffi::String, ffi::Any> Attributes(const ForNode *node) {
  return node->annotations;
}
ffi::String RegionName(const RegionStmtNode *node) {
  auto name = node->attrs->dict.Get("name_hint");
  return name ? name->cast<ffi::String>() : ffi::String("");
}

constexpr const char *kModeMerging = "MODE_MERGING";
constexpr const char *kModeZeroing = "MODE_ZEROING";
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
  auto primitive = op->ty.as<PrimType>();
  auto tuple = op->ty.as<TupleTypeNode>();
  bool is_void =
      primitive ? primitive->IsVoid() : tuple && tuple->fields.empty();
  if (!is_void || op->args.empty()) {
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

void ValidateSharedScope(const std::string &scope) {
  if (scope == "shared") {
    LOG(FATAL) << "Static shared memory (scope='shared') is not supported on "
                  "Ascend. "
               << "Use dynamic shared memory (scope='shared.dyn') with "
                  "T.alloc_shared.";
  }
}

bool RequiresStatementControlFlow(const Expr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    if (PythonIsOp(call->op, "if_then_else") ||
        PythonIsOp(call->op, "if_then_else")) {
      ICHECK_EQ(call->args.size(), 3U);
      return RequiresStatementControlFlow(call->args[0]) ||
             RequiresStatementControlFlow(call->args[1]) ||
             RequiresStatementControlFlow(call->args[2]);
    }
  }
  // Read-only expressions can remain inside a lazy PTODSL branch. Only an
  // expression that mutates state can require emitted Python statements.
  return SideEffect(expr) > CallEffectKind::kReadState;
}

std::string NormalizeVldDistribution(const std::string &dist) {
  if (dist == "NORM_B8" || dist == "NORM_B16" || dist == "NORM_B32") {
    return "NORM";
  }
  if (dist == "BLK") {
    return "BRC_BLK";
  }
  // AscendC-style UNPK4_B* aliases use the UNPK4 token in PTODSL.
  return dist.rfind("UNPK4_B", 0) == 0 ? "UNPK4" : dist;
}

std::string FP8TypeName(PrimType t) {
  ICHECK(tl::IsAscendVectorizableFP8(t))
      << "PTO Ascend FP8 type expected, got " << t;

  std::string lanes_suffix;
  switch (t.lanes()) {
  case 1:
    break;
  case 2:
  case 4:
  case 8:
    lanes_suffix = "x" + std::to_string(t.lanes());
    break;
  default:
    LOG(FATAL) << "PTO Ascend FP8 types support lanes 1, 2, 4, and 8 only, got "
               << t;
  }

  if (t.MatchesCode(kDLFloat8_e4m3) || t.MatchesCode(kDLFloat8_e4m3fn))
    return "pto.f8e4m3" + lanes_suffix;
  if (t.MatchesCode(kDLFloat8_e5m2))
    return "pto.f8e5m2" + lanes_suffix;
  LOG(FATAL) << "Unsupported PTO Ascend FP8 type: " << t;
  return "";
}

std::string DataTypeName(PrimType t) {
  if (tl::IsAscendVectorizableFP8(t))
    return FP8TypeName(t);

  ICHECK(t.IsScalar()) << "PTO scalar type expected, got " << t;
  if (t.MatchesCode(kDLBool)) {
    return "pto.i1";
  }
  if (t.MatchesCode(kDLFloat)) {
    if (t.bits() == 32)
      return "pto.f32";
    if (t.bits() == 16)
      return "pto.f16";
  } else if (t.MatchesCode(kDLFloat4_e2m1fn)) {
    // TileLang float4_e2m1fn is packed 2xFP4 per byte → PTODSL f4e2m1x2.
    return "pto.f4e2m1x2";
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    return "pto.bf16";
  } else if (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) {
    const char *prefix = t.MatchesCode(kDLUInt) ? "pto.ui" : "pto.i";
    if (t.bits() == 64)
      return std::string(prefix) + "64";
    if (t.bits() == 32)
      return std::string(prefix) + "32";
    if (t.bits() == 16)
      return std::string(prefix) + "16";
    if (t.bits() == 8)
      return std::string(prefix) + "8";
    if (t.bits() == 1)
      return "pto.i1";
  }
  LOG(FATAL) << "Unsupported PTO type: " << t;
  return "";
}

std::string AtomicTypeSuffix(PrimType t) {
  ICHECK(t.IsScalar()) << "PTO store atomic type must be scalar, got " << t;
  if (t.MatchesCode(kDLFloat) && t.bits() == 32) {
    return "f32";
  }
  if (t.MatchesCode(kDLFloat) && t.bits() == 16) {
    return "f16";
  }
  if (t.MatchesElementType(kDLBfloat, 16)) {
    return "bf16";
  }
  if (t.MatchesCode(kDLInt) && t.bits() == 32) {
    return "s32";
  }
  if (t.MatchesCode(kDLInt) && t.bits() == 16) {
    return "s16";
  }
  if (t.MatchesCode(kDLInt) && t.bits() == 8) {
    return "s8";
  }
  LOG(FATAL) << "Unsupported PTO store atomic type: " << t
             << "; expected float32, float16, bfloat16, int32, int16, or int8";
  return "";
}

// Wrap Python/TIR immediates as typed PTODSL consts so vdup/vmaxs keep the
// authored element dtype (e.g. bf16 vs default f16 inference).
std::string WrapTypedConst(const std::string &printed, PrimType dtype,
                           bool is_immediate) {
  if (!is_immediate) {
    return printed;
  }
  return "pto.const(" + printed +
         ", dtype=" + DataTypeName(dtype.WithLanes(1)) + ")";
}

std::string ScalarCastExpr(const std::string &expr, PrimType dtype,
                           const std::string &context) {
  ICHECK(dtype.IsScalar()) << context << " expects a scalar dtype, got "
                           << dtype;
  return "pto.cast(" + expr + ", " + DataTypeName(dtype) + ")";
}

// Peel scalar casts so `Cast(bf16, 0.0)` / `Cast(u16, 32767)` still count as
// immediates for typed-const wrapping.
PrimExpr PeelScalarCasts(const Expr &value) {
  PrimExpr cur = value.as_or_throw<PrimExpr>();
  while (const auto *cast = cur.as<CastNode>()) {
    if (!PythonDType(GetRef<Expr>(cast)).IsScalar() ||
        !PythonDType(cast->value).IsScalar()) {
      break;
    }
    cur = cast->value;
  }
  return cur;
}

bool IsImmediateScalar(const Expr &value) {
  PrimExpr peeled = PeelScalarCasts(value);
  return peeled.as<IntImmNode>() != nullptr ||
         peeled.as<FloatImmNode>() != nullptr;
}

std::string StripPipePrefix(const std::string &name) {
  if (name.rfind("PIPE_", 0) == 0) {
    return name.substr(5);
  }
  return name;
}

PrimType ParseDataType(const std::string &dtype_name) {
  if (dtype_name == "float" || dtype_name == "float32")
    return PrimType::Float(32);
  if (dtype_name == "half" || dtype_name == "float16")
    return PrimType::Float(16);
  if (dtype_name == "bfloat16" || dtype_name == "bfloat16_t")
    return PrimType::BFloat(16);
  if (dtype_name == "float8_e4m3" || dtype_name == "float8_e4m3fn" ||
      dtype_name == "float8_e4m3_t")
    return PrimType(kDLFloat8_e4m3fn, 8);
  if (dtype_name == "float8_e5m2" || dtype_name == "float8_e5m2_t")
    return PrimType(kDLFloat8_e5m2, 8);
  if (dtype_name == "float4_e2m1fn" || dtype_name == "float4_e2m1_t" ||
      dtype_name == "float4_e2m1fnx2" || dtype_name == "float4_e2m1x2_t")
    return PrimType(kDLFloat4_e2m1fn, 4, 2);
  if (dtype_name == "int64")
    return PrimType::Int(64);
  if (dtype_name == "int32")
    return PrimType::Int(32);
  if (dtype_name == "int16")
    return PrimType::Int(16);
  if (dtype_name == "uint64")
    return PrimType::UInt(64);
  if (dtype_name == "uint32")
    return PrimType::UInt(32);
  if (dtype_name == "uint16")
    return PrimType::UInt(16);
  if (dtype_name == "int8")
    return PrimType::Int(8);
  if (dtype_name == "uint16" || dtype_name == "uint16_t")
    return PrimType::UInt(16);
  if (dtype_name == "uint8")
    return PrimType::UInt(8);
  if (dtype_name == "uint32")
    return PrimType::UInt(32);
  if (dtype_name == "uint64")
    return PrimType::UInt(64);
  LOG(FATAL) << "Unsupported PTO dtype string: " << dtype_name;
  return PrimType::Void();
}

bool TryGetConstInt(const Expr &expr, int64_t *value) {
  if (auto constant = tl::GetConstInt(expr)) {
    *value = *constant;
    return true;
  }
  return false;
}

std::string StoreL2CacheName(const Expr &control) {
  static const char *const kNames[] = {
      "nmfv",  "nmlv",  "nmprs",  "nmred",  "naci",   "napw",
      "napi",  "nared", "wbhfv",  "wbhlv",  "wbhprs", "wbhred",
      "wtsfv", "wtslv", "wtsprs", "wtsred",
  };
  int64_t value = 0;
  ICHECK(TryGetConstInt(control, &value))
      << "PTO UB->GM MTE l2_cache_ctl must be a constant in [0, 15], got "
      << control;
  ICHECK_GE(value, 0) << "PTO UB->GM MTE l2_cache_ctl must be in [0, 15], got "
                      << value;
  ICHECK_LT(value, 16) << "PTO UB->GM MTE l2_cache_ctl must be in [0, 15], got "
                       << value;
  return kNames[value];
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.rfind(prefix, 0) == 0;
}

int VstsMaskGranularityOverride(const std::string &dist,
                                PrimType value_elem_dtype) {
  const int width = value_elem_dtype.bits();

  // PTOAS derives the ordinary vsts mask granularity from the value vector
  // element type. Packing/merge distributions are the exception: their mask
  // describes the pre-pack source lanes, so mirror PTOAS' VPTO verifier.
  if (dist == "PK_B16" && width == 8) {
    return 16;
  }
  if (dist == "PK_B32" && width == 16) {
    return 32;
  }
  if (dist == "PK_B64" && width == 32) {
    return 32;
  }
  if (dist == "PK4_B32" && (width == 4 || width == 8)) {
    return 32;
  }
  if (dist == "MRG4CHN_B8" && width == 8) {
    return 32;
  }
  if (dist == "MRG2CHN_B8" && width == 8) {
    return 16;
  }
  return 0;
}

bool IsFloat32(PrimType t) {
  return t.MatchesCode(kDLFloat) && t.bits() == 32 && t.lanes() == 1;
}

bool IsSimtLocalBufferDtype(PrimType t) {
  PrimType elem = t.IsFixedLengthVector() ? t.WithLanes(1) : t;
  if (elem.lanes() != 1)
    return false;
  return IsFloat32(elem) || elem.MatchesElementType(kDLFloat, 16) ||
         elem.MatchesElementType(kDLBfloat, 16) ||
         tl::IsAscendVectorizableFP8(elem) || elem == PrimType::UInt(32) ||
         ((elem.MatchesCode(kDLInt) || elem.MatchesCode(kDLUInt)) &&
          elem.bits() == 8);
}

bool IsFloat32Pair(PrimType t) {
  return t.MatchesCode(kDLFloat) && t.bits() == 32 && t.lanes() == 2;
}

bool IsInteger32(PrimType t) {
  return t.IsScalar() && t.bits() == 32 &&
         (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt));
}

bool IsPackedFp4Pair(PrimType dtype) {
  return dtype.MatchesCode(kDLFloat4_e2m1fn) && dtype.lanes() == 2;
}

bool IsStorageDtypeCompatible(PrimType lhs, PrimType rhs) {
  // Packed FP4 storage is represented as float4_e2m1fnx2: two logical FP4
  // elements in one physical byte.  It is intentionally non-scalar in TVM,
  // but a same-type GM<->UB MTE is a raw storage copy, not a dtype conversion.
  // Keep the exception exact so packed FP4 is never treated as compatible
  // with an unrelated 8-bit element type.
  if (IsPackedFp4Pair(lhs) || IsPackedFp4Pair(rhs)) {
    return lhs == rhs;
  }
  if (!lhs.IsScalar() || !rhs.IsScalar() || lhs.bits() != rhs.bits()) {
    return false;
  }
  if ((lhs.MatchesCode(kDLInt) || lhs.MatchesCode(kDLUInt)) &&
      (rhs.MatchesCode(kDLInt) || rhs.MatchesCode(kDLUInt))) {
    return true;
  }
  return lhs == rhs;
}

bool IsSupportedFloatMinMaxType(PrimType t) {
  if (t.IsScalar())
    return t.MatchesElementType(kDLFloat, 16) || IsFloat32(t) ||
           t.MatchesElementType(kDLBfloat, 16);
  return t.lanes() == 2 &&
         (t.MatchesElementType(kDLFloat, 16) || IsFloat32Pair(t) ||
          t.MatchesElementType(kDLBfloat, 16));
}

bool IsSupportedSIMTUnaryMathTypeImpl(PrimType t) {
  const bool scalar_or_pair = t.IsScalar() || t.lanes() == 2;
  return scalar_or_pair && (t.MatchesElementType(kDLFloat, 16) ||
                            (t.MatchesCode(kDLFloat) && t.bits() == 32));
}

bool IsSupportedSIMTUnaryMathType(PrimType t) {
  return IsSupportedSIMTUnaryMathTypeImpl(t);
}

bool IsSupportedScalarUnaryMathType(PrimType t) {
  return t.IsScalar() && (t.MatchesElementType(kDLFloat, 16) || IsFloat32(t) ||
                          t.MatchesElementType(kDLBfloat, 16));
}

bool IsSupportedSIMTAllReduceType(PrimType t) {
  return t.IsScalar() && (t.MatchesElementType(kDLFloat, 16) || IsFloat32(t) ||
                          t == PrimType::Int(32) || t == PrimType::UInt(32));
}

bool IsSupportedSIMTLocalScalarAccessType(PrimType t) {
  return t.IsScalar() && (IsSimtLocalBufferDtype(t) || t == PrimType::Int(32));
}

bool IsSupportedSIMTLocalStorageType(PrimType t) {
  return IsSimtLocalBufferDtype(t) || (t.IsScalar() && t == PrimType::Int(32));
}

std::string SIMTLocalStorageTypeName(PrimType t) {
  ICHECK(IsSupportedSIMTLocalStorageType(t));
  // LLVM stack allocations require signless integer element types. Preserve
  // TileLang's signedness at load/store boundaries instead.
  return IsInteger32(t) ? "pto.i32" : DataTypeName(t);
}

bool IsSupportedSIMTFP8ContiguousLaneCount(int lanes) {
  return lanes == 2 || lanes == 4 || lanes == 8;
}

bool IsPowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

void CheckContiguousRampStride(const Expr &index, const char *access_kind) {
  if (const auto *ramp = index.as<RampNode>()) {
    PrimExpr stride_expr = sym::Analyzer()->Simplify(ramp->stride);
    int64_t stride = 0;
    ICHECK(TryGetConstInt(stride_expr, &stride) && stride == 1)
        << "PTO SIMT vector " << access_kind
        << " emits a contiguous vector access, so Ramp stride must be 1, got "
        << ramp->stride;
  }
}

void CheckConstZero(const Expr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value) && value == 0)
      << "PTO codegen currently only supports " << name
      << " == 0 for tl.ascend_copy_gm_to_ubuf, got " << expr;
}

bool IsOpName(const ObjectRef &op, const std::string &name) {
  if (auto opt_call_op = op.as<Op>()) {
    return opt_call_op.value()->name == name;
  }
  return false;
}

int64_t ConstShapeDim(const Expr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value))
      << name << " must be static, got " << expr;
  return value;
}

int64_t ConstArgDim(const CallNode *call, size_t index, const char *name) {
  ICHECK_GT(call->args.size(), index) << name << " argument is missing";
  return ConstShapeDim(call->args[index], name);
}

// SIMD register temps (T.simd.alloc_var): lanes >= 32.
bool IsVectorLocalVarDtype(PrimType dtype) { return dtype.lanes() >= 32; }

// Keep scalar local.var as authored PTO surface values. PTODSL should model
// region rebinding with proper if/loop results.
bool IsSurfaceScalarLocalVarDtype(PrimType dtype) { return dtype.lanes() == 1; }

int64_t PhysicalElementBits(PrimType dtype) {
  if (dtype.MatchesCode(kDLBool)) {
    return 8;
  }
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

void ValidateGmBypassDtype(PrimType dtype) {
  ICHECK(dtype.IsScalar())
      << "PTO GM dcache bypass expects a scalar dtype, got " << dtype;
  const int64_t bits = PhysicalElementBits(dtype);
  ICHECK(bits == 8 || bits == 16 || bits == 32 || bits == 64)
      << "PTO GM dcache bypass only supports 1/2/4/8-byte scalar types, got "
      << dtype;
}

int64_t PhysicalElementBytes(PrimType dtype) {
  int64_t bits = PhysicalElementBits(dtype);
  ICHECK_EQ(bits % 8, 0)
      << "PTO physical element width must be byte-aligned, got " << dtype
      << " (" << bits << " bits)";
  return bits / 8;
}

PrimExpr ElementIndexToByteOffsetExpr(const PrimExpr &elem_index,
                                      PrimType dtype, const char *context,
                                      sym::Analyzer analyzer) {
  int64_t elem_bits = PhysicalElementBits(dtype);
  if (elem_bits < 8) {
    ICHECK_EQ(8 % elem_bits, 0)
        << context << " packed dtype bit width must divide one byte, got "
        << dtype;
    int64_t pack_factor = 8 / elem_bits;
    PrimExpr pack = MakeConst(PythonDType(elem_index), pack_factor);
    PrimExpr elem_mod = analyzer->Simplify(floormod(elem_index, pack));
    int64_t remainder = 0;
    ICHECK(!TryGetConstInt(elem_mod, &remainder) || remainder == 0)
        << context << " expects a packed-byte-aligned logical offset, got "
        << elem_index << " for " << dtype;
    return analyzer->Simplify(FloorDiv(elem_index, pack));
  }

  ICHECK_EQ(elem_bits % 8, 0)
      << context << " physical element width must be byte-aligned, got "
      << dtype << " (" << elem_bits << " bits)";
  int64_t elem_bytes = elem_bits / 8;
  return analyzer->Simplify(elem_index *
                            MakeConst(PythonDType(elem_index), elem_bytes));
}

int64_t ElementsToBytesCeil(int64_t elements, PrimType dtype) {
  int64_t elem_bits = PhysicalElementBits(dtype);
  return (elements * elem_bits + 7) / 8;
}

std::string PointerElementTypeName(PrimType dtype) {
  // Pointer payloads are scalar element types even when the TIR buffer
  // carries a vector dtype (e.g. float32x64 in mhc_norm_fn). Preserve the
  // packed FP4 pair mapping before peeling vector lanes.
  if (IsPackedFp4Pair(dtype)) {
    return "pto.f4e2m1x2";
  }
  PrimType elem_dtype =
      dtype.IsFixedLengthVector() ? dtype.WithLanes(1) : dtype;
  if (elem_dtype.MatchesCode(kDLBool)) {
    return "pto.i8";
  }
  return DataTypeName(elem_dtype);
}

void CheckLocalVarBuffer(const VarNode *buffer) {
  ICHECK_EQ(TensorTypeOf(buffer).shape().size(), 1U)
      << "PTO local.var only supports scalar alloc_var buffers, got rank "
      << TensorTypeOf(buffer).shape().size();
  int64_t extent = 0;
  ICHECK(TryGetConstInt(TensorTypeOf(buffer).shape()[0], &extent) &&
         extent == 1)
      << "PTO local.var only supports scalar alloc_var buffers with shape "
         "(1,), got "
      << TensorTypeOf(buffer).shape()[0];
  PrimType dtype = TensorTypeOf(buffer).dtype();
  if (dtype.MatchesCode(kDLOpaqueHandle)) {
    // local.var handle buffers carry mutable typed pointers (e.g. UB views).
    // They are initialized to None and receive a pointer on their first store.
    return;
  }
  if (IsVectorLocalVarDtype(dtype) || dtype.MatchesCode(kDLOpaqueHandle)) {
    (void)DataTypeName(dtype.WithLanes(1));
    return;
  }
  ICHECK_EQ(dtype.lanes(), 1)
      << "PTO local.var only supports scalar or full-width vector values, got "
      << dtype;
  PrimType elem_dtype = dtype.WithLanes(1);
  int bits = elem_dtype.bits();
  bool supported =
      elem_dtype.MatchesCode(kDLBool) ||
      elem_dtype.MatchesElementType(kDLBfloat, 16) ||
      elem_dtype.MatchesCode(kDLFloat8_e4m3fn) ||
      elem_dtype.MatchesCode(kDLFloat8_e5m2) ||
      (elem_dtype.MatchesCode(kDLFloat) && (bits == 16 || bits == 32)) ||
      ((elem_dtype.MatchesCode(kDLInt) || elem_dtype.MatchesCode(kDLUInt)) &&
       (bits == 1 || bits == 8 || bits == 16 || bits == 32 || bits == 64));
  ICHECK(supported)
      << "PTO local.var only supports PTO numeric scalar/vector element "
         "types, got "
      << dtype;
}

void CheckAllReduceDtype(const CallNode *call) {
  ICHECK_GE(call->args.size(), 2U)
      << "tl::AscendAllReduce call expects a value argument";
  PrimType dtype = PythonDType(call->args[1]);
  ICHECK(IsSupportedSIMTAllReduceType(dtype))
      << "PTO cross-thread allreduce currently supports float16, float32, "
         "int32, and uint32 only, got "
      << dtype
      << ". Use a supported reducer dtype, such as float32, and cast the "
         "finalized result afterward.";
}

void ValidateKernelCapabilities(const tirx::Function &func) {
  bool has_cube_block = false;
  bool has_vector_block = false;
  bool has_gemm_l1 = false;
  bool has_blockscaled_gemm_l1 = false;
  bool has_mad = false;
  bool has_gm_to_ub_mte = false;
  bool has_ub_to_gm_mte = false;
  bool has_copy_pad_setter = false;
  bool has_l0c_to_ub_mte = false;
  bool has_supported_cube_mte = false;
  (void)0; // has_unsupported_cube_op removed: nd2nz ops are emitted below

  PostOrderVisit(func->body.value(), [&](const ffi::ObjectRef &node) {
    if (const auto *block = node.as<RegionStmtNode>()) {
      if (RegionName(block) == "CUBE") {
        has_cube_block = true;
      } else if (RegionName(block) == "VECTOR") {
        has_vector_block = true;
      }
    }

    if (const auto *call = node.as<CallNode>()) {
      if ((PythonIsOp(call->op, "call_extern") ||
           PythonIsOp(call->op, "call_pure_extern")) &&
          !call->args.empty()) {
        const auto *func_name = call->args[0].as<StringImmNode>();
        if (func_name != nullptr &&
            func_name->value.find("tl::AscendAllReduce") != std::string::npos) {
          CheckAllReduceDtype(call);
        }
      }
      if (call->op.same_as(tl::ascend_gemm_l1())) {
        has_gemm_l1 = true;
      } else if (call->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
        has_blockscaled_gemm_l1 = true;
      } else if (call->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
        has_gm_to_ub_mte = true;
      } else if (call->op.same_as(tl::ascend_copy_ubuf_to_gm())) {
        has_ub_to_gm_mte = true;
      } else if (call->op.same_as(tl::ascend_set_copy_pad_value())) {
        has_copy_pad_setter = true;
      } else if (call->op.same_as(tl::ascend_copy_matrix_cc_to_ub())) {
        has_l0c_to_ub_mte = true;
      } else if (call->op.same_as(tl::ascend_copy_gm_to_cbuf()) ||
                 call->op.same_as(tl::ascend_fill_l1()) ||
                 call->op.same_as(tl::ascend_load_cbuf_to_ca()) ||
                 call->op.same_as(tl::ascend_load_cbuf_to_cb()) ||
                 call->op.same_as(tl::ascend_load_ca_sf()) ||
                 call->op.same_as(tl::ascend_load_cb_sf()) ||
                 call->op.same_as(tl::ascend_copy_ubuf_to_cbuf())) {
        has_supported_cube_mte = true;
      } else if (call->op.same_as(tl::ascend_mad()) ||
                 call->op.same_as(tl::ascend_mad_mx())) {
        has_mad = true;
      }
      // ascend_nd2nz_scatter / ascend_nd2nz_post_copy are emitted by
      // EmitAscendNd2NzScatter / the post-copy emitter below; they are
      // supported and must not trip the unsupported-op guard.
    }
  });

  ICHECK(!has_cube_block || has_gemm_l1 || has_blockscaled_gemm_l1 ||
         has_supported_cube_mte || has_mad)
      << "PTO codegen requires a supported Cube operation: GM->L1, L1 fill, "
         "L1->L0A/B, UB->L1, tl.ascend_gemm_l1, tl.ascend_blockscaled_gemm_l1, "
         "tl.ascend_mad, or tl.ascend_mad_mx";
  bool has_cube_path = has_cube_block || has_gemm_l1 ||
                       has_blockscaled_gemm_l1 || has_supported_cube_mte ||
                       has_l0c_to_ub_mte || has_mad;
  bool has_mixed_sections = has_cube_block && has_vector_block;
  ICHECK(!has_cube_path ||
         (!has_gm_to_ub_mte && !has_ub_to_gm_mte && !has_copy_pad_setter) ||
         has_mixed_sections)
      << "The same PTO function contains a CUBE block and vector GM<->UB "
         "MTE/padding operations, but the IR does not contain both CUBE and "
         "VECTOR sections required for a mixed PTO kernel.";
  ICHECK(!has_l0c_to_ub_mte || has_gemm_l1 || has_blockscaled_gemm_l1 ||
         has_mad)
      << "PTO L0C->UB MTE requires a supported CUBE kernel containing "
         "tl.ascend_gemm_l1, tl.ascend_blockscaled_gemm_l1, tl.ascend_mad, "
         "or tl.ascend_mad_mx";
}

std::string AccStoreUnitFlagArg(int64_t unit_flag_ctrl) {
  if (unit_flag_ctrl == 0)
    return "None";
  if (unit_flag_ctrl == 2)
    return "pto.AccStoreUnitFlagCtrl.CHECK_ONLY";
  if (unit_flag_ctrl == 3)
    return "pto.AccStoreUnitFlagCtrl.CHECK_AND_CLEAR";
  LOG(FATAL) << "PTO GEMM unsupported L0C store unit_flag_ctrl="
             << unit_flag_ctrl;
  return "None";
}

bool IsSupportedGemmInputDtype(PrimType dtype) {
  return dtype.MatchesElementType(kDLBfloat, 16) ||
         (dtype.MatchesCode(kDLFloat) &&
          (dtype.bits() == 16 || dtype.bits() == 32)) ||
         dtype.MatchesCode(kDLFloat8_e4m3fn) ||
         dtype.MatchesCode(kDLFloat8_e5m2) ||
         dtype.MatchesCode(kDLFloat4_e2m1fn);
}

bool IsSupportedBlockscaledGemmInputDtype(PrimType dtype) {
  return dtype.MatchesElementType(kDLBfloat, 16) ||
         dtype.MatchesCode(kDLFloat8_e4m3fn) ||
         dtype.MatchesCode(kDLFloat4_e2m1fn);
}

int64_t GemmInputPackFactor(PrimType dtype) {
  return dtype.MatchesCode(kDLFloat4_e2m1fn) ? 2 : 1;
}

int64_t GemmInputC0(PrimType dtype) {
  ICHECK(IsSupportedGemmInputDtype(dtype))
      << "PTO GEMM L1 helper only supports float16, bfloat16, float32, "
         "float8_e4m3fn, float8_e5m2, or float4_e2m1fn inputs, got "
      << dtype;
  if (dtype.MatchesCode(kDLFloat4_e2m1fn)) {
    return 64;
  }
  int64_t elem_bytes = dtype.StorageBytes();
  ICHECK_GT(elem_bytes, 0) << "Invalid PTO GEMM input dtype size: " << dtype;
  ICHECK_EQ(32 % elem_bytes, 0)
      << "PTO GEMM input dtype byte size must divide 32, got " << dtype;
  return 32 / elem_bytes;
}

bool IsSupportedCubeMteDtype(PrimType dtype) {
  return dtype.MatchesElementType(kDLBfloat, 16) ||
         dtype.MatchesCode(kDLFloat8_e4m3fn) ||
         dtype.MatchesCode(kDLFloat8_e5m2) ||
         dtype.MatchesCode(kDLFloat4_e2m1fn) ||
         (dtype.MatchesCode(kDLFloat) &&
          (dtype.bits() == 16 || dtype.bits() == 32) && dtype.lanes() == 1) ||
         (dtype.MatchesCode(kDLInt) && dtype.bits() == 8 && dtype.lanes() == 1);
}

bool IsSupportedFp8Dtype(PrimType dtype) {
  return dtype.IsScalar() && (dtype.MatchesCode(kDLFloat8_e4m3fn) ||
                              dtype.MatchesCode(kDLFloat8_e5m2));
}

bool IsSupportedMadDtypeTuple(PrimType lhs, PrimType rhs, PrimType acc) {
  if (!lhs.IsScalar() || !rhs.IsScalar() || !acc.IsScalar()) {
    return false;
  }

  bool is_same_standard_float =
      lhs == rhs &&
      (lhs.MatchesElementType(kDLBfloat, 16) ||
       (lhs.MatchesCode(kDLFloat) && (lhs.bits() == 16 || lhs.bits() == 32)));
  bool is_fp8_pair = IsSupportedFp8Dtype(lhs) && IsSupportedFp8Dtype(rhs);
  if ((is_same_standard_float || is_fp8_pair) && IsFloat32(acc)) {
    return true;
  }

  return lhs == PrimType::Int(8) && rhs == PrimType::Int(8) &&
         acc == PrimType::Int(32);
}

bool IsSupportedMadMxDtypeTuple(PrimType lhs, PrimType rhs, PrimType acc) {
  return (IsSupportedFp8Dtype(lhs) || IsPackedFp4Pair(lhs)) &&
         (IsSupportedFp8Dtype(rhs) || IsPackedFp4Pair(rhs)) && IsFloat32(acc);
}

bool IsBufferInScope(const std::string &scope,
                     const std::string &expected_scope) {
  return scope == expected_scope || scope == expected_scope + ".dyn";
}

PrimType GetAnnotatedPointerDtype(const Expr &expr, PrimType fallback_dtype) {
  if (auto pointer = expr->ty.as<PtrTypeNode>()) {
    if (auto element = pointer->element_type.as<PrimType>();
        element && !element.value().IsVoid()) {
      return element.value();
    }
  }
  const auto *call = expr.as<CallNode>();
  if (call == nullptr) {
    return fallback_dtype;
  }

  if (PythonIsOp(call->op, "tvm_access_ptr") && !call->args.empty()) {
    if (const auto *type_call = call->args[0].as<CallNode>()) {
      if (!type_call->args.empty()) {
        if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
          return ParseDataType(dtype_name->value);
        }
      }
    }
    return fallback_dtype;
  }

  if (PythonIsOp(call->op, "address_of") && call->args.size() == 1U) {
    if (const auto *load = call->args[0].as<TensorLoadNode>()) {
      return load->source.as_or_throw<TensorVar>().dtype();
    }
  }

  if (call->op.same_as(tl::access_ptr()) && call->args.size() >= 1U) {
    if (const auto *load = call->args[0].as<TensorLoadNode>()) {
      return load->source.as_or_throw<TensorVar>().dtype();
    }
  }

  return fallback_dtype;
}

// Vector-typed local buffers (S.alloc_local / S.alloc_var) have elements
// whose dtype carries lanes (e.g. float32x64). The pass pipeline widens a
// scalar element index `i` into a Ramp(i*lanes, 1, lanes). This recovers
// the scalar element index from such a Ramp.
PrimExpr VectorBufferElemIndex(const Expr &index) {
  if (const auto *ramp = index.as<RampNode>()) {
    int lanes = PythonDType(GetRef<Expr>(ramp)).lanes();
    if (lanes > 0) {
      return sym::Analyzer()->Simplify(
          floordiv(ramp->base, MakeConst(PythonDType(ramp->base), lanes)));
    }
  }
  return index.as_or_throw<PrimExpr>();
}

bool GetAddressOfIndex(const Expr &expr, PrimExpr *index,
                       const VarNode **buffer_var) {
  if (const auto *variable = expr.as<VarNode>();
      variable && PythonIsPointer(expr)) {
    *index = IntImm::Int64(0);
    *buffer_var = variable;
    return true;
  }
  const auto *call = expr.as<CallNode>();
  if (call == nullptr) {
    return false;
  }
  if (call->op.same_as(tirx::tensor_data_ptr_op())) {
    *buffer_var = call->args[0].as<VarNode>();
    *index = IntImm::Int64(0);
    return *buffer_var != nullptr;
  }
  if (call->op.same_as(tirx::ptr_byte_offset_op()) ||
      (call->op.same_as(tirx::reinterpret_op()) && PythonIsPointer(expr))) {
    PrimExpr base_index(0);
    if (!GetAddressOfIndex(call->args[0], &base_index, buffer_var))
      return false;
    auto source_dtype =
        GetAnnotatedPointerDtype(call->args[0], PrimType::UInt(8));
    auto target_dtype = GetAnnotatedPointerDtype(expr, PrimType::UInt(8));
    PrimExpr bits = prim::cast(base_index, PrimType::Int(64)) *
                    IntImm::Int64(PhysicalElementBits(source_dtype));
    if (call->op.same_as(tirx::ptr_byte_offset_op())) {
      bits = bits + prim::cast(call->args[1].as_or_throw<PrimExpr>(),
                               PrimType::Int(64)) *
                        IntImm::Int64(8);
    }
    *index = sym::Analyzer()->Simplify(
        FloorDiv(bits, IntImm::Int64(PhysicalElementBits(target_dtype))));
    return true;
  }
  if (PythonIsOp(call->op, "address_of")) {
    if (call->args.size() != 1U) {
      return false;
    }
    const auto *load = call->args[0].as<TensorLoadNode>();
    if (load == nullptr || load->indices.size() != 1U) {
      return false;
    }
    *index = load->indices[0];
    *buffer_var = load->source.as_or_throw<TensorVar>().get();
    return true;
  }
  if (PythonIsOp(call->op, "tvm_access_ptr")) {
    if (call->args.size() < 3U) {
      return false;
    }
    const auto *var = call->args[1].as<VarNode>();
    if (var == nullptr) {
      return false;
    }
    *index = call->args[2].as_or_throw<PrimExpr>();
    *buffer_var = var;
    return true;
  }
  if (call->op.same_as(tl::access_ptr())) {
    if (call->args.size() < 1U) {
      return false;
    }
    const auto *load = call->args[0].as<TensorLoadNode>();
    if (load == nullptr || load->indices.size() != 1U) {
      return false;
    }
    *index = load->indices[0];
    *buffer_var = load->source.as_or_throw<TensorVar>().get();
    return true;
  }
  return false;
}

struct PTOModeState {
  std::array<PrimExpr, 3> guards{MakeConst(PrimType::Bool(), false),
                                 MakeConst(PrimType::Bool(), false),
                                 MakeConst(PrimType::Bool(), false)};
};

struct PTOModeFlowState {
  PTOModeState fallthrough;
  PTOModeState breaks;
  PTOModeState continues;
};

class PTOModeAnalyzer : public tl::ascend::AnalysisFunctor<PTOModeFlowState(
                            const Stmt &, PTOModeState)> {
public:
  enum class Kind { kHf32, kMadDirection };
  using ModeMap =
      std::unordered_map<Call, uint8_t, ObjectPtrHash, ObjectPtrEqual>;

  static ModeMap Analyze(const tirx::Function &function, Kind kind) {
    PTOModeAnalyzer analyzer(kind);
    analyzer.VisitStmt(function->body.value(), FromMask(1));
    return std::move(analyzer.modes_);
  }

private:
  using State = PTOModeState;
  using FlowState = PTOModeFlowState;

  explicit PTOModeAnalyzer(Kind kind) : kind_(kind) {}

  static State FromMask(uint8_t mask) {
    State state;
    for (size_t mode = 0; mode < state.guards.size(); ++mode)
      state.guards[mode] =
          MakeConst(PrimType::Bool(), (mask & (1U << mode)) != 0);
    return state;
  }

  uint8_t Mask(const State &state) {
    uint8_t result = 0;
    for (size_t mode = 0; mode < state.guards.size(); ++mode)
      if (!analyzer_->CanProve(!state.guards[mode],
                               sym::ProofStrength::kSymbolicBound))
        result |= 1U << mode;
    return result;
  }

  State Merge(const State &first, const State &second) {
    State result;
    for (size_t mode = 0; mode < result.guards.size(); ++mode)
      result.guards[mode] =
          analyzer_->Simplify(first.guards[mode] || second.guards[mode]);
    return result;
  }

  State Restrict(const State &state, const PrimExpr &condition) {
    if (SideEffect(condition) != CallEffectKind::kPure)
      return state;
    State result;
    for (size_t mode = 0; mode < result.guards.size(); ++mode)
      result.guards[mode] =
          analyzer_->Simplify(state.guards[mode] && condition);
    return result;
  }

  State SetMode(const State &state, size_t mode) {
    State result;
    PrimExpr reachable = MakeConst(PrimType::Bool(), false);
    for (const PrimExpr &guard : state.guards)
      reachable = analyzer_->Simplify(reachable || guard);
    result.guards[mode] = reachable;
    return result;
  }

  FlowState AnalyzeLoop(const Stmt &body, State incoming, bool must_execute,
                        bool can_exit_normally = true) {
    uint8_t incoming_mask = Mask(incoming);
    uint8_t header = incoming_mask;
    for (int iteration = 0; iteration < 8; ++iteration) {
      FlowState body_out = VisitStmt(body, FromMask(header));
      uint8_t backedge = Mask(Merge(body_out.fallthrough, body_out.continues));
      uint8_t next = incoming_mask | backedge;
      if (next == header) {
        uint8_t exits = Mask(body_out.breaks);
        if (can_exit_normally) {
          exits |= backedge;
          if (!must_execute)
            exits |= incoming_mask;
        }
        return {FromMask(exits), {}, {}};
      }
      header = next;
    }
    LOG(FATAL) << "PTO mode loop analysis failed to converge";
    return {incoming, {}, {}};
  }

  FlowState VisitStmt_(const BindNode *op, State state) final {
    return VisitValue(op->value, state);
  }

  FlowState VisitStmt_(const IfNode *op, State state) final {
    PrimExpr condition = analyzer_->Simplify(op->condition);
    FlowState then_out = VisitStmt(op->then_case, Restrict(state, condition));
    FlowState else_out =
        op->else_case
            ? VisitStmt(op->else_case.value(), Restrict(state, !condition))
            : FlowState{Restrict(state, !condition), {}, {}};
    return {Merge(then_out.fallthrough, else_out.fallthrough),
            Merge(then_out.breaks, else_out.breaks),
            Merge(then_out.continues, else_out.continues)};
  }

  FlowState VisitStmt_(const ForNode *op, State state) final {
    PrimExpr enters = (op->step > 0 && op->begin < op->end) ||
                      (op->step < 0 && op->begin > op->end);
    if (analyzer_->CanProve(!enters)) {
      VisitStmt(op->body, {});
      return {state, {}, {}};
    }
    PrimExpr one_trip =
        enters && ((op->step > 0 && op->begin + op->step >= op->end) ||
                   (op->step < 0 && op->begin + op->step <= op->end));
    if (analyzer_->CanProve(one_trip)) {
      With<sym::ConstraintContext> scope(analyzer_, op->var == op->begin);
      FlowState body_out = VisitStmt(op->body, state);
      State exits = Merge(Merge(body_out.fallthrough, body_out.breaks),
                          body_out.continues);
      Map<PrimVar, PrimExpr> values{{op->var, op->begin}};
      for (PrimExpr &guard : exits.guards)
        guard = analyzer_->Simplify(Substitute(guard, values));
      return {exits, {}, {}};
    }
    return AnalyzeLoop(op->body, state, analyzer_->CanProve(enters));
  }

  FlowState VisitStmt_(const WhileNode *op, State state) final {
    PrimExpr condition = analyzer_->Simplify(op->condition);
    if (IsZero(condition)) {
      VisitStmt(op->body, {});
      return {state, {}, {}};
    }
    const bool always_true = IsOne(condition);
    return AnalyzeLoop(op->body, state, always_true, !always_true);
  }

  FlowState VisitStmt_(const TensorStoreNode *, State state) final {
    return {state, {}, {}};
  }

  FlowState VisitStmt_(const AssertStmtNode *, State state) final {
    return {state, {}, {}};
  }

  FlowState VisitStmt_(const SeqStmtNode *op, State state) final {
    FlowState result{state, {}, {}};
    for (const Stmt &statement : op->seq) {
      FlowState next = VisitStmt(statement, result.fallthrough);
      result.fallthrough = next.fallthrough;
      result.breaks = Merge(result.breaks, next.breaks);
      result.continues = Merge(result.continues, next.continues);
    }
    return result;
  }

  FlowState VisitStmt_(const BreakNode *, State state) final {
    return {{}, state, {}};
  }
  FlowState VisitStmt_(const ContinueNode *, State state) final {
    return {{}, {}, state};
  }
  FlowState VisitStmt_(const ReturnNode *, State) final { return {}; }

  FlowState VisitStmt_(const EvaluateNode *op, State state) final {
    return VisitValue(op->value, state);
  }

  FlowState VisitValue(const Expr &value, State state) {
    const auto *call = value.as<CallNode>();
    if (!call)
      return {state, {}, {}};
    if (call->op.same_as(tl::loop_break()))
      return {{}, state, {}};
    if (kind_ == Kind::kHf32 && call->op.same_as(tl::ascend_set_hf32_mode())) {
      ICHECK_EQ(call->args.size(), 1U)
          << "tl.ascend_set_hf32_mode expects exactly 1 argument";
      int64_t mode = 0;
      ICHECK(TryGetConstInt(call->args[0], &mode) && mode >= 0 && mode <= 2)
          << "PTO HF32 mode must be a constant in {0, 1, 2}, got "
          << call->args[0];
      return {SetMode(state, mode), {}, {}};
    }
    if (kind_ == Kind::kMadDirection &&
        call->op.same_as(tl::ascend_set_mmad_direction())) {
      ICHECK_EQ(call->args.size(), 1U);
      const auto *direction = call->args[0].as<StringImmNode>();
      ICHECK(direction && (direction->value == "m" || direction->value == "n"))
          << "PTO MAD direction expects 'm' or 'n'";
      return {SetMode(state, direction->value == "n"), {}, {}};
    }
    bool record = call->op.same_as(tl::ascend_mad()) ||
                  call->op.same_as(tl::ascend_mad_mx());
    if (call->op.same_as(tl::ascend_gemm_l1())) {
      ICHECK_GT(call->args.size(), 9U);
      const auto *dtype_name = call->args[9].as<StringImmNode>();
      ICHECK(dtype_name) << "PTO GEMM requires a constant input dtype string";
      record = kind_ == Kind::kMadDirection ||
               IsFloat32(ParseDataType(dtype_name->value));
    }
    if (kind_ == Kind::kMadDirection &&
        call->op.same_as(tl::ascend_blockscaled_gemm_l1()))
      record = true;
    if (record)
      modes_[GetRef<Call>(call)] |= Mask(state);
    return {state, {}, {}};
  }

  FlowState VisitStmt_(const RegionStmtNode *op, State state) final {
    return VisitStmt(op->body, state);
  }

  Kind kind_;
  ModeMap modes_;
  sym::Analyzer analyzer_;
};

class PTOHf32ModeAnalyzer {
public:
  using ModeMap =
      std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>;

  static ModeMap Analyze(const tirx::Function &function) {
    ModeMap result;
    for (const auto &[call, modes] :
         PTOModeAnalyzer::Analyze(function, PTOModeAnalyzer::Kind::kHf32)) {
      ICHECK_EQ(modes & (modes - 1), 0)
          << "PTO HF32 mode is control-flow dependent for GEMM " << call
          << "; set one deterministic HF32 mode before this GEMM";
      result.emplace(call, modes == 4 ? 2 : modes == 2 ? 1 : 0);
    }
    return result;
  }
};

class PTOMmadDirectionAnalyzer {
public:
  using DirectionMap =
      std::unordered_map<Call, bool, ObjectPtrHash, ObjectPtrEqual>;

  static DirectionMap Analyze(const tirx::Function &function) {
    DirectionMap result;
    for (const auto &[call, modes] : PTOModeAnalyzer::Analyze(
             function, PTOModeAnalyzer::Kind::kMadDirection)) {
      ICHECK(modes == 0 || modes == 1 || modes == 2)
          << "PTO MAD direction depends on control flow for " << call;
      result.emplace(call, modes == 2);
    }
    return result;
  }
};

// Detects whether a statement (loop body) contains a tl.loop_break() call,
// without recursing into nested for loops (their breaks are their own).
class LoopBreakDetector : public tl::ascend::AnalysisVisitor {
public:
  bool found = false;
  void VisitStmt_(const BreakNode *) override { found = true; }
  void VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(tl::loop_break())) {
      found = true;
    }
    if (!found)
      tl::ascend::AnalysisVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const ForNode *op) override { (void)op; }
  void VisitStmt_(const WhileNode *op) override { (void)op; }
};

static bool ContainsLoopBreak(const Stmt &stmt) {
  LoopBreakDetector det;
  det(stmt);
  return det.found;
}

class SimtPersistentBufferCollector final : public tl::ascend::AnalysisVisitor {
public:
  std::unordered_set<const VarNode *> Collect(const Stmt &body) {
    VisitStmt(body);

    std::unordered_set<const VarNode *> persistent_buffers;
    for (const VarNode *var : outer_local_allocations_) {
      if (simt_accesses_.count(var)) {
        persistent_buffers.insert(var);
      }
    }
    return persistent_buffers;
  }

private:
  void RecordAllocation(const TensorVar &buffer) {
    std::string scope = buffer.scope().name();
    if (simt_depth_ == 0 && (scope == "local" || scope == "local.fragment")) {
      outer_local_allocations_.insert(TensorTypeOf(buffer).get());
    }
  }

  void RecordAccess(const TensorVar &buffer) {
    if (simt_depth_ > 0) {
      simt_accesses_.insert(TensorTypeOf(buffer).get());
    }
  }

  void VisitStmt_(const BindNode *op) final {
    if (auto tensor = op->var.as<TensorVar>())
      RecordAllocation(tensor.value());
    tl::ascend::AnalysisVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const RegionStmtNode *op) final {
    const bool is_simt = RegionName(op) == "SIMT_VF";
    if (is_simt) {
      ++simt_depth_;
    }
    tl::ascend::AnalysisVisitor::VisitStmt_(op);
    if (is_simt) {
      --simt_depth_;
    }
  }

  void VisitExpr_(const TensorLoadNode *op) final {
    RecordAccess(op->source.as_or_throw<TensorVar>());
    tl::ascend::AnalysisVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const TensorStoreNode *op) final {
    RecordAccess(op->dest.as_or_throw<TensorVar>());
    tl::ascend::AnalysisVisitor::VisitStmt_(op);
  }

  int simt_depth_{0};
  std::unordered_set<const VarNode *> outer_local_allocations_;
  std::unordered_set<const VarNode *> simt_accesses_;
};

struct MixedSectionVariableInfo {
  using VarSet = std::unordered_set<const VarNode *>;

  std::unordered_map<const RegionStmtNode *, VarSet> external_vars_by_section;
  VarSet captured_vars;
};

class MixedSectionLocalVarAnalyzer {
  using VarSet = MixedSectionVariableInfo::VarSet;
  using OwnerMap = std::unordered_map<const VarNode *, const RegionStmtNode *>;

  class StructureCollector final : public tl::ascend::AnalysisVisitor {
  public:
    void Collect(const Stmt &body) { VisitStmt(body); }

    const std::vector<const RegionStmtNode *> &sections() const {
      return sections_;
    }
    const OwnerMap &owners() const { return owners_; }

  private:
    static bool IsPhysicalSection(const RegionStmtNode *op) {
      return RegionName(op) == "CUBE" || RegionName(op) == "VECTOR";
    }

    void RecordAllocation(const TensorVar &buffer) {
      if (buffer.scope().name() != "local.var") {
        return;
      }
      const VarNode *var = TensorTypeOf(buffer).get();
      auto [it, inserted] = owners_.emplace(var, current_section_);
      ICHECK(inserted || it->second == current_section_)
          << "A local.var buffer is allocated in multiple mixed-section "
             "ownership regions: "
          << buffer;
    }

    void VisitStmt_(const BindNode *op) final {
      if (auto tensor = op->var.as<TensorVar>())
        RecordAllocation(tensor.value());
    }

    void VisitStmt_(const RegionStmtNode *op) final {
      const bool is_physical = IsPhysicalSection(op);
      ICHECK(!is_physical || current_section_ == nullptr)
          << "Nested CUBE/VECTOR physical sections are not supported: "
          << RegionName(op);

      const RegionStmtNode *saved_section = current_section_;
      if (is_physical) {
        current_section_ = op;
        sections_.push_back(op);
      }
      VisitStmt(op->body);
      current_section_ = saved_section;
    }

    const RegionStmtNode *current_section_{nullptr};
    std::vector<const RegionStmtNode *> sections_;
    OwnerMap owners_;
  };

  class SectionFlowAnalyzer final
      : public tl::ascend::AnalysisFunctor<VarSet(const Stmt &, VarSet)> {
  public:
    SectionFlowAnalyzer(const RegionStmtNode *section, const OwnerMap &owners)
        : section_(section), owners_(owners) {}

    VarSet Analyze() {
      VarSet defined;
      VisitStmt(section_->body, std::move(defined));
      return external_vars_;
    }

  private:
    class ReadCollector final : public tl::ascend::AnalysisVisitor {
    public:
      ReadCollector(SectionFlowAnalyzer *parent, const VarSet &defined)
          : parent_(parent), defined_(defined) {}

      void VisitExpr_(const TensorLoadNode *op) final {
        parent_->RecordAccess(op->source.as_or_throw<TensorVar>(), defined_);
        tl::ascend::AnalysisVisitor::VisitExpr_(op);
      }

    private:
      SectionFlowAnalyzer *parent_;
      const VarSet &defined_;
    };

    static VarSet Intersect(VarSet lhs, const VarSet &rhs) {
      for (auto it = lhs.begin(); it != lhs.end();) {
        if (!rhs.count(*it)) {
          it = lhs.erase(it);
        } else {
          ++it;
        }
      }
      return lhs;
    }

    void AnalyzeExpr(const Expr &expr, const VarSet &defined) {
      ReadCollector(this, defined)(expr);
    }

    void AnalyzeExprs(const ffi::Array<PrimExpr> &exprs,
                      const VarSet &defined) {
      for (const Expr &expr : exprs) {
        AnalyzeExpr(expr, defined);
      }
    }

    bool IsLocalVar(const TensorVar &buffer) const {
      return buffer.scope().name() == "local.var";
    }

    void RecordAccess(const TensorVar &buffer, const VarSet &defined) {
      if (!IsLocalVar(buffer)) {
        return;
      }
      const VarNode *var = TensorTypeOf(buffer).get();
      auto owner = owners_.find(var);
      ICHECK(owner != owners_.end())
          << "Mixed-section local.var access has no allocation: " << buffer;
      ICHECK(owner->second == nullptr || owner->second == section_)
          << "A mixed section cannot access local.var storage allocated by a "
             "sibling section: "
          << buffer;
      ICHECK(defined.count(var) || owner->second == nullptr)
          << "Mixed-section local.var is accessed before its section-local "
             "allocation: "
          << buffer;
      if (owner->second == nullptr) {
        external_vars_.insert(var);
      }
    }

    void DefineAllocation(const TensorVar &buffer, VarSet *defined) {
      if (IsLocalVar(buffer)) {
        defined->insert(TensorTypeOf(buffer).get());
      }
    }

    VarSet VisitStmt_(const BindNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      if (op->var.as<TensorVar>())
        DefineAllocation(op->var.as_or_throw<TensorVar>(), &defined);
      return defined;
    }

    VarSet VisitStmt_(const IfNode *op, VarSet defined) final {
      AnalyzeExpr(op->condition, defined);
      VarSet then_out = VisitStmt(op->then_case, defined);
      VarSet else_out = op->else_case.has_value()
                            ? VisitStmt(op->else_case.value(), defined)
                            : defined;
      return Intersect(std::move(then_out), else_out);
    }

    VarSet VisitStmt_(const ForNode *op, VarSet defined) final {
      AnalyzeExpr(op->begin, defined);
      AnalyzeExpr((op->end - op->begin), defined);
      if (true) {
        AnalyzeExpr(op->step, defined);
      }
      VarSet body_out = VisitStmt(op->body, defined);
      return sym::Analyzer()->CanProve((op->end - op->begin) > 0) ? body_out
                                                                  : defined;
    }

    VarSet VisitStmt_(const WhileNode *op, VarSet defined) final {
      AnalyzeExpr(op->condition, defined);
      VisitStmt(op->body, defined);
      return defined;
    }

    VarSet VisitStmt_(const TensorStoreNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      AnalyzeExprs(op->indices, defined);
      RecordAccess(op->dest.as_or_throw<TensorVar>(), defined);
      if (IsLocalVar(op->dest.as_or_throw<TensorVar>())) {
        defined.insert(op->dest.as_or_throw<TensorVar>().get());
      }
      return defined;
    }

    VarSet VisitStmt_(const AssertStmtNode *op, VarSet defined) final {
      AnalyzeExpr(op->condition, defined);
      return defined;
    }

    VarSet VisitStmt_(const SeqStmtNode *op, VarSet defined) final {
      for (const Stmt &stmt : op->seq) {
        defined = VisitStmt(stmt, std::move(defined));
      }
      return defined;
    }

    VarSet VisitStmt_(const EvaluateNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      return defined;
    }

    VarSet VisitStmt_(const ReturnNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      return defined;
    }

    VarSet VisitStmt_(const RegionStmtNode *op, VarSet defined) final {
      ICHECK(RegionName(op) != "CUBE" && RegionName(op) != "VECTOR")
          << "Nested CUBE/VECTOR physical sections are not supported";
      return VisitStmt(op->body, std::move(defined));
    }

    const RegionStmtNode *section_;
    const OwnerMap &owners_;
    VarSet external_vars_;
  };

  class OuterAllocationChecker final
      : public tl::ascend::AnalysisFunctor<VarSet(const Stmt &, VarSet)> {
  public:
    OuterAllocationChecker(
        const std::unordered_map<const RegionStmtNode *, VarSet> &section_vars,
        VarSet *captured_vars)
        : section_vars_(section_vars), captured_vars_(captured_vars) {}

  private:
    static VarSet Intersect(VarSet lhs, const VarSet &rhs) {
      for (auto it = lhs.begin(); it != lhs.end();) {
        if (!rhs.count(*it)) {
          it = lhs.erase(it);
        } else {
          ++it;
        }
      }
      return lhs;
    }

    static void DefineAllocation(const TensorVar &buffer, VarSet *defined) {
      if (buffer.scope().name() == "local.var") {
        defined->insert(TensorTypeOf(buffer).get());
      }
    }

    VarSet VisitStmt_(const BindNode *op, VarSet defined) final {
      if (op->var.as<TensorVar>())
        DefineAllocation(op->var.as_or_throw<TensorVar>(), &defined);
      return defined;
    }

    VarSet VisitStmt_(const IfNode *op, VarSet defined) final {
      VarSet then_out = VisitStmt(op->then_case, defined);
      VarSet else_out = op->else_case.has_value()
                            ? VisitStmt(op->else_case.value(), defined)
                            : defined;
      return Intersect(std::move(then_out), else_out);
    }

    VarSet VisitStmt_(const ForNode *op, VarSet defined) final {
      VarSet body_out = VisitStmt(op->body, defined);
      return sym::Analyzer()->CanProve((op->end - op->begin) > 0) ? body_out
                                                                  : defined;
    }

    VarSet VisitStmt_(const WhileNode *op, VarSet defined) final {
      VisitStmt(op->body, defined);
      return defined;
    }

    VarSet VisitStmt_(const TensorStoreNode *op, VarSet defined) final {
      (void)op;
      return defined;
    }

    VarSet VisitStmt_(const AssertStmtNode *op, VarSet defined) final {
      (void)op;
      return defined;
    }

    VarSet VisitStmt_(const SeqStmtNode *op, VarSet defined) final {
      for (const Stmt &stmt : op->seq) {
        defined = VisitStmt(stmt, std::move(defined));
      }
      return defined;
    }

    VarSet VisitStmt_(const EvaluateNode *op, VarSet defined) final {
      (void)op;
      return defined;
    }

    VarSet VisitStmt_(const ReturnNode *, VarSet defined) final {
      return defined;
    }

    VarSet VisitStmt_(const RegionStmtNode *op, VarSet defined) final {
      const bool is_physical =
          RegionName(op) == "CUBE" || RegionName(op) == "VECTOR";
      if (is_physical) {
        auto section_it = section_vars_.find(op);
        ICHECK(section_it != section_vars_.end());
        for (const VarNode *var : section_it->second) {
          ICHECK(defined.count(var))
              << "Mixed-section local.var access is not dominated by an "
                 "outer allocation: "
              << var->name;
          captured_vars_->insert(var);
        }
        return defined;
      }
      return VisitStmt(op->body, std::move(defined));
    }

    const std::unordered_map<const RegionStmtNode *, VarSet> &section_vars_;
    VarSet *captured_vars_;
  };

public:
  static MixedSectionVariableInfo Analyze(const Stmt &body) {
    StructureCollector collector;
    collector.Collect(body);

    MixedSectionVariableInfo result;
    for (const RegionStmtNode *section : collector.sections()) {
      result.external_vars_by_section.emplace(
          section, SectionFlowAnalyzer(section, collector.owners()).Analyze());
    }
    OuterAllocationChecker checker(result.external_vars_by_section,
                                   &result.captured_vars);
    checker(body, VarSet{});
    return result;
  }
};

bool UsesVar(const Stmt &body, const VarNode *target) {
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (node.get() == target) {
      found = true;
    }
  });
  return found;
}

struct CopyPadState {
  bool valid{false};
  bool changed{false};
  PrimType dtype = PrimType::Void();

  bool CanMerge(const CopyPadState &other) const {
    return valid && other.valid && dtype == other.dtype &&
           (changed || other.changed);
  }
};

class CopyPadStateAnalyzer : public tl::ascend::AnalysisVisitor {
public:
  explicit CopyPadStateAnalyzer(CopyPadState entry) : state_(entry) {}

  CopyPadState state() const { return state_; }

  void VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
      ICHECK_EQ(op->args.size(), 1U);
      state_ = CopyPadState{true, true, PythonDType(op->args[0])};
      return;
    }
    tl::ascend::AnalysisVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const IfNode *op) override {
    CopyPadState entry = state_;
    CopyPadStateAnalyzer then_analyzer(entry);
    then_analyzer(op->then_case);
    CopyPadState then_state = then_analyzer.state();

    CopyPadState else_state = entry;
    if (op->else_case.has_value()) {
      CopyPadStateAnalyzer else_analyzer(entry);
      else_analyzer(op->else_case.value());
      else_state = else_analyzer.state();
    }

    if (then_state.valid && else_state.valid &&
        then_state.dtype == else_state.dtype) {
      state_ = CopyPadState{
          true,
          then_state.changed || else_state.changed,
          then_state.dtype,
      };
    } else {
      state_ = CopyPadState{};
    }
  }

  // PTO codegen restores the incoming padding state after a loop because the
  // loop may execute zero times. Mirror that reaching-state rule here.
  void VisitStmt_(const ForNode *op) override { (void)op; }
  void VisitStmt_(const WhileNode *op) override { (void)op; }

private:
  CopyPadState state_;
};

CopyPadState AnalyzeCopyPadState(const Stmt &stmt, CopyPadState entry) {
  CopyPadStateAnalyzer analyzer(entry);
  analyzer(stmt);
  return analyzer.state();
}

// Adjacent complementary ifs are safe to pair only when neither condition
// observes mutable state between the two statements.
class StableConditionChecker : public tl::ascend::AnalysisVisitor {
public:
  bool stable{true};

  void VisitExpr_(const TensorLoadNode *op) override {
    (void)op;
    stable = false;
  }

  void VisitExpr_(const CallNode *op) override {
    (void)op;
    stable = false;
  }
};

bool IsStableCondition(const Expr &condition) {
  StableConditionChecker checker;
  checker(condition);
  return checker.stable;
}

bool AreComplementaryConditions(const Expr &lhs, const Expr &rhs) {
  auto same_operands = [](const Expr &lhs_a, const Expr &lhs_b,
                          const Expr &rhs_a, const Expr &rhs_b) {
    StructuralEqual equal;
    bool same = (equal(lhs_a, rhs_a) && equal(lhs_b, rhs_b)) ||
                (equal(lhs_a, rhs_b) && equal(lhs_b, rhs_a));
    return same && IsStableCondition(lhs_a) && IsStableCondition(lhs_b);
  };

  if (const auto *lhs_eq = lhs.as<EQNode>()) {
    if (const auto *rhs_ne = rhs.as<NENode>()) {
      return same_operands(lhs_eq->a, lhs_eq->b, rhs_ne->a, rhs_ne->b);
    }
  }
  if (const auto *lhs_ne = lhs.as<NENode>()) {
    if (const auto *rhs_eq = rhs.as<EQNode>()) {
      return same_operands(lhs_ne->a, lhs_ne->b, rhs_eq->a, rhs_eq->b);
    }
  }
  return false;
}
} // namespace

CodeGenTileLangPTO::CodeGenTileLangPTO() {
  auto pass_ctx = tvm::transform::PassContext::Current();
  enable_fast_math_ =
      pass_ctx->GetConfig<bool>(tl::kEnableFastMath).value_or(false);
}

void CodeGenTileLangPTO::ValidateVecMode_(const CallNode *op, size_t mode_idx,
                                          const char *expected) {
  ICHECK_LT(mode_idx, op->args.size())
      << "PTO SIMD op has no mode argument at index " << mode_idx;
  auto mode_str = (op->args[mode_idx]).as_or_throw<StringImm>()->value;
  ICHECK_EQ(mode_str, expected)
      << "PTO codegen currently only supports " << expected
      << " for this SIMD op, got " << mode_str;
}

bool CodeGenTileLangPTO::EmitSimdMergingCall_(const CallNode *op,
                                              std::ostream &os) { // NOLINT(*)
  std::string intrinsic_name;
  if (!IsSimdMergingCall(op, &intrinsic_name)) {
    return false;
  }

  ICHECK_GE(op->args.size(), 4U) << "Legalized PTO MODE_MERGING call must "
                                    "contain a destination, operands, "
                                    "mask, and mode";
  const auto *access_ptr = op->args[0].as<CallNode>();
  ICHECK(access_ptr && ((access_ptr->op.same_as(tl::access_ptr()) &&
                         access_ptr->args.size() == 3U) ||
                        (PythonIsOp(access_ptr->op, "address_of") &&
                         access_ptr->args.size() == 1U)))
      << "Legalized PTO MODE_MERGING call must carry a vector destination, "
         "got "
      << op->args[0];
  const auto *destination = access_ptr->args[0].as<TensorLoadNode>();
  ICHECK(destination)
      << "Legalized PTO MODE_MERGING destination must address a vector buffer";

  const bool is_rmw = op->op.same_as(tl::simd_vmula()) ||
                      op->op.same_as(tl::simd_vmadd()) ||
                      op->op.same_as(tl::simd_vaxpy());
  const bool is_binary =
      op->op.same_as(tl::simd_vadd()) || op->op.same_as(tl::simd_vsub()) ||
      op->op.same_as(tl::simd_vmul()) || op->op.same_as(tl::simd_vdiv()) ||
      op->op.same_as(tl::simd_vabsdif()) || op->op.same_as(tl::simd_vmax()) ||
      op->op.same_as(tl::simd_vmin()) || op->op.same_as(tl::simd_vand()) ||
      op->op.same_as(tl::simd_vor()) || op->op.same_as(tl::simd_vxor()) ||
      op->op.same_as(tl::simd_vshl()) || op->op.same_as(tl::simd_vshr());
  const bool is_vector_scalar =
      op->op.same_as(tl::simd_vadds()) || op->op.same_as(tl::simd_vmuls()) ||
      op->op.same_as(tl::simd_vmaxs()) || op->op.same_as(tl::simd_vmins()) ||
      op->op.same_as(tl::simd_vshls()) || op->op.same_as(tl::simd_vshrs());
  const bool is_unary =
      op->op.same_as(tl::simd_vabs()) || op->op.same_as(tl::simd_vexp()) ||
      op->op.same_as(tl::simd_vln()) || op->op.same_as(tl::simd_vsqrt()) ||
      op->op.same_as(tl::simd_vneg()) || op->op.same_as(tl::simd_vrelu()) ||
      op->op.same_as(tl::simd_vnot());
  const bool is_broadcast =
      op->op.same_as(tl::simd_vdup()) || op->op.same_as(tl::simd_vdupv());
  const bool is_reduction =
      op->op.same_as(tl::simd_vcpadd()) || op->op.same_as(tl::simd_vcadd()) ||
      op->op.same_as(tl::simd_vcmax()) || op->op.same_as(tl::simd_vcmin()) ||
      op->op.same_as(tl::simd_vcgadd()) || op->op.same_as(tl::simd_vcgmax()) ||
      op->op.same_as(tl::simd_vcgmin());
  const bool is_vcvt = op->op.same_as(tl::simd_vcvt());
  ICHECK(is_rmw || is_binary || is_vector_scalar || is_unary || is_broadcast ||
         is_reduction || is_vcvt)
      << "PTO codegen does not support software MODE_MERGING for tl.simd."
      << intrinsic_name;

  size_t mask_idx = 0;
  if (is_rmw || is_binary || is_vector_scalar) {
    mask_idx = 3;
  } else {
    // Unary, reduction, vdup/vdupv, and vcvt calls all carry the predicate
    // immediately after their first source operand once the destination is
    // prepended by LegalizeSimdMerging.
    mask_idx = 2;
  }
  ICHECK_LT(mask_idx, op->args.size() - 1)
      << "PTO MODE_MERGING call has no predicate operand";

  const PrimType result_dtype = PythonDType(GetRef<Expr>(destination));
  ICHECK_GT(result_dtype.lanes(), 1)
      << "PTO MODE_MERGING destination must be vector-typed, got "
      << result_dtype;
  std::string dst_ref = GetMutableVectorRef(op->args[0], intrinsic_name);
  std::string mask = PrintExpr_(op->args[mask_idx]);
  std::string result;

  if (is_rmw) {
    ICHECK_EQ(op->args.size(), 5U)
        << "PTO SIMD read-modify-write merging op expects 5 arguments "
           "(dst, src0, src1/scalar, mask, mode)";
    std::ostringstream result_os;
    result_os << "pto." << intrinsic_name << "(";
    if (op->op.same_as(tl::simd_vaxpy())) {
      // PTODSL spells vaxpy as alpha * x + y; the mutable destination is y.
      result_os << PrintExpr_(op->args[2]) << ", " << PrintExpr_(op->args[1])
                << ", " << dst_ref;
    } else {
      result_os << dst_ref << ", " << PrintExpr_(op->args[1]) << ", "
                << PrintExpr_(op->args[2]);
    }
    result_os << ", " << mask << ")";
    result = result_os.str();
  } else {
    ffi::Array<Expr> zeroing_args;
    zeroing_args.reserve(op->args.size() - 1);
    for (size_t i = 1; i < op->args.size(); ++i) {
      zeroing_args.push_back(i + 1 == op->args.size()
                                 ? Expr(StringImm(kModeZeroing))
                                 : op->args[i]);
    }
    Call zeroing_call(result_dtype, op->op, std::move(zeroing_args), op->attrs,
                      op->ty_args, op->loc);
    result = PrintExpr_(zeroing_call);
    ICHECK(!result.empty()) << "PTO zeroing form for tl.simd." << intrinsic_name
                            << " did not produce a vector result";
  }

  PrimType result_elem = result_dtype.WithLanes(1);
  int result_bits = result_elem.bits();
  if (result_bits < 8) {
    result_bits = 8;
  }
  const std::string input_mask = mask;
  // Reduction predicates select source lanes. Their results occupy a fixed
  // low-lane layout, so merge those result lanes rather than the source mask.
  if (op->op.same_as(tl::simd_vcadd())) {
    mask = "pto.pset_b" + std::to_string(result_bits) + "(\"PAT_VL1\")";
  } else if (op->op.same_as(tl::simd_vcmax()) ||
             op->op.same_as(tl::simd_vcmin())) {
    mask = "pto.pset_b" + std::to_string(result_bits) + "(\"PAT_VL2\")";
  } else if (op->op.same_as(tl::simd_vcgadd()) ||
             op->op.same_as(tl::simd_vcgmax()) ||
             op->op.same_as(tl::simd_vcgmin())) {
    mask = "pto.pset_b" + std::to_string(result_bits) + "(\"PAT_VL8\")";
  } else if (op->op.same_as(tl::simd_vcpadd())) {
    // Pairwise adjacent-lane sums pack into the low half of the result
    // register. Write that result region unconditionally (pairs with no
    // active lanes contribute 0) and preserve the old high half.
    int half_lanes = result_dtype.lanes() / 2;
    ICHECK_GE(half_lanes, 1)
        << "PTO vcpadd merging requires a multi-lane destination, got "
        << result_dtype;
    ICHECK_LE(half_lanes, 128)
        << "PTO vcpadd merging expects at most 256 result lanes, got "
        << result_dtype;
    mask = "pto.pset_b" + std::to_string(result_bits) + "(\"PAT_VL" +
           std::to_string(half_lanes) + "\")";
  }
  std::string selection_mask = mask;
  if (is_vcvt) {
    // Conversion changes the element width. Reinterpret the predicate at the
    // destination granularity before selecting the converted result.
    selection_mask = "pto.pbitcast(" + input_mask + ", pto.mask_type(\"b" +
                     std::to_string(result_bits) + "\"))";
  }

  if (tl::IsAscendVectorizableFP8(result_elem) ||
      result_elem.MatchesCode(kDLFloat4_e2m1fn)) {
    // pto.vsel intentionally rejects low-precision arithmetic payloads. Select
    // their byte representation, then restore the authored result element type.
    result = "pto.vbitcast(pto.vsel(pto.vbitcast(" + result +
             ", pto.ui8), pto.vbitcast(" + dst_ref + ", pto.ui8), " +
             selection_mask + "), " + DataTypeName(result_elem) + ")";
  } else {
    result =
        "pto.vsel(" + result + ", " + dst_ref + ", " + selection_mask + ")";
  }
  os << dst_ref << " = " << result;
  return true;
}

void CodeGenTileLangPTO::ResetRngState_() {
  pto_rng_state_var_.clear();
  pto_rng_counter_var_.clear();
  pto_rng_normal_cache_var_.clear();
  pto_rng_has_normal_var_.clear();
  pto_rng_initialized_ = false;
  pto_rng_result_counter_ = 0;
}

void CodeGenTileLangPTO::ResetFunctionState_() {
  integer_address_vars_.clear();
  local_var_buffers_.clear();
  mixed_captured_local_vars_.clear();
  mixed_external_vars_by_section_.clear();
  mixed_entry_snapshot_ids_.clear();
  loop_condition_snapshot_ids_.clear();
  simd_pair_vars_.clear();
  gemm_emit_contexts_.clear();
  gemm_emit_context_by_call_.clear();
  gemm_zero_addr_emitted_ = false;
  inside_simtvf_body_ = false;
  in_mixed_vector_section_ = false;
  inside_mixed_section_ = false;
  copy_pad_value_counter_ = 0;
  current_copy_pad_value_id_ = -1;
  current_copy_pad_value_dtype_ = PrimType::Void();
  uniform_const_copy_pad_value_ = std::nullopt;
  native_loop_depth_ = 0;
  simdvf_nesting_depth_ = 0;
  pto_if_result_counter_ = 0;
  ResetRngState_();
}

void CodeGenTileLangPTO::AddFunction(const GlobalVar &gvar,
                                     const tirx::Function &func) {
  RegisterFunction_(gvar, func);
  current_function_name_ = GetFunctionName_(gvar);
  InitFuncState_(func);
  ResetFunctionState_();
  name_supply_->ReserveName("pto");
  name_supply_->ReserveName("scalar");
  name_supply_->ReserveName("tl");
  ValidateKernelCapabilities(func);
  persistent_buffer_vars_ =
      SimtPersistentBufferCollector().Collect(func->body.value());
  hf32_mode_by_gemm_ = PTOHf32ModeAnalyzer::Analyze(func);
  n_direction_by_mad_ = PTOMmadDirectionAnalyzer::Analyze(func);
  bool saw_copy_pad_value = false;
  bool has_uniform_const_copy_pad_value = true;
  tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr || !call->op.same_as(tl::ascend_set_copy_pad_value())) {
      return;
    }
    ICHECK_EQ(call->args.size(), 1U);
    PrimExpr value = call->args[0].as_or_throw<PrimExpr>();
    bool is_const = value.as<IntImmNode>() != nullptr ||
                    value.as<FloatImmNode>() != nullptr;
    if (!is_const) {
      has_uniform_const_copy_pad_value = false;
    } else if (!saw_copy_pad_value) {
      uniform_const_copy_pad_value_ = value;
    } else if (!ffi::StructuralEqual()(uniform_const_copy_pad_value_, value)) {
      has_uniform_const_copy_pad_value = false;
    }
    saw_copy_pad_value = true;
  });
  if (!saw_copy_pad_value || !has_uniform_const_copy_pad_value) {
    uniform_const_copy_pad_value_ = std::nullopt;
  }
  bool func_has_gemm_l1 = HasAscendGemmL1(func);
  bool func_has_blockscaled_gemm_l1 = HasAscendBlockscaledGemmL1(func);
  current_function_has_gemm_ =
      func_has_gemm_l1 || func_has_blockscaled_gemm_l1 || HasAscendMad(func);
  current_function_is_cube_ = IsAscendCubeKernel(func);
  current_function_is_mixed_ = IsAscendMixedKernel(func);
  if (current_function_is_mixed_) {
    MixedSectionVariableInfo mixed_info =
        MixedSectionLocalVarAnalyzer::Analyze(func->body.value());
    mixed_captured_local_vars_ = std::move(mixed_info.captured_vars);
    mixed_external_vars_by_section_ =
        std::move(mixed_info.external_vars_by_section);
  }

  PrintFuncDecorator_(stream);
  PrintFunctionSignature_(current_function_name_, func, stream);
  stream << ":\n";
  int func_scope = BeginScope();
  if (!current_function_is_mixed_) {
    PrintIndent();
    stream << "pto.init_core()\n";
  }
  if (func_has_gemm_l1 || func_has_blockscaled_gemm_l1) {
    std::vector<Call> gemm_calls;
    tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tl::ascend_gemm_l1()) ||
            call->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
          gemm_calls.push_back(GetRef<Call>(call));
        }
      }
    });
    ICHECK(!gemm_calls.empty());
    // Emit every helper while still at function scope. A configuration first
    // encountered inside control flow must not construct its helper there.
    for (const Call &call : gemm_calls) {
      if (call->op.same_as(tl::ascend_gemm_l1())) {
        EnsureGemmHelper(call.get());
      } else {
        EnsureBlockscaledGemmHelper(call.get());
      }
    }
  }
  PrintStmt_(func->body.value());
  EndScope(func_scope);
  stream << "\n";
  if (current_function_is_mixed_) {
    stream << current_function_name_ << " = tl.finalize_mixed_kernel("
           << current_function_name_ << ")\n\n";
  }
}

std::string CodeGenTileLangPTO::Finish() {
  std::ostringstream code;
  code << "from ptodsl import pto\n";
  code << "import tilelang.contrib.ptodsl as tl\n";
  code << "\n";
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangPTO::PrintFuncDecorator_(std::ostream &os) { // NOLINT(*)
  os << "@pto.jit(name=\"" << current_function_name_ << "\"";
  if (!current_function_is_mixed_) {
    os << ", kernel_kind=\"" << (current_function_is_cube_ ? "cube" : "vector")
       << "\"";
  }
  os << ", target=\"a5\", mode=\"explicit\"";
  if (current_function_has_gemm_ && !current_function_is_mixed_) {
    os << ", insert_sync=False";
  } else if (current_function_is_mixed_) {
    os << ", insert_sync=True";
  }
  os << ")\n";
}

void CodeGenTileLangPTO::PrintFunctionSignature_(
    const ffi::String &function_name, const tirx::Function &func,
    std::ostream &os) { // NOLINT(*)
  os << "def " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    Var v = func->params[i];
    if (i > 0) {
      os << ", ";
    }
    os << AllocVarID(v.get());
    if (auto tensor = v.as<TensorVar>()) {
      TensorVar buffer = tensor.value();
      os << ": " << PointerTypeName(TensorTypeOf(buffer).dtype(), "gm");
    } else if (auto *ptr = v->ty.as<PtrTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        std::string scope = ptr->scope.name() == "global"
                                ? std::string("gm")
                                : std::string(ptr->scope.name());
        if (scope == "global") {
          scope = "gm";
        }
        os << ": " << PointerTypeName(GetRef<PrimType>(prim), scope);
      } else {
        os << ": " << ScalarType(PythonDType(v));
      }
    } else {
      os << ": " << ScalarType(PythonDType(v));
    }
  }
  os << ")";

  for (const auto &param : func->params) {
    if (auto *ptr = param->ty.as<PtrTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType_(param.get(), GetRef<PrimType>(prim));
      }
    }
  }
}

std::string CodeGenTileLangPTO::ScalarType(PrimType t) const {
  return DataTypeName(t);
}

std::string
CodeGenTileLangPTO::PointerTypeName(PrimType t,
                                    const std::string &space) const {
  // TIR bool is an i1 scalar predicate, but buffers store bool values in
  // byte-addressable memory. FP4 buffers are rewritten to packed pairs
  // (float4_e2m1fnx2), whose PTO pointer element is one physical byte.
  const std::string elem_type = PointerElementTypeName(t);
  std::ostringstream os;
  os << "pto.ptr(" << elem_type << ", \"" << space << "\")";
  return os.str();
}

bool CodeGenTileLangPTO::NeedsCastptr_(const VarNode *buffer_var,
                                       PrimType elem_dtype) const {
  if (integer_address_vars_.count(GetRef<Var>(buffer_var))) {
    return true;
  }
  // Compare physical PTO element types. TVM bool buffers use int8 backing
  // storage even though scalar bool expressions use i1.
  PrimType storage_dtype =
      elem_dtype.MatchesCode(kDLBool) ? PrimType::Int(8) : elem_dtype;
  if (HandleTypeMatch_(buffer_var, storage_dtype)) {
    return false;
  }
  if (IsPackedFp4Pair(storage_dtype) &&
      HandleTypeMatch_(buffer_var, storage_dtype.WithLanes(1))) {
    return false;
  }
  if (storage_dtype.MatchesCode(kDLFloat4_e2m1fn) &&
      storage_dtype.lanes() == 1 &&
      HandleTypeMatch_(buffer_var, storage_dtype.WithLanes(2))) {
    return false;
  }
  // Signedness is part of the authored PTO pointer type. Do not treat
  // int/uint aliases as interchangeable: pto.load derives signedness from
  // the pointer element type, which later selects signed or unsigned scalar
  // ops.
  if (storage_dtype.lanes() == 1 && storage_dtype.bits() >= 8 &&
      storage_dtype.MatchesCode(kDLInt) &&
      HandleTypeMatch_(buffer_var, PrimType::Bool())) {
    return false;
  }
  return true;
}

std::string CodeGenTileLangPTO::ScalarPointerBase_(const VarNode *buffer_var,
                                                   PrimType elem_dtype,
                                                   const std::string &scope) {
  std::string base = GetVarID(buffer_var);
  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    if (NeedsCastptr_(buffer_var, elem_dtype)) {
      std::string pto_space =
          (scope == "shared" || scope == "shared.dyn") ? "ub" : "gm";
      base = "pto.castptr(" + base + ", " +
             PointerTypeName(elem_dtype, pto_space) + ")";
    }
  }
  return base;
}

std::pair<std::string, std::string>
CodeGenTileLangPTO::ParseHardEventPair(const std::string &hard_event) const {
  auto pos = hard_event.find('_');
  if (pos == std::string::npos) {
    return {hard_event, hard_event};
  }
  return {hard_event.substr(0, pos), hard_event.substr(pos + 1)};
}

bool CodeGenTileLangPTO::HasAscendGemmL1(const tirx::Function &func) const {
  bool found = false;
  tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_gemm_l1());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::HasAscendBlockscaledGemmL1(
    const tirx::Function &func) const {
  bool found = false;
  tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_blockscaled_gemm_l1());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::IsAscendCubeKernel(const tirx::Function &func) const {
  bool found = false;
  tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *block = node.as<RegionStmtNode>()) {
      found = RegionName(block) == "CUBE";
    }
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_gemm_l1()) ||
              call->op.same_as(tl::ascend_blockscaled_gemm_l1()) ||
              call->op.same_as(tl::ascend_mad()) ||
              call->op.same_as(tl::ascend_mad_mx()) ||
              call->op.same_as(tl::ascend_copy_gm_to_cbuf()) ||
              call->op.same_as(tl::ascend_fill_l1()) ||
              call->op.same_as(tl::ascend_load_cbuf_to_ca()) ||
              call->op.same_as(tl::ascend_load_cbuf_to_cb()) ||
              call->op.same_as(tl::ascend_load_ca_sf()) ||
              call->op.same_as(tl::ascend_load_cb_sf()) ||
              call->op.same_as(tl::ascend_copy_ubuf_to_cbuf()) ||
              call->op.same_as(tl::ascend_copy_matrix_cc_to_ub()) ||
              call->op.same_as(tl::ascend_copy_matrix_cc_to_gm());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::HasAscendMad(const tirx::Function &func) const {
  bool found = false;
  tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_mad()) ||
              call->op.same_as(tl::ascend_mad_mx());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::IsAscendMixedKernel(const tirx::Function &func) const {
  bool has_cube = false;
  bool has_vector = false;
  tl::PostOrderVisit(func->body.value(), [&](const ObjectRef &node) {
    if (const auto *block = node.as<RegionStmtNode>()) {
      has_cube = has_cube || RegionName(block) == "CUBE";
      has_vector = has_vector || RegionName(block) == "VECTOR";
    }
  });
  return has_cube && has_vector;
}

bool CodeGenTileLangPTO::IsInlineableInvariantSimdBind(
    const Expr &value) const {
  const auto *call = value.as<CallNode>();
  if (call == nullptr) {
    return false;
  }
  // Predicate masks are pure; inlining avoids mask SSA escaping across
  // range()/scf.for (e.g. SwiGLU backward one_mask used after the reduce loop).
  if (call->op.same_as(tl::simd_pset())) {
    return true;
  }
  // Broadcasts of immediates are rematerializable.
  if (!(call->op.same_as(tl::simd_vdup()) ||
        call->op.same_as(tl::simd_vdupv()))) {
    return false;
  }
  if (call->args.empty()) {
    return false;
  }
  PrimExpr scalar = PeelScalarCasts(call->args[0]);
  if (IsImmediateScalar(scalar)) {
    return true;
  }
  return false;
}

std::string CodeGenTileLangPTO::ScopeOfBuffer(const VarNode *buffer) const {
  std::string scope;
  auto it = alloc_storage_scope_.find(TensorTypeOf(buffer).get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = PythonStorageScope(TensorTypeOf(buffer));
  }
  return scope;
}

void CodeGenTileLangPTO::ExtractSimtThreadExtents(const RegionStmtNode *op,
                                                  int64_t *thread_x,
                                                  int64_t *thread_y,
                                                  int64_t *thread_z) const {
  *thread_x = *thread_y = *thread_z = 1;
  tl::PostOrderVisit(op->body, [&](const ffi::ObjectRef &node) {
    auto region = node.as<RegionStmtNode>();
    if (!region || !region->op.same_as(tirx::launch_thread_op()))
      return;
    auto tag = region->args[0].as_or_throw<StringImm>()->value;
    auto extent = tl::GetConstInt(region->args[1]);
    ICHECK(extent) << "PTO SIMT thread extent must be constant";
    if (tag == "threadIdx.x")
      *thread_x = *extent;
    else if (tag == "threadIdx.y")
      *thread_y = *extent;
    else if (tag == "threadIdx.z")
      *thread_z = *extent;
  });
}

void CodeGenTileLangPTO::EmitInlineSimtVF(const RegionStmtNode *op,
                                          int64_t thread_x, int64_t thread_y,
                                          int64_t thread_z) {
  constexpr int64_t kMaxSimtWorkitems = 2048;
  ICHECK_LE(thread_x, kMaxSimtWorkitems)
      << "PTO inline SIMT launch has more than " << kMaxSimtWorkitems
      << " workitems";
  ICHECK_LE(thread_y, kMaxSimtWorkitems / thread_x)
      << "PTO inline SIMT launch has more than " << kMaxSimtWorkitems
      << " workitems";
  ICHECK_LE(thread_z, kMaxSimtWorkitems / (thread_x * thread_y))
      << "PTO inline SIMT launch has more than " << kMaxSimtWorkitems
      << " workitems";

  PrintIndent();
  stream << "with pto.simt(" << thread_x << ", " << thread_y << ", " << thread_z
         << "):\n";
  int simt_scope = BeginScope();
  const auto body_start = stream.tellp();
  bool saved_inside_simtvf_body = inside_simtvf_body_;
  inside_simtvf_body_ = true;

  // Older TIR forms keep section-local allocations on the SBlock. Current
  // lowering emits them as AllocBuffer statements in the body.
  PrintStmt_(op->body);
  if (stream.tellp() == body_start) {
    PrintIndent();
    stream << "pass\n";
  }

  inside_simtvf_body_ = saved_inside_simtvf_body;
  EndScope(simt_scope);
}

std::string CodeGenTileLangPTO::GetPointerExpr(const VarNode *buffer_var,
                                               PrimType elem_dtype,
                                               const Expr &index) {
  std::string scope = "global";
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }

  std::string base = GetVarID(buffer_var);
  if (IsBufferInScope(scope, "shared.l1") ||
      IsBufferInScope(scope, "shared.l0a") ||
      IsBufferInScope(scope, "shared.l0b") ||
      IsBufferInScope(scope, "shared.l0c")) {
    if (IsZero(index.as_or_throw<PrimExpr>()))
      return base;
    return "pto.add(" + base + ", " +
           GetLocalByteAddrExpr(index, elem_dtype, "PTO native local address") +
           ")";
  }
  // Cast the handle to the access dtype when they differ (e.g. a raw-byte
  // shared container accessed as f32, or a GM buffer reinterpreted through
  // T.view). PTO MLIR copy ops require src/dst element byte widths to match,
  // mirroring AscendC's (__gm__/__ubuf__ T*) casts.
  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    if (NeedsCastptr_(buffer_var, elem_dtype)) {
      std::string pto_space =
          (scope == "global" || scope.empty()) ? "gm" : "ub";
      base = "pto.castptr(" + base + ", " +
             PointerTypeName(elem_dtype, pto_space) + ")";
    }
  }

  if (IsZero(index.as_or_throw<PrimExpr>())) {
    return base;
  }

  PrimExpr ptr_index = index.as_or_throw<PrimExpr>();
  if (const auto *ramp = index.as<RampNode>()) {
    CheckContiguousRampStride(index, "pointer address");
    ptr_index = sym::Analyzer()->Simplify(ramp->base);
  }

  std::string index_str;
  int64_t const_index = 0;
  const bool is_const_index = TryGetConstInt(ptr_index, &const_index);
  if (is_const_index) {
    index_str = std::to_string(const_index);
  } else {
    index_str = RemoveOutermostParentheses(PrintExpr_(ptr_index));
  }
  int64_t physical_bits = PhysicalElementBits(elem_dtype);
  if (physical_bits < 8) {
    ICHECK_EQ(8 % physical_bits, 0)
        << "PTO packed pointer dtype bit width must divide one byte, got "
        << elem_dtype;
    int pack_factor = 8 / physical_bits;
    if (is_const_index) {
      ICHECK_EQ(const_index % pack_factor, 0)
          << "PTO packed pointer offset must be aligned to packed storage "
          << "elements, got logical offset " << const_index << " for "
          << elem_dtype;
      index_str = std::to_string(const_index / pack_factor);
    } else {
      index_str =
          "((" + index_str + ") // " + std::to_string(pack_factor) + ")";
    }
  }

  if (scope == "global" || scope.empty() || scope == "shared" ||
      scope == "shared.dyn") {
    return "pto.addptr(" + base + ", " + index_str + ")";
  }

  LOG(FATAL) << "Unsupported storage scope in PTO pointer emission: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::GetPointerExpr(const VarNode *buffer,
                                               const Expr &index) {
  return GetPointerExpr(TensorTypeOf(buffer).get(),
                        TensorTypeOf(buffer).dtype(), index);
}

std::string CodeGenTileLangPTO::GetVectorLocalRef(const VarNode *buffer_var,
                                                  const Expr &index,
                                                  const std::string &context) {
  std::string name = GetVarID(buffer_var);
  PrimExpr elem_index = VectorBufferElemIndex(index);
  if (inside_simtvf_body_) {
    return name + "[" + RemoveOutermostParentheses(PrintExpr_(elem_index)) +
           "]";
  }
  int64_t slot = 0;
  ICHECK(TryGetConstInt(elem_index, &slot))
      << context
      << " requires a constant local.fragment slot outside SIMT, got "
      << elem_index;
  return name + "_tl_slot_" + std::to_string(slot);
}

std::string
CodeGenTileLangPTO::GetMutableVectorRef(const Expr &address,
                                        const std::string &op_name) {
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(address, &dst_index, &buffer_var))
      << "PTO " << op_name
      << " expects address_of/tvm_access_ptr as its destination, got "
      << address;

  auto scope_it = alloc_storage_scope_.find(buffer_var);
  std::string scope;
  if (scope_it != alloc_storage_scope_.end()) {
    scope = scope_it->second;
  } else {
    scope = PythonStorageScope(GetRef<Var>(buffer_var));
  }
  if (scope == "local.var") {
    return GetVarID(buffer_var);
  }
  ICHECK(scope == "local" || scope == "local.fragment")
      << "PTO " << op_name << " destination must be a local vector value, got "
      << scope;
  return GetVectorLocalRef(buffer_var, dst_index, "PTO " + op_name);
}

std::string CodeGenTileLangPTO::PrintCondition(const Expr &condition) {
  PrimType dtype = PythonDType(condition);
  ICHECK(dtype.IsScalar() &&
         (dtype.MatchesCode(kDLBool) || dtype.MatchesCode(kDLInt) ||
          dtype.MatchesCode(kDLUInt)))
      << "PTO control-flow condition must be a scalar integer predicate, got "
      << dtype;
  std::string value = RemoveOutermostParentheses(PrintExpr_(condition));
  // TIR follows C truthiness for all integer conditions.  Always compare at
  // the use site because a bool-typed expression may be materialized in an
  // i8 local.var by earlier lowering.  This also avoids truncating arbitrary
  // nonzero integers to i1 (which would make even values false).
  return "tl.as_logical_bool(" + value + ")";
}

std::string CodeGenTileLangPTO::ScalarLoad(const VarNode *buffer,
                                           const Expr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  if (scope == "local.var") {
    return GetVarID(TensorTypeOf(buffer).get());
  }

  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));
  if (inside_simtvf_body_ && (scope == "local.fragment" || scope == "local")) {
    ICHECK(IsSupportedSIMTLocalScalarAccessType(TensorTypeOf(buffer).dtype()))
        << "PTO SIMT local scalar load supports float16, float32, bfloat16, "
           "int8, uint8, int32, uint32, and FP8 only, got "
        << TensorTypeOf(buffer).dtype();
    std::string value = "pto.load(" + GetVarID(TensorTypeOf(buffer).get()) +
                        ", " + index_str + ")";
    if (tl::IsAscendVectorizableFP8(TensorTypeOf(buffer).dtype())) {
      return "tl.fp8_byte_load(" + GetVarID(TensorTypeOf(buffer).get()) + ", " +
             index_str + ", " + FP8TypeName(TensorTypeOf(buffer).dtype()) + ")";
    }
    if (IsInteger32(TensorTypeOf(buffer).dtype())) {
      return "pto.cast(" + value + ", " +
             DataTypeName(TensorTypeOf(buffer).dtype()) + ")";
    }
    return value;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    std::string base = ScalarPointerBase_(TensorTypeOf(buffer).get(),
                                          TensorTypeOf(buffer).dtype(), scope);
    if (inside_simtvf_body_ &&
        tl::IsAscendVectorizableFP8(TensorTypeOf(buffer).dtype())) {
      return "tl.fp8_byte_load(" + base + ", " + index_str + ", " +
             FP8TypeName(TensorTypeOf(buffer).dtype()) + ")";
    }
    return "pto.load(" + base + ", " + index_str + ")";
  }

  if (scope == "local.fragment" || scope == "local") {
    return GetVarID(TensorTypeOf(buffer).get()) + "[" + index_str + "]";
  }

  LOG(FATAL) << "Unsupported PTO scalar load scope: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::LocalVarStoreValue(const Expr &value,
                                                   PrimType dtype) {
  std::string value_expr = RemoveOutermostParentheses(PrintExpr_(value));
  if (dtype.MatchesCode(kDLOpaqueHandle)) {
    return value_expr;
  }
  bool is_literal =
      value.as<IntImmNode>() != nullptr || value.as<FloatImmNode>() != nullptr;
  if (dtype.IsScalar() && is_literal) {
    return "pto.const(" + value_expr + ", dtype=" + DataTypeName(dtype) + ")";
  }
  if (dtype.MatchesCode(kDLBool) || dtype.bits() == 1 ||
      dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) {
    return ScalarCastExpr(value_expr, dtype, "PTO local.var store");
  }
  return value_expr;
}

void CodeGenTileLangPTO::EmitScalarStore(const VarNode *buffer,
                                         const std::string &value,
                                         const Expr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));

  if (scope == "local.var") {
    stream << GetVarID(TensorTypeOf(buffer).get()) << " = " << value << "\n";
    EmitMixedEntrySnapshot(TensorTypeOf(buffer).get());
    return;
  }

  if (scope == "local.fragment" || scope == "local") {
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalScalarAccessType(TensorTypeOf(buffer).dtype()))
          << "PTO SIMT local scalar store supports float16, bfloat16, "
             "float32, int8, uint8, int32, uint32, and FP8 only, got "
          << TensorTypeOf(buffer).dtype();
      std::string store_value = value;
      if (tl::IsAscendVectorizableFP8(TensorTypeOf(buffer).dtype())) {
        stream << "tl.fp8_byte_store(" << store_value << ", "
               << GetVarID(TensorTypeOf(buffer).get()) << ", " << index_str
               << ", " << FP8TypeName(TensorTypeOf(buffer).dtype()) << ")\n";
        return;
      }
      if (IsInteger32(TensorTypeOf(buffer).dtype())) {
        store_value = "pto.cast(" + store_value + ", pto.i32)";
      }
      stream << "pto.store(" << store_value << ", "
             << GetVarID(TensorTypeOf(buffer).get()) << ", " << index_str
             << ")\n";
      return;
    }
    stream << GetVarID(TensorTypeOf(buffer).get()) << "[" << index_str
           << "] = " << value << "\n";
    return;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    std::string base = ScalarPointerBase_(TensorTypeOf(buffer).get(),
                                          TensorTypeOf(buffer).dtype(), scope);
    if (inside_simtvf_body_ &&
        tl::IsAscendVectorizableFP8(TensorTypeOf(buffer).dtype())) {
      stream << "tl.fp8_byte_store(" << value << ", " << base << ", "
             << index_str << ", " << FP8TypeName(TensorTypeOf(buffer).dtype())
             << ")\n";
      return;
    }
    stream << "pto.store(" << value << ", " << base << ", " << index_str
           << ")\n";
    return;
  }

  LOG(FATAL) << "Unsupported PTO scalar store scope: " << scope;
}

void CodeGenTileLangPTO::EmitLocalVarInitialization_(const TensorVar &buffer,
                                                     std::string vid) {
  PrimType dtype = TensorTypeOf(buffer).dtype();
  if (vid.rfind("__cond_", 0) == 0) {
    vid = "tl_cond_" + vid.substr(7);
    var_idmap_[TensorTypeOf(buffer).get()] = vid;
  } else if (!vid.empty() && vid.front() == '_') {
    // PTODSL rewrites native Python if statements through BranchHandle,
    // whose underscore-prefixed attributes are reserved for internals.
    // Give local.var values such as `_tmp_id` a unique public name so they
    // can be carried out of nested branches.
    vid = name_supply_->FreshName("tl" + vid, false);
    var_idmap_[TensorTypeOf(buffer).get()] = vid;
  }
  if (current_function_has_gemm_) {
    // Keep authored PTO surface values (not Python ints) so AST-rewritten
    // `if` / br.assign can merge tile indices across branches.
    if (IsVectorLocalVarDtype(dtype) || dtype.MatchesCode(kDLOpaqueHandle)) {
      stream << vid << " = None\n";
    } else {
      stream << vid << " = pto.const("
             << (dtype.MatchesCode(kDLFloat) ? "0.0" : "0")
             << ", dtype=" << DataTypeName(dtype) << ")\n";
    }
    RegisterHandleType_(TensorTypeOf(buffer).get(),
                        TensorTypeOf(buffer).dtype());
    EmitMixedEntrySnapshot(TensorTypeOf(buffer).get());
    return;
  }
  // Wide vector SSA; scalar values stay as PTO surface values.
  if (IsVectorLocalVarDtype(dtype) || dtype.MatchesCode(kDLOpaqueHandle)) {
    stream << vid << " = None\n";
  } else if (IsSurfaceScalarLocalVarDtype(dtype)) {
    stream << vid << " = pto.const("
           << (dtype.MatchesCode(kDLFloat) ? "0.0" : "0")
           << ", dtype=" << DataTypeName(dtype) << ")\n";
  } else {
    LOG(FATAL) << "Unsupported PTO local.var dtype: " << dtype;
  }
  EmitMixedEntrySnapshot(TensorTypeOf(buffer).get());
  RegisterHandleType_(TensorTypeOf(buffer).get(), TensorTypeOf(buffer).dtype());
}

void CodeGenTileLangPTO::EmitBufferAllocation(const TensorVar &buffer) {
  std::string scope = PythonStorageScope(TensorTypeOf(buffer));
  alloc_storage_scope_[TensorTypeOf(buffer).get()] = scope;
  ValidateSharedScope(scope);

  if (scope == "shared.dyn") {
    PrintIndent();
    std::string vid = AllocVarID(TensorTypeOf(buffer).get());
    // MergeUBAllocations assigns offsets when multiple buffers share the UB
    // arena. The generated PTODSL pointer does not require the logical extent
    // to be a constant.
    stream << vid << " = pto.castptr(pto.const(0, dtype=pto.i64), "
           << PointerTypeName(TensorTypeOf(buffer).dtype(), "ub") << ")\n";
    RegisterHandleType_(TensorTypeOf(buffer).get(),
                        TensorTypeOf(buffer).dtype());
    return;
  } else if (scope == "local.fragment" || scope == "local") {
    // Allocations nested in a runtime loop keep the same TIR Var identity and
    // are re-initialized on each iteration.  The loop/carry analysis may also
    // reserve the name before reaching this statement.  Reuse the stable
    // Python name while still emitting the initialization at this location.
    std::string vid = var_idmap_.count(TensorTypeOf(buffer).get())
                          ? GetVarID(TensorTypeOf(buffer).get())
                          : AllocVarID(TensorTypeOf(buffer).get());
    auto opt_size = tl::GetConstInt(PythonTensorSize(buffer));
    ICHECK(opt_size.has_value())
        << "PTO local.fragment currently requires constant allocation size";
    PrintIndent();
    const bool persistent =
        persistent_buffer_vars_.count(TensorTypeOf(buffer).get()) != 0;
    PrimType elem = TensorTypeOf(buffer).dtype().IsFixedLengthVector()
                        ? TensorTypeOf(buffer).dtype().WithLanes(1)
                        : TensorTypeOf(buffer).dtype();
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalStorageType(TensorTypeOf(buffer).dtype()))
          << "PTO SIMT local allocation supports float16, bfloat16, "
             "float32, int8, uint8, int32, uint32, and "
             "float8_e4m3fn/float8_e5m2, got "
          << TensorTypeOf(buffer).dtype();
      stream << vid << " = pto.alloc_buffer((" << opt_size.value() << ",), "
             << (TensorTypeOf(buffer).dtype().IsFixedLengthVector()
                     ? (tl::IsAscendVectorizableFP8(
                            TensorTypeOf(buffer).dtype())
                            ? DataTypeName(TensorTypeOf(buffer).dtype())
                            : ScalarType(elem))
                     : SIMTLocalStorageTypeName(TensorTypeOf(buffer).dtype()))
             << ")\n";
    } else if (persistent) {
      ICHECK(IsFloat32(TensorTypeOf(buffer).dtype()))
          << "PTO persistent local allocation currently supports float32 "
             "only, got "
          << TensorTypeOf(buffer).dtype();
      stream << vid << " = pto.alloc_buffer((" << opt_size.value() << ",), "
             << ScalarType(elem) << ")\n";
    } else if (TensorTypeOf(buffer).dtype().IsFixedLengthVector()) {
      // Keep vector-register slots as independent Python names.  PTOAS's AST
      // liveness analysis tracks Name stores as scf.for iter_args, but cannot
      // treat list subscript assignments as loop-carried SSA values.
      for (int64_t i = 0; i < opt_size.value(); ++i) {
        stream << vid << "_tl_slot_" << i << " = None\n";
        if (i + 1 < opt_size.value()) {
          PrintIndent();
        }
      }
    } else {
      stream << vid << " = [None] * " << opt_size.value() << "\n";
    }
  } else if (scope == "local.var") {
    PrintIndent();
    // A surrounding SIMD loop may discover this local.var as loop-carried
    // state before the allocation statement is emitted.  LocalVarID then
    // reserves its Python name in var_idmap_.  Reuse that reservation here;
    // AllocVarID would incorrectly diagnose it as a second TIR definition.
    // local_var_buffers_ records allocations that were actually emitted, so
    // a genuinely repeated allocation does not reset an accumulator.
    if (local_var_buffers_.count(TensorTypeOf(buffer).get())) {
      return;
    }
    local_var_buffers_.insert(TensorTypeOf(buffer).get());
    std::string vid = var_idmap_.count(TensorTypeOf(buffer).get())
                          ? GetVarID(TensorTypeOf(buffer).get())
                          : AllocVarID(TensorTypeOf(buffer).get());
    EmitLocalVarInitialization_(buffer, std::move(vid));
    return;
  }

  RegisterHandleType_(TensorTypeOf(buffer).get(), TensorTypeOf(buffer).dtype());
}

std::string CodeGenTileLangPTO::GetAddressOfExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1U);
  const auto *load = op->args[0].as<TensorLoadNode>();
  ICHECK(load) << "address_of expects TensorLoad";
  ICHECK_EQ(load->indices.size(), 1U)
      << "CodeGenTileLangPTO only supports flat memory";
  return GetPointerExpr(load->source.as_or_throw<TensorVar>().get(),
                        load->indices[0]);
}

std::string CodeGenTileLangPTO::GetAccessPtrExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 5U);
  auto buffer_var = (op->args[1]).as_or_throw<Var>();
  PrimType elem_dtype = PrimType::Float(32);

  if (const auto *type_call = op->args[0].as<CallNode>()) {
    if (!type_call->args.empty()) {
      if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
        elem_dtype = ParseDataType(dtype_name->value);
      }
    }
  } else if (HandleTypeMatch_(buffer_var.get(), PrimType::Float(32))) {
    elem_dtype = PrimType::Float(32);
  }

  return GetPointerExpr(buffer_var.get(), elem_dtype, op->args[2]);
}

std::string CodeGenTileLangPTO::GetAscendCopyGmUbExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 11U)
      << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";
  int64_t sid = 0;
  ICHECK(TryGetConstInt(op->args[2], &sid) && sid == 0)
      << "PTO GM->UB MTE requires sid == 0, got " << op->args[2];

  const VarNode *dst_var = nullptr;
  const VarNode *src_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimExpr src_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  PrimType src_dtype = PrimType::Void();
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO GM->UB MTE destination", &dst_var,
                   &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO GM->UB MTE source", &src_var, &src_index,
                   &src_dtype, &src_scope);
  ICHECK(dst_scope == "shared" || dst_scope == "shared.dyn")
      << "PTO GM->UB MTE destination must use shared/shared.dyn (UB) "
         "storage, got scope `"
      << dst_scope << "`";
  ICHECK(src_scope.empty() || src_scope == "global")
      << "PTO GM->UB MTE source must use global (GM) storage, got scope `"
      << src_scope << "`";
  ICHECK(IsStorageDtypeCompatible(dst_dtype, src_dtype))
      << "PTO GM->UB MTE does not support dtype conversion: source is "
      << src_dtype << ", destination is " << dst_dtype;

  int64_t data_select = 0;
  ICHECK(TryGetConstInt(op->args[7], &data_select) &&
         (data_select == 0 || data_select == 1))
      << "PTO GM->UB MTE requires dataSelect to be the constant 0 or 1, got "
      << op->args[7];
  std::string left_padding =
      RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string right_padding =
      RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  if (data_select == 0) {
    sym::Analyzer padding_analyzer;
    ICHECK(
        padding_analyzer->CanProveEqual(op->args[5].as_or_throw<PrimExpr>(),
                                        MakeZero(PythonDType(op->args[5]))) &&
        padding_analyzer->CanProveEqual(op->args[6].as_or_throw<PrimExpr>(),
                                        MakeZero(PythonDType(op->args[6]))))
        << "PTO GM->UB MTE cannot use left/right padding when dataSelect == 0";
  }
  int64_t l2_cache_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[8], &l2_cache_ctrl))
      << "PTO GM->UB MTE requires constant l2_cache_ctl, got " << op->args[8];
  ICHECK_GE(l2_cache_ctrl, 0)
      << "PTO GM->UB MTE l2_cache_ctl must be in [0, 15], got "
      << l2_cache_ctrl;
  ICHECK_LT(l2_cache_ctrl, 16)
      << "PTO GM->UB MTE l2_cache_ctl must be in [0, 15], got "
      << l2_cache_ctrl;

  std::string dst = current_function_is_cube_
                        ? GetLocalPtrExpr(op->args[0], "ub", dst_dtype)
                        : RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  // Match the Ascend target's no-padding GM->UB behavior, which emits uint8_t
  // pointers when no padding is requested
  if (data_select == 0) {
    if (src_dtype != PrimType::UInt(8)) {
      src = "pto.castptr(" + src + ", pto.ptr(pto.ui8, \"gm\"))";
    }
    if (dst_dtype != PrimType::UInt(8)) {
      dst = "pto.castptr(" + dst + ", pto.ptr(pto.ui8, \"ub\"))";
    }
  }
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[9]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[10]));

  std::ostringstream os;
  os << "pto.mte_gm_ub(" << src << ", " << dst << ", " << l2_cache_ctrl << ", "
     << burst_len << ", nburst=(" << burst_num << ", " << src_stride << ", "
     << dst_stride << ")";
  if (data_select != 0) {
    std::string pad_value;
    PrimType pad_dtype = current_copy_pad_value_dtype_;
    if (current_copy_pad_value_id_ >= 0) {
      pad_value = "tl_copy_pad_" + std::to_string(current_copy_pad_value_id_);
    } else {
      ICHECK(uniform_const_copy_pad_value_.has_value())
          << "PTO GM->UB MTE with dataSelect == 1 requires a preceding "
             "tl.ascend_set_copy_pad_value in the same function";
      pad_value = GetCopyPadValueExpr_(uniform_const_copy_pad_value_.value());
      pad_dtype = PythonDType(uniform_const_copy_pad_value_.value());
    }
    ICHECK(IsStorageDtypeCompatible(pad_dtype, dst_dtype))
        << "PTO GM->UB padding dtype mismatch: pad value is " << pad_dtype
        << ", copy elements are " << dst_dtype;
    os << ", pad=(" << pad_value << ", " << left_padding << ", "
       << right_padding << ")";
  }
  os << ")";
  return os.str();
}

std::string CodeGenTileLangPTO::GetAscendCopyUbGmExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 8U)
      << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";
  int64_t sid = 0;
  ICHECK(TryGetConstInt(op->args[2], &sid) && sid == 0)
      << "PTO UB->GM MTE requires sid == 0, got " << op->args[2];

  const VarNode *dst_var = nullptr;
  const VarNode *src_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimExpr src_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  PrimType src_dtype = PrimType::Void();
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO UB->GM MTE destination", &dst_var,
                   &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO UB->GM MTE source", &src_var, &src_index,
                   &src_dtype, &src_scope);
  ICHECK(dst_scope.empty() || dst_scope == "global")
      << "PTO UB->GM MTE destination must use global (GM) storage, got scope `"
      << dst_scope << "`";
  ICHECK(src_scope == "shared" || src_scope == "shared.dyn")
      << "PTO UB->GM MTE source must use shared/shared.dyn (UB) storage, got "
         "scope `"
      << src_scope << "`";
  ICHECK(IsStorageDtypeCompatible(dst_dtype, src_dtype))
      << "PTO UB->GM MTE does not support dtype conversion: source is "
      << src_dtype << ", destination is " << dst_dtype;
  ValidateUBCopyLayout_(src_index, src_dtype, op->args[3], op->args[4],
                        op->args[7], MakeZero(PrimType::Int(32)),
                        MakeZero(PrimType::Int(32)), false,
                        "PTO UB->GM MTE source");
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = current_function_is_cube_
                        ? GetLocalPtrExpr(op->args[1], "ub", src_dtype)
                        : RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string l2_cache = StoreL2CacheName(op->args[5]);
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  std::ostringstream os;
  os << "pto.mte_ub_gm(" << src << ", " << dst << ", " << burst_len
     << ", nburst=(" << burst_num << ", " << src_stride << ", " << dst_stride
     << "), l2_cache=\"" << l2_cache << "\")";
  return os.str();
}

void CodeGenTileLangPTO::RecordAscendCopyPadValue_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1U)
      << "tl.ascend_set_copy_pad_value expects exactly 1 argument";
  PrimType dtype = PythonDType(op->args[0]);
  int bits = dtype.bits();
  bool supported =
      dtype.IsScalar() &&
      (dtype.MatchesElementType(kDLBfloat, 16) ||
       (dtype.MatchesCode(kDLFloat) && (bits == 16 || bits == 32)) ||
       ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
        (bits == 8 || bits == 16 || bits == 32)));
  ICHECK(supported)
      << "PTO tl.ascend_set_copy_pad_value supports scalar int/uint8/16/32, "
         "float16, bfloat16, and float32 values, got "
      << dtype;
  current_copy_pad_value_id_ = copy_pad_value_counter_++;
  current_copy_pad_value_dtype_ = dtype;
  PrintIndent();
  stream << "tl_copy_pad_" << current_copy_pad_value_id_ << " = "
         << GetCopyPadValueExpr_(op->args[0]) << "\n";
}

void CodeGenTileLangPTO::EmitAscendSetAtomic(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 2U)
      << "tl.ascend_set_atomic expects exactly 2 arguments (op, typed_zero)";
  const auto *atomic_op = op->args[0].as<StringImmNode>();
  ICHECK(atomic_op)
      << "tl.ascend_set_atomic operation must be a compile-time string";
  ICHECK(atomic_op->value == "add" || atomic_op->value == "max" ||
         atomic_op->value == "min")
      << "PTO store atomic operation must be add, max, or min, got "
      << atomic_op->value;

  PrimType dtype = PythonDType(op->args[1]);
  std::string type_suffix = AtomicTypeSuffix(dtype);
  auto emit_type = [&]() {
    PrintIndent();
    stream << "pto.set_atomic_" << type_suffix << "()\n";
  };
  auto emit_op = [&]() {
    PrintIndent();
    stream << "pto.set_atomic_" << atomic_op->value << "()\n";
  };
  // Match AscendC's SetAtomicAdd/Max/Min ordering.  The two setters update
  // disjoint CTRL fields, but preserving this order makes the lowering
  // directly comparable with the reference implementation.
  if (atomic_op->value == "add") {
    emit_type();
    emit_op();
  } else {
    emit_op();
    emit_type();
  }
}

void CodeGenTileLangPTO::EmitCopyPadMerge_(int merged_value_id,
                                           PrimType expected_dtype) {
  ICHECK_GE(current_copy_pad_value_id_, 0);
  ICHECK(current_copy_pad_value_dtype_ == expected_dtype);
  PrintIndent();
  stream << "tl_copy_pad_" << merged_value_id << " = tl_copy_pad_"
         << current_copy_pad_value_id_ << "\n";
}

std::string CodeGenTileLangPTO::GetCopyPadValueExpr_(const Expr &value) {
  if (const auto *imm = value.as<IntImmNode>()) {
    std::ostringstream integer_literal;
    integer_literal << imm->value;
    return "pto.const(" + integer_literal.str() +
           ", dtype=" + ScalarType(PythonDType(value)) + ")";
  }
  if (const auto *imm = value.as<FloatImmNode>()) {
    std::ostringstream literal;
    if (std::isnan(imm->value)) {
      literal << "float('nan')";
    } else if (std::isinf(imm->value)) {
      literal << (imm->value < 0 ? "-float('inf')" : "float('inf')");
    } else if (imm->value == 0.0 && std::signbit(imm->value)) {
      literal << "-0.0";
    } else {
      literal << std::setprecision(std::numeric_limits<double>::max_digits10)
              << imm->value;
    }
    return "pto.const(" + literal.str() +
           ", dtype=" + ScalarType(PythonDType(value)) + ")";
  }
  return RemoveOutermostParentheses(PrintExpr_(value));
}

void CodeGenTileLangPTO::GetCopyEndpoint_(const Expr &expr, const char *context,
                                          const VarNode **buffer_var,
                                          PrimExpr *index, PrimType *dtype,
                                          std::string *scope) const {
  ICHECK(GetAddressOfIndex(expr, index, buffer_var))
      << context << " expects address_of/tvm_access_ptr, got " << expr;
  *dtype = GetAnnotatedPointerDtype(expr, PrimType::Void());
  ICHECK(!dtype->IsVoid()) << context << " has no element dtype annotation";

  auto scope_it = alloc_storage_scope_.find(*buffer_var);
  if (scope_it != alloc_storage_scope_.end()) {
    *scope = scope_it->second;
  } else if (const auto *ptr = (*buffer_var)->ty.as<PtrTypeNode>()) {
    *scope = ptr->scope.name();
  } else {
    scope->clear();
  }
  if (*scope == "gm") {
    *scope = "global";
  }
}

void CodeGenTileLangPTO::ValidateUBCopyLayout_(
    const Expr &index, PrimType dtype, const Expr &burst_num,
    const Expr &burst_len, const Expr &ub_stride, const Expr &left_padding,
    const Expr &right_padding, bool uses_padding, const char *context) const {
  ICHECK(dtype.IsScalar() || IsPackedFp4Pair(dtype))
      << context << " requires a scalar or packed FP4-pair dtype, got "
      << dtype;

  sym::Analyzer analyzer;
  PrimExpr elem_index = index.as_or_throw<PrimExpr>();
  if (const auto *ramp = index.as<RampNode>()) {
    CheckContiguousRampStride(index, context);
    elem_index = analyzer->Simplify(ramp->base);
  }

  PrimExpr byte_offset =
      ElementIndexToByteOffsetExpr(elem_index, dtype, context, analyzer);
  PrimExpr offset_mod = analyzer->Simplify(
      floormod(byte_offset, MakeConst(PythonDType(byte_offset), 32)));
  if (!analyzer->CanProveEqual(offset_mod, MakeZero(PythonDType(offset_mod)))) {
    int64_t offset_remainder = 0;
    ICHECK(!TryGetConstInt(offset_mod, &offset_remainder))
        << context << " address must be 32-byte aligned, but element offset "
        << elem_index << " for dtype " << dtype << " has byte remainder "
        << offset_remainder << " modulo 32";
  }

  int64_t nburst = 0;
  int64_t stride = 0;
  bool has_const_nburst = TryGetConstInt(burst_num, &nburst);
  if (has_const_nburst) {
    ICHECK_GT(nburst, 0) << context << " requires a positive burst count, got "
                         << nburst;
  }
  bool needs_row_stride = !has_const_nburst || nburst > 1;
  if (needs_row_stride) {
    // PTODSL accepts runtime SSA values for every nburst field.  Preserve a
    // dynamic TileLang row stride here.  Some packed scale copies also have a
    // narrow UB row stride when nburst > 1, so do not impose a codegen-only
    // 32-byte alignment check; keep the capacity checks below and let PTOAS
    // validate backend-specific MTE legality.
  }

  int64_t len = 0;
  int64_t left = 0;
  int64_t right = 0;
  bool has_const_len = TryGetConstInt(burst_len, &len);
  bool has_const_left = TryGetConstInt(left_padding, &left);
  bool has_const_right = TryGetConstInt(right_padding, &right);
  if (!has_const_left) {
    ICHECK(analyzer->CanProve(left_padding.as_or_throw<PrimExpr>() >=
                              MakeZero(PythonDType(left_padding))))
        << context << " requires a non-negative left padding count, got "
        << left_padding;
  }
  if (!has_const_right) {
    ICHECK(analyzer->CanProve(right_padding.as_or_throw<PrimExpr>() >=
                              MakeZero(PythonDType(right_padding))))
        << context << " requires a non-negative right padding count, got "
        << right_padding;
  }
  if (uses_padding) {
    if (!has_const_len) {
      ICHECK(analyzer->CanProve(burst_len.as_or_throw<PrimExpr>() >=
                                MakeZero(PythonDType(burst_len))))
          << context << " requires a non-negative burst length, got "
          << burst_len;
    }
    // A single burst has no following UB row, so ub_stride does not
    // participate in addressing. A dynamic count may be greater than one, so
    // it follows the conservative multi-burst stride and capacity checks.
    if (needs_row_stride && has_const_len && has_const_left &&
        has_const_right && TryGetConstInt(ub_stride, &stride)) {
      int64_t required = len + ElementsToBytesCeil(left + right, dtype);
      ICHECK_GE(stride, required)
          << context << " stride is too small for the padded row: got "
          << stride << " bytes, need at least " << required
          << " bytes (burst_len=" << len << ", leftPadding=" << left
          << ", rightPadding=" << right << ", dtype=" << dtype << ")";
    }
  }
}

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToUb(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 26U)
      << "tl.ascend_copy_matrix_cc_to_ub expects exactly 26 arguments";

  auto require_const = [&](size_t index, const char *name) {
    int64_t value = 0;
    ICHECK(TryGetConstInt(op->args[index], &value))
        << "PTO L0C->UB MTE requires constant " << name << ", got "
        << op->args[index];
    return value;
  };
  auto require_zero = [&](size_t index, const char *name) {
    int64_t value = require_const(index, name);
    ICHECK_EQ(value, 0) << "PTO L0C->UB MTE does not support " << name
                        << " != 0, got " << value;
  };

  int64_t sid = require_const(2, "sid");
  ICHECK_EQ(sid, 0) << "PTO L0C->UB MTE requires sid == 0, got " << sid;
  int64_t n_size_const = 0;
  int64_t m_size_const = 0;
  const bool has_const_n_size = TryGetConstInt(op->args[3], &n_size_const);
  const bool has_const_m_size = TryGetConstInt(op->args[4], &m_size_const);
  std::string n_size = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string m_size = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  int64_t dst_stride_const = 0;
  int64_t src_stride_const = 0;
  const bool has_const_dst_stride =
      TryGetConstInt(op->args[5], &dst_stride_const);
  const bool has_const_src_stride =
      TryGetConstInt(op->args[6], &src_stride_const);
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  ICHECK(!has_const_m_size || m_size_const > 0)
      << "PTO L0C->UB MTE requires m_size > 0";
  ICHECK(!has_const_n_size || n_size_const > 0)
      << "PTO L0C->UB MTE requires n_size > 0";
  ICHECK(!has_const_n_size || !has_const_dst_stride ||
         dst_stride_const >= n_size_const)
      << "PTO L0C->UB MTE destination stride must be at least n_size";
  ICHECK(!has_const_n_size || !has_const_src_stride ||
         src_stride_const >= n_size_const)
      << "PTO L0C->UB MTE source stride must be at least n_size";

  int64_t dual_dst_ctl = require_const(7, "dual_dst_ctl");
  ICHECK(dual_dst_ctl == 0 || dual_dst_ctl == 1 || dual_dst_ctl == 2)
      << "PTO L0C->UB MTE supports dual_dst_ctl 0 (single), 1 (M split), or "
         "2 (N split), got "
      << dual_dst_ctl;
  int64_t sub_blockid = require_const(8, "sub_blockid");
  if (dual_dst_ctl == 0) {
    ICHECK(sub_blockid == 0 || sub_blockid == 1)
        << "PTO L0C->UB MTE sub_blockid must be 0 or 1, got " << sub_blockid;
  } else {
    ICHECK_EQ(sub_blockid, 0)
        << "PTO L0C->UB split mode requires sub_blockid == 0, got "
        << sub_blockid;
  }

  require_zero(9, "clip_relu_pre");
  int64_t unit_flag_ctrl = require_const(10, "unit_flag_ctl");
  ICHECK(unit_flag_ctrl == 0 || unit_flag_ctrl == 2 || unit_flag_ctrl == 3)
      << "PTO L0C->UB MTE unit_flag_ctl must be 0, 2, or 3, got "
      << unit_flag_ctrl;
  require_zero(11, "quant_pre");
  require_zero(12, "relu_pre");
  require_zero(13, "split_en");
  int64_t nz2nd = require_const(14, "NZ2ND_en");
  ICHECK_EQ(nz2nd, 1)
      << "PTO L0C->UB MTE currently requires NZ2ND_en == 1, got " << nz2nd;
  const char *unsupported_controls[] = {"quant_post",
                                        "relu_post",
                                        "clip_relu_post",
                                        "loop_enhance_en",
                                        "eltwise_op",
                                        "eltwise_antq_en",
                                        "loop_enhance_merge_en",
                                        "C0_pad_en",
                                        "wino_post_en",
                                        "broadcast_en",
                                        "NZ2DN_en"};
  for (size_t index = 15; index < 26; ++index) {
    require_zero(index, unsupported_controls[index - 15]);
  }

  const VarNode *dst_var = nullptr;
  const VarNode *src_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimExpr src_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  PrimType src_dtype = PrimType::Void();
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO L0C->UB MTE destination", &dst_var,
                   &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO L0C->UB MTE source", &src_var, &src_index,
                   &src_dtype, &src_scope);
  ICHECK(dst_scope == "shared" || dst_scope == "shared.dyn")
      << "PTO L0C->UB MTE destination must use shared/shared.dyn (UB) "
         "storage, got scope `"
      << dst_scope << "`";
  ICHECK(IsBufferInScope(src_scope, "shared.l0c"))
      << "PTO L0C->UB MTE source must use shared.l0c (L0C) storage, got "
         "scope `"
      << src_scope << "`";
  ICHECK(IsStorageDtypeCompatible(dst_dtype, src_dtype))
      << "PTO L0C->UB MTE currently supports no-quant copies with matching "
         "storage dtypes, got source "
      << src_dtype << " and destination " << dst_dtype;
  ValidateUBCopyLayout_(
      dst_index, dst_dtype, MakeConst(PrimType::Int(32), 1),
      MakeConst(PrimType::Int(32), 0), MakeConst(PrimType::Int(32), 0),
      MakeConst(PrimType::Int(32), 0), MakeConst(PrimType::Int(32), 0), false,
      "PTO L0C->UB MTE destination");

  std::string src = GetAccPtrExpr(op->args[1], src_dtype);
  // Cube-local L1/L0C/UB allocations are represented as byte addresses in the
  // generated PTODSL. Materialize the UB address as a typed pointer instead of
  // forwarding the i64 allocation alias directly to mte_l0c_ub.
  std::string dst = GetLocalPtrExpr(op->args[0], "ub", dst_dtype);
  PrintIndent();
  stream << "pto.mte_l0c_ub(" << src << ", " << dst << ", " << m_size << ", "
         << n_size << ", " << src_stride << ", " << dst_stride;
  if (dual_dst_ctl == 0) {
    stream << ", " << sub_blockid;
  } else {
    stream << ", split=pto.SplitMode." << (dual_dst_ctl == 1 ? "M" : "N");
  }
  if (unit_flag_ctrl != 0) {
    stream << ", unit_flag=" << AccStoreUnitFlagArg(unit_flag_ctrl);
  }
  stream << ", layout=\"nz2nd\")\n";
}

std::string
CodeGenTileLangPTO::GetLocalByteAddrExpr(const Expr &index, PrimType elem_dtype,
                                         const std::string &context) {
  PrimExpr elem_index = index.as_or_throw<PrimExpr>();
  if (const auto *ramp = index.as<RampNode>()) {
    CheckContiguousRampStride(index, "local pointer offset");
    elem_index = sym::Analyzer()->Simplify(ramp->base);
  }

  int64_t const_index = 0;
  int64_t elem_bits = PhysicalElementBits(elem_dtype);
  if (elem_bits < 8) {
    ICHECK_EQ(8 % elem_bits, 0)
        << "PTO packed local pointer dtype bit width must divide one byte, got "
        << elem_dtype;
    int pack_factor = 8 / elem_bits;
    if (TryGetConstInt(elem_index, &const_index)) {
      ICHECK_EQ(const_index % pack_factor, 0)
          << context << " expects a packed-byte-aligned logical offset, got "
          << const_index << " for " << elem_dtype;
      return "pto.const(" + std::to_string(const_index / pack_factor) +
             ", dtype=pto.int64)";
    }
    LOG(FATAL) << context
               << " dynamic sub-byte local pointer offset requires native "
               << "signedness/width-aware integer division support, got "
               << elem_dtype;
    return "";
  }
  int64_t elem_bytes = PhysicalElementBytes(elem_dtype);
  if (TryGetConstInt(elem_index, &const_index)) {
    return "pto.const(" + std::to_string(const_index * elem_bytes) +
           ", dtype=pto.int64)";
  }

  std::string index_expr = RemoveOutermostParentheses(PrintExpr_(elem_index));
  std::string coerced = ScalarCastExpr(index_expr, PrimType::Int(64), context);
  if (elem_bytes == 1) {
    return coerced;
  }
  return "pto.mul(" + coerced + ", pto.const(" + std::to_string(elem_bytes) +
         ", dtype=pto.i64))";
}

std::string CodeGenTileLangPTO::GetLocalPtrExpr(const Expr &expr,
                                                const std::string &space,
                                                PrimType fallback_dtype) {
  PrimExpr index = MakeConst(PrimType::Int(32), 0);
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(expr, &index, &buffer_var))
      << "PTO local pointer expects address_of/tvm_access_ptr, got " << expr;

  PrimType elem_dtype = fallback_dtype;
  if (const auto *call = expr.as<CallNode>()) {
    if (PythonIsOp(call->op, "tvm_access_ptr") && !call->args.empty()) {
      if (auto *type_call = call->args[0].as<CallNode>()) {
        if (!type_call->args.empty()) {
          if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
            elem_dtype = ParseDataType(dtype_name->value);
          }
        }
      }
    } else if (PythonIsOp(call->op, "address_of")) {
      if (const auto *load = call->args[0].as<TensorLoadNode>()) {
        elem_dtype = load->source.as_or_throw<TensorVar>().dtype();
      }
    }
  }

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  ICHECK(scope == "shared" || scope == "shared.dyn" || scope == "shared.l1" ||
         scope == "shared.l1.dyn" || scope == "shared.l0a" ||
         scope == "shared.l0a.dyn" || scope == "shared.l0b" ||
         scope == "shared.l0b.dyn" || scope == "shared.l0c" ||
         scope == "shared.l0c.dyn" || scope == "local.fragment" ||
         scope.empty())
      << "PTO local pointer expected shared/local.fragment storage, got "
      << scope;

  std::string byte_addr =
      GetLocalByteAddrExpr(index, elem_dtype, "PTO local pointer offset");
  if (integer_address_vars_.count(GetRef<Var>(buffer_var))) {
    byte_addr =
        IsZero(index.as_or_throw<PrimExpr>())
            ? GetVarID(buffer_var)
            : "pto.add(" + GetVarID(buffer_var) + ", " + byte_addr + ")";
  }
  return "pto.castptr(" + byte_addr + ", " +
         PointerTypeName(elem_dtype, space) + ")";
}

std::string CodeGenTileLangPTO::GetE8M0ScalePtrExpr(const Expr &expr) {
  PrimExpr index = MakeConst(PrimType::Int(32), 0);
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(expr, &index, &buffer_var))
      << "PTO E8M0 scale pointer expects address_of/tvm_access_ptr, got "
      << expr;

  // Scale buffers use uint8 logical storage or an explicit uint16 pair-packed
  // view. Compute the byte address from that physical storage type, then expose
  // the address to the MX instruction as an E8M0 matrix pointer.
  PrimType storage_dtype = GetAnnotatedPointerDtype(expr, PrimType::UInt(8));
  ICHECK(storage_dtype == PrimType::UInt(8) ||
         storage_dtype == PrimType::UInt(16) ||
         storage_dtype == PrimType::Int(16))
      << "PTO E8M0 scale storage must be uint8 or pair-packed 16-bit, got "
      << storage_dtype;

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  ICHECK(scope == "shared.l1" || scope == "shared.l1.dyn")
      << "PTO E8M0 scale pointer must refer to an L1 allocation, got " << scope;

  std::string byte_addr = GetLocalByteAddrExpr(index, storage_dtype,
                                               "PTO E8M0 scale pointer offset");
  if (integer_address_vars_.count(GetRef<Var>(buffer_var))) {
    byte_addr =
        IsZero(index.as_or_throw<PrimExpr>())
            ? GetVarID(buffer_var)
            : "pto.add(" + GetVarID(buffer_var) + ", " + byte_addr + ")";
  }
  return "pto.castptr(" + byte_addr + ", pto.ptr(pto.f8e8m0, \"mat\"))";
}

std::string CodeGenTileLangPTO::GetAccPtrExpr(const Expr &expr,
                                              PrimType dtype) {
  return GetLocalPtrExpr(expr, "acc", dtype);
}

std::string CodeGenTileLangPTO::LocalVarID(const VarNode *var) {
  return GetVarID(var);
}

bool CodeGenTileLangPTO::IsLocalVarBuffer(const VarNode *var) const {
  return local_var_buffers_.count(var) != 0;
}

void CodeGenTileLangPTO::EmitMixedEntrySnapshot(const VarNode *var) {
  if (!current_function_is_mixed_ || inside_mixed_section_ ||
      !mixed_captured_local_vars_.count(var)) {
    return;
  }

  std::string value_id = LocalVarID(var);
  auto snapshot = mixed_entry_snapshot_ids_.find(var);
  if (snapshot == mixed_entry_snapshot_ids_.end()) {
    std::string snapshot_id =
        name_supply_->FreshName("tl_mixed_entry_" + value_id, false);
    snapshot = mixed_entry_snapshot_ids_.emplace(var, snapshot_id).first;
  }
  PrintIndent();
  stream << snapshot->second << " = " << value_id << "\n";
}

void CodeGenTileLangPTO::RestoreMixedSectionVariables(
    const RegionStmtNode *section) {
  auto section_vars = mixed_external_vars_by_section_.find(section);
  ICHECK(section_vars != mixed_external_vars_by_section_.end())
      << "Missing mixed-section variable analysis for " << RegionName(section);

  std::vector<std::pair<std::string, std::string>> restores;
  restores.reserve(section_vars->second.size());
  for (const VarNode *var : section_vars->second) {
    auto snapshot = mixed_entry_snapshot_ids_.find(var);
    ICHECK(snapshot != mixed_entry_snapshot_ids_.end())
        << "Mixed-section local.var is used before its outer allocation is "
           "emitted: "
        << var->name;
    restores.emplace_back(LocalVarID(var), snapshot->second);
  }
  std::sort(restores.begin(), restores.end());
  for (const auto &[value_id, snapshot_id] : restores) {
    PrintIndent();
    stream << value_id << " = " << snapshot_id << "\n";
  }
}

const CodeGenTileLangPTO::PTOGemmEmitContext &
CodeGenTileLangPTO::EnsureGemmHelper(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 12U)
      << "tl.ascend_gemm_l1 expects exactly 12 arguments";
  int64_t tile_m = ConstArgDim(op, 3, "tl.ascend_gemm_l1 M");
  int64_t tile_k = ConstArgDim(op, 4, "tl.ascend_gemm_l1 K");
  int64_t tile_n = ConstArgDim(op, 5, "tl.ascend_gemm_l1 N");
  int64_t base_k = ConstArgDim(op, 6, "tl.ascend_gemm_l1 tile_k_sub");
  int64_t trans_b = ConstArgDim(op, 7, "tl.ascend_gemm_l1 trans_b");
  ICHECK_EQ(trans_b, 1) << "PTO GEMM L1 helper currently requires trans_b=1";
  ICHECK_EQ(tile_k % base_k, 0);

  const auto *dtype_name = op->args[9].as<StringImmNode>();
  ICHECK(dtype_name) << "PTO GEMM L1 helper requires a constant input dtype "
                        "string at tl.ascend_gemm_l1 arg 9";
  PrimType input_dtype = ParseDataType(dtype_name->value);
  ICHECK(IsSupportedGemmInputDtype(input_dtype))
      << "PTO GEMM L1 helper only supports float16, bfloat16, float32, "
         "float8_e4m3fn, and float8_e5m2 inputs, got "
      << input_dtype;

  PrimType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  PrimType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(a_dtype == input_dtype)
      << "PTO GEMM L1 A pointer dtype must match input dtype " << input_dtype
      << ", got " << a_dtype;
  ICHECK(b_dtype == input_dtype)
      << "PTO GEMM L1 B pointer dtype must match input dtype " << input_dtype
      << ", got " << b_dtype;

  PrimType accum_dtype =
      GetAnnotatedPointerDtype(op->args[0], PrimType::Float(32));
  ICHECK(accum_dtype.MatchesCode(kDLFloat) && accum_dtype.bits() == 32)
      << "PTO GEMM L1 helper currently only supports float32 accum/output, got "
      << accum_dtype;

  Call call = GetRef<Call>(op);
  auto call_it = gemm_emit_context_by_call_.find(call);
  if (call_it != gemm_emit_context_by_call_.end()) {
    return gemm_emit_contexts_[call_it->second];
  }
  for (size_t i = 0; i < gemm_emit_contexts_.size(); ++i) {
    const auto &ctx = gemm_emit_contexts_[i];
    ICHECK(!ctx.blockscaled) << "PTO kernel mixes tl.ascend_gemm_l1 with "
                                "tl.ascend_blockscaled_gemm_l1";
    if (ctx.tile_m == tile_m && ctx.tile_n == tile_n && ctx.tile_k == tile_k &&
        ctx.base_k == base_k && ctx.input_dtype == input_dtype &&
        ctx.accum_dtype == accum_dtype) {
      gemm_emit_context_by_call_[call] = i;
      return ctx;
    }
  }

  size_t context_id = gemm_emit_contexts_.size();
  PTOGemmEmitContext ctx;
  ctx.blockscaled = false;
  ctx.tile_m = tile_m;
  ctx.tile_n = tile_n;
  ctx.tile_k = tile_k;
  ctx.base_k = base_k;
  ctx.input_dtype = input_dtype;
  ctx.accum_dtype = accum_dtype;
  std::string suffix = context_id == 0 ? "" : "_" + std::to_string(context_id);
  ctx.helper_name = "_tl_gemm_l1" + suffix;
  ctx.a_l0_name = "a_l0_" + std::to_string(context_id);
  ctx.b_l0_name = "b_l0_" + std::to_string(context_id);
  gemm_emit_contexts_.push_back(ctx);
  gemm_emit_context_by_call_[call] = context_id;
  const auto &emitted_ctx = gemm_emit_contexts_.back();

  int64_t input_c0 = GemmInputC0(input_dtype);
  ICHECK_EQ(base_k % input_c0, 0)
      << "PTO GEMM base_k must be divisible by input C0=" << input_c0
      << " for dtype " << input_dtype;
  int64_t sub_k_tiles = tile_k / base_k;
  int64_t sub_k_c0_blocks = base_k / input_c0;
  int64_t a_l0_stage_elems = tile_m * base_k;
  int64_t b_l0_stage_elems = base_k * tile_n;

  if (!gemm_zero_addr_emitted_) {
    PrintIndent();
    stream << "zero_addr = pto.const(0, dtype=pto.int64)\n";
    gemm_zero_addr_emitted_ = true;
  }
  PrintIndent();
  stream << emitted_ctx.a_l0_name << " = pto.castptr(zero_addr, "
         << PointerTypeName(input_dtype, "left") << ")\n";
  PrintIndent();
  stream << emitted_ctx.b_l0_name << " = pto.castptr(zero_addr, "
         << PointerTypeName(input_dtype, "right") << ")\n";
  PrintIndent();
  stream << emitted_ctx.helper_name << " = tl.PTOGemmL1Template(" << tile_m
         << ", " << tile_n << ", " << tile_k << ", " << base_k << ", "
         << sub_k_tiles << ", " << input_c0 << ", " << sub_k_c0_blocks << ", "
         << a_l0_stage_elems << ", " << b_l0_stage_elems << ")\n";
  return emitted_ctx;
}

const CodeGenTileLangPTO::PTOGemmEmitContext &
CodeGenTileLangPTO::EnsureBlockscaledGemmHelper(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 18U)
      << "tl.ascend_blockscaled_gemm_l1 expects exactly 18 arguments";
  int64_t tile_m = ConstArgDim(op, 5, "tl.ascend_blockscaled_gemm_l1 M");
  int64_t tile_k = ConstArgDim(op, 6, "tl.ascend_blockscaled_gemm_l1 K");
  int64_t tile_n = ConstArgDim(op, 7, "tl.ascend_blockscaled_gemm_l1 N");
  int64_t base_k =
      ConstArgDim(op, 8, "tl.ascend_blockscaled_gemm_l1 tile_k_sub");
  int64_t trans_b = ConstArgDim(op, 9, "tl.ascend_blockscaled_gemm_l1 trans_b");
  ICHECK_EQ(trans_b, 1)
      << "PTO blockscaled GEMM L1 helper currently requires trans_b=1";
  ICHECK_EQ(tile_k % base_k, 0);
  ICHECK_EQ(base_k % 64, 0)
      << "PTO blockscaled GEMM requires tile_k_sub multiple of 64, got "
      << base_k;

  const auto *dtype_name = op->args[11].as<StringImmNode>();
  ICHECK(dtype_name) << "PTO blockscaled GEMM L1 helper requires a constant "
                        "input dtype string at arg 11";
  PrimType input_dtype = ParseDataType(dtype_name->value);
  ICHECK(IsSupportedBlockscaledGemmInputDtype(input_dtype))
      << "PTO blockscaled GEMM L1 helper currently only supports bfloat16 and "
         "float8_e4m3fn inputs, got "
      << input_dtype;

  PrimType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  PrimType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(a_dtype == input_dtype)
      << "PTO blockscaled GEMM L1 A pointer dtype must match input dtype "
      << input_dtype << ", got " << a_dtype;
  ICHECK(b_dtype == input_dtype)
      << "PTO blockscaled GEMM L1 B pointer dtype must match input dtype "
      << input_dtype << ", got " << b_dtype;

  PrimType accum_dtype =
      GetAnnotatedPointerDtype(op->args[0], PrimType::Float(32));
  ICHECK(accum_dtype.MatchesCode(kDLFloat) && accum_dtype.bits() == 32)
      << "PTO blockscaled GEMM L1 helper currently only supports float32 "
         "accum/output, got "
      << accum_dtype;

  int64_t sf_nz_stride =
      ConstArgDim(op, 16, "tl.ascend_blockscaled_gemm_l1 sf_nz_stride");
  ICHECK_GE(sf_nz_stride, base_k / 64)
      << "PTO blockscaled GEMM requires sf_nz_stride >= tile_k_sub / 64, got "
         "sf_nz_stride="
      << sf_nz_stride << ", tile_k_sub=" << base_k;

  CheckConstZero(op->args[14], "blockscaled GEMM buf_offset");

  Call call = GetRef<Call>(op);
  auto call_it = gemm_emit_context_by_call_.find(call);
  if (call_it != gemm_emit_context_by_call_.end()) {
    return gemm_emit_contexts_[call_it->second];
  }
  for (size_t i = 0; i < gemm_emit_contexts_.size(); ++i) {
    const auto &ctx = gemm_emit_contexts_[i];
    ICHECK(ctx.blockscaled) << "PTO kernel mixes tl.ascend_gemm_l1 with "
                               "tl.ascend_blockscaled_gemm_l1";
    if (ctx.tile_m == tile_m && ctx.tile_n == tile_n && ctx.tile_k == tile_k &&
        ctx.base_k == base_k && ctx.sf_nz_stride == sf_nz_stride &&
        ctx.input_dtype == input_dtype && ctx.accum_dtype == accum_dtype) {
      gemm_emit_context_by_call_[call] = i;
      return ctx;
    }
  }

  size_t context_id = gemm_emit_contexts_.size();
  PTOGemmEmitContext ctx;
  ctx.blockscaled = true;
  ctx.tile_m = tile_m;
  ctx.tile_n = tile_n;
  ctx.tile_k = tile_k;
  ctx.base_k = base_k;
  ctx.sf_nz_stride = sf_nz_stride;
  ctx.input_dtype = input_dtype;
  ctx.accum_dtype = accum_dtype;
  std::string suffix = context_id == 0 ? "" : "_" + std::to_string(context_id);
  ctx.helper_name = "_tl_blockscaled_gemm_l1" + suffix;
  ctx.a_l0_name = "a_l0_" + std::to_string(context_id);
  ctx.b_l0_name = "b_l0_" + std::to_string(context_id);
  gemm_emit_contexts_.push_back(ctx);
  gemm_emit_context_by_call_[call] = context_id;
  const auto &emitted_ctx = gemm_emit_contexts_.back();

  int64_t input_c0 = GemmInputC0(input_dtype);
  ICHECK_EQ(base_k % input_c0, 0)
      << "PTO blockscaled GEMM base_k must be divisible by input C0="
      << input_c0 << " for dtype " << input_dtype;
  int64_t sub_k_tiles = tile_k / base_k;
  int64_t sub_k_c0_blocks = base_k / input_c0;
  int64_t input_pack_factor = GemmInputPackFactor(input_dtype);
  int64_t a_l0_stage_elems = tile_m * base_k / input_pack_factor;
  int64_t b_l0_stage_elems = tile_n * base_k / input_pack_factor;

  if (!gemm_zero_addr_emitted_) {
    PrintIndent();
    stream << "zero_addr = pto.const(0, dtype=pto.int64)\n";
    gemm_zero_addr_emitted_ = true;
  }
  PrintIndent();
  stream << emitted_ctx.a_l0_name << " = pto.castptr(zero_addr, "
         << PointerTypeName(input_dtype, "left") << ")\n";
  PrintIndent();
  stream << emitted_ctx.b_l0_name << " = pto.castptr(zero_addr, "
         << PointerTypeName(input_dtype, "right") << ")\n";
  PrintIndent();
  stream << emitted_ctx.helper_name << " = tl.PTOBlockscaledGemmL1Template("
         << tile_m << ", " << tile_n << ", " << tile_k << ", " << base_k << ", "
         << sub_k_tiles << ", " << input_c0 << ", " << sub_k_c0_blocks << ", "
         << a_l0_stage_elems << ", " << b_l0_stage_elems
         << ", sf_nz_stride=" << sf_nz_stride;
  if (input_pack_factor != 1) {
    stream << ", input_pack_factor=" << input_pack_factor;
  }
  stream << ")\n";
  return emitted_ctx;
}

void CodeGenTileLangPTO::ValidateFractalAddressAlignment_(
    const Expr &index, PrimType dtype, const char *context) const {
  const bool is_fp4_packed = IsPackedFp4Pair(dtype);
  ICHECK((dtype.IsScalar() || is_fp4_packed))
      << context << " requires a scalar or packed-FP4 dtype, got " << dtype;
  sym::Analyzer analyzer;
  int64_t elem_bytes = is_fp4_packed ? 1 : dtype.StorageBytes();
  PrimExpr byte_offset =
      analyzer->Simplify(index.as_or_throw<PrimExpr>() *
                         MakeConst(PythonDType(index), elem_bytes));
  PrimExpr offset_mod = analyzer->Simplify(
      floormod(byte_offset, MakeConst(PythonDType(byte_offset), 32)));
  if (!analyzer->CanProveEqual(offset_mod, MakeZero(PythonDType(offset_mod)))) {
    int64_t remainder = 0;
    ICHECK(!TryGetConstInt(offset_mod, &remainder))
        << context << " address must be 32-byte aligned, but element offset "
        << index << " for dtype " << dtype << " has byte remainder "
        << remainder << " modulo 32";
  }
}

void CodeGenTileLangPTO::EmitAscendCopyGmToCbuf(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 12U)
      << "tl.ascend_copy_gm_to_cbuf expects exactly 12 arguments";
  auto require_const = [&](size_t index, const char *name) {
    int64_t value = 0;
    ICHECK(TryGetConstInt(op->args[index], &value))
        << "PTO GM->L1 MTE requires constant " << name << ", got "
        << op->args[index];
    return value;
  };

  int64_t sid = require_const(2, "sid");
  int64_t src_inner_stride = 0;
  bool has_const_src_inner_stride =
      TryGetConstInt(op->args[3], &src_inner_stride);

  int64_t l2_cache_ctrl = require_const(4, "l2_cache_ctrl");

  int64_t n_value = 0;
  bool has_const_n_value = TryGetConstInt(op->args[5], &n_value);

  int64_t d_value = 0;
  bool has_const_d_value = TryGetConstInt(op->args[6], &d_value);

  int64_t src_outer_stride = 0;
  bool has_const_src_outer_stride =
      TryGetConstInt(op->args[7], &src_outer_stride);

  int64_t smallc0_en = require_const(8, "smallc0_en");
  int64_t transpose = require_const(9, "transpose");

  int64_t dst_n_value = 0;
  bool has_const_dst_n_value = TryGetConstInt(op->args[10], &dst_n_value);

  ICHECK_EQ(sid, 0) << "PTO GM->L1 MTE requires sid == 0, got " << sid;
  ICHECK(!has_const_src_inner_stride || src_inner_stride > 0)
      << "PTO GM->L1 MTE requires loop1_src_stride > 0";
  ICHECK(!has_const_src_outer_stride || src_outer_stride >= 0)
      << "PTO GM->L1 MTE requires non-negative loop4_src_stride";
  ICHECK_GE(l2_cache_ctrl, 0)
      << "PTO GM->L1 MTE l2_cache_ctrl must be in [0, 15], got "
      << l2_cache_ctrl;
  ICHECK_LT(l2_cache_ctrl, 16)
      << "PTO GM->L1 MTE l2_cache_ctrl must be in [0, 15], got "
      << l2_cache_ctrl;
  ICHECK(!has_const_n_value || n_value > 0)
      << "PTO GM->L1 MTE requires n_value > 0";
  ICHECK(!has_const_n_value || n_value <= 65535)
      << "PTO GM->L1 MTE n_value exceeds its 16-bit field: " << n_value;
  ICHECK(!has_const_d_value || d_value > 0)
      << "PTO GM->L1 MTE requires d_value > 0";
  ICHECK(!has_const_d_value || d_value < (1LL << 21))
      << "PTO GM->L1 MTE d_value exceeds its 21-bit field: " << d_value;
  ICHECK(!has_const_src_inner_stride || src_inner_stride < (1LL << 40))
      << "PTO GM->L1 MTE loop1_src_stride exceeds its 40-bit field: "
      << src_inner_stride;
  ICHECK(!has_const_src_outer_stride || src_outer_stride < (1LL << 40))
      << "PTO GM->L1 MTE loop4_src_stride exceeds its 40-bit field: "
      << src_outer_stride;
  ICHECK(smallc0_en == 0 || smallc0_en == 1)
      << "PTO GM->L1 MTE smallc0_en must be 0 or 1, got " << smallc0_en;
  ICHECK(smallc0_en == 0 || !has_const_d_value || d_value <= 4)
      << "PTO GM->L1 MTE smallc0_en requires d_value <= 4, got " << d_value;
  ICHECK(transpose == 0 || transpose == 1)
      << "PTO GM->L1 MTE transpose must be 0 or 1, got " << transpose;
  ICHECK(!has_const_dst_n_value || dst_n_value > 0)
      << "PTO GM->L1 MTE requires dst_n_value > 0";
  ICHECK(!has_const_dst_n_value || dst_n_value <= 65535)
      << "PTO GM->L1 MTE dst_n_value exceeds its 16-bit field: " << dst_n_value;

  // Packed scale factors are logically stored as 8-bit values but transferred
  // as uint16 pairs. Mirror AscendC's physical pointer casts while retaining
  // the common argument, scope, alignment, and stride validation.
  const auto *physical_dtype_imm = op->args[11].as<StringImmNode>();
  ICHECK(physical_dtype_imm)
      << "PTO GM->L1 MTE requires a constant physical_dtype string";
  bool has_physical = !physical_dtype_imm->value.empty();
  PrimType transfer_dtype = PrimType::Void();
  if (has_physical) {
    if (physical_dtype_imm->value == "int8_t") {
      // Packed FP4: two values per byte, issued as int8 DMA.
      transfer_dtype = PrimType::Int(8);
    } else if (physical_dtype_imm->value == "uint16_t") {
      // Packed E8M0 scale factors: two scales per uint16.
      transfer_dtype = PrimType::UInt(16);
    } else {
      ICHECK(false) << "PTO GM->L1 MTE supports physical_dtype int8_t (packed "
                       "FP4) or uint16_t (packed scale factors), got "
                    << physical_dtype_imm->value;
    }
  }

  const VarNode *dst_var = nullptr;
  const VarNode *src_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimExpr src_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  PrimType src_dtype = PrimType::Void();
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO GM->L1 MTE destination", &dst_var,
                   &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO GM->L1 MTE source", &src_var, &src_index,
                   &src_dtype, &src_scope);
  ICHECK(dst_scope == "shared.l1" || dst_scope == "shared.l1.dyn")
      << "PTO GM->L1 MTE destination must use shared.l1 storage, got scope `"
      << dst_scope << "`";
  ICHECK(src_scope.empty() || src_scope == "global")
      << "PTO GM->L1 MTE source must use global storage, got scope `"
      << src_scope << "`";

  if (has_physical) {
    const bool is_fp4 = dst_dtype.MatchesCode(kDLFloat4_e2m1fn);
    const bool is_pair_packed_scale = physical_dtype_imm->value == "uint16_t";
    ICHECK((!is_pair_packed_scale && is_fp4) ||
           (is_pair_packed_scale && dst_dtype.MatchesCode(kDLUInt) &&
            dst_dtype.bits() == 8))
        << "PTO GM-to-L1 copy supports int8 physical storage only for FP4 or "
           "uint16 pair-packed storage only for uint8 scale factors, got "
        << physical_dtype_imm->value << " for destination dtype " << dst_dtype;
  } else {
    ICHECK(IsStorageDtypeCompatible(dst_dtype, src_dtype))
        << "PTO GM->L1 MTE does not support dtype conversion: source is "
        << src_dtype << ", destination is " << dst_dtype;
    bool is_uint16_scale_storage =
        dst_dtype.bits() == 16 && src_dtype.bits() == 16 &&
        (dst_dtype.MatchesCode(kDLInt) || dst_dtype.MatchesCode(kDLUInt)) &&
        (src_dtype.MatchesCode(kDLInt) || src_dtype.MatchesCode(kDLUInt));
    ICHECK(IsSupportedCubeMteDtype(dst_dtype) || is_uint16_scale_storage)
        << "PTO GM->L1 MTE supports int8, float16, bfloat16, float32, and "
           "float8_e4m3fn/float8_e5m2 matrix data, plus signed or unsigned "
           "16-bit scale-factor storage; got "
        << dst_dtype;
    transfer_dtype = dst_dtype;
  }

  ValidateFractalAddressAlignment_(dst_index, transfer_dtype,
                                   "PTO GM->L1 MTE destination");
  bool has_const_source_row_elems =
      transpose == 0 ? has_const_d_value : has_const_n_value;
  int64_t source_row_elems = transpose == 0 ? d_value : n_value;
  ICHECK(!has_const_src_inner_stride || !has_const_source_row_elems ||
         src_inner_stride >= source_row_elems * transfer_dtype.StorageBytes())
      << "PTO GM->L1 MTE source row stride is smaller than the source row: "
      << src_inner_stride << " bytes for " << source_row_elems
      << " physical elements of " << transfer_dtype;

  std::string dst;
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  if (has_physical) {
    std::string byte_addr = GetLocalByteAddrExpr(
        dst_index, dst_dtype, "PTO packed L1 destination offset");
    dst = "pto.castptr(" + byte_addr + ", " +
          PointerTypeName(transfer_dtype, "mat") + ")";
    if (src_dtype != transfer_dtype) {
      src = "pto.castptr(" + src + ", " +
            PointerTypeName(transfer_dtype, "gm") + ")";
    }
  } else {
    dst = GetLocalPtrExpr(op->args[0], "mat", dst_dtype);
  }

  PrintIndent();
  stream << "pto.mte_gm_l1_frac(" << src << ", " << dst << ", "
         << (transpose == 0 ? "pto.FractalMode.ND2NZ" : "pto.FractalMode.DN2NZ")
         << ", shape=(" << PrintExpr_(op->args[5]) << ", "
         << PrintExpr_(op->args[6]) << "), src_layout=("
         << PrintExpr_(op->args[3]);
  if (!IsZero(op->args[7].as_or_throw<PrimExpr>())) {
    stream << ", " << PrintExpr_(op->args[7]);
  }
  stream << ",), dst_group=(1, 1, " << PrintExpr_(op->args[10])
         << ", 0), ctrl=(" << l2_cache_ctrl << ", "
         << (smallc0_en == 0 ? "False" : "True") << "))\n";
}

void CodeGenTileLangPTO::EmitAscendFillL1(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 7U)
      << "tl.ascend_fill_l1 expects exactly 7 arguments";

  const VarNode *dst_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  std::string dst_scope;
  GetCopyEndpoint_(op->args[0], "PTO L1 fill destination", &dst_var, &dst_index,
                   &dst_dtype, &dst_scope);
  ICHECK(dst_scope == "shared.l1" || dst_scope == "shared.l1.dyn")
      << "PTO L1 fill destination must use shared.l1 storage, got scope `"
      << dst_scope << "`";

  int64_t fill_word_bits = 0;
  ICHECK(TryGetConstInt(op->args[6], &fill_word_bits) &&
         (fill_word_bits == 16 || fill_word_bits == 32))
      << "PTO L1 fill requires fill_word_bits to be constant 16 or 32";

  PrintIndent();
  stream << "pto.raw_fill_l1(" << GetLocalPtrExpr(op->args[0], "mat", dst_dtype)
         << ", byte_offset=" << PrintExpr_(op->args[1])
         << ", raw_value=" << PrintExpr_(op->args[2])
         << ", repeat_times=" << PrintExpr_(op->args[3])
         << ", block_num_32b=" << PrintExpr_(op->args[4])
         << ", dst_gap_32b=" << PrintExpr_(op->args[5])
         << ", fill_word_bits=" << fill_word_bits << ")\n";
}

void CodeGenTileLangPTO::EmitAscendLoadCbufToL0(const CallNode *op,
                                                bool is_ca) {
  const char *name =
      is_ca ? "tl.ascend_load_cbuf_to_ca" : "tl.ascend_load_cbuf_to_cb";
  ICHECK(op->args.size() == 9U || op->args.size() == 16U)
      << name << " expects 9 or 16 arguments, got " << op->args.size();
  bool has_mx_scale = op->args.size() == 16U;

  auto require_const = [&](size_t index, const char *arg_name) {
    int64_t value = 0;
    ICHECK(TryGetConstInt(op->args[index], &value))
        << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B")
        << " MTE requires constant " << arg_name << ", got " << op->args[index];
    return value;
  };
  auto control_expr = [&](size_t index, const char *arg_name) {
    int64_t value = 0;
    if (TryGetConstInt(op->args[index], &value)) {
      ICHECK_GT(value, 0) << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B")
                          << " MTE requires positive " << arg_name << ", got "
                          << value;
    }
    return RemoveOutermostParentheses(PrintExpr_(op->args[index]));
  };
  int64_t row_start_const = 0;
  int64_t col_start_const = 0;
  bool has_const_row_start = TryGetConstInt(op->args[2], &row_start_const);
  bool has_const_col_start = TryGetConstInt(op->args[3], &col_start_const);
  if (has_const_row_start) {
    ICHECK_GE(row_start_const, 0)
        << "PTO L1->L0 MTE requires mStartPosition >= 0";
  }
  if (has_const_col_start) {
    ICHECK_GE(col_start_const, 0)
        << "PTO L1->L0 MTE requires kStartPosition >= 0";
  }
  std::string row_start = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  std::string col_start = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  // PTOAS accepts runtime scalar controls for the full-control form. Keep
  // dynamic values emitted by TileLang (notably mhc_norm_fn's tail mStep),
  // while validating positive literals here and letting PTOAS check ranges.
  std::string row_step = control_expr(4, "mStep");
  std::string col_step = control_expr(5, "kStep");
  std::string src_stride = control_expr(6, "srcStride");
  std::string dst_stride = control_expr(7, "dstStride");
  int64_t transpose = require_const(8, "transpose");
  ICHECK(transpose == 0 || transpose == 1)
      << "PTO L1->L0 MTE transpose must be 0 or 1, got " << transpose;

  const VarNode *dst_var = nullptr;
  const VarNode *src_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimExpr src_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  PrimType src_dtype = PrimType::Void();
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO L1->L0 MTE destination", &dst_var,
                   &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO L1->L0 MTE source", &src_var, &src_index,
                   &src_dtype, &src_scope);
  const char *expected_scope = is_ca ? "shared.l0a" : "shared.l0b";
  ICHECK(dst_scope == expected_scope ||
         dst_scope == expected_scope + std::string(".dyn"))
      << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B")
      << " MTE destination must use " << expected_scope << " storage, got `"
      << dst_scope << "`";
  ICHECK(src_scope == "shared.l1" || src_scope == "shared.l1.dyn")
      << "PTO L1->L0 MTE source must use shared.l1 storage, got `" << src_scope
      << "`";
  ICHECK(IsStorageDtypeCompatible(dst_dtype, src_dtype))
      << "PTO L1->L0 MTE does not support dtype conversion: source is "
      << src_dtype << ", destination is " << dst_dtype;
  ICHECK(IsSupportedCubeMteDtype(dst_dtype))
      << "PTO L1->L0 MTE supports int8, float16, bfloat16, float32, and "
         "float8_e4m3fn/float8_e5m2, got "
      << dst_dtype;
  ValidateFractalAddressAlignment_(dst_index, dst_dtype,
                                   "PTO L1->L0 MTE destination");
  ValidateFractalAddressAlignment_(src_index, src_dtype,
                                   "PTO L1->L0 MTE source");

  std::string src = GetLocalPtrExpr(op->args[1], "mat", src_dtype);
  std::string dst =
      GetLocalPtrExpr(op->args[0], is_ca ? "left" : "right", dst_dtype);
  // Preserve the controls produced by TileLang's canonical fractal-layout
  // lowering.  Reconstructing a shape and asking PTOAS to derive the controls
  // is not equivalent for every dtype: float32 uses an 8-element physical C0,
  // while the transpose shape helper aligns its axes to at least 16 elements.
  // Explicit controls also keep this lowering identical to the AscendC path.
  PrintIndent();
  stream << (is_ca ? "pto.mte_l1_l0a(" : "pto.mte_l1_l0b(") << src << ", "
         << dst << ", m_start=" << row_start << ", k_start=" << col_start
         << ", m_step=" << row_step << ", k_step=" << col_step
         << ", src_stride=" << src_stride << ", dst_stride=" << dst_stride
         << ", transpose=" << (transpose == 0 ? "False" : "True") << ")\n";

  if (!has_mx_scale) {
    return;
  }

  const VarNode *sf_var = nullptr;
  PrimExpr sf_index = MakeConst(PrimType::Int(32), 0);
  PrimType sf_dtype = PrimType::Void();
  std::string sf_scope;
  GetCopyEndpoint_(op->args[9], "PTO MX scale-factor L1 source", &sf_var,
                   &sf_index, &sf_dtype, &sf_scope);
  ICHECK(sf_scope == "shared.l1" || sf_scope == "shared.l1.dyn")
      << "PTO MX scale-factor source must use shared.l1 storage, got `"
      << sf_scope << "`";
  ICHECK(sf_dtype == PrimType::UInt(8) || sf_dtype == PrimType::UInt(16) ||
         sf_dtype == PrimType::Int(16))
      << "PTO MX scale-factor source supports uint8 or pair-packed 16-bit "
         "storage, got "
      << sf_dtype;
  ValidateFractalAddressAlignment_(sf_index, sf_dtype,
                                   "PTO MX scale-factor L1 source");

  const char *sf_arg_names[] = {"x_start", "y_start",    "x_step",
                                "y_step",  "src_stride", "dst_stride"};
  for (size_t i = 0; i < 6U; ++i) {
    int64_t value = 0;
    if (TryGetConstInt(op->args[10 + i], &value)) {
      if (i < 2U) {
        ICHECK_GE(value, 0)
            << "PTO MX " << sf_arg_names[i] << " must be non-negative";
      } else {
        ICHECK_GT(value, 0)
            << "PTO MX " << sf_arg_names[i] << " must be positive";
      }
    }
  }

  std::string sf_src = GetE8M0ScalePtrExpr(op->args[9]);
  PrintIndent();
  stream << (is_ca ? "pto.mte_l1_l0a_mx" : "pto.mte_l1_l0b_mx") << "(" << sf_src
         << ", " << dst;
  for (size_t i = 0; i < 6U; ++i) {
    stream << ", " << sf_arg_names[i] << "="
           << RemoveOutermostParentheses(PrintExpr_(op->args[10 + i]));
  }
  stream << ")\n";
}

void CodeGenTileLangPTO::EmitAscendLoadMxSf(const CallNode *op, bool is_ca) {
  const char *name = is_ca ? "tl.ascend_load_ca_sf" : "tl.ascend_load_cb_sf";
  ICHECK_EQ(op->args.size(), 8U)
      << name << " expects exactly 8 arguments, got " << op->args.size();

  auto control_expr = [&](size_t index, const char *arg_name) {
    int64_t value = 0;
    if (TryGetConstInt(op->args[index], &value)) {
      if (index < 4U) {
        ICHECK_GE(value, 0)
            << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B") << " MX " << arg_name
            << " must be non-negative, got " << value;
      } else {
        ICHECK_GT(value, 0)
            << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B") << " MX " << arg_name
            << " must be positive, got " << value;
      }
    }
    return RemoveOutermostParentheses(PrintExpr_(op->args[index]));
  };

  const VarNode *dst_var = nullptr;
  const VarNode *sf_var = nullptr;
  PrimExpr dst_index = MakeConst(PrimType::Int(32), 0);
  PrimExpr sf_index = MakeConst(PrimType::Int(32), 0);
  PrimType dst_dtype = PrimType::Void();
  PrimType sf_dtype = PrimType::Void();
  std::string dst_scope;
  std::string sf_scope;
  GetCopyEndpoint_(op->args[0], "PTO MX L0 destination", &dst_var, &dst_index,
                   &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO MX scale-factor source", &sf_var,
                   &sf_index, &sf_dtype, &sf_scope);

  const char *expected_scope = is_ca ? "shared.l0a" : "shared.l0b";
  ICHECK(IsBufferInScope(dst_scope, expected_scope))
      << name << " destination must use " << expected_scope
      << " storage, got scope " << dst_scope;
  ICHECK(sf_scope == "shared.l1" || sf_scope == "shared.l1.dyn")
      << name << " scale-factor source must use shared.l1 storage, got scope "
      << sf_scope;
  ICHECK(sf_dtype == PrimType::UInt(8) || sf_dtype == PrimType::UInt(16) ||
         sf_dtype == PrimType::Int(16))
      << name
      << " scale-factor source supports uint8 or pair-packed 16-bit "
         "storage, got "
      << sf_dtype;
  ICHECK(IsSupportedCubeMteDtype(dst_dtype))
      << name
      << " destination supports int8, float16, bfloat16, float32, "
         "float8_e4m3fn/float8_e5m2, and float4_e2m1fn, got "
      << dst_dtype;

  ValidateFractalAddressAlignment_(dst_index, dst_dtype,
                                   "PTO MX L0 destination");
  ValidateFractalAddressAlignment_(sf_index, sf_dtype,
                                   "PTO MX scale-factor source");

  std::string sf_src = GetE8M0ScalePtrExpr(op->args[1]);
  std::string dst =
      GetLocalPtrExpr(op->args[0], is_ca ? "left" : "right", dst_dtype);
  const char *arg_names[] = {"x_start", "y_start",    "x_step",
                             "y_step",  "src_stride", "dst_stride"};
  PrintIndent();
  stream << (is_ca ? "pto.mte_l1_l0a_mx(" : "pto.mte_l1_l0b_mx(") << sf_src
         << ", " << dst;
  for (size_t i = 0; i < 6U; ++i) {
    stream << ", " << arg_names[i] << "=" << control_expr(i + 2U, arg_names[i]);
  }
  stream << ")\n";
}

void CodeGenTileLangPTO::EmitAscendCrossCoreFlag(const CallNode *op,
                                                 bool is_set) {
  ICHECK_EQ(op->args.size(), 3U)
      << "tl.ascend_cross_core_" << (is_set ? "set" : "wait")
      << "_flag expects exactly 3 arguments (mode_id, pipe, flag_id)";
  int64_t mode_id = 0;
  ICHECK(TryGetConstInt(op->args[0], &mode_id))
      << "PTO cross-core flag requires a constant mode_id";
  ICHECK(mode_id == 0 || mode_id == 1 || mode_id == 2 || mode_id == 4)
      << "PTO cross-core flag mode_id must be one of 0/1/2/4, got " << mode_id;

  const auto *pipe_imm = op->args[1].as<StringImmNode>();
  ICHECK(pipe_imm) << "PTO cross-core flag requires a constant pipe string";
  PrimType event_dtype = PythonDType(op->args[2]);
  ICHECK(event_dtype.IsScalar() &&
         (event_dtype.MatchesCode(kDLInt) || event_dtype.MatchesCode(kDLUInt)))
      << "PTO cross-core flag_id must be a scalar integer, got " << event_dtype;

  std::string event_id = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  PrintIndent();
  stream << "tl.ascend_cross_core_" << (is_set ? "set" : "wait") << "_flag("
         << mode_id << ", \"" << pipe_imm->value << "\", " << event_id << ")\n";
}

void CodeGenTileLangPTO::EmitGemmRun(const PTOGemmEmitContext &ctx,
                                     const std::string &a_mat,
                                     const std::string &b_mat,
                                     const std::string &acc,
                                     const std::string &clear_accum,
                                     const std::string &unit_flag_ctrl,
                                     int64_t hf32_mode) {
  PrintIndent();
  stream << ctx.helper_name << ".run_l1_tile(" << a_mat << ", " << b_mat << ", "
         << ctx.a_l0_name << ", " << ctx.b_l0_name << ", " << acc
         << ", clear_accum=" << clear_accum
         << ", unit_flag_ctrl=" << unit_flag_ctrl;
  if (IsFloat32(ctx.input_dtype) && hf32_mode != 0) {
    // PTO names the hardware bit-47 mode ROUND_AWAY; CANN names the same
    // setting HF32TransMode::NEAREST_ZERO.
    stream << ", tf32_mode="
           << (hf32_mode == 1 ? "pto.Tf32Mode.ROUND_AWAY"
                              : "pto.Tf32Mode.ROUND_EVEN");
  }
  stream << ")\n";
}

void CodeGenTileLangPTO::EmitBlockscaledGemmRun(
    const PTOGemmEmitContext &ctx, const std::string &a_mat,
    const std::string &b_mat, const std::string &sfa_mat,
    const std::string &sfb_mat, const std::string &acc,
    const std::string &clear_accum, const std::string &sf_k_offset,
    const std::string &unit_flag_ctrl) {
  PrintIndent();
  stream << ctx.helper_name << ".run_l1_tile(" << a_mat << ", " << b_mat << ", "
         << sfa_mat << ", " << sfb_mat << ", " << ctx.a_l0_name << ", "
         << ctx.b_l0_name << ", " << acc << ", sf_k_offset=" << sf_k_offset
         << ", clear_accum=" << clear_accum
         << ", unit_flag_ctrl=" << unit_flag_ctrl << ")\n";
}

void CodeGenTileLangPTO::EmitAscendGemmL1(const CallNode *op) {
  ICHECK(!n_direction_by_mad_.at(GetRef<Call>(op)))
      << "PTO L1 GEMM does not support N MAD direction; use L0 inputs "
         "for N-directed GEMM";
  const PTOGemmEmitContext &ctx = EnsureGemmHelper(op);
  std::string acc = GetAccPtrExpr(op->args[0], PrimType::Float(32));
  std::string a_mat = GetLocalPtrExpr(op->args[1], "mat", ctx.input_dtype);
  std::string b_mat = GetLocalPtrExpr(op->args[2], "mat", ctx.input_dtype);

  int64_t hf32_mode = 0;
  if (IsFloat32(ctx.input_dtype)) {
    auto it = hf32_mode_by_gemm_.find(GetRef<Call>(op));
    ICHECK(it != hf32_mode_by_gemm_.end())
        << "PTO codegen did not analyze HF32 mode for FP32 GEMM";
    hf32_mode = it->second;
  }
  EmitGemmRun(ctx, a_mat, b_mat, acc,
              RemoveOutermostParentheses(PrintExpr_(op->args[8])),
              RemoveOutermostParentheses(PrintExpr_(op->args[11])), hf32_mode);
}

void CodeGenTileLangPTO::EmitAscendBlockscaledGemmL1(const CallNode *op) {
  ICHECK(!n_direction_by_mad_.at(GetRef<Call>(op)))
      << "PTO blockscaled L1 GEMM does not support N MAD direction";
  const PTOGemmEmitContext &ctx = EnsureBlockscaledGemmHelper(op);
  std::string acc = GetAccPtrExpr(op->args[0], PrimType::Float(32));
  std::string a_mat = GetLocalPtrExpr(op->args[1], "mat", ctx.input_dtype);
  std::string b_mat = GetLocalPtrExpr(op->args[2], "mat", ctx.input_dtype);

  std::string sfa_mat = GetE8M0ScalePtrExpr(op->args[3]);
  std::string sfb_mat = GetE8M0ScalePtrExpr(op->args[4]);

  EmitBlockscaledGemmRun(ctx, a_mat, b_mat, sfa_mat, sfb_mat, acc,
                         RemoveOutermostParentheses(PrintExpr_(op->args[10])),
                         RemoveOutermostParentheses(PrintExpr_(op->args[15])),
                         RemoveOutermostParentheses(PrintExpr_(op->args[17])));
}

void CodeGenTileLangPTO::EmitAscendMad(const CallNode *op) {
  const bool is_mx = op->op.same_as(tl::ascend_mad_mx());
  ICHECK(op->op.same_as(tl::ascend_mad()) || is_mx);
  const char *op_name = is_mx ? "mad_mx" : "mad";
  ICHECK_EQ(op->args.size(), 10U)
      << "tl.ascend_" << op_name << " expects exactly 10 arguments";

  auto get_endpoint = [&](size_t arg_index, const char *context) {
    const VarNode *buffer_var = nullptr;
    PrimExpr index = MakeConst(PrimType::Int(32), 0);
    PrimType dtype = PrimType::Void();
    std::string scope;
    GetCopyEndpoint_(op->args[arg_index], context, &buffer_var, &index, &dtype,
                     &scope);
    return std::make_pair(dtype, scope);
  };

  auto acc_info = get_endpoint(0, "PTO MAD accumulator");
  auto lhs_info = get_endpoint(1, "PTO MAD left operand");
  auto rhs_info = get_endpoint(2, "PTO MAD right operand");
  PrimType acc_dtype = acc_info.first;
  PrimType lhs_dtype = lhs_info.first;
  PrimType rhs_dtype = rhs_info.first;

  ICHECK(IsBufferInScope(lhs_info.second, "shared.l0a"))
      << "PTO " << op_name << " left operand must use shared.l0a storage, got `"
      << lhs_info.second << "`";
  ICHECK(IsBufferInScope(rhs_info.second, "shared.l0b"))
      << "PTO " << op_name
      << " right operand must use shared.l0b storage, got `" << rhs_info.second
      << "`";
  ICHECK(IsBufferInScope(acc_info.second, "shared.l0c"))
      << "PTO " << op_name << " accumulator must use shared.l0c storage, got `"
      << acc_info.second << "`";

  if (is_mx) {
    ICHECK(IsSupportedMadMxDtypeTuple(lhs_dtype, rhs_dtype, acc_dtype))
        << "PTO mad_mx supports float8_e4m3fn/float8_e5m2 left/right "
           "operands with a float32 accumulator, got left="
        << lhs_dtype << ", right=" << rhs_dtype
        << ", accumulator=" << acc_dtype;
  } else {
    ICHECK(IsSupportedMadDtypeTuple(lhs_dtype, rhs_dtype, acc_dtype))
        << "PTO mad supports matching float16, bfloat16, float32, or "
           "float8_e4m3fn/float8_e5m2 (including mixed FP8) left/right "
           "operands with a float32 accumulator, or matching int8 operands "
           "with an int32 accumulator; got left="
        << lhs_dtype << ", right=" << rhs_dtype
        << ", accumulator=" << acc_dtype;
  }

  std::string acc = GetAccPtrExpr(op->args[0], acc_dtype);
  std::string lhs = GetLocalPtrExpr(op->args[1], "left", lhs_dtype);
  std::string rhs = GetLocalPtrExpr(op->args[2], "right", rhs_dtype);
  auto direction_it = n_direction_by_mad_.find(GetRef<Call>(op));
  ICHECK(direction_it != n_direction_by_mad_.end())
      << "PTO codegen did not analyze MAD direction";
  const bool n_direction = direction_it->second;

  int64_t hf32_mode = 0;
  if (IsFloat32(lhs_dtype)) {
    auto it = hf32_mode_by_gemm_.find(GetRef<Call>(op));
    ICHECK(it != hf32_mode_by_gemm_.end())
        << "PTO codegen did not analyze HF32 mode for FP32 MAD";
    hf32_mode = it->second;
  }

  std::string m = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string k = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string n = RemoveOutermostParentheses(PrintExpr_(op->args[5]));

  int64_t gemv_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[7], &gemv_ctrl) &&
         (gemv_ctrl == 0 || gemv_ctrl == 1))
      << "PTO " << op_name << " requires constant gemv_ctrl 0 or 1, got "
      << op->args[7];

  int64_t btbuf_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[8], &btbuf_ctrl) && btbuf_ctrl == 0)
      << "PTO " << op_name << " currently only supports BTbuf_ctrl=0, got "
      << op->args[8];

  auto emit_mad_call = [&](bool clear_accumulator, int64_t unit_flag_ctrl) {
    ICHECK(unit_flag_ctrl == 0 || unit_flag_ctrl == 2 || unit_flag_ctrl == 3)
        << "PTO " << op_name
        << " unsupported unit_flag_ctrl=" << unit_flag_ctrl;
    const char *mad_fn = nullptr;
    if (is_mx) {
      mad_fn = clear_accumulator ? "pto.mad_mx" : "pto.mad_mx_acc";
    } else {
      mad_fn = clear_accumulator ? "pto.mad" : "pto.mad_acc";
    }

    PrintIndent();
    stream << mad_fn << "(" << lhs << ", " << rhs << ", " << acc << ", " << m
           << ", " << n << ", " << k;
    if (gemv_ctrl == 1) {
      stream << ", disable_gemv=True";
    }
    if (n_direction) {
      stream << ", n_dir=True";
    }
    if (IsFloat32(lhs_dtype) && hf32_mode != 0) {
      // Encode the reaching HF32 state as a per-MAD tf32_mode attribute.
      stream << ", tf32_mode="
             << (hf32_mode == 1 ? "pto.Tf32Mode.ROUND_AWAY"
                                : "pto.Tf32Mode.ROUND_EVEN");
    }
    if (unit_flag_ctrl == 2) {
      stream << ", unit_flag=\"check_only\"";
    } else if (unit_flag_ctrl == 3) {
      stream << ", unit_flag=\"check_and_set\"";
    }
    stream << ")\n";
  };

  auto emit_for_clear_accum = [&](int64_t unit_flag_ctrl) {
    int64_t zero_cmatrix = 0;
    if (TryGetConstInt(op->args[9], &zero_cmatrix)) {
      emit_mad_call(zero_cmatrix != 0, unit_flag_ctrl);
      return;
    }

    PrintIndent();
    stream << "if " << PrintCondition(op->args[9]) << ":\n";
    int clear_scope = BeginScope();
    emit_mad_call(true, unit_flag_ctrl);
    EndScope(clear_scope);
    PrintIndent();
    stream << "else:\n";
    int accumulate_scope = BeginScope();
    emit_mad_call(false, unit_flag_ctrl);
    EndScope(accumulate_scope);
  };

  auto emit_for_unit_flag = [&](auto &&self,
                                const Expr &unit_flag_expr) -> void {
    int64_t unit_flag_ctrl = 0;
    if (TryGetConstInt(unit_flag_expr, &unit_flag_ctrl)) {
      emit_for_clear_accum(unit_flag_ctrl);
      return;
    }

    PrimExpr condition = MakeConst(PrimType::Int(32), 0);
    PrimExpr true_value = MakeConst(PrimType::Int(32), 0);
    PrimExpr false_value = MakeConst(PrimType::Int(32), 0);
    if (const auto *select = unit_flag_expr.as<SelectNode>()) {
      condition = select->condition;
      true_value = select->true_value;
      false_value = select->false_value;
    } else if (const auto *call = unit_flag_expr.as<CallNode>();
               call != nullptr &&
               (PythonIsOp(call->op, "if_then_else") ||
                PythonIsOp(call->op, "if_then_else")) &&
               call->args.size() == 3U) {
      condition = call->args[0].as_or_throw<PrimExpr>();
      true_value = call->args[1].as_or_throw<PrimExpr>();
      false_value = call->args[2].as_or_throw<PrimExpr>();
    } else {
      LOG(FATAL) << "PTO " << op_name
                 << " supports dynamic unit_flag_ctrl only as Select/"
                    "if_then_else expressions whose leaves are 0, 2, or 3, "
                    "got "
                 << unit_flag_expr;
    }

    PrintIndent();
    stream << "if " << PrintCondition(condition) << ":\n";
    int true_scope = BeginScope();
    self(self, true_value);
    EndScope(true_scope);
    PrintIndent();
    stream << "else:\n";
    int false_scope = BeginScope();
    self(self, false_value);
    EndScope(false_scope);
  };

  sym::Analyzer analyzer;
  emit_for_unit_flag(emit_for_unit_flag,
                     analyzer->Simplify(op->args[6].as_or_throw<PrimExpr>()));
}

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToGm(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 25U)
      << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = GetAccPtrExpr(op->args[1], PrimType::Float(32));
  std::string sid = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  std::string n_size = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string m_size = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string l2_cache_ctrl =
      RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  int64_t unit_flag_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[9], &unit_flag_ctrl))
      << "PTO L0C-to-GM expects constant unit_flag_ctrl";
  int64_t quant_pre = 0;
  ICHECK(TryGetConstInt(op->args[10], &quant_pre) &&
         (quant_pre == 0 || quant_pre == 16))
      << "PTO L0C-to-GM currently supports quant_pre 0 (no conversion) or "
         "16 (float32-to-bfloat16), got "
      << op->args[10];

  auto check_const = [&](size_t index, int64_t expected, const char *name) {
    int64_t value = 0;
    ICHECK(TryGetConstInt(op->args[index], &value) && value == expected)
        << "PTO L0C-to-GM currently requires " << name << " == " << expected
        << ", got " << op->args[index];
  };
  check_const(8, 0, "clip_relu_pre");
  check_const(11, 0, "relu_pre");
  check_const(12, 0, "split_en");
  check_const(13, 1, "NZ2ND_en");
  check_const(14, 0, "quant_post");
  check_const(15, 0, "relu_post");
  check_const(16, 0, "clip_relu_post");
  check_const(17, 0, "loop_enhance_en");
  check_const(18, 0, "eltwise_op");
  check_const(19, 0, "eltwise_antq_en");
  check_const(20, 0, "loop_enhance_merge_en");
  check_const(21, 0, "C0_pad_en");
  check_const(22, 0, "wino_post_en");
  check_const(23, 0, "broadcast_en");
  check_const(24, 0, "NZ2DN_en");

  PrintIndent();
  stream << "pto.mte_l0c_gm(" << src << ", " << dst << ", " << m_size << ", "
         << n_size << ", " << src_stride << ", " << dst_stride << ", " << sid
         << ", " << l2_cache_ctrl;
  if (unit_flag_ctrl != 0) {
    stream << ", unit_flag=" << AccStoreUnitFlagArg(unit_flag_ctrl);
  }
  if (quant_pre == 16) {
    stream << ", pre_quant=(pto.bf16(1.0), \"f32_bf16\")";
  }
  stream << ", layout=\"nz2nd\")\n";
}

void CodeGenTileLangPTO::EmitAscendCopyUbufToCbuf(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 7U)
      << "tl.ascend_copy_ubuf_to_cbuf expects exactly 7 arguments";
  int64_t sid = 0;
  ICHECK(TryGetConstInt(op->args[2], &sid) && sid == 0)
      << "PTO UB->L1 copy currently only supports sub_blockid == 0, got "
      << op->args[2];

  std::string dst = GetLocalPtrExpr(op->args[0], "mat", PrimType::BFloat(16));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string src_gap = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string dst_gap = RemoveOutermostParentheses(PrintExpr_(op->args[6]));

  PrintIndent();
  stream << "pto.mte_ub_l1(" << src << ", " << dst << ", " << burst_len
         << ", nburst=(" << burst_num << ", " << src_gap << ", " << dst_gap
         << "))\n";
}

void CodeGenTileLangPTO::EmitAscendNd2NzPostCopy(const CallNode *op) {
  // ascend_nd2nz_post_copy(dst_l1, src_nz_ub, rows, cols, full_rows,
  //                        dst_dtype_str [, split_dim_neg2, half_extent])
  // The dual extras (args 6-7) are consumed by warpgroup_partition before
  // codegen; by the time we get here, the dst access_ptr already has the
  // sid-based offset baked in.
  ICHECK(op->args.size() == 6U || op->args.size() == 8U)
      << "tl.ascend_nd2nz_post_copy expects 6 or 8 arguments, got "
      << op->args.size();

  std::string dst = GetLocalPtrExpr(op->args[0], "mat", PrimType::BFloat(16));
  std::string src = GetLocalPtrExpr(op->args[1], "ub", PrimType::BFloat(16));

  int64_t rows = ConstArgDim(op, 2, "rows");
  int64_t cols = ConstArgDim(op, 3, "cols");
  int64_t full_rows = ConstArgDim(op, 4, "full_rows");
  std::string dst_dtype = (op->args[5]).as_or_throw<StringImm>()->value;

  int elem_bytes = (dst_dtype == "float") ? 4 : 2;
  int elems_perC0 = 32 / elem_bytes;
  int64_t burst_num = cols / elems_perC0;
  int64_t burst_len = rows;
  int64_t src_gap = 1;
  int64_t dst_gap = full_rows - rows;

  PrintIndent();
  stream << "pto.mte_ub_l1(" << src << ", " << dst << ", " << burst_len
         << ", nburst=(" << burst_num << ", " << src_gap << ", " << dst_gap
         << "))\n";
}

void CodeGenTileLangPTO::EmitAscendNd2NzScatter(const CallNode *op) {
  // ascend_nd2nz_scatter(src_ub, tmp_nz_ub, rows, cols,
  //                      dst_dtype_str, src_dtype_str)
  // Emits inline PTO SIMD calls that reorder [ROWS, COLS] data from ND
  // (row-major) to NZ (fractal) layout within UB.
  ICHECK_EQ(op->args.size(), 6U)
      << "tl.ascend_nd2nz_scatter expects exactly 6 arguments";

  int64_t rows = ConstArgDim(op, 2, "rows");
  int64_t cols = ConstArgDim(op, 3, "cols");
  std::string dst_dtype = (op->args[4]).as_or_throw<StringImm>()->value;
  std::string src_dtype = (op->args[5]).as_or_throw<StringImm>()->value;

  ICHECK(rows % 16 == 0) << "PTO nd2nz_scatter requires ROWS % 16 == 0, got "
                         << rows;

  std::string src_ptr =
      GetLocalPtrExpr(op->args[0], "ub", PrimType::BFloat(16));
  std::string dst_ptr =
      GetLocalPtrExpr(op->args[1], "ub", PrimType::BFloat(16));

  // Currently support same-type scatter only (the common case for dual_copy,
  // which rejects dtype mismatch on L0C->UB at the frontend level).
  ICHECK(src_dtype == dst_dtype)
      << "PTO nd2nz_scatter currently only supports same-type scatter; got "
      << "src=" << src_dtype << " dst=" << dst_dtype;

  int elem_bytes = (src_dtype == "float") ? 4 : 2;
  int vl_elems = 256 / elem_bytes;       // 64 for float, 128 for half/bfloat16
  int elems_per_block = 32 / elem_bytes; // 8 for float, 16 for half/bfloat16

  // 32-byte alignment (one C0 block) is the only hard requirement.
  // Partial-VL passes are handled via PAT_VL{N} masks.
  ICHECK((cols * elem_bytes) % 32 == 0)
      << "PTO nd2nz_scatter requires COLS * sizeof(elem) to be 32-byte "
      << "aligned, got COLS=" << cols << " elem_bytes=" << elem_bytes;

  int num_passes =
      static_cast<int>((cols + vl_elems - 1) / vl_elems); // ceil division
  for (int pass = 0; pass < num_passes; ++pass) {
    int remaining = static_cast<int>(cols - pass * vl_elems);
    int elems_this_pass = std::min(remaining, vl_elems);

    // Determine mask: PAT_ALL for full-VL passes, PAT_VL{N} for partial.
    std::string mask_fn;
    if (elems_this_pass >= vl_elems) {
      mask_fn = (elem_bytes == 4) ? "pto.pset_b32(\"PAT_ALL\")"
                                  : "pto.pset_b16(\"PAT_ALL\")";
    } else {
      mask_fn = (elem_bytes == 4) ? "pto.pset_b32(\"PAT_VL" +
                                        std::to_string(elems_this_pass) + "\")"
                                  : "pto.pset_b16(\"PAT_VL" +
                                        std::to_string(elems_this_pass) + "\")";
    }

    int src_offset = pass * vl_elems;
    int dst_pass_offset = pass * (static_cast<int>(rows) + 1) * vl_elems;

    for (int r = 0; r < rows; ++r) {
      int row_src_offset = src_offset + r * static_cast<int>(cols);
      // Each row advances by one 32B block (elems_per_block elements)
      // within the NZ layout. The destination pointer already includes the
      // complete row offset, so the non-post-update repeat stride must be
      // zero; a value of one would add another 32B offset and leave the first
      // row of every NZ tile uninitialized.
      int row_dst_offset = dst_pass_offset + r * elems_per_block;
      std::string var =
          "_nd2nz_v" + std::to_string(pass) + "_" + std::to_string(r);
      PrintIndent();
      stream << var << " = pto.vlds(pto.addptr(" << src_ptr << ", "
             << row_src_offset << "), pto.const(0, dtype=pto.int64),"
             << " dist=\"NORM\")\n";
      PrintIndent();
      stream << "pto.vsstb(" << var << ", pto.addptr(" << dst_ptr << ", "
             << row_dst_offset << "), " << (rows + 1) << ", 0, " << mask_fn
             << ")\n";
    }
  }
}

std::string CodeGenTileLangPTO::EmitAllReduceExpr_(const std::string &func_name,
                                                   const CallNode *op) {
  CheckAllReduceDtype(op);

  const size_t begin = func_name.find("tl::AscendAllReduce");
  ICHECK_NE(begin, std::string::npos)
      << "Cannot parse AscendAllReduce template arguments from: " << func_name;
  struct ReductionInfo {
    const char *tir_name;
    const char *pto_name;
  };
  const ReductionInfo reductions[] = {
      {"tl::SumOp", "tl.simt_allreduce_sum"},
      {"tl::MaxOp", "tl.simt_allreduce_max"},
      {"tl::MinOp", "tl.simt_allreduce_min"},
  };
  const ReductionInfo *reduction = nullptr;
  for (const ReductionInfo &candidate : reductions) {
    const std::string prefix =
        std::string("tl::AscendAllReduce<") + candidate.tir_name;
    if (func_name.find(prefix, begin) != std::string::npos) {
      reduction = &candidate;
      break;
    }
  }
  ICHECK(reduction != nullptr)
      << "PTO codegen currently maps only sum, max, and min AscendAllReduce "
         "reductions, got "
      << func_name;

  long long parsed_threads = 0;       // NOLINT(runtime/int)
  long long parsed_scale = 1;         // NOLINT(runtime/int)
  long long parsed_thread_offset = 0; // NOLINT(runtime/int)
  const std::string pattern = std::string("tl::AscendAllReduce<") +
                              reduction->tir_name + ", %lld, %lld, %lld";
  const int parsed =
      std::sscanf(func_name.c_str() + begin, pattern.c_str(), &parsed_threads,
                  &parsed_scale, &parsed_thread_offset);
  ICHECK_GE(parsed, 1)
      << "AscendAllReduce expects at least a threads template parameter: "
      << func_name;

  const int64_t threads = parsed_threads;
  const int64_t scale = parsed >= 2 ? parsed_scale : 1;
  const int64_t thread_offset = parsed >= 3 ? parsed_thread_offset : 0;

  std::string value = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string scratch = "None";
  if (op->args.size() >= 3U) {
    scratch = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  } else {
    const bool warp_fast_path =
        threads <= 32 && IsPowerOfTwo(threads) && IsPowerOfTwo(scale);
    ICHECK(threads <= scale || warp_fast_path)
        << "AscendAllReduce requires a scratch pointer outside the identity "
           "and power-of-two single-warp paths";
  }

  std::ostringstream os;
  os << reduction->pto_name << "(" << value << ", threads=" << threads
     << ", scale=" << scale << ", thread_offset=" << thread_offset
     << ", scratch=" << scratch << ")";
  return os.str();
}

void CodeGenTileLangPTO::Dispatch_(const CallNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (op->op.same_as(tirx::ptr_byte_offset_op())) {
    PrimExpr index(0);
    const VarNode *variable = nullptr;
    auto expression = GetRef<Call>(op);
    ICHECK(GetAddressOfIndex(expression, &index, &variable))
        << "PTO byte-offset pointer requires an addressable base";
    os << GetPointerExpr(
        variable, GetAnnotatedPointerDtype(expression, PrimType::UInt(8)),
        index);
    return;
  }
  if (EmitSimdMergingCall_(op, os)) {
    return;
  }
  if (PythonIsOp(op->op, "if_then_else") ||
      PythonIsOp(op->op, "if_then_else")) {
    ICHECK_EQ(op->args.size(), 3U)
        << "if_then_else expects <condition, true_value, false_value>";
    const bool requires_statement_control_flow =
        RequiresStatementControlFlow(op->args[1]) ||
        RequiresStatementControlFlow(op->args[2]);
    auto branch_value = [&](const Expr &expr) {
      std::string value = PrintExpr_(expr);
      if (PythonDType(expr).MatchesCode(kDLBool)) {
        // PTODSL typed constants require numeric bool literals.
        if (value == "True") {
          value = "1";
        } else if (value == "False") {
          value = "0";
        }
      }
      if (expr.as<IntImmNode>() != nullptr ||
          expr.as<FloatImmNode>() != nullptr) {
        return "pto.const(" + value +
               ", dtype=" + DataTypeName(PythonDType(GetRef<Expr>(op))) + ")";
      }
      if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLBool)) {
        return "tl.as_logical_bool(" + value + ")";
      }
      if (PythonDType(GetRef<Expr>(op)).IsScalar() &&
          (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
           PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt))) {
        return ScalarCastExpr(value, PythonDType(GetRef<Expr>(op)),
                              "PTO if_then_else branch");
      }
      return value;
    };

    if (requires_statement_control_flow) {
      // Stateful branch expressions may emit statements while being printed.
      // Keep those statements in the selected Python branch so PTODSL merges
      // the expression result and any live-out SSA state through the scf.if.
      std::string condition = PrintCondition(op->args[0]);
      std::string result_var =
          "_tl_if_result_" + std::to_string(pto_if_result_counter_++);

      PrintIndent();
      stream << "if " << condition << ":\n";
      int true_scope = BeginScope();
      std::string true_value = branch_value(op->args[1]);
      PrintIndent();
      stream << result_var << " = " << true_value << "\n";
      EndScope(true_scope);

      PrintIndent();
      stream << "else:\n";
      int false_scope = BeginScope();
      std::string false_value = branch_value(op->args[2]);
      PrintIndent();
      stream << result_var << " = " << false_value << "\n";
      EndScope(false_scope);

      os << result_var;
      return;
    }

    // tir.if_then_else is lazy and may guard an otherwise unsafe load.  Keep
    // the branch expressions in lambdas so the helper traces each one inside
    // the matching scf.if region.  Unlike injecting statements into `stream`,
    // this remains composable wherever an expression is printed.
    os << "tl.if_then_else(" << PrintCondition(op->args[0]);
    os << ", lambda: " << branch_value(op->args[1])
       << ", lambda: " << branch_value(op->args[2]) << ")";
    return;
  }

  if (PythonIsOp(op->op, "bitwise_and")) {
    PrintBinaryExpr_("&", PythonDType(GetRef<Expr>(op)), op->args[0],
                     op->args[1], os);
    return;
  }

  if (PythonIsOp(op->op, "bitwise_or")) {
    PrintBinaryExpr_("|", PythonDType(GetRef<Expr>(op)), op->args[0],
                     op->args[1], os);
    return;
  }

  if (PythonIsOp(op->op, "bitwise_xor")) {
    PrintBinaryExpr_("^", PythonDType(GetRef<Expr>(op)), op->args[0],
                     op->args[1], os);
    return;
  }

  const bool is_shift_left = PythonIsOp(op->op, "shift_left");
  const bool is_shift_right = PythonIsOp(op->op, "shift_right");
  if (is_shift_left || is_shift_right) {
    const char *shift_name = is_shift_left ? "shift_left" : "shift_right";
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative " << shift_name;
    PrimType dtype = PythonDType(op->args[0]);
    ICHECK(dtype.IsScalar() &&
           (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)))
        << "PTO " << shift_name << " expects a scalar integer operand, got "
        << dtype;
    ICHECK_LT(shift, dtype.bits())
        << "PTO " << shift_name << " amount " << shift
        << " must be smaller than operand bit width " << dtype.bits();

    PrintBinaryExpr_(is_shift_left ? "<<" : ">>", PythonDType(GetRef<Expr>(op)),
                     op->args[0], IntImm(PythonDType(op->args[1]), shift), os);
    return;
  }

  if (op->op.same_as(tirx::call_extern_op()) ||
      op->op.same_as(tirx::call_pure_extern_op())) {
    ICHECK_GE(op->args.size(), 1U);
    // TODO: Support more scalar ops, ref:
    // src/ascend/codegen/intrin_rule_ascend.cc
    std::string func_name = (op->args[0]).as_or_throw<StringImm>()->value;
    if (op->args.size() == 2U &&
        TryEmitUnaryMath_(func_name, op->args[1], os)) {
      return;
    }
    if (func_name.find("tl::AscendAllReduce") != std::string::npos) {
      os << EmitAllReduceExpr_(func_name, op);
      return;
    }
  }

  if (op->op.same_as(tl::warp_reduce_sum()) ||
      op->op.same_as(tl::warp_reduce_max()) ||
      op->op.same_as(tl::warp_reduce_min())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "PTO warp reduction expects exactly one argument";
    PrimType dtype = PythonDType(op->args[0]);
    ICHECK(IsSupportedSIMTAllReduceType(dtype))
        << "PTO warp reduction currently supports float16, float32, int32, "
           "and uint32 only, got "
        << dtype
        << ". Use a supported reducer dtype, such as float32, and cast the "
           "result afterward.";
    const char *name = op->op.same_as(tl::warp_reduce_sum())   ? "redux_add"
                       : op->op.same_as(tl::warp_reduce_max()) ? "redux_max"
                                                               : "redux_min";
    os << "tl." << name << "(" << PrintExpr_(op->args[0]) << ")";
    return;
  }

  bool is_atomic_add = op->op.same_as(tl::atomic_add_elem_op()) ||
                       op->op.same_as(tl::atomic_add_ret_elem_op());
  bool is_atomic_max = op->op.same_as(tl::atomic_max_elem_op()) ||
                       op->op.same_as(tl::atomic_max_ret_elem_op());
  bool is_atomic_min = op->op.same_as(tl::atomic_min_elem_op()) ||
                       op->op.same_as(tl::atomic_min_ret_elem_op());
  if (is_atomic_add || is_atomic_max || is_atomic_min) {
    ICHECK(inside_simtvf_body_)
        << "PTO atomic operations must be used inside T.SimtVF";
    ICHECK(op->args.size() == 2U || op->args.size() == 3U)
        << "PTO atomic operations expect dst_ptr, value[, memory_order]";

    if (op->args.size() == 3U) {
      const auto *memory_order = op->args[2].as<IntImmNode>();
      ICHECK(memory_order)
          << "PTO atomic memory_order must be a compile-time integer constant";
      ICHECK_EQ(memory_order->value, 0)
          << "PTO atomic operations currently support only memory_order="
             "\"relaxed\" because PTODSL/PTOAS atomic operations do not expose "
             "memory-order semantics; got memory_order id "
          << memory_order->value;
    }

    PrimType dtype =
        GetAnnotatedPointerDtype(op->args[0], PythonDType(op->args[1]));
    ICHECK(dtype.IsScalar())
        << "PTO atomic operations support scalar values only";
    // PTOAS supports 64-bit integer atomics for GM pointers.  Keep UB-space
    // legality in PTOAS's address-space-aware verifier instead of rejecting
    // all i64/u64 atomics here before PTODSL lowering.
    ICHECK((dtype.MatchesCode(kDLFloat) &&
            (dtype.bits() == 16 || dtype.bits() == 32)) ||
           ((dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
            (dtype.bits() == 32 || dtype.bits() == 64)))
        << "PTO atomic operations support float16, float32, int32, uint32, "
           "int64, and uint64, got "
        << dtype;

    const char *pto_atomic_op =
        is_atomic_add ? "atomic_add"
                      : (is_atomic_max ? "atomic_max" : "atomic_min");
    os << "pto." << pto_atomic_op << "(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]);
    // Only integer atomic_min/atomic_max need signedness
    if (!is_atomic_add &&
        (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt))) {
      os << ", signedness=\""
         << (dtype.MatchesCode(kDLUInt) ? "unsigned" : "signed") << "\"";
    }
    os << ")";
    return;
  }

  if (PythonIsOp(op->op, "gpu_storage_sync")) {
    ICHECK_GE(op->args.size(), 1U)
        << "gpu_storage_sync expects at least the storage scope argument";
    std::string sync_scope = (op->args[0]).as_or_throw<StringImm>()->value;
    if (sync_scope == "warp") {
      return;
    }
    if (sync_scope == "shared" || sync_scope == "shared.dyn") {
      PrintIndent();
      stream << (inside_simtvf_body_ ? "pto.syncthreads()\n"
                                     : "pto.pipe_barrier(pto.Pipe.ALL)\n");
      return;
    }
    LOG(FATAL) << "Unsupported PTO storage sync scope: " << sync_scope;
  }

  if (op->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
    PrintIndent();
    stream << GetAscendCopyGmUbExpr_(op) << "\n";
    return;
  }

  if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
    RecordAscendCopyPadValue_(op);
    return;
  }

  if (op->op.same_as(tl::ascend_set_atomic())) {
    EmitAscendSetAtomic(op);
    return;
  }

  if (op->op.same_as(tl::ascend_set_atomic_none())) {
    ICHECK_EQ(op->args.size(), 0U)
        << "tl.ascend_set_atomic_none expects no arguments";
    PrintIndent();
    stream << "pto.set_atomic_none()\n";
    return;
  }

  if (op->op.same_as(tl::ascend_copy_ubuf_to_gm())) {
    PrintIndent();
    stream << GetAscendCopyUbGmExpr_(op) << "\n";
    return;
  }

  if (op->op.same_as(tl::ascend_copy_gm_to_cbuf())) {
    EmitAscendCopyGmToCbuf(op);
    return;
  }

  if (op->op.same_as(tl::ascend_fill_l1())) {
    EmitAscendFillL1(op);
    return;
  }

  if (op->op.same_as(tl::ascend_load_cbuf_to_ca())) {
    EmitAscendLoadCbufToL0(op, true);
    return;
  }

  if (op->op.same_as(tl::ascend_load_cbuf_to_cb())) {
    EmitAscendLoadCbufToL0(op, false);
    return;
  }

  if (op->op.same_as(tl::ascend_load_ca_sf())) {
    EmitAscendLoadMxSf(op, true);
    return;
  }

  if (op->op.same_as(tl::ascend_load_cb_sf())) {
    EmitAscendLoadMxSf(op, false);
    return;
  }

  if (op->op.same_as(tl::ascend_cross_core_set_flag())) {
    EmitAscendCrossCoreFlag(op, true);
    return;
  }

  if (op->op.same_as(tl::ascend_cross_core_wait_flag())) {
    EmitAscendCrossCoreFlag(op, false);
    return;
  }

  if (op->op.same_as(tl::ascend_mad()) || op->op.same_as(tl::ascend_mad_mx())) {
    EmitAscendMad(op);
    return;
  }

  if (op->op.same_as(tl::ascend_gemm_l1())) {
    EmitAscendGemmL1(op);
    return;
  }
  if (op->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
    EmitAscendBlockscaledGemmL1(op);
    return;
  }

  // Scalar GM dcache bypass: MarkScalarDcacheBypass rewrites written GM
  // TensorLoad/Store into these Calls. Map to PTODSL cache-bypass helpers,
  // which lower to pto.ld_dev/st_dev.
  if (op->op.same_as(tl::ascend_read_gm_bypass_dcache())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.ascend_read_gm_bypass_dcache expects address_of(TensorLoad)";
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && PythonIsOp(addr->op, "address_of"))
        << "tl.ascend_read_gm_bypass_dcache expects address_of(TensorLoad)";
    const auto *load = addr->args[0].as<TensorLoadNode>();
    ICHECK(load && load->indices.size() == 1U)
        << "tl.ascend_read_gm_bypass_dcache expects a flat TensorLoad";
    ICHECK(PythonDType(GetRef<Expr>(load)).IsScalar())
        << "PTO GM dcache bypass only supports scalar loads, got "
        << PythonDType(GetRef<Expr>(load));
    std::string scope =
        ScopeOfBuffer(load->source.as_or_throw<TensorVar>().get());
    ICHECK(scope == "global" || scope.empty())
        << "PTO GM dcache bypass expects a global buffer, got scope " << scope;
    PrimType value_dtype = PythonDType(GetRef<Expr>(load));
    ValidateGmBypassDtype(value_dtype);
    std::string base = ScalarPointerBase_(
        load->source.as_or_throw<TensorVar>().get(), value_dtype, scope);
    std::string index =
        RemoveOutermostParentheses(PrintExpr_(load->indices[0]));
    os << "tl.read_gm_bypass_dcache(" << base << ", " << index << ", "
       << ScalarType(value_dtype) << ")";
    return;
  }

  if (op->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ascend_write_gm_bypass_dcache expects (address_of(TensorLoad), "
           "value)";
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && PythonIsOp(addr->op, "address_of"))
        << "tl.ascend_write_gm_bypass_dcache expects address_of(TensorLoad)";
    const auto *load = addr->args[0].as<TensorLoadNode>();
    ICHECK(load && load->indices.size() == 1U)
        << "tl.ascend_write_gm_bypass_dcache expects a flat TensorLoad";
    ICHECK(load->source.as_or_throw<TensorVar>().dtype().IsScalar())
        << "PTO GM dcache bypass only supports scalar stores, got "
        << load->source.as_or_throw<TensorVar>().dtype();
    std::string scope =
        ScopeOfBuffer(load->source.as_or_throw<TensorVar>().get());
    ICHECK(scope == "global" || scope.empty())
        << "PTO GM dcache bypass expects a global buffer, got scope " << scope;
    PrimType value_dtype = load->source.as_or_throw<TensorVar>().dtype();
    ValidateGmBypassDtype(value_dtype);
    // Print pointer/index/value first so any SSA side effects land before the
    // helper call.
    std::string base = ScalarPointerBase_(
        load->source.as_or_throw<TensorVar>().get(), value_dtype, scope);
    std::string index =
        RemoveOutermostParentheses(PrintExpr_(load->indices[0]));
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
    PrintIndent();
    stream << "tl.write_gm_bypass_dcache(" << base << ", " << index << ", "
           << value << ", " << ScalarType(value_dtype) << ")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_set_hf32_mode())) {
    // PTO represents HF32 as a per-MAD attribute. The mode analysis binds
    // each stateful TileLang setting to the GEMMs it reaches.
    return;
  }

  if (op->op.same_as(tl::ascend_set_mmad_direction())) {
    // The reaching direction is encoded as the per-MAD n_dir attribute.
    return;
  }

  if (op->op.same_as(tl::ascend_copy_matrix_cc_to_ub())) {
    EmitAscendCopyMatrixCcToUb(op);
    return;
  }

  if (op->op.same_as(tl::ascend_copy_matrix_cc_to_gm())) {
    EmitAscendCopyMatrixCcToGm(op);
    return;
  }

  if (op->op.same_as(tl::ascend_copy_ubuf_to_cbuf())) {
    EmitAscendCopyUbufToCbuf(op);
    return;
  }

  if (op->op.same_as(tl::ascend_nd2nz_post_copy())) {
    EmitAscendNd2NzPostCopy(op);
    return;
  }

  if (op->op.same_as(tl::ascend_nd2nz_scatter())) {
    EmitAscendNd2NzScatter(op);
    return;
  }

  if (PythonIsOp(op->op, "address_of")) {
    os << GetAddressOfExpr_(op);
    return;
  }

  if (PythonIsOp(op->op, "reinterpret")) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tirx.reinterpret expects 1 argument (value)";
    std::string value = PrintExpr_(op->args[0]);
    PrimType target_dtype = PythonDType(GetRef<Expr>(op));
    if (target_dtype.lanes() > 1) {
      // Vector reinterpret: lower to pto.vbitcast(vec, elem_type).
      os << "pto.vbitcast(" << value << ", "
         << ScalarType(target_dtype.WithLanes(1)) << ")";
    } else {
      PrimType source_dtype = PythonDType(op->args[0]);
      ICHECK(source_dtype.IsScalar())
          << "PTO scalar reinterpret expects scalar source, got "
          << source_dtype;
      ICHECK_EQ(source_dtype.bits(), target_dtype.bits())
          << "PTO scalar reinterpret requires equal bit widths, got source "
          << source_dtype << " and target " << target_dtype;
      os << "pto.bitcast(" << value << ", " << ScalarType(target_dtype) << ")";
    }
    return;
  }

  if (PythonIsOp(op->op, "tvm_access_ptr")) {
    os << GetAccessPtrExpr_(op);
    return;
  }

  if (op->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (TensorLoad, extent, rw_mask)";
    const auto *load = op->args[0].as<TensorLoadNode>();
    ICHECK(load) << "tl.access_ptr arg0 must be TensorLoad";
    os << GetPointerExpr(load->source.as_or_throw<TensorVar>().get(),
                         load->indices[0]);
    return;
  }

  if (op->op.same_as(tl::ascend_pipe_barrier())) {
    auto pipe_name = (op->args[0]).as_or_throw<StringImm>()->value;
    PrintIndent();
    stream << "pto.pipe_barrier(\"" << StripPipePrefix(pipe_name) << "\")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_set_flag()) ||
      op->op.same_as(tl::ascend_wait_flag())) {
    auto hard_event = (op->args[0]).as_or_throw<StringImm>()->value;
    auto [src_pipe, dst_pipe] = ParseHardEventPair(hard_event);
    std::string eid = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
    PrintIndent();
    stream << (op->op.same_as(tl::ascend_set_flag()) ? "pto.set_flag("
                                                     : "pto.wait_flag(")
           << "\"" << src_pipe << "\", \"" << dst_pipe << "\", event_id=" << eid
           << ")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_threadfence())) {
    ICHECK_EQ(op->args.size(), 0)
        << "tl.ascend_threadfence expects 0 arguments";
    PrintIndent();
    stream << "pto.threadfence();\n";
    return;
  }

  if (op->op.same_as(tl::ascend_get_buf()) ||
      op->op.same_as(tl::ascend_rls_buf())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.ascend_get_buf/rls_buf expects exactly 3 arguments "
           "(pipe string, buf_id, mode)";
    std::string pipe_str = (op->args[0]).as_or_throw<StringImm>()->value;
    std::string buf_id = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
    std::string mode = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
    PrintIndent();
    stream << (op->op.same_as(tl::ascend_get_buf()) ? "pto.get_buf("
                                                    : "pto.rls_buf(")
           << "\"" << StripPipePrefix(pipe_str) << "\", " << buf_id
           << ", mode=" << mode << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_pset())) {
    ICHECK_GE(op->args.size(), 1U)
        << "tl.simd.pset expects at least 1 argument (element width)";
    int64_t elem_width = 0;
    ICHECK(TryGetConstInt(op->args[0], &elem_width))
        << "tl.simd.pset element width must be constant for PTO codegen";
    std::string dist = "PAT_ALL";
    if (op->args.size() >= 2U) {
      dist = (op->args[1]).as_or_throw<StringImm>()->value;
    }
    os << "pto.pset_b" << elem_width << "(\"" << dist << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_pge())) {
    ICHECK(op->args.size() >= 1U && op->args.size() <= 2U)
        << "tl.simd.pge expects 1 or 2 arguments (element width[, "
           "distribution])";
    int64_t elem_width = 0;
    ICHECK(TryGetConstInt(op->args[0], &elem_width))
        << "tl.simd.pge element width must be constant for PTO codegen";
    ICHECK(elem_width == 8 || elem_width == 16 || elem_width == 32)
        << "PTO tl.simd.pge only supports element widths 8, 16, and 32, got "
        << elem_width;
    std::string dist = "PAT_ALL";
    if (op->args.size() == 2U) {
      const auto *dist_imm = op->args[1].as<StringImmNode>();
      ICHECK(dist_imm)
          << "tl.simd.pge distribution must be a constant string for PTO "
             "codegen";
      dist = dist_imm->value;
    }
    static const std::unordered_set<std::string> kSupportedPgePatterns = {
        "PAT_ALL", "PAT_VL1",  "PAT_VL2",  "PAT_VL3",  "PAT_VL4",
        "PAT_VL8", "PAT_VL16", "PAT_VL32", "PAT_VL64", "PAT_VL128",
        "PAT_M3",  "PAT_M4",   "PAT_H",    "PAT_Q",    "PAT_ALLF",
    };
    ICHECK(kSupportedPgePatterns.count(dist) != 0U)
        << "PTO tl.simd.pge does not support distribution " << dist
        << "; expected PAT_ALL, PAT_VL1/2/3/4/8/16/32/64/128, "
           "PAT_M3, PAT_M4, PAT_H, PAT_Q, or PAT_ALLF";
    os << "pto.pge_b" << elem_width << "(\"" << dist << "\")";
    return;
  }

  // Predicate load/store: (addr, dist) and (addr, src, dist).
  if (op->op.same_as(tl::simd_pld())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.pld expects 2 arguments (addr, dist)";
    const auto *dist_imm = op->args[1].as<StringImmNode>();
    ICHECK(dist_imm)
        << "tl.simd.pld distribution must be a constant string for PTO "
           "codegen";
    static const std::unordered_set<std::string> kSupportedPldDist = {
        "NORM", "US", "DS"};
    ICHECK(kSupportedPldDist.count(dist_imm->value) != 0U)
        << "PTO tl.simd.pld does not support distribution " << dist_imm->value
        << "; expected NORM, US, or DS";
    std::string addr = PrintExpr_(op->args[0]);
    if (GetAnnotatedPointerDtype(op->args[0], PrimType::UInt(32)) !=
        PrimType::UInt(32)) {
      addr = "pto.castptr(" + addr + ", " +
             PointerTypeName(PrimType::UInt(32), "ub") + ")";
    }
    os << "pto.plds(" << addr << ", pto.const(0), dist=\"" << dist_imm->value
       << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_pst())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.pst expects 3 arguments (addr, src, dist)";
    const auto *dist_imm = op->args[2].as<StringImmNode>();
    ICHECK(dist_imm)
        << "tl.simd.pst distribution must be a constant string for PTO "
           "codegen";
    static const std::unordered_set<std::string> kSupportedPstDist = {"NORM",
                                                                      "PK"};
    ICHECK(kSupportedPstDist.count(dist_imm->value) != 0U)
        << "PTO tl.simd.pst does not support distribution " << dist_imm->value
        << "; expected NORM or PK";
    std::string addr = PrintExpr_(op->args[0]);
    if (GetAnnotatedPointerDtype(op->args[0], PrimType::UInt(32)) !=
        PrimType::UInt(32)) {
      addr = "pto.castptr(" + addr + ", " +
             PointerTypeName(PrimType::UInt(32), "ub") + ")";
    }
    PrintIndent();
    stream << "pto.psts(" << PrintExpr_(op->args[1]) << ", " << addr
           << ", pto.const(0), dist=\"" << dist_imm->value << "\")\n";
    return;
  }

  if (op->op.same_as(tl::simd_vld())) {
    ICHECK(op->args.size() >= 2U && op->args.size() <= 3U)
        << "tl.simd.vld expects 2 or 3 arguments (addr, dist[, offset])";
    PrimType elem_dtype = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    ICHECK_GT(PythonDType(GetRef<Expr>(op)).lanes(), 1)
        << "tl.simd.vld should return a vector type, got "
        << PythonDType(GetRef<Expr>(op));
    std::string dist =
        NormalizeVldDistribution((op->args[1]).as_or_throw<StringImm>()->value);
    std::string offset =
        op->args.size() == 3U
            ? RemoveOutermostParentheses(PrintExpr_(op->args[2]))
            : "pto.const(0)";
    int pto_lanes = PythonDType(GetRef<Expr>(op)).lanes();
    if (elem_dtype.bits() < 8) {
      ICHECK_EQ((pto_lanes * elem_dtype.bits()) % 8, 0)
          << "PTO vld sub-byte vector lanes must form whole bytes, got "
          << PythonDType(GetRef<Expr>(op));
      pto_lanes = (pto_lanes * elem_dtype.bits()) / 8;
    }
    os << "pto.vlds(" << PrintExpr_(op->args[0]) << ", " << offset
       << ", pto.vreg_type(" << pto_lanes << ", " << ScalarType(elem_dtype)
       << ")";
    if (!dist.empty() && dist != "NORM") {
      os << ", dist=\"" << dist << "\"";
    }
    os << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vabsdif())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vabsdif expects 4 arguments (src0, src1, mask, mode)";
    PrimType elem_dtype = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    ICHECK(elem_dtype.MatchesCode(kDLFloat) &&
           (elem_dtype.bits() == 16 || elem_dtype.bits() == 32))
        << "PTO tl.simd.vabsdif only supports float16 and float32, got "
        << elem_dtype;
    ValidateVecMode_(op, 3);
    std::string mask = PrintExpr_(op->args[2]);
    os << "pto.vabs(pto.vsub(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << mask << "), " << mask << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vadd()) || op->op.same_as(tl::simd_vsub()) ||
      op->op.same_as(tl::simd_vmul()) || op->op.same_as(tl::simd_vdiv()) ||
      op->op.same_as(tl::simd_vmax()) || op->op.same_as(tl::simd_vmin()) ||
      op->op.same_as(tl::simd_vand()) || op->op.same_as(tl::simd_vor()) ||
      op->op.same_as(tl::simd_vxor()) || op->op.same_as(tl::simd_vshl()) ||
      op->op.same_as(tl::simd_vshr())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd binary vector op expects 4 arguments (src0, src1, mask, "
           "mode)";
    if (op->op.same_as(tl::simd_vdiv())) {
      // Per-op "precision" annotation (0=hw, 1=exact) wins over the global
      // fast-math default via the shared SFU precision resolver. Non-fp32
      // always uses the hardware instruction regardless of annotations.
      const bool is_f32 =
          PythonDType(GetRef<Expr>(op)).WithLanes(1).MatchesCode(kDLFloat) &&
          PythonDType(GetRef<Expr>(op)).WithLanes(1).bits() == 32;
      const SfuPrecision prec = ResolveSfuPrecision(
          GetRef<Call>(op), PythonDType(GetRef<Expr>(op)),
          (is_f32 && !enable_fast_math_) ? SfuPrecision::kExact
                                         : SfuPrecision::kHw);
      ICHECK_NE(static_cast<int>(prec),
                static_cast<int>(SfuPrecision::kKeepSub))
          << "PTO tl.simd.vdiv has no ftz_false precision tier; use "
             "precision=\"ftz_true\" or \"exact\"";
      if (prec == SfuPrecision::kExact) {
        ValidateVecMode_(op, 3);
        os << "tl.vdiv_precise_f32(" << PrintExpr_(op->args[0]) << ", "
           << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
        return;
      }
    }
    const char *name = nullptr;
    if (op->op.same_as(tl::simd_vadd()))
      name = "vadd";
    else if (op->op.same_as(tl::simd_vsub()))
      name = "vsub";
    else if (op->op.same_as(tl::simd_vmul()))
      name = "vmul";
    else if (op->op.same_as(tl::simd_vdiv()))
      name = "vdiv";
    else if (op->op.same_as(tl::simd_vmax()))
      name = "vmax";
    else if (op->op.same_as(tl::simd_vmin()))
      name = "vmin";
    else if (op->op.same_as(tl::simd_vand()))
      name = "vand";
    else if (op->op.same_as(tl::simd_vor()))
      name = "vor";
    else if (op->op.same_as(tl::simd_vxor()))
      name = "vxor";
    else if (op->op.same_as(tl::simd_vshl()))
      name = "vshl";
    else
      name = "vshr";
    ValidateVecMode_(op, 3);
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    return;
  }

  // --- SIMD unary: (src, mask, mode) → pto.<op>(src, mask) ---
  if (op->op.same_as(tl::simd_vabs()) || op->op.same_as(tl::simd_vexp()) ||
      op->op.same_as(tl::simd_vln()) || op->op.same_as(tl::simd_vsqrt()) ||
      op->op.same_as(tl::simd_vneg()) || op->op.same_as(tl::simd_vrelu()) ||
      op->op.same_as(tl::simd_vnot())) {
    // LegalizeSimdMerging rewrites `dst[0] = op(src, mask, MODE_MERGING)` into
    // a void call carrying dst as an explicit read-write first argument.
    const bool has_explicit_dst = op->args.size() == 4U;
    ICHECK(op->args.size() == 3U || has_explicit_dst)
        << "tl.simd unary op expects 3 arguments (src, mask, mode) or the "
           "legalized MODE_MERGING form (dst, src, mask, mode), got "
        << op->args.size() << " arguments";
    if (has_explicit_dst) {
      LOG(FATAL) << "PTO codegen does not support MODE_MERGING for "
                 << (op->op).as_or_throw<Op>()->name << " yet";
    }
    ValidateVecMode_(op, 2);
    const char *name = op->op.same_as(tl::simd_vabs())    ? "vabs"
                       : op->op.same_as(tl::simd_vexp())  ? "vexp"
                       : op->op.same_as(tl::simd_vln())   ? "vln"
                       : op->op.same_as(tl::simd_vsqrt()) ? "vsqrt"
                       : op->op.same_as(tl::simd_vneg())  ? "vneg"
                       : op->op.same_as(tl::simd_vrelu()) ? "vrelu"
                                                          : "vnot";
    PrimType input_elem = PythonDType(op->args[0]).WithLanes(1);
    int input_bits = input_elem.bits();
    if (input_bits < 8) {
      input_bits = 8;
    }
    std::string mask =
        "pto.pbitcast(" + RemoveOutermostParentheses(PrintExpr_(op->args[1])) +
        ", pto.mask_type(\"b" + std::to_string(input_bits) + "\"))";
    // Per-op SFU precision annotation (vexp/vln/vsqrt only; vabs/vneg/vrelu
    // have no precision tiers). 0 (hw) keeps the bare SFU call; 2 (ftz_false)
    // routes to the subnormal-preserving software wrapper. The shared
    // resolver guarantees ftz_false only applies to float32 operands.
    if (op->op.same_as(tl::simd_vexp()) || op->op.same_as(tl::simd_vln()) ||
        op->op.same_as(tl::simd_vsqrt())) {
      const SfuPrecision prec = ResolveSfuPrecision(
          GetRef<Call>(op), PythonDType(GetRef<Expr>(op)), SfuPrecision::kHw);
      if (prec == SfuPrecision::kKeepSub) {
        const char *helper =
            op->op.same_as(tl::simd_vexp())  ? "tl.vexp_1ulp_ftz_false"
            : op->op.same_as(tl::simd_vln()) ? "tl.vln_1ulp_ftz_false"
                                             : "tl.vsqrt_0ulp_ftz_false";
        os << helper << "(" << PrintExpr_(op->args[0]) << ", " << mask << ")";
        return;
      }
      ICHECK_EQ(static_cast<int>(prec), static_cast<int>(SfuPrecision::kHw))
          << "PTO tl.simd." << name
          << " only supports precision 0 (hw) and 2 (ftz_false), got "
          << static_cast<int>(prec);
    }
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ", " << mask
       << ")";
    return;
  }

  // --- SIMD predicate ops: pand/por/pxor/psel (src0, src1, mask);
  //     pnot (src, mask) → pto.<op>(...) ---
  if (op->op.same_as(tl::simd_pand()) || op->op.same_as(tl::simd_por()) ||
      op->op.same_as(tl::simd_pxor()) || op->op.same_as(tl::simd_pnot()) ||
      op->op.same_as(tl::simd_psel())) {
    bool is_pnot = op->op.same_as(tl::simd_pnot());
    ICHECK_EQ(op->args.size(), is_pnot ? 2U : 3U)
        << "tl.simd predicate op expects " << (is_pnot ? 2 : 3) << " arguments";
    const char *name = is_pnot                           ? "pnot"
                       : op->op.same_as(tl::simd_pand()) ? "pand"
                       : op->op.same_as(tl::simd_por())  ? "por"
                       : op->op.same_as(tl::simd_pxor()) ? "pxor"
                                                         : "psel";
    os << "pto." << name << "(" << PrintExpr_(op->args[0]);
    for (int i = 1; i < static_cast<int>(op->args.size()); ++i) {
      os << ", " << PrintExpr_(op->args[i]);
    }
    os << ")";
    return;
  }

  // Runtime tail predicate and predicate pack/unpack.
  if (op->op.same_as(tl::simd_update_mask())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.update_mask expects 2 arguments (value, width)";
    int64_t width = 0;
    ICHECK(TryGetConstInt(op->args[1], &width))
        << "tl.simd.update_mask width must be constant for PTO codegen";
    ICHECK(width == 8 || width == 16 || width == 32)
        << "PTO tl.simd.update_mask only supports widths 8, 16, and 32, got "
        << width;
    os << "pto.make_mask(" << DataTypeName(PrimType::UInt(width)) << ", "
       << PrintExpr_(op->args[0]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_ppack()) || op->op.same_as(tl::simd_punpack())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd predicate pack/unpack expects 2 arguments (src, part)";
    const auto *part_imm = op->args[1].as<StringImmNode>();
    ICHECK(part_imm)
        << "tl.simd predicate pack/unpack part must be a constant string for "
           "PTO codegen";
    ICHECK(part_imm->value == "LOWER" || part_imm->value == "HIGHER")
        << "PTO predicate pack/unpack part must be LOWER or HIGHER, got "
        << part_imm->value;
    const char *name = op->op.same_as(tl::simd_ppack()) ? "ppack" : "punpack";
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ", \""
       << part_imm->value << "\")";
    return;
  }

  // Carry and predicate-pair operations are emitted at their T.bind site so
  // each multi-result instruction is evaluated exactly once.
  if (op->op.same_as(tl::simd_vaddc()) || op->op.same_as(tl::simd_vsubc()) ||
      op->op.same_as(tl::simd_vaddcs()) || op->op.same_as(tl::simd_vsubcs()) ||
      op->op.same_as(tl::simd_vmull()) || op->op.same_as(tl::simd_pintlv()) ||
      op->op.same_as(tl::simd_pdintlv())) {
    LOG(FATAL) << op->op
               << " must be bound via T.bind for PTO codegen before its pair "
                  "elements are used";
    return;
  }

  // Leaky/parametric ReLU.
  if (op->op.same_as(tl::simd_vlrelu()) || op->op.same_as(tl::simd_vprelu())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd vlrelu/vprelu expects 3 arguments";
    if (op->op.same_as(tl::simd_vlrelu())) {
      PrimType vector_dtype = PythonDType(op->args[0]);
      std::string alpha = WrapTypedConst(
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[1]))),
          vector_dtype, IsImmediateScalar(op->args[1]));
      os << "pto.vlrelu(" << PrintExpr_(op->args[0]) << ", " << alpha << ", "
         << PrintExpr_(op->args[2]) << ")";
    } else {
      os << "pto.vprelu(" << PrintExpr_(op->args[0]) << ", "
         << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    }
    return;
  }

  // Histogram updates a mutable vector destination.
  if (op->op.same_as(tl::simd_chistv2()) ||
      op->op.same_as(tl::simd_dhistv2())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd histogram expects 4 arguments (dst, src, mask, bin)";
    const bool is_dhistv2 = op->op.same_as(tl::simd_dhistv2());
    const char *name = is_dhistv2 ? "dhistv2" : "chistv2";
    int64_t bin = 0;
    ICHECK(TryGetConstInt(op->args[3], &bin))
        << "tl.simd." << name << " bin must be a constant integer";
    ICHECK(bin == 0 || bin == 1)
        << "tl.simd." << name << " only supports bin 0 or 1, got " << bin;
    std::string dst_ref = GetMutableVectorRef(op->args[0], name);
    PrintIndent();
    stream << dst_ref << " = pto." << name << "(" << dst_ref << ", "
           << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
           << ", pto.const(" << bin << ", dtype=pto.int32))\n";
    return;
  }

  // --- SIMD vector-scalar ops: (src, scalar, mask, mode) → pto.<op>(src,
  // scalar, mask) ---
  if (op->op.same_as(tl::simd_vadds()) || op->op.same_as(tl::simd_vmuls()) ||
      op->op.same_as(tl::simd_vmaxs()) || op->op.same_as(tl::simd_vmins()) ||
      op->op.same_as(tl::simd_vshls()) || op->op.same_as(tl::simd_vshrs())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd vector-scalar op expects (src, scalar, mask, mode), got "
        << op->args.size() << " arguments";
    ValidateVecMode_(op, 3);
    const char *name = op->op.same_as(tl::simd_vadds())   ? "vadds"
                       : op->op.same_as(tl::simd_vmuls()) ? "vmuls"
                       : op->op.same_as(tl::simd_vmaxs()) ? "vmaxs"
                       : op->op.same_as(tl::simd_vmins()) ? "vmins"
                       : op->op.same_as(tl::simd_vshls()) ? "vshls"
                                                          : "vshrs";
    std::string src = PrintExpr_(op->args[0]);
    PrimType result_dtype = PythonDType(op->args[0]);
    std::string scalar = WrapTypedConst(
        RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[1]))),
        result_dtype, IsImmediateScalar(op->args[1]));
    std::string mask = PrintExpr_(op->args[2]);
    os << "pto." << name << "(" << src << ", " << scalar << ", " << mask << ")";
    return;
  }

  // Cross-lane reductions / lane squeeze: (src, mask, mode) -> pto.<op>(src,
  // mask)
  if (op->op.same_as(tl::simd_vcpadd()) || op->op.same_as(tl::simd_vcadd()) ||
      op->op.same_as(tl::simd_vcmax()) || op->op.same_as(tl::simd_vcmin()) ||
      op->op.same_as(tl::simd_vcgadd()) || op->op.same_as(tl::simd_vcgmax()) ||
      op->op.same_as(tl::simd_vcgmin()) || op->op.same_as(tl::simd_vsqz())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd cross-lane op expects 3 arguments (src, mask, mode)";
    ValidateVecMode_(op, 2,
                     op->op.same_as(tl::simd_vsqz()) ? "MODE_STORED"
                                                     : "MODE_ZEROING");
    const char *name = op->op.same_as(tl::simd_vcpadd())   ? "vcpadd"
                       : op->op.same_as(tl::simd_vcadd())  ? "vcadd"
                       : op->op.same_as(tl::simd_vcmax())  ? "vcmax"
                       : op->op.same_as(tl::simd_vcmin())  ? "vcmin"
                       : op->op.same_as(tl::simd_vcgadd()) ? "vcgadd"
                       : op->op.same_as(tl::simd_vcgmax()) ? "vcgmax"
                       : op->op.same_as(tl::simd_vcgmin()) ? "vcgmin"
                                                           : "vsqz";
    PrimType input_elem = PythonDType(op->args[0]).WithLanes(1);
    int input_bits = input_elem.bits();
    if (input_bits < 8) {
      input_bits = 8;
    }
    std::string mask =
        "pto.pbitcast(" + RemoveOutermostParentheses(PrintExpr_(op->args[1])) +
        ", pto.mask_type(\"b" + std::to_string(input_bits) + "\"))";
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ", " << mask
       << ")";
    return;
  }

  // Unsqueeze mask bits into an integer prefix-count vector. PTOAS keeps a
  // vector carrier operand for this operation, while TileLang only exposes the
  // predicate; use a zero-initialized carrier of the result type.
  if (op->op.same_as(tl::simd_vusqz())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.simd.vusqz expects 1 argument (mask)";
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    ICHECK((elem.MatchesCode(kDLInt) || elem.MatchesCode(kDLUInt)) &&
           (elem.bits() == 8 || elem.bits() == 16 || elem.bits() == 32))
        << "tl.simd.vusqz result must be an 8-, 16-, or 32-bit integer vector, "
           "got "
        << PythonDType(GetRef<Expr>(op));
    std::string mask = PrintExpr_(op->args[0]);
    std::string carrier = "pto.vdup(pto.const(0, dtype=" + DataTypeName(elem) +
                          "), " + mask + ")";
    os << "pto.vusqz(" << carrier << ", " << mask << ")";
    return;
  }

  // Elementwise compare -> predicate: (src0, src1, mask, op)
  if (op->op.same_as(tl::simd_vcmp())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vcmp expects 4 arguments (src0, src1, mask, op)";
    std::string cmp_op = (op->args[3]).as_or_throw<StringImm>()->value;
    os << "pto.vcmp(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ", \""
       << cmp_op << "\")";
    return;
  }

  // --- SIMD vector-scalar compare: (src, scalar, mask, op) → pto.vcmps(src,
  // scalar, mask, pto.CmpMode.<OP>) ---
  if (op->op.same_as(tl::simd_vcmps())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vcmps expects 4 arguments (src, scalar, mask, op)";
    std::string cmp_op = (op->args[3]).as_or_throw<StringImm>()->value;
    static const std::unordered_map<std::string, const char *> kCmpModeEnum = {
        {"eq", "EQ"}, {"ne", "NE"}, {"gt", "GT"},
        {"ge", "GE"}, {"lt", "LT"}, {"le", "LE"},
    };
    auto it = kCmpModeEnum.find(cmp_op);
    ICHECK(it != kCmpModeEnum.end())
        << "Unsupported cmp op '" << cmp_op
        << "' for tl.simd.vcmps; expected one of eq/ne/gt/ge/lt/le";
    os << "pto.vcmps(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
       << ", pto.CmpMode." << it->second << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vci())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.vci expects 2 arguments (index, order)";
    std::string order = (op->args[1]).as_or_throw<StringImm>()->value;
    // VPTO's parseOrderImmediate only accepts "ASC" / "DESC";
    // TileLang's simd layer uses "INC_ORDER" / "DEC_ORDER".
    if (order == "INC_ORDER")
      order = "ASC";
    else if (order == "DEC_ORDER")
      order = "DESC";
    ICHECK(order == "ASC" || order == "DESC")
        << "Unsupported tl.simd.vci order '" << order
        << "'; expected INC_ORDER/DEC_ORDER (or ASC/DESC)";
    // Match the authored vector element dtype (i16 for packed-ue8m0 lane math,
    // i32 for f32 gather paths). Always-i32 coercion breaks i16 vci users.
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    std::string index;
    if (IsImmediateScalar(op->args[0])) {
      index =
          "pto.const(" +
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0]))) +
          ", dtype=" + ScalarType(elem) + ")";
    } else {
      index =
          ScalarCastExpr(PrintExpr_(op->args[0]), elem, "tl.simd.vci index");
    }
    os << "pto.vci(" << index << ", order=\"" << order << "\")";
    return;
  }

  // Scalar broadcast: (scalar, mask, mode) -> pto.vdup(scalar, mask)
  if (op->op.same_as(tl::simd_vdup())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.vdup expects 3 arguments (src, mask, mode)";
    ValidateVecMode_(op, 2);
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    std::string scalar;
    if (IsImmediateScalar(op->args[0])) {
      scalar =
          "pto.const(" +
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0]))) +
          ", dtype=" + DataTypeName(elem) + ")";
    } else if (elem.MatchesCode(kDLInt) || elem.MatchesCode(kDLUInt) ||
               elem.MatchesCode(kDLBool)) {
      scalar = ScalarCastExpr(
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0]))),
          elem, "tl.simd.vdup scalar");
    } else {
      scalar =
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0])));
    }
    os << "pto.vdup(" << scalar << ", " << PrintExpr_(op->args[1]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vdupv())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vdupv expects 4 arguments (src, mask, pos, mode)";
    ValidateVecMode_(op, 3);
    std::string pos_str = (op->args[2]).as_or_throw<StringImm>()->value;
    std::string position = (pos_str == "POS_LOWEST") ? "LOWEST" : "HIGHEST";
    std::string scalar = WrapTypedConst(
        RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0]))),
        PythonDType(GetRef<Expr>(op)), IsImmediateScalar(op->args[0]));
    os << "pto.vdup(" << scalar << ", " << PrintExpr_(op->args[1])
       << ", position=\"" << position << "\")";
    return;
  }

  // Select: mask ? src0 : src1
  if (op->op.same_as(tl::simd_vsel())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.vsel expects 3 arguments (src0, src1, mask)";
    os << "pto.vsel(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    return;
  }

  // Lane select/reorder: (src, index)
  if (op->op.same_as(tl::simd_vselr())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.vselr expects 2 arguments (src, index)";
    os << "pto.vselr(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vmula()) || op->op.same_as(tl::simd_vmadd()) ||
      op->op.same_as(tl::simd_vaxpy())) {
    ICHECK_EQ(op->args.size(), 5U)
        << "PTO SIMD read-modify-write op expects 5 arguments "
           "(dst, src0, src1/scalar, mask, mode)";
    ValidateVecMode_(op, 4);
    const bool is_vmula = op->op.same_as(tl::simd_vmula());
    const bool is_vmadd = op->op.same_as(tl::simd_vmadd());
    std::string op_name = is_vmula ? "vmula" : (is_vmadd ? "vmadd" : "vaxpy");
    std::string dst_ref = GetMutableVectorRef(op->args[0], op_name);
    PrintIndent();
    stream << dst_ref << " = pto." << op_name << "(";
    if (is_vmula || is_vmadd) {
      stream << dst_ref << ", " << PrintExpr_(op->args[1]) << ", "
             << PrintExpr_(op->args[2]);
    } else {
      // PTODSL spells vaxpy as alpha * x + y; TileLang's destination is y.
      stream << PrintExpr_(op->args[2]) << ", " << PrintExpr_(op->args[1])
             << ", " << dst_ref;
    }
    stream << ", " << PrintExpr_(op->args[3]) << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_vpack())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.vpack expects 2 arguments (src, part)";
    std::string part = (op->args[1]).as_or_throw<StringImm>()->value;
    os << "pto.vpack(" << PrintExpr_(op->args[0]) << ", \"" << part << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_vunpack())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.vunpack expects 2 arguments (src, part)";
    int64_t part = -1;
    if (const auto *part_imm = op->args[1].as<IntImmNode>()) {
      part = part_imm->value.as<int64_t>().value();
    } else if (const auto *part_str = op->args[1].as<StringImmNode>()) {
      if (part_str->value == "LOWER") {
        part = 0;
      } else if (part_str->value == "HIGHER") {
        part = 1;
      }
    }
    ICHECK(part == 0 || part == 1)
        << "tl.simd.vunpack part must be LOWER/HIGHER or 0/1, got "
        << op->args[1];
    // PTOAS selects signed versus zero extension from the source vector type.
    os << "pto.vunpack(" << PrintExpr_(op->args[0]) << ", " << part << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vgatherb())) {
    ICHECK(op->args.size() == 2U || op->args.size() == 3U)
        << "tl.simd.vgatherb expects 2 or 3 arguments (base, index[, mask])";
    PrimType elem = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    std::string mask;
    if (op->args.size() == 3U) {
      mask = PrintExpr_(op->args[2]);
    } else {
      int elem_bits = elem.bits();
      if (elem_bits <= 8)
        elem_bits = 8;
      else if (elem_bits <= 16)
        elem_bits = 16;
      else
        elem_bits = 32;
      mask = "pto.pset_b" + std::to_string(elem_bits) + "(\"PAT_ALL\")";
    }
    os << "pto.vgatherb(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << mask
       << ", result_vreg_type=pto.vreg_type("
       << PythonDType(GetRef<Expr>(op)).lanes() << ", " << ScalarType(elem)
       << "))";
    return;
  }

  if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK(op->args.size() >= 4U && op->args.size() <= 6U)
        << "tl.simd.vcvt expects 4 to 6 arguments (src, mask, ...extra)";
    PrimType elem_dtype = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    // Parse trailing StringImm args: ROUND_*, SAT/NOSAT, PART_*, MODE_*
    std::string rnd, sat, part;
    for (size_t i = 2; i < op->args.size(); ++i) {
      std::string val = (op->args[i]).as_or_throw<StringImm>()->value;
      if (val.rfind("ROUND_", 0) == 0) {
        rnd = val.substr(6); // "ROUND_R" -> "R"
      } else if (val == "RS_ENABLE" || val == "RS_DISABLE") {
        sat = (val == "RS_ENABLE") ? "SAT" : "NOSAT";
      } else if (val.rfind("PART_", 0) == 0) {
        part = val.substr(5); // "PART_EVEN" -> "EVEN"
      } else if (val == "MODE_ZEROING") {
        // validated and dropped (ptodsl has no mode param)
      } else if (val == "MODE_MERGING") {
        LOG(FATAL) << "PTO codegen does not support MODE_MERGING for vcvt yet";
      } else {
        LOG(FATAL) << "PTO codegen: unrecognized vcvt argument: " << val;
      }
    }
    PrimType src_elem_dtype = PythonDType(op->args[0]).WithLanes(1);
    int src_bits = src_elem_dtype.bits();
    if (src_bits < 8) {
      src_bits = 8; // CCE has no pset_b4
    }
    std::string mask =
        "pto.pbitcast(" + RemoveOutermostParentheses(PrintExpr_(op->args[1])) +
        ", pto.mask_type(\"b" + std::to_string(src_bits) + "\"))";
    os << "pto.vcvt(" << PrintExpr_(op->args[0]) << ", "
       << ScalarType(elem_dtype) << ", " << mask;
    if (!rnd.empty()) {
      os << ", rnd=\"" << rnd << "\"";
    }
    if (!sat.empty()) {
      os << ", sat=\"" << sat << "\"";
    }
    if (!part.empty()) {
      os << ", part=\"" << part << "\"";
    }
    os << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vsts())) {
    ICHECK(op->args.size() >= 3U && op->args.size() <= 5U)
        << "tl.simd.vsts expects 3 to 5 arguments "
           "(addr, src, mask[, dist][, offset])";
    std::string dist;
    int offset_index = -1;
    if (op->args.size() >= 4U && op->args[3].as<StringImmNode>()) {
      dist = (op->args[3]).as_or_throw<StringImm>()->value;
      if (dist.rfind("ONEPT_", 0) == 0) {
        dist = "1PT_" + dist.substr(6);
      }
      if (op->args.size() == 5U) {
        offset_index = 4;
      }
    } else if (op->args.size() == 4U) {
      offset_index = 3;
    }
    std::string offset =
        offset_index >= 0
            ? RemoveOutermostParentheses(PrintExpr_(op->args[offset_index]))
            : "pto.const(0)";
    std::string vsts_mask = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
    const int mask_granularity = VstsMaskGranularityOverride(
        dist, PythonDType(op->args[1]).WithLanes(1));
    if (mask_granularity != 0) {
      vsts_mask = "pto.pbitcast(" + vsts_mask + ", pto.mask_b" +
                  std::to_string(mask_granularity) + ")";
    }
    PrintIndent();
    stream << "pto.vsts(" << PrintExpr_(op->args[1]) << ", "
           << PrintExpr_(op->args[0]) << ", " << offset << ", " << vsts_mask;
    if (!dist.empty()) {
      // Normalize legacy AscendC dist naming (ONEPT_ → 1PT_)
      // so ptodsl accepts it.
      if (dist.rfind("ONEPT_", 0) == 0) {
        dist = "1PT_" + dist.substr(6);
      }
      stream << ", dist=\"" << dist << "\"";
    }
    stream << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_vsstb())) {
    ICHECK(op->args.size() == 4U || op->args.size() == 5U)
        << "tl.simd.vsstb expects 4 or 5 arguments "
           "(src, base, packed_stride, mask[, POST_UPDATE])";
    const bool post_update = op->args.size() == 5U;
    ICHECK_EQ(PythonIsPointer(GetRef<Expr>(op)), post_update)
        << "tl.simd.vsstb post-update form must return a handle";
    PrimType stride_dtype = PythonDType(op->args[2]);
    ICHECK(stride_dtype.IsScalar() &&
           (stride_dtype.MatchesCode(kDLInt) ||
            stride_dtype.MatchesCode(kDLUInt)) &&
           stride_dtype.bits() <= 32)
        << "tl.simd.vsstb packed stride must be a scalar integer no wider "
           "than 32 bits, got "
        << stride_dtype;

    // AscendC exposes vsstb's two 16-bit stride fields as one packed int32:
    //   packed_stride[31:16] = block_stride
    //   packed_stride[15:0]  = repeat_stride
    // PTODSL models the fields separately, so preserve the AscendC/TIR API
    // semantics by unpacking them here.
    std::string block_stride;
    std::string repeat_stride;
    int64_t packed_stride = 0;
    if (TryGetConstInt(op->args[2], &packed_stride)) {
      uint32_t packed_bits = static_cast<uint32_t>(packed_stride);
      block_stride = std::to_string((packed_bits >> 16) & 0xffffU);
      repeat_stride = std::to_string(packed_bits & 0xffffU);
    } else {
      std::string stride = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
      // RuntimeValue implements integer floor division and bitwise-and, but
      // not Python's shift operators. Dividing by 2^16 and masking is
      // equivalent to extracting bits [31:16] for a 32-bit packed value,
      // including negative signed values.
      block_stride = "(((" + stride + ") // 65536) & 0xffff)";
      repeat_stride = "((" + stride + ") & 0xffff)";
    }

    std::string src = PrintExpr_(op->args[0]);
    std::string base = PrintExpr_(op->args[1]);
    std::string mask = PrintExpr_(op->args[3]);
    if (post_update) {
      const auto *post = op->args[4].as<StringImmNode>();
      ICHECK(post && post->value == "POST_UPDATE")
          << "tl.simd.vsstb POST_UPDATE argument must be POST_UPDATE";
      os << "pto.vsstb(" << src << ", " << base << ", " << block_stride << ", "
         << repeat_stride << ", " << mask << ", post_update=pto.PostUpdate.ON)";
    } else {
      PrintIndent();
      stream << "pto.vsstb(" << src << ", " << base << ", " << block_stride
             << ", " << repeat_stride << ", " << mask << ")\n";
    }
    return;
  }

  if (op->op.same_as(tl::simd_vscatter())) {
    ICHECK_EQ(op->args.size(), 4U) << "tl.simd.vscatter expects 4 arguments "
                                      "(src, base, offsets, mask)";
    // The TileLang surface accepts a byte-addressed buffer for packed
    // storage, while PTO's vscatter requires the destination pointer element
    // type to match the value register element type. Keep the authored byte
    // offsets intact and only refine the pointer type for the verifier (the
    // base expression is already an UB address produced by access_ptr).
    PrimType value_dtype = PythonDType(op->args[0]).WithLanes(1);
    std::string base = PrintExpr_(op->args[1]);
    PrimType base_dtype = GetAnnotatedPointerDtype(op->args[1], value_dtype);
    if (base_dtype != value_dtype) {
      base = "pto.castptr(" + base + ", " + PointerTypeName(value_dtype, "ub") +
             ")";
    }
    PrintIndent();
    stream << "pto.vscatter(" << PrintExpr_(op->args[0]) << ", " << base << ", "
           << PrintExpr_(op->args[2]) << ", " << PrintExpr_(op->args[3])
           << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_mem_bar())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.simd.mem_bar expects 1 argument (barrier type string)";
    std::string barrier_type = (op->args[0]).as_or_throw<StringImm>()->value;
    PrintIndent();
    stream << "pto.mem_bar(\"" << barrier_type << "\")\n";
    return;
  }

  if (op->op.same_as(tl::loop_break())) {
    ICHECK_GT(native_loop_depth_, 0)
        << "tl.loop_break encountered outside a native Python loop";
    PrintIndent();
    stream << "break\n";
    return;
  }

  if (op->op.same_as(tl::simd_vexpdif())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.vexpdif expects 3 arguments (src0, src1, mask)";
    std::string src0 = PrintExpr_(op->args[0]);
    std::string src1 = PrintExpr_(op->args[1]);
    std::string mask = PrintExpr_(op->args[2]);
    os << "pto.vexpdif(" << src0 << ", " << src1 << ", " << mask << ", \""
       << "EVEN" << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_vgather2())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.vgather2 expects 3 arguments (base, index, mask)";
    std::string base = PrintExpr_(op->args[0]);
    std::string index = PrintExpr_(op->args[1]);
    PrimType elem_dtype = PythonDType(GetRef<Expr>(op)).WithLanes(1);
    // vgather2 widens byte sources to 16-bit result lanes. Its PTO predicate
    // is defined at the result element width, whereas the TileLang intrinsic
    // carries the source-byte mask (b8). Reinterpret the predicate at the
    // result width so PTODSL/MLIR sees the required mask type without changing
    // the selected lane bits.
    int mask_bits = elem_dtype.bits();
    if (mask_bits < 8) {
      mask_bits = 8;
    }
    std::string mask =
        "pto.pbitcast(" + RemoveOutermostParentheses(PrintExpr_(op->args[2])) +
        ", pto.mask_type(\"b" + std::to_string(mask_bits) + "\"))";
    std::string result_type =
        "pto.vreg_type(" +
        std::to_string(PythonDType(GetRef<Expr>(op)).lanes()) + ", " +
        ScalarType(elem_dtype) + ")";
    os << "pto.vgather2(" << base << ", " << index << ", " << mask << ", "
       << result_type << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vintlv()) || op->op.same_as(tl::simd_vdintlv())) {
    // Pair-producing ops are handled in VisitStmt_(BindNode) where the tuple
    // result is unpacked into two SSA names. Reaching here means the call was
    // not bound (e.g. inlined), which is unsupported for PTO codegen.
    LOG(FATAL) << "tl.simd.vintlv/vdintlv must be bound via T.bind for PTO "
               << "codegen (use `a, b = T.simd.vintlv(x, y)`)";
    return;
  }

  if (op->op.same_as(tl::simd_vld2())) {
    LOG(FATAL) << "tl.simd.vld2 must be bound via T.bind for PTO codegen "
               << "(use `a, b = T.simd.vld2(addr, dist=...)`)";
    return;
  }

  if (op->op.same_as(tl::simd_pair_get())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.pair_get expects 2 arguments (pair, index)";
    const auto *var = op->args[0].as<VarNode>();
    ICHECK(var != nullptr)
        << "tl.simd.pair_get expects a bound pair variable as arg0";
    auto it = simd_pair_vars_.find(var);
    ICHECK(it != simd_pair_vars_.end())
        << "tl.simd.pair_get references an unknown pair variable " << var->name
        << " in " << GetRef<Call>(op);
    int64_t index =
        (op->args[1]).as_or_throw<IntImm>()->value.as<int64_t>().value();
    os << (index == 0 ? it->second.first : it->second.second);
    return;
  }

  if (op->op.same_as(tl::rng_init())) {
    EmitRngInit(op);
    return;
  }

  if (op->op.same_as(tl::rng_rand())) {
    os << EmitRngRand(op);
    return;
  }

  if (op->op.same_as(tl::rng_rand_float())) {
    os << EmitRngRandFloat(op);
    return;
  }

  if (auto opt_call_op = op->op.as<Op>()) {
    const auto &call_op = opt_call_op.value();
    std::string op_name = call_op->name;
    if (op->args.size() == 1U && TryEmitUnaryMath_(op_name, op->args[0], os)) {
      return;
    }
    if (StartsWith(op_name, "tl.")) {
      LOG(FATAL) << "PTO codegen does not support TileLang op `" << op_name
                 << "`. Please add a handler in CodeGenTileLangPTO or lower "
                    "it before PTO codegen.";
    }
  }

  CodeGenTileLangPY::Dispatch_(op, os);
}

void CodeGenTileLangPTO::Dispatch_(const FloatImmNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (!PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16)) {
    CodeGenTileLangPY::Dispatch_(op, os);
    return;
  }

  // Python has no bfloat16 scalar constructor.  Keep the literal as a Python
  // float here; PTODSL operations such as pto.store and pto.const coerce it
  // to the destination/authored bf16 type at the use site.
  std::ostringstream literal;
  literal << "float.fromhex('" << FlexibleHexFormat(op->value) << "')";
  MarkConst(literal.str());
  os << literal.str();
}

void CodeGenTileLangPTO::Dispatch_(const CastNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrimType from = PythonDType(op->value);
  PrimType to = PythonDType(GetRef<Expr>(op));
  ICHECK_EQ(to.lanes(), from.lanes())
      << "PTO cast expects source and target to have the same lane count, got "
      << from << " -> " << to;

  if (from == to) {
    PrintExpr_(op->value, os);
    return;
  }

  const bool from_fp8 = tl::IsAscendVectorizableFP8(from);
  const bool to_fp8 = tl::IsAscendVectorizableFP8(to);
  if (from_fp8 || to_fp8) {
    ICHECK(inside_simtvf_body_)
        << "PTO FP8 cast is currently supported only inside SIMT, got " << from
        << " -> " << to;
    if (from.MatchesCode(kDLFloat) && from.bits() == 32 && from.lanes() == 2 &&
        to_fp8) {
      os << "pto.cast(";
      PrintExpr_(op->value, os);
      os << ", " << FP8TypeName(to) << ", rounding=\"r\", saturation=\"sat\")";
      return;
    }
    LOG(FATAL) << "PTO SIMT FP8 cast currently supports float32x2 to "
                  "float8_e4m3fnx2/float8_e5m2x2 only, got "
               << from << " -> " << to;
  }

  const bool from_integer = from.MatchesCode(kDLInt) ||
                            from.MatchesCode(kDLUInt) ||
                            from.MatchesCode(kDLBool);
  const bool to_integer = to.MatchesCode(kDLInt) || to.MatchesCode(kDLUInt) ||
                          to.MatchesCode(kDLBool);
  const bool from_float =
      from.MatchesCode(kDLFloat) || from.MatchesElementType(kDLBfloat, 16);
  const bool to_float =
      to.MatchesCode(kDLFloat) || to.MatchesElementType(kDLBfloat, 16);
  const bool immediate_value = op->value.as<IntImmNode>() != nullptr ||
                               op->value.as<FloatImmNode>() != nullptr;
  const bool scalar_value = from.IsScalar() && to.IsScalar();
  if (scalar_value && from_integer && to.MatchesCode(kDLFloat) &&
      (from.MatchesCode(kDLBool) || from.bits() == 1)) {
    // Python float(runtime_i1) tries to coerce a device SSA value at trace
    // time.  Materialize the numeric boolean conversion on the device.
    os << "pto.select(" << PrintCondition(op->value)
       << ", pto.const(1.0, dtype=" << DataTypeName(to)
       << "), pto.const(0.0, dtype=" << DataTypeName(to) << "))";
    return;
  }
  if (scalar_value && from_integer && to.MatchesCode(kDLBool)) {
    // The bool result must be normalized to 0/1, not the raw integer.
    os << "tl.as_logical_bool(" << PrintExpr_(op->value) << ")";
    return;
  }
  if (scalar_value && from.MatchesCode(kDLBool) &&
      (to.MatchesCode(kDLInt) || to.MatchesCode(kDLUInt))) {
    // Materialize the boolean as a normalized 0/1 integer.
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << "pto.const(" << imm->value << ", dtype=" << DataTypeName(to) << ")";
      return;
    }
    os << "pto.select(" << PrintCondition(op->value)
       << ", pto.const(1, dtype=" << DataTypeName(to)
       << "), pto.const(0, dtype=" << DataTypeName(to) << "))";
    return;
  }
  if (!immediate_value && scalar_value &&
      ((from_float && to_integer) || (from_integer && to_float))) {
    os << ScalarCastExpr(PrintExpr_(op->value), to, "PTO scalar cast");
    return;
  }
  if (!immediate_value && from_float && to_float) {
    // Keep MLIR implementation details inside PTODSL.  pto.cast preserves
    // the shape of builtin vectors while replacing their element dtype.  Use
    // the full element dtype instead of bit width so same-width conversions
    // such as float16 <-> bfloat16 are also lowered on the device.
    os << ScalarCastExpr(PrintExpr_(op->value), to.WithLanes(1),
                         "PTO floating-point cast");
    return;
  }
  if (scalar_value && from_integer && to_integer &&
      !from.MatchesCode(kDLBool) && !to.MatchesCode(kDLBool) &&
      (from.bits() != to.bits() ||
       from.MatchesCode(kDLUInt) != to.MatchesCode(kDLUInt))) {
    // Python int(runtime_value) tries to consume a device-side SSA value while
    // PTODSL is tracing. Use pto.cast for both width changes and same-width
    // signedness changes so later scalar ops see the authored int/uint type.
    os << ScalarCastExpr(PrintExpr_(op->value), to, "PTO integer cast");
    return;
  }
  if (from_integer && to_integer && from.bits() <= to.bits()) {
    // Address-index widening should not become Python int(...), because that
    // coerces PTODSL runtime values and breaks tracing. Same-width int/uint
    // casts are handled above to preserve signedness for downstream scalar
    // lowering.
    PrintExpr_(op->value, os);
    return;
  }

  if (to.MatchesCode(kDLBool)) {
    ICHECK(from.IsScalar() && (from_integer || from_float))
        << "PTO bool cast expects a scalar numeric source, got " << from;
    // Integer sources already normalize above; runtime floating-point
    // sources already use ScalarCastExpr.
    const auto *imm = op->value.as<FloatImmNode>();
    ICHECK(imm) << "PTO bool cast expected a floating-point immediate";
    os << (imm->value != 0.0 ? "True" : "False");
    return;
  }

  if (from_integer && to_integer) {
    // All scalar integer casts were handled above. Only unsupported vector
    // narrowing remains; preserve its diagnostic instead of emitting a cast.
    LOG(FATAL) << "PTO integer cast currently supports scalar values only, got "
               << from << " -> " << to;
  }

  if (from_integer && to_float) {
    ICHECK(from.IsScalar() && to.IsScalar())
        << "PTO integer-to-float cast currently supports scalar values only, "
           "got "
        << from << " -> " << to;
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << "pto.const(" << imm->value << ", dtype=" << DataTypeName(to) << ")";
      return;
    }
    // Non-immediate scalar integer-to-float casts returned above through
    // ScalarCastExpr; only the rejected vector cases can reach here.
  }

  if (const auto *imm = op->value.as<FloatImmNode>();
      imm && scalar_value &&
      ((IsFloat32(from) && to.MatchesElementType(kDLBfloat, 16)) ||
       (from.MatchesElementType(kDLBfloat, 16) && IsFloat32(to)))) {
    os << "pto.const(float.fromhex('" << FlexibleHexFormat(imm->value)
       << "'), dtype=" << DataTypeName(to) << ")";
    return;
  }

  LOG(FATAL) << "Unsupported PTO cast: " << from << " -> " << to;
}

void CodeGenTileLangPTO::Dispatch_(const LetNode *op,
                                   std::ostream &os) { // NOLINT(*)
  // Keep Let expression-local.  In particular, a Let can appear inside a lazy
  // if_then_else branch; writing its binding to the function stream would
  // hoist potentially unsafe work (for example, division by zero) outside the
  // branch callback.  A Python lambda preserves Let's value-before-body order
  // and is evaluated wherever the containing expression is evaluated.
  std::string value = PrintExpr_(op->value);
  ICHECK(!var_idmap_.count(op->var.get()))
      << "PTO Let variable is already defined: " << op->var->name;

  std::string vid = AllocVarID(op->var.get());
  std::string body = PrintExpr_(op->body);

  bool removed = var_idmap_.erase(op->var.get());
  ICHECK(removed) << "PTO Let variable was not registered: " << op->var->name;

  os << "(lambda " << vid << ": " << body << ")(" << value << ")";
}

void CodeGenTileLangPTO::Dispatch_(const ShuffleNode *op,
                                   std::ostream &os) { // NOLINT(*)
  // Shuffle picks lanes by constant index from its concatenated inputs;
  // tl.shuffle_vec gathers them into a scalar or packed vector.
  int64_t total_lanes = 0;
  for (const Expr &vector : op->vectors) {
    PrimType dtype = PythonDType(vector);
    if (dtype.lanes() > 1) {
      ICHECK(!tl::IsAscendVectorizableFP8(dtype))
          << "PTO cannot split FP8 builtin vectors lane-by-lane, got " << dtype;
    }
    total_lanes += dtype.lanes();
  }
  ICHECK_GE(total_lanes, 1) << "PTO Shuffle expects at least one input lane";
  for (const PrimExpr &index : op->indices) {
    const auto *constant_index = index.as<IntImmNode>();
    ICHECK(constant_index != nullptr)
        << "ShuffleNode indices must be constants at PTO codegen time, got "
        << index;
    ICHECK_GE(constant_index->value, 0)
        << "ShuffleNode index must be non-negative, got "
        << constant_index->value;
    ICHECK_LT(constant_index->value, total_lanes)
        << "ShuffleNode index " << constant_index->value
        << " is outside the concatenated input with " << total_lanes
        << " lanes";
  }

  os << "tl.shuffle_vec("
     << DataTypeName(PythonDType(GetRef<Expr>(op)).WithLanes(1)) << ", (";
  for (size_t i = 0; i < op->vectors.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    PrintExpr_(op->vectors[i], os);
  }
  // A single-element Python tuple requires a trailing comma: (value,).
  if (op->vectors.size() == 1) {
    os << ",";
  }
  os << "), (";
  for (size_t i = 0; i < op->indices.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << (op->indices[i]).as_or_throw<IntImm>()->value;
  }
  if (op->indices.size() == 1) {
    os << ",";
  }
  os << "))";
}

void CodeGenTileLangPTO::Dispatch_(const MinNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (inside_simtvf_body_ &&
      tl::IsAscendVectorizableFP8(PythonDType(GetRef<Expr>(op)))) {
    LOG(FATAL) << "PTO SIMT scalar FP8 min is not supported; only +, -, and * "
                  "between non-immediate operands of the same FP8 dtype are "
                  "currently supported, got "
               << PythonDType(op->a) << " and " << PythonDType(op->b);
  }
  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
    os << "pto.min(";
    PrintExpr_(op->a, os);
    os << ", ";
    PrintExpr_(op->b, os);
    os << ")";
    return;
  }
  PrintFloatMinMax_("minimum", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}

void CodeGenTileLangPTO::Dispatch_(const MaxNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (inside_simtvf_body_ &&
      tl::IsAscendVectorizableFP8(PythonDType(GetRef<Expr>(op)))) {
    LOG(FATAL) << "PTO SIMT scalar FP8 max is not supported; only +, -, and * "
                  "between non-immediate operands of the same FP8 dtype are "
                  "currently supported, got "
               << PythonDType(op->a) << " and " << PythonDType(op->b);
  }
  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
    os << "pto.max(";
    PrintExpr_(op->a, os);
    os << ", ";
    PrintExpr_(op->b, os);
    os << ")";
    return;
  }
  PrintFloatMinMax_("maximum", PythonDType(GetRef<Expr>(op)), op->a, op->b, os);
}

void CodeGenTileLangPTO::PrintFloatMinMax_(const char *op_name, PrimType dtype,
                                           PrimExpr lhs, PrimExpr rhs,
                                           std::ostream &os) { // NOLINT(*)
  ICHECK(IsSupportedFloatMinMaxType(dtype))
      << "PTO floating min/max currently supports f16, f32, bf16, "
         "vector<2xf16>, vector<2xf32>, and vector<2xbf16>, got "
      << dtype;

  // PTO's packed min/max micro-ops omit f32x2. Apply the supported scalar f32
  // operation lane-by-lane and repack the pair instead.
  if (inside_simtvf_body_ && IsFloat32Pair(dtype)) {
    os << "tl.vectorize_binary_f32x2(pto." << op_name << ", ";
    PrintExpr_(lhs, os);
    os << ", ";
    PrintExpr_(rhs, os);
    os << ")";
    return;
  }

  os << "pto." << op_name << "(";
  PrintExpr_(lhs, os);
  os << ", ";
  PrintExpr_(rhs, os);
  os << ")";
}

bool CodeGenTileLangPTO::TryEmitUnaryMath_(const std::string &name,
                                           const Expr &arg,
                                           std::ostream &os) { // NOLINT(*)
  // Unary math mapping shared by the extern C path (call_pure_extern) and
  // the tirx intrinsic path. Add new functions as table entries only.
  struct UnaryMathForm {
    const char *function;
    bool reciprocal; // emit (1.0 / fn(x)) instead of fn(x)
  };
  static const std::unordered_map<std::string, UnaryMathForm> kForms = {
      {"expf", {"pto.exp(", false}},
      {"tirx.exp", {"pto.exp(", false}},
      {"logf", {"pto.log(", false}},
      {"tirx.log", {"pto.log(", false}},
      {"sqrt", {"pto.sqrt(", false}},
      {"sqrtf", {"pto.sqrt(", false}},
      {"tirx.sqrt", {"pto.sqrt(", false}},
      {"rsqrt", {"pto.sqrt(", true}},
      {"rsqrtf", {"pto.sqrt(", true}},
      {"tirx.rsqrt", {"pto.sqrt(", true}},
      {"fabsf", {"pto.absf(", false}},
      {"tirx.fabs", {"pto.absf(", false}},
      // f16 extern names produced by AscendMath (intrin_rule_ascend.cc).
      {"hexp", {"pto.exp(", false}},
      {"hfabs", {"pto.absf(", false}},
      {"hlog", {"pto.log(", false}},
      {"hsqrt", {"pto.sqrt(", false}},
      {"hrsqrt", {"pto.sqrt(", true}},
  };
  auto it = kForms.find(name);
  if (it == kForms.end()) {
    return false;
  }
  PrimType dtype = PythonDType(arg);
  if (inside_simtvf_body_) {
    ICHECK(IsSupportedSIMTUnaryMathType(dtype))
        << "PTO SIMT " << name
        << " currently supports f16, f32, vector<2xf16>, and vector<2xf32>, "
           "got "
        << dtype;
  } else {
    ICHECK(IsSupportedScalarUnaryMathType(dtype))
        << "PTO scalar " << name
        << " currently supports scalar f16, f32, and bf16, got " << dtype;
  }
  const UnaryMathForm &form = it->second;
  std::string value = PrintExpr_(arg);
  // PTO's packed unary micro-ops omit f32x2. Scalarize these operations just
  // like min/max above so PTOAS receives only supported scalar f32 ops.
  if (inside_simtvf_body_ && IsFloat32Pair(dtype)) {
    if (form.reciprocal) {
      os << "tl.vectorize_unary_f32x2(tl.scalar_rsqrt, " << value << ")";
    } else {
      std::string fn = form.function;
      ICHECK(!fn.empty() && fn.back() == '(');
      fn.pop_back();
      os << "tl.vectorize_unary_f32x2(" << fn << ", " << value << ")";
    }
    return true;
  }
  if (form.reciprocal) {
    os << "(1.0 / " << form.function << value << "))";
  } else {
    os << form.function << value << ")";
  }
  return true;
}

void CodeGenTileLangPTO::Dispatch_(const NotNode *op,
                                   std::ostream &os) { // NOLINT(*)
  ICHECK(PythonDType(GetRef<Expr>(op)).IsScalar() &&
         PythonDType(op->a).MatchesCode(kDLBool))
      << "PTO logical-not currently supports scalar bool predicates only, got "
      << PythonDType(op->a);
  // tl.logical_not(value) implements logical negation as value == 0.
  os << "tl.logical_not(";
  PrintExpr_(op->a, os);
  os << ")";
}

void CodeGenTileLangPTO::Dispatch_(const SelectNode *op,
                                   std::ostream &os) { // NOLINT(*)
  const bool coerce_integer_branches =
      PythonDType(GetRef<Expr>(op)).IsScalar() &&
      (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
       PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt));
  auto print_select_value = [&](const Expr &expr) {
    if (expr.as<IntImmNode>() != nullptr ||
        expr.as<FloatImmNode>() != nullptr) {
      os << "pto.const(" << PrintExpr_(expr)
         << ", dtype=" << DataTypeName(PythonDType(GetRef<Expr>(op))) << ")";
      return;
    }
    if (PythonDType(GetRef<Expr>(op)).IsScalar() &&
        PythonDType(GetRef<Expr>(op)).MatchesCode(kDLBool)) {
      os << "tl.as_logical_bool(" << PrintExpr_(expr) << ")";
      return;
    }
    if (coerce_integer_branches) {
      os << ScalarCastExpr(PrintExpr_(expr), PythonDType(GetRef<Expr>(op)),
                           "PTO select branch");
      return;
    }
    PrintExpr_(expr, os);
  };
  os << "pto.select(";
  os << PrintCondition(op->condition);
  os << ", ";
  print_select_value(op->true_value);
  os << ", ";
  print_select_value(op->false_value);
  os << ")";
}

void CodeGenTileLangPTO::Dispatch_(const ModNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)).IsScalar() &&
      (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
       PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt))) {
    os << "(" << PrintExpr_(op->a) << " % " << PrintExpr_(op->b) << ")";
    return;
  }
  CodeGenTileLangPY::Dispatch_(op, os);
}

void CodeGenTileLangPTO::Dispatch_(const DivNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (PythonDType(GetRef<Expr>(op)).IsScalar() &&
      PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt)) {
    os << "pto.div(" << PrintExpr_(op->a) << ", " << PrintExpr_(op->b) << ")";
    return;
  }
  CodeGenTileLangPY::Dispatch_(op, os);
}

void CodeGenTileLangPTO::PrintBinaryExpr_(const std::string &opstr,
                                          PrimType dtype, Expr lhs, Expr rhs,
                                          std::ostream &os) { // NOLINT(*)
  // Some lowering paths emit "%" via PrintBinaryExpr_ without going through
  // VisitExpr_(Mod). Keep integer remainder as an authored runtime `%` so
  // PTODSL preserves signedness and width (RemSI vs RemUI).
  if (dtype.IsScalar() &&
      (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) &&
      opstr == "%") {
    os << "(" << PrintExpr_(lhs) << " % " << PrintExpr_(rhs) << ")";
    return;
  }
  if (dtype.IsScalar()) {
    PrimType lhs_dtype = PythonDType(lhs);
    PrimType rhs_dtype = PythonDType(rhs);
    bool lhs_fp8 = tl::IsAscendVectorizableFP8(lhs_dtype);
    bool rhs_fp8 = tl::IsAscendVectorizableFP8(rhs_dtype);
    bool result_fp8 = tl::IsAscendVectorizableFP8(dtype);
    if (inside_simtvf_body_ && (lhs_fp8 || rhs_fp8 || result_fp8)) {
      bool supported_op = opstr == "+" || opstr == "-" || opstr == "*";
      bool has_immediate = lhs.as<IntImmNode>() != nullptr ||
                           lhs.as<FloatImmNode>() != nullptr ||
                           rhs.as<IntImmNode>() != nullptr ||
                           rhs.as<FloatImmNode>() != nullptr;
      if (supported_op && lhs_fp8 && rhs_fp8 && lhs_dtype == rhs_dtype &&
          result_fp8 && !has_immediate) {
        os << "tl.scalar_binary_fp8('" << opstr << "', " << PrintExpr_(lhs)
           << ", " << PrintExpr_(rhs) << ", " << FP8TypeName(dtype) << ")";
        return;
      }
      LOG(FATAL) << "Unsupported PTO SIMT scalar FP8 binary op '" << opstr
                 << "' with lhs dtype " << lhs_dtype << ", rhs dtype "
                 << rhs_dtype << ", and result dtype " << dtype
                 << "; only +, -, and * between non-immediate operands of "
                    "the same FP8 dtype are currently supported";
    }
    CodeGenTileLangPY::PrintBinaryExpr_(opstr, dtype, lhs, rhs, os);
    return;
  }

  ICHECK(inside_simtvf_body_)
      << "PTO vector binary expressions are only supported inside SIMT bodies";
  if (opstr == "/") {
    ICHECK(IsFloat32Pair(dtype))
        << "PTO SIMT vector division currently supports vector<2xf32> only, "
           "got "
        << dtype;
    os << "tl.vectorize_binary_f32x2(tl.scalar_div, " << PrintExpr_(lhs) << ", "
       << PrintExpr_(rhs) << ")";
    return;
  }
  if (opstr != "+" && opstr != "-" && opstr != "*") {
    LOG(FATAL) << "Unsupported PTO SIMT vector binary op: " << opstr;
  }
  if (tl::IsAscendVectorizableFP8(dtype)) {
    os << "tl.vectorize_binary_fp8('" << opstr << "', " << PrintExpr_(lhs)
       << ", " << PrintExpr_(rhs) << ", " << FP8TypeName(dtype.WithLanes(1))
       << ")";
    return;
  }
  os << "(" << PrintExpr_(lhs) << " " << opstr << " " << PrintExpr_(rhs) << ")";
}

void CodeGenTileLangPTO::Dispatch_(const RampNode *op,
                                   std::ostream &os) { // NOLINT(*)
  // Keep operands expression-local and evaluate each once, even for effectful
  // calls. A Ramp used as a value must remain a typed vector.
  os << "(lambda _ramp_base, _ramp_stride: tl.vector_from_list("
     << DataTypeName(PythonDType(GetRef<Expr>(op)).WithLanes(1)) << ", (";
  for (int lane = 0; lane < PythonDType(GetRef<Expr>(op)).lanes(); ++lane) {
    if (lane != 0) {
      os << ", ";
    }
    os << "(_ramp_base)+(_ramp_stride*" << lane << ")";
  }
  os << ")))(" << PrintExpr_(op->base) << ", " << PrintExpr_(op->stride) << ")";
}

void CodeGenTileLangPTO::Dispatch_(const BroadcastNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrimType elem_dtype = PythonDType(op->value);
  ICHECK(elem_dtype.MatchesElementType(kDLFloat, 16) || IsFloat32(elem_dtype) ||
         elem_dtype.MatchesElementType(kDLBfloat, 16))
      << "PTO vector broadcast currently supports f16, f32, and bf16 only, got "
      << elem_dtype;
  std::string value = PrintExpr_(op->value);
  os << "pto.Vec(" << DataTypeName(elem_dtype) << ", "
     << PythonDType(GetRef<Expr>(op)).lanes() << ", init=" << value << ")";
}

void CodeGenTileLangPTO::Dispatch_(const BindNode *op) {
  if (auto allocation = op->value.as<CallNode>();
      allocation && allocation->op.same_as(tirx::alloc_tensor_op())) {
    EmitAllocation(op->var.as_or_throw<TensorVar>());
    return;
  }
  if (const auto *call = op->value.as<CallNode>()) {
    if (call->op.same_as(tirx::decl_tensor_op())) {
      auto tensor = op->var.as_or_throw<TensorVar>();
      PrimExpr index(0);
      const VarNode *base = nullptr;
      bool integer_address = GetAddressOfIndex(call->args[0], &index, &base) &&
                             integer_address_vars_.count(GetRef<Var>(base));
      auto source = integer_address
                        ? GetPointerExpr(base,
                                         GetAnnotatedPointerDtype(
                                             call->args[0], tensor.dtype()),
                                         index)
                        : PrintExpr_(call->args[0]);
      if (integer_address)
        integer_address_vars_.insert(tensor);
      RegisterHandleType_(tensor.get(), tensor.dtype());
      alloc_storage_scope_[tensor.get()] = tensor.scope().name();
      PrintIndent();
      stream << AllocVarID(tensor.get()) << " = " << source << "\n";
      return;
    }
    if (IsOpName(call->op, "tl.simd.alloc")) {
      AllocVarID(op->var.get());
      ICHECK_GT(PythonDType(op->var).lanes(), 1)
          << "tl.simd.alloc should bind a vector-typed Var, got "
          << PythonDType(op->var);
      return;
    }

    // Pair-producing simd ops unpack into (low, high) SSA names. The bound var
    // is recorded so that subsequent pair_get(var, 0/1) calls resolve to the
    // correct element.
    if (call->op.same_as(tl::simd_vintlv()) ||
        call->op.same_as(tl::simd_vdintlv())) {
      std::string lhs = PrintExpr_(call->args[0]);
      std::string rhs = PrintExpr_(call->args[1]);
      std::string name =
          call->op.same_as(tl::simd_vintlv()) ? "vintlv" : "vdintlv";
      std::string pair_name = AllocVarID(op->var.get());
      std::string low = pair_name + "_low";
      std::string high = pair_name + "_high";
      PrintIndent();
      stream << low << ", " << high << " = pto." << name << "(" << lhs << ", "
             << rhs << ")\n";
      simd_pair_vars_[op->var.get()] = {low, high};
      return;
    }
    if (call->op.same_as(tl::simd_pintlv()) ||
        call->op.same_as(tl::simd_pdintlv())) {
      ICHECK_EQ(call->args.size(), 3U)
          << "tl.simd predicate interleave/deinterleave expects 3 arguments "
             "(src0, src1, width)";
      int64_t width = 0;
      ICHECK(TryGetConstInt(call->args[2], &width))
          << "tl.simd predicate interleave/deinterleave width must be "
             "constant for PTO codegen";
      ICHECK(width == 8 || width == 16 || width == 32)
          << "PTO predicate interleave/deinterleave only supports widths 8, "
             "16, and 32, got "
          << width;
      std::string name =
          call->op.same_as(tl::simd_pintlv()) ? "pintlv" : "pdintlv";
      std::string pair_name = AllocVarID(op->var.get());
      std::string low = pair_name + "_low";
      std::string high = pair_name + "_high";
      PrintIndent();
      stream << low << ", " << high << " = pto." << name << "_b" << width << "("
             << PrintExpr_(call->args[0]) << ", " << PrintExpr_(call->args[1])
             << ")\n";
      simd_pair_vars_[op->var.get()] = {low, high};
      return;
    }
    if (call->op.same_as(tl::simd_vaddc()) ||
        call->op.same_as(tl::simd_vsubc()) ||
        call->op.same_as(tl::simd_vaddcs()) ||
        call->op.same_as(tl::simd_vsubcs())) {
      const bool has_carry_in = call->op.same_as(tl::simd_vaddcs()) ||
                                call->op.same_as(tl::simd_vsubcs());
      ICHECK_EQ(call->args.size(), has_carry_in ? 4U : 3U)
          << "tl.simd carry operation has an invalid argument count";
      const char *name = call->op.same_as(tl::simd_vaddc())    ? "vaddc"
                         : call->op.same_as(tl::simd_vsubc())  ? "vsubc"
                         : call->op.same_as(tl::simd_vaddcs()) ? "vaddcs"
                                                               : "vsubcs";
      std::string pair_name = AllocVarID(op->var.get());
      std::string result = pair_name + "_result";
      std::string carry = pair_name + "_carry";
      PrintIndent();
      stream << result << ", " << carry << " = pto." << name << "("
             << PrintExpr_(call->args[0]) << ", " << PrintExpr_(call->args[1]);
      for (size_t i = 2; i < call->args.size(); ++i) {
        stream << ", " << PrintExpr_(call->args[i]);
      }
      stream << ")\n";
      // PTODSL returns (result, carry); TileLang exposes (carry, result).
      simd_pair_vars_[op->var.get()] = {carry, result};
      return;
    }
    if (call->op.same_as(tl::simd_vmull())) {
      ICHECK_EQ(call->args.size(), 3U)
          << "tl.simd.vmull expects 3 arguments (src0, src1, mask)";
      std::string pair_name = AllocVarID(op->var.get());
      std::string low = pair_name + "_low";
      std::string high = pair_name + "_high";
      PrintIndent();
      stream << low << ", " << high << " = pto.vmull("
             << PrintExpr_(call->args[0]) << ", " << PrintExpr_(call->args[1])
             << ", " << PrintExpr_(call->args[2]) << ")\n";
      simd_pair_vars_[op->var.get()] = {low, high};
      return;
    }
    if (call->op.same_as(tl::simd_vld2())) {
      std::string ptr = PrintExpr_(call->args[0]);
      std::string dist = (call->args[1]).as_or_throw<StringImm>()->value;
      std::string offset =
          call->args.size() > 2U
              ? RemoveOutermostParentheses(PrintExpr_(call->args[2]))
              : "pto.const(0)";
      std::string pair_name = AllocVarID(op->var.get());
      std::string low = pair_name + "_low";
      std::string high = pair_name + "_high";
      PrintIndent();
      stream << low << ", " << high << " = pto.vldsx2(" << ptr << ", " << offset
             << ", \"" << dist << "\")\n";
      simd_pair_vars_[op->var.get()] = {low, high};
      return;
    }
    if (call->op.same_as(tl::simd_vld()) && call->args.size() == 3U) {
      std::string ptr = PrintExpr_(call->args[0]);
      std::string dist = NormalizeVldDistribution(
          (call->args[1]).as_or_throw<StringImm>()->value);
      std::string step = RemoveOutermostParentheses(PrintExpr_(call->args[2]));
      PrimType elem_dtype = PythonDType(GetRef<Expr>(call)).WithLanes(1);
      std::string pair_name = AllocVarID(op->var.get());
      std::string value = pair_name + "_value";
      std::string next_ptr = pair_name + "_next_ptr";
      PrintIndent();
      stream << value << " = pto.vlds(" << ptr
             << ", pto.const(0), pto.vreg_type("
             << PythonDType(GetRef<Expr>(call)).lanes() << ", "
             << ScalarType(elem_dtype) << ")";
      if (!dist.empty() && dist != "NORM") {
        stream << ", dist=\"" << dist << "\"";
      }
      stream << ")\n";
      PrintIndent();
      stream << next_ptr << " = pto.addptr(" << ptr << ", " << step << ")\n";
      simd_pair_vars_[op->var.get()] = {value, next_ptr};
      return;
    }
  }

  // T.make_tensor binds a handle-typed Var to `reinterpret(handle, int_addr)`
  // and carries the element dtype + storage scope in `type_annotation`.
  // Register them so GetPointerExpr / HandleTypeMatch_ see the right type.
  if (PythonIsPointer(op->var) && op->var->ty.defined()) {
    if (auto *ptr = op->var->ty.as<PtrTypeNode>()) {
      PrimType elem_dtype = PrimType::Float(32);
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        elem_dtype = GetRef<PrimType>(prim);
        RegisterHandleType_(op->var.get(), GetRef<PrimType>(prim));
      }
      std::string scope = PythonStorageScope(op->var);
      alloc_storage_scope_[op->var.get()] = scope;
      if (const auto *source_var = op->value.as<VarNode>()) {
        if (integer_address_vars_.count(GetRef<Var>(source_var))) {
          integer_address_vars_.insert(op->var);
        }
      }

      // T.make_tensor: reinterpret(handle, int_addr) carries an int64 address.
      // PTODSL's pto.addptr expects a pointer, not i64; emit a castptr so the
      // bound variable is a properly typed gm pointer.
      const auto *call = op->value.as<CallNode>();
      if (call && PythonIsOp(call->op, "reinterpret") &&
          !PythonIsPointer(call->args[0])) {
        std::string pto_space = (scope == "global") ? "gm" : scope;
        std::string ptr_type = PointerTypeName(elem_dtype, pto_space);
        std::string addr =
            RemoveOutermostParentheses(PrintExpr_(call->args[0]));
        std::string vid = AllocVarID(op->var.get());
        PrintIndent();
        stream << vid << " = pto.castptr(" << addr << ", " << ptr_type << ")\n";
        return;
      }
    }
  }

  std::string vid;
  std::string name_hint = op->var->name;
  if (name_hint.rfind("__cond_", 0) == 0) {
    if (SideEffect(op->value) == CallEffectKind::kPure) {
      // Scheduling can reuse this predicate across tasks that mutate its input.
      // Only pure predicates can be rematerialized; state reads need a
      // snapshot.
      var_idmap_[op->var.get()] = PrintExpr_(op->value);
      return;
    }
    // PTODSL's rewritten branch object uses attribute access for live-out
    // values. Python reserves double-underscore attribute names, so normalize
    // AutoSchedule's condition temporaries before they reach the DSL.
    auto snapshot_it = loop_condition_snapshot_ids_.find(op->var);
    if (snapshot_it == loop_condition_snapshot_ids_.end()) {
      vid = name_supply_->FreshName("tl_cond_" + name_hint.substr(7), false);
    } else {
      vid = snapshot_it->second;
    }
    var_idmap_[op->var.get()] = vid;
  } else if (IsInlineableInvariantSimdBind(op->value)) {
    // Map the Var to the printed RHS so each use rematerializes the broadcast /
    // pset. Do not emit a live SSA bind that would cross range()/scf.for.
    var_idmap_[op->var.get()] = PrintExpr_(op->value);
    return;
  } else {
    vid = AllocVarID(op->var.get());
  }
  PrintSSAAssign(vid, PrintExpr_(op->value), PythonDType(op->var));
}

void CodeGenTileLangPTO::EmitAllocation(const TensorVar &tensor) {
  if (!current_function_is_cube_ || in_mixed_vector_section_) {
    EmitBufferAllocation(tensor);
    return;
  }

  const Var &buffer_var = tensor;
  std::string scope = PythonStorageScope(buffer_var);
  alloc_storage_scope_[buffer_var.get()] = scope;
  ValidateSharedScope(scope);

  if (scope == "shared.dyn" || scope == "shared.l1" ||
      scope == "shared.l1.dyn" || scope == "shared.l0a" ||
      scope == "shared.l0a.dyn" || scope == "shared.l0b" ||
      scope == "shared.l0b.dyn" || scope == "shared.l0c" ||
      scope == "shared.l0c.dyn") {
    PrintIndent();
    stream << AllocVarID(buffer_var.get())
           << " = pto.const(0, dtype=pto.int64)\n";
    integer_address_vars_.insert(buffer_var);
  } else if (scope == "local.fragment") {
    AllocVarID(buffer_var.get());
    auto opt_size = tl::GetConstInt(PythonTensorSize(tensor));
    ICHECK(opt_size.has_value())
        << "PTO local.fragment allocation expects a constant size";
  } else if (scope == "local.var") {
    // Scalar alloc_var is used by GEMM tile mapping; vector alloc_var is a
    // mutable SIMD register and is emitted as a PTODSL Vec value.
    CheckLocalVarBuffer(tensor.get());
    PrintIndent();
    local_var_buffers_.insert(buffer_var.get());
    EmitLocalVarInitialization_(tensor, AllocVarID(buffer_var.get()));
    return;
  }

  RegisterHandleType_(buffer_var.get(), tensor.dtype());
}

void CodeGenTileLangPTO::Dispatch_(const SeqStmtNode *op) {
  for (size_t i = 0; i < op->seq.size(); ++i) {
    const Stmt &stmt = op->seq[i];
    if (i + 1U < op->seq.size()) {
      const auto *first_if = stmt.as<IfNode>();
      const auto *second_if = op->seq[i + 1U].as<IfNode>();
      if (first_if != nullptr && second_if != nullptr &&
          !first_if->else_case.has_value() &&
          !second_if->else_case.has_value() && !ContainsLoopBreak(stmt) &&
          !ContainsLoopBreak(op->seq[i + 1U]) &&
          AreComplementaryConditions(first_if->condition,
                                     second_if->condition)) {
        CopyPadState symbolic_entry{
            current_copy_pad_value_id_ >= 0,
            false,
            current_copy_pad_value_dtype_,
        };
        CopyPadState first_state =
            AnalyzeCopyPadState(first_if->then_case, symbolic_entry);
        CopyPadState second_state =
            AnalyzeCopyPadState(second_if->then_case, symbolic_entry);
        if (first_state.CanMerge(second_state)) {
          int entry_pad_value_id = current_copy_pad_value_id_;
          PrimType entry_pad_value_dtype = current_copy_pad_value_dtype_;
          int merged_pad_value_id = copy_pad_value_counter_++;
          PrintIndent();
          stream << "if " << PrintCondition(first_if->condition) << ":\n";
          int first_scope = BeginScope();
          PrintStmt_(first_if->then_case);
          EmitCopyPadMerge_(merged_pad_value_id, first_state.dtype);
          EndScope(first_scope);

          current_copy_pad_value_id_ = entry_pad_value_id;
          current_copy_pad_value_dtype_ = entry_pad_value_dtype;
          PrintIndent();
          stream << "else:\n";
          int second_scope = BeginScope();
          PrintStmt_(second_if->then_case);
          EmitCopyPadMerge_(merged_pad_value_id, second_state.dtype);
          EndScope(second_scope);

          current_copy_pad_value_id_ = merged_pad_value_id;
          current_copy_pad_value_dtype_ = first_state.dtype;
          ++i;
          continue;
        }
      }
    }

    PrintStmt_(stmt);
  }
}

void CodeGenTileLangPTO::Dispatch_(const ForNode *op) {
  int entry_pad_value_id = current_copy_pad_value_id_;
  PrimType entry_pad_value_dtype = current_copy_pad_value_dtype_;

  sym::Analyzer analyzer;
  PrimExpr start = analyzer->Simplify(op->begin);
  PrimExpr extent = analyzer->Simplify((op->end - op->begin));
  PrimExpr step = analyzer->Simplify(op->step);
  PrimExpr stop = analyzer->Simplify(start + extent);
  std::string vid = AllocVarID(op->var.get());
  std::string begin = RemoveOutermostParentheses(PrintExpr_(start));
  std::string end = RemoveOutermostParentheses(PrintExpr_(stop));
  std::string step_expr = RemoveOutermostParentheses(PrintExpr_(step));
  // Explicit unrolls are normally expanded before codegen. A kUnrolled loop
  // that survives without the annotation is the explicit=False form and must
  // remain a device-side range loop.
  bool explicit_unroll_requested = false;
  std::string range_vid;
  if (op->kind == ForKind::kUnrolled) {
    auto it = Attributes(op).find("pragma_unroll_explicit");
    if (it != Attributes(op).end()) {
      auto annotated = (*it).second.try_cast<bool>();
      ICHECK(annotated.has_value())
          << "pragma_unroll_explicit must be a boolean, got " << (*it).second;
      explicit_unroll_requested = annotated.value();
    }
  }
  int64_t start_value = 0;
  int64_t extent_value = 0;
  int64_t step_value = 0;
  const bool explicit_unroll = explicit_unroll_requested &&
                               TryGetConstInt(start, &start_value) &&
                               TryGetConstInt(extent, &extent_value) &&
                               TryGetConstInt(step, &step_value);
  bool static_local_index_loop = false;
  // Scalarized local buffers are emitted as Python lists. A constant-bound
  // loop whose induction variable is used as a list slot must therefore also
  // be trace-time static, even when TIR classified the loop as serial.
  if (TryGetConstInt(start, &start_value) &&
      TryGetConstInt(extent, &extent_value) &&
      TryGetConstInt(step, &step_value)) {
    tl::PostOrderVisit(op->body, [&](const ObjectRef &node) {
      const VarNode *buffer = nullptr;
      PrimExpr index = MakeConst(PrimType::Int(32), 0);
      if (const auto *load = node.as<TensorLoadNode>()) {
        if (load->indices.size() == 1) {
          buffer = load->source.as_or_throw<TensorVar>().get();
          index = load->indices[0];
        }
      } else if (const auto *store = node.as<TensorStoreNode>()) {
        if (store->indices.size() == 1) {
          buffer = store->dest.as_or_throw<TensorVar>().get();
          index = store->indices[0];
        }
      }
      // Keep non-explicit kUnrolled loops on the device so PTOAS can honor
      // pto.range unroll hints. Vector local slots also require a constant
      // TIR index; static_range alone does not substitute that index.
      // Persistent fragments are the exception: PTOAS requires their GEP
      // index to be a compile-time constant, so a loop that indexes a
      // persistent fragment must always expand at trace time (this mirrors
      // the ascend branch, which maps kUnrolled loops to static_range).
      const bool persistent_fragment =
          buffer != nullptr &&
          persistent_buffer_vars_.count(TensorTypeOf(buffer).get()) != 0;
      if (buffer == nullptr || !index.defined() ||
          (TensorTypeOf(buffer).dtype().lanes() != 1 && !persistent_fragment) ||
          (inside_simtvf_body_ && !persistent_fragment) ||
          (op->kind == ForKind::kUnrolled && !explicit_unroll_requested &&
           !persistent_fragment)) {
        return;
      }
      std::string scope = PythonStorageScope(TensorTypeOf(buffer));
      if ((scope == "local" || scope == "local.fragment") &&
          UsesVar(index,
                  [&](const VarNode *var) { return var == op->var.get(); })) {
        static_local_index_loop = true;
      }
    });
  }
  const bool emit_static_range = (explicit_unroll || static_local_index_loop) &&
                                 !ContainsLoopBreak(op->body);
  const bool pto_unroll_hint = op->kind == ForKind::kUnrolled &&
                               !explicit_unroll_requested &&
                               !ContainsLoopBreak(op->body);
  auto unroll_factor = Attributes(op).Get("pragma_unroll_factor");
  std::optional<int64_t> factor;
  if (unroll_factor) {
    factor = tl::GetIntAnnotation(*unroll_factor);
    ICHECK(factor && *factor > 0 && *factor <= INT32_MAX)
        << "PTO pragma_unroll_factor must be a positive int32";
  }
  PrintIndent();
  if (emit_static_range) {
    // Preserve explicit TIR unrolling as trace-time Python expansion.
    stream << "for " << vid << " in pto.static_range(" << begin << ", " << end
           << ", " << step_expr << "):\n";
  } else {
    // PTODSL rewrites native range loops to device-side scf.for and infers
    // loop-carried SSA values automatically. Keep its MLIR index induction
    // variable separate from the authored TIR dtype used by the loop body.
    range_vid = name_supply_->FreshName("_tl_range_" + vid, false);
    stream << "for " << range_vid << " in "
           << (pto_unroll_hint ? "pto.range(" : "range(") << begin << ", "
           << end << ", " << step_expr;
    if (pto_unroll_hint) {
      if (factor) {
        stream << ", unroll_factor=" << *factor;
      } else {
        stream << ", unroll=\"enable\"";
      }
    }
    stream << "):\n";
  }
  int for_scope = BeginScope();
  const auto for_body_start = stream.tellp();
  if (!emit_static_range) {
    PrintIndent();
    stream << vid << " = pto.cast(" << range_vid << ", "
           << ScalarType(PythonDType(op->var)) << ")\n";
  }
  ++native_loop_depth_;
  PrintStmt_(op->body);
  --native_loop_depth_;
  if (stream.tellp() == for_body_start) {
    PrintIndent();
    stream << "pass\n";
  }
  EndScope(for_scope);
  current_copy_pad_value_id_ = entry_pad_value_id;
  current_copy_pad_value_dtype_ = entry_pad_value_dtype;
}

void CodeGenTileLangPTO::Dispatch_(const WhileNode *op) {
  int entry_pad_value_id = current_copy_pad_value_id_;
  PrimType entry_pad_value_dtype = current_copy_pad_value_dtype_;

  PrintIndent();
  stream << "while " << PrintCondition(op->condition) << ":\n";
  int while_scope = BeginScope();
  const auto while_body_start = stream.tellp();
  // PTODSL can infer a nested loop's condition snapshots as outer loop inputs
  // when an earlier break skips their definitions. Bind placeholders before
  // any break; the authored Bind still takes the actual snapshot at its use.
  // Nested while loops reuse these names instead of resetting them again.
  std::vector<Var> initialized_conditions;
  tl::PostOrderVisit(op->body, [&](const ObjectRef &node) {
    const auto *bind = node.as<BindNode>();
    if (bind == nullptr) {
      return;
    }
    std::string name_hint = bind->var->name;
    if (name_hint.rfind("__cond_", 0) != 0 ||
        SideEffect(bind->value) == CallEffectKind::kPure ||
        loop_condition_snapshot_ids_.count(bind->var)) {
      return;
    }
    std::string name =
        name_supply_->FreshName("tl_cond_" + name_hint.substr(7), false);
    loop_condition_snapshot_ids_.emplace(bind->var, name);
    initialized_conditions.push_back(bind->var);
    PrintIndent();
    stream << name
           << " = pto.const(0, dtype=" << DataTypeName(PythonDType(bind->var))
           << ")\n";
  });
  // PTODSL's native-while rewrite needs a value to be carried when it is
  // assigned on some iterations and read after the loop. Materialize a
  // read/write on every iteration for local.var values already in scope.
  std::set<std::string> carried_locals;
  tl::PostOrderVisit(op->body, [&](const ObjectRef &node) {
    const auto *store = node.as<TensorStoreNode>();
    if (store == nullptr)
      return;
    const VarNode *var = store->dest.as_or_throw<TensorVar>().get();
    if (local_var_buffers_.count(var) && var_idmap_.count(var) &&
        !store->dest.as_or_throw<TensorVar>().dtype().IsFixedLengthVector()) {
      carried_locals.insert(GetVarID(var));
    }
  });
  for (const std::string &name : carried_locals) {
    PrintIndent();
    stream << name << " = " << name << "\n";
  }
  ++native_loop_depth_;
  PrintStmt_(op->body);
  --native_loop_depth_;
  if (stream.tellp() == while_body_start) {
    PrintIndent();
    stream << "pass\n";
  }
  EndScope(while_scope);
  for (const Var &var : initialized_conditions) {
    loop_condition_snapshot_ids_.erase(var);
  }

  // A runtime loop may execute zero times, so no copy-pad state established
  // only in its body is available after the loop.
  current_copy_pad_value_id_ = entry_pad_value_id;
  current_copy_pad_value_dtype_ = entry_pad_value_dtype;
}

void CodeGenTileLangPTO::Dispatch_(const RegionStmtNode *op) {
  if (Attributes(op).count("tl.simdvf_scope")) {
    if (simdvf_nesting_depth_ == 0) {
      PrintIndent();
      stream << "with pto.vecscope():\n";
      int scope = BeginScope();
      ++simdvf_nesting_depth_;
      PrintStmt_(op->body);
      --simdvf_nesting_depth_;
      EndScope(scope);
    } else {
      PrintIndent();
      stream << "pto.mem_bar(\"VST_VLD\")\n";
      PrintStmt_(op->body);
    }
    return;
  }
  if (op->op.same_as(tirx::launch_thread_op())) {
    auto tag = op->args[0].as_or_throw<StringImm>()->value;
    auto variable = op->body_params[0].as_or_throw<PrimVar>();
    if (tag.starts_with("threadIdx.") && !inside_simtvf_body_) {
      ICHECK(!UsesVar(op->body, variable.get()))
          << "PTO thread index outside SIMT section: " << tag;
      PrintStmt_(op->body);
      return;
    }
    std::string value;
    if (tag == "blockIdx.x")
      value = "pto.get_block_idx()";
    else if (tag == "cthread")
      value = "pto.get_subblock_idx()";
    else if (tag == "threadIdx.x")
      value = "pto.get_tid_x()";
    else if (tag == "threadIdx.y")
      value = "pto.get_tid_y()";
    else if (tag == "threadIdx.z")
      value = "pto.get_tid_z()";
    else
      LOG(FATAL) << "Unsupported PTO thread tag: " << tag;
    PrintIndent();
    stream << AllocVarID(variable.get()) << " = pto.cast(" << value << ", "
           << ScalarType(variable.ty()) << ")\n";
    PrintStmt_(op->body);
    return;
  }
  if (current_function_is_mixed_ &&
      (RegionName(op) == "CUBE" || RegionName(op) == "VECTOR")) {
    bool is_vector_section = RegionName(op) == "VECTOR";
    int64_t vector_count = 2;
    if (is_vector_section) {
      if (auto opt = Attributes(op).Get("vector_count")) {
        auto count = tl::GetIntAnnotation(opt.value());
        ICHECK(count && (*count == 1 || *count == 2))
            << "Mixed-kernel vector_count must be the constant integer 1 or "
               "2, got "
            << opt.value();
        vector_count = *count;
      }
    }
    // Physical sections are independent SSA regions.  Restore all captured
    // outer local.var bindings both before and after a section so neither a
    // sibling section nor ordinary outer code can observe section-local SSA.
    RestoreMixedSectionVariables(op);
    PrintIndent();
    stream << "with tl.mixed_kernel_section(\""
           << (is_vector_section ? "vector" : "cube") << "\"):\n";
    int section_scope = BeginScope();
    bool old_in_mixed_vector_section = in_mixed_vector_section_;
    bool old_inside_mixed_section = inside_mixed_section_;
    in_mixed_vector_section_ = is_vector_section;
    inside_mixed_section_ = true;
    PrintIndent();
    stream << "pto.init_core()\n";
    int active_aiv_scope = -1;
    if (is_vector_section && vector_count == 1) {
      // The physical mixed group still has two AIVs; only AIV0 executes this
      // section's body, including its cross-core synchronization.
      PrintIndent();
      stream << "if pto.get_subblock_idx() == 0:\n";
      active_aiv_scope = BeginScope();
    }
    PrintStmt_(op->body);
    if (active_aiv_scope != -1) {
      EndScope(active_aiv_scope);
    }
    inside_mixed_section_ = old_inside_mixed_section;
    in_mixed_vector_section_ = old_in_mixed_vector_section;
    EndScope(section_scope);
    RestoreMixedSectionVariables(op);
    return;
  }

  if (RegionName(op) == "SIMT_VF") {
    int64_t thread_x = 1;
    int64_t thread_y = 1;
    int64_t thread_z = 1;
    ExtractSimtThreadExtents(op, &thread_x, &thread_y, &thread_z);
    EmitInlineSimtVF(op, thread_x, thread_y, thread_z);
    return;
  }

  if (RegionName(op) == "SIMD_VF") {
    PrintStmt_(op->body);
    return;
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::Dispatch_(const IfNode *op) {
  // Fold compile-time conditions; keep runtime control flow as native Python.
  PrimExpr cond_simp = sym::Analyzer()->Simplify(op->condition);
  if (const auto *imm = cond_simp.as<IntImmNode>()) {
    if (imm->value != 0) {
      PrintStmt_(op->then_case);
    } else if (op->else_case.has_value()) {
      PrintStmt_(op->else_case.value());
    }
    return;
  }

  int entry_pad_value_id = current_copy_pad_value_id_;
  PrimType entry_pad_value_dtype = current_copy_pad_value_dtype_;
  CopyPadState symbolic_entry{
      entry_pad_value_id >= 0,
      false,
      entry_pad_value_dtype,
  };
  CopyPadState symbolic_then =
      AnalyzeCopyPadState(op->then_case, symbolic_entry);
  CopyPadState symbolic_else = symbolic_entry;
  if (op->else_case.has_value()) {
    symbolic_else = AnalyzeCopyPadState(op->else_case.value(), symbolic_entry);
  }
  bool needs_pad_merge = symbolic_then.CanMerge(symbolic_else);

  std::string cond = PrintCondition(op->condition);

  auto set_reaching_pad_state = [&](int then_id, PrimType then_dtype,
                                    int else_id, PrimType else_dtype) {
    if (then_id >= 0 && then_id == else_id) {
      ICHECK(then_dtype == else_dtype)
          << "PTO copy padding reached the same SSA alias with inconsistent "
             "dtypes";
      current_copy_pad_value_id_ = then_id;
      current_copy_pad_value_dtype_ = then_dtype;
    } else {
      current_copy_pad_value_id_ = -1;
      current_copy_pad_value_dtype_ = PrimType::Void();
    }
  };

  PrintIndent();
  stream << "if " << cond << ":\n";
  int then_scope = BeginScope();
  const auto then_body_start = stream.tellp();
  PrintStmt_(op->then_case);
  int then_pad_value_id = current_copy_pad_value_id_;
  PrimType then_pad_value_dtype = current_copy_pad_value_dtype_;
  int merged_pad_value_id = -1;
  if (needs_pad_merge) {
    merged_pad_value_id = copy_pad_value_counter_++;
    EmitCopyPadMerge_(merged_pad_value_id, symbolic_then.dtype);
  }
  if (stream.tellp() == then_body_start) {
    PrintIndent();
    stream << "pass\n";
  }
  EndScope(then_scope);

  current_copy_pad_value_id_ = entry_pad_value_id;
  current_copy_pad_value_dtype_ = entry_pad_value_dtype;
  int else_pad_value_id = entry_pad_value_id;
  PrimType else_pad_value_dtype = entry_pad_value_dtype;
  if (op->else_case.has_value()) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    const auto else_body_start = stream.tellp();
    PrintStmt_(op->else_case.value());
    else_pad_value_id = current_copy_pad_value_id_;
    else_pad_value_dtype = current_copy_pad_value_dtype_;
    if (needs_pad_merge) {
      EmitCopyPadMerge_(merged_pad_value_id, symbolic_else.dtype);
    }
    if (stream.tellp() == else_body_start) {
      PrintIndent();
      stream << "pass\n";
    }
    EndScope(else_scope);
  } else if (needs_pad_merge) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    EmitCopyPadMerge_(merged_pad_value_id, symbolic_else.dtype);
    EndScope(else_scope);
  }

  if (needs_pad_merge) {
    current_copy_pad_value_id_ = merged_pad_value_id;
    current_copy_pad_value_dtype_ = symbolic_then.dtype;
  } else {
    set_reaching_pad_state(then_pad_value_id, then_pad_value_dtype,
                           else_pad_value_id, else_pad_value_dtype);
  }
}

void CodeGenTileLangPTO::Dispatch_(const EvaluateNode *op) {
  if (tl::GetConstInt(op->value))
    return;
  if (const auto *call = op->value.as<CallNode>();
      call != nullptr && (call->op.same_as(tl::device_assert()) ||
                          call->op.same_as(tl::device_assert_with_msg()))) {
    const bool has_message = call->op.same_as(tl::device_assert_with_msg());
    ICHECK_EQ(call->args.size(), has_message ? 2U : 1U)
        << (has_message ? "tl.device_assert_with_msg expects exactly 2 "
                          "arguments (condition, msg)"
                        : "tl.device_assert expects exactly 1 argument "
                          "(condition)");
    // TODO(PTOAS): 等pto支持打印后放开msg传递
    std::string condition = PrintCondition(call->args[0]);
    PrintIndent();
    stream << "if (" << condition << ") == 0:\n";
    int trap_scope = BeginScope();
    PrintIndent();
    stream << "pto.trap()\n";
    EndScope(trap_scope);
    return;
  }
  std::string emitted = PrintExpr_(op->value);
  if (!emitted.empty()) {
    PrintIndent();
    stream << emitted << "\n";
  }
}

void CodeGenTileLangPTO::Dispatch_(const TensorLoadNode *op,
                                   std::ostream &os) { // NOLINT(*)
  // S.alloc_var inside T.SimdVF() creates a `local.var`-scope buffer with a
  // vector dtype (e.g. float32x64). In PTODSL it is a plain Python variable
  // holding a vector register value, so load at index 0 is just the variable
  // itself. Skip the scalar-only CheckLocalVarBuffer below.
  if (op->source.as_or_throw<TensorVar>().dtype().lanes() > 1) {
    std::string scope =
        ScopeOfBuffer(op->source.as_or_throw<TensorVar>().get());
    if (scope == "local.var") {
      ICHECK_EQ(op->indices.size(), 1U)
          << "PTO vector local.var load expects a single index";
      int64_t index = 0;
      ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
          << "PTO vector local.var load expects index 0";
      os << LocalVarID(op->source.as_or_throw<TensorVar>().get());
      return;
    }
  }

  if (IsLocalVarBuffer(op->source.as_or_throw<TensorVar>().get())) {
    // Vector SSA and scalar PTO surface values both load as Python variables.
    CheckLocalVarBuffer(op->source.as_or_throw<TensorVar>().get());
    if (op->source.as_or_throw<TensorVar>().dtype().lanes() > 1) {
      os << LocalVarID(op->source.as_or_throw<TensorVar>().get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var load expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var load expects index 0";
    if (current_function_has_gemm_) {
      os << LocalVarID(op->source.as_or_throw<TensorVar>().get());
      return;
    }
    if (IsVectorLocalVarDtype(op->source.as_or_throw<TensorVar>().dtype()) ||
        IsSurfaceScalarLocalVarDtype(
            op->source.as_or_throw<TensorVar>().dtype())) {
      os << LocalVarID(op->source.as_or_throw<TensorVar>().get());
    } else {
      LOG(FATAL) << "Unsupported PTO local.var load dtype: "
                 << op->source.as_or_throw<TensorVar>().dtype();
    }
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer loads";

  PrimType value_dtype = PythonDType(GetRef<Expr>(op));
  PrimType element_dtype = op->source.as_or_throw<TensorVar>().dtype();
  if (value_dtype.lanes() > 1) {
    EmitScalarizedLoad(op, os);
    return;
  }

  ICHECK_EQ(value_dtype, element_dtype)
      << "PTO scalar TensorLoad expects value dtype to match buffer element "
         "dtype, got "
      << value_dtype << " vs " << element_dtype;

  std::string loaded =
      ScalarLoad(op->source.as_or_throw<TensorVar>().get(), op->indices[0]);
  if (value_dtype.MatchesCode(kDLBool)) {
    os << "tl.as_logical_bool(" << loaded << ")";
  } else {
    os << loaded;
  }
}

void CodeGenTileLangPTO::EmitScalarizedLoad(const TensorLoadNode *op,
                                            std::ostream &os) {
  PrimType value_dtype = PythonDType(GetRef<Expr>(op));
  PrimType element_dtype = op->source.as_or_throw<TensorVar>().dtype();
  std::string scope = ScopeOfBuffer(op->source.as_or_throw<TensorVar>().get());

  // Vector-element buffer (e.g. T.simd.alloc_local produces float32x64
  // elements): reading one slot yields a whole vector register. Emit a plain
  // Python index expression.
  if (element_dtype.lanes() > 1) {
    if (scope == "local.fragment" || scope == "local") {
      os << GetVectorLocalRef(op->source.as_or_throw<TensorVar>().get(),
                              op->indices[0], "PTO vector load");
    } else {
      std::string index_str =
          RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
      os << GetVarID(op->source.as_or_throw<TensorVar>().get()) << "["
         << index_str << "]";
    }
    return;
  }

  ICHECK_EQ(element_dtype.lanes(), 1)
      << "PTO vector TensorLoad scalarization currently expects scalar "
         "buffer elements, got "
      << element_dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsSupportedSIMTLocalStorageType(element_dtype))
        << "PTO SIMT vector TensorLoad supports float16, bfloat16, float32, "
           "int32, uint32, and float8_e4m3fn/float8_e5m2 storage, got "
        << element_dtype;
    ICHECK_EQ(value_dtype.WithLanes(1), element_dtype)
        << "PTO SIMT vector TensorLoad expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << element_dtype;
    ICHECK(!tl::IsAscendVectorizableFP8(element_dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 TensorLoad currently supports contiguous lanes 2, 4, "
           "and 8 only, got "
        << value_dtype;
    const int lanes = value_dtype.lanes();
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "load");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    if (scope == "local.fragment" || scope == "local") {
      os << "pto.load(" << GetVarID(op->source.as_or_throw<TensorVar>().get())
         << ", " << index_str << ", contiguous=" << lanes << ")";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base = ScalarPointerBase_(
          op->source.as_or_throw<TensorVar>().get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base = ScalarPointerBase_(
          op->source.as_or_throw<TensorVar>().get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector load scope: " << scope;
  }

  // Non-SIMT vector load from a scalar-element buffer: use pto.load with
  // contiguous=N to read all lanes in one shot.
  if (value_dtype.lanes() > 1) {
    const int lanes = value_dtype.lanes();
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "load");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    if (scope == "local.fragment" || scope == "local") {
      // Non-SIMT local buffers are Python lists; rebuild a typed vector from
      // their scalar lanes instead of calling pto.load on the list.
      const std::string list_ref =
          GetVarID(op->source.as_or_throw<TensorVar>().get());
      os << "tl.vector_from_list(" << DataTypeName(element_dtype) << ", (";
      for (int lane = 0; lane < lanes; ++lane) {
        if (lane != 0) {
          os << ", ";
        }
        os << list_ref << "[" << index_str;
        if (lane != 0) {
          os << " + " << lane;
        }
        os << "]";
      }
      os << "))";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base = ScalarPointerBase_(
          op->source.as_or_throw<TensorVar>().get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base = ScalarPointerBase_(
          op->source.as_or_throw<TensorVar>().get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    LOG(FATAL) << "Unsupported PTO vector load scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector TensorLoad is not supported yet";
}

void CodeGenTileLangPTO::Dispatch_(const TensorStoreNode *op) {
  // S.alloc_var inside T.SimdVF() creates a `local.var`-scope buffer with a
  // vector dtype (e.g. float32x64). In PTODSL it is a plain Python variable
  // holding a vector register value, so store at index 0 is just
  // `var = value`. Skip the scalar-only CheckLocalVarBuffer below.
  if (op->dest.as_or_throw<TensorVar>().dtype().lanes() > 1) {
    std::string scope = ScopeOfBuffer(op->dest.as_or_throw<TensorVar>().get());
    if (scope == "local.var") {
      ICHECK_EQ(op->indices.size(), 1U)
          << "PTO vector local.var store expects a single index";
      int64_t index = 0;
      ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
          << "PTO vector local.var store expects index 0";
      std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
      PrintIndent();
      stream << LocalVarID(op->dest.as_or_throw<TensorVar>().get()) << " = "
             << value << "\n";
      EmitMixedEntrySnapshot(op->dest.as_or_throw<TensorVar>().get());
      return;
    }
  }

  if (IsLocalVarBuffer(op->dest.as_or_throw<TensorVar>().get())) {
    // Vector SSA and scalar PTO surface values both store as Python variables.
    CheckLocalVarBuffer(op->dest.as_or_throw<TensorVar>().get());
    if (op->dest.as_or_throw<TensorVar>().dtype().lanes() > 1) {
      std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
      PrintIndent();
      stream << LocalVarID(op->dest.as_or_throw<TensorVar>().get()) << " = "
             << value << "\n";
      EmitMixedEntrySnapshot(op->dest.as_or_throw<TensorVar>().get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var store expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var store expects index 0";
    std::string vid = LocalVarID(op->dest.as_or_throw<TensorVar>().get());
    PrimType dtype = op->dest.as_or_throw<TensorVar>().dtype();
    std::string value;
    if (current_function_has_gemm_) {
      value = LocalVarStoreValue(op->value, dtype);
    } else if (IsVectorLocalVarDtype(dtype) ||
               dtype.MatchesCode(kDLOpaqueHandle)) {
      value = RemoveOutermostParentheses(PrintExpr_(op->value));
    } else if (IsSurfaceScalarLocalVarDtype(dtype)) {
      value = LocalVarStoreValue(op->value, dtype);
    } else {
      LOG(FATAL) << "Unsupported PTO local.var store dtype: " << dtype;
    }
    PrintIndent();
    stream << vid << " = " << value << "\n";
    EmitMixedEntrySnapshot(op->dest.as_or_throw<TensorVar>().get());
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer stores";

  if (PythonDType(op->value).lanes() > 1) {
    EmitScalarizedStore(op);
    return;
  }

  std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
  PrintIndent();
  EmitScalarStore(op->dest.as_or_throw<TensorVar>().get(), value,
                  op->indices[0]);
}

void CodeGenTileLangPTO::EmitScalarizedStore(const TensorStoreNode *op) {
  PrimType buffer_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  PrimType value_dtype = PythonDType(op->value);
  std::string scope = ScopeOfBuffer(op->dest.as_or_throw<TensorVar>().get());

  // Vector-element buffer (e.g. T.simd.alloc_local produces float32x64
  // elements): the value is a whole vector register stored into one buffer
  // slot. Emit a plain Python assignment; the buffer was allocated as a list
  // (non-SIMT) or pto.alloc_buffer (SIMT).
  if (buffer_dtype.lanes() > 1) {
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    PrintIndent();
    if (scope == "local.fragment" || scope == "local") {
      stream << GetVectorLocalRef(op->dest.as_or_throw<TensorVar>().get(),
                                  op->indices[0], "PTO vector store")
             << " = " << value << "\n";
    } else {
      std::string index_str =
          RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
      stream << GetVarID(op->dest.as_or_throw<TensorVar>().get()) << "["
             << index_str << "] = " << value << "\n";
    }
    return;
  }

  ICHECK_EQ(buffer_dtype.lanes(), 1)
      << "PTO vector TensorStore scalarization currently expects scalar "
         "buffer elements, got "
      << buffer_dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsSupportedSIMTLocalStorageType(buffer_dtype))
        << "PTO SIMT vector TensorStore supports float16, bfloat16, float32, "
           "int32, uint32, and float8_e4m3fn/float8_e5m2 storage only, got "
        << buffer_dtype;
    ICHECK_EQ(value_dtype.WithLanes(1), buffer_dtype)
        << "PTO SIMT vector TensorStore expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << buffer_dtype;
    if (TryEmitRngBroadcastStore(op)) {
      return;
    }
    ICHECK(!tl::IsAscendVectorizableFP8(buffer_dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 TensorStore currently supports contiguous lanes 2, "
           "4, and 8 only, got "
        << value_dtype;
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "store");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    PrintIndent();
    const int store_lanes = value_dtype.lanes();
    const std::string contiguous_suffix =
        store_lanes > 1 ? ", contiguous=" + std::to_string(store_lanes) : "";
    if (scope == "local.fragment" || scope == "local") {
      stream << "pto.store(" << value << ", "
             << GetVarID(op->dest.as_or_throw<TensorVar>().get()) << ", "
             << index_str << contiguous_suffix << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base =
          ScalarPointerBase_(op->dest.as_or_throw<TensorVar>().get(),
                             op->dest.as_or_throw<TensorVar>().dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base =
          ScalarPointerBase_(op->dest.as_or_throw<TensorVar>().get(),
                             op->dest.as_or_throw<TensorVar>().dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector store scope: " << scope;
  }

  // Non-SIMT vector store to a scalar-element buffer: use pto.store with
  // contiguous=N to write all lanes in one shot.
  if (value_dtype.lanes() > 1) {
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "store");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    if (scope == "local.fragment" || scope == "local") {
      // Non-SIMT local buffers are Python lists.  PTODSL pto.store does
      // not accept a list as a vector destination, so extract each lane and
      // assign it directly.
      const std::string list_ref =
          GetVarID(op->dest.as_or_throw<TensorVar>().get());
      PrintIndent();
      stream << "tl.store_vector_to_list(" << list_ref << ", " << index_str
             << ", " << value << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      PrintIndent();
      std::string base =
          ScalarPointerBase_(op->dest.as_or_throw<TensorVar>().get(),
                             op->dest.as_or_throw<TensorVar>().dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << ", contiguous=" << value_dtype.lanes() << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      PrintIndent();
      std::string base =
          ScalarPointerBase_(op->dest.as_or_throw<TensorVar>().get(),
                             op->dest.as_or_throw<TensorVar>().dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << ", contiguous=" << value_dtype.lanes() << ")\n";
      return;
    }
    LOG(FATAL) << "Unsupported PTO vector store scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector TensorStore is not supported yet";
}

bool CodeGenTileLangPTO::TryEmitRngBroadcastStore(const TensorStoreNode *op) {
  const auto *broadcast = op->value.as<BroadcastNode>();
  if (broadcast == nullptr) {
    return false;
  }
  bool contains_rng_draw = false;
  tl::PostOrderVisit(broadcast->value, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call != nullptr && (call->op.same_as(tl::rng_rand()) ||
                            call->op.same_as(tl::rng_rand_float()))) {
      contains_rng_draw = true;
    }
  });
  if (!contains_rng_draw) {
    return false;
  }
  // Each lane must evaluate the complete scalar expression independently so
  // that every RNG call advances the stream. Reusing one evaluated value for
  // both stores would incorrectly duplicate the same random output.
  ICHECK_EQ(PythonDType(GetRef<Expr>(broadcast)).lanes(), 2)
      << "PTO RNG broadcast store currently supports exactly 2 lanes, got "
      << PythonDType(GetRef<Expr>(broadcast));

  PrimExpr base_index = op->indices[0];
  if (const auto *ramp = base_index.as<RampNode>()) {
    CheckContiguousRampStride(op->indices[0], "store");
    base_index = ramp->base;
  }

  sym::Analyzer analyzer;
  for (int64_t lane = 0; lane < 2; ++lane) {
    std::string tmp = name_supply_->FreshName("_tl_rng_value");
    std::string value =
        RemoveOutermostParentheses(PrintExpr_(broadcast->value));
    PrintIndent();
    stream << tmp << " = " << value << "\n";
    PrimExpr lane_index = analyzer->Simplify(
        base_index + MakeConst(PythonDType(base_index), lane));
    PrintIndent();
    EmitScalarStore(op->dest.as_or_throw<TensorVar>().get(), tmp, lane_index);
  }
  return true;
}

void CodeGenTileLangPTO::EmitRngInit(const CallNode *op) {
  ICHECK(inside_simtvf_body_)
      << "tl.rng_init on PTO must be used inside a T.SimtVF(...) block";
  ICHECK_EQ(op->args.size(), 4U)
      << "tl.rng_init expects exactly 4 arguments (seed, seq, off, generator)";
  pto_rng_initialized_ = true;
  pto_rng_state_var_ = "_tl_rng_state";
  pto_rng_counter_var_ = "_tl_rng_counter";
  pto_rng_normal_cache_var_ = "_tl_rng_normal_cache";
  pto_rng_has_normal_var_ = "_tl_rng_has_normal";
  std::string seed = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string seq = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string off = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  PrintIndent();
  stream << pto_rng_state_var_ << " = tl.PhiloxRNG(" << seed << ", " << seq
         << ", " << off << ")\n";
  PrintIndent();
  stream << pto_rng_counter_var_ << " = pto.const(0, dtype=pto.ui64)\n";
  PrintIndent();
  stream << pto_rng_normal_cache_var_ << " = pto.const(0.0, dtype=pto.f32)\n";
  PrintIndent();
  stream << pto_rng_has_normal_var_ << " = pto.const(0, dtype=pto.i32)\n";
}

std::string CodeGenTileLangPTO::EmitRngRand(const CallNode *op) {
  ICHECK(pto_rng_initialized_)
      << "tl.rng_rand called without prior tl.rng_init";
  ICHECK(PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt) &&
         PythonDType(GetRef<Expr>(op)).bits() == 32 &&
         PythonDType(GetRef<Expr>(op)).IsScalar())
      << "tl.rng_rand expects a scalar uint32 result, got "
      << PythonDType(GetRef<Expr>(op));
  std::string result_var =
      "_tl_rng_result_" + std::to_string(pto_rng_result_counter_++);
  PrintIndent();
  stream << result_var << ", " << pto_rng_counter_var_ << " = "
         << pto_rng_state_var_ << ".rand(" << pto_rng_counter_var_ << ")\n";
  return ScalarCastExpr(result_var, PythonDType(GetRef<Expr>(op)),
                        "PTO rng_rand result");
}

std::string CodeGenTileLangPTO::EmitRngRandFloat(const CallNode *op) {
  ICHECK(pto_rng_initialized_)
      << "tl.rng_rand_float called without prior tl.rng_init";
  ICHECK_NE(PythonDType(GetRef<Expr>(op)).bits(), 64)
      << "float64 RNG (tl.rng_rand_float bit=64) is not supported on PTO";
  ICHECK_EQ(op->args.size(), 1U)
      << "tl.rng_rand_float expects 1 argument (distribution)";
  std::string dist = (op->args[0]).as_or_throw<StringImm>()->value;
  ICHECK(dist == "uniform" || dist == "normal")
      << "Unsupported RNG distribution on PTO: " << dist;
  std::string result_var =
      "_tl_rng_result_" + std::to_string(pto_rng_result_counter_++);
  PrintIndent();
  if (dist == "uniform") {
    stream << result_var << ", " << pto_rng_counter_var_ << " = "
           << pto_rng_state_var_ << ".rand_uniform(" << pto_rng_counter_var_
           << ")\n";
  } else {
    stream << result_var << ", " << pto_rng_counter_var_ << ", "
           << pto_rng_normal_cache_var_ << ", " << pto_rng_has_normal_var_
           << " = " << pto_rng_state_var_ << ".rand_normal("
           << pto_rng_counter_var_ << ", " << pto_rng_normal_cache_var_ << ", "
           << pto_rng_has_normal_var_ << ")\n";
  }
  return result_var;
}

} // namespace codegen
} // namespace tvm
