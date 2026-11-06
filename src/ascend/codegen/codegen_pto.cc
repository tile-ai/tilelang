/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"
#include "ascend/codegen/sfu_precision.h"
#include "ascend/op/nd2nz.h"

#include "ascend/op/builtin.h"
#include "backend/common/codegen/codegen_utils.h"
#include "op/builtin.h"
#include "support/check.h"

#include "ascend/ir/block.h"
#include "ascend/target_utils.h"
#include <algorithm>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
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

namespace {

constexpr const char *kModeMerging = "MODE_MERGING";
constexpr const char *kModeZeroing = "MODE_ZEROING";
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
  const auto type = op->ty.as<PrimType>();
  if (!type || !type.value().IsVoid() || op->args.empty()) {
    return false;
  }
  const auto *mode = op->args.back().as<StringImmObj>();
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
    TVM_FFI_THROW(InternalError)
        << "Static shared memory (scope='shared') is not supported on "
           "Ascend. "
        << "Use dynamic shared memory (scope='shared.dyn') with "
           "T.alloc_shared.";
  }
}

bool RequiresStatementControlFlow(const Expr &expr) {
  if (const auto *call = expr.as<CallObj>()) {
    if (call->op.same_as(prim::if_then_else_op())) {
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
    TVM_FFI_THROW(InternalError)
        << "PTO Ascend FP8 types support lanes 1, 2, 4, and 8 only, got " << t;
  }

  if (t.MatchesElementType(kDLFloat8_e4m3, 8) ||
      t.MatchesElementType(kDLFloat8_e4m3fn, 8))
    return "pto.f8e4m3" + lanes_suffix;
  if (t.MatchesElementType(kDLFloat8_e5m2, 8))
    return "pto.f8e5m2" + lanes_suffix;
  TVM_FFI_THROW(InternalError) << "Unsupported PTO Ascend FP8 type: " << t;
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
  } else if (t.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
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
  TVM_FFI_THROW(InternalError) << "Unsupported PTO type: " << t;
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
  TVM_FFI_THROW(InternalError)
      << "Unsupported PTO store atomic type: " << t
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
  while (const auto *cast = cur.as<prim::CastObj>()) {
    if (!cast->ty.as_or_throw<PrimType>().IsScalar() ||
        !cast->value.ty().IsScalar()) {
      break;
    }
    cur = cast->value;
  }
  return cur;
}

bool IsImmediateScalar(const Expr &value) {
  PrimExpr peeled = PeelScalarCasts(value);
  return peeled.as<IntImmObj>() != nullptr ||
         peeled.as<FloatImmObj>() != nullptr;
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
  TVM_FFI_THROW(InternalError)
      << "Unsupported PTO dtype string: " << dtype_name;
  return PrimType::Void();
}

bool TryGetConstInt(const Expr &expr, int64_t *value) {
  if (const auto *imm = expr.as<IntImmObj>()) {
    *value = CodegenInt(imm);
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
  PrimType elem = (!t.IsScalar()) ? t.WithLanes(1) : t;
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
  return dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) && dtype.lanes() == 2;
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

void CheckContiguousRampStride(const PrimExpr &index, const char *access_kind) {
  if (const auto *ramp = index.as<prim::RampObj>()) {
    PrimExpr stride_expr = sym::Analyzer()->Simplify(ramp->stride);
    int64_t stride = 0;
    ICHECK(TryGetConstInt(stride_expr, &stride) && stride == 1)
        << "PTO SIMT vector " << access_kind
        << " emits a contiguous vector access, so Ramp stride must be 1, got "
        << ramp->stride;
  }
}

PrimExpr SIMTVectorLocalScalarOffset(const PrimExpr &index, int element_lanes,
                                     const char *access_kind) {
  // BufferLoad/Store indices count vector slots, including Ramp indices.
  PrimExpr slot = index;
  if (const auto *ramp = index.as<prim::RampObj>()) {
    CheckContiguousRampStride(index, access_kind);
    slot = ramp->base;
  }
  ICHECK(slot.ty().IsScalar())
      << "PTO SIMT vector local " << access_kind
      << " requires a scalar index or a unit-stride Ramp, got " << index;
  return sym::Analyzer()->Simplify(slot * MakeConst(slot.ty(), element_lanes));
}

void CheckConstZero(const PrimExpr &expr, const char *name) {
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

int64_t ConstArgDim(const CallObj *call, size_t index, const char *name) {
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
    PrimExpr pack = MakeConst(elem_index.ty(), pack_factor);
    PrimExpr elem_mod = analyzer->Simplify(floormod(elem_index, pack));
    int64_t remainder = 0;
    ICHECK(!TryGetConstInt(elem_mod, &remainder) || remainder == 0)
        << context << " expects a packed-byte-aligned logical offset, got "
        << elem_index << " for " << dtype;
    return analyzer->Simplify(prim::FloorDiv(elem_index, pack));
  }

  ICHECK_EQ(elem_bits % 8, 0)
      << context << " physical element width must be byte-aligned, got "
      << dtype << " (" << elem_bits << " bits)";
  int64_t elem_bytes = elem_bits / 8;
  return analyzer->Simplify(elem_index *
                            MakeConst(elem_index.ty(), elem_bytes));
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
  PrimType elem_dtype = (!dtype.IsScalar()) ? dtype.WithLanes(1) : dtype;
  if (elem_dtype.MatchesCode(kDLBool)) {
    return "pto.i8";
  }
  return DataTypeName(elem_dtype);
}

void CheckLocalVarBuffer(const VarObj *buffer) {
  ICHECK_EQ(CodegenTensor(buffer).shape().size(), 1U)
      << "PTO local.var only supports scalar alloc_var buffers, got rank "
      << CodegenTensor(buffer).shape().size();
  int64_t extent = 0;
  ICHECK(TryGetConstInt(CodegenTensor(buffer).shape()[0], &extent) &&
         extent == 1)
      << "PTO local.var only supports scalar alloc_var buffers with shape "
         "(1,), got "
      << CodegenTensor(buffer).shape()[0];
  PrimType dtype = CodegenTensorDType(buffer);
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
      elem_dtype.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
      elem_dtype.MatchesElementType(kDLFloat8_e5m2, 8) ||
      (elem_dtype.MatchesCode(kDLFloat) && (bits == 16 || bits == 32)) ||
      ((elem_dtype.MatchesCode(kDLInt) || elem_dtype.MatchesCode(kDLUInt)) &&
       (bits == 1 || bits == 8 || bits == 16 || bits == 32 || bits == 64));
  ICHECK(supported)
      << "PTO local.var only supports PTO numeric scalar/vector element "
         "types, got "
      << dtype;
}

void CheckAllReduceDtype(const CallObj *call) {
  ICHECK_GE(call->args.size(), 2U)
      << "tl::AscendAllReduce call expects a value argument";
  PrimType dtype = call->args[1]->ty.as_or_throw<PrimType>();
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

  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body,
      [&](const RegionStmtObj *block) -> ffi::Expected<ffi::WalkResult> {
        if (!tl::ascend::IsAscendBlock(block))
          return ffi::WalkResult::Advance();
        const auto name = block->attrs.GetAttr<ffi::String>("name").value();
        if (name == "CUBE") {
          has_cube_block = true;
        } else if (name == "VECTOR") {
          has_vector_block = true;
        }
        return ffi::WalkResult::Advance();
      },
      [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if ((call->op.same_as(tirx::call_extern_op()) ||
             call->op.same_as(tirx::call_pure_extern_op())) &&
            !call->args.empty()) {
          const auto *func_name = call->args[0].as<StringImmObj>();
          if (func_name != nullptr &&
              func_name->value.find("tl::AscendAllReduce") !=
                  std::string::npos) {
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
        return ffi::WalkResult::Advance();
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
  TVM_FFI_THROW(InternalError)
      << "PTO GEMM unsupported L0C store unit_flag_ctrl=" << unit_flag_ctrl;
  return "None";
}

bool IsSupportedGemmInputDtype(PrimType dtype) {
  return dtype.MatchesElementType(kDLBfloat, 16) ||
         (dtype.MatchesCode(kDLFloat) &&
          (dtype.bits() == 16 || dtype.bits() == 32)) ||
         dtype.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
         dtype.MatchesElementType(kDLFloat8_e5m2, 8) ||
         dtype.MatchesElementType(kDLFloat4_e2m1fn, 4);
}

bool IsSupportedBlockscaledGemmInputDtype(PrimType dtype) {
  return dtype.MatchesElementType(kDLBfloat, 16) ||
         dtype.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
         dtype.MatchesElementType(kDLFloat4_e2m1fn, 4);
}

int64_t GemmInputPackFactor(PrimType dtype) {
  return dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) ? 2 : 1;
}

int64_t GemmInputC0(PrimType dtype) {
  ICHECK(IsSupportedGemmInputDtype(dtype))
      << "PTO GEMM L1 helper only supports float16, bfloat16, float32, "
         "float8_e4m3fn, float8_e5m2, or float4_e2m1fn inputs, got "
      << dtype;
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
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
         dtype.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
         dtype.MatchesElementType(kDLFloat8_e5m2, 8) ||
         dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) ||
         (dtype.MatchesCode(kDLFloat) &&
          (dtype.bits() == 16 || dtype.bits() == 32) && dtype.lanes() == 1) ||
         (dtype.MatchesCode(kDLInt) && dtype.bits() == 8 && dtype.lanes() == 1);
}

bool IsSupportedFp8Dtype(PrimType dtype) {
  return dtype.IsScalar() && (dtype.MatchesElementType(kDLFloat8_e4m3fn, 8) ||
                              dtype.MatchesElementType(kDLFloat8_e5m2, 8));
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
  if (const auto *pointer = expr->ty.as<PtrTypeObj>())
    return pointer->element_type.as_or_throw<PrimType>();
  if (const auto *tensor = expr->ty.as<TensorTypeObj>())
    return tensor->dtype;
  if (const auto *call = expr.as<CallObj>();
      call && call->op.same_as(tl::access_ptr())) {
    const auto args = call->args;
    if (!args.empty())
      if (const auto *load = args[0].as<TensorLoadObj>())
        return load->source.as_or_throw<TensorVar>().dtype();
  }
  return fallback_dtype;
}

std::string PTOAddressSpace(const std::string &scope) {
  if (scope.empty() || scope == "global" || scope == "gm")
    return "gm";
  if (scope == "shared" || scope == "shared.dyn")
    return "ub";
  if (scope == "shared.l1" || scope == "shared.l1.dyn")
    return "mat";
  if (scope == "shared.l0a" || scope == "shared.l0a.dyn")
    return "left";
  if (scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return "right";
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return "acc";
  TVM_FFI_THROW(ValueError)
      << "PTO pointer operation has unsupported address space: " << scope;
}

// Vector-typed local buffers (S.alloc_local / S.alloc_var) have elements
// whose dtype carries lanes (e.g. float32x64). The pass pipeline widens a
// scalar element index `i` into a Ramp(i*lanes, 1, lanes). This recovers
// the scalar element index from such a Ramp.
PrimExpr VectorBufferElemIndex(const PrimExpr &index) {
  if (const auto *ramp = index.as<prim::RampObj>()) {
    int lanes = ramp->ty.as_or_throw<PrimType>().lanes();
    if (lanes > 0) {
      return sym::Analyzer()->Simplify(
          floordiv(ramp->base, MakeConst(ramp->base.ty(), lanes)));
    }
  }
  return index;
}

// All offsets are accumulated in bytes before converting to the pointee type
// required by the consuming instruction. A reinterpret never changes the
// address, and a declared view remains a distinct native pointer value.
bool GetNativeByteOffset(const Expr &expr, PrimExpr *bytes,
                         const VarObj **base) {
  if (const auto *var = expr.as<VarObj>()) {
    if (!var->ty.as<PtrTypeObj>() && !var->ty.as<TensorTypeObj>())
      return false;
    *base = var;
    *bytes = MakeConst(PrimType::Int(64), 0);
    return true;
  }
  const auto *call = expr.as<CallObj>();
  if (!call)
    return false;
  const auto args = call->args;
  if (call->op.same_as(tirx::tensor_data_ptr_op()) ||
      call->op.same_as(tirx::reinterpret_op())) {
    return args.size() == 1 && GetNativeByteOffset(args[0], bytes, base);
  }
  if (call->op.same_as(tirx::ptr_byte_offset_op()) ||
      call->op.same_as(tirx::handle_add_byte_offset_op()) ||
      call->op.same_as(tl::ptr_access())) {
    if (args.size() < 2 || !GetNativeByteOffset(args[0], bytes, base))
      return false;
    PrimExpr offset = args[1].as_or_throw<PrimExpr>();
    if (call->op.same_as(tl::ptr_access())) {
      offset = ElementIndexToByteOffsetExpr(
          offset, GetAnnotatedPointerDtype(args[0], PrimType::Void()),
          "PTO pointer access", sym::Analyzer());
    }
    *bytes = sym::Analyzer()->Simplify(*bytes +
                                       prim::Cast(PrimType::Int(64), offset));
    return true;
  }
  if ((call->op.same_as(tl::access_ptr()) ||
       call->op.same_as(tirx::address_of_op())) &&
      !args.empty()) {
    const auto *load = args[0].as<TensorLoadObj>();
    if (!load || load->indices.size() != 1)
      return false;
    const auto tensor = load->source.as_or_throw<TensorVar>();
    *base = tensor.get();
    sym::Analyzer analyzer;
    *bytes = ElementIndexToByteOffsetExpr(
        load->indices[0].ty().IsScalar()
            ? load->indices[0]
            : prim::Shuffle::ExtractElement(load->indices[0], 0),
        tensor.dtype(), "PTO tensor access", analyzer);
    return true;
  }
  return false;
}

bool GetAddressOfIndex(const Expr &expr, PrimExpr *index,
                       const VarObj **buffer_var) {
  PrimExpr bytes = MakeConst(PrimType::Int(64), 0);
  if (!GetNativeByteOffset(expr, &bytes, buffer_var))
    return false;
  PrimType dtype = GetAnnotatedPointerDtype(expr, PrimType::Void());
  if (dtype.IsVoid())
    return false;
  const int64_t bits = PhysicalElementBits(dtype);
  sym::Analyzer analyzer;
  if (bits < 8) {
    TVM_FFI_CHECK(bits > 0 && 8 % bits == 0, ValueError);
    *index = analyzer->Simplify(bytes * MakeConst(bytes.ty(), 8 / bits));
  } else {
    const int64_t element_bytes = PhysicalElementBytes(dtype);
    const PrimExpr divisor = MakeConst(bytes.ty(), element_bytes);
    TVM_FFI_CHECK(analyzer->CanProveEqual(floormod(bytes, divisor),
                                          MakeConst(bytes.ty(), 0)),
                  ValueError)
        << "PTO instruction pointer offset must align to its element width";
    *index = analyzer->Simplify(floordiv(bytes, divisor));
  }
  return true;
}

PrimExpr PTOLoopCondition(const ForObj *loop) {
  return ((loop->step > 0) && (loop->begin < loop->end)) ||
         ((loop->step < 0) && (loop->begin > loop->end));
}

PrimExpr PTOSingleIteration(const ForObj *loop) {
  return PTOLoopCondition(loop) &&
         (((loop->step > 0) && (loop->end - loop->begin <= loop->step)) ||
          ((loop->step < 0) && (loop->end - loop->begin >= loop->step)));
}

class PTOHf32ModeAnalyzer : public StmtFunctor<uint8_t(const Stmt &, uint8_t)> {
public:
  using ModeMap =
      std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>;

  static ModeMap Analyze(const tirx::Function &func) {
    PTOHf32ModeAnalyzer analyzer;
    analyzer.Dispatch(func->body, kDisabled);

    ModeMap result;
    for (const auto &[call, modes] : analyzer.gemm_modes_) {
      ICHECK_NE(modes, 0);
      ICHECK_EQ(modes & (modes - 1), 0)
          << "PTO HF32 mode is control-flow dependent for GEMM " << call
          << "; set one deterministic HF32 mode before this GEMM";
      result.emplace(call, ModeFromMask(modes));
    }
    return result;
  }

private:
  static constexpr int64_t kModeCount = 3;
  static constexpr int kMaxLoopAnalysisIterations = 1 << kModeCount;
  static constexpr uint8_t kDisabled = 1U << 0;
  static constexpr uint8_t kNearestZero = 1U << 1;
  static constexpr uint8_t kNearestEven = 1U << 2;

  static uint8_t MaskFromMode(int64_t mode) {
    ICHECK_GE(mode, 0);
    ICHECK_LT(mode, kModeCount);
    return 1U << mode;
  }

  static int64_t ModeFromMask(uint8_t mask) {
    if (mask == kDisabled)
      return 0;
    if (mask == kNearestZero)
      return 1;
    ICHECK_EQ(mask, kNearestEven);
    return 2;
  }

  uint8_t AnalyzeLoop(const Stmt &body, uint8_t incoming, bool must_execute) {
    uint8_t header = incoming;
    for (int i = 0; i < kMaxLoopAnalysisIterations; ++i) {
      uint8_t body_out = Dispatch(body, header);
      uint8_t next = incoming | body_out;
      if (next == header)
        return must_execute ? body_out : header;
      header = next;
    }
    TVM_FFI_THROW(InternalError)
        << "PTO HF32 loop analysis failed to converge for body " << body;
    return incoming;
  }

  uint8_t Dispatch_(const BindObj *op, uint8_t state) final { return state; }

  uint8_t Dispatch_(const RegionStmtObj *op, uint8_t state) final {
    return Dispatch(op->body, state);
  }

  uint8_t Dispatch_(const IfObj *op, uint8_t state) final {
    uint8_t then_out = Dispatch(op->then_case, state);
    uint8_t else_out =
        op->else_case ? Dispatch(op->else_case.value(), state) : state;
    return then_out | else_out;
  }

  uint8_t Dispatch_(const ForObj *op, uint8_t state) final {
    return AnalyzeLoop(op->body, state,
                       analyzer_->CanProve(PTOLoopCondition(op)));
  }

  uint8_t Dispatch_(const WhileObj *op, uint8_t state) final {
    return AnalyzeLoop(op->body, state, false);
  }

  uint8_t Dispatch_(const TensorStoreObj *op, uint8_t state) final {
    return state;
  }

  uint8_t Dispatch_(const AssertStmtObj *op, uint8_t state) final {
    return state;
  }

  uint8_t Dispatch_(const SeqStmtObj *op, uint8_t state) final {
    for (const Stmt &stmt : op->seq) {
      state = Dispatch(stmt, state);
    }
    return state;
  }

  uint8_t Dispatch_(const EvaluateObj *op, uint8_t state) final {
    const auto *call = op->value.as<CallObj>();
    if (call == nullptr)
      return state;

    if (call->op.same_as(tl::ascend_set_hf32_mode())) {
      ICHECK_EQ(call->args.size(), 1U)
          << "tl.ascend_set_hf32_mode expects exactly 1 argument";
      int64_t mode = 0;
      ICHECK(TryGetConstInt(call->args[0], &mode) && mode >= 0 && mode <= 2)
          << "PTO HF32 mode must be a constant in {0, 1, 2}, got "
          << call->args[0];
      return MaskFromMode(mode);
    }

    if (call->op.same_as(tl::ascend_gemm_l1())) {
      ICHECK_GT(call->args.size(), 9U);
      const auto *dtype_name = call->args[9].as<StringImmObj>();
      ICHECK(dtype_name) << "PTO GEMM requires a constant input dtype string";
      if (IsFloat32(ParseDataType(dtype_name->value))) {
        gemm_modes_[GetRef<Call>(call)] |= state;
      }
    } else if (call->op.same_as(tl::ascend_mad()) ||
               call->op.same_as(tl::ascend_mad_mx())) {
      // HF32 mode applies to subsequent Cube MADs. Preserve the reaching mode
      // for FP32 L0 operations as well as L1 GEMMs so PTOAS can encode it as
      // a per-MAD tf32_mode attribute.
      ICHECK_GT(call->args.size(), 2U);
      // Keep the reaching state for every MAD.  The emitter decides whether
      // the operand tuple is FP32 and therefore needs a PTOAS tf32_mode.
      gemm_modes_[GetRef<Call>(call)] |= state;
    }
    return state;
  }

  uint8_t Dispatch_(const BreakObj *, uint8_t state) final { return state; }
  uint8_t Dispatch_(const ContinueObj *, uint8_t state) final { return state; }
  uint8_t Dispatch_(const ReturnObj *, uint8_t state) final { return state; }

  std::unordered_map<Call, uint8_t, ObjectPtrHash, ObjectPtrEqual> gemm_modes_;
  sym::Analyzer analyzer_;
};

// Retain guards for immutable SSA conditions split apart by AutoSchedule.
struct MmadDirectionState {
  PrimExpr m = MakeConst(PrimType::Bool(), false);
  PrimExpr n = MakeConst(PrimType::Bool(), false);
};
struct MmadDirectionFlowState {
  MmadDirectionState fallthrough;
  MmadDirectionState breaks;
  MmadDirectionState continues;
};

// PTODSL encodes the stateful AscendC MAD direction on each MAD operation.
class PTOMmadDirectionAnalyzer : public StmtFunctor<MmadDirectionFlowState(
                                     const Stmt &, MmadDirectionState)> {
public:
  using DirectionMap =
      std::unordered_map<Call, bool, ObjectPtrHash, ObjectPtrEqual>;

  static DirectionMap Analyze(const tirx::Function &func) {
    bool has_mad = false;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        func->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          has_mad |= IsMad(call);
          return ffi::WalkResult::Advance();
        });
    if (!has_mad) {
      return {};
    }
    PTOMmadDirectionAnalyzer analyzer;
    analyzer.Dispatch(func->body, FromMask(kM));
    DirectionMap result;
    for (const auto &[call, directions] : analyzer.mad_directions_) {
      ICHECK(directions == 0 || directions == kM || directions == kN)
          << "PTO MAD direction depends on control flow for " << call;
      // Codegen can still emit unreachable MADs. Their direction is arbitrary.
      result.emplace(call, directions == kN);
    }
    return result;
  }

private:
  using FlowState = MmadDirectionFlowState;
  static constexpr uint8_t kM = 1;
  static constexpr uint8_t kN = 2;

  using State = MmadDirectionState;

  static State FromMask(uint8_t mask) {
    return {MakeConst(PrimType::Bool(), (mask & kM) != 0),
            MakeConst(PrimType::Bool(), (mask & kN) != 0)};
  }

  uint8_t Mask(const State &state) {
    return (analyzer_->CanProve(!state.m) ? 0 : kM) |
           (analyzer_->CanProve(!state.n) ? 0 : kN);
  }

  static bool IsMad(const CallObj *call) {
    return call->op.same_as(tl::ascend_mad()) ||
           call->op.same_as(tl::ascend_mad_mx()) ||
           call->op.same_as(tl::ascend_gemm_l1()) ||
           call->op.same_as(tl::ascend_blockscaled_gemm_l1());
  }

  PrimExpr MergePredicate(const PrimExpr &a, const PrimExpr &b) {
    // Direction predicates contain only pure SSA conditions (see Restrict).
    // Stateful guards leave both paths unchanged, so repeated merges would
    // otherwise double the predicate tree before arithmetic simplification.
    if (a.same_as(b) || ffi::StructuralEqual()(a, b)) {
      return a;
    }
    return analyzer_->Simplify(a || b);
  }

  State Merge(const State &a, const State &b) {
    return {MergePredicate(a.m, b.m), MergePredicate(a.n, b.n)};
  }

  State Restrict(const State &state, const PrimExpr &condition) {
    // Buffer reads can change between two identical-looking guards. Only SSA
    // conditions may be correlated; uncertain paths must remain reachable.
    if (SideEffect(condition) != CallEffectKind::kPure) {
      return state;
    }
    return {analyzer_->Simplify(state.m && condition),
            analyzer_->Simplify(state.n && condition)};
  }

  FlowState AnalyzeLoop(const Stmt &body, State incoming, bool must_execute,
                        bool can_exit_normally = true) {
    // Widen at backedges: an induction variable or body-local SSA snapshot
    // denotes a different value on the next iteration.
    uint8_t incoming_mask = Mask(incoming);
    uint8_t header = incoming_mask;
    for (int i = 0; i < 4; ++i) {
      FlowState body_out = Dispatch(body, FromMask(header));
      uint8_t next =
          incoming_mask | Mask(Merge(body_out.fallthrough, body_out.continues));
      if (next == header) {
        // Breaks exit this loop rather than flowing back to its header. A
        // nested loop consumes its own breaks before returning to this body.
        uint8_t exits = Mask(body_out.breaks);
        if (can_exit_normally) {
          exits |= Mask(Merge(body_out.fallthrough, body_out.continues));
          if (!must_execute) {
            exits |= incoming_mask;
          }
        }
        return {FromMask(exits), {}};
      }
      header = next;
    }
    TVM_FFI_THROW(InternalError)
        << "PTO MAD direction loop analysis failed to converge";
    return {incoming, {}};
  }

  FlowState Dispatch_(const BindObj *, State state) final {
    return {state, {}};
  }
  FlowState Dispatch_(const RegionStmtObj *op, State state) final {
    return Dispatch(op->body, state);
  }
  FlowState Dispatch_(const IfObj *op, State state) final {
    PrimExpr condition = analyzer_->Simplify(op->condition);
    if (const auto *imm = condition.as<IntImmObj>()) {
      if (imm->value != 0) {
        return Dispatch(op->then_case, state);
      }
      return op->else_case ? Dispatch(op->else_case.value(), state)
                           : FlowState{state, {}};
    }
    FlowState then_out = Dispatch(op->then_case, Restrict(state, condition));
    FlowState else_out =
        op->else_case
            ? Dispatch(op->else_case.value(), Restrict(state, !condition))
            : FlowState{Restrict(state, !condition), {}};
    return {Merge(then_out.fallthrough, else_out.fallthrough),
            Merge(then_out.breaks, else_out.breaks),
            Merge(then_out.continues, else_out.continues)};
  }
  FlowState Dispatch_(const ForObj *op, State state) final {
    if (analyzer_->CanProve(!PTOLoopCondition(op))) {
      // Codegen can still emit the body. Record its MADs as unreachable
      // without allowing its direction setters to affect subsequent code.
      Dispatch(op->body, {});
      return {state, {}};
    }
    if (analyzer_->CanProve(PTOSingleIteration(op))) {
      // A single-trip loop has no backedge. Both normal completion and break
      // leave the loop, so merge them directly into its fallthrough state.
      FlowState body_out = Dispatch(op->body, state);
      State exits = Merge(Merge(body_out.fallthrough, body_out.breaks),
                          body_out.continues);
      Map<Var, PrimExpr> values{{op->var, op->begin}};
      return {{analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
                   exits.m,
                   [&](const Var &var) -> ffi::Optional<PrimExpr> {
                     return values.Get(var);
                   })),
               analyzer_->Simplify(tirx::SubstituteWithDataTypeLegalization(
                   exits.n,
                   [&](const Var &var) -> ffi::Optional<PrimExpr> {
                     return values.Get(var);
                   }))},
              {}};
    }
    return AnalyzeLoop(op->body, state,
                       analyzer_->CanProve(PTOLoopCondition(op)));
  }
  FlowState Dispatch_(const WhileObj *op, State state) final {
    PrimExpr condition = analyzer_->Simplify(op->condition);
    if (IsZero(condition)) {
      Dispatch(op->body, {});
      return {state, {}};
    }
    const bool always_true = IsOne(condition);
    // A constant-true while only exits through break. Its normal body exit is
    // a backedge, and the incoming state cannot bypass the body.
    return AnalyzeLoop(op->body, state, always_true, !always_true);
  }

  FlowState Dispatch_(const TensorStoreObj *, State state) final {
    return {state, {}};
  }
  FlowState Dispatch_(const AssertStmtObj *, State state) final {
    return {state, {}};
  }
  FlowState Dispatch_(const SeqStmtObj *op, State state) final {
    FlowState result{state, {}};
    for (const Stmt &stmt : op->seq) {
      FlowState next = Dispatch(stmt, result.fallthrough);
      result.fallthrough = next.fallthrough;
      result.breaks = Merge(result.breaks, next.breaks);
      result.continues = Merge(result.continues, next.continues);
    }
    return result;
  }
  FlowState Dispatch_(const EvaluateObj *op, State state) final {
    const auto *call = op->value.as<CallObj>();
    if (call == nullptr)
      return {state, {}};
    if (call->op.same_as(tl::loop_break())) {
      return {{}, state};
    }
    if (call->op.same_as(tl::ascend_set_mmad_direction())) {
      ICHECK_EQ(call->args.size(), 1U);
      const auto *direction = call->args[0].as<StringImmObj>();
      ICHECK(direction && (direction->value == "m" || direction->value == "n"))
          << "PTO MAD direction expects 'm' or 'n'";
      // An unreachable setter must not revive the path after a break.
      if (Mask(state) == 0) {
        return {{}, {}};
      }
      PrimExpr reachable = analyzer_->Simplify(state.m || state.n);
      return {direction->value == "n"
                  ? State{MakeConst(PrimType::Bool(), false), reachable}
                  : State{reachable, MakeConst(PrimType::Bool(), false)},
              {}};
    }
    if (IsMad(call))
      mad_directions_[GetRef<Call>(call)] |= Mask(state);
    return {state, {}};
  }

  FlowState Dispatch_(const BreakObj *, State state) final {
    return {{}, state, {}};
  }
  FlowState Dispatch_(const ContinueObj *, State state) final {
    return {{}, {}, state};
  }
  FlowState Dispatch_(const ReturnObj *, State) final { return {}; }

  std::unordered_map<Call, uint8_t, ObjectPtrHash, ObjectPtrEqual>
      mad_directions_;
  sym::Analyzer analyzer_;
};

// These emitter queries use native statement dispatch and the structural
// expression walker. Region arguments remain visible; tensor identities come
// from TensorVar definitions, not from a legacy data-pointer side table.
class PTOIRVisitor : public StmtFunctor<void(const Stmt &)> {
public:
  void VisitExpression(const Expr &expression) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        expression,
        [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          ObserveCall(call);
          return ffi::WalkResult::Advance();
        },
        [&](const TensorLoadObj *load) -> ffi::Expected<ffi::WalkResult> {
          ObserveLoad(load);
          return ffi::WalkResult::Advance();
        });
  }

protected:
  virtual void ObserveCall(const CallObj *) {}
  virtual void ObserveLoad(const TensorLoadObj *) {}
  void Dispatch_(const BindObj *op) override { VisitExpression(op->value); }
  void Dispatch_(const RegionStmtObj *op) override {
    for (const Expr &arg : op->args)
      VisitExpression(arg);
    Dispatch(op->body);
  }
  void Dispatch_(const IfObj *op) override {
    VisitExpression(op->condition);
    Dispatch(op->then_case);
    if (op->else_case)
      Dispatch(op->else_case.value());
  }
  void Dispatch_(const ForObj *op) override {
    VisitExpression(op->begin);
    VisitExpression(op->end);
    VisitExpression(op->step);
    Dispatch(op->body);
  }
  void Dispatch_(const WhileObj *op) override {
    VisitExpression(op->condition);
    Dispatch(op->body);
  }
  void Dispatch_(const TensorStoreObj *op) override {
    VisitExpression(op->value);
    for (const auto &index : op->indices)
      VisitExpression(index);
  }
  void Dispatch_(const AssertStmtObj *op) override {
    VisitExpression(op->condition);
  }
  void Dispatch_(const SeqStmtObj *op) override {
    for (const Stmt &statement : op->seq)
      Dispatch(statement);
  }
  void Dispatch_(const EvaluateObj *op) override { VisitExpression(op->value); }
  void Dispatch_(const ReturnObj *op) override { VisitExpression(op->value); }
  void Dispatch_(const BreakObj *) override {}
  void Dispatch_(const ContinueObj *) override {}
};

class LoopBreakDetector final : public PTOIRVisitor {
public:
  bool found{false};

private:
  void ObserveCall(const CallObj *op) override {
    found |= op->op.same_as(tl::loop_break());
  }
  void Dispatch_(const BreakObj *) override { found = true; }
  void Dispatch_(const ForObj *) override {}
  void Dispatch_(const WhileObj *) override {}
};

bool ContainsLoopBreak(const Stmt &body) {
  LoopBreakDetector detector;
  detector(body);
  return detector.found;
}

class SimtPersistentBufferCollector final : public PTOIRVisitor {
public:
  std::unordered_set<const VarObj *> Collect(const Stmt &body) {
    Dispatch(body);
    std::unordered_set<const VarObj *> result;
    for (const VarObj *var : outer_local_allocations_)
      if (simt_accesses_.count(var))
        result.insert(var);
    return result;
  }

private:
  void RecordAccess(const VarObj *tensor) {
    if (simt_depth_ > 0)
      simt_accesses_.insert(tensor);
  }
  void Dispatch_(const BindObj *op) final {
    if (op->var->ty.as<TensorTypeObj>()) {
      const auto scope = CodegenStorageScope(op->var);
      if (simt_depth_ == 0 && (scope == "local" || scope == "local.fragment"))
        outer_local_allocations_.insert(op->var.get());
    }
    PTOIRVisitor::Dispatch_(op);
  }
  void Dispatch_(const RegionStmtObj *op) final {
    const bool simt =
        tl::ascend::IsAscendBlock(op) &&
        op->attrs.GetAttr<ffi::String>("name").value() == "SIMT_VF";
    simt_depth_ += simt;
    PTOIRVisitor::Dispatch_(op);
    simt_depth_ -= simt;
  }
  void ObserveLoad(const TensorLoadObj *op) final {
    RecordAccess(op->source.as_or_throw<TensorVar>().get());
  }
  void Dispatch_(const TensorStoreObj *op) final {
    RecordAccess(op->dest.as_or_throw<TensorVar>().get());
    PTOIRVisitor::Dispatch_(op);
  }
  int simt_depth_{0};
  std::unordered_set<const VarObj *> outer_local_allocations_;
  std::unordered_set<const VarObj *> simt_accesses_;
};

struct MixedSectionVariableInfo {
  using VarSet = std::unordered_set<const VarObj *>;
  std::unordered_map<const RegionStmtObj *, VarSet> external_vars_by_section;
  VarSet captured_vars;
};

class MixedSectionLocalVarAnalyzer {
  using VarSet = MixedSectionVariableInfo::VarSet;
  using OwnerMap = std::unordered_map<const VarObj *, const RegionStmtObj *>;

  static bool IsPhysical(const RegionStmtObj *op) {
    if (!tl::ascend::IsAscendBlock(op))
      return false;
    const auto name = op->attrs.GetAttr<ffi::String>("name").value();
    return name == "CUBE" || name == "VECTOR";
  }
  static bool IsLocalVar(const VarObj *tensor) {
    return tensor->ty.as<tirx::TensorTypeObj>() &&
           CodegenStorageScope(GetRef<Var>(tensor)) == "local.var";
  }
  static VarSet Intersect(VarSet left, const VarSet &right) {
    for (auto it = left.begin(); it != left.end();) {
      if (right.count(*it))
        ++it;
      else
        it = left.erase(it);
    }
    return left;
  }

  class StructureCollector final : public PTOIRVisitor {
  public:
    std::vector<const RegionStmtObj *> sections;
    OwnerMap owners;

  private:
    void Dispatch_(const BindObj *op) final {
      if (!IsLocalVar(op->var.get()))
        return;
      const auto [it, inserted] = owners.emplace(op->var.get(), current_);
      TVM_FFI_CHECK(inserted || it->second == current_, ValueError)
          << "A local.var tensor has multiple mixed-section owners";
    }
    void Dispatch_(const RegionStmtObj *op) final {
      const bool physical = IsPhysical(op);
      TVM_FFI_CHECK(!physical || current_ == nullptr, ValueError)
          << "Nested CUBE/VECTOR physical sections are not supported";
      const auto *prior = current_;
      if (physical) {
        current_ = op;
        sections.push_back(op);
      }
      PTOIRVisitor::Dispatch_(op);
      current_ = prior;
    }
    const RegionStmtObj *current_{nullptr};
  };

  class Flow : public StmtFunctor<VarSet(const Stmt &, VarSet)> {
  protected:
    virtual void Read(const VarObj *, const VarSet &) {}
    void ReadExpression(const Expr &expr, const VarSet &defined) {
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
          expr,
          [&](const TensorLoadObj *load) -> ffi::Expected<ffi::WalkResult> {
            Read(load->source.as_or_throw<TensorVar>().get(), defined);
            return ffi::WalkResult::Advance();
          });
    }
    VarSet Dispatch_(const BindObj *op, VarSet defined) override {
      ReadExpression(op->value, defined);
      if (IsLocalVar(op->var.get()))
        defined.insert(op->var.get());
      return defined;
    }
    VarSet Dispatch_(const RegionStmtObj *op, VarSet defined) override {
      for (const Expr &arg : op->args)
        ReadExpression(arg, defined);
      // A native region owns its lexical definitions. Definitions produced
      // inside do not become available in a sibling or after the region.
      Dispatch(op->body, defined);
      return defined;
    }
    VarSet Dispatch_(const IfObj *op, VarSet defined) override {
      ReadExpression(op->condition, defined);
      auto left = Dispatch(op->then_case, defined);
      auto right =
          op->else_case ? Dispatch(op->else_case.value(), defined) : defined;
      return Intersect(std::move(left), right);
    }
    VarSet Dispatch_(const ForObj *op, VarSet defined) override {
      ReadExpression(op->begin, defined);
      ReadExpression(op->end, defined);
      ReadExpression(op->step, defined);
      Dispatch(op->body, defined);
      return defined;
    }
    VarSet Dispatch_(const WhileObj *op, VarSet defined) override {
      ReadExpression(op->condition, defined);
      Dispatch(op->body, defined);
      return defined;
    }
    VarSet Dispatch_(const TensorStoreObj *op, VarSet defined) override {
      ReadExpression(op->value, defined);
      for (const auto &index : op->indices)
        ReadExpression(index, defined);
      Read(op->dest.as_or_throw<TensorVar>().get(), defined);
      return defined;
    }
    VarSet Dispatch_(const AssertStmtObj *op, VarSet defined) override {
      ReadExpression(op->condition, defined);
      return defined;
    }
    VarSet Dispatch_(const SeqStmtObj *op, VarSet defined) override {
      for (const auto &statement : op->seq)
        defined = Dispatch(statement, std::move(defined));
      return defined;
    }
    VarSet Dispatch_(const EvaluateObj *op, VarSet defined) override {
      ReadExpression(op->value, defined);
      return defined;
    }
    VarSet Dispatch_(const ReturnObj *op, VarSet defined) override {
      ReadExpression(op->value, defined);
      return defined;
    }
    VarSet Dispatch_(const BreakObj *, VarSet defined) override {
      return defined;
    }
    VarSet Dispatch_(const ContinueObj *, VarSet defined) override {
      return defined;
    }
  };

  class SectionFlow final : public Flow {
  public:
    SectionFlow(const RegionStmtObj *section, const OwnerMap &owners)
        : section_(section), owners_(owners) {}
    VarSet Analyze() {
      Dispatch(section_->body, {});
      return external_;
    }

  private:
    void Read(const VarObj *tensor, const VarSet &defined) final {
      if (!IsLocalVar(tensor))
        return;
      auto owner = owners_.find(tensor);
      TVM_FFI_CHECK(owner != owners_.end(), ValueError)
          << "Mixed-section local.var access has no allocation";
      TVM_FFI_CHECK(owner->second == nullptr || owner->second == section_,
                    ValueError)
          << "A mixed section cannot access local.var storage allocated by a "
             "sibling section";
      TVM_FFI_CHECK(defined.count(tensor) || owner->second == nullptr,
                    ValueError)
          << "Mixed-section local.var is accessed before its section-local "
             "allocation";
      if (owner->second == nullptr)
        external_.insert(tensor);
    }
    VarSet Dispatch_(const RegionStmtObj *op, VarSet defined) final {
      TVM_FFI_CHECK(!IsPhysical(op), ValueError)
          << "Nested CUBE/VECTOR physical sections are not supported";
      return Flow::Dispatch_(op, std::move(defined));
    }
    const RegionStmtObj *section_;
    const OwnerMap &owners_;
    VarSet external_;
  };

  class OuterFlow final : public Flow {
  public:
    explicit OuterFlow(MixedSectionVariableInfo *info) : info_(info) {}

  private:
    VarSet Dispatch_(const RegionStmtObj *op, VarSet defined) final {
      if (!IsPhysical(op))
        return Flow::Dispatch_(op, std::move(defined));
      for (const VarObj *var : info_->external_vars_by_section.at(op)) {
        TVM_FFI_CHECK(defined.count(var), ValueError)
            << "Mixed-section local.var access is not dominated by an outer "
               "allocation: "
            << var->name;
        info_->captured_vars.insert(var);
      }
      return defined;
    }
    MixedSectionVariableInfo *info_;
  };

public:
  static MixedSectionVariableInfo Analyze(const Stmt &body) {
    StructureCollector collector;
    collector(body);
    MixedSectionVariableInfo result;
    for (const auto *section : collector.sections)
      result.external_vars_by_section.emplace(
          section, SectionFlow(section, collector.owners).Analyze());
    OuterFlow checker(&result);
    checker(body, {});
    return result;
  }
};

bool UsesVar(ffi::AnyView body, const VarObj *target) {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      body, [&](const VarObj *node) -> ffi::Expected<ffi::WalkResult> {
        if (node == target) {
          found = true;
        }
        return ffi::WalkResult::Advance();
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

class CopyPadStateAnalyzer : public PTOIRVisitor {
public:
  explicit CopyPadStateAnalyzer(CopyPadState entry) : state_(entry) {}

  CopyPadState state() const { return state_; }

  void ObserveCall(const CallObj *op) override {
    if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
      ICHECK_EQ(op->args.size(), 1U);
      state_ =
          CopyPadState{true, true, op->args[0]->ty.as_or_throw<PrimType>()};
      return;
    }
  }

  void Dispatch_(const IfObj *op) override {
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
  void Dispatch_(const ForObj *op) override { (void)op; }
  void Dispatch_(const WhileObj *op) override { (void)op; }

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
class StableConditionChecker : public PTOIRVisitor {
public:
  bool stable{true};

  void ObserveLoad(const TensorLoadObj *op) override {
    (void)op;
    stable = false;
  }

  void ObserveCall(const CallObj *op) override {
    (void)op;
    stable = false;
  }
};

bool IsStableCondition(const Expr &condition) {
  StableConditionChecker checker;
  checker.VisitExpression(condition);
  return checker.stable;
}

bool AreComplementaryConditions(const PrimExpr &lhs, const PrimExpr &rhs) {
  auto same_operands = [](const PrimExpr &lhs_a, const PrimExpr &lhs_b,
                          const PrimExpr &rhs_a, const PrimExpr &rhs_b) {
    ffi::StructuralEqual equal;
    bool same = (equal(lhs_a, rhs_a) && equal(lhs_b, rhs_b)) ||
                (equal(lhs_a, rhs_b) && equal(lhs_b, rhs_a));
    return same && IsStableCondition(lhs_a) && IsStableCondition(lhs_b);
  };

  if (const auto *lhs_eq = lhs.as<prim::EQObj>()) {
    if (const auto *rhs_ne = rhs.as<prim::NEObj>()) {
      return same_operands(lhs_eq->a, lhs_eq->b, rhs_ne->a, rhs_ne->b);
    }
  }
  if (const auto *lhs_ne = lhs.as<prim::NEObj>()) {
    if (const auto *rhs_eq = rhs.as<prim::EQObj>()) {
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

void CodeGenTileLangPTO::ValidateVecMode_(const CallObj *op, size_t mode_idx,
                                          const char *expected) {
  ICHECK_LT(mode_idx, op->args.size())
      << "PTO SIMD op has no mode argument at index " << mode_idx;
  auto mode_str = (op->args[mode_idx]).as_or_throw<StringImm>()->value;
  ICHECK_EQ(mode_str, expected)
      << "PTO codegen currently only supports " << expected
      << " for this SIMD op, got " << mode_str;
}

bool CodeGenTileLangPTO::EmitSimdMergingCall_(const CallObj *op,
                                              std::ostream &os) { // NOLINT(*)
  std::string intrinsic_name;
  if (!IsSimdMergingCall(op, &intrinsic_name)) {
    return false;
  }

  ICHECK_GE(op->args.size(), 4U) << "Legalized PTO MODE_MERGING call must "
                                    "contain a destination, operands, "
                                    "mask, and mode";
  // The native destination carries the vector element type in PtrType.
  // GetMutableVectorRef resolves tensor_data_ptr/byte offsets and the retained
  // TileLang access marker to the owning local register slot.
  const PrimType result_dtype =
      GetAnnotatedPointerDtype(op->args[0], PrimType::Void());

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
                      {result_dtype}, op->loc);
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
      result_elem.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
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
  current_unroll_factor_loop_var_ = Optional<Var>();
  current_unroll_factor_ = 0;
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
  TVM_FFI_CHECK((!func->GetAttr(tvm::attr::kDeclaration).value_or(false)),
                ValueError)
      << "PTO codegen requires a function definition";
  RegisterFunction_(gvar, func);
  current_function_name_ = GetFunctionName_(gvar);
  InitFuncState_(func);
  ResetFunctionState_();
  name_supply_->ReserveName("pto");
  name_supply_->ReserveName("scalar");
  name_supply_->ReserveName("tl");
  ValidateKernelCapabilities(func);
  persistent_buffer_vars_ = SimtPersistentBufferCollector().Collect(func->body);
  hf32_mode_by_gemm_ = PTOHf32ModeAnalyzer::Analyze(func);
  n_direction_by_mad_ = PTOMmadDirectionAnalyzer::Analyze(func);
  bool saw_copy_pad_value = false;
  bool has_uniform_const_copy_pad_value = true;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (!call->op.same_as(tl::ascend_set_copy_pad_value())) {
          return ffi::WalkResult::Advance();
        }
        ICHECK_EQ(call->args.size(), 1U);
        PrimExpr value = call->args[0].as_or_throw<PrimExpr>();
        bool is_const = value.as<IntImmObj>() != nullptr ||
                        value.as<FloatImmObj>() != nullptr;
        if (!is_const) {
          has_uniform_const_copy_pad_value = false;
        } else if (!saw_copy_pad_value) {
          uniform_const_copy_pad_value_ = value;
        } else if (!ffi::StructuralEqual()(
                       uniform_const_copy_pad_value_.value(), value)) {
          has_uniform_const_copy_pad_value = false;
        }
        saw_copy_pad_value = true;
        return ffi::WalkResult::Advance();
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
        MixedSectionLocalVarAnalyzer::Analyze(func->body);
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
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        func->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          if (call->op.same_as(tl::ascend_gemm_l1()) ||
              call->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
            gemm_calls.push_back(GetRef<Call>(call));
          }
          return ffi::WalkResult::Advance();
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
  PrintStmt_(func->body);
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
    std::ostream &os) {
  os << "def " << function_name << "(";
  for (size_t index = 0; index < func->params.size(); ++index) {
    const Var &var = func->params[index];
    if (index)
      os << ", ";
    os << AllocVarID(var.get()) << ": ";
    const auto *pointer = var->ty.as<PtrTypeObj>();
    const auto tensor = var.as<tirx::TensorVar>();
    if (pointer || tensor) {
      const PrimType element =
          tensor ? tensor->dtype()
                 : pointer->element_type.as_or_throw<PrimType>();
      std::string scope =
          tensor ? tensor->scope().name() : pointer->scope.name();
      if (scope == "global")
        scope = "gm";
      else if (scope == "shared" || scope == "shared.dyn")
        scope = "ub";
      os << PointerTypeName(element, scope);
      RegisterHandleType_(var.get(), element);
      alloc_storage_scope_[var.get()] = CodegenStorageScope(var);
    } else {
      const auto type = var->ty.as_or_throw<PrimType>();
      TVM_FFI_CHECK(type.IsScalar() && !type.IsVoid(), TypeError)
          << "PTO scalar parameters require a scalar primitive type";
      os << ScalarType(type);
    }
  }
  os << ")";
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

bool CodeGenTileLangPTO::NeedsCastptr_(const VarObj *buffer_var,
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
  if (storage_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) &&
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

std::string CodeGenTileLangPTO::ScalarPointerBase_(const VarObj *buffer_var,
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
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (found)
          return ffi::WalkResult::Advance();

        found = call->op.same_as(tl::ascend_gemm_l1());
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool CodeGenTileLangPTO::HasAscendBlockscaledGemmL1(
    const tirx::Function &func) const {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (found)
          return ffi::WalkResult::Advance();

        found = call->op.same_as(tl::ascend_blockscaled_gemm_l1());
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool CodeGenTileLangPTO::IsAscendCubeKernel(const tirx::Function &func) const {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body,
      [&](const RegionStmtObj *block) -> ffi::Expected<ffi::WalkResult> {
        if (found)
          return ffi::WalkResult::Advance();

        found = tl::ascend::IsAscendBlock(block) &&
                block->attrs.GetAttr<ffi::String>("name").value() == "CUBE";
        return ffi::WalkResult::Advance();
      },
      [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (found)
          return ffi::WalkResult::Advance();

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
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool CodeGenTileLangPTO::HasAscendMad(const tirx::Function &func) const {
  bool found = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body, [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if (found)
          return ffi::WalkResult::Advance();

        found = call->op.same_as(tl::ascend_mad()) ||
                call->op.same_as(tl::ascend_mad_mx());
        return ffi::WalkResult::Advance();
      });
  return found;
}

bool CodeGenTileLangPTO::IsAscendMixedKernel(const tirx::Function &func) const {
  bool has_cube = false;
  bool has_vector = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      func->body,
      [&](const RegionStmtObj *block) -> ffi::Expected<ffi::WalkResult> {
        if (!tl::ascend::IsAscendBlock(block))
          return ffi::WalkResult::Advance();
        const auto name = block->attrs.GetAttr<ffi::String>("name").value();
        has_cube = has_cube || name == "CUBE";
        has_vector = has_vector || name == "VECTOR";
        return ffi::WalkResult::Advance();
      });
  return has_cube && has_vector;
}

bool CodeGenTileLangPTO::IsInlineableInvariantSimdBind(
    const PrimExpr &value) const {
  const auto *call = value.as<CallObj>();
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

std::string CodeGenTileLangPTO::ScopeOfBuffer(const VarObj *buffer) const {
  std::string scope;
  auto it = alloc_storage_scope_.find(buffer);
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = CodegenStorageScope(GetRef<Var>(buffer));
  }
  return scope;
}

void CodeGenTileLangPTO::ExtractSimtThreadExtents(const RegionStmtObj *region,
                                                  int64_t *thread_x,
                                                  int64_t *thread_y,
                                                  int64_t *thread_z) const {
  *thread_x = *thread_y = *thread_z = 1;
  std::unordered_map<std::string, int64_t> seen;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      region->body,
      [&](const RegionStmtObj *launch) -> ffi::Expected<ffi::WalkResult> {
        if (!launch->op.same_as(tirx::launch_thread_op()))
          return ffi::WalkResult::Advance();
        TVM_FFI_CHECK(launch->args.size() == 2 &&
                          launch->body_params.size() == 1 &&
                          launch->result_vars.empty(),
                      ValueError)
            << "Malformed native PTO thread launch";
        const std::string tag = launch->args[0].as_or_throw<StringImm>()->value;
        int64_t *dimension = nullptr;
        if (tag == "threadIdx.x")
          dimension = thread_x;
        else if (tag == "threadIdx.y")
          dimension = thread_y;
        else if (tag == "threadIdx.z")
          dimension = thread_z;
        else
          return ffi::WalkResult::Advance();
        const int64_t extent = CodegenInt(launch->args[1]);
        TVM_FFI_CHECK_GT(extent, 0, ValueError)
            << "PTO inline SIMT launch dimensions must be positive";
        auto [it, inserted] = seen.emplace(tag, extent);
        TVM_FFI_CHECK(inserted || it->second == extent, ValueError)
            << "Conflicting PTO SIMT extents for " << tag;
        *dimension = extent;
        return ffi::WalkResult::Advance();
      });
}

void CodeGenTileLangPTO::EmitInlineSimtVF(const RegionStmtObj *op,
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

  PrintStmt_(op->body);
  if (stream.tellp() == body_start) {
    PrintIndent();
    stream << "pass\n";
  }

  inside_simtvf_body_ = saved_inside_simtvf_body;
  EndScope(simt_scope);
}

std::string CodeGenTileLangPTO::GetPointerExpr(const VarObj *buffer_var,
                                               PrimType elem_dtype,
                                               const PrimExpr &index) {
  const std::string scope = ScopeOfBuffer(buffer_var);
  const std::string pto_space = PTOAddressSpace(scope);
  std::string base = GetVarID(buffer_var);
  // Preserve the native allocation/view base in every addressable space.
  // Cube allocations use integer byte addresses; other allocations and
  // declarations can already be typed pointers. Cast before element-wise
  // addition so reinterpret views keep their physical byte addresses.
  if (NeedsCastptr_(buffer_var, elem_dtype)) {
    base = "pto.castptr(" + base + ", " +
           PointerTypeName(elem_dtype, pto_space) + ")";
  }

  if (IsZero(index)) {
    return base;
  }

  PrimExpr ptr_index = index;
  if (const auto *ramp = index.as<prim::RampObj>();
      ramp && ramp->stride.as<IntImmObj>()) {
    ptr_index = ramp->base;
  } else if (!index.ty().IsScalar()) {
    ptr_index = prim::Shuffle::ExtractElement(index, 0);
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

  return "pto.addptr(" + base + ", " + index_str + ")";
}

std::string CodeGenTileLangPTO::GetPointerExpr(const VarObj *buffer,
                                               const PrimExpr &index) {
  return GetPointerExpr(buffer, CodegenTensorDType(buffer), index);
}

std::string CodeGenTileLangPTO::GetVectorLocalRef(const VarObj *buffer_var,
                                                  const PrimExpr &index,
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
  const VarObj *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(address, &dst_index, &buffer_var))
      << "PTO " << op_name
      << " expects a typed local tensor address as its destination, got "
      << address;

  auto scope_it = alloc_storage_scope_.find(buffer_var);
  std::string scope;
  if (scope_it != alloc_storage_scope_.end()) {
    scope = scope_it->second;
  } else {
    scope = CodegenStorageScope(GetRef<Var>(buffer_var));
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
  PrimType dtype = condition->ty.as_or_throw<PrimType>();
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

std::string CodeGenTileLangPTO::ScalarLoad(const VarObj *buffer,
                                           const std::string &index_str) {
  std::string scope = ScopeOfBuffer(buffer);
  if (scope == "local.var") {
    return GetVarID(buffer);
  }

  if (inside_simtvf_body_ && (scope == "local.fragment" || scope == "local")) {
    ICHECK(IsSupportedSIMTLocalScalarAccessType(
        CodegenTensorDType(buffer).WithLanes(1)))
        << "PTO SIMT local scalar load supports float16, float32, bfloat16, "
           "int8, uint8, int32, uint32, and FP8 only, got "
        << CodegenTensorDType(buffer);
    std::string value = "pto.load(" + GetVarID(buffer) + ", " + index_str + ")";
    if (tl::IsAscendVectorizableFP8(CodegenTensorDType(buffer))) {
      return "tl.fp8_byte_load(" + GetVarID(buffer) + ", " + index_str + ", " +
             FP8TypeName(CodegenTensorDType(buffer)) + ")";
    }
    if (IsInteger32(CodegenTensorDType(buffer))) {
      return "pto.cast(" + value + ", " +
             DataTypeName(CodegenTensorDType(buffer)) + ")";
    }
    return value;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    std::string base =
        ScalarPointerBase_(buffer, CodegenTensorDType(buffer), scope);
    if (inside_simtvf_body_ &&
        tl::IsAscendVectorizableFP8(CodegenTensorDType(buffer))) {
      return "tl.fp8_byte_load(" + base + ", " + index_str + ", " +
             FP8TypeName(CodegenTensorDType(buffer)) + ")";
    }
    return "pto.load(" + base + ", " + index_str + ")";
  }

  if (scope == "local.fragment" || scope == "local") {
    return GetVarID(buffer) + "[" + index_str + "]";
  }

  TVM_FFI_THROW(InternalError)
      << "Unsupported PTO scalar load scope: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::LocalVarStoreValue(const PrimExpr &value,
                                                   PrimType dtype) {
  std::string value_expr = RemoveOutermostParentheses(PrintExpr_(value));
  if (dtype.MatchesCode(kDLOpaqueHandle)) {
    return value_expr;
  }
  bool is_literal =
      value.as<IntImmObj>() != nullptr || value.as<FloatImmObj>() != nullptr;
  if (dtype.IsScalar() && is_literal) {
    return "pto.const(" + value_expr + ", dtype=" + DataTypeName(dtype) + ")";
  }
  if (dtype.MatchesCode(kDLBool) || dtype.bits() == 1 ||
      dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt)) {
    return ScalarCastExpr(value_expr, dtype, "PTO local.var store");
  }
  return value_expr;
}

void CodeGenTileLangPTO::EmitScalarStore(const VarObj *buffer,
                                         const std::string &value,
                                         const std::string &index_str) {
  std::string scope = ScopeOfBuffer(buffer);

  if (scope == "local.var") {
    stream << GetVarID(buffer) << " = " << value << "\n";
    EmitMixedEntrySnapshot(buffer);
    return;
  }

  if (scope == "local.fragment" || scope == "local") {
    if (inside_simtvf_body_) {
      // The index addresses scalar storage, including lanes of vector locals.
      PrimType dtype = CodegenTensorDType(buffer).WithLanes(1);
      ICHECK(IsSupportedSIMTLocalScalarAccessType(dtype))
          << "PTO SIMT local scalar store supports float16, bfloat16, "
             "float32, int8, uint8, int32, uint32, and FP8 only, got "
          << dtype;
      std::string store_value = value;
      if (tl::IsAscendVectorizableFP8(dtype)) {
        stream << "tl.fp8_byte_store(" << store_value << ", "
               << GetVarID(buffer) << ", " << index_str << ", "
               << FP8TypeName(dtype) << ")\n";
        return;
      }
      if (IsInteger32(dtype)) {
        store_value = "pto.cast(" + store_value + ", pto.i32)";
      }
      stream << "pto.store(" << store_value << ", " << GetVarID(buffer) << ", "
             << index_str << ")\n";
      return;
    }
    stream << GetVarID(buffer) << "[" << index_str << "] = " << value << "\n";
    return;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    std::string base =
        ScalarPointerBase_(buffer, CodegenTensorDType(buffer), scope);
    if (inside_simtvf_body_ &&
        tl::IsAscendVectorizableFP8(CodegenTensorDType(buffer))) {
      stream << "tl.fp8_byte_store(" << value << ", " << base << ", "
             << index_str << ", " << FP8TypeName(CodegenTensorDType(buffer))
             << ")\n";
      return;
    }
    stream << "pto.store(" << value << ", " << base << ", " << index_str
           << ")\n";
    return;
  }

  TVM_FFI_THROW(InternalError)
      << "Unsupported PTO scalar store scope: " << scope;
}

void CodeGenTileLangPTO::EmitLocalVarInitialization_(const VarObj *buffer,
                                                     std::string vid) {
  PrimType dtype = CodegenTensorDType(buffer);
  if (vid.rfind("__cond_", 0) == 0) {
    vid = "tl_cond_" + vid.substr(7);
    var_idmap_[buffer] = vid;
  } else if (!vid.empty() && vid.front() == '_') {
    // PTODSL rewrites native Python if statements through BranchHandle,
    // whose underscore-prefixed attributes are reserved for internals.
    // Give local.var values such as `_tmp_id` a unique public name so they
    // can be carried out of nested branches.
    vid = name_supply_->FreshName("tl" + vid, false);
    var_idmap_[buffer] = vid;
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
    RegisterHandleType_(buffer, CodegenTensorDType(buffer));
    EmitMixedEntrySnapshot(buffer);
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
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO local.var dtype: " << dtype;
  }
  EmitMixedEntrySnapshot(buffer);
  RegisterHandleType_(buffer, CodegenTensorDType(buffer));
}

void CodeGenTileLangPTO::EmitBufferAllocation(const VarObj *buffer) {
  std::string scope = CodegenStorageScope(GetRef<Var>(buffer));
  alloc_storage_scope_[buffer] = scope;
  ValidateSharedScope(scope);

  if (scope == "shared.dyn" || scope == "shared.l1" ||
      scope == "shared.l1.dyn" || scope == "shared.l0a" ||
      scope == "shared.l0a.dyn" || scope == "shared.l0b" ||
      scope == "shared.l0b.dyn" || scope == "shared.l0c" ||
      scope == "shared.l0c.dyn") {
    PrintIndent();
    std::string vid = AllocVarID(buffer);
    // Arena passes assign view offsets; an allocation denotes the arena base.
    // L1/L0 storage can also be consumed in a vector function (for example
    // UB-to-L1 post-copy), where no Cube allocation emitter binds its name.
    stream << vid << " = pto.castptr(pto.const(0, dtype=pto.i64), "
           << PointerTypeName(CodegenTensorDType(buffer),
                              PTOAddressSpace(scope))
           << ")\n";
    RegisterHandleType_(buffer, CodegenTensorDType(buffer));
    return;
  } else if (scope == "local.fragment" || scope == "local") {
    // Allocations nested in a runtime loop keep the same TIR Var identity and
    // are re-initialized on each iteration.  The loop/carry analysis may also
    // reserve the name before reaching this statement.  Reuse the stable
    // Python name while still emitting the initialization at this location.
    std::string vid =
        var_idmap_.count(buffer) ? GetVarID(buffer) : AllocVarID(buffer);
    std::optional<int64_t> opt_size =
        CodegenAllocationElements(CodegenTensor(buffer));
    ICHECK(opt_size.has_value())
        << "PTO local.fragment currently requires constant allocation size";
    PrintIndent();
    const bool persistent = persistent_buffer_vars_.count(buffer) != 0;
    PrimType elem = (!CodegenTensorDType(buffer).IsScalar())
                        ? CodegenTensorDType(buffer).WithLanes(1)
                        : CodegenTensorDType(buffer);
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalStorageType(CodegenTensorDType(buffer)))
          << "PTO SIMT local allocation supports float16, bfloat16, "
             "float32, int8, uint8, int32, uint32, and "
             "float8_e4m3fn/float8_e5m2, got "
          << CodegenTensorDType(buffer);
      // SIMT locals use scalar-addressed storage, including vector slots.
      stream << vid << " = pto.alloc_buffer(("
             << opt_size.value() * CodegenTensorDType(buffer).lanes() << ",), "
             << SIMTLocalStorageTypeName(elem) << ")\n";
    } else if (persistent) {
      ICHECK(IsFloat32(CodegenTensorDType(buffer)))
          << "PTO persistent local allocation currently supports float32 "
             "only, got "
          << CodegenTensorDType(buffer);
      stream << vid << " = pto.alloc_buffer((" << opt_size.value() << ",), "
             << ScalarType(elem) << ")\n";
    } else if ((!CodegenTensorDType(buffer).IsScalar())) {
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
    if (local_var_buffers_.count(buffer)) {
      return;
    }
    local_var_buffers_.insert(buffer);
    std::string vid =
        var_idmap_.count(buffer) ? GetVarID(buffer) : AllocVarID(buffer);
    EmitLocalVarInitialization_(buffer, std::move(vid));
    return;
  }

  RegisterHandleType_(buffer, CodegenTensorDType(buffer));
}

std::string CodeGenTileLangPTO::GetAscendCopyGmUbExpr_(const CallObj *op) {
  ICHECK_EQ(op->args.size(), 11U)
      << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";
  int64_t sid = 0;
  ICHECK(TryGetConstInt(op->args[2], &sid) && sid == 0)
      << "PTO GM->UB MTE requires sid == 0, got " << op->args[2];

  const VarObj *dst_var = nullptr;
  const VarObj *src_var = nullptr;
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
    ICHECK(padding_analyzer->CanProveEqual(
               op->args[5].as_or_throw<PrimExpr>(),
               MakeConst(op->args[5]->ty.as_or_throw<PrimType>(), 0)) &&
           padding_analyzer->CanProveEqual(
               op->args[6].as_or_throw<PrimExpr>(),
               MakeConst(op->args[6]->ty.as_or_throw<PrimType>(), 0)))
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
      pad_dtype = uniform_const_copy_pad_value_.value().ty();
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

std::string CodeGenTileLangPTO::GetAscendCopyUbGmExpr_(const CallObj *op) {
  ICHECK_EQ(op->args.size(), 8U)
      << "tl.ascend_copy_ubuf_to_gm expects exactly 8 arguments";
  int64_t sid = 0;
  ICHECK(TryGetConstInt(op->args[2], &sid) && sid == 0)
      << "PTO UB->GM MTE requires sid == 0, got " << op->args[2];

  const VarObj *dst_var = nullptr;
  const VarObj *src_var = nullptr;
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
  ValidateUBCopyLayout_(
      src_index, src_dtype, op->args[3].as_or_throw<PrimExpr>(),
      op->args[4].as_or_throw<PrimExpr>(), op->args[7].as_or_throw<PrimExpr>(),
      MakeConst(PrimType::Int(32), 0), MakeConst(PrimType::Int(32), 0), false,
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

void CodeGenTileLangPTO::RecordAscendCopyPadValue_(const CallObj *op) {
  ICHECK_EQ(op->args.size(), 1U)
      << "tl.ascend_set_copy_pad_value expects exactly 1 argument";
  PrimType dtype = op->args[0]->ty.as_or_throw<PrimType>();
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

void CodeGenTileLangPTO::EmitAscendSetAtomic(const CallObj *op) {
  ICHECK_EQ(op->args.size(), 2U)
      << "tl.ascend_set_atomic expects exactly 2 arguments (op, typed_zero)";
  const auto *atomic_op = op->args[0].as<StringImmObj>();
  ICHECK(atomic_op)
      << "tl.ascend_set_atomic operation must be a compile-time string";
  ICHECK(atomic_op->value == "add" || atomic_op->value == "max" ||
         atomic_op->value == "min")
      << "PTO store atomic operation must be add, max, or min, got "
      << atomic_op->value;

  PrimType dtype = op->args[1]->ty.as_or_throw<PrimType>();
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
  if (const auto *imm = value.as<IntImmObj>()) {
    std::ostringstream literal;
    literal << imm->value;
    return "pto.const(" + literal.str() +
           ", dtype=" + ScalarType(value->ty.as_or_throw<PrimType>()) + ")";
  }
  if (const auto *imm = value.as<FloatImmObj>()) {
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
           ", dtype=" + ScalarType(value->ty.as_or_throw<PrimType>()) + ")";
  }
  return RemoveOutermostParentheses(PrintExpr_(value));
}

void CodeGenTileLangPTO::GetCopyEndpoint_(const Expr &expr, const char *context,
                                          const VarObj **buffer_var,
                                          PrimExpr *index, PrimType *dtype,
                                          std::string *scope) const {
  ICHECK(GetAddressOfIndex(expr, index, buffer_var))
      << context << " expects a native typed pointer, got " << expr;
  *dtype = GetAnnotatedPointerDtype(expr, PrimType::Void());
  ICHECK(!dtype->IsVoid()) << context << " has no element dtype annotation";

  auto scope_it = alloc_storage_scope_.find(*buffer_var);
  if (scope_it != alloc_storage_scope_.end()) {
    *scope = scope_it->second;
  } else {
    *scope = CodegenStorageScope(GetRef<Var>(*buffer_var));
  }
  if (*scope == "gm") {
    *scope = "global";
  }
}

void CodeGenTileLangPTO::ValidateUBCopyLayout_(
    const PrimExpr &index, PrimType dtype, const PrimExpr &burst_num,
    const PrimExpr &burst_len, const PrimExpr &ub_stride,
    const PrimExpr &left_padding, const PrimExpr &right_padding,
    bool uses_padding, const char *context) const {
  ICHECK(dtype.IsScalar() || IsPackedFp4Pair(dtype))
      << context << " requires a scalar or packed FP4-pair dtype, got "
      << dtype;

  sym::Analyzer analyzer;
  PrimExpr elem_index = index;
  if (const auto *ramp = index.as<prim::RampObj>()) {
    CheckContiguousRampStride(index, context);
    elem_index = analyzer->Simplify(ramp->base);
  }

  PrimExpr byte_offset =
      ElementIndexToByteOffsetExpr(elem_index, dtype, context, analyzer);
  PrimExpr offset_mod = analyzer->Simplify(
      floormod(byte_offset, MakeConst(byte_offset.ty(), 32)));
  if (!analyzer->CanProveEqual(offset_mod, MakeConst(offset_mod.ty(), 0))) {
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
    ICHECK(analyzer->CanProve(left_padding >= MakeConst(left_padding.ty(), 0)))
        << context << " requires a non-negative left padding count, got "
        << left_padding;
  }
  if (!has_const_right) {
    ICHECK(
        analyzer->CanProve(right_padding >= MakeConst(right_padding.ty(), 0)))
        << context << " requires a non-negative right padding count, got "
        << right_padding;
  }
  if (uses_padding) {
    if (!has_const_len) {
      ICHECK(analyzer->CanProve(burst_len >= MakeConst(burst_len.ty(), 0)))
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

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToUb(const CallObj *op) {
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

  const VarObj *dst_var = nullptr;
  const VarObj *src_var = nullptr;
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

std::string CodeGenTileLangPTO::GetLocalByteAddrExpr(
    const PrimExpr &index, PrimType elem_dtype, const std::string &context) {
  PrimExpr elem_index = index;
  if (const auto *ramp = index.as<prim::RampObj>()) {
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
    TVM_FFI_THROW(InternalError)
        << context << " dynamic sub-byte local pointer offset requires native "
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
  const auto *pointer = expr->ty.as<PtrTypeObj>();
  TVM_FFI_CHECK(pointer, TypeError)
      << "PTO local operation expects a native typed pointer";
  const std::string actual = PTOAddressSpace(pointer->scope.name());
  TVM_FFI_CHECK(actual == space, ValueError)
      << "PTO local pointer expects " << space << " address space, got "
      << actual;
  const PrimType dtype = GetAnnotatedPointerDtype(expr, fallback_dtype);
  // Keep the full native pointer expression. Rebuilding it from only a local
  // offset discards a nonzero arena/view base and aliases the wrong allocation.
  return "pto.castptr(" + PrintExpr_(expr) + ", " +
         PointerTypeName(dtype, space) + ")";
}

std::string CodeGenTileLangPTO::GetE8M0ScalePtrExpr(const Expr &expr) {
  const auto *pointer = expr->ty.as<PtrTypeObj>();
  TVM_FFI_CHECK(pointer && PTOAddressSpace(pointer->scope.name()) == "mat",
                TypeError)
      << "PTO E8M0 scale pointer must refer to an L1 allocation";
  const PrimType dtype = pointer->element_type.as_or_throw<PrimType>();
  TVM_FFI_CHECK(dtype == PrimType::UInt(8) || dtype == PrimType::UInt(16) ||
                    dtype == PrimType::Int(16),
                TypeError)
      << "PTO E8M0 scale storage must be uint8 or pair-packed 16-bit";
  return "pto.castptr(" + PrintExpr_(expr) + ", pto.ptr(pto.f8e8m0, \"mat\"))";
}

std::string CodeGenTileLangPTO::GetAccPtrExpr(const Expr &expr,
                                              PrimType dtype) {
  return GetLocalPtrExpr(expr, "acc", dtype);
}

std::string CodeGenTileLangPTO::LocalVarID(const VarObj *var) {
  return GetVarID(var);
}

bool CodeGenTileLangPTO::IsLocalVarBuffer(const VarObj *var) const {
  return local_var_buffers_.count(var) != 0;
}

void CodeGenTileLangPTO::EmitMixedEntrySnapshot(const VarObj *var) {
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
    const RegionStmtObj *section) {
  auto section_vars = mixed_external_vars_by_section_.find(section);
  ICHECK(section_vars != mixed_external_vars_by_section_.end())
      << "Missing mixed-section variable analysis for "
      << section->attrs.GetAttr<ffi::String>("name").value();

  std::vector<std::pair<std::string, std::string>> restores;
  restores.reserve(section_vars->second.size());
  for (const VarObj *var : section_vars->second) {
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
CodeGenTileLangPTO::EnsureGemmHelper(const CallObj *op) {
  ICHECK_EQ(op->args.size(), 12U)
      << "tl.ascend_gemm_l1 expects exactly 12 arguments";
  int64_t tile_m = ConstArgDim(op, 3, "tl.ascend_gemm_l1 M");
  int64_t tile_k = ConstArgDim(op, 4, "tl.ascend_gemm_l1 K");
  int64_t tile_n = ConstArgDim(op, 5, "tl.ascend_gemm_l1 N");
  int64_t base_k = ConstArgDim(op, 6, "tl.ascend_gemm_l1 tile_k_sub");
  int64_t trans_b = ConstArgDim(op, 7, "tl.ascend_gemm_l1 trans_b");
  ICHECK_EQ(trans_b, 1) << "PTO GEMM L1 helper currently requires trans_b=1";
  ICHECK_EQ(tile_k % base_k, 0);

  const auto *dtype_name = op->args[9].as<StringImmObj>();
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
CodeGenTileLangPTO::EnsureBlockscaledGemmHelper(const CallObj *op) {
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

  const auto *dtype_name = op->args[11].as<StringImmObj>();
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

  CheckConstZero(op->args[14].as_or_throw<PrimExpr>(),
                 "blockscaled GEMM buf_offset");

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
    const PrimExpr &index, PrimType dtype, const char *context) const {
  const bool is_fp4_packed = IsPackedFp4Pair(dtype);
  ICHECK((dtype.IsScalar() || is_fp4_packed))
      << context << " requires a scalar or packed-FP4 dtype, got " << dtype;
  sym::Analyzer analyzer;
  PrimExpr byte_offset =
      ElementIndexToByteOffsetExpr(index, dtype, context, analyzer);
  PrimExpr offset_mod = analyzer->Simplify(
      floormod(byte_offset, MakeConst(byte_offset.ty(), 32)));
  if (!analyzer->CanProveEqual(offset_mod, MakeConst(offset_mod.ty(), 0))) {
    int64_t remainder = 0;
    ICHECK(!TryGetConstInt(offset_mod, &remainder))
        << context << " address must be 32-byte aligned, but element offset "
        << index << " for dtype " << dtype << " has byte remainder "
        << remainder << " modulo 32";
  }
}

void CodeGenTileLangPTO::EmitAscendCopyGmToCbuf(const CallObj *op) {
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
  const auto *physical_dtype_imm = op->args[11].as<StringImmObj>();
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

  const VarObj *dst_var = nullptr;
  const VarObj *src_var = nullptr;
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
    const bool is_fp4 = dst_dtype.MatchesElementType(kDLFloat4_e2m1fn, 4);
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

  ValidateFractalAddressAlignment_(dst_index, dst_dtype,
                                   "PTO GM->L1 MTE destination");
  bool has_const_source_row_elems =
      transpose == 0 ? has_const_d_value : has_const_n_value;
  int64_t source_row_elems = transpose == 0 ? d_value : n_value;
  ICHECK(!has_const_src_inner_stride || !has_const_source_row_elems ||
         src_inner_stride >=
             source_row_elems * PhysicalElementBytes(transfer_dtype))
      << "PTO GM->L1 MTE source row stride is smaller than the source row: "
      << src_inner_stride << " bytes for " << source_row_elems
      << " physical elements of " << transfer_dtype;

  std::string dst;
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  if (has_physical) {
    // Reinterpreting packed elements changes the pointee type, not the
    // complete arena/view address carried by the native pointer.
    dst = "pto.castptr(" + PrintExpr_(op->args[0]) + ", " +
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

void CodeGenTileLangPTO::EmitAscendFillL1(const CallObj *op) {
  ICHECK_EQ(op->args.size(), 7U)
      << "tl.ascend_fill_l1 expects exactly 7 arguments";

  const VarObj *dst_var = nullptr;
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

void CodeGenTileLangPTO::EmitAscendLoadCbufToL0(const CallObj *op, bool is_ca) {
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

  const VarObj *dst_var = nullptr;
  const VarObj *src_var = nullptr;
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

  const VarObj *sf_var = nullptr;
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

void CodeGenTileLangPTO::EmitAscendLoadMxSf(const CallObj *op, bool is_ca) {
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

  const VarObj *dst_var = nullptr;
  const VarObj *sf_var = nullptr;
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

void CodeGenTileLangPTO::EmitAscendCrossCoreFlag(const CallObj *op,
                                                 bool is_set) {
  ICHECK_EQ(op->args.size(), 3U)
      << "tl.ascend_cross_core_" << (is_set ? "set" : "wait")
      << "_flag expects exactly 3 arguments (mode_id, pipe, flag_id)";
  int64_t mode_id = 0;
  ICHECK(TryGetConstInt(op->args[0], &mode_id))
      << "PTO cross-core flag requires a constant mode_id";
  ICHECK(mode_id == 0 || mode_id == 1 || mode_id == 2 || mode_id == 4)
      << "PTO cross-core flag mode_id must be one of 0/1/2/4, got " << mode_id;

  const auto *pipe_imm = op->args[1].as<StringImmObj>();
  ICHECK(pipe_imm) << "PTO cross-core flag requires a constant pipe string";
  PrimType event_dtype = op->args[2]->ty.as_or_throw<PrimType>();
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

void CodeGenTileLangPTO::EmitAscendGemmL1(const CallObj *op) {
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

void CodeGenTileLangPTO::EmitAscendBlockscaledGemmL1(const CallObj *op) {
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

void CodeGenTileLangPTO::EmitAscendMad(const CallObj *op) {
  const bool is_mx = op->op.same_as(tl::ascend_mad_mx());
  ICHECK(op->op.same_as(tl::ascend_mad()) || is_mx);
  const char *op_name = is_mx ? "mad_mx" : "mad";
  ICHECK_EQ(op->args.size(), 10U)
      << "tl.ascend_" << op_name << " expects exactly 10 arguments";

  auto get_endpoint = [&](size_t arg_index, const char *context) {
    const VarObj *buffer_var = nullptr;
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
                                const PrimExpr &unit_flag_expr) -> void {
    int64_t unit_flag_ctrl = 0;
    if (TryGetConstInt(unit_flag_expr, &unit_flag_ctrl)) {
      emit_for_clear_accum(unit_flag_ctrl);
      return;
    }

    PrimExpr condition = MakeConst(PrimType::Int(32), 0);
    PrimExpr true_value = MakeConst(PrimType::Int(32), 0);
    PrimExpr false_value = MakeConst(PrimType::Int(32), 0);
    if (const auto *select = unit_flag_expr.as<prim::SelectObj>()) {
      condition = select->condition;
      true_value = select->true_value;
      false_value = select->false_value;
    } else if (const auto *call = unit_flag_expr.as<CallObj>();
               call != nullptr &&
               (call->op.same_as(prim::if_then_else_op()) ||
                call->op.same_as(prim::if_then_else_op())) &&
               call->args.size() == 3U) {
      condition = call->args[0].as_or_throw<PrimExpr>();
      true_value = call->args[1].as_or_throw<PrimExpr>();
      false_value = call->args[2].as_or_throw<PrimExpr>();
    } else {
      TVM_FFI_THROW(InternalError)
          << "PTO " << op_name
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

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToGm(const CallObj *op) {
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

void CodeGenTileLangPTO::EmitAscendCopyUbufToCbuf(const CallObj *op) {
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

void CodeGenTileLangPTO::EmitAscendNd2NzPostCopy(const CallObj *op) {
  auto geometry = tl::ParseNd2NzGeometry(op, false);
  if (geometry.Empty())
    return;
  std::string dst = GetLocalPtrExpr(op->args[0], "mat", PrimType::BFloat(16));
  std::string src = GetLocalPtrExpr(op->args[1], "ub", PrimType::BFloat(16));
  int64_t burst_num = geometry.Bursts();
  int64_t burst_len = geometry.rows;
  int64_t src_gap = geometry.src_pitch - geometry.rows;
  int64_t dst_gap = geometry.dst_pitch - geometry.rows;

  PrintIndent();
  stream << "pto.mte_ub_l1(" << src << ", " << dst << ", " << burst_len
         << ", nburst=(" << burst_num << ", " << src_gap << ", " << dst_gap
         << "))\n";
}

void CodeGenTileLangPTO::EmitAscendNd2NzScatter(const CallObj *op) {
  auto geometry = tl::ParseNd2NzGeometry(op, true);
  if (geometry.Empty())
    return;
  int64_t rows = geometry.rows, cols = geometry.cols;
  std::string src_dtype = (op->args[5]).as_or_throw<StringImm>()->value;
  std::string dst_dtype = (op->args[6]).as_or_throw<StringImm>()->value;

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
    int dst_pass_offset =
        pass * static_cast<int>(geometry.dst_pitch) * vl_elems;

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
             << row_dst_offset << "), " << geometry.dst_pitch << ", 0, "
             << mask_fn << ")\n";
    }
  }
}

std::string CodeGenTileLangPTO::EmitAllReduceExpr_(const std::string &func_name,
                                                   const CallObj *op) {
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

void CodeGenTileLangPTO::Dispatch_(const CallObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const auto args = op->args;
  if (op->op.same_as(tirx::address_of_op())) {
    TVM_FFI_CHECK_EQ(args.size(), 1, ValueError);
    const auto *load = args[0].as<TensorLoadObj>();
    TVM_FFI_CHECK(load && load->indices.size() == 1, ValueError)
        << "PTO address_of requires a flat TensorLoad";
    // Taking an address must not emit the TensorLoad. In particular, scalar
    // GM cache-bypass calls require a pointer to the original element, not
    // its value or the beginning of the tensor.
    os << GetPointerExpr(load->source.as_or_throw<TensorVar>().get(),
                         load->indices[0]);
    return;
  }
  if (op->op.same_as(tirx::tensor_data_ptr_op())) {
    TVM_FFI_CHECK_EQ(args.size(), 1, ValueError);
    const auto tensor = args[0].as_or_throw<TensorVar>();
    const std::string base = GetVarID(tensor.get());
    if (integer_address_vars_.count(tensor)) {
      os << "pto.castptr(" << base << ", "
         << PointerTypeName(tensor.dtype(),
                            PTOAddressSpace(tensor.scope().name()))
         << ")";
    } else {
      os << base;
    }
    return;
  }
  if (op->op.same_as(tirx::ptr_byte_offset_op()) ||
      op->op.same_as(tirx::handle_add_byte_offset_op()) ||
      op->op.same_as(tl::ptr_access())) {
    TVM_FFI_CHECK(args.size() >= 2 && op->ty.as<PtrTypeObj>(), TypeError);
    const PtrType type = op->ty.as_or_throw<PtrType>();
    const PrimType element = type->element_type.as_or_throw<PrimType>();
    const std::string space = PTOAddressSpace(type->scope.name());
    PrimExpr bytes = args[1].as_or_throw<PrimExpr>();
    if (op->op.same_as(tl::ptr_access())) {
      bytes = ElementIndexToByteOffsetExpr(
          bytes, GetAnnotatedPointerDtype(args[0], PrimType::Void()),
          "PTO pointer access", sym::Analyzer());
    }
    os << "pto.castptr(pto.addptr(pto.castptr(" << PrintExpr_(args[0]) << ", "
       << PointerTypeName(PrimType::UInt(8), space) << "), "
       << PrintExpr_(bytes) << "), " << PointerTypeName(element, space) << ")";
    return;
  }
  if (op->op.same_as(tirx::reinterpret_op()) && op->ty.as<PtrTypeObj>()) {
    TVM_FFI_CHECK_EQ(args.size(), 1, ValueError);
    const PtrType type = op->ty.as_or_throw<PtrType>();
    os << "pto.castptr(" << PrintExpr_(args[0]) << ", "
       << PointerTypeName(type->element_type.as_or_throw<PrimType>(),
                          PTOAddressSpace(type->scope.name()))
       << ")";
    return;
  }

  if (EmitSimdMergingCall_(op, os)) {
    return;
  }
  if (op->op.same_as(prim::if_then_else_op()) ||
      op->op.same_as(prim::if_then_else_op())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "if_then_else expects <condition, true_value, false_value>";
    const bool requires_statement_control_flow =
        RequiresStatementControlFlow(op->args[1]) ||
        RequiresStatementControlFlow(op->args[2]);
    auto branch_value = [&](const Expr &expr) {
      std::string value = PrintExpr_(expr);
      if (expr->ty.as_or_throw<PrimType>().MatchesCode(kDLBool)) {
        // PTODSL typed constants require numeric bool literals.
        if (value == "True") {
          value = "1";
        } else if (value == "False") {
          value = "0";
        }
      }
      const PrimType dtype = op->ty.as_or_throw<PrimType>();
      if (expr.as<IntImmObj>() != nullptr ||
          expr.as<FloatImmObj>() != nullptr) {
        return "pto.const(" + value + ", dtype=" + DataTypeName(dtype) + ")";
      }
      if (dtype.MatchesCode(kDLBool)) {
        return "tl.as_logical_bool(" + value + ")";
      }
      if (dtype.IsScalar() &&
          (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt))) {
        return ScalarCastExpr(value, dtype, "PTO if_then_else branch");
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

  if (op->op.same_as(builtin_call_extern_) ||
      op->op.same_as(builtin_call_pure_extern_)) {
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
    PrimType dtype = op->args[0]->ty.as_or_throw<PrimType>();
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
      const auto *memory_order = op->args[2].as<IntImmObj>();
      ICHECK(memory_order)
          << "PTO atomic memory_order must be a compile-time integer constant";
      ICHECK_EQ(memory_order->value, 0)
          << "PTO atomic operations currently support only memory_order="
             "\"relaxed\" because PTODSL/PTOAS atomic operations do not expose "
             "memory-order semantics; got memory_order id "
          << memory_order->value;
    }

    PrimType dtype = GetAnnotatedPointerDtype(
        op->args[0], op->args[1]->ty.as_or_throw<PrimType>());
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

  if (op->op.same_as(tirx::gpu_storage_sync_op())) {
    ICHECK_EQ(op->args.size(), 1);
    const auto sync_scope = op->args[0].as_or_throw<StringImm>()->value;
    if (sync_scope == "warp") {
      return;
    }
    if (sync_scope == "shared" || sync_scope == "shared.dyn") {
      PrintIndent();
      stream << (inside_simtvf_body_ ? "pto.syncthreads()\n"
                                     : "pto.pipe_barrier(pto.Pipe.ALL)\n");
      return;
    }
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO storage sync scope: " << sync_scope;
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
  // BufferLoad/Store into these Calls. Map to PTODSL cache-bypass helpers,
  // which lower to pto.ld_dev/st_dev.

  if (op->op.same_as(tl::ascend_read_gm_bypass_dcache()) ||
      op->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
    const bool load = op->op.same_as(tl::ascend_read_gm_bypass_dcache());
    TVM_FFI_CHECK_EQ(args.size(), load ? 1 : 2, ValueError);
    const PtrType pointer = args[0]->ty.as_or_throw<PtrType>();
    TVM_FFI_CHECK(PTOAddressSpace(pointer->scope.name()) == "gm", ValueError)
        << "PTO GM dcache bypass expects a global pointer";
    const PrimType element = pointer->element_type.as_or_throw<PrimType>();
    ValidateGmBypassDtype(element);
    if (load) {
      TVM_FFI_CHECK(op->ty.as_or_throw<PrimType>() == element, TypeError);
      os << "tl.read_gm_bypass_dcache(" << PrintExpr_(args[0]) << ", 0, "
         << ScalarType(element) << ")";
    } else {
      TVM_FFI_CHECK(args[1]->ty.as_or_throw<PrimType>() == element, TypeError);
      PrintIndent();
      stream << "tl.write_gm_bypass_dcache(" << PrintExpr_(args[0]) << ", 0, "
             << PrintExpr_(args[1]) << ", " << ScalarType(element) << ")\n";
    }
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

  if (op->op.same_as(tirx::reinterpret_op())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tirx.reinterpret expects 1 argument (value)";
    std::string value = PrintExpr_(op->args[0]);
    PrimType target_dtype = op->ty.as_or_throw<PrimType>();
    if (target_dtype.lanes() > 1) {
      // Vector reinterpret: lower to pto.vbitcast(vec, elem_type).
      os << "pto.vbitcast(" << value << ", "
         << ScalarType(target_dtype.WithLanes(1)) << ")";
    } else {
      PrimType source_dtype = op->args[0]->ty.as_or_throw<PrimType>();
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

  if (op->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *load = op->args[0].as<TensorLoadObj>();
    ICHECK(load) << "tl.access_ptr arg0 must be BufferLoad";
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
      const auto *dist_imm = op->args[1].as<StringImmObj>();
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
    const auto *dist_imm = op->args[1].as<StringImmObj>();
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
    const auto *dist_imm = op->args[2].as<StringImmObj>();
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
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    PrimType elem_dtype = dtype.WithLanes(1);
    ICHECK_GT(dtype.lanes(), 1)
        << "tl.simd.vld should return a vector type, got " << dtype;
    std::string dist =
        NormalizeVldDistribution((op->args[1]).as_or_throw<StringImm>()->value);
    std::string offset =
        op->args.size() == 3U
            ? RemoveOutermostParentheses(PrintExpr_(op->args[2]))
            : "pto.const(0)";
    int pto_lanes = dtype.lanes();
    if (elem_dtype.bits() < 8) {
      ICHECK_EQ((pto_lanes * elem_dtype.bits()) % 8, 0)
          << "PTO vld sub-byte vector lanes must form whole bytes, got "
          << dtype;
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
    PrimType elem_dtype = op->ty.as_or_throw<PrimType>().WithLanes(1);
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
      const PrimType dtype = op->ty.as_or_throw<PrimType>();
      const bool is_f32 = dtype.WithLanes(1).MatchesCode(kDLFloat) &&
                          dtype.WithLanes(1).bits() == 32;
      const SfuPrecision prec = ResolveSfuPrecision(
          GetRef<Call>(op), dtype,
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
      TVM_FFI_THROW(InternalError)
          << "PTO codegen does not support MODE_MERGING for "
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
    PrimType input_elem = op->args[0]->ty.as_or_throw<PrimType>().WithLanes(1);
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
          GetRef<Call>(op), op->ty.as_or_throw<PrimType>(), SfuPrecision::kHw);
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
    const auto *part_imm = op->args[1].as<StringImmObj>();
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
    TVM_FFI_THROW(InternalError)
        << op->op
        << " must be bound via T.bind for PTO codegen before its pair "
           "elements are used";
    return;
  }

  // Leaky/parametric ReLU.
  if (op->op.same_as(tl::simd_vlrelu()) || op->op.same_as(tl::simd_vprelu())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd vlrelu/vprelu expects 3 arguments";
    if (op->op.same_as(tl::simd_vlrelu())) {
      PrimType vector_dtype = op->args[0]->ty.as_or_throw<PrimType>();
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
    PrimType result_dtype = op->args[0]->ty.as_or_throw<PrimType>();
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
    PrimType input_elem = op->args[0]->ty.as_or_throw<PrimType>().WithLanes(1);
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
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    PrimType elem = dtype.WithLanes(1);
    ICHECK((elem.MatchesCode(kDLInt) || elem.MatchesCode(kDLUInt)) &&
           (elem.bits() == 8 || elem.bits() == 16 || elem.bits() == 32))
        << "tl.simd.vusqz result must be an 8-, 16-, or 32-bit integer vector, "
           "got "
        << dtype;
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
    PrimType elem = op->ty.as_or_throw<PrimType>().WithLanes(1);
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
    PrimType elem = op->ty.as_or_throw<PrimType>().WithLanes(1);
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
        op->ty.as_or_throw<PrimType>(), IsImmediateScalar(op->args[0]));
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
    if (const auto *part_imm = op->args[1].as<IntImmObj>()) {
      part = CodegenInt(part_imm);
    } else if (const auto *part_str = op->args[1].as<StringImmObj>()) {
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
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    PrimType elem = dtype.WithLanes(1);
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
       << ", result_vreg_type=pto.vreg_type(" << dtype.lanes() << ", "
       << ScalarType(elem) << "))";
    return;
  }

  if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK(op->args.size() >= 4U && op->args.size() <= 6U)
        << "tl.simd.vcvt expects 4 to 6 arguments (src, mask, ...extra)";
    PrimType elem_dtype = op->ty.as_or_throw<PrimType>().WithLanes(1);
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
        TVM_FFI_THROW(InternalError)
            << "PTO codegen does not support MODE_MERGING for vcvt yet";
      } else {
        TVM_FFI_THROW(InternalError)
            << "PTO codegen: unrecognized vcvt argument: " << val;
      }
    }
    PrimType src_elem_dtype =
        op->args[0]->ty.as_or_throw<PrimType>().WithLanes(1);
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
    if (op->args.size() >= 4U && op->args[3].as<StringImmObj>()) {
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
        dist, op->args[1]->ty.as_or_throw<PrimType>().WithLanes(1));
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
    ICHECK_EQ(op->ty.as<PtrTypeObj>() != nullptr, post_update)
        << "tl.simd.vsstb post-update form must return a handle";
    PrimType stride_dtype = op->args[2]->ty.as_or_throw<PrimType>();
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
      const auto *post = op->args[4].as<StringImmObj>();
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
    PrimType value_dtype = op->args[0]->ty.as_or_throw<PrimType>().WithLanes(1);
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
       << "EVEN"
       << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_vgather2())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.vgather2 expects 3 arguments (base, index, mask)";
    std::string base = PrintExpr_(op->args[0]);
    std::string index = PrintExpr_(op->args[1]);
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    PrimType elem_dtype = dtype.WithLanes(1);
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
    std::string result_type = "pto.vreg_type(" + std::to_string(dtype.lanes()) +
                              ", " + ScalarType(elem_dtype) + ")";
    os << "pto.vgather2(" << base << ", " << index << ", " << mask << ", "
       << result_type << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vintlv()) || op->op.same_as(tl::simd_vdintlv())) {
    // Pair-producing ops are handled in VisitStmt_(BindObj) where the tuple
    // result is unpacked into two SSA names. Reaching here means the call was
    // not bound (e.g. inlined), which is unsupported for PTO codegen.
    TVM_FFI_THROW(InternalError)
        << "tl.simd.vintlv/vdintlv must be bound via T.bind for PTO "
        << "codegen (use `a, b = T.simd.vintlv(x, y)`)";
    return;
  }

  if (op->op.same_as(tl::simd_vld2())) {
    TVM_FFI_THROW(InternalError)
        << "tl.simd.vld2 must be bound via T.bind for PTO codegen "
        << "(use `a, b = T.simd.vld2(addr, dist=...)`)";
    return;
  }

  if (op->op.same_as(tl::simd_pair_get())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.pair_get expects 2 arguments (pair, index)";
    const auto *var = op->args[0].as<VarObj>();
    ICHECK(var != nullptr)
        << "tl.simd.pair_get expects a bound pair variable as arg0";
    auto it = simd_pair_vars_.find(var);
    ICHECK(it != simd_pair_vars_.end())
        << "tl.simd.pair_get references an unknown pair variable " << var->name
        << " in " << GetRef<Call>(op);
    int64_t index = CodegenInt(op->args[1]);
    TVM_FFI_CHECK(index == 0 || index == 1, ValueError)
        << "PTO SIMD pair_get index must be 0 or 1";
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
      TVM_FFI_THROW(InternalError)
          << "PTO codegen does not support TileLang op `" << op_name
          << "`. Please add a handler in CodeGenTileLangPTO or lower "
             "it before PTO codegen.";
    }
  }

  CodeGenTileLangPY::Dispatch_(op, os);
}

void CodeGenTileLangPTO::Dispatch_(const FloatImmObj *op,
                                   std::ostream &os) { // NOLINT(*)
  if (!op->ty.as_or_throw<PrimType>().MatchesElementType(kDLBfloat, 16)) {
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

void CodeGenTileLangPTO::Dispatch_(const prim::CastObj *op,
                                   std::ostream &os) { // NOLINT(*)
  PrimType from = op->value.ty();
  PrimType to = op->ty.as_or_throw<PrimType>();
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
    TVM_FFI_THROW(InternalError)
        << "PTO SIMT FP8 cast currently supports float32x2 to "
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
  const bool immediate_value = op->value.as<IntImmObj>() != nullptr ||
                               op->value.as<FloatImmObj>() != nullptr;
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
    if (const auto *imm = op->value.as<IntImmObj>()) {
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
  if (from_integer && to_integer && !from.IsScalar()) {
    os << "tl.vector_from_list(" << DataTypeName(to.WithLanes(1))
       << ", tuple(pto.cast(pto.cast(lane, " << DataTypeName(from.WithLanes(1))
       << "), " << DataTypeName(to.WithLanes(1))
       << ") for lane in tl.vector_to_list(" << PrintExpr_(op->value) << ")))";
    return;
  }

  if (to.MatchesCode(kDLBool)) {
    ICHECK(from.IsScalar() && (from_integer || from_float))
        << "PTO bool cast expects a scalar numeric source, got " << from;
    // Integer sources already normalize above; runtime floating-point
    // sources already use ScalarCastExpr.
    const auto *imm = op->value.as<FloatImmObj>();
    ICHECK(imm) << "PTO bool cast expected a floating-point immediate";
    os << (imm->value != 0.0 ? "True" : "False");
    return;
  }

  if (from_integer && to_integer) {
    // All scalar integer casts were handled above. Only unsupported vector
    // narrowing remains; preserve its diagnostic instead of emitting a cast.
    TVM_FFI_THROW(InternalError)
        << "PTO integer cast currently supports scalar values only, got "
        << from << " -> " << to;
  }

  if (from_integer && to_float) {
    ICHECK(from.IsScalar() && to.IsScalar())
        << "PTO integer-to-float cast currently supports scalar values only, "
           "got "
        << from << " -> " << to;
    if (const auto *imm = op->value.as<IntImmObj>()) {
      os << "pto.const(" << imm->value << ", dtype=" << DataTypeName(to) << ")";
      return;
    }
    // Non-immediate scalar integer-to-float casts returned above through
    // ScalarCastExpr; only the rejected vector cases can reach here.
  }

  if (const auto *imm = op->value.as<FloatImmObj>();
      imm && scalar_value &&
      ((IsFloat32(from) && to.MatchesElementType(kDLBfloat, 16)) ||
       (from.MatchesElementType(kDLBfloat, 16) && IsFloat32(to)))) {
    os << "pto.const(float.fromhex('" << FlexibleHexFormat(imm->value)
       << "'), dtype=" << DataTypeName(to) << ")";
    return;
  }

  TVM_FFI_THROW(InternalError)
      << "Unsupported PTO cast: " << from << " -> " << to;
}

void CodeGenTileLangPTO::Dispatch_(const prim::LetObj *op,
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

void CodeGenTileLangPTO::Dispatch_(const prim::ShuffleObj *op,
                                   std::ostream &os) { // NOLINT(*)
  // Shuffle picks lanes by constant index from its concatenated inputs;
  // tl.shuffle_vec gathers them into a scalar or packed vector.
  int64_t total_lanes = 0;
  for (const PrimExpr &vector : op->vectors) {
    PrimType dtype = vector.ty();
    if (dtype.lanes() > 1) {
      ICHECK(!tl::IsAscendVectorizableFP8(dtype))
          << "PTO cannot split FP8 builtin vectors lane-by-lane, got " << dtype;
    }
    total_lanes += dtype.lanes();
  }
  ICHECK_GE(total_lanes, 1) << "PTO Shuffle expects at least one input lane";
  for (const PrimExpr &index : op->indices) {
    const auto *constant_index = index.as<IntImmObj>();
    ICHECK(constant_index != nullptr)
        << "ShuffleObj indices must be constants at PTO codegen time, got "
        << index;
    ICHECK_GE(constant_index->value, 0)
        << "ShuffleObj index must be non-negative, got "
        << constant_index->value;
    ICHECK_LT(constant_index->value, total_lanes)
        << "ShuffleObj index " << constant_index->value
        << " is outside the concatenated input with " << total_lanes
        << " lanes";
  }

  os << "tl.shuffle_vec("
     << DataTypeName(op->ty.as_or_throw<PrimType>().WithLanes(1)) << ", (";
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

namespace {
void ValidatePTOShift(const PrimExpr &value, const PrimExpr &amount) {
  int64_t shift = 0;
  TVM_FFI_CHECK(TryGetConstInt(amount, &shift) && shift >= 0, ValueError)
      << "PTO shifts require a constant non-negative amount";
  const PrimType type = value.ty();
  TVM_FFI_CHECK(type.IsScalar() && type.MatchesCode(kDLInt, kDLUInt), TypeError)
      << "PTO shifts expect a scalar integer operand";
  TVM_FFI_CHECK_LT(shift, type.bits(), ValueError)
      << "PTO shift amount must be smaller than operand bit width";
}
} // namespace

void CodeGenTileLangPTO::Dispatch_(const prim::LShiftObj *op,
                                   std::ostream &os) {
  ValidatePTOShift(op->a, op->b);
  PrintBinaryExpr_("<<", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPTO::Dispatch_(const prim::RShiftObj *op,
                                   std::ostream &os) {
  ValidatePTOShift(op->a, op->b);
  PrintBinaryExpr_(">>", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}

void CodeGenTileLangPTO::Dispatch_(const prim::MinObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const PrimType result_type = op->ty.as_or_throw<PrimType>();
  if (inside_simtvf_body_ && tl::IsAscendVectorizableFP8(result_type)) {
    TVM_FFI_THROW(InternalError)
        << "PTO SIMT scalar FP8 min is not supported; only +, -, and * "
           "between non-immediate operands of the same FP8 dtype are "
           "currently supported, got "
        << op->a.ty() << " and " << op->b.ty();
  }
  if (result_type.MatchesCode(kDLInt) || result_type.MatchesCode(kDLUInt)) {
    os << "pto.min(";
    PrintExpr_(op->a, os);
    os << ", ";
    PrintExpr_(op->b, os);
    os << ")";
    return;
  }
  PrintFloatMinMax_("minimum", result_type, op->a, op->b, os);
}

void CodeGenTileLangPTO::Dispatch_(const prim::MaxObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const PrimType result_type = op->ty.as_or_throw<PrimType>();
  if (inside_simtvf_body_ && tl::IsAscendVectorizableFP8(result_type)) {
    TVM_FFI_THROW(InternalError)
        << "PTO SIMT scalar FP8 max is not supported; only +, -, and * "
           "between non-immediate operands of the same FP8 dtype are "
           "currently supported, got "
        << op->a.ty() << " and " << op->b.ty();
  }
  if (result_type.MatchesCode(kDLInt) || result_type.MatchesCode(kDLUInt)) {
    os << "pto.max(";
    PrintExpr_(op->a, os);
    os << ", ";
    PrintExpr_(op->b, os);
    os << ")";
    return;
  }
  PrintFloatMinMax_("maximum", result_type, op->a, op->b, os);
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
  PrimType dtype = arg->ty.as_or_throw<PrimType>();
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

void CodeGenTileLangPTO::Dispatch_(const prim::NotObj *op,
                                   std::ostream &os) { // NOLINT(*)
  ICHECK(op->ty.as_or_throw<PrimType>().IsScalar() &&
         op->a.ty().MatchesCode(kDLBool))
      << "PTO logical-not currently supports scalar bool predicates only, got "
      << op->a.ty();
  // tl.logical_not(value) implements logical negation as value == 0.
  os << "tl.logical_not(";
  PrintExpr_(op->a, os);
  os << ")";
}

void CodeGenTileLangPTO::Dispatch_(const prim::SelectObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const PrimType result_type = op->ty.as_or_throw<PrimType>();
  const bool coerce_integer_branches =
      result_type.IsScalar() &&
      (result_type.MatchesCode(kDLInt) || result_type.MatchesCode(kDLUInt));
  auto print_select_value = [&](const PrimExpr &expr) {
    if (expr.as<IntImmObj>() != nullptr || expr.as<FloatImmObj>() != nullptr) {
      os << "pto.const(" << PrintExpr_(expr)
         << ", dtype=" << DataTypeName(result_type) << ")";
      return;
    }
    if (result_type.IsScalar() && result_type.MatchesCode(kDLBool)) {
      os << "tl.as_logical_bool(" << PrintExpr_(expr) << ")";
      return;
    }
    if (coerce_integer_branches) {
      os << ScalarCastExpr(PrintExpr_(expr), result_type, "PTO select branch");
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

void CodeGenTileLangPTO::Dispatch_(const prim::ModObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  if (dtype.IsScalar() &&
      (dtype.MatchesCode(kDLInt) || dtype.MatchesCode(kDLUInt))) {
    os << "(" << PrintExpr_(op->a) << " % " << PrintExpr_(op->b) << ")";
    return;
  }
  CodeGenTileLangPY::Dispatch_(op, os);
}

void CodeGenTileLangPTO::Dispatch_(const prim::DivObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  if (dtype.IsScalar() && dtype.MatchesCode(kDLInt)) {
    os << "pto.div(" << PrintExpr_(op->a) << ", " << PrintExpr_(op->b) << ")";
    return;
  }
  CodeGenTileLangPY::Dispatch_(op, os);
}

void CodeGenTileLangPTO::PrintBinaryExpr_(const std::string &opstr,
                                          PrimType dtype, PrimExpr lhs,
                                          PrimExpr rhs,
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
    PrimType lhs_dtype = lhs.ty();
    PrimType rhs_dtype = rhs.ty();
    bool lhs_fp8 = tl::IsAscendVectorizableFP8(lhs_dtype);
    bool rhs_fp8 = tl::IsAscendVectorizableFP8(rhs_dtype);
    bool result_fp8 = tl::IsAscendVectorizableFP8(dtype);
    if (inside_simtvf_body_ && (lhs_fp8 || rhs_fp8 || result_fp8)) {
      bool supported_op = opstr == "+" || opstr == "-" || opstr == "*";
      bool has_immediate =
          lhs.as<IntImmObj>() != nullptr || lhs.as<FloatImmObj>() != nullptr ||
          rhs.as<IntImmObj>() != nullptr || rhs.as<FloatImmObj>() != nullptr;
      if (supported_op && lhs_fp8 && rhs_fp8 && lhs_dtype == rhs_dtype &&
          result_fp8 && !has_immediate) {
        os << "tl.scalar_binary_fp8('" << opstr << "', " << PrintExpr_(lhs)
           << ", " << PrintExpr_(rhs) << ", " << FP8TypeName(dtype) << ")";
        return;
      }
      TVM_FFI_THROW(InternalError)
          << "Unsupported PTO SIMT scalar FP8 binary op '" << opstr
          << "' with lhs dtype " << lhs_dtype << ", rhs dtype " << rhs_dtype
          << ", and result dtype " << dtype
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
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO SIMT vector binary op: " << opstr;
  }
  if (tl::IsAscendVectorizableFP8(dtype)) {
    os << "tl.vectorize_binary_fp8('" << opstr << "', " << PrintExpr_(lhs)
       << ", " << PrintExpr_(rhs) << ", " << FP8TypeName(dtype.WithLanes(1))
       << ")";
    return;
  }
  os << "(" << PrintExpr_(lhs) << " " << opstr << " " << PrintExpr_(rhs) << ")";
}

void CodeGenTileLangPTO::Dispatch_(const prim::RampObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  // Keep operands expression-local and evaluate each once, even for effectful
  // calls. A Ramp used as a value must remain a typed vector.
  os << "(lambda _ramp_base, _ramp_stride: tl.vector_from_list("
     << DataTypeName(dtype.WithLanes(1)) << ", (";
  for (int lane = 0; lane < dtype.lanes(); ++lane) {
    if (lane != 0) {
      os << ", ";
    }
    os << "(_ramp_base)+(_ramp_stride*" << lane << ")";
  }
  os << ")))("
     << ScalarCastExpr(PrintExpr_(op->base), dtype.WithLanes(1), "Ramp base")
     << ", "
     << ScalarCastExpr(PrintExpr_(op->stride), dtype.WithLanes(1),
                       "Ramp stride")
     << ")";
}

void CodeGenTileLangPTO::Dispatch_(const prim::BroadcastObj *op,
                                   std::ostream &os) { // NOLINT(*)
  PrimType elem_dtype = op->value.ty();
  ICHECK(elem_dtype.MatchesElementType(kDLFloat, 16) || IsFloat32(elem_dtype) ||
         elem_dtype.MatchesElementType(kDLBfloat, 16) ||
         elem_dtype.MatchesCode(kDLInt, kDLUInt, kDLBool))
      << "Unsupported PTO vector broadcast element type: " << elem_dtype;
  std::string value = PrintExpr_(op->value);
  os << "pto.Vec(" << DataTypeName(elem_dtype) << ", "
     << op->ty.as_or_throw<PrimType>().lanes() << ", init=" << value << ")";
}

void CodeGenTileLangPTO::Dispatch_(const BindObj *op) {
  if (const auto *call = op->value.as<CallObj>()) {
    if (call->op.same_as(tirx::alloc_tensor_op()))
      return EmitTensorAllocation_(op, call);
    if (call->op.same_as(tirx::decl_tensor_op()))
      return EmitTensorDeclaration_(op, call);
  }
  if (const auto *call = op->value.as<CallObj>()) {
    if (IsOpName(call->op, "tl.simd.alloc")) {
      AllocVarID(op->var.get());
      ICHECK_GT(op->var->ty.as_or_throw<PrimType>().lanes(), 1)
          << "tl.simd.alloc should bind a vector-typed Var, got "
          << op->var->ty.as_or_throw<PrimType>();
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
      PrimType elem_dtype = call->ty.as_or_throw<PrimType>().WithLanes(1);
      std::string pair_name = AllocVarID(op->var.get());
      std::string value = pair_name + "_value";
      std::string next_ptr = pair_name + "_next_ptr";
      PrintIndent();
      stream << value << " = pto.vlds(" << ptr
             << ", pto.const(0), pto.vreg_type("
             << call->ty.as_or_throw<PrimType>().lanes() << ", "
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

  if (const auto *pointer = op->var->ty.as<PtrTypeObj>()) {
    const auto element = pointer->element_type.as_or_throw<PrimType>();
    RegisterHandleType_(op->var.get(), element);
    alloc_storage_scope_[op->var.get()] = pointer->scope.name();
    if (const auto *source = op->value.as<VarObj>()) {
      if (integer_address_vars_.count(GetRef<Var>(source)))
        integer_address_vars_.insert(op->var);
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
  } else if (op->value->ty.as<PrimTypeObj>() &&
             IsInlineableInvariantSimdBind(op->value.as_or_throw<PrimExpr>())) {
    // Map the Var to the printed RHS so each use rematerializes the broadcast /
    // pset. Do not emit a live SSA bind that would cross range()/scf.for.
    var_idmap_[op->var.get()] = PrintExpr_(op->value);
    return;
  } else {
    vid = AllocVarID(op->var.get());
  }
  PrintSSAAssign(vid, PrintExpr_(op->value), op->var->ty);
}

void CodeGenTileLangPTO::EmitTensorAllocation_(const BindObj *op,
                                               const CallObj *) {
  const TensorVar tensor = op->var.as_or_throw<TensorVar>();
  if (!current_function_is_cube_ || in_mixed_vector_section_) {
    EmitBufferAllocation(tensor.get());
    return;
  }
  const std::string scope = tensor.scope().name();
  alloc_storage_scope_[tensor.get()] = scope;
  ValidateSharedScope(scope);
  if (scope == "shared.dyn" || scope == "shared.l1" ||
      scope == "shared.l1.dyn" || scope == "shared.l0a" ||
      scope == "shared.l0a.dyn" || scope == "shared.l0b" ||
      scope == "shared.l0b.dyn" || scope == "shared.l0c" ||
      scope == "shared.l0c.dyn") {
    PrintIndent();
    stream << AllocVarID(tensor.get()) << " = pto.const(0, dtype=pto.i64)\n";
    integer_address_vars_.insert(tensor);
  } else if (scope == "local.fragment") {
    AllocVarID(tensor.get());
    (void)CodegenAllocationElements(tensor);
  } else if (scope == "local.var") {
    CheckLocalVarBuffer(tensor.get());
    PrintIndent();
    local_var_buffers_.insert(tensor.get());
    EmitLocalVarInitialization_(tensor.get(), AllocVarID(tensor.get()));
    return;
  } else {
    TVM_FFI_THROW(ValueError)
        << "Unsupported PTO Cube allocation scope: " << scope;
  }
  RegisterHandleType_(tensor.get(), tensor.dtype());
}

void CodeGenTileLangPTO::EmitTensorDeclaration_(const BindObj *op,
                                                const CallObj *call) {
  const TensorVar tensor = op->var.as_or_throw<TensorVar>();
  TVM_FFI_CHECK_EQ(call->args.size(), 1, ValueError);
  const std::string value = PrintExpr_(call->args[0]);
  RegisterHandleType_(tensor.get(), tensor.dtype());
  alloc_storage_scope_[tensor.get()] = tensor.scope().name();
  PrintIndent();
  stream << AllocVarID(tensor.get()) << " = " << value << "\n";
  if (const auto *source = call->args[0].as<VarObj>())
    if (integer_address_vars_.count(GetRef<Var>(source)))
      integer_address_vars_.insert(tensor);
}

void CodeGenTileLangPTO::Dispatch_(const RegionStmtObj *op) {
  if (tl::ascend::IsAscendBlock(op))
    return EmitBlockRegion_(op);
  if (op->op.same_as(tirx::launch_thread_op())) {
    TVM_FFI_CHECK(op->args.size() == 2 && op->body_params.size() == 1 &&
                      op->result_vars.empty(),
                  ValueError);
    const auto var = op->body_params[0].as_or_throw<PrimVar>();
    const std::string tag = op->args[0].as_or_throw<StringImm>()->value;
    const bool simt =
        tag == "threadIdx.x" || tag == "threadIdx.y" || tag == "threadIdx.z";
    if (simt && !inside_simtvf_body_) {
      TVM_FFI_CHECK(!UsesVar(op->body, var.get()), ValueError)
          << "PTO codegen cannot use " << tag << " outside a SIMT section";
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
      TVM_FFI_THROW(ValueError) << "Unsupported PTO thread tag: " << tag;
    const auto prior = var_idmap_.find(var.get());
    std::optional<std::string> saved;
    if (prior != var_idmap_.end())
      saved = prior->second;
    if (saved)
      var_idmap_.erase(var.get());
    PrintIndent();
    stream << AllocVarID(var.get()) << " = pto.cast(" << value << ", "
           << ScalarType(var.ty()) << ")\n";
    PrintStmt_(op->body);
    if (saved)
      var_idmap_[var.get()] = *saved;
    else
      var_idmap_.erase(var.get());
    return;
  }
  if (op->op->name == "tl.attr") {
    const auto attribute = DecodeTileLangAttribute(op);
    TVM_FFI_CHECK(op->body_params.empty() && op->result_vars.empty(),
                  ValueError)
        << "PTO attribute regions cannot carry hidden bindings";
    if (attribute.key == "pragma_unroll_factor") {
      const auto var = attribute.node.cast<PrimVar>();
      const int64_t factor =
          attribute.value.cast<IntImm>()->value.as<int64_t>().value();
      TVM_FFI_CHECK(factor > 0 && factor <= std::numeric_limits<int32_t>::max(),
                    ValueError)
          << "PTO pragma_unroll_factor requires a positive signless i32 value";
      const auto prior_var = current_unroll_factor_loop_var_;
      const auto prior_factor = current_unroll_factor_;
      current_unroll_factor_loop_var_ = var;
      current_unroll_factor_ = factor;
      PrintStmt_(op->body);
      current_unroll_factor_loop_var_ = prior_var;
      current_unroll_factor_ = prior_factor;
      return;
    }
    if (attribute.key == "tl.simdvf_scope") {
      if (simdvf_nesting_depth_ == 0) {
        PrintIndent();
        stream << "with pto.vecscope():\n";
        const int scope = BeginScope();
        ++simdvf_nesting_depth_;
        PrintSuite_(op->body);
        --simdvf_nesting_depth_;
        EndScope(scope);
      } else {
        PrintIndent();
        stream << "pto.mem_bar(\"VST_VLD\")\n";
        PrintStmt_(op->body);
      }
      return;
    }
    if (attribute.key == "tl.simtvf_scope" ||
        attribute.key == "tilelang_assume" || attribute.key == "tl.assume") {
      PrintStmt_(op->body);
      return;
    }
  }
  CodeGenTileLangPY::Dispatch_(op);
}

void CodeGenTileLangPTO::Dispatch_(const SeqStmtObj *op) {
  for (size_t i = 0; i < op->seq.size(); ++i) {
    const Stmt &stmt = op->seq[i];
    if (i + 1U < op->seq.size()) {
      const auto *first_if = stmt.as<IfObj>();
      const auto *second_if = op->seq[i + 1U].as<IfObj>();
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

void CodeGenTileLangPTO::Dispatch_(const ForObj *op) {
  int entry_pad_value_id = current_copy_pad_value_id_;
  PrimType entry_pad_value_dtype = current_copy_pad_value_dtype_;

  sym::Analyzer analyzer;
  PrimExpr start = analyzer->Simplify(op->begin);
  PrimExpr extent = analyzer->Simplify((op->end - op->begin));
  PrimExpr step = analyzer->Simplify(op->step);
  PrimExpr stop = analyzer->Simplify(op->end);
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
    auto it = op->annotations.find("pragma_unroll_explicit");
    if (it != op->annotations.end()) {
      explicit_unroll_requested = CodegenBool((*it).second);
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
    auto check_local_index = [&](const VarObj *buffer, const PrimExpr &index) {
      // Keep non-explicit kUnrolled loops on the device so PTOAS can honor
      // pto.range unroll hints. Vector local slots also require a constant
      // TIR index; static_range alone does not substitute that index.
      // Persistent fragments are the exception: PTOAS requires their GEP
      // index to be a compile-time constant, so a loop that indexes a
      // persistent fragment must always expand at trace time (this mirrors
      // the ascend branch, which maps kUnrolled loops to static_range).
      const bool persistent_fragment =
          persistent_buffer_vars_.count(buffer) != 0;
      if ((CodegenTensorDType(buffer).lanes() != 1 && !persistent_fragment) ||
          (inside_simtvf_body_ && !persistent_fragment) ||
          (op->kind == ForKind::kUnrolled && !explicit_unroll_requested &&
           !persistent_fragment)) {
        return;
      }
      std::string scope = CodegenStorageScope(GetRef<Var>(buffer));
      if ((scope == "local" || scope == "local.fragment") &&
          UsesVar(index, op->var.get())) {
        static_local_index_loop = true;
      }
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        op->body,
        [&](const TensorLoadObj *load) -> ffi::Expected<ffi::WalkResult> {
          if (load->indices.size() == 1)
            check_local_index(load->source.as_or_throw<TensorVar>().get(),
                              load->indices[0]);
          return ffi::WalkResult::Advance();
        },
        [&](const TensorStoreObj *store) -> ffi::Expected<ffi::WalkResult> {
          if (store->indices.size() == 1)
            check_local_index(store->dest.as_or_throw<TensorVar>().get(),
                              store->indices[0]);
          return ffi::WalkResult::Advance();
        });
  }
  const bool emit_static_range = (explicit_unroll || static_local_index_loop) &&
                                 !ContainsLoopBreak(op->body);
  const bool pto_unroll_hint = op->kind == ForKind::kUnrolled &&
                               !explicit_unroll_requested &&
                               !ContainsLoopBreak(op->body);
  const bool has_unroll_factor =
      current_unroll_factor_loop_var_.has_value() &&
      current_unroll_factor_loop_var_.value().same_as(op->var);
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
      if (has_unroll_factor) {
        stream << ", unroll_factor=" << current_unroll_factor_;
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
           << ScalarType(op->var->ty.as_or_throw<PrimType>()) << ")\n";
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

void CodeGenTileLangPTO::Dispatch_(const WhileObj *op) {
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
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      op->body, [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
        std::string name_hint = bind->var->name;
        if (name_hint.rfind("__cond_", 0) != 0 ||
            SideEffect(bind->value) == CallEffectKind::kPure ||
            loop_condition_snapshot_ids_.count(bind->var)) {
          return ffi::WalkResult::Advance();
        }
        std::string name =
            name_supply_->FreshName("tl_cond_" + name_hint.substr(7), false);
        loop_condition_snapshot_ids_.emplace(bind->var, name);
        initialized_conditions.push_back(bind->var);
        PrintIndent();
        stream << name << " = pto.const(0, dtype="
               << DataTypeName(bind->var->ty.as_or_throw<PrimType>()) << ")\n";
        return ffi::WalkResult::Advance();
      });
  // PTODSL's native-while rewrite needs a value to be carried when it is
  // assigned on some iterations and read after the loop. Materialize a
  // read/write on every iteration for local.var values already in scope.
  std::set<std::string> carried_locals;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      op->body,
      [&](const TensorStoreObj *store) -> ffi::Expected<ffi::WalkResult> {
        const auto tensor = store->dest.as_or_throw<TensorVar>();
        const VarObj *var = tensor.get();
        if (local_var_buffers_.count(var) && var_idmap_.count(var) &&
            tensor.dtype().IsScalar()) {
          carried_locals.insert(GetVarID(var));
        }
        return ffi::WalkResult::Advance();
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

void CodeGenTileLangPTO::EmitBlockRegion_(const RegionStmtObj *op) {
  TVM_FFI_CHECK(op->args.empty(), ValueError)
      << "Ascend block dynamic annotations must be consumed before codegen";
  const auto annotations =
      op->attrs.GetAttr<ffi::Map<ffi::String, ffi::Any>>("annotations").value();
  const auto name = op->attrs.GetAttr<ffi::String>("name").value();
  if (current_function_is_mixed_ && (name == "CUBE" || name == "VECTOR")) {
    bool is_vector_section = name == "VECTOR";
    int64_t vector_count = 2;
    if (is_vector_section) {
      if (auto opt = annotations.Get("vector_count")) {
        vector_count = opt.value().cast<IntImm>()->value.as<int64_t>().value();
        ICHECK(vector_count == 1 || vector_count == 2)
            << "Mixed-kernel vector_count must be the constant integer 1 or "
               "2, got "
            << opt.value();
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

  if (name == "SIMT_VF") {
    int64_t thread_x = 1;
    int64_t thread_y = 1;
    int64_t thread_z = 1;
    ExtractSimtThreadExtents(op, &thread_x, &thread_y, &thread_z);
    EmitInlineSimtVF(op, thread_x, thread_y, thread_z);
    return;
  }

  if (name == "SIMD_VF") {
    PrintStmt_(op->body);
    return;
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::Dispatch_(const IfObj *op) {
  // Fold compile-time conditions; keep runtime control flow as native Python.
  PrimExpr cond_simp = sym::Analyzer()->Simplify(op->condition);
  if (const auto *imm = cond_simp.as<IntImmObj>()) {
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

void CodeGenTileLangPTO::Dispatch_(const EvaluateObj *op) {
  if (op->value.as<IntImmObj>())
    return;
  if (const auto *call = op->value.as<CallObj>();
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

void CodeGenTileLangPTO::Dispatch_(const TensorLoadObj *op,
                                   std::ostream &os) { // NOLINT(*)
  const auto tensor = op->source.as_or_throw<TensorVar>();
  // S.alloc_var inside T.SimdVF() creates a `local.var`-scope buffer with a
  // vector dtype (e.g. float32x64). In PTODSL it is a plain Python variable
  // holding a vector register value, so load at index 0 is just the variable
  // itself. Skip the scalar-only CheckLocalVarBuffer below.
  if (tensor.dtype().lanes() > 1) {
    std::string scope = ScopeOfBuffer(tensor.get());
    if (scope == "local.var") {
      ICHECK_EQ(op->indices.size(), 1U)
          << "PTO vector local.var load expects a single index";
      int64_t index = 0;
      ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
          << "PTO vector local.var load expects index 0";
      os << LocalVarID(tensor.get());
      return;
    }
  }

  if (IsLocalVarBuffer(tensor.get())) {
    // Vector SSA and scalar PTO surface values both load as Python variables.
    CheckLocalVarBuffer(tensor.get());
    if (tensor.dtype().lanes() > 1) {
      os << LocalVarID(tensor.get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var load expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var load expects index 0";
    if (current_function_has_gemm_) {
      os << LocalVarID(tensor.get());
      return;
    }
    if (IsVectorLocalVarDtype(tensor.dtype()) ||
        IsSurfaceScalarLocalVarDtype(tensor.dtype())) {
      os << LocalVarID(tensor.get());
    } else {
      TVM_FFI_THROW(InternalError)
          << "Unsupported PTO local.var load dtype: " << tensor.dtype();
    }
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer loads";

  PrimType value_dtype = op->ty.as_or_throw<PrimType>();
  PrimType element_dtype = tensor.dtype();
  if (value_dtype.lanes() > 1) {
    EmitScalarizedLoad(op, os);
    return;
  }

  ICHECK_EQ(value_dtype, element_dtype)
      << "PTO scalar BufferLoad expects value dtype to match buffer element "
         "dtype, got "
      << value_dtype << " vs " << element_dtype;

  std::string loaded = ScalarLoad(
      tensor.get(), RemoveOutermostParentheses(PrintExpr_(op->indices[0])));
  if (value_dtype.MatchesCode(kDLBool)) {
    os << "tl.as_logical_bool(" << loaded << ")";
  } else {
    os << loaded;
  }
}

std::vector<std::string>
CodeGenTileLangPTO::MaterializeIndexLanes(const PrimExpr &index,
                                          int element_lanes) {
  ICHECK(index.ty().IsFixedLengthVector());
  std::string value = PrintExpr_(index);
  std::string name = name_supply_->FreshName("tl_indices");
  PrintIndent();
  stream << name << " = tl.vector_to_list(" << value << ")\n";
  std::vector<std::string> result;
  for (int i = 0; i < index.ty().lanes(); ++i) {
    std::string slot = ScalarCastExpr(name + "[" + std::to_string(i) + "]",
                                      index.ty().WithLanes(1), "vector index");
    // First honor the authored index width, then scale in address width.
    if (element_lanes > 1 && index.ty().bits() < 64) {
      slot = ScalarCastExpr(slot, PrimType::Int(64), "vector slot offset");
    }
    for (int j = 0; j < element_lanes; ++j) {
      result.push_back(element_lanes == 1
                           ? slot
                           : "(" + slot + " * " +
                                 std::to_string(element_lanes) + " + " +
                                 std::to_string(j) + ")");
    }
  }
  return result;
}

void CodeGenTileLangPTO::EmitScalarizedLoad(const TensorLoadObj *op,
                                            std::ostream &os) {
  PrimType value_dtype = op->ty.as_or_throw<PrimType>();
  const auto tensor = op->source.as_or_throw<TensorVar>();
  PrimType element_dtype = tensor.dtype();
  std::string scope = ScopeOfBuffer(tensor.get());
  PrimExpr index = op->indices[0];

  const auto *ramp = index.as<prim::RampObj>();
  if (index.ty().IsFixedLengthVector() &&
      !(ramp && IsOne(sym::Analyzer()->Simplify(ramp->stride)))) {
    ICHECK(element_dtype.IsScalar() ||
           (inside_simtvf_body_ &&
            (scope == "local.fragment" || scope == "local")))
        << "PTO vector-element gather requires SIMT local storage";
    ICHECK(!tl::IsAscendVectorizableFP8(element_dtype))
        << "PTO FP8 gather requires supported scalar lane conversion";
    // Keep the address computation expression-local: this load may occur in
    // a Let body or a lazy branch. The lambda evaluates the vector index once.
    std::string value = PrintExpr_(index);
    std::string name = name_supply_->FreshName("tl_indices");
    ICHECK_EQ(index.ty().lanes() * element_dtype.lanes(), value_dtype.lanes());
    os << "(lambda " << name << ": tl.vector_from_list("
       << DataTypeName(value_dtype.WithLanes(1)) << ", (";
    for (int i = 0; i < index.ty().lanes(); ++i) {
      std::string slot =
          ScalarCastExpr(name + "[" + std::to_string(i) + "]",
                         index.ty().WithLanes(1), "vector index");
      if (!element_dtype.IsScalar() && index.ty().bits() < 64) {
        slot = ScalarCastExpr(slot, PrimType::Int(64), "vector slot offset");
      }
      for (int j = 0; j < element_dtype.lanes(); ++j) {
        std::string offset = element_dtype.IsScalar()
                                 ? slot
                                 : "(" + slot + " * " +
                                       std::to_string(element_dtype.lanes()) +
                                       " + " + std::to_string(j) + ")";
        os << ScalarLoad(tensor.get(), offset) << ", ";
      }
    }
    os << ")))(tl.vector_to_list(" << value << "))";
    return;
  }

  if (inside_simtvf_body_ && !element_dtype.IsScalar() &&
      (scope == "local.fragment" || scope == "local")) {
    // Convert vector slots to scalar offsets for the existing contiguous load.
    index = SIMTVectorLocalScalarOffset(index, element_dtype.lanes(), "load");
    element_dtype = element_dtype.WithLanes(1);
  }

  // Vector-element buffer (e.g. T.simd.alloc_local produces float32x64
  // elements): reading one slot yields a whole vector register. Emit a plain
  // Python index expression.
  if (element_dtype.lanes() > 1) {
    if (scope == "local.fragment" || scope == "local") {
      os << GetVectorLocalRef(tensor.get(), index, "PTO vector load");
    } else {
      std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));
      os << GetVarID(tensor.get()) << "[" << index_str << "]";
    }
    return;
  }

  ICHECK_EQ(element_dtype.lanes(), 1)
      << "PTO vector BufferLoad scalarization currently expects scalar "
         "buffer elements, got "
      << element_dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsSupportedSIMTLocalStorageType(element_dtype))
        << "PTO SIMT vector BufferLoad supports float16, bfloat16, float32, "
           "int32, uint32, and float8_e4m3fn/float8_e5m2 storage, got "
        << element_dtype;
    ICHECK_EQ(value_dtype.WithLanes(1), element_dtype)
        << "PTO SIMT vector BufferLoad expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << element_dtype;
    ICHECK(!tl::IsAscendVectorizableFP8(element_dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 BufferLoad currently supports contiguous lanes 2, 4, "
           "and 8 only, got "
        << value_dtype;
    const int lanes = value_dtype.lanes();
    PrimExpr scalar_index = index;
    if (const auto *ramp = index.as<prim::RampObj>()) {
      CheckContiguousRampStride(index, "load");
      scalar_index = ramp->base;
    }
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(scalar_index));
    if (scope == "local.fragment" || scope == "local") {
      std::string value = "pto.load(" + GetVarID(tensor.get()) + ", " +
                          index_str + ", contiguous=" + std::to_string(lanes) +
                          ")";
      // Local integer storage is signless; restore the authored signedness.
      os << (IsInteger32(element_dtype)
                 ? ScalarCastExpr(value, element_dtype,
                                  "PTO SIMT local vector load")
                 : value);
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base = ScalarPointerBase_(tensor.get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base = ScalarPointerBase_(tensor.get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO SIMT vector load scope: " << scope;
  }

  // Non-SIMT vector load from a scalar-element buffer: use pto.load with
  // contiguous=N to read all lanes in one shot.
  if (value_dtype.lanes() > 1) {
    const int lanes = value_dtype.lanes();
    PrimExpr scalar_index = index;
    if (const auto *ramp = index.as<prim::RampObj>()) {
      CheckContiguousRampStride(index, "load");
      scalar_index = ramp->base;
    }
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(scalar_index));
    if (scope == "local.fragment" || scope == "local") {
      // Non-SIMT local buffers are Python lists; rebuild a typed vector from
      // their scalar lanes instead of calling pto.load on the list.
      const std::string list_ref = GetVarID(tensor.get());
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
      std::string base = ScalarPointerBase_(tensor.get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base = ScalarPointerBase_(tensor.get(), element_dtype, scope);
      os << "pto.load(" << base << ", " << index_str << ", contiguous=" << lanes
         << ")";
      return;
    }
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO vector load scope: " << scope;
  }

  TVM_FFI_THROW(InternalError)
      << "PTO non-SIMT vector BufferLoad is not supported yet";
}

void CodeGenTileLangPTO::Dispatch_(const TensorStoreObj *op) {
  const auto tensor = op->dest.as_or_throw<TensorVar>();
  // S.alloc_var inside T.SimdVF() creates a `local.var`-scope buffer with a
  // vector dtype (e.g. float32x64). In PTODSL it is a plain Python variable
  // holding a vector register value, so store at index 0 is just
  // `var = value`. Skip the scalar-only CheckLocalVarBuffer below.
  if (tensor.dtype().lanes() > 1) {
    std::string scope = ScopeOfBuffer(tensor.get());
    if (scope == "local.var") {
      ICHECK_EQ(op->indices.size(), 1U)
          << "PTO vector local.var store expects a single index";
      int64_t index = 0;
      ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
          << "PTO vector local.var store expects index 0";
      std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
      PrintIndent();
      stream << LocalVarID(tensor.get()) << " = " << value << "\n";
      EmitMixedEntrySnapshot(tensor.get());
      return;
    }
  }

  if (IsLocalVarBuffer(tensor.get())) {
    // Vector SSA and scalar PTO surface values both store as Python variables.
    CheckLocalVarBuffer(tensor.get());
    if (tensor.dtype().lanes() > 1) {
      std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
      PrintIndent();
      stream << LocalVarID(tensor.get()) << " = " << value << "\n";
      EmitMixedEntrySnapshot(tensor.get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var store expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var store expects index 0";
    std::string vid = LocalVarID(tensor.get());
    PrimType dtype = tensor.dtype();
    std::string value;
    if (current_function_has_gemm_) {
      value = LocalVarStoreValue(op->value, dtype);
    } else if (IsVectorLocalVarDtype(dtype) ||
               dtype.MatchesCode(kDLOpaqueHandle)) {
      value = RemoveOutermostParentheses(PrintExpr_(op->value));
    } else if (IsSurfaceScalarLocalVarDtype(dtype)) {
      value = LocalVarStoreValue(op->value, dtype);
    } else {
      TVM_FFI_THROW(InternalError)
          << "Unsupported PTO local.var store dtype: " << dtype;
    }
    PrintIndent();
    stream << vid << " = " << value << "\n";
    EmitMixedEntrySnapshot(tensor.get());
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer stores";

  if (op->value.ty().lanes() > 1) {
    EmitScalarizedStore(op);
    return;
  }

  std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
  PrintIndent();
  EmitScalarStore(tensor.get(), value,
                  RemoveOutermostParentheses(PrintExpr_(op->indices[0])));
}

void CodeGenTileLangPTO::EmitScalarizedStore(const TensorStoreObj *op) {
  const auto tensor = op->dest.as_or_throw<TensorVar>();
  PrimType buffer_dtype = tensor.dtype();
  PrimType value_dtype = op->value.ty();
  std::string scope = ScopeOfBuffer(tensor.get());
  PrimExpr index = op->indices[0];

  const auto *ramp = index.as<prim::RampObj>();
  if (index.ty().IsFixedLengthVector() &&
      !(ramp && IsOne(sym::Analyzer()->Simplify(ramp->stride)))) {
    ICHECK(buffer_dtype.IsScalar() ||
           (inside_simtvf_body_ &&
            (scope == "local.fragment" || scope == "local")))
        << "PTO vector-element scatter requires SIMT local storage";
    ICHECK(!tl::IsAscendVectorizableFP8(buffer_dtype))
        << "PTO FP8 scatter requires supported scalar lane conversion";
    auto indices = MaterializeIndexLanes(index, buffer_dtype.lanes());
    ICHECK_EQ(indices.size(), value_dtype.lanes());
    std::string value = PrintExpr_(op->value);
    std::string name = name_supply_->FreshName("tl_values");
    PrintIndent();
    stream << name << " = tl.vector_to_list(" << value << ")\n";
    for (int i = 0; i < value_dtype.lanes(); ++i) {
      PrintIndent();
      EmitScalarStore(tensor.get(), name + "[" + std::to_string(i) + "]",
                      indices[i]);
    }
    return;
  }

  if (inside_simtvf_body_ && !buffer_dtype.IsScalar() &&
      (scope == "local.fragment" || scope == "local")) {
    // Match the scalar-addressed allocation and contiguous load path.
    index = SIMTVectorLocalScalarOffset(index, buffer_dtype.lanes(), "store");
    buffer_dtype = buffer_dtype.WithLanes(1);
  }

  // Vector-element buffer (e.g. T.simd.alloc_local produces float32x64
  // elements): the value is a whole vector register stored into one buffer
  // slot outside SIMT. Emit an assignment to its Python vector value.
  if (buffer_dtype.lanes() > 1) {
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    PrintIndent();
    if (scope == "local.fragment" || scope == "local") {
      stream << GetVectorLocalRef(tensor.get(), index, "PTO vector store")
             << " = " << value << "\n";
    } else {
      std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));
      stream << GetVarID(tensor.get()) << "[" << index_str << "] = " << value
             << "\n";
    }
    return;
  }

  ICHECK_EQ(buffer_dtype.lanes(), 1)
      << "PTO vector BufferStore scalarization currently expects scalar "
         "buffer elements, got "
      << buffer_dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsSupportedSIMTLocalStorageType(buffer_dtype))
        << "PTO SIMT vector BufferStore supports float16, bfloat16, float32, "
           "int32, uint32, and float8_e4m3fn/float8_e5m2 storage only, got "
        << buffer_dtype;
    ICHECK_EQ(value_dtype.WithLanes(1), buffer_dtype)
        << "PTO SIMT vector BufferStore expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << buffer_dtype;
    if (TryEmitRngBroadcastStore(op, index)) {
      return;
    }
    ICHECK(!tl::IsAscendVectorizableFP8(buffer_dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 BufferStore currently supports contiguous lanes 2, "
           "4, and 8 only, got "
        << value_dtype;
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    PrimExpr scalar_index = index;
    if (const auto *ramp = index.as<prim::RampObj>()) {
      CheckContiguousRampStride(index, "store");
      scalar_index = ramp->base;
    }
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(scalar_index));
    PrintIndent();
    const int store_lanes = value_dtype.lanes();
    const std::string contiguous_suffix =
        store_lanes > 1 ? ", contiguous=" + std::to_string(store_lanes) : "";
    if (scope == "local.fragment" || scope == "local") {
      stream << "pto.store(" << value << ", " << GetVarID(tensor.get()) << ", "
             << index_str << contiguous_suffix << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base =
          ScalarPointerBase_(tensor.get(), tensor.dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base =
          ScalarPointerBase_(tensor.get(), tensor.dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO SIMT vector store scope: " << scope;
  }

  // Non-SIMT vector store to a scalar-element buffer: use pto.store with
  // contiguous=N to write all lanes in one shot.
  if (value_dtype.lanes() > 1) {
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    PrimExpr scalar_index = index;
    if (const auto *ramp = index.as<prim::RampObj>()) {
      CheckContiguousRampStride(index, "store");
      scalar_index = ramp->base;
    }
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(scalar_index));
    if (scope == "local.fragment" || scope == "local") {
      // Non-SIMT local buffers are Python lists.  PTODSL pto.store does
      // not accept a list as a vector destination, so extract each lane and
      // assign it directly.
      const std::string list_ref = GetVarID(tensor.get());
      PrintIndent();
      stream << "tl.store_vector_to_list(" << list_ref << ", " << index_str
             << ", " << value << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      PrintIndent();
      std::string base =
          ScalarPointerBase_(tensor.get(), tensor.dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << ", contiguous=" << value_dtype.lanes() << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      PrintIndent();
      std::string base =
          ScalarPointerBase_(tensor.get(), tensor.dtype(), scope);
      stream << "pto.store(" << value << ", " << base << ", " << index_str
             << ", contiguous=" << value_dtype.lanes() << ")\n";
      return;
    }
    TVM_FFI_THROW(InternalError)
        << "Unsupported PTO vector store scope: " << scope;
  }

  TVM_FFI_THROW(InternalError)
      << "PTO non-SIMT vector BufferStore is not supported yet";
}

bool CodeGenTileLangPTO::TryEmitRngBroadcastStore(const TensorStoreObj *op,
                                                  const PrimExpr &index) {
  const auto *broadcast = op->value.as<prim::BroadcastObj>();
  if (broadcast == nullptr) {
    return false;
  }
  bool contains_rng_draw = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      broadcast->value,
      [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
        if ((call->op.same_as(tl::rng_rand()) ||
             call->op.same_as(tl::rng_rand_float()))) {
          contains_rng_draw = true;
        }
        return ffi::WalkResult::Advance();
      });
  if (!contains_rng_draw) {
    return false;
  }
  // Each lane must evaluate the complete scalar expression independently so
  // that every RNG call advances the stream. Reusing one evaluated value for
  // both stores would incorrectly duplicate the same random output.
  ICHECK_EQ(broadcast->ty.as_or_throw<PrimType>().lanes(), 2)
      << "PTO RNG broadcast store currently supports exactly 2 lanes, got "
      << broadcast->ty.as_or_throw<PrimType>();

  PrimExpr base_index = index;
  if (const auto *ramp = base_index.as<prim::RampObj>()) {
    CheckContiguousRampStride(index, "store");
    base_index = ramp->base;
  }

  const auto tensor = op->dest.as_or_throw<TensorVar>();
  sym::Analyzer analyzer;
  for (int64_t lane = 0; lane < 2; ++lane) {
    std::string tmp = name_supply_->FreshName("_tl_rng_value");
    std::string value =
        RemoveOutermostParentheses(PrintExpr_(broadcast->value));
    PrintIndent();
    stream << tmp << " = " << value << "\n";
    PrimExpr lane_index =
        analyzer->Simplify(base_index + MakeConst(base_index.ty(), lane));
    PrintIndent();
    EmitScalarStore(tensor.get(), tmp,
                    RemoveOutermostParentheses(PrintExpr_(lane_index)));
  }
  return true;
}

void CodeGenTileLangPTO::EmitRngInit(const CallObj *op) {
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

std::string CodeGenTileLangPTO::EmitRngRand(const CallObj *op) {
  ICHECK(pto_rng_initialized_)
      << "tl.rng_rand called without prior tl.rng_init";
  const PrimType dtype = op->ty.as_or_throw<PrimType>();
  ICHECK(dtype.MatchesCode(kDLUInt) && dtype.bits() == 32 && dtype.IsScalar())
      << "tl.rng_rand expects a scalar uint32 result, got " << dtype;
  std::string result_var =
      "_tl_rng_result_" + std::to_string(pto_rng_result_counter_++);
  PrintIndent();
  stream << result_var << ", " << pto_rng_counter_var_ << " = "
         << pto_rng_state_var_ << ".rand(" << pto_rng_counter_var_ << ")\n";
  return ScalarCastExpr(result_var, dtype, "PTO rng_rand result");
}

std::string CodeGenTileLangPTO::EmitRngRandFloat(const CallObj *op) {
  ICHECK(pto_rng_initialized_)
      << "tl.rng_rand_float called without prior tl.rng_init";
  ICHECK_NE(op->ty.as_or_throw<PrimType>().bits(), 64)
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
