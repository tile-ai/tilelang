/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"

#include "ascend/op/builtin.h"
#include "backend/common/codegen/codegen_utils.h"
#include "backend/common/target_utils.h"
#include "support/check.h"
#include "tvm/ir/repr.h"

#include <algorithm>
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>

namespace tvm {
namespace codegen {

using namespace tirx;

namespace {

void ValidateSharedScope(const std::string &scope) {
  if (scope == "shared") {
    LOG(FATAL) << "Static shared memory (scope='shared') is not supported on "
                  "Ascend. "
               << "Use dynamic shared memory (scope='shared.dyn') with "
                  "T.alloc_shared.";
  }
}

bool RequiresStatementControlFlow(const PrimExpr &expr) {
  return SideEffect(expr) > CallEffectKind::kPure;
}

std::string FP8TypeName(DataType t) {
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

  if (t.is_float8_e4m3() || t.is_float8_e4m3fn())
    return "pto.f8e4m3" + lanes_suffix;
  if (t.is_float8_e5m2())
    return "pto.f8e5m2" + lanes_suffix;
  LOG(FATAL) << "Unsupported PTO Ascend FP8 type: " << t;
  return "";
}

std::string DataTypeName(DataType t) {
  if (tl::IsAscendVectorizableFP8(t))
    return FP8TypeName(t);

  ICHECK(t.is_scalar()) << "PTO scalar type expected, got " << t;
  if (t.is_bool()) {
    return "pto.i1";
  }
  if (t.is_float()) {
    if (t.bits() == 32)
      return "pto.f32";
    if (t.bits() == 16)
      return "pto.f16";
  } else if (t.is_float4_e2m1fn()) {
    // TileLang float4_e2m1fn is packed 2xFP4 per byte → PTODSL f4e2m1x2.
    return "pto.f4e2m1x2";
  } else if (t.is_bfloat16()) {
    return "pto.bf16";
  } else if (t.is_int() || t.is_uint()) {
    const char *prefix = t.is_uint() ? "pto.ui" : "pto.i";
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

std::string SignedIntegerTypeName(DataType t) {
  ICHECK(t.is_scalar() && t.is_int())
      << "PTO signed integer scalar type expected, got " << t;
  switch (t.bits()) {
  case 8:
    return "pto.si8";
  case 16:
    return "pto.si16";
  case 32:
    return "pto.si32";
  case 64:
    return "pto.si64";
  default:
    LOG(FATAL) << "Unsupported PTO signed integer type: " << t;
  }
  return "";
}

std::string AtomicTypeSuffix(DataType t) {
  ICHECK(t.is_scalar()) << "PTO store atomic type must be scalar, got " << t;
  if (t.is_float() && t.bits() == 32) {
    return "f32";
  }
  if (t.is_float() && t.bits() == 16) {
    return "f16";
  }
  if (t.is_bfloat16()) {
    return "bf16";
  }
  if (t.is_int() && t.bits() == 32) {
    return "s32";
  }
  if (t.is_int() && t.bits() == 16) {
    return "s16";
  }
  if (t.is_int() && t.bits() == 8) {
    return "s8";
  }
  LOG(FATAL) << "Unsupported PTO store atomic type: " << t
             << "; expected float32, float16, bfloat16, int32, int16, or int8";
  return "";
}

// Wrap Python/TIR immediates as typed PTODSL consts so vdup/vmaxs keep the
// authored element dtype (e.g. bf16 vs default f16 inference).
std::string WrapTypedConst(const std::string &printed, DataType dtype,
                              bool is_immediate) {
  if (!is_immediate) {
    return printed;
  }
  return "pto.const(" + printed +
         ", dtype=" + DataTypeName(dtype.element_of()) + ")";
}

std::string ScalarCastExpr(const std::string &expr, DataType dtype,
                              const std::string &context) {
  ICHECK(dtype.is_scalar()) << "PTO scalar cast expects scalar dtype, got "
                            << dtype;
  return "tl.scalar_cast(" + expr + ", " + DataTypeName(dtype) +
         ", context=\"" + context + "\")";
}

// Peel scalar casts so `Cast(bf16, 0.0)` / `Cast(u16, 32767)` still count as
// immediates for typed-const wrapping.
PrimExpr PeelScalarCasts(const PrimExpr &value) {
  PrimExpr cur = value;
  while (const auto *cast = cur.as<CastNode>()) {
    if (!cast->dtype.is_scalar() || !cast->value.dtype().is_scalar()) {
      break;
    }
    cur = cast->value;
  }
  return cur;
}

bool IsImmediateScalar(const PrimExpr &value) {
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

DataType ParseDataType(const std::string &dtype_name) {
  if (dtype_name == "float" || dtype_name == "float32")
    return DataType::Float(32);
  if (dtype_name == "half" || dtype_name == "float16")
    return DataType::Float(16);
  if (dtype_name == "bfloat16" || dtype_name == "bfloat16_t")
    return DataType::BFloat(16);
  if (dtype_name == "float8_e4m3" || dtype_name == "float8_e4m3fn" ||
      dtype_name == "float8_e4m3_t")
    return DataType::Float8E4M3FN();
  if (dtype_name == "float8_e5m2" || dtype_name == "float8_e5m2_t")
    return DataType::Float8E5M2();
  if (dtype_name == "float4_e2m1fn" || dtype_name == "float4_e2m1_t" ||
      dtype_name == "float4_e2m1fnx2" || dtype_name == "float4_e2m1x2_t")
    return DataType::Float4E2M1FN(2);
  if (dtype_name == "int64")
    return DataType::Int(64);
  if (dtype_name == "int32")
    return DataType::Int(32);
  if (dtype_name == "int16")
    return DataType::Int(16);
  if (dtype_name == "uint64")
    return DataType::UInt(64);
  if (dtype_name == "uint32")
    return DataType::UInt(32);
  if (dtype_name == "uint16")
    return DataType::UInt(16);
  if (dtype_name == "int8")
    return DataType::Int(8);
  if (dtype_name == "uint16" || dtype_name == "uint16_t")
    return DataType::UInt(16);
  if (dtype_name == "uint8")
    return DataType::UInt(8);
  if (dtype_name == "uint32")
    return DataType::UInt(32);
  if (dtype_name == "uint64")
    return DataType::UInt(64);
  LOG(FATAL) << "Unsupported PTO dtype string: " << dtype_name;
  return DataType::Void();
}

bool TryGetConstInt(const PrimExpr &expr, int64_t *value) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    *value = imm->value;
    return true;
  }
  return false;
}

std::string StoreL2CacheName(const PrimExpr &control) {
  static const char *const kNames[] = {
      "nmfv",  "nmlv",  "nmprs", "nmred", "naci",   "napw",
      "napi",  "nared", "wbhfv", "wbhlv", "wbhprs", "wbhred",
      "wtsfv", "wtslv", "wtsprs", "wtsred",
  };
  int64_t value = 0;
  ICHECK(TryGetConstInt(control, &value))
      << "PTO UB->GM MTE l2_cache_ctl must be a constant in [0, 15], got "
      << control;
  ICHECK_GE(value, 0)
      << "PTO UB->GM MTE l2_cache_ctl must be in [0, 15], got " << value;
  ICHECK_LT(value, 16)
      << "PTO UB->GM MTE l2_cache_ctl must be in [0, 15], got " << value;
  return kNames[value];
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.rfind(prefix, 0) == 0;
}

int VstsMaskGranularityOverride(const std::string &dist,
                                   DataType value_elem_dtype) {
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

bool IsFloat32(DataType t) {
  return t.is_float() && t.bits() == 32 && t.lanes() == 1;
}

bool IsSimtLocalBufferDtype(DataType t) {
  DataType elem = t.is_vector() ? t.element_of() : t;
  if (elem.lanes() != 1) return false;
  return IsFloat32(elem) || elem.is_float16() || elem.is_bfloat16() ||
         tl::IsAscendVectorizableFP8(elem) ||
         elem == DataType::UInt(32) ||
         ((elem.is_int() || elem.is_uint()) && elem.bits() == 8);
}

bool IsFloat32Pair(DataType t) {
  return t.is_float() && t.bits() == 32 && t.lanes() == 2;
}

bool IsInteger32(DataType t) {
  return t.is_scalar() && t.bits() == 32 && (t.is_int() || t.is_uint());
}

bool IsPackedFp4Pair(DataType dtype) {
  return dtype.is_float4_e2m1fn() && dtype.lanes() == 2;
}

bool IsStorageDtypeCompatible(DataType lhs, DataType rhs) {
  // Packed FP4 storage is represented as float4_e2m1fnx2: two logical FP4
  // elements in one physical byte.  It is intentionally non-scalar in TVM,
  // but a same-type GM<->UB MTE is a raw storage copy, not a dtype conversion.
  // Keep the exception exact so packed FP4 is never treated as compatible
  // with an unrelated 8-bit element type.
  if (IsPackedFp4Pair(lhs) || IsPackedFp4Pair(rhs)) {
    return lhs == rhs;
  }
  if (!lhs.is_scalar() || !rhs.is_scalar() || lhs.bits() != rhs.bits()) {
    return false;
  }
  if ((lhs.is_int() || lhs.is_uint()) && (rhs.is_int() || rhs.is_uint())) {
    return true;
  }
  return lhs == rhs;
}

bool IsSupportedFloatMinMaxType(DataType t) {
  if (t.is_scalar())
    return t.is_float16() || IsFloat32(t) || t.is_bfloat16();
  return t.lanes() == 2 &&
         (t.is_float16() || IsFloat32Pair(t) || t.is_bfloat16());
}

bool IsSupportedSIMTUnaryMathTypeImpl(DataType t) {
  const bool scalar_or_pair = t.is_scalar() || t.lanes() == 2;
  return scalar_or_pair && (t.is_float16() || (t.is_float() && t.bits() == 32));
}

bool IsSupportedSIMTUnaryMathType(DataType t) {
  return IsSupportedSIMTUnaryMathTypeImpl(t);
}

bool IsSupportedScalarUnaryMathType(DataType t) {
  return t.is_scalar() && (t.is_float16() || IsFloat32(t) || t.is_bfloat16());
}

bool IsSupportedSIMTAllReduceType(DataType t) {
  return t.is_scalar() && (t.is_float16() || IsFloat32(t) ||
                           t == DataType::Int(32) || t == DataType::UInt(32));
}

bool IsSupportedSIMTLocalScalarAccessType(DataType t) {
  return t.is_scalar() &&
         (IsSimtLocalBufferDtype(t) || t == DataType::Int(32));
}

bool IsSupportedSIMTLocalStorageType(DataType t) {
  return IsSimtLocalBufferDtype(t) ||
         (t.is_scalar() && t == DataType::Int(32));
}

std::string SIMTLocalStorageTypeName(DataType t) {
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
  if (const auto *ramp = index.as<RampNode>()) {
    PrimExpr stride_expr = arith::Analyzer().Simplify(ramp->stride);
    int64_t stride = 0;
    ICHECK(TryGetConstInt(stride_expr, &stride) && stride == 1)
        << "PTO SIMT vector " << access_kind
        << " emits a contiguous vector access, so Ramp stride must be 1, got "
        << ramp->stride;
  }
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

int64_t ConstShapeDim(const PrimExpr &expr, const char *name) {
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
bool IsVectorLocalVarDtype(DataType dtype) { return dtype.lanes() >= 32; }

// Keep scalar local.var as authored PTO surface values. PTODSL should model
// region rebinding with proper if/loop results.
bool IsSurfaceScalarLocalVarDtype(DataType dtype) {
  return dtype.lanes() == 1;
}

int64_t PhysicalElementBits(DataType dtype) {
  if (dtype.is_bool()) {
    return 8;
  }
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

void ValidateGmBypassDtype(DataType dtype) {
  ICHECK(dtype.is_scalar())
      << "PTO GM dcache bypass expects a scalar dtype, got " << dtype;
  const int64_t bits = PhysicalElementBits(dtype);
  ICHECK(bits == 8 || bits == 16 || bits == 32 || bits == 64)
      << "PTO GM dcache bypass only supports 1/2/4/8-byte scalar types, got "
      << dtype;
}

int64_t PhysicalElementBytes(DataType dtype) {
  int64_t bits = PhysicalElementBits(dtype);
  ICHECK_EQ(bits % 8, 0)
      << "PTO physical element width must be byte-aligned, got " << dtype
      << " (" << bits << " bits)";
  return bits / 8;
}

PrimExpr ElementIndexToByteOffsetExpr(const PrimExpr &elem_index,
                                         DataType dtype, const char *context,
                                         arith::Analyzer *analyzer) {
  int64_t elem_bits = PhysicalElementBits(dtype);
  if (elem_bits < 8) {
    ICHECK_EQ(8 % elem_bits, 0)
        << context << " packed dtype bit width must divide one byte, got "
        << dtype;
    int64_t pack_factor = 8 / elem_bits;
    PrimExpr pack = make_const(elem_index.dtype(), pack_factor);
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
  return analyzer->Simplify(
      elem_index * make_const(elem_index.dtype(), elem_bytes));
}

int64_t ElementsToBytesCeil(int64_t elements, DataType dtype) {
  int64_t elem_bits = PhysicalElementBits(dtype);
  return (elements * elem_bits + 7) / 8;
}

std::string PointerElementTypeName(DataType dtype) {
  // Pointer payloads are scalar element types even when the TIR buffer
  // carries a vector dtype (e.g. float32x64 in mhc_norm_fn). Preserve the
  // packed FP4 pair mapping before peeling vector lanes.
  if (IsPackedFp4Pair(dtype)) {
    return "pto.f4e2m1x2";
  }
  DataType elem_dtype = dtype.is_vector() ? dtype.element_of() : dtype;
  if (elem_dtype.is_bool()) {
    return "pto.i8";
  }
  return DataTypeName(elem_dtype);
}

void CheckLocalVarBuffer(const BufferNode *buffer) {
  ICHECK_EQ(buffer->shape.size(), 1U)
      << "PTO local.var only supports scalar alloc_var buffers, got rank "
      << buffer->shape.size();
  int64_t extent = 0;
  ICHECK(TryGetConstInt(buffer->shape[0], &extent) && extent == 1)
      << "PTO local.var only supports scalar alloc_var buffers with shape "
         "(1,), got "
      << buffer->shape[0];
  DataType dtype = buffer->dtype;
  if (dtype.is_handle()) {
    // local.var handle buffers carry mutable typed pointers (e.g. UB views).
    // They are initialized to None and receive a pointer on their first store.
    return;
  }
  if (IsVectorLocalVarDtype(dtype) || dtype.is_handle()) {
    (void)DataTypeName(dtype.element_of());
    return;
  }
  ICHECK_EQ(dtype.lanes(), 1)
      << "PTO local.var only supports scalar or full-width vector values, got "
      << dtype;
  DataType elem_dtype = dtype.element_of();
  int bits = elem_dtype.bits();
  bool supported =
      elem_dtype.is_bool() || elem_dtype.is_bfloat16() ||
      elem_dtype.is_float8_e4m3fn() || elem_dtype.is_float8_e5m2() ||
      (elem_dtype.is_float() && (bits == 16 || bits == 32)) ||
      ((elem_dtype.is_int() || elem_dtype.is_uint()) &&
       (bits == 1 || bits == 8 || bits == 16 || bits == 32 || bits == 64));
  ICHECK(supported)
      << "PTO local.var only supports PTO numeric scalar/vector element "
         "types, got "
      << dtype;
}

void CheckAllReduceDtype(const CallNode *call) {
  ICHECK_GE(call->args.size(), 2U)
      << "tl::AscendAllReduce call expects a value argument";
  DataType dtype = call->args[1].dtype();
  ICHECK(IsSupportedSIMTAllReduceType(dtype))
      << "PTO cross-thread allreduce currently supports float16, float32, "
         "int32, and uint32 only, got "
      << dtype
      << ". Use a supported reducer dtype, such as float32, and cast the "
         "finalized result afterward.";
}

void ValidateKernelCapabilities(const PrimFunc &func) {
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
  (void)0;  // has_unsupported_cube_op removed: nd2nz ops are emitted below

  PostOrderVisit(func->body, [&](const ffi::ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      if (block->name_hint == "CUBE") {
        has_cube_block = true;
      } else if (block->name_hint == "VECTOR") {
        has_vector_block = true;
      }
    }

    if (const auto *call = node.as<CallNode>()) {
      if ((call->op.same_as(builtin::call_extern()) ||
           call->op.same_as(builtin::call_pure_extern())) &&
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
  bool has_cube_path =
      has_cube_block || has_gemm_l1 || has_blockscaled_gemm_l1 ||
      has_supported_cube_mte || has_l0c_to_ub_mte || has_mad;
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

std::string MadUnitFlagArg(int64_t unit_flag_ctrl) {
  if (unit_flag_ctrl == 0)
    return "None";
  if (unit_flag_ctrl == 2)
    return "pto.MadUnitFlagMode.CHECK_ONLY";
  if (unit_flag_ctrl == 3)
    return "pto.MadUnitFlagMode.CHECK_AND_SET";
  LOG(FATAL) << "PTO MAD unsupported unit_flag_ctrl=" << unit_flag_ctrl;
  return "None";
}

bool IsSupportedGemmInputDtype(DataType dtype) {
  return dtype.is_bfloat16() ||
         (dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
         dtype.is_float8_e4m3fn() || dtype.is_float8_e5m2() ||
         dtype.is_float4_e2m1fn();
}

bool IsSupportedBlockscaledGemmInputDtype(DataType dtype) {
  return dtype.is_bfloat16() || dtype.is_float8_e4m3fn() ||
         dtype.is_float4_e2m1fn();
}

int64_t GemmInputPackFactor(DataType dtype) {
  return dtype.is_float4_e2m1fn() ? 2 : 1;
}

int64_t GemmInputC0(DataType dtype) {
  ICHECK(IsSupportedGemmInputDtype(dtype))
      << "PTO GEMM L1 helper only supports float16, bfloat16, float32, "
         "float8_e4m3fn, float8_e5m2, or float4_e2m1fn inputs, got "
      << dtype;
  if (dtype.is_float4_e2m1fn()) {
    return 64;
  }
  int64_t elem_bytes = dtype.bytes();
  ICHECK_GT(elem_bytes, 0) << "Invalid PTO GEMM input dtype size: " << dtype;
  ICHECK_EQ(32 % elem_bytes, 0)
      << "PTO GEMM input dtype byte size must divide 32, got " << dtype;
  return 32 / elem_bytes;
}

bool IsSupportedCubeMteDtype(DataType dtype) {
  return dtype.is_bfloat16() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2() || dtype.is_float4_e2m1fn() ||
         (dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32) &&
          dtype.lanes() == 1) ||
         (dtype.is_int() && dtype.bits() == 8 && dtype.lanes() == 1);
}

bool IsSupportedFp8Dtype(DataType dtype) {
  return dtype.is_scalar() &&
         (dtype.is_float8_e4m3fn() || dtype.is_float8_e5m2());
}

bool IsSupportedMadDtypeTuple(DataType lhs, DataType rhs, DataType acc) {
  if (!lhs.is_scalar() || !rhs.is_scalar() || !acc.is_scalar()) {
    return false;
  }

  bool is_same_standard_float =
      lhs == rhs &&
      (lhs.is_bfloat16() ||
       (lhs.is_float() && (lhs.bits() == 16 || lhs.bits() == 32)));
  bool is_fp8_pair =
      IsSupportedFp8Dtype(lhs) && IsSupportedFp8Dtype(rhs);
  if ((is_same_standard_float || is_fp8_pair) && IsFloat32(acc)) {
    return true;
  }

  return lhs == DataType::Int(8) && rhs == DataType::Int(8) &&
         acc == DataType::Int(32);
}

bool IsSupportedMadMxDtypeTuple(DataType lhs, DataType rhs, DataType acc) {
  return (IsSupportedFp8Dtype(lhs) || IsPackedFp4Pair(lhs)) &&
         (IsSupportedFp8Dtype(rhs) || IsPackedFp4Pair(rhs)) &&
         IsFloat32(acc);
}

bool IsBufferInScope(const std::string &scope,
                        const std::string &expected_scope) {
  return scope == expected_scope || scope == expected_scope + ".dyn";
}

DataType GetAnnotatedPointerDtype(const PrimExpr &expr,
                                  DataType fallback_dtype) {
  const auto *call = expr.as<CallNode>();
  if (call == nullptr) {
    return fallback_dtype;
  }

  if (call->op.same_as(builtin::tvm_access_ptr()) && !call->args.empty()) {
    if (const auto *type_call = call->args[0].as<CallNode>()) {
      if (!type_call->args.empty()) {
        if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
          return ParseDataType(dtype_name->value);
        }
      }
    }
    return fallback_dtype;
  }

  if (call->op.same_as(builtin::address_of()) && call->args.size() == 1U) {
    if (const auto *load = call->args[0].as<BufferLoadNode>()) {
      return load->buffer->dtype;
    }
  }

  if (call->op.same_as(tl::access_ptr()) && call->args.size() >= 1U) {
    if (const auto *load = call->args[0].as<BufferLoadNode>()) {
      return load->buffer->dtype;
    }
  }

  return fallback_dtype;
}

// Vector-typed local buffers (S.alloc_local / S.alloc_var) have elements
// whose dtype carries lanes (e.g. float32x64). The pass pipeline widens a
// scalar element index `i` into a Ramp(i*lanes, 1, lanes). This recovers
// the scalar element index from such a Ramp.
PrimExpr VectorBufferElemIndex(const PrimExpr &index) {
  if (const auto *ramp = index.as<RampNode>()) {
    int lanes = ramp->dtype.lanes();
    if (lanes > 0) {
      return arith::Analyzer().Simplify(
          floordiv(ramp->base, make_const(ramp->base.dtype(), lanes)));
    }
  }
  return index;
}

bool GetAddressOfIndex(const PrimExpr &expr, PrimExpr *index,
                       const VarNode **buffer_var) {
  const auto *call = expr.as<CallNode>();
  if (call == nullptr) {
    return false;
  }
  if (call->op.same_as(builtin::address_of())) {
    if (call->args.size() != 1U) {
      return false;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load == nullptr || load->indices.size() != 1U) {
      return false;
    }
    *index = load->indices[0];
    *buffer_var = load->buffer->data.get();
    return true;
  }
  if (call->op.same_as(builtin::tvm_access_ptr())) {
    if (call->args.size() < 3U) {
      return false;
    }
    const auto *var = call->args[1].as<VarNode>();
    if (var == nullptr) {
      return false;
    }
    *index = call->args[2];
    *buffer_var = var;
    return true;
  }
  if (call->op.same_as(tl::access_ptr())) {
    if (call->args.size() < 1U) {
      return false;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load == nullptr || load->indices.size() != 1U) {
      return false;
    }
    *index = load->indices[0];
    *buffer_var = load->buffer->data.get();
    return true;
  }
  return false;
}

class PTOHf32ModeAnalyzer : public StmtFunctor<uint8_t(const Stmt &, uint8_t)> {
public:
  using ModeMap =
      std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>;

  static ModeMap Analyze(const PrimFunc &func) {
    PTOHf32ModeAnalyzer analyzer;
    analyzer.VisitStmt(func->body, kDisabled);

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
      uint8_t body_out = VisitStmt(body, header);
      uint8_t next = incoming | body_out;
      if (next == header)
        return must_execute ? body_out : header;
      header = next;
    }
    LOG(FATAL) << "PTO HF32 loop analysis failed to converge for body " << body;
    return incoming;
  }

  uint8_t VisitStmt_(const BindNode *op, uint8_t state) final { return state; }

  uint8_t VisitStmt_(const AttrStmtNode *op, uint8_t state) final {
    return VisitStmt(op->body, state);
  }

  uint8_t VisitStmt_(const IfThenElseNode *op, uint8_t state) final {
    uint8_t then_out = VisitStmt(op->then_case, state);
    uint8_t else_out =
        op->else_case ? VisitStmt(op->else_case.value(), state) : state;
    return then_out | else_out;
  }

  uint8_t VisitStmt_(const ForNode *op, uint8_t state) final {
    return AnalyzeLoop(op->body, state, analyzer_.CanProve(op->extent > 0));
  }

  uint8_t VisitStmt_(const WhileNode *op, uint8_t state) final {
    return AnalyzeLoop(op->body, state, false);
  }

  uint8_t VisitStmt_(const AllocBufferNode *op, uint8_t state) final {
    return state;
  }

  uint8_t VisitStmt_(const DeclBufferNode *op, uint8_t state) final {
    return state;
  }

  uint8_t VisitStmt_(const BufferStoreNode *op, uint8_t state) final {
    return state;
  }

  uint8_t VisitStmt_(const AssertStmtNode *op, uint8_t state) final {
    return state;
  }

  uint8_t VisitStmt_(const SeqStmtNode *op, uint8_t state) final {
    for (const Stmt &stmt : op->seq) {
      state = VisitStmt(stmt, state);
    }
    return state;
  }

  uint8_t VisitStmt_(const EvaluateNode *op, uint8_t state) final {
    const auto *call = op->value.as<CallNode>();
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
      const auto *dtype_name = call->args[9].as<StringImmNode>();
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

  uint8_t VisitStmt_(const SBlockNode *op, uint8_t state) final {
    if (op->init) {
      state |= VisitStmt(op->init.value(), state);
    }
    return VisitStmt(op->body, state);
  }

  uint8_t VisitStmt_(const SBlockRealizeNode *op, uint8_t state) final {
    uint8_t block_out = VisitStmt(op->block, state);
    return is_one(op->predicate) ? block_out : state | block_out;
  }

  std::unordered_map<Call, uint8_t, ObjectPtrHash, ObjectPtrEqual> gemm_modes_;
  arith::Analyzer analyzer_;
};

// Detects whether a statement (loop body) contains a tl.loop_break() call,
// without recursing into nested for loops (their breaks are their own).
class LoopBreakDetector : public tirx::StmtExprVisitor {
public:
  bool found = false;
  void VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(tl::loop_break())) {
      found = true;
    }
    if (!found)
      StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const ForNode *op) override { (void)op; }
  void VisitStmt_(const WhileNode *op) override { (void)op; }
};

static bool ContainsLoopBreak(const Stmt &stmt) {
  LoopBreakDetector det;
  det(stmt);
  return det.found;
}
bool ContainsSimdVectorStore(const Stmt &stmt) {
  bool found = false;
  tirx::PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::simd_vsts());
    }
  });
  return found;
}

class SimtPersistentBufferCollector final : public StmtExprVisitor {
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
  void RecordAllocation(const Buffer &buffer) {
    std::string scope = buffer.scope();
    if (simt_depth_ == 0 && (scope == "local" || scope == "local.fragment")) {
      outer_local_allocations_.insert(buffer->data.get());
    }
  }

  void RecordAccess(const Buffer &buffer) {
    if (simt_depth_ > 0) {
      simt_accesses_.insert(buffer->data.get());
    }
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    RecordAllocation(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    const bool is_simt = op->name_hint == "SIMT_VF";
    if (is_simt) {
      ++simt_depth_;
    }
    for (const Buffer &buffer : op->alloc_buffers) {
      RecordAllocation(buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
    if (is_simt) {
      --simt_depth_;
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    RecordAccess(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    RecordAccess(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  int simt_depth_{0};
  std::unordered_set<const VarNode *> outer_local_allocations_;
  std::unordered_set<const VarNode *> simt_accesses_;
};

struct MixedSectionVariableInfo {
  using VarSet = std::unordered_set<const VarNode *>;

  std::unordered_map<const SBlockNode *, VarSet> external_vars_by_section;
  VarSet captured_vars;
};

class MixedSectionLocalVarAnalyzer {
  using VarSet = MixedSectionVariableInfo::VarSet;
  using OwnerMap =
      std::unordered_map<const VarNode *, const SBlockNode *>;

  class StructureCollector final : public StmtExprVisitor {
  public:
    void Collect(const Stmt &body) { VisitStmt(body); }

    const std::vector<const SBlockNode *> &sections() const { return sections_; }
    const OwnerMap &owners() const { return owners_; }

  private:
    static bool IsPhysicalSection(const SBlockNode *op) {
      return op->name_hint == "CUBE" || op->name_hint == "VECTOR";
    }

    void RecordAllocation(const Buffer &buffer) {
      if (buffer.scope() != "local.var") {
        return;
      }
      const VarNode *var = buffer->data.get();
      auto [it, inserted] = owners_.emplace(var, current_section_);
      ICHECK(inserted || it->second == current_section_)
          << "A local.var buffer is allocated in multiple mixed-section "
             "ownership regions: "
          << buffer;
    }

    void VisitStmt_(const AllocBufferNode *op) final {
      RecordAllocation(op->buffer);
    }

    void VisitStmt_(const SBlockNode *op) final {
      const bool is_physical = IsPhysicalSection(op);
      ICHECK(!is_physical || current_section_ == nullptr)
          << "Nested CUBE/VECTOR physical sections are not supported: "
          << op->name_hint;

      const SBlockNode *saved_section = current_section_;
      if (is_physical) {
        current_section_ = op;
        sections_.push_back(op);
      }
      for (const Buffer &buffer : op->alloc_buffers) {
        RecordAllocation(buffer);
      }
      if (op->init.defined()) {
        VisitStmt(op->init.value());
      }
      VisitStmt(op->body);
      current_section_ = saved_section;
    }

    const SBlockNode *current_section_{nullptr};
    std::vector<const SBlockNode *> sections_;
    OwnerMap owners_;
  };

  class SectionFlowAnalyzer final
      : public StmtFunctor<VarSet(const Stmt &, VarSet)> {
  public:
    SectionFlowAnalyzer(const SBlockNode *section, const OwnerMap &owners)
        : section_(section), owners_(owners) {}

    VarSet Analyze() {
      VarSet defined;
      for (const Buffer &buffer : section_->alloc_buffers) {
        DefineAllocation(buffer, &defined);
      }
      if (section_->init.defined()) {
        defined = VisitStmt(section_->init.value(), std::move(defined));
      }
      VisitStmt(section_->body, std::move(defined));
      return external_vars_;
    }

  private:
    class ReadCollector final : public StmtExprVisitor {
    public:
      ReadCollector(SectionFlowAnalyzer *parent, const VarSet &defined)
          : parent_(parent), defined_(defined) {}

      void VisitExpr_(const BufferLoadNode *op) final {
        parent_->RecordAccess(op->buffer, defined_);
        StmtExprVisitor::VisitExpr_(op);
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

    void AnalyzeExpr(const PrimExpr &expr, const VarSet &defined) {
      ReadCollector(this, defined)(expr);
    }

    void AnalyzeExprs(const ffi::Array<PrimExpr> &exprs,
                      const VarSet &defined) {
      for (const PrimExpr &expr : exprs) {
        AnalyzeExpr(expr, defined);
      }
    }

    bool IsLocalVar(const Buffer &buffer) const {
      return buffer.scope() == "local.var";
    }

    void RecordAccess(const Buffer &buffer, const VarSet &defined) {
      if (!IsLocalVar(buffer)) {
        return;
      }
      const VarNode *var = buffer->data.get();
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

    void DefineAllocation(const Buffer &buffer, VarSet *defined) {
      if (IsLocalVar(buffer)) {
        defined->insert(buffer->data.get());
      }
    }

    VarSet VisitStmt_(const BindNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      return defined;
    }

    VarSet VisitStmt_(const AttrStmtNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      return VisitStmt(op->body, std::move(defined));
    }

    VarSet VisitStmt_(const IfThenElseNode *op, VarSet defined) final {
      AnalyzeExpr(op->condition, defined);
      VarSet then_out = VisitStmt(op->then_case, defined);
      VarSet else_out = op->else_case.defined()
                            ? VisitStmt(op->else_case.value(), defined)
                            : defined;
      return Intersect(std::move(then_out), else_out);
    }

    VarSet VisitStmt_(const ForNode *op, VarSet defined) final {
      AnalyzeExpr(op->min, defined);
      AnalyzeExpr(op->extent, defined);
      if (op->step.defined()) {
        AnalyzeExpr(op->step.value(), defined);
      }
      VarSet body_out = VisitStmt(op->body, defined);
      return arith::Analyzer().CanProve(op->extent > 0) ? body_out : defined;
    }

    VarSet VisitStmt_(const WhileNode *op, VarSet defined) final {
      AnalyzeExpr(op->condition, defined);
      VisitStmt(op->body, defined);
      return defined;
    }

    VarSet VisitStmt_(const AllocBufferNode *op, VarSet defined) final {
      DefineAllocation(op->buffer, &defined);
      return defined;
    }

    VarSet VisitStmt_(const DeclBufferNode *op, VarSet defined) final {
      (void)op;
      return defined;
    }

    VarSet VisitStmt_(const BufferStoreNode *op, VarSet defined) final {
      AnalyzeExpr(op->value, defined);
      AnalyzeExprs(op->indices, defined);
      if (op->predicate.defined()) {
        AnalyzeExpr(op->predicate.value(), defined);
      }
      RecordAccess(op->buffer, defined);
      if (IsLocalVar(op->buffer)) {
        defined.insert(op->buffer->data.get());
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

    VarSet VisitStmt_(const SBlockNode *op, VarSet defined) final {
      ICHECK(op->name_hint != "CUBE" && op->name_hint != "VECTOR")
          << "Nested CUBE/VECTOR physical sections are not supported";
      for (const Buffer &buffer : op->alloc_buffers) {
        DefineAllocation(buffer, &defined);
      }
      if (op->init.defined()) {
        defined = VisitStmt(op->init.value(), std::move(defined));
      }
      return VisitStmt(op->body, std::move(defined));
    }

    VarSet VisitStmt_(const SBlockRealizeNode *op, VarSet defined) final {
      AnalyzeExprs(op->iter_values, defined);
      AnalyzeExpr(op->predicate, defined);
      VarSet block_out = VisitStmt(op->block, defined);
      if (arith::Analyzer().CanProve(op->predicate)) {
        return block_out;
      }
      return Intersect(std::move(block_out), defined);
    }

    const SBlockNode *section_;
    const OwnerMap &owners_;
    VarSet external_vars_;
  };

  class OuterAllocationChecker final
      : public StmtFunctor<VarSet(const Stmt &, VarSet)> {
  public:
    OuterAllocationChecker(
        const std::unordered_map<const SBlockNode *, VarSet> &section_vars,
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

    static void DefineAllocation(const Buffer &buffer, VarSet *defined) {
      if (buffer.scope() == "local.var") {
        defined->insert(buffer->data.get());
      }
    }

    VarSet VisitStmt_(const BindNode *op, VarSet defined) final {
      (void)op;
      return defined;
    }

    VarSet VisitStmt_(const AttrStmtNode *op, VarSet defined) final {
      return VisitStmt(op->body, std::move(defined));
    }

    VarSet VisitStmt_(const IfThenElseNode *op, VarSet defined) final {
      VarSet then_out = VisitStmt(op->then_case, defined);
      VarSet else_out = op->else_case.defined()
                            ? VisitStmt(op->else_case.value(), defined)
                            : defined;
      return Intersect(std::move(then_out), else_out);
    }

    VarSet VisitStmt_(const ForNode *op, VarSet defined) final {
      VarSet body_out = VisitStmt(op->body, defined);
      return arith::Analyzer().CanProve(op->extent > 0) ? body_out : defined;
    }

    VarSet VisitStmt_(const WhileNode *op, VarSet defined) final {
      VisitStmt(op->body, defined);
      return defined;
    }

    VarSet VisitStmt_(const AllocBufferNode *op, VarSet defined) final {
      DefineAllocation(op->buffer, &defined);
      return defined;
    }

    VarSet VisitStmt_(const DeclBufferNode *op, VarSet defined) final {
      (void)op;
      return defined;
    }

    VarSet VisitStmt_(const BufferStoreNode *op, VarSet defined) final {
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

    VarSet VisitStmt_(const SBlockNode *op, VarSet defined) final {
      const bool is_physical =
          op->name_hint == "CUBE" || op->name_hint == "VECTOR";
      if (is_physical) {
        auto section_it = section_vars_.find(op);
        ICHECK(section_it != section_vars_.end());
        for (const VarNode *var : section_it->second) {
          ICHECK(defined.count(var))
              << "Mixed-section local.var access is not dominated by an "
                 "outer allocation: "
              << var->name_hint;
          captured_vars_->insert(var);
        }
        return defined;
      }

      for (const Buffer &buffer : op->alloc_buffers) {
        DefineAllocation(buffer, &defined);
      }
      if (op->init.defined()) {
        defined = VisitStmt(op->init.value(), std::move(defined));
      }
      return VisitStmt(op->body, std::move(defined));
    }

    VarSet VisitStmt_(const SBlockRealizeNode *op, VarSet defined) final {
      VarSet block_out = VisitStmt(op->block, defined);
      if (arith::Analyzer().CanProve(op->predicate)) {
        return block_out;
      }
      return Intersect(std::move(block_out), defined);
    }

    const std::unordered_map<const SBlockNode *, VarSet> &section_vars_;
    VarSet *captured_vars_;
  };

public:
  static MixedSectionVariableInfo Analyze(const Stmt &body) {
    StructureCollector collector;
    collector.Collect(body);

    MixedSectionVariableInfo result;
    for (const SBlockNode *section : collector.sections()) {
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
  DataType dtype;

  bool CanMerge(const CopyPadState &other) const {
    return valid && other.valid && dtype == other.dtype &&
           (changed || other.changed);
  }
};

class CopyPadStateAnalyzer : public tirx::StmtExprVisitor {
public:
  explicit CopyPadStateAnalyzer(CopyPadState entry) : state_(entry) {}

  CopyPadState state() const { return state_; }

  void VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
      ICHECK_EQ(op->args.size(), 1U);
      state_ = CopyPadState{true, true, op->args[0].dtype()};
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) override {
    CopyPadState entry = state_;
    CopyPadStateAnalyzer then_analyzer(entry);
    then_analyzer(op->then_case);
    CopyPadState then_state = then_analyzer.state();

    CopyPadState else_state = entry;
    if (op->else_case.defined()) {
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
class StableConditionChecker : public tirx::StmtExprVisitor {
public:
  bool stable{true};

  void VisitExpr_(const BufferLoadNode *op) override {
    (void)op;
    stable = false;
  }

  void VisitExpr_(const CallNode *op) override {
    (void)op;
    stable = false;
  }
};

bool IsStableCondition(const PrimExpr &condition) {
  StableConditionChecker checker;
  checker(condition);
  return checker.stable;
}

bool AreComplementaryConditions(const PrimExpr &lhs, const PrimExpr &rhs) {
  auto same_operands = [](const PrimExpr &lhs_a, const PrimExpr &lhs_b,
                          const PrimExpr &rhs_a, const PrimExpr &rhs_b) {
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
      pass_ctx->GetConfig<Bool>(tl::kEnableFastMath, Bool(false)).value();
}

void CodeGenTileLangPTO::ValidateVecMode_(const CallNode *op, size_t mode_idx,
                                          const char *expected) {
  ICHECK_LT(mode_idx, op->args.size())
      << "PTO SIMD op has no mode argument at index " << mode_idx;
  auto mode_str = Downcast<StringImm>(op->args[mode_idx])->value;
  ICHECK_EQ(mode_str, expected)
      << "PTO codegen currently only supports " << expected
      << " for this SIMD op, got " << mode_str;
}

void CodeGenTileLangPTO::AddFunction(const GlobalVar &gvar,
                                     const PrimFunc &func) {
  RegisterFunction_(gvar, func);
  current_function_name_ = GetFunctionName_(gvar);
  InitFuncState_(func);
  name_supply_->ReserveName("pto");
  name_supply_->ReserveName("scalar");
  name_supply_->ReserveName("tl");
  ValidateKernelCapabilities(func);
  fragment_info_.clear();
  local_var_buffers_.clear();
  current_unroll_factor_loop_var_ = Optional<Var>();
  current_unroll_factor_ = 0;
  mixed_captured_local_vars_.clear();
  mixed_external_vars_by_section_.clear();
  mixed_entry_snapshot_ids_.clear();
  simd_pair_vars_.clear();
  gemm_emit_contexts_.clear();
  gemm_emit_context_by_call_.clear();
  gemm_zero_addr_emitted_ = false;
  inside_simtvf_body_ = false;
  persistent_buffer_vars_ = SimtPersistentBufferCollector().Collect(func->body);
  hf32_mode_by_gemm_ = PTOHf32ModeAnalyzer::Analyze(func);
  copy_pad_value_counter_ = 0;
  current_copy_pad_value_id_ = -1;
  current_copy_pad_value_dtype_ = DataType::Void();
  uniform_const_copy_pad_value_ = PrimExpr();
  bool saw_copy_pad_value = false;
  bool has_uniform_const_copy_pad_value = true;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr ||
        !call->op.same_as(tl::ascend_set_copy_pad_value())) {
      return;
    }
    ICHECK_EQ(call->args.size(), 1U);
    PrimExpr value = call->args[0];
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
    uniform_const_copy_pad_value_ = PrimExpr();
  }
  bool func_has_gemm_l1 = HasAscendGemmL1(func);
  bool func_has_blockscaled_gemm_l1 = HasAscendBlockscaledGemmL1(func);
  current_function_has_gemm_ =
      func_has_gemm_l1 || func_has_blockscaled_gemm_l1 || HasAscendMad(func);
  current_function_is_cube_ = IsAscendCubeKernel(func);
  current_function_is_mixed_ = IsAscendMixedKernel(func);
  in_mixed_vector_section_ = false;
  inside_mixed_section_ = false;
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
  if (func_has_gemm_l1 || func_has_blockscaled_gemm_l1) {
    std::vector<Call> gemm_calls;
    tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
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
  code << "from ptodsl import pto, scalar\n";
  code << "import tilelang.contrib.ptodsl as tl\n";
  code << "\n";
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangPTO::PrintFuncDecorator_(std::ostream &os) { // NOLINT(*)
  os << "@pto.jit(name=\"" << current_function_name_ << "\"";
  if (!current_function_is_mixed_) {
    os << ", kernel_kind=\""
       << (current_function_is_cube_ ? "cube" : "vector") << "\"";
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
    const ffi::String &function_name, const PrimFunc &func,
    std::ostream &os) { // NOLINT(*)
  os << "def " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    Var v = func->params[i];
    if (i > 0) {
      os << ", ";
    }
    os << AllocVarID(v.get());
    if (func->buffer_map.count(v)) {
      Buffer buffer = func->buffer_map[v];
      os << ": " << PointerTypeName(buffer->dtype, "gm");
    } else if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        std::string scope =
            ptr->storage_scope.empty() ? "gm" : ptr->storage_scope;
        if (scope == "global") {
          scope = "gm";
        }
        os << ": " << PointerTypeName(prim->dtype, scope);
      } else {
        os << ": " << ScalarType(v->dtype);
      }
    } else {
      os << ": " << ScalarType(v->dtype);
    }
  }
  os << ")";

  for (const auto &param : func->params) {
    if (auto *ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType_(param.get(), prim->dtype);
      }
    }
  }
}

std::string CodeGenTileLangPTO::ScalarType(DataType t) const {
  return DataTypeName(t);
}

std::string CodeGenTileLangPTO::PointerTypeName(DataType t,
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
                                           DataType elem_dtype) const {
  // Compare physical PTO element types. TVM bool buffers use int8 backing
  // storage even though scalar bool expressions use i1.
  DataType storage_dtype =
      elem_dtype.is_bool() ? DataType::Int(8) : elem_dtype;
  if (HandleTypeMatch_(buffer_var, storage_dtype)) {
    return false;
  }
  if (IsPackedFp4Pair(storage_dtype) &&
      HandleTypeMatch_(buffer_var, storage_dtype.element_of())) {
    return false;
  }
  if (storage_dtype.is_float4_e2m1fn() && storage_dtype.lanes() == 1 &&
      HandleTypeMatch_(buffer_var, storage_dtype.with_lanes(2))) {
    return false;
  }
  // Signedness is part of the authored PTO pointer type. Do not treat
  // int/uint aliases as interchangeable: scalar.load derives signedness from
  // the pointer element type, which later selects signed or unsigned scalar ops.
  if (storage_dtype.lanes() == 1 && storage_dtype.bits() >= 8 &&
      storage_dtype.is_int() && HandleTypeMatch_(buffer_var, DataType::Bool())) {
    return false;
  }
  return true;
}

std::string CodeGenTileLangPTO::ScalarPointerBase_(
    const VarNode *buffer_var, DataType elem_dtype, const std::string &scope) {
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

bool CodeGenTileLangPTO::HasAscendGemmL1(const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_gemm_l1());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::HasAscendBlockscaledGemmL1(
    const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_blockscaled_gemm_l1());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::IsAscendCubeKernel(const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *block = node.as<SBlockNode>()) {
      found = block->name_hint == "CUBE";
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
              call->op.same_as(tl::ascend_copy_ubuf_to_cbuf()) ||
              call->op.same_as(tl::ascend_copy_matrix_cc_to_ub()) ||
              call->op.same_as(tl::ascend_copy_matrix_cc_to_gm());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::HasAscendMad(const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_mad()) ||
              call->op.same_as(tl::ascend_mad_mx());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::IsAscendMixedKernel(const PrimFunc &func) const {
  bool has_cube = false;
  bool has_vector = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      has_cube = has_cube || block->name_hint == "CUBE";
      has_vector = has_vector || block->name_hint == "VECTOR";
    }
  });
  return has_cube && has_vector;
}

bool CodeGenTileLangPTO::IsInlineableInvariantSimdBind(
    const PrimExpr &value) const {
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

std::string CodeGenTileLangPTO::ScopeOfBuffer(const BufferNode *buffer) const {
  std::string scope;
  auto it = alloc_storage_scope_.find(buffer->data.get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }
  return scope;
}

void CodeGenTileLangPTO::ExtractSimtThreadExtents(const SBlockNode *op,
                                                  int64_t *thread_x,
                                                  int64_t *thread_y,
                                                  int64_t *thread_z) const {
  *thread_x = 1;
  *thread_y = 1;
  *thread_z = 1;

  tirx::PostOrderVisit(
      op->body, [&](const ffi::ObjectRef &node) {
        const auto *attr = node.as<AttrStmtNode>();
        if (attr == nullptr || attr->attr_key != tirx::attr::thread_extent) {
          return;
        }
        const auto *iv = attr->node.as<IterVarNode>();
        if (!iv) {
          return;
        }

        int64_t *dimension = nullptr;
        if (iv->thread_tag == "threadIdx.x") {
          dimension = thread_x;
        } else if (iv->thread_tag == "threadIdx.y") {
          dimension = thread_y;
        } else if (iv->thread_tag == "threadIdx.z") {
          dimension = thread_z;
        } else {
          return;
        }

        int64_t value = 0;
        ICHECK(TryGetConstInt(attr->value, &value))
            << "PTO inline SIMT launch dimensions must be static, got "
            << attr->value;
        ICHECK_GT(value, 0)
            << "PTO inline SIMT launch dimensions must be positive, got "
            << value;
        *dimension = value;
      });
}

void CodeGenTileLangPTO::EmitInlineSimtVF(const SBlockNode *op,
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
  for (const Buffer &buffer : op->alloc_buffers) {
    EmitBufferAllocation(buffer);
  }
  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
  if (stream.tellp() == body_start) {
    PrintIndent();
    stream << "pass\n";
  }

  inside_simtvf_body_ = saved_inside_simtvf_body;
  EndScope(simt_scope);
}

std::string CodeGenTileLangPTO::GetPointerExpr(const VarNode *buffer_var,
                                                  DataType elem_dtype,
                                                  const PrimExpr &index) {
  std::string scope = "global";
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }

  std::string base = GetVarID(buffer_var);
  // Cast the handle to the access dtype when they differ (e.g. a raw-byte
  // shared container accessed as f32, or a GM buffer reinterpreted through
  // T.view). PTO MLIR copy ops require src/dst element byte widths to match,
  // mirroring AscendC's (__gm__/__ubuf__ T*) casts.
  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    if (NeedsCastptr_(buffer_var, elem_dtype)) {
      std::string pto_space =
          (scope == "global" || scope.empty()) ? "gm" : "ub";
      base = "pto.castptr(" + base + ", " + PointerTypeName(elem_dtype, pto_space) +
             ")";
    }
  }

  if (is_zero(index)) {
    return base;
  }

  PrimExpr ptr_index = index;
  if (const auto *ramp = index.as<RampNode>()) {
    CheckContiguousRampStride(index, "pointer address");
    ptr_index = arith::Analyzer().Simplify(ramp->base);
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
      index_str = "((" + index_str + ") // " + std::to_string(pack_factor) + ")";
    }
  }

  if (scope == "global" || scope.empty() || scope == "shared" ||
      scope == "shared.dyn") {
    return "pto.addptr(" + base + ", " + index_str + ")";
  }

  LOG(FATAL) << "Unsupported storage scope in PTO pointer emission: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::GetPointerExpr(const BufferNode *buffer,
                                                  const PrimExpr &index) {
  return GetPointerExpr(buffer->data.get(), buffer->dtype, index);
}

std::string
CodeGenTileLangPTO::GetVectorLocalRef(const VarNode *buffer_var,
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
CodeGenTileLangPTO::GetMutableVectorRef(const PrimExpr &address,
                                           const std::string &op_name) {
  PrimExpr dst_index;
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
    scope = GetPtrStorageScope(GetRef<Var>(buffer_var));
  }
  if (scope == "local.var") {
    return GetVarID(buffer_var);
  }
  ICHECK(scope == "local" || scope == "local.fragment")
      << "PTO " << op_name << " destination must be a local vector value, got "
      << scope;
  return GetVectorLocalRef(buffer_var, dst_index, "PTO " + op_name);
}

std::string CodeGenTileLangPTO::PrintCondition(const PrimExpr &condition) {
  DataType dtype = condition.dtype();
  ICHECK(dtype.is_scalar() &&
         (dtype.is_bool() || dtype.is_int() || dtype.is_uint()))
      << "PTO control-flow condition must be a scalar integer predicate, got "
      << dtype;
  std::string value = RemoveOutermostParentheses(PrintExpr_(condition));
  // TIR follows C truthiness for all integer conditions.  Always compare at
  // the use site because a bool-typed expression may be materialized in an
  // i8 local.var by earlier lowering.  This also avoids truncating arbitrary
  // nonzero integers to i1 (which would make even values false).
  return "tl.as_logical_bool(" + value + ")";
}

std::string CodeGenTileLangPTO::ScalarLoad(const BufferNode *buffer,
                                              const PrimExpr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  if (scope == "local.var") {
    return GetVarID(buffer->data.get());
  }

  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));
  if (inside_simtvf_body_ && (scope == "local.fragment" || scope == "local")) {
    ICHECK(IsSupportedSIMTLocalScalarAccessType(buffer->dtype))
        << "PTO SIMT local scalar load supports float16, float32, bfloat16, "
           "int8, uint8, int32, uint32, and FP8 only, got "
        << buffer->dtype;
    std::string value =
        "scalar.load(" + GetVarID(buffer->data.get()) + ", " + index_str + ")";
    if (IsInteger32(buffer->dtype)) {
      return "scalar.cast(" + value + ", " + DataTypeName(buffer->dtype) + ")";
    }
    return value;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    std::string base =
        ScalarPointerBase_(buffer->data.get(), buffer->dtype, scope);
    return "scalar.load(" + base + ", " + index_str + ")";
  }

  if (scope == "local.fragment" || scope == "local") {
    return GetVarID(buffer->data.get()) + "[" + index_str + "]";
  }

  LOG(FATAL) << "Unsupported PTO scalar load scope: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::LocalVarStoreValue(const PrimExpr &value,
                                                   DataType dtype) {
  std::string value_expr =
      RemoveOutermostParentheses(PrintExpr_(value));
  if (dtype.is_handle()) {
    return value_expr;
  }
  bool is_literal = value.as<IntImmNode>() != nullptr ||
                    value.as<FloatImmNode>() != nullptr;
  if (dtype.is_scalar() && is_literal) {
    return "pto.const(" + value_expr + ", dtype=" + DataTypeName(dtype) +
           ")";
  }
  if (dtype.is_bool() || dtype.bits() == 1 || dtype.is_int() ||
      dtype.is_uint()) {
    return ScalarCastExpr(value_expr, dtype, "PTO local.var store");
  }
  return value_expr;
}

void CodeGenTileLangPTO::EmitScalarStore(const BufferNode *buffer,
                                            const std::string &value,
                                            const PrimExpr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));

  if (scope == "local.var") {
    stream << GetVarID(buffer->data.get()) << " = " << value << "\n";
    EmitMixedEntrySnapshot(buffer->data.get());
    return;
  }

  if (scope == "local.fragment" || scope == "local") {
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalScalarAccessType(buffer->dtype))
          << "PTO SIMT local scalar store supports float16, bfloat16, "
             "float32, int8, uint8, int32, uint32, and FP8 only, got "
          << buffer->dtype;
      std::string store_value = value;
      if (IsInteger32(buffer->dtype)) {
        store_value = "scalar.cast(" + store_value + ", pto.i32)";
      }
      stream << "scalar.store(" << store_value << ", "
             << GetVarID(buffer->data.get()) << ", " << index_str << ")\n";
      return;
    }
    stream << GetVarID(buffer->data.get()) << "[" << index_str
           << "] = " << value << "\n";
    return;
  }

  if (scope == "shared" || scope == "shared.dyn" || scope == "global" ||
      scope.empty()) {
    std::string base =
        ScalarPointerBase_(buffer->data.get(), buffer->dtype, scope);
    stream << "scalar.store(" << value << ", " << base << ", " << index_str
           << ")\n";
    return;
  }

  LOG(FATAL) << "Unsupported PTO scalar store scope: " << scope;
}

void CodeGenTileLangPTO::EmitBufferAllocation(const Buffer &buffer) {
  std::string scope = GetPtrStorageScope(buffer->data);
  alloc_storage_scope_[buffer->data.get()] = scope;
  ValidateSharedScope(scope);

  if (scope == "shared.dyn") {
    PrintIndent();
    std::string vid = AllocVarID(buffer->data.get());
    // MergeUBAllocations assigns offsets when multiple buffers share the UB
    // arena. The generated PTODSL pointer does not require the logical extent
    // to be a constant.
    stream << vid << " = pto.castptr(pto.const(0, dtype=pto.i64), "
           << PointerTypeName(buffer->dtype, "ub") << ")\n";
    RegisterHandleType_(buffer->data.get(), buffer->dtype);
    return;
  } else if (scope == "local.fragment" || scope == "local") {
    // Allocations nested in a runtime loop keep the same TIR Var identity and
    // are re-initialized on each iteration.  The loop/carry analysis may also
    // reserve the name before reaching this statement.  Reuse the stable
    // Python name while still emitting the initialization at this location.
    std::string vid = var_idmap_.count(buffer->data.get())
                          ? GetVarID(buffer->data.get())
                          : AllocVarID(buffer->data.get());
    auto alloc = AllocBuffer(buffer);
    auto opt_size = alloc.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO local.fragment currently requires constant allocation size";
    PrintIndent();
    const bool persistent =
        persistent_buffer_vars_.count(buffer->data.get()) != 0;
    DataType elem = buffer->dtype.is_vector() ? buffer->dtype.element_of()
                                              : buffer->dtype;
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalStorageType(buffer->dtype))
          << "PTO SIMT local allocation supports float16, bfloat16, "
             "float32, int8, uint8, int32, uint32, and "
             "float8_e4m3fn/float8_e5m2, got "
          << buffer->dtype;
      stream << vid << " = pto.alloc_buffer((" << opt_size.value() << ",), "
             << (buffer->dtype.is_vector()
                     ? (tl::IsAscendVectorizableFP8(buffer->dtype)
                            ? DataTypeName(buffer->dtype)
                            : ScalarType(elem))
                     : SIMTLocalStorageTypeName(buffer->dtype))
             << ")\n";
    } else if (persistent) {
      ICHECK(IsFloat32(buffer->dtype))
          << "PTO persistent local allocation currently supports float32 "
             "only, got "
          << buffer->dtype;
      stream << vid << " = pto.alloc_buffer((" << opt_size.value() << ",), "
             << ScalarType(elem) << ")\n";
    } else if (buffer->dtype.is_vector()) {
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
    if (local_var_buffers_.count(buffer->data.get())) {
      return;
    }
    local_var_buffers_.insert(buffer->data.get());
    DataType dtype = buffer->dtype;
    std::string vid = var_idmap_.count(buffer->data.get())
                          ? GetVarID(buffer->data.get())
                          : AllocVarID(buffer->data.get());
    if (vid.rfind("__cond_", 0) == 0) {
      vid = "tl_cond_" + vid.substr(7);
      var_idmap_[buffer->data.get()] = vid;
    } else if (!vid.empty() && vid.front() == '_') {
      // PTODSL rewrites native Python if statements through BranchHandle,
      // whose underscore-prefixed attributes are reserved for internals.
      // Give local.var values such as `_tmp_id` a unique public name so they
      // can be carried out of nested branches.
      vid = name_supply_->FreshName("tl" + vid, false);
      var_idmap_[buffer->data.get()] = vid;
    }
    if (current_function_has_gemm_) {
      // Keep authored PTO surface values (not Python ints) so AST-rewritten
      // `if` / br.assign can merge tile indices across branches.
      if (IsVectorLocalVarDtype(dtype) || dtype.is_handle()) {
        stream << vid << " = None\n";
      } else {
        stream << vid << " = pto.const(" << (dtype.is_float() ? "0.0" : "0")
               << ", dtype=" << DataTypeName(dtype) << ")\n";
      }
      RegisterHandleType_(buffer->data.get(), buffer->dtype);
      EmitMixedEntrySnapshot(buffer->data.get());
      return;
    }
    // Wide vector SSA; scalar values stay as PTO surface values.
    if (IsVectorLocalVarDtype(dtype) || dtype.is_handle()) {
      stream << vid << " = None\n";
    } else if (IsSurfaceScalarLocalVarDtype(dtype)) {
      stream << vid << " = pto.const(" << (dtype.is_float() ? "0.0" : "0")
             << ", dtype="
             << DataTypeName(dtype)
             << ")\n";
    } else {
      LOG(FATAL) << "Unsupported PTO local.var dtype: " << dtype;
    }
    EmitMixedEntrySnapshot(buffer->data.get());
  }

  RegisterHandleType_(buffer->data.get(), buffer->dtype);
}

std::string CodeGenTileLangPTO::GetAddressOfExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1U);
  const auto *load = op->args[0].as<BufferLoadNode>();
  ICHECK(load) << "address_of expects BufferLoad";
  ICHECK_EQ(load->indices.size(), 1U)
      << "CodeGenTileLangPTO only supports flat memory";
  return GetPointerExpr(load->buffer.get(), load->indices[0]);
}

std::string CodeGenTileLangPTO::GetAccessPtrExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 5U);
  auto buffer_var = Downcast<Var>(op->args[1]);
  DataType elem_dtype = DataType::Float(32);

  if (const auto *type_call = op->args[0].as<CallNode>()) {
    if (!type_call->args.empty()) {
      if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
        elem_dtype = ParseDataType(dtype_name->value);
      }
    }
  } else if (HandleTypeMatch_(buffer_var.get(), DataType::Float(32))) {
    elem_dtype = DataType::Float(32);
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
  PrimExpr dst_index;
  PrimExpr src_index;
  DataType dst_dtype;
  DataType src_dtype;
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO GM->UB MTE destination", &dst_var,
                      &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO GM->UB MTE source", &src_var,
                      &src_index, &src_dtype, &src_scope);
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
    arith::Analyzer padding_analyzer;
    ICHECK(padding_analyzer.CanProveEqual(
               op->args[5], make_zero(op->args[5].dtype())) &&
           padding_analyzer.CanProveEqual(
               op->args[6], make_zero(op->args[6].dtype())))
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
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[9]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[10]));

  std::ostringstream os;
  os << "pto.mte_gm_ub(" << src << ", " << dst << ", " << l2_cache_ctrl
     << ", " << burst_len << ", nburst=(" << burst_num << ", "
     << src_stride << ", " << dst_stride << ")";
  if (data_select != 0) {
    std::string pad_value;
    DataType pad_dtype = current_copy_pad_value_dtype_;
    if (current_copy_pad_value_id_ >= 0) {
      pad_value = "tl_copy_pad_" +
                  std::to_string(current_copy_pad_value_id_);
    } else {
      ICHECK(uniform_const_copy_pad_value_.defined())
          << "PTO GM->UB MTE with dataSelect == 1 requires a preceding "
             "tl.ascend_set_copy_pad_value in the same function";
      pad_value = GetCopyPadValueExpr_(uniform_const_copy_pad_value_);
      pad_dtype = uniform_const_copy_pad_value_.dtype();
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
  PrimExpr dst_index;
  PrimExpr src_index;
  DataType dst_dtype;
  DataType src_dtype;
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO UB->GM MTE destination", &dst_var,
                      &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO UB->GM MTE source", &src_var,
                      &src_index, &src_dtype, &src_scope);
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
                           op->args[7], make_zero(DataType::Int(32)),
                           make_zero(DataType::Int(32)), false,
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
  DataType dtype = op->args[0].dtype();
  int bits = dtype.bits();
  bool supported =
      dtype.is_scalar() && (dtype.is_bfloat16() ||
                            (dtype.is_float() && (bits == 16 || bits == 32)) ||
                            ((dtype.is_int() || dtype.is_uint()) &&
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

  DataType dtype = op->args[1].dtype();
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
                                            DataType expected_dtype) {
  ICHECK_GE(current_copy_pad_value_id_, 0);
  ICHECK(current_copy_pad_value_dtype_ == expected_dtype);
  PrintIndent();
  stream << "tl_copy_pad_" << merged_value_id << " = tl_copy_pad_"
         << current_copy_pad_value_id_ << "\n";
}

std::string CodeGenTileLangPTO::GetCopyPadValueExpr_(const PrimExpr &value) {
  if (const auto *imm = value.as<IntImmNode>()) {
    return "pto.const(" + std::to_string(imm->value) +
           ", dtype=" + ScalarType(value.dtype()) + ")";
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
           ", dtype=" + ScalarType(value.dtype()) + ")";
  }
  return RemoveOutermostParentheses(PrintExpr_(value));
}

void CodeGenTileLangPTO::GetCopyEndpoint_(const PrimExpr &expr,
                                             const char *context,
                                             const VarNode **buffer_var,
                                             PrimExpr *index, DataType *dtype,
                                             std::string *scope) const {
  ICHECK(GetAddressOfIndex(expr, index, buffer_var))
      << context << " expects address_of/tvm_access_ptr, got " << expr;
  *dtype = GetAnnotatedPointerDtype(expr, DataType::Void());
  ICHECK(!dtype->is_void()) << context << " has no element dtype annotation";

  auto scope_it = alloc_storage_scope_.find(*buffer_var);
  if (scope_it != alloc_storage_scope_.end()) {
    *scope = scope_it->second;
  } else if (const auto *ptr =
                 (*buffer_var)->type_annotation.as<PointerTypeNode>()) {
    *scope = ptr->storage_scope;
  } else {
    scope->clear();
  }
  if (*scope == "gm") {
    *scope = "global";
  }
}

void CodeGenTileLangPTO::ValidateUBCopyLayout_(
    const PrimExpr &index, DataType dtype, const PrimExpr &burst_num,
    const PrimExpr &burst_len, const PrimExpr &ub_stride,
    const PrimExpr &left_padding, const PrimExpr &right_padding,
    bool uses_padding, const char *context) const {
  ICHECK(dtype.is_scalar() || IsPackedFp4Pair(dtype))
      << context << " requires a scalar or packed FP4-pair dtype, got "
      << dtype;

  arith::Analyzer analyzer;
  PrimExpr elem_index = index;
  if (const auto *ramp = index.as<RampNode>()) {
    CheckContiguousRampStride(index, context);
    elem_index = analyzer.Simplify(ramp->base);
  }

  PrimExpr byte_offset =
      ElementIndexToByteOffsetExpr(elem_index, dtype, context, &analyzer);
  PrimExpr offset_mod = analyzer.Simplify(
      floormod(byte_offset, make_const(byte_offset.dtype(), 32)));
  if (!analyzer.CanProveEqual(offset_mod, make_zero(offset_mod.dtype()))) {
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
    ICHECK(analyzer.CanProve(
        left_padding >= make_zero(left_padding.dtype())))
        << context << " requires a non-negative left padding count, got "
        << left_padding;
  }
  if (!has_const_right) {
    ICHECK(analyzer.CanProve(
        right_padding >= make_zero(right_padding.dtype())))
        << context << " requires a non-negative right padding count, got "
        << right_padding;
  }
  if (uses_padding) {
    if (!has_const_len) {
      ICHECK(analyzer.CanProve(
          burst_len >= make_zero(burst_len.dtype())))
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
  int64_t n_size = require_const(3, "n_size");
  int64_t m_size = require_const(4, "m_size");
  int64_t dst_stride = require_const(5, "loop_dst_stride");
  int64_t src_stride = require_const(6, "loop_src_stride");
  ICHECK_GT(m_size, 0) << "PTO L0C->UB MTE requires m_size > 0";
  ICHECK_GT(n_size, 0) << "PTO L0C->UB MTE requires n_size > 0";
  ICHECK_GE(dst_stride, n_size)
      << "PTO L0C->UB MTE destination stride must be at least n_size, got "
      << dst_stride << " < " << n_size;
  ICHECK_GE(src_stride, n_size)
      << "PTO L0C->UB MTE source stride must be at least n_size, got "
      << src_stride << " < " << n_size;

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
  PrimExpr dst_index;
  PrimExpr src_index;
  DataType dst_dtype;
  DataType src_dtype;
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO L0C->UB MTE destination", &dst_var,
                      &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO L0C->UB MTE source", &src_var,
                      &src_index, &src_dtype, &src_scope);
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
      dst_index, dst_dtype, make_const(DataType::Int(32), 1),
      make_const(DataType::Int(32), 0), make_const(DataType::Int(32), 0),
      make_const(DataType::Int(32), 0), make_const(DataType::Int(32), 0), false,
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
    const PrimExpr &index, DataType elem_dtype, const std::string &context) {
  PrimExpr elem_index = index;
  if (const auto *ramp = index.as<RampNode>()) {
    CheckContiguousRampStride(index, "local pointer offset");
    elem_index = arith::Analyzer().Simplify(ramp->base);
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
  std::string coerced =
      ScalarCastExpr(index_expr, DataType::Int(64), context);
  if (elem_bytes == 1) {
    return coerced;
  }
  return "scalar.muli(" + coerced + ", pto.const(" +
         std::to_string(elem_bytes) + ", dtype=pto.int64))";
}

std::string CodeGenTileLangPTO::GetLocalPtrExpr(const PrimExpr &expr,
                                                   const std::string &space,
                                                   DataType fallback_dtype) {
  PrimExpr index;
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(expr, &index, &buffer_var))
      << "PTO local pointer expects address_of/tvm_access_ptr, got " << expr;

  DataType elem_dtype = fallback_dtype;
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) && !call->args.empty()) {
      if (auto *type_call = call->args[0].as<CallNode>()) {
        if (!type_call->args.empty()) {
          if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
            elem_dtype = ParseDataType(dtype_name->value);
          }
        }
      }
    } else if (call->op.same_as(builtin::address_of())) {
      if (const auto *load = call->args[0].as<BufferLoadNode>()) {
        elem_dtype = load->buffer->dtype;
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
         scope == "shared.l0c.dyn" ||
         scope == "local.fragment" || scope.empty())
      << "PTO local pointer expected shared/local.fragment storage, got "
      << scope;

  std::string byte_addr =
      GetLocalByteAddrExpr(index, elem_dtype, "PTO local pointer offset");
  return "pto.castptr(" + byte_addr + ", " + PointerTypeName(elem_dtype, space) +
         ")";
}

std::string CodeGenTileLangPTO::GetE8M0ScalePtrExpr(const PrimExpr &expr) {
  PrimExpr index;
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(expr, &index, &buffer_var))
      << "PTO E8M0 scale pointer expects address_of/tvm_access_ptr, got "
      << expr;

  // Scale buffers use uint8 logical storage or an explicit uint16 pair-packed
  // view. Compute the byte address from that physical storage type, then expose
  // the address to the MX instruction as an E8M0 matrix pointer.
  DataType storage_dtype = GetAnnotatedPointerDtype(expr, DataType::UInt(8));
  ICHECK(storage_dtype.is_uint() &&
         (storage_dtype.bits() == 8 || storage_dtype.bits() == 16))
      << "PTO E8M0 scale storage must be uint8 or pair-packed uint16, got "
      << storage_dtype;

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  ICHECK(scope == "shared.l1" || scope == "shared.l1.dyn")
      << "PTO E8M0 scale pointer must refer to an L1 allocation, got " << scope;

  std::string byte_addr = GetLocalByteAddrExpr(index, storage_dtype,
                                               "PTO E8M0 scale pointer offset");
  return "pto.castptr(" + byte_addr + ", pto.ptr(pto.f8e8m0, \"mat\"))";
}

std::string CodeGenTileLangPTO::GetAccPtrExpr(const PrimExpr &expr,
                                                 DataType dtype) {
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
    const SBlockNode *section) {
  auto section_vars = mixed_external_vars_by_section_.find(section);
  ICHECK(section_vars != mixed_external_vars_by_section_.end())
      << "Missing mixed-section variable analysis for " << section->name_hint;

  std::vector<std::pair<std::string, std::string>> restores;
  restores.reserve(section_vars->second.size());
  for (const VarNode *var : section_vars->second) {
    auto snapshot = mixed_entry_snapshot_ids_.find(var);
    ICHECK(snapshot != mixed_entry_snapshot_ids_.end())
        << "Mixed-section local.var is used before its outer allocation is "
           "emitted: "
        << var->name_hint;
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
  DataType input_dtype = ParseDataType(dtype_name->value);
  ICHECK(IsSupportedGemmInputDtype(input_dtype))
      << "PTO GEMM L1 helper only supports float16, bfloat16, float32, "
         "float8_e4m3fn, and float8_e5m2 inputs, got "
      << input_dtype;

  DataType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  DataType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(a_dtype == input_dtype)
      << "PTO GEMM L1 A pointer dtype must match input dtype " << input_dtype
      << ", got " << a_dtype;
  ICHECK(b_dtype == input_dtype)
      << "PTO GEMM L1 B pointer dtype must match input dtype " << input_dtype
      << ", got " << b_dtype;

  DataType accum_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  ICHECK(accum_dtype.is_float() && accum_dtype.bits() == 32)
      << "PTO GEMM L1 helper currently only supports float32 accum/output, got "
      << accum_dtype;

  Call call = GetRef<Call>(op);
  auto call_it = gemm_emit_context_by_call_.find(call);
  if (call_it != gemm_emit_context_by_call_.end()) {
    return gemm_emit_contexts_[call_it->second];
  }
  for (size_t i = 0; i < gemm_emit_contexts_.size(); ++i) {
    const auto &ctx = gemm_emit_contexts_[i];
    ICHECK(!ctx.blockscaled)
        << "PTO kernel mixes tl.ascend_gemm_l1 with "
           "tl.ascend_blockscaled_gemm_l1";
    if (ctx.tile_m == tile_m && ctx.tile_n == tile_n &&
        ctx.tile_k == tile_k && ctx.base_k == base_k &&
        ctx.input_dtype == input_dtype && ctx.accum_dtype == accum_dtype) {
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
  DataType input_dtype = ParseDataType(dtype_name->value);
  ICHECK(IsSupportedBlockscaledGemmInputDtype(input_dtype))
      << "PTO blockscaled GEMM L1 helper currently only supports bfloat16 and "
         "float8_e4m3fn inputs, got "
      << input_dtype;

  DataType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  DataType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(a_dtype == input_dtype)
      << "PTO blockscaled GEMM L1 A pointer dtype must match input dtype "
      << input_dtype << ", got " << a_dtype;
  ICHECK(b_dtype == input_dtype)
      << "PTO blockscaled GEMM L1 B pointer dtype must match input dtype "
      << input_dtype << ", got " << b_dtype;

  DataType accum_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  ICHECK(accum_dtype.is_float() && accum_dtype.bits() == 32)
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
    ICHECK(ctx.blockscaled)
        << "PTO kernel mixes tl.ascend_gemm_l1 with "
           "tl.ascend_blockscaled_gemm_l1";
    if (ctx.tile_m == tile_m && ctx.tile_n == tile_n &&
        ctx.tile_k == tile_k && ctx.base_k == base_k &&
        ctx.sf_nz_stride == sf_nz_stride &&
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
  stream << emitted_ctx.helper_name << " = tl.PTOBlockscaledGemmL1Template(" << tile_m
         << ", " << tile_n << ", " << tile_k << ", " << base_k << ", "
         << sub_k_tiles << ", " << input_c0 << ", " << sub_k_c0_blocks << ", "
         << a_l0_stage_elems << ", " << b_l0_stage_elems << ", sf_nz_stride="
         << sf_nz_stride;
  if (input_pack_factor != 1) {
    stream << ", input_pack_factor=" << input_pack_factor;
  }
  stream << ")\n";
  return emitted_ctx;
}

void CodeGenTileLangPTO::ValidateFractalAddressAlignment_(
    const PrimExpr &index, DataType dtype, const char *context) const {
  const bool is_fp4_packed = IsPackedFp4Pair(dtype);
  ICHECK((dtype.is_scalar() || is_fp4_packed))
      << context << " requires a scalar or packed-FP4 dtype, got " << dtype;
  arith::Analyzer analyzer;
  int64_t elem_bytes = is_fp4_packed ? 1 : dtype.bytes();
  PrimExpr byte_offset = analyzer.Simplify(
      index * make_const(index.dtype(), elem_bytes));
  PrimExpr offset_mod = analyzer.Simplify(
      floormod(byte_offset, make_const(byte_offset.dtype(), 32)));
  if (!analyzer.CanProveEqual(offset_mod, make_zero(offset_mod.dtype()))) {
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
  bool has_const_src_inner_stride = TryGetConstInt(op->args[3], &src_inner_stride);

  int64_t l2_cache_ctrl = require_const(4, "l2_cache_ctrl");

  int64_t n_value = 0;
  bool has_const_n_value = TryGetConstInt(op->args[5], &n_value);

  int64_t d_value = 0;
  bool has_const_d_value = TryGetConstInt(op->args[6], &d_value);

  int64_t src_outer_stride = 0;
  bool has_const_src_outer_stride = TryGetConstInt(op->args[7], &src_outer_stride);

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
      << "PTO GM->L1 MTE dst_n_value exceeds its 16-bit field: "
      << dst_n_value;

  // Packed scale factors are logically stored as 8-bit values but transferred
  // as uint16 pairs. Mirror AscendC's physical pointer casts while retaining
  // the common argument, scope, alignment, and stride validation.
  const auto *physical_dtype_imm = op->args[11].as<StringImmNode>();
  ICHECK(physical_dtype_imm)
      << "PTO GM->L1 MTE requires a constant physical_dtype string";
  bool has_physical = !physical_dtype_imm->value.empty();
  DataType transfer_dtype;
  if (has_physical) {
    if (physical_dtype_imm->value == "int8_t") {
      // Packed FP4: two values per byte, issued as int8 DMA.
      transfer_dtype = DataType::Int(8);
    } else if (physical_dtype_imm->value == "uint16_t") {
      // Packed E8M0 scale factors: two scales per uint16.
      transfer_dtype = DataType::UInt(16);
    } else {
      ICHECK(false) << "PTO GM->L1 MTE supports physical_dtype int8_t (packed "
                       "FP4) or uint16_t (packed scale factors), got "
                    << physical_dtype_imm->value;
    }
  }

  const VarNode *dst_var = nullptr;
  const VarNode *src_var = nullptr;
  PrimExpr dst_index;
  PrimExpr src_index;
  DataType dst_dtype;
  DataType src_dtype;
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO GM->L1 MTE destination", &dst_var,
                      &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO GM->L1 MTE source", &src_var,
                      &src_index, &src_dtype, &src_scope);
  ICHECK(dst_scope == "shared.l1" || dst_scope == "shared.l1.dyn")
      << "PTO GM->L1 MTE destination must use shared.l1 storage, got scope `"
      << dst_scope << "`";
  ICHECK(src_scope.empty() || src_scope == "global")
      << "PTO GM->L1 MTE source must use global storage, got scope `"
      << src_scope << "`";

  if (has_physical) {
    const bool is_fp4 = dst_dtype.is_float4_e2m1fn();
    const bool is_pair_packed_scale = physical_dtype_imm->value == "uint16_t";
    ICHECK((!is_pair_packed_scale && is_fp4) ||
           (is_pair_packed_scale && dst_dtype.is_uint() && dst_dtype.bits() == 8))
        << "PTO GM-to-L1 copy supports int8 physical storage only for FP4 or "
           "uint16 pair-packed storage only for uint8 scale factors, got "
        << physical_dtype_imm->value << " for destination dtype " << dst_dtype;
  } else {
    ICHECK(IsStorageDtypeCompatible(dst_dtype, src_dtype))
        << "PTO GM->L1 MTE does not support dtype conversion: source is "
        << src_dtype << ", destination is " << dst_dtype;
    bool is_uint16_scale_storage =
        dst_dtype == DataType::UInt(16) && src_dtype == DataType::UInt(16);
    ICHECK(IsSupportedCubeMteDtype(dst_dtype) || is_uint16_scale_storage)
        << "PTO GM->L1 MTE supports int8, float16, bfloat16, float32, and "
           "float8_e4m3fn/float8_e5m2 matrix data, plus uint16 scale-factor "
           "storage; got "
        << dst_dtype;
    transfer_dtype = dst_dtype;
  }

  ValidateFractalAddressAlignment_(dst_index, transfer_dtype,
                                      "PTO GM->L1 MTE destination");
  bool has_const_source_row_elems =
      transpose == 0 ? has_const_d_value : has_const_n_value;
  int64_t source_row_elems = transpose == 0 ? d_value : n_value;
  ICHECK(!has_const_src_inner_stride || !has_const_source_row_elems ||
         src_inner_stride >= source_row_elems * transfer_dtype.bytes())
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
  if (!is_zero(op->args[7])) {
    stream << ", " << PrintExpr_(op->args[7]);
  }
  stream << ",), dst_group=(1, 1, " << PrintExpr_(op->args[10])
         << ", 0), ctrl=(" << l2_cache_ctrl << ", "
         << (smallc0_en == 0 ? "False" : "True") << "))\n";
}

void CodeGenTileLangPTO::EmitAscendFillL1(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 7U) << "tl.ascend_fill_l1 expects exactly 7 arguments";

  const VarNode *dst_var = nullptr;
  PrimExpr dst_index;
  DataType dst_dtype;
  std::string dst_scope;
  GetCopyEndpoint_(op->args[0], "PTO L1 fill destination", &dst_var, &dst_index, &dst_dtype, &dst_scope);
  ICHECK(dst_scope == "shared.l1" || dst_scope == "shared.l1.dyn")
      << "PTO L1 fill destination must use shared.l1 storage, got scope `"
      << dst_scope << "`";

  int64_t fill_word_bits = 0;
  ICHECK(TryGetConstInt(op->args[6], &fill_word_bits) && (fill_word_bits == 16 || fill_word_bits == 32))
      << "PTO L1 fill requires fill_word_bits to be constant 16 or 32";

  PrintIndent();
  stream << "pto.raw_fill_l1("
         << GetLocalPtrExpr(op->args[0], "mat", dst_dtype)
         << ", byte_offset=" << PrintExpr_(op->args[1])
         << ", raw_value=" << PrintExpr_(op->args[2])
         << ", repeat_times=" << PrintExpr_(op->args[3])
         << ", block_num_32b=" << PrintExpr_(op->args[4])
         << ", dst_gap_32b=" << PrintExpr_(op->args[5])
         << ", fill_word_bits=" << fill_word_bits << ")\n";
}

void CodeGenTileLangPTO::EmitAscendLoadCbufToL0(const CallNode *op,
                                                 bool is_ca) {
  const char *name = is_ca ? "tl.ascend_load_cbuf_to_ca"
                           : "tl.ascend_load_cbuf_to_cb";
  ICHECK(op->args.size() == 9U || op->args.size() == 16U)
      << name << " expects 9 or 16 arguments, got " << op->args.size();
  bool has_mx_scale = op->args.size() == 16U;

  auto require_const = [&](size_t index, const char *arg_name) {
    int64_t value = 0;
    ICHECK(TryGetConstInt(op->args[index], &value))
        << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B")
        << " MTE requires constant " << arg_name << ", got "
        << op->args[index];
    return value;
  };
  auto control_expr = [&](size_t index, const char *arg_name) {
    int64_t value = 0;
    if (TryGetConstInt(op->args[index], &value)) {
      ICHECK_GT(value, 0)
          << "PTO " << (is_ca ? "L1->L0A" : "L1->L0B")
          << " MTE requires positive " << arg_name << ", got " << value;
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
  std::string row_start =
      RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  std::string col_start =
      RemoveOutermostParentheses(PrintExpr_(op->args[3]));
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
  PrimExpr dst_index;
  PrimExpr src_index;
  DataType dst_dtype;
  DataType src_dtype;
  std::string dst_scope;
  std::string src_scope;
  GetCopyEndpoint_(op->args[0], "PTO L1->L0 MTE destination", &dst_var,
                      &dst_index, &dst_dtype, &dst_scope);
  GetCopyEndpoint_(op->args[1], "PTO L1->L0 MTE source", &src_var,
                      &src_index, &src_dtype, &src_scope);
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
  std::string dst = GetLocalPtrExpr(op->args[0], is_ca ? "left" : "right",
                                       dst_dtype);
  // Preserve the controls produced by TileLang's canonical fractal-layout
  // lowering.  Reconstructing a shape and asking PTOAS to derive the controls
  // is not equivalent for every dtype: float32 uses an 8-element physical C0,
  // while the transpose shape helper aligns its axes to at least 16 elements.
  // Explicit controls also keep this lowering identical to the AscendC path.
  PrintIndent();
  stream << (is_ca ? "pto.mte_l1_l0a(" : "pto.mte_l1_l0b(") << src << ", "
         << dst << ", m_start=" << row_start << ", k_start=" << col_start
         << ", m_step=" << row_step << ", k_step=" << col_step
         << ", src_stride=" << src_stride
         << ", dst_stride=" << dst_stride
         << ", transpose=" << (transpose == 0 ? "False" : "True") << ")\n";

  if (!has_mx_scale) {
    return;
  }

  const VarNode *sf_var = nullptr;
  PrimExpr sf_index;
  DataType sf_dtype;
  std::string sf_scope;
  GetCopyEndpoint_(op->args[9], "PTO MX scale-factor L1 source", &sf_var,
                      &sf_index, &sf_dtype, &sf_scope);
  ICHECK(sf_scope == "shared.l1" || sf_scope == "shared.l1.dyn")
      << "PTO MX scale-factor source must use shared.l1 storage, got `"
      << sf_scope << "`";
  ICHECK(sf_dtype.is_scalar() && sf_dtype.is_uint() &&
         (sf_dtype.bits() == 8 || sf_dtype.bits() == 16))
      << "PTO MX scale-factor source supports uint8 or pair-packed uint16 "
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

void CodeGenTileLangPTO::EmitAscendCrossCoreFlag(const CallNode *op,
                                                  bool is_set) {
  ICHECK_EQ(op->args.size(), 3U)
      << "tl.ascend_cross_core_" << (is_set ? "set" : "wait")
      << "_flag expects exactly 3 arguments (mode_id, pipe, flag_id)";
  int64_t mode_id = 0;
  ICHECK(TryGetConstInt(op->args[0], &mode_id))
      << "PTO cross-core flag requires a constant mode_id";
  ICHECK(mode_id == 0 || mode_id == 1 || mode_id == 2 || mode_id == 4)
      << "PTO cross-core flag mode_id must be one of 0/1/2/4, got "
      << mode_id;

  const auto *pipe_imm = op->args[1].as<StringImmNode>();
  ICHECK(pipe_imm) << "PTO cross-core flag requires a constant pipe string";
  DataType event_dtype = op->args[2].dtype();
  ICHECK(event_dtype.is_scalar() &&
         (event_dtype.is_int() || event_dtype.is_uint()))
      << "PTO cross-core flag_id must be a scalar integer, got "
      << event_dtype;

  // mode_id=0: inter-core FFTS sync -> pto.set/wait_cross_block (asc form).
  // mode_id=4: AIC<->AIV intra-block sync -> pto.set/wait_intra_block (asc).
  if (mode_id == 0 || mode_id == 4) {
    std::string printed_event_id =
        RemoveOutermostParentheses(PrintExpr_(op->args[2]));
    int64_t event_value = 0;
    std::string event_id =
        TryGetConstInt(op->args[2], &event_value)
            ? std::to_string(event_value)
            : "scalar.index_cast(" + printed_event_id + ")";
    const bool cross = mode_id == 0;
    std::string op_name = is_set ? (cross ? "set_cross_block" : "set_intra_block")
                                 : (cross ? "wait_cross_block"
                                          : "wait_intra_block");
    PrintIndent();
    stream << "pto." << op_name << "(\""
           << StripPipePrefix(pipe_imm->value) << "\", " << event_id << ")\n";
    return;
  }

  // Modes 1/2 stay on the PTODSL compatibility helper, which maps them to
  // PTOAS pto.sync operations with the authored mode/pipe/event triple.
  std::string event_id =
      RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  PrintIndent();
  stream << "tl.ascend_cross_core_" << (is_set ? "set" : "wait")
         << "_flag(" << mode_id << ", \"" << pipe_imm->value << "\", "
         << event_id << ")\n";
}

void CodeGenTileLangPTO::EmitGemmRun(const PTOGemmEmitContext &ctx,
                                        const std::string &a_mat,
                                        const std::string &b_mat,
                                        const std::string &acc,
                                        const std::string &clear_accum,
                                        const std::string &unit_flag_ctrl,
                                        int64_t hf32_mode) {
  PrintIndent();
  stream << ctx.helper_name << ".run_l1_tile(" << a_mat << ", " << b_mat
         << ", " << ctx.a_l0_name << ", " << ctx.b_l0_name << ", " << acc
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
    const std::string &b_mat,
    const std::string &sfa_mat, const std::string &sfb_mat,
    const std::string &acc, const std::string &clear_accum,
    const std::string &sf_k_offset, const std::string &unit_flag_ctrl) {
  PrintIndent();
  stream << ctx.helper_name << ".run_l1_tile(" << a_mat << ", " << b_mat << ", "
         << sfa_mat << ", " << sfb_mat << ", " << ctx.a_l0_name << ", "
         << ctx.b_l0_name << ", " << acc << ", sf_k_offset=" << sf_k_offset
         << ", clear_accum=" << clear_accum
         << ", unit_flag_ctrl=" << unit_flag_ctrl << ")\n";
}

void CodeGenTileLangPTO::EmitAscendGemmL1(const CallNode *op) {
  const PTOGemmEmitContext &ctx = EnsureGemmHelper(op);
  std::string acc = GetAccPtrExpr(op->args[0], DataType::Float(32));
  std::string a_mat =
      GetLocalPtrExpr(op->args[1], "mat", ctx.input_dtype);
  std::string b_mat =
      GetLocalPtrExpr(op->args[2], "mat", ctx.input_dtype);

  int64_t hf32_mode = 0;
  if (IsFloat32(ctx.input_dtype)) {
    auto it = hf32_mode_by_gemm_.find(GetRef<Call>(op));
    ICHECK(it != hf32_mode_by_gemm_.end())
        << "PTO codegen did not analyze HF32 mode for FP32 GEMM";
    hf32_mode = it->second;
  }
  EmitGemmRun(
      ctx, a_mat, b_mat, acc,
      RemoveOutermostParentheses(PrintExpr_(op->args[8])),
      RemoveOutermostParentheses(PrintExpr_(op->args[11])), hf32_mode);
}

void CodeGenTileLangPTO::EmitAscendBlockscaledGemmL1(const CallNode *op) {
  const PTOGemmEmitContext &ctx = EnsureBlockscaledGemmHelper(op);
  std::string acc = GetAccPtrExpr(op->args[0], DataType::Float(32));
  std::string a_mat =
      GetLocalPtrExpr(op->args[1], "mat", ctx.input_dtype);
  std::string b_mat =
      GetLocalPtrExpr(op->args[2], "mat", ctx.input_dtype);

  std::string sfa_mat = GetE8M0ScalePtrExpr(op->args[3]);
  std::string sfb_mat = GetE8M0ScalePtrExpr(op->args[4]);

  EmitBlockscaledGemmRun(
      ctx, a_mat, b_mat, sfa_mat, sfb_mat, acc,
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
    PrimExpr index;
    DataType dtype;
    std::string scope;
    GetCopyEndpoint_(op->args[arg_index], context, &buffer_var, &index,
                        &dtype, &scope);
    return std::make_pair(dtype, scope);
  };

  auto acc_info = get_endpoint(0, "PTO MAD accumulator");
  auto lhs_info = get_endpoint(1, "PTO MAD left operand");
  auto rhs_info = get_endpoint(2, "PTO MAD right operand");
  DataType acc_dtype = acc_info.first;
  DataType lhs_dtype = lhs_info.first;
  DataType rhs_dtype = rhs_info.first;

  ICHECK(IsBufferInScope(lhs_info.second, "shared.l0a"))
      << "PTO " << op_name
      << " left operand must use shared.l0a storage, got `" << lhs_info.second
      << "`";
  ICHECK(IsBufferInScope(rhs_info.second, "shared.l0b"))
      << "PTO " << op_name
      << " right operand must use shared.l0b storage, got `" << rhs_info.second
      << "`";
  ICHECK(IsBufferInScope(acc_info.second, "shared.l0c"))
      << "PTO " << op_name
      << " accumulator must use shared.l0c storage, got `" << acc_info.second
      << "`";

  if (is_mx) {
    ICHECK(IsSupportedMadMxDtypeTuple(lhs_dtype, rhs_dtype, acc_dtype))
        << "PTO mad_mx supports float8_e4m3fn/float8_e5m2 left/right "
           "operands with a float32 accumulator, got left="
        << lhs_dtype << ", right=" << rhs_dtype << ", accumulator="
        << acc_dtype;
  } else {
    ICHECK(IsSupportedMadDtypeTuple(lhs_dtype, rhs_dtype, acc_dtype))
        << "PTO mad supports matching float16, bfloat16, float32, or "
           "float8_e4m3fn/float8_e5m2 (including mixed FP8) left/right "
           "operands with a float32 accumulator, or matching int8 operands "
           "with an int32 accumulator; got left="
        << lhs_dtype << ", right=" << rhs_dtype << ", accumulator="
        << acc_dtype;
  }

  std::string acc = GetAccPtrExpr(op->args[0], acc_dtype);
  std::string lhs = GetLocalPtrExpr(op->args[1], "left", lhs_dtype);
  std::string rhs = GetLocalPtrExpr(op->args[2], "right", rhs_dtype);

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
      << "PTO " << op_name
      << " requires constant gemv_ctrl 0 or 1, got " << op->args[7];

  int64_t btbuf_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[8], &btbuf_ctrl) && btbuf_ctrl == 0)
      << "PTO " << op_name
      << " currently only supports BTbuf_ctrl=0, got " << op->args[8];

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
    stream << mad_fn << "(" << lhs << ", " << rhs << ", " << acc << ", "
           << m << ", " << n << ", " << k;
    if (gemv_ctrl == 1) {
      stream << ", disable_gemv=True";
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

    PrimExpr condition;
    PrimExpr true_value;
    PrimExpr false_value;
    if (const auto *select = unit_flag_expr.as<SelectNode>()) {
      condition = select->condition;
      true_value = select->true_value;
      false_value = select->false_value;
    } else if (const auto *call = unit_flag_expr.as<CallNode>();
               call != nullptr &&
               (call->op.same_as(builtin::if_then_else()) ||
                call->op.same_as(tirx::builtin::if_then_else())) &&
               call->args.size() == 3U) {
      condition = call->args[0];
      true_value = call->args[1];
      false_value = call->args[2];
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

  arith::Analyzer analyzer;
  emit_for_unit_flag(emit_for_unit_flag, analyzer.Simplify(op->args[6]));
}

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToGm(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 25U)
      << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = GetAccPtrExpr(op->args[1], DataType::Float(32));
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

  std::string dst =
      GetLocalPtrExpr(op->args[0], "mat", DataType::BFloat(16));
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

  std::string dst = GetLocalPtrExpr(op->args[0], "mat", DataType::BFloat(16));
  std::string src = GetLocalPtrExpr(op->args[1], "ub", DataType::BFloat(16));

  int64_t rows = ConstArgDim(op, 2, "rows");
  int64_t cols = ConstArgDim(op, 3, "cols");
  int64_t full_rows = ConstArgDim(op, 4, "full_rows");
  std::string dst_dtype = Downcast<StringImm>(op->args[5])->value;

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
  std::string dst_dtype = Downcast<StringImm>(op->args[4])->value;
  std::string src_dtype = Downcast<StringImm>(op->args[5])->value;

  ICHECK(rows % 16 == 0) << "PTO nd2nz_scatter requires ROWS % 16 == 0, got "
                         << rows;

  std::string src_ptr = GetLocalPtrExpr(op->args[0], "ub", DataType::BFloat(16));
  std::string dst_ptr = GetLocalPtrExpr(op->args[1], "ub", DataType::BFloat(16));

  // Currently support same-type scatter only (the common case for dual_copy,
  // which rejects dtype mismatch on L0C->UB at the frontend level).
  ICHECK(src_dtype == dst_dtype)
      << "PTO nd2nz_scatter currently only supports same-type scatter; got "
      << "src=" << src_dtype << " dst=" << dst_dtype;

  int elem_bytes = (src_dtype == "float") ? 4 : 2;
  int vl_elems = 256 / elem_bytes;       // 64 for float, 128 for half/bfloat16
  int elems_per_block = 32 / elem_bytes;  // 8 for float, 16 for half/bfloat16

  // 32-byte alignment (one C0 block) is the only hard requirement.
  // Partial-VL passes are handled via PAT_VL{N} masks.
  ICHECK((cols * elem_bytes) % 32 == 0)
      << "PTO nd2nz_scatter requires COLS * sizeof(elem) to be 32-byte "
      << "aligned, got COLS=" << cols << " elem_bytes=" << elem_bytes;

  int num_passes =
      static_cast<int>((cols + vl_elems - 1) / vl_elems);  // ceil division
  for (int pass = 0; pass < num_passes; ++pass) {
    int remaining = static_cast<int>(cols - pass * vl_elems);
    int elems_this_pass = std::min(remaining, vl_elems);

    // Determine mask: PAT_ALL for full-VL passes, PAT_VL{N} for partial.
    std::string mask_fn;
    if (elems_this_pass >= vl_elems) {
      mask_fn = (elem_bytes == 4) ? "pto.pset_b32(\"PAT_ALL\")"
                                  : "pto.pset_b16(\"PAT_ALL\")";
    } else {
      mask_fn =
          (elem_bytes == 4)
              ? "pto.pset_b32(\"PAT_VL" + std::to_string(elems_this_pass) + "\")"
              : "pto.pset_b16(\"PAT_VL" + std::to_string(elems_this_pass) +
                    "\")";
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

std::string
CodeGenTileLangPTO::EmitAllReduceExpr_(const std::string &func_name,
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

void CodeGenTileLangPTO::VisitExpr_(const CallNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->op.same_as(builtin::if_then_else()) ||
      op->op.same_as(tirx::builtin::if_then_else())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "if_then_else expects <condition, true_value, false_value>";
    const bool requires_statement_control_flow =
        RequiresStatementControlFlow(op->args[1]) ||
        RequiresStatementControlFlow(op->args[2]);
    auto branch_value = [&](const PrimExpr &expr) {
      std::string value = PrintExpr_(expr);
      if (expr.dtype().is_bool()) {
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
               ", dtype=" + DataTypeName(op->dtype) + ")";
      }
      if (op->dtype.is_bool()) {
        return "tl.as_logical_bool(" + value + ")";
      }
      if (op->dtype.is_scalar() && (op->dtype.is_int() || op->dtype.is_uint())) {
        return ScalarCastExpr(value, op->dtype, "PTO if_then_else branch");
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

  if (op->op.same_as(builtin::bitwise_and())) {
    PrintBinaryExpr_("&", op->dtype, op->args[0], op->args[1], os);
    return;
  }

  if (op->op.same_as(builtin::bitwise_or())) {
    PrintBinaryExpr_("|", op->dtype, op->args[0], op->args[1], os);
    return;
  }

  if (op->op.same_as(builtin::bitwise_xor())) {
    PrintBinaryExpr_("^", op->dtype, op->args[0], op->args[1], os);
    return;
  }

  if (op->op.same_as(builtin::shift_left())) {
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative shift_left";
    PrintBinaryExpr_("*", op->dtype, op->args[0],
                     IntImm(op->args[0].dtype(), 1LL << shift), os);
    return;
  }

  if (op->op.same_as(builtin::shift_right())) {
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative shift_right";
    DataType dtype = op->args[0].dtype();
    ICHECK(dtype.is_scalar() && (dtype.is_int() || dtype.is_uint()))
        << "PTO shift_right expects a scalar integer operand, got " << dtype;
    ICHECK_LT(shift, dtype.bits())
        << "PTO shift_right amount " << shift
        << " must be smaller than operand bit width " << dtype.bits();

    if (dtype.is_uint()) {
      DataType signless_dtype = DataType::Int(dtype.bits());
      os << "tl.ushr(" << PrintExpr_(op->args[0])
         << ", pto.const(" << shift << ", dtype="
         << ScalarType(signless_dtype) << "), "
         << ScalarType(signless_dtype) << ", " << DataTypeName(dtype)
         << ", context=\"PTO unsigned shift_right\")";
      return;
    }

    // LowerIntrin uses signed x >> (bits - 1) to extract the sign mask for
    // floordiv/floormod correction.  Materializing 1 << (bits - 1) in the
    // original signed dtype is invalid (for example, 2^31 is not int32).
    // Emit the exact arithmetic-shift result without an out-of-range literal.
    if (dtype.is_int() && shift == dtype.bits() - 1) {
      PrimExpr non_negative = op->args[0] >= make_const(dtype, 0);
      os << "scalar.select(" << PrintCondition(non_negative)
         << ", pto.const(0, dtype=" << DataTypeName(dtype)
         << "), pto.const(-1, dtype=" << DataTypeName(dtype) << "))";
      return;
    }

    PrintBinaryExpr_("//", op->dtype, op->args[0],
                     IntImm(dtype, 1LL << shift), os);
    return;
  }

  if (op->op.same_as(builtin_call_extern_) ||
      op->op.same_as(builtin_call_pure_extern_)) {
    ICHECK_GE(op->args.size(), 1U);
    // TODO: Support more scalar ops, ref: src/ascend/codegen/intrin_rule_ascend.cc
    std::string func_name = Downcast<StringImm>(op->args[0])->value;
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
    const char *name = op->op.same_as(tl::warp_reduce_sum())   ? "redux_add"
                       : op->op.same_as(tl::warp_reduce_max()) ? "redux_max"
                                                               : "redux_min";
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ")";
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

    DataType dtype = GetAnnotatedPointerDtype(op->args[0], op->args[1].dtype());
    ICHECK(dtype.is_scalar())
        << "PTO atomic operations support scalar values only";
    // PTOAS supports 64-bit integer atomics for GM pointers.  Keep UB-space
    // legality in PTOAS's address-space-aware verifier instead of rejecting
    // all i64/u64 atomics here before PTODSL lowering.
    ICHECK((dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
           ((dtype.is_int() || dtype.is_uint()) &&
            (dtype.bits() == 32 || dtype.bits() == 64)))
        << "PTO atomic operations support float16, float32, int32, uint32, "
           "int64, and uint64, got "
        << dtype;

    const char *pto_atomic_op =
        is_atomic_add ? "atomic_add"
                      : (is_atomic_max ? "atomic_max" : "atomic_min");
    os << "pto." << pto_atomic_op << "(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]);
    if (dtype.is_int() || dtype.is_uint()) {
      os << ", signedness=\"" << (dtype.is_uint() ? "unsigned" : "signed")
         << "\"";
    }
    os << ")";
    return;
  }

  if (op->op.same_as(builtin::tvm_storage_sync())) {
    ICHECK_GE(op->args.size(), 1U)
        << "tvm_storage_sync expects at least the storage scope argument";
    std::string sync_scope = Downcast<StringImm>(op->args[0])->value;
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

  if (op->op.same_as(tl::ascend_cross_core_set_flag())) {
    EmitAscendCrossCoreFlag(op, true);
    return;
  }

  if (op->op.same_as(tl::ascend_cross_core_wait_flag())) {
    EmitAscendCrossCoreFlag(op, false);
    return;
  }

  if (op->op.same_as(tl::ascend_mad()) ||
      op->op.same_as(tl::ascend_mad_mx())) {
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
  // BufferLoad/Store into these Calls. Map to PTODSL cache-bypass helpers, which pass to pto.load_scalar/store_scalar.
  if (op->op.same_as(tl::ascend_read_gm_bypass_dcache())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.ascend_read_gm_bypass_dcache expects address_of(BufferLoad)";
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && addr->op.same_as(builtin::address_of()))
        << "tl.ascend_read_gm_bypass_dcache expects address_of(BufferLoad)";
    const auto *load = addr->args[0].as<BufferLoadNode>();
    ICHECK(load && load->indices.size() == 1U)
        << "tl.ascend_read_gm_bypass_dcache expects a flat BufferLoad";
    ICHECK(load->dtype.is_scalar())
        << "PTO GM dcache bypass only supports scalar loads, got "
        << load->dtype;
    std::string scope = ScopeOfBuffer(load->buffer.get());
    ICHECK(scope == "global" || scope.empty())
        << "PTO GM dcache bypass expects a global buffer, got scope " << scope;
    DataType value_dtype = load->dtype;
    ValidateGmBypassDtype(value_dtype);
    std::string base =
        ScalarPointerBase_(load->buffer->data.get(), value_dtype, scope);
    std::string index =
        RemoveOutermostParentheses(PrintExpr_(load->indices[0]));
    os << "tl.read_gm_bypass_dcache(" << base << ", " << index << ", "
       << ScalarType(value_dtype) << ")";
    return;
  }

  if (op->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ascend_write_gm_bypass_dcache expects (address_of(BufferLoad), "
           "value)";
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && addr->op.same_as(builtin::address_of()))
        << "tl.ascend_write_gm_bypass_dcache expects address_of(BufferLoad)";
    const auto *load = addr->args[0].as<BufferLoadNode>();
    ICHECK(load && load->indices.size() == 1U)
        << "tl.ascend_write_gm_bypass_dcache expects a flat BufferLoad";
    ICHECK(load->buffer->dtype.is_scalar())
        << "PTO GM dcache bypass only supports scalar stores, got "
        << load->buffer->dtype;
    std::string scope = ScopeOfBuffer(load->buffer.get());
    ICHECK(scope == "global" || scope.empty())
        << "PTO GM dcache bypass expects a global buffer, got scope " << scope;
    DataType value_dtype = load->buffer->dtype;
    ValidateGmBypassDtype(value_dtype);
    // Print pointer/index/value first so any SSA side effects land before the
    // helper call.
    std::string base =
        ScalarPointerBase_(load->buffer->data.get(), value_dtype, scope);
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

  if (op->op.same_as(builtin::address_of())) {
    os << GetAddressOfExpr_(op);
    return;
  }

  if (op->op.same_as(tirx::builtin::reinterpret())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tirx.reinterpret expects 1 argument (value)";
    std::string value = PrintExpr_(op->args[0]);
    DataType target_dtype = op->dtype;
    if (target_dtype.lanes() > 1) {
      // Vector reinterpret: lower to pto.vbitcast(vec, elem_type).
      os << "pto.vbitcast(" << value << ", "
         << ScalarType(target_dtype.element_of()) << ")";
    } else {
      DataType source_dtype = op->args[0].dtype();
      ICHECK(source_dtype.is_scalar())
          << "PTO scalar reinterpret expects scalar source, got " << source_dtype;
      ICHECK_EQ(source_dtype.bits(), target_dtype.bits())
          << "PTO scalar reinterpret requires equal bit widths, got source "
          << source_dtype << " and target " << target_dtype;
      auto signless_integer_dtype = [](DataType dtype) {
        ICHECK(dtype.is_int() || dtype.is_uint())
            << "PTO scalar bitcast signless dtype expects integer, got " << dtype;
        return DataType::Int(dtype.bits());
      };
      DataType bitcast_dtype = target_dtype;
      std::string final_dtype;
      if (target_dtype.is_int() || target_dtype.is_uint()) {
        bitcast_dtype = signless_integer_dtype(target_dtype);
        if (target_dtype.is_uint()) {
          final_dtype = ScalarType(target_dtype);
        }
      }
      std::string source_dtype_arg;
      if (source_dtype.is_int() || source_dtype.is_uint()) {
        source_dtype_arg = ScalarType(signless_integer_dtype(source_dtype));
      }
      os << "tl.scalar_bitcast(" << value << ", " << ScalarType(bitcast_dtype);
      if (!source_dtype_arg.empty()) {
        os << ", source_dtype=" << source_dtype_arg;
      }
      if (!final_dtype.empty()) {
        os << ", final_dtype=" << final_dtype;
      }
      os << ")";
    }
    return;
  }

  if (op->op.same_as(builtin::tvm_access_ptr())) {
    os << GetAccessPtrExpr_(op);
    return;
  }

  if (op->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *load = op->args[0].as<BufferLoadNode>();
    ICHECK(load) << "tl.access_ptr arg0 must be BufferLoad";
    os << GetPointerExpr(load->buffer.get(), load->indices[0]);
    return;
  }

  if (op->op.same_as(tl::ascend_pipe_barrier())) {
    auto pipe_name = Downcast<StringImm>(op->args[0])->value;
    PrintIndent();
    stream << "pto.pipe_barrier(\"" << StripPipePrefix(pipe_name) << "\")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_set_flag()) ||
      op->op.same_as(tl::ascend_wait_flag())) {
    auto hard_event = Downcast<StringImm>(op->args[0])->value;
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
    std::string pipe_str = Downcast<StringImm>(op->args[0])->value;
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
      dist = Downcast<StringImm>(op->args[1])->value;
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
    if (GetAnnotatedPointerDtype(op->args[0], DataType::UInt(32)) !=
        DataType::UInt(32)) {
      addr = "pto.castptr(" + addr + ", " +
             PointerTypeName(DataType::UInt(32), "ub") + ")";
    }
    os << "pto.plds(" << addr << ", pto.const(0), dist=\""
       << dist_imm->value << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_pst())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.pst expects 3 arguments (addr, src, dist)";
    const auto *dist_imm = op->args[2].as<StringImmNode>();
    ICHECK(dist_imm)
        << "tl.simd.pst distribution must be a constant string for PTO "
           "codegen";
    static const std::unordered_set<std::string> kSupportedPstDist = {
        "NORM", "PK"};
    ICHECK(kSupportedPstDist.count(dist_imm->value) != 0U)
        << "PTO tl.simd.pst does not support distribution " << dist_imm->value
        << "; expected NORM or PK";
    std::string addr = PrintExpr_(op->args[0]);
    if (GetAnnotatedPointerDtype(op->args[0], DataType::UInt(32)) !=
        DataType::UInt(32)) {
      addr = "pto.castptr(" + addr + ", " +
             PointerTypeName(DataType::UInt(32), "ub") + ")";
    }
    PrintIndent();
    stream << "pto.psts(" << PrintExpr_(op->args[1]) << ", " << addr
           << ", pto.const(0), dist=\"" << dist_imm->value << "\")\n";
    return;
  }

  if (op->op.same_as(tl::simd_vld())) {
    ICHECK(op->args.size() >= 2U && op->args.size() <= 3U)
        << "tl.simd.vld expects 2 or 3 arguments (addr, dist[, offset])";
    DataType elem_dtype = op->dtype.element_of();
    ICHECK_GT(op->dtype.lanes(), 1)
        << "tl.simd.vld should return a vector type, got " << op->dtype;
    std::string dist = Downcast<StringImm>(op->args[1])->value;
    // AscendC-style UNPK4_B* aliases → PTODSL UNPK4 token.
    if (dist.rfind("UNPK4_B", 0) == 0) {
      dist = "UNPK4";
    }
    std::string offset =
        op->args.size() == 3U
            ? RemoveOutermostParentheses(PrintExpr_(op->args[2]))
            : "pto.const(0)";
    int pto_lanes = op->dtype.lanes();
    if (elem_dtype.bits() < 8) {
      ICHECK_EQ((pto_lanes * elem_dtype.bits()) % 8, 0)
          << "PTO vld sub-byte vector lanes must form whole bytes, got "
          << op->dtype;
      pto_lanes = (pto_lanes * elem_dtype.bits()) / 8;
    }
    os << "pto.vlds(" << PrintExpr_(op->args[0]) << ", " << offset
       << ", pto.vreg_type(" << pto_lanes << ", "
       << ScalarType(elem_dtype) << ")";
    if (!dist.empty() && dist != "NORM") {
      os << ", dist=\"" << dist << "\"";
    }
    os << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vabsdif())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vabsdif expects 4 arguments (src0, src1, mask, mode)";
    DataType elem_dtype = op->dtype.element_of();
    ICHECK(elem_dtype.is_float() &&
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
    if (op->op.same_as(tl::simd_vdiv()) && !enable_fast_math_ &&
        op->dtype.element_of().is_float() &&
        op->dtype.element_of().bits() == 32) {
      ValidateVecMode_(op, 3);
      os << "tl.vdiv_precise_f32(" << PrintExpr_(op->args[0]) << ", "
         << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
      return;
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
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd unary op expects 3 arguments (src, mask, mode)";
    ValidateVecMode_(op, 2);
    const char *name = op->op.same_as(tl::simd_vabs())    ? "vabs"
                       : op->op.same_as(tl::simd_vexp())  ? "vexp"
                       : op->op.same_as(tl::simd_vln())   ? "vln"
                       : op->op.same_as(tl::simd_vsqrt()) ? "vsqrt"
                       : op->op.same_as(tl::simd_vneg())  ? "vneg"
                       : op->op.same_as(tl::simd_vrelu()) ? "vrelu"
                                                          : "vnot";
    DataType input_elem = op->args[0].dtype().element_of();
    int input_bits = input_elem.bits();
    if (input_bits < 8) {
      input_bits = 8;
    }
    std::string mask = "pto.pbitcast(" +
                       RemoveOutermostParentheses(PrintExpr_(op->args[1])) +
                       ", pto.mask_type(\"b" + std::to_string(input_bits) +
                       "\"))";
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
    os << "pto.make_mask(" << DataTypeName(DataType::UInt(width)) << ", "
       << PrintExpr_(op->args[0]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_ppack()) ||
      op->op.same_as(tl::simd_punpack())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd predicate pack/unpack expects 2 arguments (src, part)";
    const auto *part_imm = op->args[1].as<StringImmNode>();
    ICHECK(part_imm)
        << "tl.simd predicate pack/unpack part must be a constant string for "
           "PTO codegen";
    ICHECK(part_imm->value == "LOWER" || part_imm->value == "HIGHER")
        << "PTO predicate pack/unpack part must be LOWER or HIGHER, got "
        << part_imm->value;
    const char *name =
        op->op.same_as(tl::simd_ppack()) ? "ppack" : "punpack";
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ", \""
       << part_imm->value << "\")";
    return;
  }

  // Carry and predicate-pair operations are emitted at their T.bind site so
  // each multi-result instruction is evaluated exactly once.
  if (op->op.same_as(tl::simd_vaddc()) ||
      op->op.same_as(tl::simd_vsubc()) ||
      op->op.same_as(tl::simd_vaddcs()) ||
      op->op.same_as(tl::simd_vsubcs()) ||
      op->op.same_as(tl::simd_vmull()) ||
      op->op.same_as(tl::simd_pintlv()) ||
      op->op.same_as(tl::simd_pdintlv())) {
    LOG(FATAL) << op->op
               << " must be bound via T.bind for PTO codegen before its pair "
                  "elements are used";
    return;
  }

  // Leaky/parametric ReLU.
  if (op->op.same_as(tl::simd_vlrelu()) ||
      op->op.same_as(tl::simd_vprelu())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd vlrelu/vprelu expects 3 arguments";
    if (op->op.same_as(tl::simd_vlrelu())) {
      DataType vector_dtype = op->args[0].dtype();
      std::string alpha = WrapTypedConst(
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[1]))),
          vector_dtype, IsImmediateScalar(op->args[1]));
      os << "pto.vlrelu(" << PrintExpr_(op->args[0]) << ", " << alpha << ", "
         << PrintExpr_(op->args[2]) << ")";
    } else {
      os << "pto.vprelu(" << PrintExpr_(op->args[0]) << ", "
         << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
         << ")";
    }
    return;
  }

  // Cumulative histogram updates a mutable vector destination.
  if (op->op.same_as(tl::simd_chistv2())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.chistv2 expects 4 arguments (dst, src, mask, bin)";
    int64_t bin = 0;
    ICHECK(TryGetConstInt(op->args[3], &bin))
        << "tl.simd.chistv2 bin must be a constant integer";
    ICHECK(bin == 0 || bin == 1)
        << "tl.simd.chistv2 only supports bin 0 or 1, got " << bin;
    std::string dst_ref = GetMutableVectorRef(op->args[0], "chistv2");
    PrintIndent();
    stream << dst_ref << " = pto.chistv2(" << dst_ref << ", "
           << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
           << ", pto.const(" << bin << ", dtype=pto.int32))\n";
    return;
  }

  // --- SIMD vector-scalar ops: (src, scalar, mask, mode) → pto.<op>(src,
  // scalar, mask) ---
  if (op->op.same_as(tl::simd_vadds()) || op->op.same_as(tl::simd_vmuls()) ||
      op->op.same_as(tl::simd_vmaxs()) || op->op.same_as(tl::simd_vmins()) ||
      op->op.same_as(tl::simd_vshls()) || op->op.same_as(tl::simd_vshrs())) {
    // LegalizeSimdMerging rewrites
    //   dst = op(src, scalar, mask, MODE_MERGING)
    // to a void call carrying dst as an explicit read-write first argument.
    // Keep the ordinary four-argument expression form for zeroing operations.
    const bool has_explicit_dst = op->args.size() == 5U;
    ICHECK(op->args.size() == 4U || has_explicit_dst)
        << "tl.simd vector-scalar op expects (src, scalar, mask, mode) or "
           "(dst, src, scalar, mask, mode), got "
        << op->args.size() << " arguments";
    const size_t arg_base = has_explicit_dst ? 1U : 0U;
    auto mode = Downcast<StringImm>(op->args[arg_base + 3])->value;
    ICHECK(mode == "MODE_ZEROING" || mode == "MODE_MERGING")
        << "PTO vector-scalar op expects MODE_ZEROING or MODE_MERGING, got "
        << mode;
    const char *name = op->op.same_as(tl::simd_vadds())   ? "vadds"
                       : op->op.same_as(tl::simd_vmuls()) ? "vmuls"
                       : op->op.same_as(tl::simd_vmaxs()) ? "vmaxs"
                       : op->op.same_as(tl::simd_vmins()) ? "vmins"
                       : op->op.same_as(tl::simd_vshls()) ? "vshls"
                                                          : "vshrs";
    std::string src = PrintExpr_(op->args[arg_base]);
    DataType result_dtype = op->args[arg_base].dtype();
    std::string scalar = WrapTypedConst(
        RemoveOutermostParentheses(
            PrintExpr_(PeelScalarCasts(op->args[arg_base + 1]))),
        result_dtype, IsImmediateScalar(op->args[arg_base + 1]));
    std::string mask = PrintExpr_(op->args[arg_base + 2]);
    std::string result;
    {
      std::ostringstream result_os;
      result_os << "pto." << name << "(" << src << ", " << scalar << ", "
                << mask << ")";
      result = result_os.str();
    }
    if (mode == "MODE_MERGING") {
      // PTODSL vector-scalar ops use zeroing semantics for inactive lanes.
      if (has_explicit_dst) {
        // Newer pipelines legalize merging to an explicit read-write
        // destination, which may differ from the source operand.
        std::string dst_ref = GetMutableVectorRef(op->args[0], name);
        os << dst_ref << " = pto.vsel(" << result << ", " << dst_ref << ", "
           << mask << ")";
      } else {
        // Compatibility with support_pto_codegen before
        // LegalizeSimdMerging was added to the Ascend pipeline.
        os << "pto.vsel(" << result << ", " << src << ", " << mask << ")";
      }
    } else {
      os << result;
    }
    return;
  }

  // Cross-lane reductions / lane squeeze: (src, mask, mode) -> pto.<op>(src,
  // mask)
  if (op->op.same_as(tl::simd_vcpadd()) ||
      op->op.same_as(tl::simd_vcadd()) || op->op.same_as(tl::simd_vcmax()) ||
      op->op.same_as(tl::simd_vcmin()) || op->op.same_as(tl::simd_vcgadd()) ||
      op->op.same_as(tl::simd_vcgmax()) || op->op.same_as(tl::simd_vcgmin()) ||
      op->op.same_as(tl::simd_vsqz())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd cross-lane op expects 3 arguments (src, mask, mode)";
    ValidateVecMode_(op, 2,
                     op->op.same_as(tl::simd_vsqz()) ? "MODE_STORED"
                                                     : "MODE_ZEROING");
    const char *name = op->op.same_as(tl::simd_vcpadd())   ? "vcpadd"
                       : op->op.same_as(tl::simd_vcadd())   ? "vcadd"
                       : op->op.same_as(tl::simd_vcmax())  ? "vcmax"
                       : op->op.same_as(tl::simd_vcmin())  ? "vcmin"
                       : op->op.same_as(tl::simd_vcgadd()) ? "vcgadd"
                       : op->op.same_as(tl::simd_vcgmax()) ? "vcgmax"
                       : op->op.same_as(tl::simd_vcgmin()) ? "vcgmin"
                                                           : "vsqz";
    DataType input_elem = op->args[0].dtype().element_of();
    int input_bits = input_elem.bits();
    if (input_bits < 8) {
      input_bits = 8;
    }
    std::string mask = "pto.pbitcast(" +
                       RemoveOutermostParentheses(PrintExpr_(op->args[1])) +
                       ", pto.mask_type(\"b" + std::to_string(input_bits) +
                       "\"))";
    os << "pto." << name << "(" << PrintExpr_(op->args[0]) << ", " << mask
       << ")";
    return;
  }

  // Elementwise compare -> predicate: (src0, src1, mask, op)
  if (op->op.same_as(tl::simd_vcmp())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vcmp expects 4 arguments (src0, src1, mask, op)";
    std::string cmp_op = Downcast<StringImm>(op->args[3])->value;
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
    std::string cmp_op = Downcast<StringImm>(op->args[3])->value;
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
    std::string order = Downcast<StringImm>(op->args[1])->value;
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
    DataType elem = op->dtype.element_of();
    std::string index;
    if (IsImmediateScalar(op->args[0])) {
      index = "pto.const(" +
              RemoveOutermostParentheses(
                  PrintExpr_(PeelScalarCasts(op->args[0]))) +
              ", dtype=" + ScalarType(elem) + ")";
    } else {
      index = ScalarCastExpr(PrintExpr_(op->args[0]), elem,
                                "tl.simd.vci index");
    }
    os << "pto.vci(" << index << ", order=\"" << order << "\")";
    return;
  }

  // Scalar broadcast: (scalar, mask, mode) -> pto.vdup(scalar, mask)
  if (op->op.same_as(tl::simd_vdup())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.simd.vdup expects 3 arguments (src, mask, mode)";
    ValidateVecMode_(op, 2);
    DataType elem = op->dtype.element_of();
    std::string scalar;
    if (IsImmediateScalar(op->args[0])) {
      scalar = "pto.const(" +
               RemoveOutermostParentheses(
                   PrintExpr_(PeelScalarCasts(op->args[0]))) +
               ", dtype=" + DataTypeName(elem) + ")";
    } else if (elem.is_int() || elem.is_uint() || elem.is_bool()) {
      scalar = ScalarCastExpr(
          RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0]))),
          elem, "tl.simd.vdup scalar");
    } else {
      scalar = RemoveOutermostParentheses(
          PrintExpr_(PeelScalarCasts(op->args[0])));
    }
    os << "pto.vdup(" << scalar << ", " << PrintExpr_(op->args[1]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vdupv())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vdupv expects 4 arguments (src, mask, pos, mode)";
    ValidateVecMode_(op, 3);
    std::string pos_str = Downcast<StringImm>(op->args[2])->value;
    std::string position = (pos_str == "POS_LOWEST") ? "LOWEST" : "HIGHEST";
    std::string scalar = WrapTypedConst(
        RemoveOutermostParentheses(PrintExpr_(PeelScalarCasts(op->args[0]))),
        op->dtype, IsImmediateScalar(op->args[0]));
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
    std::string part = Downcast<StringImm>(op->args[1])->value;
    os << "pto.vpack(" << PrintExpr_(op->args[0]) << ", \"" << part << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_vgatherb())) {
    ICHECK(op->args.size() == 2U || op->args.size() == 3U)
        << "tl.simd.vgatherb expects 2 or 3 arguments (base, index[, mask])";
    DataType elem = op->dtype.element_of();
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
       << ", result_vreg_type=pto.vreg_type(" << op->dtype.lanes() << ", "
       << ScalarType(elem) << "))";
    return;
  }

  if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK(op->args.size() >= 4U && op->args.size() <= 6U)
        << "tl.simd.vcvt expects 4 to 6 arguments (src, mask, ...extra)";
    DataType elem_dtype = op->dtype.element_of();
    // Parse trailing StringImm args: ROUND_*, SAT/NOSAT, PART_*, MODE_*
    std::string rnd, sat, part;
    for (size_t i = 2; i < op->args.size(); ++i) {
      std::string val = Downcast<StringImm>(op->args[i])->value;
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
    DataType src_elem_dtype = op->args[0].dtype().element_of();
    int src_bits = src_elem_dtype.bits();
    if (src_bits < 8) {
      src_bits = 8; // CCE has no pset_b4
    }
    std::string mask = "pto.pbitcast(" +
                       RemoveOutermostParentheses(PrintExpr_(op->args[1])) +
                       ", pto.mask_type(\"b" + std::to_string(src_bits) +
                       "\"))";
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
      dist = Downcast<StringImm>(op->args[3])->value;
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
        dist, op->args[1].dtype().element_of());
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
    ICHECK_EQ(op->dtype.is_handle(), post_update)
        << "tl.simd.vsstb post-update form must return a handle";
    DataType stride_dtype = op->args[2].dtype();
    ICHECK(stride_dtype.is_scalar() &&
           (stride_dtype.is_int() || stride_dtype.is_uint()) &&
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
      std::string stride =
          RemoveOutermostParentheses(PrintExpr_(op->args[2]));
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
        os << "pto.vsstb(" << src << ", " << base << ", "
           << block_stride << ", " << repeat_stride << ", " << mask
           << ", post_update=pto.PostUpdate.ON)";
      } else {
        PrintIndent();
        stream << "pto.vsstb(" << src << ", " << base << ", "
               << block_stride << ", " << repeat_stride << ", " << mask
               << ")\n";
      }
      return;
  }

  if (op->op.same_as(tl::simd_vscatter())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vscatter expects 4 arguments "
           "(src, base, offsets, mask)";
    // The TileLang surface accepts a byte-addressed buffer for packed
    // storage, while PTO's vscatter requires the destination pointer element
    // type to match the value register element type. Keep the authored byte
    // offsets intact and only refine the pointer type for the verifier (the
    // base expression is already an UB address produced by access_ptr).
    DataType value_dtype = op->args[0].dtype().element_of();
    std::string base = PrintExpr_(op->args[1]);
    DataType base_dtype = GetAnnotatedPointerDtype(op->args[1], value_dtype);
    if (base_dtype != value_dtype) {
      base = "pto.castptr(" + base + ", " + PointerTypeName(value_dtype, "ub") +
             ")";
    }
    PrintIndent();
    stream << "pto.vscatter(" << PrintExpr_(op->args[0]) << ", "
           << base << ", " << PrintExpr_(op->args[2])
           << ", " << PrintExpr_(op->args[3]) << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_mem_bar())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.simd.mem_bar expects 1 argument (barrier type string)";
    std::string barrier_type = Downcast<StringImm>(op->args[0])->value;
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
    DataType elem_dtype = op->dtype.element_of();
    // vgather2 widens byte sources to 16-bit result lanes. Its PTO predicate
    // is defined at the result element width, whereas the TileLang intrinsic
    // carries the source-byte mask (b8). Reinterpret the predicate at the
    // result width so PTODSL/MLIR sees the required mask type without changing
    // the selected lane bits.
    int mask_bits = elem_dtype.bits();
    if (mask_bits < 8) {
      mask_bits = 8;
    }
    std::string mask = "pto.pbitcast(" +
                       RemoveOutermostParentheses(PrintExpr_(op->args[2])) +
                       ", pto.mask_type(\"b" + std::to_string(mask_bits) +
                       "\"))";
    std::string result_type = "pto.vreg_type(" +
                              std::to_string(op->dtype.lanes()) + ", " +
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
        << "tl.simd.pair_get references an unknown pair variable";
    int64_t index = Downcast<IntImm>(op->args[1])->value;
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
    if (op->args.size() == 1U &&
        TryEmitUnaryMath_(op_name, op->args[0], os)) {
      return;
    }
    if (StartsWith(op_name, "tl.")) {
      LOG(FATAL) << "PTO codegen does not support TileLang op `" << op_name
                 << "`. Please add a handler in CodeGenTileLangPTO or lower "
                    "it before PTO codegen.";
    }
  }

  CodeGenTileLangPY::VisitExpr_(op, os);
}

void CodeGenTileLangPTO::VisitExpr_(const FloatImmNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (!op->dtype.is_bfloat16()) {
    CodeGenTileLangPY::VisitExpr_(op, os);
    return;
  }

  // Python has no bfloat16 scalar constructor.  Keep the literal as a Python
  // float here; PTODSL operations such as scalar.store and pto.const coerce it
  // to the destination/authored bf16 type at the use site.
  std::ostringstream literal;
  literal << "float.fromhex('" << FlexibleHexFormat(op->value) << "')";
  MarkConst(literal.str());
  os << literal.str();
}

void CodeGenTileLangPTO::VisitExpr_(const CastNode *op,
                                    std::ostream &os) { // NOLINT(*)
  DataType from = op->value.dtype();
  DataType to = op->dtype;
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
    if (from.is_float() && from.bits() == 32 && from.lanes() == 2 && to_fp8) {
      os << "pto.convert(";
      PrintExpr_(op->value, os);
      os << ", " << FP8TypeName(to)
         << ", rounding=\"r\", saturation=\"sat\")";
      return;
    }
    LOG(FATAL) << "PTO SIMT FP8 cast currently supports float32x2 to "
                  "float8_e4m3fnx2/float8_e5m2x2 only, got "
               << from << " -> " << to;
  }

  bool from_integer = from.is_int() || from.is_uint() || from.is_bool();
  bool to_integer = to.is_int() || to.is_uint() || to.is_bool();
  bool from_float = from.is_float() || from.is_bfloat16();
  bool to_float = to.is_float() || to.is_bfloat16();
  bool immediate_value = op->value.as<IntImmNode>() != nullptr ||
                         op->value.as<FloatImmNode>() != nullptr;
  bool scalar_value = from.is_scalar() && to.is_scalar();
  if (from.is_scalar() && to.is_scalar() && from_integer && to.is_float() &&
      (from.is_bool() || from.bits() == 1)) {
    // Python float(runtime_i1) tries to coerce a device SSA value at trace
    // time.  Materialize the numeric boolean conversion on the device.
    os << "scalar.select(" << PrintCondition(op->value)
       << ", pto.const(1.0, dtype=" << DataTypeName(to)
       << "), pto.const(0.0, dtype=" << DataTypeName(to) << "))";
    return;
  }
  if (!immediate_value && scalar_value &&
      ((from_float && to_integer) ||
       (from_integer && to_float && !inside_simtvf_body_))) {
    os << "tl.scalar_cast(" << PrintExpr_(op->value) << ", "
       << DataTypeName(to) << ", context=\"PTO scalar cast\")";
    return;
  }
  if (!immediate_value && from_float && to_float &&
      from.lanes() == to.lanes() &&
      from.element_of() != to.element_of()) {
    // Keep MLIR implementation details inside PTODSL.  scalar.cast preserves
    // the shape of builtin vectors while replacing their element dtype.  Use
    // the full element dtype instead of bit width so same-width conversions
    // such as float16 <-> bfloat16 are also lowered on the device.
    os << "tl.scalar_cast(" << PrintExpr_(op->value) << ", "
       << DataTypeName(to.element_of())
       << ", context=\"PTO floating-point cast\")";
    return;
  }
  if (from.is_scalar() && to.is_scalar() && from_integer && to_integer &&
      !from.is_bool() && !to.is_bool() &&
      (from.bits() != to.bits() || from.is_uint() != to.is_uint())) {
    // Python int(runtime_value) tries to consume a device-side SSA value while
    // PTODSL is tracing. Use scalar.cast for both width changes and same-width
    // signedness changes so later scalar ops see the authored int/uint type.
    os << "tl.scalar_cast(";
    PrintExpr_(op->value, os);
    os << ", " << ScalarType(to)
       << ", context=\"PTO integer cast\")";
    return;
  }
  if (from_integer && to_integer && from.lanes() == to.lanes() &&
      (from.bits() <= to.bits() || from.bits() == 1 || to.bits() == 1 ||
       from.is_bool() || to.is_bool())) {
    // Address-index widening and boolean representation casts should not become
    // Python int(...)/bool(...), because that coerces PTODSL runtime values and
    // breaks tracing. Same-width int/uint casts are handled above to preserve
    // signedness for downstream scalar lowering.
    PrintExpr_(op->value, os);
    return;
  }

  if (to.is_bool()) {
    ICHECK(from.is_scalar() && (from_integer || from_float))
        << "PTO bool cast expects a scalar numeric source, got " << from;
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << (imm->value != 0 ? "True" : "False");
      return;
    }
    if (const auto *imm = op->value.as<FloatImmNode>()) {
      os << (imm->value != 0.0 ? "True" : "False");
      return;
    }
    os << "(";
    PrintExpr_(op->value, os);
    os << " != pto.const(0, dtype=" << DataTypeName(from) << "))";
    return;
  }

  if (from.is_bool() && to_integer) {
    ICHECK(from.is_scalar() && to.is_scalar())
        << "PTO bool-to-integer cast expects scalar types, got " << from
        << " -> " << to;
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << "pto.const(" << (imm->value != 0 ? 1 : 0)
         << ", dtype=" << DataTypeName(to) << ")";
      return;
    }
    os << "scalar.select(";
    PrintExpr_(op->value, os);
    os << ", pto.const(1, dtype=" << DataTypeName(to)
       << "), pto.const(0, dtype=" << DataTypeName(to) << "))";
    return;
  }

  if (from_integer && to_integer) {
    ICHECK(from.is_scalar() && to.is_scalar())
        << "PTO integer cast currently supports scalar values only, got "
        << from << " -> " << to;
    os << "scalar.cast(";
    PrintExpr_(op->value, os);
    os << ", " << DataTypeName(to) << ")";
    return;
  }

  if (from_integer && to_float) {
    ICHECK(from.is_scalar() && to.is_scalar())
        << "PTO integer-to-float cast currently supports scalar values only, "
           "got "
        << from << " -> " << to;
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << "pto.const(" << imm->value << ", dtype=" << DataTypeName(to) << ")";
      return;
    }
    if (inside_simtvf_body_) {
      // PTOAS accepts only signless i32/i64 as integer operands of pto.convert;
      // authored TIR integer values may carry signed/unsigned MLIR types.
      // Normalize the payload while retaining the authored signedness in the
      // conversion attribute. Widen narrow integers to the nearest supported
      // PTO conversion width before emitting the SIMT operation.
      const std::string pto_convert_src_type =
          from.bits() > 32 ? "pto.i64" : "pto.i32";
      os << "pto.convert(scalar.cast(";
      PrintExpr_(op->value, os);
      os << ", " << pto_convert_src_type << "), " << DataTypeName(to)
         << ", rounding=\"r\", saturation=\"nosat\", signedness=\""
         << (from.is_uint() ? "unsigned" : "signed") << "\")";
      return;
    } else {
      // TODO: Emit scalar.sitofp/uitofp for the corresponding TIR
      // arith.sitofp/uitofp once the PTODSL scalar.xxx and pto.xxx
      // interfaces are unified.
      ICHECK(false)
          << "PTO dynamic integer-to-float cast outside SIMT is temporarily "
             "unsupported, got "
          << from << " -> " << to;
    }
  }

  const bool from_float32_pair = IsFloat32Pair(from);
  const bool to_float32_pair = IsFloat32Pair(to);
  const bool from_half_pair =
      from.lanes() == 2 && (from.is_float16() || from.is_bfloat16());
  const bool to_half_pair =
      to.lanes() == 2 && (to.is_float16() || to.is_bfloat16());
  if ((from_float32_pair && to_half_pair) ||
      (from_half_pair && to_float32_pair)) {
    ICHECK(inside_simtvf_body_)
        << "PTO packed float cast is supported only inside SIMT, got " << from
        << " -> " << to;
    const char *to_pto_type = to_float32_pair   ? "pto.f32x2"
                              : to.is_float16() ? "pto.f16x2"
                                                : "pto.bf16x2";
    os << "pto.convert(";
    PrintExpr_(op->value, os);
    os << ", " << to_pto_type << ", rounding=\"r\", saturation=\"nosat\")";
    return;
  }

  const bool supported_float_cast = from.is_scalar() && to.is_scalar() &&
                                    ((IsFloat32(from) && to.is_bfloat16()) ||
                                     (from.is_bfloat16() && IsFloat32(to)));
  if (supported_float_cast) {
    if (const auto *imm = op->value.as<FloatImmNode>()) {
      os << "pto.const(float.fromhex('" << FlexibleHexFormat(imm->value)
         << "'), dtype=" << DataTypeName(to) << ")";
      return;
    }
    os << "scalar.cast(";
    PrintExpr_(op->value, os);
    os << ", " << DataTypeName(to) << ")";
    return;
  }

  LOG(FATAL) << "Unsupported PTO cast: " << from << " -> " << to;
}

void CodeGenTileLangPTO::VisitExpr_(const LetNode *op,
                                    std::ostream &os) { // NOLINT(*)
  // Keep Let expression-local.  In particular, a Let can appear inside a lazy
  // if_then_else branch; writing its binding to the function stream would
  // hoist potentially unsafe work (for example, division by zero) outside the
  // branch callback.  A Python lambda preserves Let's value-before-body order
  // and is evaluated wherever the containing expression is evaluated.
  std::string value = PrintExpr_(op->value);
  ICHECK(!var_idmap_.count(op->var.get()))
      << "PTO Let variable is already defined: " << op->var->name_hint;

  std::string vid = AllocVarID(op->var.get());
  std::string body = PrintExpr_(op->body);

  bool removed = var_idmap_.erase(op->var.get());
  ICHECK(removed) << "PTO Let variable was not registered: "
                  << op->var->name_hint;

  os << "(lambda " << vid << ": " << body << ")(" << value << ")";
}

void CodeGenTileLangPTO::VisitExpr_(const MinNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    // pto.fmin/fmax only accept floating dtypes and there is no SIMT-domain
    // integer min/max op yet, so integer min/max go through the runtime
    // scalar helpers.
    // TODO: switch to the unified pto.min/max once the scalar/SIMT op
    // unification lands and scalar.min/max are deprecated.
    os << "scalar.min(";
    PrintExpr_(op->a, os);
    os << ", ";
    PrintExpr_(op->b, os);
    os << ")";
    return;
  }
  PrintFloatMinMax_("min", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::VisitExpr_(const MaxNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    // TODO: switch to the unified pto.min/max once the scalar/SIMT op
    // unification lands and scalar.min/max are deprecated.
    os << "scalar.max(";
    PrintExpr_(op->a, os);
    os << ", ";
    PrintExpr_(op->b, os);
    os << ")";
    return;
  }
  PrintFloatMinMax_("max", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::PrintFloatMinMax_(const char *op_name,
                                              DataType dtype, PrimExpr lhs,
                                              PrimExpr rhs,
                                              std::ostream &os) { // NOLINT(*)
  ICHECK(IsSupportedFloatMinMaxType(dtype))
      << "PTO floating min/max currently supports f16, f32, bf16, "
         "vector<2xf16>, vector<2xf32>, and vector<2xbf16>, got "
      << dtype;

  // PTO's packed min/max micro-ops omit f32x2. Apply the supported scalar f32
  // operation lane-by-lane and repack the pair instead.
  if (inside_simtvf_body_ && IsFloat32Pair(dtype)) {
    os << "tl.vectorize_binary_f32x2(pto.f" << op_name << ", ";
    PrintExpr_(lhs, os);
    os << ", ";
    PrintExpr_(rhs, os);
    os << ")";
    return;
  }
  if (!inside_simtvf_body_) {
    ICHECK(dtype.is_scalar())
        << "PTO floating min/max outside SIMT requires a scalar dtype, got "
        << dtype;
    os << "scalar." << op_name << "(";
    PrintExpr_(lhs, os);
    os << ", ";
    PrintExpr_(rhs, os);
    os << ")";
    return;
  }

  // TODO: Use the unified pto.min/max once the scalar/SIMT op unification
  // lands; pto.fmin/fmax and scalar.min/max both collapse into the unified
  // operations.
  os << "pto.f" << op_name << "(";
  PrintExpr_(lhs, os);
  os << ", ";
  PrintExpr_(rhs, os);
  os << ")";
}

bool CodeGenTileLangPTO::TryEmitUnaryMath_(const std::string &name,
                                              const PrimExpr &arg,
                                              std::ostream &os) { // NOLINT(*)
  // Unary math mapping shared by the extern C path (call_pure_extern) and
  // the tirx intrinsic path. Add new functions as table entries only.
  struct UnaryMathForm {
    const char *simt;   // form emitted inside T.SimtVF bodies
    const char *scalar; // form emitted in the scalar domain
    bool reciprocal;    // emit (1.0 / fn(x)) instead of fn(x)
  };
  static const std::unordered_map<std::string, UnaryMathForm> kForms = {
      {"expf", {"pto.exp(", "scalar.exp(", false}},
      {"tirx.exp", {"pto.exp(", "scalar.exp(", false}},
      {"logf", {"pto.log(", "scalar.log(", false}},
      {"tirx.log", {"pto.log(", "scalar.log(", false}},
      {"sqrt", {"pto.sqrt(", "scalar.sqrt(", false}},
      {"sqrtf", {"pto.sqrt(", "scalar.sqrt(", false}},
      {"tirx.sqrt", {"pto.sqrt(", "scalar.sqrt(", false}},
      {"rsqrt", {"pto.sqrt(", "scalar.sqrt(", true}},
      {"rsqrtf", {"pto.sqrt(", "scalar.sqrt(", true}},
      {"tirx.rsqrt", {"pto.sqrt(", "scalar.sqrt(", true}},
  };
  auto it = kForms.find(name);
  if (it == kForms.end()) {
    return false;
  }
  DataType dtype = arg.dtype();
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
      std::string simt_op = form.simt;
      ICHECK(!simt_op.empty() && simt_op.back() == '(');
      simt_op.pop_back();
      os << "tl.vectorize_unary_f32x2(" << simt_op << ", " << value << ")";
    }
    return true;
  }
  const char *fn = inside_simtvf_body_ ? form.simt : form.scalar;
  if (form.reciprocal) {
    os << "(1.0 / " << fn << value << "))";
  } else {
    os << fn << value << ")";
  }
  return true;
}

void CodeGenTileLangPTO::VisitExpr_(const NotNode *op,
                                    std::ostream &os) { // NOLINT(*)
  ICHECK(op->dtype.is_scalar() && op->a.dtype().is_bool())
      << "PTO logical-not currently supports scalar bool predicates only, got "
      << op->a.dtype();
  // tl.logical_not(value) implements logical negation as value == 0.
  os << "tl.logical_not(";
  PrintExpr_(op->a, os);
  os << ")";
}

void CodeGenTileLangPTO::VisitExpr_(const SelectNode *op,
                                    std::ostream &os) { // NOLINT(*)
  const bool coerce_integer_branches =
      op->dtype.is_scalar() && (op->dtype.is_int() || op->dtype.is_uint());
  auto print_select_value = [&](const PrimExpr &expr) {
    if (expr.as<IntImmNode>() != nullptr ||
        expr.as<FloatImmNode>() != nullptr) {
      os << "pto.const(" << PrintExpr_(expr)
         << ", dtype=" << DataTypeName(op->dtype) << ")";
      return;
    }
    if (op->dtype.is_scalar() && op->dtype.is_bool()) {
      os << "tl.as_logical_bool(" << PrintExpr_(expr) << ")";
      return;
    }
    if (coerce_integer_branches) {
      os << ScalarCastExpr(PrintExpr_(expr), op->dtype,
                              "PTO select branch");
      return;
    }
    PrintExpr_(expr, os);
  };
  os << "scalar.select(";
  os << PrintCondition(op->condition);
  os << ", ";
  print_select_value(op->true_value);
  os << ", ";
  print_select_value(op->false_value);
  os << ")";
}

void CodeGenTileLangPTO::VisitExpr_(const ModNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_scalar() && (op->dtype.is_int() || op->dtype.is_uint())) {
    os << "(" << PrintExpr_(op->a) << " % " << PrintExpr_(op->b) << ")";
    return;
  }
  CodeGenTileLangPY::VisitExpr_(op, os);
}

void CodeGenTileLangPTO::VisitExpr_(const DivNode *op,
                                    std::ostream &os) { // NOLINT(*)
  CodeGenTileLangPY::VisitExpr_(op, os);
}

void CodeGenTileLangPTO::PrintBinaryExpr_(const std::string &opstr,
                                          DataType dtype, PrimExpr lhs,
                                          PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
  // shift_right and similar paths emit "//"/"%" via PrintBinaryExpr_ without
  // going through VisitExpr_(Div/Mod). Keep integer remainder as an authored
  // runtime `%` so PTODSL preserves signedness and width (RemSI vs RemUI).
  if (dtype.is_scalar() && (dtype.is_int() || dtype.is_uint()) &&
      opstr == "%") {
    os << "(" << PrintExpr_(lhs) << " % " << PrintExpr_(rhs) << ")";
    return;
  }
  if (dtype.is_scalar()) {
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
    os << "tl.vectorize_binary_f32x2(tl.scalar_div, " << PrintExpr_(lhs)
       << ", " << PrintExpr_(rhs) << ")";
    return;
  }
  if (opstr != "+" && opstr != "-" && opstr != "*") {
    LOG(FATAL) << "Unsupported PTO SIMT vector binary op: " << opstr;
  }
  os << "(" << PrintExpr_(lhs) << " " << opstr << " " << PrintExpr_(rhs) << ")";
}

void CodeGenTileLangPTO::VisitExpr_(const BroadcastNode *op,
                                    std::ostream &os) { // NOLINT(*)
  DataType elem_dtype = op->value.dtype();
  ICHECK(elem_dtype.is_float16() || IsFloat32(elem_dtype) ||
         elem_dtype.is_bfloat16())
      << "PTO vector broadcast currently supports f16, f32, and bf16 only, got "
      << elem_dtype;
  std::string value = PrintExpr_(op->value);
  os << "pto.Vec(" << DataTypeName(elem_dtype) << ", " << op->dtype.lanes()
     << ", init=" << value << ")";
}

void CodeGenTileLangPTO::VisitStmt_(const DeclBufferNode *op) { (void)op; }

void CodeGenTileLangPTO::VisitStmt_(const BindNode *op) {
  if (const auto *call = op->value.as<CallNode>()) {
    if (IsOpName(call->op, "tl.simd.alloc")) {
      AllocVarID(op->var.get());
      DataType elem_dtype = op->var.dtype().element_of();
      ICHECK_GT(op->var.dtype().lanes(), 1)
          << "tl.simd.alloc should bind a vector-typed Var, got "
          << op->var.dtype();
      FragmentInfo info;
      info.lanes = op->var.dtype().lanes();
      info.dtype = elem_dtype;
      fragment_info_[op->var.get()] = info;
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
      stream << low << ", " << high << " = pto." << name << "_b" << width
             << "(" << PrintExpr_(call->args[0]) << ", "
             << PrintExpr_(call->args[1]) << ")\n";
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
             << PrintExpr_(call->args[0]) << ", "
             << PrintExpr_(call->args[1]);
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
             << PrintExpr_(call->args[0]) << ", "
             << PrintExpr_(call->args[1]) << ", "
             << PrintExpr_(call->args[2]) << ")\n";
      simd_pair_vars_[op->var.get()] = {low, high};
      return;
    }
    if (call->op.same_as(tl::simd_vld2())) {
      std::string ptr = PrintExpr_(call->args[0]);
      std::string dist = Downcast<StringImm>(call->args[1])->value;
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
  }

  // T.make_tensor binds a handle-typed Var to `reinterpret(handle, int_addr)`
  // and carries the element dtype + storage scope in `type_annotation`.
  // Register them so GetPointerExpr / HandleTypeMatch_ see the right type.
  if (op->var.dtype().is_handle() && op->var->type_annotation.defined()) {
    if (auto *ptr = op->var->type_annotation.as<PointerTypeNode>()) {
      DataType elem_dtype = DataType::Float(32);
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        elem_dtype = prim->dtype;
        RegisterHandleType_(op->var.get(), prim->dtype);
      }
      std::string scope = GetPtrStorageScope(op->var);
      alloc_storage_scope_[op->var.get()] = scope;

      // T.make_tensor: reinterpret(handle, int_addr) carries an int64 address.
      // PTODSL's pto.addptr expects a pointer, not i64; emit a castptr so the
      // bound variable is a properly typed gm pointer.
      const auto *call = op->value.as<CallNode>();
      if (call && call->op.same_as(builtin::reinterpret()) &&
          !call->args[0]->dtype.is_handle()) {
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
  std::string name_hint = op->var->name_hint;
  if (name_hint.rfind("__cond_", 0) == 0) {
    // PTODSL's rewritten branch object uses attribute access for live-out
    // values. Python reserves double-underscore attribute names, so normalize
    // AutoSchedule's condition temporaries before they reach the DSL.
    vid = "tl_cond_" + name_hint.substr(7);
    var_idmap_[op->var.get()] = vid;
  } else if (IsInlineableInvariantSimdBind(op->value)) {
    // Map the Var to the printed RHS so each use rematerializes the broadcast /
    // pset. Do not emit a live SSA bind that would cross range()/scf.for.
    var_idmap_[op->var.get()] = PrintExpr_(op->value);
    return;
  } else {
    vid = AllocVarID(op->var.get());
  }
  PrintSSAAssign(vid, PrintExpr_(op->value), op->var.dtype());
}

void CodeGenTileLangPTO::VisitStmt_(const AllocBufferNode *op) {
  if (!current_function_is_cube_ || in_mixed_vector_section_) {
    EmitBufferAllocation(op->buffer);
    return;
  }

  const Var &buffer_var = op->buffer->data;
  std::string scope = GetPtrStorageScope(buffer_var);
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
  } else if (scope == "local.fragment") {
    AllocVarID(buffer_var.get());
    auto alloc_ref = GetRef<AllocBuffer>(op);
    auto opt_size = alloc_ref.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO local.fragment allocation expects a constant size";
    FragmentInfo info;
    info.lanes = static_cast<int>(opt_size.value());
    info.dtype = op->buffer->dtype;
    fragment_info_[buffer_var.get()] = info;
  } else if (scope == "local.var") {
    // Scalar alloc_var is used by GEMM tile mapping; vector alloc_var is a
    // mutable SIMD register and is emitted as a PTODSL Vec value.
    CheckLocalVarBuffer(op->buffer.get());
    PrintIndent();
    local_var_buffers_.insert(buffer_var.get());
    DataType dtype = op->buffer->dtype;
    std::string vid = AllocVarID(buffer_var.get());
    if (vid.rfind("__cond_", 0) == 0) {
      vid = "tl_cond_" + vid.substr(7);
      var_idmap_[buffer_var.get()] = vid;
    } else if (!vid.empty() && vid.front() == '_') {
      // See EmitBufferAllocation: BranchHandle result names may not start
      // with an underscore.
      vid = name_supply_->FreshName("tl" + vid, false);
      var_idmap_[buffer_var.get()] = vid;
    }
    if (current_function_has_gemm_) {
      // Keep authored PTO surface values (not Python ints) so AST-rewritten
      // `if` / br.assign can merge tile indices across branches.
      if (IsVectorLocalVarDtype(dtype) || dtype.is_handle()) {
        stream << vid << " = None\n";
      } else {
        stream << vid << " = pto.const(" << (dtype.is_float() ? "0.0" : "0")
               << ", dtype=" << DataTypeName(dtype) << ")\n";
      }
      RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
      EmitMixedEntrySnapshot(buffer_var.get());
      return;
    }
    if (IsVectorLocalVarDtype(dtype) || dtype.is_handle()) {
      stream << vid << " = None\n";
    } else if (IsSurfaceScalarLocalVarDtype(dtype)) {
      stream << vid << " = pto.const(" << (dtype.is_float() ? "0.0" : "0")
             << ", dtype="
             << DataTypeName(dtype)
             << ")\n";
    } else {
      LOG(FATAL) << "Unsupported PTO local.var dtype: " << dtype;
    }
    EmitMixedEntrySnapshot(buffer_var.get());
  }

  RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
}

void CodeGenTileLangPTO::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "pragma_unroll_factor") {
    const auto *loop_var = op->node.as<VarNode>();
    const auto *factor = op->value.as<IntImmNode>();
    ICHECK(loop_var != nullptr)
        << "pragma_unroll_factor must annotate a loop variable";
    ICHECK(factor != nullptr)
        << "pragma_unroll_factor must be a constant integer";
    ICHECK_GT(factor->value, 0)
        << "PTO pragma_unroll_factor must be a positive integer, got "
        << factor->value;
    ICHECK_LE(factor->value, std::numeric_limits<int32_t>::max())
        << "PTO pragma_unroll_factor must fit a signless i32 attribute, got "
        << factor->value;

    Optional<Var> old_loop_var = current_unroll_factor_loop_var_;
    const int64_t old_factor = current_unroll_factor_;
    current_unroll_factor_loop_var_ = GetRef<Var>(loop_var);
    current_unroll_factor_ = factor->value;
    VisitStmt(op->body);
    current_unroll_factor_loop_var_ = old_loop_var;
    current_unroll_factor_ = old_factor;
    return;
  }

  if (op->attr_key == tirx::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x") {
      // Cast pto.get_block_idx() (default i64) with the dtype of TIR launch
      // IterVar (normally i32) once
      std::string vid = AllocVarID(iv->var.get());
      PrintIndent();
      stream << vid << " = tl.scalar_cast(pto.get_block_idx(), "
             << ScalarType(iv->var.dtype())
             << ", context=\"PTO block index cast\")\n";
      VisitStmt(op->body);
      return;
    }

    const bool is_simt_thread = iv->thread_tag == "threadIdx.x" ||
                                iv->thread_tag == "threadIdx.y" ||
                                iv->thread_tag == "threadIdx.z";
    if (!inside_simtvf_body_ && is_simt_thread) {
      ICHECK(!UsesVar(op->body, iv->var.get()))
          << "PTO codegen cannot use " << iv->thread_tag
          << " outside a SIMT section; move the use into T.SimtVF";
      VisitStmt(op->body);
      return;
    }

    std::string vid = AllocVarID(iv->var.get());
    std::string thread_value;
    if (iv->thread_tag == "cthread") {
      thread_value = "pto.get_subblock_idx()";
    } else if (iv->thread_tag == "threadIdx.x") {
      thread_value = "pto.get_tid_x()";
    } else if (iv->thread_tag == "threadIdx.y") {
      thread_value = "pto.get_tid_y()";
    } else if (iv->thread_tag == "threadIdx.z") {
      thread_value = "pto.get_tid_z()";
    } else {
      LOG(FATAL) << "Unsupported PTO thread tag: " << iv->thread_tag;
    }
    PrintIndent();
    stream << vid << " = " << thread_value << "\n";
    VisitStmt(op->body);
    return;
  }

  if (op->attr_key == "tl.simdvf_scope") {
    if (simdvf_nesting_depth_ == 0) {
      PrintIndent();
      stream << "with pto.vecscope():\n";
      int scope = BeginScope();
      simdvf_nesting_depth_++;
      VisitStmt(op->body);
      simdvf_nesting_depth_--;
      EndScope(scope);
    } else {
      PrintIndent();
      stream << "pto.mem_bar(\"VST_VLD\")\n";
      VisitStmt(op->body);
    }
    return;
  }

  if (op->attr_key == "tl.simtvf_scope") {
    VisitStmt(op->body);
    return;
  }

  VisitStmt(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const SeqStmtNode *op) {
  for (size_t i = 0; i < op->seq.size(); ++i) {
    const Stmt &stmt = op->seq[i];
    if (i + 1U < op->seq.size()) {
      const auto *first_if = stmt.as<IfThenElseNode>();
      const auto *second_if = op->seq[i + 1U].as<IfThenElseNode>();
      if (first_if != nullptr && second_if != nullptr &&
          !first_if->else_case.defined() &&
          !second_if->else_case.defined() &&
          !ContainsLoopBreak(stmt) &&
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
          DataType entry_pad_value_dtype = current_copy_pad_value_dtype_;
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

void CodeGenTileLangPTO::VisitStmt_(const ForNode *op) {
  int entry_pad_value_id = current_copy_pad_value_id_;
  DataType entry_pad_value_dtype = current_copy_pad_value_dtype_;

  arith::Analyzer analyzer;
  PrimExpr start = analyzer.Simplify(op->min);
  PrimExpr extent = analyzer.Simplify(op->extent);
  PrimExpr step = op->step.has_value() ? analyzer.Simplify(op->step.value())
                                       : make_const(op->loop_var.dtype(), 1);
  PrimExpr stop = analyzer.Simplify(start + extent);
  std::string vid = AllocVarID(op->loop_var.get());
  std::string begin = RemoveOutermostParentheses(PrintExpr_(start));
  std::string end = RemoveOutermostParentheses(PrintExpr_(stop));
  std::string step_expr = RemoveOutermostParentheses(PrintExpr_(step));
  // Explicit unrolls are normally expanded before codegen. A kUnrolled loop
  // that survives without the annotation is the explicit=False form and must
  // remain a device-side range loop.
  bool explicit_unroll_requested = false;
  std::string range_vid;
  if (op->kind == tirx::ForKind::kUnrolled) {
    auto it = op->annotations.find(tirx::attr::pragma_unroll_explicit);
    if (it != op->annotations.end()) {
      auto annotated = (*it).second.try_cast<bool>();
      ICHECK(annotated.has_value())
          << "pragma_unroll_explicit must be a boolean, got " << (*it).second;
      explicit_unroll_requested = annotated.value();
    }
  }
  int64_t start_value = 0;
  int64_t extent_value = 0;
  int64_t step_value = 0;
  const bool explicit_unroll =
      explicit_unroll_requested && TryGetConstInt(start, &start_value) &&
      TryGetConstInt(extent, &extent_value) && TryGetConstInt(step, &step_value);
  bool static_local_index_loop = false;
  // Scalarized local buffers are emitted as Python lists. A constant-bound
  // loop whose induction variable is used as a list slot must therefore also
  // be trace-time static, even when TIR classified the loop as serial.
  if (TryGetConstInt(start, &start_value) &&
      TryGetConstInt(extent, &extent_value) &&
      TryGetConstInt(step, &step_value)) {
    tirx::PostOrderVisit(op->body, [&](const ObjectRef &node) {
      const BufferNode *buffer = nullptr;
      PrimExpr index;
      if (const auto *load = node.as<BufferLoadNode>()) {
        if (load->indices.size() == 1) {
          buffer = load->buffer.get();
          index = load->indices[0];
        }
      } else if (const auto *store = node.as<BufferStoreNode>()) {
        if (store->indices.size() == 1) {
          buffer = store->buffer.get();
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
          persistent_buffer_vars_.count(buffer->data.get()) != 0;
      if (buffer == nullptr || !index.defined() ||
          (buffer->dtype.lanes() != 1 && !persistent_fragment) ||
          (inside_simtvf_body_ && !persistent_fragment) ||
          (op->kind == tirx::ForKind::kUnrolled &&
           !explicit_unroll_requested && !persistent_fragment)) {
        return;
      }
      std::string scope = GetPtrStorageScope(buffer->data);
      if ((scope == "local" || scope == "local.fragment") &&
          tirx::UsesVar(index, [&](const VarNode *var) {
            return var == op->loop_var.get();
          })) {
        static_local_index_loop = true;
      }
    });
  }
  const bool emit_static_range =
      (explicit_unroll || static_local_index_loop) &&
      !ContainsLoopBreak(op->body);
  const bool pto_unroll_hint = op->kind == tirx::ForKind::kUnrolled &&
                               !explicit_unroll_requested &&
                               !ContainsLoopBreak(op->body);
  const bool has_unroll_factor =
      current_unroll_factor_loop_var_.defined() &&
      current_unroll_factor_loop_var_.value().same_as(op->loop_var);
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
  if (!emit_static_range) {
    PrintIndent();
    stream << vid << " = tl.scalar_cast(" << range_vid << ", "
           << ScalarType(op->loop_var.dtype())
           << ", context=\"PTO loop induction cast\")\n";
  }
  ++native_loop_depth_;
  PrintStmt_(op->body);
  --native_loop_depth_;
  EndScope(for_scope);
  current_copy_pad_value_id_ = entry_pad_value_id;
  current_copy_pad_value_dtype_ = entry_pad_value_dtype;
}

void CodeGenTileLangPTO::VisitStmt_(const WhileNode *op) {
  int entry_pad_value_id = current_copy_pad_value_id_;
  DataType entry_pad_value_dtype = current_copy_pad_value_dtype_;

  PrintIndent();
  stream << "while " << PrintCondition(op->condition) << ":\n";
  int while_scope = BeginScope();
  ++native_loop_depth_;
  PrintStmt_(op->body);
  --native_loop_depth_;
  EndScope(while_scope);

  // A runtime loop may execute zero times, so no copy-pad state established
  // only in its body is available after the loop.
  current_copy_pad_value_id_ = entry_pad_value_id;
  current_copy_pad_value_dtype_ = entry_pad_value_dtype;
}

void CodeGenTileLangPTO::VisitStmt_(const SBlockNode *op) {
  if (current_function_is_mixed_ &&
      (op->name_hint == "CUBE" || op->name_hint == "VECTOR")) {
    bool is_vector_section = op->name_hint == "VECTOR";
    int64_t vector_count = 2;
    if (is_vector_section) {
      if (auto opt = op->annotations.Get("vector_count")) {
        const auto *count = opt.value().as<IntImmNode>();
        ICHECK(count != nullptr && (count->value == 1 || count->value == 2))
            << "Mixed-kernel vector_count must be the constant integer 1 or "
               "2, got "
            << opt.value();
        vector_count = count->value;
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
    int active_aiv_scope = -1;
    if (is_vector_section && vector_count == 1) {
      // The physical mixed group still has two AIVs; only AIV0 executes this
      // section's body, including its cross-core synchronization.
      PrintIndent();
      stream << "if pto.get_subblock_idx() == 0:\n";
      active_aiv_scope = BeginScope();
    }
    if (op->init.defined()) {
      PrintStmt_(op->init.value());
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

  if (op->name_hint == "SIMT_VF") {
    int64_t thread_x = 1;
    int64_t thread_y = 1;
    int64_t thread_z = 1;
    ExtractSimtThreadExtents(op, &thread_x, &thread_y, &thread_z);
    EmitInlineSimtVF(op, thread_x, thread_y, thread_z);
    return;
  }

  if (op->name_hint == "SIMD_VF") {
    for (const Buffer &buf : op->alloc_buffers) {
      EmitBufferAllocation(buf);
    }
    if (op->init.defined()) {
      PrintStmt_(op->init.value());
    }
    PrintStmt_(op->body);
    return;
  }

  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const IfThenElseNode *op) {
  // Fold compile-time conditions; keep runtime control flow as native Python.
  PrimExpr cond_simp = arith::Analyzer().Simplify(op->condition);
  if (const auto *imm = cond_simp.as<IntImmNode>()) {
    if (imm->value != 0) {
      PrintStmt_(op->then_case);
    } else if (op->else_case.defined()) {
      PrintStmt_(op->else_case.value());
    }
    return;
  }

  int entry_pad_value_id = current_copy_pad_value_id_;
  DataType entry_pad_value_dtype = current_copy_pad_value_dtype_;
  CopyPadState symbolic_entry{
      entry_pad_value_id >= 0,
      false,
      entry_pad_value_dtype,
  };
  CopyPadState symbolic_then =
      AnalyzeCopyPadState(op->then_case, symbolic_entry);
  CopyPadState symbolic_else = symbolic_entry;
  if (op->else_case.defined()) {
    symbolic_else =
        AnalyzeCopyPadState(op->else_case.value(), symbolic_entry);
  }
  bool needs_pad_merge = symbolic_then.CanMerge(symbolic_else);

  std::string cond = PrintCondition(op->condition);

  auto set_reaching_pad_state = [&](int then_id, DataType then_dtype,
                                    int else_id, DataType else_dtype) {
    if (then_id >= 0 && then_id == else_id) {
      ICHECK(then_dtype == else_dtype)
          << "PTO copy padding reached the same SSA alias with inconsistent "
             "dtypes";
      current_copy_pad_value_id_ = then_id;
      current_copy_pad_value_dtype_ = then_dtype;
    } else {
      current_copy_pad_value_id_ = -1;
      current_copy_pad_value_dtype_ = DataType::Void();
    }
  };

  PrintIndent();
  stream << "if " << cond << ":\n";
  int then_scope = BeginScope();
  PrintStmt_(op->then_case);
  int then_pad_value_id = current_copy_pad_value_id_;
  DataType then_pad_value_dtype = current_copy_pad_value_dtype_;
  int merged_pad_value_id = -1;
  if (needs_pad_merge) {
    merged_pad_value_id = copy_pad_value_counter_++;
    EmitCopyPadMerge_(merged_pad_value_id, symbolic_then.dtype);
  }
  EndScope(then_scope);

  current_copy_pad_value_id_ = entry_pad_value_id;
  current_copy_pad_value_dtype_ = entry_pad_value_dtype;
  int else_pad_value_id = entry_pad_value_id;
  DataType else_pad_value_dtype = entry_pad_value_dtype;
  if (op->else_case.defined()) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt_(op->else_case.value());
    else_pad_value_id = current_copy_pad_value_id_;
    else_pad_value_dtype = current_copy_pad_value_dtype_;
    if (needs_pad_merge) {
      EmitCopyPadMerge_(merged_pad_value_id, symbolic_else.dtype);
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

void CodeGenTileLangPTO::VisitStmt_(const EvaluateNode *op) {
  if (is_const_int(op->value))
    return;
  if (const auto *call = op->value.as<CallNode>();
      call != nullptr &&
      (call->op.same_as(tl::device_assert()) ||
       call->op.same_as(tl::device_assert_with_msg()))) {
    const bool has_message =
        call->op.same_as(tl::device_assert_with_msg());
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

void CodeGenTileLangPTO::VisitExpr_(const BufferLoadNode *op,
                                    std::ostream &os) { // NOLINT(*)
  // S.alloc_var inside T.SimdVF() creates a `local.var`-scope buffer with a
  // vector dtype (e.g. float32x64). In PTODSL it is a plain Python variable
  // holding a vector register value, so load at index 0 is just the variable
  // itself. Skip the scalar-only CheckLocalVarBuffer below.
  if (op->buffer->dtype.lanes() > 1) {
    std::string scope = ScopeOfBuffer(op->buffer.get());
    if (scope == "local.var") {
      ICHECK_EQ(op->indices.size(), 1U)
          << "PTO vector local.var load expects a single index";
      int64_t index = 0;
      ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
          << "PTO vector local.var load expects index 0";
      os << LocalVarID(op->buffer->data.get());
      return;
    }
  }

  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // Vector SSA and scalar PTO surface values both load as Python variables.
    CheckLocalVarBuffer(op->buffer.get());
    if (op->buffer->dtype.lanes() > 1) {
      os << LocalVarID(op->buffer->data.get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var load expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var load expects index 0";
    if (current_function_has_gemm_) {
      os << LocalVarID(op->buffer->data.get());
      return;
    }
    if (IsVectorLocalVarDtype(op->buffer->dtype) ||
        IsSurfaceScalarLocalVarDtype(op->buffer->dtype)) {
      os << LocalVarID(op->buffer->data.get());
    } else {
      LOG(FATAL) << "Unsupported PTO local.var load dtype: "
                 << op->buffer->dtype;
    }
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer loads";
  ICHECK(!op->predicate.defined())
      << "CodeGenTileLangPTO does not support predicated loads yet";

  DataType value_dtype = op->dtype;
  DataType element_dtype = op->buffer->dtype;
  if (value_dtype.lanes() > 1) {
    EmitScalarizedLoad(op, os);
    return;
  }

  ICHECK_EQ(value_dtype, element_dtype)
      << "PTO scalar BufferLoad expects value dtype to match buffer element "
         "dtype, got "
      << value_dtype << " vs " << element_dtype;

  std::string loaded = ScalarLoad(op->buffer.get(), op->indices[0]);
  if (value_dtype.is_bool()) {
    os << "tl.as_logical_bool(" << loaded << ")";
  } else {
    os << loaded;
  }
}

void CodeGenTileLangPTO::EmitScalarizedLoad(const BufferLoadNode *op,
                                            std::ostream &os) {
  DataType value_dtype = op->dtype;
  DataType element_dtype = op->buffer->dtype;
  std::string scope = ScopeOfBuffer(op->buffer.get());

  // Vector-element buffer (e.g. T.simd.alloc_local produces float32x64
  // elements): reading one slot yields a whole vector register. Emit a plain
  // Python index expression.
  if (element_dtype.lanes() > 1) {
    if (scope == "local.fragment" || scope == "local") {
      os << GetVectorLocalRef(op->buffer->data.get(), op->indices[0],
                                 "PTO vector load");
    } else {
      std::string index_str =
          RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
      os << GetVarID(op->buffer->data.get()) << "[" << index_str << "]";
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
    ICHECK_EQ(value_dtype.element_of(), element_dtype)
        << "PTO SIMT vector BufferLoad expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << element_dtype;
    ICHECK(!tl::IsAscendVectorizableFP8(element_dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 BufferLoad currently supports contiguous lanes 2, 4, "
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
      os << "scalar.load(" << GetVarID(op->buffer->data.get()) << ", "
         << index_str << ", contiguous=" << lanes << ")";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base =
          ScalarPointerBase_(op->buffer->data.get(), element_dtype, scope);
      os << "scalar.load(" << base << ", " << index_str
         << ", contiguous=" << lanes << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base =
          ScalarPointerBase_(op->buffer->data.get(), element_dtype, scope);
      os << "scalar.load(" << base << ", "
         << index_str << ", contiguous=" << lanes << ")";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector load scope: " << scope;
  }

  // Non-SIMT vector load from a scalar-element buffer: use scalar.load with
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
      // their scalar lanes instead of calling scalar.load on the list.
      const std::string list_ref = GetVarID(op->buffer->data.get());
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
      std::string base =
          ScalarPointerBase_(op->buffer->data.get(), element_dtype, scope);
      os << "scalar.load(" << base << ", " << index_str
         << ", contiguous=" << lanes << ")";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base =
          ScalarPointerBase_(op->buffer->data.get(), element_dtype, scope);
      os << "scalar.load(" << base << ", "
         << index_str << ", contiguous=" << lanes << ")";
      return;
    }
    LOG(FATAL) << "Unsupported PTO vector load scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector BufferLoad is not supported yet";
}

void CodeGenTileLangPTO::VisitStmt_(const BufferStoreNode *op) {
  // S.alloc_var inside T.SimdVF() creates a `local.var`-scope buffer with a
  // vector dtype (e.g. float32x64). In PTODSL it is a plain Python variable
  // holding a vector register value, so store at index 0 is just
  // `var = value`. Skip the scalar-only CheckLocalVarBuffer below.
  if (op->buffer->dtype.lanes() > 1) {
    std::string scope = ScopeOfBuffer(op->buffer.get());
    if (scope == "local.var") {
      ICHECK_EQ(op->indices.size(), 1U)
          << "PTO vector local.var store expects a single index";
      int64_t index = 0;
      ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
          << "PTO vector local.var store expects index 0";
      std::string value =
          RemoveOutermostParentheses(PrintExpr_(op->value));
      PrintIndent();
      stream << LocalVarID(op->buffer->data.get()) << " = " << value << "\n";
      EmitMixedEntrySnapshot(op->buffer->data.get());
      return;
    }
  }

  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // Vector SSA and scalar PTO surface values both store as Python variables.
    CheckLocalVarBuffer(op->buffer.get());
    if (op->buffer->dtype.lanes() > 1) {
      std::string value =
          RemoveOutermostParentheses(PrintExpr_(op->value));
      PrintIndent();
      stream << LocalVarID(op->buffer->data.get()) << " = " << value << "\n";
      EmitMixedEntrySnapshot(op->buffer->data.get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var store expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var store expects index 0";
    std::string vid = LocalVarID(op->buffer->data.get());
    DataType dtype = op->buffer->dtype;
    std::string value;
    if (current_function_has_gemm_) {
      value = LocalVarStoreValue(op->value, dtype);
    } else if (IsVectorLocalVarDtype(dtype) || dtype.is_handle()) {
      value = RemoveOutermostParentheses(PrintExpr_(op->value));
    } else if (IsSurfaceScalarLocalVarDtype(dtype)) {
      value = LocalVarStoreValue(op->value, dtype);
    } else {
      LOG(FATAL) << "Unsupported PTO local.var store dtype: " << dtype;
    }
    PrintIndent();
    stream << vid << " = " << value << "\n";
    EmitMixedEntrySnapshot(op->buffer->data.get());
    return;
  }

  ICHECK_EQ(op->indices.size(), 1)
      << "CodeGenTileLangPTO only supports flat buffer stores";
  ICHECK(!op->predicate.defined())
      << "CodeGenTileLangPTO does not support predicated stores yet";

  if (op->value.dtype().lanes() > 1) {
    EmitScalarizedStore(op);
    return;
  }

  std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
  PrintIndent();
  EmitScalarStore(op->buffer.get(), value, op->indices[0]);
}

void CodeGenTileLangPTO::EmitScalarizedStore(const BufferStoreNode *op) {
  DataType buffer_dtype = op->buffer->dtype;
  DataType value_dtype = op->value.dtype();
  std::string scope = ScopeOfBuffer(op->buffer.get());

  // Vector-element buffer (e.g. T.simd.alloc_local produces float32x64
  // elements): the value is a whole vector register stored into one buffer
  // slot. Emit a plain Python assignment; the buffer was allocated as a list
  // (non-SIMT) or pto.alloc_buffer (SIMT).
  if (buffer_dtype.lanes() > 1) {
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    PrintIndent();
    if (scope == "local.fragment" || scope == "local") {
      stream << GetVectorLocalRef(op->buffer->data.get(), op->indices[0],
                                     "PTO vector store")
             << " = " << value << "\n";
    } else {
      std::string index_str =
          RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
      stream << GetVarID(op->buffer->data.get()) << "[" << index_str
             << "] = " << value << "\n";
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
    ICHECK_EQ(value_dtype.element_of(), buffer_dtype)
        << "PTO SIMT vector BufferStore expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << buffer_dtype;
    if (TryEmitRngBroadcastStore(op)) {
      return;
    }
    ICHECK(!tl::IsAscendVectorizableFP8(buffer_dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 BufferStore currently supports contiguous lanes 2, "
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
      stream << "scalar.store(" << value << ", "
             << GetVarID(op->buffer->data.get()) << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      std::string base = ScalarPointerBase_(
          op->buffer->data.get(), op->buffer->dtype, scope);
      stream << "scalar.store(" << value << ", " << base << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      std::string base = ScalarPointerBase_(
          op->buffer->data.get(), op->buffer->dtype, scope);
      stream << "scalar.store(" << value << ", " << base << ", " << index_str
             << contiguous_suffix << ")\n";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector store scope: " << scope;
  }

  // Non-SIMT vector store to a scalar-element buffer: use scalar.store with
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
      // Non-SIMT local buffers are Python lists.  PTODSL scalar.store does
      // not accept a list as a vector destination, so extract each lane and
      // assign it directly.
      const std::string list_ref = GetVarID(op->buffer->data.get());
      PrintIndent();
      stream << "tl.store_vector_to_list(" << list_ref << ", " << index_str
             << ", " << value << ")\n";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      PrintIndent();
      std::string base =
          ScalarPointerBase_(op->buffer->data.get(), op->buffer->dtype, scope);
      stream << "scalar.store(" << value << ", " << base << ", " << index_str
             << ", contiguous=" << value_dtype.lanes() << ")\n";
      return;
    }
    if (scope == "global" || scope.empty()) {
      PrintIndent();
      std::string base =
          ScalarPointerBase_(op->buffer->data.get(), op->buffer->dtype, scope);
      stream << "scalar.store(" << value << ", " << base << ", " << index_str
             << ", contiguous=" << value_dtype.lanes() << ")\n";
      return;
    }
    LOG(FATAL) << "Unsupported PTO vector store scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector BufferStore is not supported yet";
}

bool CodeGenTileLangPTO::TryEmitRngBroadcastStore(
    const BufferStoreNode *op) {
  const auto *broadcast = op->value.as<BroadcastNode>();
  if (broadcast == nullptr) {
    return false;
  }
  bool contains_rng_draw = false;
  tirx::PostOrderVisit(broadcast->value, [&](const ObjectRef &node) {
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
  ICHECK_EQ(broadcast->dtype.lanes(), 2)
      << "PTO RNG broadcast store currently supports exactly 2 lanes, got "
      << broadcast->dtype;

  PrimExpr base_index = op->indices[0];
  if (const auto *ramp = base_index.as<RampNode>()) {
    CheckContiguousRampStride(op->indices[0], "store");
    base_index = ramp->base;
  }

  arith::Analyzer analyzer;
  for (int64_t lane = 0; lane < 2; ++lane) {
    std::string tmp = name_supply_->FreshName("_tl_rng_value");
    std::string value =
        RemoveOutermostParentheses(PrintExpr_(broadcast->value));
    PrintIndent();
    stream << tmp << " = " << value << "\n";
    PrimExpr lane_index =
        analyzer.Simplify(base_index + make_const(base_index.dtype(), lane));
    PrintIndent();
    EmitScalarStore(op->buffer.get(), tmp, lane_index);
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
  stream << pto_rng_counter_var_
         << " = pto.const(0, dtype=pto.ui64)\n";
  PrintIndent();
  stream << pto_rng_normal_cache_var_
         << " = pto.const(0.0, dtype=pto.f32)\n";
  PrintIndent();
  stream << pto_rng_has_normal_var_
         << " = pto.const(0, dtype=pto.i32)\n";
}

std::string CodeGenTileLangPTO::EmitRngRand(const CallNode *op) {
  ICHECK(pto_rng_initialized_)
      << "tl.rng_rand called without prior tl.rng_init";
  ICHECK(op->dtype.is_uint() && op->dtype.bits() == 32 &&
         op->dtype.is_scalar())
      << "tl.rng_rand expects a scalar uint32 result, got " << op->dtype;
  std::string result_var =
      "_tl_rng_result_" + std::to_string(pto_rng_result_counter_++);
  PrintIndent();
  stream << result_var << ", " << pto_rng_counter_var_ << " = "
         << pto_rng_state_var_ << ".rand(" << pto_rng_counter_var_ << ")\n";
  return ScalarCastExpr(result_var, op->dtype, "PTO rng_rand result");
}

std::string CodeGenTileLangPTO::EmitRngRandFloat(const CallNode *op) {
  ICHECK(pto_rng_initialized_)
      << "tl.rng_rand_float called without prior tl.rng_init";
  ICHECK_NE(op->dtype.bits(), 64)
      << "float64 RNG (tl.rng_rand_float bit=64) is not supported on PTO";
  ICHECK_EQ(op->args.size(), 1U)
      << "tl.rng_rand_float expects 1 argument (distribution)";
  std::string dist = Downcast<StringImm>(op->args[0])->value;
  ICHECK(dist == "uniform" || dist == "normal")
      << "Unsupported RNG distribution on PTO: " << dist;
  std::string result_var =
      "_tl_rng_result_" + std::to_string(pto_rng_result_counter_++);
  PrintIndent();
  if (dist == "uniform") {
    stream << result_var << ", " << pto_rng_counter_var_ << " = "
           << pto_rng_state_var_ << ".rand_uniform("
           << pto_rng_counter_var_ << ")\n";
  } else {
    stream << result_var << ", " << pto_rng_counter_var_ << ", "
           << pto_rng_normal_cache_var_ << ", " << pto_rng_has_normal_var_
           << " = " << pto_rng_state_var_ << ".rand_normal("
           << pto_rng_counter_var_ << ", " << pto_rng_normal_cache_var_
           << ", " << pto_rng_has_normal_var_ << ")\n";
  }
  return result_var;
}

} // namespace codegen
} // namespace tvm
