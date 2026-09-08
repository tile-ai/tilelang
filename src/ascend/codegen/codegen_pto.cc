/*!
 * \file ascend/codegen/codegen_pto.cc
 * \brief Utility to generate PTO Python source.
 */
#include "ascend/codegen/codegen_pto.h"

#include "backend/common/codegen/codegen_utils.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"
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
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using namespace tirx;

namespace {

std::string PtoFP8TypeName(DataType t) {
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

std::string PtoTypeName(DataType t) {
  if (tl::IsAscendVectorizableFP8(t))
    return PtoFP8TypeName(t);

  // PTO stores E2M1 as one byte containing a pair of logical FP4 values.
  // TileLang expresses FP4 logically; PTO stores each pair as one f4e2m1x2
  // element.
  if (t.is_float4_e2m1fn()) {
    return "pto.f4e2m1x2";
  }
  ICHECK(t.is_scalar()) << "PTO scalar type expected, got " << t;
  // pto.i1 is the logical predicate type used by PTOAS SSA expressions. It
  // is not the physical representation of a bool element in GM; pointer
  // types use the byte-backed mapping in PtoPointerElementTypeName below.
  if (t.is_bool())
    return "pto.i1";
  if (t.is_float()) {
    if (t.bits() == 32)
      return "pto.f32";
    if (t.bits() == 16)
      return "pto.f16";
  } else if (t.is_bfloat16()) {
    return "pto.bf16";
  } else if (t.is_uint()) {
    if (t.bits() == 64)
      return "pto.ui64";
    if (t.bits() == 32)
      return "pto.ui32";
    if (t.bits() == 16)
      return "pto.ui16";
    if (t.bits() == 8)
      return "pto.ui8";
  } else if (t.is_int()) {
    if (t.bits() == 64)
      return "pto.si64";
    if (t.bits() == 32)
      return "pto.si32";
    if (t.bits() == 16)
      return "pto.si16";
    if (t.bits() == 8)
      return "pto.si8";
    if (t.bits() == 1)
      return "pto.i1";
  }
  LOG(FATAL) << "Unsupported PTO type: " << t;
  return "";
}

std::string PtoPointerElementTypeName(DataType t) {
  // GM bool buffers use one byte per element. Keep their pointer payload as
  // pto.i8 because PTOAS device scalar loads/stores require integer payloads.
  if (t.is_bool())
    return "pto.i8";
  return PtoTypeName(t);
}

std::string PtoSignedIntegerTypeName(DataType t) {
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

std::string StripPipePrefix(const std::string &name) {
  if (name.rfind("PIPE_", 0) == 0) {
    return name.substr(5);
  }
  return name;
}

DataType ParsePTODtype(const std::string &dtype_name) {
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
  if (dtype_name == "uint8")
    return DataType::UInt(8);
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

std::string PtoIntraBlockEventId(const PrimExpr &event_id,
                                 const std::string &printed_event_id) {
  int64_t value = 0;
  if (TryGetConstInt(event_id, &value)) {
    return std::to_string(value);
  }
  return "scalar.index_cast(" + printed_event_id + ")";
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.rfind(prefix, 0) == 0;
}

bool IsFloat32(DataType t) {
  return t.is_float() && t.bits() == 32 && t.lanes() == 1;
}

bool IsFloat32Pair(DataType t) {
  return t.is_float() && t.bits() == 32 && t.lanes() == 2;
}

bool IsPTOInteger32(DataType t) {
  return t.is_scalar() && t.bits() == 32 && (t.is_int() || t.is_uint());
}

constexpr const char *kPtoModeZeroing = "MODE_ZEROING";
constexpr const char *kPtoModeMerging = "MODE_MERGING";

void RejectPtoSimdMerging(const CallNode *op) {
  auto call_op = op->op.as<Op>();
  if (!call_op.has_value() || !StartsWith(call_op.value()->name, "tl.simd.") ||
      op->args.empty()) {
    return;
  }
  const auto *mode = op->args.back().as<StringImmNode>();
  if (mode != nullptr && mode->value == kPtoModeMerging) {
    LOG(FATAL) << "PTO codegen does not support MODE_MERGING for "
               << call_op.value()->name
               << ": physical pto.* SIMD operations have no merge-mode "
                  "operand; use tl.vmi.* with pmode=\"merge\" or lower the "
                  "merge into an explicit read-modify-write sequence";
  }
}

void CheckPtoSimdMode(const CallNode *op, size_t mode_index,
                      const char *op_name) {
  ICHECK_LT(mode_index, op->args.size())
      << op_name << " is missing its mode argument";
  const auto *mode = op->args[mode_index].as<StringImmNode>();
  ICHECK(mode) << op_name << " mode must be a constant string";
  ICHECK_EQ(mode->value, kPtoModeZeroing)
      << "PTO codegen currently supports only MODE_ZEROING for " << op_name
      << "; MODE_MERGING is not representable by the physical pto.* op";
}

void CheckPtoVdivPrecision(const CallNode *op, bool enable_fast_math) {
  if (!IsFloat32(op->dtype.element_of())) {
    return;
  }

  if (auto precision = op->annotations.Get("precision")) {
    const auto *value = precision.value().as<IntImmNode>();
    ICHECK(value)
        << "tl.simd.vdiv precision annotation must be an integer code";
    ICHECK_EQ(value->value, 0)
        << "PTO codegen cannot lower precise tl.simd.vdiv: PTOAS "
           "pto.vdiv has no precision operand; vector high-precision vdiv "
           "requires a newer PTOAS version; use "
           "precision=\"ftz_true\" for the vector hardware operation";
    return;
  }

  ICHECK(enable_fast_math)
      << "PTO codegen cannot lower precise default tl.simd.vdiv: fast math "
         "is disabled; vector high-precision vdiv requires a newer PTOAS "
         "version. Use precision=\"ftz_true\" or enable "
         "tl.enable_fast_math";
}

bool IsPTOLocalVarScalarDtype(DataType dtype) {
  return dtype.is_int() || dtype.is_uint() || IsFloat32(dtype) ||
         dtype.is_handle();
}

std::string PtoLocalVarInitialValue(DataType dtype) {
  if (dtype.is_int() || dtype.is_uint()) {
    return "pto.const(0, dtype=pto.int64)";
  }
  return "pto.const(0, dtype=" + PtoTypeName(dtype) + ")";
}

std::string PtoLocalVarStoreValue(DataType dtype, const std::string &value) {
  if (dtype.is_handle()) {
    return value;
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return "_tl_wrap_surface_value(_tl_coerce_i64(" + value +
           ", context=\"PTO local.var store\"))";
  }
  return "scalar.cast(" + value + ", " + PtoTypeName(dtype) + ")";
}

bool IsSupportedPTOFloatMinMaxType(DataType t) {
  if (t.is_scalar())
    return t.is_float16() || IsFloat32(t) || t.is_bfloat16();
  return t.lanes() == 2 &&
         (t.is_float16() || IsFloat32Pair(t) || t.is_bfloat16());
}

bool IsSupportedPTOSIMTUnaryMathType(DataType t) {
  const bool scalar_or_pair = t.is_scalar() || t.lanes() == 2;
  return scalar_or_pair && (t.is_float16() || (t.is_float() && t.bits() == 32));
}

bool IsSupportedPTOScalarUnaryMathType(DataType t) {
  return t.is_scalar() && (t.is_float16() || IsFloat32(t) || t.is_bfloat16());
}

bool IsSupportedPTOSIMTAllReduceType(DataType t) {
  return t.is_scalar() && (t.is_float16() || IsFloat32(t) ||
                           t == DataType::Int(32) || t == DataType::UInt(32));
}

bool IsSupportedSIMTLocalScalarAccessType(DataType t) {
  return t.is_scalar() && (t.is_float16() || IsFloat32(t) ||
                           t == DataType::Int(32) || t == DataType::UInt(32));
}

bool IsSupportedSIMTLocalStorageType(DataType t) {
  return t.is_scalar() && (t.is_float16() || IsFloat32(t) || t.is_bfloat16() ||
                           t == DataType::Int(32) || t == DataType::UInt(32) ||
                           tl::IsAscendVectorizableFP8(t));
}

std::string PtoSIMTLocalStorageTypeName(DataType t) {
  ICHECK(IsSupportedSIMTLocalStorageType(t));
  // LLVM stack allocations require signless integer element types. Preserve
  // TileLang's signedness at load/store boundaries instead.
  return IsPTOInteger32(t) ? "pto.i32" : PtoTypeName(t);
}

bool IsSupportedSIMTFP8ContiguousLaneCount(int lanes) {
  return lanes == 2 || lanes == 4 || lanes == 8;
}

bool IsPowerOfTwo(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

int64_t PtoPhysicalElementBits(DataType dtype) {
  // The dcache-bypass instructions operate on physical GM element width. A
  // logical bool is therefore counted as its byte-backed storage width.
  if (dtype.is_bool())
    return 8;
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

void ValidatePtoGmBypassDtype(DataType dtype) {
  // PTOAS ld_dev/st_dev accept only 1/2/4/8-byte scalar integer payloads;
  // floating-point values are handled by same-width bitcasts in the wrapper.
  ICHECK(dtype.is_scalar())
      << "PTO GM dcache bypass expects a scalar dtype, got " << dtype;
  const int64_t bits = PtoPhysicalElementBits(dtype);
  ICHECK(bits == 8 || bits == 16 || bits == 32 || bits == 64)
      << "PTO GM dcache bypass only supports 1/2/4/8-byte scalar types, got "
      << dtype;
}

bool IntegerLiteralFits(DataType dtype, int64_t value) {
  if (!dtype.is_scalar()) {
    return false;
  }
  if (dtype.is_int()) {
    if (dtype.bits() >= 63) {
      return true;
    }
    const int64_t min_value = -(1LL << (dtype.bits() - 1));
    const int64_t max_value = (1LL << (dtype.bits() - 1)) - 1;
    return value >= min_value && value <= max_value;
  }
  if (dtype.is_uint()) {
    if (value < 0) {
      return false;
    }
    if (dtype.bits() >= 63) {
      return true;
    }
    const int64_t max_value = (1LL << dtype.bits()) - 1;
    return value <= max_value;
  }
  return false;
}

void PrintIntegerLiteralForRuntimeBinary(int64_t value, DataType anchor_dtype,
                                         std::ostream &os) {
  if (IntegerLiteralFits(anchor_dtype, value)) {
    os << value;
    return;
  }
  os << "pto.const(" << value << ", dtype=pto.i64)";
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

PrimExpr NormalizePackedFP4Index(const PrimExpr &index, DataType elem_dtype,
                                 const char *pointer_kind) {
  if (!elem_dtype.is_float4_e2m1fn() || index.dtype().lanes() == 1) {
    return index;
  }

  const auto *ramp = index.as<RampNode>();
  ICHECK(ramp) << "PTO packed FP4 " << pointer_kind
               << " expects a contiguous Ramp index, got " << index;
  PrimExpr stride = arith::Analyzer().Simplify(ramp->stride);
  int64_t stride_value = 0;
  ICHECK(TryGetConstInt(stride, &stride_value) && stride_value == 1)
      << "PTO packed FP4 " << pointer_kind
      << " requires a unit-stride Ramp index, got " << ramp->stride;

  const int lanes = index.dtype().lanes();
  return arith::Analyzer().Simplify(
      floordiv(ramp->base, make_const(ramp->base.dtype(), lanes)));
}

void CheckConstZero(const PrimExpr &expr, const char *name) {
  int64_t value = 0;
  ICHECK(TryGetConstInt(expr, &value) && value == 0)
      << "PTO codegen currently only supports " << name
      << " == 0 for tl.ascend_copy_gm_to_ubuf, got " << expr;
}

std::string PtoStoreL2CacheToken(int64_t value) {
  switch (value) {
  case 0:
    return "nmfv";
  case 1:
    return "nmlv";
  case 2:
    return "nmprs";
  case 3:
    return "nmred";
  case 4:
    return "naci";
  case 5:
    return "napw";
  case 6:
    return "napi";
  case 7:
    return "nared";
  case 8:
    return "wbhfv";
  case 9:
    return "wbhlv";
  case 10:
    return "wbhprs";
  case 11:
    return "wbhred";
  case 12:
    return "wtsfv";
  case 13:
    return "wtslv";
  case 14:
    return "wtsprs";
  case 15:
    return "wtsred";
  default:
    LOG(FATAL) << "Unsupported PTO store l2 cache control value: " << value;
    return "nmfv";
  }
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

void CheckPTOLocalVarBuffer(const BufferNode *buffer) {
  ICHECK_EQ(buffer->shape.size(), 1U)
      << "PTO local.var only supports scalar alloc_var buffers, got rank "
      << buffer->shape.size();
  int64_t extent = 0;
  ICHECK(TryGetConstInt(buffer->shape[0], &extent) && extent == 1)
      << "PTO local.var only supports scalar alloc_var buffers with shape "
         "(1,), got "
      << buffer->shape[0];

  DataType dtype = buffer->dtype;
  if (dtype.lanes() > 1) {
    // VMI registers are logical vectors; integer, floating-point and
    // low-precision element types are all valid here.
    return;
  }
  ICHECK(IsPTOLocalVarScalarDtype(dtype))
      << "PTO local.var only supports integer, floating-point, or handle "
         "scalar values, got "
      << dtype;
}

void CheckPTOAllReduceDtype(const CallNode *call) {
  ICHECK_GE(call->args.size(), 2U)
      << "tl::AscendAllReduce call expects a value argument";
  DataType dtype = call->args[1].dtype();
  ICHECK(IsSupportedPTOSIMTAllReduceType(dtype))
      << "PTO cross-thread allreduce currently supports float16, float32, "
         "int32, and uint32 only, got "
      << dtype
      << ". Use a supported reducer dtype, such as float32, and cast the "
         "finalized result afterward.";
}

void CheckPTOKernel(const PrimFunc &func) {
  bool has_cube_block = false;
  bool has_supported_cube_op = false;
  bool has_unsupported_cube_op = false;

  PostOrderVisit(func->body, [&](const ffi::ObjectRef &node) {
    if (const auto *block = node.as<SBlockNode>()) {
      if (block->name_hint == "CUBE") {
        has_cube_block = true;
      }
    }

    if (const auto *call = node.as<CallNode>()) {
      if ((call->op.same_as(builtin::call_extern()) ||
           call->op.same_as(builtin::call_pure_extern())) &&
          !call->args.empty()) {
        const auto *func_name = call->args[0].as<StringImmNode>();
        if (func_name != nullptr &&
            func_name->value.find("tl::AscendAllReduce") != std::string::npos) {
          CheckPTOAllReduceDtype(call);
        }
      }
      if (call->op.same_as(tl::ascend_gemm_l1()) ||
          call->op.same_as(tl::ascend_blockscaled_gemm_l1()) ||
          call->op.same_as(tl::ascend_mad()) ||
          call->op.same_as(tl::ascend_mad_mx())) {
        has_supported_cube_op = true;
      } else if (call->op.same_as(tl::ascend_nd2nz_scatter()) ||
                 call->op.same_as(tl::ascend_nd2nz_post_copy())) {
        has_unsupported_cube_op = true;
      }
    }
  });

  ICHECK(!has_cube_block || has_supported_cube_op)
      << "PTO codegen found a cube block without a supported cube operation";
  ICHECK(!has_unsupported_cube_op)
      << "PTO codegen does not support this Ascend cube op yet.";
}

std::string AccStoreUnitFlagArg(int64_t unit_flag_ctrl) {
  if (unit_flag_ctrl == 0)
    return "";
  if (unit_flag_ctrl == 2)
    return "pto.AccStoreUnitFlagCtrl.CHECK_ONLY";
  if (unit_flag_ctrl == 3)
    return "pto.AccStoreUnitFlagCtrl.CHECK_AND_CLEAR";
  LOG(FATAL) << "PTO L0C store unsupported unit_flag_ctrl=" << unit_flag_ctrl;
  return "";
}

std::string MadUnitFlagArg(int64_t unit_flag_ctrl) {
  if (unit_flag_ctrl == 0)
    return "";
  if (unit_flag_ctrl == 2)
    return "pto.MadUnitFlagMode.CHECK_ONLY";
  if (unit_flag_ctrl == 3)
    return "pto.MadUnitFlagMode.CHECK_AND_SET";
  LOG(FATAL) << "PTO MAD unsupported unit_flag_ctrl=" << unit_flag_ctrl;
  return "";
}

bool IsSupportedPTOGemmInputDtype(DataType dtype) {
  return IsFloat32(dtype) || dtype.is_float16() || dtype.is_bfloat16() ||
         dtype.is_float8_e4m3fn() || dtype.is_float4_e2m1fn();
}

bool IsSamePTOStorageDtype(DataType lhs, DataType rhs) {
  return lhs == rhs || (lhs.is_float4_e2m1fn() && rhs.is_float4_e2m1fn());
}

bool IsPTOPackedFP4Storage(DataType dtype) {
  return dtype.is_float4_e2m1fn() && dtype.lanes() == 2;
}

int64_t PTOStorageElementBytes(DataType dtype) {
  if (IsPTOPackedFP4Storage(dtype)) {
    return static_cast<int64_t>(dtype.bits()) * dtype.lanes() / 8;
  }
  ICHECK(dtype.is_scalar())
      << "PTO storage dtype must be scalar or packed FP4 x2, got " << dtype;
  return dtype.bytes();
}

bool IsPTOStorageDtypeCompatible(DataType lhs, DataType rhs) {
  // Reuse the canonical storage-equivalence rule, including FP4 packed
  // storage whose lane metadata may differ between GM and UB endpoints.
  if (IsSamePTOStorageDtype(lhs, rhs)) {
    return true;
  }
  if (!lhs.is_scalar() || !rhs.is_scalar()) {
    return false;
  }
  // Signed and unsigned integer payloads are interchangeable when their
  // physical element widths match.
  return lhs.bits() == rhs.bits() && (lhs.is_int() || lhs.is_uint()) &&
         (rhs.is_int() || rhs.is_uint());
}

int64_t PTOGemmInputPackFactor(DataType dtype) {
  return dtype.is_float4_e2m1fn() ? 2 : 1;
}

// L1/L0A/L0B matrices use a fixed 16-row fractal. The K fractal depends on
// the element width and is obtained from PTOGemmInputC0.
constexpr int64_t kAscendMFractal = 16;

int64_t PTOGemmInputC0(DataType dtype) {
  ICHECK(IsSupportedPTOGemmInputDtype(dtype))
      << "PTO GEMM L1 helper currently only supports float32, float16, "
         "bfloat16, float8_e4m3fn, or float4_e2m1fn inputs, got "
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
          return ParsePTODtype(dtype_name->value);
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

  return fallback_dtype;
}

DataType GetAscendMadInputDtype(const CallNode *call) {
  ICHECK_EQ(call->args.size(), 10U)
      << "tl.ascend_mad expects exactly 10 arguments";
  return GetAnnotatedPointerDtype(
      call->args[1],
      GetAnnotatedPointerDtype(call->args[2], DataType::BFloat(16)));
}

void EmitPTOHf32ModeArgument(std::ostream &os, int64_t hf32_mode) {
  if (hf32_mode == 0) {
    return;
  }
  ICHECK(hf32_mode == 1 || hf32_mode == 2)
      << "PTO HF32 mode must be 0, 1, or 2, got " << hf32_mode;
  // PTO names the hardware bit-47 mode ROUND_AWAY; CANN names the same
  // setting HF32TransMode::NEAREST_ZERO.
  os << ", tf32_mode="
     << (hf32_mode == 1 ? "pto.Tf32Mode.ROUND_AWAY"
                        : "pto.Tf32Mode.ROUND_EVEN");
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
    for (const auto &[call, modes] : analyzer.cube_call_modes_) {
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
      if (IsFloat32(ParsePTODtype(dtype_name->value))) {
        cube_call_modes_[GetRef<Call>(call)] |= state;
      }
    } else if (call->op.same_as(tl::ascend_mad())) {
      if (IsFloat32(GetAscendMadInputDtype(call))) {
        cube_call_modes_[GetRef<Call>(call)] |= state;
      }
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

  std::unordered_map<Call, uint8_t, ObjectPtrHash, ObjectPtrEqual>
      cube_call_modes_;
  arith::Analyzer analyzer_;
};

bool IsSupportedPTOCopyPadDtype(DataType dtype) {
  return dtype.is_scalar() &&
         (dtype.is_bfloat16() ||
          (dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
          ((dtype.is_int() || dtype.is_uint()) &&
           (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32)));
}

// Resolve the TIR pad-register setter that reaches each padded GM->UB copy.
// This is a codegen binding analysis only: PTOAS remains responsible for
// lowering the pad requirement to hardware setters and optimizing their use.
struct PTOPadValueAnalysis {
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      binding_by_copy;
  std::unordered_map<Call, int64_t, ObjectPtrHash, ObjectPtrEqual>
      binding_by_setter;
  std::unordered_map<int64_t, DataType> dtype_by_binding;
};

struct PTOPadState {
  enum class Kind : uint8_t { kUnknown, kKnown, kConflict };

  Kind kind{Kind::kUnknown};
  Call setter;
  int64_t binding_id{-1};
  DataType dtype;
};

bool SamePTOPadState(const PTOPadState &lhs, const PTOPadState &rhs) {
  if (lhs.kind != rhs.kind) {
    return false;
  }
  if (lhs.kind != PTOPadState::Kind::kKnown) {
    return true;
  }
  return lhs.setter.same_as(rhs.setter) && lhs.binding_id == rhs.binding_id &&
         lhs.dtype == rhs.dtype;
}

PTOPadState MergePTOPadStates(const PTOPadState &lhs, const PTOPadState &rhs) {
  // A copy cannot safely name one branch-local Python binding unless both
  // paths carry the exact same setter definition.
  if (lhs.kind == PTOPadState::Kind::kConflict ||
      rhs.kind == PTOPadState::Kind::kConflict) {
    return {PTOPadState::Kind::kConflict, {}, -1, DataType::Void()};
  }
  if (lhs.kind == PTOPadState::Kind::kUnknown ||
      rhs.kind == PTOPadState::Kind::kUnknown) {
    return {PTOPadState::Kind::kUnknown, {}, -1, DataType::Void()};
  }
  if (SamePTOPadState(lhs, rhs)) {
    return lhs;
  }
  return {PTOPadState::Kind::kConflict, {}, -1, DataType::Void()};
}

class PTOPadValueAnalyzer final
    : public StmtFunctor<PTOPadState(const Stmt &, PTOPadState)> {
public:
  static PTOPadValueAnalysis Analyze(const PrimFunc &func) {
    PTOPadValueAnalyzer analyzer;
    analyzer.VisitStmt(func->body, PTOPadState{});
    return std::move(analyzer.result_);
  }

private:
  int64_t RegisterSetter(const Call &setter) {
    auto it = result_.binding_by_setter.find(setter);
    if (it != result_.binding_by_setter.end()) {
      return it->second;
    }
    ICHECK_EQ(setter->args.size(), 1U)
        << "tl.ascend_set_copy_pad_value expects exactly 1 argument";
    DataType dtype = setter->args[0].dtype();
    ICHECK(IsSupportedPTOCopyPadDtype(dtype))
        << "PTO copy padding supports scalar int/uint8/16/32, float16, "
           "bfloat16, and float32 values, got "
        << dtype;
    const int64_t id = next_binding_id_++;
    result_.binding_by_setter.emplace(setter, id);
    result_.dtype_by_binding.emplace(id, dtype);
    return id;
  }

  PTOPadState AnalyzeLoop(const Stmt &body, const PTOPadState &incoming,
                          bool must_execute) {
    // Use a small finite-state fixed point so loop-carried setters are only
    // accepted when the same reaching definition is stable on every iteration.
    PTOPadState header = incoming;
    constexpr int kMaxIterations = 128;
    for (int i = 0; i < kMaxIterations; ++i) {
      PTOPadState body_out = VisitStmt(body, header);
      PTOPadState next =
          must_execute ? body_out : MergePTOPadStates(incoming, body_out);
      if (SamePTOPadState(next, header)) {
        return next;
      }
      header = next;
    }
    LOG(FATAL)
        << "PTO copy pad-value analysis failed to converge for loop body "
        << body;
    return incoming;
  }

  PTOPadState VisitStmt_(const BindNode *op, PTOPadState state) final {
    (void)op;
    return state;
  }

  PTOPadState VisitStmt_(const AttrStmtNode *op, PTOPadState state) final {
    return VisitStmt(op->body, state);
  }

  PTOPadState VisitStmt_(const IfThenElseNode *op, PTOPadState state) final {
    PTOPadState then_out = VisitStmt(op->then_case, state);
    PTOPadState else_out =
        op->else_case ? VisitStmt(op->else_case.value(), state) : state;
    return MergePTOPadStates(then_out, else_out);
  }

  PTOPadState VisitStmt_(const ForNode *op, PTOPadState state) final {
    arith::Analyzer analyzer;
    bool must_execute = analyzer.CanProve(op->extent > 0);
    return AnalyzeLoop(op->body, state, must_execute);
  }

  PTOPadState VisitStmt_(const WhileNode *op, PTOPadState state) final {
    return AnalyzeLoop(op->body, state, false);
  }

  PTOPadState VisitStmt_(const AllocBufferNode *op, PTOPadState state) final {
    (void)op;
    return state;
  }

  PTOPadState VisitStmt_(const DeclBufferNode *op, PTOPadState state) final {
    (void)op;
    return state;
  }

  PTOPadState VisitStmt_(const BufferStoreNode *op, PTOPadState state) final {
    (void)op;
    return state;
  }

  PTOPadState VisitStmt_(const AssertStmtNode *op, PTOPadState state) final {
    (void)op;
    return state;
  }

  PTOPadState VisitStmt_(const SeqStmtNode *op, PTOPadState state) final {
    for (const Stmt &stmt : op->seq) {
      state = VisitStmt(stmt, state);
    }
    return state;
  }

  PTOPadState VisitStmt_(const EvaluateNode *op, PTOPadState state) final {
    const auto *call_node = op->value.as<CallNode>();
    if (call_node == nullptr) {
      return state;
    }
    Call call = GetRef<Call>(call_node);
    if (call->op.same_as(tl::ascend_set_copy_pad_value())) {
      const int64_t id = RegisterSetter(call);
      return {PTOPadState::Kind::kKnown, call, id, call->args[0].dtype()};
    }
    if (!call->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
      return state;
    }

    ICHECK_EQ(call->args.size(), 11U)
        << "tl.ascend_copy_gm_to_ubuf expects exactly 11 arguments";
    int64_t data_select = 0;
    ICHECK(TryGetConstInt(call->args[7], &data_select) &&
           (data_select == 0 || data_select == 1))
        << "PTO GM->UB MTE requires dataSelect to be the constant 0 or 1, got "
        << call->args[7];
    if (data_select == 1) {
      ICHECK(state.kind == PTOPadState::Kind::kKnown)
          << "PTO GM->UB MTE with dataSelect == 1 requires a unique preceding "
             "tl.ascend_set_copy_pad_value on every control-flow path";
      result_.binding_by_copy[call] = state.binding_id;
    }
    return state;
  }

  PTOPadState VisitStmt_(const SBlockNode *op, PTOPadState state) final {
    if (op->init.defined()) {
      state = VisitStmt(op->init.value(), state);
    }
    return VisitStmt(op->body, state);
  }

  PTOPadState VisitStmt_(const SBlockRealizeNode *op, PTOPadState state) final {
    PTOPadState block_out = VisitStmt(op->block, state);
    if (is_one(op->predicate)) {
      return block_out;
    }
    if (is_zero(op->predicate)) {
      return state;
    }
    return MergePTOPadStates(state, block_out);
  }

  PTOPadValueAnalysis result_;
  int64_t next_binding_id_{0};
};

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

bool UsesVar(const Stmt &body, const VarNode *target) {
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (node.get() == target) {
      found = true;
    }
  });
  return found;
}

} // namespace

CodeGenTileLangPTO::CodeGenTileLangPTO() {
  auto pass_ctx = tvm::transform::PassContext::Current();
  enable_fast_math_ =
      pass_ctx->GetConfig<Bool>(tl::kEnableFastMath, Bool(false)).value();
}

void CodeGenTileLangPTO::AddFunction(const GlobalVar &gvar,
                                     const PrimFunc &func) {
  RegisterFunction_(gvar, func);
  current_function_name_ = GetFunctionName_(gvar);
  InitFuncState_(func);
  CheckPTOKernel(func);
  fragment_info_.clear();
  local_var_buffers_.clear();
  inside_simtvf_body_ = false;
  inside_dynamic_control_flow_ = 0;
  rng_state_var_.clear();
  persistent_buffer_vars_ = SimtPersistentBufferCollector().Collect(func->body);
  current_function_has_mixed_sections_ = false;
  int64_t cube_section_count = 0;
  int64_t vector_section_count = 0;
  int64_t vector_count = 2;
  std::optional<int64_t> cthread_extent;
  std::optional<int64_t> inconsistent_cthread_extent;
  bool has_nonconstant_cthread_extent = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>()) {
      const auto *iter_var = attr->node.as<IterVarNode>();
      if (attr->attr_key == tirx::attr::thread_extent && iter_var != nullptr &&
          iter_var->thread_tag == "cthread") {
        const auto *extent = attr->value.as<IntImmNode>();
        if (extent == nullptr) {
          has_nonconstant_cthread_extent = true;
        } else if (cthread_extent.has_value()) {
          if (cthread_extent.value() != extent->value) {
            inconsistent_cthread_extent = extent->value;
          }
        } else {
          cthread_extent = extent->value;
        }
      }
    }
    if (const auto *block = node.as<SBlockNode>()) {
      if (block->name_hint == "CUBE") {
        ++cube_section_count;
      } else if (block->name_hint == "VECTOR") {
        ++vector_section_count;
        if (auto count_annotation = block->annotations.Get("vector_count")) {
          const auto *count = count_annotation.value().as<IntImmNode>();
          ICHECK(count != nullptr) << "PTO VECTOR section vector_count must be "
                                      "an integer, got "
                                   << count_annotation.value();
          vector_count = count->value;
        }
      }
    }
  });
  ICHECK_LE(cube_section_count, 1)
      << "PTO codegen currently supports at most one CUBE section per function";
  ICHECK_LE(vector_section_count, 1) << "PTO codegen currently supports at "
                                        "most one VECTOR section per function";
  const bool has_cube_section = cube_section_count != 0;
  const bool has_vector_section = vector_section_count != 0;
  const bool has_mixed_sections = has_cube_section && has_vector_section;
  if (has_mixed_sections &&
      (cthread_extent.has_value() || has_nonconstant_cthread_extent)) {
    ICHECK(!has_nonconstant_cthread_extent)
        << "PTO cthread extent must be a constant integer";
    ICHECK(!inconsistent_cthread_extent.has_value())
        << "PTO function contains inconsistent cthread extents ("
        << cthread_extent.value() << " and "
        << inconsistent_cthread_extent.value() << ")";
    ICHECK_EQ(cthread_extent.value(), 2)
        << "PTO backend cannot express __mix__(1, " << cthread_extent.value()
        << "); PTOAS currently supports only AIV count 2";
    ICHECK_EQ(cthread_extent.value(), vector_count)
        << "PTO VECTOR section cthread extent must match vector_count; got "
        << cthread_extent.value() << " and " << vector_count;
  }
  if (has_mixed_sections) {
    ICHECK_EQ(vector_count, 2)
        << "PTO mixed kernel currently supports only T.Vector(vector=2); "
           "T.Vector(vector=1) requires PTOAS AIV-count support";
  }
  current_function_has_mixed_sections_ = has_mixed_sections;
  gemm_emit_ctx_ = PTOGemmEmitContext();
  hf32_mode_by_cube_call_ = PTOHf32ModeAnalyzer::Analyze(func);
  PTOPadValueAnalysis pad_analysis = PTOPadValueAnalyzer::Analyze(func);
  pad_binding_by_copy_ = std::move(pad_analysis.binding_by_copy);
  pad_binding_by_setter_ = std::move(pad_analysis.binding_by_setter);
  pad_binding_dtype_by_id_ = std::move(pad_analysis.dtype_by_binding);
  blockscaled_gemm_emit_ctx_ = PTOBlockscaledGemmEmitContext();
  current_function_has_gemm_ = HasAscendGemm(func);
  has_gemm_l1_ = has_gemm_l1_ || HasAscendGemmL1(func);

  PrintFuncDecorator_(stream);
  PrintFunctionSignature_(current_function_name_, func, stream);
  stream << ":\n";
  int func_scope = BeginScope();
  if (current_function_has_gemm_) {
    const CallNode *first_gemm = nullptr;
    const CallNode *first_blockscaled_gemm = nullptr;
    tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (first_gemm == nullptr && call->op.same_as(tl::ascend_gemm_l1())) {
          first_gemm = call;
        } else if (first_blockscaled_gemm == nullptr &&
                   call->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
          first_blockscaled_gemm = call;
        }
      }
    });
    if (first_gemm != nullptr) {
      EnsurePTOGemmHelper(first_gemm);
    }
    if (first_blockscaled_gemm != nullptr) {
      EnsurePTOBlockscaledGemmHelper(first_blockscaled_gemm);
    }
  }
  PrintStmt_(func->body);
  EndScope(func_scope);
  stream << "\n";
}

std::string CodeGenTileLangPTO::Finish() {
  std::ostringstream code;
  code << "from ptodsl import pto, scalar\n";
  // These wrappers adapt TileLang logical dtypes to PTOAS ld_dev/st_dev,
  // which are the AICore scalar operations that bypass the GM data cache.
  code << "from tilelang.contrib.ptodsl.dcache_bypass import (\n"
          "  pto_read_gm_bypass_dcache as _tl_pto_read_gm_bypass_dcache,\n"
          "  pto_write_gm_bypass_dcache as _tl_pto_write_gm_bypass_dcache,\n"
          ")\n";
  code << "from tilelang.contrib.ptodsl.simt import (\n"
          "  scalar_div as _tl_scalar_div,\n"
          "  scalar_rsqrt as _tl_scalar_rsqrt,\n"
          "  simt_allreduce_max as _tl_simt_allreduce_max,\n"
          "  simt_allreduce_min as _tl_simt_allreduce_min,\n"
          "  simt_allreduce_sum as _tl_simt_allreduce_sum,\n"
          "  vectorize_binary_f32x2 as _tl_vectorize_binary_f32x2,\n"
          "  vectorize_unary_f32x2 as _tl_vectorize_unary_f32x2,\n"
          ")\n";
  code << "from ptodsl._ops import _coerce_i64 as _tl_coerce_i64\n";
  code << "from ptodsl._surface_values import wrap_surface_value as "
          "_tl_wrap_surface_value\n";
  if (has_gemm_l1_) {
    code << "from tilelang.contrib.ptodsl.gemm import PTOGemmL1Template, "
            "PTOBlockscaledGemmL1Template\n";
  }
  if (uses_rng_) {
    code << "from tilelang.contrib.ptodsl.rng import PhiloxRNG\n";
  }
  code << "\n";
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangPTO::PrintFuncDecorator_(std::ostream &os) { // NOLINT(*)
  // Leave kernel_kind inferred so PTOAS can split explicit CUBE/VECTOR
  // sections into their corresponding backend children.
  auto kernel_kind = "vector";
  if (current_function_has_gemm_) {
    kernel_kind = "cube";
  }
  os << "@pto.jit(name=\"" << current_function_name_ << "\"";
  if (!current_function_has_mixed_sections_) {
    os << ", kernel_kind=\"" << kernel_kind << "\"";
  }
  os << ", target=\"a5\", mode=\"explicit\"";
  if (current_function_has_gemm_) {
    os << ", insert_sync=False";
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
      os << ": " << PtoPtrType(buffer->dtype, "gm");
    } else if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        auto pto_space = PtoSpaceForStorageScope(ptr->storage_scope);
        ICHECK(pto_space.has_value())
            << "Unsupported PTO pointer storage scope: " << ptr->storage_scope;
        os << ": " << PtoPtrType(prim->dtype, *pto_space);
      } else {
        os << ": " << PtoScalarType(v->dtype);
      }
    } else {
      os << ": " << PtoScalarType(v->dtype);
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

std::string CodeGenTileLangPTO::PtoScalarType(DataType t) const {
  return PtoTypeName(t);
}

std::string CodeGenTileLangPTO::PtoPtrType(DataType t,
                                           const std::string &space) const {
  std::ostringstream os;
  os << "pto.ptr(" << PtoPointerElementTypeName(t) << ", \"" << space << "\")";
  return os.str();
}

bool CodeGenTileLangPTO::NeedsPtoCastptr_(const VarNode *buffer_var,
                                          DataType elem_dtype) const {
  // Bypass accesses are lowered through an integer payload pointer. In
  // particular, bool GM storage is i8 while its expression type is i1.
  DataType storage_dtype = elem_dtype.is_bool() ? DataType::Int(8) : elem_dtype;
  return !HandleTypeMatch_(buffer_var, storage_dtype);
}

std::string CodeGenTileLangPTO::PtoScalarPointerBase_(
    const VarNode *buffer_var, DataType elem_dtype, const std::string &scope) {
  // address_of(BufferLoad) carries the logical element dtype, but PTOAS
  // ld_dev/st_dev need a typed GM pointer for the physical payload width.
  std::string base = GetVarID(buffer_var);
  if (scope == "global" || scope.empty()) {
    if (NeedsPtoCastptr_(buffer_var, elem_dtype)) {
      base = "pto.castptr(" + base + ", " + PtoPtrType(elem_dtype, "gm") + ")";
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

bool CodeGenTileLangPTO::HasAscendGemm(const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_gemm_l1()) ||
              call->op.same_as(tl::ascend_blockscaled_gemm_l1()) ||
              call->op.same_as(tl::ascend_mad()) ||
              call->op.same_as(tl::ascend_mad_mx());
    }
  });
  return found;
}

bool CodeGenTileLangPTO::HasAscendGemmL1(const PrimFunc &func) const {
  bool found = false;
  tirx::PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (found)
      return;
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ascend_gemm_l1()) ||
              call->op.same_as(tl::ascend_blockscaled_gemm_l1());
    }
  });
  return found;
}

std::optional<std::string>
CodeGenTileLangPTO::PtoSpaceForStorageScope(const std::string &scope) const {
  if (scope == "global" || scope.empty())
    return std::string("gm");
  if (scope == "shared" || scope == "shared.dyn")
    return std::string("ub");
  if (scope == "shared.l1" || scope == "shared.l1.dyn")
    return std::string("mat");
  if (scope == "shared.l0a" || scope == "shared.l0a.dyn")
    return std::string("left");
  if (scope == "shared.l0b" || scope == "shared.l0b.dyn")
    return std::string("right");
  if (scope == "shared.l0c" || scope == "shared.l0c.dyn")
    return std::string("acc");
  return std::nullopt;
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
  ICHECK(rng_state_var_.empty())
      << "PTO RNG state must not escape its defining SIMT VF section";
  inside_simtvf_body_ = true;

  // Older TIR forms keep section-local allocations on the SBlock. Current
  // lowering emits them as AllocBuffer statements in the body.
  for (const Buffer &buffer : op->alloc_buffers) {
    EmitPtoBufferAllocation(buffer);
  }
  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
  if (stream.tellp() == body_start) {
    PrintIndent();
    stream << "pass\n";
  }

  // PhiloxRNG contains section-local SSA state. A sibling SIMT section must
  // initialize its own stream instead of referring to this section's helper.
  rng_state_var_.clear();
  inside_simtvf_body_ = saved_inside_simtvf_body;
  EndScope(simt_scope);
}

std::string CodeGenTileLangPTO::GetPtoPointerExpr(const VarNode *buffer_var,
                                                  DataType elem_dtype,
                                                  const PrimExpr &index) {
  std::string scope = "global";
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }

  auto pto_space = PtoSpaceForStorageScope(scope);
  std::string base = GetVarID(buffer_var);
  if (pto_space.has_value() && *pto_space != "gm") {
    if (HandleTypeMatch_(buffer_var, DataType::Int(8)) ||
        !HandleTypeMatch_(buffer_var, elem_dtype)) {
      base = "pto.castptr(" + base + ", " + PtoPtrType(elem_dtype, *pto_space) +
             ")";
    }
  }

  PrimExpr normalized_index =
      NormalizePackedFP4Index(index, elem_dtype, "pointer offset");
  if (is_zero(normalized_index)) {
    return base;
  }

  std::string index_str;
  int64_t const_index = 0;
  if (TryGetConstInt(normalized_index, &const_index)) {
    index_str = std::to_string(const_index);
  } else {
    index_str = RemoveOutermostParentheses(PrintExpr_(normalized_index));
  }

  if (pto_space.has_value()) {
    return "pto.addptr(" + base + ", " + index_str + ")";
  }

  LOG(FATAL) << "Unsupported storage scope in PTO pointer emission: " << scope;
  return "";
}

std::string CodeGenTileLangPTO::GetPtoPointerExpr(const BufferNode *buffer,
                                                  const PrimExpr &index) {
  return GetPtoPointerExpr(buffer->data.get(), buffer->dtype, index);
}

std::string CodeGenTileLangPTO::PtoScalarLoad(const BufferNode *buffer,
                                              const PrimExpr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  if (scope == "local.var") {
    return GetVarID(buffer->data.get());
  }

  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));
  if (inside_simtvf_body_ && (scope == "local.fragment" || scope == "local")) {
    ICHECK(IsSupportedSIMTLocalScalarAccessType(buffer->dtype))
        << "PTO SIMT local scalar load currently supports float16, float32, "
           "int32, and uint32 only, got "
        << buffer->dtype;
    std::string value =
        "scalar.load(" + GetVarID(buffer->data.get()) + ", " + index_str + ")";
    if (IsPTOInteger32(buffer->dtype)) {
      return "scalar.cast(" + value + ", " + PtoTypeName(buffer->dtype) + ")";
    }
    return value;
  }

  auto pto_space = PtoSpaceForStorageScope(scope);
  if (pto_space.has_value()) {
    const VarNode *buffer_var = buffer->data.get();
    std::string base = GetVarID(buffer_var);
    if (*pto_space != "gm") {
      const bool need_cast = HandleTypeMatch_(buffer_var, DataType::Int(8)) ||
                             !HandleTypeMatch_(buffer_var, buffer->dtype);
      if (need_cast) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(buffer->dtype, *pto_space) + ")";
      }
    }
    return "scalar.load(" + base + ", " + index_str + ")";
  }

  if (scope == "local.fragment" || scope == "local") {
    return GetVarID(buffer->data.get()) + "[" + index_str + "]";
  }

  LOG(FATAL) << "Unsupported PTO scalar load scope: " << scope;
  return "";
}

void CodeGenTileLangPTO::EmitPtoScalarStore(const BufferNode *buffer,
                                            const std::string &value,
                                            const PrimExpr &index) {
  std::string scope = ScopeOfBuffer(buffer);
  std::string index_str = RemoveOutermostParentheses(PrintExpr_(index));

  if (scope == "local.var") {
    stream << GetVarID(buffer->data.get()) << " = " << value << "\n";
    return;
  }

  if (scope == "local.fragment" || scope == "local") {
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalScalarAccessType(buffer->dtype))
          << "PTO SIMT local scalar store currently supports float16, "
             "float32, int32, and uint32 only, got "
          << buffer->dtype;
      std::string store_value = value;
      if (IsPTOInteger32(buffer->dtype)) {
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

  auto pto_space = PtoSpaceForStorageScope(scope);
  if (pto_space.has_value()) {
    const VarNode *buffer_var = buffer->data.get();
    std::string base = GetVarID(buffer_var);
    if (*pto_space != "gm") {
      const bool need_cast = HandleTypeMatch_(buffer_var, DataType::Int(8)) ||
                             !HandleTypeMatch_(buffer_var, buffer->dtype);
      if (need_cast) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(buffer->dtype, *pto_space) + ")";
      }
    }
    stream << "scalar.store(" << value << ", " << base << ", " << index_str
           << ")\n";
    return;
  }

  LOG(FATAL) << "Unsupported PTO scalar store scope: " << scope;
}

void CodeGenTileLangPTO::EmitPtoBufferAllocation(const Buffer &buffer) {
  std::string scope = GetPtrStorageScope(buffer->data);
  alloc_storage_scope_[buffer->data.get()] = scope;

  auto pto_space = PtoSpaceForStorageScope(scope);
  if (pto_space.has_value() && *pto_space != "gm") {
    PrintIndent();
    std::string vid = AllocVarID(buffer->data.get());
    auto alloc = AllocBuffer(buffer);
    auto opt_size = alloc.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO shared allocation currently requires constant allocation size";
    stream << vid << " = pto.castptr(pto.const(0, dtype=pto.i64), "
           << PtoPtrType(buffer->dtype, *pto_space) << ")\n";
    RegisterHandleType_(buffer->data.get(), buffer->dtype);
    return;
  } else if (scope == "local.fragment" || scope == "local") {
    std::string vid = AllocVarID(buffer->data.get());
    auto alloc = AllocBuffer(buffer);
    auto opt_size = alloc.ConstantAllocationSize();
    ICHECK(opt_size.has_value())
        << "PTO local.fragment currently requires constant allocation size";
    PrintIndent();
    const bool persistent =
        persistent_buffer_vars_.count(buffer->data.get()) != 0;
    if (inside_simtvf_body_) {
      ICHECK(IsSupportedSIMTLocalStorageType(buffer->dtype))
          << "PTO SIMT local allocation currently supports float16, "
             "bfloat16, float32, int32, uint32, and "
             "float8_e4m3fn/float8_e5m2, got "
          << buffer->dtype;
      stream << vid << " = pto.alloc_buffer((" << opt_size.value() << ",), "
             << PtoSIMTLocalStorageTypeName(buffer->dtype) << ")\n";
    } else if (persistent) {
      ICHECK(IsFloat32(buffer->dtype))
          << "PTO persistent local allocation currently supports float32 "
             "only, got "
          << buffer->dtype;
      stream << vid << " = pto.alloc_buffer((" << opt_size.value()
             << ",), pto.f32)\n";
    } else {
      stream << vid << " = [None] * " << opt_size.value() << "\n";
    }
  } else if (scope == "local.var") {
    PrintIndent();
    stream << AllocVarID(buffer->data.get()) << " = 0\n";
  }

  RegisterHandleType_(buffer->data.get(), buffer->dtype);
}

std::string CodeGenTileLangPTO::GetAddressOfExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 1U);
  // TIR address_of is represented as address_of(BufferLoad): BufferLoad
  // supplies the base buffer and index, while this node denotes only an
  // address and must not be lowered as a normal GM load.
  const auto *load = op->args[0].as<BufferLoadNode>();
  ICHECK(load) << "address_of expects BufferLoad";
  ICHECK_EQ(load->indices.size(), 1U)
      << "CodeGenTileLangPTO only supports flat memory";
  return GetPtoPointerExpr(load->buffer.get(), load->indices[0]);
}

std::string CodeGenTileLangPTO::GetAccessPtrExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 5U);
  auto buffer_var = Downcast<Var>(op->args[1]);
  DataType elem_dtype = DataType::Float(32);

  if (const auto *type_call = op->args[0].as<CallNode>()) {
    if (!type_call->args.empty()) {
      if (const auto *dtype_name = type_call->args[0].as<StringImmNode>()) {
        elem_dtype = ParsePTODtype(dtype_name->value);
      }
    }
  } else if (HandleTypeMatch_(buffer_var.get(), DataType::Float(32))) {
    elem_dtype = DataType::Float(32);
  }

  return GetPtoPointerExpr(buffer_var.get(), elem_dtype, op->args[2]);
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
  GetPTOCopyEndpoint_(op->args[0], "PTO GM->UB MTE destination", &dst_var,
                      &dst_index, &dst_dtype, &dst_scope);
  GetPTOCopyEndpoint_(op->args[1], "PTO GM->UB MTE source", &src_var,
                      &src_index, &src_dtype, &src_scope);
  ICHECK(dst_scope == "shared" || dst_scope == "shared.dyn")
      << "PTO GM->UB MTE destination must use shared/shared.dyn (UB) "
         "storage, got scope `"
      << dst_scope << "`";
  ICHECK(src_scope.empty() || src_scope == "global")
      << "PTO GM->UB MTE source must use global (GM) storage, got scope `"
      << src_scope << "`";
  ICHECK(IsPTOStorageDtypeCompatible(dst_dtype, src_dtype))
      << "PTO GM->UB MTE does not support dtype conversion: source is "
      << src_dtype << ", destination is " << dst_dtype;

  int64_t data_select = 0;
  ICHECK(TryGetConstInt(op->args[7], &data_select) &&
         (data_select == 0 || data_select == 1))
      << "PTO GM->UB MTE requires dataSelect to be the constant 0 or 1, got "
      << op->args[7];
  int64_t left_padding = 0;
  int64_t right_padding = 0;
  ICHECK(TryGetConstInt(op->args[5], &left_padding) && left_padding >= 0)
      << "PTO GM->UB MTE requires non-negative constant leftPadding, got "
      << op->args[5];
  ICHECK(TryGetConstInt(op->args[6], &right_padding) && right_padding >= 0)
      << "PTO GM->UB MTE requires non-negative constant rightPadding, got "
      << op->args[6];
  ICHECK(data_select != 0 || (left_padding == 0 && right_padding == 0))
      << "PTO GM->UB MTE cannot use left/right padding when dataSelect == 0";
  int64_t l2_cache_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[8], &l2_cache_ctrl))
      << "PTO GM->UB MTE requires constant l2_cache_ctl, got " << op->args[8];
  ICHECK_GE(l2_cache_ctrl, 0)
      << "PTO GM->UB MTE l2_cache_ctl must be in [0, 3], got " << l2_cache_ctrl;
  ICHECK_LT(l2_cache_ctrl, 4)
      << "PTO GM->UB MTE l2_cache_ctl must be in [0, 3], got " << l2_cache_ctrl;

  ValidatePTOUBCopyLayout_(dst_index, dst_dtype, op->args[3], op->args[4],
                           op->args[10], op->args[5], op->args[6],
                           data_select != 0, "PTO GM->UB MTE destination");

  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[9]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[10]));

  std::ostringstream os;
  os << "pto.mte_gm_ub(" << src << ", " << dst << ", " << l2_cache_ctrl << ", "
     << burst_len << ", nburst=(" << burst_num << ", " << src_stride << ", "
     << dst_stride << ")";
  if (data_select != 0) {
    auto binding_it = pad_binding_by_copy_.find(GetRef<Call>(op));
    ICHECK(binding_it != pad_binding_by_copy_.end())
        << "PTO GM->UB MTE with dataSelect == 1 has no analyzed pad value";
    const int64_t binding_id = binding_it->second;
    auto dtype_it = pad_binding_dtype_by_id_.find(binding_id);
    ICHECK(dtype_it != pad_binding_dtype_by_id_.end())
        << "PTO GM->UB MTE has an unknown pad binding " << binding_id;
    ICHECK(IsPTOStorageDtypeCompatible(dtype_it->second, dst_dtype))
        << "PTO GM->UB padding dtype mismatch: pad value is "
        << dtype_it->second << ", copy elements are " << dst_dtype;
    os << ", pad=(_tl_pad_" << binding_id << ", " << left_padding << ", "
       << right_padding << ")";
  }
  os << ")";
  return os.str();
}

std::string CodeGenTileLangPTO::GetPTOCopyPadValueExpr_(const PrimExpr &value) {
  if (const auto *imm = value.as<IntImmNode>()) {
    return "pto.const(" + std::to_string(imm->value) +
           ", dtype=" + PtoScalarType(value.dtype()) + ")";
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
           ", dtype=" + PtoScalarType(value.dtype()) + ")";
  }
  return RemoveOutermostParentheses(PrintExpr_(value));
}

void CodeGenTileLangPTO::GetPTOCopyEndpoint_(const PrimExpr &expr,
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

void CodeGenTileLangPTO::ValidatePTOUBCopyLayout_(
    const PrimExpr &index, DataType dtype, const PrimExpr &burst_num,
    const PrimExpr &burst_len, const PrimExpr &ub_stride,
    const PrimExpr &left_padding, const PrimExpr &right_padding,
    bool uses_padding, const char *context) const {
  const bool is_packed_fp4 = IsPTOPackedFP4Storage(dtype);
  ICHECK((dtype.is_scalar() || is_packed_fp4) &&
         PTOStorageElementBytes(dtype) > 0)
      << context
      << " requires a scalar or packed FP4 x2 byte-addressable dtype, got "
      << dtype;
  const int64_t element_bytes = PTOStorageElementBytes(dtype);

  arith::Analyzer analyzer;
  PrimExpr byte_offset =
      analyzer.Simplify(index * make_const(index.dtype(), element_bytes));
  PrimExpr offset_mod = analyzer.Simplify(
      floormod(byte_offset, make_const(byte_offset.dtype(), 32)));
  if (!analyzer.CanProveEqual(offset_mod, make_zero(offset_mod.dtype()))) {
    int64_t offset_remainder = 0;
    ICHECK(!TryGetConstInt(offset_mod, &offset_remainder))
        << context << " address must be 32-byte aligned, but element offset "
        << index << " for dtype " << dtype << " has byte remainder "
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
    ICHECK(TryGetConstInt(ub_stride, &stride))
        << context
        << " requires a constant UB row stride for potentially multi-burst "
           "copies, got "
        << ub_stride;
    ICHECK_EQ(stride % 32, 0)
        << context
        << " row stride must be 32-byte aligned for potentially multi-burst "
           "copies, got "
        << stride << " bytes";
  }

  int64_t len = 0;
  int64_t left = 0;
  int64_t right = 0;
  ICHECK(TryGetConstInt(left_padding, &left) &&
         TryGetConstInt(right_padding, &right))
      << context << " requires constant left/right padding";
  if (uses_padding) {
    ICHECK(TryGetConstInt(burst_len, &len))
        << context << " requires a constant burst length for padding";
    if (needs_row_stride) {
      int64_t required = len + (left + right) * element_bytes;
      ICHECK_GE(stride, required)
          << context << " stride is too small for the padded row: got "
          << stride << " bytes, need at least " << required
          << " bytes (burst_len=" << len << ", leftPadding=" << left
          << ", rightPadding=" << right << ", dtype=" << dtype << ")";
    }
  }
}

std::string CodeGenTileLangPTO::GetAscendCopyUbGmExpr_(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 8U);
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  DataType src_dtype =
      GetAnnotatedPointerDtype(op->args[1], DataType::Float(32));
  std::string src = GetPtoLocalPtrExpr(op->args[1], "ub", src_dtype);
  std::string burst_num = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string burst_len = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  int64_t l2_cache_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[5], &l2_cache_ctrl))
      << "PTO UB-to-GM copy expects constant l2_cache_ctrl";
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[7]));

  std::ostringstream os;
  os << "pto.mte_ub_gm(" << src << ", " << dst << ", " << burst_len
     << ", nburst=(" << burst_num << ", " << src_stride << ", " << dst_stride
     << "), l2_cache=\"" << PtoStoreL2CacheToken(l2_cache_ctrl) << "\")";
  return os.str();
}

std::string CodeGenTileLangPTO::GetPtoLocalByteAddrExpr(
    const PrimExpr &index, DataType elem_dtype, const std::string &context) {
  PrimExpr normalized_index =
      NormalizePackedFP4Index(index, elem_dtype, context.c_str());
  int64_t const_index = 0;
  int64_t elem_bytes = elem_dtype.bytes();
  if (TryGetConstInt(normalized_index, &const_index)) {
    return "pto.const(" + std::to_string(const_index * elem_bytes) +
           ", dtype=pto.int64)";
  }

  std::string index_expr =
      RemoveOutermostParentheses(PrintExpr_(normalized_index));
  std::string coerced =
      "_tl_coerce_i64(" + index_expr + ", context=\"" + context + "\")";
  if (elem_bytes == 1) {
    return coerced;
  }
  return "scalar.muli(" + coerced + ", pto.const(" +
         std::to_string(elem_bytes) + ", dtype=pto.int64))";
}

std::string CodeGenTileLangPTO::GetPtoLocalPtrExpr(const PrimExpr &expr,
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
            elem_dtype = ParsePTODtype(dtype_name->value);
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
  std::optional<std::string> actual_space;
  if (!scope.empty()) {
    actual_space = PtoSpaceForStorageScope(scope);
  }
  ICHECK(actual_space.has_value() || scope == "local.fragment" || scope.empty())
      << "PTO local pointer expected shared/local.fragment storage, got "
      << scope;
  if (actual_space.has_value()) {
    ICHECK_EQ(*actual_space, space)
        << "PTO local pointer storage scope does not match requested space";
  }

  std::string byte_addr =
      GetPtoLocalByteAddrExpr(index, elem_dtype, "PTO local pointer offset");
  return "pto.castptr(" + byte_addr + ", " + PtoPtrType(elem_dtype, space) +
         ")";
}

std::string CodeGenTileLangPTO::GetPtoE8M0ScalePtrExpr(const PrimExpr &expr) {
  PrimExpr index;
  const VarNode *buffer_var = nullptr;
  ICHECK(GetAddressOfIndex(expr, &index, &buffer_var))
      << "PTO E8M0 scale pointer expects address_of/tvm_access_ptr, got "
      << expr;

  // Scale buffers retain their logical uint8 E8M0 element type. The preceding
  // GM-to-L1 transfer may instead use a uint16 pair-packed physical view, but
  // that view must not change the byte address used by the MX load. Accept an
  // explicit uint16 access pointer as well for already-materialized views.
  DataType storage_dtype = GetAnnotatedPointerDtype(expr, DataType::UInt(8));
  ICHECK(storage_dtype.is_uint() &&
         (storage_dtype.bits() == 8 || storage_dtype.bits() == 16))
      << "PTO blockscaled GEMM scale storage must be logical uint8 or an "
         "explicit pair-packed uint16 view, got "
      << storage_dtype;

  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  ICHECK(scope == "shared.l1" || scope == "shared.l1.dyn")
      << "PTO E8M0 scale pointer must refer to an L1 allocation, got " << scope;

  std::string byte_addr = GetPtoLocalByteAddrExpr(
      index, storage_dtype, "PTO E8M0 scale pointer offset");
  return "pto.castptr(" + byte_addr + ", pto.ptr(pto.f8e8m0, \"mat\"))";
}

std::string CodeGenTileLangPTO::GetPtoAccPtrExpr(const PrimExpr &expr,
                                                 DataType dtype) {
  return GetPtoLocalPtrExpr(expr, "acc", dtype);
}

std::string CodeGenTileLangPTO::GetPtoL0APtrExpr(const PrimExpr &expr,
                                                 DataType dtype) {
  return GetPtoLocalPtrExpr(expr, "left", dtype);
}

std::string CodeGenTileLangPTO::GetPtoL0BPtrExpr(const PrimExpr &expr,
                                                 DataType dtype) {
  return GetPtoLocalPtrExpr(expr, "right", dtype);
}

std::string CodeGenTileLangPTO::GetPtoMatPtrExpr(const PrimExpr &expr,
                                                 DataType dtype) {
  return GetPtoLocalPtrExpr(expr, "mat", dtype);
}

std::string CodeGenTileLangPTO::GetPtoUbPtrExpr(const PrimExpr &expr,
                                                DataType dtype) {
  return GetPtoLocalPtrExpr(expr, "ub", dtype);
}

std::string CodeGenTileLangPTO::LocalVarID(const VarNode *var) {
  return GetVarID(var);
}

bool CodeGenTileLangPTO::IsLocalVarBuffer(const VarNode *var) const {
  return local_var_buffers_.count(var) != 0;
}

std::vector<const VarNode *>
CodeGenTileLangPTO::CollectLoopCarriedLocalVars(const Stmt &body) const {
  std::vector<const VarNode *> carry_vars;
  std::unordered_set<const VarNode *> seen;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    const auto *store = node.as<BufferStoreNode>();
    if (store == nullptr) {
      return;
    }
    const VarNode *var = store->buffer->data.get();
    if (!IsLocalVarBuffer(var) || store->buffer->dtype.lanes() <= 1) {
      return;
    }
    if (seen.insert(var).second) {
      carry_vars.push_back(var);
    }
  });
  return carry_vars;
}

bool CodeGenTileLangPTO::IsVmiLocalRegisterBuffer(
    const BufferNode *buffer) const {
  return !inside_simtvf_body_ && ScopeOfBuffer(buffer) == "local" &&
         buffer->dtype.lanes() > 1;
}

void CodeGenTileLangPTO::CheckVmiLocalRegisterIndex(
    const BufferNode *buffer, const PrimExpr &index) const {
  bool is_constant = true;
  tirx::PostOrderVisit(index, [&](const ObjectRef &node) {
    if (node.as<VarNode>()) {
      is_constant = false;
    }
  });
  ICHECK(is_constant)
      << "PTO VMI local register buffer `" << buffer->name
      << "` requires a compile-time constant index. Register arrays cannot "
         "be accessed inside T.unroll(..., explicit=False) or other runtime "
         "loops; use T.unroll(..., explicit=True), got "
      << index;
}

void CodeGenTileLangPTO::EnsurePTOGemmHelper(const CallNode *op) {
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
  DataType input_dtype = ParsePTODtype(dtype_name->value);
  ICHECK(IsSupportedPTOGemmInputDtype(input_dtype))
      << "PTO GEMM L1 helper currently only supports float32, float16, "
         "bfloat16, float8_e4m3fn, or float4_e2m1fn inputs, got "
      << input_dtype;
  ICHECK(!input_dtype.is_float4_e2m1fn())
      << "PTO GEMM L1 does not support packed float4 inputs; use the "
         "blockscaled GEMM path so codegen can preserve the FP4 storage "
         "layout";

  DataType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  DataType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(IsSamePTOStorageDtype(a_dtype, input_dtype))
      << "PTO GEMM L1 A pointer dtype must match input dtype " << input_dtype
      << ", got " << a_dtype;
  ICHECK(IsSamePTOStorageDtype(b_dtype, input_dtype))
      << "PTO GEMM L1 B pointer dtype must match input dtype " << input_dtype
      << ", got " << b_dtype;

  DataType accum_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  ICHECK(accum_dtype.is_float() && accum_dtype.bits() == 32)
      << "PTO GEMM L1 helper currently only supports float32 accum/output, got "
      << accum_dtype;

  if (gemm_emit_ctx_.initialized) {
    ICHECK_EQ(gemm_emit_ctx_.tile_m, tile_m);
    ICHECK_EQ(gemm_emit_ctx_.tile_n, tile_n);
    ICHECK_EQ(gemm_emit_ctx_.tile_k, tile_k);
    ICHECK_EQ(gemm_emit_ctx_.base_k, base_k);
    ICHECK(gemm_emit_ctx_.input_dtype == input_dtype)
        << "PTO GEMM L1 helper saw inconsistent input dtype: expected "
        << gemm_emit_ctx_.input_dtype << ", got " << input_dtype;
    ICHECK(gemm_emit_ctx_.accum_dtype == accum_dtype)
        << "PTO GEMM L1 helper saw inconsistent accum/output dtype: expected "
        << gemm_emit_ctx_.accum_dtype << ", got " << accum_dtype;
    return;
  }

  gemm_emit_ctx_.initialized = true;
  gemm_emit_ctx_.tile_m = tile_m;
  gemm_emit_ctx_.tile_n = tile_n;
  gemm_emit_ctx_.tile_k = tile_k;
  gemm_emit_ctx_.base_k = base_k;
  gemm_emit_ctx_.input_dtype = input_dtype;
  gemm_emit_ctx_.accum_dtype = accum_dtype;
  gemm_emit_ctx_.helper_name = "_tl_gemm_l1";
  gemm_emit_ctx_.a_l0_name = "a_l0_0";
  gemm_emit_ctx_.b_l0_name = "b_l0_0";

  int64_t input_c0 = PTOGemmInputC0(input_dtype);
  ICHECK_EQ(base_k % input_c0, 0)
      << "PTO GEMM base_k must be divisible by input C0=" << input_c0
      << " for dtype " << input_dtype;
  int64_t sub_k_tiles = tile_k / base_k;
  int64_t sub_k_c0_blocks = base_k / input_c0;
  int64_t a_l0_stage_elems = tile_m * base_k;
  int64_t b_l0_stage_elems = base_k * tile_n;

  PrintIndent();
  stream << "zero_addr = pto.const(0, dtype=pto.int64)\n";
  PrintIndent();
  stream << gemm_emit_ctx_.a_l0_name << " = pto.castptr(zero_addr, "
         << PtoPtrType(input_dtype, "left") << ")\n";
  PrintIndent();
  stream << gemm_emit_ctx_.b_l0_name << " = pto.castptr(zero_addr, "
         << PtoPtrType(input_dtype, "right") << ")\n";
  PrintIndent();
  stream << gemm_emit_ctx_.helper_name << " = PTOGemmL1Template(" << tile_m
         << ", " << tile_n << ", " << tile_k << ", " << base_k << ", "
         << sub_k_tiles << ", " << input_c0 << ", " << sub_k_c0_blocks << ", "
         << a_l0_stage_elems << ", " << b_l0_stage_elems << ")\n";
}

void CodeGenTileLangPTO::EnsurePTOBlockscaledGemmHelper(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 18U)
      << "tl.ascend_blockscaled_gemm_l1 expects exactly 18 arguments";
  int64_t tile_m = ConstArgDim(op, 5, "tl.ascend_blockscaled_gemm_l1 M");
  int64_t tile_k = ConstArgDim(op, 6, "tl.ascend_blockscaled_gemm_l1 K");
  int64_t tile_n = ConstArgDim(op, 7, "tl.ascend_blockscaled_gemm_l1 N");
  int64_t base_k =
      ConstArgDim(op, 8, "tl.ascend_blockscaled_gemm_l1 tile_k_sub");
  int64_t trans_b = ConstArgDim(op, 9, "tl.ascend_blockscaled_gemm_l1 trans_b");
  int64_t sf_nz_stride =
      ConstArgDim(op, 16, "tl.ascend_blockscaled_gemm_l1 sf_nz_stride");
  ICHECK_EQ(trans_b, 1)
      << "PTO blockscaled GEMM L1 helper currently requires trans_b=1";
  ICHECK_EQ(tile_k % base_k, 0);
  ICHECK_EQ(base_k % 64, 0)
      << "PTO blockscaled GEMM tile_k_sub must be divisible by 64, got "
      << base_k;
  ICHECK_GT(sf_nz_stride, 0)
      << "PTO blockscaled GEMM sf_nz_stride must be positive, got "
      << sf_nz_stride;
  CheckConstZero(op->args[14], "blockscaled GEMM buf_offset");

  const auto *input_dtype_name = op->args[11].as<StringImmNode>();
  ICHECK(input_dtype_name)
      << "PTO blockscaled GEMM L1 helper requires a constant input dtype "
         "string at tl.ascend_blockscaled_gemm_l1 arg 11";
  DataType input_dtype = ParsePTODtype(input_dtype_name->value);
  ICHECK(input_dtype.is_float8_e4m3fn() || input_dtype.is_float4_e2m1fn())
      << "PTO blockscaled GEMM L1 helper currently supports only "
         "float8_e4m3fn or float4_e2m1fn inputs, got "
      << input_dtype;

  const auto *scale_dtype_name = op->args[12].as<StringImmNode>();
  ICHECK(scale_dtype_name)
      << "PTO blockscaled GEMM L1 helper requires a constant scale dtype "
         "string at tl.ascend_blockscaled_gemm_l1 arg 12";
  ICHECK(scale_dtype_name->value == "uint16_t")
      << "PTO blockscaled GEMM L1 helper currently requires pair-packed "
         "uint16_t scales, got "
      << scale_dtype_name->value;

  const auto *accum_dtype_name = op->args[13].as<StringImmNode>();
  ICHECK(accum_dtype_name)
      << "PTO blockscaled GEMM L1 helper requires a constant accum dtype "
         "string at tl.ascend_blockscaled_gemm_l1 arg 13";
  DataType accum_dtype = ParsePTODtype(accum_dtype_name->value);
  ICHECK(accum_dtype.is_float() && accum_dtype.bits() == 32)
      << "PTO blockscaled GEMM L1 helper currently supports float32 "
         "accum/output, got "
      << accum_dtype;

  DataType a_dtype = GetAnnotatedPointerDtype(op->args[1], input_dtype);
  DataType b_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK(IsSamePTOStorageDtype(a_dtype, input_dtype))
      << "PTO blockscaled GEMM A pointer dtype must match input dtype "
      << input_dtype << ", got " << a_dtype;
  ICHECK(IsSamePTOStorageDtype(b_dtype, input_dtype))
      << "PTO blockscaled GEMM B pointer dtype must match input dtype "
      << input_dtype << ", got " << b_dtype;

  DataType sfa_dtype =
      GetAnnotatedPointerDtype(op->args[3], DataType::UInt(16));
  DataType sfb_dtype =
      GetAnnotatedPointerDtype(op->args[4], DataType::UInt(16));
  ICHECK(sfa_dtype.is_uint() && sfa_dtype.bits() == 16)
      << "PTO blockscaled GEMM SFA pointer must be uint16, got " << sfa_dtype;
  ICHECK(sfb_dtype.is_uint() && sfb_dtype.bits() == 16)
      << "PTO blockscaled GEMM SFB pointer must be uint16, got " << sfb_dtype;

  DataType acc_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  ICHECK(acc_dtype == accum_dtype)
      << "PTO blockscaled GEMM accumulator pointer dtype must match "
      << accum_dtype << ", got " << acc_dtype;

  if (blockscaled_gemm_emit_ctx_.initialized) {
    ICHECK_EQ(blockscaled_gemm_emit_ctx_.tile_m, tile_m);
    ICHECK_EQ(blockscaled_gemm_emit_ctx_.tile_n, tile_n);
    ICHECK_EQ(blockscaled_gemm_emit_ctx_.tile_k, tile_k);
    ICHECK_EQ(blockscaled_gemm_emit_ctx_.base_k, base_k);
    ICHECK_EQ(blockscaled_gemm_emit_ctx_.sf_nz_stride, sf_nz_stride);
    ICHECK(blockscaled_gemm_emit_ctx_.input_dtype == input_dtype)
        << "PTO blockscaled GEMM helper saw inconsistent input dtype";
    ICHECK(blockscaled_gemm_emit_ctx_.accum_dtype == accum_dtype)
        << "PTO blockscaled GEMM helper saw inconsistent accum/output dtype";
    return;
  }

  blockscaled_gemm_emit_ctx_.initialized = true;
  blockscaled_gemm_emit_ctx_.tile_m = tile_m;
  blockscaled_gemm_emit_ctx_.tile_n = tile_n;
  blockscaled_gemm_emit_ctx_.tile_k = tile_k;
  blockscaled_gemm_emit_ctx_.base_k = base_k;
  blockscaled_gemm_emit_ctx_.sf_nz_stride = sf_nz_stride;
  blockscaled_gemm_emit_ctx_.input_dtype = input_dtype;
  blockscaled_gemm_emit_ctx_.accum_dtype = accum_dtype;
  blockscaled_gemm_emit_ctx_.helper_name = "_tl_blockscaled_gemm_l1";
  blockscaled_gemm_emit_ctx_.a_l0_name = "blockscaled_a_l0_0";
  blockscaled_gemm_emit_ctx_.b_l0_name = "blockscaled_b_l0_0";

  int64_t input_c0 = PTOGemmInputC0(input_dtype);
  ICHECK_EQ(base_k % input_c0, 0)
      << "PTO blockscaled GEMM tile_k_sub must be divisible by input C0="
      << input_c0 << " for dtype " << input_dtype;
  int64_t sub_k_tiles = tile_k / base_k;
  int64_t sub_k_c0_blocks = base_k / input_c0;
  int64_t input_pack_factor = PTOGemmInputPackFactor(input_dtype);
  int64_t a_l0_stage_elems = tile_m * base_k / input_pack_factor;
  int64_t b_l0_stage_elems = tile_n * base_k / input_pack_factor;

  PrintIndent();
  stream << "blockscaled_zero_addr = pto.const(0, dtype=pto.int64)\n";
  PrintIndent();
  stream << blockscaled_gemm_emit_ctx_.a_l0_name << " = pto.castptr("
         << "blockscaled_zero_addr, " << PtoPtrType(input_dtype, "left")
         << ")\n";
  PrintIndent();
  stream << blockscaled_gemm_emit_ctx_.b_l0_name << " = pto.castptr("
         << "blockscaled_zero_addr, " << PtoPtrType(input_dtype, "right")
         << ")\n";
  PrintIndent();
  stream << blockscaled_gemm_emit_ctx_.helper_name
         << " = PTOBlockscaledGemmL1Template(" << tile_m << ", " << tile_n
         << ", " << tile_k << ", " << base_k << ", " << sub_k_tiles << ", "
         << input_c0 << ", " << sub_k_c0_blocks << ", " << a_l0_stage_elems
         << ", " << b_l0_stage_elems << ", " << sf_nz_stride;
  if (input_pack_factor != 1) {
    stream << ", input_pack_factor=" << input_pack_factor;
  }
  stream << ")\n";
}

void CodeGenTileLangPTO::EmitAscendCopyGmToCbuf(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 12U)
      << "tl.ascend_copy_gm_to_cbuf expects exactly 12 arguments";
  CheckConstZero(op->args[2], "sid");
  CheckConstZero(op->args[7], "loop4_src_stride");

  const std::string physical_dtype = Downcast<StringImm>(op->args[11])->value;
  DataType dst_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::BFloat(16));
  const bool is_fp4 = dst_dtype.is_float4_e2m1fn();
  const bool is_pair_packed_scale = physical_dtype == "uint16_t";
  ICHECK(physical_dtype.empty() || (is_fp4 && physical_dtype == "int8_t") ||
         (is_pair_packed_scale && dst_dtype.is_uint() && dst_dtype.bits() == 8))
      << "PTO GM-to-L1 copy supports int8 physical storage only for FP4 or "
         "uint16 pair-packed storage only for uint8 scale factors, got "
      << physical_dtype << " for destination dtype " << dst_dtype;

  int64_t transpose = 0;
  ICHECK(TryGetConstInt(op->args[9], &transpose))
      << "PTO GM-to-L1 fractal copy expects constant transpose flag";

  std::string dst =
      GetPtoLocalPtrExpr(op->args[0], "mat", DataType::BFloat(16));
  std::string src = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  DataType src_dtype = GetAnnotatedPointerDtype(op->args[1], dst_dtype);
  if (is_fp4 && !src_dtype.is_float4_e2m1fn()) {
    src = "pto.castptr(" + src + ", " + PtoPtrType(dst_dtype, "gm") + ")";
  }
  if (is_pair_packed_scale) {
    ICHECK(src_dtype.is_uint() && src_dtype.bits() == 8)
        << "PTO uint16 pair-packed GM-to-L1 copy requires uint8 scale "
           "source storage, got "
        << src_dtype;
    // The Final TIR count is already halved for two E8M0 values per uint16.
    // Cast both existing byte-addressed pointers so that the MTE moves that
    // packed physical view without changing the underlying byte offsets.
    src = "pto.castptr(" + src + ", " + PtoPtrType(DataType::UInt(16), "gm") +
          ")";
    dst = "pto.castptr(" + dst + ", " + PtoPtrType(DataType::UInt(16), "mat") +
          ")";
  }
  std::string l2_cache_ctrl =
      RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string n_value = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string d_value = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string smallc0_en = RemoveOutermostParentheses(PrintExpr_(op->args[8]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string dst_n_value =
      RemoveOutermostParentheses(PrintExpr_(op->args[10]));

  PrintIndent();
  stream << "pto.mte_gm_l1_frac(" << src << ", " << dst << ", "
         << (transpose == 0 ? "pto.FractalMode.ND2NZ" : "pto.FractalMode.DN2NZ")
         << ", shape=(" << n_value << ", " << d_value << "), src_layout=("
         << src_stride << ",), dst_group=(1, 1, " << dst_n_value
         << ", 0), ctrl=(" << l2_cache_ctrl << ", "
         << (smallc0_en == "0" ? "False" : smallc0_en) << "))\n";
}

void CodeGenTileLangPTO::EmitAscendFillL1(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 7U)
      << "tl.ascend_fill_l1 expects exactly 7 arguments";
  int64_t fill_word_bits = 0;
  ICHECK(TryGetConstInt(op->args[6], &fill_word_bits) &&
         (fill_word_bits == 16 || fill_word_bits == 32))
      << "tl.ascend_fill_l1 fill_word_bits must be a constant 16 or 32";

  DataType fill_dtype = DataType::UInt(static_cast<int>(fill_word_bits));
  std::string dst = GetPtoLocalPtrExpr(op->args[0], "mat", fill_dtype);
  std::string byte_offset = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
  std::string raw_value = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  std::string repeat_times =
      RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string block_num_32b =
      RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string dst_gap_32b = RemoveOutermostParentheses(PrintExpr_(op->args[5]));

  PrintIndent();
  stream << "pto.raw_fill_l1(" << dst << ", " << byte_offset << ", "
         << raw_value << ", repeat_times=" << repeat_times
         << ", block_num_32b=" << block_num_32b
         << ", dst_gap_32b=" << dst_gap_32b
         << ", fill_word_bits=" << fill_word_bits << ")\n";
}

void CodeGenTileLangPTO::EmitAscendLoadCbufToL0(const CallNode *op,
                                                bool is_l0a) {
  ICHECK(op->args.size() == 9U || op->args.size() == 16U)
      << "PTO L1-to-L0 load expects 9 or 16 arguments";

  const char *intrinsic_name =
      is_l0a ? "tl.ascend_load_cbuf_to_ca" : "tl.ascend_load_cbuf_to_cb";
  std::string m_start = RemoveOutermostParentheses(PrintExpr_(op->args[2]));
  std::string k_start = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string m_step = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string k_step = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[7]));
  int64_t transpose = 0;
  ICHECK(TryGetConstInt(op->args[8], &transpose))
      << intrinsic_name << " requires a constant transpose flag";

  if (op->args.size() == 9U) {
    DataType dtype =
        GetAnnotatedPointerDtype(op->args[0], DataType::BFloat(16));
    std::string dst = is_l0a ? GetPtoL0APtrExpr(op->args[0], dtype)
                             : GetPtoL0BPtrExpr(op->args[0], dtype);
    std::string src = GetPtoMatPtrExpr(op->args[1], dtype);

    PrintIndent();
    stream << (is_l0a ? "pto.mte_l1_l0a(" : "pto.mte_l1_l0b(") << src << ", "
           << dst << ", m_start=" << m_start << ", k_start=" << k_start
           << ", m_step=" << m_step << ", k_step=" << k_step
           << ", src_stride=" << src_stride << ", dst_stride=" << dst_stride;
    if (transpose != 0) {
      stream << ", transpose=True";
    }
    stream << ")\n";
  } else {
    DataType source_dtype =
        GetAnnotatedPointerDtype(op->args[1], DataType::Float8E4M3FN());
    DataType destination_dtype =
        GetAnnotatedPointerDtype(op->args[0], source_dtype);
    ICHECK(source_dtype.is_float8_e4m3fn() || source_dtype.is_float4_e2m1fn())
        << "PTO blockscaled L0 staging currently supports only float8_e4m3fn "
           "or float4_e2m1fn data, got "
        << source_dtype;
    ICHECK(IsSamePTOStorageDtype(destination_dtype, source_dtype))
        << "PTO blockscaled L0 staging source and destination dtypes must "
           "match";

    std::string source = GetPtoLocalPtrExpr(op->args[1], "mat", source_dtype);
    std::string destination = GetPtoLocalPtrExpr(
        op->args[0], is_l0a ? "left" : "right", destination_dtype);
    std::string scale_source = GetPtoE8M0ScalePtrExpr(op->args[9]);

    PrintIndent();
    stream << "pto.mte_l1_l0" << (is_l0a ? "a" : "b") << "(" << source << ", "
           << destination << ", m_start=" << m_start << ", k_start=" << k_start
           << ", m_step=" << m_step << ", k_step=" << k_step
           << ", src_stride=" << src_stride << ", dst_stride=" << dst_stride;
    if (transpose != 0) {
      stream << ", transpose=True";
    }
    stream << ")\n";

    PrintIndent();
    stream
        << "pto.mte_l1_l0" << (is_l0a ? "a" : "b") << "_mx(" << scale_source
        << ", " << destination
        << ", x_start=" << RemoveOutermostParentheses(PrintExpr_(op->args[10]))
        << ", y_start=" << RemoveOutermostParentheses(PrintExpr_(op->args[11]))
        << ", x_step=" << RemoveOutermostParentheses(PrintExpr_(op->args[12]))
        << ", y_step=" << RemoveOutermostParentheses(PrintExpr_(op->args[13]))
        << ", src_stride="
        << RemoveOutermostParentheses(PrintExpr_(op->args[14]))
        << ", dst_stride="
        << RemoveOutermostParentheses(PrintExpr_(op->args[15])) << ")\n";
  }
}

void CodeGenTileLangPTO::EmitPTOGemmRun(const std::string &a_mat,
                                        const std::string &b_mat,
                                        const std::string &acc,
                                        const std::string &clear_accum,
                                        const std::string &unit_flag_ctrl,
                                        int64_t hf32_mode) {
  PrintIndent();
  stream << gemm_emit_ctx_.helper_name << ".run_l1_tile(" << a_mat << ", "
         << b_mat << ", " << gemm_emit_ctx_.a_l0_name << ", "
         << gemm_emit_ctx_.b_l0_name << ", " << acc
         << ", clear_accum=" << clear_accum
         << ", unit_flag_ctrl=" << unit_flag_ctrl;
  if (IsFloat32(gemm_emit_ctx_.input_dtype)) {
    EmitPTOHf32ModeArgument(stream, hf32_mode);
  }
  stream << ")\n";
}

void CodeGenTileLangPTO::EmitAscendGemmL1(const CallNode *op) {
  EnsurePTOGemmHelper(op);
  std::string acc = GetPtoAccPtrExpr(op->args[0], DataType::Float(32));
  std::string a_mat = GetPtoMatPtrExpr(op->args[1], gemm_emit_ctx_.input_dtype);
  std::string b_mat = GetPtoMatPtrExpr(op->args[2], gemm_emit_ctx_.input_dtype);

  int64_t hf32_mode = 0;
  if (IsFloat32(gemm_emit_ctx_.input_dtype)) {
    auto it = hf32_mode_by_cube_call_.find(GetRef<Call>(op));
    ICHECK(it != hf32_mode_by_cube_call_.end())
        << "PTO codegen did not analyze HF32 mode for FP32 GEMM";
    hf32_mode = it->second;
  }
  EmitPTOGemmRun(
      a_mat, b_mat, acc, RemoveOutermostParentheses(PrintExpr_(op->args[8])),
      RemoveOutermostParentheses(PrintExpr_(op->args[11])), hf32_mode);
}

void CodeGenTileLangPTO::EmitUnitFlagDispatch(
    const PrimExpr &unit_flag_expr,
    const std::function<std::string(int64_t)> &map_unit_flag,
    const std::function<void(const std::string &)> &emit_operation) {
  int64_t unit_flag_value = 0;
  if (TryGetConstInt(unit_flag_expr, &unit_flag_value)) {
    emit_operation(map_unit_flag(unit_flag_value));
    return;
  }

  if (const auto *select = unit_flag_expr.as<SelectNode>()) {
    std::string select_cond =
        RemoveOutermostParentheses(PrintExpr_(select->condition));
    PrintIndent();
    stream << "if " << select_cond << ":\n";
    int then_scope = BeginScope();
    EmitUnitFlagDispatch(select->true_value, map_unit_flag, emit_operation);
    EndScope(then_scope);
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    EmitUnitFlagDispatch(select->false_value, map_unit_flag, emit_operation);
    EndScope(else_scope);
    return;
  }

  std::string unit_flag_expr_text =
      RemoveOutermostParentheses(PrintExpr_(unit_flag_expr));
  // PTODSL's default AST rewrite converts these native Python conditions to
  // structured device control flow. The public TileLang contract restricts
  // runtime values to 0, 2, or 3, so the final branch represents 2.
  PrintIndent();
  stream << "if " << unit_flag_expr_text << " == 0:\n";
  int no_flag_scope = BeginScope();
  emit_operation(map_unit_flag(0));
  EndScope(no_flag_scope);
  PrintIndent();
  stream << "elif " << unit_flag_expr_text << " == 3:\n";
  int set_scope = BeginScope();
  emit_operation(map_unit_flag(3));
  EndScope(set_scope);
  PrintIndent();
  stream << "else:\n";
  int check_scope = BeginScope();
  emit_operation(map_unit_flag(2));
  EndScope(check_scope);
}

void CodeGenTileLangPTO::EmitAscendMad(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 10U)
      << "tl.ascend_mad expects exactly 10 arguments";
  int64_t gemv_ctrl = ConstArgDim(op, 7, "tl.ascend_mad gemv_ctrl");
  int64_t btbuf_ctrl = ConstArgDim(op, 8, "tl.ascend_mad BTbuf_ctrl");
  ICHECK_EQ(btbuf_ctrl, 0) << "PTO MAD does not support nonzero BTbuf_ctrl";

  std::string dst = GetPtoAccPtrExpr(op->args[0], DataType::Float(32));
  DataType input_dtype = GetAscendMadInputDtype(op);
  std::string lhs = GetPtoL0APtrExpr(op->args[1], input_dtype);
  std::string rhs = GetPtoL0BPtrExpr(op->args[2], input_dtype);

  int64_t hf32_mode = 0;
  if (IsFloat32(input_dtype)) {
    auto it = hf32_mode_by_cube_call_.find(GetRef<Call>(op));
    ICHECK(it != hf32_mode_by_cube_call_.end())
        << "PTO codegen did not analyze HF32 mode for FP32 L0 MAD";
    hf32_mode = it->second;
  }

  auto emit_mad = [&](bool clear_accum, const std::string &unit_flag) {
    PrintIndent();
    stream << (clear_accum ? "pto.mad(" : "pto.mad_acc(") << lhs << ", " << rhs
           << ", " << dst << ", " << PrintExpr_(op->args[3]) << ", "
           << PrintExpr_(op->args[5]) << ", " << PrintExpr_(op->args[4]);
    if (!unit_flag.empty()) {
      stream << ", unit_flag=" << unit_flag;
    }
    if (gemv_ctrl != 0) {
      stream << ", disable_gemv=True";
    }
    EmitPTOHf32ModeArgument(stream, hf32_mode);
    stream << ")\n";
  };
  auto emit_mad_with_unit_flag = [&](bool clear_accum) {
    EmitUnitFlagDispatch(op->args[6], MadUnitFlagArg,
                         [&](const std::string &unit_flag) {
                           emit_mad(clear_accum, unit_flag);
                         });
  };

  int64_t zero_c = 0;
  if (TryGetConstInt(op->args[9], &zero_c)) {
    emit_mad_with_unit_flag(zero_c != 0);
    return;
  }

  std::string zero_c_expr = RemoveOutermostParentheses(PrintExpr_(op->args[9]));
  PrintIndent();
  stream << "with pto.if_(" << zero_c_expr << ") as clear_br:\n";
  int clear_scope = BeginScope();
  PrintIndent();
  stream << "with clear_br.then_:\n";
  int then_scope = BeginScope();
  emit_mad_with_unit_flag(true);
  EndScope(then_scope);
  PrintIndent();
  stream << "with clear_br.else_:\n";
  int else_scope = BeginScope();
  emit_mad_with_unit_flag(false);
  EndScope(else_scope);
  EndScope(clear_scope);
}

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToUb(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 26U)
      << "tl.ascend_copy_matrix_cc_to_ub expects exactly 26 arguments";
  DataType dst_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  std::string dst = GetPtoLocalPtrExpr(op->args[0], "ub", dst_dtype);
  std::string src = GetPtoAccPtrExpr(op->args[1], DataType::Float(32));
  std::string n_size = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string m_size = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string dst_stride = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string src_stride = RemoveOutermostParentheses(PrintExpr_(op->args[6]));
  std::string sub_blockid = RemoveOutermostParentheses(PrintExpr_(op->args[8]));

  int64_t dual_dst_ctl = 0;
  int64_t quant_pre = 0;
  int64_t split_en = 0;
  auto check_const = [&](size_t index, int64_t expected, const char *name) {
    int64_t value = 0;
    ICHECK(TryGetConstInt(op->args[index], &value) && value == expected)
        << "PTO L0C-to-UB currently requires " << name << " == " << expected
        << ", got " << op->args[index];
  };
  check_const(2, 0, "sid");
  ICHECK(TryGetConstInt(op->args[7], &dual_dst_ctl) && dual_dst_ctl >= 0 &&
         dual_dst_ctl <= 2)
      << "PTO L0C-to-UB expects dual_dst_ctl in {0, 1, 2}";
  ICHECK(TryGetConstInt(op->args[11], &quant_pre))
      << "PTO L0C-to-UB expects constant quant_pre";
  ICHECK(quant_pre == 0 || quant_pre == 1 || quant_pre == 16)
      << "PTO L0C-to-UB currently supports quant_pre 0, 1, or 16";
  DataType expected_dst_dtype = DataType::Float(32);
  if (quant_pre == 1) {
    expected_dst_dtype = DataType::Float(16);
  } else if (quant_pre == 16) {
    expected_dst_dtype = DataType::BFloat(16);
  }
  ICHECK_EQ(dst_dtype, expected_dst_dtype)
      << "PTO L0C-to-UB destination dtype must match quant_pre " << quant_pre
      << ", got " << dst_dtype;
  ICHECK(TryGetConstInt(op->args[13], &split_en) && split_en == 0)
      << "PTO L0C-to-UB currently requires split_en=0";
  check_const(14, 1, "NZ2ND_en");
  if (dual_dst_ctl != 0) {
    check_const(8, 0, "sub_blockid with dual destination mode");
    check_const(11, 0, "quant_pre with dual destination mode");
  }
  check_const(9, 0, "clip_relu_pre");
  check_const(12, 0, "relu_pre");
  check_const(15, 0, "quant_post");
  check_const(16, 0, "relu_post");
  check_const(17, 0, "clip_relu_post");
  check_const(18, 0, "loop_enhance_en");
  check_const(19, 0, "eltwise_op");
  check_const(20, 0, "eltwise_antq_en");
  check_const(21, 0, "loop_enhance_merge_en");
  check_const(22, 0, "C0_pad_en");
  check_const(23, 0, "wino_post_en");
  check_const(24, 0, "broadcast_en");
  check_const(25, 0, "NZ2DN_en");

  auto emit_store = [&](const std::string &unit_flag) {
    PrintIndent();
    stream << "pto.mte_l0c_ub(" << src << ", " << dst << ", " << m_size << ", "
           << n_size << ", " << src_stride << ", " << dst_stride;
    if (dual_dst_ctl == 0) {
      stream << ", " << sub_blockid;
    } else {
      stream << ", split=pto.SplitMode." << (dual_dst_ctl == 1 ? "M" : "N");
    }
    if (!unit_flag.empty()) {
      stream << ", unit_flag=" << unit_flag;
    }
    if (quant_pre == 1) {
      stream << ", pre_quant=(pto.f16(1.0), \"f32_f16\")";
    } else if (quant_pre == 16) {
      stream << ", pre_quant=(pto.bf16(1.0), \"f32_bf16\")";
    }
    stream << ", layout=\"nz2nd\")\n";
  };
  EmitUnitFlagDispatch(op->args[10], AccStoreUnitFlagArg, emit_store);
}

void CodeGenTileLangPTO::EmitPTOBlockscaledGemmRun(
    const std::string &a_mat, const std::string &b_mat,
    const std::string &sfa_e8m0_mat, const std::string &sfb_e8m0_mat,
    const std::string &acc, const std::string &sf_k_offset,
    const std::string &clear_accum, const std::string &unit_flag_ctrl) {
  PrintIndent();
  stream << blockscaled_gemm_emit_ctx_.helper_name << ".run_l1_tile(" << a_mat
         << ", " << b_mat << ", " << sfa_e8m0_mat << ", " << sfb_e8m0_mat
         << ", " << blockscaled_gemm_emit_ctx_.a_l0_name << ", "
         << blockscaled_gemm_emit_ctx_.b_l0_name << ", " << acc
         << ", sf_k_offset=" << sf_k_offset << ", clear_accum=" << clear_accum
         << ", unit_flag_ctrl=" << unit_flag_ctrl << ")\n";
}

void CodeGenTileLangPTO::EmitAscendBlockscaledGemmL1(const CallNode *op) {
  EnsurePTOBlockscaledGemmHelper(op);
  std::string acc = GetPtoAccPtrExpr(op->args[0], DataType::Float(32));
  std::string a_mat = GetPtoLocalPtrExpr(
      op->args[1], "mat", blockscaled_gemm_emit_ctx_.input_dtype);
  std::string b_mat = GetPtoLocalPtrExpr(
      op->args[2], "mat", blockscaled_gemm_emit_ctx_.input_dtype);
  std::string sfa_e8m0_mat = GetPtoE8M0ScalePtrExpr(op->args[3]);
  std::string sfb_e8m0_mat = GetPtoE8M0ScalePtrExpr(op->args[4]);

  EmitPTOBlockscaledGemmRun(
      a_mat, b_mat, sfa_e8m0_mat, sfb_e8m0_mat, acc,
      RemoveOutermostParentheses(PrintExpr_(op->args[15])),
      RemoveOutermostParentheses(PrintExpr_(op->args[10])),
      RemoveOutermostParentheses(PrintExpr_(op->args[17])));
}

void CodeGenTileLangPTO::EmitAscendMadMx(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 10U)
      << "tl.ascend_mad_mx expects exactly 10 arguments";

  DataType input_dtype =
      GetAnnotatedPointerDtype(op->args[1], DataType::Float8E4M3FN());
  DataType rhs_dtype = GetAnnotatedPointerDtype(op->args[2], input_dtype);
  ICHECK((input_dtype.is_float8_e4m3fn() || input_dtype.is_float4_e2m1fn()) &&
         IsSamePTOStorageDtype(rhs_dtype, input_dtype))
      << "PTO blockscaled L0 MAD currently supports float8_e4m3fn or "
         "float4_e2m1fn inputs";
  DataType acc_dtype =
      GetAnnotatedPointerDtype(op->args[0], DataType::Float(32));
  ICHECK(IsFloat32(acc_dtype)) << "PTO blockscaled L0 MAD currently supports "
                                  "only float32 accumulator, got "
                               << acc_dtype;
  int64_t gemv_ctrl = 0;
  int64_t btbuf_ctrl = 0;
  ICHECK(TryGetConstInt(op->args[7], &gemv_ctrl) && gemv_ctrl == 1)
      << "PTO blockscaled L0 MAD requires gemv_ctrl=1";
  ICHECK(TryGetConstInt(op->args[8], &btbuf_ctrl) && btbuf_ctrl == 0)
      << "PTO blockscaled L0 MAD requires BTbuf_ctrl=0";

  std::string acc = GetPtoAccPtrExpr(op->args[0], DataType::Float(32));
  std::string lhs = GetPtoLocalPtrExpr(op->args[1], "left", input_dtype);
  std::string rhs = GetPtoLocalPtrExpr(op->args[2], "right", input_dtype);
  std::string m = RemoveOutermostParentheses(PrintExpr_(op->args[3]));
  std::string k = RemoveOutermostParentheses(PrintExpr_(op->args[4]));
  std::string n = RemoveOutermostParentheses(PrintExpr_(op->args[5]));
  std::string clear_accum = RemoveOutermostParentheses(PrintExpr_(op->args[9]));

  auto emit_mad = [&](bool clear, const std::string &unit_flag) {
    PrintIndent();
    stream << "pto." << (clear ? "mad_mx" : "mad_mx_acc") << "(" << lhs << ", "
           << rhs << ", " << acc << ", " << m << ", " << n << ", " << k;
    if (!unit_flag.empty()) {
      stream << ", unit_flag=" << unit_flag;
    }
    stream << ", disable_gemv=True, sat=\"sat\")\n";
  };
  auto emit_mad_with_unit_flag = [&](bool clear) {
    EmitUnitFlagDispatch(
        op->args[6], MadUnitFlagArg,
        [&](const std::string &unit_flag) { emit_mad(clear, unit_flag); });
  };

  int64_t clear_accum_value = 0;
  if (TryGetConstInt(op->args[9], &clear_accum_value)) {
    emit_mad_with_unit_flag(clear_accum_value != 0);
    return;
  }

  PrintIndent();
  stream << "if " << clear_accum << ":\n";
  int clear_scope = BeginScope();
  emit_mad_with_unit_flag(true);
  EndScope(clear_scope);
  PrintIndent();
  stream << "else:\n";
  int clear_else_scope = BeginScope();
  emit_mad_with_unit_flag(false);
  EndScope(clear_else_scope);
}

void CodeGenTileLangPTO::EmitAscendCopyMatrixCcToGm(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 25U)
      << "tl.ascend_copy_matrix_cc_to_gm expects exactly 25 arguments";
  std::string dst = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string src = GetPtoAccPtrExpr(op->args[1], DataType::Float(32));
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

std::string
CodeGenTileLangPTO::EmitPTOAllReduceExpr_(const std::string &func_name,
                                          const CallNode *op) {
  CheckPTOAllReduceDtype(op);

  const size_t begin = func_name.find("tl::AscendAllReduce");
  ICHECK_NE(begin, std::string::npos)
      << "Cannot parse AscendAllReduce template arguments from: " << func_name;
  struct ReductionInfo {
    const char *tir_name;
    const char *pto_name;
  };
  const ReductionInfo reductions[] = {
      {"tl::SumOp", "_tl_simt_allreduce_sum"},
      {"tl::MaxOp", "_tl_simt_allreduce_max"},
      {"tl::MinOp", "_tl_simt_allreduce_min"},
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

std::string
CodeGenTileLangPTO::PrintVmiAnnotationValue(const std::string &key,
                                            const ObjectRef &value) {
  if (key == "to_dtype") {
    if (const auto *dtype_name = value.as<StringImmNode>()) {
      const std::string &name = dtype_name->value;
      // PTODSL int-to-int widening requires a signed/unsigned source
      // (si16/ui16), not TVM's signless i16.
      if (name == "si8")
        return "pto.si8";
      if (name == "si16")
        return "pto.si16";
      if (name == "si32")
        return "pto.si32";
      if (name == "si64")
        return "pto.si64";
      return PtoScalarType(ParsePTODtype(name));
    }
  }

  if (const auto *expr = value.as<PrimExprNode>()) {
    return PrintExpr_(GetRef<PrimExpr>(expr));
  }

  std::ostringstream os;
  os << value;
  return os.str();
}

// Returns true if a tl.vmi.vcvt narrows logical BF16 values to FP4.  PTO only
// exposes FP4 as f4e2m1x2, so this call packs its BF16 input in codegen.
static bool
IsVmiVcvtToFp4(const std::vector<std::pair<std::string, ObjectRef>> &kwargs) {
  for (const auto &[key, value] : kwargs) {
    if (key == "to_dtype") {
      if (const auto *str = value.as<StringImmNode>()) {
        return str->value == "float4_e2m1fn";
      }
    }
  }
  return false;
}

void CodeGenTileLangPTO::PrintPtoVmiCall_(const CallNode *op,
                                          std::ostream &os) {
  auto opt_call_op = op->op.as<Op>();
  ICHECK(opt_call_op.has_value());
  std::string op_name = opt_call_op.value()->name;
  ICHECK(StartsWith(op_name, "tl.vmi."))
      << "Expected a tl.vmi.* call, got " << op_name;

  std::vector<std::pair<std::string, ObjectRef>> kwargs;
  kwargs.reserve(op->annotations.size());
  for (const auto &[key, value] : op->annotations) {
    std::string key_str = key;
    if (key_str == "loc" || key_str == "ip") {
      continue;
    }
    kwargs.emplace_back(std::move(key_str), value);
  }
  std::sort(kwargs.begin(), kwargs.end(), [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  if (op_name == "tl.vmi.vcvt" && IsVmiVcvtToFp4(kwargs)) {
    ICHECK_EQ(op->args.size(), 1U)
        << "Packed FP4 vcvt expects exactly one BF16 source vector";
    const DataType source_dtype = op->args[0].dtype();
    ICHECK(source_dtype.is_vector() && source_dtype.element_of().is_bfloat16())
        << "Packed FP4 vcvt only supports bfloat16 source vectors, got "
        << source_dtype;
    ICHECK_EQ(source_dtype.lanes() % 2, 0)
        << "Packed FP4 vcvt requires an even BF16 lane count, got "
        << source_dtype.lanes();
  }

  os << "pto." << op_name.substr(3) << "(";
  bool needs_comma = false;

  auto print_scalar_literal_value = [&](const PrimExpr &arg) {
    if (const auto *imm = arg.as<IntImmNode>()) {
      if (imm->dtype == DataType::Bool()) {
        os << (imm->value ? "True" : "False");
      } else {
        os << imm->value;
      }
      return;
    }
    if (const auto *imm = arg.as<FloatImmNode>()) {
      os << "float.fromhex('" << FlexibleHexFormat(imm->value) << "')";
      return;
    }
    PrintExpr_(arg, os);
  };

  // PTODSL vgather/vgatherb/vscatter take a single pointer operand. TileLang
  // lowers buffer addresses to (ptr, elem_offset); fold them with addptr here.
  auto print_ptr_with_offset = [&](const PrimExpr &ptr,
                                   const PrimExpr &offset) {
    if (is_zero(offset)) {
      os << PrintExpr_(ptr);
      return;
    }
    os << "pto.addptr(" << PrintExpr_(ptr) << ", "
       << RemoveOutermostParentheses(PrintExpr_(offset)) << ")";
  };

  if (op_name == "tl.vmi.vgather" || op_name == "tl.vmi.vgatherb") {
    ICHECK_EQ(op->args.size(), 4U)
        << op_name << " expects (ptr, offset, offsets, mask)";
    print_ptr_with_offset(op->args[0], op->args[1]);
    os << ", " << PrintExpr_(op->args[2]) << ", " << PrintExpr_(op->args[3]);
    needs_comma = true;
  } else if (op_name == "tl.vmi.vscatter") {
    ICHECK_EQ(op->args.size(), 5U)
        << op_name << " expects (value, ptr, offset, offsets, mask)";
    os << PrintExpr_(op->args[0]) << ", ";
    print_ptr_with_offset(op->args[1], op->args[2]);
    os << ", " << PrintExpr_(op->args[3]) << ", " << PrintExpr_(op->args[4]);
    needs_comma = true;
  } else if (op_name == "tl.vmi.vstore") {
    auto dist_mode_it = op->annotations.find("dist_mode");
    ObjectRef dist_mode = dist_mode_it != op->annotations.end()
                              ? (*dist_mode_it).second
                              : ObjectRef();
    const bool is_dintlv_store =
        dist_mode.defined() && dist_mode.as<StringImmNode>() != nullptr &&
        Downcast<StringImm>(dist_mode)->value == "dintlv";
    if (is_dintlv_store) {
      ICHECK_GE(op->args.size(), 4U)
          << op_name
          << " with dist_mode=dintlv expects (even, odd, ptr, offset[, mask])";
      os << "(" << PrintExpr_(op->args[0]) << ", " << PrintExpr_(op->args[1])
         << ")";
      for (size_t i = 2; i < op->args.size(); ++i) {
        os << ", " << PrintExpr_(op->args[i]);
      }
      needs_comma = true;
    } else {
      for (size_t i = 0; i < op->args.size(); ++i) {
        const PrimExpr &arg = op->args[i];
        if (needs_comma) {
          os << ", ";
        }
        os << PrintExpr_(arg);
        needs_comma = true;
      }
    }
  } else {
    for (size_t i = 0; i < op->args.size(); ++i) {
      const PrimExpr &arg = op->args[i];
      if (needs_comma) {
        os << ", ";
      }
      const bool is_scalar_literal =
          arg.as<FloatImmNode>() != nullptr || arg.as<IntImmNode>() != nullptr;
      const bool should_wrap_typed_literal =
          (op_name == "tl.vmi.vbrc" || op_name == "tl.vmi.vci") && i == 0 &&
          is_scalar_literal;
      if (should_wrap_typed_literal) {
        // PTODSL needs typed literal scalars for these VMI sources; preserve
        // the literal dtype instead of assuming every source is f32.
        os << PtoScalarType(arg.dtype()) << "(";
        print_scalar_literal_value(arg);
        os << ")";
      } else if (op_name == "tl.vmi.vcvt" && i == 0 && IsVmiVcvtToFp4(kwargs) &&
                 arg.dtype().is_vector() &&
                 arg.dtype().element_of().is_bfloat16()) {
        // PTOAS vcvt only accepts bf16x2 -> f4x2; pair the bf16 source
        // (vinterpret_cast is a physical no-op) inline. Note: pto.vmi.bf16x2
        // is the BF16x2Type dtype, distinct from the MLIR vector pto.bf16x2.
        os << "pto.vmi.vinterpret_cast(" << PrintExpr_(arg)
           << ", to_dtype=pto.vmi.bf16x2)";
      } else {
        os << PrintExpr_(arg);
      }
      needs_comma = true;
    }
  }

  for (const auto &[key, value] : kwargs) {
    if (needs_comma) {
      os << ", ";
    }
    os << key << "=";
    if (op_name == "tl.vmi.vcvt" && key == "to_dtype" && op->dtype.is_int()) {
      os << PtoSignedIntegerTypeName(op->dtype.element_of());
    } else if (op_name == "tl.vmi.vload" && key == "size" &&
               op->dtype.element_of().is_float4_e2m1fn()) {
      // TileLang VMI exposes FP4 lane counts logically.  PTO represents each
      // f4e2m1x2 byte as one physical VMI element, so its vload size must be
      // expressed in packed-pair units.  The frontend has already converted
      // the corresponding address offset to the same physical unit.
      os << "(" << PrintVmiAnnotationValue(key, value) << " // 2)";
    } else {
      os << PrintVmiAnnotationValue(key, value);
    }
    needs_comma = true;
  }
  os << ")";
}

void CodeGenTileLangPTO::PrintPtoSelectValue_(const PrimExpr &value,
                                              DataType dtype,
                                              std::ostream &os) {
  const std::string pto_dtype = PtoTypeName(dtype);
  ICHECK(!pto_dtype.empty()) << "Unsupported PTO select value dtype: " << dtype;

  if (const auto *int_imm = value.as<IntImmNode>()) {
    os << "pto.const(" << int_imm->value << ", dtype=" << pto_dtype << ")";
    return;
  }
  if (const auto *float_imm = value.as<FloatImmNode>()) {
    os << "pto.const(float.fromhex('" << FlexibleHexFormat(float_imm->value)
       << "'), dtype=" << pto_dtype << ")";
    return;
  }
  PrintExpr_(value, os);
}

void CodeGenTileLangPTO::PrintPtoSelect_(const PrimExpr &condition,
                                         const PrimExpr &true_value,
                                         const PrimExpr &false_value,
                                         DataType dtype, std::ostream &os) {
  os << "scalar.select(";
  PrintExpr_(condition, os);
  os << ", ";
  PrintPtoSelectValue_(true_value, dtype, os);
  os << ", ";
  PrintPtoSelectValue_(false_value, dtype, os);
  os << ")";
}

void CodeGenTileLangPTO::PrintPtoIfThenElse_(const CallNode *op,
                                             std::ostream &os) {
  ICHECK_EQ(op->args.size(), 3U) << "if_then_else expects 3 arguments";

  std::string condition = RemoveOutermostParentheses(PrintExpr_(op->args[0]));
  std::string result = name_supply_->FreshName("_tl_if_then_else_result");

  PrimExpr simplified_condition = arith::Analyzer().Simplify(op->args[0]);
  int64_t constant_condition = 0;
  const bool is_dynamic =
      !TryGetConstInt(simplified_condition, &constant_condition);

  PrintIndent();
  stream << "if " << condition << ":\n";
  int then_scope = BeginScope();
  if (is_dynamic) {
    ++inside_dynamic_control_flow_;
  }
  std::ostringstream true_value;
  PrintPtoSelectValue_(op->args[1], op->dtype, true_value);
  if (is_dynamic) {
    --inside_dynamic_control_flow_;
  }
  PrintIndent();
  stream << result << " = " << true_value.str() << "\n";
  EndScope(then_scope);

  PrintIndent();
  stream << "else:\n";
  int else_scope = BeginScope();
  if (is_dynamic) {
    ++inside_dynamic_control_flow_;
  }
  std::ostringstream false_value;
  PrintPtoSelectValue_(op->args[2], op->dtype, false_value);
  if (is_dynamic) {
    --inside_dynamic_control_flow_;
  }
  PrintIndent();
  stream << result << " = " << false_value.str() << "\n";
  EndScope(else_scope);

  os << result;
}

void CodeGenTileLangPTO::VisitExpr_(const CallNode *op,
                                    std::ostream &os) { // NOLINT(*)
  // MODE_MERGING is legalized before codegen on the Ascend pipeline. The
  // resulting read-modify-write call still carries MODE_MERGING as its last
  // argument, so reject both raw and legalized forms before an arity check
  // can obscure the unsupported PTO limitation.
  RejectPtoSimdMerging(op);

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

  if (op->op.same_as(builtin::if_then_else())) {
    PrintPtoIfThenElse_(op, os);
    return;
  }

  if (op->op.same_as(builtin::shift_left())) {
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative shift_left";
    ICHECK_LT(shift, 63) << "PTO codegen shift_left is too large: " << shift;
    const int64_t factor = 1LL << shift;
    // PTOAS does not yet expose a scalar shift-left operation. Keep the
    // multiply fallback in the operand's fixed-width type rather than
    // widening it and changing overflow semantics.
    ICHECK(IntegerLiteralFits(op->args[0].dtype(), factor))
        << "PTO codegen cannot preserve fixed-width shift_left semantics for "
           "shift factor "
        << factor << " with operand dtype " << op->args[0].dtype();
    os << "(";
    PrintExpr_(op->args[0], os);
    os << " * ";
    PrintIntegerLiteralForRuntimeBinary(factor, op->args[0].dtype(), os);
    os << ")";
    return;
  }

  if (op->op.same_as(builtin::shift_right())) {
    int64_t shift = 0;
    ICHECK(TryGetConstInt(op->args[1], &shift) && shift >= 0)
        << "PTO codegen only supports constant non-negative shift_right";
    ICHECK_LT(shift, 63) << "PTO codegen shift_right is too large: " << shift;
    // TODO: Replace this floor-division fallback with PTOAS signed/unsigned
    // scalar shift-right once those operations are supported. The i64
    // constant workaround is intentionally limited to the existing positive
    // index case where the widened divisor avoids int32 literal overflow.
    os << "(";
    PrintExpr_(op->args[0], os);
    os << " // ";
    PrintIntegerLiteralForRuntimeBinary(1LL << shift, op->args[0].dtype(), os);
    os << ")";
    return;
  }

  if (op->op.same_as(builtin_call_extern_) ||
      op->op.same_as(builtin_call_pure_extern_)) {
    ICHECK_GE(op->args.size(), 1U);
    std::string func_name = Downcast<StringImm>(op->args[0])->value;
    if (op->args.size() == 2U &&
        TryEmitPtoUnaryMath_(func_name, op->args[1], os)) {
      return;
    }
    if (func_name.find("tl::AscendAllReduce") != std::string::npos) {
      os << EmitPTOAllReduceExpr_(func_name, op);
      return;
    }
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
    ICHECK((dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32)) ||
           ((dtype.is_int() || dtype.is_uint()) && dtype.bits() == 32))
        << "PTO atomic operations support float16, float32, int32, and "
           "uint32, got "
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

  if (op->op.same_as(tl::ascend_set_atomic())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ascend_set_atomic expects (op_str, typed_zero)";
    const std::string atomic_op = Downcast<StringImm>(op->args[0])->value;
    ICHECK(atomic_op == "add" || atomic_op == "max" || atomic_op == "min")
        << "tl.ascend_set_atomic op must be add/max/min, got " << atomic_op;

    const DataType dtype = op->args[1].dtype();
    const char *dtype_api = nullptr;
    if (dtype == DataType::Float(32)) {
      dtype_api = "set_atomic_f32";
    } else if (dtype == DataType::Float(16)) {
      dtype_api = "set_atomic_f16";
    } else if (dtype == DataType::Int(16)) {
      dtype_api = "set_atomic_s16";
    } else if (dtype == DataType::Int(32)) {
      dtype_api = "set_atomic_s32";
    } else if (dtype == DataType::Int(8)) {
      dtype_api = "set_atomic_s8";
    } else if (dtype == DataType::BFloat(16)) {
      dtype_api = "set_atomic_bf16";
    }
    ICHECK(dtype_api != nullptr)
        << "Unsupported PTO store-atomic dtype: " << dtype;

    // AscendC already lowers T.set_atomic / T.set_atomic_none in
    // codegen_ascend.cc. Explicit T.set_atomic only needed this PTO
    // VisitExpr_ path (e.g. GEMM split-K / multi-core flush).
    PrintIndent();
    if (atomic_op == "add") {
      stream << "pto." << dtype_api << "(); pto.set_atomic_add()\n";
    } else {
      stream << "pto.set_atomic_" << atomic_op << "(); pto." << dtype_api
             << "()\n";
    }
    return;
  }

  if (op->op.same_as(tl::ascend_set_atomic_none())) {
    ICHECK_EQ(op->args.size(), 0U)
        << "tl.ascend_set_atomic_none expects no arguments";
    PrintIndent();
    stream << "pto.set_atomic_none()\n";
    return;
  }

  if (op->op.same_as(tl::ascend_set_copy_pad_value())) {
    auto binding_it = pad_binding_by_setter_.find(GetRef<Call>(op));
    ICHECK(binding_it != pad_binding_by_setter_.end())
        << "PTO copy padding setter was not analyzed";
    const int64_t binding_id = binding_it->second;
    auto dtype_it = pad_binding_dtype_by_id_.find(binding_id);
    ICHECK(dtype_it != pad_binding_dtype_by_id_.end())
        << "PTO copy padding setter has an unknown binding " << binding_id;
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.ascend_set_copy_pad_value expects exactly 1 argument";
    ICHECK_EQ(op->args[0].dtype(), dtype_it->second)
        << "PTO copy padding setter binding dtype changed from "
        << dtype_it->second << " to " << op->args[0].dtype();

    // Keep the setter value as a Python SSA binding. This is not a hardware
    // set_mov_pad_val; PTOAS receives it later as the MTE pad requirement and
    // owns any setter deduplication/placement optimization.
    std::string value = GetPTOCopyPadValueExpr_(op->args[0]);
    PrintIndent();
    stream << "_tl_pad_" << binding_id << " = " << value << "\n";
    return;
  }

  if (op->op.same_as(tl::ascend_copy_gm_to_ubuf())) {
    PrintIndent();
    stream << GetAscendCopyGmUbExpr_(op) << "\n";
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

  // MarkScalarDcacheBypass rewrites scalar accesses to writable GM buffers
  // into explicit intrinsics. The nested address_of(BufferLoad) is an lvalue
  // address descriptor, not a second GM load; extract its buffer and index and
  // lower the intrinsic to the PTODSL bypass wrapper.
  if (op->op.same_as(tl::ascend_read_gm_bypass_dcache())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.ascend_read_gm_bypass_dcache expects address_of(BufferLoad)";
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && addr->op.same_as(builtin::address_of()))
        << "tl.ascend_read_gm_bypass_dcache expects address_of(BufferLoad)";
    const auto *load = addr->args[0].as<BufferLoadNode>();
    ICHECK(load && load->indices.size() == 1U)
        << "tl.ascend_read_gm_bypass_dcache expects a flat BufferLoad";
    ICHECK(load->dtype.is_scalar() && op->dtype == load->dtype)
        << "PTO GM dcache bypass read requires a scalar result matching the "
           "addressed BufferLoad dtype, got result "
        << op->dtype << " and load " << load->dtype;
    std::string scope = ScopeOfBuffer(load->buffer.get());
    ICHECK(scope == "global" || scope.empty())
        << "PTO GM dcache bypass read expects a global buffer, got scope `"
        << scope << "`";
    DataType value_dtype = load->dtype;
    ValidatePtoGmBypassDtype(value_dtype);
    std::string base =
        PtoScalarPointerBase_(load->buffer->data.get(), value_dtype, scope);
    std::string index =
        RemoveOutermostParentheses(PrintExpr_(load->indices[0]));
    os << "_tl_pto_read_gm_bypass_dcache(" << base << ", " << index << ", "
       << PtoScalarType(value_dtype) << ")";
    return;
  }

  if (op->op.same_as(tl::ascend_write_gm_bypass_dcache())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ascend_write_gm_bypass_dcache expects "
           "(address_of(BufferLoad), value)";
    const auto *addr = op->args[0].as<CallNode>();
    ICHECK(addr && addr->op.same_as(builtin::address_of()))
        << "tl.ascend_write_gm_bypass_dcache expects address_of(BufferLoad)";
    const auto *load = addr->args[0].as<BufferLoadNode>();
    ICHECK(load && load->indices.size() == 1U)
        << "tl.ascend_write_gm_bypass_dcache expects a flat BufferLoad";
    ICHECK(load->dtype.is_scalar() && op->args[1].dtype().is_scalar())
        << "PTO GM dcache bypass write supports scalar stores only, got "
        << load->dtype << " and " << op->args[1].dtype();
    ICHECK_EQ(op->args[1].dtype().bits(), load->dtype.bits())
        << "PTO GM dcache bypass write requires matching value/load bit width, "
           "got value "
        << op->args[1].dtype() << " and load " << load->dtype;
    std::string scope = ScopeOfBuffer(load->buffer.get());
    ICHECK(scope == "global" || scope.empty())
        << "PTO GM dcache bypass write expects a global buffer, got scope `"
        << scope << "`";
    DataType value_dtype = load->buffer->dtype;
    ValidatePtoGmBypassDtype(value_dtype);
    // Print pointer, index, and value before emitting the statement because
    // printing an operand may flush auxiliary SSA statements into the stream.
    std::string base =
        PtoScalarPointerBase_(load->buffer->data.get(), value_dtype, scope);
    std::string index =
        RemoveOutermostParentheses(PrintExpr_(load->indices[0]));
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->args[1]));
    PrintIndent();
    stream << "_tl_pto_write_gm_bypass_dcache(" << base << ", " << index << ", "
           << value << ", " << PtoScalarType(value_dtype) << ")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_fill_l1())) {
    EmitAscendFillL1(op);
    return;
  }

  if (op->op.same_as(tl::ascend_load_cbuf_to_ca()) ||
      op->op.same_as(tl::ascend_load_cbuf_to_cb())) {
    EmitAscendLoadCbufToL0(op, op->op.same_as(tl::ascend_load_cbuf_to_ca()));
    return;
  }

  if (op->op.same_as(tl::ascend_gemm_l1())) {
    EmitAscendGemmL1(op);
    return;
  }

  if (op->op.same_as(tl::ascend_set_hf32_mode())) {
    // PTO represents HF32 as a per-MAD attribute. The mode analysis binds
    // each stateful TileLang setting to the GEMMs it reaches.
    return;
  }

  if (op->op.same_as(tl::ascend_mad())) {
    EmitAscendMad(op);
    return;
  }

  if (op->op.same_as(tl::ascend_copy_matrix_cc_to_ub())) {
    EmitAscendCopyMatrixCcToUb(op);
    return;
  }

  if (op->op.same_as(tl::ascend_blockscaled_gemm_l1())) {
    EmitAscendBlockscaledGemmL1(op);
    return;
  }

  if (op->op.same_as(tl::ascend_mad_mx())) {
    EmitAscendMadMx(op);
    return;
  }

  if (op->op.same_as(tl::ascend_copy_matrix_cc_to_gm())) {
    EmitAscendCopyMatrixCcToGm(op);
    return;
  }

  if (op->op.same_as(builtin::address_of())) {
    os << GetAddressOfExpr_(op);
    return;
  }

  if (op->op.same_as(builtin::tvm_access_ptr())) {
    os << GetAccessPtrExpr_(op);
    return;
  }

  if (op->op.same_as(tl::ascend_pipe_barrier())) {
    auto pipe_name = Downcast<StringImm>(op->args[0])->value;
    PrintIndent();
    stream << "pto.pipe_barrier(\"" << StripPipePrefix(pipe_name) << "\")\n";
    return;
  }

  if (op->op.same_as(tl::simd_mem_bar())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.simd.mem_bar expects exactly one barrier type";
    auto barrier_type = Downcast<StringImm>(op->args[0])->value;
    PrintIndent();
    stream << "pto.mem_bar(pto.BarrierType." << barrier_type << ")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_cross_core_set_flag())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.ascend_cross_core_set_flag expects exactly 3 arguments "
           "(mode_id int, pipe string, flag_id)";
    int64_t mode_id = 0;
    ICHECK(TryGetConstInt(op->args[0], &mode_id))
        << "PTO cross-core set mode_id must be a compile-time integer";
    // mode_id=0: inter-core FFTS sync -> pto.set_cross_block
    // mode_id=4: AIC<->AIV intra-block sync -> pto.set_intra_block
    ICHECK(mode_id == 0 || mode_id == 4)
        << "PTO cross-core set only supports mode_id=0 via "
           "pto.set_cross_block or mode_id=4 via pto.set_intra_block, got "
        << mode_id;
    const auto *pipe = op->args[1].as<StringImmNode>();
    ICHECK(pipe) << "PTO cross-core set pipe must be a string";
    std::string event_id = PtoIntraBlockEventId(
        op->args[2], RemoveOutermostParentheses(PrintExpr_(op->args[2])));
    PrintIndent();
    stream << (mode_id == 0 ? "pto.set_cross_block(\""
                            : "pto.set_intra_block(\"")
           << StripPipePrefix(pipe->value) << "\", " << event_id << ")\n";
    return;
  }

  if (op->op.same_as(tl::ascend_cross_core_wait_flag())) {
    ICHECK_EQ(op->args.size(), 3U)
        << "tl.ascend_cross_core_wait_flag expects exactly 3 arguments "
           "(mode_id int, pipe string, flag_id)";
    int64_t mode_id = 0;
    ICHECK(TryGetConstInt(op->args[0], &mode_id))
        << "PTO cross-core wait mode_id must be a compile-time integer";
    // mode_id=0: inter-core FFTS sync -> pto.wait_cross_block
    // mode_id=4: AIC<->AIV intra-block sync -> pto.wait_intra_block
    ICHECK(mode_id == 0 || mode_id == 4)
        << "PTO cross-core wait only supports mode_id=0 via "
           "pto.wait_cross_block or mode_id=4 via pto.wait_intra_block, got "
        << mode_id;
    const auto *pipe = op->args[1].as<StringImmNode>();
    ICHECK(pipe) << "PTO cross-core wait pipe must be a string";
    std::string event_id = PtoIntraBlockEventId(
        op->args[2], RemoveOutermostParentheses(PrintExpr_(op->args[2])));
    PrintIndent();
    stream << (mode_id == 0 ? "pto.wait_cross_block(\""
                            : "pto.wait_intra_block(\"")
           << StripPipePrefix(pipe->value) << "\", " << event_id << ")\n";
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

  if (op->op.same_as(tl::simd_vld())) {
    ICHECK(op->args.size() >= 2U && op->args.size() <= 3U)
        << "tl.simd.vld expects 2 or 3 arguments (addr, dist[, offset])";
    DataType elem_dtype = op->dtype.element_of();
    ICHECK_GT(op->dtype.lanes(), 1)
        << "tl.simd.vld should return a vector type, got " << op->dtype;
    std::string dist = Downcast<StringImm>(op->args[1])->value;
    std::string offset =
        op->args.size() == 3U
            ? RemoveOutermostParentheses(PrintExpr_(op->args[2]))
            : "pto.const(0)";
    os << "pto.vlds(" << PrintExpr_(op->args[0]) << ", " << offset
       << ", pto.vreg_type(" << op->dtype.lanes() << ", "
       << PtoScalarType(elem_dtype) << ")";
    if (!dist.empty() && dist != "NORM") {
      os << ", dist=\"" << dist << "\"";
    }
    os << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vld2())) {
    ICHECK(op->args.size() >= 2U && op->args.size() <= 3U)
        << "tl.simd.vld2 expects 2 or 3 arguments (addr, dist[, offset])";
    std::string dist = Downcast<StringImm>(op->args[1])->value;
    std::string offset =
        op->args.size() == 3U
            ? RemoveOutermostParentheses(PrintExpr_(op->args[2]))
            : "pto.const(0)";
    os << "pto.vldsx2(" << PrintExpr_(op->args[0]) << ", " << offset << ", \""
       << dist << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_pair_get())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.simd.pair_get expects exactly 2 arguments";
    os << "(" << PrintExpr_(op->args[0]) << ")[" << PrintExpr_(op->args[1])
       << "]";
    return;
  }

  if (op->op.same_as(tl::simd_vadd())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.simd.vadd expects 4 arguments (src0, src1, mask, mode)";
    CheckPtoSimdMode(op, 3, "tl.simd.vadd");
    os << "pto.vadd(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vmul()) || op->op.same_as(tl::simd_vdiv()) ||
      op->op.same_as(tl::simd_vmax()) || op->op.same_as(tl::simd_vor())) {
    ICHECK_EQ(op->args.size(), 4U) << "PTO binary SIMD op expects 4 arguments";
    const char *pto_op = op->op.same_as(tl::simd_vmul())   ? "vmul"
                         : op->op.same_as(tl::simd_vdiv()) ? "vdiv"
                         : op->op.same_as(tl::simd_vmax()) ? "vmax"
                                                           : "vor";
    const std::string op_name = std::string("tl.simd.") + pto_op;
    CheckPtoSimdMode(op, 3, op_name.c_str());
    if (op->op.same_as(tl::simd_vdiv())) {
      CheckPtoVdivPrecision(op, enable_fast_math_);
    }
    os << "pto." << pto_op << "(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vmuls())) {
    ICHECK_EQ(op->args.size(), 4U) << "tl.simd.vmuls expects 4 arguments";
    CheckPtoSimdMode(op, 3, "tl.simd.vmuls");
    os << "pto.vmuls(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vdup())) {
    ICHECK_EQ(op->args.size(), 3U) << "tl.simd.vdup expects 3 arguments";
    CheckPtoSimdMode(op, 2, "tl.simd.vdup");
    const DataType target_dtype = op->dtype.element_of();
    const DataType source_dtype = op->args[0].dtype();
    const std::string pto_dtype = PtoScalarType(target_dtype);

    const bool source_integer = source_dtype.is_int() || source_dtype.is_uint();
    const bool target_integer = target_dtype.is_int() || target_dtype.is_uint();
    const bool source_float =
        source_dtype.is_float() || source_dtype.is_bfloat16();
    const bool target_float =
        target_dtype.is_float() || target_dtype.is_bfloat16();

    // PTOAS infers the vdup result element type from the scalar operand and
    // mask; it does not see TileLang's result dtype.  Keep literals typed and
    // coerce dynamic scalars when the frontend source type differs from the
    // requested vector element type.
    auto print_vdup_scalar = [&](DataType scalar_dtype) {
      const std::string scalar_pto_dtype = PtoScalarType(scalar_dtype);
      if (const auto *imm = op->args[0].as<IntImmNode>()) {
        if (scalar_dtype.is_bool()) {
          os << (imm->value ? "True" : "False");
        } else {
          os << scalar_pto_dtype << "(" << imm->value << ")";
        }
        return;
      }
      if (const auto *imm = op->args[0].as<FloatImmNode>()) {
        os << scalar_pto_dtype << "(float.fromhex('"
           << FlexibleHexFormat(imm->value) << "'))";
        return;
      }
      if (source_dtype != scalar_dtype) {
        os << "scalar.cast(" << PrintExpr_(op->args[0]) << ", " << pto_dtype
           << ")";
      } else {
        PrintExpr_(op->args[0], os);
      }
    };

    if ((source_float && target_integer) || (source_integer && target_float)) {
      ICHECK_EQ(source_dtype.bits(), target_dtype.bits())
          << "PTO tl.simd.vdup cross-domain conversion requires matching "
             "widths, got "
          << source_dtype << " -> " << target_dtype;
      os << "pto.vcvt(pto.vdup(";
      print_vdup_scalar(source_dtype);
      os << ", " << PrintExpr_(op->args[1]) << "), " << pto_dtype << ", "
         << PrintExpr_(op->args[1]);
      if (source_float) {
        os << ", rnd=\"Z\", sat=\"SAT\"";
      } else {
        os << ", rnd=\"R\"";
      }
      os << ")";
      return;
    }

    os << "pto.vdup(";
    print_vdup_scalar(target_dtype);
    os << ", " << PrintExpr_(op->args[1]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vcmax()) || op->op.same_as(tl::simd_vcadd())) {
    ICHECK_EQ(op->args.size(), 3U) << "tl.simd.vcmax/vcadd expects 3 arguments";
    CheckPtoSimdMode(op, 2,
                     op->op.same_as(tl::simd_vcmax()) ? "tl.simd.vcmax"
                                                      : "tl.simd.vcadd");
    os << (op->op.same_as(tl::simd_vcmax()) ? "pto.vcmax(" : "pto.vcadd(")
       << PrintExpr_(op->args[0]) << ", " << PrintExpr_(op->args[1]) << ")";
    return;
  }

  if (op->op.same_as(tl::simd_vexpdif())) {
    ICHECK_EQ(op->args.size(), 4U) << "tl.simd.vexpdif expects 4 arguments";
    int64_t part = ConstArgDim(op, 3, "tl.simd.vexpdif part");
    ICHECK(part == 0 || part == 1)
        << "tl.simd.vexpdif part must be 0 (EVEN) or 1 (ODD), got " << part;
    os << "pto.vexpdif(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2])
       << ", part=\"" << (part == 0 ? "EVEN" : "ODD") << "\")";
    return;
  }

  if (op->op.same_as(tl::simd_vcvt())) {
    ICHECK_GE(op->args.size(), 3U)
        << "tl.simd.vcvt expects source, mask, and mode";
    CheckPtoSimdMode(op, op->args.size() - 1, "tl.simd.vcvt");
    os << "pto.vcvt(" << PrintExpr_(op->args[0]) << ", "
       << PtoScalarType(op->dtype.element_of()) << ", "
       << PrintExpr_(op->args[1]);
    for (size_t i = 2; i + 1 < op->args.size(); ++i) {
      const auto *value = op->args[i].as<StringImmNode>();
      ICHECK(value) << "tl.simd.vcvt option " << i
                    << " must be a constant string, got " << op->args[i];
      const std::string &token = value->value;
      if (StartsWith(token, "ROUND_")) {
        ICHECK(token == "ROUND_R" || token == "ROUND_A" || token == "ROUND_F" ||
               token == "ROUND_C" || token == "ROUND_Z" || token == "ROUND_O" ||
               token == "ROUND_H")
            << "Unsupported tl.simd.vcvt rounding token: " << token;
        os << ", rnd=\"" << token.substr(6) << "\"";
      } else if (token == "RS_ENABLE" || token == "RS_DISABLE") {
        os << ", sat=\"" << (token == "RS_ENABLE" ? "SAT" : "NOSAT") << "\"";
      } else if (StartsWith(token, "PART_")) {
        const std::string part = token.substr(5);
        ICHECK(part == "EVEN" || part == "ODD" || part == "P0" ||
               part == "P1" || part == "P2" || part == "P3")
            << "Unsupported tl.simd.vcvt part token: " << token;
        // PTODSL expects P0..P3, while the frontend token is PART_P0..P3.
        os << ", part=\"" << part << "\"";
      } else {
        LOG(FATAL) << "Unsupported tl.simd.vcvt option token: " << token;
      }
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
      if (StartsWith(dist, "ONEPT_")) {
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
    PrintIndent();
    stream << "pto.vsts(" << PrintExpr_(op->args[1]) << ", "
           << PrintExpr_(op->args[0]) << ", " << offset << ", "
           << PrintExpr_(op->args[2]);
    if (!dist.empty()) {
      stream << ", dist=\"" << dist << "\"";
    }
    stream << ")\n";
    return;
  }

  if (op->op.same_as(tl::simd_vsstb())) {
    ICHECK(op->args.size() == 4U || op->args.size() == 5U)
        << "tl.simd.vsstb expects 4 or 5 arguments";
    std::string stride = PrintExpr_(op->args[2]);
    os << "pto.vsstb(" << PrintExpr_(op->args[0]) << ", "
       << PrintExpr_(op->args[1]) << ", ((" << stride << " >> 16) & 65535), ("
       << stride << " & 65535), " << PrintExpr_(op->args[3]);
    if (op->args.size() == 5U) {
      os << ", post_update=pto.PostUpdate.ON";
    }
    os << ")";
    return;
  }

  if (auto opt_call_op = op->op.as<Op>()) {
    const auto &call_op = opt_call_op.value();
    std::string op_name = call_op->name;
    if (op->args.size() == 1U &&
        TryEmitPtoUnaryMath_(op_name, op->args[0], os)) {
      return;
    }
    if (StartsWith(op_name, "tl.vmi.")) {
      if (op_name == "tl.vmi.pair_get") {
        ICHECK_EQ(op->args.size(), 2U)
            << "tl.vmi.pair_get expects exactly 2 arguments";
        os << "(" << PrintExpr_(op->args[0]) << ")[" << PrintExpr_(op->args[1])
           << "]";
        return;
      }
      PrintPtoVmiCall_(op, os);
      return;
    }
    if (op->op.same_as(tl::rng_rand()) ||
        op->op.same_as(tl::rng_rand_float())) {
      os << EmitRngDrawExpr(op);
      return;
    }
    if (op_name == "tl.loop_break") {
      os << "break";
      return;
    }
    if (op_name == "tir.reinterpret" || op_name == "tirx.reinterpret") {
      ICHECK_EQ(op->args.size(), 1U) << op_name << " expects one argument";
      os << "pto.vbitcast(" << PrintExpr_(op->args[0]) << ", "
         << PtoScalarType(op->dtype.element_of()) << ")";
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
      os << ", " << PtoFP8TypeName(to)
         << ", rounding=\"r\", saturation=\"sat\")";
      return;
    }
    LOG(FATAL) << "PTO SIMT FP8 cast currently supports float32x2 to "
                  "float8_e4m3fnx2/float8_e5m2x2 only, got "
               << from << " -> " << to;
  }

  const bool from_integer = from.is_int() || from.is_uint();
  const bool to_integer = to.is_int() || to.is_uint();
  const bool from_float = from.is_float() || from.is_bfloat16();
  const bool to_float = to.is_float() || to.is_bfloat16();

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
    os << " != pto.const(0, dtype=" << PtoTypeName(from) << "))";
    return;
  }

  if (from.is_bool() && to_integer) {
    ICHECK(from.is_scalar() && to.is_scalar())
        << "PTO bool-to-integer cast expects scalar types, got " << from
        << " -> " << to;
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << "pto.const(" << (imm->value != 0 ? 1 : 0)
         << ", dtype=" << PtoTypeName(to) << ")";
      return;
    }
    os << "scalar.select(";
    PrintExpr_(op->value, os);
    os << ", pto.const(1, dtype=" << PtoTypeName(to)
       << "), pto.const(0, dtype=" << PtoTypeName(to) << "))";
    return;
  }

  if (from_integer && to_integer) {
    ICHECK(from.is_scalar() && to.is_scalar())
        << "PTO integer cast currently supports scalar values only, got "
        << from << " -> " << to;
    os << "scalar.cast(";
    PrintExpr_(op->value, os);
    os << ", " << PtoTypeName(to) << ")";
    return;
  }

  if (from_integer && to_float) {
    ICHECK(from.is_scalar() && to.is_scalar())
        << "PTO integer-to-float cast currently supports scalar values only, "
           "got "
        << from << " -> " << to;
    if (const auto *imm = op->value.as<IntImmNode>()) {
      os << "pto.const(" << imm->value << ", dtype=" << PtoTypeName(to) << ")";
      return;
    }
    if (inside_simtvf_body_) {
      // pto.convert is a SIMT operation and cannot represent a conversion in
      // the regular scalar domain.
      os << "pto.convert(";
      PrintExpr_(op->value, os);
      os << ", " << PtoTypeName(to)
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
         << "'), dtype=" << PtoTypeName(to) << ")";
      return;
    }
    os << "scalar.cast(";
    PrintExpr_(op->value, os);
    os << ", " << PtoTypeName(to) << ")";
    return;
  }

  LOG(FATAL) << "Unsupported PTO cast: " << from << " -> " << to;
}

void CodeGenTileLangPTO::VisitExpr_(const FloatImmNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (op->dtype.is_bfloat16()) {
    // The base class prints float16 literals via Python float(...), which has
    // no bf16 equivalent; emit an explicitly typed PTO constant instead.
    std::ostringstream temp;
    temp << "pto.const(float.fromhex('" << FlexibleHexFormat(op->value)
         << "'), dtype=pto.bf16)";
    MarkConst(temp.str());
    os << temp.str();
    return;
  }
  CodeGenTileLangPY::VisitExpr_(op, os);
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
  PrintPtoFloatMinMax_("min", op->dtype, op->a, op->b, os);
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
  PrintPtoFloatMinMax_("max", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::PrintPtoFloatMinMax_(const char *op_name,
                                              DataType dtype, PrimExpr lhs,
                                              PrimExpr rhs,
                                              std::ostream &os) { // NOLINT(*)
  ICHECK(IsSupportedPTOFloatMinMaxType(dtype))
      << "PTO floating min/max currently supports f16, f32, bf16, "
         "vector<2xf16>, vector<2xf32>, and vector<2xbf16>, got "
      << dtype;

  // PTO's packed min/max micro-ops omit f32x2. Apply the supported scalar f32
  // operation lane-by-lane and repack the pair instead.
  if (inside_simtvf_body_ && IsFloat32Pair(dtype)) {
    os << "_tl_vectorize_binary_f32x2(pto.f" << op_name << ", ";
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

bool CodeGenTileLangPTO::TryEmitPtoUnaryMath_(const std::string &name,
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
    ICHECK(IsSupportedPTOSIMTUnaryMathType(dtype))
        << "PTO SIMT " << name
        << " currently supports f16, f32, vector<2xf16>, and vector<2xf32>, "
           "got "
        << dtype;
  } else {
    ICHECK(IsSupportedPTOScalarUnaryMathType(dtype))
        << "PTO scalar " << name
        << " currently supports scalar f16, f32, and bf16, got " << dtype;
  }
  const UnaryMathForm &form = it->second;
  std::string value = PrintExpr_(arg);
  // PTO's packed unary micro-ops omit f32x2. Scalarize these operations just
  // like min/max above so PTOAS receives only supported scalar f32 ops.
  if (inside_simtvf_body_ && IsFloat32Pair(dtype)) {
    if (form.reciprocal) {
      os << "_tl_vectorize_unary_f32x2(_tl_scalar_rsqrt, " << value << ")";
    } else {
      std::string simt_op = form.simt;
      ICHECK(!simt_op.empty() && simt_op.back() == '(');
      simt_op.pop_back();
      os << "_tl_vectorize_unary_f32x2(" << simt_op << ", " << value << ")";
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

void CodeGenTileLangPTO::VisitExpr_(const AndNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("&", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::VisitExpr_(const OrNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("|", op->dtype, op->a, op->b, os);
}

void CodeGenTileLangPTO::PrintPtoLogicalNot_(const PrimExpr &value,
                                             std::ostream &os) {
  ICHECK(value.dtype().is_scalar() && value.dtype().is_bool())
      << "PTO logical not expects a scalar boolean operand, got "
      << value.dtype();
  os << "scalar.select(";
  PrintExpr_(value, os);
  os << ", pto.const(0, dtype=pto.i1), pto.const(1, dtype=pto.i1))";
}

void CodeGenTileLangPTO::VisitExpr_(const NotNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (const auto *imm = op->a.as<IntImmNode>()) {
    ICHECK(imm->dtype.is_bool())
        << "PTO logical not only accepts a boolean constant, got "
        << imm->dtype;
    os << (imm->value ? "False" : "True");
    return;
  }
  auto can_invert_comparison = [](DataType dtype) {
    return dtype.is_int() || dtype.is_uint();
  };
  if (const auto *eq = op->a.as<EQNode>();
      eq != nullptr && can_invert_comparison(eq->a.dtype())) {
    PrintBinaryExpr_("!=", op->dtype, eq->a, eq->b, os);
    return;
  }
  if (const auto *ne = op->a.as<NENode>();
      ne != nullptr && can_invert_comparison(ne->a.dtype())) {
    PrintBinaryExpr_("==", op->dtype, ne->a, ne->b, os);
    return;
  }
  if (const auto *lt = op->a.as<LTNode>();
      lt != nullptr && can_invert_comparison(lt->a.dtype())) {
    PrintBinaryExpr_(">=", op->dtype, lt->a, lt->b, os);
    return;
  }
  if (const auto *le = op->a.as<LENode>();
      le != nullptr && can_invert_comparison(le->a.dtype())) {
    PrintBinaryExpr_(">", op->dtype, le->a, le->b, os);
    return;
  }
  if (const auto *gt = op->a.as<GTNode>();
      gt != nullptr && can_invert_comparison(gt->a.dtype())) {
    PrintBinaryExpr_("<=", op->dtype, gt->a, gt->b, os);
    return;
  }
  if (const auto *ge = op->a.as<GENode>();
      ge != nullptr && can_invert_comparison(ge->a.dtype())) {
    PrintBinaryExpr_("<", op->dtype, ge->a, ge->b, os);
    return;
  }
  PrintPtoLogicalNot_(op->a, os);
}

void CodeGenTileLangPTO::VisitExpr_(const SelectNode *op,
                                    std::ostream &os) { // NOLINT(*)
  PrintPtoSelect_(op->condition, op->true_value, op->false_value, op->dtype,
                  os);
}

void CodeGenTileLangPTO::VisitExpr_(const LetNode *op,
                                    std::ostream &os) { // NOLINT(*)
  // Bind once as a statement, then return the bound name. Preserve Let
  // once-eval (no Substitute). VisitStmt_(For) prints bounds before the
  // header so these assignments land on their own lines. After #382
  // (shape >= 0) some Persistent extents keep Lets that used to fold.
  std::string value = PrintExpr_(op->value);
  ICHECK(!var_idmap_.count(op->var.get()));
  PrintIndent();
  stream << AllocVarID(op->var.get()) << " = " << value << "\n";
  os << PrintExpr_(op->body);
  bool removed = var_idmap_.erase(op->var.get());
  ICHECK(removed);
}

void CodeGenTileLangPTO::PrintBinaryExpr_(const std::string &opstr,
                                          DataType dtype, PrimExpr lhs,
                                          PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)
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
    os << "_tl_vectorize_binary_f32x2(_tl_scalar_div, " << PrintExpr_(lhs)
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
  ICHECK(IsFloat32(elem_dtype))
      << "PTO vector broadcast currently supports float32 only, got "
      << elem_dtype;
  std::string value = PrintExpr_(op->value);
  os << "pto.Vec(pto.f32, " << op->dtype.lanes() << ", init=" << value << ")";
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
  }

  PrintSSAAssign(AllocVarID(op->var.get()), PrintExpr_(op->value),
                 op->var.dtype());
}

void CodeGenTileLangPTO::VisitStmt_(const AllocBufferNode *op) {
  const Var &buffer_var = op->buffer->data;
  std::string scope = GetPtrStorageScope(buffer_var);

  // VMI/SIMD mutable registers are local.var buffers even for vector-only
  // kernels (which do not enter the GEMM allocation path).
  if (scope == "local.var") {
    CheckPTOLocalVarBuffer(op->buffer.get());
    PrintIndent();
    local_var_buffers_.insert(buffer_var.get());
    if (op->buffer->dtype.is_handle()) {
      stream << AllocVarID(buffer_var.get()) << " = None\n";
    } else if (op->buffer->dtype.lanes() > 1) {
      stream << AllocVarID(buffer_var.get()) << " = pto.vmi.vreg("
             << op->buffer->dtype.lanes() << ", "
             << PtoTypeName(op->buffer->dtype.element_of()) << ")\n";
    } else {
      stream << AllocVarID(buffer_var.get()) << " = "
             << PtoLocalVarInitialValue(op->buffer->dtype) << "\n";
    }
    RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
    return;
  }

  // Local buffers retain their ordinary lexical allocation semantics even
  // when another part of the function contains GEMM.  In particular, SIMT
  // local buffers must materialize as pto.alloc_buffer instead of falling
  // through the GEMM address-only allocation path.
  if (scope == "local" || scope == "local.fragment") {
    EmitPtoBufferAllocation(op->buffer);
    return;
  }

  if (!current_function_has_gemm_) {
    EmitPtoBufferAllocation(op->buffer);
    return;
  }

  alloc_storage_scope_[buffer_var.get()] = scope;

  auto pto_space = PtoSpaceForStorageScope(scope);
  if (pto_space.has_value() && *pto_space != "gm") {
    PrintIndent();
    stream << AllocVarID(buffer_var.get())
           << " = pto.castptr(pto.const(0, dtype=pto.i64), "
           << PtoPtrType(op->buffer->dtype, *pto_space) << ")\n";
  }

  RegisterHandleType_(buffer_var.get(), op->buffer->dtype);
}

void CodeGenTileLangPTO::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tirx::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x") {
      var_idmap_[iv->var.get()] = "pto.get_block_idx()";
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
    VisitStmt(op->body);
    return;
  }

  if (op->attr_key == "tl.simtvf_scope") {
    VisitStmt(op->body);
    return;
  }

  VisitStmt(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const ForNode *op) {
  // Trace-time Python iteration is safe only for explicitly unrolled loops
  // whose full triplet is known. Every other loop must remain device-side.
  arith::Analyzer analyzer;
  PrimExpr start = analyzer.Simplify(op->min);
  PrimExpr extent = analyzer.Simplify(op->extent);
  PrimExpr step = op->step.has_value() ? analyzer.Simplify(op->step.value())
                                       : make_const(op->loop_var.dtype(), 1);
  PrimExpr stop = analyzer.Simplify(start + extent);

  int64_t start_value = 0;
  int64_t extent_value = 0;
  int64_t step_value = 0;
  const bool use_static_range = op->kind == tirx::ForKind::kUnrolled &&
                                TryGetConstInt(start, &start_value) &&
                                TryGetConstInt(extent, &extent_value) &&
                                TryGetConstInt(step, &step_value);

  // Print bounds before the header so nested Let assignments land above it.
  std::string start_str = PrintExpr_(start);
  std::string stop_str = PrintExpr_(stop);
  std::string step_str = PrintExpr_(step);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  const std::vector<const VarNode *> carry_vars =
      use_static_range ? std::vector<const VarNode *>{}
                       : CollectLoopCarriedLocalVars(op->body);
  if (use_static_range) {
    stream << "for " << vid << " in pto.static_range(" << start_str << ", "
           << stop_str << ", " << step_str << "):\n";
  } else if (!carry_vars.empty()) {
    // Python range + PTODSL ast_rewrite infers scf.for iter_args from
    // ``acc = f(acc, ...)`` stores of an outer vector local.var.
    stream << "for " << vid << " in range(" << start_str << ", " << stop_str
           << ", " << step_str << "):\n";
  } else {
    stream << "for " << vid << " in range(" << start_str << ", " << stop_str
           << ", " << step_str << "):\n";
  }
  int scope = BeginScope();
  if (!use_static_range) {
    ++inside_dynamic_control_flow_;
  }
  PrintStmt_(op->body);
  EndScope(scope);
  if (!use_static_range) {
    --inside_dynamic_control_flow_;
  }
}

void CodeGenTileLangPTO::VisitStmt_(const WhileNode *op) {
  ++inside_dynamic_control_flow_;
  std::string cond = RemoveOutermostParentheses(PrintExpr_(op->condition));
  PrintIndent();
  stream << "while " << cond << ":\n";
  int while_scope = BeginScope();
  if (tirx::is_no_op(op->body)) {
    PrintIndent();
    stream << "pass\n";
  } else {
    PrintStmt_(op->body);
  }
  EndScope(while_scope);
  --inside_dynamic_control_flow_;
}

void CodeGenTileLangPTO::VisitStmt_(const SBlockNode *op) {
  if (op->name_hint == "CUBE" || op->name_hint == "VECTOR") {
    if (!current_function_has_mixed_sections_) {
      const bool is_empty = op->alloc_buffers.empty() && !op->init.defined() &&
                            tirx::is_no_op(op->body);
      if (is_empty) {
        PrintIndent();
        stream << "pass\n";
        return;
      }
      for (const Buffer &buf : op->alloc_buffers) {
        EmitPtoBufferAllocation(buf);
      }
      if (op->init.defined()) {
        PrintStmt_(op->init.value());
      }
      PrintStmt_(op->body);
      return;
    }
    PrintIndent();
    stream << "with pto.section(\""
           << (op->name_hint == "CUBE" ? "cube" : "vector") << "\"):\n";
    int section_scope = BeginScope();
    bool is_empty = op->alloc_buffers.empty() && !op->init.defined() &&
                    tirx::is_no_op(op->body);
    if (is_empty) {
      PrintIndent();
      stream << "pass\n";
    }
    for (const Buffer &buf : op->alloc_buffers) {
      EmitPtoBufferAllocation(buf);
    }
    if (op->init.defined()) {
      PrintStmt_(op->init.value());
    }
    PrintStmt_(op->body);
    EndScope(section_scope);
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
    PrintIndent();
    stream << "with pto.vecscope():\n";
    int vecscope = BeginScope();
    const auto body_start = stream.tellp();
    for (const Buffer &buf : op->alloc_buffers) {
      EmitPtoBufferAllocation(buf);
    }
    if (op->init.defined()) {
      PrintStmt_(op->init.value());
    }
    PrintStmt_(op->body);
    if (stream.tellp() == body_start) {
      PrintIndent();
      stream << "pass\n";
    }
    EndScope(vecscope);
    return;
  }

  if (op->init.defined()) {
    PrintStmt_(op->init.value());
  }
  PrintStmt_(op->body);
}

void CodeGenTileLangPTO::VisitStmt_(const IfThenElseNode *op) {
  PrimExpr simplified_condition = arith::Analyzer().Simplify(op->condition);
  int64_t constant_condition = 0;
  const bool is_dynamic =
      !TryGetConstInt(simplified_condition, &constant_condition);
  std::string cond = RemoveOutermostParentheses(PrintExpr_(op->condition));
  PrintIndent();
  stream << "if " << cond << ":\n";
  int if_scope = BeginScope();
  if (is_dynamic) {
    ++inside_dynamic_control_flow_;
  }
  PrintStmt_(op->then_case);
  if (is_dynamic) {
    --inside_dynamic_control_flow_;
  }
  EndScope(if_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    if (is_dynamic) {
      ++inside_dynamic_control_flow_;
    }
    PrintStmt_(op->else_case.value());
    if (is_dynamic) {
      --inside_dynamic_control_flow_;
    }
    EndScope(else_scope);
  }
}

void CodeGenTileLangPTO::VisitStmt_(const EvaluateNode *op) {
  if (is_const_int(op->value))
    return;
  if (const auto *call = op->value.as<CallNode>()) {
    if (call->op.same_as(tl::rng_init())) {
      EmitRngInit(call);
      return;
    }
    if (call->op.same_as(tl::loop_break())) {
      PrintIndent();
      stream << "break\n";
      return;
    }
  }
  std::string emitted = PrintExpr_(op->value);
  if (!emitted.empty()) {
    PrintIndent();
    stream << emitted << "\n";
  }
}

void CodeGenTileLangPTO::EmitRngInit(const CallNode *op) {
  ICHECK_EQ(op->args.size(), 4U) << "tl.rng_init expects exactly 4 arguments "
                                    "(seed, seq, off, generator)";
  ICHECK(inside_simtvf_body_)
      << "tl.rng_init on PTO must be used inside a T.SimtVF(...) block";
  ICHECK_EQ(inside_dynamic_control_flow_, 0)
      << "PTO RNG initialization is not supported inside dynamic control "
         "flow; RNG state is tracked at trace time, so initialize it in "
         "straight-line code before entering a dynamic branch or loop";
  // args[3] (generator string) is intentionally ignored, matching the Ascend
  // backend: both targets fix Philox as the generator.
  uses_rng_ = true;
  rng_state_var_ = name_supply_->FreshName("_tl_rng_state");
  PrintIndent();
  stream << rng_state_var_ << " = PhiloxRNG(" << PrintExpr_(op->args[0]) << ", "
         << PrintExpr_(op->args[1]) << ", " << PrintExpr_(op->args[2]) << ")\n";
}

std::string CodeGenTileLangPTO::EmitRngDrawExpr(const CallNode *op) {
  ICHECK(inside_simtvf_body_)
      << "PTO RNG draw must be used inside a T.SimtVF(...) block";
  ICHECK(!rng_state_var_.empty())
      << "PTO RNG draw requires a preceding T.rng_init call in the same "
         "T.SimtVF(...) block";
  ICHECK(inside_dynamic_control_flow_ == 0)
      << "PTO RNG draw is not supported inside dynamic control flow; RNG "
         "state is tracked at trace time, so use straight-line code or "
         "statically unrolled loops (pto.static_range)";
  if (op->op.same_as(tl::rng_rand())) {
    ICHECK_EQ(op->args.size(), 0U) << "tl.rng_rand expects no arguments";
    return rng_state_var_ + ".rand()";
  }
  ICHECK_EQ(op->args.size(), 1U)
      << "tl.rng_rand_float expects exactly one argument (distribution)";
  ICHECK_NE(op->dtype.bits(), 64)
      << "float64 RNG (tl.rng_rand_float bit=64) is not supported on PTO, "
         "matching the Ascend backend";
  const auto *dist_imm = op->args[0].as<StringImmNode>();
  ICHECK(dist_imm)
      << "tl.rng_rand_float distribution must be a constant string, got "
      << op->args[0];
  const std::string dist = dist_imm->value;
  ICHECK(dist == "uniform" || dist == "normal")
      << "Unsupported PTO RNG distribution: " << dist;
  return rng_state_var_ + ".rand_" + dist + "()";
}

bool CodeGenTileLangPTO::TryEmitRngBroadcastStore(const BufferStoreNode *op) {
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
    PrintIndent();
    stream << tmp << " = "
           << RemoveOutermostParentheses(PrintExpr_(broadcast->value)) << "\n";
    PrimExpr lane_index =
        analyzer.Simplify(base_index + make_const(base_index.dtype(), lane));
    PrintIndent();
    EmitPtoScalarStore(op->buffer.get(), tmp, lane_index);
  }
  return true;
}

void CodeGenTileLangPTO::VisitExpr_(const BufferLoadNode *op,
                                    std::ostream &os) { // NOLINT(*)
  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // T.alloc_var lowers to local.var[0]. Read it back as the scalar surface
    // value used by GEMM tile mapping, not as a general PTO buffer load.
    CheckPTOLocalVarBuffer(op->buffer.get());
    if (op->buffer->dtype.lanes() > 1) {
      os << LocalVarID(op->buffer->data.get());
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var load expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var load expects index 0";
    os << LocalVarID(op->buffer->data.get());
    return;
  }

  if (IsVmiLocalRegisterBuffer(op->buffer.get())) {
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO VMI local register buffers must be flattened before codegen";
    ICHECK(!op->predicate.defined())
        << "PTO VMI local register buffers do not support predicated loads";
    CheckVmiLocalRegisterIndex(op->buffer.get(), op->indices[0]);
    os << GetVarID(op->buffer->data.get()) << "["
       << RemoveOutermostParentheses(PrintExpr_(op->indices[0])) << "]";
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

  os << PtoScalarLoad(op->buffer.get(), op->indices[0]);
}

void CodeGenTileLangPTO::EmitScalarizedLoad(const BufferLoadNode *op,
                                            std::ostream &os) {
  DataType value_dtype = op->dtype;
  DataType element_dtype = op->buffer->dtype;
  ICHECK_EQ(element_dtype.lanes(), 1)
      << "PTO vector BufferLoad scalarization currently expects scalar "
         "buffer elements, got "
      << element_dtype;

  if (inside_simtvf_body_) {
    ICHECK(IsSupportedSIMTLocalStorageType(element_dtype))
        << "PTO SIMT vector BufferLoad currently supports float16, bfloat16, "
           "float32, int32, uint32, and float8_e4m3fn/float8_e5m2 storage, "
           "got "
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
    std::string scope = ScopeOfBuffer(op->buffer.get());
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
    if (auto pto_space = PtoSpaceForStorageScope(scope)) {
      std::string base = GetVarID(op->buffer->data.get());
      if (*pto_space != "gm" &&
          (HandleTypeMatch_(op->buffer->data.get(), DataType::Int(8)) ||
           !HandleTypeMatch_(op->buffer->data.get(), element_dtype))) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(element_dtype, *pto_space) + ")";
      }
      os << "scalar.load(" << base << ", " << index_str
         << ", contiguous=" << lanes << ")";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector load scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector BufferLoad is not supported yet";
}

void CodeGenTileLangPTO::VisitStmt_(const BufferStoreNode *op) {
  if (IsLocalVarBuffer(op->buffer->data.get())) {
    // T.alloc_var lowers to local.var[0]. Store it as a scalar surface value
    // for GEMM tile mapping, not as a general PTO buffer store.
    CheckPTOLocalVarBuffer(op->buffer.get());
    if (op->buffer->dtype.lanes() > 1) {
      PrintIndent();
      stream << LocalVarID(op->buffer->data.get()) << " = "
             << RemoveOutermostParentheses(PrintExpr_(op->value)) << "\n";
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO local.var store expects a scalar buffer";
    int64_t index = 0;
    ICHECK(TryGetConstInt(op->indices[0], &index) && index == 0)
        << "PTO local.var store expects index 0";
    PrintIndent();
    stream << LocalVarID(op->buffer->data.get()) << " = "
           << PtoLocalVarStoreValue(
                  op->buffer->dtype,
                  RemoveOutermostParentheses(PrintExpr_(op->value)))
           << "\n";
    return;
  }

  if (IsVmiLocalRegisterBuffer(op->buffer.get())) {
    ICHECK_EQ(op->indices.size(), 1U)
        << "PTO VMI local register buffers must be flattened before codegen";
    ICHECK(!op->predicate.defined())
        << "PTO VMI local register buffers do not support predicated stores";
    CheckVmiLocalRegisterIndex(op->buffer.get(), op->indices[0]);
    PrintIndent();
    stream << GetVarID(op->buffer->data.get()) << "["
           << RemoveOutermostParentheses(PrintExpr_(op->indices[0]))
           << "] = " << RemoveOutermostParentheses(PrintExpr_(op->value))
           << "\n";
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
  EmitPtoScalarStore(op->buffer.get(), value, op->indices[0]);
}

void CodeGenTileLangPTO::EmitScalarizedStore(const BufferStoreNode *op) {
  ICHECK_EQ(op->buffer->dtype.lanes(), 1)
      << "PTO vector BufferStore scalarization currently expects scalar "
         "buffer elements, got "
      << op->buffer->dtype;

  if (inside_simtvf_body_) {
    DataType value_dtype = op->value.dtype();
    ICHECK(IsSupportedSIMTLocalStorageType(op->buffer->dtype))
        << "PTO SIMT vector BufferStore currently supports float16, "
           "bfloat16, float32, int32, uint32, and "
           "float8_e4m3fn/float8_e5m2 storage only, got "
        << op->buffer->dtype;
    ICHECK_EQ(value_dtype.element_of(), op->buffer->dtype)
        << "PTO SIMT vector BufferStore expects the value element dtype to "
           "match the buffer element dtype, got "
        << value_dtype << " vs " << op->buffer->dtype;
    if (TryEmitRngBroadcastStore(op)) {
      return;
    }
    ICHECK(!tl::IsAscendVectorizableFP8(op->buffer->dtype) ||
           IsSupportedSIMTFP8ContiguousLaneCount(value_dtype.lanes()))
        << "PTO SIMT FP8 BufferStore currently supports contiguous lanes 2, "
           "4, and 8 only, got "
        << value_dtype;
    std::string scope = ScopeOfBuffer(op->buffer.get());
    std::string value = RemoveOutermostParentheses(PrintExpr_(op->value));
    std::string index_str =
        RemoveOutermostParentheses(PrintExpr_(op->indices[0]));
    if (const auto *ramp = op->indices[0].as<RampNode>()) {
      CheckContiguousRampStride(op->indices[0], "store");
      index_str = RemoveOutermostParentheses(PrintExpr_(ramp->base));
    }
    PrintIndent();
    if (scope == "local.fragment" || scope == "local") {
      stream << "scalar.store(" << value << ", "
             << GetVarID(op->buffer->data.get()) << ", " << index_str << ")\n";
      return;
    }
    if (auto pto_space = PtoSpaceForStorageScope(scope)) {
      std::string base = GetVarID(op->buffer->data.get());
      if (*pto_space != "gm" &&
          (HandleTypeMatch_(op->buffer->data.get(), DataType::Int(8)) ||
           !HandleTypeMatch_(op->buffer->data.get(), op->buffer->dtype))) {
        base = "pto.castptr(" + base + ", " +
               PtoPtrType(op->buffer->dtype, *pto_space) + ")";
      }
      stream << "scalar.store(" << value << ", " << base << ", " << index_str
             << ")\n";
      return;
    }
    LOG(FATAL) << "Unsupported PTO SIMT vector store scope: " << scope;
  }

  LOG(FATAL) << "PTO non-SIMT vector BufferStore is not supported yet";
}

} // namespace codegen
} // namespace tvm
