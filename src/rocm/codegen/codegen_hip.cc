#include "compat/analyzer.h"
/*!
 * \file target/codegen.cc
 */

#include "codegen_hip.h"
#include "op/prim.h"
#include "support/check.h"
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt_functor.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend/common/codegen/codegen_py.h"
#include "backend/common/target_utils.h"
#include "compat/expr.h"
#include "compat/visit.h"
#include "rocm/op/builtin.h"
#include "transform/common/storage_size.h"
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

namespace tvm {
namespace codegen {

using namespace ffi;
using namespace prim;

namespace {

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
  if (ptr_call->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(ptr_call->args.size(), 3U)
        << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
    const auto *buffer_load = ptr_call->args[0].as<TensorLoadNode>();
    ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
    return buffer_load->source.as_or_throw<TensorVar>().dtype();
  }
  return std::nullopt;
}

// Convert an element-wise vector expression into the scalar expression for a
// single lane.  In particular, this keeps BufferLoad nodes inside a scalar
// Select instead of materializing both vector-valued branches before the
// ternary is emitted.  Unlike CUDA's Select visitor (introduced in #2843),
// which materializes the condition and both branches as SSA temporaries before
// extracting lanes, HIP intentionally preserves lazy branch evaluation here.
class VectorLaneScalarizer : public tvm::ExprFunctor<PrimExpr(const Expr &)> {
public:
  explicit VectorLaneScalarizer(int lane) : lane_(lane) {}

private:
  PrimExpr DispatchDefault_(const ffi::Object *node) final {
    auto expression = GetRef<ffi::ObjectRef>(node).as_or_throw<PrimExpr>();
    return expression.ty().IsScalar()
               ? expression
               : Shuffle::ExtractElement(expression, lane_);
  }

  PrimExpr Dispatch_(const RampNode *node) final {
    return Dispatch(node->base) +
           Dispatch(node->stride) * IntImm(node->stride.ty(), lane_);
  }

  PrimExpr Dispatch_(const BroadcastNode *node) final {
    return Dispatch(node->value);
  }

  PrimExpr Dispatch_(const LetNode *node) final {
    return Dispatch(tl::Substitute(
        node->body, {{node->var.as_or_throw<PrimVar>(), node->value}}));
  }

  PrimExpr Dispatch_(const TensorLoadNode *node) final {
    int element_lanes = node->source.as_or_throw<TensorVar>().dtype().lanes();
    Array<PrimExpr> indices;
    for (const PrimExpr &index : node->indices)
      indices.push_back(VectorLaneScalarizer(lane_ / element_lanes)(index));
    PrimExpr load = TensorLoad(node->source, indices);
    return element_lanes == 1
               ? load
               : Shuffle::ExtractElement(load, lane_ % element_lanes);
  }

  PrimExpr Dispatch_(const SelectNode *node) final {
    return Select(Dispatch(node->condition), Dispatch(node->true_value),
                  Dispatch(node->false_value));
  }

  PrimExpr Dispatch_(const NotNode *node) final {
    return Not(Dispatch(node->a));
  }

#define TL_HIP_SCALAR_BINARY(Kind)                                             \
  PrimExpr Dispatch_(const Kind##Node *node) final {                           \
    return Kind(Dispatch(node->a), Dispatch(node->b));                         \
  }
  TL_HIP_SCALAR_BINARY(Add)
  TL_HIP_SCALAR_BINARY(Sub)
  TL_HIP_SCALAR_BINARY(Mul)
  TL_HIP_SCALAR_BINARY(Div)
  TL_HIP_SCALAR_BINARY(Mod)
  TL_HIP_SCALAR_BINARY(FloorDiv)
  TL_HIP_SCALAR_BINARY(FloorMod)
  TL_HIP_SCALAR_BINARY(Min)
  TL_HIP_SCALAR_BINARY(Max)
  TL_HIP_SCALAR_BINARY(EQ)
  TL_HIP_SCALAR_BINARY(NE)
  TL_HIP_SCALAR_BINARY(LT)
  TL_HIP_SCALAR_BINARY(LE)
  TL_HIP_SCALAR_BINARY(GT)
  TL_HIP_SCALAR_BINARY(GE)
  TL_HIP_SCALAR_BINARY(And)
  TL_HIP_SCALAR_BINARY(Or)
  TL_HIP_SCALAR_BINARY(LShift)
  TL_HIP_SCALAR_BINARY(RShift)
  TL_HIP_SCALAR_BINARY(BitwiseAnd)
  TL_HIP_SCALAR_BINARY(BitwiseOr)
  TL_HIP_SCALAR_BINARY(BitwiseXor)
#undef TL_HIP_SCALAR_BINARY

  PrimExpr Dispatch_(const ShuffleNode *node) final {
    int output_lane = PythonDType(GetRef<Expr>(node)).IsScalar() ? 0 : lane_;
    ICHECK_LT(output_lane, node->indices.size());
    auto index = tl::GetConstInt(node->indices[output_lane]);
    ICHECK(index) << "Vector lane extraction needs constant shuffle indices";
    int64_t source_lane = *index;
    for (const PrimExpr &vector : node->vectors) {
      ICHECK(!vector.ty().IsScalableVector());
      if (source_lane < vector.ty().lanes())
        return VectorLaneScalarizer(static_cast<int>(source_lane))(vector);
      source_lane -= vector.ty().lanes();
    }
    TVM_FFI_THROW(ValueError) << "Shuffle lane is outside the source vectors";
  }

  PrimExpr Dispatch_(const CastNode *node) final {
    PrimExpr value = Dispatch(node->value);
    PrimType dtype = node->ty.as_or_throw<PrimType>().WithLanes(1);
    return value.ty() == dtype ? value : Cast(dtype, value);
  }

  PrimExpr Dispatch_(const CallNode *node) final {
    PrimType result_type = node->ty.as_or_throw<PrimType>();
    if (!result_type.IsFixedLengthVector())
      return DispatchDefault_(node);
    if (node->op.same_as(tirx::reinterpret_op())) {
      PrimType source_type = PythonDType(node->args[0]);
      ICHECK_EQ(source_type.lanes() * source_type.bits(),
                result_type.lanes() * result_type.bits());
      auto reinterpret = [](PrimExpr value, PrimType type) {
        return value.ty() == type ? value
                                  : Call(type, tirx::reinterpret_op(), {value})
                                        .as_or_throw<PrimExpr>();
      };
      if (source_type.bits() == result_type.bits())
        return reinterpret(Dispatch(node->args[0]), result_type.WithLanes(1));
      PrimType source_bits = PrimType::UInt(source_type.bits());
      PrimType result_bits = PrimType::UInt(result_type.bits());
      if (source_type.bits() < result_type.bits()) {
        ICHECK_EQ(result_type.bits() % source_type.bits(), 0);
        int factor = result_type.bits() / source_type.bits();
        PrimExpr packed = tl::MakeConst(result_bits, 0);
        for (int component = 0; component < factor; ++component) {
          PrimExpr value =
              VectorLaneScalarizer(lane_ * factor + component)(node->args[0]);
          PrimExpr bits = Cast(result_bits, reinterpret(value, source_bits));
          packed = packed | (bits << IntImm(result_bits,
                                            component * source_type.bits()));
        }
        return reinterpret(packed, result_type.WithLanes(1));
      }
      ICHECK_EQ(source_type.bits() % result_type.bits(), 0);
      int factor = source_type.bits() / result_type.bits();
      PrimExpr value = VectorLaneScalarizer(lane_ / factor)(node->args[0]);
      PrimExpr bits = reinterpret(value, source_bits);
      int offset = (lane_ % factor) * result_type.bits();
      return reinterpret(Cast(result_bits, bits >> IntImm(source_bits, offset)),
                         result_type.WithLanes(1));
    }
    Array<Expr> arguments;
    for (const Expr &argument : node->args)
      arguments.push_back(argument.as<PrimExpr>() ? Expr(Dispatch(argument))
                                                  : argument);
    auto scalar = [&](size_t index) {
      return arguments[index].as_or_throw<PrimExpr>();
    };
    if (node->op.same_as(tl::add2()))
      return scalar(0) + scalar(1);
    if (node->op.same_as(tl::sub2()))
      return scalar(0) - scalar(1);
    if (node->op.same_as(tl::mul2()))
      return scalar(0) * scalar(1);
    if (node->op.same_as(tl::fma2()))
      return scalar(0) * scalar(1) + scalar(2);
    if (node->op.same_as(tl::max2()))
      return Select(scalar(0) > scalar(1), scalar(0), scalar(1));
    if (node->op.same_as(tl::min2()))
      return Select(scalar(0) < scalar(1), scalar(0), scalar(1));
    if (node->op.same_as(tl::abs2())) {
      PrimExpr value = scalar(0);
      if (value.ty().MatchesElementType(kDLBfloat, 16))
        return Call(value.ty(), tirx::call_pure_extern_op(),
                    {StringImm("__habs"), value})
            .as_or_throw<PrimExpr>();
      return Select(value >= tl::MakeConst(value.ty(), 0), value, -value);
    }
    return Call(result_type.WithLanes(1), node->op, arguments, node->attrs,
                node->ty_args, node->loc)
        .as_or_throw<PrimExpr>();
  }

  int lane_;
};

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

} // namespace

static std::string GetFP8Type(PrimType type) {
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
  } else if (lanes == 16) {
    vec = "_16";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2, 4, 8, 16) "
           "for FP8";
  }
  if (type.MatchesCode(kDLFloat8_e4m3fn) ||
      type.MatchesCode(kDLFloat8_e4m3fnuz) ||
      type.MatchesCode(kDLFloat8_e4m3) ||
      type.code() == kDLFloat8_e4m3b11fnuz) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.MatchesCode(kDLFloat8_e5m2) ||
             type.MatchesCode(kDLFloat8_e5m2fnuz) ||
             type.code() == kDLFloat8_e5m2) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.code() == kDLFloat8_e8m0fnu) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Unsupported FP8 type in HIP codegen: " << type;
  }
  return stream.str();
}

// Returns the HIP C type string for a float4_e2m1fn PrimType.
// Only used on gfx950; caller sets enable_fp4_ before invoking.
static std::string GetFP4Type(PrimType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  if (type.IsScalar()) {
    stream << "fp4_e2_t";
  } else if (lanes == 2) {
    stream << "fp4_e2_2_t";
  } else if (lanes == 4) {
    stream << "fp4_e2_4_t";
  } else if (lanes == 8) {
    stream << "fp4_e2_8_t";
  } else if (lanes == 16) {
    stream << "fp4_e2_16_t";
  } else if (lanes == 32) {
    stream << "fp4_e2_32_t";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Only support scalar and vector types of width (2,4,8,16,32) "
           "for FP4 on HIP";
  }
  return stream.str();
}

/*!
 * \brief Replace patterns with replacement strings.
 * \note should use std::format instead when codebase is ported to C++20.
 */
class Replacer {
public:
  void register_rule(const std::string &pattern,
                     const std::string &replacement) {
    _rules.emplace_back(pattern, replacement);
  }
  std::string rewrite(std::string str) {
    for (auto &&rule : _rules) {
      auto [pattern, replacement] = rule;
      size_t len = pattern.size();
      size_t new_len = replacement.size();
      size_t pos = str.find(pattern);
      while (pos != std::string::npos) {
        str = str.replace(pos, len, replacement);
        pos = str.find(pattern, pos + new_len);
      }
    }
    return str;
  }
  void empty_rules() { _rules.clear(); }

private:
  std::vector<std::pair<std::string, std::string>> _rules;
};

CodeGenTileLangHIP::CodeGenTileLangHIP() { restrict_keyword_ = "__restrict__"; }

void CodeGenTileLangHIP::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" __global__ ";
}

void CodeGenTileLangHIP::PrintExtraAttrs(const tirx::Function &function,
                                         std::ostream &output) {
  PrimExpr threads(1);
  tl::PostOrderVisit(*function->body, [&](const ffi::ObjectRef &object) {
    const auto *region = object.as<RegionStmtNode>();
    if (region && region->op.same_as(tirx::launch_thread_op()) &&
        std::string(region->args[0].as_or_throw<StringImm>()->value)
            .starts_with("threadIdx."))
      threads = threads * region->args[1].as_or_throw<PrimExpr>();
  });
  auto count = tl::GetConstInt(tl::arith::Simplify(sym::Analyzer(), threads));
  if (count && *count >= 64)
    output << " __launch_bounds__(" << *count << ")";
}

void CodeGenTileLangHIP::EmitPreamble_(std::ostream &os) const {
  // hip must need a header file.
  os << "#include <hip/hip_runtime.h>\n";
  if (need_mma_h_) {
    os << "#include <mma.h>\n";
  }

  if (enable_fp8_) {
    os << "#include <tl_templates/hip/hip_fp8.h>\n";
  }

  if (enable_fp4_) {
    os << "#include <tl_templates/hip/hip_fp4.h>\n";
  }

  if (need_cooperative_groups_) {
    os << "#include <hip/hip_cooperative_groups.h>\n";
  }

  os << "#include <tl_templates/hip/gemm.h>\n";
  os << "#include <tl_templates/hip/copy.h>\n";
  os << "#include <tl_templates/hip/reduce.h>\n";
  os << "#include <tl_templates/hip/scan.h>\n";
  os << "#include <tl_templates/hip/ldsm.h>\n";
  os << "#include <tl_templates/hip/threadblock_swizzle.h>\n";
  os << "#include <tl_templates/hip/debug.h>\n";
  os << "\n";
}

void CodeGenTileLangHIP::Dispatch_(const ForNode *node) {
  if (node->kind == ForKind::kUnrolled) {
    PrintIndent();
    stream << "#pragma unroll\n";
  }
  CodeGenC::Dispatch_(node);
}

void CodeGenTileLangHIP::BindThreadIndex(const PrimVar &variable,
                                         const ffi::String &thread_tag) {
  ICHECK(!var_idmap_.count(variable.get()));
  var_idmap_[variable.get()] =
      CastFromTo(thread_tag, PrimType::UInt(32), variable.ty());
}

void CodeGenTileLangHIP::PrintType(const PrimType &t,
                                   std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.MatchesCode(kDLOpaqueHandle)) {
    ICHECK(t.IsScalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.IsVoid()) {
    os << "void";
    return;
  }

  bool fail = false;
  if (t.MatchesCode(kDLFloat)) {
    switch (t.bits()) {
    case 16:
      if (t.IsScalar()) {
        os << "half_t";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp16 vector elements.
        //
        // half4 is stored as uint2
        //
        // h4.x is emitted as *(half2*)(&(u2.x)).x
        // h4.y is emitted as *(half2*)(&(u2.x)).y
        // h4.z is emitted as *(half2*)(&(u2.y)).x
        // h4.w is emitted as *(half2*)(&(u2.y)).y
        //
        ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
        os << "uint" << lanes / 2;
      } else {
        fail = true;
      }
      break;
    case 32:
      if (lanes <= 4) {
        os << "float";
      } else if (lanes <= 8) {
        // Emit CUDA code to access fp32 vector elements for 4 < lanes <= 8.
        //
        // float8 is stored as ulonglong4
        //
        // f8.v1 is emitted as *(float2*)(&(ul4.x)).x
        // f8.v2 is emitted as *(float2*)(&(ul4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for float type with lanes > 4";
        os << "ulonglong" << lanes / 2;
      } else if (lanes == 16) {
        // float32x16: GCC vector extension type used by MFMA accumulators.
        os << "float32x16";
        return;
      } else if (lanes == 32) {
        // float32x32: GCC vector extension type used by MFMA accumulators.
        os << "float32x32";
        return;
      } else {
        fail = true;
      }
      break;
    case 64:
      os << "double";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.IsScalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    if (t.IsScalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else if (lanes == 16) {
      // bfloat16x16: struct { bfloat16_t data[16]; } used by MFMA accumulators.
      os << "bfloat16x16";
      return;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.MatchesCode(kDLFloat8_e4m3fn, kDLFloat8_e5m2, kDLFloat8_e4m3fnuz,
                           kDLFloat8_e5m2fnuz, kDLFloat8_e4m3,
                           kDLFloat8_e8m0fnu)) {
    enable_fp8_ = true;
    os << GetFP8Type(t);
    return;
  } else if (t.MatchesCode(kDLFloat4_e2m1fn)) {
    // FP4 E2M1 is only supported on gfx950 (CDNA4). Setting enable_fp4_ will
    // cause Finish() to include hip_fp4.h which is itself guarded by
    // #if defined(__gfx950__), so this is safe to emit on all HIP targets
    // (the compiler will reject FP4 kernel code on non-gfx950 anyway).
    enable_fp4_ = true;
    if (t.lanes() <= 32) {
      os << GetFP4Type(t);
    } else {
      TVM_FFI_THROW(InternalError)
          << "Cannot convert FP4 type with " << t.lanes()
          << " lanes to HIP type (max 32 lanes supported)";
    }
    return;
  } else if (t == PrimType::Bool()) {
    os << "bool";
    return;
  } else if ((t.MatchesCode(kDLBool) && !t.IsScalar())) {
    // HIP does not support bool vectors.
    // Use ushort vectors for directly addressable lanes and packed unsigned
    // integers for wider vectors.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    } else if (n == 8) {
      os << "uint8_t";
      return;
    } else if (n == 16) {
      os << "uint16_t";
      return;
    } else if (n == 32) {
      os << "uint";
      return;
    }
  } else if (t.MatchesCode(kDLUInt) || t.MatchesCode(kDLInt)) {
    if (t.MatchesCode(kDLUInt)) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.IsScalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int8_t";
        return;
      } else if (t.lanes() == 16) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 32) {
        os << "int";
        return;
      } else {
        TVM_FFI_THROW(InternalError)
            << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.IsScalar()) {
        os << "int";
        return;
      } else if (t.lanes() == 4) {
        os << "int16_t";
        return;
      } else if (t.lanes() == 8) {
        // directly 8 4-bit int in integer.
        os << "int";
        return;
      } else if (t.lanes() == 16) {
        os << "int2";
        return;
      } else if (t.lanes() == 32) {
        os << "int4";
        return;
      } else if (t.lanes() == 64) {
        os << "int8";
        return;
      } else {
        TVM_FFI_THROW(InternalError)
            << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        os << "int4";
        return;
      } else if (!t.MatchesCode(kDLUInt) && t.IsScalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.IsScalar()) {
        os << "short";
      } else if (t.lanes() <= 4) {
        os << "short" << lanes;
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int16 vector elements.
        //
        // short4 is stored as int2
        //
        // s4.x is emitted as *(short2*)(&(i2.x)).x
        // s4.y is emitted as *(short2*)(&(i2.x)).y
        // s4.z is emitted as *(short2*)(&(i2.y)).x
        // s4.w is emitted as *(short2*)(&(i2.y)).y
        //
        ICHECK_EQ(t.lanes() % 2, 0)
            << "only support even lane for shorT type with lanes > 4";
        os << "int" << t.lanes() / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.IsScalar()) {
        os << "int";
      } else if (t.lanes() <= 4) {
        os << "int" << t.lanes();
      } else if (t.lanes() <= 8) {
        // Emit CUDA code to access int32 vector elements for 4 < lanes <= 8.
        //
        // int8 is stored as longlong4
        //
        // i8.v1 is emitted as *(int2*)(&(l4.x)).x
        // i8.v2 is emitted as *(int2*)(&(l4.x)).y
        //
        ICHECK_EQ(lanes % 2, 0)
            << "only support even lane for int32 type with lanes > 4";
        os << "longlong" << lanes / 2;
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.IsScalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1) {
      return;
    }
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError)
      << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangHIP::PrintVecBinaryOp(const std::string &op,
                                          const PrimType &t, PrimExpr lhs,
                                          PrimExpr rhs,
                                          std::ostream &os) { // NOLINT(*)

  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), PythonDType(lhs));
    std::string vrhs = SSAGetID(PrintExpr(rhs), PythonDType(rhs));

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
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
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangHIP::Dispatch_(const SelectNode *op, std::ostream &os) {
  if (!PythonDType(op->condition).IsFixedLengthVector()) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  TVM_FFI_ICHECK(PythonDType(op->false_value) ==
                     PythonDType(GetRef<Expr>(op)) &&
                 PythonDType(op->true_value) == PythonDType(GetRef<Expr>(op)) &&
                 PythonDType(GetRef<Expr>(op)).lanes() ==
                     PythonDType(op->condition).lanes());

  std::string result = name_supply_->FreshName("_");
  auto scalarize_lane = [&](int lane) {
    PrimExpr lane_select = VectorLaneScalarizer(lane)(GetRef<PrimExpr>(op));
    ICHECK(PythonDType(lane_select).IsScalar());
    return lane_select;
  };
  this->PrintIndent();
  this->PrintType(PythonDType(GetRef<Expr>(op)), stream);
  stream << ' ' << result << "{};\n";
  for (int i = 0; i < PythonDType(GetRef<Expr>(op)).lanes(); ++i) {
    PrintVecElemStore(result, PythonDType(GetRef<Expr>(op)), i,
                      PrintExpr(scalarize_lane(i)));
  }
  os << result;
}

void CodeGenTileLangHIP::Dispatch_(const NotNode *op, std::ostream &os) {
  if (!PythonDType(GetRef<Expr>(op)).IsFixedLengthVector()) {
    CodeGenC::Dispatch_(op, os);
    return;
  }

  std::string result = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(PythonDType(GetRef<Expr>(op)), stream);
  stream << ' ' << result << ";\n";
  int ssa_scope = BeginScope();
  {
    std::string value = SSAGetID(PrintExpr(op->a), PythonDType(op->a));
    for (int i = 0; i < PythonDType(GetRef<Expr>(op)).lanes(); ++i) {
      std::ostringstream lane;
      lane << "!bool(";
      PrintVecElemLoad(value, PythonDType(op->a), i, lane);
      lane << ')';
      PrintVecElemStore(result, PythonDType(GetRef<Expr>(op)), i, lane.str());
    }
  }
  EndScope(ssa_scope);
  os << result;
}

void CodeGenTileLangHIP::PrintVecElemLoad(const std::string &vec,
                                          const PrimType &t, int i,
                                          std::ostream &os) { // NOLINT(*)
  if (t.IsScalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  bool is_packed_bool =
      (t.MatchesCode(kDLBool) && !t.IsScalar()) && t.lanes() >= 8;
  bool is_packed_int4 =
      t.bits() == 4 && (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt));
  ICHECK(i >= 0 && i < (is_packed_bool                       ? t.lanes()
                        : is_packed_int4                     ? t.lanes()
                        : t.MatchesCode(kDLFloat4_e2m1fn)    ? 32
                        : t.bits() == 8                      ? 16
                        : (t.lanes() == 16)                  ? 16
                        : (t.lanes() == 32)                  ? 32
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
  if (is_packed_bool) {
    std::ostringstream packed_byte;
    packed_byte << "((const unsigned char*)(&(" << vec << ")))[" << i / 8
                << "]";
    os << "((static_cast<unsigned int>(" << packed_byte.str() << ") >> "
       << i % 8 << ") & 0x01u)";
  } else if (t.bits() == 8 &&
             (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt))) {
    std::string type_name = t.MatchesCode(kDLInt) ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    }
  } else if ((t.lanes() == 16 || t.lanes() == 32) && t.bits() == 32 &&
             t.MatchesCode(kDLFloat)) {
    // float32x16/float32x32: __attribute__((__vector_size__(...))) supports
    // subscript.
    os << vec << "[" << i << "]";
  } else if (t.lanes() == 16 && t.MatchesElementType(kDLBfloat, 16)) {
    // bfloat16x16: struct { bfloat16_t data[16]; }
    os << vec << ".data[" << i << "]";
  } else if (t.MatchesElementType(kDLFloat, 16)) {
    os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    os << "((bfloat16x2*)(&(" << vec << "." << access[i / 2] << ")))->"
       << access[i % 2];
  } else if (is_packed_int4) {
    std::ostringstream packed_byte;
    packed_byte << "((const unsigned char*)(&(" << vec << ")))[" << i / 2
                << "]";
    int shift = i % 2 * 4;
    if (t.MatchesCode(kDLUInt)) {
      os << "((static_cast<unsigned int>(" << packed_byte.str() << ") >> "
         << shift << ") & 0x0fu)";
    } else {
      os << "((static_cast<int>((static_cast<unsigned int>("
         << packed_byte.str() << ") >> " << shift << ") & 0x0fu) ^ 8) - 8)";
    }
  } else if (t.MatchesCode(kDLFloat8_e4m3fn, kDLFloat8_e5m2, kDLFloat8_e4m3fnuz,
                           kDLFloat8_e5m2fnuz, kDLFloat8_e4m3,
                           kDLFloat8_e8m0fnu) &&
             !t.MatchesCode(kDLFloat8_e8m0fnu)) {
    if (t.lanes() == 2) {
      os << "tl_fp8x2_get_lane<" << GetFP8Type(t.WithLanes(1)) << ">(" << vec
         << ", " << i << ")";
      return;
    }
    std::string element = vec;
    int lanes = t.lanes();
    int lane = i;
    while (lanes > 4) {
      int group_size = lanes / 2;
      element += lane < group_size ? ".x" : ".y";
      lane %= group_size;
      lanes = group_size;
    }
    element += "." + std::string(1, access[lane]);
    os << element;
  } else if (t.MatchesCode(kDLFloat4_e2m1fn)) {
    std::string element = vec;
    int lanes = t.lanes();
    int lane = i;
    while (lanes > 2) {
      int group_size = lanes / 2;
      element += lane < group_size ? ".x" : ".y";
      lane %= group_size;
      lanes = group_size;
    }
    os << element << (lane == 0 ? ".x()" : ".y()");
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "short";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "int";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "uint";
      } else if (t.MatchesCode(kDLFloat)) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    os << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
       << ")))->" << access[i % 2];
  } else {
    os << vec << "." << access[i];
  }
}

void CodeGenTileLangHIP::PrintVecElemStore(const std::string &vec,
                                           const PrimType &t, int i,
                                           const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};

  bool is_packed_bool =
      (t.MatchesCode(kDLBool) && !t.IsScalar()) && t.lanes() >= 8;
  bool is_packed_int4 =
      t.bits() == 4 && (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt));
  ICHECK(i >= 0 && i < (is_packed_bool                       ? t.lanes()
                        : is_packed_int4                     ? t.lanes()
                        : t.MatchesCode(kDLFloat4_e2m1fn)    ? 32
                        : t.bits() == 8                      ? 16
                        : (t.lanes() == 16)                  ? 16
                        : (t.lanes() == 32)                  ? 32
                        : (t.bits() == 16 || t.bits() == 32) ? 8
                                                             : 4));
  if (is_packed_bool) {
    std::ostringstream packed_byte;
    packed_byte << "((unsigned char*)(&(" << vec << ")))[" << i / 8 << "]";
    stream << packed_byte.str() << " = ";
    if (i % 8 != 0) {
      stream << "(" << packed_byte.str() << " & ~(0x01u << " << i % 8
             << ")) | ";
    }
    stream << "((static_cast<unsigned int>(" << value << ") & 0x01u) << "
           << i % 8 << ");\n";
  } else if (t.bits() == 8 &&
             (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt))) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "=" << "(" << value
             << ");\n";
    } else {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first lane of each carrier word.
      if (i % 4 != 0) {
        stream << "static_cast<unsigned int>(" << ac << ") & ~(0x000000ffu << "
               << i % 4 * 8 << ") |";
      }
      stream << "(static_cast<unsigned int>(static_cast<unsigned char>("
             << value << ")) << " << i % 4 * 8 << ");\n";
    }
  } else if ((t.lanes() == 16 || t.lanes() == 32) && t.bits() == 32 &&
             t.MatchesCode(kDLFloat)) {
    // float32x16/float32x32: __attribute__((__vector_size__(...))) supports
    // subscript.
    stream << vec << "[" << i << "] = " << value << ";\n";
  } else if (t.lanes() == 16 && t.MatchesElementType(kDLBfloat, 16)) {
    // bfloat16x16: struct { bfloat16_t data[16]; }
    stream << vec << ".data[" << i << "] = " << value << ";\n";
  } else if (t.MatchesElementType(kDLFloat, 16)) {
    stream << "*((half_t*)(&(((half2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << "))) = " << value << ";\n";
  } else if (t.MatchesElementType(kDLBfloat, 16)) {
    stream << "((bfloat16_t*)(&(" << vec << "." << access[i / 2] << ")))["
           << (i % 2) << "] = " << value << ";\n";
  } else if (is_packed_int4) {
    std::ostringstream packed_byte;
    packed_byte << "((unsigned char*)(&(" << vec << ")))[" << i / 2 << "]";
    stream << packed_byte.str() << " = ";
    if (i % 2 != 0) {
      stream << "(" << packed_byte.str() << " & 0x0fu) | ";
    }
    stream << "((static_cast<unsigned int>(" << value << ") & 0x0fu) << "
           << i % 2 * 4 << ");\n";
  } else if (t.MatchesCode(kDLFloat8_e4m3fn, kDLFloat8_e5m2, kDLFloat8_e4m3fnuz,
                           kDLFloat8_e5m2fnuz, kDLFloat8_e4m3,
                           kDLFloat8_e8m0fnu) &&
             !t.MatchesCode(kDLFloat8_e8m0fnu)) {
    if (t.lanes() == 2) {
      stream << "tl_fp8x2_set_lane(" << vec << ", " << i << ", " << value
             << ");\n";
      return;
    }
    std::string element = vec;
    int lanes = t.lanes();
    int lane = i;
    while (lanes > 4) {
      int group_size = lanes / 2;
      element += lane < group_size ? ".x" : ".y";
      lane %= group_size;
      lanes = group_size;
    }
    stream << element << "." << access[lane] << " = " << value << ";\n";
  } else if (t.MatchesCode(kDLFloat4_e2m1fn)) {
    std::string element = vec;
    int lanes = t.lanes();
    int lane = i;
    while (lanes > 2) {
      int group_size = lanes / 2;
      element += lane < group_size ? ".x" : ".y";
      lane %= group_size;
      lanes = group_size;
    }
    stream << element << (lane == 0 ? ".set_x(" : ".set_y(") << value << ");\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "short";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.MatchesCode(kDLInt)) {
        type_name = "int";
      } else if (t.MatchesCode(kDLUInt)) {
        type_name = "uint";
      } else if (t.MatchesCode(kDLFloat)) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenTileLangHIP::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    // TODO: change to __builtin_amdgcn_s_barrier() later
    // __syncthreads() will add a vmcnt(0) to final asm, which will break all
    // buffer_load_async and buffer_load_async...lds
    // in tilelang vmcnt should be managed explicitly by cp_async_wait.
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenTileLangHIP::PrintStorageScope(
    const runtime::MemoryScope &storage_scope,
    std::ostream &os) { // NOLINT(*)
  std::string scope = storage_scope.name();
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
         "all global arrays as input instead";
  if (scope == "shared") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ __align__(1024) ";
  }
}

std::string CodeGenTileLangHIP::CastFromTo(std::string value,
                                           const PrimType &from,
                                           const PrimType &target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.MatchesElementType(kDLFloat, 16) &&
      (target.MatchesCode(kDLInt) || target.MatchesCode(kDLUInt)) &&
      target.bits() == 8) {
    os << "(";
    if (target.MatchesCode(kDLUInt)) {
      os << "u";
    }
    os << "int)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangHIP::Dispatch_(const ShuffleNode *op,
                                   std::ostream &os) { // NOLINT(*)
  // For bfloat16x2 / float16x2 construction from two scalar lanes, emit the
  // HIP pack intrinsic instead of the invalid `uint1(a, b)` that CodeGenC
  // would generate (HIP's uint1 has no two-argument constructor).
  // `uint1{value}` aggregate initialisation is valid: uint1 is defined by ROCm
  // as HIP_vector_type<unsigned int, 1> which has a single .x member.
  PrimType t = PythonDType(GetRef<Expr>(op));
  bool is_bf16x2 = t.MatchesElementType(kDLBfloat, 16) && t.lanes() == 2;
  bool is_fp16x2 = t.MatchesElementType(kDLFloat, 16) && t.lanes() == 2;
  if ((is_bf16x2 || is_fp16x2) && op->vectors.size() == 2 &&
      PythonDType(op->vectors[0]).lanes() == 1 &&
      PythonDType(op->vectors[1]).lanes() == 1) {
    std::string e0 = PrintExpr(op->vectors[0]);
    std::string e1 = PrintExpr(op->vectors[1]);
    if (is_bf16x2) {
      os << "uint1{__pack_bfloat162(" << e0 << ", " << e1 << ")}";
    } else {
      os << "uint1{__pack_half2(" << e0 << ", " << e1 << ")}";
    }
    return;
  }
  // Default path for all other shuffle patterns.
  CodeGenC::Dispatch_(op, os);
}

void CodeGenTileLangHIP::Dispatch_(const CastNode *op, std::ostream &os) {
  PrimType from_ty = PythonDType(op->value);
  PrimType target_ty = PythonDType(GetRef<Expr>(op));
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.IsScalar())
    return CodeGenC::Dispatch_(op, os);

  // ---------------------------------------------------------------------------
  // Vectorized FP4 <-> float/half/bfloat16 conversions (gfx950 only).
  // These use the pair-wise helpers from hip_fp4.h.  We process two lanes at a
  // time: odd-indexed lane is the "high" nibble, even-indexed is "low".
  // Only enabled when the FP4 header is included (enable_fp4_ is set by
  // PrintType when a float4 type is encountered).
  // ---------------------------------------------------------------------------
  int fp4_lanes = from_ty.lanes();
  // Pairwise cast: process 2 FP4 lanes at a time via packed uint8_t byte.
  // Supported lane widths: 2, 4, 8, 16, 32 (all even widths up to FP4x32).
  bool fp4_pair_cast = (fp4_lanes == 2 || fp4_lanes == 4 || fp4_lanes == 8 ||
                        fp4_lanes == 16 || fp4_lanes == 32);

  // FP4 -> float16 : use __tl_cvt_fp4x2_to_half2 per 2-element pair
  if (from_ty.MatchesCode(kDLFloat4_e2m1fn) &&
      target_ty.MatchesElementType(kDLFloat, 16) && fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    // Iterate over pairs: src is stored as fp4_e2_{lanes}_t; we access the
    // packed byte for each pair via reinterpret as uint8_t array.
    // Materialize the uint1 result into a local variable to avoid taking the
    // address of a prvalue (C++ UB).
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::string tmp = name_supply_->FreshName("_fp4h2_");
      this->PrintIndent();
      stream << "uint1 " << tmp << " = __tl_cvt_fp4x2_to_half2(((uint8_t*)&("
             << src << "))[" << i / 2 << "]);\n";
      PrintVecElemStore(sret, target_ty, i, "((half_t*)(&(" + tmp + ")))[0]");
      PrintVecElemStore(sret, target_ty, i + 1,
                        "((half_t*)(&(" + tmp + ")))[1]");
    }
    os << sret;
    return;
  }

  // FP4 -> float32 : use __tl_cvt_fp4x2_to_float2 per 2-element pair
  if (from_ty.MatchesCode(kDLFloat4_e2m1fn) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 32 &&
      fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream val;
      val << "__tl_cvt_fp4x2_to_float2(((uint8_t*)&(" << src << "))[" << i / 2
          << "])";
      std::string tmp = name_supply_->FreshName("_fp4f2_");
      this->PrintIndent();
      stream << "float2 " << tmp << " = " << val.str() << ";\n";
      PrintVecElemStore(sret, target_ty, i, tmp + ".x");
      PrintVecElemStore(sret, target_ty, i + 1, tmp + ".y");
    }
    os << sret;
    return;
  }

  // FP4 -> double : use __tl_cvt_fp4x2_to_double2 per 2-element pair
  if (from_ty.MatchesCode(kDLFloat4_e2m1fn) &&
      target_ty.MatchesCode(kDLFloat) && target_ty.bits() == 64 &&
      fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream val;
      val << "__tl_cvt_fp4x2_to_double2(((uint8_t*)&(" << src << "))[" << i / 2
          << "])";
      std::string tmp = name_supply_->FreshName("_fp4d2_");
      this->PrintIndent();
      stream << "double2 " << tmp << " = " << val.str() << ";\n";
      PrintVecElemStore(sret, target_ty, i, tmp + ".x");
      PrintVecElemStore(sret, target_ty, i + 1, tmp + ".y");
    }
    os << sret;
    return;
  }

  // FP4 -> bfloat16 : use __tl_cvt_fp4x2_to_bfloat162 per 2-element pair
  if (from_ty.MatchesCode(kDLFloat4_e2m1fn) &&
      target_ty.MatchesElementType(kDLBfloat, 16) && fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream val;
      val << "__tl_cvt_fp4x2_to_bfloat162(((uint8_t*)&(" << src << "))["
          << i / 2 << "])";
      std::string tmp = name_supply_->FreshName("_fp4bf2_");
      this->PrintIndent();
      stream << "uint1 " << tmp << " = " << val.str() << ";\n";
      PrintVecElemStore(sret, target_ty, i,
                        "((bfloat16_t*)(&(" + tmp + ")))[0]");
      PrintVecElemStore(sret, target_ty, i + 1,
                        "((bfloat16_t*)(&(" + tmp + ")))[1]");
    }
    os << sret;
    return;
  }

  // float16 -> FP4
  if (from_ty.MatchesElementType(kDLFloat, 16) &&
      target_ty.MatchesCode(kDLFloat4_e2m1fn) && fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream h0, h1;
      PrintVecElemLoad(src, from_ty, i, h0);
      PrintVecElemLoad(src, from_ty, i + 1, h1);
      std::string tmp = name_supply_->FreshName("_h2fp4_");
      this->PrintIndent();
      stream << "uint1 " << tmp << " = uint1{__pack_half2(" << h0.str() << ", "
             << h1.str() << ")};\n";
      this->PrintIndent();
      stream << "((uint8_t*)&(" << sret << "))[" << i / 2
             << "] = __tl_cvt_half2_to_fp4x2(" << tmp << ");\n";
    }
    os << sret;
    return;
  }

  // float32 -> FP4
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 32 &&
      target_ty.MatchesCode(kDLFloat4_e2m1fn) && fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream f0, f1;
      PrintVecElemLoad(src, from_ty, i, f0);
      PrintVecElemLoad(src, from_ty, i + 1, f1);
      this->PrintIndent();
      stream << "((uint8_t*)&(" << sret << "))[" << i / 2
             << "] = __tl_cvt_float2_to_fp4x2("
             << "float2{" << f0.str() << ", " << f1.str() << "});\n";
    }
    os << sret;
    return;
  }

  // double -> FP4
  if (from_ty.MatchesCode(kDLFloat) && from_ty.bits() == 64 &&
      target_ty.MatchesCode(kDLFloat4_e2m1fn) && fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream d0, d1;
      PrintVecElemLoad(src, from_ty, i, d0);
      PrintVecElemLoad(src, from_ty, i + 1, d1);
      this->PrintIndent();
      stream << "((uint8_t*)&(" << sret << "))[" << i / 2
             << "] = __tl_cvt_double2_to_fp4x2("
             << "double2{" << d0.str() << ", " << d1.str() << "});\n";
    }
    os << sret;
    return;
  }

  // bfloat16 -> FP4
  if (from_ty.MatchesElementType(kDLBfloat, 16) &&
      target_ty.MatchesCode(kDLFloat4_e2m1fn) && fp4_pair_cast) {
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(target_ty, stream);
    stream << ' ' << sret << ";\n";
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0; i < fp4_lanes; i += 2) {
      std::ostringstream b0, b1;
      PrintVecElemLoad(src, from_ty, i, b0);
      PrintVecElemLoad(src, from_ty, i + 1, b1);
      std::string tmp = name_supply_->FreshName("_bf2fp4_");
      this->PrintIndent();
      stream << "uint1 " << tmp << " = uint1{__pack_bfloat162(" << b0.str()
             << ", " << b1.str() << ")};\n";
      this->PrintIndent();
      stream << "((uint8_t*)&(" << sret << "))[" << i / 2
             << "] = __tl_cvt_bfloat162_to_fp4x2(" << tmp << ");\n";
    }
    os << sret;
    return;
  }

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
  {
    std::string src = SSAGetID(PrintExpr(op->value), from_ty);
    for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
      std::ostringstream val;
      val << "(";
      PrintType(target_ty.WithLanes(1), val);
      val << ")(";
      PrintVecElemLoad(src, from_ty, i, val);
      val << ")";
      PrintVecElemStore(sret, target_ty, i, val.str());
    }
  }
  os << sret;
}

void CodeGenTileLangHIP::PrintCallExtern(Type ret_type, String global_symbol,
                                         const Array<Expr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
  PrimType ret_dtype =
      (IsVoidType(ret_type) ? PrimType::Void()
                            : ret_type.as_or_throw<PrimType>());
  if (ret_dtype.IsFixedLengthVector()) {
    //
    // Emit an unsupported vector call
    //
    // v = intrin_f((float4*)A[0], (float4*)B[0])
    //
    // as
    //
    // float4 __ret;
    // {
    //   float4 __arg0 = ((float4*)A)[0];
    //   float4 __arg1 = ((float4*)B)[0];
    //   __ret.x = intrin_f(__arg0.x, __arg1.x);
    //   __ret.y = intrin_f(__arg0.y, __arg1.y);
    //   __ret.z = intrin_f(__arg0.z, __arg1.z);
    //   __ret.w = intrin_f(__arg0.w, __arg1.w);
    // }
    // v = __ret;
    //
    // Declare the result vector.
    std::string sret = name_supply_->FreshName("_");
    this->PrintIndent();
    this->PrintType(ret_dtype, stream);
    stream << ' ' << sret << ";\n";
    {
      // Load arguments.
      std::vector<std::string> sargs;
      size_t arg_begin = static_cast<size_t>(skip_first_arg);
      for (size_t i = arg_begin; i < args.size(); ++i) {
        std::string val = SSAGetID(PrintExpr(args[i]), PythonDType(args[i]));
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
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
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangHIP::GetBufferRef(const PrimType &t,
                                             const VarNode *buffer,
                                             PrimExpr index) {
  Var buffer_var = GetRef<Var>(buffer);
  const VarNode *buffer_var_node = buffer_var.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var_node);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var_node)) {
    scope = alloc_storage_scope_.at(buffer_var_node).name();
  }

  // FP4 scalar access on gfx950: redirect to tl_fp4_packed_load helper.
  // Non-scalar FP4 accesses fall through to the normal path (the vector
  // types fp4_e2_4_t etc. are directly addressable as structs).
  if (t.MatchesCode(kDLFloat4_e2m1fn) && t.IsScalar()) {
    std::string idx_str = PrintExpr(index);
    auto packed_it = fp4_packed_buffers_.find(buffer_var);
    if (packed_it != fp4_packed_buffers_.end()) {
      // Packed local buffer: use the pre-allocated fp4_e2_2_t array.
      os << "tl_fp4_packed_load(" << packed_it->second << ", " << idx_str
         << ")";
    } else {
      // Non-packed (e.g. shared) buffer: reinterpret as fp4_e2_2_t*.
      os << "tl_fp4_packed_load((fp4_e2_2_t*)" << vid << ", " << idx_str << ")";
    }
    return os.str();
  }

  // bool is_vol = IsVolatile(buffer_var.get());
  // always false for tl cutlass backend.
  bool is_vol = false;

  auto ptr_cast = [this, is_vol, scope](PrimType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(runtime::MemoryScope::Get(scope), ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  PrimType buffer_element_dtype =
      GetRef<Var>(buffer).as_or_throw<TensorVar>().dtype();

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var_node, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }

  if (scope.empty()) {
    scope = PythonStorageScope(GetRef<Var>(buffer));
  }
  // local.var is a scalar — no indexing needed.
  if (scope == "local.var") {
    os << vid;
    return os.str();
  }

  std::string index_str = PrintExpr(index);
  if (t.bits() == 4 || (t.bits() == 1 && t.MatchesCode(kDLInt))) {
    // This is a special case, because CodegenCUDA::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();

    os << "*(" << "(" << ptr_cast(t) << vid << ")" << " + " << index_str
       << " / " << div_factor << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

void CodeGenTileLangHIP::Dispatch_(const CallNode *op, std::ostream &os) {
  auto print_extern_call_stmt = [&](std::string name, size_t start = 0,
                                    size_t end = 0) {
    this->PrintIndent();
    this->stream << name << "(";
    for (size_t i = start; i < op->args.size() - end; i++) {
      if (i > start)
        this->stream << ", ";
      this->stream << this->PrintExpr(op->args[i]);
    }
    this->stream << ");\n";
  };
  if (PythonIsOp(op->op, "reinterpret") && op->args.size() == 1U &&
      op->ty.as<PrimTypeNode>() && op->args[0]->ty.as<PrimTypeNode>() &&
      !PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat4_e2m1fn) &&
      !PythonDType(op->args[0]).MatchesCode(kDLFloat4_e2m1fn)) {
    ICHECK_EQ(PythonDType(GetRef<Expr>(op)).lanes() *
                  PythonDType(GetRef<Expr>(op)).bits(),
              PythonDType(op->args[0]).lanes() *
                  PythonDType(op->args[0]).bits())
        << "reinterpret expects source and target to have the same number of "
           "bits";
    os << "__builtin_bit_cast(";
    this->PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << ")";
  } else if (PythonIsOp(op->op, "ptx_cp_async") &&
             !op->op.same_as(tl::ptx_cp_async())) {
    // args[0] = dst_access_ptr, args[1] = src_access_ptr, args[2] = bytes,
    // args[3] = predicate (optional)
    ICHECK(op->args.size() == 3 || op->args.size() == 4)
        << "ptx_cp_async expects 3 or 4 arguments (dst_access_ptr, "
           "src_access_ptr, bytes, [predicate])";
    std::string dst = this->PrintExpr(op->args[0]);
    std::string src = this->PrintExpr(op->args[1]);
    std::string size = this->PrintExpr(op->args[2]);
    this->PrintIndent();
    if (op->args.size() == 3) {
      // Non-predicated version
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << ", " << src
                   << ");\n";
    } else {
      // Predicated version
      std::string condition = this->PrintExpr(op->args[3]);
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << ", " << src << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(tl::ptx_cp_async())) {
    int total_bytes = GetTileLangCPAsyncTransferBytes(op);
    std::string dst = this->PrintExpr(op->args[0]);
    std::string src = this->PrintExpr(op->args[1]);
    std::string size = std::to_string(total_bytes);
    this->PrintIndent();
    if (op->args.size() == 3) {
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << ", " << src
                   << ");\n";
    } else {
      std::string condition = this->PrintExpr(op->args[3]);
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << ", " << src << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(Op::Get("tirx.ptx.cp_async_commit_group"))) {
    TVM_FFI_CHECK(op->args.size() == 3 &&
                      op->args[0].as_or_throw<StringImm>()->value == "async" &&
                      op->args[1].as_or_throw<StringImm>()->value ==
                          "commit_group" &&
                      op->args[2].as_or_throw<StringImm>()->value.empty(),
                  ValueError)
        << "HIP requires a plain cp.async.commit_group operation";
    print_extern_call_stmt("tl::cp_async_commit", op->args.size());
  } else if (op->op.same_as(Op::Get("tirx.ptx.cp_async_wait_group"))) {
    TVM_FFI_CHECK(op->args.size() == 4 &&
                      op->args[1].as_or_throw<StringImm>()->value == "async" &&
                      op->args[2].as_or_throw<StringImm>()->value ==
                          "wait_group" &&
                      op->args[3].as_or_throw<StringImm>()->value.empty(),
                  ValueError)
        << "HIP requires a plain cp.async.wait_group operation";
    auto groups = tl::GetConstInt(op->args[0]);
    TVM_FFI_CHECK(groups && *groups >= 0, ValueError)
        << "HIP async wait requires a constant nonnegative group count";
    print_extern_call_stmt("tl::cp_async_wait<" + std::to_string(*groups) + ">",
                           op->args.size());
  } else if (PythonIsOp(op->op, "ptx_commit_group")) {
    print_extern_call_stmt("tl::cp_async_commit");
  } else if (PythonIsOp(op->op, "ptx_wait_group")) {
    int n = op->args[0].as_or_throw<IntImm>()->value.as<int>().value();
    std::string func_name = "tl::cp_async_wait<" + std::to_string(n) + ">";
    print_extern_call_stmt(func_name, 1);
  } else if (PythonIsOp(op->op, "create_barriers")) {
    this->PrintIndent();
    int barrier_count =
        op->args[0].as_or_throw<IntImm>()->value.as<int>().value();
    std::string barrier_name = "_mbarrier";
    this->stream << "__shared__ uint64_t " << barrier_name << "["
                 << barrier_count << "];\n";
  } else if (PythonIsOp(op->op, "ptx_arrive_barrier")) {
    print_extern_call_stmt("tl::mbarrier_arrive");
  } else if (PythonIsOp(op->op, "ptx_init_barrier_thread_count")) {
    print_extern_call_stmt("tl::mbarrier_init");
  } else if (PythonIsOp(op->op, "ptx_arrive_barrier_expect_tx")) {
    print_extern_call_stmt("tl::mbarrier_arrive_expect_tx");
  } else if (PythonIsOp(op->op, "ptx_cp_async_barrier")) {
    print_extern_call_stmt("tl::mbarrier_cp_async_arrive");
  } else if (op->op.same_as(tl::mbarrier_expect_tx())) {
    print_extern_call_stmt("tl::mbarrier_expect_tx");
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    print_extern_call_stmt("tl::mbarrier_wait");
  } else if (op->op.same_as(tl::ptx_stmatrix())) {
    int trans = op->args[0].as_or_throw<IntImm>()->value.as<int>().value();
    int num = op->args[1].as_or_throw<IntImm>()->value.as<int>().value();
    bool has_shape =
        op->args.size() >= 4 && op->args.back().as<StringImmNode>();
    if (has_shape) {
      ICHECK_EQ(op->args.back().as_or_throw<StringImm>()->value, "m8n8")
          << "HIP stmatrix codegen only supports m8n8";
    }
    std::string func_name = "tl::ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2, has_shape ? 1 : 0);
  } else if (op->op.same_as(tl::wait_wgmma())) {
    this->PrintIndent();
    int num_mma = op->args[0].as_or_throw<IntImm>()->value.as<int>().value();
    this->stream << "tl::wait_wgmma<" << std::to_string(num_mma) << ">();\n";
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "__pack_half2(" << this->PrintExpr(op->args[0]) << ", "
       << this->PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::sync_grid())) {
    this->need_cooperative_groups_ = true;
    this->PrintIndent();
    this->stream << "cooperative_groups::this_grid().sync();\n";
  } else if (op->op.same_as(tl::sync_warp())) {
    // AMD wavefronts execute in lockstep, so no runtime barrier is needed for
    // intra-wavefront convergence and the mask argument is ignored. A compiler
    // barrier is still required: without one the backend may reorder the LDS
    // accesses that a warp-synchronous reduction depends on.
    // __builtin_amdgcn_wave_barrier() emits no instructions and only constrains
    // scheduling.
    this->PrintIndent();
    this->stream << "__builtin_amdgcn_wave_barrier();\n";
  } else if (op->op.same_as(tl::any_sync())) {
    ICHECK_EQ(op->args.size(), 2U) << "tl.any_sync expects <mask, predicate>.";
    // HIP __any takes only the predicate; the mask is ignored because
    // wavefront execution is always convergent across the full wave.
    os << "__any(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::all_sync())) {
    ICHECK_EQ(op->args.size(), 2U) << "tl.all_sync expects <mask, predicate>.";
    os << "__all(" << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ballot_sync())) {
    ICHECK_EQ(op->args.size(), 2U)
        << "tl.ballot_sync expects <mask, predicate>.";
    // HIP __ballot returns uint64 natively, covering every lane of the
    // wavefront; the CUDA-style mask argument is ignored.
    os << "((unsigned long long)__ballot(" << PrintExpr(op->args[1]) << "))";
  } else if (op->op.same_as(tl::ballot())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.ballot expects <predicate>.";
    os << "((unsigned long long)__ballot(" << PrintExpr(op->args[0]) << "))";
  } else if (op->op.same_as(tl::activemask())) {
    ICHECK(op->args.empty()) << "tl.activemask takes no arguments.";
    os << "((unsigned long long)__ballot(1))";
  } else if (op->op.same_as(tl::syncthreads_count())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.syncthreads_count expects <predicate>.";
    os << "__syncthreads_count(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::syncthreads_and())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.syncthreads_and expects <predicate>.";
    os << "__syncthreads_and(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::syncthreads_or())) {
    ICHECK_EQ(op->args.size(), 1U) << "tl.syncthreads_or expects <predicate>.";
    os << "__syncthreads_or(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::shfl_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_sync expects <mask, value, src_lane, width>.";
    // HIP __shfl takes only (value, src_lane, width); the mask is ignored.
    os << "__shfl(" << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2])
       << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_xor_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_xor_sync expects <mask, value, lane_mask, width>.";
    os << "__shfl_xor(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_down_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_down_sync expects <mask, value, delta, width>.";
    os << "__shfl_down(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::shfl_up_sync())) {
    ICHECK_EQ(op->args.size(), 4U)
        << "tl.shfl_up_sync expects <mask, value, delta, width>.";
    os << "__shfl_up(" << PrintExpr(op->args[1]) << ", "
       << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3]) << ")";
  } else if (op->op.same_as(tl::match_any_sync()) ||
             op->op.same_as(tl::match_all_sync())) {
    TVM_FFI_THROW(InternalError)
        << "tl." << op->op << " is not supported on HIP: the "
        << "__match_{any,all}_sync primitives have no HIP equivalent.";
  } else if (op->op.same_as(tl::add2()) || op->op.same_as(tl::sub2()) ||
             op->op.same_as(tl::mul2()) || op->op.same_as(tl::fma2()) ||
             op->op.same_as(tl::max2()) || op->op.same_as(tl::min2()) ||
             op->op.same_as(tl::abs2())) {
    // Packed x2 element-wise math intrinsics.
    // HIP currently only supports float32x2 (float2).
    std::string op_name;
    if (op->op.same_as(tl::add2()))
      op_name = "add2";
    else if (op->op.same_as(tl::sub2()))
      op_name = "sub2";
    else if (op->op.same_as(tl::mul2()))
      op_name = "mul2";
    else if (op->op.same_as(tl::fma2()))
      op_name = "fma2";
    else if (op->op.same_as(tl::max2()))
      op_name = "max2";
    else if (op->op.same_as(tl::min2()))
      op_name = "min2";
    else
      op_name = "abs2";

    os << "tl::" << op_name << "(";
    os << PrintExpr(op->args[0]);
    for (size_t i = 1; i < op->args.size(); ++i) {
      os << ", " << PrintExpr(op->args[i]);
    }
    os << ")";
  } else if (op->op.same_as(tl::ds_read_tr16_b64())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.ds_read_tr16_b64 expects one argument (smem_access_ptr).";
    os << "tl::ds_read_tr16_b64(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ds_read_tr8_b64())) {
    ICHECK_EQ(op->args.size(), 1U)
        << "tl.ds_read_tr8_b64 expects one argument (smem_access_ptr).";
    os << "tl::ds_read_tr8_b64(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__ldg())) {
    // HIP fallback: regular load
    const TensorLoadNode *bl = op->args[0].as<TensorLoadNode>();
    ICHECK(bl) << "T.__ldg expects a BufferLoad as the first argument.";
    ICHECK_EQ(bl->indices.size(), 1)
        << "T.__ldg currently supports flattened 1D buffer accesses.";
    const VarNode *buffer = bl->source.as_or_throw<Var>().get();
    PrimExpr base = bl->indices[0];
    auto buffer_ref =
        this->GetBufferRef(PythonDType(GetRef<Expr>(op)), buffer, base);
    os << buffer_ref;
  } else if (PythonIsOp(op->op, "tvm_fill_fragment")) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (PythonIsOp(op->op, "tvm_load_matrix_sync")) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::load_matrix_sync(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[6], os);
    os << ")";
  } else if (PythonIsOp(op->op, "tvm_store_matrix_sync")) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::store_matrix_sync(";
    this->PrintExpr(op->args[5], os);
    os << ", ";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[6], os);
    if (const StringImmNode *str = op->args[7].as<StringImmNode>()) {
      os << ", nvcuda::wmma::mem_" << str->value;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid parameters";
    }
    os << ")";
  } else if (PythonIsOp(op->op, "tvm_mma_sync")) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (PythonIsOp(op->op, "tvm_bmma_sync")) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(tl::tvm_mfma())) {
    // arg 0: prefix: {otype}_{intrM}x{intrN}x{intrK}_{itype}
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: float16, float32, ...
    // arg 4: B precision: float16, float32, ...
    // arg 5: C precision: float32, float64, ...
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index

    ICHECK(op->args.size() == 12U)
        << "Invalid number of arguments for tvm_mfma";
    std::string prefix = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    ICHECK(A_layout == "row" || B_layout == "row")
        << "Matrix core only support row major";
    // map for dtype -> float32x4 -> float4
    std::unordered_map<std::string, std::string> dtype_map = {
        {"int8", "char"},
        {"int32", "int"},
        {"int8x4", "int32_t"},
        {"int8x8", "int64_t"},
        {"int8x16", "int32x4"},
        {"int32x4", "int32x4"},
        {"int32x16", "int32x16"},
        {"float16", "half"},
        {"float32", "float"},
        {"float64", "double"},
        {"float16x4", "float16x4"},
        {"float16x8", "float16x8"},
        {"bfloat16x4", "bfloat16x4_vec"},
        {"bfloat16x8", "bfloat16x8_vec"},
        {"float32x4", "float32x4"},
        {"float8_e4m3fnuzx4", "fp8_e4_4_t"},
        {"float8_e4m3fnx4", "fp8_e4_4_t"},
        {"float8_e4m3fnuzx8", "long"},
        {"float8_e4m3fnx8", "long"},
        {"float8_e5m2fnuzx4", "fp8_e5_4_t"},
        {"float8_e5m2fnuzx8", "long"},
        {"float8_e5m2x4", "fp8_e5_4_t"},
        {"float8_e5m2x8", "long"},
        {"float32x16", "float32x16"},
        {"float32x32", "float32x32"}};
    std::string call_mfma_code = R"({
      *((({C_dtype}*){c_ref}) + {c_bias}) = {mfma_buildin}(*((({A_dtype}*){a_ref}) + {a_bias}),
                    *((({B_dtype}*){b_ref}) + {b_bias}),
                    *((({C_dtype}*){c_ref}) + {c_bias}), 0, 0, 0);
    })";
    std::string mfma_buildin = "__builtin_amdgcn_mfma_" + prefix;
    Replacer replacer;

    replacer.register_rule("{mfma_buildin}", mfma_buildin);
    replacer.register_rule("{A_dtype}", dtype_map[A_dtype]);
    replacer.register_rule("{B_dtype}", dtype_map[B_dtype]);
    replacer.register_rule("{C_dtype}", dtype_map[C_dtype]);
    replacer.register_rule("{a_ref}", a_ref);
    replacer.register_rule("{a_bias}", a_bias);
    replacer.register_rule("{b_ref}", b_ref);
    replacer.register_rule("{b_bias}", b_bias);
    replacer.register_rule("{c_ref}", c_ref);
    replacer.register_rule("{c_bias}", c_bias);
    os << replacer.rewrite(call_mfma_code);
  } else if (op->op.same_as(tl::tvm_rdna_wmma())) {
    // arg 0: shape string, e.g. "f32_16x16x16_f16_w32"
    // arg 1: A layout: "row"/"col"
    // arg 2: B layout: "row"/"col"
    // arg 3: A dtype, e.g. "float16x8"
    // arg 4: B dtype
    // arg 5: C dtype, e.g. "float32x8"
    // arg 6: A data pointer
    // arg 7: A element index (in units of the vectorized type)
    // arg 8: B data pointer
    // arg 9: B element index
    // arg 10: C data pointer
    // arg 11: C element index
    ICHECK(op->args.size() == 12U) << "tvm_rdna_wmma expects 12 arguments";
    std::string shape = op->args[0].as_or_throw<StringImm>()->value;
    std::string A_layout = op->args[1].as_or_throw<StringImm>()->value;
    std::string B_layout = op->args[2].as_or_throw<StringImm>()->value;
    std::string A_dtype = op->args[3].as_or_throw<StringImm>()->value;
    std::string B_dtype = op->args[4].as_or_throw<StringImm>()->value;
    std::string C_dtype = op->args[5].as_or_throw<StringImm>()->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);

    // Get RDNA Generation
    ICHECK(target_.has_value()) << "CodeGenTileLangHIP target is not set";
    int rdna_gen = tvm::tl::TargetGetRDNAGeneration(*target_);
    ICHECK(rdna_gen == 11 || rdna_gen == 12)
        << "Unsupported RDNA target for WMMA: gfx" << target_.value()->str();

    // Determine wmma builtin name from shape
    // shape = "f32_16x16x16_f16_w32" ->
    // "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12" For gfx12 targets use
    // the _gfx12 suffix variant, which gfx11 targets don't have.
    std::string wmma_builtin = "__builtin_amdgcn_wmma_" + shape;

    int ab_half_elems = 16;
    std::string ab_vec_typedef = "tl_v16f16";
    if (rdna_gen == 12) {
      wmma_builtin += "_gfx12";
      ab_half_elems = 8;
      ab_vec_typedef = "tl_v8f16";
    }

    // Emit the WMMA call.
    // For gfx12:
    // Signature: v8f32 = wmma_builtin(v8f16 a, v8f16 b, v8f32 c)
    // where v8f16 = __fp16 x 8, v8f32 = float x 8.
    // For gfx11:
    // Signature: v8f32 = wmma_builtin(v16f16 a, v16f16 b, v8f32 c)
    // where v16f16 = __fp16 x 16, v8f32 = float x 8.
    //
    // A/B buffers hold half_t (fp16), C/D buffers hold float.
    // Each element index accesses a packed vector of 8/16 elements.
    //
    // Using typedef'd vector types for the cast:
    //   typedef __attribute__((__vector_size__(8/16 * sizeof(__fp16)))) __fp16
    //   tl_v8/16f16; typedef __attribute__((__vector_size__(8 *
    //   sizeof(float)))) float tl_v8f32;
    std::string call_wmma_code = R"({
      typedef __attribute__((__vector_size__({ab_half_elems} * sizeof(__fp16)))) __fp16 {ab_vec_typedef};
      typedef __attribute__((__vector_size__(8 * sizeof(float)))) float tl_v8f32;
      *((tl_v8f32*){c_ref} + {c_bias}) = {wmma_builtin}(
          *(({ab_vec_typedef}*){a_ref} + {a_bias}),
          *(({ab_vec_typedef}*){b_ref} + {b_bias}),
          *((tl_v8f32*){c_ref} + {c_bias}));
    })";
    Replacer wmma_replacer;
    wmma_replacer.register_rule("{wmma_builtin}", wmma_builtin);
    wmma_replacer.register_rule("{ab_half_elems}",
                                std::to_string(ab_half_elems));
    wmma_replacer.register_rule("{ab_vec_typedef}", ab_vec_typedef);
    wmma_replacer.register_rule("{a_ref}", a_ref);
    wmma_replacer.register_rule("{a_bias}", a_bias);
    wmma_replacer.register_rule("{b_ref}", b_ref);
    wmma_replacer.register_rule("{b_bias}", b_bias);
    wmma_replacer.register_rule("{c_ref}", c_ref);
    wmma_replacer.register_rule("{c_bias}", c_bias);
    os << wmma_replacer.rewrite(call_wmma_code);
  } else if (PythonIsOp(op->op, "thread_return")) {
    os << "return";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(tl::no_set_max_nreg())) {
    // HIP doesn't need explicit register management like CUDA
    // This is a no-op for HIP
    return;
  } else if (op->op.same_as(tl::annotate_producer_reg_dealloc()) ||
             op->op.same_as(tl::annotate_consumer_reg_alloc())) {
    return;
  } else if (op->op.same_as(tl::warp_reduce_sum())) {
    os << "tl::warp_reduce_sum(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_max())) {
    os << "tl::warp_reduce_max(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_min())) {
    os << "tl::warp_reduce_min(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitand())) {
    os << "tl::warp_reduce_bitand(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::warp_reduce_bitor())) {
    os << "tl::warp_reduce_bitor(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::atomic_add_elem_op())) {
    // atomic_add_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicAdd(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_add_ret_elem_op())) {
    // atomic_add_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "AtomicAddRet(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::atomic_addx2_elem_op())) {
    // atomic_addx2_elem_op(dst_ptr, src_ptr[, memory_order])
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_ptr = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicAddx2(" << dst_ptr << ", " << src_ptr;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_addx2_ret_elem_op())) {
    // atomic_addx2_ret_elem_op(dst_ptr, src_ptr[, memory_order]) -> returns
    // the previous two values
    os << "AtomicAddx2Ret(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::atomic_addx4_elem_op())) {
    // atomic_addx4_elem_op(dst_ptr, src_ptr[, memory_order])
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_ptr = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicAddx4(" << dst_ptr << ", " << src_ptr;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_addx4_ret_elem_op())) {
    // atomic_addx4_ret_elem_op(dst_ptr, src_ptr[, memory_order]) -> returns
    // the previous four values
    os << "AtomicAddx4Ret(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::atomic_load_elem_op())) {
    // atomic_load_elem_op(src_ptr, memory_order) -> returns loaded value
    os << "AtomicLoad(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::atomic_store_elem_op())) {
    // atomic_store_elem_op(dst_ptr, value, memory_order)
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string value = PrintExpr(op->args[1]);
    std::string memory_order = PrintExpr(op->args[2]);
    this->PrintIndent();
    this->stream << "AtomicStore(" << dst_ptr << ", " << value << ", "
                 << memory_order << ");\n";
  } else if (op->op.same_as(tl::atomic_or_elem_op())) {
    // atomic_or_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicOr(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_max_elem_op())) {
    // atomic_max_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicMax(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_max_ret_elem_op())) {
    // atomic_max_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "AtomicMaxRet(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else if (op->op.same_as(tl::atomic_min_elem_op())) {
    // atomic_min_elem_op(dst_ptr, src_value[, memory_order])
    std::string dst_ptr = PrintExpr(op->args[0]);
    std::string src_value = PrintExpr(op->args[1]);
    this->PrintIndent();
    this->stream << "AtomicMin(" << dst_ptr << ", " << src_value;
    if (op->args.size() > 2) {
      this->stream << ", " << PrintExpr(op->args[2]);
    }
    this->stream << ");\n";
  } else if (op->op.same_as(tl::atomic_min_ret_elem_op())) {
    // atomic_min_ret_elem_op(dst_ptr, src_value[, memory_order]) -> returns
    // prev value
    os << "AtomicMinRet(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]);
    if (op->args.size() > 2) {
      os << ", " << PrintExpr(op->args[2]);
    }
    os << ")";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenTileLangHIP::Dispatch_(const RegionStmtNode *node) {
  if (auto swizzle = node->attrs->dict.Get("threadblock_swizzle_pattern")) {
    auto fields = swizzle->cast<tvm::Tuple>()->fields;
    auto name = fields[0].as_or_throw<StringImm>()->value;
    auto panel = tl::GetConstInt(fields[1]);
    TVM_FFI_CHECK(panel && *panel > 0, ValueError)
        << "Invalid threadblock swizzle panel";
    TVM_FFI_CHECK(name == "rasterization2DRow" ||
                      name == "rasterization2DColumn",
                  ValueError)
        << "Unsupported ROCm threadblock swizzle: " << name;
    PrintIndent();
    stream << "const dim3 blockIdx = tl::" << name << "<" << *panel << ">();\n";
  }
  CodeGenC::Dispatch_(node);
}

void CodeGenTileLangHIP::Dispatch_(const BindNode *op) {
  const auto *allocation = op->value.as<CallNode>();
  if (!allocation || !allocation->op.same_as(tirx::alloc_tensor_op())) {
    CodeGenC::Dispatch_(op);
    return;
  }
  auto tensor = op->var.as_or_throw<TensorVar>();
  auto annotations =
      allocation->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict;
  alloc_storage_scope_[tensor.get()] = tensor.scope();
  std::string vid = AllocVarID(tensor.get());

  this->PrintIndent();
  std::string scope = PythonStorageScope(tensor);
  PrimType dtype = tensor.dtype();

  // FP4 scalar local buffers use packed storage (2 elements per byte).
  // Skip the normal type+scope header; emit the packed type directly below.
  bool is_fp4_scalar_local = dtype.MatchesCode(kDLFloat4_e2m1fn) &&
                             dtype.IsScalar() && scope == "local";

  if (!is_fp4_scalar_local) {
    PrintStorageScope(tensor.scope(), stream);
    PrintType(dtype, stream);
  }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    auto opt_size = tl::GetConstInt(PythonTensorSize(tensor));
    ICHECK(opt_size.has_value() && opt_size.value() > 0)
        << "Can only handle constant size stack allocation for now";
    size_t constant_size = static_cast<size_t>(opt_size.value());

    if (scope == "shared") {
      if (dtype.MatchesCode(kDLFloat4_e2m1fn) && dtype.IsScalar()) {
        constant_size = (constant_size + 1) / 2;
      } else if (dtype == PrimType::Int(4) || dtype == PrimType::UInt(4) ||
                 dtype == PrimType::Int(1)) {
        size_t elements_per_storage_word = 32 / dtype.bits();
        constant_size = (constant_size + elements_per_storage_word - 1) /
                        elements_per_storage_word;
      }
    }

    if (is_fp4_scalar_local) {
      // Use fp4_e2_2_t (2 elements per byte) as packed storage for FP4
      // local register arrays.  Record the mapping so BufferLoad/Store
      // can emit tl_fp4_packed_load / tl_fp4_packed_store calls.
      enable_fp4_ = true;
      auto vid_packed = vid + "_packed";
      stream << "fp4_e2_2_t " << vid_packed << '[' << (constant_size + 1) / 2
             << "];\n";
      fp4_packed_buffers_[tensor] = vid_packed;
    } else if (scope == "local.var") {
      // Single-element variable: emit an initializer so the value is defined.
      // Default to 0; respect the user-provided tl.local_var_init annotation.
      PrimExpr init = tl::MakeConst(tensor.dtype(), 0);
      auto init_it = annotations.find(tl::attr::kLocalVarInit);
      if (init_it != annotations.end()) {
        PrimExpr user_init = (*init_it).second.cast<PrimExpr>();
        if (!PythonDType(user_init).IsVoid() &&
            PythonDType(user_init) != tensor.dtype()) {
          user_init = prim::Cast(tensor.dtype(), user_init);
        }
        init = user_init;
      }
      stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
    } else {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    }
  }

  RegisterHandleType(tensor.get(), tensor.dtype());
}

void CodeGenTileLangHIP::Dispatch_(const TensorStoreNode *op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

  PrimType value_dtype = PythonDType(op->value);
  PrimType element_dtype = op->dest.as_or_throw<TensorVar>().dtype();
  Var buffer_var = op->dest.as_or_throw<Var>();

  // FP4 scalar store: use tl_fp4_packed_store to correctly handle nibble-level
  // writes without corrupting the neighbouring nibble.
  if (element_dtype.MatchesCode(kDLFloat4_e2m1fn) && element_dtype.IsScalar() &&
      value_dtype.IsScalar()) {
    std::string idx_str = PrintExpr(op->indices[0]);
    std::string value = this->PrintExpr(op->value);
    this->PrintIndent();
    auto packed_it = fp4_packed_buffers_.find(buffer_var);
    if (packed_it != fp4_packed_buffers_.end()) {
      stream << "tl_fp4_packed_store(" << packed_it->second << ", " << idx_str
             << ", " << value << ");\n";
    } else {
      stream << "tl_fp4_packed_store((fp4_e2_2_t*)"
             << GetVarID(buffer_var.get()) << ", " << idx_str << ", " << value
             << ");\n";
    }
    return;
  }

  // Default path for all other types.
  CodeGenC::Dispatch_(op);
}

void CodeGenTileLangHIP::Dispatch_(const RampNode *op, std::ostream &os) {
  int lanes = op->lanes.as_or_throw<IntImm>()->value.as<int>().value();
  ICHECK_LE(lanes, 4)
      << "ValueError: Ramp of more than 4 lanes is not allowed.";
  os << "(make_";
  PrintType(PythonDType(GetRef<Expr>(op)), os);
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")" << "+(" << PrintExpr(op->stride)
       << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenTileLangHIP::Dispatch_(const BroadcastNode *op,
                                   std::ostream &os) { // NOLINT(*)
  int lanes = op->lanes.as_or_throw<IntImm>()->value.as<int>().value();
  if ((PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
       PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) &&
      PythonDType(GetRef<Expr>(op)).bits() == 8 && lanes == 4) {
    // make_int8x4
    auto p = tl::GetConstInt(op->value);
    ICHECK(p);
    int64_t v = *p & 0xFF;
    v = (v << 24) | (v << 16) | (v << 8) | v;
    if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) {
      os << "(uint)" << v;
    } else {
      os << "(int)" << v;
    }
    return;
  }

  if (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLFloat, 16)) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << '(';
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      os << "__pack_half2(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16)) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << '(';
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      os << "__pack_bfloat162(" << v << ", " << v << ")";
    }
    os << ')';
    return;
  }

  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat) &&
      PythonDType(GetRef<Expr>(op)).bits() == 32 &&
      PythonDType(GetRef<Expr>(op)).lanes() == 8) {
    std::string v = PrintExpr(op->value);
    // HIP does not allow taking the address of a temporary, so use a union
    // to reinterpret float2 as unsigned long long without UB or temp-address.
    os << "make_ulonglong4(";
    for (int i = 0; i < 4; ++i) {
      if (i != 0)
        os << ", ";
      os << "([&]{ union { float2 f; unsigned long long u; } _tmp;"
         << " _tmp.f = make_float2(" << v << ", " << v
         << "); return _tmp.u; }())";
    }
    os << ')';
    return;
  }

  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat) &&
      PythonDType(GetRef<Expr>(op)).bits() == 32 &&
      (PythonDType(GetRef<Expr>(op)).lanes() == 16 ||
       PythonDType(GetRef<Expr>(op)).lanes() == 32)) {
    // float32x16/float32x32: GCC vector extension — initialize with compound
    // literal.
    std::string v = PrintExpr(op->value);
    os << "(float32x" << PythonDType(GetRef<Expr>(op)).lanes() << "){";
    for (int i = 0; i < PythonDType(GetRef<Expr>(op)).lanes(); ++i) {
      if (i != 0)
        os << ", ";
      os << v;
    }
    os << "}";
    return;
  }

  if (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16) &&
      PythonDType(GetRef<Expr>(op)).lanes() == 16) {
    // bfloat16x16: struct aggregate initializer.
    std::string v = PrintExpr(op->value);
    os << "bfloat16x16{";
    for (int i = 0; i < 16; ++i) {
      if (i != 0)
        os << ", ";
      os << v;
    }
    os << "}";
    return;
  }

  if ((PythonDType(GetRef<Expr>(op)).MatchesCode(kDLInt) ||
       PythonDType(GetRef<Expr>(op)).MatchesCode(kDLUInt)) &&
      PythonDType(GetRef<Expr>(op)).bits() == 4) {
    bool fail = false;
    auto p = tl::GetConstInt(op->value);
    ICHECK(p);
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

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangHIP *p) { // NOLINT(*)
  // Type code is kBFloat
  if (PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16)) {
    os << "bfloat16_t(";
    FloatImm const_f32 = FloatImm(PrimType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    return;
  } else if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat8_e4m3fnuz) ||
             PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat8_e4m3) ||
             PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat8_e4m3fn)) {
    os << "fp8_e4_t(";
    FloatImm const_f32 = FloatImm(PrimType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    return;
  } else if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat4_e2m1fn)) {
    // fp4_e2_t only has a raw-byte constructor, so build the constant from a
    // float at runtime via the device helper rather than the `T(<float>)` form
    // used by the wider types above.
    os << "__tl_float_to_fp4(";
    FloatImm const_f32 = FloatImm(PrimType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    return;
  }
  // Type code is kFloat
  switch (PythonDType(GetRef<Expr>(op)).bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((PythonDType(GetRef<Expr>(op)).bits() == 32) ? "HUGE_VALF"
                                                            : "HUGE_VAL");
    } else if (std::isnan(op->value)) {
      temp << ((PythonDType(GetRef<Expr>(op)).bits() == 32) ? "NAN" : "NAN");
    } else {
      temp << std::scientific << op->value;
      if (PythonDType(GetRef<Expr>(op)).bits() == 32)
        temp << 'f';
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    os << "half_t" << '(';
    FloatImm const_f32 = FloatImm(PrimType::Float(32), op->value);
    PrintConst(const_f32.get(), os, p);
    os << ')';
    break;
  }
  default:
    TVM_FFI_THROW(InternalError)
        << "Bad bit-width for float: " << PythonDType(GetRef<Expr>(op)) << "\n";
  }
}

void CodeGenTileLangHIP::Dispatch_(const FloatImmNode *op,
                                   std::ostream &os) { // NOLINT(*)
  // An fp4 constant is printed as a `__tl_float_to_fp4(...)` call declared in
  // hip_fp4.h. Printing it can happen without PrintType ever running on the fp4
  // type (e.g. a scalar store to an external fp4 buffer), so enable the header
  // here too -- this is the only path that feeds an fp4 FloatImm to PrintConst.
  if (PythonDType(GetRef<Expr>(op)).MatchesCode(kDLFloat4_e2m1fn)) {
    enable_fp4_ = true;
  }
  PrintConst(op, os, this);
}

void CodeGenTileLangHIP::HandleVolatileLoads(const std::string &value,
                                             const TensorLoadNode *op,
                                             std::ostream &os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLFloat, 16) ||
       PythonDType(GetRef<Expr>(op)).MatchesElementType(kDLBfloat, 16)) &&
      IsVolatile(op->source.as_or_throw<Var>().get())) {
    os << "(";
    PrintType(PythonDType(GetRef<Expr>(op)), os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenTileLangHIP::PrintVecElemLoadExpr(const PrimType &t, int i,
                                              const std::string &value,
                                              std::ostream &os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt))) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8
         << "))";
      return;
    }
  }

  if (t.MatchesElementType(kDLFloat, 16)) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_half2(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if (t.MatchesElementType(kDLBfloat, 16) && t.lanes() != 16) {
    if (i == 0) {
      os << "make_";
      PrintType(t, os);
      os << '(';
    }
    if (i % 2 == 0) {
      os << "__pack_bfloat162(" << value;
    } else {
      os << "," << value << ")";
      if (i != t.lanes() - 1) {
        os << ",";
      } else {
        os << ")";
      }
    }
    return;
  }

  if ((t.lanes() == 16 || t.lanes() == 32) && t.bits() == 32 &&
      t.MatchesCode(kDLFloat)) {
    // float32x16/float32x32: compound literal.
    if (i == 0)
      os << "(float32x" << t.lanes() << "){";
    os << value;
    if (i != t.lanes() - 1)
      os << ",";
    else
      os << "}";
    return;
  }

  if (t.lanes() == 16 && t.MatchesElementType(kDLBfloat, 16)) {
    // bfloat16x16: struct aggregate initializer.
    if (i == 0)
      os << "bfloat16x16{";
    os << value;
    if (i != t.lanes() - 1)
      os << ",";
    else
      os << "}";
    return;
  }

  if (i == 0) {
    os << "make_";
    PrintType(t, os);
    os << "(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << ")";
  }
  return;
}

void CodeGenTileLangHIP::AddFunction(const GlobalVar &global,
                                     const tirx::Function &function) {
  fp4_packed_buffers_.clear();
  CodeGenC::AddFunction(global, function);
}

} // namespace codegen
} // namespace tvm
