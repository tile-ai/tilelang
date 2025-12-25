/*!
 * \file target/codegen.cc
 */

#include "codegen_cuda.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "./ptx.h"
#include "./utils.h"
#include "arith/pattern_match.h"

namespace tvm {
namespace codegen {
using namespace tvm::tl::codegen;
using namespace ffi;

struct CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(t, name);
    }
    return "";
  }
};

struct CUDAFastMathTan : public CUDAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      // `__tanf` seems to produce some values too deviant from numpy tan
      // version. So, let's use just `tanf` instead.
      case 32:
        return name + 'f';
      case 16:
        return 'h' + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct CUDAIEEEMath {
  std::string operator()(DataType t, std::string name,
                         std::string rounding_mode) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + "_" + rounding_mode;
    } else if (t.is_float() && t.bits() == 64) {
      return "__d" + name + "_" + rounding_mode;
    }
    return "";
  }
};

static std::string GetTileLangFP8Type(DataType type) {
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
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else {
    LOG(FATAL)
        << "Only support scalar and vector types of width (2, 4, 8, 16, 32) "
           "for FP8";
  }
  if (type.is_float8_e4m3() || type.is_float8_e4m3fn()) {
    stream << "fp8_e4" << vec << "_t";
  } else if (type.is_float8_e5m2()) {
    stream << "fp8_e5" << vec << "_t";
  } else if (type.is_float8_e8m0fnu()) {
    stream << "fp8_e8" << vec << "_t";
  } else {
    LOG(FATAL) << "Unsupported FP8 type in CUDA codegen but got " << type;
  }
  return stream.str();
}

std::string GetTileLangFP6Type(DataType type) {
  std::stringstream stream;
  int32_t lanes = type.lanes();
  std::string vec;
  if (type.is_scalar()) {
    vec = "";
  } else if (lanes == 2) {
    vec = "x2";
  } else if (lanes == 4) {
    vec = "x4";
  } else if (lanes == 8) {
    vec = "x8";
  } else if (lanes == 16) {
    vec = "x16";
  } else {
    LOG(FATAL)
        << "Only support scalar and vector types of width (2, 4) for FP6";
  }
  stream << "__nv_fp6";
  std::string suffix;
  if (type.code() == DataType::kFloat6_e2m3fn) {
    suffix = "_e2m3";
  } else if (type.code() == DataType::kFloat6_e3m2fn) {
    suffix = "_e3m2";
  } else {
    LOG(FATAL) << "Unsupported FP6 type in CUDA codegen";
  }
  stream << vec << suffix;
  return stream.str();
}

std::string GetTileLangFP4Type(DataType type) {
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
  } else if (lanes == 16) {
    vec = "_16";
  } else if (lanes == 32) {
    vec = "_32";
  } else if (lanes == 64) {
    vec = "_64";
  } else {
    LOG(FATAL) << "Only support scalar and vector types of width (2, 4, 8, 16, "
                  "32, 64) for FP4";
  }

  std::string suffix;
  if (type.code() == DataType::kFloat4_e2m1fn) {
    suffix = "_e2";
  } else {
    LOG(FATAL) << "Unsupported FP4 type in CUDA codegen";
  }

  stream << "fp4" << suffix << vec << "_t";
  return stream.str();
}

CodeGenTileLangCUDA::CodeGenTileLangCUDA() {
  restrict_keyword_ = "__restrict__";
  vid_global_barrier_state_ =
      name_supply_->FreshName(runtime::symbol::tvm_global_barrier_state);
  vid_global_barrier_expect_ = name_supply_->FreshName("__barrier_expect");
  ICHECK_EQ(vid_global_barrier_state_,
            runtime::symbol::tvm_global_barrier_state);
}

void CodeGenTileLangCUDA::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" __global__ ";
}

class LaunchConfigExtractor : public tir::StmtVisitor {
private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->var->name_hint == "threadIdx.x" ||
          iv->thread_tag == "threadIdx.x") {
        threadIdx_x_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.y" ||
                 iv->thread_tag == "threadIdx.y") {
        threadIdx_y_ext = op->value;
      } else if (iv->var->name_hint == "threadIdx.z" ||
                 iv->thread_tag == "threadIdx.z") {
        threadIdx_z_ext = op->value;
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

public:
  PrimExpr threadIdx_x_ext = Integer(1);
  PrimExpr threadIdx_y_ext = Integer(1);
  PrimExpr threadIdx_z_ext = Integer(1);
};

void CodeGenTileLangCUDA::PrintExtraAttrs(const PrimFunc &f) {
  LaunchConfigExtractor extractor;
  extractor(f->body);
  arith::Analyzer analyzer;
  PrimExpr threadIdx_ext =
      analyzer.Simplify(extractor.threadIdx_x_ext * extractor.threadIdx_y_ext *
                        extractor.threadIdx_z_ext);
  if (const IntImmNode *const threadIdx_ext_int =
          threadIdx_ext.as<IntImmNode>()) {
    if (threadIdx_ext_int->value == 1) {
      // unable to extract the number of threads per block, hence directly
      // return
      return;
    }
    stream << " __launch_bounds__(" << threadIdx_ext_int->value << ", 1)";
  }
}

std::string CodeGenTileLangCUDA::Finish() {
  if (need_mma_h_) {
    decl_stream << "#include <mma.h>\n";
  }
  if (need_mma_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/mma.h>\n";
  }
  if (need_wgmma_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/wgmma.h>\n";
  }
  if (need_tcgen05mma_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/tcgen05mma.h>\n";
  }
  if (need_mma_sm70_instruction_h_) {
    decl_stream << "#include <tl_templates/cuda/instruction/mma_sm70.h>\n";
  }
  if (need_tcgen05_common_h_) {
    decl_stream << "#include <tl_templates/cuda/tcgen_05.h>\n";
  }
  if (enable_fp8_) {
    decl_stream << "#include <tl_templates/cuda/cuda_fp8.h>\n";
  }
  if (enable_fp4_) {
    decl_stream << "#include <tl_templates/cuda/cuda_fp4.h>\n";
  }

  if (need_math_constants_h_) {
    decl_stream << "#include <math_constants.h>\n";
  }

  if (need_cooperative_groups_) {
    decl_stream << "#include <cooperative_groups.h>\n";
  }

  if (need_curand_kernel_h_) {
    decl_stream << "#include <curand_kernel.h>\n";
  }

  decl_stream << "#include <tl_templates/cuda/gemm.h>\n";
  if (enable_sparse_gemm_) {
    decl_stream << "#include <tl_templates/cuda/gemm_sp.h>\n";
  }
  decl_stream << "#include <tl_templates/cuda/copy.h>\n";
  decl_stream << "#include <tl_templates/cuda/reduce.h>\n";
  decl_stream << "#include <tl_templates/cuda/ldsm.h>\n";
  decl_stream << "#include <tl_templates/cuda/threadblock_swizzle.h>\n";
  decl_stream << "#include <tl_templates/cuda/debug.h>\n";
  decl_stream << "#ifdef ENABLE_BF16\n";
  decl_stream << "#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>\n";
  decl_stream << "#endif\n";

  if (need_global_barrier_) {
    decl_stream << "__device__ unsigned " << vid_global_barrier_state_
                << " = 0;\n";
  }
  decl_stream << "\n";

  return CodeGenC::Finish();
}

void CodeGenTileLangCUDA::VisitStmt_(const tir::ForNode *op) {
  if (op->kind == tir::ForKind::kUnrolled) {
    PrintIndent();
    if (unroll_factor.count(op->loop_var.get())) {
      stream << "#pragma unroll "
             << PrintExpr(unroll_factor[op->loop_var.get()]) << "\n";
    } else {
      stream << "#pragma unroll\n";
    }
  }
  std::string extent =
      PrintExpr(arith::Analyzer().Simplify(op->extent + op->min));
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  std::string start = PrintExpr(op->min);
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = " << start << "; " << vid << " < " << extent
         << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangCUDA::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  var_idmap_[iv->var.get()] =
      CastFromTo(iv->thread_tag, DataType::UInt(32), iv->var.dtype());
}

void CodeGenTileLangCUDA::PrintType(DataType t, std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK(t.is_scalar()) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }

  if (t == tl::cuTensorMapType()) {
    os << "CUtensorMap";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
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
      } else if (lanes <= 16) {
        ICHECK_EQ(lanes % 4, 0) << "only support (mod 4 = 0) lanes for half "
                                   "type of more than 8 lanes";
        os << "ulonglong" << lanes / 4;
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
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
    if (!fail && (lanes > 4 && lanes <= 8 && t.bits() == 32))
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "bfloat16_t";
    } else if (lanes <= 8) {
      ICHECK_EQ(lanes % 2, 0) << "only support even lane for half type";
      os << "uint" << lanes / 2;
    } else if (lanes <= 16) {
      ICHECK_EQ(lanes % 4, 0) << "only support (mod 4 = 0) lanes for half type "
                                 "of more than 8 lanes";
      os << "ulonglong" << lanes / 4;
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t.is_float8()) {
    enable_fp8_ = true;
    os << GetTileLangFP8Type(t);
    return;
  } else if (t.is_float6()) {
    enable_fp6_ = true;
    if (t.lanes() <= 4) {
      os << GetTileLangFP6Type(t);
    }
    return;
  } else if (t.is_float4()) {
    enable_fp4_ = true;
    if (t.lanes() <= 64) {
      os << GetTileLangFP4Type(t);
    } else {
      fail = true;
    }
    return;
  } else if (t == DataType::Bool()) {
    os << "bool";
    return;
  } else if (t.is_vector_bool()) {
    // CUDA does not support bool vectors.
    // Use ushort vectors to represent instead.
    int n = t.lanes();
    if (n <= 4) {
      os << "ushort" << n;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
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
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 4: {
      if (t.is_scalar()) {
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
        LOG(FATAL) << "Cannot convert type " << t << " to CUDA type!";
      }
    }
    case 8: {
      if (t.lanes() == 4) {
        // directly 4 8 bit int in integer.
        enable_int8_ = true;

        // We use int for int8x4 instead of char4 because using char4 is
        // likely to produce extra instructions to pack four int8 elements
        // into 32-bit data.
        os << "int";
        return;
      } else if (t.lanes() == 8) {
        enable_int8_ = true;
        os << "int2";
        return;
      } else if (t.lanes() == 16) {
        enable_int8_ = true;
        os << "int4";
        return;
      } else if (t.lanes() == 32) {
        enable_int8_ = true;
        os << "longlong4";
        return;
      } else if (!t.is_uint() && t.is_scalar()) {
        os << "signed char";
        break;
      } else {
        os << "char";
        break;
      }
    }
    case 16: {
      if (t.is_scalar()) {
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
      if (t.is_scalar()) {
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
      if (t.is_scalar()) {
        os << "int64_t";
      } else if (t.lanes() == 2) {
        os << "longlong2";
      } else if (t.lanes() == 3) {
        os << "longlong3";
      } else if (t.lanes() == 4) {
        os << "longlong4";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
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
  LOG(FATAL) << "Cannot convert type " << t << " to CUDA type";
}

void CodeGenTileLangCUDA::PrintVecBinaryOp(const std::string &op, DataType t,
                                           PrimExpr lhs, PrimExpr rhs,
                                           std::ostream &os) { // NOLINT(*)
  // Declare the result.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(t, stream);
  stream << ' ' << sret << ";\n";
  int ssa_scope = BeginScope();
  {
    // Unpack into individual ops.
    std::string vlhs = SSAGetID(PrintExpr(lhs), lhs.dtype());
    std::string vrhs = SSAGetID(PrintExpr(rhs), rhs.dtype());

    for (int i = 0, lanes = t.lanes(); i < lanes; ++i) {
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
  EndScope(ssa_scope);
  os << sret;
}

void CodeGenTileLangCUDA::PrintVecElemLoad(const std::string &vec, DataType t,
                                           int i,
                                           std::ostream &os) { // NOLINT(*)
  if (t.is_scalar()) {
    os << vec;
    return;
  }

  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 256 / t.bits())
      << "i: " << i << " t: " << t << " t.bits(): " << t.bits()
      << " t.lanes(): " << t.lanes();
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    std::string type_name = t.is_int() ? "char" : "unsigned char";
    if (t.lanes() == 2 || t.lanes() == 3) {
      os << vec << "." << access[i % t.lanes()];
    } else if (t.lanes() <= 16) {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      os << "((" << type_name << ")(" << ac << " >> " << i % 4 * 8 << "))";
    } else {
      ICHECK(t.lanes() == 32);
      std::string ac = vec + "." + access[i / 8];
      os << "((" << type_name << ")(" << ac << " >> " << i % 8 * 8 << "))";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 8) {
      os << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
         << access[i % 2];
    } else {
      os << "(((half2*)(&(" << vec << "." << access[i / 4] << "))) + "
         << (i / 2 % 2) << ")->" << access[i % 2];
    }
  } else if (t.is_bfloat16()) {
    if (t.lanes() <= 8) {
      os << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
         << access[i % 2];
    } else {
      os << "(((nv_bfloat162*)(&(" << vec << "." << access[i / 4] << "))) + "
         << (i / 2 % 2) << ")->" << access[i % 2];
    }
  } else if (t.is_float8()) {
    os << vec;
    // fp8_e5_32_t
    if (t.lanes() >= 32)
      os << "." << access[i / 16];
    // fp8_e5_16_t
    if (t.lanes() >= 16)
      os << "." << access[(i % 16) / 8];
    // fp8_e5_8_t
    if (t.lanes() >= 8)
      os << "." << access[(i % 8) / 4];
    // fp8_e5_4_t or fp8_e5_2_t
    os << "." << access[i % 4];
  } else if (t.is_float4_e2m1fn()) {
    os << vec;
    // fp4_e2_64_t
    if (t.lanes() >= 64)
      os << "." << access[i / 32];
    // fp4_e2_32_t
    if (t.lanes() >= 32)
      os << "." << access[(i % 32) / 16];
    // fp4_e2_16_t
    if (t.lanes() >= 16)
      os << "." << access[(i % 16) / 8];
    // fp4_e2_8_t
    if (t.lanes() >= 8)
      os << "." << access[(i % 8) / 4];
    // fp4_e2_4_t or fp4_e2_2_t
    os << "." << access[i % 4];
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
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

void CodeGenTileLangCUDA::PrintVecElemStore(const std::string &vec, DataType t,
                                            int i, const std::string &value) {
  this->PrintIndent();
  static const char access[] = {'x', 'y', 'z', 'w'};
  ICHECK(i >= 0 && i < 256 / t.bits());
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (t.lanes() == 2 || t.lanes() == 3) {
      stream << vec << '.' << access[i % t.lanes()] << "="
             << "(" << value << ");\n";
    } else if (t.lanes() <= 16) {
      std::string ac = t.lanes() == 4 ? vec : (vec + "." + access[i / 4]);
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 4 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 4 * 8 << ");\n";
    } else {
      ICHECK(t.lanes() == 32);
      std::string ac = vec + "." + access[i / 8];
      stream << ac << "=";
      // Do not read the first undef lane.
      if (i != 0) {
        stream << ac << " & ~(0x000000ff << " << i % 8 * 8 << ") |";
      }
      stream << "(" << value << " << " << i % 8 * 8 << ");\n";
    }
  } else if (t.is_float16()) {
    if (t.lanes() <= 8) {
      stream << "((half2*)(&(" << vec << "." << access[i / 2] << ")))->"
             << access[i % 2] << " = " << value << ";\n";
    } else {
      stream << "(((half2*)(&(" << vec << "." << access[i / 4] << "))) + "
             << (i / 2 % 2) << ")->" << access[i % 2] << " = " << value
             << ";\n";
    }
  } else if (t.is_bfloat16()) {
    if (t.lanes() <= 8) {
      stream << "((nv_bfloat162*)(&(" << vec << "." << access[i / 2] << ")))->"
             << access[i % 2] << " = " << value << ";\n";
    } else {
      stream << "(((nv_bfloat162*)(&(" << vec << "." << access[i / 4]
             << "))) + " << (i / 2 % 2) << ")->" << access[i % 2] << " = "
             << value << ";\n";
    }
  } else if (t.is_float8()) {
    stream << vec;
    // fp8_e5_32_t
    if (t.lanes() >= 32)
      stream << "." << access[i / 16];
    // fp8_e5_16_t
    if (t.lanes() >= 16)
      stream << "." << access[(i % 16) / 8];
    // fp8_e5_8_t
    if (t.lanes() >= 8)
      stream << "." << access[(i % 8) / 4];
    // fp8_e5_4_t or fp8_e5_2_t
    stream << "." << access[i % 4] << " = " << value << ";\n";
  } else if (t.lanes() > 4 && t.lanes() <= 8) {
    std::string type_name;
    if (t.bits() == 16) {
      if (t.is_int()) {
        type_name = "short";
      } else if (t.is_uint()) {
        type_name = "ushort";
      }
    } else if (t.bits() == 32) {
      if (t.is_int()) {
        type_name = "int";
      } else if (t.is_uint()) {
        type_name = "uint";
      } else if (t.is_float()) {
        type_name = "float";
      }
    }
    ICHECK(!type_name.empty());
    stream << "((" << type_name << "2*)(&(" << vec << "." << access[i / 2]
           << ")))->" << access[i % 2] << " = " << value << ";\n";
  } else if (t.is_float4_e2m1fn()) {
    stream << vec;
    // fp4_e2_64_t
    if (t.lanes() >= 64)
      stream << "." << access[i / 32];
    // fp4_e2_32_t
    if (t.lanes() >= 32)
      stream << "." << access[(i % 32) / 16];
    // fp4_e2_16_t
    if (t.lanes() >= 16)
      stream << "." << access[(i % 16) / 8];
    // fp4_e2_8_t
    if (t.lanes() >= 8)
      stream << "." << access[(i % 8) / 4];
    // fp4_e2_4_t or fp4_e2_2_t
    stream << "." << access[i % 4] << " = " << value << ";\n";
  } else {
    stream << vec << "." << access[i] << " = " << value << ";\n";
  }
}

void CodeGenTileLangCUDA::PrintStorageSync(const CallNode *op) {
  auto args = op->args;
  const std::string &sync = args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // DO nothing.
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    if (args.size() == 1) {
      this->stream << "__syncthreads();\n";
    } else if (args.size() == 2) {
      auto barrier_id = args[1].as<IntImmNode>()->value;
      this->stream << "tl::__sync_thread_partial<" << barrier_id << ">();\n";
    } else if (args.size() == 3) {
      auto barrier_id = args[1].as<IntImmNode>()->value;
      auto thread_count = args[2].as<IntImmNode>()->value;
      this->stream << "tl::__sync_thread_partial<" << barrier_id << ", "
                   << thread_count << ">();\n";
    } else {
      LOG(FATAL) << "Invalid number of arguments for storage sync: "
                 << args.size();
    }
  } else if (sync == "global") {
    if (!need_global_barrier_) {
      need_global_barrier_ = true;
    }
    // global synchronizer
    std::string is_load = PrintExpr(op->args[1]);
    std::string num_blocks = PrintExpr(op->args[2]);
    this->PrintIndent();
    // In theory only threadfence is needed
    // but we observed problems with only threadfence
    this->stream << "__threadfence_system();\n";
    this->PrintIndent();
    this->stream << "if (" << is_load << ") {\n";
    int wb = this->BeginScope();
    this->PrintIndent();
    this->stream << "atomicAdd(&" << vid_global_barrier_state_ << ", 1);\n";
    this->PrintIndent();
    std::string ptr = name_supply_->FreshName("pf");
    this->stream << "volatile unsigned* " << ptr << " = &"
                 << vid_global_barrier_state_ << ";\n";
    this->PrintIndent();
    this->stream << vid_global_barrier_expect_ << " += " << num_blocks << ";\n";
    this->PrintIndent();
    this->stream << "while (" << ptr << "[0] < " << vid_global_barrier_expect_
                 << ");\n";
    this->EndScope(wb);
    this->PrintIndent();
    this->stream << "}\n";
    this->PrintIndent();
    this->stream << "__syncthreads();\n";
  }
}

void CodeGenTileLangCUDA::PrintStorageScope(const std::string &scope,
                                            std::ostream &os) { // NOLINT(*)
  ICHECK_NE(scope, "global")
      << "Cannot allocate global memory when targeting CUDA. You must pass "
         "all global arrays as input instead";
  if (scope == "shared" || scope == "shared.barrier") {
    os << "__shared__ ";
  } else if (scope == "shared.dyn") {
    os << "extern __shared__ __align__(1024) ";
  }
}

std::string CodeGenTileLangCUDA::CastFromTo(std::string value, DataType from,
                                            DataType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")";
  if (from.is_float16() && (target.is_int() || target.is_uint()) &&
      target.bits() == 8) {
    os << "(";
    if (target.is_uint()) {
      os << "u";
    }
    os << "int)";
  }
  if ((from.is_float16() || from.is_bfloat16()) && target.is_float8()) {
    os << "(float)";
  }
  os << value << ")";
  return os.str();
}

void CodeGenTileLangCUDA::VisitExpr_(const CastNode *op, std::ostream &os) {
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  ICHECK_EQ(target_ty.lanes(), from_ty.lanes());

  // Emit simple C-style type conversion.
  if (from_ty.is_scalar())
    return CodeGenC::VisitExpr_(op, os);

  // We could emit make_float4 like calls, but the emitted code looks
  // too compact to read. Emit this as vectorized unary ops.
  std::string sret = name_supply_->FreshName("_");
  this->PrintIndent();
  this->PrintType(target_ty, stream);
  stream << ' ' << sret << ";\n";
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

  // Handle conversion from float16 to float32
  if (from_ty.is_float16() && target_ty.is_float() && target_ty.bits() == 32) {
    // Use __half22float2 for vectorized conversion (half2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__half22float2", "half2", "float2");
      return;
    }
  }

  // Handle conversion from float32 to float16
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_float16()) {
    // Use __float22half2_rn for vectorized conversion (float2 -> half2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22half2_rn", "float2", "half2");
      return;
    }
  }

  // Handle conversion from bfloat16 to float32
  if (from_ty.is_bfloat16() && target_ty.is_float() && target_ty.bits() == 32) {
    // Use __bfloat1622float2 for vectorized conversion (bfloat162 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__bfloat1622float2", "__nv_bfloat162", "float2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from float32 to bfloat16
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_bfloat16()) {
    // Use __float22bfloat162_rn for vectorized conversion (float2 -> bfloat162)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__float22bfloat162_rn", "float2", "__nv_bfloat162",
                          "", false, true);
      return;
    }
  }

  // Handle conversion from float32 to float8 (E4M3/E5M2)
  if (from_ty.is_float() && from_ty.bits() == 32 &&
      tl::IsCudaVectorizableFP8(target_ty)) {
    bool target_type_is_e4m3 =
        target_ty.is_float8_e4m3() || target_ty.is_float8_e4m3fn();
    std::string type_suffix = target_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";

    // Use __nv_cvt_float2_to_fp8x2 for vectorized conversion (float2 -> fp8x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      std::string extra_args = ", __NV_SATFINITE, " + type_suffix;
      PrintVectorizedCast("__nv_cvt_float2_to_fp8x2", "float2",
                          "__nv_fp8x2_storage_t", extra_args, false, true);
      return;
    }
  }

  // Handle conversion from float8 (E4M3/E5M2) to float32
  if (tl::IsCudaVectorizableFP8(from_ty) && target_ty.is_float()) {
    bool from_type_is_e4m3 =
        from_ty.is_float8_e4m3() || from_ty.is_float8_e4m3fn();
    std::string type_suffix = from_type_is_e4m3 ? "__NV_E4M3" : "__NV_E5M2";

    // Use __tl_cvt_fp8x2_to_float2 for vectorized conversion (fp8x2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp8x2_to_float2", "__nv_fp8x2_storage_t",
                          "float2", ", " + type_suffix, true, false);
      return;
    }
  }

  // Handle conversion from float16 to float4 (E2M1)
  if (from_ty.is_float16() && target_ty.is_float4_e2m1fn()) {
    // Use __tl_cvt_half2_to_fp4x2 for vectorized conversion (half2 -> fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_half2_to_fp4x2", "half2", "uint8_t", "",
                          false, true);
      return;
    }
  }

  // Handle conversion from float32 to float4 (E2M1)
  if (from_ty.is_float() && target_ty.is_float4_e2m1fn()) {
    // Use __tl_cvt_float2_to_fp4x2 for vectorized conversion (float2 -> fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_float2_to_fp4x2", "float2", "uint8_t", "",
                          false, true);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to float16
  if (from_ty.is_float4_e2m1fn() && target_ty.is_float16()) {
    // Use __tl_cvt_fp4x2_to_half2 for vectorized conversion (fp4x2 -> half2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_half2", "uint8_t", "half2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to float32
  if (from_ty.is_float4_e2m1fn() && target_ty.is_float()) {
    // Use __tl_cvt_fp4x2_to_float2 for vectorized conversion (fp4x2 -> float2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_float2", "uint8_t", "float2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from double to float4 (E2M1)
  if (from_ty.is_float() && from_ty.bits() == 64 &&
      target_ty.is_float4_e2m1fn()) {
    // Use __tl_cvt_double2_to_fp4x2 for vectorized conversion (double2 ->
    // fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_double2_to_fp4x2", "double2", "uint8_t", "",
                          false, true);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to double
  if (from_ty.is_float4_e2m1fn() && target_ty.is_float() &&
      target_ty.bits() == 64) {
    // Use __tl_cvt_fp4x2_to_double2 for vectorized conversion (fp4x2 ->
    // double2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_double2", "uint8_t", "double2", "",
                          true, false);
      return;
    }
  }

  // Handle conversion from bfloat16 to float4 (E2M1)
  if (from_ty.is_bfloat16() && target_ty.is_float4_e2m1fn()) {
    // Use __tl_cvt_bfloat162_to_fp4x2 for vectorized conversion (bfloat162 ->
    // fp4x2)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_bfloat162_to_fp4x2", "__nv_bfloat162",
                          "uint8_t", "", false, true);
      return;
    }
  }

  // Handle conversion from float4 (E2M1) to bfloat16
  if (from_ty.is_float4_e2m1fn() && target_ty.is_bfloat16()) {
    // Use __tl_cvt_fp4x2_to_bfloat162 for vectorized conversion (fp4x2 ->
    // bfloat162)
    if (lanes == 2 || lanes == 4 || lanes == 8) {
      PrintVectorizedCast("__tl_cvt_fp4x2_to_bfloat162", "uint8_t",
                          "__nv_bfloat162", "", true, false);
      return;
    }
  }

  // Fallback: elementwise cast
  for (int i = 0, lanes = from_ty.lanes(); i < lanes; ++i) {
    std::ostringstream val;
    val << "(";
    PrintType(target_ty.element_of(), val);
    val << ")(";
    PrintVecElemLoad(src, from_ty, i, val);
    val << ")";
    PrintVecElemStore(sret, target_ty, i, val.str());
  }

  os << sret;
}

void CodeGenTileLangCUDA::VisitExpr_(const MinNode *op, std::ostream &os) {
  // TODO(wt): Consider vectorized reduction and impl for other dtypes
  DataType t = op->dtype;

  // Standard min/max functions don't support bfloat16 or float16
  if ((t.is_bfloat16() || t.is_float16()) && t.is_scalar()) {
    os << "cutlass::fast_min(" << PrintExpr(op->a) << ", " << PrintExpr(op->b)
       << ")";
    return;
  }

  // For float32 and float64 scalar, use standard min functions
  if (t.is_float() && t.is_scalar()) {
    if (t.bits() == 32 || t.bits() == 64) {
      os << "min(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
      return;
    }
  }

  // For all other scalar types (int, uint), use default implementation
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangCUDA::VisitExpr_(const MaxNode *op, std::ostream &os) {
  // TODO(wt): Consider vectorized reduction and impl for other dtypes
  DataType t = op->dtype;

  // Standard min/max functions don't support bfloat16 or float16
  if ((t.is_bfloat16() || t.is_float16()) && t.is_scalar()) {
    os << "cutlass::fast_max(" << PrintExpr(op->a) << ", " << PrintExpr(op->b)
       << ")";
    return;
  }

  // For float32 and float64 scalar, use standard max functions
  if (t.is_float() && t.is_scalar()) {
    if (t.bits() == 32 || t.bits() == 64) {
      os << "max(" << PrintExpr(op->a) << ", " << PrintExpr(op->b) << ")";
      return;
    }
  }

  // For all other scalar types (int, uint), use default implementation
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangCUDA::PrintCallExtern(Type ret_type, String global_symbol,
                                          const Array<PrimExpr> &args,
                                          bool skip_first_arg,
                                          std::ostream &os) { // NOLINT(*)
  DataType ret_dtype = GetRuntimeDataType(ret_type);
  if (ret_dtype.is_fixed_length_vector()) {
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
        std::string val = SSAGetID(PrintExpr(args[i]), args[i].dtype());
        sargs.push_back(std::move(val));
      }

      // Emit a scalar call for each lane.
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
  } else {
    CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg,
                              os);
  }
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangCUDA::GetBufferRef(DataType t,
                                              const BufferNode *buffer,
                                              PrimExpr index) {
  const VarNode *buffer_var = buffer->data.get();
  std::ostringstream os;
  std::string vid = GetVarID(buffer_var);
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  // bool is_vol = IsVolatile(buffer_var);
  // always false for tl cutlass backend.
  bool is_vol = false;

  auto ptr_cast = [this, is_vol, scope](DataType pointed_to) {
    std::ostringstream ptr_os;
    ptr_os << "(";
    if (is_vol) {
      ptr_os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, ptr_os);
    }
    PrintType(pointed_to, ptr_os);
    ptr_os << "*)";
    return ptr_os.str();
  };

  DataType buffer_element_dtype = buffer->dtype;

  std::string buffer_str = vid;
  if (!HandleTypeMatch(buffer_var, buffer_element_dtype) || is_vol) {
    std::stringstream temp;
    temp << "(" << ptr_cast(buffer_element_dtype) << vid << ")";
    buffer_str = temp.str();
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }
  if (scope == "local.var" || scope.find("local.descriptor") == 0) {
    os << vid;
    return os.str();
  }
  std::string index_str = PrintExpr(index);
  if ((t.bits() == 4 && !t.is_float4()) || (t.bits() == 1 && t.is_int())) {
    // This is a special case, because CodegenCUDA::PrintType()
    // returns "int" for bool and for 4-bit integers. In most cases,
    // we divide by the number of lanes to determine the index.
    // However, the backing type for scalar int4 and scalar bool is
    // int32.  Therefore, we need to divide by the ratio of their
    // sizes in that case.
    int div_factor = (t.lanes() == 1) ? (32 / t.bits()) : t.lanes();
    index_str =
        PrintExpr(arith::Analyzer().Simplify(truncdiv(index, div_factor)));

    os << "*((" << ptr_cast(t) << vid << ")" << " + " << index_str << ")";
  } else if (t == buffer_element_dtype) {
    os << buffer_str << "[" << index_str << "]";
  } else {
    // Fix fp4 pointer arithmetic: fp4 elements are 4-bit packed 2 per byte.
    // fp4* + n incorrectly advances n bytes (skipping 2n elements).
    int div_factor = 1;
    if (buffer_element_dtype.is_float4() && buffer_element_dtype.lanes() == 1) {
      div_factor = 2;
    }
    index_str =
        PrintExpr(arith::Analyzer().Simplify(truncdiv(index, div_factor)));

    os << "*" << ptr_cast(t) << "(" << buffer_str << " + " << index_str << ")";
  }

  return os.str();
}

std::string CodeGenTileLangCUDA::GetVecLoad(DataType t,
                                            const BufferNode *buffer,
                                            PrimExpr base) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }

  if (scope != "global" || t.bits() * t.lanes() <= 128) {
    return this->CodeGenC::GetVecLoad(t, buffer, base);
  }
  ICHECK_EQ(t.bits() * t.lanes(), 256)
      << "Unsupported vector load size: " << t.bits() * t.lanes();
  auto buffer_ref = this->GetBufferRef(t, buffer, base);
  std::ostringstream os;
  os << "tl::ld_global_256(&(" << buffer_ref << "))";
  return os.str();
}

void CodeGenTileLangCUDA::PrintVecStore(const BufferNode *buffer, DataType t,
                                        PrimExpr base,
                                        const std::string &value) {
  const VarNode *buffer_var = buffer->data.get();
  std::string scope;
  if (alloc_storage_scope_.count(buffer_var)) {
    scope = alloc_storage_scope_.at(buffer_var);
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(buffer->data);
  }

  if (scope != "global" || t.bits() * t.lanes() <= 128) {
    this->CodeGenC::PrintVecStore(buffer, t, base, value);
    return;
  }
  ICHECK_EQ(t.bits() * t.lanes(), 256)
      << "Unsupported vector load size: " << t.bits() * t.lanes();
  auto buffer_ref = this->GetBufferRef(t, buffer, base);
  this->PrintIndent();
  this->stream << "tl::st_global_256(&(" << buffer_ref << "), " << value
               << ");\n";
}

/**
 * @brief Emit CUDA/TensorLib-specific code for a call expression.
 *
 * This visitor handles CallNode intrinsics and builtins that require emitting
 * CUDA/TL-specific code (inline PTX/ASM sequences, TensorLanguage runtime
 * calls, WMMA/TMA helpers, barriers, cp.async primitives, index-map based
 * stores, reinterpret/packing helpers, and various mma/ldmatrix patterns). The
 * function writes the generated code to the provided output stream and falls
 * back to the C codegen for unrecognized calls.
 *
 * The method recognizes and emits code for (non-exhaustive): cp.async and its
 * commit/wait variants, tma_load/store and im2col variants, ptX
 * ldmatrix/stmatrix helpers, mbarrier APIs, cooperative grid sync, WMMA/legacy
 * MMA intrinsics (fill/load/store/mma/bmma/ptx_mma/ptx_mma_sp), low-level PTX
 * asm helpers (ldg32, cp_async bulk/init/arrive/wait barriers), reinterpret
 * paths for special small-float encodings (e.g., float4 e2m1fn), tl::tl_gemm
 * and related external calls, and other TL runtime calls.
 *
 * Side effects:
 * - Emits to `os` and the internal codegen output stream.
 * - May set internal feature flags (e.g., need_cooperative_groups_,
 * need_mma_h_, need_cast_smem_ptr_to_int_, enable_sparse_gemm_).
 * - May open/close SSA scopes and mutate internal variable mappings.
 * - May call LOG(FATAL) / CHECK / ICHECK on invalid or unsupported argument
 *   patterns.
 *
 * @param op The call node to generate code for; the function inspects op->op
 *           and op->args to determine the appropriate emission.
 * @param os  Output stream to receive expression-level output when the caller
 *            expects an expression result (some paths write directly to the
 *            member stream instead).
 */
void CodeGenTileLangCUDA::VisitExpr_(const CallNode *op, std::ostream &os) {
  auto print_extern_call_stmt = [&](std::string name, size_t start = 0,
                                    size_t end = 0) {
    // Cache context into a private ss, otherwise the let node may generate
    // within the function call arguments.
    std::ostringstream ss;

    for (size_t i = start; i < op->args.size() - end; i++) {
      if (i > start)
        ss << ", ";
      ss << this->PrintExpr(op->args[i]);
    }

    this->PrintIndent();
    this->stream << name << "(";
    this->stream << ss.str();
    this->stream << ");\n";
  };
  auto print_mbarrier_obj = [&](PrimExpr barrier_id) {
    std::ostringstream ss;
    if (barrier_id.as<IntImmNode>()) {
      // incase the barrier_id is an integer, we need to print the barrier_id as
      // an integer
      ss << mbarrier_name_ << "[" << barrier_id << "]";
    } else {
      // otherwise may be a T.get_mbarrier() call or BufferLoad Node
      // we need to print the barrier_id as a string
      ss << this->PrintExpr(barrier_id);
    }
    return ss.str();
  };
  if (op->op.same_as(builtin::ptx_cp_async())) {
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    // use size of argument list to indicate whether or not to use predicated
    // cp.async
    if (op->args.size() == 5) {
      this->PrintIndent();
      this->stream << "tl::cp_async_gs<" << size << ">(" << dst << "+"
                   << dst_offset << ", " << src << "+" << src_offset << ");\n";
    } else {
      std::string condition = this->PrintExpr(op->args[5]);
      this->PrintIndent();
      this->stream << "tl::cp_async_gs_conditional<" << size << ">(" << dst
                   << "+" << dst_offset << ", " << src << "+" << src_offset
                   << ", " << condition << ");\n";
    }
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    print_extern_call_stmt("tl::cp_async_commit");
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    int n = Downcast<IntImm>(op->args[0])->value;
    std::string func_name = "tl::cp_async_wait<" + std::to_string(n) + ">";
    print_extern_call_stmt(func_name, 1);
  } else if (op->op.same_as(builtin::create_barriers())) {
    this->PrintIndent();
    int barrier_count = Downcast<IntImm>(op->args[0])->value;
    auto mbarrier_storage_name = mbarrier_name_ + "_mem";
    this->stream << "__shared__ uint64_t " << mbarrier_storage_name << "["
                 << barrier_count << "];\n";
    this->PrintIndent();
    this->stream << "auto " << mbarrier_name_ << " = reinterpret_cast<"
                 << mbarrier_dtype_ << "*>(" << mbarrier_storage_name << ");\n";
  } else if (op->op.same_as(tl::get_mbarrier())) {
    ICHECK_EQ(op->args.size(), 1);
    std::string barrier_id = this->PrintExpr(op->args[0]);
    os << mbarrier_name_ + "[" + barrier_id + "]";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
    if (op->args.size() == 1) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      this->stream << mbarrier_obj << ".arrive();\n";
    } else if (op->args.size() == 3) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto cta_id = this->PrintExpr(op->args[1]);
      auto pred = this->PrintExpr(op->args[2]);
      this->stream << mbarrier_obj << ".arrive(" << cta_id << ", " << pred
                   << ");\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier "
                 << op->args.size();
    }
  } else if (op->op.same_as(builtin::ptx_init_barrier_thread_count())) {
    ICHECK_EQ(op->args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto arrive_count = this->PrintExpr(op->args[1]);
    this->stream << mbarrier_obj << ".init(" << arrive_count << ");\n";
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    if (op->args.size() == 2) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto transaction_bytes = this->PrintExpr(op->args[1]);
      this->stream << mbarrier_obj << ".arrive_and_expect_tx("
                   << transaction_bytes << ");\n";
    } else if (op->args.size() == 4) {
      this->PrintIndent();
      auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
      auto transaction_bytes = this->PrintExpr(op->args[1]);
      auto cta_id = this->PrintExpr(op->args[2]);
      auto pred = this->PrintExpr(op->args[3]);
      this->stream << mbarrier_obj << ".arrive_and_expect_tx("
                   << transaction_bytes << ", " << cta_id << ", " << pred
                   << ");\n";
    } else {
      LOG(FATAL) << "Invalid parameter  for tl::arrive_barrier_expect_tx "
                 << op->args.size();
    }
  } else if (op->op.same_as(builtin::ptx_cp_async_barrier())) {
    print_extern_call_stmt("tl::mbarrier_cp_async_arrive");
  } else if (op->op.same_as(tl::ptx_fence_barrier_init())) {
    print_extern_call_stmt("tl::fence_barrier_init");
  } else if (op->op.same_as(tl::ptx_cp_async_barrier_noinc())) {
    print_extern_call_stmt("tl::mbarrier_cp_async_arrive_noinc");
  } else if (op->op.same_as(tl::mbarrier_expect_tx())) {
    ICHECK_EQ(op->args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto transaction_bytes = this->PrintExpr(op->args[1]);
    this->stream << mbarrier_obj << ".expect_transaction(" << transaction_bytes
                 << ");\n";
  } else if (op->op.same_as(tl::mbarrier_wait_parity())) {
    ICHECK_EQ(op->args.size(), 2);
    this->PrintIndent();
    auto mbarrier_obj = print_mbarrier_obj(op->args[0]);
    auto phase = this->PrintExpr(op->args[1]);
    this->stream << mbarrier_obj << ".wait(" << phase << ");\n";
  } else if (op->op.same_as(tl::ptx_init_tensor_memory())) {
    print_extern_call_stmt("tl::tmem_allocate");
  } else if (op->op.same_as(tl::ptx_deallocate_tensor_memory())) {
    print_extern_call_stmt("tl::tmem_deallocate");
  } else if (op->op.same_as(tl::no_set_max_nreg())) {
    return;
  } else if (op->op.same_as(tl::tma_load())) {
    std::ostringstream ss;
    ICHECK_GE(op->args.size(), 2);
    auto eviction_policy =
        this->eviction_policy_names_
            [op->args[op->args.size() - 1].as<IntImmNode>()->value];
    // Simplify the code by using the default eviction policy
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load<tl::CacheHintSm90::" << eviction_policy << ">(";
    } else {
      ss << "tl::tma_load(";
    }
    auto desc = op->args[0];
    ss << this->PrintExpr(desc) << ", ";
    ss << print_mbarrier_obj(op->args[1]) << ", ";
    for (size_t i = 2; i < op->args.size() - 1; i++) {
      if (i > 2)
        ss << ", ";
      ss << this->PrintExpr(op->args[i]);
    }
    ss << ");\n";
    this->PrintIndent();
    this->stream << ss.str();
  } else if (op->op.same_as(tl::tma_load_im2col())) {
    std::stringstream ss;
    auto eviction_policy =
        this->eviction_policy_names_
            [op->args[op->args.size() - 1].as<IntImmNode>()->value];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_load_im2col<tl::CacheHintSm90::" << eviction_policy << ">";
    } else {
      ss << "tl::tma_load_im2col";
    }
    print_extern_call_stmt(ss.str(), 0, 1);
  } else if (op->op.same_as(tl::tma_store())) {
    std::stringstream ss;
    auto need_reduce = op->args[op->args.size() - 2].as<IntImmNode>()->value;
    if (need_reduce) {
      print_extern_call_stmt("tl::tma_store_add", 0, 2);
      return;
    }
    auto eviction_policy =
        this->eviction_policy_names_
            [op->args[op->args.size() - 1].as<IntImmNode>()->value];
    if (eviction_policy != "EVICT_NORMAL") {
      ss << "tl::tma_store<tl::CacheHintSm90::" << eviction_policy << ">";
    } else {
      ss << "tl::tma_store";
    }
    print_extern_call_stmt(ss.str(), 0, 2);
  } else if (op->op.same_as(tl::ptx_ldmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl::ptx_ldmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::ptx_stmatrix())) {
    int trans = Downcast<IntImm>(op->args[0])->value;
    int num = Downcast<IntImm>(op->args[1])->value;
    std::string func_name = "tl::ptx_stmatrix_x" + std::to_string(num);
    if (trans == 1)
      func_name += "_trans";
    print_extern_call_stmt(func_name, 2);
  } else if (op->op.same_as(tl::fence_proxy_async())) {
    print_extern_call_stmt("tl::fence_proxy_async");
  } else if (op->op.same_as(tl::tma_store_arrive())) {
    print_extern_call_stmt("tl::tma_store_arrive");
  } else if (op->op.same_as(tl::tma_store_wait())) {
    print_extern_call_stmt("tl::tma_store_wait<0>");
  } else if (op->op.same_as(tl::warpgroup_arrive())) {
    print_extern_call_stmt("tl::warpgroup_arrive");
  } else if (op->op.same_as(tl::warpgroup_commit_batch())) {
    print_extern_call_stmt("tl::warpgroup_commit_batch");
  } else if (op->op.same_as(tl::warpgroup_wait())) {
    this->PrintIndent();
    int num_mma = Downcast<IntImm>(op->args[0])->value;
    this->stream << "tl::warpgroup_wait<" << std::to_string(num_mma)
                 << ">();\n";
  } else if (op->op.same_as(tl::warpgroup_fence_operand())) {
    ICHECK_EQ(op->args.size(), 4U);
    std::string dtype = Downcast<StringImm>(op->args[0])->value;
    std::string data_ptr = this->PrintExpr(op->args[1]);
    std::string offset = this->PrintExpr(op->args[2]);
    std::string num_regs = this->PrintExpr(op->args[3]);
    auto dtype_enum = tl::codegen::ptx::DTypeFromString(dtype);
    std::string cast_type = "uint32_t";
    if (dtype_enum == tl::codegen::ptx::DataType::kFloat32 ||
        dtype_enum == tl::codegen::ptx::DataType::kTensorFloat32) {
      cast_type = "float";
    }
    this->PrintIndent();
    this->stream << "tl::warpgroup_fence_operand(reinterpret_cast<" << cast_type
                 << "*>(" << data_ptr << " + " << offset << "), " << num_regs
                 << ");\n";
  } else if (op->op.same_as(tl::set_max_nreg())) {
    this->PrintIndent();
    int nreg = Downcast<IntImm>(op->args[0])->value;
    int is_inc = Downcast<IntImm>(op->args[1])->value;
    std::string func_name =
        is_inc ? "tl::warpgroup_reg_alloc" : "tl::warpgroup_reg_dealloc";
    this->stream << func_name << "<" << std::to_string(nreg) << ">();\n";
  } else if (op->op.same_as(tl::wait_wgmma())) {
    this->PrintIndent();
    int num_mma = Downcast<IntImm>(op->args[0])->value;
    this->stream << "tl::wait_wgmma<" << std::to_string(num_mma) << ">();\n";
  } else if (op->op.same_as(tl::pack_b16())) {
    os << "__pack_half2(" << this->PrintExpr(op->args[0]) << ", "
       << this->PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::sync_grid())) {
    this->need_cooperative_groups_ = true;
    this->PrintIndent();
    this->stream << "cooperative_groups::this_grid().sync();\n";
  } else if (op->op.same_as(tl::loop_break())) {
    this->PrintIndent();
    this->stream << "break;\n";
  } else if (op->op.same_as(builtin::tvm_fill_fragment())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 6U);
    os << "nvcuda::wmma::fill_fragment(";
    this->PrintExpr(op->args[0], os);
    os << "[";
    this->PrintExpr(op->args[4], os);
    os << "], ";
    this->PrintExpr(op->args[5], os);
    os << ")";
  } else if (op->op.same_as(builtin::tvm_load_matrix_sync())) {
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
  } else if (op->op.same_as(builtin::tvm_store_matrix_sync())) {
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
      LOG(FATAL) << "Invalid parameters";
    }
    os << ")";
  } else if (op->op.same_as(builtin::tvm_mma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::mma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::tvm_bmma_sync())) {
    need_mma_h_ = true;
    ICHECK_EQ(op->args.size(), 8U);
    os << "nvcuda::wmma::bmma_sync(";
    for (int i = 0; i < 4; ++i) {
      this->PrintExpr(op->args[i * 2], os);
      os << "[";
      this->PrintExpr(op->args[i * 2 + 1], os);
      os << "]" << ((i < 3) ? ", " : ")");
    }
  } else if (op->op.same_as(builtin::ptx_mma())) {
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
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_mma_instruction_h_ = true;
    this->PrintIndent();
    std::string mma_call =
        "tl::mma_sync<(AType), (BType), (CType), (M), (N), (K), (TransA), "
        "(TransB)>(reinterpret_cast<(CRegType)*>((C_ptr) + (C_offset)), "
        "reinterpret_cast<const (ARegType)*>((A_ptr) + (A_offset)), "
        "reinterpret_cast<const (BRegType)*>((B_ptr) + (B_offset)));\n";
    tl::codegen::Replacer replacer;

    // TODO(lei): Type Workaround for TF32, should be removed when
    // we introduced tfloat32_t in the frontend.
    std::string AType = tl::codegen::ptx::DTypeEnumToString(dtype_a_enum);
    if (AType == "tl::DataType::kFloat32") {
      AType = "tl::DataType::kTensorFloat32";
    }
    std::string BType = tl::codegen::ptx::DTypeEnumToString(dtype_b_enum);
    if (BType == "tl::DataType::kFloat32") {
      BType = "tl::DataType::kTensorFloat32";
    }
    std::string ARegType = tl::codegen::GetMMARegisterType(dtype_a_enum);
    if (ARegType == "float") {
      ARegType = "uint32_t";
    }
    std::string BRegType = tl::codegen::GetMMARegisterType(dtype_b_enum);
    if (BRegType == "float") {
      BRegType = "uint32_t";
    }

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(TransA)", A_layout == "row" ? "false" : "true");
    replacer.register_rule("(TransB)", B_layout == "row" ? "false" : "true");
    replacer.register_rule("(ARegType)", ARegType);
    replacer.register_rule("(BRegType)", BRegType);
    replacer.register_rule("(CRegType)",
                           tl::codegen::GetMMARegisterType(dtype_c_enum));
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", a_bias);
    replacer.register_rule("(B_ptr)", b_ref);
    replacer.register_rule("(B_offset)", b_bias);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_bias);
    this->stream << replacer.rewrite(mma_call);
  } else if (op->op.same_as(tl::ptx_mma_sm70())) {
    // arg 0: shape: mXnXkX
    // arg 1: A layout: row/col
    // arg 2: B layout: row/col
    // arg 3: A precision: fp16
    // arg 4: B precision: fp16
    // arg 5: C precision: fp16, fp32
    // arg 6: A multiplicand
    // arg 7: A multiplicand index
    // arg 8: B multiplicand
    // arg 9: B multiplicand index
    // arg 10: C accumulator
    // arg 11: C accumulator index
    // arg 12: saturate
    ICHECK_EQ(op->args.size(), 12U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_bias = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_bias = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_bias = this->PrintExpr(op->args[11]);
    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_mma_sm70_instruction_h_ = true;
    this->PrintIndent();
    std::string mma_call =
        "tl::mma_sync_sm70<(AType), (BType), (CType), (M), (N), (K), (TransA), "
        "(TransB)>(reinterpret_cast<(CRegType)*>((C_ptr) + (C_offset)), "
        "reinterpret_cast<const (ARegType)*>((A_ptr) + (A_offset)), "
        "reinterpret_cast<const (BRegType)*>((B_ptr) + (B_offset)));\n";
    tl::codegen::Replacer replacer;

    replacer.register_rule("(AType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_a_enum));
    replacer.register_rule("(BType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_b_enum));
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(TransA)", A_layout == "row" ? "false" : "true");
    replacer.register_rule("(TransB)", B_layout == "row" ? "false" : "true");
    replacer.register_rule("(ARegType)",
                           tl::codegen::GetMMARegisterType(dtype_a_enum));
    replacer.register_rule("(BRegType)",
                           tl::codegen::GetMMARegisterType(dtype_b_enum));
    replacer.register_rule("(CRegType)",
                           tl::codegen::GetMMARegisterType(dtype_c_enum));
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", a_bias);
    replacer.register_rule("(B_ptr)", b_ref);
    replacer.register_rule("(B_offset)", b_bias);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_bias);
    this->stream << replacer.rewrite(mma_call);
  } else if (op->op.same_as(builtin::ptx_mma_sp())) {
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
    // arg 12: metadata
    // arg 13: metadata index
    // arg 14: sparse_selector
    // arg 15: saturate
    ICHECK_EQ(op->args.size(), 16U);
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    std::string A_layout = Downcast<StringImm>(op->args[1])->value;
    std::string B_layout = Downcast<StringImm>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_ref = this->PrintExpr(op->args[6]);
    std::string a_offset = this->PrintExpr(op->args[7]);
    std::string b_ref = this->PrintExpr(op->args[8]);
    std::string b_offset = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_offset = this->PrintExpr(op->args[11]);
    std::string metadata = this->PrintExpr(op->args[12]);
    std::string metadata_offset = this->PrintExpr(op->args[13]);
    std::string sparse_selector = this->PrintExpr(op->args[14]);
    bool saturate = Downcast<Bool>(op->args[15])->value;
    this->PrintIndent();
    std::string asm_code = PrintMMAAssembly(
        shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype, a_ref, a_offset,
        b_ref, b_offset, c_ref, c_offset, metadata, metadata_offset,
        sparse_selector, "", true, saturate);
    this->stream << asm_code;
  } else if (op->op.same_as(tl::ptx_wgmma_ss())) {
    // arg 0: dtype
    // arg 1: shape
    // arg 2: A_layout
    // arg 3: B_layout
    // arg 4: A_dtype
    // arg 5: B_dtype
    // arg 6: C_dtype
    // arg 7: multiplicand_a
    // arg 8: multiplicand_b
    // arg 9: accumulator
    // arg 10: saturate
    ICHECK_EQ(op->args.size(), 15U) << "ptx_wgmma_ss args is " << op->args;
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    bool a_is_k_major = Downcast<Bool>(op->args[1])->value;
    bool b_is_k_major = Downcast<Bool>(op->args[2])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[5])->value;
    std::string a_desc = this->PrintExpr(op->args[6]);
    std::string A_offset = this->PrintExpr(op->args[7]);
    std::string b_desc = this->PrintExpr(op->args[8]);
    std::string B_offset = this->PrintExpr(op->args[9]);
    std::string c_ref = this->PrintExpr(op->args[10]);
    std::string c_offset = this->PrintExpr(op->args[11]);
    std::string scale_out = this->PrintExpr(op->args[12]);
    bool scale_in_a = Downcast<Bool>(op->args[13])->value;
    bool scale_in_b = Downcast<Bool>(op->args[14])->value;

    const bool a_is_shared = true;
    this->PrintIndent();
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);
    need_wgmma_instruction_h_ = true;
    std::string wgmma_asm_code =
        "tl::wgmma_ss<(AType), (BType), (CType), (M), (N), (K), (tnspA), "
        "(tnspB), (scaleA), (scaleB)>(uint64_t((desc_a) + (A_offset)), "
        "uint64_t((desc_b) + (B_offset)), ((uint32_t*)((C))), (scale_out));\n";
    // replace patterns
    tl::codegen::Replacer replacer;

    std::string AType = tl::codegen::ptx::DTypeEnumToString(A_dtype);
    if (AType == "tl::DataType::kFloat32") {
      AType = "tl::DataType::kTensorFloat32";
    }
    std::string BType = tl::codegen::ptx::DTypeEnumToString(B_dtype);
    if (BType == "tl::DataType::kFloat32") {
      BType = "tl::DataType::kTensorFloat32";
    }

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(C_dtype));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(tnspA)", a_is_k_major ? "false" : "true");
    replacer.register_rule("(tnspB)", b_is_k_major ? "false" : "true");
    replacer.register_rule("(scaleA)", scale_in_a ? "1" : "-1");
    replacer.register_rule("(scaleB)", scale_in_b ? "1" : "-1");
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref + " + " + c_offset);
    replacer.register_rule("(scale_out)", scale_out);
    wgmma_asm_code = replacer.rewrite(wgmma_asm_code);
    this->stream << wgmma_asm_code;
  } else if (op->op.same_as(tl::ptx_wgmma_rs())) {
    // arg 0: shape
    // arg 1: B_layout
    // arg 2: A_dtype
    // arg 3: B_dtype
    // arg 4: C_dtype
    // arg 5: multiplicand_a
    // arg 6: multiplicand_a offset
    // arg 7: multiplicand_b descriptor
    // arg 8: multiplicand_b offset
    // arg 9: accumulator
    // arg 10: accumulator offset
    // arg 11: scale_out
    // arg 12: scale_in_a
    // arg 13: scale_in_b
    ICHECK_EQ(op->args.size(), 14U) << "ptx_wgmma_rs args is " << op->args;
    std::string shape = Downcast<StringImm>(op->args[0])->value;
    bool b_is_k_major = Downcast<Bool>(op->args[1])->value;
    std::string A_dtype = Downcast<StringImm>(op->args[2])->value;
    std::string B_dtype = Downcast<StringImm>(op->args[3])->value;
    std::string C_dtype = Downcast<StringImm>(op->args[4])->value;
    std::string a_ref = this->PrintExpr(op->args[5]);
    std::string A_offset = this->PrintExpr(op->args[6]);
    std::string b_desc = this->PrintExpr(op->args[7]);
    std::string B_offset = this->PrintExpr(op->args[8]);
    std::string c_ref = this->PrintExpr(op->args[9]);
    std::string c_offset = this->PrintExpr(op->args[10]);
    std::string scale_out = this->PrintExpr(op->args[11]);
    bool scale_in_a = Downcast<Bool>(op->args[12])->value;
    bool scale_in_b = Downcast<Bool>(op->args[13])->value;

    auto dtype_a_enum = tl::codegen::ptx::DTypeFromString(A_dtype);
    auto dtype_b_enum = tl::codegen::ptx::DTypeFromString(B_dtype);
    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);
    auto [m, n, k] = tl::codegen::ptx::ParseMMAShape(shape);

    need_wgmma_instruction_h_ = true;
    this->PrintIndent();
    std::string wgmma_call =
        "tl::wgmma_rs<(AType), (BType), (CType), (M), (N), (K), (tnspA), "
        "(tnspB), (scaleA), (scaleB)>(reinterpret_cast<const "
        "uint32_t*>((A_ptr) + (A_offset)), "
        "uint64_t((desc_b) + (B_offset)), "
        "reinterpret_cast<uint32_t*>((C_ptr) + (C_offset)), "
        "(scale_out));\n";

    tl::codegen::Replacer replacer;
    std::string AType = tl::codegen::ptx::DTypeEnumToString(A_dtype);
    if (AType == "tl::DataType::kFloat32") {
      AType = "tl::DataType::kTensorFloat32";
    }
    std::string BType = tl::codegen::ptx::DTypeEnumToString(B_dtype);
    if (BType == "tl::DataType::kFloat32") {
      BType = "tl::DataType::kTensorFloat32";
    }

    replacer.register_rule("(AType)", AType);
    replacer.register_rule("(BType)", BType);
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(M)", std::to_string(m));
    replacer.register_rule("(N)", std::to_string(n));
    replacer.register_rule("(K)", std::to_string(k));
    replacer.register_rule("(tnspA)", "false");
    replacer.register_rule("(tnspB)", b_is_k_major ? "false" : "true");
    replacer.register_rule("(scaleA)", scale_in_a ? "1" : "-1");
    replacer.register_rule("(scaleB)", scale_in_b ? "1" : "-1");
    replacer.register_rule("(A_ptr)", a_ref);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C_ptr)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(scale_out)", scale_out);
    wgmma_call = replacer.rewrite(wgmma_call);
    this->stream << wgmma_call;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ss())) {
    ICHECK_EQ(op->args.size(), 14U)
        << "ptx_tcgen05_mma_ss args is " << op->args;
    std::string C_dtype = Downcast<StringImm>(op->args[0])->value;
    std::string a_desc = this->PrintExpr(op->args[1]);
    std::string A_offset = this->PrintExpr(op->args[2]);
    std::string b_desc = this->PrintExpr(op->args[3]);
    std::string B_offset = this->PrintExpr(op->args[4]);
    std::string c_ref = this->PrintExpr(op->args[5]);
    std::string c_offset = this->PrintExpr(op->args[6]);
    PrimExpr desc_expr = op->args[7];
    std::string scale_out = this->PrintExpr(op->args[8]);
    std::string mask0 = this->PrintExpr(op->args[9]);
    std::string mask1 = this->PrintExpr(op->args[10]);
    std::string mask2 = this->PrintExpr(op->args[11]);
    std::string mask3 = this->PrintExpr(op->args[12]);
    bool enable_ws = Downcast<Bool>(op->args[13])->value;

    auto dtype_c_enum = tl::codegen::ptx::DTypeFromString(C_dtype);

    need_tcgen05mma_instruction_h_ = true;
    this->PrintIndent();
    std::string tcgen05_call =
        "tl::(tcgen05_name)<(CType)>(uint64_t((desc_a) + (A_offset)), "
        "uint64_t((desc_b) + (B_offset)), (*reinterpret_cast<uint32_t*>((C))) "
        "+ (C_offset), "
        "(scale_out), static_cast<uint32_t>((desc_val)), (mask0), (mask1), "
        "(mask2), (mask3));\n";
    tl::codegen::Replacer replacer;
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_c_enum));
    replacer.register_rule("(desc_a)", a_desc);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(tcgen05_name)",
                           enable_ws ? "tcgen05mma_ws_ss" : "tcgen05mma_ss");
    replacer.register_rule("(scale_out)", scale_out);
    replacer.register_rule("(desc_val)", this->PrintExpr(desc_expr));
    replacer.register_rule("(mask0)", mask0);
    replacer.register_rule("(mask1)", mask1);
    replacer.register_rule("(mask2)", mask2);
    replacer.register_rule("(mask3)", mask3);
    tcgen05_call = replacer.rewrite(tcgen05_call);
    this->stream << tcgen05_call;
  } else if (op->op.same_as(tl::ptx_tcgen05_mma_ts())) {
    // TS: A from TMEM, B from SMEM (desc)
    ICHECK_EQ(op->args.size(), 13U)
        << "ptx_tcgen05_mma_ts args is " << op->args;
    std::string kind_dtype = Downcast<StringImm>(op->args[0])->value;
    std::string a_ref = this->PrintExpr(op->args[1]);
    std::string A_offset = this->PrintExpr(op->args[2]);
    std::string b_desc = this->PrintExpr(op->args[3]);
    std::string B_offset = this->PrintExpr(op->args[4]);
    std::string c_ref = this->PrintExpr(op->args[5]);
    std::string c_offset = this->PrintExpr(op->args[6]);
    PrimExpr desc_expr = op->args[7];
    std::string scale_out = this->PrintExpr(op->args[8]);
    std::string mask0 = this->PrintExpr(op->args[9]);
    std::string mask1 = this->PrintExpr(op->args[10]);
    std::string mask2 = this->PrintExpr(op->args[11]);
    std::string mask3 = this->PrintExpr(op->args[12]);

    auto dtype_enum = tl::codegen::ptx::DTypeFromString(kind_dtype);

    need_tcgen05mma_instruction_h_ = true;
    this->PrintIndent();
    std::string tcgen05_call =
        "tl::tcgen05mma_ts<(CType)>( (*reinterpret_cast<uint32_t*>((A))) + "
        "(A_offset), "
        "uint64_t((desc_b) + (B_offset)), (*reinterpret_cast<uint32_t*>((C))) "
        "+ (C_offset), "
        "(scale_out), static_cast<uint32_t>((desc_val)), (mask0), (mask1), "
        "(mask2), (mask3));\n";
    tl::codegen::Replacer replacer;
    replacer.register_rule("(CType)",
                           tl::codegen::ptx::DTypeEnumToString(dtype_enum));
    replacer.register_rule("(A)", a_ref);
    replacer.register_rule("(A_offset)", A_offset);
    replacer.register_rule("(desc_b)", b_desc);
    replacer.register_rule("(B_offset)", B_offset);
    replacer.register_rule("(C)", c_ref);
    replacer.register_rule("(C_offset)", c_offset);
    replacer.register_rule("(scale_out)", scale_out);
    replacer.register_rule("(desc_val)", this->PrintExpr(desc_expr));
    replacer.register_rule("(mask0)", mask0);
    replacer.register_rule("(mask1)", mask1);
    replacer.register_rule("(mask2)", mask2);
    replacer.register_rule("(mask3)", mask3);
    tcgen05_call = replacer.rewrite(tcgen05_call);
    this->stream << tcgen05_call;
  } else if (op->op.same_as(tl::tcgen05_mma_arrive())) {
    ICHECK_EQ(op->args.size(), 1U) << "tcgen05_mma_arrive expects 1 argument";
    need_tcgen05_common_h_ = true;
    this->PrintIndent();
    this->stream << "tl::tcgen05_mma_arrive(" << this->PrintExpr(op->args[0])
                 << ");\n";
  } else if (op->op.same_as(builtin::ptx_ldmatrix())) {
    // arg 0: whether the matrix is loaded in column major format or not.
    // arg 1: number of matrices to load.
    // arg 2: The data type in the matrix, .b16 is the only accepted data type.
    // arg 3: pointer to local buffer.
    // arg 4: The offset of the element to store in the local buffer.
    // arg 5: pointer to the shared memory buffer to load.
    // arg 6: The offset of the start element of the row to load in shared
    // memory.
    ICHECK_EQ(op->args.size(), 7U);
    bool trans = Downcast<Bool>(op->args[0])->value;
    int num = Downcast<Integer>(op->args[1])->value;
    std::string type = Downcast<StringImm>(op->args[2])->value;
    std::string local_ptr = this->PrintExpr(op->args[3]);
    std::string local_elem_offset = this->PrintExpr(op->args[4]);
    std::string smem_ptr = this->PrintExpr(op->args[5]);
    if (trans && op->dtype.bits() == 8) {
      // Since ldmatrix assumes that a matrix element is 16 bit, it cannot
      // properly transpose an int8 matrix.
      std::string smem_stride = this->PrintExpr(op->args[6]);
      ICHECK(num == 4);
      os << "for (int i = 0; i < 16; ++i) {\n";
      os << local_ptr << "[" + local_elem_offset + " + i] = " << smem_ptr
         << "[(i % 8) / 4 * " + smem_stride +
                " * 16 + (threadIdx.x % 4) * 4 * " + smem_stride +
                "+ (i % 4) * " + smem_stride +
                " + threadIdx.x / 4 + (i / 8) * 8];\n";
      os << "}\n";
    } else {
      std::string smem_elem_offset = this->PrintExpr(op->args[6]);
      std::string func_name = "tl::ptx_ldmatrix_x" + std::to_string(num);
      if (trans == 1)
        func_name += "_trans";
      this->PrintIndent();
      this->stream << func_name << "(" << smem_ptr << " + " << smem_elem_offset
                   << ", " << local_ptr << " + " << local_elem_offset << ");\n";
    }
  } else if (op->op.same_as(builtin::mma_store())) {
    int m = Downcast<Integer>(op->args[0])->value;
    int n = Downcast<Integer>(op->args[1])->value;
    std::string dst = this->PrintExpr(op->args[2]);
    std::string src = this->PrintExpr(op->args[3]);
    std::string src_offset = this->PrintExpr(op->args[4]);
    PrimExpr stride = op->args[5];

    ICHECK(m == 16 && n == 16)
        << "Only m == 16 && n == 16 case supported for now";

    // Each thread in a warp holds a certain number of elements of an MMA
    // output. For example, if we compute a 16x16 tile using MMA, each thread
    // holds 8 elements in its registers. So conceptually, a warp memory is
    // organized as a 32x8 block. A map from a 16x16 tile to a 32x8 block of
    // memory is specified by the index map below.

    // To store the 32x8 output back to a 16x16 tile in shared or global memory,
    // we invert this map to determine the output location for each 8 element.

    const auto index_map_func = ffi::Function::GetGlobal(
        "tir.index_map.shared_16x16_to_mma_32x8_layout");

    IndexMap index_map;
    if (!index_map_func) {
      Var i, j;

      // The index map is defined as follows:
      index_map = IndexMap(
          {i, j}, {4 * FloorMod(i, 8) + FloorDiv(FloorMod(j, 8), 2),
                   4 * FloorDiv(j, 8) + FloorDiv(i, 8) * 2 + FloorMod(j, 2)});
    } else {
      index_map = IndexMap::FromFunc(2, *index_map_func);
    }

    arith::Analyzer analyzer;
    auto inverse_index_map =
        index_map.Inverse({Range(0, m), Range(0, n)}, &analyzer);
    auto indices_16x16 = inverse_index_map->final_indices;

    // "//" and "%" in the index map are translated to FloorDiv/Mod, but the
    // plain Div/Mod are fine. FloorDiv/Mod are supposed to be lowered before
    // they reach codegen, so manually replace them to the plain ones here.
    class LowerFloorDivMod : public ExprMutator {
    public:
      PrimExpr VisitExpr_(const FloorDivNode *op) {
        return tir::Div(this->VisitExpr(op->a), this->VisitExpr(op->b));
      }
      PrimExpr VisitExpr_(const FloorModNode *op) {
        return tir::Mod(this->VisitExpr(op->a), this->VisitExpr(op->b));
      }
    };

    auto dst_ind =
        LowerFloorDivMod()(indices_16x16[0] * stride + indices_16x16[1]);

    var_idmap_[inverse_index_map->initial_indices[0].get()] = "threadIdx.x";
    var_idmap_[inverse_index_map->initial_indices[1].get()] = "local_id";
    if (op->dtype.bits() == 16) {
      os << "for (int local_id = 0; local_id < 8; local_id+=2) {\n";
      os << "*((uint *)&" << dst << "[" + this->PrintExpr(dst_ind) + "])"
         << " = "
         << "*((uint *)&" << src << "[" << src_offset << " + local_id]);\n";
      os << "}\n";
    } else {
      os << "for (int local_id = 0; local_id < 8; ++local_id) {\n";
      os << dst << "[" + this->PrintExpr(dst_ind) + "]" << " = " << src << "["
         << src_offset << " + local_id];\n";
      os << "}\n";
    }

  } else if (op->op.same_as(builtin::mma_fill())) {
    std::string num_elem = this->PrintExpr(op->args[0]);
    std::string dst = this->PrintExpr(op->args[1]);
    std::string dst_offset = this->PrintExpr(op->args[2]);

    os << "for (int i = 0; i < " << num_elem << "; ++i) {\n";
    os << dst << "[" << dst_offset << " + i] = 0.0;";
    os << "}\n";
  } else if (op->op.same_as(builtin::ptx_cp_async())) {
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    need_cast_smem_ptr_to_int_ = true;
    // use size of argument list to indicate whether or not to use predicated
    // cp.async
    if (op->args.size() == 5) {
      this->stream << PrintCpAsyncAssembly(dst, dst_offset, src, src_offset,
                                           size);
    } else {
      this->stream << PrintPredicatedCpAsyncAssembly(
          dst, dst_offset, src, src_offset, size, this->PrintExpr(op->args[5]));
    }
  } else if (op->op.same_as(builtin::ptx_cp_async_bulk())) {
    need_cast_smem_ptr_to_int_ = true;
    std::string dst = this->PrintExpr(op->args[0]);
    std::string dst_offset = this->PrintExpr(op->args[1]);
    std::string src = this->PrintExpr(op->args[2]);
    std::string src_offset = this->PrintExpr(op->args[3]);
    std::string size = this->PrintExpr(op->args[4]);
    int barrier_id = Downcast<IntImm>(op->args[5])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier =
        barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintCpAsyncBulkAsm(dst, dst_offset, src, src_offset, size,
                                        barrier);
  } else if (op->op.same_as(builtin::ptx_commit_group())) {
    this->stream << "__asm__ __volatile__(\"cp.async.commit_group;\");\n\n";
  } else if (op->op.same_as(builtin::ptx_wait_group())) {
    int n = Downcast<IntImm>(op->args[0])->value;
    this->stream << "__asm__ __volatile__(\"cp.async.wait_group " << n
                 << ";\");\n\n";
  } else if (op->op.same_as(builtin::ptx_init_barrier_thread_count())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier =
        barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    std::string thread_count = this->PrintExpr(op->args[1]);
    this->stream << PrintInitBarrierThreadCountAsm(barrier, thread_count);
  } else if (op->op.same_as(builtin::ptx_arrive_barrier())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier =
        barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintArriveBarrierAsm(barrier);
  } else if (op->op.same_as(builtin::ptx_arrive_barrier_expect_tx())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier =
        barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    std::string byte_count = this->PrintExpr(op->args[1]);
    this->stream << PrintArriveBarrierExpectTxAsm(barrier, byte_count);
  } else if (op->op.same_as(builtin::ptx_wait_barrier())) {
    need_cast_smem_ptr_to_int_ = true;
    int barrier_id = Downcast<IntImm>(op->args[0])->value;
    CHECK(barrier_id < barrier_count_);
    std::string barrier =
        barrier_name_ + "[" + std::to_string(barrier_id) + "]";
    this->stream << PrintWaitBarrierAsm(barrier);
  } else if (op->op.same_as(builtin::ptx_ldg32())) {
    /*
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        // " @p ld.global.nc.f32 %0, [%1];}\n"t
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
        : "=f"(reg)
        : "l"(addr), "r"((int)guard)
    );
    */

    // get local
    std::string reg = this->PrintExpr(op->args[0]);
    // get guard
    std::string guard = this->PrintExpr(op->args[1]);
    const BufferLoadNode *addr_buffer = op->args[2].as<BufferLoadNode>();
    std::string global_addr = this->PrintExpr(addr_buffer->indices[0]);
    std::string global_buffer = this->PrintExpr(addr_buffer->buffer->data);
    std::string local_addr = this->PrintExpr(op->args[3]);
    this->stream << "asm volatile (\n";
    this->stream << "\"{.reg .pred p;\\n\"\n";
    this->stream << "\" setp.ne.b32 p, %2, 0;\\n\"\n";
    this->stream << "\" @!p mov.b32 %0, 0;\\n\"\n";
    this->stream << "\" @p ld.global.nc.f32 %0, [%1];}\\n\"\n";
    // stream << "\" @p ld.global.nc.L2::128B.f32 %0, [%1];}\\n\"\n" ;
    stream << ": \"=f\"(" << reg << "[" << local_addr << "]"
           << ")\n";
    stream << ": \"l\"((void*)(" << global_buffer << "+" << global_addr
           << ")), \"r\"((int)" << guard << ")\n";
    stream << ");\n";
  } else if (op->op.same_as(tl::__ldg())) {
    // Explicit read-only cached load. Preferred form: __ldg(BufferLoad(...)).
    // Fallback form: __ldg(buffer, index)
    const BufferLoadNode *bl = nullptr;
    if (!op->args.empty()) {
      bl = op->args[0].as<BufferLoadNode>();
    }
    if (bl == nullptr) {
      LOG(FATAL) << "T.__ldg expects a BufferLoad as the first argument.";
    }
    const BufferNode *buffer = bl->buffer.get();
    ICHECK_EQ(bl->indices.size(), 1)
        << "T.__ldg currently supports flattened 1D buffer accesses.";
    PrimExpr base = bl->indices[0];
    // Emit __ldg(&buffer_ref)
    auto buffer_ref = this->GetBufferRef(op->dtype, buffer, base);
    os << "__ldg(&(" << buffer_ref << "))";
  } else if (op->op.same_as(builtin::reinterpret())) {
    DataType tgt_dtype = op->dtype;
    DataType src_dtype = op->args[0]->dtype;
    PrimExpr value = op->args[0];

    // Handle float4_e2m1fn reinterpret
    if (!src_dtype.is_float4_e2m1fn() && !tgt_dtype.is_float4_e2m1fn()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    if (src_dtype == tgt_dtype || tgt_dtype.lanes() * tgt_dtype.bits() ==
                                      src_dtype.lanes() * src_dtype.bits()) {
      return CodeGenC::VisitExpr_(op, os);
    }
    CHECK_EQ(tgt_dtype.lanes(), src_dtype.lanes())
        << "E2M1 float4 reinterpret expects source and target to have the same "
           "number of lanes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;
    CHECK_EQ(tgt_dtype.bytes(), src_dtype.bytes())
        << "E2M1 float4 reinterpret expects source and target to have the same "
           "number of bytes. "
        << "Source dtype: " << src_dtype << ", Target dtype: " << tgt_dtype;

    int lanes = tgt_dtype.lanes();

    int ssa_scope = BeginScope();
    if (lanes == 1) {
      // The case of lane=1 is same as the normal reinterpret,
      // except that we allow the src and dst dtype to have different number of
      // bits.
      std::string rhs = SSAGetID(PrintExpr(value), src_dtype);
      os << "(*(";
      this->PrintType(tgt_dtype, os);
      os << " *)(&(" << rhs << ")))";
    } else if (lanes == 2) {
      if (tgt_dtype.is_float4_e2m1fn()) {
        // We view the source as an uint16, and then extract bits of two fp4
        // numbers, and finally reinterpret the result as fp4x2.
        value =
            tir::Call(DataType::UInt(16), tir::builtin::reinterpret(), {value});
        tir::Var temp_var("temp_var", DataType::UInt(16));
        value =
            tir::Let(temp_var, value,
                     tir::Cast(DataType::UInt(8),
                               (temp_var & IntImm(DataType::UInt(16), 0xF)) |
                                   ((temp_var >> 4) &
                                    IntImm(DataType::UInt(16), 0xF0))));
      } else {
        value = tir::Cast(
            DataType::UInt(16),
            tir::Call(DataType::UInt(8), tir::builtin::reinterpret(), {value}));
        tir::Var temp_var("temp_var", DataType::UInt(16));
        value =
            tir::Let(temp_var, value,
                     (temp_var & IntImm(DataType::UInt(16), 0xF)) |
                         ((temp_var & IntImm(DataType::UInt(16), 0xF0)) << 4));
      }
      os << PrintExpr(
          tir::Call(tgt_dtype, tir::builtin::reinterpret(), {value}));
    } else if (lanes == 4) {
      if (tgt_dtype.is_float4_e2m1fn()) {
        // We view the source as an uint32, and then extract bits of four fp4
        // numbers, and finally reinterpret the result as fp4x4.
        value =
            tir::Call(DataType::UInt(32), tir::builtin::reinterpret(), {value});
        tir::Var temp_var("temp_var", DataType::UInt(32));
        value = tir::Let(
            temp_var, value,
            tir::Cast(
                DataType::UInt(16),
                (temp_var & IntImm(DataType::UInt(32), 0xF)) |
                    ((temp_var >> 4) & IntImm(DataType::UInt(32), 0xF0)) |
                    ((temp_var >> 8) & IntImm(DataType::UInt(32), 0xF00)) |
                    ((temp_var >> 12) & IntImm(DataType::UInt(32), 0xF000))));
      } else {
        value = tir::Cast(DataType::UInt(32),
                          tir::Call(DataType::UInt(16),
                                    tir::builtin::reinterpret(), {value}));
        tir::Var temp_var("temp_var", DataType::UInt(32));
        value = tir::Let(
            temp_var, value,
            (temp_var & IntImm(DataType::UInt(32), 0xF)) |
                ((temp_var & IntImm(DataType::UInt(32), 0xF0)) << 4) |
                ((temp_var & IntImm(DataType::UInt(32), 0xF00)) << 8) |
                ((temp_var & IntImm(DataType::UInt(32), 0xF000)) << 12));
      }
      os << PrintExpr(
          tir::Call(tgt_dtype, tir::builtin::reinterpret(), {value}));
    } else {
      LOG(FATAL) << "Invalid number of lanes for float4_e2m1fn reinterpret: "
                 << lanes;
    }
    EndScope(ssa_scope);
  } else if (op->op.same_as(builtin::thread_return())) {
    os << "return";
  } else if (op->op.same_as(tl::tl_gemm())) {
    ICHECK(op->args.size() == 4) << "tl_gemm expects 4 arguments <op_instance, "
                                    "A_ptr, B_ptr, C_ptr>, but got "
                                 << op->args.size();
    auto op_instance = Downcast<StringImm>(op->args[0]);
    this->PrintCallExtern(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                          op_instance->value, op->args, true, os);
  } else if (op->op.same_as(tl::tl_gemm_sp())) {
    ICHECK(op->args.size() == 5)
        << "tl_gemm_sp expects 5 arguments <op_instance, A_ptr, B_ptr, C_ptr, "
           "E_ptr>, but got "
        << op->args.size();
    auto op_instance = Downcast<StringImm>(op->args[0]);
    enable_sparse_gemm_ = true;
    this->PrintCallExtern(GetType(tvm::ffi::GetRef<PrimExpr>(op)),
                          op_instance->value, op->args, true, os);
  } else if (op->op.same_as(tl::get_lane_idx())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_lane_idx expects at most one argument <warp_size>.";
    os << "tl::get_lane_idx(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx_sync())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_warp_idx_sync expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx_sync(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_idx())) {
    ICHECK_LE(op->args.size(), 1)
        << "tl.get_warp_idx expects at most one argument <warp_size>.";
    os << "tl::get_warp_idx(";
    if (!op->args.empty()) {
      os << PrintExpr(op->args[0]);
    }
    os << ")";
  } else if (op->op.same_as(tl::get_warp_group_idx())) {
    ICHECK_LE(op->args.size(), 2)
        << "tl.get_warp_group_idx expects <warp_size, warps_per_group>.";
    os << "tl::get_warp_group_idx(";
    for (size_t i = 0; i < op->args.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << PrintExpr(op->args[i]);
    }
    os << ")";
  } else if (op->op.same_as(tl::tl_shuffle_elect())) {
    os << "tl::tl_shuffle_elect<" << PrintExpr(op->args[0]) << ">()";
  } else if (op->op.same_as(tl::initialize_wgmma_descriptor())) {
    ICHECK(op->args.size() == 5)
        << "tl_initialize_wgmma_descriptor expects 5 arguments but got "
        << op->args.size();
    auto descriptor = op->args[0];
    auto start_address = op->args[1];
    auto layout_type = op->args[2];
    auto leading_byte_offset = op->args[3];
    auto stride_byte_offset = op->args[4];
    os << "tl::initialize_wgmma_descriptor<" << PrintExpr(layout_type) << ", "
       << PrintExpr(leading_byte_offset) << ", "
       << PrintExpr(stride_byte_offset) << ">(" << PrintExpr(descriptor) << ", "
       << PrintExpr(start_address) << ")";
  } else if (op->op.same_as(tl::initialize_tcgen05_descriptor())) {
    ICHECK(op->args.size() == 7)
        << "tl_initialize_tcgen05_descriptor expects 7 arguments but got "
        << op->args.size();
    auto descriptor = op->args[0];
    auto start_address = op->args[1];
    auto leading_byte_offset = op->args[2];
    auto stride_byte_offset = op->args[3];
    auto base_offset = op->args[4];
    auto leading_abs = op->args[5];
    auto swizzle_mode = op->args[6];
    os << "tl::initialize_tcgen05_descriptor(" << PrintExpr(descriptor) << ", "
       << PrintExpr(start_address) << ", " << PrintExpr(leading_byte_offset)
       << ", " << PrintExpr(stride_byte_offset) << ", "
       << PrintExpr(base_offset) << ", " << PrintExpr(leading_abs) << ", "
       << PrintExpr(swizzle_mode) << ")";
  } else if (op->op.same_as(tl::increase_descriptor_offset())) {
    ICHECK(op->args.size() == 2)
        << "tl_increase_descriptor_offset expects 2 arguments but got "
        << op->args.size();
    auto descriptor = op->args[0];
    auto offset = op->args[1];
    os << "tl::increase_descriptor_offset<int>(" << PrintExpr(descriptor)
       << ", " << PrintExpr(offset) << ")";
  } else if (op->op.same_as(tl::__exp())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "exp");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__exp10())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "exp10");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__log())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "log");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__log2())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "log2");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__log10())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "log10");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__tan())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "tan");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__cos())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "cos");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::__sin())) {
    CUDAFastMath math_func;
    std::string func_name = math_func(op->dtype, "sin");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_add())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fadd", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ieee_sub())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fsub", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ieee_mul())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fmul", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::ieee_fmaf())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[3])->value;
    std::string func_name = math_func(op->dtype, "fmaf", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(tl::ieee_frcp())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[1])->value;
    std::string func_name = math_func(op->dtype, "frcp", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_fsqrt())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[1])->value;
    std::string func_name = math_func(op->dtype, "fsqrt", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_frsqrt())) {
    CUDAIEEEMath math_func;
    std::string func_name = math_func(op->dtype, "frsqrt", "rn");
    os << func_name << "(" << PrintExpr(op->args[0]) << ")";
  } else if (op->op.same_as(tl::ieee_fdiv())) {
    CUDAIEEEMath math_func;
    std::string rounding_mode = Downcast<StringImm>(op->args[2])->value;
    std::string func_name = math_func(op->dtype, "fdiv", rounding_mode);
    os << func_name << "(" << PrintExpr(op->args[0]) << ", "
       << PrintExpr(op->args[1]) << ")";
  } else if (op->op.same_as(tl::rng_init())) {
    this->need_curand_kernel_h_ = true;
    this->curand_philox_state = name_supply_->FreshName("__philox_state");
    this->PrintIndent();
    this->stream << "curandStatePhilox4_32_10_t " << this->curand_philox_state
                 << ";\n";
    this->PrintIndent();
    this->stream << "curand_init(" << PrintExpr(op->args[0]) << ", "
                 << PrintExpr(op->args[1]) << ", " << PrintExpr(op->args[2])
                 << ", &" << this->curand_philox_state << ");\n";
    // Store state_var for later use by rng_rand
  } else if (op->op.same_as(tl::rng_rand())) {
    this->need_curand_kernel_h_ = true;
    os << "curand(&" << this->curand_philox_state << ")";
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
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangCUDA::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == tir::attr::fragment_shape) {
    const VarNode *buffer = op->node.as<VarNode>();
    const StringImmNode *shape_str = op->value.as<StringImmNode>();
    fragment_shapes[buffer] = shape_str->value;
  } else if (op->attr_key == tir::attr::fragment_layout) {
    const VarNode *buffer = op->node.as<VarNode>();
    const StringImmNode *layout_str = op->value.as<StringImmNode>();
    fragment_layouts[buffer] = layout_str->value;
  } else if (op->attr_key == tir::attr::async_commit_queue_scope) {
    const IntImmNode *queue_id = op->value.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    this->VisitStmt(op->body);
    auto commit_group = Call(DataType::Void(), builtin::ptx_commit_group(), {});
    this->VisitExpr(commit_group, this->stream);
    return;
  } else if (op->attr_key == tir::attr::async_wait_queue_scope) {
    auto wait_attrs = GetAsyncWaitAttributes(op);
    auto queue_id = wait_attrs.first.as<IntImmNode>();
    ICHECK(queue_id && queue_id->value == 0)
        << "For CUDA, the index of an async queue must be 0.";
    auto wait_cnt = wait_attrs.second;
    auto wait_group =
        Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt});
    this->VisitExpr(wait_group, this->stream);
    auto inner = op->body.as<AttrStmtNode>();
    ICHECK(inner);
    this->VisitStmt(inner->body);
    return;
  } else if (op->attr_key == "threadblock_swizzle_pattern") {
    this->PrintIndent();
    const StringImmNode *pattern = op->value.as<StringImmNode>();
    ICHECK(pattern);
    this->stream << "const dim3 blockIdx = " << pattern->value << "();\n";
    this->VisitStmt(op->body);
    return;
  } else if (op->attr_key == "pragma_unroll_factor") {
    const IntImmNode *factor = op->value.as<IntImmNode>();
    ICHECK(factor);
    unroll_factor[op->node.as<VarNode>()] = Downcast<IntImm>(factor);
  }

  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangCUDA::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());
  this->PrintIndent();
  std::string scope = GetPtrStorageScope(op->buffer_var);
  const VarNode *buffer = op->buffer_var.as<VarNode>();
  if (scope.find("wmma.") == 0) {
    if (scope == "wmma.matrix_a" || scope == "wmma.matrix_b") {
      ICHECK(op->dtype == DataType::Float(16) ||
             op->dtype == DataType::Int(8) || op->dtype == DataType::UInt(8) ||
             op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
             op->dtype == DataType::Int(1) || op->dtype == DataType::BFloat(16))
          << "Matrix_a and matrix_b only support half or char or unsigned char "
          << "or uint4 or int4 or int1 type for now";
    } else {
      ICHECK(op->dtype == DataType::Float(16) ||
             op->dtype == DataType::Float(32) || op->dtype == DataType::Int(32))
          << "Accumulator only support half, float and int type for now";
    }
    PrintWmmaScope(scope, op->dtype, buffer, stream);
  } else if (scope == "local.descriptor.wgmma") {
    stream << "tl::GmmaDescriptor " << vid << ";\n";
  } else if (scope == "local.descriptor.tcgen05_smem") {
    stream << "tl::Tcgen05SMemDescriptor " << vid << ";\n";
  } else if (scope == "local.descriptor.tcgen05_instr") {
    stream << "tl::Tcgen05InstrDescriptor " << vid << ";\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
  }

  if (scope == "shared.dyn") {
    stream << ' ' << vid << "[];\n";
  } else {
    size_t constant_size = op->ConstantAllocationSize();
    ICHECK_GT(constant_size, 0)
        << "Can only handle constant size stack allocation for now, but get "
        << constant_size << " for " << op->buffer_var->name_hint;
    if (scope.find("wmma.") == 0) {
      constant_size = GetWmmaFragmentSize(scope, buffer, constant_size);
    }
    if ((op->dtype == DataType::Int(4) || op->dtype == DataType::UInt(4) ||
         op->dtype == DataType::Int(1)) &&
        scope == "shared") {
      constant_size = constant_size / (32 / op->dtype.bits());
    }
    if (scope == "shared") {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    } else if (scope == "shared.barrier") {
      auto v_id_mem = vid + "_mem";
      stream << ' ' << v_id_mem << "[" << constant_size << "];\n";
      PrintIndent();
      stream << "auto " << vid << " = reinterpret_cast<" << mbarrier_dtype_
             << "*>(" << v_id_mem << ");\n";
    } else if (scope == "local") {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    } else if (scope == "local.var") {
      PrimExpr init = tir::make_const(op->dtype, 0);
      auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
      if (init_it != op->annotations.end()) {
        PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
        if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
          user_init = tir::Cast(op->dtype, user_init);
        }
        init = user_init;
      }
      stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
    } else if (scope.find("local.descriptor") != 0) {
      ICHECK(false) << "Unsupported scope: " << scope;
    }
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenTileLangCUDA::VisitStmt_(const EvaluateNode *op) {
  if (is_const_int(op->value))
    return;
  const CallNode *call = op->value.as<CallNode>();
  if (call && call->op.same_as(builtin::tvm_global_barrier_kinit())) {
    PrintIndent();
    stream << "__shared__ unsigned " << vid_global_barrier_expect_ << ";\n";
    PrintIndent();
    stream << "if (threadIdx.x == 0) {\n";
    PrintIndent();
    stream << "  " << vid_global_barrier_expect_ << " = 0;\n";
    PrintIndent();
    stream << "}\n";
  }
  if (call && (call->op.same_as(tvm::tl::device_assert()))) {
    std::string cond = PrintExpr(call->args[0]);
    this->PrintIndent();
    stream << "device_assert(" << cond << ");\n";
  } else if (call && call->op.same_as(tvm::tl::device_assert_with_msg())) {
    std::string cond = PrintExpr(call->args[0]);
    std::string msg_expr = PrintExpr(call->args[1]);
    this->PrintIndent();
    stream << "device_assert_with_msg(" << cond << ", " << msg_expr << ");\n";
  } else {
    CodeGenC::VisitStmt_(op);
  }
}

void CodeGenTileLangCUDA::VisitExpr_(const RampNode *op, std::ostream &os) {
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  CHECK_LE(lanes, 4) << "Translate Ramp Node " << tvm::ffi::GetRef<Ramp>(op)
                     << " with " << lanes << " lanes is not allowed.";
  os << "(make_";
  PrintType(op->dtype, os);
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr(op->base) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << "))";
}

void CodeGenTileLangCUDA::VisitExpr_(const BufferLoadNode *op,
                                     std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1)
      << "Load from non-flat memory not supported.";
  ICHECK(!op->predicate.defined())
      << "Predicated buffer load is not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;

  int lanes = op->dtype.lanes();
  // declare type.
  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string ref = GetBufferRef(op->dtype, op->buffer.get(), index);
    HandleVolatileLoads(ref, op, os);
  } else {
    bool can_vector_load = false;
    arith::PVar<PrimExpr> base;
    int ramp_lanes = value_dtype.lanes() / element_dtype.lanes();
    if (arith::ramp(base, 1, ramp_lanes).Match(index)) {
      const RampNode *ramp = index.as<RampNode>();
      ICHECK(ramp);
      can_vector_load = true;
      // arith::ModularSet me = arith::Analyzer().modular_set(ramp->base);
      // The condition: {k * coeff + base} divisible by the alignment for any k
      // if (me->coeff % op->dtype.lanes() == 0 && me->base % op->dtype.lanes()
      // == 0) {
      //   can_vector_load = true;
      // }
    }

    if (can_vector_load) {
      std::string ref = GetVecLoad(op->dtype, op->buffer.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(index), index.dtype());
      std::string vid = GetVarID(buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenTileLangCUDA::VisitStmt_(const BufferStoreNode *op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";
  ICHECK(!op->predicate.defined())
      << "Predicated buffer store is not supported.";

  DataType value_dtype = op->value.dtype();
  DataType element_dtype = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];
  Var buffer_var = op->buffer->data;

  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string value = this->PrintExpr(op->value);
    std::string ref =
        this->GetBufferRef(value_dtype, op->buffer.get(), index_expr);
    this->PrintIndent();
    stream << ref << " = " << value << ";\n";
  } else {
    arith::PVar<PrimExpr> base;
    int ramp_lanes = value_dtype.lanes() / element_dtype.lanes();
    if (arith::ramp(base, 1, ramp_lanes).Match(index_expr)) {
      std::string value = this->PrintExpr(op->value);
      this->PrintVecStore(op->buffer.get(), value_dtype, base.Eval(), value);
    } else {
      // The assignment below introduces side-effect, and the resulting value
      // cannot be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements separately
      std::string index = SSAGetID(PrintExpr(index_expr), index_expr.dtype());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
      std::string vid = GetVarID(buffer_var.get());
      for (int i = 0; i < value_dtype.lanes(); ++i) {
        this->PrintIndent();
        DataType elem_type = value_dtype.element_of();
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          stream << "((";
          if (buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          stream << vid;
        }
        stream << '[';
        PrintVecElemLoad(index, index_expr.dtype(), i, stream);
        stream << "] = ";
        PrintVecElemLoad(value, op->value.dtype(), i, stream);
        stream << ";\n";
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenTileLangCUDA::VisitExpr_(const BroadcastNode *op,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = static_cast<int>(Downcast<IntImm>(op->lanes)->value);
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() == 8) {
    if (lanes == 4) {
      // make_int8x4
      const int64_t *p = as_const_int(op->value);
      ICHECK(p);
      int64_t v = *p & 0xFF;
      v = (v << 24) | (v << 16) | (v << 8) | v;
      if (op->dtype.is_uint()) {
        os << "(uint)" << v;
      } else {
        os << "(int)" << v;
      }
      return;
    } else if (lanes == 32) {
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

  if (op->dtype.is_float16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_half2(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_float16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
  }

  if (op->dtype.is_bfloat16()) {
    std::string v = PrintExpr(op->value);
    os << "make_";
    PrintType(op->dtype, os);
    os << '(';
    if (lanes <= 8) {
      for (int i = 0; i < lanes / 2; ++i) {
        if (i != 0)
          os << ", ";
        os << "__pack_nv_bfloat162(" << v << ", " << v << ")";
      }
    } else {
      for (int i = 0; i < lanes / 4; ++i) {
        if (i != 0)
          os << ", ";
        os << "tl::pack_bfloat16x4(" << v << ", " << v << ", " << v << ", " << v
           << ")";
      }
    }
    os << ')';
    return;
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
    ICHECK(p);
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

inline void PrintConst(const FloatImmNode *op, std::ostream &os,
                       CodeGenTileLangCUDA *p) { // NOLINT(*)
  // Type code is kBFloat/kFloat16
  // which is indeed CUTLASS supported types currently
  if (op->dtype.is_bfloat16() || op->dtype.is_float16()) {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << "std::numeric_limits<";
      p->PrintType(op->dtype, temp);
      temp << ">::infinity()";
    } else if (std::isnan(op->value)) {
      temp << "std::numeric_limits<";
      p->PrintType(op->dtype, temp);
      temp << ">::quiet_NaN()";
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
  case 64:
  case 32: {
    std::ostringstream temp;
    if (std::isinf(op->value)) {
      if (op->value < 0) {
        temp << "-";
      }
      temp << ((op->dtype.bits() == 32) ? "CUDART_INF_F" : "CUDART_INF");
      p->need_math_constants_h_ = true;
    } else if (std::isnan(op->value)) {
      temp << ((op->dtype.bits() == 32) ? "CUDART_NAN_F" : "CUDART_NAN");
      p->need_math_constants_h_ = true;
    } else {
      temp << std::hexfloat << op->value;
      if (op->dtype.bits() == 32)
        temp << 'f';
      temp << "/*" << std::scientific << op->value << "*/";
    }
    p->MarkConst(temp.str());
    os << temp.str();
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenTileLangCUDA::VisitExpr_(const FloatImmNode *op,
                                     std::ostream &os) { // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenTileLangCUDA::PrintWmmaScope(const std::string &scope, DataType t,
                                         const VarNode *variable,
                                         std::ostream &os) {
  std::stringstream type;
  PrintType(t, type);
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  if ((t.is_int() || t.is_uint()) && t.bits() < 8 && t.lanes() == 1) {
    type.str(std::string());
    if (t.is_int()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::s4";
      } else if (t.bits() == 1) {
        type << "nvcuda::wmma::experimental::precision::b1";
      } else {
        LOG(FATAL) << "Unhandled integer type for wmma fragment!";
      }
    } else if (t.is_uint()) {
      if (t.bits() == 4) {
        type << "nvcuda::wmma::experimental::precision::u4";
      } else {
        LOG(FATAL) << "Unhandled integer type for wmma fragment!";
      }
    }
  }
  if (scope == "wmma.matrix_a") {
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_a";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, " << shape_str << ", "
       << type.str() << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.matrix_b") {
    std::string layout_str = fragment_layouts[variable];
    ICHECK_NE(layout_str, "") << "Layout must be defined for matrix_b";
    os << "nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, " << shape_str << ", "
       << type.str() << ", nvcuda::wmma::" << layout_str << ">";
  } else if (scope == "wmma.accumulator") {
    os << "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, " << shape_str
       << ", " << type.str() << ">";
  }
}

int32_t CodeGenTileLangCUDA::GetWmmaFragmentSize(const std::string &scope,
                                                 const VarNode *variable,
                                                 int32_t size) {
  ICHECK(fragment_shapes.count(variable))
      << "Cannot find shape of the wmma fragment " << variable->name_hint;
  std::string shape_str = fragment_shapes.at(variable);
  std::pair<int32_t, int32_t> dim = GetWmmaFragmentDimSize(shape_str, scope);
  if (dim.first * dim.second != 0)
    return size / dim.first / dim.second;
  else
    return 0;
}

void CodeGenTileLangCUDA::HandleVolatileLoads(const std::string &value,
                                              const BufferLoadNode *op,
                                              std::ostream &os) {
  // Cast away volatile qualifier for fp16 types. That is, only loads and
  // stores are volatile. The loaded objects are not marked as volatile.
  //
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) &&
      IsVolatile(op->buffer->data.get())) {
    os << "(";
    PrintType(op->dtype, os);
    os << ")(" << value << ")";
  } else {
    os << value;
  }
}

void CodeGenTileLangCUDA::PrintVecElemLoadExpr(DataType t, int i,
                                               const std::string &value,
                                               std::ostream &os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (!(t.lanes() == 2 || t.lanes() == 3)) {
      if (i != 0) {
        os << "|";
      }
      os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8
         << "))";
      return;
    }
  }

  if (t.is_float16()) {
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

  if (t.is_bfloat16()) {
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

void CodeGenTileLangCUDA::PrintFunctionSignature(const String &function_name,
                                                 const PrimFunc &func,
                                                 std::ostream &os) {
  PrintFuncPrefix(os);
  CodeGenC::PrintType(func->ret_type, os);
  CodeGenC::PrintExtraAttrs(func, os);
  bool no_alias = func->HasNonzeroAttr(tir::attr::kNoAlias);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          func->GetAttr<ffi::Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tir::Var &v : opt.value())
      non_restrict.insert(v.get());
  }
  // Read-only param indices attribute, if present.
  std::unordered_set<int> ro_param_indices;
  if (auto opt =
          func->GetAttr<ffi::Array<Integer>>("tl.readonly_param_indices")) {
    for (const auto &idx : opt.value()) {
      ro_param_indices.insert(static_cast<int>(Downcast<Integer>(idx)->value));
    }
  }
  os << " " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    tir::Var v = func->params[i];
    std::string vid = AllocVarID(v.get());

    if (i > 0) {
      os << ", ";
    }

    if (v.dtype().is_handle()) {
      // work around for grid constant parameters.
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (ptr->storage_scope == "grid_constant") {
          os << "__grid_constant__ const ";
          CodeGenC::PrintType(ptr->element_type, os);
          os << ' ' << vid;
          continue;
        }
      }

      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, os);
      }
      // If marked read-only, emit const qualifier before type.
      if (ro_param_indices.count(static_cast<int>(i))) {
        os << "const ";
      }
      CodeGenC::PrintType(GetType(v), os);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && !non_restrict.count(v.get())) {
        PrintRestrict(v, os);
      }
    } else {
      CodeGenC::PrintType(GetType(v), os);
    }
    os << ' ' << vid;
  }
  os << ")";

  // Register handle data type
  // TODO(tvm-team): consider simply keep type info in the
  // type annotation(via a normalizing rewriting).
  for (const auto &param : func->params) {
    if (auto *ptr = param->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(param.get(), prim->dtype);
      }
    }
  }
}

void CodeGenTileLangCUDA::AddFunction(const GlobalVar &gvar,
                                      const PrimFunc &f) {
  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol)
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          f->GetAttr<ffi::Array<tir::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tir::Var &v : opt.value())
      non_restrict.insert(v.get());
  }
  // Read-only param indices attribute, if present.
  std::unordered_set<int> ro_param_indices;
  if (auto opt = f->GetAttr<ffi::Array<Integer>>("tl.readonly_param_indices")) {
    for (const auto &idx : opt.value()) {
      ro_param_indices.insert(static_cast<int>(Downcast<Integer>(idx)->value));
    }
  }

  this->PrintFuncPrefix(stream);
  CodeGenC::PrintType(f->ret_type, stream);
  this->PrintExtraAttrs(f);

  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0)
      stream << ", ";
    if (v.dtype().is_handle()) {
      // work around for grid constant parameters.
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
      // If marked read-only, emit const qualifier before type.
      if (ro_param_indices.count(static_cast<int>(i))) {
        stream << "const ";
      }
      CodeGenC::PrintType(GetType(v), stream);
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }

      if (no_alias && !non_restrict.count(v.get())) {
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
}

} // namespace codegen
} // namespace tvm
