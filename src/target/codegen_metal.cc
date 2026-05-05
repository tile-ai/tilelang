/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file codegen_metal.cc
 */
#include "codegen_metal.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "../op/builtin.h"
#include "runtime/metal/metal_module.h"
#include "runtime/thread_storage_scope.h"
#include "target/build_common.h"

namespace tvm {
namespace codegen {

void CodeGenTileLangMetal::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenTileLangMetal::CodeGenTileLangMetal(Target target) : target_(target) {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream << "using namespace metal;\n\n";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int[2];\n"
              << "};\n\n";
}

// Inline MSL helpers for storage-only FP8 emulation (e4m3 / e5m2).
// Apple Silicon (M4 Max and earlier; M5 NAX is FP16/INT8 only) has NO native
// FP8 ALU support, so FP8 is realised as `uchar` storage with explicit
// dequantize-on-load / quantize-on-store. The helpers mirror the IEEE 754
// derived encoding from the OFP8 spec (E4M3 with finite-only encoding, E5M2
// IEEE-style with NaN/Inf).
void CodeGenTileLangMetal::PrintFP8Prelude(std::ostream &os) {
  os <<
      "// FP8 storage-only emulation helpers (MSL has no native float8 type).\n"
      "// See OCP \"OFP8 Formats for Deep Learning\" v1.0 spec.\n"
      "inline half __tvm_fp8_e4m3_to_half(uchar x) {\n"
      "  ushort sign = (ushort)(x & 0x80) << 8;\n"
      "  ushort mant = (ushort)(x & 0x07);\n"
      "  ushort exp = (ushort)((x >> 3) & 0x0F);\n"
      "  ushort h;\n"
      "  if (exp == 0) {\n"
      "    if (mant == 0) {\n"
      "      h = sign;\n"
      "    } else {\n"
      "      // subnormal: e4m3 value = mant * 2^-9. After shifting the\n"
      "      // mantissa so the leading 1 hits bit 2 (0x4), the half\n"
      "      // biased exponent is (e + 7), not (e + 8).\n"
      "      ushort m = mant;\n"
      "      ushort e = 1;\n"
      "      while ((m & 0x4) == 0) { m <<= 1; e -= 1; }\n"
      "      m &= 0x3;\n"
      "      h = (ushort)(sign | ((ushort)(e + 7) << 10) | (ushort)(m << 8));\n"
      "    }\n"
      "  } else if (exp == 0x0F && mant == 0x07) {\n"
      "    h = (ushort)(sign | 0x7E00);\n"
      "  } else {\n"
      "    h = (ushort)(sign | ((ushort)(exp + 8) << 10) | (ushort)(mant << 7));\n"
      "  }\n"
      "  return as_type<half>(h);\n"
      "}\n"
      "inline half __tvm_fp8_e5m2_to_half(uchar x) {\n"
      "  ushort h = ((ushort)x) << 8;\n"
      "  return as_type<half>(h);\n"
      "}\n"
      "inline uchar __tvm_half_to_fp8_e4m3(half v) {\n"
      "  ushort h = as_type<ushort>(v);\n"
      "  ushort sign = (h >> 8) & 0x80;\n"
      "  short he = (short)((h >> 10) & 0x1F);\n"
      "  ushort hm = h & 0x3FF;\n"
      "  if (he == 0x1F) {\n"
      "    return (uchar)(sign | 0x7F);\n"
      "  }\n"
      "  short e = he - 8;\n"
      "  if (e >= 0x0F) {\n"
      "    return (uchar)(sign | 0x7E);\n"
      "  }\n"
      "  if (e <= 0) {\n"
      "    if (e < -3) return (uchar)sign;\n"
      "    ushort m = hm | 0x400;\n"
      "    ushort shift = (ushort)(7 + 1 - e);\n"
      "    ushort round_bit = (ushort)1 << (shift - 1);\n"
      "    ushort sticky = m & (round_bit - 1);\n"
      "    ushort q = m >> shift;\n"
      "    ushort rem = m & ((round_bit << 1) - 1);\n"
      "    if (rem > round_bit || (rem == round_bit && (q & 1))) q += 1;\n"
      "    (void)sticky;\n"
      "    return (uchar)(sign | (q & 0x7F));\n"
      "  }\n"
      "  ushort q = hm >> 7;\n"
      "  ushort rem = hm & 0x7F;\n"
      "  if (rem > 0x40 || (rem == 0x40 && (q & 1))) {\n"
      "    q += 1;\n"
      "    if (q == 0x08) { q = 0; e += 1; }\n"
      "    if (e >= 0x0F) return (uchar)(sign | 0x7E);\n"
      "  }\n"
      "  return (uchar)(sign | (ushort)(e << 3) | (q & 0x07));\n"
      "}\n"
      "inline uchar __tvm_half_to_fp8_e5m2(half v) {\n"
      "  ushort h = as_type<ushort>(v);\n"
      "  ushort sign = h & 0x8000;\n"
      "  ushort exp = (h >> 10) & 0x1F;\n"
      "  ushort mant = h & 0x3FF;\n"
      "  if (exp == 0x1F) {\n"
      "    if (mant != 0) return (uchar)((sign >> 8) | 0x7E);\n"
      "    return (uchar)((sign >> 8) | 0x7C);\n"
      "  }\n"
      "  ushort q = mant >> 8;\n"
      "  ushort rem = mant & 0xFF;\n"
      "  if (rem > 0x80 || (rem == 0x80 && (q & 1))) {\n"
      "    q += 1;\n"
      "    if (q == 0x4) { q = 0; exp += 1; }\n"
      "    if (exp == 0x1F) return (uchar)((sign >> 8) | 0x7C);\n"
      "  }\n"
      "  return (uchar)((sign >> 8) | (uchar)(exp << 2) | (uchar)(q & 0x3));\n"
      "}\n\n";
}

// Vector FP8 cast helpers (lanes = 2, 3, 4). Storage rules:
//   lanes 2-4 -> ucharN (matches PrintType output)
// Each helper just calls the scalar variant per lane. Keeping the vector type
// at the IR level lets subsequent passes preserve their vector loads/stores.
// The compiler is free to scalarise these back into per-lane calls; the goal
// here is to keep the IR-level type information intact.
void CodeGenTileLangMetal::PrintFP8VectorPrelude(std::ostream &os) {
  os <<
      "// Vector FP8 helpers (lanes 2/3/4 use ucharN packed storage).\n"
      "inline half2 __tvm_fp8_e4m3_to_half_v2(uchar2 x) {\n"
      "  return half2(__tvm_fp8_e4m3_to_half(x.x), __tvm_fp8_e4m3_to_half(x.y));\n"
      "}\n"
      "inline half3 __tvm_fp8_e4m3_to_half_v3(uchar3 x) {\n"
      "  return half3(__tvm_fp8_e4m3_to_half(x.x), __tvm_fp8_e4m3_to_half(x.y), __tvm_fp8_e4m3_to_half(x.z));\n"
      "}\n"
      "inline half4 __tvm_fp8_e4m3_to_half_v4(uchar4 x) {\n"
      "  return half4(__tvm_fp8_e4m3_to_half(x.x), __tvm_fp8_e4m3_to_half(x.y), __tvm_fp8_e4m3_to_half(x.z), __tvm_fp8_e4m3_to_half(x.w));\n"
      "}\n"
      "inline half2 __tvm_fp8_e5m2_to_half_v2(uchar2 x) {\n"
      "  return half2(__tvm_fp8_e5m2_to_half(x.x), __tvm_fp8_e5m2_to_half(x.y));\n"
      "}\n"
      "inline half3 __tvm_fp8_e5m2_to_half_v3(uchar3 x) {\n"
      "  return half3(__tvm_fp8_e5m2_to_half(x.x), __tvm_fp8_e5m2_to_half(x.y), __tvm_fp8_e5m2_to_half(x.z));\n"
      "}\n"
      "inline half4 __tvm_fp8_e5m2_to_half_v4(uchar4 x) {\n"
      "  return half4(__tvm_fp8_e5m2_to_half(x.x), __tvm_fp8_e5m2_to_half(x.y), __tvm_fp8_e5m2_to_half(x.z), __tvm_fp8_e5m2_to_half(x.w));\n"
      "}\n"
      "inline uchar2 __tvm_half_to_fp8_e4m3_v2(half2 v) {\n"
      "  return uchar2(__tvm_half_to_fp8_e4m3(v.x), __tvm_half_to_fp8_e4m3(v.y));\n"
      "}\n"
      "inline uchar3 __tvm_half_to_fp8_e4m3_v3(half3 v) {\n"
      "  return uchar3(__tvm_half_to_fp8_e4m3(v.x), __tvm_half_to_fp8_e4m3(v.y), __tvm_half_to_fp8_e4m3(v.z));\n"
      "}\n"
      "inline uchar4 __tvm_half_to_fp8_e4m3_v4(half4 v) {\n"
      "  return uchar4(__tvm_half_to_fp8_e4m3(v.x), __tvm_half_to_fp8_e4m3(v.y), __tvm_half_to_fp8_e4m3(v.z), __tvm_half_to_fp8_e4m3(v.w));\n"
      "}\n"
      "inline uchar2 __tvm_half_to_fp8_e5m2_v2(half2 v) {\n"
      "  return uchar2(__tvm_half_to_fp8_e5m2(v.x), __tvm_half_to_fp8_e5m2(v.y));\n"
      "}\n"
      "inline uchar3 __tvm_half_to_fp8_e5m2_v3(half3 v) {\n"
      "  return uchar3(__tvm_half_to_fp8_e5m2(v.x), __tvm_half_to_fp8_e5m2(v.y), __tvm_half_to_fp8_e5m2(v.z));\n"
      "}\n"
      "inline uchar4 __tvm_half_to_fp8_e5m2_v4(half4 v) {\n"
      "  return uchar4(__tvm_half_to_fp8_e5m2(v.x), __tvm_half_to_fp8_e5m2(v.y), __tvm_half_to_fp8_e5m2(v.z), __tvm_half_to_fp8_e5m2(v.w));\n"
      "}\n\n";
}

void CodeGenTileLangMetal::AddFunction(const GlobalVar &gvar,
                                       const PrimFunc &func) {
  // NOTE: There is no inter-function calls among Metal kernels.
  // For now we keep the metal codegen without inter-function call
  // process.
  // We can switch to follow the flow with inter-function call process
  // after the Metal function declaration is properly printed.
  // In Metal, for PrimFuncs with signature
  //    def func(A: Buffer, B: Buffer, x: int, y: float) -> None
  // where there are trailing pod parameters, the codegen emits a struct
  //    struct func_params{ x: int; y: float; }
  // for the function. In the flow of inter-function call process,
  // the struct will be emitted for every time a function is declared.
  // So consequently there are duplicate appearances of a same struct,
  // which makes the Metal compiler unable to recognize.

  // clear previous generated state.
  this->InitFuncState(func);
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->FreshName("v_");

  // add to alloc buffer type.
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  // Function header.
  this->stream << "kernel void "
               << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  size_t limit =
      target_->GetAttr<Integer>("max_function_args").value().IntValue();
  if (func->params.size() > limit) {
    LOG(WARNING) << "Probably you won't be able to execute your kernel due to "
                    "high number of "
                    "buffers in the kernel";
  }
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v.dtype().is_handle())
      break;
    this->stream << "  ";
    std::string vid = AllocVarID(v.get());
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, this->stream);
    }
    PrintType(GetType(v), this->stream);
    // Register handle data type
    // TODO(tvm-team): consider simply keep type info in the
    // type annotation(via a normalizing rewriting).
    if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(v.get(), prim->dtype);
      }
    }
    this->stream << ' ' << vid << " [[ buffer(" << i << ") ]],\n";
  }
  // Setup normal arguments.
  size_t nargs = func->params.size() - num_buffer;
  std::string varg = name_supply_->FreshName("arg");
  if (nargs != 0) {
    std::string arg_buf_type =
        static_cast<std::string>(global_symbol.value()) + "_args_t";
    this->stream << "  constant " << arg_buf_type << "& " << varg
                 << " [[ buffer(" << num_buffer << ") ]],\n";
    // declare the struct
    decl_stream << "struct " << arg_buf_type << " {\n";
    for (size_t i = num_buffer; i < func->params.size(); ++i) {
      Var v = func->params[i];
      ICHECK(!v.dtype().is_handle());
      std::string vid = AllocVarID(v.get());
      std::ostringstream vref;
      if (v.dtype().bits() == 32) {
        decl_stream << "  ";
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vid << "[2];\n";
        vref << varg << "." << vid << "[0]";
      } else if (v.dtype().bits() == 64) {
        decl_stream << "  ";
        PrintType(v.dtype(), decl_stream);
        decl_stream << " " << vid << ";\n";
        vref << varg << "." << vid;
      } else {
        // For non 32bit type, ref through arg union.
        decl_stream << "  __TVMArgUnion " << vid << ";\n";
        vref << varg << "." << vid << ".v_";
        PrintType(v.dtype(), vref);
      }
      var_idmap_[v.get()] = vref.str();
    }
    decl_stream << "};\n\n";
  }
  // Setup the thread group info.
  ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");
  int work_dim = 0;
  auto launch_params =
      func->GetAttr<ffi::Array<ffi::String>>(tir::attr::kKernelLaunchParams)
          .value();
  for (const auto &tag : launch_params) {
    if (tag != runtime::launch_param::kUseDynamicSharedMemoryTag) {
      runtime::ThreadScope scope = runtime::ThreadScope::Create(tag);
      work_dim = std::max(work_dim, scope.dim_index + 1);
    }
  }

  if (work_dim != 0) {
    // use ushort by default for now
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    stream << " blockIdx [[threadgroup_position_in_grid]],\n";
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    stream << " threadIdx [[thread_position_in_threadgroup]]\n";
  }
  thread_work_dim_ = work_dim;

  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  this->PrintStmt(func->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangMetal::BindThreadIndex(const IterVar &iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  // if we only have threadIdx.x
  // metal will directly print as threadIdx
  std::string vname = iv->thread_tag;
  if (thread_work_dim_ <= 1) {
    vname = vname.substr(0, iv->thread_tag.length() - 2);
  }
  var_idmap_[iv->var.get()] =
      CastFromTo(vname, DataType::UInt(thread_index_bits_), iv->var.dtype());
}

void CodeGenTileLangMetal::PrintType(DataType t,
                                     std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }

  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  if (t.is_float() && t.bits() == 16 && lanes > 4 && lanes <= 8 &&
      lanes % 2 == 0) {
    os << "uint" << lanes / 2;
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    if (lanes == 3) {
      os << "packed_";
    }
    switch (t.bits()) {
    case 16:
      os << "half";
      break;
    case 32:
      os << "float";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
    case 8:
      os << "char";
      break;
    case 16:
      os << "short";
      break;
    case 32:
      os << "int";
      break;
    case 64:
      os << "long";
      break;
    case 1:
      os << "bool";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && lanes == 1)
      return;
    if (!fail && (lanes >= 2 && lanes <= 4)) {
      os << lanes;
      return;
    }
  } else if (t.is_bfloat16()) {
    os << "bfloat";
    return;
  } else if (t.is_float8()) {
    // FP8 is storage-only on Metal: print as `uchar`/`ucharN` and emit explicit
    // dequantize/quantize helpers via the FP8 prelude. Caller-side casts must
    // route through __tvm_fp8_*_to_half / __tvm_half_to_fp8_*.
    enable_fp8_ = true;
    if (lanes == 1) {
      os << "uchar";
      return;
    }
    if (lanes >= 2 && lanes <= 4) {
      os << "uchar" << lanes;
      return;
    }
    if (lanes == 8) {
      os << "uint2";
      return;
    }
    if (lanes == 16) {
      os << "uint4";
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
}

void CodeGenTileLangMetal::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "global barrier not supported";
  }
}

void CodeGenTileLangMetal::PrintVecElemLoad(const std::string &vec, DataType t,
                                            int i,
                                            std::ostream &os) { // NOLINT(*)
  if (t.is_float16() && t.lanes() > 4) {
    os << "((thread half*)(&" << vec << "))[" << i << "]";
  } else {
    os << vec << "[" << i << "]";
  }
}

void CodeGenTileLangMetal::PrintVecElemStore(const std::string &vec, DataType t,
                                             int i, const std::string &value) {
  this->PrintIndent();
  if (t.is_float16() && t.lanes() > 4) {
    stream << "((thread half*)(&" << vec << "))[" << i << "] = " << value
           << ";\n";
  } else {
    stream << vec << "[" << i << "]"
           << " = " << value << ";\n";
  }
}

void CodeGenTileLangMetal::PrintStorageScope(const std::string &scope,
                                             std::ostream &os) { // NOLINT(*)
  if (scope == "global") {
    os << "device ";
  } else if (scope == "shared" || scope == "shared.dyn") {
    os << "threadgroup ";
  } else if (scope == "local") {
    os << "thread ";
  } else {
    LOG(FATAL) << "Unknown storage scope `" << scope << "`";
  }
}

void CodeGenTileLangMetal::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  size_t constant_size = op->ConstantAllocationSize();
  ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";

  auto scope = GetPtrStorageScope(op->buffer_var);
  alloc_storage_scope_[op->buffer_var.get()] = scope;
  if (scope == "metal.simdgroup") {
    ICHECK(op->dtype == DataType::Float(16) ||
           op->dtype == DataType::Float(32) ||
           op->dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported, but got "
        << op->dtype;
    ICHECK(constant_size % 64 == 0) << "Only 8x8 matrix is supported, but got "
                                    << constant_size << " bytes\n";

    std::ostringstream dtype_os;
    PrintType(op->dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[op->buffer_var.get()] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << constant_size / 64 << "];\n";
  } else if (scope == "local.var") {
    ICHECK(op->dtype.is_scalar())
        << "Vector local.var allocation is not supported.";
    ICHECK_EQ(constant_size, 1)
        << "Only scalar local.var allocation is supported.";
    PrimExpr init = tir::make_const(op->dtype, 0);
    auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
    if (init_it != op->annotations.end()) {
      PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
      if (!user_init.dtype().is_void() && user_init.dtype() != op->dtype) {
        user_init = tir::Cast(op->dtype, user_init);
      }
      init = user_init;
    }
    PrintType(op->dtype, stream);
    stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(op->dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenTileLangMetal::VisitExpr_(const BufferLoadNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::string scope;
  auto it = alloc_storage_scope_.find(op->buffer->data.get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(op->buffer->data);
  }
  if (scope == "local.var") {
    ICHECK_EQ(op->indices.size(), 1)
        << "Load from non-flat local.var memory not supported.";
    ICHECK(op->dtype.is_scalar()) << "Vector local.var load is not supported.";
    auto index = op->indices[0].as<IntImmNode>();
    ICHECK(index && index->value == 0)
        << "local.var load requires scalar index 0.";
    os << GetVarID(op->buffer->data.get());
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangMetal::VisitStmt_(const BufferStoreNode *op) {
  std::string scope;
  auto it = alloc_storage_scope_.find(op->buffer->data.get());
  if (it != alloc_storage_scope_.end()) {
    scope = it->second;
  }
  if (scope.empty()) {
    scope = GetPtrStorageScope(op->buffer->data);
  }
  if (scope == "local.var") {
    ICHECK_EQ(op->indices.size(), 1)
        << "Store to non-flat local.var memory not supported.";
    ICHECK(op->value.dtype().is_scalar())
        << "Vector local.var store is not supported.";
    auto index = op->indices[0].as<IntImmNode>();
    ICHECK(index && index->value == 0)
        << "local.var store requires scalar index 0.";
    this->PrintIndent();
    stream << GetVarID(op->buffer->data.get()) << " = " << PrintExpr(op->value)
           << ";\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitExpr_(const SelectNode *op,
                                      std::ostream &os) { // NOLINT(*)
  os << "select(" << PrintExpr(op->false_value) << ", "
     << PrintExpr(op->true_value) << ", " << PrintExpr(op->condition) << ")";
}

void CodeGenTileLangMetal::VisitExpr_(const BroadcastNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::string v = PrintExpr(op->value);
  int lanes = op->dtype.lanes();
  if (op->dtype.is_float16() && lanes > 4 && lanes % 2 == 0) {
    os << "uint" << lanes / 2 << "(";
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      os << "as_type<uint>(half2(" << v << ", " << v << "))";
    }
    os << ')';
  } else {
    PrintType(op->dtype, os);
    os << "(";
    for (int i = 0; i < lanes; ++i) {
      if (i != 0)
        os << ", ";
      os << v;
    }
    os << ')';
  }
}

void CodeGenTileLangMetal::VisitExpr_(const CallNode *op,
                                      std::ostream &os) { // NOLINT(*)
  CHECK(!op->op.as<GlobalVarNode>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls PrimFunc "
      << op->op;
  auto f_check_simdgroup_shape = [](PrimExpr col, PrimExpr row) {
    ICHECK(col->IsInstance<IntImmNode>() && row->IsInstance<IntImmNode>())
        << "Only constant shape is supported for simdgroup matrix, but got "
        << col << "x" << row;
    int col_val = col.as<IntImmNode>()->value;
    int row_val = row.as<IntImmNode>()->value;
    ICHECK(col_val == 8 && row_val == 8)
        << "Only 8x8 matrix is supported, but got " << col_val << "x"
        << row_val;
  };
  if (op->op.same_as(builtin::make_filled_simdgroup_matrix())) {
    ICHECK_EQ(op->args.size(), 5);
    Var var = Downcast<Var>(op->args[0]);
    // Get the data type of the simdgroup matrix
    auto it = simdgroup_dtype_.find(var.get());
    ICHECK(it != simdgroup_dtype_.end())
        << "Cannot find variable allocation for simdgroup: " << var;
    const std::string &dtype_str = it->second;
    f_check_simdgroup_shape(op->args[3], op->args[4]);
    os << PrintExpr(var) << "[" << PrintExpr(op->args[1])
       << "] = make_filled_simdgroup_matrix<" << dtype_str << ", "
       << PrintExpr(op->args[3]) << ", " << PrintExpr(op->args[4]) << ">("
       << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_load())) {
    ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_load(" << PrintExpr(op->args[0]) << "["
       << PrintExpr(op->args[1]) << "], " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_store())) {
    ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_store(" << PrintExpr(op->args[0]) << "["
       << PrintExpr(op->args[1]) << "], " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_multiply_accumulate())) {
    ICHECK_EQ(op->args.size(), 8);
    os << "simdgroup_multiply_accumulate("                                 //
       << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "], " //
       << PrintExpr(op->args[2]) << "[" << PrintExpr(op->args[3]) << "], " //
       << PrintExpr(op->args[4]) << "[" << PrintExpr(op->args[5]) << "], " //
       << PrintExpr(op->args[6]) << "[" << PrintExpr(op->args[7]) << "])";
  } else if (op->op.same_as(builtin::reinterpret())) {
    // generate as_type<TYPE>(ARG)
    os << "(as_type<";
    this->PrintType(op->dtype, os);
    os << ">(";
    this->PrintExpr(op->args[0], os);
    os << "))";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangMetal::VisitExpr_(const CastNode *op,
                                      std::ostream &os) { // NOLINT(*)
  DataType from_ty = op->value.dtype();
  DataType target_ty = op->dtype;
  if (target_ty.is_float8() || from_ty.is_float8()) {
    enable_fp8_ = true;
    ICHECK_EQ(target_ty.lanes(), from_ty.lanes())
        << "FP8 vector cast lanes must match: " << from_ty << " -> "
        << target_ty;
    auto fp8_to_half = [&](DataType ft, std::string val) {
      const char *helper = ft.code() == DataType::kFloat8_e5m2
                               ? "__tvm_fp8_e5m2_to_half"
                               : "__tvm_fp8_e4m3_to_half";
      return std::string(helper) + "(" + val + ")";
    };
    auto half_to_fp8 = [&](DataType tt, std::string val) {
      const char *helper = tt.code() == DataType::kFloat8_e5m2
                               ? "__tvm_half_to_fp8_e5m2"
                               : "__tvm_half_to_fp8_e4m3";
      return std::string(helper) + "(" + val + ")";
    };
    if (target_ty.lanes() == 1) {
      std::string val = PrintExpr(op->value);
      if (from_ty.is_float8() && !target_ty.is_float8()) {
        std::string h = fp8_to_half(from_ty, val);
        if (target_ty == DataType::Float(16)) {
          os << h;
        } else {
          os << "((";
          PrintType(target_ty, os);
          os << ")(" << h << "))";
        }
      } else if (!from_ty.is_float8() && target_ty.is_float8()) {
        std::string h = from_ty == DataType::Float(16)
                            ? val
                            : "((half)(" + val + "))";
        os << half_to_fp8(target_ty, h);
      } else {
        std::string h = fp8_to_half(from_ty, val);
        os << half_to_fp8(target_ty, h);
      }
      return;
    }
    // Vector path (lanes 2/3/4): route through the vector helpers which
    // wrap the scalar helpers per-lane while preserving the vector type at
    // the IR level. Wider widths must be lowered to scalar casts upstream.
    int lanes = target_ty.lanes();
    if (lanes == 2 || lanes == 3 || lanes == 4) {
      enable_fp8_vector_ = true;
      auto fp8_to_half_vec = [&](DataType ft) {
        const char *base = ft.code() == DataType::kFloat8_e5m2
                               ? "__tvm_fp8_e5m2_to_half"
                               : "__tvm_fp8_e4m3_to_half";
        return std::string(base) + "_v" + std::to_string(lanes);
      };
      auto half_to_fp8_vec = [&](DataType tt) {
        const char *base = tt.code() == DataType::kFloat8_e5m2
                               ? "__tvm_half_to_fp8_e5m2"
                               : "__tvm_half_to_fp8_e4m3";
        return std::string(base) + "_v" + std::to_string(lanes);
      };
      std::string val = PrintExpr(op->value);
      if (from_ty.is_float8() && !target_ty.is_float8()) {
        std::string h = fp8_to_half_vec(from_ty) + "(" + val + ")";
        if (target_ty == DataType::Float(16, lanes)) {
          os << h;
        } else {
          os << "((";
          PrintType(target_ty, os);
          os << ")(" << h << "))";
        }
        return;
      } else if (!from_ty.is_float8() && target_ty.is_float8()) {
        std::string h_val = val;
        if (from_ty != DataType::Float(16, lanes)) {
          h_val = "((half" + std::to_string(lanes) + ")(" + val + "))";
        }
        os << half_to_fp8_vec(target_ty) << "(" << h_val << ")";
        return;
      } else {
        std::string h = fp8_to_half_vec(from_ty) + "(" + val + ")";
        os << half_to_fp8_vec(target_ty) << "(" << h << ")";
        return;
      }
    }
    LOG(FATAL) << "Vector FP8 casts (lanes=" << lanes
               << ") not yet supported by Metal storage-only FP8 emulation."
               << " Currently only lanes 2/3/4 are wired through inline"
               << " helpers; wider widths must be lowered to scalar casts.";
  }
  CodeGenC::VisitExpr_(op, os);
}

std::string CodeGenTileLangMetal::Finish() {
  std::ostringstream prelude;
  if (enable_fp8_) {
    PrintFP8Prelude(prelude);
  }
  if (enable_fp8_vector_) {
    PrintFP8VectorPrelude(prelude);
  }
  std::string base = CodeGenC::Finish();
  if (prelude.str().empty())
    return base;
  const std::string anchor = "using namespace metal;\n";
  auto pos = base.find(anchor);
  if (pos == std::string::npos) {
    return prelude.str() + base;
  }
  pos += anchor.size();
  return base.substr(0, pos) + "\n" + prelude.str() + base.substr(pos);
}

void CodeGenTileLangMetal::VisitExpr_(const FloatImmNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::ostringstream temp;
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      temp << "-";
    }
    temp << "INFINITY";
  } else if (std::isnan(op->value)) {
    temp << "NAN";
  } else {
    temp << std::scientific << op->value;
    if (op->dtype.bits() == 32)
      temp << 'f';
    else if (op->dtype.bits() == 16)
      temp << 'h';
  }
  MarkConst(temp.str());
  os << temp.str();
}

ffi::Module BuildTileLangMetal(IRModule mod, Target target) {
  bool output_ssa = false;
  mod = tir::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  std::unordered_map<std::string, std::string> smap;
  const auto fmetal_compile =
      tvm::ffi::Function::GetGlobal("tvm_callback_metal_compile");
  std::string fmt = fmetal_compile ? "metallib" : "metal";

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto global_symbol =
        kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenTileLangMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenTileLangMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    if (fmetal_compile) {
      fsource = (*fmetal_compile)(fsource, target).cast<std::string>();
    }
    smap[func_name] = fsource;
  }

  return MetalModuleCreate(smap, ExtractFuncInfo(mod), fmt, source_maker.str());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_metal", BuildTileLangMetal);
}
} // namespace codegen
} // namespace tvm
