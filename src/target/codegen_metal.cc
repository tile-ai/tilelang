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
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "op/builtin.h"
#include "runtime/metal/metal_module.h"
#include "runtime/thread_storage_scope.h"
#include "target/build_common.h"

namespace tvm {
namespace codegen {

void CodeGenTileLangMetal::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
  emitted_frag_lane_vars_ = false;
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenTileLangMetal::CodeGenTileLangMetal(Target target) : target_(target) {
  decl_stream << "#include <metal_stdlib>\n";
  decl_stream
      << "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  decl_stream << "using namespace metal;\n\n";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int[2];\n"
              << "};\n\n";
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
  TVM_FFI_ICHECK(global_symbol.has_value())
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
      TVM_FFI_ICHECK(!v.dtype().is_handle());
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
  TVM_FFI_ICHECK_EQ(name_supply_->FreshName("threadIdx"), "threadIdx");
  TVM_FFI_ICHECK_EQ(name_supply_->FreshName("blockIdx"), "blockIdx");
  int work_dim = 0;
  auto launch_params =
      func->GetAttr<ffi::Array<ffi::String>>(tirx::attr::kKernelLaunchParams)
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
    stream << " __gridDim [[threadgroups_per_grid]],\n";
    stream << "  ";
    PrintType(DataType::UInt(thread_index_bits_, work_dim), stream);
    stream << " threadIdx [[thread_position_in_threadgroup]]\n";
  }
  thread_work_dim_ = work_dim;

  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  if (work_dim > 0) {
    this->PrintIndent();
    stream << "const ushort __lane = ((uint)threadIdx.x) % 32;\n";
    this->PrintIndent();
    stream << "const ushort __qid = __lane >> 2;\n";
    this->PrintIndent();
    stream << "const ushort __base_row = (__qid & 4) | ((__lane >> 1) & 3);\n";
    this->PrintIndent();
    stream << "const ushort __base_col = ((__qid & 2) | (__lane & 1)) * 4;\n";
    emitted_frag_lane_vars_ = true;
  }
  this->PrintStmt(func->body);
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenTileLangMetal::BindThreadIndex(const IterVar &iv) {
  TVM_FFI_ICHECK(!var_idmap_.count(iv->var.get()));
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
    TVM_FFI_ICHECK_EQ(lanes, 1) << "do not yet support vector types";
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
  }
  LOG(FATAL) << "Cannot convert type " << t << " to Metal type";
}

void CodeGenTileLangMetal::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    // Metal has no per-simdgroup barrier; simdgroup_barrier synchronizes
    // the entire threadgroup (same as threadgroup_barrier).  We emit the
    // narrower intrinsic so the source documents the intended scope even
    // though the hardware effect is identical.
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

void CodeGenTileLangMetal::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "threadblock_swizzle_pattern") {
    std::string func_name;
    int panel_size = 0;
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_tuple()) && call->args.size() >= 2) {
        const auto *name_node = call->args[0].as<StringImmNode>();
        const auto *size_node = call->args[1].as<IntImmNode>();
        TVM_FFI_ICHECK(name_node && size_node);
        func_name = name_node->value;
        panel_size = static_cast<int>(size_node->value);
      }
    }
    TVM_FFI_ICHECK(!func_name.empty() && panel_size > 0);

    this->PrintIndent();
    stream << "{ ";
    if (func_name == "rasterization2DRow") {
      stream << "const uint __bi = blockIdx.x + blockIdx.y * __gridDim.x; "
                "const uint __gs = __gridDim.x * __gridDim.y; "
                "const uint __ps = "
             << panel_size
             << "u * __gridDim.x; "
                "const uint __po = __bi % __ps; "
                "const uint __pi = __bi / __ps; "
                "const uint __tp = (__gs + __ps - 1u) / __ps; "
                "const uint __st = __pi + 1u < __tp ? "
             << panel_size
             << "u : (__gs - __pi * __ps) / __gridDim.x; "
                "const uint3 blockIdx = uint3("
                "(__pi & 1u) ? __gridDim.x - 1u - __po / __st : __po / __st, "
                "__po % __st + __pi * "
             << panel_size
             << "u, "
                "blockIdx.z);\n";
    } else {
      stream << "const uint __bi = blockIdx.x + blockIdx.y * __gridDim.x; "
                "const uint __gs = __gridDim.x * __gridDim.y; "
                "const uint __ps = "
             << panel_size
             << "u * __gridDim.y; "
                "const uint __po = __bi % __ps; "
                "const uint __pi = __bi / __ps; "
                "const uint __tp = (__gs + __ps - 1u) / __ps; "
                "const uint __st = __pi + 1u < __tp ? "
             << panel_size
             << "u : (__gs - __pi * __ps) / __gridDim.y; "
                "const uint3 blockIdx = uint3("
                "__po % __st + __pi * "
             << panel_size
             << "u, "
                "(__pi & 1u) ? __gridDim.y - 1u - __po / __st : __po / __st, "
                "blockIdx.z);\n";
    }
    this->VisitStmt(op->body);
    this->PrintIndent();
    stream << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitStmt_(const ForNode *op) {
  auto *ext_imm = op->extent.as<IntImmNode>();
  if (ext_imm && ext_imm->value > 4) {
    PrintIndent();
    stream << "#pragma clang loop unroll(disable)\n";
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitStmt_(const AllocBufferNode *op) {
  TVM_FFI_ICHECK(op->buffer.defined());
  std::string vid = AllocVarID(op->buffer->data.get());

  this->PrintIndent();
  size_t constant_size = 1;
  for (const auto &dim : op->buffer->shape) {
    const IntImmNode *dim_imm = dim.as<IntImmNode>();
    TVM_FFI_ICHECK(dim_imm)
        << "Can only handle constant size stack allocation for now";
    constant_size *= dim_imm->value;
  }
  TVM_FFI_ICHECK_GT(constant_size, 0)
      << "Can only handle constant size stack allocation for now";

  DataType dtype = op->buffer->dtype;
  auto scope = GetPtrStorageScope(op->buffer->data);
  alloc_storage_scope_[op->buffer->data.get()] = scope;
  if (scope == "metal.cooperative_tensor") {
    TVM_FFI_ICHECK(dtype == DataType::Float(16) ||
                   dtype == DataType::Float(32) ||
                   dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported for "
           "cooperative_tensor, but got "
        << dtype;
    TVM_FFI_ICHECK(constant_size % 64 == 0)
        << "cooperative_tensor buffer size must be multiple of 64, got "
        << constant_size;

    std::ostringstream dtype_os;
    PrintType(dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    cooperative_tensor_dtype_[op->buffer->data.get()] = dtype_str;
    int elems_per_thread = constant_size / 32;
    stream << "thread " << dtype_str << " " << vid << '[' << elems_per_thread
           << "];\n";
    if (dtype_str == "float" && elems_per_thread >= 16 &&
        elems_per_thread % 16 == 0) {
      int num_c_tiles = elems_per_thread / 16;
      ct_c_inlined_.insert(op->buffer->data.get());
      this->PrintIndent();
      stream
          << "constexpr auto __pct_desc = mpp::tensor_ops::matmul2d_descriptor("
          << "16, 32, 16, false, false, true, "
          << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);"
             "\n";
      this->PrintIndent();
      stream << "mpp::tensor_ops::matmul2d<__pct_desc, "
                "metal::execution_simdgroup> __pct_op;\n";
      for (int t = 0; t < num_c_tiles; t++) {
        this->PrintIndent();
        stream << "auto __pct_c" << t
               << " = __pct_op.get_destination_cooperative_tensor<"
               << "decltype(__pct_op.get_left_input_cooperative_tensor<half, "
                  "half, float>()), "
               << "decltype(__pct_op.get_right_input_cooperative_tensor<half, "
                  "half, float>()), float>(); "
               << "for (ushort __i = 0; __i < 16; __i++) __pct_c" << t
               << "[__i] = 0.0f;\n";
      }
    }
  } else if (scope == "metal.simdgroup") {
    TVM_FFI_ICHECK(dtype == DataType::Float(16) ||
                   dtype == DataType::Float(32) ||
                   dtype == DataType::BFloat(16))
        << "Only float16, float32, and bfloat16 are supported, but got "
        << dtype;
    TVM_FFI_ICHECK(constant_size % 64 == 0)
        << "Only 8x8 matrix is supported, but got " << constant_size
        << " bytes\n";

    std::ostringstream dtype_os;
    PrintType(dtype, dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[op->buffer->data.get()] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << constant_size / 64 << "];\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(dtype, stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(op->buffer->data.get(), dtype);
}

void CodeGenTileLangMetal::EnsureCooperativeTensorBuffer(const Var &var) {
  if (cooperative_tensor_dtype_.count(var.get()) != 0) {
    return;
  }
  auto type_it = handle_data_type_.find(var.get());
  TVM_FFI_ICHECK(type_it != handle_data_type_.end())
      << "Cannot find variable allocation for cooperative_tensor: " << var;
  std::ostringstream dtype_os;
  PrintType(type_it->second, dtype_os);
  cooperative_tensor_dtype_[var.get()] = dtype_os.str();
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

std::string
CodeGenTileLangMetal::GetAddrSpaceOf(const PrimExpr &ptr_expr) const {
  if (auto *call = ptr_expr.as<CallNode>()) {
    if (call->op.same_as(builtin::address_of())) {
      if (auto *load = call->args[0].as<BufferLoadNode>()) {
        auto it = alloc_storage_scope_.find(load->buffer->data.get());
        if (it != alloc_storage_scope_.end()) {
          const std::string &scope = it->second;
          if (scope == "shared" || scope == "shared.dyn") {
            return "threadgroup";
          }
          if (scope == "local" || scope == "metal.cooperative_tensor") {
            return "thread";
          }
          if (scope == "global") {
            return "device";
          }
        }
      }
    }
    for (const auto &arg : call->args) {
      std::string result = GetAddrSpaceOf(arg);
      if (!result.empty()) {
        return result;
      }
    }
  }
  if (auto *var = ptr_expr.as<VarNode>()) {
    auto it = alloc_storage_scope_.find(var);
    if (it != alloc_storage_scope_.end()) {
      const std::string &scope = it->second;
      if (scope == "shared" || scope == "shared.dyn") {
        return "threadgroup";
      }
      if (scope == "local" || scope == "metal.cooperative_tensor") {
        return "thread";
      }
      if (scope == "global") {
        return "device";
      }
    }
  }
  return "thread";
}

void CodeGenTileLangMetal::VisitExpr_(const CallNode *op,
                                      std::ostream &os) { // NOLINT(*)
  TVM_FFI_ICHECK(!op->op.as<GlobalVarNode>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls PrimFunc "
      << op->op;
  auto f_check_simdgroup_shape = [](PrimExpr col, PrimExpr row) {
    TVM_FFI_ICHECK(col->IsInstance<IntImmNode>() &&
                   row->IsInstance<IntImmNode>())
        << "Only constant shape is supported for simdgroup matrix, but got "
        << col << "x" << row;
    int col_val = col.as<IntImmNode>()->value;
    int row_val = row.as<IntImmNode>()->value;
    TVM_FFI_ICHECK(col_val == 8 && row_val == 8)
        << "Only 8x8 matrix is supported, but got " << col_val << "x"
        << row_val;
  };
  if (op->op.same_as(builtin::make_filled_simdgroup_matrix())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 5);
    Var var = Downcast<Var>(op->args[0]);
    // Get the data type of the simdgroup matrix
    auto it = simdgroup_dtype_.find(var.get());
    TVM_FFI_ICHECK(it != simdgroup_dtype_.end())
        << "Cannot find variable allocation for simdgroup: " << var;
    const std::string &dtype_str = it->second;
    f_check_simdgroup_shape(op->args[3], op->args[4]);
    os << PrintExpr(var) << "[" << PrintExpr(op->args[1])
       << "] = make_filled_simdgroup_matrix<" << dtype_str << ", "
       << PrintExpr(op->args[3]) << ", " << PrintExpr(op->args[4]) << ">("
       << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_load())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_load(" << PrintExpr(op->args[0]) << "["
       << PrintExpr(op->args[1]) << "], " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_store())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 7);
    f_check_simdgroup_shape(op->args[4], op->args[5]);
    os << "simdgroup_store(" << PrintExpr(op->args[0]) << "["
       << PrintExpr(op->args[1]) << "], " << PrintExpr(op->args[2]) << ", "
       << PrintExpr(op->args[3]) << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(builtin::simdgroup_multiply_accumulate())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 8);
    os << "simdgroup_multiply_accumulate("                                 //
       << PrintExpr(op->args[0]) << "[" << PrintExpr(op->args[1]) << "], " //
       << PrintExpr(op->args[2]) << "[" << PrintExpr(op->args[3]) << "], " //
       << PrintExpr(op->args[4]) << "[" << PrintExpr(op->args[5]) << "], " //
       << PrintExpr(op->args[6]) << "[" << PrintExpr(op->args[7]) << "])";
  } else if (op->op.same_as(tl::cooperative_tensor_fill())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 5);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string val = PrintExpr(op->args[2]);
    int rows = op->args[3].as<IntImmNode>()->value;
    int cols = op->args[4].as<IntImmNode>()->value;
    int elems_per_tile = rows * cols / 32;
    Var fill_v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(fill_v);
    bool is_inlined = ct_c_inlined_.count(fill_v.get()) > 0;
    auto *fill_idx_imm = op->args[1].as<IntImmNode>();
    os << "for (ushort __i = 0; __i < " << elems_per_tile << "; __i++) " << var
       << "[" << idx << " * " << elems_per_tile << " + __i] = " << val;
    if (is_inlined && fill_idx_imm) {
      os << "; for (ushort __i = 0; __i < " << elems_per_tile << "; __i++) "
         << "__pct_c" << fill_idx_imm->value << "[__i] = " << val;
    }
  } else if (op->op.same_as(tl::cooperative_tensor_load())) {
    TVM_FFI_ICHECK_GE(op->args.size(), 11);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string src_ptr = PrintExpr(op->args[2]);
    std::string stride = PrintExpr(op->args[3]);
    int rows = op->args[4].as<IntImmNode>()->value;
    int cols = op->args[5].as<IntImmNode>()->value;
    Var v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v.get());
    TVM_FFI_ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(op->args[2]);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    os << "{ " << addr_space << " " << dtype << "* __src = (" << addr_space
       << " " << dtype << "*)" << src_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << (nfrag_r * nfrag_c * 8) << " + " << elem_offset << "]) = "
           << "*(" << addr_space << " " << dtype << "4*)(&__src[__r0 * "
           << stride << " + __c0]); "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << (nfrag_r * nfrag_c * 8) << " + " << (elem_offset + 4) << "]) = "
           << "*(" << addr_space << " " << dtype << "4*)(&__src[__r1 * "
           << stride << " + __c0]); } ";
        elem_offset += 8;
      }
    }
    os << "}";
  } else if (op->op.same_as(tl::cooperative_tensor_store())) {
    TVM_FFI_ICHECK_GE(op->args.size(), 11);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string dst_ptr = PrintExpr(op->args[2]);
    std::string stride = PrintExpr(op->args[3]);
    int rows = op->args[4].as<IntImmNode>()->value;
    int cols = op->args[5].as<IntImmNode>()->value;
    Var v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v.get());
    TVM_FFI_ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(op->args[2]);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    int total_elems = nfrag_r * nfrag_c * 8;
    bool is_inlined = ct_c_inlined_.count(v.get()) > 0;
    auto *store_idx_imm = op->args[1].as<IntImmNode>();
    if (is_inlined && store_idx_imm) {
      int mma_tiles_per_store = total_elems / 16;
      int base_pct = store_idx_imm->value * mma_tiles_per_store;
      for (int t = 0; t < mma_tiles_per_store; t++) {
        os << "for (ushort __i = 0; __i < 16; __i++) " << var << "[" << idx
           << " * " << total_elems << " + " << (t * 16) << " + __i] = "
           << "__pct_c" << (base_pct + t) << "[__i]; ";
      }
    }
    os << "{ " << addr_space << " " << dtype << "* __dst = (" << addr_space
       << " " << dtype << "*)" << dst_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(" << addr_space << " " << dtype << "4*)(&__dst[__r0 * "
           << stride << " + __c0]) = "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << total_elems << " + " << elem_offset << "]); "
           << "*(" << addr_space << " " << dtype << "4*)(&__dst[__r1 * "
           << stride << " + __c0]) = "
           << "*(thread " << dtype << "4*)(&" << var << "[" << idx << " * "
           << total_elems << " + " << (elem_offset + 4) << "]); } ";
        elem_offset += 8;
      }
    }
    os << "}";
  } else if (op->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
    TVM_FFI_ICHECK_GE(op->args.size(), 13);
    int M = op->args[8].as<IntImmNode>()->value;
    int N = op->args[9].as<IntImmNode>()->value;
    int K = op->args[10].as<IntImmNode>()->value;
    bool trans_a = op->args[11].as<IntImmNode>()->value != 0;
    bool trans_b = op->args[12].as<IntImmNode>()->value != 0;

    std::string a_var = PrintExpr(op->args[2]);
    std::string a_idx = PrintExpr(op->args[3]);
    std::string b_var = PrintExpr(op->args[4]);
    std::string b_idx = PrintExpr(op->args[5]);
    std::string c_var = PrintExpr(op->args[0]);
    std::string c_idx = PrintExpr(op->args[1]);

    Var a_v = Downcast<Var>(op->args[2]);
    Var c_v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(a_v);
    EnsureCooperativeTensorBuffer(Downcast<Var>(op->args[4]));
    EnsureCooperativeTensorBuffer(c_v);
    auto a_it = cooperative_tensor_dtype_.find(a_v.get());
    auto c_it = cooperative_tensor_dtype_.find(c_v.get());
    TVM_FFI_ICHECK(a_it != cooperative_tensor_dtype_.end());
    TVM_FFI_ICHECK(c_it != cooperative_tensor_dtype_.end());
    std::string a_dtype = a_it->second;
    std::string c_dtype = c_it->second;

    int a_elems = M * K / 32;
    int b_elems = K * N / 32;
    int c_elems = M * N / 32;

    TVM_FFI_ICHECK(M == 32 || N == 32 || K == 32)
        << "MPP matmul2d requires at least one of M, N, K to be 32, got " << M
        << "x" << N << "x" << K;

    bool c_inlined = ct_c_inlined_.count(c_v.get()) > 0;
    auto *c_idx_imm = op->args[1].as<IntImmNode>();
    bool c_idx_const = c_inlined && c_idx_imm != nullptr;
    if (c_idx_const) {
      os << "{ "
         << "constexpr auto __desc = mpp::tensor_ops::matmul2d_descriptor(" << M
         << ", " << N << ", " << K << ", " << (trans_a ? "true" : "false")
         << ", " << (trans_b ? "true" : "false") << ", true, "
         << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)"
            "; "
         << "mpp::tensor_ops::matmul2d<__desc, metal::execution_simdgroup> "
            "__op; "
         << "auto __ct_a = __op.get_left_input_cooperative_tensor<" << a_dtype
         << ", " << a_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_b = __op.get_right_input_cooperative_tensor<" << a_dtype
         << ", " << a_dtype << ", " << c_dtype << ">(); "
         << "for (ushort __i = 0; __i < " << a_elems << "; __i++) "
         << "__ct_a[__i] = " << a_var << "[" << a_idx << " * " << a_elems
         << " + __i]; "
         << "for (ushort __i = 0; __i < " << b_elems << "; __i++) "
         << "__ct_b[__i] = " << b_var << "[" << b_idx << " * " << b_elems
         << " + __i]; "
         << "__op.run(__ct_a, __ct_b, __pct_c" << c_idx_imm->value << "); }";
    } else {
      os << "{ "
         << "constexpr auto __desc = mpp::tensor_ops::matmul2d_descriptor(" << M
         << ", " << N << ", " << K << ", " << (trans_a ? "true" : "false")
         << ", " << (trans_b ? "true" : "false") << ", true, "
         << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)"
            "; "
         << "mpp::tensor_ops::matmul2d<__desc, metal::execution_simdgroup> "
            "__op; "
         << "auto __ct_a = __op.get_left_input_cooperative_tensor<" << a_dtype
         << ", " << a_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_b = __op.get_right_input_cooperative_tensor<" << a_dtype
         << ", " << a_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_c = __op.get_destination_cooperative_tensor<"
         << "decltype(__ct_a), decltype(__ct_b), " << c_dtype << ">(); "
         << "for (ushort __i = 0; __i < " << a_elems << "; __i++) "
         << "__ct_a[__i] = " << a_var << "[" << a_idx << " * " << a_elems
         << " + __i]; "
         << "for (ushort __i = 0; __i < " << b_elems << "; __i++) "
         << "__ct_b[__i] = " << b_var << "[" << b_idx << " * " << b_elems
         << " + __i]; "
         << "for (ushort __i = 0; __i < " << c_elems << "; __i++) "
         << "__ct_c[__i] = " << c_var << "[" << c_idx << " * " << c_elems
         << " + __i]; "
         << "__op.run(__ct_a, __ct_b, __ct_c); "
         << "for (ushort __i = 0; __i < " << c_elems << "; __i++) " << c_var
         << "[" << c_idx << " * " << c_elems << " + __i] = __ct_c[__i]; }";
    }
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
  mod = tirx::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  ffi::Map<ffi::String, ffi::Bytes> smap;
  const auto fmetal_compile =
      tvm::ffi::Function::GetGlobal("tvm_callback_metal_compile");
  std::string fmt = fmetal_compile ? "metallib" : "metal";

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto global_symbol =
        kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenTileLangMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenTileLangMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    if (fmetal_compile) {
      smap.Set(func_name,
               (*fmetal_compile)(fsource, target).cast<ffi::Bytes>());
      continue;
    }
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  return MetalModuleCreate(std::move(smap), ExtractFuncInfo(mod),
                           ffi::String(fmt), ffi::String(source_maker.str()));
}

ffi::Module BuildTileLangMetalWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  mod = tirx::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  ffi::Map<ffi::String, ffi::Bytes> smap;

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "CodeGenTileLangMetal: Can only take PrimFunc";
    auto global_symbol =
        kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenTileLangMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenTileLangMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  return MetalModuleCreate(std::move(smap), ExtractFuncInfo(mod),
                           ffi::String("metal"),
                           ffi::String(source_maker.str()));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_metal", BuildTileLangMetal)
      .def("target.build.tilelang_metal_without_compile",
           BuildTileLangMetalWithoutCompile);
}
} // namespace codegen
} // namespace tvm
