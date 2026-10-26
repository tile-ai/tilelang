#include "compat/analyzer.h"
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
#include "op/prim.h"
#include "transform/common/pointer_value_type.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "tirx_compat/build_common.h"
using tvm::codegen::ExtractFuncInfo;
#include "tirx_compat/metal_fallback_module.h"

namespace tvm::tl::codegen_impl {
namespace attr = tvm::tl::codegen_attr;
using namespace tvm::prim;

namespace {

bool IsPackedHalfVector(const PrimType &type) {
  return type.bits() == 16 && type.MatchesCode(kDLFloat, kDLBfloat) &&
         type.lanes() > 4 && type.lanes() <= 8 && type.lanes() % 2 == 0;
}

} // namespace

Var CodeGenMetal::GetMatrixBufferVar(const Expr &data) {
  if (const auto *var = data.as<VarNode>()) {
    return ffi::GetRef<Var>(var);
  }
  if (const auto *call = data.as<CallNode>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) &&
      call->args.size() == 1) {
    const auto *buffer = call->args[0].as<VarNode>();
    TVM_FFI_ICHECK(buffer && buffer->ty.as<TensorTypeNode>())
        << "Metal simdgroup data operands expect tensor_data_ptr to project a "
           "TensorVar";
    return ffi::GetRef<Var>(buffer);
  }
  TVM_FFI_THROW(InternalError) << "Metal simdgroup data operands must be a Var "
                                  "or tensor_data_ptr(TensorVar), but got "
                               << data;
}

void CodeGenMetal::InitFuncState(const Function &f) {
  CodeGenC::InitFuncState(f);
  analyzer_ = sym::Analyzer();
  simdgroup_dtype_.clear();
  // analyze the data;
  for (Var arg : f->params) {
    if (arg->ty.as<PtrTypeNode>()) {
      alloc_storage_scope_[arg.get()] = runtime::MemoryScope::Global();
    }
  }
  AnalyzeMetalFunction(f);
}

CodeGenMetal::CodeGenMetal(Target target) : target_(target) {
  header_stream_ << "#include <metal_stdlib>\n";
  header_stream_
      << "#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n";
  restrict_keyword_ = "__restrict";
  header_stream_ << "using namespace metal;\n\n";
  header_stream_ << "union __TVMArgUnion {\n"
                 << " int v_int[2];\n"
                 << "};\n\n";
}

void CodeGenMetal::AddFunction(const GlobalVar &gvar, const Function &func) {
  TVM_FFI_CHECK(func->body.has_value(), ValueError)
      << "Kernel code generation requires a function body";
  // NOTE: There is no inter-function calls among Metal kernels.
  // For now we keep the metal codegen without inter-function call
  // process.
  // We can switch to follow the flow with inter-function call process
  // after the Metal function declaration is properly printed.
  // In Metal, for Functions with signature
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
  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  TVM_FFI_ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect Function to have the global_symbol attribute";

  // Function header.
  EmitLocation(func->loc);
  if (total_threads_ > 0) {
    stream << "[[kernel, max_total_threads_per_threadgroup(" << total_threads_
           << ")]] void ";
  } else {
    stream << "kernel void ";
  }
  stream << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  size_t limit = target_->GetAttr<int64_t>("max_function_args").value();
  if (func->params.size() > limit) {
    LOG(WARNING) << "Probably you won't be able to execute your kernel due to "
                    "high number of "
                    "buffers in the kernel";
  }
  std::unordered_set<int64_t> read_only;
  std::unordered_set<int64_t> non_restrict;
  for (const IntImm &index :
       func->GetAttr(attr::kReadOnlyParamIndices).value_or({})) {
    TVM_FFI_CHECK(index->value >= 0 && index->value < func->params.size(),
                  ValueError)
        << "Read-only parameter index is out of range";
    read_only.insert(static_cast<int64_t>(index->value));
  }
  for (const IntImm &index :
       func->GetAttr(attr::kNonRestrictParamIndices).value_or({})) {
    TVM_FFI_CHECK(index->value >= 0 && index->value < func->params.size(),
                  ValueError)
        << "Non-restrict parameter index is out of range";
    non_restrict.insert(static_cast<int64_t>(index->value));
  }
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v->ty.as<PtrTypeNode>())
      break;
    this->stream << "  ";
    if (read_only.count(i))
      stream << "const ";
    std::string vid = AllocVarID(v.get());
    auto it = alloc_storage_scope_.find(v.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, this->stream);
    }
    PrintType(v->ty, this->stream);
    if (func->HasNonzeroAttr(tirx::attr::kNoAlias.name) &&
        !non_restrict.count(i))
      PrintRestrict(v, stream);
    // Register handle data type
    // TODO(tvm-team): consider simply keep type info in the
    // type annotation(via a normalizing rewriting).
    if (auto *ptr = v->ty.as<PtrTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType(v.get(), ffi::GetRef<PrimType>(prim));
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
      PrimType value_type = v->ty.as_or_throw<PrimType>();
      TVM_FFI_ICHECK(!value_type.IsVoid());
      std::string vid = AllocVarID(v.get());
      std::ostringstream vref;
      if (value_type.bits() == 32) {
        decl_stream << "  ";
        PrintType(value_type, decl_stream);
        decl_stream << " " << vid << "[2];\n";
        vref << varg << "." << vid << "[0]";
      } else if (value_type.bits() == 64) {
        decl_stream << "  ";
        PrintType(value_type, decl_stream);
        decl_stream << " " << vid << ";\n";
        vref << varg << "." << vid;
      } else {
        // For non 32bit type, ref through arg union.
        decl_stream << "  __TVMArgUnion " << vid << ";\n";
        vref << varg << "." << vid << ".v_";
        PrintType(value_type, vref);
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
      func->GetAttr(tvm::tirx::attr::kKernelLaunchParams).value();
  for (const auto &tag : launch_params) {
    if (tag != runtime::launch_param::kUseDynamicSharedMemoryTag) {
      support::GPUThreadTag scope = support::GPUThreadTag::Parse(tag);
      work_dim = std::max(work_dim, scope.dim_index + 1);
    }
  }

  if (work_dim != 0) {
    // use ushort by default for now
    stream << "  ";
    PrintType(PrimType::UInt(thread_index_bits_, work_dim), stream);
    stream << " blockIdx [[threadgroup_position_in_grid]],\n";
    stream << "  ";
    PrintType(PrimType::UInt(thread_index_bits_, work_dim), stream);
    stream << " __gridDim [[threadgroups_per_grid]],\n  ";
    PrintType(PrimType::UInt(thread_index_bits_, work_dim), stream);
    stream << " threadIdx [[thread_position_in_threadgroup]]";
    if (uses_cooperative_tensors_)
      stream << ",\n  uint __simd_group_id [[simdgroup_index_in_threadgroup]]";
    stream << "\n";
  }
  thread_work_dim_ = work_dim;

  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  if (needs_fragment_lane_vars_) {
    PrintIndent();
    stream << "const ushort __lane = "
              "__metal_get_thread_index_in_simdgroup(ushort());\n";
    PrintIndent();
    stream << "const ushort __quad = __lane >> 2;\n";
    PrintIndent();
    stream << "const ushort __base_row = (__quad & 4) | ((__lane >> 1) & 3);\n";
    PrintIndent();
    stream << "const ushort __base_col = ((__quad & 2) | (__lane & 1)) * 4;\n";
  }
  this->PrintStmt(func->body.value());
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenMetal::BindThreadIndex(const PrimVar &var,
                                   const ffi::String &thread_tag) {
  TVM_FFI_ICHECK(!var_idmap_.count(var.get()));
  support::GPUThreadTag ts = support::GPUThreadTag::Parse(thread_tag);
  TVM_FFI_ICHECK(ts.rank == 0 || ts.rank == 1)
      << "Unsupported Metal thread tag " << thread_tag;
  TVM_FFI_ICHECK_GE(ts.dim_index, 0);
  TVM_FFI_ICHECK_LT(ts.dim_index, 3);
  // if we only have threadIdx.x
  // metal will directly print as threadIdx
  std::string vname = thread_tag;
  if (thread_work_dim_ <= 1) {
    vname = vname.substr(0, thread_tag.length() - 2);
  }
  var_idmap_[var.get()] =
      CastFromTo(vname, PrimType::UInt(thread_index_bits_), var.ty());
  tl::arith::Bind(
      analyzer_, var,
      Range(tl::MakeConst(var.ty(), 0),
            tl::CastValue(thread_extents_.at(thread_tag), var.ty())));
  if (thread_tag == "blockIdx.x")
    block_indices_[0] = var;
  if (thread_tag == "blockIdx.y")
    block_indices_[1] = var;
}

void CodeGenMetal::PrintType(const PrimType &t, std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (IsPackedHalfVector(t)) {
    if (lanes == 6)
      os << "packed_";
    os << "uint" << lanes / 2;
    return;
  }
  if (lanes > 4) {
    os << "array<";
    PrintType(t.WithLanes(1), os);
    os << ", " << lanes << ">";
    return;
  }
  if (t.IsVoid()) {
    os << "void";
    return;
  }
  if (t.MatchesCode(kDLBool) && lanes <= 4) {
    os << "bool";
    if (lanes != 1)
      os << lanes;
    return;
  }
  bool fail = false;
  if (t.code() == DLDataTypeCode::kDLFloat) {
    // Need to care about sizes and alignment of half3/float3 because tirx
    // representation might not be aware of Metal half3/float3 details and can
    // treat them as just three elements, while sizes and alignmnents of
    // half3/float3 are one element more (half3-8 bytes/ float13 - 16bytes).
    // Example of problematic pattern: filling of threadgroup packed array using
    // float3 elements by threads concurrently can lead to datarace and wrong
    // data in threadgroup shared array. packed_(half3/float3) are exactly
    // datatypes dealing with 3 elements and per-element alignment
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
  } else if (t.MatchesCode(DLDataTypeCode::kDLUInt, DLDataTypeCode::kDLInt)) {
    if (t.MatchesCode(DLDataTypeCode::kDLUInt)) {
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
  } else if (t.MatchesElementType(DLDataTypeCode::kDLBfloat, 16)) {
    if (lanes == 3)
      os << "packed_";
    os << "bfloat";
    if (lanes == 1)
      return;
    if (lanes >= 2 && lanes <= 4) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError)
      << "Cannot convert type " << ffi::DLDataTypeToString(t->dtype)
      << " to Metal type";
}

void CodeGenMetal::PrintStorageSync(const CallNode *op) {
  const std::string &sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "shared" || sync == "shared.dyn") {
    this->PrintIndent();
    this->stream << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  } else if (sync == "global") {
    TVM_FFI_THROW(InternalError) << "global barrier not supported";
  }
}

void CodeGenMetal::PrintVecElemLoad(const std::string &vec, const PrimType &t,
                                    int i,
                                    std::ostream &os) { // NOLINT(*)
  if (t.IsScalar()) {
    os << vec;
    return;
  }
  if (IsPackedHalfVector(t)) {
    os << "((thread const ";
    PrintType(t.WithLanes(1), os);
    os << "*)(&" << vec << "))[" << i << "]";
    return;
  }
  os << vec << "[" << i << "]";
}

void CodeGenMetal::PrintVecElemStore(const std::string &vec, const PrimType &t,
                                     int i, const std::string &value) {
  this->PrintIndent();
  if (IsPackedHalfVector(t)) {
    stream << "((thread ";
    PrintType(t.WithLanes(1), stream);
    stream << "*)(&" << vec << "))[" << i << "] = " << value << ";\n";
    return;
  }
  stream << vec << "[" << i << "]"
         << " = " << value << ";\n";
}

void CodeGenMetal::PrintStorageScope(const runtime::MemoryScope &scope,
                                     std::ostream &os) { // NOLINT(*)
  static const auto scope_shared = runtime::MemoryScope::Get("shared");
  static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");
  static const auto scope_local = runtime::MemoryScope::Get("local");
  if (scope == runtime::MemoryScope::Global()) {
    os << "device ";
  } else if (scope == scope_shared || scope == scope_shared_dyn) {
    os << "threadgroup ";
  } else if (scope == scope_local || scope.name().starts_with("local.") ||
             scope.name().starts_with("metal.")) {
    os << "thread ";
  } else {
    TVM_FFI_THROW(InternalError)
        << "Unknown storage scope `" << scope.name() << "`";
  }
}

void CodeGenMetal::Dispatch_(const BindNode *op) {
  if (const auto *call = op->value.as<CallNode>(); call) {
    if (call->op.same_as(tirx::alloc_tensor_op()))
      return DispatchAllocTensor(op, call);
    if (call->op.same_as(tirx::decl_tensor_op()))
      return DispatchDeclTensor(op, call);
  }
  if (auto value = op->value.as<PrimExpr>(); value && active_mlx_panel_ > 0) {
    std::ostringstream coordinate;
    if (TryPrintMlxCoordinate(*value, coordinate)) {
      PrintIndent();
      PrintType(op->var->ty, stream);
      stream << " " << AllocVarID(op->var.get()) << " = " << coordinate.str()
             << ";\n";
      return;
    }
  }
  // Stateful reads cannot be substituted after the underlying state changes.
  if (auto prim_value = op->value.as<PrimExpr>();
      prim_value && SideEffect(prim_value.value()) <= CallEffectKind::kPure) {
    tl::arith::Bind(analyzer_, op->var, prim_value.value());
  }
  const auto *pointer_type = op->var->ty.as<PtrTypeNode>();
  if (pointer_type == nullptr) {
    return CodeGenC::Dispatch_(op);
  }

  const runtime::MemoryScope &storage_scope = pointer_type->scope;
  alloc_storage_scope_[op->var.get()] = storage_scope;
  RegisterHandleTypeFromPointer(op->var, &op->value);
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    TVM_FFI_ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
    return;
  }

  PrintIndent();
  PrintStorageScope(storage_scope, stream);
  PrintType(pointer_type->element_type, stream);
  stream << "* " << AllocVarID(op->var.get()) << " = (";
  PrintStorageScope(storage_scope, stream);
  PrintType(pointer_type->element_type, stream);
  stream << "*)" << value << ";\n";
}

void CodeGenMetal::DispatchAllocTensor(const BindNode *op,
                                       const CallNode *buffer_call) {
  static const auto scope_metal_simdgroup =
      runtime::MemoryScope::Get("metal.simdgroup");
  tvm::Tuple shape = tvm::Tuple(
      buffer_call->ty_args[0].as_or_throw<tirx::TensorType>()->shape);
  DLDataType dtype =
      buffer_call->ty_args[0].as_or_throw<tirx::TensorType>()->dtype->dtype;
  runtime::MemoryScope scope =
      buffer_call->ty_args[0].as_or_throw<tirx::TensorType>()->scope;
  TensorVar buffer = op->var.as_or_throw<TensorVar>();
  DictAttrs annotations =
      buffer_call->attrs.has_value()
          ? buffer_call->attrs.value().as_or_throw<DictAttrs>()
          : DictAttrs(ffi::Map<ffi::String, ffi::Any>{});
  TVM_FFI_ICHECK(buffer.defined());
  std::string vid = AllocVarID(buffer.get());

  this->PrintIndent();
  // Compute a compile-time upper bound on the number of buffer elements.
  size_t constant_size = 1;
  for (const auto &dim : shape->fields) {
    const auto *dim_imm = dim.as<IntImmNode>();
    int64_t dim_size =
        dim_imm
            ? static_cast<int64_t>(dim_imm->value)
            : tl::arith::ConstIntBoundOf(analyzer_, dim.as_or_throw<PrimExpr>())
                  ->max_value;
    if (dim_imm == nullptr) {
      // An integer dtype's intrinsic maximum is not a program-derived
      // allocation bound.
      TVM_FFI_ICHECK(dim_size != sym::ConstIntBound::kPosInf)
          << "Metal allocation extent requires a finite compile-time upper "
             "bound, but got "
          << dim;
      PrimExpr dtype_max_value = max_value(dim->ty.as_or_throw<PrimType>());
      if (const auto *dtype_max = dtype_max_value.as<IntImmNode>()) {
        TVM_FFI_ICHECK_LT(dim_size, dtype_max->value)
            << "Metal allocation extent requires a finite compile-time upper "
               "bound, but got "
            << dim;
      }
    }
    TVM_FFI_ICHECK_GT(dim_size, 0)
        << "Metal allocation extent requires a positive compile-time upper "
           "bound, but got "
        << dim;
    TVM_FFI_ICHECK_LE(static_cast<uint64_t>(dim_size),
                      std::numeric_limits<size_t>::max() / constant_size)
        << "Metal allocation element count is too large to represent";
    constant_size *= static_cast<size_t>(dim_size);
  }

  alloc_storage_scope_[buffer.get()] = scope;
  if (scope.name() == "metal.cooperative_tensor") {
    AllocateCooperativeTensor(buffer, constant_size);
  } else if (scope == scope_metal_simdgroup) {
    bool supported_simdgroup_dtype = PrimType(dtype) == PrimType::Float(16) ||
                                     PrimType(dtype) == PrimType::Float(32) ||
                                     PrimType(dtype) == PrimType::BFloat(16);
    TVM_FFI_ICHECK(supported_simdgroup_dtype)
        << "Only float16, float32, and bfloat16 are supported, but got "
        << ffi::DLDataTypeToString(dtype);
    TVM_FFI_ICHECK(constant_size % 64 == 0)
        << "Only 8x8 matrix is supported, but got " << constant_size
        << " bytes\n";

    std::ostringstream dtype_os;
    PrintType(PrimType(dtype), dtype_os);
    std::string dtype_str = dtype_os.str();
    simdgroup_dtype_[buffer.get()] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << constant_size / 64 << "];\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(PrimType(dtype), stream);
    stream << ' ' << vid << '[' << constant_size << "];\n";
  }

  RegisterHandleType(buffer.get(), PrimType(dtype));
  if (annotations->dict.count(tvm::tirx::attr::kVolatile.name)) {
    MarkVolatile(buffer.get());
  }
}

void CodeGenMetal::Dispatch_(const prim::SelectNode *op,
                             std::ostream &os) { // NOLINT(*)
  PrimType result_type = op->ty.as_or_throw<PrimType>();
  if (result_type.lanes() > 4) {
    std::string condition =
        SSAGetID(PrintExpr(op->condition), op->condition.ty());
    std::string true_value = SSAGetID(PrintExpr(op->true_value), result_type);
    std::string false_value = SSAGetID(PrintExpr(op->false_value), result_type);
    std::string result = name_supply_->FreshName("selected");
    PrintIndent();
    PrintType(result_type, stream);
    stream << " " << result << ";\n";
    for (int lane = 0; lane < result_type.lanes(); ++lane) {
      std::ostringstream value;
      value << '(';
      PrintVecElemLoad(condition, op->condition.ty(), lane, value);
      value << " ? ";
      PrintVecElemLoad(true_value, result_type, lane, value);
      value << " : ";
      PrintVecElemLoad(false_value, result_type, lane, value);
      value << ')';
      PrintVecElemStore(result, result_type, lane, value.str());
    }
    os << result;
    return;
  }
  os << "select(" << PrintExpr(op->false_value) << ", "
     << PrintExpr(op->true_value) << ", " << PrintExpr(op->condition) << ")";
}

void CodeGenMetal::Dispatch_(const prim::BroadcastNode *op,
                             std::ostream &os) { // NOLINT(*)
  PrimType result_type = op->ty.as_or_throw<PrimType>();
  if (result_type.lanes() > 4) {
    std::string value = SSAGetID(PrintExpr(op->value), op->value.ty());
    std::string result = name_supply_->FreshName("broadcast");
    PrintIndent();
    PrintType(result_type, stream);
    stream << " " << result << ";\n";
    for (int lane = 0; lane < result_type.lanes(); ++lane)
      PrintVecElemStore(result, result_type, lane, value);
    os << result;
    return;
  }
  std::string v = PrintExpr(op->value);
  int lanes = op->ty.as_or_throw<PrimType>().lanes();
  PrintType(op->ty.as_or_throw<PrimType>(), os);
  os << "(";
  for (int i = 0; i < lanes; ++i) {
    if (i != 0)
      os << ", ";
    os << v;
  }
  os << ')';
}

void CodeGenMetal::Dispatch_(const prim::CastNode *node, std::ostream &output) {
  if (TryPrintMlxCoordinate(ffi::GetRef<PrimExpr>(node), output))
    return;
  PrimType destination = node->ty.as_or_throw<PrimType>();
  PrimType source = node->value.ty();
  if (source.lanes() <= 4 && destination.lanes() <= 4) {
    CodeGenC::Dispatch_(node, output);
    return;
  }
  TVM_FFI_ICHECK_EQ(source.lanes(), destination.lanes());
  std::string input = SSAGetID(PrintExpr(node->value), source);
  std::string result = name_supply_->FreshName("cast_result");
  PrintIndent();
  PrintType(destination, stream);
  stream << " " << result << ";\n";
  for (int lane = 0; lane < destination.lanes(); ++lane) {
    std::ostringstream value;
    value << '(';
    PrintType(destination.WithLanes(1), value);
    value << ")(";
    PrintVecElemLoad(input, source, lane, value);
    value << ')';
    PrintVecElemStore(result, destination, lane, value.str());
  }
  output << result;
}

void CodeGenMetal::PrintVecBinaryOp(const std::string &operation,
                                    const PrimType &result_type, PrimExpr left,
                                    PrimExpr right, std::ostream &output) {
  if (result_type.lanes() <= 4 && left.ty().lanes() <= 4 &&
      right.ty().lanes() <= 4) {
    CodeGenC::PrintVecBinaryOp(operation, result_type, left, right, output);
    return;
  }
  std::string left_value = SSAGetID(PrintExpr(left), left.ty());
  std::string right_value = SSAGetID(PrintExpr(right), right.ty());
  std::string result = name_supply_->FreshName("vector_result");
  PrintIndent();
  PrintType(result_type, stream);
  stream << " " << result << ";\n";
  for (int lane = 0; lane < result_type.lanes(); ++lane) {
    std::ostringstream value;
    bool function_call = std::isalpha(operation[0]);
    if (function_call)
      value << operation;
    value << '(';
    PrintVecElemLoad(left_value, left.ty(), lane, value);
    value << (function_call ? ", " : " " + operation + " ");
    PrintVecElemLoad(right_value, right.ty(), lane, value);
    value << ')';
    PrintVecElemStore(result, result_type, lane, value.str());
  }
  output << result;
}

void CodeGenMetal::Dispatch_(const CallNode *op,
                             std::ostream &os) { // NOLINT(*)
  if (const auto *operation = op->op.as<OpNode>()) {
    if (operation->name == "tirx.metal.cooperative_tensor_fill") {
      PrintCooperativeFill(op, os);
      return;
    }
    if (operation->name == "tirx.metal.cooperative_tensor_load" ||
        operation->name == "tirx.metal.cooperative_tensor_store") {
      PrintCooperativeCopy(
          op, operation->name == "tirx.metal.cooperative_tensor_store", os);
      return;
    }
    if (operation->name ==
        "tirx.metal.cooperative_tensor_multiply_accumulate") {
      PrintCooperativeMultiply(op, os);
      return;
    }
  }
  if (op->op.same_as(tirx::atomic_add_op())) {
    TVM_FFI_CHECK_EQ(op->args.size(), 2, ValueError);
    PrimType value_type = op->args[1]->ty.as_or_throw<PrimType>();
    auto pointer_type = op->args[0]->ty.as_or_throw<PtrType>();
    TVM_FFI_CHECK(value_type.IsScalar() && value_type.bits() == 32 &&
                      value_type.MatchesCode(kDLInt, kDLUInt),
                  TypeError)
        << "Metal atomic add supports scalar int32 and uint32 values";
    TVM_FFI_CHECK(pointer_type->scope == runtime::MemoryScope::Global() ||
                      pointer_type->scope.name().starts_with("shared"),
                  ValueError)
        << "Metal atomic add requires device or threadgroup storage";
    os << "atomic_fetch_add_explicit(reinterpret_cast<";
    PrintStorageScope(pointer_type->scope, os);
    os << (value_type.MatchesCode(kDLInt) ? "atomic_int" : "atomic_uint")
       << " *>(";
    PrintExpr(op->args[0], os);
    os << "), ";
    PrintExpr(op->args[1], os);
    os << ", memory_order_relaxed)";
    return;
  }
  TVM_FFI_ICHECK(!op->op.as<GlobalVarNode>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls Function "
      << op->op;
  auto f_check_simdgroup_shape = [](PrimExpr col, PrimExpr row) {
    TVM_FFI_ICHECK(col->IsInstance<IntImmNode>() &&
                   row->IsInstance<IntImmNode>())
        << "Only constant shape is supported for simdgroup matrix, but got "
        << col << "x" << row;
    int col_val = col.as<IntImmNode>()->value.as<int>().value();
    int row_val = row.as<IntImmNode>()->value.as<int>().value();
    TVM_FFI_ICHECK(col_val == 8 && row_val == 8)
        << "Only 8x8 matrix is supported, but got " << col_val << "x"
        << row_val;
  };

  static const Op make_filled_simdgroup_matrix_op =
      Op::Get("tirx.metal.make_filled_simdgroup_matrix");
  static const Op simdgroup_load_op = Op::Get("tirx.metal.simdgroup_load");
  static const Op simdgroup_store_op = Op::Get("tirx.metal.simdgroup_store");
  static const Op simdgroup_multiply_accumulate_op =
      Op::Get("tirx.metal.simdgroup_multiply_accumulate");

  if (op->op.same_as(make_filled_simdgroup_matrix_op)) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 5);
    Var var = GetMatrixBufferVar(op->args[0]);
    // Get the data type of the simdgroup matrix
    auto it = simdgroup_dtype_.find(var.get());
    TVM_FFI_ICHECK(it != simdgroup_dtype_.end())
        << "Cannot find variable allocation for simdgroup: " << var;
    const std::string &dtype_str = it->second;
    f_check_simdgroup_shape(op->args[3].as_or_throw<PrimExpr>(),
                            op->args[4].as_or_throw<PrimExpr>());
    os << PrintExpr(var) << "[" << PrintExpr(op->args[1])
       << "] = make_filled_simdgroup_matrix<" << dtype_str << ", "
       << PrintExpr(op->args[3]) << ", " << PrintExpr(op->args[4]) << ">("
       << PrintExpr(op->args[2]) << ")";
  } else if (op->op.same_as(simdgroup_load_op)) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 7);
    Var var = GetMatrixBufferVar(op->args[0]);
    f_check_simdgroup_shape(op->args[4].as_or_throw<PrimExpr>(),
                            op->args[5].as_or_throw<PrimExpr>());
    os << "simdgroup_load(" << PrintExpr(var) << "[" << PrintExpr(op->args[1])
       << "], " << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3])
       << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(simdgroup_store_op)) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 7);
    Var var = GetMatrixBufferVar(op->args[0]);
    f_check_simdgroup_shape(op->args[4].as_or_throw<PrimExpr>(),
                            op->args[5].as_or_throw<PrimExpr>());
    os << "simdgroup_store(" << PrintExpr(var) << "[" << PrintExpr(op->args[1])
       << "], " << PrintExpr(op->args[2]) << ", " << PrintExpr(op->args[3])
       << ", 0, " << PrintExpr(op->args[6]) << ")";
  } else if (op->op.same_as(simdgroup_multiply_accumulate_op)) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 8);
    Var d = GetMatrixBufferVar(op->args[0]);
    Var a = GetMatrixBufferVar(op->args[2]);
    Var b = GetMatrixBufferVar(op->args[4]);
    Var c = GetMatrixBufferVar(op->args[6]);
    os << "simdgroup_multiply_accumulate("                       //
       << PrintExpr(d) << "[" << PrintExpr(op->args[1]) << "], " //
       << PrintExpr(a) << "[" << PrintExpr(op->args[3]) << "], " //
       << PrintExpr(b) << "[" << PrintExpr(op->args[5]) << "], " //
       << PrintExpr(c) << "[" << PrintExpr(op->args[7]) << "])";
  } else if (op->op.same_as(tirx::ptr_byte_offset_op()) ||
             op->op.same_as(tirx::handle_add_byte_offset_op())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 2U);
    const auto *pointer_type = op->ty.as<PtrTypeNode>();
    TVM_FFI_ICHECK(pointer_type) << "Metal pointer byte offsets must have a "
                                    "pointer result type, but got "
                                 << op->ty;

    os << "((";
    PrintStorageScope(pointer_type->scope, os);
    PrintType(pointer_type->element_type, os);
    os << "*)(((";
    PrintStorageScope(pointer_type->scope, os);
    os << "char*)";
    PrintExpr(op->args[0], os);
    os << ") + ";
    PrintExpr(op->args[1], os);
    os << "))";
  } else if (op->op.same_as(tirx::reinterpret_op())) {
    if (!op->ty.as<PrimTypeNode>() || !op->args[0]->ty.as<PrimTypeNode>()) {
      return CodeGenC::Dispatch_(op, os);
    }
    // generate as_type<TYPE>(ARG)
    os << "(as_type<";
    this->PrintType(op->ty.as_or_throw<PrimType>(), os);
    os << ">(";
    this->PrintExpr(op->args[0], os);
    os << "))";
  } else {
    CodeGenC::Dispatch_(op, os);
  }
}

void CodeGenMetal::Dispatch_(const FloatImmNode *op,
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
    if (op->ty.as_or_throw<PrimType>().bits() == 32)
      temp << 'f';
    else if (op->ty.as_or_throw<PrimType>().bits() == 16)
      temp << 'h';
  }
  MarkConst(temp.str());
  os << temp.str();
}

ffi::Module BuildMetal(IRModule mod, Target target) {
  bool output_ssa = false;
  mod = tl::RewritePointerValueTypes()(std::move(mod));

  std::ostringstream source_maker;
  // Per-kernel payload: kernel-name -> bytes (MSL source for fmt="metal" /
  // compiled metallib bytes for fmt="metallib").  See
  // tasks/...-tvm-unify-device-module... "Per-kernel smap shape" — uniform
  // Map<String, Bytes> across all multi-shader backends.
  ffi::Map<ffi::String, ffi::Bytes> smap;
  const auto compile_name = mod->GetAttr(tvm::tl::codegen_attr::kCompile)
                                .value_or("tvm_callback_metal_compile");
  auto fmetal_compile = tvm::ffi::Function::GetGlobal(compile_name);
  if (mod->GetAttr(tvm::tl::codegen_attr::kSourceOnly).value_or(false)) {
    fmetal_compile = std::nullopt;
  } else {
    TVM_FFI_CHECK(fmetal_compile ||
                      !mod->GetAttr(tvm::tl::codegen_attr::kCompile),
                  ValueError)
        << "Unknown Metal compilation callback " << compile_name;
  }
  const auto postproc_name = mod->GetAttr(tvm::tl::codegen_attr::kPostproc)
                                 .value_or("tvm_callback_metal_postproc");
  const auto fmetal_postproc = tvm::ffi::Function::GetGlobal(postproc_name);
  TVM_FFI_CHECK(fmetal_postproc ||
                    !mod->GetAttr(tvm::tl::codegen_attr::kPostproc),
                ValueError)
      << "Unknown Metal postprocessing callback " << postproc_name;
  // Default payload format. A callback may override it per the contract below.
  std::string fmt = fmetal_compile ? "metallib" : "metal";
  bool fmt_locked = false;

  for (auto kv : mod->functions) {
    TVM_FFI_ICHECK(kv.second->IsInstance<FunctionNode>())
        << "CodeGenMetal: Can only take Function";
    auto global_symbol = kv.second->GetAttr(tvm::attr::kGlobalSymbol);
    TVM_FFI_ICHECK(global_symbol.has_value());
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenMetal cg(target);
    cg.Init(output_ssa);
    cg.ConfigureModule(mod);
    auto f = kv.second.as_or_throw<Function>();
    auto calling_conv = f->GetAttr(tvm::attr::kCallingConv);
    TVM_FFI_ICHECK(calling_conv.has_value())
        << "CodeGenMetal: expected kCallingConv attribute to be set.";
    TVM_FFI_ICHECK(calling_conv.value() == CallingConv::kDeviceKernelLaunch)
        << "CodeGenMetal: expect calling_conv equals "
           "CallingConv::kDeviceKernelLaunch, but got "
        << static_cast<int>(calling_conv.value());

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    if (fmetal_postproc) {
      fsource = (*fmetal_postproc)(fsource, target).cast<std::string>();
    }
    source_maker << fsource << "\n";
    if (fmetal_compile) {
      ffi::Any ret = (*fmetal_compile)(fsource, target);
      // Backward-compatible contract for tvm_callback_metal_compile:
      //   * returning a str/bytes    -> treated as a compiled metallib payload
      //   * returning (payload, fmt) -> the callback declares the payload
      //   format
      std::string kernel_fmt;
      if (auto ret_tuple = ret.try_cast<ffi::Array<ffi::Any>>()) {
        TVM_FFI_ICHECK_EQ(ret_tuple->size(), 2)
            << "tvm_callback_metal_compile must return either a payload or a "
               "(payload, format) pair, but got a tuple of size "
            << ret_tuple->size();
        fsource = (*ret_tuple)[0].cast<std::string>();
        kernel_fmt = (*ret_tuple)[1].cast<std::string>();
        TVM_FFI_ICHECK(kernel_fmt == "metal" || kernel_fmt == "metallib")
            << "tvm_callback_metal_compile returned unsupported format \""
            << kernel_fmt << "\"; expected \"metal\" or \"metallib\"";
      } else {
        // Backward-compatible behavior
        fsource = ret.cast<std::string>();
        kernel_fmt = "metallib";
      }
      // All kernels of a module share a single declared format
      TVM_FFI_ICHECK(!fmt_locked || fmt == kernel_fmt)
          << "tvm_callback_metal_compile returned inconsistent formats across "
             "kernels: \""
          << fmt << "\" vs \"" << kernel_fmt << "\"";
      fmt = kernel_fmt;
      fmt_locked = true;
    }
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  // The aggregated MSL source dump is preserved in the in-memory source
  // map keyed by "metal" — only used by InspectSource and never serialized.
  ffi::Map<ffi::String, ffi::String> source;
  source.Set("metal", source_maker.str());
  return target::MetalModuleCreateWithFallback(
      std::move(smap), ffi::String(fmt), ExtractFuncInfo(mod),
      std::move(source));
}

void RegisterMetalCodegen() {
  static bool registered = false;
  if (registered)
    return;
  registered = true;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.codegen.build_metal", BuildMetal);
}
} // namespace tvm::tl::codegen_impl

TVM_FFI_STATIC_INIT_BLOCK() { tvm::tl::codegen_impl::RegisterMetalCodegen(); }
