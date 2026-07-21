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
 * \file metal/codegen/codegen_metal.cc
 */
#include "codegen_metal.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "runtime/thread_storage_scope.h"
#include "target/build_common.h"
#include "target/metal/metal_fallback_module.h"

namespace tvm {
namespace codegen {

namespace {

class CooperativeTensorUseCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const AllocBufferNode *op) final {
    if (op->buffer.scope() == "metal.cooperative_tensor") {
      uses_cooperative_tensor = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::cooperative_tensor_load()) ||
        op->op.same_as(tl::cooperative_tensor_store())) {
      uses_cooperative_tensor = true;
      needs_fragment_lane_vars = true;
    } else if (op->op.same_as(tl::cooperative_tensor_fill()) ||
               op->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
      uses_cooperative_tensor = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool uses_cooperative_tensor{false};
  bool needs_fragment_lane_vars{false};
};

} // namespace

void CodeGenTileLangMetal::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
  emitted_frag_lane_vars_ = false;
  emitted_metal_simdgroup_id_ = false;
  thread_idx_x_var_ = Var();
  block_idx_x_var_ = Var();
  block_idx_y_var_ = Var();
  active_mlx_swizzle_panel_ = 0;
  active_mlx_swizzle_log_ = 0;
  ct_c_inlined_.clear();
  ct_c_storage_elided_.clear();
  cooperative_tensor_dtype_.clear();
  simdgroup_dtype_.clear();
  CooperativeTensorUseCollector ct_collector;
  ct_collector(f->body);
  uses_cooperative_tensor_ = ct_collector.uses_cooperative_tensor;
  needs_fragment_lane_vars_ = ct_collector.needs_fragment_lane_vars;
  // analyze the data;
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      alloc_storage_scope_[arg.get()] = "global";
    }
  }
}

CodeGenTileLangMetal::CodeGenTileLangMetal(Target target) : target_(target) {
  restrict_keyword_ = "__restrict";
  decl_stream << "union __TVMArgUnion {\n"
              << " int v_int[2];\n"
              << "};\n\n";
}

std::string CodeGenTileLangMetal::Finish() {
  std::ostringstream code;
  code << "#include <metal_stdlib>\n";
  code << "#include <metal_simdgroup>\n";
  if (uses_cooperative_tensor_) {
    code << "#include "
            "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";
  }
  code << "#define TILELANG_PRAGMA_UNROLL _Pragma(\"clang loop "
          "unroll(full)\")\n";
  code << "using namespace metal;\n\n";
  code << decl_stream.str();
  code << fwd_decl_stream.str();
  code << stream.str();
  return code.str();
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
  int64_t max_total_threads_per_threadgroup = 0;
  if (auto opt_thread_extent =
          func->GetAttr<ffi::Map<ffi::String, PrimExpr>>("thread_extent")) {
    int64_t total_threads = 1;
    bool has_thread_extent = false;
    bool is_static_thread_extent = true;
    for (ffi::String tag :
         {ffi::String("threadIdx.x"), ffi::String("threadIdx.y"),
          ffi::String("threadIdx.z")}) {
      auto extent = opt_thread_extent.value().Get(tag);
      if (!extent.has_value()) {
        continue;
      }
      auto *extent_imm = extent.value().as<IntImmNode>();
      if (!extent_imm) {
        is_static_thread_extent = false;
        break;
      }
      total_threads *= extent_imm->value;
      has_thread_extent = true;
    }
    if (has_thread_extent && is_static_thread_extent && total_threads > 0) {
      max_total_threads_per_threadgroup = total_threads;
    }
  }
  if (max_total_threads_per_threadgroup > 0) {
    this->stream << "[[kernel, max_total_threads_per_threadgroup("
                 << max_total_threads_per_threadgroup << ")]] void ";
  } else {
    this->stream << "kernel void ";
  }
  this->stream << static_cast<std::string>(global_symbol.value()) << "(";

  // Buffer arguments
  size_t num_buffer = 0;
  size_t limit =
      target_->GetAttr<Integer>("max_function_args").value().IntValue();
  if (func->params.size() > limit) {
    LOG(WARNING) << "Probably you won't be able to execute your kernel due to "
                    "high number of "
                    "buffers in the kernel";
  }
  bool no_alias = func->HasNonzeroAttr(tirx::attr::kNoAlias);
  std::unordered_set<const VarNode *> non_restrict;
  if (auto opt =
          func->GetAttr<ffi::Array<tirx::Var>>(tl::attr::kNonRestrictParams)) {
    for (const tirx::Var &v : opt.value()) {
      non_restrict.insert(v.get());
    }
  }
  std::unordered_set<int> readonly_param_indices;
  if (auto opt =
          func->GetAttr<ffi::Array<Integer>>("tl.readonly_param_indices")) {
    for (const auto &idx : opt.value()) {
      readonly_param_indices.insert(static_cast<int>(idx->value));
    }
  }
  for (size_t i = 0; i < func->params.size(); ++i, ++num_buffer) {
    Var v = func->params[i];
    if (!v.dtype().is_handle())
      break;
    this->stream << "  ";
    std::string vid = AllocVarID(v.get());
    if (readonly_param_indices.count(static_cast<int>(i))) {
      this->stream << "const ";
    }
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
    if (no_alias && !non_restrict.count(v.get())) {
      PrintRestrict(v, this->stream);
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
    stream << " threadIdx [[thread_position_in_threadgroup]],\n";
    stream << "  uint __simd_group_id [[simdgroup_index_in_threadgroup]]\n";
    emitted_metal_simdgroup_id_ = true;
  }
  thread_work_dim_ = work_dim;

  // the function scope.
  stream << ") {\n";
  int func_scope = this->BeginScope();
  if (needs_fragment_lane_vars_) {
    EnsureFragmentLaneVars();
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
  if (iv->thread_tag == "threadIdx.x") {
    thread_idx_x_var_ = iv->var;
  } else if (iv->thread_tag == "blockIdx.x") {
    block_idx_x_var_ = iv->var;
  } else if (iv->thread_tag == "blockIdx.y") {
    block_idx_y_var_ = iv->var;
  }
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
  if ((t.is_float16() || t.is_bfloat16()) && lanes > 4 && lanes <= 8 &&
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
  } else if (t.is_bfloat16() && t.lanes() > 4) {
    os << "((thread bfloat*)(&" << vec << "))[" << i << "]";
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
  } else if (t.is_bfloat16() && t.lanes() > 4) {
    stream << "((thread bfloat*)(&" << vec << "))[" << i << "] = " << value
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
  } else if (scope == "local" || scope == "local.var") {
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
    } else if (func_name == "rasterization2DColumn") {
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
    } else if (func_name == "rasterization2DMLX") {
      TVM_FFI_ICHECK_EQ(panel_size & (panel_size - 1), 0)
          << "Metal MLX swizzle panel size must be a power of two, got "
          << panel_size;
      int swizzle_log = 0;
      while ((1 << swizzle_log) < panel_size) {
        swizzle_log++;
      }
      if (thread_work_dim_ <= 1) {
        stream << "const uint3 __physical_blockIdx = "
                  "uint3(blockIdx, 0u, 0u); ";
      } else if (thread_work_dim_ == 2) {
        stream << "const uint3 __physical_blockIdx = "
                  "uint3(blockIdx.x, blockIdx.y, 0u); ";
      } else {
        stream << "const uint3 __physical_blockIdx = "
                  "uint3(blockIdx.x, blockIdx.y, blockIdx.z); ";
      }
      stream << "const uint3 blockIdx = uint3("
             << "(__physical_blockIdx.x >> " << swizzle_log << "), "
             << "((__physical_blockIdx.y << " << swizzle_log
             << ") + (__physical_blockIdx.x & " << (panel_size - 1) << "u)), "
             << "__physical_blockIdx.z);\n";
      int old_panel = active_mlx_swizzle_panel_;
      int old_log = active_mlx_swizzle_log_;
      active_mlx_swizzle_panel_ = panel_size;
      active_mlx_swizzle_log_ = swizzle_log;
      this->VisitStmt(op->body);
      active_mlx_swizzle_panel_ = old_panel;
      active_mlx_swizzle_log_ = old_log;
      this->PrintIndent();
      stream << "}\n";
      return;
    } else {
      LOG(FATAL) << "Unknown Metal threadblock swizzle pattern `" << func_name
                 << "`";
    }
    this->VisitStmt(op->body);
    this->PrintIndent();
    stream << "}\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::VisitStmt_(const ForNode *op) {
  auto *min_imm = op->min.as<IntImmNode>();
  auto *ext_imm = op->extent.as<IntImmNode>();
  // Unroll small constant-bound loops at codegen time so loop variables become
  // IntImm constants. Persistent cooperative_tensor C uses named Metal CT
  // objects (__pct_c0, __pct_c1, ...), so its tile index must be constant.
  bool body_has_alloc = false;
  bool needs_ct_index_const = false;
  auto check_ct_index = [&](const PrimExpr &buffer, const PrimExpr &idx) {
    if (idx.as<IntImmNode>()) {
      return;
    }
    if (auto *var = buffer.as<tirx::VarNode>()) {
      Var var_ref = ffi::GetRef<Var>(var);
      needs_ct_index_const = ct_c_inlined_.count(var_ref) != 0 ||
                             cooperative_tensor_dtype_.count(var_ref) != 0;
    }
  };
  tirx::PostOrderVisit(op->body, [&](const ffi::ObjectRef &node) {
    if (node->IsInstance<tirx::AllocBufferNode>()) {
      body_has_alloc = true;
    }
    if (auto *call = node.as<tirx::CallNode>()) {
      if (call->op.same_as(tl::cooperative_tensor_fill()) ||
          call->op.same_as(tl::cooperative_tensor_load()) ||
          call->op.same_as(tl::cooperative_tensor_store()) ||
          call->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
        check_ct_index(call->args[0], call->args[1]);
      }
    }
  });
  if (ext_imm && min_imm && ext_imm->value >= 2 && ext_imm->value <= 4 &&
      !body_has_alloc && needs_ct_index_const) {
    int64_t start = min_imm->value;
    int64_t extent = ext_imm->value;
    for (int64_t i = 0; i < extent; i++) {
      ffi::Map<tirx::Var, PrimExpr> vmap;
      vmap.Set(op->loop_var, IntImm(op->loop_var->dtype, start + i));
      tirx::Stmt body = tirx::Substitute(op->body, vmap);
      // Substitute leaves expressions such as (0 * 2 + 1); fold them so
      // cooperative_tensor indices can become IntImm.
      arith::Analyzer analyzer;
      class ConstFolder : public tirx::StmtExprMutator {
      public:
        explicit ConstFolder(arith::Analyzer *analyzer) : analyzer_(analyzer) {}

        PrimExpr VisitExpr(const PrimExpr &expr) final {
          PrimExpr visited = tirx::StmtExprMutator::VisitExpr(expr);
          if (!visited.as<IntImmNode>() && !visited.as<tirx::VarNode>()) {
            return analyzer_->Simplify(visited);
          }
          return visited;
        }

      private:
        arith::Analyzer *analyzer_;
      };
      body = ConstFolder(&analyzer)(std::move(body));
      std::vector<const tirx::VarNode *> local_vars;
      tirx::PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
        if (auto *bind = node.as<tirx::BindNode>()) {
          local_vars.push_back(bind->var.get());
        } else if (auto *inner_for = node.as<tirx::ForNode>()) {
          local_vars.push_back(inner_for->loop_var.get());
        }
      });
      this->VisitStmt(body);
      for (auto *var : local_vars) {
        var_idmap_.erase(var);
      }
    }
    return;
  }
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
    uses_cooperative_tensor_ = true;
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
    cooperative_tensor_dtype_[op->buffer->data] = dtype_str;
    int elems_per_thread = constant_size / 32;
    bool can_inline_c = dtype_str == "float" && elems_per_thread >= 16 &&
                        elems_per_thread % 16 == 0;
    int num_c_tiles = can_inline_c ? elems_per_thread / 16 : 0;
    bool elide_c_storage = can_inline_c && num_c_tiles <= 4;
    if (!elide_c_storage) {
      stream << "thread " << dtype_str << " " << vid << '[' << elems_per_thread
             << "];\n";
    }
    if (can_inline_c) {
      ct_c_inlined_.insert(op->buffer->data);
      if (elide_c_storage) {
        ct_c_storage_elided_.insert(op->buffer->data);
      }
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
                  "half, float>()), float>();\n";
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
    simdgroup_dtype_[op->buffer->data] = dtype_str;
    stream << "simdgroup_" << dtype_str << "8x8 " << vid << '['
           << constant_size / 64 << "];\n";
  } else {
    PrintStorageScope(scope, stream);
    PrintType(dtype, stream);
    if (scope == "local.var") {
      PrimExpr init = tirx::make_const(op->buffer->dtype, 0);
      auto init_it = op->annotations.find(tl::attr::kLocalVarInit);
      if (init_it != op->annotations.end()) {
        PrimExpr user_init = Downcast<PrimExpr>((*init_it).second);
        if (!user_init.dtype().is_void() &&
            user_init.dtype() != op->buffer->dtype) {
          user_init = tirx::Cast(op->buffer->dtype, user_init);
        }
        init = user_init;
      }
      stream << ' ' << vid << " = " << PrintExpr(init) << ";\n";
    } else {
      stream << ' ' << vid << '[' << constant_size << "];\n";
    }
  }

  RegisterHandleType(op->buffer->data.get(), dtype);
}

void CodeGenTileLangMetal::VisitExpr_(const BufferLoadNode *op,
                                      std::ostream &os) {
  if (GetPtrStorageScope(op->buffer->data) == "local.var") {
    TVM_FFI_ICHECK_EQ(op->indices.size(), 1)
        << "Load from non-flat local.var not supported.";
    os << GetVarID(op->buffer->data.get());
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangMetal::VisitStmt_(const BufferStoreNode *op) {
  if (GetPtrStorageScope(op->buffer->data) == "local.var") {
    TVM_FFI_ICHECK_EQ(op->indices.size(), 1)
        << "Store to non-flat local.var not supported.";
    TVM_FFI_ICHECK(!op->predicate.defined())
        << "Predicated local.var store is not supported.";
    PrintIndent();
    stream << GetVarID(op->buffer->data.get()) << " = " << PrintExpr(op->value)
           << ";\n";
    return;
  }
  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangMetal::EnsureCooperativeTensorBuffer(const Var &var) {
  if (cooperative_tensor_dtype_.count(var) != 0) {
    return;
  }
  auto type_it = handle_data_type_.find(var.get());
  TVM_FFI_ICHECK(type_it != handle_data_type_.end())
      << "Cannot find variable allocation for cooperative_tensor: " << var;
  std::ostringstream dtype_os;
  PrintType(type_it->second, dtype_os);
  cooperative_tensor_dtype_[var] = dtype_os.str();
}

void CodeGenTileLangMetal::EnsureFragmentLaneVars() {
  if (emitted_frag_lane_vars_) {
    return;
  }
  this->PrintIndent();
  stream << "const ushort __lane = "
            "__metal_get_thread_index_in_simdgroup(ushort());\n";
  this->PrintIndent();
  stream << "const ushort __qid = __lane >> 2;\n";
  this->PrintIndent();
  stream << "const ushort __base_row = (__qid & 4) | ((__lane >> 1) & 3);\n";
  this->PrintIndent();
  stream << "const ushort __base_col = ((__qid & 2) | (__lane & 1)) * 4;\n";
  emitted_frag_lane_vars_ = true;
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
  if ((op->dtype.is_float16() || op->dtype.is_bfloat16()) && lanes > 4 &&
      lanes % 2 == 0) {
    os << "uint" << lanes / 2 << "(";
    for (int i = 0; i < lanes / 2; ++i) {
      if (i != 0)
        os << ", ";
      if (op->dtype.is_float16()) {
        os << "as_type<uint>(half2(" << v << ", " << v << "))";
      } else if (op->dtype.is_bfloat16()) {
        os << "as_type<uint>(bfloat2(" << v << ", " << v << "))";
      }
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

void CodeGenTileLangMetal::VisitExpr_(const AddNode *op,
                                      std::ostream &os) { // NOLINT(*)
  if (TryPrintMlxSwizzleExpr(ffi::GetRef<PrimExpr>(op), os)) {
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

void CodeGenTileLangMetal::VisitExpr_(const CastNode *op,
                                      std::ostream &os) { // NOLINT(*)
  std::ostringstream value;
  if (TryPrintMlxSwizzleExpr(op->value, value)) {
    os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
    return;
  }
  CodeGenC::VisitExpr_(op, os);
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

std::string
CodeGenTileLangMetal::GetPointeeTypeOf(const PrimExpr &ptr_expr,
                                       const std::string &fallback) {
  if (auto *call = ptr_expr.as<CallNode>()) {
    if (call->op.same_as(builtin::address_of())) {
      if (auto *load = call->args[0].as<BufferLoadNode>()) {
        std::ostringstream os;
        PrintType(load->buffer->dtype, os);
        return os.str();
      }
    }
    for (const auto &arg : call->args) {
      std::string result = GetPointeeTypeOf(arg, "");
      if (!result.empty()) {
        return result;
      }
    }
  }
  if (auto *var = ptr_expr.as<VarNode>()) {
    auto it = handle_data_type_.find(var);
    if (it != handle_data_type_.end()) {
      std::ostringstream os;
      PrintType(it->second, os);
      return os.str();
    }
  }
  return fallback;
}

bool CodeGenTileLangMetal::IsThreadIdxXExpr(const PrimExpr &expr) const {
  if (!thread_idx_x_var_.defined()) {
    return false;
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return IsThreadIdxXExpr(cast->value);
  }
  if (const auto *var = expr.as<VarNode>()) {
    return ffi::GetRef<Var>(var).same_as(thread_idx_x_var_);
  }
  return false;
}

bool CodeGenTileLangMetal::IsBlockIdxExpr(const PrimExpr &expr, int dim) const {
  const PrimExpr *current = &expr;
  while (const auto *cast = current->as<CastNode>()) {
    current = &cast->value;
  }
  const auto *var = current->as<VarNode>();
  if (!var) {
    return false;
  }
  Var var_ref = ffi::GetRef<Var>(var);
  if (dim == 0) {
    return block_idx_x_var_.defined() && var_ref.same_as(block_idx_x_var_);
  }
  if (dim == 1) {
    return block_idx_y_var_.defined() && var_ref.same_as(block_idx_y_var_);
  }
  return false;
}

bool CodeGenTileLangMetal::IsConstIntExpr(const PrimExpr &expr,
                                          int64_t value) const {
  const auto *imm = expr.as<IntImmNode>();
  return imm && imm->value == value;
}

bool CodeGenTileLangMetal::IsMlxPanelRemainderExpr(const PrimExpr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  int64_t mask = active_mlx_swizzle_panel_ - 1;
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::bitwise_and()) && call->args.size() == 2) {
      return (IsBlockIdxExpr(call->args[0], 0) &&
              IsConstIntExpr(call->args[1], mask)) ||
             (IsBlockIdxExpr(call->args[1], 0) &&
              IsConstIntExpr(call->args[0], mask));
    }
  }
  if (const auto *mod = expr.as<ModNode>()) {
    return IsBlockIdxExpr(mod->a, 0) &&
           IsConstIntExpr(mod->b, active_mlx_swizzle_panel_);
  }
  if (const auto *mod = expr.as<FloorModNode>()) {
    return IsBlockIdxExpr(mod->a, 0) &&
           IsConstIntExpr(mod->b, active_mlx_swizzle_panel_);
  }
  return false;
}

bool CodeGenTileLangMetal::IsMlxPanelRowExpr(const PrimExpr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  if (active_mlx_swizzle_log_ > 0) {
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(builtin::shift_left()) && call->args.size() == 2) {
        return IsBlockIdxExpr(call->args[0], 1) &&
               IsConstIntExpr(call->args[1], active_mlx_swizzle_log_);
      }
    }
  }
  if (const auto *mul = expr.as<MulNode>()) {
    return (IsBlockIdxExpr(mul->a, 1) &&
            IsConstIntExpr(mul->b, active_mlx_swizzle_panel_)) ||
           (IsBlockIdxExpr(mul->b, 1) &&
            IsConstIntExpr(mul->a, active_mlx_swizzle_panel_));
  }
  return active_mlx_swizzle_panel_ == 1 && IsBlockIdxExpr(expr, 1);
}

bool CodeGenTileLangMetal::IsMlxLogicalBlockXExpr(const PrimExpr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::shift_right()) && call->args.size() == 2) {
      return IsBlockIdxExpr(call->args[0], 0) &&
             IsConstIntExpr(call->args[1], active_mlx_swizzle_log_);
    }
  }
  if (const auto *div = expr.as<DivNode>()) {
    return IsBlockIdxExpr(div->a, 0) &&
           IsConstIntExpr(div->b, active_mlx_swizzle_panel_);
  }
  if (const auto *div = expr.as<FloorDivNode>()) {
    return IsBlockIdxExpr(div->a, 0) &&
           IsConstIntExpr(div->b, active_mlx_swizzle_panel_);
  }
  return active_mlx_swizzle_panel_ == 1 && IsBlockIdxExpr(expr, 0);
}

bool CodeGenTileLangMetal::IsMlxLogicalBlockYExpr(const PrimExpr &expr) const {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }
  if (IsMlxPanelRemainderExpr(expr)) {
    return true;
  }
  if (const auto *add = expr.as<AddNode>()) {
    return (IsMlxPanelRowExpr(add->a) && IsMlxPanelRemainderExpr(add->b)) ||
           (IsMlxPanelRowExpr(add->b) && IsMlxPanelRemainderExpr(add->a));
  }
  return false;
}

bool CodeGenTileLangMetal::TryPrintMlxLogicalYAffineExpr(const PrimExpr &expr,
                                                         std::ostream &os) {
  if (active_mlx_swizzle_panel_ <= 0) {
    return false;
  }

  std::vector<PrimExpr> terms;
  std::function<void(const PrimExpr &)> flatten_add =
      [&](const PrimExpr &current) {
        if (const auto *add = current.as<AddNode>()) {
          flatten_add(add->a);
          flatten_add(add->b);
        } else {
          terms.push_back(current);
        }
      };
  flatten_add(expr);
  if (terms.size() < 2) {
    return false;
  }

  enum class TermKind { kOther, kPanelRow, kRemainder };
  struct Term {
    TermKind kind{TermKind::kOther};
    int64_t coeff{1};
    PrimExpr expr;
    bool used{false};
  };

  auto strip_cast = [](PrimExpr current) {
    while (const auto *cast = current.as<CastNode>()) {
      current = cast->value;
    }
    return current;
  };

  auto parse_const = [](const PrimExpr &current, int64_t *value) {
    if (const auto *imm = current.as<IntImmNode>()) {
      *value = imm->value;
      return true;
    }
    return false;
  };

  std::vector<Term> classified;
  classified.reserve(terms.size());
  bool saw_panel_row = false;
  for (const PrimExpr &term_expr : terms) {
    Term term;
    term.expr = term_expr;
    PrimExpr base = strip_cast(term_expr);
    int64_t coeff = 1;
    if (const auto *mul = base.as<MulNode>()) {
      int64_t lhs_const = 0;
      int64_t rhs_const = 0;
      if (parse_const(mul->a, &lhs_const)) {
        coeff = lhs_const;
        base = strip_cast(mul->b);
      } else if (parse_const(mul->b, &rhs_const)) {
        coeff = rhs_const;
        base = strip_cast(mul->a);
      }
    }

    if (IsMlxPanelRowExpr(base)) {
      term.kind = TermKind::kPanelRow;
      term.coeff = coeff;
      saw_panel_row = true;
    } else if (IsBlockIdxExpr(base, 1) &&
               coeff % active_mlx_swizzle_panel_ == 0) {
      term.kind = TermKind::kPanelRow;
      term.coeff = coeff / active_mlx_swizzle_panel_;
      saw_panel_row = true;
    } else if (IsMlxPanelRemainderExpr(base)) {
      term.kind = TermKind::kRemainder;
      term.coeff = coeff;
    }
    classified.push_back(std::move(term));
  }

  std::vector<std::string> output_terms;
  for (size_t i = 0; i < classified.size(); ++i) {
    Term &row = classified[i];
    if (row.used || row.kind != TermKind::kPanelRow) {
      continue;
    }
    for (size_t j = 0; j < classified.size(); ++j) {
      Term &rem = classified[j];
      if (rem.used || rem.kind != TermKind::kRemainder ||
          rem.coeff != row.coeff) {
        continue;
      }
      std::ostringstream term_os;
      if (row.coeff == 1) {
        term_os << "blockIdx.y";
      } else {
        term_os << "(blockIdx.y * " << row.coeff << ")";
      }
      output_terms.push_back(term_os.str());
      row.used = true;
      rem.used = true;
      break;
    }
  }

  if (!saw_panel_row) {
    for (Term &term : classified) {
      if (term.used || term.kind != TermKind::kRemainder) {
        continue;
      }
      std::ostringstream term_os;
      if (term.coeff == 1) {
        term_os << "blockIdx.y";
      } else {
        term_os << "(blockIdx.y * " << term.coeff << ")";
      }
      output_terms.push_back(term_os.str());
      term.used = true;
    }
  }

  bool combined_mlx_y = !output_terms.empty();
  if (!combined_mlx_y) {
    return false;
  }
  for (const Term &term : classified) {
    if (term.used) {
      continue;
    }
    output_terms.push_back(PrintExpr(term.expr));
  }
  os << "(";
  for (size_t i = 0; i < output_terms.size(); ++i) {
    if (i != 0) {
      os << " + ";
    }
    os << output_terms[i];
  }
  os << ")";
  return true;
}

bool CodeGenTileLangMetal::TryPrintMlxSwizzleExpr(const PrimExpr &expr,
                                                  std::ostream &os) {
  if (TryPrintMlxLogicalYAffineExpr(expr, os)) {
    return true;
  }
  if (IsMlxLogicalBlockXExpr(expr)) {
    os << "blockIdx.x";
    return true;
  }
  if (IsMlxLogicalBlockYExpr(expr)) {
    os << "blockIdx.y";
    return true;
  }
  return false;
}

void CodeGenTileLangMetal::PrintSimdgroupIndexExpr(int64_t group_mask,
                                                   int64_t group_shift,
                                                   std::ostream &os) const {
  std::string base = "((int)__simd_group_id)";
  if (group_mask >= 0) {
    base = "(" + base + " & " + std::to_string(group_mask) + ")";
  }
  if (group_shift > 0) {
    os << "(" << base << " >> " << group_shift << ")";
  } else {
    os << base;
  }
}

bool CodeGenTileLangMetal::TryPrintSimdgroupIndexExpr(const CallNode *op,
                                                      std::ostream &os) {
  if (!emitted_metal_simdgroup_id_ || !op->op.same_as(builtin::shift_right()) ||
      op->args.size() != 2) {
    return false;
  }
  const auto *shift = op->args[1].as<IntImmNode>();
  if (!shift || shift->value < 5) {
    return false;
  }
  int64_t group_shift = shift->value - 5;
  if (IsThreadIdxXExpr(op->args[0])) {
    PrintSimdgroupIndexExpr(-1, group_shift, os);
    return true;
  }
  const auto *and_call = op->args[0].as<CallNode>();
  if (!and_call || !and_call->op.same_as(builtin::bitwise_and()) ||
      and_call->args.size() != 2) {
    return false;
  }
  int64_t mask = -1;
  if (IsThreadIdxXExpr(and_call->args[0])) {
    const auto *mask_imm = and_call->args[1].as<IntImmNode>();
    if (!mask_imm) {
      return false;
    }
    mask = mask_imm->value;
  } else if (IsThreadIdxXExpr(and_call->args[1])) {
    const auto *mask_imm = and_call->args[0].as<IntImmNode>();
    if (!mask_imm) {
      return false;
    }
    mask = mask_imm->value;
  } else {
    return false;
  }
  if (mask < 0 || (mask & 31) != 31) {
    return false;
  }
  PrintSimdgroupIndexExpr(mask >> 5, group_shift, os);
  return true;
}

void CodeGenTileLangMetal::VisitExpr_(const CallNode *op,
                                      std::ostream &os) { // NOLINT(*)
  TVM_FFI_ICHECK(!op->op.as<GlobalVarNode>())
      << "CodegenMetal does not support inter-function calls, "
      << "but expression " << ffi::GetRef<Call>(op) << " calls PrimFunc "
      << op->op;
  if (TryPrintSimdgroupIndexExpr(op, os)) {
    return;
  }
  if (TryPrintMlxSwizzleExpr(ffi::GetRef<PrimExpr>(op), os)) {
    return;
  }
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
    auto it = simdgroup_dtype_.find(var);
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
    bool is_inlined = ct_c_inlined_.count(fill_v) > 0;
    bool storage_elided = ct_c_storage_elided_.count(fill_v) > 0;
    auto *fill_idx_imm = op->args[1].as<IntImmNode>();
    if (is_inlined && fill_idx_imm) {
      os << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < "
         << elems_per_tile << "; __i++) "
         << "__pct_c" << fill_idx_imm->value << "[__i] = " << val;
    } else {
      TVM_FFI_ICHECK(!storage_elided)
          << "Elided cooperative tensor C storage requires constant fill index";
      os << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < "
         << elems_per_tile << "; __i++) " << var << "[" << idx << " * "
         << elems_per_tile << " + __i] = " << val;
    }
  } else if (op->op.same_as(tl::cooperative_tensor_load())) {
    uses_cooperative_tensor_ = true;
    TVM_FFI_ICHECK_GE(op->args.size(), 11);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string src_ptr = PrintExpr(op->args[2]);
    std::string stride = PrintExpr(op->args[3]);
    int rows = op->args[4].as<IntImmNode>()->value;
    int cols = op->args[5].as<IntImmNode>()->value;
    Var v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v);
    TVM_FFI_ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(op->args[2]);
    std::string src_dtype = GetPointeeTypeOf(op->args[2], dtype);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    int total_elems = nfrag_r * nfrag_c * 8;
    bool is_inlined = ct_c_inlined_.count(v) > 0;
    bool storage_elided = ct_c_storage_elided_.count(v) > 0;
    auto *load_idx_imm = op->args[1].as<IntImmNode>();
    std::string direct_c_load_name;
    if (storage_elided) {
      TVM_FFI_ICHECK(is_inlined && load_idx_imm)
          << "Elided cooperative tensor C storage requires constant load index";
      direct_c_load_name =
          std::string("__pct_c") + std::to_string(load_idx_imm->value);
    }
    std::string src_addr_space = "const " + addr_space;
    os << "{ " << src_addr_space << " " << src_dtype << "* __src = ("
       << src_addr_space << " " << src_dtype << "*)" << src_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(thread " << dtype << "4*)(&";
        if (!direct_c_load_name.empty()) {
          os << direct_c_load_name << "[" << elem_offset << "]";
        } else {
          os << var << "[" << idx << " * " << (nfrag_r * nfrag_c * 8) << " + "
             << elem_offset << "]";
        }
        os << ") = ";
        std::string load0 = std::string("*(") + src_addr_space + " " +
                            src_dtype + "4*)(&__src[__r0 * " + stride +
                            " + __c0])";
        if (src_dtype == dtype) {
          os << load0;
        } else {
          os << dtype << "4(" << load0 << ")";
        }
        os << "; "
           << "*(thread " << dtype << "4*)(&";
        if (!direct_c_load_name.empty()) {
          os << direct_c_load_name << "[" << (elem_offset + 4) << "]";
        } else {
          os << var << "[" << idx << " * " << (nfrag_r * nfrag_c * 8) << " + "
             << (elem_offset + 4) << "]";
        }
        os << ") = ";
        std::string load1 = std::string("*(") + src_addr_space + " " +
                            src_dtype + "4*)(&__src[__r1 * " + stride +
                            " + __c0])";
        if (src_dtype == dtype) {
          os << load1;
        } else {
          os << dtype << "4(" << load1 << ")";
        }
        os << "; } ";
        elem_offset += 8;
      }
    }
    os << "}";
    if (is_inlined && load_idx_imm && direct_c_load_name.empty()) {
      int mma_tiles_per_load = total_elems / 16;
      int base_pct = load_idx_imm->value * mma_tiles_per_load;
      for (int t = 0; t < mma_tiles_per_load; t++) {
        os << "; TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < 16; __i++) "
              "__pct_c"
           << (base_pct + t) << "[__i] = " << var << "[" << idx << " * "
           << total_elems << " + " << (t * 16) << " + __i];";
      }
    }
  } else if (op->op.same_as(tl::cooperative_tensor_store())) {
    uses_cooperative_tensor_ = true;
    TVM_FFI_ICHECK_GE(op->args.size(), 11);
    std::string var = PrintExpr(op->args[0]);
    std::string idx = PrintExpr(op->args[1]);
    std::string dst_ptr = PrintExpr(op->args[2]);
    std::string stride = PrintExpr(op->args[3]);
    int rows = op->args[4].as<IntImmNode>()->value;
    int cols = op->args[5].as<IntImmNode>()->value;
    Var v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(v);
    auto it = cooperative_tensor_dtype_.find(v);
    TVM_FFI_ICHECK(it != cooperative_tensor_dtype_.end());
    std::string dtype = it->second;
    std::string addr_space = GetAddrSpaceOf(op->args[2]);
    std::string dst_dtype = GetPointeeTypeOf(op->args[2], dtype);
    int frag_rows = 16, frag_cols = 16;
    int nfrag_r = rows / frag_rows;
    int nfrag_c = cols / frag_cols;
    int total_elems = nfrag_r * nfrag_c * 8;
    bool is_inlined = ct_c_inlined_.count(v) > 0;
    bool storage_elided = ct_c_storage_elided_.count(v) > 0;
    auto *store_idx_imm = op->args[1].as<IntImmNode>();
    TVM_FFI_ICHECK(!(storage_elided && !(is_inlined && store_idx_imm)))
        << "Elided cooperative tensor C storage requires constant store index";
    os << "{ " << addr_space << " " << dst_dtype << "* __dst = (" << addr_space
       << " " << dst_dtype << "*)" << dst_ptr << "; ";
    int elem_offset = 0;
    for (int fr = 0; fr < nfrag_r; fr++) {
      for (int fc = 0; fc < nfrag_c; fc++) {
        int row_off = fr * frag_rows;
        int col_off = fc * frag_cols;
        auto emit_store_value = [&](const std::string &value) {
          if (dst_dtype == dtype) {
            os << value;
          } else {
            os << dst_dtype << "4(" << value << ")";
          }
        };
        os << "{ "
           << "ushort __r0 = __base_row + " << row_off << "; "
           << "ushort __r1 = __r0 + 8; "
           << "ushort __c0 = __base_col + " << col_off << "; "
           << "*(" << addr_space << " " << dst_dtype << "4*)(&__dst[__r0 * "
           << stride << " + __c0]) = ";
        if (is_inlined && store_idx_imm) {
          int pct_idx =
              store_idx_imm->value * (total_elems / 16) + elem_offset / 16;
          int pct_elem = elem_offset % 16;
          std::string value0 = std::string("*(thread ") + dtype +
                               "4*)(&__pct_c" + std::to_string(pct_idx) + "[" +
                               std::to_string(pct_elem) + "])";
          std::string value1 = std::string("*(thread ") + dtype +
                               "4*)(&__pct_c" + std::to_string(pct_idx) + "[" +
                               std::to_string(pct_elem + 4) + "])";
          emit_store_value(value0);
          os << "; "
             << "*(" << addr_space << " " << dst_dtype << "4*)(&__dst[__r1 * "
             << stride << " + __c0]) = ";
          emit_store_value(value1);
          os << "; } ";
        } else {
          std::string value0 = std::string("*(thread ") + dtype + "4*)(&" +
                               var + "[" + idx + " * " +
                               std::to_string(total_elems) + " + " +
                               std::to_string(elem_offset) + "])";
          std::string value1 = std::string("*(thread ") + dtype + "4*)(&" +
                               var + "[" + idx + " * " +
                               std::to_string(total_elems) + " + " +
                               std::to_string(elem_offset + 4) + "])";
          emit_store_value(value0);
          os << "; "
             << "*(" << addr_space << " " << dst_dtype << "4*)(&__dst[__r1 * "
             << stride << " + __c0]) = ";
          emit_store_value(value1);
          os << "; } ";
        }
        elem_offset += 8;
      }
    }
    os << "}";
  } else if (op->op.same_as(tl::cooperative_tensor_multiply_accumulate())) {
    uses_cooperative_tensor_ = true;
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
    Var b_v = Downcast<Var>(op->args[4]);
    Var c_v = Downcast<Var>(op->args[0]);
    EnsureCooperativeTensorBuffer(a_v);
    EnsureCooperativeTensorBuffer(b_v);
    EnsureCooperativeTensorBuffer(c_v);
    auto a_it = cooperative_tensor_dtype_.find(a_v);
    auto b_it = cooperative_tensor_dtype_.find(b_v);
    auto c_it = cooperative_tensor_dtype_.find(c_v);
    TVM_FFI_ICHECK(a_it != cooperative_tensor_dtype_.end());
    TVM_FFI_ICHECK(b_it != cooperative_tensor_dtype_.end());
    TVM_FFI_ICHECK(c_it != cooperative_tensor_dtype_.end());
    std::string a_dtype = a_it->second;
    std::string b_dtype = b_it->second;
    std::string c_dtype = c_it->second;

    int a_elems = M * K / 32;
    int b_elems = K * N / 32;
    int c_elems = M * N / 32;

    TVM_FFI_ICHECK(M == 32 || N == 32 || K == 32)
        << "MPP matmul2d requires at least one of M, N, K to be 32, got " << M
        << "x" << N << "x" << K;

    bool c_inlined = ct_c_inlined_.count(c_v) > 0;
    bool c_storage_elided = ct_c_storage_elided_.count(c_v) > 0;
    auto *c_idx_imm = op->args[1].as<IntImmNode>();
    bool c_idx_const = c_inlined && c_idx_imm != nullptr;
    TVM_FFI_ICHECK(!(c_storage_elided && !c_idx_const))
        << "Elided cooperative tensor C storage requires constant MMA index";
    bool can_reuse_pct_op =
        c_idx_const && M == 16 && N == 32 && K == 16 && !trans_a && !trans_b;
    if (c_idx_const) {
      os << "{ ";
      if (can_reuse_pct_op) {
        os << "auto __ct_a = __pct_op.get_left_input_cooperative_tensor<"
           << a_dtype << ", " << b_dtype << ", " << c_dtype << ">(); "
           << "auto __ct_b = __pct_op.get_right_input_cooperative_tensor<"
           << a_dtype << ", " << b_dtype << ", " << c_dtype << ">(); ";
      } else {
        os << "constexpr auto __desc = mpp::tensor_ops::matmul2d_descriptor("
           << M << ", " << N << ", " << K << ", "
           << (trans_a ? "true" : "false") << ", "
           << (trans_b ? "true" : "false") << ", true, "
           << "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate)"
              "; "
           << "mpp::tensor_ops::matmul2d<__desc, metal::execution_simdgroup> "
              "__op; "
           << "auto __ct_a = __op.get_left_input_cooperative_tensor<" << a_dtype
           << ", " << b_dtype << ", " << c_dtype << ">(); "
           << "auto __ct_b = __op.get_right_input_cooperative_tensor<"
           << a_dtype << ", " << b_dtype << ", " << c_dtype << ">(); ";
      }
      os << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << a_elems
         << "; __i++) "
         << "__ct_a[__i] = " << a_var << "[" << a_idx << " * " << a_elems
         << " + __i]; "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << b_elems
         << "; __i++) "
         << "__ct_b[__i] = " << b_var << "[" << b_idx << " * " << b_elems
         << " + __i]; " << (can_reuse_pct_op ? "__pct_op" : "__op")
         << ".run(__ct_a, __ct_b, __pct_c" << c_idx_imm->value << "); }";
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
         << ", " << b_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_b = __op.get_right_input_cooperative_tensor<" << a_dtype
         << ", " << b_dtype << ", " << c_dtype << ">(); "
         << "auto __ct_c = __op.get_destination_cooperative_tensor<"
         << "decltype(__ct_a), decltype(__ct_b), " << c_dtype << ">(); "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << a_elems
         << "; __i++) "
         << "__ct_a[__i] = " << a_var << "[" << a_idx << " * " << a_elems
         << " + __i]; "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << b_elems
         << "; __i++) "
         << "__ct_b[__i] = " << b_var << "[" << b_idx << " * " << b_elems
         << " + __i]; "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << c_elems
         << "; __i++) "
         << "__ct_c[__i] = " << c_var << "[" << c_idx << " * " << c_elems
         << " + __i]; "
         << "__op.run(__ct_a, __ct_b, __ct_c); "
         << "TILELANG_PRAGMA_UNROLL for (ushort __i = 0; __i < " << c_elems
         << "; __i++) " << c_var << "[" << c_idx << " * " << c_elems
         << " + __i] = __ct_c[__i]; }";
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
  const auto fmetal_postproc =
      tvm::ffi::Function::GetGlobal("tilelang_callback_metal_postproc");
  std::string fmt = "metal";

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
    if (fmetal_postproc) {
      fsource = (*fmetal_postproc)(fsource, target).cast<std::string>();
    }
    smap.Set(func_name, ffi::Bytes(std::move(fsource)));
  }

  ffi::Map<ffi::String, ffi::String> source;
  source.Set("metal", source_maker.str());
  return tvm::target::MetalModuleCreateWithFallback(
      std::move(smap), ffi::String(fmt), ExtractFuncInfo(mod),
      std::move(source));
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

  ffi::Map<ffi::String, ffi::String> source;
  source.Set("metal", source_maker.str());
  return tvm::target::MetalModuleCreateWithFallback(
      std::move(smap), ffi::String("metal"), ExtractFuncInfo(mod),
      std::move(source));
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
