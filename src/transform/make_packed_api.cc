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
 * \file make_packed_api.cc
 * \brief Preserve TileLang output ownership around native packed workers.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../runtime/output_module.h"
#include "common/unused_tensor_alias.h"
#include "size_var_defaults.h"
#include "tensor_descriptor.h"
#include "tvm_ffi_binder.h"

namespace tvm::tl::transform {
namespace {
using tirx::Function;
using namespace tirx;

class TensorDescriptorRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit TensorDescriptorRewriter(
      ffi::Array<Var> params, std::unordered_map<const VarObj *, Var> handles)
      : params_(std::move(params)), handles_(std::move(handles)) {}

  UnchangedOr<Expr> Mutate_(const CallObj *call,
                            InplaceMode inplace_mode) final {
    if (!call->op.same_as(input_tensor_descriptor_op())) {
      return StmtExprMutator::Mutate_(call, inplace_mode);
    }
    int index = InputTensorDescriptorIndex(call, params_);
    auto found = handles_.find(params_[index].get());
    TVM_FFI_ICHECK(found != handles_.end())
        << "Missing decoded input descriptor for parameter " << index;
    return found->second;
  }

private:
  ffi::Array<Var> params_;
  std::unordered_map<const VarObj *, Var> handles_;
};

class ReturnRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) override {
    if (input.as<ExprObj>())
      return ffi::Unchanged();
    return StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit ReturnRewriter(Var ret_var) : ret_var_(ret_var) {}

  UnchangedOr<Stmt> Mutate_(const ForObj *node,
                            InplaceMode inplace_mode) override {
    // Unannotated single-iteration loops execute without a threadpool launch.
    bool is_parallel =
        node->kind == ForKind::kParallel &&
        !(prim::IsOne(Range::GetExtentFromBeginEnd(node->begin, node->end)) &&
          node->HasTrivialStep() && node->annotations.empty());
    if (is_parallel)
      in_parallel_ += 1;
    Stmt ret = StmtExprMutator::Mutate_(node, inplace_mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(node));
    if (is_parallel)
      in_parallel_ -= 1;
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *node,
                            InplaceMode inplace_mode) override {
    bool is_parallel = node->op.same_as(tirx::parallel_launch_op());
    if (is_parallel)
      in_parallel_ += 1;
    auto ret = StmtExprMutator::Mutate_(node, inplace_mode);
    if (is_parallel)
      in_parallel_ -= 1;
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const ReturnObj *node,
                            InplaceMode inplace_mode) override {
    TVM_FFI_ICHECK_EQ(in_parallel_, 0)
        << "Return cannot be used in parallel scope.";
    return WriteToOut(
        this->Mutate(node->value, inplace_mode).ValueOrUnchanged(node->value));
  }

private:
  struct ConvertedInfo {
    int type_index{-1};
    Expr expr;
  };

  ConvertedInfo ConvertForFFI(Expr val) {
    ConvertedInfo info{-1, val};

    if (IsVoidType(val->ty)) {
      info.type_index = ffi::TypeIndex::kTVMFFINone;
      info.expr = IntImm::Int64(0);
      return info;
    }

    // convert val's data type to FFI data type, return type code
    if (val->ty.as<PtrTypeObj>()) {
      info.type_index = ffi::TypeIndex::kTVMFFIOpaquePtr;
      info.expr = val;
      return info;
    }

    PrimExpr prim_val = val.as_or_throw<PrimExpr>();
    PrimType dtype = prim_val.ty();
    if (dtype.MatchesCode(DLDataTypeCode::kDLBool)) {
      info.type_index = ffi::TypeIndex::kTVMFFIBool;
      info.expr = prim::Cast(PrimType::Int(64), prim_val);

    } else if (dtype.MatchesCode(DLDataTypeCode::kDLInt,
                                 DLDataTypeCode::kDLUInt)) {
      info.type_index = ffi::TypeIndex::kTVMFFIInt;
      info.expr = prim::Cast(PrimType::Int(64), prim_val);
    } else if (dtype.code() == DLDataTypeCode::kDLFloat) {
      info.type_index = ffi::TypeIndex::kTVMFFIFloat;
      info.expr = prim::Cast(PrimType::Float(64), prim_val);
    } else if (dtype.IsVoid()) {
      info.type_index = ffi::TypeIndex::kTVMFFINone;
      info.expr = IntImm::Int64(0);
    } else {
      TVM_FFI_THROW(InternalError)
          << "data type " << dtype->dtype << " not supported yet";
    }
    return info;
  }

  Stmt WriteToOut(Expr val) {
    auto info = ConvertForFFI(val);
    Stmt store_tindex =
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {ret_var_, IntImm::Int32(0),
                       IntImm::Int32(tirx::kTVMFFIAnyTypeIndex),
                       IntImm::Int32(info.type_index)})
                     .as_or_throw<PrimExpr>());
    Stmt store_zero_padding = Evaluate(
        Call(PrimType::Int(32), tirx::abi_field_set_op(),
             {ret_var_, IntImm::Int32(0),
              IntImm::Int32(tirx::kTVMFFIAnyZeroPadding), IntImm::Int32(0)})
            .as_or_throw<PrimExpr>());
    Stmt store_val =
        Evaluate(Call(PrimType::Int(32), tirx::abi_field_set_op(),
                      {ret_var_, IntImm::Int32(0),
                       IntImm::Int32(tirx::kTVMFFIAnyUnionValue), info.expr})
                     .as_or_throw<PrimExpr>());
    Stmt ret_zero = Return(IntImm::Int32(0));
    // Return(call_returning_void()) still evaluates the call. The FFI value
    // slot stores zero, never a void expression or the empty Tuple object.
    if (info.type_index == ffi::TypeIndex::kTVMFFINone &&
        !(val.as<TupleObj>() && val.as<TupleObj>()->fields.empty())) {
      return SeqStmt({Evaluate(val), store_tindex, store_zero_padding,
                      store_val, ret_zero});
    }
    return SeqStmt({store_tindex, store_zero_padding, store_val, ret_zero});
  }

  Var ret_var_;
  int in_parallel_{0};
};

class SubroutineCallRewriter : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static ffi::Optional<Stmt>
  Apply(const ffi::Map<GlobalVar, ffi::String> &packed_func_methods,
        Stmt stmt) {
    auto rewriter =
        ffi::make_object<SubroutineCallRewriter>(packed_func_methods);
    stmt =
        rewriter->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    if (rewriter->made_change_) {
      return stmt;
    } else {
      return std::nullopt;
    }
  }

public:
  explicit SubroutineCallRewriter(
      const ffi::Map<GlobalVar, ffi::String> &packed_func_methods)
      : packed_func_methods(packed_func_methods) {}

private:
  UnchangedOr<Expr> Mutate_(const CallObj *op,
                            InplaceMode inplace_mode) override {
    auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();

    if (auto *gvar_ptr = node->op.as<GlobalVarObj>()) {
      auto gvar = ffi::GetRef<GlobalVar>(gvar_ptr);
      if (auto symbol = packed_func_methods.Get(gvar)) {
        ffi::Array<Expr> cpacked_args;
        cpacked_args.push_back(StringImm(symbol.value()));
        for (const Expr &arg : node->args) {
          cpacked_args.push_back(arg);
        }

        // The native C packed ABI passes self_handle separately. All operands
        // after the symbol are actual arguments; no trailing handle is needed.
        made_change_ = true;
        return Call(node->ty, tirx::call_cpacked_op(), cpacked_args);
      }
    }

    return node;
  }
  const ffi::Map<GlobalVar, ffi::String> &packed_func_methods;
  bool made_change_{false};
};

ffi::Optional<ffi::String> RequiresPackedAPI(const Function &func) {
  // Native declarations have a real, empty SeqStmt. They are not definitions
  // to rewrite into a packed wrapper or to include in the source host graph.
  if (func->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return std::nullopt;
  // A function with an explicit calling convention has already been
  // lowered, and should not be modified.
  if (auto opt = func->GetAttr(tvm::attr::kCallingConv)) {
    if (opt.value() != CallingConv::kDefault) {
      return std::nullopt;
    }
  }

  // Internal function calls do not need the ffi::Function API
  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  if (!global_symbol.has_value()) {
    return std::nullopt;
  }

  return global_symbol.value();
}

Function MakePackedFunction(Function func) {
  auto global_symbol = RequiresPackedAPI(func);
  if (!global_symbol.has_value()) {
    return StripSizeVarDefaults(std::move(func));
  }
  std::string name_hint = global_symbol.value();

  Target target = [&]() {
    auto opt = func->GetAttr(tvm::attr::kTarget);
    TVM_FFI_ICHECK(opt) << "MakePackedAPI required the function to be "
                           "annotated with tvm::attr::kTarget.name ("
                        << tvm::attr::kTarget.name
                        << "), but the function only has attributes "
                        << func->attrs;
    return opt.value();
  }();
  int target_device_type = target->GetTargetDeviceType();

  // A function without a host target has already been lowered.
  auto opt_target_host = target->GetHost();
  if (!opt_target_host)
    return StripSizeVarDefaults(std::move(func));
  Target target_host = opt_target_host.value();

  func = LowerTensorDescriptorParameters(std::move(func));
  auto size_var_defaults = CollectSizeVarDefaults(func);
  func = StripSizeVarDefaults(std::move(func));
  auto *func_ptr = func.CopyOnWrite();
  // Layout preparation can introduce pure parameter aliases even when the
  // source body uses only shape values. Direct packed callers do not always
  // pass through host/device splitting, which performs the same cleanup.
  func_ptr->body = RemoveUnusedTensorAliases(func_ptr->body);

  // Data field definitions
  Var v_self_handle("self_handle", PtrType::VoidPtrType());
  Var v_packed_args("args", PtrType::VoidPtrType());
  Var v_num_packed_args("num_args", PrimType::Int(32));
  Var v_result("result", PtrType::VoidPtrType());

  // The device context
  PrimVar device_id("dev_id");
  IntImm device_type(PrimType::Int(32), target_device_type);

  // Create TVMFFIABIBuilder and decode all packed args
  TVMFFIABIBuilder binder(name_hint, func_ptr->params, v_packed_args,
                          v_num_packed_args, device_type,
                          device_id.as_or_throw<PrimExpr>());
  // TileLang owns its established argument and storage conventions. These
  // choices are private to this binder, not attributes on TIRx functions.
  binder.SetTensorArgumentErrorKind("RuntimeError");
  binder.SetTypeIndexErrorKind("RuntimeError");
  binder.AllowPackedSubByteTensors();
  binder.AllowFP8DTypeAliases();
  binder.AllowTF32StorageAlias();
  binder.SetSizeVarDefaults(std::move(size_var_defaults));
  binder.AllowNullUnusedTensors(func_ptr->body);
  binder.DecodeAllParams();

  auto result = binder.Finalize();
  bool need_set_device = result.var_defs.count(device_id.get());

  // signature: (void* handle, TVMFFIAny* packed_args, int num_args, TVMFFIAny*
  // v_result)
  ffi::Array<Var> args{v_self_handle, v_packed_args, v_num_packed_args,
                       v_result};

  // reset global symbol to attach prefix
  func =
      WithAttrs(std::move(func),
                {{tvm::attr::kCallingConv.name, CallingConv::kCPackedFunc},
                 {tvm::attr::kTarget.name, target_host},
                 {tvm::attr::kGlobalSymbol.name,
                  ffi::symbol::tvm_ffi_symbol_prefix + global_symbol.value()}});

  Stmt body = ffi::make_object<TensorDescriptorRewriter>(
                  func_ptr->params, std::move(result.tensor_handles))
                  ->Mutate(func_ptr->body, InplaceMode::kDisallow)
                  .ValueOrUnchanged(func_ptr->body);
  body = ffi::make_object<ReturnRewriter>(v_result)
             ->Mutate(body, InplaceMode::kAllow)
             .ValueOrUnchanged(body);
  body =
      RegionStmt(tirx::compute_scope_op(), {StringImm(name_hint + "_compute_")},
                 {}, DictAttrs(), body);
  // Set device context
  if (need_set_device) {
    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      PrimVar status("set_device_status", PrimType::Int(32));
      Stmt set_device = Bind(
          status,
          Call(PrimType::Int(32), tirx::call_extern_op(),
               {StringImm("TVMBackendSetDevice"), device_type, device_id}));
      Stmt check_status =
          If(status != IntImm::Int32(0),
             SeqStmt({Evaluate(
                 Call(PrimType::Void(), tirx::throw_last_error_op(), {}))}));
      body = SeqStmt({set_device, check_status, body});
    }
  }

  // Return error code of zero on success
  body = SeqStmt({body, Return(IntImm::Int32(0))});

  // Prepend tensor declarations and alignment assumptions.
  result.decl_tensors.push_back(body);
  body = SeqStmt(result.decl_tensors);
  if (!result.asserts.empty()) {
    result.asserts.push_back(body);
    body = SeqStmt(result.asserts);
  }
  if (need_set_device) {
    body = RegionStmt(tirx::device_context_op(), {device_type, device_id}, {},
                      DictAttrs(), body);
  }
  if (!result.init_nest.empty()) {
    result.init_nest.push_back(body);
    body = SeqStmt(result.init_nest);
  }
  func_ptr->body = body;
  func_ptr->params = args;

  ffi::Array<Var> undefined = UndefinedVars(func_ptr->body, func_ptr->params);
  TVM_FFI_ICHECK_EQ(undefined.size(), 0)
      << "In Function " << name_hint << " variables " << undefined
      << " are used, but are not passed in as API arguments";

  func_ptr->ret_type = PrimType::Int(32);
  func_ptr->ty = func_ptr->func_type_annotation();

  // return the function.
  return func;
}

IRModule LowerPackedFunctions(IRModule mod) {
  ffi::Map<GlobalVar, ffi::String> packed_func_methods;
  for (const auto &[gvar, base_func] : mod->functions) {
    if (auto opt = base_func.as<Function>()) {
      auto function = opt.value();
      if (auto global_symbol = RequiresPackedAPI(function)) {
        packed_func_methods.Set(gvar, global_symbol.value());
      }
    }
  }

  IRModuleObj *mptr = mod.CopyOnWrite();
  IRModule updates;

  for (const auto &[gvar, base_func] : mptr->functions) {
    if (auto opt = base_func.as<Function>()) {
      auto func = opt.value();
      if (func->GetAttr(tvm::attr::kDeclaration).value_or(false))
        continue;
      auto orig_func = func;

      if (auto body =
              SubroutineCallRewriter::Apply(packed_func_methods, func->body)) {
        func.CopyOnWrite()->body = body.value();
      }

      func = MakePackedFunction(std::move(func));

      if (!func.same_as(orig_func)) {
        updates->Add(gvar, func);
      }
    }
  }

  if (updates->functions.size()) {
    mod.CopyOnWrite()->Update(updates);
  }
  return mod;
}

// A retained host graph owns its GlobalVar definitions independently of the
// packed module. Packing and device filtering can remove or rename the public
// entry; a bare Map would then leave unbound metadata keys. Use a module and
// rename all internal references by identity, including function attributes.
IRModule SourceHostModule(
    const ffi::Map<GlobalVar, tirx::Function> &functions,
    const ffi::Map<ffi::String, ffi::Array<GlobalInfo>> &global_infos) {
  ffi::Map<GlobalVar, GlobalVar> globals;
  for (const auto &[global, function] : functions) {
    globals.Set(global,
                GlobalVar(ffi::make_object<GlobalVarObj>(*global.get())));
  }
  auto remap = [&globals](const GlobalVar &global) {
    return globals.Get(global).value_or(global);
  };
  IRModule source;
  source.CopyOnWrite()->global_infos = global_infos;
  for (const auto &[global, function] : functions) {
    auto rewritten =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(function, remap)
            .as_or_throw<tirx::Function>();
    rewritten.CopyOnWrite()->attrs =
        ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(function->attrs, remap)
            .as_or_throw<DictAttrs>();
    source->Add(globals.at(global), rewritten);
  }
  return source;
}

// Output allocation and compute share native TensorType contracts. Separating
// them into two typed workers reuses the native binder for both input shapes
// and the allocator's actual output descriptors, including strides/offsets.
// A C++ frame encloses both calls so early returns never bypass owned cleanup.
tirx::Function AllocationWorker(const tirx::Function &function,
                                OutputEntrySpec *entry) {
  ffi::Array<Var> inputs;
  std::unordered_set<int> output_indices;
  for (const auto &slot : entry->outputs)
    output_indices.insert(slot.parameter_index);
  for (size_t i = 0; i < function->params.size(); ++i) {
    if (!output_indices.count(static_cast<int>(i)))
      inputs.push_back(function->params[i]);
  }
  Var frame("output_frame", PtrType::VoidPtrType());
  inputs.push_back(frame);
  ffi::Array<Stmt> body;
  auto resolver =
      ffi::Function::GetGlobal("tl.tvm_ffi.resolve_output_storage_dtype");
  TVM_FFI_CHECK(resolver.has_value(), RuntimeError)
      << "Missing output storage dtype resolver";
  for (size_t i = 0; i < entry->outputs.size(); ++i) {
    auto &slot = entry->outputs[i];
    auto tensor =
        function->params[slot.parameter_index].as_or_throw<tirx::TensorVar>();
    PrimType dtype = tensor.dtype();
    slot.storage_dtype = (*resolver)(dtype->dtype).cast<DLDataType>();
    int logical_bits = dtype.bits() * dtype.lanes();
    int storage_bits = slot.storage_dtype.bits * slot.storage_dtype.lanes;
    TVM_FFI_CHECK(logical_bits > 0 && storage_bits >= logical_bits &&
                      storage_bits % logical_bits == 0,
                  ValueError)
        << "Incompatible output storage for " << tensor->name;
    int factor = storage_bits / logical_bits;
    TVM_FFI_CHECK(factor == 1 || !tensor.shape().empty(), ValueError)
        << "Packed output dtype requires at least one dimension";
    ffi::Array<Expr> args{StringImm("tl.runtime.AllocateOutput"), frame,
                          IntImm::Int32(static_cast<int>(i))};
    for (size_t dim = 0; dim < tensor.shape().size(); ++dim) {
      PrimExpr extent = prim::Cast(PrimType::Int(64), tensor.shape()[dim]);
      body.push_back(AssertStmt(extent >= 0, StringImm("ValueError"),
                                {StringImm("Negative output dimension")}));
      if (dim + 1 == tensor.shape().size() && factor > 1) {
        auto packing = IntImm::Int64(factor);
        body.push_back(AssertStmt(
            prim::FloorMod(extent, packing) == 0, StringImm("RuntimeError"),
            {StringImm("Packed output dimension is not storage-aligned")}));
        extent = prim::FloorDiv(extent, packing);
      }
      args.push_back(extent);
    }
    body.push_back(
        Evaluate(Call(PrimType::Int(32), tirx::call_packed_op(), args)));
  }
  auto attrs = DictAttrs(ffi::Map<ffi::String, ffi::Any>{
      {"global_symbol", ffi::String(entry->allocate_name)},
      {"target", function->GetAttr(tvm::attr::kTarget).value()},
      {"tirx.noalias", false}});
  return WithSizeVarDefaults(tirx::Function(inputs, SeqStmt(body),
                                            PrimType::Void(), attrs,
                                            function->loc),
                             CollectSizeVarDefaults(function));
}
} // namespace

tvm::transform::Pass MakePackedAPI() {
  auto pass = [](IRModule module, tvm::transform::PassContext context) {
    IRModule updates;
    // Source adapters consume the typed launch graph, before packed workers
    // replace its tensor parameters with TVMFFIAny slots. Keep native Functions
    // in module metadata so serialization retains variable identity, lexical
    // control flow, and the actual argument order selected by launch lowering.
    ffi::Map<GlobalVar, tirx::Function> source_hosts;
    for (const auto &[global, base] : module->functions) {
      auto function = base.as<tirx::Function>();
      auto target = base->GetAttr(tvm::attr::kTarget);
      // Keep private helpers in the source host call graph even though only
      // its public entry needs packed-argument policies below.
      if (function &&
          !(*function)->GetAttr(tvm::attr::kDeclaration).value_or(false) &&
          target && (*target)->GetHost() &&
          base->GetAttr(tvm::attr::kCallingConv)
                  .value_or(CallingConv::kDefault) == CallingConv::kDefault) {
        source_hosts.Set(global, StripSizeVarDefaults(*function));
      }
    }
    // A source launcher consumes this typed graph directly. Requiring its
    // unused DLPack wrapper would reject valid raw-pointer contracts, such as
    // an odd int4 element offset which has no whole-byte descriptor offset.
    // The lowering entry selects this only when native host codegen is absent.
    if (module->GetAttr<bool>("tl.source_host_only").value_or(false)) {
      IRModule stripped;
      for (const auto &[global, base] : module->functions) {
        if (auto function = base.as<tirx::Function>()) {
          auto rewritten = StripSizeVarDefaults(*function);
          if (!rewritten.same_as(*function))
            stripped->Add(global, rewritten);
        }
      }
      if (!stripped->functions.empty())
        module.CopyOnWrite()->Update(stripped);
      if (!source_hosts.empty()) {
        module = WithAttr(module, "tl.source_host_module",
                          SourceHostModule(source_hosts, module->global_infos));
        module = WithAttr(module, "tl.source_host_schema", int64_t{2});
      }
      return module;
    }
    std::vector<OutputEntrySpec> entries;
    std::unordered_set<std::string> symbols;
    for (const auto &[global, base] : module->functions) {
      if (auto symbol = base->GetAttr(tvm::attr::kGlobalSymbol))
        symbols.insert(*symbol);
    }
    for (const auto &[global, base] : module->functions) {
      auto maybe_function = base.as<tirx::Function>();
      if (!maybe_function)
        continue;
      tirx::Function function = *maybe_function;
      if (function->GetAttr(tvm::attr::kDeclaration).value_or(false) ||
          !function->GetAttr<bool>("tilelang_callee_allocated_outputs")
               .value_or(false))
        continue;
      auto symbol = function->GetAttr(tvm::attr::kGlobalSymbol);
      auto target = function->GetAttr(tvm::attr::kTarget);
      if (!symbol || !target || !(*target)->GetHost() ||
          function->GetAttr(tvm::attr::kCallingConv)
                  .value_or(CallingConv::kDefault) != CallingConv::kDefault)
        continue;
      auto output_attr =
          function->GetAttr<ffi::Array<int64_t>>("tilelang_out_idx");
      if (!output_attr || output_attr->empty())
        continue;
      OutputEntrySpec entry;
      entry.public_name = *symbol;
      entry.allocate_name = "__tl_output_allocate_" + entry.public_name;
      entry.compute_name = "__tl_output_compute_" + entry.public_name;
      entry.parameter_count = static_cast<int>(function->params.size());
      entry.device_type = (*target)->GetTargetDeviceType();
      entry.is_main = function->HasNonzeroAttr(tirx::attr::kIsEntryFunc.name);
      TVM_FFI_CHECK(symbols.insert(entry.allocate_name).second &&
                        symbols.insert(entry.compute_name).second,
                    ValueError)
          << "Output worker symbol collides with an existing export";
      std::unordered_set<int> indices;
      for (int64_t raw : *output_attr) {
        int64_t index = raw < 0 ? raw + entry.parameter_count : raw;
        TVM_FFI_CHECK(index >= 0 && index < entry.parameter_count &&
                          indices.insert(static_cast<int>(index)).second,
                      ValueError)
            << "Invalid or duplicate output parameter " << raw;
        auto tensor = function->params[index].as<tirx::TensorVar>();
        TVM_FFI_CHECK(tensor.has_value(), TypeError)
            << "Output parameter must be a tensor";
        entry.outputs.push_back({static_cast<int>(index), DLDataType{},
                                 static_cast<int>(tensor->shape().size())});
      }
      auto allocator = AllocationWorker(function, &entry);
      updates->Add(GlobalVar(entry.allocate_name), allocator);
      function =
          WithAttr(function, "global_symbol", ffi::String(entry.compute_name));
      function = WithoutAttr(function, tirx::attr::kIsEntryFunc);
      function = WithoutAttr(function, "tilelang_callee_allocated_outputs");
      updates->Add(global, function);
      entries.push_back(std::move(entry));
    }
    if (!updates->functions.empty())
      module.CopyOnWrite()->Update(updates);
    module = LowerPackedFunctions(std::move(module));
    if (!source_hosts.empty()) {
      module = WithAttr(module, "tl.source_host_module",
                        SourceHostModule(source_hosts, module->global_infos));
      module = WithAttr(module, "tl.source_host_schema", int64_t{2});
    }
    return entries.empty() ? module
                           : WithAttr(module, kOutputModuleSchemaAttr,
                                      EncodeOutputSchema(entries));
  };
  return tvm::transform::CreateModulePass(pass, 0, "tl.MakePackedAPI");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MakePackedAPI", MakePackedAPI);
}
} // namespace tvm::tl::transform
