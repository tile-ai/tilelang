/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file split_host_device.cc
 * \brief Split device function from host.
 */
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../op/builtin.h"
#include "common/assume.h"
#include "common/attr.h"
#include "tir/analysis/var_use_def_analysis.h"
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {
using namespace ffi;
using namespace tirx;

// This pass traverses the AST, split the target function into host part and
// device part and copies all assume attribute statements to the device side.

// 1. Traverse AST and collect all assume statements into host_assumes_.
// 2. Until the first AttrStmtNode with tvm::attr::kTarget.
// 3. Call SplitDeviceFunc, which will create a new device function and replace
//    the original body with a call to that function.
class HostDeviceSplitter : public tirx::StmtMutator {
public:
  explicit HostDeviceSplitter(IRModule *device_mod,
                              std::function<GlobalVar()> var_supply)
      : device_mod_(device_mod), var_supply_(std::move(var_supply)) {}

  void SetNonRestrictParams(const Array<tirx::Var> &params) {
    for (auto param : params) {
      non_restrict_params_.push_back(param);
    }
  }

  void SetClusterDims(Array<Integer> cluster_dims) {
    cluster_dims_ = std::move(cluster_dims);
  }

  void SetHostFuncSignature(const tirx::PrimFunc &func) {
    host_buffer_map_ = func->buffer_map;
  }

  tirx::Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      found_device_region_ = true;
      auto device_target = op->node.as<tvm::Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    } else if (op->attr_key == tirx::attr::tilelang_assume) {
      // NOTE(chaofan): the assumes collected here must be in host-side.
      //    This is because when the collector reaches the split region,
      //    it will start to split and return. For safety, we add a check here.
      ICHECK(!found_device_region_)
          << "Assumes collection should not be in device region.";
      // We first push back the outside assume, then visit the child.
      // So when moving assumes to device side, we need to do the building
      // process in a reverse order.
      host_assumes_.push_back(op);
    }
    return tirx::StmtMutator::VisitStmt_(op);
  }

  tirx::Stmt VisitStmt_(const tirx::EvaluateNode *op) final {
    auto stmt = GetRef<tirx::Stmt>(op);
    // There should be no assume in evaluate form after InjectAssumes.
    ICHECK(!IsAssumeInEvaluateForm(stmt))
        << "Unexpected assume in evaluate form. Please run InjectAssumes pass "
           "first.";
    return tirx::StmtMutator::VisitStmt_(op);
  }

  tirx::Stmt ForceSplit(tirx::Stmt body, tvm::Target device_target) {
    return SplitDeviceFunc(std::move(body), std::move(device_target));
  }

  bool found_device_region() const { return found_device_region_; }

private:
  bool found_device_region_{false};
  Map<tirx::Var, tirx::Buffer> host_buffer_map_;
  Array<tirx::Var> non_restrict_params_;
  Optional<Array<Integer>> cluster_dims_{std::nullopt};
  Optional<String> code_block_source_{std::nullopt};
  Optional<String> code_block_entry_name_{std::nullopt};

  static void SortDeviceParams(std::vector<tirx::Var> *params) {
    std::sort(params->begin(), params->end(),
              [](const tirx::Var &a, const tirx::Var &b) {
                auto sort_key = [](const tirx::Var &var) {
                  return std::tuple{
                      !var->dtype.is_handle(),
                      var->name_hint,
                  };
                };
                return sort_key(a) < sort_key(b);
              });
  }

  std::tuple<Array<tirx::Var>, Array<tirx::Buffer>>
  CollectSourceKernelSignature() const {
    std::vector<tirx::Var> params;
    std::unordered_set<std::string> seen_vars;

    auto push = [&](const tirx::Var &var) {
      if (var.defined() && seen_vars.insert(var->name_hint).second) {
        params.push_back(var);
      }
    };

    Array<tirx::Buffer> buffers_to_declare;
    for (const auto &kv : host_buffer_map_) {
      const tirx::Buffer &buf = kv.second;
      push(buf->data);
      buffers_to_declare.push_back(buf);
      for (const PrimExpr &dim : buf->shape) {
        if (const auto *var = dim.as<tirx::VarNode>()) {
          push(GetRef<tirx::Var>(var));
        }
      }
      for (const PrimExpr &stride : buf->strides) {
        if (const auto *var = stride.as<tirx::VarNode>()) {
          push(GetRef<tirx::Var>(var));
        }
      }
      if (const auto *var = buf->elem_offset.as<tirx::VarNode>()) {
        push(GetRef<tirx::Var>(var));
      }
    }

    SortDeviceParams(&params);
    return {Array<tirx::Var>(params.begin(), params.end()), buffers_to_declare};
  }

  class SourceKernelAttrExtractor : public tirx::StmtMutator {
  public:
    static Stmt Extract(Stmt body, Optional<String> *code_block_source,
                        Optional<String> *code_block_entry_name) {
      SourceKernelAttrExtractor extractor(code_block_source,
                                          code_block_entry_name);
      return extractor(std::move(body));
    }

  private:
    explicit SourceKernelAttrExtractor(Optional<String> *code_block_source,
                                       Optional<String> *code_block_entry_name)
        : code_block_source_(code_block_source),
          code_block_entry_name_(code_block_entry_name) {}

    Stmt VisitStmt_(const tirx::AttrStmtNode *op) final {
      if (op->attr_key == tl::attr::kCodeBlockSource) {
        if (auto str = op->value.as<StringImmNode>()) {
          *code_block_source_ = str->value;
        } else {
          LOG(FATAL) << "Expected `" << tl::attr::kCodeBlockSource
                     << "` AttrStmt to carry a StringImm value, but got "
                     << op->value->GetTypeKey();
        }
        return VisitStmt(op->body);
      }

      if (op->attr_key == tl::attr::kCodeBlockEntryName) {
        if (auto str = op->value.as<StringImmNode>()) {
          *code_block_entry_name_ = str->value;
        } else {
          LOG(FATAL) << "Expected `" << tl::attr::kCodeBlockEntryName
                     << "` AttrStmt to carry a StringImm value, but got "
                     << op->value->GetTypeKey();
        }
        return VisitStmt(op->body);
      }

      return tirx::StmtMutator::VisitStmt_(op);
    }

    Optional<String> *code_block_source_;
    Optional<String> *code_block_entry_name_;
  };

  class BufferUseRemapper : public tirx::StmtExprMutator {
  public:
    explicit BufferUseRemapper(Map<tirx::Buffer, tirx::Buffer> buffer_remap)
        : explicit_buffer_remap_(std::move(buffer_remap)) {}

  private:
    tirx::Buffer VisitBufferUse(const tirx::Buffer &buffer) final {
      if (auto it = explicit_buffer_remap_.find(buffer);
          it != explicit_buffer_remap_.end()) {
        return (*it).second;
      }
      return tirx::StmtExprMutator::VisitBufferUse(buffer);
    }

    Map<tirx::Buffer, tirx::Buffer> explicit_buffer_remap_;
  };

  // Wrap body with assumes, substituting variables in assumes with the
  // corresponding variables in the device body based on name_hint matching.
  // This substitution is necessary because host-side assume variables may be
  // different Var objects from device-side parameters, even if they have the
  // same name. We always perform substitution to ensure ConvertSSA sees
  // consistent variable references.
  Stmt wrapBodyWithHostSideAssumes(
      Stmt body,
      const std::unordered_map<std::string, tirx::Var> &name_to_var) {
    // Build substitution map: assume_var -> body_var
    // Always substitute if we find a matching name, regardless of whether
    // it's the same object. This ensures ConvertSSA treats them as the same
    // variable.
    auto substitute_func =
        [&name_to_var](const tirx::Var &var) -> Optional<PrimExpr> {
      auto it = name_to_var.find(var->name_hint);
      if (it != name_to_var.end()) {
        return it->second;
      }
      return Optional<PrimExpr>();
    };

    for (auto it = host_assumes_.rbegin(); it != host_assumes_.rend(); ++it) {
      // Substitute variables in the assume condition
      PrimExpr original_node = Downcast<PrimExpr>((*it)->node);
      PrimExpr substituted_node =
          tirx::Substitute(original_node, substitute_func);
      body = AttrStmt(substituted_node, tirx::attr::tilelang_assume,
                      (*it)->value, body);
    }
    return body;
  }

  tirx::Stmt SplitDeviceFunc(tirx::Stmt body, tvm::Target device_target) {
    code_block_source_ = std::nullopt;
    code_block_entry_name_ = std::nullopt;
    body = SourceKernelAttrExtractor::Extract(
        std::move(body), &code_block_source_, &code_block_entry_name_);

    // Normal kernels infer device parameters from use-def of the device body.
    // Source kernels have no meaningful DSL body, so their device signature
    // must be reconstructed explicitly from the host PrimFunc signature and
    // buffer metadata.
    auto [old_params, buffers_to_declare] =
        [&]() -> std::tuple<Array<tirx::Var>, Array<tirx::Buffer>> {
      if (code_block_source_) {
        return CollectSourceKernelSignature();
      }

      tirx::VarUseDefAnalyzer use_def(/*defined_vars=*/{},
                                      /*visit_thread_extent=*/true);
      use_def(body);

      std::vector<tirx::Var> params{use_def.undefined_.begin(),
                                    use_def.undefined_.end()};
      SortDeviceParams(&params);
      return {Array<tirx::Var>(params.begin(), params.end()),
              use_def.undefined_buffers_};
    }();

    // Create new parameter variables for the device function to avoid sharing
    // Var objects with the host function. This prevents ConvertSSA from
    // incorrectly renaming variables when it processes multiple functions.
    Array<tirx::Var> params;
    Map<tirx::Var, PrimExpr> var_remap;
    std::unordered_map<std::string, tirx::Var> name_to_var;
    for (const auto &old_var : old_params) {
      tirx::Var new_var(old_var->name_hint, old_var->type_annotation);
      params.push_back(new_var);
      var_remap.Set(old_var, new_var);
      name_to_var[old_var->name_hint] = new_var;
    }

    // Substitute old variables with new ones in the body
    body = tirx::Substitute(body, var_remap);

    // Also remap buffers to use new variables
    Array<tirx::Buffer> new_buffers_to_declare;
    Map<tirx::Buffer, tirx::Buffer> buffer_remap;
    for (const auto &buf : buffers_to_declare) {
      auto new_shape = buf->shape.Map(
          [&](const PrimExpr &e) { return tirx::Substitute(e, var_remap); });
      auto new_strides = buf->strides.Map(
          [&](const PrimExpr &e) { return tirx::Substitute(e, var_remap); });
      auto new_elem_offset = tirx::Substitute(buf->elem_offset, var_remap);
      auto new_data = var_remap.count(buf->data)
                          ? Downcast<tirx::Var>(var_remap[buf->data])
                          : buf->data;
      tirx::Buffer new_buf(new_data, buf->dtype, new_shape, new_strides,
                           new_elem_offset, buf->name, buf->data_alignment,
                           buf->offset_factor, buf->buffer_type,
                           buf->axis_separators, buf->span);
      buffer_remap.Set(buf, new_buf);
      new_buffers_to_declare.push_back(new_buf);
    }
    body = BufferUseRemapper(buffer_remap)(std::move(body));
    buffers_to_declare = new_buffers_to_declare;

    // CodeGenCPU is used for some device-side targets, such as
    // "ext_dev", and expects to be able to return a int32_t status
    // code.

    bool can_propagate_errors = [&]() {
      auto kind = device_target->GetTargetDeviceType();
      return kind == kDLCPU || kind == kDLExtDev || kind == kDLHexagon;
    }();
    IntImm success(DataType::Int(32), 0);
    Type kernel_ret_type;
    if (can_propagate_errors) {
      kernel_ret_type = PrimType(DataType::Int(32));
      body = tirx::SeqStmt::Flatten(body, tirx::Evaluate(ret(success)));
    } else {
      kernel_ret_type = VoidType();
    }

    // Declare necessary buffers for the device side.
    for (tirx::Buffer buf : buffers_to_declare) {
      body = tirx::SeqStmt({tirx::DeclBuffer(buf), std::move(body)});
    }

    // Copy assumes from host-side to device-side, with variable substitution.
    // This must be done after DeclBuffer so that assumes are at the outermost
    // level of the function body. This ensures ConvertSSA correctly identifies
    // that assume variables refer to function parameters.
    body = wrapBodyWithHostSideAssumes(body, name_to_var);

    // Remap non_restrict_params to use new parameter variables
    Array<tirx::Var> remapped_non_restrict_params;
    for (const auto &old_var : non_restrict_params_) {
      if (var_remap.count(old_var)) {
        remapped_non_restrict_params.push_back(
            Downcast<tirx::Var>(var_remap[old_var]));
      } else {
        remapped_non_restrict_params.push_back(old_var);
      }
    }

    tirx::PrimFunc device_func(params, body, kernel_ret_type);
    Map<String, Any> device_attrs = {
        {tvm::attr::kTarget, device_target},
        {tirx::attr::kNoAlias, true},
        {tirx::attr::kIsGlobalFunc, true},
        {tl::attr::kNonRestrictParams, remapped_non_restrict_params}};
    if (cluster_dims_.defined()) {
      device_attrs.Set("cluster_dims", cluster_dims_.value());
    }
    if (code_block_source_) {
      device_attrs.Set(tl::attr::kCodeBlockSource, code_block_source_.value());
    }
    device_func = WithAttrs(std::move(device_func), device_attrs);

    GlobalVar kernel_symbol_global = var_supply_();
    if (code_block_entry_name_) {
      kernel_symbol_global = GlobalVar(code_block_entry_name_.value());
    }

    (*device_mod_)->Add(kernel_symbol_global, device_func);
    // Use old_params as call arguments (host-side variables)
    Array<PrimExpr> args =
        old_params.Map([](const tirx::Var &var) -> PrimExpr { return var; });

    if (can_propagate_errors) {
      tirx::Var kernel_error_code("kernel_error_code", success->dtype);
      tirx::Call kernel_call(success->dtype, kernel_symbol_global, args);
      tirx::Stmt assert_success = tirx::AssertStmt(
          kernel_error_code == success, tirx::StringImm("RuntimeError"),
          Array<tirx::StringImm>(
              {tirx::StringImm("Error executing compute kernel")}));
      tirx::Stmt let_check = tirx::SeqStmt(
          {tirx::Bind(kernel_error_code, kernel_call), assert_success});

      return let_check;

    } else {
      return tirx::Evaluate(
          tirx::Call(DataType::Void(), kernel_symbol_global, args));
    }
  }

  // target ir module
  IRModule *device_mod_;
  // Generate new GlobalVar for the kernel
  std::function<GlobalVar()> var_supply_;
  // Collect assumes in host side
  Array<const tirx::AttrStmtNode *> host_assumes_;
};

tirx::PrimFunc SplitHostDevice(tirx::PrimFunc func, IRModule *device_mod,
                               std::function<GlobalVar()> var_supply) {
  HostDeviceSplitter splitter(device_mod, std::move(var_supply));
  splitter.SetHostFuncSignature(func);
  // Propagate non-restrict parameter list from host func to device kernels
  if (auto opt =
          func->GetAttr<Array<tirx::Var>>(tl::attr::kNonRestrictParams)) {
    splitter.SetNonRestrictParams(opt.value());
    // Remove the attribute from host-side PrimFunc; it only matters for device
    // codegen.
    func = tvm::WithoutAttr(std::move(func), tl::attr::kNonRestrictParams);
  }
  // Propagate cluster_dims from host func to device kernel.
  // LowerOpaqueBlock sets this attr on the pre-split kernel; after splitting
  // it must live on the device side so the codegen can emit a cluster launch.
  if (auto opt = func->GetAttr<Array<Integer>>("cluster_dims")) {
    splitter.SetClusterDims(opt.value());
    func = tvm::WithoutAttr(std::move(func), "cluster_dims");
  }

  if (auto body = splitter(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  } else if (!splitter.found_device_region()) {
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      auto device_target = target.value().WithoutHost();
      if (device_target.defined() &&
          func->HasNonzeroAttr(tirx::attr::kIsEntryFunc) &&
          tirx::is_no_op(func->body)) {
        if (auto forced = splitter.ForceSplit(func->body, device_target);
            !forced.same_as(func->body)) {
          func.CopyOnWrite()->body = forced;
        }
      }
    }
  }
  return func;
}

namespace transform {

tvm::transform::Pass SplitHostDevice() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext ctx) {
    tvm::GlobalVarSupply global_var_supply(mod);

    IRModule device_mod = IRModule(Map<GlobalVar, BaseFunc>({}));
    IRModule updates = IRModule(Map<GlobalVar, BaseFunc>({}));

    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<tirx::PrimFunc>()) {
        tirx::PrimFunc func = opt.value();

        auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
        auto name_prefix = global_symbol.value_or(gvar->name_hint);
        auto kernel_name = name_prefix + "_kernel";
        auto var_supply = [&global_var_supply, &kernel_name]() -> GlobalVar {
          return global_var_supply->FreshGlobal(kernel_name, false);
        };

        func = ::tvm::tl::SplitHostDevice(std::move(func), &device_mod,
                                          var_supply);
        if (!func.same_as(base_func)) {
          updates->Add(gvar, func);
        }
      }
    }
    mod->Update(updates);
    mod->Update(device_mod);
    return tirx::transform::ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.SplitHostDevice",
                                          {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.SplitHostDevice", SplitHostDevice);
}

} // namespace transform
} // namespace tl
} // namespace tvm
