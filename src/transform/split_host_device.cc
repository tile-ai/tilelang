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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "tir/analysis/var_use_def_analysis.h"

namespace tvm {
namespace tl {
using namespace ffi;
namespace tir = tvm::tir;

class HostDeviceSplitter : public tir::StmtMutator {
public:
  explicit HostDeviceSplitter(IRModule *device_mod,
                              std::function<GlobalVar()> var_supply)
      : device_mod_(device_mod), var_supply_(std::move(var_supply)) {}

  tir::Stmt VisitStmt_(const tir::AttrStmtNode *op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      found_device_region_ = true;
      auto device_target = op->node.as<tvm::Target>().value().WithoutHost();
      return SplitDeviceFunc(op->body, device_target);
    }
    return tir::StmtMutator::VisitStmt_(op);
  }

  tir::Stmt ForceSplit(tir::Stmt body, tvm::Target device_target) {
    return SplitDeviceFunc(std::move(body), std::move(device_target));
  }

  bool found_device_region() const { return found_device_region_; }

private:
  bool found_device_region_{false};

  tir::Stmt SplitDeviceFunc(tir::Stmt body, tvm::Target device_target) {
    auto [params, buffers_to_declare] =
        [&]() -> std::tuple<Array<tir::Var>, Array<tir::Buffer>> {
      tir::VarUseDefAnalyzer use_def(/*defined_vars=*/{},
                                     /*visit_thread_extent=*/true);
      use_def(body);

      // Sort first by variable type, then by variable name
      std::vector<tir::Var> params{use_def.undefined_.begin(),
                                   use_def.undefined_.end()};
      std::sort(params.begin(), params.end(),
                [](const tir::Var &a, const tir::Var &b) {
                  auto sort_key = [](const tir::Var &var) {
                    return std::tuple{
                        !var->dtype.is_handle(),
                        var->name_hint,
                    };
                  };
                  return sort_key(a) < sort_key(b);
                });
      return {params, use_def.undefined_buffers_};
    }();

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
      body = tir::SeqStmt::Flatten(body, tir::Evaluate(ret(success)));
    } else {
      kernel_ret_type = VoidType();
    }

    for (tir::Buffer buf : buffers_to_declare) {
      body = tir::DeclBuffer(buf, std::move(body));
    }
    tir::PrimFunc device_func(params, body, kernel_ret_type);
    device_func =
        WithAttrs(std::move(device_func), {{tvm::attr::kTarget, device_target},
                                           {tir::attr::kNoAlias, true},
                                           {tir::attr::kIsGlobalFunc, true}});

    GlobalVar kernel_symbol_global = var_supply_();
    (*device_mod_)->Add(kernel_symbol_global, device_func);
    Array<PrimExpr> args =
        params.Map([](const tir::Var &var) -> PrimExpr { return var; });

    if (can_propagate_errors) {
      tir::Var kernel_error_code("kernel_error_code", success->dtype);
      tir::Call kernel_call(success->dtype, kernel_symbol_global, args);
      tir::AssertStmt assert_success(
          kernel_error_code == success,
          tir::StringImm("Error executing compute kernel"), tir::Evaluate(0));
      tir::LetStmt let_check(kernel_error_code, kernel_call, assert_success);

      return let_check;

    } else {
      return tir::Evaluate(
          tir::Call(DataType::Void(), kernel_symbol_global, args));
    }
  }

  // target ir module
  IRModule *device_mod_;
  // Generate new GlobalVar for the kernel
  std::function<GlobalVar()> var_supply_;
};

tir::PrimFunc SplitHostDevice(tir::PrimFunc func, IRModule *device_mod,
                              std::function<GlobalVar()> var_supply) {
  HostDeviceSplitter splitter(device_mod, std::move(var_supply));

  if (auto body = splitter(func->body); !body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  } else if (!splitter.found_device_region()) {
    if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
      auto device_target = target.value().WithoutHost();
      if (device_target.defined() &&
          func->HasNonzeroAttr(tir::attr::kIsEntryFunc) &&
          tir::is_no_op(func->body)) {
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
      if (auto opt = base_func.as<tir::PrimFunc>()) {
        tir::PrimFunc func = opt.value();

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
    return tir::transform::ConvertSSA()(mod);
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.SplitHostDevice",
                                          {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.SplitHostDevice", SplitHostDevice);
}

} // namespace transform
} // namespace tl
} // namespace tvm
