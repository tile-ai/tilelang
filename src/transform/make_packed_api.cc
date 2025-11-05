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
 * \file make_packed_api.cc Lower PrimFunc to use the packed function API.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "arg_binder.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;
using namespace ffi;
static constexpr const char *kDeviceContextVar = "device_api_context";

namespace {
class ReturnRewriter : public StmtMutator {
public:
  explicit ReturnRewriter(Var ret_var, Var ret_tcode)
      : ret_var_(std::move(ret_var)), ret_tcode_(std::move(ret_tcode)) {}

  Stmt VisitStmt_(const ForNode *node) override {
    if (node->kind == ForKind::kParallel)
      in_parallel_ += 1;
    Stmt ret = StmtMutator::VisitStmt_(node);
    if (node->kind == ForKind::kParallel)
      in_parallel_ -= 1;
    return ret;
  }

  Stmt VisitStmt_(const EvaluateNode *node) override {
    Stmt ret = StmtMutator::VisitStmt_(node);
    const EvaluateNode *eval = ret.as<EvaluateNode>();
    ICHECK(eval);
    if (const CallNode *call = eval->value.as<CallNode>()) {
      if (call->op.same_as(builtin::ret())) {
        ICHECK_EQ(in_parallel_, 0)
            << "tir.ret cannot be used in parallel scope.";
        ICHECK_EQ(call->args.size(), 1) << "tir.ret expect a single argument.";
        ret = WriteToOut(call->args[0]);
      }
    }
    return ret;
  }

private:
  struct ConvertedInfo {
    int type_index{-1};
    PrimExpr expr;
    Buffer dummy_val_buffer;
    Buffer dummy_tcode_buffer;
  };

  ConvertedInfo ConvertForFFI(const PrimExpr &val) {
    ConvertedInfo info;

    // convert val's data type to FFI data type, return type code
    DataType dtype = val.dtype();
    if (dtype.is_int() || dtype.is_uint()) {
      info.type_index = ffi::TypeIndex::kTVMFFIInt;
      info.expr = Cast(DataType::Int(64), val);
    } else if (dtype.is_float()) {
      info.type_index = ffi::TypeIndex::kTVMFFIFloat;
      info.expr = Cast(DataType::Float(64), val);
    } else if (dtype.is_void()) {
      info.type_index = ffi::TypeIndex::kTVMFFINone;
      info.expr = val;
    } else {
      LOG(FATAL) << "data type " << dtype << " not supported yet";
    }

    // If multiple return locations have the same data type, use the
    // same dummy buffer declaration.
    auto it = dummy_val_buffer_map_.find(info.type_index);
    if (it != dummy_val_buffer_map_.end()) {
      info.dummy_val_buffer = it->second;
    } else {
      info.dummy_val_buffer =
          Buffer(ret_var_, info.expr.dtype(), {1}, {1}, ConstInt32(0),
                 ret_var_->name_hint, 0, 0, kDefault);
      dummy_val_buffer_map_[info.type_index] = info.dummy_val_buffer;
    }

    // The type_index is always a 32-bit int, so we don't need to have a
    // separate map.
    if (!dummy_tcode_buffer_.defined()) {
      dummy_tcode_buffer_ =
          Buffer(ret_tcode_, DataType::Int(32), {1}, {1}, ConstInt32(0),
                 ret_tcode_->name_hint, 0, 0, kDefault);
    }
    info.dummy_tcode_buffer = dummy_tcode_buffer_;

    return info;
  }

  Stmt WriteToOut(const PrimExpr &val) {
    auto info = ConvertForFFI(val);
    Stmt store_val = BufferStore(info.dummy_val_buffer, info.expr, {0});
    Stmt store_tcode =
        BufferStore(info.dummy_tcode_buffer, info.type_index, {0});
    Stmt ret_zero = Evaluate(tvm::ret(0));
    return SeqStmt({store_val, store_tcode, ret_zero});
  }

  Var ret_var_;
  Var ret_tcode_;
  int in_parallel_{0};

  std::unordered_map<int, Buffer> dummy_val_buffer_map_;
  Buffer dummy_tcode_buffer_;
};

Stmt RewriteReturn(Stmt body, Var ret_var, Var ret_tcode) {
  ReturnRewriter rewriter(std::move(ret_var), std::move(ret_tcode));
  return rewriter(std::move(body));
}

class SubroutineCallRewriter : public StmtExprMutator {
public:
  static Optional<Stmt> Apply(const Map<GlobalVar, String> &packed_func_methods,
                              Stmt stmt) {
    SubroutineCallRewriter rewriter(packed_func_methods);
    stmt = rewriter.VisitStmt(stmt);
    if (rewriter.made_change_) {
      return stmt;
    } else {
      return std::nullopt;
    }
  }

private:
  explicit SubroutineCallRewriter(
      const Map<GlobalVar, String> &packed_func_methods)
      : packed_func_methods(packed_func_methods) {}

  PrimExpr VisitExpr_(const CallNode *op) override {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    if (auto *gvar_ptr = node->op.as<GlobalVarNode>()) {
      auto gvar = tvm::ffi::GetRef<GlobalVar>(gvar_ptr);
      if (auto symbol = packed_func_methods.Get(gvar)) {
        Array<PrimExpr> cpacked_args;
        cpacked_args.push_back(tir::StringImm(symbol.value()));
        for (auto arg : node->args) {
          cpacked_args.push_back(arg);
        }

        // push an empty handle to be compatible with current cpacked convention
        cpacked_args.push_back(tir::make_zero(DataType::Handle()));
        made_change_ = true;
        return tir::Call(node->dtype, tir::builtin::tvm_call_cpacked(),
                         cpacked_args);
      }
    }

    return node;
  }
  const Map<GlobalVar, String> &packed_func_methods;
  bool made_change_{false};
};

} // namespace

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, const std::string &msg) {
  return AssertStmt(std::move(lhs) == std::move(rhs), tvm::tir::StringImm(msg),
                    Evaluate(0));
}

inline Stmt MakeAssertNotNull(PrimExpr ptr, const std::string &msg) {
  Call isnull(DataType::Bool(), builtin::isnullptr(), {std::move(ptr)});
  return AssertStmt(!isnull, tvm::tir::StringImm(msg), Evaluate(0));
}

/* \brief Return the global_symbol of the function, if it should be updated
 *
 * \param func The function to be inspected
 *
 * \returns The global_symbol to be used for the function at call
 * sites, or std::nullopt if the function is to remain unchanged.
 */
Optional<String> RequiresPackedAPI(const PrimFunc &func) {
  // A function with an explicit calling convention has already been
  // lowered, and should not be modified.
  if (auto opt = func->GetAttr<Integer>(tvm::attr::kCallingConv)) {
    if (CallingConv(opt.value()->value) != CallingConv::kDefault) {
      return std::nullopt;
    }
  }

  // Internal function calls do not need the PackedFunc API
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  if (!global_symbol) {
    return std::nullopt;
  }

  return global_symbol;
}

PrimFunc MakePackedAPI(PrimFunc func) {
  auto global_symbol = RequiresPackedAPI(func);
  if (!global_symbol) {
    return func;
  }
  std::string name_hint = global_symbol.value();

  Target target = [&]() {
    auto opt = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(opt) << "MakePackedAPI required the function to be annotated with "
                   "tvm::attr::kTarget ("
                << tvm::attr::kTarget
                << "), but the function only has attributes " << func->attrs;
    return opt.value();
  }();
  int target_device_type = target->GetTargetDeviceType();

  // A function without a host target has already been lowered.
  Target target_host;
  if (auto opt = target->GetHost()) {
    target_host = opt.value();
  } else {
    return func;
  }

  auto *func_ptr = func.CopyOnWrite();
  const Stmt nop = Evaluate(0);
  int num_args = static_cast<int>(func_ptr->params.size());

  // Data field definitions
  // The packed fields
  Var v_packed_args("args", DataType::Handle());
  Buffer buf_packed_arg_type_ids =
      decl_buffer({IntImm(DataType::Int(32), func_ptr->params.size())},
                  DataType::Int(32), "arg_type_ids");
  Var v_num_packed_args("num_args", DataType::Int(32));
  Var v_out_ret_value("out_ret_value", PointerType(PrimType(DataType::Void())));
  Var v_out_ret_tcode("out_ret_tcode",
                      PointerType(PrimType(DataType::Int(32))));
  Var v_resource_handle("resource_handle", DataType::Handle());
  // The arguments of the function.

  // The device context
  Var device_id("dev_id");
  Integer device_type(target_device_type);
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after init
  std::vector<Stmt> seq_init, seq_check, arg_buffer_declarations;
  std::unordered_map<const VarNode *, PrimExpr> vmap;
  ArgBinder binder(&vmap);
  std::vector<Stmt> shape_checks;
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  bool disable_dynamic_tail_split =
      ctxt->GetConfig<Bool>(kDisableDynamicTailSplit, Bool(true)).value();

  // ---------------------------
  // local function definitions
  // load i-th argument as type t
  auto f_arg_value = [&](DataType t, int i) {
    Array<PrimExpr> call_args{
        v_packed_args, IntImm(DataType::Int(32), i),
        IntImm(DataType::Int(32), builtin::kTVMValueContent)};
    // load 64 bit version
    DataType api_type = APIType(t);
    PrimExpr res = Call(api_type, builtin::tvm_struct_get(), call_args);
    // cast to the target version.
    if (api_type != t) {
      res = Cast(t, res);
    }
    return res;
  };

  // Find the device API context argument based on name
  for (const auto &param : func_ptr->params) {
    if (param->name_hint == kDeviceContextVar) {
      num_args--;
      v_resource_handle = param;
      break;
    }
  }

  // Assert correct type codes for each argument.  This must be done
  // *before* any initialization steps produced by
  // `binder.BindDLTensor()`.  The validity of those initialization
  // steps depends on the correct types being present, and must not
  // occur before the type codes are actually checked.
  seq_init.push_back(
      MakeAssertEQ(v_num_packed_args, num_args, [&]() -> std::string {
        std::ostringstream error_message;
        error_message << name_hint << ": num_args should be " << num_args;
        return error_message.str();
      }()));

  seq_init.push_back(MakeAssertNotNull(
      v_packed_args, name_hint + ": TVMValue* arg pointer was NULL"));
  seq_init.push_back(MakeAssertNotNull(
      buf_packed_arg_type_ids->data, name_hint + ": int* type_codes was NULL"));

  seq_init.emplace_back(DeclBuffer(buf_packed_arg_type_ids, nop));

  // Need to delay binding of the buffers, in case some arguments also
  // appear in the buffer.
  std::vector<std::pair<PrimExpr, Var>> var_def;
  std::vector<std::pair<Var, Buffer>> buffer_def;

  for (int i = 0; i < static_cast<int>(func_ptr->params.size()); ++i) {
    Var param = func_ptr->params[i];

    // Ignore the device context argument, as it will still be passed
    // as a native argument.
    if (param->name_hint == kDeviceContextVar) {
      continue;
    }

    var_def.emplace_back(f_arg_value(param.dtype(), i), param);
    if (func_ptr->buffer_map.count(param)) {
      buffer_def.emplace_back(param, func_ptr->buffer_map[param]);
    }

    // type code checks
    Var type_index(param->name_hint + ".code", DataType::Int(32));
    seq_init.emplace_back(LetStmt(
        type_index,
        BufferLoad(buf_packed_arg_type_ids, {IntImm(DataType::Int(32), i)}),
        nop));
    DataType t = param.dtype();
    if (t.is_handle()) {
      std::ostringstream msg;
      msg << name_hint << ": Expect arg[" << i << "] to be pointer";
      seq_init.emplace_back(
          AssertStmt(type_index == ffi::TypeIndex::kTVMFFINone ||
                         type_index == ffi::TypeIndex::kTVMFFIOpaquePtr ||
                         type_index == ffi::TypeIndex::kTVMFFIDLTensorPtr ||
                         type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin,
                     tvm::tir::StringImm(msg.str()), nop));
    } else if (t.is_int() || t.is_uint()) {
      std::ostringstream msg;
      msg << name_hint << ": Expect arg[" << i << "] to be int";
      seq_init.emplace_back(AssertStmt(type_index == kDLInt,
                                       tvm::tir::StringImm(msg.str()), nop));
    } else {
      ICHECK(t.is_float());
      std::ostringstream msg;
      msg << name_hint << ": Expect arg[" << i << "] to be float";
      seq_init.emplace_back(AssertStmt(type_index == kDLFloat,
                                       tvm::tir::StringImm(msg.str()), nop));
    }
  }

  Array<Var> args{v_packed_args,     buf_packed_arg_type_ids->data,
                  v_num_packed_args, v_out_ret_value,
                  v_out_ret_tcode,   v_resource_handle};

  // Arg definitions are defined before buffer binding to avoid the use before
  // def errors.
  //
  // For example, for auto broadcasting, checks are required to guarantee that
  // either 0 or the original stride will be correctly used. Checks here have
  // to use the args that may have no let binding yet. Therefore, hoisting let
  // binding for args before buffer declaration is needed.
  for (const auto &[expr, param] : var_def) {
    binder.Bind(param, expr, name_hint + "." + param->name_hint, true);
  }

  for (const auto &kv : buffer_def) {
    binder.BindDLTensor(kv.second, device_type, device_id, kv.first,
                        name_hint + "." + kv.first->name_hint);
    arg_buffer_declarations.push_back(DeclBuffer(kv.second, nop));
  }

  func =
      WithAttrs(std::move(func),
                {{tvm::attr::kCallingConv, Integer(CallingConv::kCPackedFunc)},
                 {tvm::attr::kTarget, target_host}});
  Stmt body = RewriteReturn(func_ptr->body, v_out_ret_value, v_out_ret_tcode);
  body = AttrStmt(make_zero(DataType::Int(32)), tir::attr::compute_scope,
                  StringImm(name_hint + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    auto node = String("default");
    seq_check.push_back(AttrStmt(node, tir::attr::device_id, device_id, nop));
    seq_check.push_back(
        AttrStmt(node, tir::attr::device_type, device_type, nop));

    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      Stmt set_device =
          Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(),
                        {StringImm(runtime::symbol::tvm_set_device),
                         device_type, device_id}));
      body = SeqStmt({set_device, body});
    }
  }

  // (zhengju) For dynamic constraint, we need to check the buffer shape and
  // dtype to make sure the buffer can be vectorized.
  for (const auto &kv : buffer_def) {
    if (disable_dynamic_tail_split) {
      Optional<Integer> opt_dynamic_alignment =
          ctxt->GetConfig(kDynamicAlignment, Optional<Integer>());
      int dynamic_alignment = opt_dynamic_alignment.value_or(Integer(8))->value;
      // The vectorize dimension will be the last dimension of the buffer
      auto vectorize_dim = kv.second->shape[kv.second->shape.size() - 1];
      auto shape_vectorize_expr = [&]() -> PrimExpr {
        PrimExpr result = IntImm(kv.second->DefaultIndexType(), 1);
        result = result * vectorize_dim;
        result = FloorMod(result, IntImm(result->dtype, dynamic_alignment));
        return result;
      }();
      shape_checks.emplace_back(AssertStmt(
          shape_vectorize_expr == 0,
          tvm::tir::StringImm(
              kv.second->name +
              ": Vectorize dimension in buffer must be divisible by " +
              std::to_string(dynamic_alignment)),
          nop));
    }
  }

  // Return error code of zero on success
  body = SeqStmt({body, Evaluate(ret(Integer(0)))});

  if (!disable_dynamic_tail_split) {
    body = MergeNest({seq_init, binder.init_nest(), seq_check, binder.asserts(),
                      arg_buffer_declarations},
                     body);
  } else {
    body = MergeNest({seq_init, binder.init_nest(), seq_check, binder.asserts(),
                      arg_buffer_declarations, shape_checks},
                     body);
  }

  func_ptr->body = body;
  func_ptr->params = args;

  Array<Var> undefined = UndefinedVars(func_ptr->body, func_ptr->params);
  ICHECK_EQ(undefined.size(), 0)
      << "In PrimFunc " << name_hint << " variables " << undefined
      << " are used, but are not passed in as API arguments";

  func_ptr->buffer_map = Map<Var, Buffer>();
  func_ptr->ret_type = PrimType(DataType::Int(32)); // return the function.
  return func;
}

tvm::transform::Pass MakePackedAPI() {
  using tvm::transform::Pass;
  auto pass_func = [](IRModule mod, const tvm::transform::PassContext &ctx) {
    Map<GlobalVar, String> packed_func_methods;
    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        const auto &prim_func = opt.value();
        if (auto global_symbol = RequiresPackedAPI(prim_func)) {
          packed_func_methods.Set(gvar, global_symbol.value());
        }
      }
    }

    IRModuleNode *mptr = mod.CopyOnWrite();
    IRModule updates;

    for (const auto &[gvar, base_func] : mptr->functions) {
      if (auto opt = base_func.as<PrimFunc>()) {
        auto func = opt.value();
        auto orig_func = func;

        if (auto body = SubroutineCallRewriter::Apply(packed_func_methods,
                                                      func->body)) {
          func.CopyOnWrite()->body = body.value();
        }
        func = MakePackedAPI(std::move(func));

        if (!func.same_as(orig_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    if (!updates->functions.empty()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.MakePackedAPI", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MakePackedAPI",
                        []() { return MakePackedAPI(); });
}

} // namespace tl
} // namespace tvm
