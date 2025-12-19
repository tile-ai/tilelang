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
#include <tvm/ffi/extra/module.h>
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

#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "arg_binder.h"
#include "merge_if_stmt.h"
#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;
using namespace ffi;

namespace {
class ReturnRewriter : public StmtMutator {
public:
  explicit ReturnRewriter(Var ret_var) : ret_var_(ret_var) {}

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
  };

  ConvertedInfo ConvertForFFI(const PrimExpr &val) {
    ConvertedInfo info;

    // convert val's data type to FFI data type, return type code
    DataType dtype = val.dtype();
    if (dtype.is_bool()) {
      info.type_index = ffi::TypeIndex::kTVMFFIBool;
      info.expr = Cast(DataType::Int(64), val);

    } else if (dtype.is_int() || dtype.is_uint()) {
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

    return info;
  }

  Stmt WriteToOut(PrimExpr val) {
    auto info = ConvertForFFI(val);
    Stmt store_tindex = tir::Evaluate(
        tir::Call(DataType::Int(32), tir::builtin::tvm_struct_set(),
                  {ret_var_, IntImm(DataType::Int(32), 0),
                   IntImm(DataType::Int(32), tir::builtin::kTVMFFIAnyTypeIndex),
                   IntImm(DataType::Int(32), info.type_index)}));
    Stmt store_zero_padding = tir::Evaluate(tir::Call(
        DataType::Int(32), tir::builtin::tvm_struct_set(),
        {ret_var_, IntImm(DataType::Int(32), 0),
         IntImm(DataType::Int(32), tir::builtin::kTVMFFIAnyZeroPadding),
         IntImm(DataType::Int(32), 0)}));
    Stmt store_val = tir::Evaluate(tir::Call(
        DataType::Int(32), tir::builtin::tvm_struct_set(),
        {ret_var_, IntImm(DataType::Int(32), 0),
         IntImm(DataType::Int(32), tir::builtin::kTVMFFIAnyUnionValue),
         info.expr}));
    Stmt ret_zero = Evaluate(tvm::ret(0));
    return SeqStmt({store_tindex, store_zero_padding, store_val, ret_zero});
  }

  Var ret_var_;
  int in_parallel_{0};
};

class SubroutineCallRewriter : public StmtExprMutator {
public:
  static ffi::Optional<Stmt>
  Apply(const ffi::Map<GlobalVar, ffi::String> &packed_func_methods,
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
      const ffi::Map<GlobalVar, ffi::String> &packed_func_methods)
      : packed_func_methods(packed_func_methods) {}

  PrimExpr VisitExpr_(const CallNode *op) override {
    auto node = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    if (auto *gvar_ptr = node->op.as<GlobalVarNode>()) {
      auto gvar = ffi::GetRef<GlobalVar>(gvar_ptr);
      if (auto symbol = packed_func_methods.Get(gvar)) {
        ffi::Array<PrimExpr> cpacked_args;
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
  const ffi::Map<GlobalVar, ffi::String> &packed_func_methods;
  bool made_change_{false};
};

} // namespace

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, std::string msg) {
  return AssertStmt(lhs == rhs, tvm::tir::StringImm(msg), Evaluate(0));
}

inline Stmt MakeAssertNotNull(PrimExpr ptr, std::string msg) {
  Call isnull(DataType::Bool(), builtin::isnullptr(), {ptr});
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
  // set the global symbol to the packed function name
  const Stmt nop = Evaluate(0);
  int num_args = static_cast<int>(func_ptr->params.size());

  // Data field definitions
  // The packed fields
  Var v_self_handle("self_handle", DataType::Handle());
  Var v_packed_args("args", DataType::Handle());
  Var v_num_packed_args("num_args", DataType::Int(32));
  Var v_result("result", PointerType(PrimType(DataType::Void())));

  // The device context
  Var device_id("dev_id");
  Integer device_type(target_device_type);
  // seq_init gives sequence of initialization
  // seq_check gives sequence of later checks after init
  std::vector<Stmt> seq_init, seq_check, arg_buffer_declarations;
  std::unordered_map<const VarNode *, PrimExpr> vmap;
  ArgBinder binder(&vmap);

  // ---------------------------
  // local function definitions
  // load i-th argument as type t
  auto f_load_arg_value = [&](DataType arg_type, int i) {
    ffi::Array<PrimExpr> call_args{
        v_packed_args, IntImm(DataType::Int(32), i),
        IntImm(DataType::Int(32), builtin::kTVMFFIAnyUnionValue)};
    // load 64 bit version
    DataType api_type = APIType(arg_type);
    PrimExpr res = Call(api_type, builtin::tvm_struct_get(), call_args);
    // cast to the target version.
    if (api_type != arg_type) {
      res = Cast(arg_type, res);
    }
    return res;
  };

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

  if (num_args > 0) {
    seq_init.push_back(
        MakeAssertNotNull(v_packed_args, name_hint + ": args pointer is NULL"));
  }

  // Need to delay binding of the buffers, in case some arguments also
  // appear in the buffer.
  std::vector<std::pair<PrimExpr, Var>> var_def;
  std::vector<std::pair<Var, Buffer>> buffer_def;

  // First, collect a reverse map from Buffer->data var to parameter var so we
  // can detect whether a buffer is actually used by the function body. In
  // addition, collect variables that appear in the buffer's shape/stride so we
  // can consider uses of those symbols as a use of the buffer itself.
  std::unordered_map<const VarNode *, const VarNode *> data_var2param;
  std::unordered_map<const VarNode *, std::vector<const VarNode *>>
      shape_var2params;
  for (const auto &kv : func_ptr->buffer_map) {
    const Var &param = kv.first;
    const Buffer &buf = kv.second;
    data_var2param[buf->data.get()] = param.get();
    auto record_shape_vars = [&](const PrimExpr &e) {
      PostOrderVisit(e, [&](const ObjectRef &n) {
        if (const auto *v = n.as<VarNode>()) {
          shape_var2params[v].push_back(param.get());
        }
      });
    };
    for (const PrimExpr &e : buf->shape)
      record_shape_vars(e);
    for (const PrimExpr &e : buf->strides)
      record_shape_vars(e);
    if (buf->elem_offset.defined())
      record_shape_vars(buf->elem_offset);
  }

  // A visitor that records
  //  - which parameter buffers are used via their data var (load/store/direct),
  //  - which shape/stride/offset symbols are referenced in the body.
  // Shape symbols are not immediately attributed to all carrier buffers here;
  // a minimal carrier set is selected after visiting.
  struct UsedBufferDetector : public StmtExprVisitor {
    UsedBufferDetector(
        const std::unordered_map<const VarNode *, const VarNode *> &data2param,
        const std::unordered_map<const VarNode *, std::vector<const VarNode *>>
            &shape2params)
        : data2param(data2param), shape2params(shape2params) {}
    void VisitExpr_(const VarNode *op) override {
      auto it = data2param.find(op);
      if (it != data2param.end()) {
        used_params_by_data.insert(it->second);
      }
      auto it2 = shape2params.find(op);
      if (it2 != shape2params.end()) {
        used_shape_vars.insert(op);
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) override {
      auto it = data2param.find(op->buffer->data.get());
      if (it != data2param.end()) {
        used_params_by_data.insert(it->second);
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) override {
      auto it = data2param.find(op->buffer->data.get());
      if (it != data2param.end()) {
        used_params_by_data.insert(it->second);
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    const std::unordered_map<const VarNode *, const VarNode *> &data2param;
    const std::unordered_map<const VarNode *, std::vector<const VarNode *>>
        &shape2params;
    std::unordered_set<const VarNode *> used_params_by_data;
    std::unordered_set<const VarNode *> used_shape_vars;
  };

  UsedBufferDetector detector(data_var2param, shape_var2params);
  detector(func_ptr->body);

  // Build the packed argument handling. While doing so, keep track of whether
  // each parameter buffer is actually used. Unused input buffers can be
  // nullable and do not require DLTensor field dereferences.
  //
  // Start from buffers used via data-var (definitely non-NULL), then for each
  // referenced shape symbol pick a minimal "carrier" buffer that provides the
  // symbol. Prefer carriers that are already used-by-data; otherwise pick one
  // arbitrary carrier to ensure the symbol is bound.
  std::unordered_set<const VarNode *> used_param_buffers =
      detector.used_params_by_data;
  for (const VarNode *sym : detector.used_shape_vars) {
    auto it = shape_var2params.find(sym);
    if (it == shape_var2params.end())
      continue;
    const auto &carriers = it->second;
    bool has_used_carrier = false;
    for (const VarNode *p : carriers) {
      if (used_param_buffers.count(p)) {
        has_used_carrier = true;
        break;
      }
    }
    // NOTE: With the new nullable shape binding logic in
    // ArgBinder::BindDLTensors, we no longer need to force one carrier to be
    // non-NULL. The binder will:
    // 1. Assert that at least one carrier is non-NULL at runtime
    // 2. Use cascaded if_then_else to read from the first non-NULL carrier
    // So we can allow all carriers to be nullable.
    // if (!has_used_carrier && !carriers.empty()) {
    //   used_param_buffers.insert(carriers.front());
    // }
  }

  for (int i = 0; i < static_cast<int>(func_ptr->params.size()); ++i) {
    Var param = func_ptr->params[i];
    PrimExpr arg_value;
    // type index checks
    Var type_index(param->name_hint + ".type_index", DataType::Int(32));
    seq_init.push_back(LetStmt(
        type_index,
        tir::Call(DataType::Int(32), builtin::tvm_struct_get(),
                  {v_packed_args, IntImm(DataType::Int(32), i),
                   IntImm(DataType::Int(32), builtin::kTVMFFIAnyTypeIndex)}),
        nop));
    DataType dtype = param.dtype();
    if (dtype.is_handle()) {
      std::ostringstream msg;
      // Prefer the Buffer name if available; otherwise, fall back to param name
      // (trim _handle).
      std::string display_name;
      auto it_buf = func_ptr->buffer_map.find(param);
      if (it_buf != func_ptr->buffer_map.end()) {
        const auto &kv = *it_buf;
        display_name = kv.second->data->name_hint;
      } else {
        display_name = param->name_hint;
        const char *suffix = "_handle";
        if (display_name.size() >= 7 &&
            display_name.compare(display_name.size() - 7, 7, suffix) == 0) {
          display_name.erase(display_name.size() - 7);
        }
      }
      msg << "kernel " << name_hint << " input " << display_name
          << " expected pointer or tensor handle";
      seq_init.emplace_back(
          AssertStmt(type_index == ffi::TypeIndex::kTVMFFINone ||
                         type_index == ffi::TypeIndex::kTVMFFIOpaquePtr ||
                         type_index == ffi::TypeIndex::kTVMFFIDLTensorPtr ||
                         type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin,
                     tvm::tir::StringImm(msg.str()), nop));
      // if type_index is Tensor, we need to add the offset of the DLTensor
      // header which always equals 16 bytes, this ensures that T.handle always
      // shows up as a DLTensor*
      const int64_t object_cell_offset = sizeof(TVMFFIObject);
      static_assert(object_cell_offset == 24);
      arg_value = f_load_arg_value(param.dtype(), i);
      PrimExpr handle_from_tensor =
          Call(DataType::Handle(), tir::builtin::handle_add_byte_offset(),
               {arg_value, IntImm(DataType::Int(32), object_cell_offset)});
      arg_value = Select(type_index == ffi::TypeIndex::kTVMFFITensor,
                         handle_from_tensor, arg_value);
    } else if (dtype.is_bool()) {
      std::ostringstream msg;
      msg << "kernel " << name_hint << " scalar " << param->name_hint
          << " expected boolean";
      seq_init.emplace_back(
          AssertStmt(type_index == ffi::TypeIndex::kTVMFFIBool ||
                         type_index == ffi::TypeIndex::kTVMFFIInt,
                     tvm::tir::StringImm(msg.str()), nop));
      arg_value =
          Cast(DataType::Bool(), f_load_arg_value(DataType::Int(64), i));

    } else if (dtype.is_int() || dtype.is_uint()) {
      std::ostringstream msg;
      msg << "kernel " << name_hint << " scalar " << param->name_hint
          << " expected integer";
      seq_init.emplace_back(
          AssertStmt(type_index == ffi::TypeIndex::kTVMFFIInt ||
                         type_index == ffi::TypeIndex::kTVMFFIBool,
                     tvm::tir::StringImm(msg.str()), nop));
      arg_value = f_load_arg_value(param.dtype(), i);
    } else {
      ICHECK(dtype.is_float());
      std::ostringstream msg;
      msg << "kernel " << name_hint << " scalar " << param->name_hint
          << " expected float";
      seq_init.emplace_back(
          AssertStmt(type_index == ffi::TypeIndex::kTVMFFIFloat ||
                         type_index == ffi::TypeIndex::kTVMFFIInt ||
                         type_index == ffi::TypeIndex::kTVMFFIBool,
                     tvm::tir::StringImm(msg.str()), nop));
      // use select so we can also handle int conversion to bool
      arg_value = tir::Select(
          type_index == ffi::TypeIndex::kTVMFFIFloat,
          /* true_value = */ f_load_arg_value(param.dtype(), i),
          /* false_value = */
          Cast(param.dtype(), f_load_arg_value(DataType::Int(64), i)));
    }
    var_def.emplace_back(arg_value, param);
    if (func_ptr->buffer_map.count(param)) {
      // buffer binding now depends on type index
      // if the index is Tensor handle, we need to offset to get the DLTensor*
      buffer_def.emplace_back(param, func_ptr->buffer_map[param]);
    }
  }

  // signature: (void* handle, TVMFFIAny* packed_args, int num_args, TVMFFIAny*
  // v_result)
  ffi::Array<Var> args{v_self_handle, v_packed_args, v_num_packed_args,
                       v_result};

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

  binder.BindDLTensors(buffer_def, device_type, device_id, name_hint,
                       used_param_buffers);
  for (const auto &[var, buffer] : buffer_def) {
    // Prefer buffer data var name in diagnostics to avoid exposing low-level
    // handle vars
    arg_buffer_declarations.push_back(DeclBuffer(buffer, nop));
  }

  // reset global symbol to attach prefix
  func = WithAttrs(
      std::move(func),
      {{tvm::attr::kCallingConv, static_cast<int>(CallingConv::kCPackedFunc)},
       {tvm::attr::kTarget, target_host},
       {tvm::attr::kGlobalSymbol,
        ffi::symbol::tvm_ffi_symbol_prefix + global_symbol.value()}});

  Stmt body = ReturnRewriter(v_result)(func_ptr->body);
  body = AttrStmt(make_zero(DataType::Int(32)), tir::attr::compute_scope,
                  StringImm(name_hint + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    ffi::Any node = ffi::String("default");
    seq_check.push_back(AttrStmt(node, tir::attr::device_id, device_id, nop));
    seq_check.push_back(
        AttrStmt(node, tir::attr::device_type, device_type, nop));

    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      Stmt set_device =
          Evaluate(Call(DataType::Int(32), tir::builtin::tvm_call_packed(),
                        {StringImm(runtime::symbol::tvm_set_device),
                         device_type, device_id}));
      body = SeqStmt({set_device, body});
    }
  }

  // Return error code of zero on success
  body = SeqStmt({body, Evaluate(ret(Integer(0)))});

  body = MergeNest({seq_init, binder.init_nest(), seq_check, binder.asserts(),
                    arg_buffer_declarations},
                   body);
  func_ptr->body = body;
  func_ptr->params = args;

  ffi::Array<Var> undefined = UndefinedVars(body, func_ptr->params);

  ICHECK_EQ(undefined.size(), 0)
      << "In PrimFunc " << name_hint << " variables " << undefined
      << " are used, but are not passed in as API arguments";

  func_ptr->buffer_map = ffi::Map<Var, Buffer>();
  func_ptr->ret_type = PrimType(DataType::Int(32));
  // return the function.
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
        func = MergeIfStmtSubstitute(func);

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
