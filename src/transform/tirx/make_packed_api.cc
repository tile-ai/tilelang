#include "compat/metadata.h"
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
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

#include "tirx_compat/transform.h"
#include "transform/common/packed_api.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

namespace {
class ReturnRewriter : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) override {
    if (input.as<ExprNode>())
      return ffi::Unchanged();
    return tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);
  }
  explicit ReturnRewriter(Var ret_var) : ret_var_(ret_var) {}

  UnchangedOr<Stmt> Mutate_(const ForNode *node,
                            InplaceMode inplace_mode) override {

    bool is_parallel =
        node->kind == ForKind::kParallel &&
        !(prim::IsOne(Range::GetExtentFromBeginEnd(node->begin, node->end)) &&
          node->HasTrivialStep() && node->annotations.empty());
    if (is_parallel)
      in_parallel_ += 1;
    Stmt ret = tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode)
                   .ValueOrUnchanged(ffi::GetRef<Stmt>(node));
    if (is_parallel)
      in_parallel_ -= 1;
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode inplace_mode) override {
    bool is_parallel = node->op.same_as(tirx::parallel_launch_op());
    if (is_parallel)
      in_parallel_ += 1;
    auto ret = tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode);
    if (is_parallel)
      in_parallel_ -= 1;
    return ret;
  }

  UnchangedOr<Stmt> Mutate_(const ReturnNode *node,
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

    if (const auto *tuple = val.as<TupleNode>();
        tuple && tuple->fields.empty()) {
      info.type_index = ffi::TypeIndex::kTVMFFINone;
      info.expr = IntImm::Int64(0);
      return info;
    }

    if (val->ty.as<PtrTypeNode>()) {
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
      info.expr = prim_val;
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
    return SeqStmt({store_tindex, store_zero_padding, store_val, ret_zero});
  }

  Var ret_var_;
  int in_parallel_{0};
};

class SubroutineCallRewriter : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
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
  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) override {
    auto node = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();

    if (auto *gvar_ptr = node->op.as<GlobalVarNode>()) {
      auto gvar = ffi::GetRef<GlobalVar>(gvar_ptr);
      if (auto symbol = packed_func_methods.Get(gvar)) {
        ffi::Array<Expr> cpacked_args;
        cpacked_args.push_back(StringImm(symbol.value()));
        for (const Expr &arg : node->args) {
          cpacked_args.push_back(arg);
        }

        cpacked_args.push_back(tvm::tirx::ConstHandle(0));
        made_change_ = true;
        return Call(node->ty, tirx::call_cpacked_op(), cpacked_args);
      }
    }

    return node;
  }
  const ffi::Map<GlobalVar, ffi::String> &packed_func_methods;
  bool made_change_{false};
};

} // namespace

ffi::Optional<ffi::String> RequiresPackedAPI(const Function &func) {

  if (auto opt = func->GetAttr(tvm::attr::kCallingConv)) {
    if (opt.value() != CallingConv::kDefault) {
      return std::nullopt;
    }
  }

  auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
  if (!global_symbol.has_value()) {
    return std::nullopt;
  }

  return global_symbol.value();
}

Function MakePackedAPI(Function func) {
  if (!func->body.has_value())
    return func;
  auto global_symbol = RequiresPackedAPI(func);
  if (!global_symbol.has_value()) {
    return func;
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

  auto opt_target_host = target->GetHost();
  if (!opt_target_host)
    return func;
  Target target_host = opt_target_host.value();

  auto *func_ptr = func.CopyOnWrite();

  Var v_self_handle("self_handle", PtrType::VoidPtrType());
  Var v_packed_args("args", PtrType::VoidPtrType());
  Var v_num_packed_args("num_args", PrimType::Int(32));
  Var v_result("result", PtrType::VoidPtrType());

  PrimVar device_id("dev_id");
  IntImm device_type(PrimType::Int(32), target_device_type);
  auto nullable_parameters =
      func->GetAttr<ffi::Array<Var>>("tirx.nullable_params")
          .value_or(ffi::Array<Var>{});
  if (!nullable_parameters.empty()) {
    std::unordered_set<const VarNode *> used;
    for (const Var &variable : UndefinedVars(func_ptr->body.value()))
      used.insert(variable.get());
    for (const Var &parameter : nullable_parameters) {
      TVM_FFI_CHECK(!used.count(parameter.get()), ValueError)
          << "Nullable tensor parameter " << parameter->name
          << " has data uses";
    }
  }

  auto result = tirx_passes::DecodePackedAPIArguments(
      name_hint, func_ptr->params, v_packed_args, v_num_packed_args,
      device_type, device_id.as_or_throw<PrimExpr>(),
      func->GetAttr<ffi::Map<Var, PrimType>>("tirx.packed_storage_dtypes")
          .value_or(ffi::Map<Var, PrimType>{}),
      nullable_parameters,
      func->GetAttr<ffi::Array<Var>>("tirx.required_metadata")
          .value_or(ffi::Array<Var>{}));
  bool need_set_device = result.bindings.count(device_id);
  if (!need_set_device && target_device_type == kDLCUDA) {
    result.definitions.push_back(
        Bind(device_id, Call(PrimType::Int(32), tirx::call_packed_op(),
                             {StringImm("tl.runtime.GetCUDADeviceIndex")})));
    need_set_device = true;
  }

  ffi::Array<Var> args{v_self_handle, v_packed_args, v_num_packed_args,
                       v_result};

  func =
      WithAttrs(std::move(func),
                {{tvm::attr::kCallingConv.name, CallingConv::kCPackedFunc},
                 {tvm::attr::kTarget.name, target_host},
                 {tvm::attr::kGlobalSymbol.name,
                  ffi::symbol::tvm_ffi_symbol_prefix + global_symbol.value()}});

  Stmt body = ffi::make_object<ReturnRewriter>(v_result)
                  ->Mutate(func_ptr->body.value(), InplaceMode::kAllow)
                  .ValueOrUnchanged(func_ptr->body.value());
  body =
      RegionStmt(tirx::compute_scope_op(), {StringImm(name_hint + "_compute_")},
                 {}, DictAttrs(), body);

  if (need_set_device) {
    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      PrimVar status("set_device_status", PrimType::Int(32));
      Stmt set_device = Bind(
          status,
          Call(PrimType::Int(32), tirx::call_extern_op(),
               {StringImm("TVMBackendSetDevice"), device_type, device_id}));
      Stmt check_status = If(
          status != IntImm::Int32(0),
          SeqStmt(
              {Evaluate(Call(PrimType::Int(32), tirx::throw_last_error_op(), {})
                            .as_or_throw<PrimExpr>())}));
      body = SeqStmt({set_device, check_status, body});
    }
  }

  body = SeqStmt({body, Return(IntImm::Int32(0))});

  result.tensor_declarations.push_back(body);
  body = SeqStmt(result.tensor_declarations);
  if (!result.assertions.empty()) {
    result.assertions.push_back(body);
    body = SeqStmt(result.assertions);
  }
  if (need_set_device) {
    body = RegionStmt(tirx::device_context_op(), {device_type, device_id}, {},
                      DictAttrs(), body);
  }
  if (!result.definitions.empty()) {
    result.definitions.push_back(body);
    body = SeqStmt(result.definitions);
  }
  func_ptr->body = body;
  func_ptr->params = args;

  ffi::Array<Var> undefined =
      UndefinedVars(func_ptr->body.value(), func_ptr->params);
  TVM_FFI_ICHECK_EQ(undefined.size(), 0)
      << "In Function " << name_hint << " variables " << undefined
      << " are used, but are not passed in as API arguments";

  func_ptr->ret_type = PrimType::Int(32);

  auto attributes = func_ptr->attrs->dict;
  attributes.erase("tirx.nullable_params");
  attributes.erase("tirx.required_metadata");
  attributes.erase("tirx.packed_storage_dtypes");
  func_ptr->attrs = DictAttrs(attributes);

  return func;
}

} // namespace

Pass MakePackedAPI() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    ffi::Map<GlobalVar, ffi::String> packed_func_methods;
    for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto function = opt.value();
        if (auto global_symbol = RequiresPackedAPI(function)) {
          packed_func_methods.Set(gvar, global_symbol.value());
        }
      }
    }

    IRModuleNode *mptr = mod.CopyOnWrite();
    IRModule updates;

    for (const auto &[gvar, base_func] : mptr->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto func = opt.value();
        if (!func->body.has_value())
          continue;
        auto orig_func = func;

        if (auto body = SubroutineCallRewriter::Apply(packed_func_methods,
                                                      func->body.value())) {
          func.CopyOnWrite()->body = body.value();
        }

        func = MakePackedAPI(std::move(func));

        if (!func.same_as(orig_func)) {
          updates->Add(gvar, func);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tl.MakePackedAPI");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.MakePackedAPI",
                                   [] { return MakePackedAPI(); });
}

} // namespace tvm::tl::tirx_passes
