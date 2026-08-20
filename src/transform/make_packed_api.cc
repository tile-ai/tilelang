/*!
 * \file make_packed_api.cc Lower PrimFunc to use the packed function API.
 */
#include "support/check.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstddef>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "common/attr.h"
#include "merge_if_stmt.h"
#include "tir/transforms/ir_utils.h"
#include "tvm_ffi_binder.h"

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;

namespace {
constexpr const char *kTileLangOutIdx = "tilelang_out_idx";
constexpr const char *kOutputStorageDTypeResolver =
    "tl.tvm_ffi.resolve_output_storage_dtype";

struct OutputStorageInfo {
  DataType dtype;
  Array<PrimExpr> shape;
  Optional<PrimExpr> packing_condition;
};

OutputStorageInfo ResolveOutputStorage(const Buffer &buffer) {
  auto resolver = Function::GetGlobal(kOutputStorageDTypeResolver);
  ICHECK(resolver.has_value())
      << "Callee-allocated TVM-FFI outputs require global function `"
      << kOutputStorageDTypeResolver << "` to be registered";

  Any storage_dtype_result = (*resolver)(buffer->dtype);
  DataType storage_dtype = storage_dtype_result.cast<DataType>();
  const int logical_bits = buffer->dtype.bits() * buffer->dtype.lanes();
  const int storage_bits = storage_dtype.bits() * storage_dtype.lanes();
  ICHECK_GT(logical_bits, 0) << "Invalid logical dtype " << buffer->dtype
                             << " for output buffer " << buffer->name;
  ICHECK_GE(storage_bits, logical_bits)
      << "Torch storage dtype " << storage_dtype << " for logical dtype "
      << buffer->dtype << " cannot use fewer bits per storage element";
  ICHECK_EQ(storage_bits % logical_bits, 0)
      << "Torch storage dtype " << storage_dtype << " for logical dtype "
      << buffer->dtype << " must contain an integral number of logical values";

  Array<PrimExpr> storage_shape = buffer->shape;
  Optional<PrimExpr> packing_condition;
  const int packing_factor = storage_bits / logical_bits;
  if (packing_factor > 1) {
    ICHECK(!storage_shape.empty()) << "Packed output dtype " << buffer->dtype
                                   << " requires at least one shape dimension";
    PrimExpr logical_extent = storage_shape.back();
    PrimExpr factor = make_const(logical_extent.dtype(), packing_factor);
    packing_condition = floormod(logical_extent, factor) == 0;
    storage_shape.Set(storage_shape.size() - 1,
                      floordiv(logical_extent, factor));
  }
  return {storage_dtype, storage_shape, packing_condition};
}

Stmt StoreFFIAny(PrimExpr result, int type_index, PrimExpr value) {
  return SeqStmt(
      {Evaluate(Call(DataType::Int(32), builtin::tvm_struct_set(),
                     {result, IntImm(DataType::Int(32), 0),
                      IntImm(DataType::Int(32), builtin::kTVMFFIAnyTypeIndex),
                      IntImm(DataType::Int(32), type_index)})),
       Evaluate(Call(DataType::Int(32), builtin::tvm_struct_set(),
                     {result, IntImm(DataType::Int(32), 0),
                      IntImm(DataType::Int(32), builtin::kTVMFFIAnyZeroPadding),
                      IntImm(DataType::Int(32), 0)})),
       Evaluate(Call(DataType::Int(32), builtin::tvm_struct_set(),
                     {result, IntImm(DataType::Int(32), 0),
                      IntImm(DataType::Int(32), builtin::kTVMFFIAnyUnionValue),
                      std::move(value)}))});
}

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
      info.type_index = TypeIndex::kTVMFFIBool;
      info.expr = Cast(DataType::Int(64), val);

    } else if (dtype.is_int() || dtype.is_uint()) {
      info.type_index = TypeIndex::kTVMFFIInt;
      info.expr = Cast(DataType::Int(64), val);
    } else if (dtype.is_float()) {
      info.type_index = TypeIndex::kTVMFFIFloat;
      info.expr = Cast(DataType::Float(64), val);
    } else if (dtype.is_void()) {
      info.type_index = TypeIndex::kTVMFFINone;
      info.expr = val;
    } else {
      LOG(FATAL) << "data type " << dtype << " not supported yet";
    }

    return info;
  }

  Stmt WriteToOut(PrimExpr val) {
    auto info = ConvertForFFI(val);
    Stmt store_tindex = tirx::Evaluate(tirx::Call(
        DataType::Int(32), tirx::builtin::tvm_struct_set(),
        {ret_var_, IntImm(DataType::Int(32), 0),
         IntImm(DataType::Int(32), tirx::builtin::kTVMFFIAnyTypeIndex),
         IntImm(DataType::Int(32), info.type_index)}));
    Stmt store_zero_padding = tirx::Evaluate(tirx::Call(
        DataType::Int(32), tirx::builtin::tvm_struct_set(),
        {ret_var_, IntImm(DataType::Int(32), 0),
         IntImm(DataType::Int(32), tirx::builtin::kTVMFFIAnyZeroPadding),
         IntImm(DataType::Int(32), 0)}));
    Stmt store_val = tirx::Evaluate(tirx::Call(
        DataType::Int(32), tirx::builtin::tvm_struct_set(),
        {ret_var_, IntImm(DataType::Int(32), 0),
         IntImm(DataType::Int(32), tirx::builtin::kTVMFFIAnyUnionValue),
         info.expr}));
    Stmt ret_zero = Evaluate(tvm::ret(0));
    return SeqStmt({store_tindex, store_zero_padding, store_val, ret_zero});
  }

  Var ret_var_;
  int in_parallel_{0};
};

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
      auto gvar = GetRef<GlobalVar>(gvar_ptr);
      if (auto symbol = packed_func_methods.Get(gvar)) {
        Array<PrimExpr> cpacked_args;
        cpacked_args.push_back(tirx::StringImm(symbol.value()));
        for (auto arg : node->args) {
          cpacked_args.push_back(arg);
        }

        // push an empty handle to be compatible with current cpacked convention
        cpacked_args.push_back(tirx::make_zero(DataType::Handle()));
        made_change_ = true;
        return tirx::Call(node->dtype, tirx::builtin::tvm_call_cpacked(),
                          cpacked_args);
      }
    }

    return node;
  }
  const Map<GlobalVar, String> &packed_func_methods;
  bool made_change_{false};
};

struct AssumeRuntimeCheck {
  PrimExpr condition;
  Array<StringImm> message_parts;
  Span span;
};

class AssumeRuntimeCheckExtractor : public StmtMutator {
public:
  static Stmt Extract(Stmt body,
                      std::vector<AssumeRuntimeCheck> *runtime_checks) {
    AssumeRuntimeCheckExtractor extractor(runtime_checks);
    return extractor(std::move(body));
  }

private:
  explicit AssumeRuntimeCheckExtractor(
      std::vector<AssumeRuntimeCheck> *runtime_checks)
      : runtime_checks_(runtime_checks) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tl::attr::kAssumeRequiresRuntimeCheck) {
      return StmtMutator::VisitStmt_(op);
    }

    PrimExpr condition = Downcast<PrimExpr>(op->node);
    Array<StringImm> message_parts;
    if (const auto *message = op->value.as<StringImmNode>()) {
      message_parts.push_back(GetRef<StringImm>(message));
    } else {
      std::ostringstream os;
      os << "Assume: " << condition;
      message_parts.push_back(StringImm(os.str()));
    }

    if (conditional_scope_depth_ != 0) {
      // Conditional checks cannot be moved to the packed-function entry
      // without changing their execution semantics. Keep only the paired
      // tl.assume, matching SplitHostDevice's conservative lifting policy.
      return VisitStmt(op->body);
    }
    runtime_checks_->push_back({condition, std::move(message_parts), op->span});
    return VisitStmt(op->body);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    ++conditional_scope_depth_;
    Stmt result = StmtMutator::VisitStmt_(op);
    --conditional_scope_depth_;
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    ++conditional_scope_depth_;
    Stmt result = StmtMutator::VisitStmt_(op);
    --conditional_scope_depth_;
    return result;
  }

  Stmt VisitStmt_(const WhileNode *op) final {
    ++conditional_scope_depth_;
    Stmt result = StmtMutator::VisitStmt_(op);
    --conditional_scope_depth_;
    return result;
  }

  std::vector<AssumeRuntimeCheck> *runtime_checks_;
  int conditional_scope_depth_{0};
};

} // namespace

inline Stmt MakeAssertEQ(PrimExpr lhs, PrimExpr rhs, std::string msg) {
  return AssertStmt(lhs == rhs, tvm::tirx::StringImm("RuntimeError"),
                    Array<tvm::tirx::StringImm>({tvm::tirx::StringImm(msg)}));
}

inline Stmt MakeAssertNotNull(PrimExpr ptr, std::string msg) {
  Call isnull(DataType::Bool(), builtin::isnullptr(), {ptr});
  return AssertStmt(!isnull, tvm::tirx::StringImm("RuntimeError"),
                    Array<tvm::tirx::StringImm>({tvm::tirx::StringImm(msg)}));
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

  // Source kernels must stay as direct GlobalVar calls until
  // LowerDeviceKernelLaunch can turn them into device launches using the
  // selected external CUDA entry symbol.
  if (func->GetAttr<String>(tl::attr::kCodeBlockSource)) {
    return std::nullopt;
  }

  // Internal function calls do not need the PackedFunc API
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  if (!global_symbol) {
    return std::nullopt;
  }

  return global_symbol;
}

std::vector<int> GetCalleeAllocatedOutputIndices(const PrimFunc &func) {
  auto output_attr = func->GetAttr<Array<Integer>>(kTileLangOutIdx);
  if (!output_attr || output_attr.value().empty() ||
      !func->HasNonzeroAttr(tirx::attr::kIsEntryFunc)) {
    return {};
  }

  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  if (!target || target.value()->kind->name != "cuda") {
    return {};
  }
  auto target_host = target.value()->GetHost();
  if (!target_host || target_host.value()->kind->name != "c") {
    return {};
  }

  std::vector<int> indices;
  std::unordered_set<int> seen;
  const int num_params = static_cast<int>(func->params.size());
  for (const Integer &raw_index : output_attr.value()) {
    int index = raw_index.IntValue();
    if (index < 0) {
      index += num_params;
    }
    ICHECK_GE(index, 0) << kTileLangOutIdx << " index " << raw_index
                        << " is out of range for a function with " << num_params
                        << " parameters";
    ICHECK_LT(index, num_params) << kTileLangOutIdx << " index " << raw_index
                                 << " is out of range for a function with "
                                 << num_params << " parameters";
    ICHECK(seen.insert(index).second)
        << kTileLangOutIdx << " contains duplicate index " << raw_index;

    const Var &param = func->params[index];
    auto buffer_it = func->buffer_map.find(param);
    ICHECK(buffer_it != func->buffer_map.end())
        << kTileLangOutIdx << " index " << raw_index
        << " does not refer to a buffer parameter";
    indices.push_back(index);
  }
  return indices;
}

PrimFunc
MakePackedAPI(PrimFunc func,
              const std::vector<int> &callee_allocated_output_indices) {
  auto global_symbol = RequiresPackedAPI(func);
  if (!global_symbol) {
    return func;
  }
  const bool use_callee_allocated_output_abi =
      !callee_allocated_output_indices.empty();
  std::unordered_set<int> callee_allocated_output_index_set(
      callee_allocated_output_indices.begin(),
      callee_allocated_output_indices.end());
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
  std::vector<AssumeRuntimeCheck> assume_runtime_checks;
  func_ptr->body = AssumeRuntimeCheckExtractor::Extract(func_ptr->body,
                                                        &assume_runtime_checks);
  // set the global symbol to the packed function name
  const Stmt nop = Evaluate(0);
  int num_args = static_cast<int>(func_ptr->params.size());
  if (use_callee_allocated_output_abi) {
    // The callee-allocated entry omits result buffers and appends one tensor
    // anchor.
    // The anchor supplies both the target device and the EnvTensorAllocator
    // context, including for otherwise scalar-only kernels.
    num_args -= static_cast<int>(callee_allocated_output_indices.size());
    num_args += 1;
  }

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
  TVMFFIABIBuilder binder(&vmap);
  TVMFFIABIBuilder output_binder(&vmap);

  // ---------------------------
  // local function definitions
  // load i-th argument as type t
  auto f_load_arg_value = [&](DataType arg_type, int i) {
    Array<PrimExpr> call_args{
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

  // Output shapes are evaluated by the callee-allocated entry itself, so every
  // symbol they reference must be treated as a runtime-required shape symbol
  // even if the original kernel body does not otherwise mention it.
  if (use_callee_allocated_output_abi) {
    for (int output_index : callee_allocated_output_indices) {
      const Var &param = func_ptr->params[output_index];
      auto it = func_ptr->buffer_map.find(param);
      ICHECK(it != func_ptr->buffer_map.end())
          << "Callee-allocated output parameter " << output_index << " of "
          << name_hint << " must be a buffer";
      for (const PrimExpr &shape : (*it).second->shape) {
        PostOrderVisit(shape, [&](const ObjectRef &node) {
          if (const auto *var = node.as<VarNode>()) {
            detector.used_shape_vars.insert(var);
          }
        });
      }
    }
  }

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
    // TVMFFIABIBuilder::BindDLTensors, we no longer need to force one carrier
    // to be non-NULL. The binder will:
    // 1. Assert that at least one carrier is non-NULL at runtime
    // 2. Use cascaded if_then_else to read from the first non-NULL carrier
    // So we can allow all carriers to be nullable.
    // if (!has_used_carrier && !carriers.empty()) {
    //   used_param_buffers.insert(carriers.front());
    // }
  }

  int packed_arg_index = 0;
  for (int i = 0; i < static_cast<int>(func_ptr->params.size()); ++i) {
    if (callee_allocated_output_index_set.count(i)) {
      continue;
    }
    Var param = func_ptr->params[i];
    PrimExpr arg_value;
    // type index checks
    Var type_index(param->name_hint + ".type_index", DataType::Int(32));
    seq_init.push_back(SeqStmt(
        {tirx::Bind(
             type_index,
             tirx::Call(
                 DataType::Int(32), builtin::tvm_struct_get(),
                 {v_packed_args, IntImm(DataType::Int(32), packed_arg_index),
                  IntImm(DataType::Int(32), builtin::kTVMFFIAnyTypeIndex)})),
         nop}));
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
      seq_init.emplace_back(AssertStmt(
          type_index == TypeIndex::kTVMFFINone ||
              type_index == TypeIndex::kTVMFFIOpaquePtr ||
              type_index == TypeIndex::kTVMFFIDLTensorPtr ||
              type_index >= TypeIndex::kTVMFFIStaticObjectBegin,
          tvm::tirx::StringImm("RuntimeError"),
          Array<tvm::tirx::StringImm>({tvm::tirx::StringImm(msg.str())})));
      // if type_index is Tensor, we need to add the offset of the DLTensor
      // header which always equals 16 bytes, this ensures that T.handle always
      // shows up as a DLTensor*
      const int64_t object_cell_offset = sizeof(TVMFFIObject);
      static_assert(object_cell_offset == 24);
      arg_value = f_load_arg_value(param.dtype(), packed_arg_index);
      PrimExpr handle_from_tensor =
          Call(DataType::Handle(), tirx::builtin::handle_add_byte_offset(),
               {arg_value, IntImm(DataType::Int(32), object_cell_offset)});
      arg_value = Select(type_index == TypeIndex::kTVMFFITensor,
                         handle_from_tensor, arg_value);
    } else if (dtype.is_bool()) {
      std::ostringstream msg;
      msg << "kernel " << name_hint << " scalar " << param->name_hint
          << " expected boolean";
      seq_init.emplace_back(AssertStmt(
          type_index == TypeIndex::kTVMFFIBool ||
              type_index == TypeIndex::kTVMFFIInt,
          tvm::tirx::StringImm("RuntimeError"),
          Array<tvm::tirx::StringImm>({tvm::tirx::StringImm(msg.str())})));
      arg_value = Cast(DataType::Bool(),
                       f_load_arg_value(DataType::Int(64), packed_arg_index));

    } else if (dtype.is_int() || dtype.is_uint()) {
      std::ostringstream msg;
      msg << "kernel " << name_hint << " scalar " << param->name_hint
          << " expected integer";
      seq_init.emplace_back(AssertStmt(
          type_index == TypeIndex::kTVMFFIInt ||
              type_index == TypeIndex::kTVMFFIBool,
          tvm::tirx::StringImm("RuntimeError"),
          Array<tvm::tirx::StringImm>({tvm::tirx::StringImm(msg.str())})));
      arg_value = f_load_arg_value(param.dtype(), packed_arg_index);
    } else {
      ICHECK(dtype.is_float());
      std::ostringstream msg;
      msg << "kernel " << name_hint << " scalar " << param->name_hint
          << " expected float";
      seq_init.emplace_back(AssertStmt(
          type_index == TypeIndex::kTVMFFIFloat ||
              type_index == TypeIndex::kTVMFFIInt ||
              type_index == TypeIndex::kTVMFFIBool,
          tvm::tirx::StringImm("RuntimeError"),
          Array<tvm::tirx::StringImm>({tvm::tirx::StringImm(msg.str())})));
      // use select so we can also handle int conversion to bool
      arg_value =
          tirx::Select(type_index == TypeIndex::kTVMFFIFloat,
                       /* true_value = */
                       f_load_arg_value(param.dtype(), packed_arg_index),
                       /* false_value = */
                       Cast(param.dtype(), f_load_arg_value(DataType::Int(64),
                                                            packed_arg_index)));
    }
    var_def.emplace_back(arg_value, param);
    if (func_ptr->buffer_map.count(param)) {
      // buffer binding now depends on type index
      // if the index is Tensor handle, we need to offset to get the DLTensor*
      buffer_def.emplace_back(param, func_ptr->buffer_map[param]);
    }
    ++packed_arg_index;
  }

  if (use_callee_allocated_output_abi) {
    const int anchor_index = packed_arg_index++;
    Var anchor_type_index("allocator_anchor.type_index", DataType::Int(32));
    seq_init.push_back(SeqStmt(
        {tirx::Bind(
             anchor_type_index,
             tirx::Call(
                 DataType::Int(32), builtin::tvm_struct_get(),
                 {v_packed_args, IntImm(DataType::Int(32), anchor_index),
                  IntImm(DataType::Int(32), builtin::kTVMFFIAnyTypeIndex)})),
         nop}));
    seq_init.emplace_back(AssertStmt(
        anchor_type_index == TypeIndex::kTVMFFITensor,
        StringImm("RuntimeError"),
        Array<StringImm>(
            {StringImm(name_hint + ": allocator anchor must be a Tensor")})));

    PrimExpr anchor_object = f_load_arg_value(DataType::Handle(), anchor_index);
    seq_init.push_back(MakeAssertNotNull(
        anchor_object, name_hint + ": allocator anchor is NULL"));
    PrimExpr anchor_tensor = Call(
        DataType::Handle(), builtin::handle_add_byte_offset(),
        {anchor_object, IntImm(DataType::Int(64),
                               static_cast<int64_t>(sizeof(TVMFFIObject)))});
    PrimExpr anchor_device_type =
        Call(DataType::Int(32), builtin::tvm_struct_get(),
             {anchor_tensor, IntImm(DataType::Int(32), 0),
              IntImm(DataType::Int(32), builtin::kDLTensorDeviceType)});
    seq_init.push_back(MakeAssertEQ(
        anchor_device_type, IntImm(DataType::Int(32), device_type.IntValue()),
        name_hint + ": allocator anchor has the wrong device type"));
    PrimExpr anchor_device_id =
        Call(DataType::Int(32), builtin::tvm_struct_get(),
             {anchor_tensor, IntImm(DataType::Int(32), 0),
              IntImm(DataType::Int(32), builtin::kDLTensorDeviceId)});
    binder.Bind(device_id, anchor_device_id,
                name_hint + ".allocator_anchor.device_id", true);
  }

  ICHECK_EQ(packed_arg_index, num_args);

  // signature: (void* handle, TVMFFIAny* packed_args, int num_args, TVMFFIAny*
  // v_result)
  Array<Var> args{v_self_handle, v_packed_args, v_num_packed_args, v_result};

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
                       used_param_buffers, detector.used_shape_vars);
  for (const auto &[var, buffer] : buffer_def) {
    // Prefer buffer data var name in diagnostics to avoid exposing low-level
    // handle vars
    arg_buffer_declarations.push_back(DeclBuffer(buffer));
  }

  std::vector<Stmt> output_allocation;
  std::vector<Stmt> callee_allocated_output_return;
  std::vector<std::pair<Var, Buffer>> output_buffer_def;
  std::vector<PrimExpr> output_object_handles;
  Var output_storage("callee_allocated_output_storage", DataType::Handle());
  Var output_shape_storage("callee_allocated_output_shape_storage",
                           DataType::Handle());
  Var output_prototype_storage("callee_allocated_output_prototype_storage",
                               DataType::Handle());

  if (use_callee_allocated_output_abi) {
    const int num_outputs =
        static_cast<int>(callee_allocated_output_indices.size());
    std::vector<OutputStorageInfo> output_storage_info;
    output_storage_info.reserve(num_outputs);
    const int num_storage_cells =
        num_outputs == 1 ? num_outputs : 2 * num_outputs;
    int num_output_dims = 0;
    for (int param_index : callee_allocated_output_indices) {
      Buffer buffer = func_ptr->buffer_map[func_ptr->params[param_index]];
      output_storage_info.push_back(ResolveOutputStorage(buffer));
      num_output_dims +=
          static_cast<int>(output_storage_info.back().shape.size());
    }
    output_allocation.push_back(SeqStmt(
        {tirx::Bind(output_storage,
                    Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                         {StringImm("tvm_ffi_any"),
                          IntImm(DataType::Int(32), num_storage_cells)})),
         nop}));
    output_allocation.push_back(SeqStmt(
        {tirx::Bind(
             output_shape_storage,
             Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                  {StringImm("shape"),
                   IntImm(DataType::Int(32), std::max(1, num_output_dims))})),
         nop}));
    output_allocation.push_back(SeqStmt(
        {tirx::Bind(output_prototype_storage,
                    Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                         {StringImm("array"),
                          IntImm(DataType::Int(32), num_outputs)})),
         nop}));

    for (int output_ordinal = 0; output_ordinal < num_outputs;
         ++output_ordinal) {
      const Optional<PrimExpr> &packing_condition =
          output_storage_info[output_ordinal].packing_condition;
      if (packing_condition) {
        int param_index = callee_allocated_output_indices[output_ordinal];
        Buffer buffer = func_ptr->buffer_map[func_ptr->params[param_index]];
        output_allocation.emplace_back(AssertStmt(
            packing_condition.value(), StringImm("RuntimeError"),
            Array<StringImm>({StringImm(
                name_hint + ": packed output " + buffer->name +
                " has a final dimension that is not storage-aligned")})));
      }
    }

    std::unordered_set<const VarNode *> used_output_buffers;
    std::unordered_set<const VarNode *> output_shape_vars;
    constexpr int64_t kAnySize = sizeof(TVMFFIAny);
    constexpr int64_t kAnyValueOffset = offsetof(TVMFFIAny, v_ptr);
    static_assert(kAnySize == 16);
    static_assert(kAnyValueOffset == 8);
    int shape_offset = 0;

    for (int output_ordinal = 0; output_ordinal < num_outputs;
         ++output_ordinal) {
      int param_index = callee_allocated_output_indices[output_ordinal];
      Var param = func_ptr->params[param_index];
      auto buffer_it = func_ptr->buffer_map.find(param);
      ICHECK(buffer_it != func_ptr->buffer_map.end());
      Buffer buffer = (*buffer_it).second;
      const OutputStorageInfo &storage_info =
          output_storage_info[output_ordinal];

      for (const PrimExpr &shape : buffer->shape) {
        PostOrderVisit(shape, [&](const ObjectRef &node) {
          if (const auto *var = node.as<VarNode>()) {
            output_shape_vars.insert(var);
            ICHECK(vmap.count(var))
                << "Cannot infer symbolic variable " << GetRef<Var>(var)
                << " required by callee-allocated output buffer "
                << buffer->name << " in " << name_hint
                << "; bind it from an input tensor or scalar parameter";
          }
        });
      }

      for (size_t dim = 0; dim < storage_info.shape.size(); ++dim) {
        output_allocation.push_back(
            Evaluate(Call(DataType::Int(32), builtin::tvm_struct_set(),
                          {output_shape_storage,
                           IntImm(DataType::Int(32), shape_offset + dim),
                           IntImm(DataType::Int(32), builtin::kInt64ArrayElem),
                           Cast(DataType::Int(64), storage_info.shape[dim])})));
      }
      PrimExpr shape_ptr =
          Call(DataType::Handle(), builtin::handle_add_byte_offset(),
               {output_shape_storage,
                IntImm(DataType::Int(64),
                       shape_offset * static_cast<int64_t>(sizeof(int64_t)))});
      shape_offset += static_cast<int>(storage_info.shape.size());

      auto set_prototype_field = [&](builtin::TVMStructFieldKind field,
                                     PrimExpr value) {
        output_allocation.push_back(Evaluate(
            Call(DataType::Int(32), builtin::tvm_struct_set(),
                 {output_prototype_storage,
                  IntImm(DataType::Int(32), output_ordinal),
                  IntImm(DataType::Int(32), field), std::move(value)})));
      };
      set_prototype_field(builtin::kDLTensorData,
                          make_zero(DataType::Handle()));
      set_prototype_field(builtin::kDLTensorShape, shape_ptr);
      set_prototype_field(builtin::kDLTensorStrides,
                          make_zero(DataType::Handle()));
      set_prototype_field(builtin::kDLTensorNDim,
                          IntImm(DataType::Int(32), storage_info.shape.size()));
      set_prototype_field(builtin::kDLTensorTypeCode,
                          IntImm(DataType::UInt(8), storage_info.dtype.code()));
      set_prototype_field(builtin::kDLTensorTypeBits,
                          IntImm(DataType::UInt(8), storage_info.dtype.bits()));
      set_prototype_field(
          builtin::kDLTensorTypeLanes,
          IntImm(DataType::UInt(16), storage_info.dtype.lanes()));
      set_prototype_field(builtin::kDLTensorByteOffset,
                          IntImm(DataType::UInt(64), 0));
      set_prototype_field(builtin::kDLTensorDeviceId, device_id);
      set_prototype_field(builtin::kDLTensorDeviceType, device_type);
      PrimExpr prototype = Call(
          DataType::Handle(), builtin::tvm_struct_get(),
          {output_prototype_storage, IntImm(DataType::Int(32), output_ordinal),
           IntImm(DataType::Int(32), builtin::kDLTensorAddr)});
      PrimExpr output_slot = Call(
          DataType::Handle(), builtin::handle_add_byte_offset(),
          {output_storage, IntImm(DataType::Int(64), output_ordinal * kAnySize +
                                                         kAnyValueOffset)});
      Var allocation_status(buffer->name + ".allocation_status",
                            DataType::Int(32));
      output_allocation.push_back(
          SeqStmt({tirx::Bind(allocation_status,
                              Call(DataType::Int(32), builtin::call_extern(),
                                   {StringImm("TVMFFIEnvTensorAlloc"),
                                    prototype, output_slot})),
                   nop}));

      std::vector<Stmt> allocation_failure_cleanup;
      for (int previous = 0; previous < output_ordinal; ++previous) {
        PrimExpr previous_handle =
            Call(DataType::Handle(), builtin::tvm_struct_get(),
                 {output_storage, IntImm(DataType::Int(32), previous),
                  IntImm(DataType::Int(32), builtin::kTVMFFIAnyUnionValue)});
        allocation_failure_cleanup.push_back(
            Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                          {StringImm("TVMFFIObjectDecRef"), previous_handle})));
      }
      allocation_failure_cleanup.push_back(Evaluate(ret(allocation_status)));
      output_allocation.push_back(
          IfThenElse(allocation_status != 0,
                     SeqStmt::Flatten(allocation_failure_cleanup)));

      PrimExpr output_object =
          Call(DataType::Handle(), builtin::tvm_struct_get(),
               {output_storage, IntImm(DataType::Int(32), output_ordinal),
                IntImm(DataType::Int(32), builtin::kTVMFFIAnyUnionValue)});
      output_object_handles.push_back(output_object);
      PrimExpr output_tensor = Call(
          DataType::Handle(), builtin::handle_add_byte_offset(),
          {output_object, IntImm(DataType::Int(64),
                                 static_cast<int64_t>(sizeof(TVMFFIObject)))});
      output_binder.Bind(param, output_tensor,
                         name_hint + "." + param->name_hint, true);
      output_buffer_def.emplace_back(param, buffer);
      used_output_buffers.insert(param.get());
      arg_buffer_declarations.push_back(DeclBuffer(buffer));
    }

    output_binder.BindDLTensors(output_buffer_def, device_type, device_id,
                                name_hint, used_output_buffers,
                                output_shape_vars);

    if (num_outputs == 1) {
      callee_allocated_output_return.push_back(StoreFFIAny(
          v_result, TypeIndex::kTVMFFITensor, output_object_handles[0]));
    } else {
      PrimExpr array_args = Call(
          DataType::Handle(), builtin::handle_add_byte_offset(),
          {output_storage, IntImm(DataType::Int(64), num_outputs * kAnySize)});
      for (int i = 0; i < num_outputs; ++i) {
        PrimExpr arg_cell =
            Call(DataType::Handle(), builtin::handle_add_byte_offset(),
                 {array_args, IntImm(DataType::Int(64), i * kAnySize)});
        callee_allocated_output_return.push_back(StoreFFIAny(
            arg_cell, TypeIndex::kTVMFFITensor, output_object_handles[i]));
      }

      Var array_status("callee_allocated_output_array_status",
                       DataType::Int(32));
      callee_allocated_output_return.push_back(SeqStmt(
          {tirx::Bind(array_status,
                      Call(DataType::Int(32),
                           ::tvm::tl::tvm_ffi_call_with_result(),
                           {StringImm("ffi.Array"), array_args,
                            IntImm(DataType::Int(32), num_outputs), v_result})),
           nop}));
      for (const PrimExpr &output_object : output_object_handles) {
        callee_allocated_output_return.push_back(
            Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                          {StringImm("TVMFFIObjectDecRef"), output_object})));
      }
      callee_allocated_output_return.push_back(
          IfThenElse(array_status != 0, Evaluate(ret(array_status))));
    }
    callee_allocated_output_return.push_back(Evaluate(ret(Integer(0))));
  }

  // reset global symbol to attach prefix
  func = WithAttrs(
      std::move(func),
      {{tvm::attr::kCallingConv, static_cast<int>(CallingConv::kCPackedFunc)},
       {tvm::attr::kTarget, target_host},
       {tvm::attr::kGlobalSymbol,
        symbol::tvm_ffi_symbol_prefix + global_symbol.value()}});

  Stmt body = ReturnRewriter(v_result)(func_ptr->body);
  for (auto it = assume_runtime_checks.rbegin();
       it != assume_runtime_checks.rend(); ++it) {
    Stmt assertion = AssertStmt(it->condition, StringImm("RuntimeError"),
                                it->message_parts, it->span);
    body = SeqStmt::Flatten(std::move(assertion), std::move(body));
  }
  body = AttrStmt(make_zero(DataType::Int(32)), tirx::attr::compute_scope,
                  StringImm(name_hint + "_compute_"), body);
  // Set device context
  if (vmap.count(device_id.get())) {
    Any node = String("default");
    seq_check.push_back(AttrStmt(node, tirx::attr::device_id, device_id, nop));
    seq_check.push_back(
        AttrStmt(node, tirx::attr::device_type, device_type, nop));

    if (runtime::DeviceAPI::NeedSetDevice(target_device_type)) {
      Stmt set_device =
          Evaluate(Call(DataType::Int(32), tirx::builtin::tvm_call_packed(),
                        {StringImm(runtime::symbol::tvm_set_device),
                         device_type, device_id}));
      body = SeqStmt({set_device, body});
    }
  }

  if (use_callee_allocated_output_abi) {
    std::vector<Stmt> body_and_return{body};
    body_and_return.insert(body_and_return.end(),
                           callee_allocated_output_return.begin(),
                           callee_allocated_output_return.end());
    body = SeqStmt::Flatten(body_and_return);
  } else {
    // Return error code of zero on success.
    body = SeqStmt({body, Evaluate(ret(Integer(0)))});
  }

  body = MergeNest({output_binder.InitNest(), output_binder.Asserts(),
                    arg_buffer_declarations},
                   body);
  if (!output_allocation.empty()) {
    std::vector<Stmt> allocation_and_body = output_allocation;
    allocation_and_body.push_back(body);
    body = SeqStmt::Flatten(allocation_and_body);
  }
  body = MergeNest({seq_init, binder.InitNest(), seq_check, binder.Asserts()},
                   body);
  func_ptr->body = body;
  func_ptr->params = args;

  Array<Var> undefined = UndefinedVars(body, func_ptr->params);

  ICHECK_EQ(undefined.size(), 0)
      << "In PrimFunc " << name_hint << " variables " << undefined
      << " are used, but are not passed in as API arguments";

  func_ptr->buffer_map = Map<Var, Buffer>();
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

        std::vector<int> callee_allocated_output_indices =
            GetCalleeAllocatedOutputIndices(func);
        func = MakePackedAPI(std::move(func), callee_allocated_output_indices);
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
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.MakePackedAPI",
                        []() { return MakePackedAPI(); });
}

} // namespace tl
} // namespace tvm
