#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/dtype.h"
#include "op/prim.h"
#include "transform/tirx_passes.h"

#include "common/attr.h"
#include "compat/expr.h"
#include "compat/visit.h"

#include "transform/common/packed_api.h"
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/type.h>

#include <algorithm>
#include <unordered_set>

namespace tvm::tl {
namespace {

using tirx::TensorVar;

constexpr const char *kOutputIndices = "tilelang_out_idx";
constexpr const char *kAllocateOutputs = "tilelang_callee_allocated_outputs";

Expr Field(const Expr &base, int index, int field, Type type) {
  return Call(type, tirx::abi_field_get_op(),
              {base, IntImm::Int32(index), IntImm::Int32(field)});
}

PrimExpr IntField(const Expr &base, int index, int field) {
  return Field(base, index, field, PrimType::Int(32)).as_or_throw<PrimExpr>();
}

Stmt SetField(const Expr &base, int index, int field, Expr value) {
  return Evaluate(
      Call(PrimType::Int(32), tirx::abi_field_set_op(),
           {base, IntImm::Int32(index), IntImm::Int32(field), value}));
}

Expr ByteOffset(const Expr &pointer, int64_t offset) {
  return Call(PtrType::VoidPtrType(), tirx::handle_add_byte_offset_op(),
              {pointer, IntImm::Int64(offset)});
}

Stmt Check(const PrimExpr &condition, const ffi::String &message) {
  return AssertStmt(condition, StringImm("RuntimeError"), {StringImm(message)});
}

Stmt StoreResult(const Expr &result, int type_index, const Expr &value) {
  return SeqStmt(
      {SetField(result, 0, tirx::kTVMFFIAnyTypeIndex,
                IntImm::Int32(type_index)),
       SetField(result, 0, tirx::kTVMFFIAnyZeroPadding, IntImm::Int32(0)),
       SetField(result, 0, tirx::kTVMFFIAnyUnionValue, value)});
}

Stmt Release(const Expr &object) {
  return Evaluate(Call(PrimType::Int(32), tirx::call_extern_op(),
                       {StringImm("TVMFFIObjectDecRef"), object}));
}

ffi::Array<int> OutputIndices(const tirx::Function &function) {
  auto enabled = function->attrs->dict.Get(kAllocateOutputs);
  if (!enabled || GetIntAnnotation(enabled.value()).value_or(0) == 0 ||
      !function->GetAttr(tirx::attr::kIsEntryFunc).value_or(false)) {
    return {};
  }
  auto indices = function->GetAttr<ffi::Array<ffi::Any>>(kOutputIndices);
  TVM_FFI_CHECK(indices && !indices.value().empty(), ValueError)
      << kAllocateOutputs << " requires non-empty " << kOutputIndices;
  ffi::Array<int> result;
  std::unordered_set<int> seen;
  int count = static_cast<int>(function->params.size());
  for (const ffi::Any &raw : indices.value()) {
    auto integer = GetIntAnnotation(raw);
    TVM_FFI_CHECK(integer && *integer >= -count && *integer < count, ValueError)
        << kOutputIndices << " contains an out-of-range parameter index";
    int index = *integer < 0 ? *integer + count : *integer;
    TVM_FFI_CHECK(seen.insert(index).second, ValueError)
        << kOutputIndices << " contains duplicate index " << index;
    TVM_FFI_CHECK(function->params[index].as<tirx::TensorVar>(), TypeError)
        << kOutputIndices << " index " << index << " is not a tensor parameter";
    result.push_back(index);
  }
  return result;
}

struct OutputStorage {
  TensorVar tensor;
  PrimType dtype;
  ffi::Array<PrimExpr> shape;
  ffi::Optional<PrimExpr> packing_condition;
};

OutputStorage ResolveStorage(const TensorVar &tensor) {
  auto resolver =
      ffi::Function::GetGlobal("tl.tvm_ffi.resolve_output_storage_dtype");
  TVM_FFI_CHECK(resolver, ValueError)
      << "Callee-allocated outputs require "
         "tl.tvm_ffi.resolve_output_storage_dtype";
  PrimType storage((*resolver)(tensor.dtype()->dtype).cast<DLDataType>());
  int logical_bits = tensor.dtype().bits() * tensor.dtype().lanes();
  int storage_bits = storage.bits() * storage.lanes();
  TVM_FFI_CHECK(logical_bits > 0 && storage_bits >= logical_bits &&
                    storage_bits % logical_bits == 0,
                ValueError)
      << "Output storage dtype " << storage
      << " must contain an integral number of " << tensor.dtype() << " values";
  ffi::Array<PrimExpr> shape = tensor.shape();
  ffi::Optional<PrimExpr> condition;
  if (storage_bits > logical_bits) {
    TVM_FFI_CHECK(!shape.empty(), ValueError)
        << "Packed output requires a shape dimension";
    PrimExpr factor = IntImm(shape.back().ty(), storage_bits / logical_bits);
    condition = prim::FloorMod(shape.back(), factor) == 0;
    shape.Set(shape.size() - 1, prim::FloorDiv(shape.back(), factor));
  }
  return {tensor, storage, shape, condition};
}

class RuntimeAssumptions : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    ++conditional_depth_;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfNode *node,
                                 ffi::InplaceMode mode) final {
    ++conditional_depth_;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *node,
                                 ffi::InplaceMode mode) final {
    ++conditional_depth_;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto rewritten = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                         .as_or_throw<RegionStmt>();
    auto condition =
        rewritten->attrs.GetAttr<PrimExpr>(attr::kAssumeRequiresRuntimeCheck);
    if (!condition)
      return rewritten;
    auto message = rewritten->attrs.GetAttr<ffi::String>("tl.assume_message")
                       .value_or("Assumption failed");
    auto attributes = rewritten->attrs->dict;
    attributes.erase(attr::kAssumeRequiresRuntimeCheck);
    attributes.erase("tl.assume_message");
    rewritten.CopyOnWrite()->attrs = DictAttrs(attributes);
    if (conditional_depth_ != 0)
      return rewritten;
    return SeqStmt({Check(condition.value(), message), rewritten});
  }

  int conditional_depth_{0};
};

class OutputReturnRewriter : public tvm::tl::StmtExprMutator {
public:
  explicit OutputReturnRewriter(Stmt epilogue) : epilogue_(epilogue) {}
  using tvm::tl::StmtExprMutator::Mutate;

private:
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    bool parallel = node->kind == ForKind::kParallel;
    parallel_depth_ += parallel;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    parallel_depth_ -= parallel;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    bool parallel = node->op.same_as(tirx::parallel_launch_op());
    parallel_depth_ += parallel;
    auto result = tvm::tl::StmtExprMutator::Mutate_(node, mode);
    parallel_depth_ -= parallel;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ReturnNode *node,
                                 ffi::InplaceMode mode) final {
    TVM_FFI_CHECK(parallel_depth_ == 0, ValueError)
        << "Return cannot be used in parallel scope";
    auto tuple = node->value.as<TupleNode>();
    auto type = node->value->ty.as<PrimType>();
    TVM_FFI_CHECK(node->value->ty.as<NoneTypeNode>() ||
                      (tuple && tuple->fields.empty()) ||
                      (type && type.value().IsVoid()),
                  ValueError)
        << "Callee-allocated outputs cannot be combined with a scalar return "
           "value";
    return epilogue_;
  }

  Stmt epilogue_;
  int parallel_depth_{0};
};

tirx::Function MakeOutputAPI(tirx::Function function,
                             const ffi::Array<int> &indices) {
  auto target = function->GetAttr(tvm::attr::kTarget);
  TVM_FFI_CHECK(target && target.value()->GetHost(), ValueError)
      << "MakePackedAPI requires a target with a host for callee-allocated "
         "outputs";
  ffi::String name = function->GetAttr(tvm::attr::kGlobalSymbol).value();
  Var self("self_handle", PtrType::VoidPtrType());
  Var packed_args("args", PtrType::VoidPtrType());
  PrimVar num_args("num_args", PrimType::Int(32));
  Var result("result", PtrType::VoidPtrType());
  Var anchor("allocator_anchor", PtrType::VoidPtrType());
  PrimVar device_id("dev_id", PrimType::Int(32));
  PrimExpr device_type = IntImm::Int32(target.value()->GetTargetDeviceType());
  std::unordered_set<int> omitted(indices.begin(), indices.end());
  ffi::Array<Var> inputs;
  for (size_t index = 0; index < function->params.size(); ++index) {
    if (!omitted.count(index))
      inputs.push_back(function->params[index]);
  }
  int anchor_index = inputs.size();
  inputs.push_back(anchor);
  auto decoded = tirx_passes::DecodePackedAPIArguments(
      name, inputs, packed_args, num_args, device_type, device_id,
      function->GetAttr<ffi::Map<Var, PrimType>>("tirx.packed_storage_dtypes")
          .value_or(ffi::Map<Var, PrimType>{}),
      function->GetAttr<ffi::Array<Var>>("tirx.nullable_params")
          .value_or(ffi::Array<Var>{}),
      function->GetAttr<ffi::Array<Var>>("tirx.required_metadata")
          .value_or(ffi::Array<Var>{}));
  ffi::Array<Stmt> initialization = decoded.definitions;
  initialization.push_back(
      Check(IntField(packed_args, anchor_index, tirx::kTVMFFIAnyTypeIndex) ==
                ffi::TypeIndex::kTVMFFITensor,
            name + ": allocator anchor must be a Tensor"));
  initialization.push_back(
      Check(prim::Not(Call(PrimType::Bool(), tirx::isnullptr_op(), {anchor})
                          .as_or_throw<PrimExpr>()),
            name + ": allocator anchor is NULL"));
  initialization.push_back(
      Check(IntField(anchor, 0, tirx::kDLTensorDeviceType) == device_type,
            name + ": allocator anchor has the wrong device type"));
  PrimExpr anchor_device = IntField(anchor, 0, tirx::kDLTensorDeviceId);
  if (decoded.bindings.count(device_id)) {
    initialization.push_back(
        Check(device_id == anchor_device,
              name + ": allocator anchor has the wrong device id"));
  } else {
    initialization.push_back(Bind(device_id, anchor_device));
  }

  ffi::Array<Stmt> validation = decoded.assertions;
  std::vector<OutputStorage> outputs;
  int shape_count = 0;
  sym::Analyzer analyzer;
  auto bind_or_check = [&](const PrimExpr &actual, const PrimExpr &expected,
                           const ffi::String &description) {
    PrimExpr cast_expected = tl::CastValue(expected, actual.ty());
    if (auto variable = actual.as<PrimVar>();
        variable && !decoded.bindings.count(variable.value())) {
      initialization.push_back(Bind(variable.value(), cast_expected));
      decoded.bindings.Set(variable.value(), cast_expected);
    } else if (!tl::arith::CanProve(analyzer, actual == cast_expected)) {
      validation.push_back(
          Check(actual == cast_expected, name + ": " + description));
    }
  };
  for (int index : indices) {
    TensorVar tensor = function->params[index].as_or_throw<TensorVar>();
    TVM_FFI_CHECK(tensor.scope().name() == "global", ValueError)
        << "Callee-allocated output must use global storage";
    for (const PrimExpr &extent : tensor.shape()) {
      PostOrderVisit(extent, [&](const ffi::ObjectRef &object) {
        if (const auto *variable = object.as<VarNode>()) {
          TVM_FFI_CHECK(decoded.bindings.count(ffi::GetRef<Var>(variable)),
                        ValueError)
              << "Cannot infer symbolic variable " << variable->name
              << " required by output " << tensor->name
              << "; bind it from an input tensor or scalar parameter";
        }
      });
      if (!tl::arith::CanProve(analyzer, extent >= 0)) {
        validation.push_back(Check(
            extent >= 0, name + ": output dimensions must be nonnegative"));
      }
    }
    auto pattern = tirx::MatchShapeStridePattern(tensor.ty());
    TVM_FFI_CHECK(pattern, ValueError)
        << "Callee-allocated output requires a strided tensor type";
    bind_or_check(pattern.value().get<2>(), IntImm::Int32(0),
                  "output offset must be zero");
    auto strides = pattern.value().get<1>();
    PrimExpr compact_stride = IntImm(tensor.ty()->DefaultIndexType(), 1);
    for (size_t dimension = tensor.shape().size(); dimension > 0; --dimension) {
      bind_or_check(strides[dimension - 1], compact_stride,
                    "output strides must be compact");
      compact_stride = compact_stride * tensor.shape()[dimension - 1];
    }
    outputs.push_back(ResolveStorage(tensor));
    if (outputs.back().packing_condition) {
      validation.push_back(Check(
          outputs.back().packing_condition.value(),
          name + ": packed output final dimension is not storage-aligned"));
    }
    shape_count += outputs.back().shape.size();
  }

  Var objects("output_objects", PtrType::VoidPtrType());
  Var shapes("output_shapes", PtrType::VoidPtrType());
  Var prototypes("output_prototypes", PtrType::VoidPtrType());
  int count = outputs.size();
  auto stack = [](const char *kind, int size) {
    return Call(PtrType::VoidPtrType(), tirx::stack_alloca_op(),
                {StringImm(kind), IntImm::Int32(size)});
  };
  ffi::Array<Stmt> allocation{
      Bind(objects, stack("tvm_ffi_any", count)),
      Bind(shapes, stack("shape", std::max(1, shape_count))),
      Bind(prototypes, stack("array", count))};
  ffi::Array<Expr> handles;
  ffi::Array<Stmt> declarations = decoded.tensor_declarations;
  int shape_offset = 0;
  Expr null_pointer =
      tirx::reinterpret(PtrType::VoidPtrType(), IntImm(PrimType::UInt(64), 0));
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    const auto &output = outputs[ordinal];
    for (size_t dimension = 0; dimension < output.shape.size(); ++dimension) {
      allocation.push_back(
          SetField(shapes, shape_offset + dimension, tirx::kInt64ArrayElem,
                   tl::CastValue(output.shape[dimension], PrimType::Int(64))));
    }
    auto field = [&](int kind, Expr value) {
      allocation.push_back(SetField(prototypes, ordinal, kind, value));
    };
    field(tirx::kDLTensorData, null_pointer);
    field(
        tirx::kDLTensorShape,
        tirx::reinterpret(PtrType(PrimType::Int(64)),
                          ByteOffset(shapes, shape_offset * sizeof(int64_t))));
    shape_offset += output.shape.size();
    field(tirx::kDLTensorStrides, null_pointer);
    field(tirx::kDLTensorNDim, IntImm::Int32(output.shape.size()));
    field(tirx::kDLTensorTypeCode,
          IntImm(PrimType::UInt(8), output.dtype.code()));
    field(tirx::kDLTensorTypeBits,
          IntImm(PrimType::UInt(8), output.dtype.bits()));
    field(tirx::kDLTensorTypeLanes,
          IntImm(PrimType::UInt(16), output.dtype.lanes()));
    field(tirx::kDLTensorByteOffset, IntImm(PrimType::UInt(64), 0));
    field(tirx::kDLTensorDeviceId, device_id);
    field(tirx::kDLTensorDeviceType, device_type);
    Expr prototype =
        Field(prototypes, ordinal, tirx::kDLTensorAddr, PtrType::VoidPtrType());
    Expr slot = ByteOffset(objects, ordinal * sizeof(TVMFFIAny) +
                                        offsetof(TVMFFIAny, v_ptr));
    PrimVar status(output.tensor->name + ".allocation_status",
                   PrimType::Int(32));
    allocation.push_back(Bind(
        status, Call(PrimType::Int(32), tirx::call_extern_op(),
                     {StringImm("TVMFFIEnvTensorAlloc"), prototype, slot})));
    ffi::Array<Stmt> cleanup;
    for (const Expr &handle : handles)
      cleanup.push_back(Release(handle));
    cleanup.push_back(Return(status));
    allocation.push_back(If(status != 0, SeqStmt(cleanup)));
    Expr handle = Field(objects, ordinal, tirx::kTVMFFIAnyUnionValue,
                        PtrType::VoidPtrType());
    handles.push_back(handle);
    Expr tensor_ptr = ByteOffset(handle, sizeof(TVMFFIObject));
    Expr data = Field(tensor_ptr, 0, tirx::kDLTensorData,
                      PtrType(output.tensor.dtype(), output.tensor.scope()));
    declarations.push_back(
        Bind(output.tensor, Call(output.tensor.ty(), tirx::decl_tensor_op(),
                                 {data}, std::nullopt, {output.tensor.ty()})));
  }

  ffi::Array<Stmt> return_statements;
  if (count == 1) {
    return_statements.push_back(
        StoreResult(result, ffi::TypeIndex::kTVMFFITensor, handles[0]));
  } else {
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      return_statements.push_back(
          StoreResult(ByteOffset(objects, ordinal * sizeof(TVMFFIAny)),
                      ffi::TypeIndex::kTVMFFITensor, handles[ordinal]));
    }
    PrimVar status("output_array_status", PrimType::Int(32));
    return_statements.push_back(Bind(
        status,
        Call(PrimType::Int(32), tirx::call_packed_op(),
             {StringImm("tl.runtime.CallGlobalWithResult"),
              StringImm("ffi.Array"), objects, IntImm::Int32(count), result})));
    for (const Expr &handle : handles)
      return_statements.push_back(Release(handle));
    return_statements.push_back(If(status != 0, Return(status)));
  }
  return_statements.push_back(Return(IntImm::Int32(0)));
  Stmt epilogue = SeqStmt(return_statements);
  Stmt body = ffi::make_object<OutputReturnRewriter>(epilogue)
                  ->Mutate(function->body.value(), ffi::InplaceMode::kDisallow)
                  .ValueOrUnchanged(function->body.value());
  body = RegionStmt(tirx::compute_scope_op(), {StringImm(name + "_compute_")},
                    {}, DictAttrs(), body);
  if (runtime::DeviceAPI::NeedSetDevice(
          target.value()->GetTargetDeviceType())) {
    PrimVar status("set_device_status", PrimType::Int(32));
    validation.push_back(
        Bind(status,
             Call(PrimType::Int(32), tirx::call_extern_op(),
                  {StringImm("TVMBackendSetDevice"), device_type, device_id})));
    validation.push_back(If(status != 0, Return(status)));
  }
  ffi::Array<Stmt> execution{SeqStmt(validation), SeqStmt(allocation),
                             SeqStmt(declarations), body, epilogue};
  initialization.push_back(RegionStmt(tirx::device_context_op(),
                                      {device_type, device_id}, {}, DictAttrs(),
                                      SeqStmt(execution)));
  auto *node = function.CopyOnWrite();
  node->params = {self, packed_args, num_args, result};
  node->body = SeqStmt(initialization);
  node->ret_type = PrimType::Int(32);
  auto attributes = node->attrs->dict;
  attributes.erase("tirx.nullable_params");
  attributes.erase("tirx.required_metadata");
  attributes.erase("tirx.packed_storage_dtypes");
  node->attrs = DictAttrs(attributes);
  auto undefined = tvm::UndefinedVars(node->body.value(), node->params);
  TVM_FFI_CHECK(undefined.empty(), ValueError)
      << "Unbound output ABI variables: " << undefined;
  return WithAttrs(
      std::move(function),
      {{tvm::attr::kCallingConv.name, CallingConv::kCPackedFunc},
       {tvm::attr::kGlobalSymbol.name,
        ffi::symbol::tvm_ffi_symbol_prefix + name},
       {tvm::attr::kTarget.name, target.value()->GetHost().value()}});
}

} // namespace

tvm::transform::Pass MakePackedAPI() {
  auto simplify =
      ffi::Function::GetGlobalRequired("tl.transform.Simplify")(false)
          .cast<tvm::transform::Pass>();
  auto prepare = [simplify](IRModule module,
                            tvm::transform::PassContext context) {
    IRModule updates;
    std::unordered_set<const GlobalVarNode *> output_entries;
    for (const auto &[global, base] : module->functions) {
      auto optional = base.as<tirx::Function>();
      if (!optional || !optional.value()->body ||
          optional.value()
                  ->GetAttr(tvm::attr::kCallingConv)
                  .value_or(CallingConv::kDefault) != CallingConv::kDefault ||
          !optional.value()->GetAttr(tvm::attr::kGlobalSymbol))
        continue;
      tirx::Function function = optional.value();
      IRModule candidate;
      candidate->Add(global, function);
      function = simplify(candidate, context)
                     ->Lookup(global)
                     .as_or_throw<tirx::Function>();
      Stmt body =
          ffi::make_object<RuntimeAssumptions>()
              ->Mutate(function->body.value(), ffi::InplaceMode::kDisallow)
              .ValueOrUnchanged(function->body.value());
      function.CopyOnWrite()->body = body;
      auto indices = OutputIndices(function);
      auto required_metadata = tvm::UndefinedVars(body);
      std::unordered_set<const VarNode *> live_parameters;
      for (const Var &variable : required_metadata)
        live_parameters.insert(variable.get());
      for (int index : indices) {
        live_parameters.insert(function->params[index].get());
        for (const Var &variable :
             tvm::UndefinedVars(function->params[index]->ty))
          required_metadata.push_back(variable);
      }
      ffi::Array<Var> nullable_parameters;
      for (const Var &parameter : function->params) {
        if (parameter.as<TensorVar>() &&
            !live_parameters.count(parameter.get()))
          nullable_parameters.push_back(parameter);
      }
      if (!nullable_parameters.empty()) {
        function = WithAttrs(std::move(function),
                             {{"tirx.nullable_params", nullable_parameters},
                              {"tirx.required_metadata", required_metadata}});
      }
      ffi::Map<Var, PrimType> storage_dtypes;
      for (const Var &parameter : function->params) {
        auto tensor = parameter.as<TensorVar>();
        if (!tensor || tensor.value().dtype().MatchesCode(kDLBool) ||
            (tensor.value().dtype().bits() >= 8 &&
             !tensor.value().dtype().MatchesCode(kDLFloat8_e4m3) &&
             !tensor.value().dtype().MatchesCode(kDLFloat4_e2m1fn) &&
             !::tvm::tl::IsCustomDType(tensor.value().dtype())))
          continue;
        auto resolver = ffi::Function::GetGlobalRequired(
            "tl.tvm_ffi.resolve_output_storage_dtype");
        PrimType storage(
            resolver(tensor.value().dtype()->dtype).cast<DLDataType>());
        if (storage != tensor.value().dtype())
          storage_dtypes.Set(parameter, storage);
      }
      if (!storage_dtypes.empty()) {
        function = WithAttr(std::move(function), "tirx.packed_storage_dtypes",
                            storage_dtypes);
      }
      if (!indices.empty()) {
        output_entries.insert(global.get());
        function = MakeOutputAPI(std::move(function), indices);
      }
      updates->Add(global, function);
    }
    if (!output_entries.empty()) {
      for (const auto &[global, function] : module->functions) {
        PostOrderVisit(function, [&](const ffi::ObjectRef &object) {
          if (const auto *call = object.as<CallNode>()) {
            const auto *callee = call->op.as<GlobalVarNode>();
            TVM_FFI_CHECK(!callee || !output_entries.count(callee), ValueError)
                << "Callee-allocated entry functions cannot be called as "
                   "subroutines";
          }
        });
      }
    }
    module.CopyOnWrite()->Update(updates);
    return module;
  };
  return tvm::transform::Sequential(
      {tvm::transform::CreateModulePass(prepare, 0, "tl.PreparePackedAPI"),
       tl::tirx_passes::ConvertSSA(), tirx_passes::MakePackedAPI()},
      "tl.MakePackedAPI");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MakePackedAPI", MakePackedAPI);
}

} // namespace tvm::tl
