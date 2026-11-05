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
 * \brief Stage TileLang contracts around native host/device splitting.
 */
#include "../ir/alias_analysis.h"
#include "../ir/annotations.h"
#include "../ir/lexical_scope.h"
#include "../op/native_utils.h"
#include "common/unused_tensor_alias.h"
#include "tensor_descriptor.h"
#include "upstream_pass_bridge.h"
#include <string>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace transform {
tvm::transform::Pass LowerDeviceKernelLaunch();
}
namespace {

// Raw CUDA source has no tensor uses in its placeholder body. Explicit native
// operands retain the complete source signature through use-def and outlining.
// Native outlining keeps the baseline ordering: pointers first, then scalars,
// with each group sorted by name. A TensorVar's type also exposes its symbolic
// shape/stride/layout dependencies, without an out-of-band buffer_map.
class SourceKernelCaptureStager : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit SourceKernelCaptureStager(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      if (parameter.as<tirx::TensorVar>())
        tensors_.push_back(parameter);
    }
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                                 InplaceMode mode) final {
    if (region->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(region, mode);
    auto fields = DecodeAttributeRegion(region);
    auto key = fields.at("key").cast<ffi::String>();
    auto body = Mutate(region->body, mode).ValueOrUnchanged(region->body);
    // Replace the frontend's possibly partial list with the finalized native
    // signature. This also supports native IR clients constructing the source
    // region without the Python language facade.
    if (key == "code_block_signature") {
      return tl::MakeLexicalScope(body, region->loc);
    }
    if (key == "code_block_source") {
      auto encoded = EncodeAnnotations(ffi::Map<ffi::String, ffi::Any>{
          {"node", ffi::Any()},
          {"key", ffi::String("code_block_signature")},
          {"value", tensors_}});
      body =
          SeqStmt(RegionStmt(Op::Get("tl.attr"), {encoded.payload}, {},
                             DictAttrs({{"schema_version", 1},
                                        {"annotation_schema", encoded.schema}}),
                             body, {}, region->loc));
    }
    if (key != "code_block_source" && body.same_as(region->body))
      return ffi::Unchanged();
    return RegionStmt(region->op, region->args, region->body_params,
                      region->attrs, body, region->result_vars, region->loc);
  }

private:
  ffi::Array<Expr> tensors_;
};

class SourceKernelAttrExtractor : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                                 InplaceMode mode) final {
    if (region->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(region, mode);
    auto fields = DecodeAttributeRegion(region);
    auto key = fields.at("key").cast<ffi::String>();
    if (key != "code_block_source" && key != "code_block_entry_name" &&
        key != "code_block_signature")
      return StmtExprMutator::Mutate_(region, mode);
    if (key != "code_block_signature") {
      auto value = fields.at("value");
      ffi::String text = value.as<StringImmObj>()
                             ? value.as<StringImmObj>()->value
                             : value.cast<ffi::String>();
      TVM_FFI_CHECK(!text.empty(), ValueError)
          << "CUDA source kernel `" << key << "` must not be empty";
      auto &slot = key == "code_block_source" ? source : entry;
      TVM_FFI_CHECK(!slot || slot.value() == text, ValueError)
          << "Conflicting CUDA source kernel `" << key
          << "` values in one device region";
      slot = text;
    }
    auto body = Mutate(region->body, mode).ValueOrUnchanged(region->body);
    return tl::MakeLexicalScope(body, region->loc);
  }

  ffi::Optional<ffi::String> source;
  ffi::Optional<ffi::String> entry;
};

class SourceKernelSymbolRewriter : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit SourceKernelSymbolRewriter(ffi::Map<GlobalVar, GlobalVar> remap)
      : remap_(std::move(remap)) {}

  ffi::UnchangedOr<Expr> Mutate_(const GlobalVarObj *var, InplaceMode) final {
    if (auto replacement = remap_.Get(ffi::GetRef<GlobalVar>(var)))
      return replacement.value();
    return ffi::Unchanged();
  }

private:
  ffi::Map<GlobalVar, GlobalVar> remap_;
};

tvm::transform::Pass FinalizeSourceKernels() {
  return tvm::transform::CreateModulePass(
      [](IRModule module, tvm::transform::PassContext) {
        ffi::Map<GlobalVar, BaseFunc> updated;
        ffi::Map<GlobalVar, GlobalVar> renamed;
        for (const auto &[symbol, base] : module->functions) {
          auto function = base.as<tirx::Function>();
          if (!function || function.value()
                               ->GetAttr(tvm::attr::kDeclaration)
                               .value_or(false)) {
            updated.Set(symbol, base);
            continue;
          }
          auto extractor = ffi::make_object<SourceKernelAttrExtractor>();
          auto original = function.value()->body;
          auto body = extractor->Mutate(original).ValueOrUnchanged(original);
          if (!extractor->source) {
            TVM_FFI_CHECK(!extractor->entry, ValueError)
                << "CUDA source entry metadata requires code_block_source";
            updated.Set(symbol, base);
            continue;
          }
          auto target = function.value()->GetAttr<Target>(tvm::attr::kTarget);
          TVM_FFI_CHECK(target && target.value()->kind->name == "cuda" &&
                            !target.value()->GetHost(),
                        ValueError)
              << "CUDASourceCodeKernel requires an outlined CUDA device region";
          auto result = function.value();
          result.CopyOnWrite()->body = body;
          result =
              WithAttr(result, "code_block_source", extractor->source.value());
          GlobalVar destination = symbol;
          if (extractor->entry) {
            auto name = extractor->entry.value();
            if (name != symbol->name_hint) {
              destination = GlobalVar(name, symbol->loc);
              renamed.Set(symbol, destination);
            }
            result = WithAttr(result, "code_block_entry_name", name);
          }
          result = WithAttr(result, tvm::attr::kGlobalSymbol.name,
                            destination->name_hint);
          updated.Set(destination, result);
        }
        // Only final symbols collide: a generated outlining name may be freed
        // when a different source kernel receives its explicit entry name.
        std::unordered_set<std::string> names;
        for (const auto &[symbol, unused] : updated) {
          TVM_FFI_CHECK(names.insert(symbol->name_hint).second, ValueError)
              << "Duplicate CUDA source entry `" << symbol->name_hint
              << "`; raw CUDA source entry names cannot be renamed";
        }
        if (!renamed.empty()) {
          auto rewriter = ffi::make_object<SourceKernelSymbolRewriter>(renamed);
          ffi::Map<GlobalVar, BaseFunc> rewritten;
          for (const auto &[symbol, base] : updated) {
            auto function = base.as<tirx::Function>();
            if (function && !function.value()
                                 ->GetAttr(tvm::attr::kDeclaration)
                                 .value_or(false)) {
              auto result = function.value();
              auto original = result->body;
              result.CopyOnWrite()->body =
                  rewriter->Mutate(original).ValueOrUnchanged(original);
              rewritten.Set(symbol, result);
            } else {
              rewritten.Set(symbol, base);
            }
          }
          updated = std::move(rewritten);
        }
        return IRModule(updated, module->attrs, module->global_infos);
      },
      0, "tl.FinalizeSourceKernels");
}

// The packed input descriptor exists on the host. Simplification may inline a
// pure getter into a device region, so pin each original parameter's handle at
// entry before outlining. Device code captures the pointer, never reconstructs
// the descriptor from a flattened data view.
class TensorDescriptorStager : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit TensorDescriptorStager(const tirx::Function &function)
      : parameters_(function->params) {}

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    if (!call->op.same_as(input_tensor_descriptor_op()))
      return StmtExprMutator::Mutate_(call, mode);
    int index = InputTensorDescriptorIndex(call, parameters_);
    if (auto found = descriptors_.find(index); found != descriptors_.end())
      return found->second;
    Var pointer(std::string(parameters_[index]->name) + "_descriptor",
                PtrType::VoidPtrType(), call->loc);
    prefix.push_back(Bind(pointer, ffi::GetRef<Expr>(call), call->loc));
    descriptors_.emplace(index, pointer);
    return pointer;
  }

  ffi::Array<Stmt> prefix;

private:
  ffi::Array<Var> parameters_;
  std::unordered_map<int, Var> descriptors_;
};

class HostAssumptionStager : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit HostAssumptionStager(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      if (parameter.as<PrimVar>())
        host_vars_.insert(parameter.get());
      VisitNativePostOrder(parameter->ty, [&](const ffi::ObjectRef &node) {
        if (const auto *variable = node.as<VarObj>();
            variable && variable->ty.as<PrimTypeObj>())
          host_vars_.insert(variable);
      });
    }
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    // Facts apply to following statements in this sequence. A nested control
    // region restores its incoming set, so a branch-local fact cannot leak to
    // its sibling or to an earlier device launch.
    const size_t incoming = host_assumptions_.size();
    ffi::Array<Stmt> statements;
    for (const Stmt &statement : op->seq) {
      Stmt rewritten = Mutate(statement, mode).ValueOrUnchanged(statement);
      statements.push_back(rewritten);
      if (device_checks_ == nullptr)
        AppendHostFacts(rewritten);
    }
    host_assumptions_.erase(host_assumptions_.begin() + incoming,
                            host_assumptions_.end());
    return SeqStmt(statements, op->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    // The old CPU lowering removed singleton grid loops before splitting.
    // Native loops can retain this scope. A literal, non-wrapping one-trip
    // loop with no early exit still executes its body unconditionally; do
    // not lose the checked assumption merely because that loop remains.
    const bool once = IsUnconditionalSingleIteration(op);
    auto incoming = constant_iterations_;
    if (once)
      constant_iterations_.Set(op->var, op->begin);
    else
      ++conditional_depth_;
    auto result = StmtExprMutator::Mutate_(op, mode);
    constant_iterations_ = std::move(incoming);
    if (!once)
      --conditional_depth_;
    return result;
  }
  ffi::UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    ++conditional_depth_;
    auto result = StmtExprMutator::Mutate_(op, mode);
    --conditional_depth_;
    return result;
  }
  ffi::UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode mode) final {
    ++conditional_depth_;
    auto result = StmtExprMutator::Mutate_(op, mode);
    --conditional_depth_;
    return result;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    bool starts_device =
        device_checks_ == nullptr && (op->op->name == "tirx.device_scope" ||
                                      op->op->name == "tirx.device_entry" ||
                                      op->op.same_as(tirx::launch_thread_op()));
    if (starts_device) {
      ffi::Array<Stmt> checks;
      device_checks_ = &checks;
      auto rewritten = StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(
          ffi::GetRef<Stmt>(op));
      device_checks_ = nullptr;
      // Only facts over already captured values belong in the device body.
      // Compare Var identity, never names, and let native outlining renew the
      // definitions and the inserted uses together. An unrelated host scalar
      // must not become a new device argument just because it has a fact.
      std::unordered_set<const VarObj *> available;
      for (const Var &variable : UndefinedVars(ffi::GetRef<Stmt>(op)))
        available.insert(variable.get());
      ffi::Array<Stmt> body;
      for (const PrimExpr &condition : host_assumptions_) {
        bool captured = true;
        for (const Var &variable : UndefinedVars(condition))
          captured &= available.count(variable.get()) != 0;
        if (captured)
          body.push_back(Evaluate(prim::assume(condition, op->loc), op->loc));
      }
      if (!body.empty()) {
        const auto *region = rewritten.as<RegionStmtObj>();
        TVM_FFI_ICHECK(region);
        body.push_back(region->body);
        rewritten = RegionStmt(region->op, region->args, region->body_params,
                               region->attrs, SeqStmt(body, region->loc),
                               region->result_vars, region->loc);
      }
      checks.push_back(rewritten);
      return SeqStmt(checks, op->loc);
    }
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    auto fields = DecodeAttributeRegion(op);
    if (fields.at("key").cast<ffi::String>() !=
        "tl.assume_requires_runtime_check")
      return StmtExprMutator::Mutate_(op, mode);
    PrimExpr condition = SubstituteLoopVars(fields.at("node").cast<PrimExpr>(),
                                            constant_iterations_);
    ffi::Array<Stmt> prefix;
    if (conditional_depth_ == 0 && HostEvaluable(condition)) {
      auto message = fields.at("value").cast<ffi::String>();
      Stmt check = AssertStmt(condition, StringImm("RuntimeError"),
                              {StringImm(message)}, op->loc);
      if (device_checks_) {
        device_checks_->push_back(check);
        // Preserve the optimizer fact on the host as well, strictly after its
        // runtime check. Repeated simplification cannot justify the check
        // using the condition that the check itself is meant to validate.
        device_checks_->push_back(
            Evaluate(prim::assume(condition, op->loc), op->loc));
      } else {
        prefix.push_back(check);
      }
    }
    // A conditional/device-dependent assume remains an optimizer hint in the
    // body. Like the baseline it does not become an unconditional host check.
    auto body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    prefix.push_back(tl::MakeLexicalScope(body, op->loc));
    return SeqStmt(prefix, op->loc);
  }

private:
  static bool IsUnconditionalSingleIteration(const ForObj *loop) {
    if (loop->kind != ForKind::kDefault)
      return false;
    const auto *begin = loop->begin.as<IntImmObj>();
    const auto *end = loop->end.as<IntImmObj>();
    const auto *step = loop->step.as<IntImmObj>();
    if (!begin || !end || !step || step->value <= 0 ||
        begin->value >= end->value)
      return false;
    // Compute outside the IR dtype. In int8, (120, 121, 127) wraps and
    // iterates again; a mathematical trip count of one would be unsound.
    ffi::BigInt next = begin->value + step->value;
    PrimType type = loop->var.ty();
    ffi::BigInt maximum =
        (ffi::BigInt(1) << (type.bits() - (type.code() == kDLInt ? 1 : 0))) - 1;
    if (next < end->value || next > maximum)
      return false;
    bool early_exit = false;
    VisitNativePostOrder(loop->body, [&](const ffi::ObjectRef &node) {
      if (node.as<BreakObj>() || node.as<ContinueObj>() || node.as<ReturnObj>())
        early_exit = true;
      if (const auto *call = node.as<CallObj>()) {
        if (const auto *op = call->op.as<OpObj>();
            op && op->name == "tl.loop_break")
          early_exit = true;
      }
    });
    return !early_exit;
  }

  void AppendHostFacts(const Stmt &statement) {
    if (const auto *sequence = statement.as<SeqStmtObj>()) {
      for (const Stmt &child : sequence->seq)
        AppendHostFacts(child);
      return;
    }
    const auto *evaluate = statement.as<EvaluateObj>();
    const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
    if (call && call->op.same_as(prim::assume_op())) {
      PrimExpr condition = SubstituteLoopVars(
          call->args[0].as_or_throw<PrimExpr>(), constant_iterations_);
      if (HostEvaluable(condition))
        host_assumptions_.push_back(condition);
    }
  }

  bool HostEvaluable(const PrimExpr &condition) const {
    if (!condition.ty().MatchesCode(kDLBool) || !condition.ty().IsScalar())
      return false;
    bool valid = true;
    VisitNativePostOrder(condition, [&](const ffi::ObjectRef &node) {
      if (node.as<CallObj>() || node.as<TensorLoadObj>())
        valid = false;
      if (const auto *var = node.as<VarObj>(); var && !host_vars_.count(var))
        valid = false;
    });
    return valid;
  }
  std::unordered_set<const VarObj *> host_vars_;
  std::vector<PrimExpr> host_assumptions_;
  ffi::Map<Var, PrimExpr> constant_iterations_;
  ffi::Array<Stmt> *device_checks_{nullptr};
  int conditional_depth_{0};
};

// Native launch lowering may retain a GlobalVar call between functions with
// the same target. TileLang emits outlined CPU compute functions in a separate
// runtime module, so calls from the host wrapper cross an external ABI even
// when both targets are identical. Device-internal calls remain GlobalVars.
class CPUComputeCallFinalizer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  explicit CPUComputeCallFinalizer(
      std::unordered_map<const GlobalVarObj *, ffi::String> exports)
      : exports_(std::move(exports)) {
    for (const auto &[symbol, name] : exports_)
      export_names_.insert(name);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *binding,
                                 InplaceMode mode) final {
    const auto *call = binding->value.as<CallObj>();
    const auto status_type = binding->var->ty.as<PrimType>();
    if (call && status_type && status_type.value() == PrimType::Int(32)) {
      bool compute_call = exports_.count(call->op.as<GlobalVarObj>());
      if (call->op.same_as(tirx::call_extern_op()) && !call->args.empty()) {
        if (const auto *name = call->args[0].as<StringImmObj>())
          compute_call |= export_names_.count(name->value);
      }
      if (compute_call)
        status_vars_.insert(binding->var.get());
    }
    return StmtExprMutator::Mutate_(binding, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const AssertStmtObj *check,
                                 InplaceMode mode) final {
    // Native outlining creates this exact status guard. Replacing a compute
    // error with its generic guard text would discard the original ValueError
    // and pointer constraint diagnostic already stored in the FFI TLS slot.
    const auto *condition = check->condition.as<prim::EQObj>();
    const auto *status = condition ? condition->a.as<VarObj>() : nullptr;
    const auto *success = condition ? condition->b.as<IntImmObj>() : nullptr;
    if (status && status_vars_.count(status) && success &&
        success->value == 0 && check->error_kind->value == "RuntimeError" &&
        check->message_parts.size() == 1 &&
        check->message_parts[0]->value == "Error executing compute kernel") {
      return If(
          prim::Not(check->condition),
          SeqStmt(Evaluate(Call(PrimType::Void(), tirx::throw_last_error_op(),
                                {}, std::nullopt, {}, check->loc),
                           check->loc)),
          std::nullopt, check->loc);
    }
    return StmtExprMutator::Mutate_(check, mode);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    auto current = StmtExprMutator::Mutate_(call, mode)
                       .ValueOrUnchanged(ffi::GetRef<Expr>(call))
                       .as_or_throw<Call>();
    const auto *symbol = current->op.as<GlobalVarObj>();
    auto exported = exports_.find(symbol);
    if (exported == exports_.end())
      return current;
    required_exports.insert(symbol);
    ffi::Array<Expr> arguments{StringImm(exported->second)};
    for (const Expr &argument : current->args)
      arguments.push_back(argument);
    Type return_type = IsVoidType(current->ty) ? PrimType::Void() : current->ty;
    return Call(return_type, tirx::call_extern_op(), arguments, current->attrs,
                current->ty_args, current->loc);
  }

  std::unordered_set<const GlobalVarObj *> required_exports;

private:
  std::unordered_map<const GlobalVarObj *, ffi::String> exports_;
  std::unordered_set<ffi::String> export_names_;
  std::unordered_set<const VarObj *> status_vars_;
};

tvm::transform::Pass FinalizeCPUComputeCalls() {
  return tvm::transform::CreateModulePass(
      [](IRModule module, tvm::transform::PassContext) {
        std::unordered_map<const GlobalVarObj *, ffi::String> exports;
        for (const auto &[symbol, base] : module->functions) {
          auto function = base.as<tirx::Function>();
          if (!function || !function.value()
                                ->GetAttr(tirx::attr::kIsGlobalFunc)
                                .value_or(false))
            continue;
          auto target = function.value()->GetAttr(tvm::attr::kTarget);
          if (!target || target.value()->GetHost() ||
              target.value()->GetTargetDeviceType() != kDLCPU)
            continue;
          exports.emplace(symbol.get(), function.value()
                                            ->GetAttr(tvm::attr::kGlobalSymbol)
                                            .value_or(symbol->name_hint));
        }
        if (exports.empty())
          return module;

        auto functions = module->functions;
        std::unordered_set<const GlobalVarObj *> required_exports;
        for (const auto &[symbol, base] : module->functions) {
          auto function = base.as<tirx::Function>();
          if (!function ||
              function.value()
                  ->GetAttr(tvm::attr::kDeclaration)
                  .value_or(false) ||
              function.value()
                  ->GetAttr(tirx::attr::kIsGlobalFunc)
                  .value_or(false))
            continue;
          auto target = function.value()->GetAttr(tvm::attr::kTarget);
          if (!target || !target.value()->GetHost() ||
              target.value()->GetHost().value()->GetTargetDeviceType() !=
                  kDLCPU)
            continue;
          auto finalizer = ffi::make_object<CPUComputeCallFinalizer>(exports);
          auto original = function.value()->body;
          auto body = finalizer->Mutate(original, InplaceMode::kDisallow);
          if (!body.UnchangedOrSameAs(original)) {
            auto updated = function.value();
            updated.CopyOnWrite()->body = std::move(body).ValueUnchecked();
            functions.Set(symbol, updated);
          }
          required_exports.insert(finalizer->required_exports.begin(),
                                  finalizer->required_exports.end());
        }
        if (required_exports.empty())
          return module;
        for (const auto &[symbol, base] : module->functions) {
          if (!required_exports.count(symbol.get()))
            continue;
          auto function = base.as_or_throw<tirx::Function>();
          if (!function->GetAttr(tvm::attr::kGlobalSymbol)) {
            functions.Set(symbol,
                          WithAttr(function, tvm::attr::kGlobalSymbol.name,
                                   exports.at(symbol.get())));
          }
        }
        return IRModule(functions, module->attrs, module->global_infos);
      },
      0, "tl.FinalizeCPUComputeCalls");
}
} // namespace

tvm::transform::Pass PrepareHostDevice(bool lower_launches) {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    function = LowerTensorDescriptorParameters(std::move(function));
    auto source_captures =
        ffi::make_object<SourceKernelCaptureStager>(function);
    auto unstaged = function->body;
    function.CopyOnWrite()->body =
        source_captures->Mutate(unstaged).ValueOrUnchanged(unstaged);
    auto descriptors = ffi::make_object<TensorDescriptorStager>(function);
    auto unpinned = function->body;
    auto pinned = descriptors->Mutate(unpinned).ValueOrUnchanged(unpinned);
    if (!descriptors->prefix.empty()) {
      descriptors->prefix.push_back(pinned);
      pinned = SeqStmt(descriptors->prefix, function->loc);
    }
    function.CopyOnWrite()->body = pinned;
    auto original = function->body;
    function.CopyOnWrite()->body = RemoveUnusedTensorAliases(original);
    function = StageNonRestrictForSplit(function);
    auto stager = ffi::make_object<HostAssumptionStager>(function);
    auto body = function->body;
    function.CopyOnWrite()->body = stager->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  auto stage =
      tirx::transform::CreateFunctionPass(pass, 0, "tl.StageHostAssumptions");
  auto finalize_aliases = tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        if (!function->GetAttr<bool>("tl.host_may_alias").value_or(false))
          return function;
        // The packed host wrapper may outline a compute_scope over unpacked
        // tensors. It cannot assume those captures are independent when the
        // source permits aliasing. Device functions already carry their
        // precise per-pointer exceptions and do not inherit this host marker.
        function = WithAttr(function, "tirx.noalias", false);
        return WithoutAttr(function, "tl.host_may_alias");
      },
      0, "tl.FinalizeSplitAliasContract");
  ffi::Array<tvm::transform::Pass> passes{
      stage, upstream_pass_bridge::OutlineDeviceRegions(),
      FinalizeSourceKernels()};
  if (lower_launches) {
    // The full entry point must consume the same pending configuration and
    // final storage contract as OutlineDeviceRegions followed by the late
    // TileLang pass. The native lowerer alone only handles ordinary calls.
    passes.push_back(transform::LowerDeviceKernelLaunch());
    passes.push_back(FinalizeCPUComputeCalls());
  }
  passes.push_back(finalize_aliases);
  return tvm::transform::Sequential(passes, lower_launches
                                                ? "tl.SplitHostDevice"
                                                : "tl.OutlineDeviceRegions");
}
tvm::transform::Pass SplitHostDevice() { return PrepareHostDevice(true); }
tvm::transform::Pass OutlineDeviceRegions() { return PrepareHostDevice(false); }
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.transform.SplitHostDevice", SplitHostDevice)
      .def("tl.transform.OutlineDeviceRegions", OutlineDeviceRegions);
}
} // namespace tvm::tl
