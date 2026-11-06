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
 * \file kernel_rewriter.h
 * \brief Rewrite native TileLang kernel bodies with their lexical constraints.
 */

#pragma once

#include "ascend/analysis/storage.h"
#include "ascend/ir/task.h"
#include "ascend/transform/source_scopes.h"
#include "transform/common/constr_visitor.h"
#include <tvm/tirx/op/region.h>

#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

struct TilelangKernelContext {
  SeqStmt body;
  ConstrSet outer_ctx;
  ffi::Optional<PrimVar> outer_sid;
  // Raw launch hints and materialized tl.kernel fields describe the same
  // contract. They are read here and remain on their original native carrier.
  ScheduleUnitMetadata kernel_annotations;
  ParameterAliasContract parameter_alias_contract;
};

using TilelangKernelRewrite =
    std::function<SeqStmt(const TilelangKernelContext &)>;

namespace kernel_rewriter_detail {

inline bool IsKernelAnchor(const RegionStmtObj *op) {
  return op->op->name == "tl.launch_kernel" ||
         (op->op.same_as(tirx::launch_thread_op()) &&
          op->args[0].as_or_throw<StringImm>()->value == "blockIdx.x");
}

inline bool IsKernelBoundary(const RegionStmtObj *op) {
  if (op->op->name == "tl.launch_kernel" ||
      op->op.same_as(tirx::launch_thread_op()))
    return true;
  auto attribute = ReadSourceAttribute(op);
  return attribute && attribute->key == "tl.kernel";
}

inline bool HasNestedKernelAnchor(const SeqStmt &body) {
  return ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
             body,
             [](const ffi::ObjectRef &object)
                 -> ffi::Expected<ffi::WalkResult> {
               // Only executable statement scopes can introduce kernels.
               if (object.as<ExprObj>() || object.as<DictAttrsObj>())
                 return ffi::WalkResult::Skip();
               if (const auto *region = object.as<RegionStmtObj>()) {
                 if (IsAscendBlock(region))
                   return ffi::WalkResult::Skip();
                 if (IsKernelAnchor(region))
                   return ffi::WalkResult::Interrupt(
                       ffi::VisitInterrupt(object));
               }
               return ffi::WalkResult::Advance();
             })
      .has_value();
}

inline bool IsStandaloneSubcoreBoundary(const RegionStmtObj *op) {
  return op->op.same_as(tirx::launch_thread_op()) &&
         op->args[0].as_or_throw<StringImm>()->value == "cthread" &&
         !HasNestedKernelAnchor(op->body);
}

inline bool HasInnerBoundary(const SeqStmt &body) {
  if (body->seq.size() != 1)
    return false;
  const auto *inner = body->seq[0].as<RegionStmtObj>();
  return inner && IsKernelBoundary(inner);
}

struct CollectedKernel {
  const RegionStmtObj *container;
  TilelangKernelContext context;
};

class ContextCollector : public ConstrVisitor {
public:
  using ConstrVisitor::Visit_;
  std::vector<CollectedKernel> kernels;

  Result Visit_(const RegionStmtObj *op) final {
    if (IsAscendBlock(op))
      return std::nullopt;
    auto old_sid = sid_;
    auto old_annotations = annotations_;
    bool old_chain = in_kernel_chain_;
    const RegionStmtObj *old_pending = pending_;

    if (IsKernelAnchor(op)) {
      in_kernel_chain_ = true;
      annotations_.clear();
      if (op->op->name == "tl.launch_kernel") {
        auto schema = op->attrs.GetAttr<ffi::Array<ffi::Any>>("hint_schema")
                          .value_or(ffi::Array<ffi::Any>{"map"});
        auto hints =
            DecodeAnnotations(op->args[4].as_or_throw<tvm::Tuple>(), schema)
                .cast<ffi::Map<ffi::Any, ffi::Any>>();
        for (const auto &[key, value] : hints)
          annotations_.Set(key.cast<ffi::String>(), value);
      }
    }
    // An external subcore launch may itself be the native kernel boundary.
    // A subcore enclosing multiple kernels must leave each kernel separate.
    if (!in_kernel_chain_ && IsStandaloneSubcoreBoundary(op))
      in_kernel_chain_ = true;
    if (auto attribute = ReadSourceAttribute(op);
        in_kernel_chain_ && attribute && attribute->key == "tl.kernel") {
      for (const auto &[key, value] :
           attribute->value.cast<ScheduleUnitMetadata>())
        annotations_.Set(key, value);
    }
    if (op->op.same_as(tirx::launch_thread_op()) &&
        op->args[0].as_or_throw<StringImm>()->value == "cthread")
      sid_ = op->body_params[0].as_or_throw<PrimVar>();
    if (op->op->name == "tl.launch_kernel") {
      if (auto tags = op->attrs.GetAttr<ffi::Array<ffi::String>>("grid_tags"))
        for (size_t i = 0; i < tags->size(); ++i)
          if ((*tags)[i] == "cthread")
            sid_ = op->body_params[i].as_or_throw<PrimVar>();
    }

    if (in_kernel_chain_ && IsKernelBoundary(op) && !HasInnerBoundary(op->body))
      pending_ = op;
    // The shared visitor evaluates operands first and binds body parameters
    // before Visit_(SeqStmt). Capture the context at that exact definition
    // boundary instead of reconstructing range facts with a second analyzer.
    auto result = ConstrVisitor::Visit_(op);
    sid_ = std::move(old_sid);
    annotations_ = std::move(old_annotations);
    in_kernel_chain_ = old_chain;
    pending_ = old_pending;
    return result;
  }

  Result Visit_(const SeqStmtObj *op) final {
    if (pending_ && op == pending_->body.get()) {
      kernels.push_back(
          {pending_,
           {ffi::GetRef<SeqStmt>(op), GetConstrSet(), sid_, annotations_}});
      return std::nullopt;
    }
    return ConstrVisitor::Visit_(op);
  }

private:
  ffi::Optional<PrimVar> sid_;
  ScheduleUnitMetadata annotations_;
  bool in_kernel_chain_{false};
  const RegionStmtObj *pending_{nullptr};
};

class KernelRewriter : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  KernelRewriter(const TilelangKernelRewrite &rewrite,
                 const std::vector<CollectedKernel> &kernels)
      : rewrite_(rewrite), kernels_(kernels) {
    for (size_t i = 0; i < kernels.size(); ++i)
      contexts_.emplace(kernels[i].container, i);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    auto found = contexts_.find(op);
    if (found == contexts_.end())
      return StmtExprMutator::Mutate_(op, mode);
    const auto &kernel = kernels_[found->second];
    SeqStmt body = rewrite_(kernel.context);
    ++rewritten;
    RegionStmt result = ffi::GetRef<RegionStmt>(op);
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }

  size_t rewritten{0};

private:
  const TilelangKernelRewrite &rewrite_;
  const std::vector<CollectedKernel> &kernels_;
  std::unordered_map<const RegionStmtObj *, size_t> contexts_;
};

} // namespace kernel_rewriter_detail

inline tirx::Function
RewriteTilelangKernels(tirx::Function function, const char *pass_name,
                       const TilelangKernelRewrite &rewrite,
                       bool require_kernel = true) {
  TVM_FFI_CHECK((!function->GetAttr(tvm::attr::kDeclaration).value_or(false)),
                ValueError)
      << pass_name << " requires a function body";
  SeqStmt normalized = NormalizeTaskScopes(function->body);
  if (!normalized.same_as(function->body))
    function.CopyOnWrite()->body = std::move(normalized);
  auto collector = ffi::make_object<kernel_rewriter_detail::ContextCollector>();
  collector->Visit(function->body);
  if (collector->kernels.empty()) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (target && (*target)->kind->name == "ascend")
      collector->kernels.push_back(
          {nullptr, {function->body, ConstrSet{}, std::nullopt, {}}});
  }
  for (auto &kernel : collector->kernels)
    kernel.context.parameter_alias_contract =
        CollectAscendParameterAliasContract(function,
                                            kernel.context.kernel_annotations);
  if (require_kernel)
    TVM_FFI_CHECK(!collector->kernels.empty(), ValueError)
        << pass_name << " could not find a TileLang kernel";
  if (collector->kernels.size() == 1 &&
      collector->kernels[0].container == nullptr) {
    auto body = rewrite(collector->kernels[0].context);
    function.CopyOnWrite()->body = std::move(body);
    return function;
  }
  auto rewriter = ffi::make_object<kernel_rewriter_detail::KernelRewriter>(
      rewrite, collector->kernels);
  const SeqStmt &body = function->body;
  auto result = rewriter->Mutate(body);
  TVM_FFI_CHECK_EQ(rewriter->rewritten, collector->kernels.size(), ValueError)
      << "TileLang kernel rewrite did not consume every collected context";
  if (!result.IsUnchanged()) {
    function.CopyOnWrite()->body = std::move(result).ValueUnchecked();
  }
  return function;
}

} // namespace tvm::tl::ascend
