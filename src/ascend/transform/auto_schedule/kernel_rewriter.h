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

#include "native_storage.h"
#include "native_task_ir.h"
#include "transform/common/constr_visitor.h"

#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm::tl::ascend {

struct TilelangKernelContext {
  RegionStmt root;
  ConstrSet outer_ctx;
  ffi::Optional<PrimVar> outer_sid;
  // Raw launch hints and materialized tl.kernel fields describe the same
  // contract. They are read here and remain on their original native carrier.
  ScheduleUnitMetadata kernel_annotations;
  NativeParameterAliasContract parameter_alias_contract;
};

using TilelangKernelRewrite =
    std::function<RegionStmt(const TilelangKernelContext &)>;

namespace kernel_rewriter_detail {

inline bool IsRoot(const RegionStmtObj *op) {
  auto name = NativeBlockName(op);
  return name && name.value() == "tilelang_root";
}

inline bool IsKernelAnchor(const RegionStmtObj *op) {
  return op->op->name == "tl.launch_kernel" ||
         (op->op.same_as(tirx::launch_thread_op()) &&
          op->args[0].as_or_throw<StringImm>()->value == "blockIdx.x");
}

inline bool IsKernelBoundary(const RegionStmtObj *op) {
  if (op->op->name == "tl.launch_kernel" ||
      op->op.same_as(tirx::launch_thread_op()) || IsRoot(op))
    return true;
  auto attribute = ReadTaskAttribute(op);
  return attribute && attribute->key == "tl.kernel";
}

inline bool HasInnerBoundary(const SeqStmt &body) {
  if (body->seq.size() != 1)
    return false;
  const auto *inner = body->seq[0].as<RegionStmtObj>();
  return inner && IsKernelBoundary(inner);
}

struct CollectedKernel {
  const RegionStmtObj *container;
  bool existing_root;
  TilelangKernelContext context;
};

class ContextCollector : public ConstrVisitor {
public:
  using ConstrVisitor::Visit_;
  std::vector<CollectedKernel> kernels;

  Result Visit_(const RegionStmtObj *op) final {
    auto old_sid = sid_;
    auto old_annotations = annotations_;
    bool old_chain = in_kernel_chain_;
    const RegionStmtObj *old_pending = pending_;
    bool old_existing = pending_existing_;

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
    if (auto attribute = ReadTaskAttribute(op);
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

    bool root = IsRoot(op);
    if (root || (in_kernel_chain_ && IsKernelBoundary(op) &&
                 !HasInnerBoundary(op->body))) {
      pending_ = op;
      pending_existing_ = root;
    }
    // The shared visitor evaluates operands first and binds body parameters
    // before Visit_(SeqStmt). Capture the context at that exact definition
    // boundary instead of reconstructing range facts with a second analyzer.
    auto result = ConstrVisitor::Visit_(op);
    sid_ = std::move(old_sid);
    annotations_ = std::move(old_annotations);
    in_kernel_chain_ = old_chain;
    pending_ = old_pending;
    pending_existing_ = old_existing;
    return result;
  }

  Result Visit_(const SeqStmtObj *op) final {
    if (pending_ && op == pending_->body.get()) {
      RegionStmt root =
          pending_existing_
              ? ffi::GetRef<RegionStmt>(pending_)
              : MakeNativeAttribute(
                    nullptr, "tl.sblock",
                    ScheduleUnitMetadata{{"name", ffi::String("tilelang_root")},
                                         {"no_realize", true}},
                    ffi::GetRef<SeqStmt>(op), pending_->loc);
      kernels.push_back({pending_,
                         pending_existing_,
                         {root, GetConstrSet(), sid_, annotations_}});
      return std::nullopt;
    }
    return ConstrVisitor::Visit_(op);
  }

private:
  ffi::Optional<PrimVar> sid_;
  ScheduleUnitMetadata annotations_;
  bool in_kernel_chain_{false};
  const RegionStmtObj *pending_{nullptr};
  bool pending_existing_{false};
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
    RegionStmt root = rewrite_(kernel.context);
    TVM_FFI_CHECK(IsRoot(root.get()), ValueError)
        << "TileLang kernel rewrite must preserve its native root region";
    ++rewritten;
    if (kernel.existing_root)
      return root;
    RegionStmt result = ffi::GetRef<RegionStmt>(op);
    result.CopyOnWrite()->body = SeqStmt(ffi::Array<Stmt>{root}, op->body->loc);
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
  auto collector = ffi::make_object<kernel_rewriter_detail::ContextCollector>();
  collector->Visit(function->body);
  for (auto &kernel : collector->kernels)
    kernel.context.parameter_alias_contract =
        CollectNativeParameterAliasContract(function,
                                            kernel.context.kernel_annotations);
  if (require_kernel)
    TVM_FFI_CHECK(!collector->kernels.empty(), ValueError)
        << pass_name << " could not find a TileLang kernel";
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
