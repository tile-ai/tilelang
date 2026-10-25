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
 * \brief Shared traversal for rewriting every TileLang kernel in a PrimFunc.
 */

#pragma once

#include <tvm/tirx/function.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <utility>
#include <vector>

#include "transform/common/constr_visitor.h"

namespace tvm::tl::ascend {

inline ffi::String KernelRegionName(const RegionStmtNode *region) {
  return region->attrs.GetAttr<ffi::String>("name_hint").value_or("");
}

struct TilelangKernelContext {
  RegionStmt root;
  ConstrSet outer_ctx;
  ffi::Optional<PrimVar> outer_sid;
};

using TilelangKernelRewrite =
    std::function<RegionStmt(const TilelangKernelContext &)>;

namespace kernel_rewriter_detail {

class TilelangKernelContextCollector : public ConstrVisitor {
public:
  std::vector<TilelangKernelContext> contexts;

protected:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    if (KernelRegionName(node) == "tilelang_root") {
      contexts.push_back({ffi::GetRef<RegionStmt>(node), GetConstrSet(), sid_});
      return std::nullopt;
    }
    auto previous_sid = sid_;
    bool launch = node->op.same_as(tirx::launch_thread_op()) ||
                  node->op.same_as(Op::Get("tl.launch_grid"));
    if (launch && node->args.size() == 2 &&
        node->args[0].as_or_throw<StringImm>()->value == "cthread") {
      sid_ = node->body_params[0].as_or_throw<PrimVar>();
    }
    ffi::Optional<ffi::VisitInterrupt> result;
    if (launch && node->op.same_as(Op::Get("tl.launch_grid"))) {
      auto variable = node->body_params[0].as_or_throw<PrimVar>();
      auto extent = node->args[1].as_or_throw<PrimExpr>();
      auto guard = MakeGuard(
          variable, Range::FromMinExtent(MakeConst(extent.ty(), 0), extent));
      result = Visit(node->body);
    } else {
      result = ConstrVisitor::Visit_(node);
    }
    sid_ = previous_sid;
    return result;
  }

private:
  ffi::Optional<PrimVar> sid_;
};

class TilelangKernelRewriter : public tirx::StmtExprMutator {
public:
  TilelangKernelRewriter(TilelangKernelRewrite rewrite,
                         std::vector<TilelangKernelContext> contexts)
      : rewrite_(std::move(rewrite)), contexts_(std::move(contexts)) {}
  size_t count{0};

protected:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    if (KernelRegionName(node) != "tilelang_root") {
      return StmtExprMutator::Mutate_(node, mode);
    }
    TVM_FFI_ICHECK_LT(count, contexts_.size());
    const auto &context = contexts_[count++];
    TVM_FFI_ICHECK(context.root.same_as(ffi::GetRef<RegionStmt>(node)));
    return rewrite_(context);
  }

private:
  TilelangKernelRewrite rewrite_;
  std::vector<TilelangKernelContext> contexts_;
};

} // namespace kernel_rewriter_detail

inline tirx::Function
RewriteTilelangKernels(tirx::Function function, const char *pass_name,
                       const TilelangKernelRewrite &rewrite,
                       bool require_kernel = true) {
  auto collector = ffi::make_object<
      kernel_rewriter_detail::TilelangKernelContextCollector>();
  if (function->body)
    collector->Visit(function->body.value());
  if (require_kernel) {
    TVM_FFI_CHECK(!collector->contexts.empty(), ValueError)
        << pass_name << " could not find a tilelang_root";
  }
  if (!function->body || collector->contexts.empty())
    return function;
  auto rewriter =
      ffi::make_object<kernel_rewriter_detail::TilelangKernelRewriter>(
          rewrite, collector->contexts);
  SeqStmt original = function->body.value();
  function.CopyOnWrite()->body =
      rewriter->Mutate(original).ValueOrUnchanged(original);
  TVM_FFI_ICHECK_EQ(rewriter->count, collector->contexts.size());
  return function;
}

} // namespace tvm::tl::ascend
