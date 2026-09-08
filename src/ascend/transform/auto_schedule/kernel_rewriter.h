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

#include <functional>
#include <utility>
#include <vector>

#include <tvm/ffi/optional.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "support/check.h"
#include "transform/common/constr_visitor.h"

namespace tvm {
namespace tl {

struct TilelangKernelContext {
  tirx::SBlock root;
  // Constraints established by statements that enclose this root.
  ConstrSet outer_ctx;
  ffi::Optional<tirx::Var> outer_sid;
};

using TilelangKernelRewrite =
    std::function<tirx::SBlock(const TilelangKernelContext &)>;

// Walk the complete PrimFunc and rewrite every tilelang_root independently.
// This preserves arbitrary outer wrappers and naturally supports both single-
// and multi-kernel functions. When require_kernel is true, the pass fails if
// no tilelang_root is found.
tirx::PrimFunc RewriteTilelangKernels(tirx::PrimFunc func,
                                      const char *pass_name,
                                      const TilelangKernelRewrite &rewrite,
                                      bool require_kernel = true);

// Inline implementation.

using namespace tirx;
using ffi::GetRef;
using ffi::Optional;

namespace kernel_rewriter_detail {

// Reuse ConstrVisitor's scope rules instead of duplicating them in the
// mutating traversal below. Contexts are consumed in the same traversal order.
class TilelangKernelContextCollector : public ConstrVisitor {
public:
  static std::vector<TilelangKernelContext> Collect(const Stmt &stmt) {
    TilelangKernelContextCollector collector;
    collector(stmt);
    return std::move(collector.contexts_);
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    Optional<Var> outer_sid = sid_;
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iter_var = op->node.as<IterVarNode>()) {
        if (iter_var->thread_tag == "cthread")
          sid_ = iter_var->var;
      }
    }
    ConstrVisitor::VisitStmt_(op);
    sid_ = std::move(outer_sid);
  }

  void VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root") {
      ConstrVisitor::VisitStmt_(op);
      return;
    }
    contexts_.push_back({GetRef<SBlock>(op), GetConstrSet(), sid_});
  }

  Optional<Var> sid_;
  std::vector<TilelangKernelContext> contexts_;
};

class TilelangKernelRewriter : public StmtMutator {
public:
  TilelangKernelRewriter(
      const TilelangKernelRewrite &rewrite,
      const std::vector<TilelangKernelContext> &kernel_contexts)
      : rewrite_(rewrite), kernel_contexts_(kernel_contexts) {}

  Stmt Rewrite(const Stmt &stmt) { return VisitStmt(stmt); }
  int NumRewrittenKernels() const { return num_rewritten_kernels_; }

private:
  Stmt VisitStmt_(const SBlockNode *op) final {
    if (op->name_hint != "tilelang_root")
      return StmtMutator::VisitStmt_(op);
    ICHECK_LT(static_cast<size_t>(num_rewritten_kernels_),
              kernel_contexts_.size())
        << "TileLang kernel rewrite encountered an uncollected tilelang_root";
    const TilelangKernelContext &context =
        kernel_contexts_[num_rewritten_kernels_];
    ICHECK(context.root.same_as(GetRef<SBlock>(op)))
        << "TileLang kernel context traversal order mismatch";
    ++num_rewritten_kernels_;
    SBlock rewritten = rewrite_(context);
    ICHECK(rewritten.defined()) << "TileLang kernel rewrite returned null";
    return rewritten;
  }

  const TilelangKernelRewrite &rewrite_;
  const std::vector<TilelangKernelContext> &kernel_contexts_;
  int num_rewritten_kernels_{0};
};

} // namespace kernel_rewriter_detail

inline PrimFunc RewriteTilelangKernels(PrimFunc func, const char *pass_name,
                                       const TilelangKernelRewrite &rewrite,
                                       bool require_kernel) {
  std::vector<TilelangKernelContext> kernel_contexts =
      kernel_rewriter_detail::TilelangKernelContextCollector::Collect(
          func->body);
  kernel_rewriter_detail::TilelangKernelRewriter rewriter(rewrite,
                                                          kernel_contexts);
  Stmt body = rewriter.Rewrite(func->body);
  ICHECK_EQ(static_cast<size_t>(rewriter.NumRewrittenKernels()),
            kernel_contexts.size())
      << "TileLang kernel rewrite did not consume every collected context";
  if (require_kernel) {
    ICHECK_GT(rewriter.NumRewrittenKernels(), 0)
        << pass_name << " could not find a tilelang_root";
  }
  func.CopyOnWrite()->body = std::move(body);
  return func;
}

} // namespace tl
} // namespace tvm
