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
 * \file unroll_loop_skip_vf.cc
 * \brief Expand explicit native loops while preserving Ascend VF bodies.
 */
#include "ascend/ir/block.h"
#include "ir/annotations.h"
#include "ir/loop_annotations.h"
#include "ir/loop_math.h"
#include "ir/stmt_functor.h"
#include "op/builtin.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class TargetedLoopControl : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  bool found{false};
  Optional<VisitInterrupt> Visit_(const BreakObj *) final {
    found = true;
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const ContinueObj *) final {
    found = true;
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const ForObj *) final { return std::nullopt; }
  Optional<VisitInterrupt> Visit_(const WhileObj *) final {
    return std::nullopt;
  }
  Optional<VisitInterrupt> Visit_(const CallObj *call) final {
    if (call->op.same_as(tl::loop_break()))
      found = true;
    return StmtExprVisitor::Visit_(call);
  }
};

class ExplicitLoopUnroller : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (ascend::GetVFKind(op) != ascend::VFKind::kNone)
      return Unchanged();
    return LayoutMapMutator::Mutate_(op, mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    auto original = GetRef<Stmt>(op);
    auto result =
        LayoutMapMutator::Mutate_(op, mode).ValueOrUnchanged(original);
    const auto *loop = result.as<ForObj>();
    auto annotations = ReadLoopAnnotations(loop);
    auto explicit_unroll = annotations.Get("pragma_unroll_explicit");
    if (loop->kind != ForKind::kUnrolled || !explicit_unroll ||
        !AnnotationAsBool(*explicit_unroll))
      return result;
    int count = ConstantUnrollTripCount(loop, analyzer_);
    TVM_FFI_CHECK_GE(count, 0, ValueError)
        << "T.unroll(..., explicit=True) requires a constant loop extent "
           "and a non-wrapping native induction variable";
    auto control = ffi::make_object<TargetedLoopControl>();
    control->Visit(loop->body);
    TVM_FFI_CHECK(!control->found, ValueError)
        << "Cannot explicitly expand a loop containing break or continue "
           "targeting that loop; use explicit=False";
    Array<Stmt> statements;
    for (int index = 0; index < count; ++index) {
      // Native RenewDef renews local definitions, including tensor types,
      // region binders and dynamic metadata uses. Free captures keep identity.
      auto copy = tl::RenewDef(tirx::Function({}, loop->body));
      auto substitution = ffi::make_object<LayoutMapMutator>();
      substitution->VarRemapSet(
          loop->var,
          analyzer_->Simplify(UnrolledLoopValue(loop, index, analyzer_)));
      auto body = substitution->Mutate(copy->body).ValueOrUnchanged(copy->body);
      for (const Stmt &statement : body->seq)
        statements.push_back(statement);
    }
    return SeqStmt(statements, loop->loc);
  }

private:
  sym::Analyzer analyzer_;
};
} // namespace

namespace transform {
tvm::transform::Pass UnrollLoopSkipVF() {
  auto pass = [](tirx::Function function, const IRModule &,
                 const tvm::transform::PassContext &) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto rewriter = ffi::make_object<ExplicitLoopUnroller>();
      function.CopyOnWrite()->body =
          rewriter->Mutate(function->body).ValueOrUnchanged(function->body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.UnrollLoopSkipVF");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.UnrollLoopSkipVF",
                                   UnrollLoopSkipVF);
}
} // namespace transform
} // namespace tvm::tl
