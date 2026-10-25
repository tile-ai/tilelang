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
 * \brief Explicitly unroll requested loops outside Ascend VF blocks.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <limits>

#include "compat/expr.h"
#include "op/builtin.h"

namespace tvm::tl {
namespace {

class TargetedControlDetector : public tirx::StmtExprVisitor {
public:
  bool found{false};

protected:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const ForNode *) final {
    return std::nullopt;
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const WhileNode *) final {
    return std::nullopt;
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const BreakNode *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const ContinueNode *) final {
    found = true;
    return std::nullopt;
  }
  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallNode *node) final {
    if (node->op.same_as(loop_break()))
      found = true;
    return StmtExprVisitor::Visit_(node);
  }
};

class ExplicitLoopUnroller : public tirx::StmtExprMutator {
protected:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto name = node->attrs.GetAttr<ffi::String>("name_hint").value_or("");
    if (name == "SIMD_VF" || name == "SIMT_VF" ||
        node->attrs->dict.count("tl.simdvf_scope") ||
        node->attrs->dict.count("tl.simtvf_scope")) {
      return ffi::Unchanged{};
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto loop = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<For>();
    auto request = loop->annotations.Get("pragma_unroll_explicit");
    if (loop->kind != ForKind::kUnrolled || !request)
      return loop;
    auto explicit_unroll = GetIntAnnotation(*request);
    TVM_FFI_CHECK(explicit_unroll &&
                      (*explicit_unroll == 0 || *explicit_unroll == 1),
                  ValueError)
        << "pragma_unroll_explicit must be a boolean";
    if (!*explicit_unroll)
      return loop;
    auto step = GetConstInt(analyzer_->Simplify(loop->step));
    auto distance = GetConstInt(analyzer_->Simplify(loop->end - loop->begin));
    TVM_FFI_CHECK(step && *step != 0 && distance, ValueError)
        << "T.unroll(..., explicit=True) requires a constant trip count and "
           "nonzero step";
    TVM_FFI_CHECK(SideEffect(loop->begin) == CallEffectKind::kPure &&
                      SideEffect(loop->end) == CallEffectKind::kPure,
                  ValueError)
        << "Explicit unrolling requires loop bounds without mutable reads";
    int64_t count = 0;
    if ((*step > 0 && *distance > 0) || (*step < 0 && *distance < 0)) {
      uint64_t magnitude = *distance > 0 ? uint64_t(*distance)
                                         : uint64_t(0) - uint64_t(*distance);
      uint64_t stride =
          *step > 0 ? uint64_t(*step) : uint64_t(0) - uint64_t(*step);
      uint64_t trips = 1 + (magnitude - 1) / stride;
      TVM_FFI_CHECK(trips <= std::numeric_limits<int>::max(), ValueError)
          << "Explicit unroll trip count is too large";
      count = static_cast<int64_t>(trips);
    }
    auto control = ffi::make_object<TargetedControlDetector>();
    control->Visit(loop->body);
    TVM_FFI_CHECK(!control->found, ValueError)
        << "Explicit unrolling cannot expand break or continue targeting the "
           "loop";
    ffi::Array<Stmt> expanded;
    for (int64_t iteration = 0; iteration < count; ++iteration) {
      PrimExpr index = analyzer_->Simplify(
          loop->begin + MakeConst(loop->var.ty(), iteration) * loop->step);
      Stmt body = Substitute(loop->body,
                             ffi::Map<PrimVar, PrimExpr>{{loop->var, index}});
      auto renewed = tirx::RenewDef(tirx::Function({}, SeqStmt(body)));
      expanded.push_back(RegionStmt(tirx::scope_op(), {}, {}, DictAttrs(),
                                    renewed->body.value(), {}, loop->loc));
    }
    return SeqStmt(expanded, loop->loc);
  }

private:
  sym::Analyzer analyzer_;
};

} // namespace

namespace transform {
tvm::transform::Pass UnrollLoopSkipVF() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto rewriter = ffi::make_object<ExplicitLoopUnroller>();
    SeqStmt original = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(original).ValueOrUnchanged(original);
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
