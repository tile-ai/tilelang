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
 * \file common.h
 * \brief Common utilities for TL transforms
 */

#ifndef TILELANG_TRANSFORM_COMMON_LOOP_FUSION_UTILS_H_
#define TILELANG_TRANSFORM_COMMON_LOOP_FUSION_UTILS_H_

#include "compat/visit.h"
#include "op/parallel.h"
#include "op/reducer.h"
#include "op/utils.h"

#include <tvm/tirx/stmt_functor_with_analyzer.h>

#include <limits>
#include <vector>

namespace tvm::tl {

class FragmentAccessDetector {
public:
  void Collect(const Stmt &statement) {
    PostOrderVisit(statement, [this](const ffi::ObjectRef &object) {
      if (const auto *load = object.as<TensorLoadNode>()) {
        if (auto tensor = load->source.as<tirx::TensorVar>()) {
          has_fragment_access_ |= IsFragmentBuffer(tensor.value()) ||
                                  IsReducerV2Buffer(tensor.value());
        }
      } else if (const auto *store = object.as<TensorStoreNode>()) {
        if (auto tensor = store->dest.as<tirx::TensorVar>()) {
          has_fragment_access_ |= IsFragmentBuffer(tensor.value()) ||
                                  IsReducerV2Buffer(tensor.value());
        }
      }
    });
  }

  bool HasFragmentAccess() const { return has_fragment_access_; }

private:
  bool has_fragment_access_{false};
};

class ParallelLoopFuser : public tirx::StmtExprMutatorWithAnalyzer {
public:
  explicit ParallelLoopFuser(sym::Analyzer analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer),
        analyzer_context_(std::move(analyzer)) {}

  static Stmt Fuse(const Stmt &statement) {
    return ffi::make_object<ParallelLoopFuser>(sym::Analyzer())
        ->Mutate(statement, ffi::InplaceMode::kDisallow)
        .ValueOrUnchanged(statement);
  }

protected:
  virtual bool PreserveParallelLoopNest(const ForNode *) const { return false; }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    FragmentAccessDetector detector;
    detector.Collect(node->body);
    if (detector.HasFragmentAccess() || PreserveParallelLoopNest(node)) {
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    }
    std::vector<const ForNode *> loops;
    std::vector<int64_t> extents;
    int64_t total = 1;
    int bits = 32;
    const ForNode *current = node;
    while (current && current->kind == ForKind::kParallel &&
           IsZero(current->begin) && IsOne(current->step)) {
      if (current != node && !current->annotations.empty()) {
        break;
      }
      auto extent =
          GetConstInt(analyzer_->Simplify(current->end - current->begin));
      if (!extent || *extent <= 0 || (*extent & (*extent - 1)) != 0 ||
          total > std::numeric_limits<int64_t>::max() / *extent) {
        return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
      }
      total *= *extent;
      bits = std::max(bits, current->var.ty().bits());
      loops.push_back(current);
      extents.push_back(*extent);
      current = current->body->seq.size() == 1
                    ? current->body->seq[0].as<ForNode>()
                    : nullptr;
    }
    if (loops.size() < 2) {
      return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
    }
    if (total > std::numeric_limits<int32_t>::max()) {
      bits = 64;
    }
    if (auto layout = node->annotations.Get(attr::kParallelLoopLayout)) {
      if (layout.value().as_or_throw<Fragment>()->InputDim() != loops.size()) {
        return ffi::GetRef<Stmt>(node);
      }
    }

    PrimType type = PrimType::Int(bits);
    PrimVar fused_var("parallel_fused", type);
    ffi::Map<PrimVar, PrimExpr> replacements;
    int64_t stride = 1;
    for (size_t axis = loops.size(); axis-- > 0;) {
      PrimExpr index = FloorMod(FloorDiv(fused_var, MakeConst(type, stride)),
                                MakeConst(type, extents[axis]));
      replacements.Set(loops[axis]->var,
                       Cast(loops[axis]->var.ty(), analyzer_->Simplify(index)));
      stride *= extents[axis];
    }
    PrimExpr extent = MakeConst(type, total);
    Stmt body = Substitute(loops.back()->body, replacements);
    For result(fused_var, MakeConst(type, 0), extent, MakeConst(type, 1),
               ForKind::kParallel, SeqStmt(body), node->annotations, node->loc);
    if (auto annotation = node->annotations.Get(attr::kParallelLoopLayout)) {
      auto layout = annotation.value().as_or_throw<Fragment>();
      result.CopyOnWrite()->annotations.Set(
          attr::kParallelLoopLayout,
          layout->Reshape({extent}, analyzer_context_));
    }
    return result;
  }

private:
  sym::Analyzer analyzer_context_;
};

} // namespace tvm::tl

#endif
