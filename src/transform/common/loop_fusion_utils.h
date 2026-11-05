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
 * \file loop_fusion_utils.h
 * \brief Fuse eligible native parallel loop nests without changing scope
 * identity.
 */
#ifndef TILELANG_LOOP_FUSION_UTILS_H_
#define TILELANG_LOOP_FUSION_UTILS_H_
#include "../../op/native_utils.h"
#include "../../op/utils.h"
#include "../layout_inference/native_policy.h"
#include "ir/stmt_functor.h"

namespace tvm::tl {
using namespace tirx;
using namespace prim;

class FragmentAccessDetector {
public:
  void Collect(const Stmt &stmt) {
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        stmt,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *load = object.as<TensorLoadObj>()) {
            if (auto tensor = load->source.as<TensorVar>())
              has_fragment_ |= IsFragmentBuffer(*tensor);
          } else if (const auto *store = object.as<TensorStoreObj>()) {
            if (auto tensor = store->dest.as<TensorVar>())
              has_fragment_ |= IsFragmentBuffer(*tensor);
          }
          return ffi::WalkResult::Advance();
        });
  }
  bool HasFragmentAccess() const { return has_fragment_; }

private:
  bool has_fragment_ = false;
};

class ParallelLoopFuser : public StmtExprMutatorWithAnalyzer {
public:
  explicit ParallelLoopFuser(sym::AnalyzerObj *analyzer,
                             NativeLayoutInferenceOptions options = {})
      : StmtExprMutatorWithAnalyzer(analyzer), options_(std::move(options)) {}
  static Stmt Fuse(const Stmt &stmt,
                   NativeLayoutInferenceOptions options = {}) {
    auto analyzer = CreateTileLangAnalyzer();
    auto fuser =
        ffi::make_object<ParallelLoopFuser>(analyzer.get(), std::move(options));
    return fuser->Mutate(stmt).ValueOrUnchanged(stmt);
  }

protected:
  virtual bool PreserveParallelLoopNest(const ForObj *loop) const {
    // An operand-backed layout describes the original logical dimensions.
    // Preserve its sole-child parallel nest until the layout consumer lowers
    // it; fusion here cannot reshape the enclosing metadata region.
    for (const auto &region : regions_) {
      if (region->op->name != "tl.attr")
        continue;
      auto fields = DecodeAttributeRegion(region.get());
      if (fields.at("key").cast<ffi::String>() != "tl.loop.annotations" ||
          !fields.at("value").cast<ffi::Map<ffi::String, ffi::Any>>().count(
              attr::kParallelLoopLayout))
        continue;
      const auto *current = SingleStatement(region->body).as<ForObj>();
      while (current && current->kind == ForKind::kParallel) {
        if (current == loop)
          return true;
        current = SingleStatement(current->body).as<ForObj>();
      }
    }
    return options_.preserve_parallel_nest &&
           options_.preserve_parallel_nest(loop, regions_);
  }
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) override {
    if (options_.IsOpaque(op))
      return ffi::Unchanged();
    regions_.push_back(ffi::GetRef<RegionStmt>(op));
    auto result = StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    regions_.pop_back();
    return result;
  }
  ffi::UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) override {
    FragmentAccessDetector detector;
    detector.Collect(op->body);
    if (detector.HasFragmentAccess() || PreserveParallelLoopNest(op))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    std::vector<const ForObj *> chain;
    const ForObj *current = op;
    while (current && current->kind == ForKind::kParallel &&
           IsZero(current->begin) && IsOne(current->step)) {
      chain.push_back(current);
      current = SingleStatement(current->body).as<ForObj>();
    }
    if (chain.size() < 2)
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    for (auto loop : chain) {
      auto extent =
          TryConstInt(GetSyncedAnalyzer()->Simplify(LoopExtent(loop)));
      if (!extent || *extent <= 0 || (*extent & (*extent - 1)) != 0)
        return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    }
    PrimType index_type = op->var.ty();
    PrimExpr total = IntImm(index_type, 1);
    std::string name;
    for (const auto *loop : chain) {
      // A mixed-width nest must keep its original induction-variable types.
      if (loop->var.ty() != index_type)
        return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
      total = total * LoopExtent(loop);
      name += loop->var->name + "_";
    }
    PrimVar fused(name + "fused", index_type);
    ffi::Map<Var, PrimExpr> replacements;
    PrimExpr stride = IntImm(index_type, 1);
    for (size_t i = chain.size(); i > 0; --i) {
      const auto *loop = chain[i - 1];
      replacements.Set(
          loop->var, GetSyncedAnalyzer()->Simplify(
                         floormod(floordiv(fused, stride), LoopExtent(loop))));
      stride = stride * LoopExtent(loop);
    }
    auto body = SubstituteLoopVars(chain.back()->body, replacements);
    For result(fused, IntImm(index_type, 0), total, IntImm(index_type, 1),
               ForKind::kParallel, SeqStmt(body), op->annotations, op->loc);
    // Only static hints live in For.annotations. A caller retaining dynamic
    // hints in a tl.attr/tile call must rebind them before requesting fusion.
    if (auto old = result->annotations.Get(attr::kParallelLoopLayout)) {
      auto layout = old->cast<Fragment>();
      if (layout->InputDim() == chain.size())
        result.CopyOnWrite()->annotations.Set(
            attr::kParallelLoopLayout,
            layout->Reshape({total}, GetSyncedAnalyzer()));
      else
        result.CopyOnWrite()->annotations.erase(attr::kParallelLoopLayout);
    }
    return result;
  }

private:
  NativeLayoutInferenceOptions options_;
  ffi::Array<RegionStmt> regions_;
};
} // namespace tvm::tl
#endif // TILELANG_LOOP_FUSION_UTILS_H_
