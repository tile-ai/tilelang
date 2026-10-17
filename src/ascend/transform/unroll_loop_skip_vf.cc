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
#include "support/check.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <utility>

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Renames Var and Buffer definitions in each unrolled body copy so every
// iteration gets a fresh SSA identity.
class UnrolledBodyDefFreshener : public StmtExprMutator {
public:
  PrimExpr VisitExpr_(const VarNode *op) final {
    Var var = GetRef<Var>(op);
    auto it = var_remap_.find(var);
    if (it != var_remap_.end()) {
      return (*it).second;
    }
    return var;
  }

  Buffer VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    Var data;
    if (alloc_data) {
      data = FreshVar(buffer->data);
    } else {
      PrimExpr remapped_data = VisitExpr(buffer->data);
      ICHECK(remapped_data.as<VarNode>())
          << "Buffer data must remain a Var after freshening definitions";
      data = Downcast<Var>(remapped_data);
    }

    auto visit_expr = [this](const PrimExpr &expr) {
      return this->VisitExpr(expr);
    };
    Buffer new_buffer = buffer;
    auto writer = new_buffer.CopyOnWrite();
    writer->data = std::move(data);
    writer->shape = buffer->shape.Map(visit_expr);
    writer->strides = buffer->strides.Map(visit_expr);
    writer->elem_offset = visit_expr(buffer->elem_offset);
    buffer_remap_.Set(buffer, new_buffer);
    return new_buffer;
  }

private:
  Var FreshVar(const Var &var) {
    Var new_var = Var(make_object<VarNode>(*var.get()));
    var_remap_.Set(var, new_var);
    return new_var;
  }

  Map<Var, Var> var_remap_;
};

class ExplicitLoopUnroller : public StmtExprMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    // VF bodies are consumed by their dedicated lowering.  Returning the block
    // itself also preserves explicit unroll loops nested anywhere in the body.
    if (op->name_hint == "SIMD_VF" || op->name_hint == "SIMT_VF") {
      return GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    // Keep the scope-marker check as a fallback for already-normalized IR in
    // which a VF marker is present without its named SBlock wrapper.
    if (op->attr_key == "tl.simdvf_scope" ||
        op->attr_key == "tl.simtvf_scope") {
      return GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Process nested loops first so every explicit request outside a VF block
    // is materialized, including requests nested under a preserved loop.
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *loop = stmt.as<ForNode>();
    ICHECK(loop != nullptr);
    if (!IsExplicitUnroll(loop)) {
      return stmt;
    }
    return Unroll(loop);
  }

private:
  static bool IsExplicitUnroll(const ForNode *op) {
    if (op->kind != ForKind::kUnrolled) {
      return false;
    }
    auto it = op->annotations.find(tirx::attr::pragma_unroll_explicit);
    if (it == op->annotations.end()) {
      return false;
    }
    if (auto explicit_unroll = (*it).second.as<bool>()) {
      return explicit_unroll.value();
    }
    if (auto explicit_unroll = (*it).second.as<Bool>()) {
      return explicit_unroll.value();
    }
    if (auto explicit_unroll = (*it).second.as<IntImm>()) {
      ICHECK(explicit_unroll.value()->value == 0 ||
             explicit_unroll.value()->value == 1)
          << "pragma_unroll_explicit must be a boolean";
      return explicit_unroll.value()->value != 0;
    }
    ICHECK(false) << "pragma_unroll_explicit must be a boolean";
    return false;
  }

  int GetExtent(const ForNode *op) {
    PrimExpr extent = analyzer_.Simplify(op->extent);
    const auto *value = extent.as<IntImmNode>();
    ICHECK(value != nullptr)
        << "T.unroll(..., explicit=True) requires a constant loop extent";
    ICHECK_GE(value->value, 0)
        << "T.unroll(..., explicit=True) requires a non-negative loop extent";
    ICHECK_LE(value->value, std::numeric_limits<int>::max())
        << "T.unroll(..., explicit=True) loop extent is too large";
    return static_cast<int>(value->value);
  }

  Stmt Unroll(const ForNode *op) {
    int extent = GetExtent(op);
    if (extent == 0) {
      return Evaluate(0);
    }

    Map<Var, PrimExpr> substitutions;
    Array<Stmt> unrolled;
    for (int i = 0; i < extent; ++i) {
      PrimExpr iteration;
      if (op->step.defined() && !is_one(op->step.value())) {
        iteration =
            op->min + make_const(op->loop_var.dtype(), i) * op->step.value();
      } else {
        iteration = op->min + make_const(op->loop_var.dtype(), i);
      }
      substitutions.Set(op->loop_var, iteration);
      Stmt body = Substitute(op->body, substitutions);
      body = UnrolledBodyDefFreshener()(std::move(body));
      unrolled.push_back(body);
    }
    return SeqStmt::Flatten(unrolled);
  }

  arith::Analyzer analyzer_;
};

Stmt UnrollLoopSkipVF(Stmt stmt) {
  Stmt result = ExplicitLoopUnroller()(stmt);
  return result.same_as(stmt) ? result : ConvertSSA(result);
}

} // namespace

namespace transform {

using namespace tirx::transform;

Pass UnrollLoopSkipVF() {
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    auto *node = func.CopyOnWrite();
    node->body = tl::UnrollLoopSkipVF(func->body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.UnrollLoopSkipVF", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.UnrollLoopSkipVF", UnrollLoopSkipVF);
}

} // namespace transform

} // namespace tl
} // namespace tvm
