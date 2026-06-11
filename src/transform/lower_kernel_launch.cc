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
 * \file lower_kernel_launch.cc
 * \brief Lower target-neutral T.Kernel loops to target-specific loop forms.
 */

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "common/attr.h"
#include "tir/transforms/ir_utils.h"

#include <utility>

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tirx;

namespace {

enum class KernelLaunchLoweringKind {
  kSerial,
  kThreadBinding,
};

bool IsKernelLaunchAnnotation(const String &key) {
  return key == tilelang_kernel_scope || key == tilelang_kernel_num_blocks ||
         key == tilelang_kernel_num_threads ||
         key == tilelang_kernel_dim_kind || key == tilelang_kernel_dim_axis ||
         key == tilelang_kernel_thread_default;
}

bool HasKernelLaunchAnnotation(const Map<String, Any> &annotations) {
  for (const auto &kv : annotations) {
    if (IsKernelLaunchAnnotation(kv.first)) {
      return true;
    }
  }
  return false;
}

Map<String, Any>
StripKernelLaunchAnnotations(const Map<String, Any> &annotations) {
  Map<String, Any> result;
  for (const auto &kv : annotations) {
    if (!IsKernelLaunchAnnotation(kv.first)) {
      result.Set(kv.first, kv.second);
    }
  }
  return result;
}

Optional<Integer> GetIntegerAnnotation(const Map<String, Any> &annotations,
                                       const char *key) {
  auto it = annotations.find(key);
  if (it == annotations.end()) {
    return Optional<Integer>(std::nullopt);
  }
  if (auto value = (*it).second.try_cast<Integer>()) {
    return value.value();
  }
  LOG(FATAL) << "Expected `" << key << "` to be an Integer, but got "
             << (*it).second.GetTypeKey();
  return Optional<Integer>(std::nullopt);
}

bool HasAnnotation(const Map<String, Any> &annotations, const char *key) {
  return annotations.find(key) != annotations.end();
}

bool IsOne(const PrimExpr &expr, arith::Analyzer *analyzer) {
  return is_const_int(expr, 1) || analyzer->CanProveEqual(expr, 1);
}

String KernelThreadTag(int64_t dim_kind, int64_t axis) {
  static const char *const block_tags[] = {"blockIdx.x", "blockIdx.y",
                                           "blockIdx.z"};
  static const char *const thread_tags[] = {"threadIdx.x", "threadIdx.y",
                                            "threadIdx.z"};
  ICHECK_GE(axis, 0);
  ICHECK_LT(axis, 3);
  if (dim_kind == kTilelangKernelDimBlock) {
    return String(block_tags[axis]);
  }
  ICHECK_EQ(dim_kind, kTilelangKernelDimThread);
  return String(thread_tags[axis]);
}

Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var, String thread_tag,
                      Stmt body) {
  IterVar iter_var(/*dom=*/Range::FromMinExtent(std::move(min), extent),
                   /*var=*/std::move(var),
                   /*iter_type=*/IterVarType::kThreadIndex,
                   /*thread_tag=*/std::move(thread_tag));
  return AttrStmt(/*node=*/std::move(iter_var),
                  /*attr_key=*/tirx::attr::thread_extent,
                  /*value=*/std::move(extent),
                  /*body=*/std::move(body));
}

class KernelLaunchLower : public StmtExprMutator {
public:
  explicit KernelLaunchLower(KernelLaunchLoweringKind kind) : kind_(kind) {}

private:
  using Parent = StmtExprMutator;

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(Parent::VisitStmt_(op));
    auto dim_kind =
        GetIntegerAnnotation(loop->annotations, tilelang_kernel_dim_kind);
    if (!dim_kind.defined()) {
      return loop;
    }

    auto axis =
        GetIntegerAnnotation(loop->annotations, tilelang_kernel_dim_axis);
    ICHECK(axis.defined()) << "T.Kernel dimension loop is missing `"
                           << tilelang_kernel_dim_axis << "`";

    bool is_default_thread =
        dim_kind.value()->value == kTilelangKernelDimThread &&
        HasAnnotation(loop->annotations, tilelang_kernel_thread_default);
    bool is_thread = dim_kind.value()->value == kTilelangKernelDimThread;
    Map<String, Any> annotations =
        StripKernelLaunchAnnotations(loop->annotations);

    if (kind_ == KernelLaunchLoweringKind::kSerial) {
      if (is_thread) {
        if (!is_default_thread && !IsOne(loop->extent, &analyzer_)) {
          LOG(WARNING)
              << "T.Kernel thread extent `" << loop->extent
              << "` is ignored by serial backends such as CPU; "
                 "thread bindings are lowered as a single logical thread.";
        }
        Map<Var, PrimExpr> var_map;
        var_map.Set(loop->loop_var, make_const(loop->loop_var.dtype(), 0));
        return Substitute(loop->body, var_map);
      }
      return For(loop->loop_var, loop->min, loop->extent, ForKind::kSerial,
                 loop->body, /*thread_binding=*/std::nullopt, annotations,
                 loop->step, loop->span);
    }

    ICHECK(annotations.empty())
        << "T.Kernel hardware dimension loops cannot carry annotations after "
           "kernel launch lowering";
    return MakeLaunchThread(
        loop->min, loop->extent, loop->loop_var,
        KernelThreadTag(dim_kind.value()->value, axis.value()->value),
        loop->body);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = Downcast<SBlock>(Parent::VisitStmt_(op));
    if (!HasKernelLaunchAnnotation(block->annotations)) {
      return block;
    }
    Map<String, Any> annotations =
        StripKernelLaunchAnnotations(block->annotations);
    return SBlock(block->iter_vars, block->reads, block->writes,
                  block->name_hint, block->body, block->init,
                  block->alloc_buffers, block->match_buffers, annotations);
  }

  KernelLaunchLoweringKind kind_;
  arith::Analyzer analyzer_;
};

PrimFunc LowerKernelLaunchInFunc(PrimFunc func, KernelLaunchLoweringKind kind) {
  KernelLaunchLower lower(kind);
  Stmt body = lower(func->body);
  if (!body.same_as(func->body)) {
    func.CopyOnWrite()->body = body;
  }
  return func;
}

} // namespace

namespace transform {

tirx::transform::Pass LowerKernelLaunchPass(KernelLaunchLoweringKind kind,
                                            const char *pass_name) {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LowerKernelLaunchInFunc(std::move(f), kind);
  };
  return CreatePrimFuncPass(pass_func, 0, pass_name, {});
}

tirx::transform::Pass LowerKernelLaunchToSerial() {
  return LowerKernelLaunchPass(KernelLaunchLoweringKind::kSerial,
                               "tl.LowerKernelLaunchToSerial");
}

tirx::transform::Pass LowerKernelLaunchToThreadBinding() {
  return LowerKernelLaunchPass(KernelLaunchLoweringKind::kThreadBinding,
                               "tl.LowerKernelLaunchToThreadBinding");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.transform.LowerKernelLaunchToSerial", LowerKernelLaunchToSerial)
      .def("tl.transform.LowerKernelLaunchToThreadBinding",
           LowerKernelLaunchToThreadBinding);
}

} // namespace transform
} // namespace tl
} // namespace tvm
