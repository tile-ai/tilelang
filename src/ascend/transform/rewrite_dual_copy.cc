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
 * \file rewrite_dual_copy.cc
 * \brief Lower Ascend dual-copy operations using the enclosing cthread sid.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "ascend/op/builtin.h"
#include "ascend/op/utils.h"
#include "op/copy.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;
using ffi::Map;
using ffi::ObjectRef;
using ffi::Optional;
using ffi::String;

namespace {

Call RewriteRegion(const Call &region_call, int split_axis,
                   const PrimExpr &offset, const PrimExpr &split_extent) {
  AccessRegion access = NormalizeToAccessRegion(region_call, kAccessReadWrite);
  BufferRegion buffer_region = access.region;
  Array<Range> ranges = buffer_region->region;
  ICHECK_GE(split_axis, 0);
  ICHECK_LT(split_axis, static_cast<int>(ranges.size()));

  Array<PrimExpr> indices;
  indices.reserve(ranges.size());
  for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
    PrimExpr index = ranges[dimension]->min;
    if (static_cast<int>(dimension) == split_axis)
      index = index + offset;
    indices.push_back(index);
  }

  Array<PrimExpr> arguments = {BufferLoad(buffer_region->buffer, indices),
                               Integer(access.access_mask)};
  for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
    arguments.push_back(static_cast<int>(dimension) == split_axis
                            ? split_extent
                            : ranges[dimension]->extent);
  }
  return Call(DataType::Handle(), region(), arguments, region_call->annotations,
              region_call->span);
}

Map<String, ObjectRef>
StripDualCopyAnnotations(const Map<String, ObjectRef> &annotations,
                         bool strip_double) {
  Map<String, ObjectRef> result;
  for (const auto &[key, value] : annotations) {
    if (key == "dual_dst_ctl" || key == "unit_flag_ctrl" ||
        (strip_double && key == "double"))
      continue;
    result.Set(key, value);
  }
  return result;
}

enum class DualCopyPath {
  kNotDualCopy,
  kHardwareL0CToUB,
  kSoftwareGMToUB,
  kSoftwareUBToGM,
  kSoftwareUBToL1,
  kUnsupported,
};

DualCopyPath ClassifyDualCopy(const Copy &copy) {
  if (!copy->annotations.Get("dual_dst_ctl"))
    return DualCopyPath::kNotDualCopy;

  int dual_control = copy->GetDualDstCtl();
  if (dual_control != 1 && dual_control != 2)
    return DualCopyPath::kUnsupported;

  bool destination_is_double = copy->annotations.Get("double").has_value();
  if (!destination_is_double && IsL0CBuffer(copy->src) &&
      IsSharedBuffer(copy->dst)) {
    return DualCopyPath::kHardwareL0CToUB;
  }
  if (!destination_is_double && IsGlobalBuffer(copy->src) &&
      IsSharedBuffer(copy->dst)) {
    return DualCopyPath::kSoftwareGMToUB;
  }
  if (destination_is_double && IsSharedBuffer(copy->src) &&
      IsGlobalBuffer(copy->dst)) {
    return DualCopyPath::kSoftwareUBToGM;
  }
  if (destination_is_double && IsSharedBuffer(copy->src) &&
      IsL1Buffer(copy->dst)) {
    return DualCopyPath::kSoftwareUBToL1;
  }
  return DualCopyPath::kUnsupported;
}

bool IsSoftwareDualCopy(DualCopyPath path) {
  return path == DualCopyPath::kSoftwareGMToUB ||
         path == DualCopyPath::kSoftwareUBToGM ||
         path == DualCopyPath::kSoftwareUBToL1;
}

int SplitAxis(const Array<Range> &ranges, int dual_control, DualCopyPath path) {
  TVM_FFI_CHECK(!ranges.empty(), ValueError)
      << "dual_copy requires at least one-dimensional regions.";
  if (ranges.size() == 1) {
    TVM_FFI_CHECK(IsSoftwareDualCopy(path), ValueError)
        << "Ascend L0C->UB hardware dual_copy requires at least "
           "two-dimensional regions.";
    return 0;
  }
  return static_cast<int>(ranges.size()) - 2 + (dual_control == 1 ? 0 : 1);
}

void ValidateStaticSplitConstraints(const Copy &copy, DualCopyPath path,
                                    const Call &call) {
  int dual_control = copy->GetDualDstCtl();
  int source_axis = SplitAxis(copy->src_range, dual_control, path);
  int destination_axis = SplitAxis(copy->dst_range, dual_control, path);
  const int64_t *source_extent =
      as_const_int(copy->src_range[source_axis]->extent);
  const int64_t *destination_extent =
      as_const_int(copy->dst_range[destination_axis]->extent);
  if (source_extent == nullptr || destination_extent == nullptr)
    return;

  bool source_is_full = path == DualCopyPath::kHardwareL0CToUB ||
                        path == DualCopyPath::kSoftwareGMToUB;
  int64_t full_extent = source_is_full ? *source_extent : *destination_extent;
  int64_t half_extent = source_is_full ? *destination_extent : *source_extent;
  TVM_FFI_CHECK(full_extent % 2 == 0 && full_extent / 2 == half_extent,
                ValueError)
      << "dual_copy requires an exact 2:1 extent ratio on its split axis, "
         "but got source extent "
      << *source_extent << " and destination extent " << *destination_extent
      << ". Operation: " << call;

  if (path == DualCopyPath::kHardwareL0CToUB && dual_control == 2) {
    TVM_FFI_CHECK(*source_extent % 32 == 0, ValueError)
        << "Ascend L0C->UB N-split dual_copy requires the source N extent "
           "to be a multiple of 32, but got "
        << *source_extent << ". Operation: " << call;
  }
}

bool CallNeedsSoftwareSid(const CallNode *op) {
  static const Op &copy_op = Op::Get("tl.tileop.copy");
  if (!op->op.same_as(copy_op) ||
      !op->annotations.Get("dual_dst_ctl").has_value())
    return false;

  Copy copy(op->args, op->annotations);
  return IsSoftwareDualCopy(ClassifyDualCopy(copy));
}

class UnscopedDualCopyFinder : public StmtExprVisitor {
public:
  static bool NeedsSid(const Stmt &stmt) {
    UnscopedDualCopyFinder finder;
    finder(stmt);
    return finder.needs_sid_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    const auto *iter_var = op->node.as<IterVarNode>();
    bool is_cthread = op->attr_key == tirx::attr::thread_extent &&
                      iter_var != nullptr && iter_var->thread_tag == "cthread";
    if (is_cthread)
      ++cthread_depth_;
    StmtExprVisitor::VisitStmt_(op);
    if (is_cthread)
      --cthread_depth_;
  }

  void VisitExpr_(const CallNode *op) final {
    if (cthread_depth_ == 0 && CallNeedsSoftwareSid(op))
      needs_sid_ = true;
    if (!needs_sid_)
      StmtExprVisitor::VisitExpr_(op);
  }

  int cthread_depth_{0};
  bool needs_sid_{false};
};

class DualCopyRewriter : public StmtMutator {
public:
  explicit DualCopyRewriter(bool allow_root_sid_synthesis)
      : allow_root_sid_synthesis_(allow_root_sid_synthesis) {}

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent)
      return StmtMutator::VisitStmt_(op);
    const auto *iter_var = op->node.as<IterVarNode>();
    if (iter_var == nullptr || iter_var->thread_tag != "cthread")
      return StmtMutator::VisitStmt_(op);

    bool is_unscoped_cthread = vector_block_depth_ == 0;
    if (is_unscoped_cthread)
      ++unscoped_cthread_depth_;
    Optional<Var> outer_sid = sid_;
    Optional<PrimExpr> outer_sid_extent = sid_extent_;
    sid_ = iter_var->var;
    sid_extent_ = op->value;
    Stmt body = VisitStmt(op->body);
    sid_ = std::move(outer_sid);
    sid_extent_ = std::move(outer_sid_extent);
    if (is_unscoped_cthread)
      --unscoped_cthread_depth_;
    if (body.same_as(op->body))
      return ffi::GetRef<Stmt>(op);
    return AttrStmt(op->node, op->attr_key, op->value, std::move(body),
                    op->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    bool is_vector_block = op->name_hint == "VECTOR";
    if (is_vector_block)
      ++vector_block_depth_;
    Stmt result = StmtMutator::VisitStmt_(op);
    if (is_vector_block)
      --vector_block_depth_;
    return result;
  }

  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    Stmt result;
    if (!allow_root_sid_synthesis_ || op->block->name_hint != "tilelang_root" ||
        sid_.has_value() ||
        !UnscopedDualCopyFinder::NeedsSid(ffi::GetRef<Stmt>(op))) {
      result = StmtMutator::VisitStmt_(op);
    } else {
      Var sid("sid", DataType::Int(32));
      PrimExpr extent = Integer(2);
      IterVar sid_thread(Range(0, extent), sid, IterVarType::kThreadIndex,
                         "cthread");
      sid_ = sid;
      sid_extent_ = extent;
      Stmt body = StmtMutator::VisitStmt_(op);
      sid_ = std::nullopt;
      sid_extent_ = std::nullopt;
      result = AttrStmt(sid_thread, tirx::attr::thread_extent, extent,
                        std::move(body), op->span);
    }

    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt rewritten = StmtMutator::VisitStmt_(op);
    const auto *evaluate = rewritten.as<EvaluateNode>();
    ICHECK(evaluate != nullptr);
    const auto *call_node = evaluate->value.as<CallNode>();
    if (call_node == nullptr)
      return rewritten;
    Call call = ffi::GetRef<Call>(call_node);

    static const Op &copy_op = Op::Get("tl.tileop.copy");
    if (!call->op.same_as(copy_op))
      return rewritten;
    Copy copy(call->args, call->annotations);
    DualCopyPath path = ClassifyDualCopy(copy);
    if (path == DualCopyPath::kNotDualCopy) {
      return rewritten;
    }
    TVM_FFI_CHECK(path != DualCopyPath::kUnsupported, ValueError)
        << "RewriteDualCopy supports only L0C->UB and GM->UB dual_copy "
           "operations whose source is the full tile, and UB->GM and UB->L1 "
           "operations whose destination is the full tile; got "
        << copy->src.scope() << " -> " << copy->dst.scope()
        << ". Operation: " << call;

    ValidateStaticSplitConstraints(copy, path, call);
    if (path == DualCopyPath::kHardwareL0CToUB)
      return rewritten;

    Var sid = RequireSid(call);
    int dual_control = copy->GetDualDstCtl();
    int source_axis = SplitAxis(copy->src_range, dual_control, path);
    int destination_axis = SplitAxis(copy->dst_range, dual_control, path);
    if (path == DualCopyPath::kSoftwareGMToUB) {
      PrimExpr split_extent = copy->dst_range[destination_axis]->extent;
      Call source = RewriteRegion(Downcast<Call>(call->args[0]), source_axis,
                                  sid * split_extent, split_extent);
      return Evaluate(Call(call->dtype, copy_op,
                           {source, Downcast<Call>(call->args[1])},
                           StripDualCopyAnnotations(call->annotations, false)));
    }

    PrimExpr split_extent = copy->src_range[source_axis]->extent;
    Call destination =
        RewriteRegion(Downcast<Call>(call->args[1]), destination_axis,
                      sid * split_extent, split_extent);
    return Evaluate(Call(call->dtype, copy_op,
                         {Downcast<Call>(call->args[0]), destination},
                         StripDualCopyAnnotations(call->annotations, true)));
  }

  Var RequireSid(const Call &call) const {
    if (!allow_root_sid_synthesis_ &&
        (vector_block_depth_ == 0 || unscoped_cthread_depth_ != 0)) {
      TVM_FFI_THROW(ValueError)
          << "RewriteDualCopy requires software dual_copy operations in a "
             "manual Ascend pipeline to be inside an explicit T.Vector() "
             "block within T.Kernel(); T.MixedKernel is not supported in "
             "manual mode. Operation: "
          << call;
    }
    TVM_FFI_CHECK(sid_.has_value(), ValueError)
        << "RewriteDualCopy found a dual-copy operation outside a cthread "
           "scope and outside a tilelang_root where one can be created. "
           "Operation: "
        << call;
    const int64_t *extent =
        sid_extent_.has_value() ? as_const_int(sid_extent_.value()) : nullptr;
    TVM_FFI_CHECK(extent != nullptr && *extent == 2, ValueError)
        << "RewriteDualCopy requires a cthread extent of 2, got "
        << (sid_extent_.has_value() ? sid_extent_.value() : PrimExpr())
        << ". Operation: " << call;
    return sid_.value();
  }

  Optional<Var> sid_;
  Optional<PrimExpr> sid_extent_;
  int vector_block_depth_{0};
  int unscoped_cthread_depth_{0};
  bool allow_root_sid_synthesis_;
};

} // namespace

tvm::transform::Pass RewriteDualCopy() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &pass_context) {
    bool enable_auto_schedule =
        pass_context->GetConfig<Bool>(kEnableAutoSchedule, Bool(true)).value();
    func.CopyOnWrite()->body =
        DualCopyRewriter(enable_auto_schedule)(func->body);
    return func;
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.RewriteDualCopy",
                                             {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteDualCopy", RewriteDualCopy);
}

} // namespace tl
} // namespace tvm
