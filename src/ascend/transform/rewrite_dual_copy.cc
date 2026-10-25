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

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/native_visitor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "op/copy.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::Optional;
using ffi::String;

namespace {

BlockAnnotations CallAnnotations(const CallNode *call) {
  return call->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>()->dict;
}

bool IsCthread(const RegionStmtNode *node) {
  return (node->op.same_as(tirx::launch_thread_op()) ||
          node->op.same_as(Op::Get("tl.launch_grid"))) &&
         node->args.size() == 2 &&
         node->args[0].as_or_throw<StringImm>()->value == "cthread";
}

Call RewriteRegion(const Expr &region_call, int split_axis,
                   const PrimExpr &offset, const PrimExpr &split_extent) {
  AccessRegion access = NormalizeToAccessRegion(region_call, kAccessReadWrite);
  TensorRegion buffer_region = access.region;
  Array<Range> ranges = buffer_region->region;
  ICHECK_GE(split_axis, 0);
  ICHECK_LT(split_axis, static_cast<int>(ranges.size()));

  Array<PrimExpr> indices;
  indices.reserve(ranges.size());
  for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
    PrimExpr index = ranges[dimension]->begin;
    if (static_cast<int>(dimension) == split_axis)
      index = index + offset;
    indices.push_back(index);
  }

  Array<Expr> arguments = {TensorLoad(buffer_region->source, indices),
                           IntImm::Int32(access.access_mask)};
  for (size_t dimension = 0; dimension < ranges.size(); ++dimension) {
    arguments.push_back(static_cast<int>(dimension) == split_axis
                            ? split_extent
                            : ranges[dimension]->extent);
  }
  auto call = region_call.as<CallNode>();
  return Call(std::nullopt, region(), arguments,
              call ? call->attrs : std::nullopt, {}, region_call->loc);
}

Map<String, Any> StripDualCopyAnnotations(const Map<String, Any> &annotations,
                                          bool strip_double) {
  Map<String, Any> result;
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

DualCopyPath ClassifyDualCopy(const AscendCopy &copy) {
  if (!copy->annotations.Get("dual_dst_ctl"))
    return DualCopyPath::kNotDualCopy;

  int dual_control = copy->dual_dst_ctl;
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

void ValidateStaticSplitConstraints(const AscendCopy &copy, DualCopyPath path,
                                    const Call &call) {
  int dual_control = copy->dual_dst_ctl;
  int source_axis = SplitAxis(copy->src_range, dual_control, path);
  int destination_axis = SplitAxis(copy->dst_range, dual_control, path);
  auto source_extent = GetConstInt(copy->src_range[source_axis]->extent);
  auto destination_extent =
      GetConstInt(copy->dst_range[destination_axis]->extent);
  if (!source_extent || !destination_extent)
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
  if (!IsAscendCopyCall(op) ||
      !CallAnnotations(op).Get("dual_dst_ctl").has_value())
    return false;

  AscendCopy copy(op->args, CallAnnotations(op));
  return IsSoftwareDualCopy(ClassifyDualCopy(copy));
}

class UnscopedDualCopyFinder : public ascend::AnalysisVisitor {
public:
  static bool NeedsSid(const Stmt &stmt) {
    auto finder = ffi::make_object<UnscopedDualCopyFinder>();
    finder->Visit(stmt);
    return finder->needs_sid_;
  }

private:
  void VisitStmt_(const RegionStmtNode *op) final {
    bool is_cthread = IsCthread(op);
    if (is_cthread)
      ++cthread_depth_;
    ascend::AnalysisVisitor::VisitStmt_(op);
    if (is_cthread)
      --cthread_depth_;
  }

  void VisitExpr_(const CallNode *op) final {
    if (cthread_depth_ == 0 && CallNeedsSoftwareSid(op))
      needs_sid_ = true;
    if (!needs_sid_)
      ascend::AnalysisVisitor::VisitExpr_(op);
  }

  int cthread_depth_{0};
  bool needs_sid_{false};
};

class DualCopyRewriter : public ascend::NativeMutator {
public:
  explicit DualCopyRewriter(bool allow_root_sid_synthesis)
      : allow_root_sid_synthesis_(allow_root_sid_synthesis) {}

private:
  Stmt VisitStmt_(const RegionStmtNode *op) final {
    auto previous_sid = sid_;
    auto previous_extent = sid_extent_;
    auto name = ascend::KernelRegionName(op);
    bool vector_region = name == "VECTOR";
    bool launch = IsCthread(op);
    bool unscoped = launch && vector_block_depth_ == 0;
    bool synthesize = allow_root_sid_synthesis_ && name == "tilelang_root" &&
                      !sid_ &&
                      UnscopedDualCopyFinder::NeedsSid(ffi::GetRef<Stmt>(op));
    if (vector_region)
      ++vector_block_depth_;
    if (unscoped)
      ++unscoped_cthread_depth_;
    if (launch) {
      sid_ = op->body_params[0].as_or_throw<PrimVar>();
      sid_extent_ = op->args[1].as_or_throw<PrimExpr>();
    } else if (synthesize) {
      sid_ = PrimVar("sid", PrimType::Int(32));
      sid_extent_ = IntImm::Int32(2);
    }
    Stmt result = ascend::NativeMutator::VisitStmt_(op);
    if (synthesize) {
      result =
          RegionStmt(Op::Get("tl.launch_grid"),
                     {StringImm("cthread"), sid_extent_.value()},
                     {sid_.value()}, DictAttrs(), SeqStmt(result), {}, op->loc);
    }
    sid_ = previous_sid;
    sid_extent_ = previous_extent;
    if (unscoped)
      --unscoped_cthread_depth_;
    if (vector_region)
      --vector_block_depth_;
    return result;
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    Stmt rewritten = ascend::NativeMutator::VisitStmt_(op);
    const auto *evaluate = rewritten.as<EvaluateNode>();
    ICHECK(evaluate != nullptr);
    const auto *call_node = evaluate->value.as<CallNode>();
    if (call_node == nullptr)
      return rewritten;
    Call call = ffi::GetRef<Call>(call_node);

    if (!IsAscendCopyCall(call_node))
      return rewritten;
    AscendCopy copy(call->args, CallAnnotations(call.get()));
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

    PrimVar sid = RequireSid(call);
    int dual_control = copy->dual_dst_ctl;
    int source_axis = SplitAxis(copy->src_range, dual_control, path);
    int destination_axis = SplitAxis(copy->dst_range, dual_control, path);
    if (path == DualCopyPath::kSoftwareGMToUB) {
      PrimExpr split_extent = copy->dst_range[destination_axis]->extent;
      Call source = RewriteRegion(call->args[0], source_axis,
                                  sid * split_extent, split_extent);
      return Evaluate(Call(call->ty, call->op, {source, call->args[1]},
                           DictAttrs(StripDualCopyAnnotations(
                               CallAnnotations(call.get()), false))));
    }

    PrimExpr split_extent = copy->src_range[source_axis]->extent;
    Call destination = RewriteRegion(call->args[1], destination_axis,
                                     sid * split_extent, split_extent);
    return Evaluate(Call(call->ty, call->op, {call->args[0], destination},
                         DictAttrs(StripDualCopyAnnotations(
                             CallAnnotations(call.get()), true))));
  }

  PrimVar RequireSid(const Call &call) const {
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
    auto extent = sid_extent_ ? GetConstInt(sid_extent_.value()) : std::nullopt;
    TVM_FFI_CHECK(extent && *extent == 2, ValueError)
        << "RewriteDualCopy requires a cthread extent of 2, got "
        << sid_extent_.value_or(IntImm::Int32(0)) << ". Operation: " << call;
    return sid_.value();
  }

  Optional<PrimVar> sid_;
  Optional<PrimExpr> sid_extent_;
  int vector_block_depth_{0};
  int unscoped_cthread_depth_{0};
  bool allow_root_sid_synthesis_;
};

} // namespace

tvm::transform::Pass RewriteDualCopy() {
  auto pass_func = [](tirx::Function func, const IRModule &,
                      const tvm::transform::PassContext &pass_context) {
    bool enable_auto_schedule =
        pass_context->GetConfig<bool>(kEnableAutoSchedule).value_or(true);
    if (!func->body)
      return func;
    auto rewriter = ffi::make_object<DualCopyRewriter>(enable_auto_schedule);
    func.CopyOnWrite()->body = rewriter->VisitStmt(func->body.value());
    return func;
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0,
                                             "tl.RewriteDualCopy");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteDualCopy", RewriteDualCopy);
}

} // namespace tl
} // namespace tvm
