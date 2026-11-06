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
 * \file ascend/transform/rewrite_dual_copy.cc
 * \brief Split native software dual copies using the lexical AIV subcore.
 */

#include "ascend/ir/block.h"
#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ir/annotations.h"
#include <tvm/tirx/op/region.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <utility>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;
using namespace prim;

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
  if (copy->dual_dst_ctl != 1 && copy->dual_dst_ctl != 2)
    return DualCopyPath::kUnsupported;
  bool double_destination = copy->annotations.Get("double").has_value();
  if (!double_destination && IsL0CBuffer(copy->src) &&
      IsSharedBuffer(copy->dst))
    return DualCopyPath::kHardwareL0CToUB;
  if (!double_destination && IsGlobalBuffer(copy->src) &&
      IsSharedBuffer(copy->dst))
    return DualCopyPath::kSoftwareGMToUB;
  if (double_destination && IsSharedBuffer(copy->src) &&
      IsGlobalBuffer(copy->dst))
    return DualCopyPath::kSoftwareUBToGM;
  if (double_destination && IsSharedBuffer(copy->src) && IsL1Buffer(copy->dst))
    return DualCopyPath::kSoftwareUBToL1;
  return DualCopyPath::kUnsupported;
}

bool IsSoftwareDualCopy(DualCopyPath path) {
  return path == DualCopyPath::kSoftwareGMToUB ||
         path == DualCopyPath::kSoftwareUBToGM ||
         path == DualCopyPath::kSoftwareUBToL1;
}

int SplitAxis(const Array<Range> &ranges, int control, DualCopyPath path) {
  TVM_FFI_CHECK(!ranges.empty(), ValueError)
      << "dual_copy requires at least one-dimensional regions.";
  if (ranges.size() == 1) {
    TVM_FFI_CHECK(IsSoftwareDualCopy(path), ValueError)
        << "Ascend L0C->UB hardware dual_copy requires at least "
           "two-dimensional regions.";
    return 0;
  }
  return static_cast<int>(ranges.size()) - 2 + (control == 1 ? 0 : 1);
}

std::optional<int64_t> Constant(const PrimExpr &value) {
  if (const auto *imm = value.as<IntImmObj>())
    return static_cast<int64_t>(imm->value);
  return std::nullopt;
}

void ValidateStaticSplitConstraints(const AscendCopy &copy, DualCopyPath path,
                                    const Call &call) {
  int source_axis = SplitAxis(copy->src_range, copy->dual_dst_ctl, path);
  int destination_axis = SplitAxis(copy->dst_range, copy->dual_dst_ctl, path);
  auto source = Constant(copy->src_range[source_axis]->extent);
  auto destination = Constant(copy->dst_range[destination_axis]->extent);
  if (!source || !destination)
    return;
  bool source_is_full = path == DualCopyPath::kHardwareL0CToUB ||
                        path == DualCopyPath::kSoftwareGMToUB;
  int64_t full = source_is_full ? *source : *destination;
  int64_t half = source_is_full ? *destination : *source;
  TVM_FFI_CHECK(full % 2 == 0 && full / 2 == half, ValueError)
      << "dual_copy requires an exact 2:1 extent ratio on its split axis, "
         "but got source extent "
      << *source << " and destination extent " << *destination
      << ". Operation: " << call;
  if (path == DualCopyPath::kHardwareL0CToUB && copy->dual_dst_ctl == 2)
    TVM_FFI_CHECK(*source % 32 == 0, ValueError)
        << "Ascend L0C->UB N-split dual_copy requires the source N extent "
           "to be a multiple of 32, but got "
        << *source << ". Operation: " << call;
}

bool BlockIs(const RegionStmtObj *region, const char *name) {
  if (ascend::IsAscendBlock(region))
    return region->attrs.GetAttr<ffi::String>("name").value() == name;
  return false;
}

bool IsKernel(const RegionStmtObj *region) {
  if (region->op->name == "tl.launch_kernel")
    return true;
  // MaterializeKernelLaunch preserves a kernel's annotation owner even
  // when it has no hints. Its subcore belongs inside that owner, just as
  // it did inside the frontend launch region. The finder must also stop
  // here so the enclosing grid does not acquire the same subcore twice.
  if (region->op->name == "tl.attr" &&
      ReadAnnotations(region).at("key").cast<String>() == "tl.kernel")
    return true;
  return region->op.same_as(launch_thread_op()) &&
         region->args[0].as_or_throw<StringImm>()->value == "blockIdx.x";
}

struct SubcoreScope {
  PrimVar sid;
  PrimExpr extent;
  PrimExpr begin;
  PrimExpr step;
};

// Accept both frontend launch regions and their materialized native form.
// The variable remains the actual region definition, never a name heuristic.
std::optional<SubcoreScope> ReadSubcore(const RegionStmtObj *region) {
  if (region->op.same_as(launch_thread_op()) &&
      region->args[0].as_or_throw<StringImm>()->value == "cthread") {
    auto sid = region->body_params[0].as_or_throw<PrimVar>();
    return SubcoreScope{sid, region->args[1].as_or_throw<PrimExpr>(),
                        IntImm(sid.ty(), 0), IntImm(sid.ty(), 1)};
  }
  if (region->op->name != "tl.launch_kernel")
    return std::nullopt;
  auto tags = region->attrs.GetAttr<Array<String>>("grid_tags");
  if (!tags)
    return std::nullopt;
  std::optional<SubcoreScope> result;
  for (size_t i = 0; i < tags->size(); ++i) {
    if ((*tags)[i] != "cthread")
      continue;
    TVM_FFI_CHECK(!result, ValueError)
        << "RewriteDualCopy requires at most one cthread kernel axis";
    result = SubcoreScope{region->body_params[i].as_or_throw<PrimVar>(),
                          region->args[1]
                              .as_or_throw<tvm::Tuple>()
                              ->fields[i]
                              .as_or_throw<PrimExpr>(),
                          region->args[0]
                              .as_or_throw<tvm::Tuple>()
                              ->fields[i]
                              .as_or_throw<PrimExpr>(),
                          region->args[2]
                              .as_or_throw<tvm::Tuple>()
                              ->fields[i]
                              .as_or_throw<PrimExpr>()};
  }
  return result;
}

class UnscopedDualCopyFinder : public StmtExprVisitor {
public:
  static bool NeedsSid(const Stmt &stmt) {
    auto finder = make_object<UnscopedDualCopyFinder>();
    finder->Visit(stmt);
    return finder->needs_sid_;
  }
  using StmtExprVisitor::Visit_;

  Optional<VisitInterrupt> Visit_(const RegionStmtObj *op) final {
    // A nested kernel or explicit subcore owns its own region. Its copy must
    // not force an unrelated parent kernel to acquire a subcore scope.
    if (IsKernel(op) || ReadSubcore(op))
      return std::nullopt;
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (IsAscendCopyCall(op)) {
      auto decoded = DecodeTileOpArguments(op);
      needs_sid_ |= IsSoftwareDualCopy(
          ClassifyDualCopy(AscendCopy(decoded.operands, decoded.annotations)));
    }
    return needs_sid_ ? std::nullopt : StmtExprVisitor::Visit_(op);
  }

private:
  bool needs_sid_{false};
};

TensorRegion RewriteRegion(const Expr &operand, int axis,
                           const PrimExpr &offset, const PrimExpr &extent) {
  auto region = NormalizeToTensorRegion(operand);
  Array<Range> ranges = region->region;
  ranges.Set(axis, Range::FromMinExtent(ranges[axis]->begin + offset, extent));
  return TensorRegion(region->source, ranges, region->ty, region->loc);
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

class DualCopyRewriter : public StmtExprMutator {
public:
  explicit DualCopyRewriter(bool allow_root_sid_synthesis)
      : allow_root_sid_synthesis_(allow_root_sid_synthesis) {}
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    bool vector = BlockIs(op, "VECTOR");
    if (vector)
      ++vector_depth_;
    auto previous = subcore_;
    auto scope = ReadSubcore(op);
    bool unscoped = scope && vector_depth_ == 0;
    if (unscoped)
      ++unscoped_cthread_depth_;
    if (scope)
      subcore_ = scope;

    bool synthesize = allow_root_sid_synthesis_ && IsKernel(op) && !subcore_ &&
                      UnscopedDualCopyFinder::NeedsSid(op->body);
    if (synthesize) {
      PrimVar sid("sid", PrimType::Int(32));
      subcore_ = SubcoreScope{sid, IntImm::Int32(2), IntImm::Int32(0),
                              IntImm::Int32(1)};
    }
    auto result =
        StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
    if (synthesize) {
      auto region = result.as_or_throw<RegionStmt>();
      region.CopyOnWrite()->body = SeqStmt(
          RegionStmt(launch_thread_op(),
                     {StringImm("cthread"), subcore_->extent}, {subcore_->sid},
                     DictAttrs(), region->body, {}, op->loc),
          op->loc);
      result = region;
    }
    subcore_ = std::move(previous);
    if (unscoped)
      --unscoped_cthread_depth_;
    if (vector)
      --vector_depth_;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    Stmt rewritten =
        StmtExprMutator::Mutate_(op, mode).ValueOrUnchanged(GetRef<Stmt>(op));
    const auto *evaluate = rewritten.as<EvaluateObj>();
    const auto *call_node = evaluate->value.as<CallObj>();
    if (!call_node || !IsAscendCopyCall(call_node))
      return rewritten;
    Call call = GetRef<Call>(call_node);
    auto decoded = DecodeTileOpArguments(call_node);
    AscendCopy copy(decoded.operands, decoded.annotations);
    DualCopyPath path = ClassifyDualCopy(copy);
    if (path == DualCopyPath::kNotDualCopy)
      return rewritten;
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
    int source_axis = SplitAxis(copy->src_range, copy->dual_dst_ctl, path);
    int destination_axis = SplitAxis(copy->dst_range, copy->dual_dst_ctl, path);
    bool source_is_full = path == DualCopyPath::kSoftwareGMToUB;
    PrimExpr extent = source_is_full ? copy->dst_range[destination_axis]->extent
                                     : copy->src_range[source_axis]->extent;
    size_t full_operand = source_is_full ? 0 : 1;
    decoded.operands.Set(
        full_operand,
        RewriteRegion(decoded.operands[full_operand],
                      source_is_full ? source_axis : destination_axis,
                      sid * extent, extent));
    return Evaluate(
        MakeAnnotatedCall(
            call->ty, call->op.as_or_throw<Op>(), decoded.operands,
            StripDualCopyAnnotations(decoded.annotations, !source_is_full),
            call->loc),
        op->loc);
  }

private:
  PrimVar RequireSid(const Call &call) const {
    if (!allow_root_sid_synthesis_ &&
        (vector_depth_ == 0 || unscoped_cthread_depth_ != 0))
      TVM_FFI_THROW(ValueError)
          << "RewriteDualCopy requires software dual_copy operations in a "
             "manual Ascend pipeline to be inside an explicit T.Vector() "
             "block within T.Kernel(); T.MixedKernel is not supported in "
             "manual mode. Operation: "
          << call;
    TVM_FFI_CHECK(subcore_.has_value(), ValueError)
        << "RewriteDualCopy found a dual-copy operation outside a cthread "
           "scope and outside a native kernel where one can be created. "
           "Operation: "
        << call;
    auto extent = Constant(subcore_->extent);
    TVM_FFI_CHECK(extent && *extent == 2 && IsZero(subcore_->begin) &&
                      IsOne(subcore_->step),
                  ValueError)
        << "RewriteDualCopy requires a cthread extent of 2 with domain [0, 2) "
           "and step 1, got "
        << subcore_->extent << ". Operation: " << call;
    return subcore_->sid;
  }

  std::optional<SubcoreScope> subcore_;
  int vector_depth_{0};
  int unscoped_cthread_depth_{0};
  bool allow_root_sid_synthesis_;
};
} // namespace

tvm::transform::Pass RewriteDualCopy() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) {
        if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
          return function;
        CheckWellFormed(function);
        bool automatic =
            context->GetConfig<bool>(kEnableAutoSchedule).value_or(true);
        auto rewriter = make_object<DualCopyRewriter>(automatic);
        function.CopyOnWrite()->body =
            rewriter->Mutate(function->body).ValueOrUnchanged(function->body);
        CheckWellFormed(function);
        return function;
      },
      0, "tl.RewriteDualCopy");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.RewriteDualCopy",
                                   RewriteDualCopy);
}
} // namespace tvm::tl
