#include "access_analysis.h"

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/parallel.h"
#include "op/region.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/*!
 * \brief Check whether two regions have intersections.
 * \param region1 The first region.
 * \param region2 The second region.
 * \return Whether region1 and region2 have intersections.
 */
bool MayConflict(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

BufferRegionCollector::BufferRegionCollector(
    Map<Var, Buffer> buffer_data_to_buffer, Target target)
    : buffer_data_to_buffer_(buffer_data_to_buffer), target_(target) {}

Array<BufferRegion> BufferRegionCollector::GetReads() const { return reads_; }

Array<BufferRegion> BufferRegionCollector::GetWrites() const { return writes_; }

bool BufferRegionCollector::GetGlobalCopyPattern() const {
  return is_global_copy_pattern_;
}

bool BufferRegionCollector::GetTmaCopyPattern() const { return is_tma_copy_; }

bool BufferRegionCollector::HasNonCopyTileOp() const {
  return has_non_copy_tile_op_;
}

bool BufferRegionCollector::IsGlobalLikeBuffer(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || (buffer.defined() && buffer.scope().empty());
}

void BufferRegionCollector::HandleTileOp(const TileOperator &tile_op) {
  if (tile_op.as<RegionOpNode>()) {
    return;
  }
  if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
    BufferRegionCollector nested(buffer_data_to_buffer_, target_);
    nested(parallel->GetRoot());
    reads_.insert(reads_.end(), nested.GetReads().begin(),
                  nested.GetReads().end());
    writes_.insert(writes_.end(), nested.GetWrites().begin(),
                   nested.GetWrites().end());
    is_global_copy_pattern_ =
        is_global_copy_pattern_ || nested.GetGlobalCopyPattern();
    is_tma_copy_ = is_tma_copy_ || nested.GetTmaCopyPattern();
    has_non_copy_tile_op_ = has_non_copy_tile_op_ || nested.HasNonCopyTileOp();
    return;
  }
  AccessRegions access = tile_op->GetAccessRegions();
  reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
  writes_.insert(writes_.end(), access.writes.begin(), access.writes.end());
  if (const auto *copy = tile_op.as<CopyNode>()) {
    if (IsGlobalLikeBuffer(copy->src) && IsSharedBuffer(copy->dst)) {
      is_global_copy_pattern_ = true;
    }
  }
  // Im2Col always uses TMA on Hopper.
  if (const auto *im2col = tile_op.as<Im2ColOpNode>()) {
    if (IsGlobalLikeBuffer(im2col->src_) && IsSharedBuffer(im2col->dst_)) {
      is_global_copy_pattern_ = true;
      if (TargetIsHopper(target_)) {
        is_tma_copy_ = true;
      }
    }
    return;
  }
  if (!tile_op.as<CopyNode>()) {
    has_non_copy_tile_op_ = true;
  }
}

void BufferRegionCollector::VisitStmt_(const BufferStoreNode *op) {
  Buffer store_buffer = op->buffer;
  Array<PrimExpr> indices = op->indices;
  // convert indices to region
  Array<Range> region;
  for (const auto &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  auto store_region = BufferRegion(store_buffer, region);
  writes_.push_back(store_region);

  is_global_read_ = false;
  this->VisitExpr(op->value);
  if (is_global_read_ && IsSharedBuffer(store_buffer)) {
    is_global_copy_pattern_ = true;
  }
  is_global_read_ = false;
}

void BufferRegionCollector::VisitExpr_(const BufferLoadNode *op) {
  auto load_buffer = op->buffer;
  Array<PrimExpr> indices = op->indices;
  // convert indices to region
  Array<Range> region;
  for (const auto &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  auto load_region = BufferRegion(load_buffer, region);
  reads_.push_back(load_region);

  if (IsGlobalLikeBuffer(op->buffer) && !within_condition_expr_) {
    // skip condition expr of if_then_else node
    // shared[i] = T.if_then_else(global[i] < n, register_a[i], register_b[i])
    // is not a global read shared[i] = T.if_then_else(global[i] < n,
    // global_a[i], global_b[i]) is a global read
    is_global_read_ = true;
  }
}

void BufferRegionCollector::VisitExpr_(const CallNode *op) {
  if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
    HandleTileOp(tile_op);
    StmtExprVisitor::VisitExpr_(op);
    return;
  }
  if (op->op.same_as(builtin::address_of())) {
    BufferRegion buffer_region;
    if (const auto *load = op->args[0].as<BufferLoadNode>()) {
      buffer_region = BufferRegion::FullRegion(load->buffer);
    } else if (const auto *var_node = op->args[0].as<VarNode>()) {
      Var data_var = GetRef<Var>(var_node);
      auto it = buffer_data_to_buffer_.find(data_var);
      if (it != buffer_data_to_buffer_.end()) {
        buffer_region = BufferRegion::FullRegion((*it).second);
      }
    }
    if (buffer_region.defined()) {
      // because we only care about the buffer itself instead of indices
      reads_.push_back(buffer_region);
    }
  } else if (op->op.same_as(builtin::tvm_access_ptr())) {
    const VarNode *buffer_var = op->args[1].as<VarNode>();
    ICHECK(buffer_var);
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(buffer_var));
    if (it != buffer_data_to_buffer_.end()) {
      const Buffer &buffer = (*it).second;
      const BufferRegion buffer_region = BufferRegion::FullRegion(buffer);
      // because we only care about the buffer itself instead of indices
      reads_.push_back(buffer_region);
    }
  } else if (op->op.same_as(builtin::if_then_else())) {
    within_condition_expr_ = true;
    this->VisitExpr(op->args[0]);
    within_condition_expr_ = false;
    for (auto i = 1; i < op->args.size(); i++) {
      this->VisitExpr(op->args[i]);
    }
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

void BufferRegionCollector::VisitStmt_(const IfThenElseNode *op) {
  within_condition_expr_ = true;
  this->VisitExpr(op->condition);
  within_condition_expr_ = false;
  this->VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    within_condition_expr_ = true;
    this->VisitStmt(op->else_case.value());
    within_condition_expr_ = false;
  }
}

} // namespace tl
} // namespace tvm
