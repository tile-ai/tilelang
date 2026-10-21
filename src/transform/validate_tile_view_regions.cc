/*!
 * \file validate_tile_view_regions.cc
 * \brief Validate regions that must be representable as Sunmmio tile views.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../op/builtin.h"
#include "../op/comm.h"
#include "../op/copy.h"
#include "../op/operator.h"
#include "../op/transpose.h"
#include "../op/utils.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

class ValidateTileViewRegionsPass : public StmtExprVisitor {
public:
  static PrimFunc Run(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsSunmmio(target.value())) {
      return f;
    }

    ValidateTileViewRegionsPass validator;
    validator.target_ = target.value();
    validator.VisitStmt(f->body);
    return f;
  }

private:
  struct DimLayoutInfo {
    bool has_layout{false};
    bool has_dynamic_outer_mode{false};
    bool is_blockwise{false};
    bool is_blockwise_non_major_dim{false};
    int64_t coalesced_extent{1};
    int64_t inner_static_mode_shape{1};
  };

  struct ScopeFrame {
    LayoutMap layout_map;
    LayoutMap global_layout_map;
  };

  enum class DivisibilityProof {
    kDivisible,
    kNotDivisible,
    kUnknown,
  };

  void VisitStmt_(const BlockNode *op) final {
    PushScope(op);
    StmtExprVisitor::VisitStmt_(op);
    scope_stack_.pop_back();
  }

  void VisitExpr_(const CallNode *op) final {
    if (IsCopyTileOp(op)) {
      ValidateTwoRegionOp(op, "tl.tileop.copy", /*arg_base=*/0);
    } else if (IsTransposeTileOp(op)) {
      ValidateTwoRegionOp(op, "tl.tileop.transpose", /*arg_base=*/0);
    } else if (IsDmaCopyOp(op)) {
      ValidateTwoRegionOp(op, "tl.dma_copy", /*arg_base=*/0);
    } else if (IsCommBroadcastOp(op)) {
      ValidateTwoRegionOp(op, "tl.tileop.comm_broadcast", /*arg_base=*/0);
    } else if (IsCommPutOp(op)) {
      ValidateTwoRegionOp(op, "tl.tileop.comm_put", /*arg_base=*/0);
    } else if (IsCommAllgatherOp(op)) {
      ValidateAllgatherOp(op);
    } else if (IsCommAllreduceOp(op)) {
      ValidateAllreduceOp(op);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  static bool IsCopyTileOp(const CallNode *call) {
    return call->op.same_as(Copy::Get());
  }

  static bool IsTransposeTileOp(const CallNode *call) {
    return call->op.same_as(Transpose::Get());
  }

  static bool IsDmaCopyOp(const CallNode *call) {
    return call->op.same_as(dma_copy());
  }

  static bool IsCommBroadcastOp(const CallNode *call) {
    return call->op.same_as(BroadcastOp::Get());
  }

  static bool IsCommPutOp(const CallNode *call) {
    return call->op.same_as(PutOp::Get());
  }

  static bool IsCommAllgatherOp(const CallNode *call) {
    return call->op.same_as(AllgatherOp::Get());
  }

  static bool IsCommAllreduceOp(const CallNode *call) {
    return call->op.same_as(AllreduceOp::Get());
  }

  void PushScope(const BlockNode *block) {
    ScopeFrame frame;
    if (!scope_stack_.empty()) {
      frame = scope_stack_.back();
    }

    CollectLayoutAnnotation(block, attr::kLayoutMap, &frame.layout_map);
    CollectLayoutAnnotation(block, attr::kGlobalLayoutMap,
                            &frame.global_layout_map);
    scope_stack_.push_back(std::move(frame));
  }

  void CollectLayoutAnnotation(const BlockNode *block, const char *attr_key,
                               LayoutMap *dst) {
    if (!block->annotations.count(attr_key)) {
      return;
    }

    auto layout_map_obj = block->annotations.Get(attr_key).value();
    if (auto layout_map = layout_map_obj.as<LayoutMap>()) {
      for (const auto &[buffer, layout] : layout_map.value()) {
        dst->Set(buffer, layout);
      }
      return;
    }

    if (auto layout_map = layout_map_obj.as<Map<Var, Layout>>()) {
      std::vector<Buffer> block_buffers;
      for (const Buffer &buffer : block->alloc_buffers) {
        AddBufferIfMissing(&block_buffers, buffer);
      }
      for (const BufferRegion &region : block->reads) {
        AddBufferIfMissing(&block_buffers, region->buffer);
      }
      for (const BufferRegion &region : block->writes) {
        AddBufferIfMissing(&block_buffers, region->buffer);
      }
      for (const MatchBufferRegion &match_buffer : block->match_buffers) {
        AddBufferIfMissing(&block_buffers, match_buffer->buffer);
      }

      for (const auto &[buffer_var, layout] : layout_map.value()) {
        bool found = false;
        for (const Buffer &buffer : block_buffers) {
          if (buffer->data.same_as(buffer_var)) {
            dst->Set(buffer, layout);
            found = true;
          }
        }
        ICHECK(found) << attr_key
                      << " annotation references unknown buffer var "
                      << buffer_var << ".";
      }
      return;
    }

    LOG(FATAL) << "Unsupported " << attr_key << " annotation type.";
  }

  static void AddBufferIfMissing(std::vector<Buffer> *buffers,
                                 const Buffer &buffer) {
    for (const Buffer &existing : *buffers) {
      if (existing.same_as(buffer)) {
        return;
      }
    }
    buffers->push_back(buffer);
  }

  void ValidateTwoRegionOp(const CallNode *call, const char *op_name,
                           size_t arg_base) {
    ICHECK_GE(call->args.size(), arg_base + 2)
        << op_name << " expects at least src and dst region arguments";

    BufferRegion src = NormalizeToBufferRegion(call->args[arg_base]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[arg_base + 1]);

    ValidateRegionCanFormTileView(src, op_name, "src");
    ValidateRegionCanFormTileView(dst, op_name, "dst");
  }

  void ValidateAllgatherOp(const CallNode *call) {
    ICHECK_GE(call->args.size(), 4U)
        << "tl.tileop.comm_allgather expects at least send, recv, direction, "
           "and size";

    const auto *direction_imm = call->args[2].as<IntImmNode>();
    ICHECK(direction_imm)
        << "tl.tileop.comm_allgather direction must be IntImm";
    int64_t direction = direction_imm->value;
    ICHECK(direction == 0 || direction == 1 || direction == 2)
        << "Invalid direction value for tl.tileop.comm_allgather: "
        << direction;

    int64_t axis = -1;
    if (call->args.size() > 4) {
      const auto *axis_imm = call->args[4].as<IntImmNode>();
      ICHECK(axis_imm) << "tl.tileop.comm_allgather axis must be IntImm";
      axis = axis_imm->value;
    }

    ValidateAllgatherRegions(NormalizeToBufferRegion(call->args[0]),
                             NormalizeToBufferRegion(call->args[1]), direction,
                             axis, "tl.tileop.comm_allgather");
  }

  void ValidateAllreduceOp(const CallNode *call) {
    ICHECK(call->args.size() == 8 || call->args.size() == 9)
        << "tl.tileop.comm_allreduce expects 8 or 9 inputs, got "
        << call->args.size();

    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    BufferRegion row_allgather = NormalizeToBufferRegion(call->args[2]);
    BufferRegion col_allgather = NormalizeToBufferRegion(call->args[3]);

    const auto *type_imm = call->args[4].as<StringImmNode>();
    ICHECK(type_imm)
        << "tl.tileop.comm_allreduce reduce type must be StringImm";

    const auto *direction_imm = call->args[5].as<IntImmNode>();
    ICHECK(direction_imm)
        << "tl.tileop.comm_allreduce direction must be IntImm";
    int64_t direction = direction_imm->value;
    ICHECK(direction == 0 || direction == 1 || direction == 2)
        << "Invalid direction value for tl.tileop.comm_allreduce: "
        << direction;

    const auto *clear_imm = call->args[6].as<IntImmNode>();
    ICHECK(clear_imm) << "tl.tileop.comm_allreduce clear must be IntImm";
    bool should_clear = clear_imm->value != 0;
    bool needs_dst_copy = direction == 2 && !should_clear;
    ICHECK_EQ(call->args.size(), needs_dst_copy ? 9U : 8U)
        << "T.comm.all_reduce dst_copy must be provided only for direction=all "
           "with clear=false";
    PrimExpr dst_copy_expr =
        call->args.size() == 9 ? call->args[7] : PrimExpr();
    ValidateAllreduceContract(
        call->args[0], call->args[1], call->args[2], call->args[3],
        dst_copy_expr, Downcast<StringImm>(call->args[4]),
        static_cast<int>(direction), should_clear, target_.value(), &analyzer_);

    ValidateRegionCanFormTileView(src, "tl.tileop.comm_allreduce", "src");
    ValidateRegionCanFormTileView(dst, "tl.tileop.comm_allreduce", "dst");

    BufferRegion row_gather_send = src;
    const BufferRegion &col_gather_send = src;
    if (direction == 2 && should_clear) {
      row_gather_send = dst;
    } else if (needs_dst_copy) {
      BufferRegion dst_copy = NormalizeToBufferRegion(dst_copy_expr);
      ValidateRegionCanFormTileView(dst_copy, "tl.tileop.comm_allreduce",
                                    "dst_copy");
      row_gather_send = dst_copy;
    }

    if (direction == 0 || direction == 2) {
      ValidateAllgatherRegions(row_gather_send, row_allgather, /*direction=*/0,
                               /*axis=*/-1, "tl.tileop.comm_allreduce.row");
    }
    if (direction == 1 || direction == 2) {
      ValidateAllgatherRegions(col_gather_send, col_allgather, /*direction=*/1,
                               /*axis=*/-1, "tl.tileop.comm_allreduce.col");
    }
  }

  void ValidateAllgatherRegions(const BufferRegion &send,
                                const BufferRegion &recv, int64_t direction,
                                int64_t axis, const char *op_name) {
    ValidateRegionCanFormTileView(send, op_name, "send");

    auto mesh = GetSunmmioMeshConfig(target_.value());
    int64_t recv_num = 1;
    if (direction == 0) {
      recv_num = mesh.ncol;
    } else if (direction == 1) {
      recv_num = mesh.nrow;
    } else {
      recv_num = mesh.nrow * mesh.ncol;
    }

    int recv_rank = static_cast<int>(recv->region.size());
    ICHECK_GT(recv_rank, 0) << op_name << " recv must have at least one dim.";
    if (axis >= 0) {
      ICHECK_EQ(recv->region.size(), send->region.size())
          << "In axis mode, recv and send must have the same rank.";
      ICHECK_LT(axis, static_cast<int64_t>(send->region.size()))
          << "allgather axis " << axis << " is out of range for send rank "
          << send->region.size() << ".";
    }
    if (axis > 0) {
      ICHECK_EQ(axis, static_cast<int64_t>(send->region.size()) - 1)
          << "Only axis = 0 or last dim of send is supported; got axis=" << axis
          << " for send rank=" << send->region.size();
      ICHECK_EQ(recv->region.size(), send->region.size())
          << "In axis mode, recv and send must have the same rank.";
    }

    int slice_axis = (axis > 0) ? (recv_rank - 1) : 0;
    PrimExpr recv_num_expr = IntImm(DataType::Int(32), recv_num);
    ICHECK(CanProveDivisible(recv->region[slice_axis]->extent, recv_num_expr))
        << op_name << " recv extent along slice axis " << slice_axis << " ("
        << recv->region[slice_axis]->extent
        << ") must be divisible by recv_num (" << recv_num << ").";
    PrimExpr slot_extent = analyzer_.Simplify(
        floordiv(recv->region[slice_axis]->extent, recv_num_expr));
    if (axis >= 0) {
      ICHECK(analyzer_.CanProveEqual(slot_extent,
                                     send->region[slice_axis]->extent))
          << op_name << " recv slot extent along axis " << slice_axis
          << " must equal send extent " << send->region[slice_axis]->extent
          << ", but got " << slot_extent << ".";
    }

    auto make_slab = [&](PrimExpr slot_start, PrimExpr extent) {
      Array<Range> ranges;
      for (int dim = 0; dim < recv_rank; ++dim) {
        if (dim == slice_axis) {
          PrimExpr base =
              analyzer_.Simplify(recv->region[dim]->min + slot_start * extent);
          ranges.push_back(Range::FromMinExtent(base, extent));
        } else {
          ranges.push_back(recv->region[dim]);
        }
      }
      return BufferRegion(recv->buffer, ranges);
    };

    int gather_axis_for_validation = -1;
    if (axis >= 0 ||
        (axis < 0 && recv->region.size() == send->region.size() + 1)) {
      gather_axis_for_validation = slice_axis;
    }

    auto validate_slab_range = [&](int64_t slab_count, const PrimExpr &extent,
                                   const char *operand_name) {
      for (int64_t idx = 0; idx < slab_count; ++idx) {
        ValidateAllgatherRecvSlab(
            make_slab(IntImm(extent.dtype(), idx), extent), op_name,
            operand_name, gather_axis_for_validation, slot_extent);
      }
    };

    validate_slab_range(recv_num, slot_extent, "recv_slab");
    if (direction == 2) {
      PrimExpr row_extent = analyzer_.Simplify(
          IntImm(DataType::Int(32), mesh.ncol) * slot_extent);
      validate_slab_range(mesh.nrow, row_extent, "recv_row_slab");
    }
  }

  void ValidateAllgatherRecvSlab(const BufferRegion &region,
                                 const char *op_name, const char *operand_name,
                                 int gather_axis, const PrimExpr &slot_extent) {
    if (gather_axis < 0 ||
        analyzer_.CanProveEqual(region->region[gather_axis]->extent, 1)) {
      ValidateRegionCanFormTileView(region, op_name, operand_name);
      return;
    }

    ValidateBasicDim(region, gather_axis, op_name, operand_name);
    const Range &gather_range = region->region[gather_axis];
    ICHECK(CanProveDivisible(gather_range->min, slot_extent))
        << op_name << " " << operand_name << " gather-axis min at dim "
        << gather_axis << " for buffer " << region->buffer->name
        << " must align to allgather slot extent " << slot_extent
        << ", but got min=" << gather_range->min << ".";
    ICHECK(CanProveDivisible(gather_range->extent, slot_extent))
        << op_name << " " << operand_name << " gather-axis extent at dim "
        << gather_axis << " for buffer " << region->buffer->name
        << " must be a multiple of allgather slot extent " << slot_extent
        << ", but got extent=" << gather_range->extent << ".";

    Array<Range> tile_ranges;
    for (int dim = 0; dim < static_cast<int>(region->region.size()); ++dim) {
      const Range &range = region->region[dim];
      if (dim == gather_axis) {
        tile_ranges.push_back(Range::FromMinExtent(range->min, 1));
      } else {
        tile_ranges.push_back(range);
      }
    }
    ValidateRegionCanFormTileView(BufferRegion(region->buffer, tile_ranges),
                                  op_name, operand_name);
  }

  void ValidateRegionCanFormTileView(const BufferRegion &region,
                                     const char *op_name,
                                     const char *operand_name) {
    ICHECK(region.defined())
        << op_name << " " << operand_name << " region is undefined";
    ICHECK(region->buffer.defined())
        << op_name << " " << operand_name << " buffer is undefined";
    ICHECK(!region->region.empty())
        << op_name << " " << operand_name
        << " region must have at least one dimension";
    ICHECK_EQ(region->region.size(), region->buffer->shape.size())
        << op_name << " " << operand_name << " region rank "
        << region->region.size() << " does not match buffer "
        << region->buffer->name << " rank " << region->buffer->shape.size();

    std::vector<std::optional<int64_t>> static_extents(region->region.size());
    std::vector<int> non_unit_dims;
    for (size_t dim = 0; dim < region->region.size(); ++dim) {
      ValidateRangeDefined(region, dim, op_name, operand_name);
      static_extents[dim] =
          ValidateBasicDim(region, dim, op_name, operand_name);

      const Range &range = region->region[dim];
      if (analyzer_.CanProveEqual(range->extent, 1)) {
        continue;
      }

      non_unit_dims.push_back(static_cast<int>(dim));
    }

    if (non_unit_dims.empty()) {
      return;
    }

    Layout layout = LookupLayout(region->buffer, op_name, operand_name);
    const size_t tile_dim_count = std::min<size_t>(2, non_unit_dims.size());
    const size_t tile_dim_begin = non_unit_dims.size() - tile_dim_count;
    for (size_t idx = 0; idx < non_unit_dims.size(); ++idx) {
      size_t dim = static_cast<size_t>(non_unit_dims[idx]);
      if (idx >= tile_dim_begin) {
        ValidateTileViewDim(region, layout, dim, static_extents[dim], op_name,
                            operand_name);
      }
    }
  }

  void ValidateRangeDefined(const BufferRegion &region, size_t dim,
                            const char *op_name, const char *operand_name) {
    const Range &range = region->region[dim];
    ICHECK(range.defined())
        << op_name << " " << operand_name << " range at dim " << dim
        << " is undefined for buffer " << region->buffer->name;
    ICHECK(range->extent.defined())
        << op_name << " " << operand_name << " extent at dim " << dim
        << " is undefined for buffer " << region->buffer->name;
  }

  std::optional<int64_t> TryGetStaticInt(const PrimExpr &expr) {
    PrimExpr simplified = analyzer_.Simplify(expr);
    if (const auto *imm = simplified.as<IntImmNode>()) {
      return imm->value;
    }
    return std::nullopt;
  }

  bool CanProveDivisible(const PrimExpr &value, const PrimExpr &divisor) {
    return ProveDivisibility(value, divisor) == DivisibilityProof::kDivisible;
  }

  DivisibilityProof ProveDivisibility(const PrimExpr &value,
                                      const PrimExpr &divisor) {
    PrimExpr simplified_divisor = analyzer_.Simplify(divisor);
    if (analyzer_.CanProveEqual(simplified_divisor, 0)) {
      return DivisibilityProof::kNotDivisible;
    }
    PrimExpr remainder =
        analyzer_.Simplify(floormod(value, simplified_divisor));
    if (analyzer_.CanProveEqual(remainder, 0)) {
      return DivisibilityProof::kDivisible;
    }
    if (analyzer_.CanProve(remainder != make_zero(remainder.dtype()))) {
      return DivisibilityProof::kNotDivisible;
    }
    return DivisibilityProof::kUnknown;
  }

  std::optional<int64_t> ValidateBasicDim(const BufferRegion &region,
                                          size_t dim, const char *op_name,
                                          const char *operand_name) {
    const Range &range = region->region[dim];
    const PrimExpr &buffer_shape = region->buffer->shape[dim];
    const PrimExpr &region_min = range->min;
    const PrimExpr &region_extent = range->extent;

    std::optional<int64_t> static_extent = TryGetStaticInt(region_extent);
    if (!static_extent.has_value()) {
      LOG(WARNING) << op_name << " " << operand_name << " region extent at dim "
                   << dim << " for buffer " << region->buffer->name
                   << " is not a compile-time constant; skip tile_view "
                      "validation for this dimension. extent="
                   << region_extent << ".";
      return std::nullopt;
    }
    ICHECK_GT(static_extent.value(), 0)
        << op_name << " " << operand_name << " region extent at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be positive, but got extent=" << region_extent << ".";

    ICHECK(!analyzer_.CanProve(region_min < make_zero(region_min.dtype())))
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name
        << " must be non-negative, but got min=" << region_min << ".";
    PrimExpr region_end = analyzer_.Simplify(region_min + region_extent);
    ICHECK(!analyzer_.CanProve(region_end > buffer_shape))
        << op_name << " " << operand_name << " region at dim " << dim
        << " for buffer " << region->buffer->name
        << " must stay within buffer shape " << buffer_shape
        << ", but got min=" << region_min << " and extent=" << region_extent
        << ".";
    if (!analyzer_.CanProve(region_end <= buffer_shape)) {
      LOG(WARNING) << op_name << " " << operand_name << " region at dim " << dim
                   << " for buffer " << region->buffer->name
                   << " cannot be proven within buffer shape " << buffer_shape
                   << "; skip bounds validation for this dimension. min="
                   << region_min << ", extent=" << region_extent
                   << ", end=" << region_end << ".";
    }

    return static_extent;
  }

  void ValidateTileViewDim(const BufferRegion &region, const Layout &layout,
                           size_t dim,
                           const std::optional<int64_t> &static_extent,
                           const char *op_name, const char *operand_name) {
    if (!static_extent.has_value()) {
      return;
    }

    const Range &range = region->region[dim];
    const PrimExpr &buffer_shape = region->buffer->shape[dim];
    const PrimExpr &region_min = range->min;
    const PrimExpr &region_extent = range->extent;

    DivisibilityProof min_alignment =
        ProveDivisibility(region_min, region_extent);
    ICHECK(min_alignment != DivisibilityProof::kNotDivisible)
        << op_name << " " << operand_name << " region min at dim " << dim
        << " for buffer " << region->buffer->name
        << " must align to region extent " << region_extent
        << ", but got min=" << region_min << ".";
    if (min_alignment == DivisibilityProof::kUnknown) {
      LOG(WARNING) << op_name << " " << operand_name << " region min at dim "
                   << dim << " for buffer " << region->buffer->name
                   << " cannot be proven aligned to region extent "
                   << region_extent
                   << "; skip alignment validation for this dimension. min="
                   << region_min << ".";
    }

    DimLayoutInfo info = GetDimLayoutInfo(layout, dim, region->buffer);
    if (!info.has_layout) {
      return;
    }

    if (info.has_dynamic_outer_mode) {
      const bool splits_inner_static_mode =
          info.inner_static_mode_shape % static_extent.value() == 0;
      const bool covers_whole_inner_static_modes =
          static_extent.value() % info.inner_static_mode_shape == 0;
      ICHECK(splits_inner_static_mode || covers_whole_inner_static_modes)
          << op_name << " " << operand_name << " region extent at dim " << dim
          << " for buffer " << region->buffer->name
          << " must be compatible with dynamic layout inner static mode shape "
          << info.inner_static_mode_shape
          << ", but got extent=" << region_extent << ".";
      return;
    }

    DivisibilityProof shape_divisibility =
        ProveDivisibility(buffer_shape, region_extent);
    ICHECK(shape_divisibility != DivisibilityProof::kNotDivisible)
        << op_name << " " << operand_name << " region extent at dim " << dim
        << " for buffer " << region->buffer->name
        << " must divide buffer shape " << buffer_shape
        << ", but got extent=" << region_extent << ".";
    if (shape_divisibility == DivisibilityProof::kUnknown) {
      LOG(WARNING) << op_name << " " << operand_name << " region extent at dim "
                   << dim << " for buffer " << region->buffer->name
                   << " cannot be proven to divide buffer shape "
                   << buffer_shape
                   << "; skip buffer-shape divisibility validation for this "
                      "dimension. extent="
                   << region_extent << ".";
    }

    if (info.is_blockwise) {
      const bool divides_coalesced_extent =
          info.coalesced_extent % static_extent.value() == 0;
      const bool covers_whole_coalesced_blocks =
          static_extent.value() % info.coalesced_extent == 0;
      ICHECK(divides_coalesced_extent || covers_whole_coalesced_blocks)
          << op_name << " " << operand_name
          << " blockwise region extent at dim " << dim << " for buffer "
          << region->buffer->name
          << " must be compatible with coalesced extent "
          << info.coalesced_extent << ", but got extent=" << region_extent
          << ".";
      const bool splits_innermost_block =
          static_extent.value() < info.inner_static_mode_shape;
      if (divides_coalesced_extent && splits_innermost_block) {
        ICHECK(!info.is_blockwise_non_major_dim)
            << op_name << " " << operand_name
            << " blockwise region extent at dim " << dim << " for buffer "
            << region->buffer->name << " splits innermost block extent "
            << info.inner_static_mode_shape
            << " and must not split the non-major dimension, but got extent="
            << region_extent << ".";
      }
      return;
    }

    if (info.coalesced_extent < static_extent.value()) {
      ICHECK_EQ(static_extent.value() % info.coalesced_extent, 0)
          << op_name << " " << operand_name << " region extent at dim " << dim
          << " for buffer " << region->buffer->name
          << " must be compatible with coalesced extent "
          << info.coalesced_extent << ", but got extent=" << region_extent
          << ".";
    } else {
      ICHECK_EQ(info.coalesced_extent % static_extent.value(), 0)
          << op_name << " " << operand_name << " region extent at dim " << dim
          << " for buffer " << region->buffer->name
          << " must be compatible with coalesced extent "
          << info.coalesced_extent << ", but got extent=" << region_extent
          << ".";
    }
  }

  DimLayoutInfo GetDimLayoutInfo(const Layout &layout, size_t dim,
                                 const Buffer &buffer) {
    DimLayoutInfo info;
    info.has_layout = true;

    if (const auto *cute = layout.as<CuteLayoutNode>()) {
      Array<PrimExpr> mode_shapes =
          cute->GetModeShapeOfDim(static_cast<int>(dim));
      Array<PrimExpr> mode_strides =
          cute->GetModeStrideOfDim(static_cast<int>(dim));
      ICHECK_EQ(mode_shapes.size(), mode_strides.size())
          << "CuteLayout mode shape/stride mismatch for buffer " << buffer->name
          << " dim " << dim << ".";
      ICHECK(!mode_shapes.empty()) << "CuteLayout has no modes for buffer "
                                   << buffer->name << " dim " << dim << ".";

      const auto *inner_shape = mode_shapes[0].as<IntImmNode>();
      if (!inner_shape) {
        LOG(WARNING) << "CuteLayout inner mode shape for buffer "
                     << buffer->name << " dim " << dim
                     << " is not a compile-time constant; skip tile_view "
                        "layout validation for this dimension. shape="
                     << mode_shapes[0] << ".";
        info.has_layout = false;
        return info;
      }
      info.inner_static_mode_shape = inner_shape->value;

      info.is_blockwise = mode_shapes.size() > 1;
      if (info.is_blockwise) {
        std::optional<bool> non_major =
            TryGetBlockwiseNonMajorDim(cute, static_cast<int>(dim), buffer);
        if (!non_major.has_value()) {
          info.has_layout = false;
          return info;
        }
        info.is_blockwise_non_major_dim = non_major.value();
      }

      const bool outer_dynamic =
          mode_shapes.size() > 1 &&
          !mode_shapes[mode_shapes.size() - 1].as<IntImmNode>();
      info.has_dynamic_outer_mode = outer_dynamic;
      info.coalesced_extent =
          ComputeDimCoalescedExtent(mode_shapes, mode_strides);
      return info;
    }

    std::optional<int64_t> shape_value = TryGetStaticInt(buffer->shape[dim]);
    if (!shape_value.has_value()) {
      LOG(WARNING) << "Non-CuteLayout buffer " << buffer->name << " dim " << dim
                   << " has non-static buffer.shape; skip tile_view layout "
                      "validation for this dimension. shape="
                   << buffer->shape[dim] << ".";
      info.has_layout = false;
      return info;
    }
    ICHECK_GT(shape_value.value(), 0)
        << "Non-CuteLayout buffer " << buffer->name << " dim " << dim
        << " requires positive buffer.shape, but got " << buffer->shape[dim]
        << ".";
    info.coalesced_extent = shape_value.value();
    info.inner_static_mode_shape = shape_value.value();
    return info;
  }

  std::optional<bool> TryGetBlockwiseNonMajorDim(const CuteLayoutNode *layout,
                                                 int dim,
                                                 const Buffer &buffer) {
    auto dim_levels = layout->GetDimLevels();
    int64_t min_innermost_stride = std::numeric_limits<int64_t>::max();
    std::optional<int64_t> dim_innermost_stride;
    bool found_blockwise_dim = false;

    for (size_t candidate_dim = 0; candidate_dim < dim_levels.size();
         ++candidate_dim) {
      if (dim_levels[candidate_dim].IntValue() <= 1) {
        continue;
      }
      found_blockwise_dim = true;
      Array<PrimExpr> strides =
          layout->GetModeStrideOfDim(static_cast<int>(candidate_dim));
      ICHECK(!strides.empty())
          << "Blockwise CuteLayout has no strides for buffer " << buffer->name
          << " dim " << candidate_dim << ".";
      std::optional<int64_t> stride = TryGetStaticInt(strides[0]);
      if (!stride.has_value()) {
        LOG(WARNING) << "Blockwise CuteLayout innermost stride for buffer "
                     << buffer->name << " dim " << candidate_dim
                     << " is not a compile-time constant; skip tile_view "
                        "layout validation for this dimension. stride="
                     << strides[0] << ".";
        return std::nullopt;
      }
      min_innermost_stride = std::min(min_innermost_stride, stride.value());
      if (static_cast<int>(candidate_dim) == dim) {
        dim_innermost_stride = stride;
      }
    }

    ICHECK(found_blockwise_dim && dim_innermost_stride.has_value())
        << "Blockwise CuteLayout metadata missing for buffer " << buffer->name
        << " dim " << dim << ".";
    return dim_innermost_stride.value() == min_innermost_stride;
  }

  int64_t ComputeDimCoalescedExtent(const Array<PrimExpr> &mode_shapes,
                                    const Array<PrimExpr> &mode_strides) {
    std::vector<std::pair<int64_t, int64_t>> static_modes;
    for (size_t i = 0; i < mode_shapes.size(); ++i) {
      const auto *shape = mode_shapes[i].as<IntImmNode>();
      const auto *stride = mode_strides[i].as<IntImmNode>();
      if (!shape || !stride) {
        break;
      }
      if (shape->value == 1) {
        continue;
      }
      static_modes.push_back({shape->value, stride->value});
    }

    ICHECK(!static_modes.empty())
        << "Cannot compute coalesced extent from layout with no static modes.";
    std::sort(static_modes.begin(), static_modes.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    int64_t coalesced_extent = static_modes[0].first;
    int64_t running_stride = static_modes[0].second;
    for (size_t i = 1; i < static_modes.size(); ++i) {
      if (coalesced_extent * running_stride != static_modes[i].second) {
        break;
      }
      coalesced_extent *= static_modes[i].first;
    }
    return coalesced_extent;
  }

  Layout LookupLayout(const Buffer &buffer, const char *op_name,
                      const char *operand_name) const {
    if (!scope_stack_.empty()) {
      const ScopeFrame &frame = scope_stack_.back();
      if (auto layout = LookupLayoutInMap(frame.layout_map, buffer)) {
        return layout.value();
      }
      if (auto layout = LookupLayoutInMap(frame.global_layout_map, buffer)) {
        return layout.value();
      }
    }

    ICHECK(false) << op_name << " " << operand_name << " buffer "
                  << buffer->name
                  << " requires layout_map/global_layout_map metadata for "
                     "tile_view validation.";
    return Layout();
  }

  std::optional<Layout> LookupLayoutInMap(const LayoutMap &layout_map,
                                          const Buffer &buffer) const {
    if (layout_map.count(buffer)) {
      return layout_map[buffer];
    }

    for (const auto &kv : layout_map) {
      const Buffer &candidate = kv.first;
      if (candidate->data.same_as(buffer->data) ||
          candidate->data->name_hint == buffer->data->name_hint) {
        return kv.second;
      }
    }
    return std::nullopt;
  }

  arith::Analyzer analyzer_;
  std::vector<ScopeFrame> scope_stack_;
  std::optional<Target> target_;
};

} // namespace

tvm::transform::Pass ValidateTileViewRegions() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ValidateTileViewRegionsPass::Run(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ValidateTileViewRegions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ValidateTileViewRegions",
                        ValidateTileViewRegions);
}

} // namespace tl
} // namespace tvm
