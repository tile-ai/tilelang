/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"

#include <cstdint>
#include <limits>

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/cute_layout.h"
#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "../tileview/reduce_tileview_planner.h"
#include "../tileview/tileview.h"
#include "../transform/common/attr.h"
#include "../transform/loop_partition.h"
#include "builtin.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ir/expr.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

PrimExpr MakeUnsignedAllOnes(DataType dtype) {
  ICHECK(dtype.is_uint()) << "Expected unsigned integer dtype, got " << dtype;
  int bits = dtype.bits();
  ICHECK_GT(bits, 0) << "Unsigned integer dtype must have positive width.";
  ICHECK_LE(bits, 64) << "Unsupported unsigned integer dtype width " << bits;

  uint64_t value = bits == 64 ? std::numeric_limits<uint64_t>::max()
                              : ((uint64_t{1} << bits) - uint64_t{1});
  return make_const(dtype, value);
}

PrimExpr MakeSignedMinimum(DataType dtype) {
  ICHECK(dtype.is_int()) << "Expected signed integer dtype, got " << dtype;
  int bits = dtype.bits();
  ICHECK_GT(bits, 0) << "Signed integer dtype must have positive width.";
  ICHECK_LE(bits, 64) << "Unsupported signed integer dtype width " << bits;

  int64_t value = bits == 64 ? std::numeric_limits<int64_t>::min()
                             : -(int64_t{1} << (bits - 1));
  return make_const(dtype, value);
}

PrimExpr MakeSignedMaximum(DataType dtype) {
  ICHECK(dtype.is_int()) << "Expected signed integer dtype, got " << dtype;
  int bits = dtype.bits();
  ICHECK_GT(bits, 0) << "Signed integer dtype must have positive width.";
  ICHECK_LE(bits, 64) << "Unsupported signed integer dtype width " << bits;

  int64_t value = bits == 64 ? std::numeric_limits<int64_t>::max()
                             : ((int64_t{1} << (bits - 1)) - int64_t{1});
  return make_const(dtype, value);
}

void ValidateSunmmioReduceDType(const ReduceType &type, DataType src_dtype,
                                DataType dst_dtype) {
  if (!(type->isBitAnd() || type->isBitOr() || type->isBitXor())) {
    return;
  }
  auto is_supported = [](DataType dtype) {
    return (dtype.is_int() || dtype.is_uint()) && dtype.bits() == 32 &&
           dtype.lanes() == 1;
  };
  ICHECK(is_supported(src_dtype) && is_supported(dst_dtype) &&
         src_dtype == dst_dtype)
      << "Sunmmio bitwise reduction requires input and output dtypes to be "
         "both int32 or both uint32, got "
      << src_dtype << " and " << dst_dtype;
}

} // namespace

// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

ReduceOp::ReduceOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<ReduceOpNode> node = tvm::ffi::make_object<ReduceOpNode>();
  // Accept BufferRegion/BufferLoad for src/dst
  node->srcRegion_ = NormalizeToBufferRegion(args[0]);
  node->dstRegion_ = NormalizeToBufferRegion(args[1]);
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  node->type = ReduceType(reduce_type);
  if (node->type->isBitAnd() || node->type->isBitOr() ||
      node->type->isBitXor()) {
    auto is_integer = [](DataType dtype) {
      return dtype.is_int() || dtype.is_uint();
    };
    ICHECK(is_integer(node->src->dtype) && is_integer(node->dst->dtype))
        << "Bitwise reduction requires integer input and output dtypes, got "
        << node->src->dtype << " and " << node->dst->dtype;
  }
  node->clear = args[4].as<Bool>().value();
  data_ = std::move(node);
}

TileOperator ReduceOpNode::Clone() const {
  auto op = tvm::ffi::make_object<ReduceOpNode>(*this);
  return ReduceOp(op);
}

TileOperator CumSumOpNode::Clone() const {
  auto op = tvm::ffi::make_object<CumSumOpNode>(*this);
  return CumSumOp(op);
}

PrimExpr ReduceOpNode::MakeInitValue() const {
  auto dst_dtype = dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  if (type->isSum()) {
    return make_zero(dst->dtype);
  } else if (type->isAbsSum()) {
    return make_zero(dst->dtype);
  } else if (type->isMax()) {
    if (is_int) {
      return MakeSignedMinimum(dst->dtype);
    } else if (is_uint) {
      return make_const(dst->dtype, 0);
    } else {
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isMin()) {
    if (is_int) {
      return MakeSignedMaximum(dst->dtype);
    } else if (is_uint) {
      return MakeUnsignedAllOnes(dst->dtype);
    } else {
      return make_const(dst->dtype, INFINITY);
    }
  } else if (type->isAbsMax()) {
    return make_const(dst->dtype, 0);
  } else if (type->isBitAnd()) {
    if (is_int) {
      return make_const(dst->dtype, -1);
    } else if (is_uint) {
      return MakeUnsignedAllOnes(dst->dtype);
    } else {
      // Should not arrive here
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isBitOr()) {
    return make_zero(dst->dtype);
  } else if (type->isBitXor()) {
    return make_zero(dst->dtype);
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return PrimExpr();
  }
}

PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &acc,
                                  const PrimExpr &b) const {
  PrimExpr rhs = b;
  if (acc->dtype != rhs->dtype) {
    rhs = Cast(acc->dtype, rhs);
  }
  if (type->isSum()) {
    return acc + rhs;
  } else if (type->isAbsSum()) {
    return acc + tvm::abs(rhs);
  } else if (type->isMax()) {
    return Max(acc, rhs);
  } else if (type->isMin()) {
    return Min(acc, rhs);
  } else if (type->isAbsMax()) {
    return Max(acc, tvm::abs(rhs));
  } else if (type->isBitAnd()) {
    return acc & rhs;
  } else if (type->isBitOr()) {
    return acc | rhs;
  } else if (type->isBitXor()) {
    return acc ^ rhs;
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return PrimExpr();
  }
}

std::string ReduceOpNode::MakeCodegenReducer() const {
  if (type->isSum()) {
    return "tl::SumOp";
  } else if (type->isAbsSum()) {
    return "tl::SumOp";
  } else if (type->isMax()) {
    return "tl::MaxOp";
  } else if (type->isMin()) {
    return "tl::MinOp";
  } else if (type->isAbsMax()) {
    return "tl::MaxOp";
  } else if (type->isBitAnd()) {
    return "tl::BitAndOp";
  } else if (type->isBitOr()) {
    return "tl::BitOrOp";
  } else if (type->isBitXor()) {
    return "tl::BitXorOp";
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return "";
  }
}

static Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

static Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;

  auto fwd = InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));

  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }

  auto reducer_layout =
      Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt)
          ->CondenseReplicateVar()
          ->BindThreadRange(src_layout->ThreadRange());
  return reducer_layout;
}

Stmt ReduceOpNode::MakeSunmmioTileReduce(const LowerArgs &T,
                                         arith::Analyzer *analyzer) const {
  auto src_scope = this->src.scope();
  auto dst_scope = this->dst.scope();

  // Sunmmio target requires buffers to be in shared.rsram scope.
  // The lowering logic here implements tile-level reduction using a
  // combination of element-wise accumulation and specialized hardware
  // primitives.
  ICHECK(src_scope == kSunmmioScopeRSRAM && dst_scope == kSunmmioScopeRSRAM)
      << "For Sunmmio target, Reduce operator src and dst must be in "
         "shared.rsram scope, but got "
      << src_scope << " and " << dst_scope;

  // Helper to find TileView metadata for a buffer, supporting name-hint
  // fallback. This is necessary because TVM may rename buffers (e.g.,
  // adding suffixes like _1, _2) during lowering, which causes direct
  // pointer-based lookup in tileview_map to fail.
  auto find_tileview = [&](const Buffer &buf) -> Optional<TileView> {
    if (T.tileview_map.count(buf->data)) {
      return T.tileview_map.at(buf->data);
    }
    // Fallback: match by name hint, ignoring common suffixes like _1, _2.
    auto simplify_name = [](std::string name) {
      if (name.size() > 2 && name[name.size() - 2] == '_') {
        return name.substr(0, name.size() - 2);
      }
      return name;
    };
    std::string target_name = simplify_name(buf->data->name_hint);
    for (const auto &kv : T.tileview_map) {
      if (simplify_name(kv.first->name_hint) == target_name) {
        return kv.second;
      }
    }
    return std::nullopt;
  };

  Optional<TileView> opt_tv_src = find_tileview(this->src);
  Optional<TileView> opt_tv_dst = find_tileview(this->dst);
  bool allow_tiled_reduction =
      !(this->type->isBitAnd() || this->type->isBitOr() ||
        this->type->isBitXor());

  ReduceTileViewPlan plan = PlanReduceTileViews(
      this->srcRegion_, this->dstRegion_, this->dim,
      {opt_tv_src, opt_tv_dst, allow_tiled_reduction}, T.layout_map,
      GetSunmmioTileProcessorConfig(T.target), analyzer);

  TileView tv_src = plan.src_tileview;
  TileView tv_dst = plan.dst_tileview;
  Array<PrimExpr> source_domain = plan.source_domain;
  int src_ndim = static_cast<int>(source_domain.size());
  int dst_ndim = static_cast<int>(this->dstRegion_->region.size());

  // Map physical buffer dimensions to their logical tile sizes based on
  // IndexMap. E.g., if IndexMap is [-2, -1] and TileShape is [32, 32], the
  // last two dims are tiled.
  std::unordered_map<int, PrimExpr> src_dim_to_tile_size;
  for (size_t i = 0; i < tv_src->IndexMap().size(); i++) {
    PrimExpr idx_expr = analyzer->Simplify(tv_src->IndexMap()[i]);
    const auto *idx_ptr = idx_expr.as<IntImmNode>();
    ICHECK(idx_ptr) << "TileView IndexMap must be constant integers, but got "
                    << tv_src->IndexMap()[i];
    int dim = idx_ptr->value;
    if (dim < 0)
      dim += src_ndim;
    src_dim_to_tile_size[dim] = tv_src->TileShape()[i];
  }

  // Check if the dimension we are reducing along is a tiled dimension.
  bool is_dim_tiled = plan.reduce_tile_axis >= 0;

  // Derive the shape of the single-tile accumulation buffer 'acc'.
  // It has the same rank as 'src', but non-tiled dimensions are collapsed
  // to 1.
  Array<PrimExpr> single_tile_acc_shape;
  single_tile_acc_shape.reserve(src_ndim);
  for (int i = 0; i < src_ndim; ++i) {
    if (src_dim_to_tile_size.count(i)) {
      single_tile_acc_shape.push_back(src_dim_to_tile_size[i]);
    } else {
      single_tile_acc_shape.push_back(1);
    }
  }

  // single_tile_shape for potential intermediate result.
  // Only needed if the reduction dimension is tiled and we are not clearing
  // (i.e. accumulating into existing dst values), since in that case we need to
  // store the per-tile reduction results before accumulating back to dst.
  Array<PrimExpr> single_tile_res_shape;
  std::unordered_map<int, PrimExpr> dst_dim_to_tile_size;
  if (is_dim_tiled) {
    single_tile_res_shape.reserve(dst_ndim);
    for (size_t i = 0; i < tv_dst->IndexMap().size(); i++) {
      const auto *idx_ptr = tv_dst->IndexMap()[i].as<IntImmNode>();
      ICHECK(idx_ptr);
      int dim = idx_ptr->value;
      if (dim < 0)
        dim += dst_ndim;
      dst_dim_to_tile_size[dim] = tv_dst->TileShape()[i];
    }
    for (int i = 0; i < dst_ndim; ++i) {
      if (dst_dim_to_tile_size.count(i)) {
        single_tile_res_shape.push_back(dst_dim_to_tile_size[i]);
      } else {
        single_tile_res_shape.push_back(1);
      }
    }
  }

  // Create the 'acc' buffer in shared.rsram to hold intermediate reduction
  // values for a single tile.
  auto create_tile_buffer = [&](std::string name, Array<PrimExpr> shape) {
    return decl_buffer(shape, this->dst->dtype, name, kSunmmioScopeRSRAM);
  };

  Buffer acc =
      create_tile_buffer(this->dst->name + "_acc", single_tile_acc_shape);

  Optional<Buffer> dst_res;
  if (is_dim_tiled && !this->clear) {
    dst_res =
        create_tile_buffer(this->dst->name + "_res", single_tile_res_shape);
  }

  // Create outer loop variables for each dimension of the source tensor.
  // Tiled dimensions use ceildiv so tail tiles remain in the loop nest.
  Array<IterVar> loop_vars;
  loop_vars.reserve(src_ndim);
  for (int i = 0; i < src_ndim; i++) {
    PrimExpr extent = source_domain[i];
    if (src_dim_to_tile_size.count(i)) {
      extent = analyzer->Simplify(ceildiv(extent, src_dim_to_tile_size[i]));
    }
    Var var("i" + std::to_string(i), extent->dtype);
    loop_vars.push_back({Range(0, extent), var, IterVarType::kDataPar});
  }

  // Identify which buffer dimensions are tiled and map them to tile axes (0,
  // 1, ...).
  std::unordered_map<int, int> src_buf_dim_to_tile_axis;
  for (size_t i = 0; i < tv_src->IndexMap().size(); i++) {
    int dim = tv_src->IndexMap()[i].as<IntImmNode>()->value;
    if (dim < 0)
      dim += src_ndim;
    src_buf_dim_to_tile_axis[dim] = i;
  }

  std::unordered_map<int, int> dst_buf_dim_to_tile_axis;
  for (size_t i = 0; i < tv_dst->IndexMap().size(); i++) {
    int dim = tv_dst->IndexMap()[i].as<IntImmNode>()->value;
    if (dim < 0)
      dim += dst_ndim;
    dst_buf_dim_to_tile_axis[dim] = i;
  }

  Array<PrimExpr> src_tile_shape = tv_src->TileShape();
  Array<PrimExpr> dst_tile_shape = tv_dst->TileShape();
  auto dst_dim_to_src_dim = [&](int dst_dim) {
    for (int src_dim = 0; src_dim < src_ndim; ++src_dim) {
      if (plan.src_dim_to_dst_dim[src_dim] == dst_dim) {
        return src_dim;
      }
    }
    LOG(FATAL) << "Could not map dst dim " << dst_dim
               << " back to a source dim for Sunmmio reduction.";
    return -1;
  };

  // Create interior loop variables (ki, kj, ...) for one element-wise tile
  // loop nest. These Vars are loop binders, so each sibling loop nest must get
  // its own Var objects to keep TIR SSA-clean.
  auto make_interior_vars = [](size_t rank) {
    Array<Var> vars;
    vars.reserve(rank);
    for (size_t i = 0; i < rank; i++) {
      vars.push_back(Var("k" + std::string{char('i' + i)}));
    }
    return vars;
  };

  // Map output tiled axes to the interior variables used by the corresponding
  // source axes. The mapping is derived per loop nest, so dst indices and src
  // indices in the same nest reference the same Var objects.
  auto map_dst_interior_vars = [&](const Array<Var> &src_interior_vars) {
    Array<Var> mapped;
    mapped.reserve(tv_dst->IndexMap().size());
    for (size_t i = 0; i < tv_dst->IndexMap().size(); i++) {
      int dst_dim_val = tv_dst->IndexMap()[i].as<IntImmNode>()->value;
      if (dst_dim_val < 0)
        dst_dim_val += dst_ndim;

      int target_src_dim = dst_dim_to_src_dim(dst_dim_val);

      bool found = false;
      for (size_t j = 0; j < tv_src->IndexMap().size(); j++) {
        int src_dim_val = tv_src->IndexMap()[j].as<IntImmNode>()->value;
        if (src_dim_val < 0)
          src_dim_val += src_ndim;
        if (src_dim_val == target_src_dim) {
          mapped.push_back(src_interior_vars[j]);
          found = true;
          break;
        }
      }
      ICHECK(found) << "Could not map dst tiled axis to src tiled axis";
    }
    return mapped;
  };

  // Helper functions to generate indices for element-wise TIR access.
  // Large buffers use (loop_var * tile_size + interior_var), while 'acc' uses
  // (interior_var).
  auto get_src_indices = [&](const Array<Var> &src_interior_vars) {
    Array<PrimExpr> indices;
    for (int i = 0; i < src_ndim; i++) {
      PrimExpr base = this->srcRegion_->region[i]->min;
      if (src_buf_dim_to_tile_axis.count(i)) {
        int axis = src_buf_dim_to_tile_axis.at(i);
        indices.push_back(base + loop_vars[i]->var * src_tile_shape[axis] +
                          src_interior_vars[axis]);
      } else {
        indices.push_back(base + loop_vars[i]->var);
      }
    }
    return indices;
  };

  auto get_acc_indices = [&](const Array<Var> &src_interior_vars) {
    Array<PrimExpr> indices;
    for (int i = 0; i < src_ndim; i++) {
      if (src_buf_dim_to_tile_axis.count(i)) {
        indices.push_back(src_interior_vars[src_buf_dim_to_tile_axis.at(i)]);
      } else {
        indices.push_back(make_zero(DataType::Int(32)));
      }
    }
    return indices;
  };

  auto get_dst_indices = [&](const Array<Var> &dst_interior_vars_mapped) {
    Array<PrimExpr> indices;
    for (int i = 0; i < dst_ndim; i++) {
      PrimExpr base = this->dstRegion_->region[i]->min;
      if (dst_buf_dim_to_tile_axis.count(i)) {
        int axis = dst_buf_dim_to_tile_axis.at(i);
        int src_dim = dst_dim_to_src_dim(i);
        indices.push_back(base +
                          loop_vars[src_dim]->var * dst_tile_shape[axis] +
                          dst_interior_vars_mapped[axis]);
      } else {
        if (dst_ndim == src_ndim && i == this->dim) {
          indices.push_back(base);
        } else {
          int src_dim = dst_dim_to_src_dim(i);
          indices.push_back(base + loop_vars[src_dim]->var);
        }
      }
    }
    return indices;
  };

  auto get_res_indices = [&](const Array<Var> &dst_interior_vars_mapped) {
    Array<PrimExpr> indices;
    for (int i = 0; i < dst_ndim; i++) {
      if (dst_buf_dim_to_tile_axis.count(i)) {
        indices.push_back(
            dst_interior_vars_mapped[dst_buf_dim_to_tile_axis.at(i)]);
      } else {
        indices.push_back(make_zero(DataType::Int(32)));
      }
    }
    return indices;
  };

  // Wraps a Stmt with interior loops (ki, kj, ...) representing element-wise
  // execution of a Tile. The innermost loop is marked as vectorized.
  auto wrap_interior = [&](Stmt b, const Array<Var> &i_vars,
                           const Array<PrimExpr> &t_shape,
                           const std::vector<int> &axes) {
    for (int i = static_cast<int>(axes.size()) - 1; i >= 0; i--) {
      int axis = axes[i];
      ForKind kind = (i == static_cast<int>(axes.size()) - 1)
                         ? ForKind::kVectorized
                         : ForKind::kSerial;
      For loop(i_vars[axis], 0, t_shape[axis], kind, b);
      auto *n = loop.CopyOnWrite();
      n->annotations.Set(attr::tile_interior, Integer(1));
      n->annotations.Set(attr::tile_interior_axis, Integer(axis));
      b = loop;
    }
    return b;
  };

  std::vector<int> all_src_axes;
  all_src_axes.reserve(src_tile_shape.size());
  for (size_t i = 0; i < src_tile_shape.size(); i++)
    all_src_axes.push_back(i);
  std::vector<int> all_dst_axes;
  all_dst_axes.reserve(dst_tile_shape.size());
  for (size_t i = 0; i < dst_tile_shape.size(); i++)
    all_dst_axes.push_back(i);

  // Identify which interior axis corresponds to the reduction dimension.
  int reduce_tile_axis = plan.reduce_tile_axis;

  // Construct the reduction logic using a guarded multi-step process.
  // This allows unifying spatial and reduction loops while maintaining
  // correct Tile semantics.

  // Step 1: Accumulate (Tile-to-Tile accumulation using element-wise
  // operations). If the reduced tiled dimension has a tail tile, invalid lanes
  // must be masked on the source load before the in-tile reduce runs.
  auto make_reduce_axis_tail_predicate = [&](const Array<Var> &interior_vars) {
    ICHECK(is_dim_tiled);
    ICHECK_GE(reduce_tile_axis, 0);
    PrimExpr tile_extent = src_tile_shape[reduce_tile_axis];
    PrimExpr logical_index = loop_vars[this->dim]->var * tile_extent +
                             interior_vars[reduce_tile_axis];
    return logical_index < source_domain[this->dim];
  };

  auto make_accumulate_stmt = [&](bool use_tail_predicate) {
    Array<Var> accumulate_vars = make_interior_vars(src_tile_shape.size());
    Array<PrimExpr> accumulate_src_idx = get_src_indices(accumulate_vars);
    Array<PrimExpr> accumulate_acc_idx = get_acc_indices(accumulate_vars);

    PrimExpr src_value;
    if (use_tail_predicate) {
      PrimExpr predicate = make_reduce_axis_tail_predicate(accumulate_vars);
      // The select explicitly defines both the valid lane value and the
      // masked-off value, so the load itself does not need a BufferLoad
      // predicate.  Keeping the predicate only here avoids emitting two
      // separate mask/select sequences in SunMMIO MLIR codegen.
      src_value =
          if_then_else(predicate, BufferLoad(this->src, accumulate_src_idx),
                       this->MakeInitValue());
    } else {
      src_value = BufferLoad(this->src, accumulate_src_idx);
    }

    Stmt stmt = BufferStore(
        acc, this->MakeReduce(BufferLoad(acc, accumulate_acc_idx), src_value),
        accumulate_acc_idx);
    return wrap_interior(stmt, accumulate_vars, src_tile_shape, all_src_axes);
  };

  Stmt accumulate_stmt = make_accumulate_stmt(/*use_tail_predicate=*/false);
  if (is_dim_tiled) {
    PrimExpr reduce_tile_extent = src_tile_shape[reduce_tile_axis];
    PrimExpr reduce_extent_remainder = analyzer->Simplify(
        floormod(source_domain[this->dim], reduce_tile_extent));
    PrimExpr zero_remainder = make_zero(reduce_extent_remainder.dtype());
    bool has_reduce_axis_tail =
        !analyzer->CanProve(reduce_extent_remainder == zero_remainder);
    if (has_reduce_axis_tail) {
      PrimExpr full_reduce_tile = analyzer->Simplify(
          loop_vars[this->dim]->var * reduce_tile_extent + reduce_tile_extent <=
          source_domain[this->dim]);
      Stmt tail_accumulate_stmt =
          make_accumulate_stmt(/*use_tail_predicate=*/true);
      if (analyzer->CanProve(full_reduce_tile)) {
        // Keep the aligned fast path.
      } else if (analyzer->CanProve(!full_reduce_tile)) {
        accumulate_stmt = tail_accumulate_stmt;
      } else {
        accumulate_stmt =
            IfThenElse(full_reduce_tile, accumulate_stmt, tail_accumulate_stmt);
      }
    }
  }

  // Step 2: Init (Guarded: only run on the first step of the reduction loop)
  Array<Var> init_vars = make_interior_vars(src_tile_shape.size());
  Array<PrimExpr> init_acc_idx = get_acc_indices(init_vars);
  Stmt init_stmt;
  if (this->clear || is_dim_tiled) {
    // If we are reducing along a tiled axis, we MUST initialize 'acc' with the
    // neutral value (e.g. 0 for sum) even if clear=False. This is because 'acc'
    // is a full tile (e.g., [32, 32]), but 'dst' is already reduced (e.g.,
    // [32]). If we load 'dst' into every element of 'acc' along the reduction
    // axis, the in-tile hardware reduction will accumulate the initial value 32
    // times (resulting in "multiple additions"). The correct approach for
    // clear=False + is_dim_tiled is to accumulate the pure reduced result into
    // 'dst' during the finalize step.
    init_stmt = BufferStore(acc, this->MakeInitValue(), init_acc_idx);
  } else {
    // For non-tiled axis reduction, 'acc' shape matches 'dst' shape within the
    // tile, so we can safely load the initial value.
    Array<Var> init_dst_vars = map_dst_interior_vars(init_vars);
    Array<PrimExpr> init_dst_idx = get_dst_indices(init_dst_vars);
    PrimExpr init_val = BufferLoad(this->dst, init_dst_idx);
    init_stmt = BufferStore(acc, init_val, init_acc_idx);
  }
  init_stmt = wrap_interior(init_stmt, init_vars, src_tile_shape, all_src_axes);

  // Step 3: Finalize (Guarded: run on the last step of the reduction loop)
  Stmt finalize_stmt;
  if (is_dim_tiled) {
    // If reducing along a tiled axis, we need a specialized in-tile
    // reduction.
    std::string in_tile_reduce_type;
    if (this->type->isSum() || this->type->isAbsSum())
      in_tile_reduce_type = "sum";
    else if (this->type->isMax() || this->type->isAbsMax())
      in_tile_reduce_type = "max";
    else if (this->type->isMin())
      in_tile_reduce_type = "min";
    else
      LOG(FATAL) << "Unsupported reduce type for Sunmmio";

    // Direct write-back to Out_shared using BufferRegion to save memory and
    // IR depth.
    Array<Range> dst_ranges;
    for (int i = 0; i < dst_ndim; i++) {
      if (dst_dim_to_tile_size.count(i)) {
        int src_dim = dst_dim_to_src_dim(i);
        PrimExpr min = this->dstRegion_->region[i]->min +
                       loop_vars[src_dim]->var * dst_dim_to_tile_size[i];
        dst_ranges.push_back(
            Range::FromMinExtent(min, dst_dim_to_tile_size[i]));
      } else if (dst_ndim == src_ndim && i == this->dim) {
        dst_ranges.push_back(
            Range::FromMinExtent(this->dstRegion_->region[i]->min, 1));
      } else {
        int src_dim = dst_dim_to_src_dim(i);
        dst_ranges.push_back(Range::FromMinExtent(
            this->dstRegion_->region[i]->min + loop_vars[src_dim]->var, 1));
      }
    }
    PrimExpr dst_region = MakeRegionExpr(this->dst, dst_ranges, 2);

    Array<Range> acc_ranges;
    for (int i = 0; i < src_ndim; i++) {
      acc_ranges.push_back(
          Range::FromMinExtent(make_zero(acc->shape[i]->dtype), acc->shape[i]));
    }
    PrimExpr acc_region = MakeRegionExpr(acc, acc_ranges, 1);

    if (this->clear) {
      finalize_stmt =
          Evaluate(Call(DataType::Handle(), vector_core_in_tile_reduce(),
                        {StringImm(in_tile_reduce_type), dst_region, acc_region,
                         IntImm(DataType::Int(32), reduce_tile_axis)}));
    } else {
      ICHECK(dst_res.defined());
      Buffer dst_res_buf = dst_res.value();

      // 1. Create a region for dst_res
      Array<Range> res_ranges;
      for (int i = 0; i < dst_ndim; i++) {
        res_ranges.push_back(Range::FromMinExtent(
            make_zero(dst_res_buf->shape[i]->dtype), dst_res_buf->shape[i]));
      }
      PrimExpr res_region = MakeRegionExpr(dst_res_buf, res_ranges, 1);

      // 2. Init dst_res with 0 (not strictly necessary if vector_core
      // overwrites, but safe) Wait, vector_core_in_tile_reduce overwrites the
      // dst region. We do not need to init dst_res manually.

      // 3. vector_core_in_tile_reduce from acc to dst_res
      Stmt in_tile_reduce_stmt =
          Evaluate(Call(DataType::Handle(), vector_core_in_tile_reduce(),
                        {StringImm(in_tile_reduce_type), res_region, acc_region,
                         IntImm(DataType::Int(32), reduce_tile_axis)}));

      // 4. Final accumulation from dst_res to dst
      Array<Var> finalize_src_vars = make_interior_vars(src_tile_shape.size());
      Array<Var> finalize_dst_vars = map_dst_interior_vars(finalize_src_vars);
      Array<PrimExpr> finalize_dst_idx = get_dst_indices(finalize_dst_vars);
      Array<PrimExpr> finalize_res_idx = get_res_indices(finalize_dst_vars);
      auto make_finalize_update = [&](const PrimExpr &dst_val,
                                      const PrimExpr &res_val) {
        PrimExpr update;
        if (this->type->isSum() || this->type->isAbsSum()) {
          update = dst_val + res_val;
        } else if (this->type->isMax() || this->type->isAbsMax()) {
          update = Max(dst_val, res_val);
        } else if (this->type->isMin()) {
          update = Min(dst_val, res_val);
        } else {
          LOG(FATAL) << "Unsupported reduce type for Sunmmio accumulation";
        }
        return update;
      };
      PrimExpr update =
          make_finalize_update(BufferLoad(this->dst, finalize_dst_idx),
                               BufferLoad(dst_res_buf, finalize_res_idx));
      Stmt add_res_stmt = BufferStore(this->dst, update, finalize_dst_idx);
      if (dst_tile_shape.empty()) {
        // Reducing a rank-1 tile into a scalar dst removes all surviving tile
        // axes.  The scalar dst still lives in RSRAM and must be exposed to
        // SunMMIO codegen as a tile operation so the aligned 1D bridge can
        // perform the 64B read-modify-write.
        ICHECK_EQ(dst_ndim, 1)
            << "Sunmmio scalar reduce finalize expects a rank-1 destination "
               "buffer region";
        int aligned_elems = GetSunmmioRsramAlignmentElems(
            GetSunmmioTileProcessorConfig(T.target).rsram_align_bytes,
            this->dst->dtype);
        ICHECK_GT(aligned_elems, 0)
            << "Sunmmio scalar reduce finalize expects a positive RSRAM "
               "alignment tile extent";
        Var ki("ki");
        Array<PrimExpr> scalar_dst_idx{this->dstRegion_->region[0]->min + ki};
        PrimExpr scalar_update =
            make_finalize_update(BufferLoad(this->dst, scalar_dst_idx),
                                 BufferLoad(dst_res_buf, finalize_res_idx));
        Stmt predicated_store =
            BufferStore(this->dst, scalar_update, scalar_dst_idx, ki == 0);
        Map<String, ObjectRef> interior_annotations;
        interior_annotations.Set(attr::tile_interior, Integer(1));
        interior_annotations.Set(attr::tile_interior_axis, Integer(0));
        predicated_store =
            For(ki, 0, Integer(aligned_elems), ForKind::kVectorized,
                predicated_store, std::nullopt, interior_annotations);
        Var i0("i0");
        Map<String, ObjectRef> domain_annotations;
        domain_annotations.Set(attr::kTileDomain, Array<PrimExpr>{Integer(1)});
        domain_annotations.Set(attr::tile_tile_size,
                               Array<PrimExpr>{Integer(aligned_elems)});
        domain_annotations.Set(attr::tile_execution_axis, Integer(0));
        domain_annotations.Set(attr::tile_execution_domain_axes,
                               Array<PrimExpr>{Integer(0)});
        domain_annotations.Set(attr::tile_scope_entry, Integer(1));
        add_res_stmt = For(i0, 0, 1, ForKind::kSerial, predicated_store,
                           std::nullopt, domain_annotations);
      } else {
        add_res_stmt = wrap_interior(add_res_stmt, finalize_dst_vars,
                                     dst_tile_shape, all_dst_axes);
      }

      finalize_stmt = SeqStmt({in_tile_reduce_stmt, add_res_stmt});
    }
  } else {
    // If reduction is NOT along a tiled axis (e.g. Batch axis), 'acc' already
    // holds the result.
    Array<Var> store_vars = make_interior_vars(src_tile_shape.size());
    Array<Var> store_dst_vars = map_dst_interior_vars(store_vars);
    Array<PrimExpr> store_acc_idx = get_acc_indices(store_vars);
    Array<PrimExpr> store_dst_idx = get_dst_indices(store_dst_vars);
    Stmt store_stmt =
        BufferStore(this->dst, BufferLoad(acc, store_acc_idx), store_dst_idx);
    store_stmt =
        wrap_interior(store_stmt, store_vars, src_tile_shape, all_src_axes);
    finalize_stmt = store_stmt;
  }

  Var rv = loop_vars[this->dim]->var;
  PrimExpr r_extent = loop_vars[this->dim]->dom->extent;

  // Combine steps into a single body with guards.
  Stmt body = SeqStmt({IfThenElse(rv == 0, init_stmt), accumulate_stmt,
                       IfThenElse(rv == r_extent - 1, finalize_stmt)});

  // 6. Wrap with the final legal tile-loop contract directly.
  // IMPORTANT: For memory reuse of 'acc', the reduction loop MUST be the
  // innermost.
  Map<String, ObjectRef> root_annotations;
  root_annotations.Set(attr::kTileDomain, source_domain);
  root_annotations.Set(attr::tile_tile_size, tv_src->TileShape());
  Array<PrimExpr> execution_domain_axes;
  execution_domain_axes.reserve(plan.execution_domain_axes.size());
  std::unordered_map<int, int> src_dim_to_execution_axis;
  for (size_t axis = 0; axis < plan.execution_domain_axes.size(); ++axis) {
    int src_dim = plan.execution_domain_axes[axis];
    execution_domain_axes.push_back(Integer(src_dim));
    src_dim_to_execution_axis[src_dim] = static_cast<int>(axis);
  }
  root_annotations.Set(attr::tile_execution_domain_axes, execution_domain_axes);

  // Identify the outermost tiled loop index to set 'tile.scope_entry'.
  int outermost_tiled_idx = -1;
  for (int i = 0; i < src_ndim; i++) {
    if (i != this->dim && src_dim_to_execution_axis.count(i)) {
      outermost_tiled_idx = i;
      break;
    }
  }
  if (outermost_tiled_idx == -1 && src_dim_to_execution_axis.count(this->dim)) {
    outermost_tiled_idx = this->dim;
  }

  // Wrap the reduction loop first (innermost).
  {
    int i = this->dim;
    Map<String, ObjectRef> loop_anno;
    if (src_dim_to_execution_axis.count(i)) {
      loop_anno.Set(attr::tile_execution_axis,
                    Integer(src_dim_to_execution_axis.at(i)));
    }
    if (i == outermost_tiled_idx) {
      loop_anno.Set(attr::tile_scope_entry, Integer(1));
    }
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kSerial, body, std::nullopt, loop_anno);
  }

  // Wrap spatial loops (outer loops).
  for (int i = src_ndim - 1; i >= 0; i--) {
    if (i == this->dim)
      continue;
    Map<String, ObjectRef> loop_anno;
    if (src_dim_to_execution_axis.count(i)) {
      loop_anno.Set(attr::tile_execution_axis,
                    Integer(src_dim_to_execution_axis.at(i)));
    }
    if (i == outermost_tiled_idx) {
      loop_anno.Set(attr::tile_scope_entry, Integer(1));
    }
    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kSerial, body, std::nullopt, loop_anno);
  }

  For root = Downcast<For>(body);
  auto *root_ptr = root.CopyOnWrite();
  for (const auto &kv : root_annotations) {
    root_ptr->annotations.Set(kv.first, kv.second);
  }
  body = root;

  // Preserve the reduction-temporary roles structurally. LowerOpaqueBlock
  // transfers this per-buffer map to the corresponding Allocate annotations.
  Map<Var, Integer> reduce_temp_roles;
  reduce_temp_roles.Set(acc->data, Integer(static_cast<int>(
                                       ReduceRegisterTempRole::kAccumulator)));
  if (dst_res.defined()) {
    reduce_temp_roles.Set(
        dst_res.value()->data,
        Integer(static_cast<int>(ReduceRegisterTempRole::kResult)));
  }
  Map<String, ObjectRef> block_annotations;
  block_annotations.Set(attr::kSunmmioReduceRegisterTemp, reduce_temp_roles);

  // Finally, wrap the body in a Block so the accumulator temporaries are
  // allocated with the reduction scope.
  Array<Buffer> alloc_buffers;
  alloc_buffers.push_back(acc);
  if (dst_res.defined()) {
    alloc_buffers.push_back(dst_res.value());
  }

  body = BlockRealize({}, Bool(true),
                      Block({}, {}, {}, "reduce_tile_op", body, std::nullopt,
                            alloc_buffers, {}, block_annotations));

  return body;
}

/**
 * @brief Lower the Reduce operator to a TIR statement.
 *
 * Lowers a ReduceOpNode operating on fragment-scoped buffers into a sequence of
 * TIR statements implementing: optional initialization, thread-local reduction
 * (unrolled inner loops), inter-thread reduction via a runtime AllReduce call
 * (Hopper targets use `NamedBarrier` instead of the default
 * `SyncThreadsBarrier`), and an optional accumulation or copy back to the
 * destination buffer when a temporary clear buffer is used.
 *
 * Behavior notes:
 * - Only supports src and dst in "local.fragment" scope; otherwise it checks
 *   and aborts with "Reduce for shared memory not implemented.".
 * - Supports both 1D reductions (scalar output) and reductions along a single
 *   extra dimension; validates layout dimensionality consistency.
 * - If `clear` is set (or for sum/abssum reductions), an initial value is
 *   written to the clear buffer; for non-clearing sum/abssum a duplicate
 *   temporary buffer is allocated and accumulated back into dst after
 * reduction.
 * - Performs iterator compression for local reduction loops using `analyzer`.
 * - Detects parallel thread splitting from the normalized iterator sum and
 *   emits a call to a templated `tl::AllReduce<...>::run`
 *   via `builtin::call_extern`. For sufficiently large reducing thread counts
 *   (> 32) a workspace is allocated via T.AddWorkspace and passed to the
 *   AllReduce call.
 * - The final body is wrapped in parallel loops over the destination spatial
 *   dimensions and partitioned by the lowering thread variable. If a temporary
 *   clear buffer is used, it is allocated for the body.
 *
 * @param T Lowering context providing buffer and layout maps, thread bounds,
 *          target information, thread variable, and workspace allocation
 * helper.
 * @param analyzer Analyzer used for iterator compression and arithmetic
 * normalization.
 * @return Stmt Lowered TIR statement implementing the reduction.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_buffer = [&](const Buffer &buf) {
    if (T.buffer_remap.count(buf))
      return T.buffer_remap[buf];
    return buf;
  };

  auto src_scope = this->src.scope();
  auto dst_scope = this->dst.scope();

  if (TargetIsSunmmio(T.target)) {
    ValidateSunmmioReduceDType(this->type, this->src->dtype, this->dst->dtype);
    return MakeSunmmioTileReduce(T, analyzer);
  }

  if (src_scope == "local.fragment" && dst_scope == "local.fragment") {

    auto src_buffer = get_buffer(this->src);
    auto dst_buffer = get_buffer(this->dst);
    auto src_layout = T.layout_map[this->src].as<Fragment>().value();
    auto dst_layout = T.layout_map[this->dst].as<Fragment>().value();
    auto red_layout = ComputeReducerLayout(src_layout, dim);
    auto src_dim = src_layout->InputDim();
    auto dst_dim = dst_layout->InputDim();

    auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

    if (is_1d_reduce) {
      ICHECK(is_one(dst_layout->OutputShape().back()))
          << "Reduce for scalar not implemented.";
    } else {
      ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
    }

    Array<IterVar> dst_vars;
    for (size_t i = 0; i < dst_dim; ++i) {
      Var var = Var(std::string{char('i' + i)});
      dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                                 IterVarType::kDataPar));
    }

    Array<IterVar> src_vars;
    if (!is_1d_reduce) {
      src_vars = dst_vars;
    }
    Range reduce_dom(0, src_layout->InputShape()[this->dim]);
    IterVar reduce_iv(reduce_dom, Var("rv"), IterVarType::kDataPar);
    src_vars.insert(src_vars.begin() + this->dim, reduce_iv);

    auto src_indices = src_layout->Forward(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto dst_indices = dst_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto red_indices = red_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

    Array<Stmt> stmts;

    auto require_init = this->clear;
    if (this->type->isSum() || this->type->isAbsSum() ||
        this->type->isBitAnd() || this->type->isBitOr() ||
        this->type->isBitXor()) {
      require_init = true;
    }

    auto clear_buffer = dst_buffer;
    auto need_duplicate = false;
    auto need_update = false;
    if ((this->type->isSum() || this->type->isAbsSum()) && !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if (this->type->isBitAnd() && !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if ((this->type->isBitOr() || this->type->isBitXor()) &&
               !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if ((this->type->isMax() || this->type->isMin() ||
                this->type->isAbsMax()) &&
               !this->clear) {
      need_duplicate = true;
      need_update = true;
    }

    // red_layout should always contain dst_layout
    // if we can prove they are the same, no need to duplicate buffer
    // otherwise, red_layout contains more replicated dimensions than dst_layout
    if (!analyzer->CanProve(dst_layout->ReplicateExtent() ==
                            red_layout->ReplicateExtent())) {
      need_duplicate = true;
    }
    ICHECK(!analyzer->CanProve(dst_layout->ReplicateExtent() >
                               red_layout->ReplicateExtent()))
        << "Inconsistent layouts between src and dst in ReduceOp: "
        << "dst_layout=" << dst_layout << "red_layout=" << red_layout;

    if (need_duplicate) {
      // Create a new buffer with same shape and dtype as dst_buffer
      clear_buffer = decl_buffer(red_layout->OutputShape(), dst_buffer->dtype,
                                 dst_buffer->name + "_clear",
                                 GetPtrStorageScope(dst_buffer->data));
    }
    // make reduce-init stmt
    // For max/min/absmax with clear=false and need_duplicate, we still need to
    // initialize the temporary buffer with identity values since the original
    // dst values will be combined later via need_update
    if (require_init ||
        (need_duplicate && (this->type->isMax() || this->type->isMin() ||
                            this->type->isAbsMax()))) {
      stmts.push_back(
          BufferStore(clear_buffer, this->MakeInitValue(), red_indices));
    }

    // make thread-local reduce
    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
      auto [expr, var] = CompressIterator(src_indices[i], src_vars,
                                          src_vars[this->dim]->var, analyzer);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }

    Stmt reduce_local = BufferStore(
        clear_buffer,
        this->MakeReduce(BufferLoad(clear_buffer, red_indices),
                         BufferLoad(src_buffer, src_indice_compressed)),
        red_indices);

    for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
      reduce_local =
          For(src_var_compressed[i]->var, 0, src_var_compressed[i]->dom->extent,
              ForKind::kUnrolled, reduce_local, std::nullopt,
              {{tir::attr::pragma_unroll_explicit, Bool(false)}});
    }
    stmts.push_back(reduce_local);

    auto src_thread = src_layout->ForwardThread(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
    auto iter_sum =
        arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);
    for (const auto &iter_split : iter_sum->args) {
      auto mark = iter_split->source->source.as<Var>();
      ICHECK(mark) << "Not a normalized iterator: " << iter_split->source;
      if (mark.value().same_as(src_vars[this->dim]->var)) {
        // `scale` is the stride of participating threads in the thread index
        // space.  When the thread-to-data mapping for the reduce dimension is
        // normalized as  threadIdx = source * scale + ...,
        //   * scale == 1  means threads are contiguous (0, 1, 2, ...),
        //   * scale  > 1  means threads are interleaved (0, scale, 2*scale,
        //     ...).
        // Both cases use the recursive XOR-butterfly reduce.
        // `extent` is the number of distinct thread positions along the reduce
        // dimension, so reducing_threads = extent * scale covers the full
        // thread index range that participates in the reduction.
        auto scale = as_const_int(iter_split->scale);
        auto extent = as_const_int(iter_split->extent);
        ICHECK(scale != nullptr && extent != nullptr);
        if (*extent == 1)
          continue;

        int reducing_threads = (*extent) * (*scale);
        std::stringstream ss;

        auto thread_offset = T.thread_bounds->min;
        if (TargetHasSMVersionGE(T.target, 90)) {
          auto all_threads = T.thread_bounds->extent;
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ", tl::NamedBarrier<" << all_threads << ">>::run";
        } else {
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ">::run";
        }
        Array<PrimExpr> thread_reduce_args = {
            StringImm(ss.str()), BufferLoad(clear_buffer, red_indices)};
        // The butterfly reduce path needs one shared-memory slot per
        // thread in the block.
        if (reducing_threads > 32) {
          int workspace_size =
              static_cast<int>(*as_const_int(T.thread_bounds->extent));
          PrimExpr workspace =
              T.AddWorkspace(workspace_size, clear_buffer->dtype);
          thread_reduce_args.push_back(workspace);
        }
        auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                         thread_reduce_args);
        stmts.push_back(BufferStore(clear_buffer, call, red_indices));
      }
    }

    // Layout status in the loop:
    //     clear_buffer: red_layout
    //     dst_buffer:   dst_layout
    //     loop_layout:  red_layout
    // At each step of the loop, we do reduction on
    // `clear_buffer[red_layout(loop_idx)]`
    //   and then transfer it to `dst_buffer[dst_layout(loop_idx)]`
    // However, since the red_layout is larger than dst_layout, not all write
    // operations are valid We need to add predicate to guard the write
    // operations
    PrimExpr predicate = Bool(true);
    {
      // dst_indices is the same as loop_indices
      auto dst_th_indices = dst_indices;
      dst_th_indices.push_back(T.thread_var);
      // 1. compute loop_idx based on thread: [dst_indices, T.thread_var] =>
      // [loop_indices]
      auto inv = dst_layout->Inverse()->Forward(dst_th_indices);
      inv.pop_back(); // remove replicate var
      // 2. ensure computed loop_idx maps back to the same [loop_indices]
      for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
        predicate = predicate && (inv[i] == dst_vars[i]->var);
      }
      // 3. simplify predicate
      predicate = analyzer->Simplify(predicate);
    }
    if (need_duplicate) {
      PrimExpr update;
      if (need_update) {
        PrimExpr src_val = BufferLoad(clear_buffer, red_indices);
        PrimExpr dst_val = BufferLoad(dst_buffer, dst_indices);
        if (this->type->isAbsSum() || this->type->isAbsMax()) {
          dst_val = tvm::abs(dst_val);
        }
        if (this->type->isSum() || this->type->isAbsSum()) {
          update = dst_val + src_val;
        } else if (this->type->isBitAnd()) {
          update = this->clear ? src_val : bitwise_and(dst_val, src_val);
        } else if (this->type->isBitOr()) {
          update = bitwise_or(dst_val, src_val);
        } else if (this->type->isBitXor()) {
          update = bitwise_xor(dst_val, src_val);
        } else if (this->type->isMax() || this->type->isAbsMax()) {
          update = Max(dst_val, src_val);
        } else if (this->type->isMin()) {
          update = Min(dst_val, src_val);
        } else {
          LOG(FATAL) << "Unsupported reduce type: " << this->type->type;
        }
      } else {
        update = BufferLoad(clear_buffer, red_indices);
      }
      auto store = BufferStore(dst_buffer, update, dst_indices);
      if (analyzer->CanProve(predicate)) {
        stmts.push_back(store);
      } else {
        stmts.push_back(IfThenElse(predicate, store));
      }
    }

    auto body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
    for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
      body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                 ForKind::kParallel, body);
    }

    if (dst_layout->InputDim() > 0) {
      body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                           red_layout);
      body = PragmaUnrollLoop(Downcast<For>(body));
    } else {
      auto guard = (T.thread_var == T.thread_bounds->min);
      body = IfThenElse(guard, body);
    }

    if (need_duplicate) {
      body = Allocate(clear_buffer->data, clear_buffer->dtype,
                      clear_buffer->shape, const_true(), body);
    }
    return body;
  }

  LOG(FATAL) << "Reduce for buffers in scope (" << src_scope << ", "
             << dst_scope << ") is not implemented.";
  return Stmt();
}

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (TargetIsSunmmio(T.target)) {
    ValidateSunmmioReduceDType(this->type, this->src->dtype, this->dst->dtype);
  }

  // Sunmmio RSRAM reduce: derive the dst layout from the src. A surviving
  // blocked (ZZ) structure is kept via DeriveLayoutLike; any row-major result
  // is alignment-padded so each row stays RSRAM-aligned.
  if (src.scope() == kSunmmioScopeRSRAM && dst.scope() == kSunmmioScopeRSRAM) {
    LayoutMap result;
    // Always propose when src has a layout — TryAssign handles priority.
    // No guard on !T.layout_map.count(dst): level-based priority in the
    // pass will accept or reject appropriately.
    if (T.layout_map.count(src)) {
      auto *src_cute = T.layout_map[src].as<CuteLayoutNode>();
      bool reduces_blocked_dim =
          src_cute && src_cute->GetDimLevels()[dim].IntValue() > 1;
      int align_bytes =
          GetSunmmioTileProcessorConfig(T.target).rsram_align_bytes;
      // True if `layout` has any blocked dim (dim_levels > 1), i.e. already-
      // aligned tiling; a plain row-major (all dim_levels == 1) returns false.
      auto HasBlockedDim = [](const Layout &layout) {
        if (const auto *cute = layout.as<CuteLayoutNode>()) {
          for (const Integer &lvl : cute->GetDimLevels()) {
            if (lvl.IntValue() > 1)
              return true;
          }
        }
        return false;
      };
      // Keep DeriveLayoutLike's result only when it preserves blocks (already
      // block-aligned); any row-major outcome (blocked-axis reduce, chained
      // unblocked src, failed derivation) goes through MakeAlignedRowMajor.
      Layout dst_layout;
      if (!reduces_blocked_dim) {
        auto derived = DeriveLayoutLike(T.layout_map[src], dst->shape);
        if (derived.defined() && HasBlockedDim(derived.value()))
          dst_layout = derived.value();
      }
      if (!dst_layout.defined())
        dst_layout =
            sunmmio::MakeAlignedRowMajor(dst->shape, dst->dtype, align_bytes);
      result.Set(dst, dst_layout);
    }
    return result;
  }

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();
    auto reducer_layout = ComputeReducerLayout(src_layout, this->dim);

    if (!T.layout_map.count(dst)) {
      return {{dst, reducer_layout}};
    }

    auto orig_dst_layout = T.layout_map.Get(dst).value().as<Fragment>().value();
    ICHECK(reducer_layout->InputDim() == orig_dst_layout->InputDim());

    auto indices = InputPlaceholders(reducer_layout->InputDim());
    arith::Analyzer analyzer;
    for (size_t i = 0; i < indices.size(); i++) {
      analyzer.Bind(Downcast<Var>(indices[i]),
                    Range(0, reducer_layout->InputShape()[i]));
    }
    if (!ProveFragmentContains(orig_dst_layout, reducer_layout, indices,
                               indices, analyzer)) {
      std::ostringstream oss;
      oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
          << src << "\n"
          << "src_layout = " << src_layout << "\n"
          << "reducer_layout = " << reducer_layout << "\n"
          << "orig_dst_layout = " << orig_dst_layout << "\n"
          << "You may need to use a shared memory to transform the "
             "layout";
      throw LayoutConflictException(oss.str());
    }
  }
  return {};
}

TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Normalize "Buffer" to BufferRegion. Use the shape of the buffer as the
// ranges.
static BufferRegion ConvertBufferToBufferRegion(const Buffer &buf) {
  Array<Range> ranges;
  for (PrimExpr extent : buf->shape) {
    ranges.push_back(Range(IntImm(extent->dtype, 0), extent));
  }
  return BufferRegion(buf, ranges);
}

CumSumOp::CumSumOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  /// CumSum constructor arguments:
  /// - src: input buffer
  /// - dst: output buffer
  /// - dim: dimension to cumsum
  /// - reverse: whether to cumsum in reverse order
  CHECK_EQ(args.size(), 4);
  ObjectPtr<CumSumOpNode> node = tvm::ffi::make_object<CumSumOpNode>();
  // node->src = vmap[GetVarFromAccessPtr(args[0])];
  // node->dst = vmap[GetVarFromAccessPtr(args[1])];
  node->srcRegion_ = NormalizeToBufferRegion(args[0]);
  node->dstRegion_ = NormalizeToBufferRegion(args[1]);
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  node->dim = args[2].as<IntImm>().value()->value;
  node->reverse = args[3].as<Bool>().value();
  CHECK_LT(node->dim, static_cast<int>(node->src->shape.size()))
      << "The dim of cumsum should be less than the number of dimensions. Got "
         "dim="
      << node->dim << ", but src has " << node->src->shape.size() << " dims.";

  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (IsFragmentBuffer(this->src) && IsFragmentBuffer(this->dst)) {
    LOG(FATAL) << "CumSum for fragment not implemented, please raise an issue "
                  "if you need this feature.";
  } else if (this->src.scope() == "shared.dyn" ||
             this->src.scope() == "shared") {
    ICHECK(this->dst.scope() == "shared.dyn" || this->dst.scope() == "shared");
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    Array<PrimExpr> args;

    // Build access pointers from regions locally
    PrimExpr srcPtr = MakeAccessPtrFromRegion(srcRegion_, 1);
    PrimExpr dstPtr = MakeAccessPtrFromRegion(dstRegion_, 2);

    // Use region extents instead of buffer shape for correct slice handling
    Array<PrimExpr> src_extents;
    for (const auto &range : srcRegion_->region) {
      src_extents.push_back(range->extent);
    }
    int ndim = static_cast<int>(src_extents.size());

    if (ndim == 1) {
      ICHECK_EQ(dim, 0) << "Cumulative sum over a 1D buffer only supports dim "
                           "= 0.";
      ss << "tl::CumSum1D<" << threads << ", " << (reverse ? "true" : "false")
         << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0]};
    } else if (ndim == 2) {
      ss << "tl::CumSum2D<" << threads << ", " << dim << ", "
         << (reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0],
              src_extents[1]};
    } else {
      LOG(FATAL) << "CumSum currently supports only 1D or 2D buffers, got "
                 << ndim << "D.";
    }
    return Evaluate(Call(dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower cumsum for " << this->src.scope() << " and "
                  << this->dst.scope();
  }

  return Stmt();
}

LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  // Only infer layout in strict mode
  if (level != InferLevel::kStrict) {
    return {};
  }

  LayoutMap result_map;

  auto make_linear_layout = [](const Buffer &buf) -> Layout {
    return makeLinearLayout(buf->shape);
  };

  auto check_or_set_linear_layout = [&](const Buffer &buf) {
    if (!IsSharedBuffer(buf))
      return;

    Layout linear_layout = make_linear_layout(buf);
    if (T.layout_map.count(buf)) {
      // Check if existing layout is linear
      Layout existing = T.layout_map.Get(buf).value().as<Layout>().value();
      ICHECK(StructuralEqual()(existing, linear_layout))
          << "CumSum requires linear layout for shared buffer " << buf->name
          << ", but got non-linear layout.";
    } else {
      result_map.Set(buf, linear_layout);
    }
  };

  check_or_set_linear_layout(src);
  check_or_set_linear_layout(dst);

  return result_map;
}

TIR_REGISTER_TL_TILE_OP(CumSumOp, cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReduceOpNode::RegisterReflection();
  CumSumOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
