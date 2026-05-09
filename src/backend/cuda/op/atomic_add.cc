/*!
 * \file tl/backend/cuda/op/atomic_add.cc
 * \brief CUDA implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"

#include "backend/cuda/op/copy.h"
#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

namespace {

Layout ComputeLinearLayout(const Buffer &shared_tensor) {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }
  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

bool UseTMA(const AtomicAddNode &op) {
  if (auto val = op.annotations.Get("use_tma")) {
    if (auto int_val = val->as<IntImmNode>()) {
      if (int_val->value != 0) {
        ICHECK(!op.src_value.defined())
            << "TMA is not supported when using TiledAtomicAdd with PrimExpr "
               "as value.";
        return true;
      }
    }
  }
  return false;
}

Array<IterVar> MakeIterVars(const AtomicAddNode &op) {
  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < op.dst_range.size(); i++) {
    if (is_one(op.dst_range[i]->extent)) {
      continue;
    }
    Var var = Var(std::string{char('i' + idx)}, op.dst_range[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, op.dst_range[i]->extent), var, IterVarType::kDataPar});
  }

  if (loop_vars.empty()) {
    Var var = Var("i");
    loop_vars.push_back({Range(0, 1), var, IterVarType::kDataPar});
  }

  return loop_vars;
}

Array<PrimExpr> MakeIndices(const AtomicAddNode &op, const Array<IterVar> &ivs,
                            int src_dst) {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? op.src_range : op.dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) {
      indices.push_back(ranges[i]->min);
    } else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }

  ICHECK(idx == ivs.size() || (idx == 0 && ivs.size() == 1))
      << "Unmatched indices: idx = " << idx << ", ivs.size() = " << ivs.size()
      << ", dst name = " << op.dst->name;
  return indices;
}

PrimExpr MakePredicate(const AtomicAddNode &op, arith::Analyzer *analyzer,
                       const Array<IterVar> &ivs, Array<PrimExpr> extents,
                       int src_dst) {
  Array<Range> ranges = src_dst == 0 ? op.src_range : op.dst_range;
  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent)) {
      continue;
    }
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty()) {
    return {};
  }
  PrimExpr cond = cond_list[0];
  for (size_t i = 1; i < cond_list.size(); i++) {
    cond = And(cond, cond_list[i]);
  }
  return cond;
}

For MakeSIMTLoop(const AtomicAddNode &op, arith::Analyzer *analyzer) {
  Array<IterVar> loop_vars = MakeIterVars(op);
  ICHECK(!loop_vars.empty()) << "MakeIterVars in AtomicOp should not return "
                                "empty vars (at least 1 var)";

  for (const auto &iv : loop_vars) {
    analyzer->Bind(iv->var, iv->dom);
  }

  ICHECK(loop_vars.size() <= op.dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << op.dst_range.size()
      << ", dst = " << op.dst->name;

  Array<PrimExpr> dst_indices = MakeIndices(op, loop_vars, 1);
  Array<PrimExpr> new_args;

  PrimExpr dst_predicate =
      MakePredicate(op, analyzer, loop_vars, op.dst->shape, 1);

  PrimExpr src_value_arg;

  if (!op.src_value.defined()) {
    ICHECK(loop_vars.size() <= op.src_range.size())
        << "loop_vars.size() = " << loop_vars.size()
        << ", src_range.size() = " << op.src_range.size()
        << ", src = " << op.src->name << ", dst = " << op.dst->name;

    Array<PrimExpr> src_indices = MakeIndices(op, loop_vars, 0);
    PrimExpr src_predicate =
        MakePredicate(op, analyzer, loop_vars, op.src->shape, 0);
    src_value_arg = BufferLoad(op.src, src_indices);
  } else {
    src_value_arg = op.src_value;
  }

  if (src_value_arg->dtype != op.dst->dtype) {
    src_value_arg = Cast(op.dst->dtype, src_value_arg);
  }

  DataType idx_dtype =
      dst_indices.empty() ? DataType::Int(32) : dst_indices[0].dtype();
  PrimExpr dst_ptr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(op.dst, dst_indices), make_const(idx_dtype, 1),
            make_const(DataType::Int(32), 3)});

  new_args.push_back(dst_ptr);
  new_args.push_back(src_value_arg);
  new_args.push_back(op.GetMemoryOrder());

  auto annotations = op.annotations;
  annotations.erase("use_tma");
  Call atomicadd_call =
      tvm::tir::Call(op.dst->dtype, op.GetElemOp(), new_args, annotations);

  Stmt body = tvm::tir::Evaluate(atomicadd_call);

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

LayoutMap InferSIMTLayout(const AtomicAddNode &op, const LayoutInferArgs &T,
                          InferLevel) {
  if (IsFragmentBuffer(op.src) && IsFragmentBuffer(op.dst)) {
    if (T.layout_map.count(op.src) && T.layout_map.count(op.dst)) {
      Layout src_layout = T.layout_map.at(op.src);
      Layout dst_layout = T.layout_map.at(op.dst);
      ICHECK(StructuralEqual()(src_layout, dst_layout))
          << "AtomicAdd requires src and dst to have the same layout, but got "
          << "src layout: " << src_layout << ", dst layout: " << dst_layout
          << " for src buffer: " << op.src->name
          << ", dst buffer: " << op.dst->name;
    }
  }
  return {};
}

} // namespace

struct AtomicAdd {
  static Stmt LowerSIMT(const AtomicAddNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
    auto simt_loop = MakeSIMTLoop(op, analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(fused_loop, loop_layout, T.thread_var, analyzer,
                             T.layout_map, par_op->GetPredicate(T.thread_var));
  }

  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &T, InferLevel level) {
    if (!UseTMA(op)) {
      return InferSIMTLayout(op, T, level);
    }

    Map<Buffer, Layout> result_map;
    Buffer shared_tensor = op.src;
    Array<Range> shared_range = op.src_range;
    bool is_tma_1d = shared_range.size() == 1;
    if (is_tma_1d) {
      return result_map;
    }

    if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
      int dim = shared_tensor->shape.size();
      const int64_t mat_stride = *as_const_int(shared_tensor->shape[dim - 2]);
      const int64_t mat_continuous =
          *as_const_int(shared_tensor->shape[dim - 1]);
      Layout swizzle_layout_2d =
          makeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                 shared_tensor->dtype.bits(), /*k_inner=*/true);
      if (StructuralEqual()(swizzle_layout_2d, makeLinearLayout(Array<PrimExpr>{
                                                   Integer(mat_stride),
                                                   Integer(mat_continuous)}))) {
        result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
      } else {
        result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                          swizzle_layout_2d, shared_tensor));
      }
    }

    return result_map;
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (!UseTMA(op)) {
      return LowerSIMT(op, T, analyzer);
    }

    // For AtomicAdd with TMA: src is shared memory, dst is global memory.
    Buffer shared_tensor = op.src;
    Buffer global_tensor = op.dst;
    Array<Range> shared_range = op.src_range;
    Array<Range> global_range = op.dst_range;

    TMADesc desc;
    desc.rank = global_tensor->shape.size();
    ICHECK(desc.rank >= 1 && desc.rank <= 5)
        << "TMA reduce only supports 1-5 dimensions, got " << desc.rank;

    ICHECK(global_tensor->dtype == shared_tensor->dtype)
        << "AtomicAdd between buffer " << shared_tensor->name << " and "
        << global_tensor->name << " with different data type "
        << shared_tensor->dtype << " and " << global_tensor->dtype;

    desc.data_type = to_CUtensorMapDataType(global_tensor->dtype);
    desc.global_addr = global_tensor->data;
    desc.global_shape = ReverseArray(global_tensor->shape);
    Array<PrimExpr> global_coords =
        ReverseArray(global_range.Map([](Range r) { return r->min; }));

    if (!global_tensor->strides.empty()) {
      desc.global_stride = ReverseArray(global_tensor->strides);
    } else {
      PrimExpr stride = 1;
      desc.global_stride.reserve(desc.rank);
      for (size_t i = 0; i < desc.rank; i++) {
        desc.global_stride.push_back(stride);
        stride *= desc.global_shape[i];
      }
    }
    desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
      return cast(DataType::Int(64), e) * global_tensor->dtype.bytes();
    });

    desc.smem_box =
        ReverseArray(global_range.Map([](Range r) { return r->extent; }));
    desc.smem_stride = Array<PrimExpr>(desc.rank, PrimExpr(1));
    desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
    desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    auto linear_layout = makeLinearLayout(shared_tensor->shape);
    Buffer shared_tensor_unmapped = shared_tensor;
    desc.interleave = static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
    Layout shared_layout;
    if (T.layout_map.count(shared_tensor)) {
      shared_layout = T.layout_map.at(shared_tensor);
      ICHECK(T.buffer_remap.count(shared_tensor))
          << "shared_tensor: " << shared_tensor->name
          << " not found in buffer_remap";
      shared_tensor = T.buffer_remap.at(shared_tensor);
    }
    if (!shared_layout.defined()) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
    } else if (StructuralEqual()(shared_layout, linear_layout)) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
    } else {
      ICHECK(shared_layout->InputDim() >= 2) << "Cannot detect TMA layout.";
      const int ndim = static_cast<int>(shared_layout->InputDim());
      auto stride = as_const_int(shared_layout->InputShape()[ndim - 2]);
      auto continuous = as_const_int(shared_layout->InputShape()[ndim - 1]);
      ICHECK(stride != nullptr && continuous != nullptr);
      if (StructuralEqual()(shared_layout, makeQuarterBankSwizzleLayout(
                                               shared_tensor_unmapped))) {
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
      } else if (StructuralEqual()(
                     shared_layout,
                     makeHalfBankSwizzleLayout(shared_tensor_unmapped))) {
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
      } else if (StructuralEqual()(
                     shared_layout,
                     makeFullBankSwizzleLayout(shared_tensor_unmapped))) {
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
      } else if (StructuralEqual()(
                     shared_layout,
                     makeGemmABLayoutPadded(*stride, *continuous,
                                            shared_tensor->dtype.bits()))) {
        DLOG(WARNING)
            << "AtomicAdd TMA cannot support a padded layout for src: "
            << op.src->name << ", dst: " << op.dst->name
            << " fallback to none swizzle";
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
      } else {
        DLOG(WARNING) << "AtomicAdd TMA unsupported swizzle layout for src: "
                      << op.src->name << ", dst: " << op.dst->name
                      << " fallback to none swizzle";
        desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
      }
    }

    auto inner_box_dim = as_const_int(desc.smem_box[0]);
    ICHECK(inner_box_dim != nullptr)
        << "inner_box_dim must be a constant integer for TMA atomic add";
    int instruction_dim = *inner_box_dim;
    if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
      instruction_dim = 64 / shared_tensor->dtype.bytes();
    } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
      instruction_dim = 128 / shared_tensor->dtype.bytes();
    }
    if (instruction_dim > 256) {
      ICHECK((*inner_box_dim) % 256 == 0)
          << "inner_box_dim: " << *inner_box_dim << " is not divisible by 256";
      instruction_dim = 256;
    }
    ICHECK((*inner_box_dim) % instruction_dim == 0)
        << "inner_box_dim: " << *inner_box_dim
        << " is not divisible by instruction_dim: " << instruction_dim;
    desc.smem_box.Set(0, PrimExpr(instruction_dim));

    int inner_box_dim_bytes = instruction_dim * shared_tensor->dtype.bytes();
    struct SwizzleCheck {
      int swizzle;
      int max_dim;
    };
    static const std::vector<SwizzleCheck> swizzle_checks = {
        {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B), 32},
        {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B), 64},
        {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B), 128},
    };
    for (const auto &check : swizzle_checks) {
      if (desc.swizzle == check.swizzle &&
          inner_box_dim_bytes > check.max_dim) {
        DLOG(WARNING) << "AtomicAdd TMA cannot support swizzled layout with "
                         "inner_box_dim_bytes > "
                      << check.max_dim;
      }
    }

    Array<PrimExpr> shared_indices;
    for (auto r : shared_range) {
      shared_indices.push_back(r->min);
    }
    std::vector<PrimExpr> shared_strides;
    PrimExpr shared_stride = 1;
    for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
      auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
      shared_strides.insert(shared_strides.begin(), shared_stride);
      shared_stride *= s;
    }
    PrimExpr shared_offset = 0;
    for (size_t i = 0; i < shared_indices.size(); i++) {
      shared_offset += shared_indices[i] * shared_strides[i];
    }

    Call create_descriptor = Call(DataType::Handle(), create_tma_descriptor(),
                                  desc.EncodeCallArgs());

    PrimExpr total_elements = 1;
    for (auto e : desc.smem_box) {
      total_elements *= e;
    }

    auto op_annotations = op.annotations;
    op_annotations.erase("use_tma");

    Stmt tma_reduce;
    if ((*inner_box_dim) != instruction_dim) {
      Var loop_var("i");
      int loop_extent = (*inner_box_dim) / instruction_dim;

      Array<PrimExpr> args;
      args.reserve(desc.rank + 4);
      args.push_back(create_descriptor);
      PrimExpr shared_addr = shared_tensor.access_ptr(
          1, DataType::Handle(), 1, shared_offset + total_elements * loop_var,
          total_elements);
      args.push_back(shared_addr);
      Array<PrimExpr> loop_global_coords = global_coords;
      loop_global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
      for (auto coord : loop_global_coords) {
        args.push_back(coord);
      }
      args.push_back(1);
      args.push_back(0);
      tma_reduce = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                       Evaluate(Call(DataType::Handle(), tma_store(), args,
                                     op_annotations)));
    } else {
      Array<PrimExpr> args;
      args.reserve(desc.rank + 4);
      args.push_back(create_descriptor);
      PrimExpr shared_addr = shared_tensor.access_ptr(
          1, DataType::Handle(), 1, shared_offset, total_elements);
      args.push_back(shared_addr);
      for (auto coord : global_coords) {
        args.push_back(coord);
      }
      args.push_back(1);
      args.push_back(0);
      tma_reduce =
          Evaluate(Call(DataType::Handle(), tma_store(), args, op_annotations));
    }

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_reduce);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                {IntImm(DataType::Int(32), 0)})));
    return IfThenElse(EQ(T.thread_var, T.thread_bounds->min),
                      SeqStmt(std::move(seq)));
  }
};

} // namespace cuda

namespace {

bool MatchCudaAtomicAddTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "cuda.AtomicAdd",
      MatchCudaAtomicAddTarget,
      cuda::AtomicAdd::InferLayout,
      cuda::AtomicAdd::Lower,
  });
  return true;
}

const bool cuda_atomic_add_registered = RegisterCudaAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm
