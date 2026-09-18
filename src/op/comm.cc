/*!
 * \file tl/op/comm.cc
 * \brief Implementation of Inter-core Communication Operators
 */

#include "comm.h"

#include <cstdint>
#include <tvm/tir/op.h>

#include "../layout/cute_layout.h"
#include "../target/sunmmio_utils.h"
#include "../target/utils.h"
#include "reduce.h"
#include "utils.h"

namespace tvm {
namespace tl {

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)
TIR_DEFINE_TL_BUILTIN(broadcast_)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
// src_region, dst_region,
// direction(0: horizontal/row, 1: vertical/col),
// mask(i64 direction-local bitmask of receiving cores),
// src_offset_byte,
// optional src_core

using namespace tir;

namespace {

PrimExpr I32Imm(int64_t value) { return IntImm(DataType::Int(32), value); }

PrimExpr I64Imm(int64_t value) { return IntImm(DataType::Int(64), value); }

PrimExpr AsI32(PrimExpr value) {
  if (value.dtype() == DataType::Int(32)) {
    return value;
  }
  return Cast(DataType::Int(32), value);
}

PrimExpr AsI64(PrimExpr value) {
  if (value.dtype() == DataType::Int(64)) {
    return value;
  }
  return Cast(DataType::Int(64), value);
}

Stmt AssertPutDistinctCores(PrimExpr src_core, PrimExpr dst_core, Stmt body,
                            arith::Analyzer *analyzer) {
  PrimExpr distinct = AsI32(src_core) != AsI32(dst_core);
  if (analyzer) {
    distinct = analyzer->Simplify(distinct);
  }
  if (const auto *imm = distinct.as<IntImmNode>()) {
    ICHECK_NE(imm->value, 0)
        << "T.comm.put requires src_core and dst_core to be different";
    return body;
  }
  return AssertStmt(distinct,
                    StringImm("T.comm.put requires src_core and dst_core to "
                              "be different"),
                    body);
}

PrimExpr MakeFullLocalMask(int axis_len) {
  ICHECK_GE(axis_len, 0);
  ICHECK_LE(axis_len, 64)
      << "tl.broadcast_ local mask currently supports at most 64 receivers";
  uint64_t mask =
      axis_len == 64 ? ~uint64_t{0} : ((uint64_t{1} << axis_len) - 1);
  return I64Imm(static_cast<int64_t>(mask));
}

PrimExpr MakeLocalSingleBit(PrimExpr local_index) {
  if (const auto *imm = local_index.as<IntImmNode>()) {
    ICHECK_GE(imm->value, 0);
    ICHECK_LT(imm->value, 64)
        << "tl.broadcast_ local mask currently supports indices in [0, 64)";
    return I64Imm(static_cast<int64_t>(uint64_t{1} << imm->value));
  }
  return I64Imm(1) << AsI64(local_index);
}

PrimExpr MakeHorizontalMask(int mesh_ncol) {
  return MakeFullLocalMask(mesh_ncol);
}

PrimExpr MakeVerticalMask(int mesh_nrow) {
  return MakeFullLocalMask(mesh_nrow);
}

void AppendBroadcastArgs(Array<PrimExpr> *args, PrimExpr src_region,
                         PrimExpr dst_region, int direction, PrimExpr mask,
                         PrimExpr src_offset_byte, PrimExpr src_core) {
  args->push_back(src_region);
  args->push_back(dst_region);
  args->push_back(I32Imm(direction));
  args->push_back(mask);
  args->push_back(src_offset_byte);
  args->push_back(src_core);
}

Stmt MakeBroadcastLeaf(PrimExpr src_region, PrimExpr dst_region, int direction,
                       PrimExpr mask, PrimExpr src_offset_byte) {
  Array<PrimExpr> args;
  args.push_back(src_region);
  args.push_back(dst_region);
  args.push_back(I32Imm(direction));
  args.push_back(mask);
  args.push_back(src_offset_byte);
  return Evaluate(Call(DataType::Handle(), broadcast_(), args));
}

} // namespace

/*!
 * \brief Sunmmio SRAM layout inference for symmetric comm ops.
 *
 * Propagates layout between src and dst via DeriveLayoutLike.
 * Always proposes — TryAssign handles priority and conflict detection.
 *
 * No validation here: the level-based priority system in TryAssign
 * determines whether proposals are accepted or rejected.  When a buffer
 * is immutable (T.annotate_layout / DRAM metadata), TryAssign catches
 * any kStrict conflict.  kCommon proposals against immutable entries
 * are silently rejected — they arise from BFS noise (e.g., derived from
 * kFree defaults) and are not real constraints.
 */
static LayoutMap SunmmioCommInferLayout(const LayoutInferArgs &T,
                                        const Buffer &src, const Buffer &dst,
                                        InferLevel level) {
  // Comm ops are propagation-only — no hard constraints at kStrict.
  if (level >= InferLevel::kStrict)
    return {};

  LayoutMap result;
  bool src_has = T.layout_map.count(src);
  bool dst_has = T.layout_map.count(dst);

  // ZN/MXZNN WSRAM dst: its src is staged as ZZ/MXZNZ (the transfer does
  // ZZ->ZN or MXZNZ->MXZNN), so infer the matching source layout. dst is
  // already determined (from GEMM); propose nothing for it.
  if (dst_has && dst.scope() == kSunmmioScopeWSRAM &&
      !sunmmio::IsZZLike(T.layout_map[dst])) {
    int rank = static_cast<int>(src->shape.size());
    if (IsSunmmioSramScope(src.scope()) && rank >= 2) {
      Array<Integer> axes{Integer(rank - 2), Integer(rank - 1)};
      Layout src_layout = sunmmio::IsMXDType(src->dtype)
                              ? sunmmio::MakeMXZNZ(src->shape, axes, src->dtype)
                              : sunmmio::MakeZZ(src->shape, axes,
                                                GetSunmmioLayoutBlockShape(
                                                    T.target, src->dtype));
      result.Set(src, src_layout);
    }
    return result;
  }

  // Propagate: derive layout for each side from the other.
  if (src_has && IsSunmmioSramScope(dst.scope())) {
    auto derived =
        DeriveLayoutLikeForDType(T.layout_map[src], dst->shape, dst->dtype);
    if (derived.defined()) {
      result.Set(dst, derived.value());
    }
  }
  if (dst_has && IsSunmmioSramScope(src.scope())) {
    auto derived =
        DeriveLayoutLikeForDType(T.layout_map[dst], src->shape, src->dtype);
    if (derived.defined()) {
      result.Set(src, derived.value());
    }
  }

  return result;
}

void CheckSunmmioCommBuffers(const char *op_name, const Buffer &src,
                             const Buffer &dst) {
  bool valid_src = src.scope() == "global" || src.scope().empty() ||
                   IsSunmmioSramScope(src.scope());
  ICHECK(valid_src && IsSunmmioSramScope(dst.scope()))
      << op_name << " expects src to be global or Sunmmio SRAM and dst to be "
      << "Sunmmio SRAM, got src buffer " << src->name << " scope "
      << src.scope() << " and dst buffer " << dst->name << " scope "
      << dst.scope();
}

BroadcastOp::BroadcastOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
  ObjectPtr<BroadcastOpNode> node = tvm::ffi::make_object<BroadcastOpNode>();
  node->src_expr = args[0];
  node->dst_expr = args[1];
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);

    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  node->size = Downcast<IntImm>(args[2]);
  node->src_core = args[3];
  node->direction = Downcast<IntImm>(args[4])->value;
  // Optional trailing positional arg; default 0.
  if (args.size() > 5) {
    node->srcOffsetByte_ = Downcast<IntImm>(args[5])->value;
  }
  data_ = std::move(node);
}

TileOperator BroadcastOpNode::Clone() const {
  auto op = tvm::ffi::make_object<BroadcastOpNode>(*this);
  return BroadcastOp(op);
}

LayoutMap BroadcastOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  CheckSunmmioCommBuffers("T.comm.broadcast", src, dst);
  return SunmmioCommInferLayout(T, src, dst, level);
}

Stmt BroadcastOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Broadcast only supports SUNMMIO targets.";
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

  // check for valid core id
  if (src_core.as<IntImmNode>()) {
    int src_core_val = src_core.as<IntImmNode>()->value;
    ICHECK(src_core_val >= 0 and src_core_val < mesh_nrow * mesh_ncol)
        << "Source core id " << src_core_val << " out of range [0, "
        << mesh_nrow * mesh_ncol << ")";
  }

  // check for src and dst buffer sizes
  PrimExpr src_elements = 1;
  for (size_t i = 0; i < src_range.size(); i++) {
    src_elements *= src_range[i]->extent;
  }
  src_elements = analyzer->Simplify(src_elements);
  PrimExpr dst_elements = 1;
  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_elements *= dst_range[i]->extent;
  }
  dst_elements = analyzer->Simplify(dst_elements);
  ICHECK(Downcast<IntImm>(src_elements)->value <=
         Downcast<IntImm>(dst_elements)->value)
      << "Source buffer size larger than destination buffer size: "
      << src_elements << " vs " << dst_elements;
  ICHECK(size->value <= Downcast<IntImm>(src_elements)->value)
      << "Broadcast size larger than data size: " << size->value << " vs "
      << Downcast<IntImm>(src_elements)->value;

  // check for size
  PrimExpr broadcast_elements;
  if (size->value < 0) {
    broadcast_elements = src_elements;
  } else {
    broadcast_elements = size;
  }
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(src_elements)->value)
      << "Broadcast size Larger than source buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(src_elements)->value;
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(dst_elements)->value)
      << "Broadcast size larger than destination buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(dst_elements)->value;

  // check for valid direction
  if (direction != 0 and direction != 1 and direction != 2) {
    LOG(FATAL) << "Invalid broadcast direction " << direction
               << ", must be 0 (horizontal) or 1 (vertical) or 2 (all).";
  }

  PrimExpr src_offset_imm = IntImm(DataType::Int(32), srcOffsetByte_);
  if (direction == 0 or direction == 1) {
    // 1D broadcast
    Array<PrimExpr> args;
    PrimExpr mask = direction == 0 ? MakeHorizontalMask(mesh_ncol)
                                   : MakeVerticalMask(mesh_nrow);
    AppendBroadcastArgs(&args,
                        MakeRegionExpr(src, src_range, /*access_mask=*/1),
                        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
                        direction, mask, src_offset_imm, src_core);
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return broadcast;
  } else {
    // 2D broadcast
    PrimExpr src_core_i32 = AsI32(src_core);
    PrimExpr src_core_col =
        analyzer->Simplify(floormod(src_core_i32, I32Imm(mesh_ncol)));

    Array<Stmt> seq;
    // vertical broadcast
    Array<PrimExpr> args;
    AppendBroadcastArgs(
        &args, MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        /*direction=*/1, MakeVerticalMask(mesh_nrow), src_offset_imm, src_core);
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    seq.push_back(broadcast);
    // horizontal broadcast
    for (int i = 0; i < mesh_nrow; i++) {
      Array<PrimExpr> args;
      PrimExpr row_src_core =
          analyzer->Simplify(I32Imm(i * mesh_ncol) + src_core_col);
      AppendBroadcastArgs(&args,
                          MakeRegionExpr(dst, dst_range, /*access_mask=*/1),
                          MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
                          /*direction=*/0, MakeHorizontalMask(mesh_ncol),
                          src_offset_imm, row_src_core);
      Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
      seq.push_back(broadcast);
    }
    return SeqStmt::Flatten(seq);
  }
}

TIR_REGISTER_TL_TILE_OP(BroadcastOp, comm_broadcast)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

PutOp::PutOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<PutOpNode> node = tvm::ffi::make_object<PutOpNode>();
  node->src_expr = args[0];
  node->dst_expr = args[1];
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  node->size = Downcast<IntImm>(args[2]);
  node->src_core = args[3];
  node->dst_core = args[4];
  data_ = std::move(node);
}

TileOperator PutOpNode::Clone() const {
  auto op = tvm::ffi::make_object<PutOpNode>(*this);
  return PutOp(op);
}

LayoutMap PutOpNode::InferLayout(const LayoutInferArgs &T,
                                 InferLevel level) const {
  CheckSunmmioCommBuffers("T.comm.put", src, dst);
  return SunmmioCommInferLayout(T, src, dst, level);
}

Stmt PutOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Put only supports SUNMMIO targets.";
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

  // check for valid core id
  if (src_core.as<IntImmNode>()) {
    int src_core_val = src_core.as<IntImmNode>()->value;
    ICHECK(src_core_val >= 0 and src_core_val < mesh_nrow * mesh_ncol)
        << "Source core id " << src_core_val << " out of range [0, "
        << mesh_nrow * mesh_ncol << ")";
  }
  if (dst_core.as<IntImmNode>()) {
    int dst_core_val = dst_core.as<IntImmNode>()->value;
    ICHECK(dst_core_val >= 0 and dst_core_val < mesh_nrow * mesh_ncol)
        << "Destination core id " << dst_core_val << " out of range [0, "
        << mesh_nrow * mesh_ncol << ")";
  }
  ICHECK(!analyzer || !analyzer->CanProve(AsI32(src_core) == AsI32(dst_core)))
      << "T.comm.put requires src_core and dst_core to be different";

  // check for src and dst buffer sizes
  PrimExpr src_elements = 1;
  for (size_t i = 0; i < src_range.size(); i++) {
    src_elements *= src_range[i]->extent;
  }
  src_elements = analyzer->Simplify(src_elements);
  PrimExpr dst_elements = 1;
  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_elements *= dst_range[i]->extent;
  }
  dst_elements = analyzer->Simplify(dst_elements);
  ICHECK(Downcast<IntImm>(src_elements)->value <=
         Downcast<IntImm>(dst_elements)->value)
      << "Source buffer size larger than destination buffer size: "
      << src_elements << " vs " << dst_elements;
  ICHECK(size->value <= Downcast<IntImm>(src_elements)->value)
      << "Put size larger than data size: " << size->value << " vs "
      << Downcast<IntImm>(src_elements)->value;

  // check for size
  PrimExpr broadcast_elements;
  if (size->value < 0) {
    broadcast_elements = src_elements;
  } else {
    broadcast_elements = size;
  }
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(src_elements)->value)
      << "Put size Larger than source buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(src_elements)->value;
  ICHECK((Downcast<IntImm>(broadcast_elements)->value) <=
         Downcast<IntImm>(dst_elements)->value)
      << "Put size larger than destination buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value) << " vs "
      << Downcast<IntImm>(dst_elements)->value;

  auto make_put_broadcast = [&](PrimExpr src_region, PrimExpr dst_region,
                                int direction, PrimExpr mask,
                                PrimExpr bcast_src_core) {
    Array<PrimExpr> args;
    AppendBroadcastArgs(&args, src_region, dst_region, direction,
                        analyzer->Simplify(mask), I32Imm(0),
                        analyzer->Simplify(bcast_src_core));
    return Evaluate(Call(DataType::Handle(), broadcast_(), args));
  };

  const auto *src_core_imm = src_core.as<IntImmNode>();
  const auto *dst_core_imm = dst_core.as<IntImmNode>();
  if (!src_core_imm || !dst_core_imm) {
    PrimExpr src_core_i32 = AsI32(src_core);
    PrimExpr dst_core_i32 = AsI32(dst_core);
    PrimExpr mesh_ncol_expr = I32Imm(mesh_ncol);
    PrimExpr src_core_row =
        analyzer->Simplify(floordiv(src_core_i32, mesh_ncol_expr));
    PrimExpr src_core_col =
        analyzer->Simplify(floormod(src_core_i32, mesh_ncol_expr));
    PrimExpr dst_core_row =
        analyzer->Simplify(floordiv(dst_core_i32, mesh_ncol_expr));
    PrimExpr dst_core_col =
        analyzer->Simplify(floormod(dst_core_i32, mesh_ncol_expr));
    PrimExpr intermediate_core =
        analyzer->Simplify(dst_core_row * mesh_ncol_expr + src_core_col);

    PrimExpr src_read = MakeRegionExpr(src, src_range, /*access_mask=*/1);
    PrimExpr dst_write = MakeRegionExpr(dst, dst_range, /*access_mask=*/2);
    PrimExpr dst_read = MakeRegionExpr(dst, dst_range, /*access_mask=*/1);

    Stmt horizontal =
        make_put_broadcast(src_read, dst_write, /*direction=*/0,
                           MakeLocalSingleBit(dst_core_col), src_core_i32);
    Stmt vertical =
        make_put_broadcast(src_read, dst_write, /*direction=*/1,
                           MakeLocalSingleBit(dst_core_row), src_core_i32);
    Array<Stmt> diagonal_seq;
    diagonal_seq.push_back(
        make_put_broadcast(src_read, dst_write, /*direction=*/1,
                           MakeLocalSingleBit(dst_core_row), src_core_i32));
    diagonal_seq.push_back(make_put_broadcast(
        dst_read, dst_write, /*direction=*/0, MakeLocalSingleBit(dst_core_col),
        intermediate_core));
    Stmt diagonal = SeqStmt::Flatten(diagonal_seq);

    Stmt routed = IfThenElse(
        src_core_row == dst_core_row, horizontal,
        IfThenElse(src_core_col == dst_core_col, vertical, diagonal));
    return AssertPutDistinctCores(src_core_i32, dst_core_i32, routed, analyzer);
  }

  int src_core_val = src_core_imm->value;
  int dst_core_val = dst_core_imm->value;
  ICHECK_NE(src_core_val, dst_core_val)
      << "T.comm.put requires src_core and dst_core to be different";
  int src_core_row = src_core_val / mesh_ncol;
  int src_core_col = src_core_val % mesh_ncol;
  int dst_core_row = dst_core_val / mesh_ncol;
  int dst_core_col = dst_core_val % mesh_ncol;

  if (src_core_row == dst_core_row) {
    // 1D put via horizontal communication
    return make_put_broadcast(
        MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        /*direction=*/0, MakeLocalSingleBit(I32Imm(dst_core_col)), src_core);
  } else if (src_core_col == dst_core_col) {
    // 1D put via vertical communication
    return make_put_broadcast(
        MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2),
        /*direction=*/1, MakeLocalSingleBit(I32Imm(dst_core_row)), src_core);
  } else {
    Array<Stmt> seq;
    // vertical transfer from src core to intermediate core
    int intermediate_core_id = dst_core_row * mesh_ncol + src_core_col;
    Stmt put1 = make_put_broadcast(
        MakeRegionExpr(src, src_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2), /*direction=*/1,
        MakeLocalSingleBit(I32Imm(dst_core_row)), src_core);
    seq.push_back(put1);
    // horizontal transfer from intermediate core to dst core
    PrimExpr intermediate_core = I32Imm(intermediate_core_id);
    Stmt put2 = make_put_broadcast(
        MakeRegionExpr(dst, dst_range, /*access_mask=*/1),
        MakeRegionExpr(dst, dst_range, /*access_mask=*/2), /*direction=*/0,
        MakeLocalSingleBit(I32Imm(dst_core_col)), intermediate_core);
    seq.push_back(put2);
    return SeqStmt::Flatten(seq);
  }
}

TIR_REGISTER_TL_TILE_OP(PutOp, comm_put)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

AllgatherOp::AllgatherOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
  ICHECK_GE(args.size(), 4)
      << "Allgather expects at least send, recv, direction, and size.";
  ObjectPtr<AllgatherOpNode> node = tvm::ffi::make_object<AllgatherOpNode>();
  node->send = args[0];
  node->recv = args[1];
  node->direction = Downcast<IntImm>(args[2])->value;
  node->size = Downcast<IntImm>(args[3]);
  // axis is optional for backwards compatibility: -1 = legacy mode.
  node->axis = (args.size() > 4) ? Downcast<IntImm>(args[4])->value : -1;
  // cid is optional during migration: new Python frontend calls pass the
  // blockIdx.x binding, while older internal C++ call sites may still omit it.
  node->cid = (args.size() > 5) ? args[5] : PrimExpr();
  node->annotations = annotations;
  data_ = std::move(node);
}

bool IsCommunicationOp(const Call &call) {
  return call->op.same_as(BroadcastOp::Get()) ||
         call->op.same_as(PutOp::Get()) || call->op.same_as(AllgatherOp::Get());
}

CommunicationDirections GetCommunicationSourceDirections(const Call &call,
                                                         Target target) {
  ICHECK(IsCommunicationOp(call));
  TileOperator tile_op = ParseOperator(call);
  ICHECK(tile_op.defined());
  if (const auto *broadcast = tile_op.as<BroadcastOpNode>()) {
    ICHECK(broadcast->direction >= 0 && broadcast->direction <= 2)
        << "Invalid broadcast direction " << broadcast->direction;
    // "all" starts vertically; its horizontal phase reads the destination.
    return broadcast->direction == 0 ? CommunicationDirections::kHorizontal
                                     : CommunicationDirections::kVertical;
  }
  if (const auto *allgather = tile_op.as<AllgatherOpNode>()) {
    ICHECK(allgather->direction >= 0 && allgather->direction <= 2)
        << "Invalid allgather direction " << allgather->direction;
    // "all" starts horizontally; its vertical phase reads the destination.
    return allgather->direction == 1 ? CommunicationDirections::kVertical
                                     : CommunicationDirections::kHorizontal;
  }
  if (const auto *put = tile_op.as<PutOpNode>()) {
    const auto *src_core = put->src_core.as<IntImmNode>();
    const auto *dst_core = put->dst_core.as<IntImmNode>();
    if (src_core == nullptr || dst_core == nullptr) {
      return CommunicationDirections::kHorizontalAndVertical;
    }

    int mesh_ncol = GetSunmmioMeshConfig(target).ncol;
    bool same_row = src_core->value / mesh_ncol == dst_core->value / mesh_ncol;
    // Diagonal put starts vertically; its horizontal phase reads dst.
    return same_row ? CommunicationDirections::kHorizontal
                    : CommunicationDirections::kVertical;
  }
  LOG(FATAL) << "Expected broadcast, put, or allgather communication op";
  return CommunicationDirections::kNone;
}

TileOperator AllgatherOpNode::Clone() const {
  auto op = tvm::ffi::make_object<AllgatherOpNode>(*this);
  return AllgatherOp(op);
}

// Not yet complete; it will be further refined later
LayoutMap AllgatherOpNode::ComputeLayout(const LayoutInferArgs &T,
                                         InferLevel level, Buffer src,
                                         Buffer dst) const {
  if (level >= InferLevel::kStrict) {
    return {};
  }

  CheckSunmmioCommBuffers("T.comm.all_gather", src, dst);
  return SunmmioCommInferLayout(T, src, dst, level);
}

LayoutMap AllgatherOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  Buffer src_buffer = NormalizeToBufferRegion(send)->buffer;
  Buffer recv_buffer = NormalizeToBufferRegion(recv)->buffer;
  return ComputeLayout(T, level, src_buffer, recv_buffer);
}

Stmt AllgatherOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Allgather only supports SUNMMIO targets.";
  auto mesh = GetSunmmioMeshConfig(target);
  int mesh_nrow = mesh.nrow;
  int mesh_ncol = mesh.ncol;

  Array<Range> send_range, recv_range;
  auto send_region = NormalizeToBufferRegion(send);
  auto recv_region = NormalizeToBufferRegion(recv);
  send_range = send_region->region;
  recv_range = recv_region->region;

  int recv_num = 1;
  if (direction == 0) { // horizontal
    recv_num = mesh_ncol;
  } else if (direction == 1) { // vertical
    recv_num = mesh_nrow;
  } else if (direction == 2) { // all
    recv_num = mesh_nrow * mesh_ncol;
  } else {
    // invalid direction
    ICHECK(false) << "Invalid direction value for allgather: " << direction;
  }

  PrimExpr send_elements = 1;
  for (size_t i = 0; i < send_range.size(); i++) {
    send_elements *= send_range[i]->extent;
  }
  send_elements = analyzer->Simplify(send_elements);
  PrimExpr recv_elements = 1;
  for (size_t i = 0; i < recv_range.size(); i++) {
    recv_elements *= recv_range[i]->extent;
  }
  recv_elements = analyzer->Simplify(recv_elements);
  // check for buffer sizes
  ICHECK(Downcast<IntImm>(send_elements)->value * recv_num <=
         Downcast<IntImm>(recv_elements)->value)
      << "Receive buffer size not enough for allgather: required "
      << (Downcast<IntImm>(send_elements)->value * recv_num) << ", but got "
      << Downcast<IntImm>(recv_elements)->value;
  PrimExpr bcast_elements = (size->value < 0) ? send_elements : PrimExpr(size);
  ICHECK(Downcast<IntImm>(bcast_elements)->value <=
         Downcast<IntImm>(send_elements)->value)
      << "Allgather size larger than send buffer size: "
      << Downcast<IntImm>(bcast_elements)->value << " vs "
      << Downcast<IntImm>(send_elements)->value;

  ICHECK(cid.defined())
      << "Allgather dynamic-region lowering requires current core id.";

  // Unified lowering for all axis modes. Every per-core contribution lands
  // in an equal-size slot along a single axis of recv ("slice_axis"). Slot
  // offset is encoded in the region's `min`, not in a separate offset arg.
  //
  //   axis == -1 (sentinel, "no axis") or axis == 0 -> slice_axis = 0
  //     legacy recv [K, d0, ...]:        slot_extent = 1
  //     axis=0 recv [K*d0, d1, ...]:     slot_extent = d0
  //   axis > 0 (last dim of send)        -> slice_axis = recv_rank - 1
  //     axis=-1 recv [d0, ..., K*dn-1]:  slot_extent = dn-1
  Buffer send_buffer = send_region->buffer;
  Buffer recv_buffer = recv_region->buffer;
  int recv_rank = static_cast<int>(recv_range.size());
  ICHECK_GT(recv_rank, 0) << "Allgather recv must have at least one dim.";
  if (axis > 0) {
    ICHECK_EQ(axis, static_cast<int>(send_range.size()) - 1)
        << "Only axis = last dim of send is supported; got axis=" << axis
        << " for send rank=" << send_range.size();
    ICHECK_EQ(recv_range.size(), send_range.size())
        << "In axis mode, recv and send must have the same rank.";
  }
  int slice_axis = (axis > 0) ? (recv_rank - 1) : 0;
  ICHECK(analyzer->CanProve(FloorMod(recv_range[slice_axis]->extent,
                                     IntImm(DataType::Int(32), recv_num)) == 0))
      << "Recv extent along slice axis " << slice_axis << " ("
      << recv_range[slice_axis]->extent << ") must be divisible by recv_num ("
      << recv_num << ").";
  PrimExpr slot_extent_p1 = analyzer->Simplify(FloorDiv(
      recv_range[slice_axis]->extent, IntImm(DataType::Int(32), recv_num)));

  // Build a sub-region of recv spanning a single slab of width `slot_extent`
  // along `slice_axis`, starting at slot index `slot_start`. All other dims
  // pass through `recv_range` unchanged.
  auto make_slab = [&](PrimExpr slot_start, PrimExpr slot_extent) {
    Array<Range> ranges;
    for (int d = 0; d < recv_rank; d++) {
      if (d == slice_axis) {
        PrimExpr base =
            analyzer->Simplify(recv_range[d]->min + slot_start * slot_extent);
        ranges.push_back(Range::FromMinExtent(base, slot_extent));
      } else {
        ranges.push_back(recv_range[d]);
      }
    }
    return ranges;
  };

  Array<Stmt> bcast_stmts;

  // Propagate src_offset_byte (read from this op's annotations) into each
  // BroadcastOp we construct, so every emitted broadcast_() carries it before
  // the optional src_core and target metadata. No AttrStmt wrapping needed.
  int src_offset_byte = GetSrcOffsetByte();
  PrimExpr src_offset_imm = IntImm(DataType::Int(32), src_offset_byte);

  PrimExpr current_core = cid;
  PrimExpr mesh_ncol_expr = IntImm(current_core.dtype(), mesh_ncol);
  PrimExpr current_row =
      analyzer->Simplify(floordiv(current_core, mesh_ncol_expr));
  PrimExpr current_col =
      analyzer->Simplify(floormod(current_core, mesh_ncol_expr));

  auto emit_from_send = [&](Array<Range> dst_ranges, int bcast_dir,
                            PrimExpr mask) {
    bcast_stmts.push_back(MakeBroadcastLeaf(
        MakeRegionExpr(send_buffer, send_range, /*access_mask=*/1),
        MakeRegionExpr(recv_buffer, dst_ranges, /*access_mask=*/2), bcast_dir,
        analyzer->Simplify(mask), src_offset_imm));
  };

  auto emit_from_recv = [&](Array<Range> slab_ranges, int bcast_dir,
                            PrimExpr mask) {
    bcast_stmts.push_back(MakeBroadcastLeaf(
        MakeRegionExpr(recv_buffer, slab_ranges, /*access_mask=*/1),
        MakeRegionExpr(recv_buffer, slab_ranges, /*access_mask=*/2), bcast_dir,
        analyzer->Simplify(mask), src_offset_imm));
  };

  if (direction == 0) { // horizontal
    emit_from_send(make_slab(current_col, slot_extent_p1), /*bcast_dir=*/0,
                   MakeHorizontalMask(mesh_ncol));
  } else if (direction == 1) { // vertical
    emit_from_send(make_slab(current_row, slot_extent_p1), /*bcast_dir=*/1,
                   MakeVerticalMask(mesh_nrow));
  } else { // direction == 2 ("all")
    // Phase 1: vertical. The current core lands at its row-major global slot.
    emit_from_send(make_slab(current_core, slot_extent_p1), /*bcast_dir=*/1,
                   MakeVerticalMask(mesh_nrow));

    // Phase 2: horizontal. A column's phase-1 slots are strided by ncol in
    // row-major core order, so forward one per-source-row slab at a time.
    // This preserves the existing recv slot mapping: slot = row*ncol + col.
    for (int row = 0; row < mesh_nrow; ++row) {
      PrimExpr global_slot =
          analyzer->Simplify(I32Imm(row * mesh_ncol) + current_col);
      emit_from_recv(make_slab(global_slot, slot_extent_p1),
                     /*bcast_dir=*/0, MakeHorizontalMask(mesh_ncol));
    }
  }

  return SeqStmt::Flatten(bcast_stmts);
}

TIR_REGISTER_TL_TILE_OP(AllgatherOp, comm_allgather)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

AllreduceOp::AllreduceOp(Array<PrimExpr> args,
                         Map<String, ObjectRef> annotations) {
  ICHECK(args.size() == 9 || args.size() == 10)
      << "Allreduce expects 9 or 10 inputs, got " << args.size();
  ObjectPtr<AllreduceOpNode> node = tvm::ffi::make_object<AllreduceOpNode>();
  node->src = args[0];
  node->dst = args[1];
  node->row_allgather = args[2];
  node->col_allgather = args[3];

  node->type = Downcast<StringImm>(args[4]);
  node->direction = Downcast<IntImm>(args[5])->value;
  node->dim = Downcast<IntImm>(args[6]);
  node->clear = Downcast<IntImm>(args[7]);
  if (args.size() == 10) {
    node->dst_copy = args[8];
    node->cid = args[9];
  } else {
    node->cid = args[8];
  }
  data_ = std::move(node);
}

TileOperator AllreduceOpNode::Clone() const {
  auto op = tvm::ffi::make_object<AllreduceOpNode>(*this);
  return AllreduceOp(op);
}

// Not yet complete; it will be further refined later
LayoutMap AllreduceOpNode::ComputeLayout(const LayoutInferArgs &T,
                                         InferLevel level, Buffer src,
                                         Buffer dst, int dim) const {
  if (level >= InferLevel::kStrict)
    return {};

  (void)dim;
  CheckSunmmioCommBuffers("T.comm.all_reduce", src, dst);
  return SunmmioCommInferLayout(T, src, dst, level);
}

LayoutMap AllreduceOpNode::InferLayout(const LayoutInferArgs &T,
                                       InferLevel level) const {
  LayoutMap lm;

  bool should_clear = clear.as<Bool>().value();
  ICHECK(should_clear || dst_copy.defined())
      << "Allreduce clear=false requires a dst_copy temporary buffer.";
  ICHECK(cid.defined())
      << "Allreduce dynamic allgather lowering requires current core id.";

  LayoutInferArgs local_T = T;
  LayoutMap known_layout = T.layout_map;
  LayoutMap proposed_layout;
  local_T.layout_map = known_layout;

  auto merge_layout = [&](const LayoutMap &layout) {
    for (const auto &kv : layout) {
      if (proposed_layout.count(kv.first)) {
        Layout existing = proposed_layout[kv.first];
        if (!IsSameLayout(existing, kv.second, T.analyzer)) {
          LOG(FATAL) << "Allreduce layout conflict on buffer \""
                     << kv.first->name << "\""
                     << "\n  existing: " << existing->DebugOutput()
                     << "\n  proposed: " << kv.second->DebugOutput();
        }
        continue;
      }
      proposed_layout.Set(kv.first, kv.second);
      lm.Set(kv.first, kv.second);
      known_layout.Set(kv.first, kv.second);
    }
    local_T.layout_map = known_layout;
  };

  auto infer_reduce = [&](PrimExpr reduce_src, PrimExpr reduce_dst,
                          PrimExpr reduce_dim, bool reduce_clear) {
    Array<PrimExpr> args;
    args.push_back(reduce_src);
    args.push_back(reduce_dst);
    args.push_back(type);
    args.push_back(reduce_dim);
    args.push_back(Bool(reduce_clear));
    ReduceOp reduce_op = ReduceOp(args);
    merge_layout(reduce_op->InferLayout(local_T, level));
  };

  auto infer_allgather = [&](PrimExpr send, PrimExpr recv, int gather_dir) {
    Array<PrimExpr> args;
    args.push_back(send);
    args.push_back(recv);
    args.push_back(IntImm(DataType::Int(32), gather_dir));
    args.push_back(IntImm(DataType::Int(32), -1)); // size
    args.push_back(IntImm(DataType::Int(32), -1)); // axis
    args.push_back(cid);
    AllgatherOp allgather_op = AllgatherOp(args);
    merge_layout(allgather_op->InferLayout(local_T, level));
  };

  if (should_clear) {
    infer_reduce(src, dst, dim, /*reduce_clear=*/true);

    if (direction == 1 || direction == 2) {
      infer_allgather(dst, col_allgather, /*gather_dir=*/1);
      infer_reduce(col_allgather, dst, IntImm(DataType::Int(32), 0),
                   /*reduce_clear=*/true);
    }

    if (direction == 0 || direction == 2) {
      infer_allgather(dst, row_allgather, /*gather_dir=*/0);
      infer_reduce(row_allgather, dst, IntImm(DataType::Int(32), 0),
                   /*reduce_clear=*/true);
    }
  } else {
    infer_reduce(src, dst_copy, dim, /*reduce_clear=*/true);

    if (direction == 1 || direction == 2) {
      infer_allgather(dst_copy, col_allgather, /*gather_dir=*/1);
      infer_reduce(col_allgather, direction == 1 ? dst : dst_copy,
                   IntImm(DataType::Int(32), 0),
                   /*reduce_clear=*/direction == 2);
    }

    if (direction == 0 || direction == 2) {
      infer_allgather(dst_copy, row_allgather, /*gather_dir=*/0);
      infer_reduce(row_allgather, dst, IntImm(DataType::Int(32), 0),
                   /*reduce_clear=*/false);
    }
  }

  return lm;
}

Stmt AllreduceOpNode::Lower(const LowerArgs &T,
                            arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(TargetIsSunmmio(target)) << "Allreduce only supports SUNMMIO targets.";

  ICHECK(direction == 0 || direction == 1 || direction == 2)
      << "Invalid allreduce direction " << direction
      << ", must be 0 (row-wise) or 1 (column-wise) or 2 (all).";

  bool should_clear = clear.as<Bool>().value();
  ICHECK(should_clear || dst_copy.defined())
      << "Allreduce clear=false requires a dst_copy temporary buffer.";
  ICHECK(cid.defined())
      << "Allreduce dynamic allgather lowering requires current core id.";

  Array<Stmt> stmts;

  auto append_reduce = [&](PrimExpr reduce_src, PrimExpr reduce_dst,
                           PrimExpr reduce_dim, bool reduce_clear) {
    Array<PrimExpr> args;
    args.push_back(reduce_src);
    args.push_back(reduce_dst);
    args.push_back(type);
    args.push_back(reduce_dim);
    args.push_back(Bool(reduce_clear));
    ReduceOp reduce_op = ReduceOp(args);
    stmts.push_back(reduce_op->Lower(T, analyzer));
  };

  auto append_allgather = [&](PrimExpr send, PrimExpr recv, int gather_dir) {
    Array<PrimExpr> args;
    args.push_back(send);
    args.push_back(recv);
    args.push_back(IntImm(DataType::Int(32), gather_dir));
    args.push_back(IntImm(DataType::Int(32), -1)); // size
    args.push_back(IntImm(DataType::Int(32), -1)); // axis
    args.push_back(cid);
    AllgatherOp allgather_op = AllgatherOp(args);
    stmts.push_back(allgather_op->Lower(T, analyzer));
  };

  auto append_row_stage = [&](PrimExpr send, PrimExpr reduce_dst,
                              bool reduce_clear) {
    append_allgather(send, row_allgather, /*gather_dir=*/0);
    append_reduce(row_allgather, reduce_dst, IntImm(DataType::Int(32), 0),
                  reduce_clear);
  };

  auto append_col_stage = [&](PrimExpr send, PrimExpr reduce_dst,
                              bool reduce_clear) {
    append_allgather(send, col_allgather, /*gather_dir=*/1);
    append_reduce(col_allgather, reduce_dst, IntImm(DataType::Int(32), 0),
                  reduce_clear);
  };

  if (should_clear) {
    append_reduce(src, dst, dim, /*reduce_clear=*/true);

    if (direction == 1 || direction == 2) {
      append_col_stage(dst, dst, /*reduce_clear=*/true);
    }

    if (direction == 0 || direction == 2) {
      append_row_stage(dst, dst, /*reduce_clear=*/true);
    }
  } else {
    append_reduce(src, dst_copy, dim, /*reduce_clear=*/true);

    if (direction == 1 || direction == 2) {
      append_col_stage(dst_copy, direction == 1 ? dst : dst_copy,
                       /*reduce_clear=*/direction == 2);
    }

    if (direction == 0 || direction == 2) {
      append_row_stage(dst_copy, dst, /*reduce_clear=*/false);
    }
  }

  return SeqStmt::Flatten(stmts);
}

TIR_REGISTER_TL_TILE_OP(AllreduceOp, comm_allreduce)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  PutOpNode::RegisterReflection();
  BroadcastOpNode::RegisterReflection();
  AllgatherOpNode::RegisterReflection();
  AllreduceOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
