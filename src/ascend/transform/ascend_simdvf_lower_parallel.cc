// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_simdvf_lower_parallel.cc
 * \brief Lower T.Parallel loops inside SIMD_VF blocks to Ascend SIMD
 *        return-value vector instructions.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "op/utils.h"

#include <functional>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using ffi::Array;

class AscendSimdVFLowerParallel : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    AscendSimdVFLowerParallel substituter;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return ffi::GetRef<PrimFunc>(fptr);
  }

private:
  bool inside_simdvf_{false};

  const VarNode *parallel_var_{nullptr};
  int64_t parallel_extent_{0};

  const VarNode *outer_parallel_var_{nullptr};
  Var outer_serial_var_;

  int64_t vreg_size_{0};
  Var chunk_var_;
  Var mask_var_;

  int frag_counter_{0};
  int64_t elem_bits_{0};
  std::vector<Stmt> scalar_loads_;

  static int64_t ComputeVRegSize(DataType dtype) {
    return 8 * 32 / (dtype.bits() / 8);
  }

  static PrimExpr MatchIndexDType(const PrimExpr &value, DataType dtype) {
    return value.dtype() == dtype ? value : Cast(dtype, value);
  }

  static Array<PrimExpr> InferCompactStrides(const Array<PrimExpr> &shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<PrimExpr> strides(ndim);
    PrimExpr stride = IntImm(shape.back().dtype(), 1);
    for (int i = ndim - 1; i >= 0; --i) {
      strides[i] = stride;
      stride = stride * shape[i];
    }
    return Array<PrimExpr>(strides.begin(), strides.end());
  }

  // True when a row-major walk of the region is a contiguous element sequence,
  // so a VReg-sized vsts from each linearized chunk address is legal. Leading
  // unit-extent dims (the in-loop multi-buffer version slot) are skipped.
  static bool IsContiguousRegion(const Buffer &buf,
                                 const std::vector<int64_t> &extents) {
    int ndim = static_cast<int>(extents.size());
    ICHECK_EQ(static_cast<int>(buf->shape.size()), ndim);
    Array<PrimExpr> strides =
        buf->strides.empty() ? InferCompactStrides(buf->shape) : buf->strides;
    ICHECK_EQ(static_cast<int>(strides.size()), ndim);
    int64_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      if (extents[i] == 1) {
        continue;
      }
      const auto *stride = strides[i].as<IntImmNode>();
      if (!stride || stride->value != expected) {
        return false;
      }
      expected *= extents[i];
    }
    return true;
  }

  static bool TryGetConstExtents(const Array<Range> &ranges,
                                 std::vector<int64_t> *extents,
                                 int64_t *total) {
    extents->clear();
    *total = 1;
    for (const Range &range : ranges) {
      const auto *ext = range->extent.as<IntImmNode>();
      if (!ext) {
        return false;
      }
      extents->push_back(ext->value);
      *total *= ext->value;
    }
    return true;
  }

  // Delinearize a flattened region offset. Mins may be symbolic (the physical
  // version index after MaterializeMultiBuffer).
  static Array<PrimExpr> MakeRegionIndices(const Array<Range> &ranges,
                                           const std::vector<int64_t> &extents,
                                           PrimExpr flat_idx) {
    int ndim = static_cast<int>(ranges.size());
    ICHECK_EQ(static_cast<int>(extents.size()), ndim);
    DataType dtype = flat_idx.dtype();
    if (ndim == 1) {
      return {ranges[0]->min +
              MatchIndexDType(flat_idx, ranges[0]->min.dtype())};
    }
    Array<PrimExpr> idxs;
    PrimExpr remaining = flat_idx;
    for (int i = ndim - 1; i >= 1; --i) {
      PrimExpr ext = IntImm(dtype, extents[i]);
      PrimExpr coord = FloorMod(remaining, ext);
      idxs.insert(idxs.begin(),
                  ranges[i]->min +
                      MatchIndexDType(coord, ranges[i]->min.dtype()));
      remaining = FloorDiv(remaining, ext);
    }
    idxs.insert(idxs.begin(),
                ranges[0]->min +
                    MatchIndexDType(remaining, ranges[0]->min.dtype()));
    return idxs;
  }

  // vsts_norm lowers to asc_storealign and requires a 32-byte-aligned base.
  // Prove the linearized region byte offset is a multiple of 32 (mirrors
  // insert_nd2nz.cc:HasAlignedScatterBase); symbolic version indices remain
  // provable when the per-version stride is 32-byte aligned.
  static void ValidateFillBaseAlignment(const Buffer &buf,
                                        const Array<Range> &ranges) {
    arith::Analyzer analyzer;
    DataType index_dtype = DataType::Int(64);
    PrimExpr offset = cast(index_dtype, buf->elem_offset);
    PrimExpr compact_stride = make_const(index_dtype, 1);
    for (size_t axis = buf->shape.size(); axis-- > 0;) {
      PrimExpr stride = buf->strides.empty()
                            ? compact_stride
                            : cast(index_dtype, buf->strides[axis]);
      offset = offset + cast(index_dtype, ranges[axis]->min) * stride;
      compact_stride = compact_stride * cast(index_dtype, buf->shape[axis]);
    }
    PrimExpr element_bytes =
        make_const(index_dtype, (buf->dtype.bits() + 7) / 8);
    PrimExpr alignment = make_const(index_dtype, 32);
    if (!analyzer.CanProveEqual(FloorMod(offset * element_bytes, alignment),
                                make_const(index_dtype, 0))) {
      TVM_FFI_THROW(ValueError)
          << "T.fill inside SimdVF requires a 32-byte-aligned region base "
             "address, got region ["
          << ranges << "] of buffer " << buf->name
          << " with linearized byte offset " << offset * element_bytes
          << " (element size " << (buf->dtype.bits() + 7) / 8
          << " bytes). Use a region whose base is 32-byte aligned.";
    }
  }

  static bool CanLowerFillRegion(const Buffer &buf, const Array<Range> &ranges,
                                 int64_t vreg_size, int64_t *total,
                                 std::vector<int64_t> *extents) {
    if (ranges.empty() || static_cast<int>(ranges.size()) !=
                              static_cast<int>(buf->shape.size())) {
      return false;
    }
    if (!TryGetConstExtents(ranges, extents, total)) {
      return false;
    }
    if (*total <= 0 || *total % vreg_size != 0) {
      return false;
    }
    if (!IsContiguousRegion(buf, *extents)) {
      return false;
    }
    // Structural checks passed; report a misaligned base with a clear error.
    ValidateFillBaseAlignment(buf, ranges);
    return true;
  }

  Var MakeReg(DataType dtype) {
    std::string name = "vreg_" + std::to_string(frag_counter_++);
    int vl = ComputeVRegSize(dtype);
    return Var(name, dtype.with_lanes(vl));
  }

  static Stmt MakeSeq(Array<Stmt> seq) {
    if (seq.empty()) {
      return Evaluate(IntImm(DataType::Int(32), 0));
    }
    if (seq.size() == 1) {
      return seq[0];
    }
    return SeqStmt(seq);
  }

  static Stmt AppendStmt(Array<Stmt> seq, const Stmt &tail) {
    seq.push_back(tail);
    return MakeSeq(seq);
  }

  static Stmt BindStmt(const Var &var, const PrimExpr &value, Stmt body) {
    return SeqStmt({tirx::Bind(var, value), body});
  }

  static int64_t GetVarStride(const PrimExpr &expr, const VarNode *var) {
    if (const auto *v = expr.as<VarNode>()) {
      return (v == var) ? 1 : 0;
    }
    if (expr.as<IntImmNode>() || expr.as<FloatImmNode>()) {
      return 0;
    }
    if (const auto *add = expr.as<AddNode>()) {
      int64_t sa = GetVarStride(add->a, var);
      int64_t sb = GetVarStride(add->b, var);
      return (sa < 0 || sb < 0) ? -1 : sa + sb;
    }
    if (const auto *sub = expr.as<SubNode>()) {
      int64_t sa = GetVarStride(sub->a, var);
      int64_t sb = GetVarStride(sub->b, var);
      return (sa < 0 || sb < 0) ? -1 : sa - sb;
    }
    if (const auto *mul = expr.as<MulNode>()) {
      if (const auto *imm = mul->b.as<IntImmNode>()) {
        int64_t sa = GetVarStride(mul->a, var);
        return sa < 0 ? -1 : sa * imm->value;
      }
      if (const auto *imm = mul->a.as<IntImmNode>()) {
        int64_t sb = GetVarStride(mul->b, var);
        return sb < 0 ? -1 : sb * imm->value;
      }
      int64_t sa = GetVarStride(mul->a, var);
      int64_t sb = GetVarStride(mul->b, var);
      return (sa != 0 && sb != 0) ? -1 : ((sa == 0) ? sb : sa);
    }
    if (const auto *fd = expr.as<FloorDivNode>()) {
      int64_t sa = GetVarStride(fd->a, var);
      if (sa == 0) {
        return 0;
      }
      if (const auto *imm = fd->b.as<IntImmNode>()) {
        return (sa > 0 && sa % imm->value == 0) ? sa / imm->value : -1;
      }
      return -1;
    }
    if (const auto *fm = expr.as<FloorModNode>()) {
      return GetVarStride(fm->a, var) == 0 ? 0 : -1;
    }
    if (const auto *cast = expr.as<CastNode>()) {
      return GetVarStride(cast->value, var);
    }
    return -1;
  }

  static Stmt StripAttr(const Stmt &body) {
    Stmt cur = body;
    while (const auto *attr = cur.as<AttrStmtNode>()) {
      cur = attr->body;
    }
    return cur;
  }

  int64_t GetStrideInStore(const Stmt &body, const VarNode *var) {
    auto get = [&](const BufferStoreNode *store) -> int64_t {
      PrimExpr flat = store->indices[0];
      for (size_t d = 1; d < store->indices.size(); ++d) {
        flat = flat * store->buffer->shape[d] + store->indices[d];
      }
      return GetVarStride(flat, var);
    };
    Stmt b = StripAttr(body);
    if (const auto *store = b.as<BufferStoreNode>()) {
      return get(store);
    }
    if (const auto *seq = b.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (const auto *store = s.as<BufferStoreNode>()) {
          return get(store);
        }
      }
    }
    return -1;
  }

  PrimExpr SubstituteIndex(const PrimExpr &index) {
    class Replacer : public ExprMutator {
      const VarNode *pv_;
      PrimExpr pr_;
      const VarNode *ov_;
      PrimExpr or_;

    public:
      Replacer(const VarNode *pv, PrimExpr pr, const VarNode *ov, PrimExpr or_)
          : pv_(pv), pr_(pr), ov_(ov), or_(or_) {}

      PrimExpr VisitExpr_(const VarNode *op) override {
        if (pv_ && op == pv_) {
          return pr_;
        }
        if (ov_ && op == ov_) {
          return or_;
        }
        return ffi::GetRef<PrimExpr>(op);
      }
    };

    PrimExpr par_repl = chunk_var_ * IntImm(DataType::Int(32), vreg_size_);
    PrimExpr outer_repl = outer_parallel_var_
                              ? PrimExpr(outer_serial_var_)
                              : PrimExpr(IntImm(DataType::Int(32), 0));
    return Replacer(parallel_var_, par_repl, outer_parallel_var_,
                    outer_repl)(index);
  }

  static PrimExpr MakeAccessPtr(const Buffer &buffer,
                                const Array<PrimExpr> &idx, int rw_mask) {
    return Call(DataType::Handle(), tl::access_ptr(),
                {BufferLoad(buffer, idx), IntImm(DataType::Int(32), 1),
                 IntImm(DataType::Int(32), rw_mask)});
  }

  PrimExpr MakeAddressOf(const Buffer &buffer,
                         const Array<PrimExpr> &orig_indices, int rw_mask) {
    Array<PrimExpr> idx;
    for (const auto &i : orig_indices) {
      idx.push_back(SubstituteIndex(i));
    }
    return MakeAccessPtr(buffer, idx, rw_mask);
  }

  Var EmitUnary(const PrimExpr &value, DataType dtype, const Op &op,
                Array<Stmt> *stmts) {
    Var src = DecomposeExpr(value, dtype, stmts);
    Var dst = MakeReg(dtype);
    stmts->push_back(
        tirx::Bind(dst, Call(dst->dtype, op,
                             {src, mask_var_, StringImm("MODE_ZEROING")})));
    return dst;
  }

  Var EmitBinary(const PrimExpr &lhs_expr, const PrimExpr &rhs_expr,
                 DataType dtype, const Op &op, Array<Stmt> *stmts) {
    Var lhs = DecomposeExpr(lhs_expr, dtype, stmts);
    Var rhs = DecomposeExpr(rhs_expr, dtype, stmts);
    Var dst = MakeReg(dtype);
    stmts->push_back(tirx::Bind(
        dst, Call(dst->dtype, op,
                  {lhs, rhs, mask_var_, StringImm("MODE_ZEROING")})));
    return dst;
  }

  Var DecomposeExpr(const PrimExpr &expr, DataType dtype, Array<Stmt> *stmts) {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      Var reg = MakeReg(dtype);
      bool depends_on_vreg = false;
      if (parallel_var_) {
        for (const auto &idx : load->indices) {
          if (GetVarStride(idx, parallel_var_) != 0) {
            depends_on_vreg = true;
            break;
          }
        }
      }

      PrimExpr addr = MakeAddressOf(load->buffer, load->indices, /*rw_mask=*/1);
      PrimExpr vld_call =
          Call(reg->dtype, simd_vld(),
               {addr, StringImm(depends_on_vreg ? "NORM" : "BRC_B32")});
      if (!depends_on_vreg && outer_parallel_var_) {
        scalar_loads_.push_back(tirx::Bind(reg, vld_call));
      } else {
        stmts->push_back(tirx::Bind(reg, vld_call));
      }
      return reg;
    }

    if (const auto *n = expr.as<AddNode>()) {
      return EmitBinary(n->a, n->b, dtype, simd_vadd(), stmts);
    }
    if (const auto *n = expr.as<SubNode>()) {
      if (is_zero(n->a)) {
        return EmitUnary(n->b, dtype, simd_vneg(), stmts);
      }
      return EmitBinary(n->a, n->b, dtype, simd_vsub(), stmts);
    }
    if (const auto *n = expr.as<MulNode>()) {
      return EmitBinary(n->a, n->b, dtype, simd_vmul(), stmts);
    }
    if (const auto *n = expr.as<DivNode>()) {
      return EmitBinary(n->a, n->b, dtype, simd_vdiv(), stmts);
    }

    if (const auto *call = expr.as<CallNode>()) {
      if (const auto *opn = call->op.as<OpNode>()) {
        std::string n = opn->name;
        if (n == "tir.exp" && call->args.size() == 1) {
          return EmitUnary(call->args[0], dtype, simd_vexp(), stmts);
        }
        if ((n == "tir.log" || n == "tir.ln") && call->args.size() == 1) {
          return EmitUnary(call->args[0], dtype, simd_vln(), stmts);
        }
        if (n == "tir.sqrt" && call->args.size() == 1) {
          return EmitUnary(call->args[0], dtype, simd_vsqrt(), stmts);
        }
        if (n == "tir.fabs" && call->args.size() == 1) {
          return EmitUnary(call->args[0], dtype, simd_vabs(), stmts);
        }
      }
    }

    if (expr.as<FloatImmNode>() || expr.as<IntImmNode>()) {
      Var reg = MakeReg(dtype);
      stmts->push_back(
          tirx::Bind(reg, Call(reg->dtype, simd_vdup(),
                               {expr, mask_var_, StringImm("MODE_ZEROING")})));
      return reg;
    }

    LOG(FATAL) << "AscendSimdVFLowerParallel: unsupported expr: " << expr;
    return Var();
  }

  Stmt LowerBufferStore(const BufferStoreNode *store, DataType dtype) {
    Array<Stmt> stmts;
    Var result = DecomposeExpr(store->value, dtype, &stmts);
    Stmt vsts_stmt = Evaluate(
        Call(DataType::Void(), simd_vsts(),
             {MakeAddressOf(store->buffer, store->indices, /*rw_mask=*/2),
              result, mask_var_, StringImm("NORM_B32")}));
    return AppendStmt(stmts, vsts_stmt);
  }

  bool CanLowerBody(const Stmt &body) {
    Stmt b = StripAttr(body);
    if (b.as<BufferStoreNode>()) {
      return true;
    }
    if (const auto *s = b.as<SeqStmtNode>()) {
      for (const Stmt &st : s->seq) {
        if (!st.as<BufferStoreNode>()) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  Stmt LowerParallelBody(const Stmt &body, DataType dtype) {
    Stmt b = StripAttr(body);
    if (const auto *s = b.as<BufferStoreNode>()) {
      return LowerBufferStore(s, dtype);
    }
    if (const auto *s = b.as<SeqStmtNode>()) {
      Array<Stmt> lowered;
      for (const Stmt &st : s->seq) {
        lowered.push_back(LowerBufferStore(st.as<BufferStoreNode>(), dtype));
      }
      return MakeSeq(lowered);
    }
    LOG(FATAL) << "AscendSimdVFLowerParallel: bad parallel body: "
               << b->GetTypeKey();
    return Stmt();
  }

  DataType InferDType(const Stmt &body) {
    Stmt b = StripAttr(body);
    if (const auto *s = b.as<BufferStoreNode>()) {
      return s->buffer->dtype;
    }
    if (const auto *s = b.as<SeqStmtNode>()) {
      for (const Stmt &st : s->seq) {
        if (const auto *bs = st.as<BufferStoreNode>()) {
          return bs->buffer->dtype;
        }
      }
    }
    return DataType::Float(32);
  }

  Stmt VisitStmt_(const SBlockNode *op) override {
    if (op->name_hint != "SIMD_VF") {
      return StmtExprMutator::VisitStmt_(op);
    }

    bool saved = inside_simdvf_;
    int saved_cnt = frag_counter_;
    inside_simdvf_ = true;
    frag_counter_ = 0;

    Stmt new_body = VisitStmt(op->body);

    inside_simdvf_ = saved;
    frag_counter_ = saved_cnt;

    if (new_body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }

    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  op->init, op->alloc_buffers, op->match_buffers,
                  op->annotations);
  }

  Stmt VisitStmt_(const ForNode *op) override {
    if (!inside_simdvf_ || op->kind != ForKind::kParallel) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt stripped = StripAttr(op->body);
    const auto *inner = stripped.as<ForNode>();
    if (inner && inner->kind == ForKind::kParallel) {
      if (const auto *t = inner->body.as<ForNode>()) {
        if (t->kind == ForKind::kParallel) {
          LOG(FATAL) << "3D+ parallel not supported";
        }
      }
      if (!CanLowerBody(inner->body)) {
        return StmtExprMutator::VisitStmt_(op);
      }

      int64_t outer_stride = GetStrideInStore(inner->body, op->loop_var.get());
      int64_t inner_stride =
          GetStrideInStore(inner->body, inner->loop_var.get());

      if (inner_stride == 1) {
        return LowerParallel(inner, op);
      }
      if (outer_stride == 1) {
        return LowerParallel(op, inner);
      }

      LOG(FATAL) << "2D parallel requires one dimension with stride=1, got "
                 << "outer stride=" << outer_stride
                 << " inner stride=" << inner_stride;
    }

    if (!CanLowerBody(op->body)) {
      return StmtExprMutator::VisitStmt_(op);
    }
    return LowerParallel(op, nullptr);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (!inside_simdvf_) {
      return StmtExprMutator::VisitStmt_(op);
    }

    const auto *call = op->value.as<CallNode>();
    if (!call) {
      return StmtExprMutator::VisitStmt_(op);
    }

    if (const auto *opnode = call->op.as<OpNode>()) {
      if (opnode->name == "tl.tileop.fill") {
        return LowerFill(call);
      }
      if (IsAscendCopyCall(call)) {
        return LowerCopy(call);
      }
      if (opnode->name == "tl.tileop.reduce") {
        return LowerReduce(call);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt LowerFill(const CallNode *call) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    BufferRegion buf_region = NormalizeToBufferRegion(call->args[0]);
    Buffer buf = buf_region->buffer;
    Array<Range> ranges = buf_region->region;
    PrimExpr value = call->args[1];
    DataType dtype = buf->dtype;

    if (value->dtype != dtype) {
      value = Cast(dtype, value);
    }

    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();

    // Fill the *region*, not the physical allocation. After
    // MaterializeMultiBuffer, an in-loop T.fill of a logical UB becomes a
    // leading unit-extent version slice (e.g. [v:v+1, 0:64] on shape [2, 64]).
    // That is a contiguous fill whose element count is divisible by the VReg
    // size and whose base address is 32-byte aligned (the per-version stride
    // is a multiple of the VReg byte width).
    int64_t total = 0;
    std::vector<int64_t> extents;
    ICHECK(CanLowerFillRegion(buf, ranges, vreg_size_, &total, &extents))
        << "T.fill inside SimdVF requires a contiguous region whose element "
           "count is divisible by VReg size ("
        << vreg_size_ << "), got region " << buf_region << " of buffer " << buf;

    int64_t nchunks = total / vreg_size_;
    Var frag = MakeReg(dtype);
    Var mask("mask", DataType::Bool(256));
    Var chunk("vchunk", DataType::Int(32));

    PrimExpr mask_value =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});
    PrimExpr flat_idx = chunk * IntImm(DataType::Int(32), vreg_size_);
    Array<PrimExpr> idxs = MakeRegionIndices(ranges, extents, flat_idx);

    PrimExpr addr = MakeAccessPtr(buf, idxs, /*rw_mask=*/2);
    Stmt store = Evaluate(Call(DataType::Void(), simd_vsts(),
                               {addr, frag, mask, StringImm("NORM_B32")}));
    Stmt loop =
        For(chunk, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, store);
    PrimExpr vdup_call = Call(frag->dtype, simd_vdup(),
                              {value, mask, StringImm("MODE_ZEROING")});

    Stmt result = MakeSeq(
        {tirx::Bind(mask, mask_value), tirx::Bind(frag, vdup_call), loop});
    frag_counter_ = saved_frag_counter;
    return result;
  }

  Stmt LowerCopy(const CallNode *call) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    Buffer src_buf = src_region->buffer;
    Buffer dst_buf = dst_region->buffer;
    Array<Range> src_ranges = src_region->region;
    Array<Range> dst_ranges = dst_region->region;
    DataType src_dtype = src_buf->dtype;
    DataType dst_dtype = dst_buf->dtype;
    bool is_cast = (dst_dtype != src_dtype);

    ICHECK(!is_cast || (src_dtype.is_float() &&
                        (dst_dtype.is_float16() || dst_dtype.is_bfloat16())))
        << "T.copy inside SimdVF: cast copy only supports f32->bf16 or "
           "f32->half";

    vreg_size_ = ComputeVRegSize(src_dtype);
    int src_elem_bits = src_dtype.bits();
    int dst_elem_bits = dst_dtype.bits();

    auto is_full = [](const Array<Range> &ranges, const Array<PrimExpr> &shape,
                      int64_t *total) -> bool {
      *total = 1;
      for (size_t i = 0; i < ranges.size(); i++) {
        const auto *mi = ranges[i]->min.as<IntImmNode>();
        const auto *ei = ranges[i]->extent.as<IntImmNode>();
        const auto *si = shape[i].as<IntImmNode>();
        if (!mi || !ei || !si || mi->value != 0 || ei->value != si->value) {
          return false;
        }
        *total *= ei->value;
      }
      return true;
    };

    int64_t src_total = 0, dst_total = 0;
    ICHECK(is_full(src_ranges, src_buf->shape, &src_total) &&
           is_full(dst_ranges, dst_buf->shape, &dst_total))
        << "T.copy inside SimdVF requires full-buffer regions";
    ICHECK_EQ(src_total, dst_total) << "T.copy inside SimdVF requires src and "
                                       "dst to have same element count";
    ICHECK_EQ(src_total % vreg_size_, 0)
        << "T.copy inside SimdVF requires total elements divisible by VReg "
           "size ("
        << vreg_size_ << "), got " << src_total;

    int64_t nchunks = src_total / vreg_size_;
    Var load_frag = MakeReg(src_dtype);
    Var cast_frag;
    if (is_cast) {
      cast_frag = MakeReg(dst_dtype);
    }

    Var chunk("vchunk", DataType::Int(32));
    mask_var_ = Var("mask", DataType::Bool(256));
    int mask_elem_bits = is_cast ? dst_elem_bits : src_elem_bits;
    PrimExpr mask_value =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), mask_elem_bits), StringImm("PAT_ALL")});

    auto flat_addr = [&](const Buffer &buf, int ndim, int rw_mask) -> PrimExpr {
      PrimExpr flat_idx = chunk * IntImm(DataType::Int(32), vreg_size_);
      Array<PrimExpr> idxs;
      if (ndim == 1) {
        idxs.push_back(flat_idx);
      } else {
        PrimExpr r = flat_idx;
        for (int i = ndim - 1; i >= 1; i--) {
          idxs.insert(idxs.begin(), FloorMod(r, buf->shape[i]));
          r = FloorDiv(r, buf->shape[i]);
        }
        idxs.insert(idxs.begin(), r);
      }
      return MakeAccessPtr(buf, idxs, rw_mask);
    };

    PrimExpr src_addr = flat_addr(src_buf, src_ranges.size(), /*rw_mask=*/1);
    PrimExpr dst_addr = flat_addr(dst_buf, dst_ranges.size(), /*rw_mask=*/2);
    PrimExpr vld_call =
        Call(load_frag->dtype, simd_vld(), {src_addr, StringImm("NORM")});

    Array<Stmt> body;
    body.push_back(tirx::Bind(load_frag, vld_call));
    if (is_cast) {
      PrimExpr vcvt_call = Call(cast_frag->dtype, simd_vcvt(),
                                {load_frag, mask_var_, StringImm("ROUND_R"),
                                 StringImm("RS_ENABLE"), StringImm("PART_EVEN"),
                                 StringImm("MODE_ZEROING")});
      body.push_back(tirx::Bind(cast_frag, vcvt_call));
      body.push_back(Evaluate(
          Call(DataType::Void(), simd_vsts(),
               {dst_addr, cast_frag, mask_var_, StringImm("NORM_B32")})));
    } else {
      body.push_back(Evaluate(
          Call(DataType::Void(), simd_vsts(),
               {dst_addr, load_frag, mask_var_, StringImm("NORM_B32")})));
    }

    Stmt loop = For(chunk, IntImm(DataType::Int(32), 0),
                    IntImm(DataType::Int(32), nchunks), ForKind::kSerial,
                    MakeSeq(body));
    Stmt result = MakeSeq({tirx::Bind(mask_var_, mask_value), loop});
    frag_counter_ = saved_frag_counter;
    return result;
  }

  Stmt LowerReduce(const CallNode *call) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    BufferRegion src_region = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    Buffer src = src_region->buffer;
    Buffer dst = dst_region->buffer;
    Array<Range> sr = src_region->region;
    Array<Range> dr = dst_region->region;

    std::string rtype = call->args[2].as<StringImmNode>()->value;
    int dim = static_cast<int>(call->args[3].as<IntImmNode>()->value);
    bool clear = call->args[4].as<Bool>().value();

    DataType dtype = src->dtype;
    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();

    ICHECK(sr.size() == 2 && dr.size() == 1 && (dim == 0 || dim == 1))
        << "T.reduce inside SimdVF requires 2D src -> 1D dst, dim=0 or 1";

    auto is_full = [](const Array<Range> &ranges, const Array<PrimExpr> &shape,
                      int64_t *total) -> bool {
      *total = 1;
      for (size_t i = 0; i < ranges.size(); i++) {
        const auto *mi = ranges[i]->min.as<IntImmNode>();
        const auto *ei = ranges[i]->extent.as<IntImmNode>();
        const auto *si = shape[i].as<IntImmNode>();
        if (!mi || !ei || !si || mi->value != 0 || ei->value != si->value) {
          return false;
        }
        *total *= ei->value;
      }
      return true;
    };

    int64_t st = 0, dt = 0;
    ICHECK(is_full(sr, src->shape, &st) && is_full(dr, dst->shape, &dt))
        << "T.reduce inside SimdVF requires full-buffer regions";

    int64_t reduce_extent = Downcast<IntImm>(sr[dim]->extent)->value;
    int64_t outer_extent = Downcast<IntImm>(sr[1 - dim]->extent)->value;
    ICHECK_EQ(reduce_extent % vreg_size_, 0)
        << "T.reduce requires dim extent multiple of VReg (" << vreg_size_
        << "), got " << reduce_extent;
    int64_t nchunks = reduce_extent / vreg_size_;

    const Op &element_op = (rtype == "sum") ? simd_vadd() : simd_vmax();
    const Op &reduce_op = (rtype == "sum") ? simd_vcadd() : simd_vcmax();
    PrimExpr init_value = (rtype == "sum")
                              ? PrimExpr(FloatImm(dtype, 0.0))
                              : PrimExpr(FloatImm(dtype, -3.402823e+38));

    auto i32 = [](int64_t v) { return IntImm(DataType::Int(32), v); };
    auto addr1d = [&](const Buffer &buf, const PrimExpr &i0, int rw_mask) {
      return MakeAccessPtr(buf, {i0}, rw_mask);
    };
    auto addr2d = [&](const Buffer &buf, const PrimExpr &i0, const PrimExpr &i1,
                      int rw_mask) {
      return MakeAccessPtr(buf, {i0, i1}, rw_mask);
    };

    mask_var_ = Var("mask", DataType::Bool(256));
    PrimExpr mask_value =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});
    Var rep("vrepeat", DataType::Int(32));

    if (dim == 1) {
      Array<Stmt> row;
      Var acc = MakeReg(dtype);
      row.push_back(tirx::Bind(
          acc, Call(acc->dtype, simd_vdup(),
                    {init_value, mask_var_, StringImm("MODE_ZEROING")})));

      for (int64_t ck = 0; ck < nchunks; ++ck) {
        PrimExpr col = i32(ck * vreg_size_);
        Var loaded = MakeReg(dtype);
        Var reduced = MakeReg(dtype);
        Var next_acc = MakeReg(dtype);
        row.push_back(tirx::Bind(
            loaded,
            Call(loaded->dtype, simd_vld(),
                 {addr2d(src, rep, col, /*rw_mask=*/1), StringImm("NORM")})));
        row.push_back(tirx::Bind(
            reduced, Call(reduced->dtype, reduce_op,
                          {loaded, mask_var_, StringImm("MODE_ZEROING")})));
        row.push_back(tirx::Bind(next_acc, Call(next_acc->dtype, element_op,
                                                {acc, reduced, mask_var_,
                                                 StringImm("MODE_ZEROING")})));
        acc = next_acc;
      }

      PrimExpr dst_off = rep * i32(vreg_size_);
      if (!clear) {
        Var old = MakeReg(dtype);
        Var merged = MakeReg(dtype);
        row.push_back(tirx::Bind(old, Call(old->dtype, simd_vld(),
                                           {addr1d(dst, dst_off, /*rw_mask=*/1),
                                            StringImm("BRC_B32")})));
        row.push_back(tirx::Bind(
            merged, Call(merged->dtype, element_op,
                         {acc, old, mask_var_, StringImm("MODE_ZEROING")})));
        acc = merged;
      }
      row.push_back(Evaluate(Call(DataType::Void(), simd_vsts(),
                                  {addr1d(dst, dst_off, /*rw_mask=*/2), acc,
                                   mask_var_, StringImm("NORM_B32")})));

      Stmt row_body =
          MakeSeq({tirx::Bind(mask_var_, mask_value), MakeSeq(row)});
      Stmt result =
          For(rep, i32(0), i32(outer_extent), ForKind::kSerial, row_body);
      frag_counter_ = saved_frag_counter;
      return result;
    }

    Var chunk("vchunk", DataType::Int(32));
    PrimExpr col = chunk * i32(vreg_size_);
    Array<Stmt> chunk_body;
    Var acc = MakeReg(dtype);
    chunk_body.push_back(tirx::Bind(
        acc,
        Call(acc->dtype, simd_vld(),
             {addr2d(src, i32(0), col, /*rw_mask=*/1), StringImm("NORM")})));

    for (int64_t row_idx = 1; row_idx < outer_extent; ++row_idx) {
      Var loaded = MakeReg(dtype);
      Var next_acc = MakeReg(dtype);
      chunk_body.push_back(tirx::Bind(
          loaded, Call(loaded->dtype, simd_vld(),
                       {addr2d(src, i32(row_idx), col, /*rw_mask=*/1),
                        StringImm("NORM")})));
      chunk_body.push_back(tirx::Bind(
          next_acc, Call(next_acc->dtype, element_op,
                         {acc, loaded, mask_var_, StringImm("MODE_ZEROING")})));
      acc = next_acc;
    }

    if (!clear) {
      Var old = MakeReg(dtype);
      Var merged = MakeReg(dtype);
      chunk_body.push_back(tirx::Bind(
          old, Call(old->dtype, simd_vld(),
                    {addr1d(dst, col, /*rw_mask=*/1), StringImm("BRC_B32")})));
      chunk_body.push_back(tirx::Bind(
          merged, Call(merged->dtype, element_op,
                       {acc, old, mask_var_, StringImm("MODE_ZEROING")})));
      acc = merged;
    }
    chunk_body.push_back(Evaluate(Call(DataType::Void(), simd_vsts(),
                                       {addr1d(dst, col, /*rw_mask=*/2), acc,
                                        mask_var_, StringImm("NORM_B32")})));

    Stmt loop =
        For(chunk, i32(0), i32(nchunks), ForKind::kSerial, MakeSeq(chunk_body));
    Stmt result = MakeSeq({tirx::Bind(mask_var_, mask_value), loop});
    frag_counter_ = saved_frag_counter;
    return result;
  }

  Stmt LowerParallel(const ForNode *vreg_for, const ForNode *serial_for) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    int64_t vreg_ext = Downcast<IntImm>(vreg_for->extent)->value;
    int64_t serial_ext =
        serial_for ? Downcast<IntImm>(serial_for->extent)->value : 1;

    DataType dtype = InferDType(vreg_for->body);
    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();
    ICHECK_EQ(vreg_ext % vreg_size_, 0)
        << "T.Parallel inside SimdVF requires VReg extent divisible by VReg "
           "size ("
        << vreg_size_ << "), got " << vreg_ext;
    int64_t nchunks = vreg_ext / vreg_size_;

    const VarNode *saved_pv = parallel_var_;
    const VarNode *saved_ov = outer_parallel_var_;
    int64_t saved_extent = parallel_extent_;

    parallel_var_ = vreg_for->loop_var.get();
    parallel_extent_ = vreg_ext;
    if (serial_for) {
      outer_parallel_var_ = serial_for->loop_var.get();
      outer_serial_var_ = Var("vserial", DataType::Int(32));
    } else {
      outer_parallel_var_ = nullptr;
    }

    chunk_var_ = Var("vchunk", DataType::Int(32));
    mask_var_ = Var("mask", DataType::Bool(256));
    scalar_loads_.clear();

    PrimExpr mask_value =
        Call(DataType::Bool(256), simd_pset(),
             {IntImm(DataType::Int(32), elem_bits_), StringImm("PAT_ALL")});

    Stmt body = LowerParallelBody(vreg_for->body, dtype);
    Stmt masked_body = BindStmt(mask_var_, mask_value, body);
    Stmt inner_loop =
        For(chunk_var_, IntImm(DataType::Int(32), 0),
            IntImm(DataType::Int(32), nchunks), ForKind::kSerial, masked_body);

    Stmt per_iter_body = inner_loop;
    if (!scalar_loads_.empty()) {
      Array<Stmt> seq;
      for (const Stmt &s : scalar_loads_) {
        seq.push_back(s);
      }
      seq.push_back(per_iter_body);
      per_iter_body = MakeSeq(seq);
      scalar_loads_.clear();
    }

    Stmt result;
    if (serial_for) {
      result = For(outer_serial_var_, IntImm(DataType::Int(32), 0),
                   IntImm(DataType::Int(32), serial_ext), ForKind::kSerial,
                   per_iter_body);
    } else {
      result = per_iter_body;
    }

    parallel_var_ = saved_pv;
    parallel_extent_ = saved_extent;
    outer_parallel_var_ = saved_ov;
    frag_counter_ = saved_frag_counter;
    return result;
  }
};

using namespace tirx::transform;

tvm::transform::Pass AscendSimdVFLowerParallel() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AscendSimdVFLowerParallel::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendSimdVFLowerParallel", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendSimdVFLowerParallel",
                        AscendSimdVFLowerParallel);
}

} // namespace tl
} // namespace tvm
