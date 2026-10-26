#include "compat/analyzer.h"
// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_simdvf_lower_parallel.cc
 * \brief Lower T.Parallel loops inside SIMD_VF blocks to Ascend SIMD
 *        return-value vector instructions.
 */

#include "op/prim.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/transform/native_visitor.h"
#include "compat/expr.h"
#include "op/utils.h"

#include <functional>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using ffi::Array;

class AscendSimdVFLowerParallel : public ascend::NativeMutator {
public:
  static tirx::Function Substitute(tirx::Function function) {
    if (!function->body)
      return function;
    auto substituter = ffi::make_object<AscendSimdVFLowerParallel>();
    auto original = function->body.value();
    function.CopyOnWrite()->body = SeqStmt(substituter->VisitStmt(original));
    return function;
  }

private:
  bool inside_simdvf_{false};

  const VarNode *parallel_var_{nullptr};
  int64_t parallel_extent_{0};

  const VarNode *outer_parallel_var_{nullptr};
  PrimVar outer_serial_var_{ffi::UnsafeInit{}};

  int64_t vreg_size_{0};
  PrimVar chunk_var_{ffi::UnsafeInit{}};
  PrimVar mask_var_{ffi::UnsafeInit{}};
  PrimExpr parallel_begin_ = IntImm::Int32(0);
  PrimExpr outer_begin_ = IntImm::Int32(0);

  int frag_counter_{0};
  int64_t elem_bits_{0};
  std::vector<Stmt> scalar_loads_;

  static int64_t ComputeVRegSize(PrimType dtype) {
    return 8 * 32 / (dtype.bits() / 8);
  }

  static PrimExpr MatchIndexDType(const PrimExpr &value, PrimType dtype) {
    return value.ty() == dtype ? value : Cast(dtype, value);
  }

  static Array<PrimExpr> InferCompactStrides(const Array<PrimExpr> &shape) {
    int ndim = static_cast<int>(shape.size());
    std::vector<PrimExpr> strides(ndim, IntImm::Int32(1));
    PrimExpr stride = IntImm(shape.back().ty(), 1);
    for (int i = ndim - 1; i >= 0; --i) {
      strides[i] = stride;
      stride = stride * shape[i];
    }
    return Array<PrimExpr>(strides.begin(), strides.end());
  }

  // True when a row-major walk of the region is a contiguous element sequence,
  // so a VReg-sized vsts from each linearized chunk address is legal. Leading
  // unit-extent dims (the in-loop multi-buffer version slot) are skipped.
  static bool IsContiguousRegion(const TensorVar &buf,
                                 const std::vector<int64_t> &extents) {
    int ndim = static_cast<int>(extents.size());
    ICHECK_EQ(static_cast<int>(buf.shape().size()), ndim);
    auto pattern = tirx::MatchShapeStridePattern(buf.ty());
    if (!pattern)
      return false;
    const auto &strides = pattern.value().get<1>();
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
      extents->push_back(ext->value.as<int64_t>().value());
      *total *= ext->value.as<int64_t>().value();
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
    PrimType dtype = flat_idx.ty();
    if (ndim == 1) {
      return {ranges[0]->begin +
              MatchIndexDType(flat_idx, ranges[0]->begin.ty())};
    }
    Array<PrimExpr> idxs;
    PrimExpr remaining = flat_idx;
    for (int i = ndim - 1; i >= 1; --i) {
      PrimExpr ext = IntImm(dtype, extents[i]);
      PrimExpr coord = FloorMod(remaining, ext);
      idxs.insert(idxs.begin(),
                  ranges[i]->begin +
                      MatchIndexDType(coord, ranges[i]->begin.ty()));
      remaining = FloorDiv(remaining, ext);
    }
    idxs.insert(idxs.begin(),
                ranges[0]->begin +
                    MatchIndexDType(remaining, ranges[0]->begin.ty()));
    return idxs;
  }

  // vsts_norm lowers to asc_storealign and requires a 32-byte-aligned base.
  // Prove the linearized region byte offset is a multiple of 32 (mirrors
  // insert_nd2nz.cc:HasAlignedScatterBase); symbolic version indices remain
  // provable when the per-version stride is 32-byte aligned.
  static void ValidateFillBaseAlignment(const TensorVar &buf,
                                        const Array<Range> &ranges) {
    sym::Analyzer analyzer;
    PrimType index_dtype = PrimType::Int(64);
    Array<PrimExpr> indices;
    for (const auto &range : ranges)
      indices.push_back(range->begin);
    auto offsets = buf.ty()->ElemOffset(indices);
    TVM_FFI_CHECK_EQ(offsets.size(), 1, ValueError);
    PrimExpr offset = tl::CastValue(offsets[0], index_dtype);
    PrimExpr element_bytes =
        tl::MakeConst(index_dtype, (buf.dtype().bits() + 7) / 8);
    PrimExpr alignment = tl::MakeConst(index_dtype, 32);
    if (!tl::arith::CanProveEqual(analyzer,
                                  FloorMod(offset * element_bytes, alignment),
                                  tl::MakeConst(index_dtype, 0))) {
      TVM_FFI_THROW(ValueError)
          << "T.fill inside SimdVF requires a 32-byte-aligned region base "
             "address, got region ["
          << ranges << "] of buffer " << buf->name
          << " with linearized byte offset " << offset * element_bytes
          << " (element size " << (buf.dtype().bits() + 7) / 8
          << " bytes). Use a region whose base is 32-byte aligned.";
    }
  }

  static bool CanLowerFillRegion(const TensorVar &buf,
                                 const Array<Range> &ranges, int64_t vreg_size,
                                 int64_t *total,
                                 std::vector<int64_t> *extents) {
    if (ranges.empty() || static_cast<int>(ranges.size()) !=
                              static_cast<int>(buf.shape().size())) {
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

  PrimVar MakeReg(PrimType dtype) {
    std::string name = "vreg_" + std::to_string(frag_counter_++);
    int vl = ComputeVRegSize(dtype);
    return PrimVar(name, dtype.WithLanes(vl));
  }

  static Stmt MakeSeq(Array<Stmt> seq) {
    if (seq.empty()) {
      return Evaluate(IntImm(PrimType::Int(32), 0));
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

  static Stmt BindStmt(const PrimVar &var, const Expr &value, Stmt body) {
    return SeqStmt({Bind(var, value), body});
  }

  static int64_t GetVarStride(const PrimExpr &expression, const VarNode *node) {
    auto variable = ffi::GetRef<Var>(node).as_or_throw<PrimVar>();
    sym::Analyzer analyzer;
    auto next = tl::Substitute(
        expression, {{variable, variable + IntImm(variable.ty(), 1)}});
    return GetConstInt(tl::arith::Simplify(analyzer, next - expression))
        .value_or(-1);
  }

  static Stmt StripAttr(const Stmt &body) {
    Stmt cur = body;
    while (const auto *sequence = cur.as<SeqStmtNode>()) {
      if (sequence->seq.size() != 1)
        break;
      cur = sequence->seq[0];
    }
    return cur;
  }

  int64_t GetStrideInStore(const Stmt &body, const VarNode *var) {
    auto get = [&](const TensorStoreNode *store) -> int64_t {
      auto offsets =
          store->dest.as_or_throw<TensorVar>().ty()->ElemOffset(store->indices);
      TVM_FFI_CHECK_EQ(offsets.size(), 1, ValueError);
      PrimExpr flat = offsets[0];
      return GetVarStride(flat, var);
    };
    Stmt b = StripAttr(body);
    if (const auto *store = b.as<TensorStoreNode>()) {
      return get(store);
    }
    if (const auto *seq = b.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        if (const auto *store = s.as<TensorStoreNode>()) {
          return get(store);
        }
      }
    }
    return -1;
  }

  PrimExpr SubstituteIndex(const PrimExpr &index) {
    ffi::Map<PrimVar, PrimExpr> replacements;
    if (parallel_var_) {
      auto variable = ffi::GetRef<Var>(parallel_var_).as_or_throw<PrimVar>();
      replacements.Set(
          variable,
          parallel_begin_ +
              tl::CastValue(chunk_var_ * IntImm(chunk_var_.ty(), vreg_size_),
                            variable.ty()));
    }
    if (outer_parallel_var_) {
      auto variable =
          ffi::GetRef<Var>(outer_parallel_var_).as_or_throw<PrimVar>();
      replacements.Set(variable, outer_begin_ + tl::CastValue(outer_serial_var_,
                                                              variable.ty()));
    }
    return tl::Substitute(index, replacements);
  }

  static Expr MakeAccessPtr(const TensorVar &buffer, const Array<PrimExpr> &idx,
                            int rw_mask) {
    return Call(std::nullopt, tl::access_ptr(),
                {TensorLoad(buffer, idx), IntImm(PrimType::Int(32), 1),
                 IntImm(PrimType::Int(32), rw_mask)});
  }

  Expr MakeAddressOf(const TensorVar &buffer,
                     const Array<PrimExpr> &orig_indices, int rw_mask) {
    Array<PrimExpr> idx;
    for (const auto &i : orig_indices) {
      idx.push_back(SubstituteIndex(i));
    }
    return MakeAccessPtr(buffer, idx, rw_mask);
  }

  PrimVar EmitUnary(const PrimExpr &value, PrimType dtype, const Op &op,
                    Array<Stmt> *stmts) {
    PrimVar src = DecomposeExpr(value, dtype, stmts);
    PrimVar dst = MakeReg(dtype);
    stmts->push_back(Bind(
        dst, Call(dst.ty(), op, {src, mask_var_, StringImm("MODE_ZEROING")})));
    return dst;
  }

  PrimVar EmitBinary(const PrimExpr &lhs_expr, const PrimExpr &rhs_expr,
                     PrimType dtype, const Op &op, Array<Stmt> *stmts) {
    PrimVar lhs = DecomposeExpr(lhs_expr, dtype, stmts);
    PrimVar rhs = DecomposeExpr(rhs_expr, dtype, stmts);
    PrimVar dst = MakeReg(dtype);
    stmts->push_back(
        Bind(dst, Call(dst.ty(), op,
                       {lhs, rhs, mask_var_, StringImm("MODE_ZEROING")})));
    return dst;
  }

  PrimVar DecomposeExpr(const PrimExpr &expr, PrimType dtype,
                        Array<Stmt> *stmts) {
    if (const auto *load = expr.as<TensorLoadNode>()) {
      PrimVar reg = MakeReg(dtype);
      int64_t stride = 0;
      if (parallel_var_) {
        auto offsets = load->source.as_or_throw<TensorVar>().ty()->ElemOffset(
            load->indices);
        TVM_FFI_CHECK_EQ(offsets.size(), 1, ValueError);
        stride = GetVarStride(offsets[0], parallel_var_);
        TVM_FFI_CHECK(stride == 0 || stride == 1, ValueError)
            << "SimdVF vector loads require a contiguous or broadcast source";
      }
      bool depends_on_vreg = stride != 0;

      Expr addr = MakeAddressOf(load->source.as_or_throw<TensorVar>(),
                                load->indices, /*rw_mask=*/1);
      Expr vld_call =
          Call(reg.ty(), simd_vld(),
               {addr, StringImm(depends_on_vreg ? "NORM" : "BRC_B32")});
      if (!depends_on_vreg && outer_parallel_var_) {
        scalar_loads_.push_back(Bind(reg, vld_call));
      } else {
        stmts->push_back(Bind(reg, vld_call));
      }
      return reg;
    }

    if (const auto *n = expr.as<AddNode>()) {
      return EmitBinary(n->a, n->b, dtype, simd_vadd(), stmts);
    }
    if (const auto *n = expr.as<SubNode>()) {
      if (IsZero(n->a)) {
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
        if (n == "prim.exp" && call->args.size() == 1) {
          return EmitUnary(call->args[0].as_or_throw<PrimExpr>(), dtype,
                           simd_vexp(), stmts);
        }
        if ((n == "prim.log" || n == "prim.ln") && call->args.size() == 1) {
          return EmitUnary(call->args[0].as_or_throw<PrimExpr>(), dtype,
                           simd_vln(), stmts);
        }
        if (n == "prim.sqrt" && call->args.size() == 1) {
          return EmitUnary(call->args[0].as_or_throw<PrimExpr>(), dtype,
                           simd_vsqrt(), stmts);
        }
        if (n == "prim.fabs" && call->args.size() == 1) {
          return EmitUnary(call->args[0].as_or_throw<PrimExpr>(), dtype,
                           simd_vabs(), stmts);
        }
      }
    }

    if (expr.as<FloatImmNode>() || expr.as<IntImmNode>()) {
      PrimVar reg = MakeReg(dtype);
      stmts->push_back(
          Bind(reg, Call(reg.ty(), simd_vdup(),
                         {expr, mask_var_, StringImm("MODE_ZEROING")})));
      return reg;
    }

    LOG(FATAL) << "AscendSimdVFLowerParallel: unsupported expr: " << expr;
    TVM_FFI_THROW(InternalError) << "Unreachable";
  }

  Stmt LowerBufferStore(const TensorStoreNode *store, PrimType dtype) {
    Array<Stmt> stmts;
    PrimVar result = DecomposeExpr(store->value, dtype, &stmts);
    Stmt vsts_stmt =
        Evaluate(Call(PrimType::Void(), simd_vsts(),
                      {MakeAddressOf(store->dest.as_or_throw<TensorVar>(),
                                     store->indices, /*rw_mask=*/2),
                       result, mask_var_, StringImm("NORM_B32")}));
    return AppendStmt(stmts, vsts_stmt);
  }

  bool CanLowerBody(const Stmt &body) {
    Stmt b = StripAttr(body);
    if (b.as<TensorStoreNode>()) {
      return true;
    }
    if (const auto *s = b.as<SeqStmtNode>()) {
      for (const Stmt &st : s->seq) {
        if (!st.as<TensorStoreNode>()) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  Stmt LowerParallelBody(const Stmt &body, PrimType dtype) {
    Stmt b = StripAttr(body);
    if (const auto *s = b.as<TensorStoreNode>()) {
      return LowerBufferStore(s, dtype);
    }
    if (const auto *s = b.as<SeqStmtNode>()) {
      Array<Stmt> lowered;
      for (const Stmt &st : s->seq) {
        lowered.push_back(LowerBufferStore(st.as<TensorStoreNode>(), dtype));
      }
      return MakeSeq(lowered);
    }
    LOG(FATAL) << "AscendSimdVFLowerParallel: bad parallel body: "
               << b->GetTypeKey();
    TVM_FFI_THROW(InternalError) << "Unreachable";
  }

  PrimType InferDType(const Stmt &body) {
    Stmt b = StripAttr(body);
    if (const auto *s = b.as<TensorStoreNode>()) {
      return s->dest.as_or_throw<TensorVar>().dtype();
    }
    if (const auto *s = b.as<SeqStmtNode>()) {
      for (const Stmt &st : s->seq) {
        if (const auto *bs = st.as<TensorStoreNode>()) {
          return bs->dest.as_or_throw<TensorVar>().dtype();
        }
      }
    }
    return PrimType::Float(32);
  }

  Stmt VisitStmt_(const RegionStmtNode *op) override {
    auto name = op->attrs.GetAttr<ffi::String>("name_hint").value_or("");
    if (name == "SIMT_VF")
      return ffi::GetRef<Stmt>(op);
    if (name != "SIMD_VF")
      return ascend::NativeMutator::VisitStmt_(op);
    bool saved = inside_simdvf_;
    int saved_counter = frag_counter_;
    inside_simdvf_ = true;
    frag_counter_ = 0;
    SeqStmt body(VisitStmt(op->body));
    inside_simdvf_ = saved;
    frag_counter_ = saved_counter;
    auto result = ffi::GetRef<RegionStmt>(op);
    result.CopyOnWrite()->body = body;
    return result;
  }

  Stmt VisitStmt_(const ForNode *op) override {
    if (!inside_simdvf_ || op->kind != ForKind::kParallel) {
      return ascend::NativeMutator::VisitStmt_(op);
    }

    Stmt stripped = StripAttr(op->body);
    const auto *inner = stripped.as<ForNode>();
    if (inner && inner->kind == ForKind::kParallel) {
      if (const auto *t = StripAttr(inner->body).as<ForNode>()) {
        if (t->kind == ForKind::kParallel) {
          LOG(FATAL) << "3D+ parallel not supported";
        }
      }
      if (!CanLowerBody(inner->body)) {
        return ascend::NativeMutator::VisitStmt_(op);
      }

      int64_t outer_stride = GetStrideInStore(inner->body, op->var.get());
      int64_t inner_stride = GetStrideInStore(inner->body, inner->var.get());

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
      return ascend::NativeMutator::VisitStmt_(op);
    }
    return LowerParallel(op, nullptr);
  }

  Stmt VisitStmt_(const EvaluateNode *op) override {
    if (!inside_simdvf_) {
      return ascend::NativeMutator::VisitStmt_(op);
    }

    const auto *call = op->value.as<CallNode>();
    if (!call) {
      return ascend::NativeMutator::VisitStmt_(op);
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
    return ascend::NativeMutator::VisitStmt_(op);
  }

  Stmt LowerFill(const CallNode *call) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    TensorRegion buf_region = NormalizeToBufferRegion(call->args[0]);
    TensorVar buf = buf_region->source.as_or_throw<TensorVar>();
    Array<Range> ranges = buf_region->region;
    PrimExpr value = call->args[1].as_or_throw<PrimExpr>();
    PrimType dtype = buf.dtype();

    if (value.ty() != dtype) {
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
    PrimVar frag = MakeReg(dtype);
    PrimVar mask("mask", PrimType::Bool(256));
    PrimVar chunk("vchunk", PrimType::Int(32));

    Expr mask_value =
        Call(PrimType::Bool(256), simd_pset(),
             {IntImm(PrimType::Int(32), elem_bits_), StringImm("PAT_ALL")});
    PrimExpr flat_idx = chunk * IntImm(PrimType::Int(32), vreg_size_);
    Array<PrimExpr> idxs = MakeRegionIndices(ranges, extents, flat_idx);

    Expr addr = MakeAccessPtr(buf, idxs, /*rw_mask=*/2);
    Stmt store = Evaluate(Call(PrimType::Void(), simd_vsts(),
                               {addr, frag, mask, StringImm("NORM_B32")}));
    Stmt loop = For(chunk, IntImm(PrimType::Int(32), 0),
                    IntImm(PrimType::Int(32), nchunks), IntImm::Int32(1),
                    ForKind::kDefault, SeqStmt(store));
    Expr vdup_call =
        Call(frag.ty(), simd_vdup(), {value, mask, StringImm("MODE_ZEROING")});

    Stmt result =
        MakeSeq({Bind(mask, mask_value), Bind(frag, vdup_call), loop});
    frag_counter_ = saved_frag_counter;
    return result;
  }

  Stmt LowerCopy(const CallNode *call) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    TensorRegion src_region = NormalizeToBufferRegion(call->args[0]);
    TensorRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    TensorVar src_buf = src_region->source.as_or_throw<TensorVar>();
    TensorVar dst_buf = dst_region->source.as_or_throw<TensorVar>();
    Array<Range> src_ranges = src_region->region;
    Array<Range> dst_ranges = dst_region->region;
    PrimType src_dtype = src_buf.dtype();
    PrimType dst_dtype = dst_buf.dtype();
    bool is_cast = (dst_dtype != src_dtype);

    ICHECK(!is_cast || (src_dtype == PrimType::Float(32) &&
                        (dst_dtype == PrimType::Float(16) ||
                         dst_dtype == PrimType::BFloat(16))))
        << "T.copy inside SimdVF: cast copy only supports f32->bf16 or "
           "f32->half";

    vreg_size_ = ComputeVRegSize(src_dtype);
    int src_elem_bits = src_dtype.bits();
    int dst_elem_bits = dst_dtype.bits();

    auto is_full = [](const Array<Range> &ranges, const Array<PrimExpr> &shape,
                      int64_t *total) -> bool {
      *total = 1;
      for (size_t i = 0; i < ranges.size(); i++) {
        const auto *mi = ranges[i]->begin.as<IntImmNode>();
        const auto *ei = ranges[i]->extent.as<IntImmNode>();
        const auto *si = shape[i].as<IntImmNode>();
        if (!mi || !ei || !si || mi->value != 0 ||
            ei->value.as<int64_t>().value() != si->value) {
          return false;
        }
        *total *= ei->value.as<int64_t>().value();
      }
      return true;
    };

    int64_t src_total = 0, dst_total = 0;
    ICHECK(is_full(src_ranges, src_buf.shape(), &src_total) &&
           is_full(dst_ranges, dst_buf.shape(), &dst_total))
        << "T.copy inside SimdVF requires full-buffer regions";
    ICHECK_EQ(src_total, dst_total) << "T.copy inside SimdVF requires src and "
                                       "dst to have same element count";
    ICHECK_EQ(src_total % vreg_size_, 0)
        << "T.copy inside SimdVF requires total elements divisible by VReg "
           "size ("
        << vreg_size_ << "), got " << src_total;

    int64_t nchunks = src_total / vreg_size_;
    PrimVar load_frag = MakeReg(src_dtype);
    PrimVar cast_frag{ffi::UnsafeInit{}};
    if (is_cast) {
      cast_frag = MakeReg(dst_dtype);
    }

    PrimVar chunk("vchunk", PrimType::Int(32));
    mask_var_ = PrimVar("mask", PrimType::Bool(256));
    int mask_elem_bits = is_cast ? dst_elem_bits : src_elem_bits;
    Expr mask_value =
        Call(PrimType::Bool(256), simd_pset(),
             {IntImm(PrimType::Int(32), mask_elem_bits), StringImm("PAT_ALL")});

    auto flat_addr = [&](const TensorVar &buf, int ndim, int rw_mask) -> Expr {
      PrimExpr flat_idx = chunk * IntImm(PrimType::Int(32), vreg_size_);
      Array<PrimExpr> idxs;
      if (ndim == 1) {
        idxs.push_back(flat_idx);
      } else {
        PrimExpr r = flat_idx;
        for (int i = ndim - 1; i >= 1; i--) {
          idxs.insert(idxs.begin(), FloorMod(r, buf.shape()[i]));
          r = FloorDiv(r, buf.shape()[i]);
        }
        idxs.insert(idxs.begin(), r);
      }
      return MakeAccessPtr(buf, idxs, rw_mask);
    };

    Expr src_addr = flat_addr(src_buf, src_ranges.size(), /*rw_mask=*/1);
    Expr dst_addr = flat_addr(dst_buf, dst_ranges.size(), /*rw_mask=*/2);
    Expr vld_call =
        Call(load_frag.ty(), simd_vld(), {src_addr, StringImm("NORM")});

    Array<Stmt> body;
    body.push_back(Bind(load_frag, vld_call));
    if (is_cast) {
      Expr vcvt_call = Call(cast_frag.ty(), simd_vcvt(),
                            {load_frag, mask_var_, StringImm("ROUND_R"),
                             StringImm("RS_ENABLE"), StringImm("PART_EVEN"),
                             StringImm("MODE_ZEROING")});
      body.push_back(Bind(cast_frag, vcvt_call));
      body.push_back(Evaluate(
          Call(PrimType::Void(), simd_vsts(),
               {dst_addr, cast_frag, mask_var_, StringImm("NORM_B32")})));
    } else {
      body.push_back(Evaluate(
          Call(PrimType::Void(), simd_vsts(),
               {dst_addr, load_frag, mask_var_, StringImm("NORM_B32")})));
    }

    Stmt loop = For(chunk, IntImm(PrimType::Int(32), 0),
                    IntImm(PrimType::Int(32), nchunks), IntImm::Int32(1),
                    ForKind::kDefault, SeqStmt(MakeSeq(body)));
    Stmt result = MakeSeq({Bind(mask_var_, mask_value), loop});
    frag_counter_ = saved_frag_counter;
    return result;
  }

  Stmt LowerReduce(const CallNode *call) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    TensorRegion src_region = NormalizeToBufferRegion(call->args[0]);
    TensorRegion dst_region = NormalizeToBufferRegion(call->args[1]);
    TensorVar src = src_region->source.as_or_throw<TensorVar>();
    TensorVar dst = dst_region->source.as_or_throw<TensorVar>();
    Array<Range> sr = src_region->region;
    Array<Range> dr = dst_region->region;

    std::string rtype = call->args[2].as<StringImmNode>()->value;
    int dim = GetConstInt(call->args[3]).value();
    bool clear = GetConstInt(call->args[4]).value() != 0;

    PrimType dtype = src.dtype();
    vreg_size_ = ComputeVRegSize(dtype);
    elem_bits_ = dtype.bits();

    ICHECK(sr.size() == 2 && dr.size() == 1 && (dim == 0 || dim == 1))
        << "T.reduce inside SimdVF requires 2D src -> 1D dst, dim=0 or 1";

    auto is_full = [](const Array<Range> &ranges, const Array<PrimExpr> &shape,
                      int64_t *total) -> bool {
      *total = 1;
      for (size_t i = 0; i < ranges.size(); i++) {
        const auto *mi = ranges[i]->begin.as<IntImmNode>();
        const auto *ei = ranges[i]->extent.as<IntImmNode>();
        const auto *si = shape[i].as<IntImmNode>();
        if (!mi || !ei || !si || mi->value != 0 ||
            ei->value.as<int64_t>().value() != si->value) {
          return false;
        }
        *total *= ei->value.as<int64_t>().value();
      }
      return true;
    };

    int64_t st = 0, dt = 0;
    ICHECK(is_full(sr, src.shape(), &st) && is_full(dr, dst.shape(), &dt))
        << "T.reduce inside SimdVF requires full-buffer regions";

    int64_t reduce_extent = GetConstInt(sr[dim]->extent).value();
    int64_t outer_extent = GetConstInt(sr[1 - dim]->extent).value();
    ICHECK_EQ(reduce_extent % vreg_size_, 0)
        << "T.reduce requires dim extent multiple of VReg (" << vreg_size_
        << "), got " << reduce_extent;
    int64_t nchunks = reduce_extent / vreg_size_;

    const Op &element_op = (rtype == "sum") ? simd_vadd() : simd_vmax();
    const Op &reduce_op = (rtype == "sum") ? simd_vcadd() : simd_vcmax();
    PrimExpr init_value = (rtype == "sum")
                              ? PrimExpr(FloatImm(dtype, 0.0))
                              : PrimExpr(FloatImm(dtype, -3.402823e+38));

    auto i32 = [](int64_t v) { return IntImm(PrimType::Int(32), v); };
    auto addr1d = [&](const TensorVar &buf, const PrimExpr &i0, int rw_mask) {
      return MakeAccessPtr(buf, {i0}, rw_mask);
    };
    auto addr2d = [&](const TensorVar &buf, const PrimExpr &i0,
                      const PrimExpr &i1, int rw_mask) {
      return MakeAccessPtr(buf, {i0, i1}, rw_mask);
    };

    mask_var_ = PrimVar("mask", PrimType::Bool(256));
    Expr mask_value =
        Call(PrimType::Bool(256), simd_pset(),
             {IntImm(PrimType::Int(32), elem_bits_), StringImm("PAT_ALL")});
    PrimVar rep("vrepeat", PrimType::Int(32));

    if (dim == 1) {
      Array<Stmt> row;
      PrimVar acc = MakeReg(dtype);
      row.push_back(
          Bind(acc, Call(acc.ty(), simd_vdup(),
                         {init_value, mask_var_, StringImm("MODE_ZEROING")})));

      for (int64_t ck = 0; ck < nchunks; ++ck) {
        PrimExpr col = i32(ck * vreg_size_);
        PrimVar loaded = MakeReg(dtype);
        PrimVar reduced = MakeReg(dtype);
        PrimVar next_acc = MakeReg(dtype);
        row.push_back(Bind(loaded, Call(loaded.ty(), simd_vld(),
                                        {addr2d(src, rep, col, /*rw_mask=*/1),
                                         StringImm("NORM")})));
        row.push_back(Bind(
            reduced, Call(reduced.ty(), reduce_op,
                          {loaded, mask_var_, StringImm("MODE_ZEROING")})));
        row.push_back(Bind(next_acc, Call(next_acc.ty(), element_op,
                                          {acc, reduced, mask_var_,
                                           StringImm("MODE_ZEROING")})));
        acc = next_acc;
      }

      PrimExpr dst_off = rep;
      if (!clear) {
        PrimVar old = MakeReg(dtype);
        PrimVar merged = MakeReg(dtype);
        row.push_back(Bind(old, Call(old.ty(), simd_vld(),
                                     {addr1d(dst, dst_off, /*rw_mask=*/1),
                                      StringImm("BRC_B32")})));
        row.push_back(Bind(
            merged, Call(merged.ty(), element_op,
                         {acc, old, mask_var_, StringImm("MODE_ZEROING")})));
        acc = merged;
      }
      row.push_back(Evaluate(Call(PrimType::Void(), simd_vsts(),
                                  {addr1d(dst, dst_off, /*rw_mask=*/2), acc,
                                   mask_var_, StringImm("ONEPT_B32")})));

      Stmt row_body = MakeSeq({Bind(mask_var_, mask_value), MakeSeq(row)});
      Stmt result = For(rep, i32(0), i32(outer_extent), IntImm::Int32(1),
                        ForKind::kDefault, SeqStmt(row_body));
      frag_counter_ = saved_frag_counter;
      return result;
    }

    PrimVar chunk("vchunk", PrimType::Int(32));
    PrimExpr col = chunk * i32(vreg_size_);
    Array<Stmt> chunk_body;
    PrimVar acc = MakeReg(dtype);
    chunk_body.push_back(Bind(
        acc,
        Call(acc.ty(), simd_vld(),
             {addr2d(src, i32(0), col, /*rw_mask=*/1), StringImm("NORM")})));

    for (int64_t row_idx = 1; row_idx < outer_extent; ++row_idx) {
      PrimVar loaded = MakeReg(dtype);
      PrimVar next_acc = MakeReg(dtype);
      chunk_body.push_back(
          Bind(loaded, Call(loaded.ty(), simd_vld(),
                            {addr2d(src, i32(row_idx), col, /*rw_mask=*/1),
                             StringImm("NORM")})));
      chunk_body.push_back(Bind(
          next_acc, Call(next_acc.ty(), element_op,
                         {acc, loaded, mask_var_, StringImm("MODE_ZEROING")})));
      acc = next_acc;
    }

    if (!clear) {
      PrimVar old = MakeReg(dtype);
      PrimVar merged = MakeReg(dtype);
      chunk_body.push_back(Bind(
          old, Call(old.ty(), simd_vld(),
                    {addr1d(dst, col, /*rw_mask=*/1), StringImm("BRC_B32")})));
      chunk_body.push_back(
          Bind(merged, Call(merged.ty(), element_op,
                            {acc, old, mask_var_, StringImm("MODE_ZEROING")})));
      acc = merged;
    }
    chunk_body.push_back(Evaluate(Call(PrimType::Void(), simd_vsts(),
                                       {addr1d(dst, col, /*rw_mask=*/2), acc,
                                        mask_var_, StringImm("NORM_B32")})));

    Stmt loop = For(chunk, i32(0), i32(nchunks), IntImm::Int32(1),
                    ForKind::kDefault, SeqStmt(MakeSeq(chunk_body)));
    Stmt result = MakeSeq({Bind(mask_var_, mask_value), loop});
    frag_counter_ = saved_frag_counter;
    return result;
  }

  Stmt LowerParallel(const ForNode *vreg_for, const ForNode *serial_for) {
    int saved_frag_counter = frag_counter_;
    frag_counter_ = 0;

    sym::Analyzer analyzer;
    TVM_FFI_CHECK(IsOne(vreg_for->step) &&
                      (!serial_for || IsOne(serial_for->step)),
                  ValueError)
        << "SimdVF parallel loops require a unit step";
    int64_t vreg_ext =
        GetConstInt(
            tl::arith::Simplify(analyzer, vreg_for->end - vreg_for->begin))
            .value();
    int64_t serial_ext =
        serial_for
            ? GetConstInt(tl::arith::Simplify(analyzer, serial_for->end -
                                                            serial_for->begin))
                  .value()
            : 1;

    Stmt scalar_body = vreg_for->body;
    if (serial_for && !CanLowerBody(scalar_body))
      scalar_body = serial_for->body;
    PrimType dtype = InferDType(scalar_body);
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

    parallel_var_ = vreg_for->var.get();
    parallel_extent_ = vreg_ext;
    parallel_begin_ = vreg_for->begin;
    outer_begin_ = serial_for ? serial_for->begin : IntImm::Int32(0);
    if (serial_for) {
      outer_parallel_var_ = serial_for->var.get();
      outer_serial_var_ = PrimVar("vserial", PrimType::Int(32));
    } else {
      outer_parallel_var_ = nullptr;
    }

    chunk_var_ = PrimVar("vchunk", PrimType::Int(32));
    mask_var_ = PrimVar("mask", PrimType::Bool(256));
    scalar_loads_.clear();

    Expr mask_value =
        Call(PrimType::Bool(256), simd_pset(),
             {IntImm(PrimType::Int(32), elem_bits_), StringImm("PAT_ALL")});

    Stmt body = LowerParallelBody(scalar_body, dtype);
    Stmt masked_body = BindStmt(mask_var_, mask_value, body);
    Stmt inner_loop = For(chunk_var_, IntImm(PrimType::Int(32), 0),
                          IntImm(PrimType::Int(32), nchunks), IntImm::Int32(1),
                          ForKind::kDefault, SeqStmt(masked_body));

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

    Stmt result = per_iter_body;
    if (serial_for) {
      result = For(outer_serial_var_, IntImm(PrimType::Int(32), 0),
                   IntImm(PrimType::Int(32), serial_ext), IntImm::Int32(1),
                   ForKind::kDefault, SeqStmt(per_iter_body));
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
  auto pass_func =
      [=](tirx::Function f, IRModule m,
          tvm::transform::PassContext ctx) -> ffi::Optional<tirx::Function> {
    return AscendSimdVFLowerParallel::Substitute(std::move(f));
  };
  return CreateFunctionPass(pass_func, 0, "tl.AscendSimdVFLowerParallel");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AscendSimdVFLowerParallel",
                        AscendSimdVFLowerParallel);
}

} // namespace tl
} // namespace tvm
