/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"
#include "builtin.h"
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt_functor.h>

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

// Decoded form of a tl.region(...) bridge call.
struct DecodedRegionCall {
  Buffer buffer;
  ffi::Array<Range> ranges;
  int access_mask;
};

// Decode a tl.region call's args:
//   args[0]: BufferLoad whose indices are per-axis minima.
//   args[1]: constant int access mask (1=read, 2=write, 3=read-write).
//   args[2 + i]: extent of axis i (supports dynamic PrimExpr).
// A min encoded as a stride-1 Ramp carries its extent in the Ramp lanes; the
// lanes must agree with the provided extent when both are constant.
DecodedRegionCall DecodeRegionCall(const ffi::Array<PrimExpr> &args) {
  size_t n = args.size();
  size_t ndim = n - 2;
  const auto *load = args[0].as<BufferLoadNode>();
  ICHECK(load);
  ICHECK(load->indices.size() == ndim)
      << "load->indices.size() = " << load->indices << " ndim = " << ndim;
  ffi::Array<Range> ranges;
  for (size_t i = 0; i < ndim; i++) {
    PrimExpr index = load->indices[i];
    PrimExpr extent = args[2 + i];
    if (const auto *ramp = index.as<RampNode>()) {
      const auto *stride_imm = ramp->stride.as<IntImmNode>();
      ICHECK(stride_imm && stride_imm->value == 1)
          << "tl.region expects stride-1 Ramp for index";
      if (const auto *lanes_imm = ramp->lanes.as<IntImmNode>()) {
        if (const auto *ext_imm = extent.as<IntImmNode>()) {
          ICHECK_EQ(lanes_imm->value, ext_imm->value)
              << "Ramp lanes and provided extent must match";
        }
      }
      ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
    } else {
      ranges.push_back(Range::FromMinExtent(index, extent));
    }
  }
  const int64_t *mask = as_const_int(args[1]);
  ICHECK(mask) << "tl.region access mask must be a constant int, but got "
               << args[1];
  return {load->buffer, std::move(ranges), static_cast<int>(*mask)};
}

} // namespace

bool IsBufferLikeExpr(const PrimExpr &expr) {
  if (expr.as<BufferLoadNode>() || expr.as<BufferRegionNode>()) {
    return true;
  }
  if (const auto *call = expr.as<CallNode>()) {
    return (call->op.same_as(region()));
  }
  return false;
}

BufferRegion NormalizeToBufferRegion(const PrimExpr &arg) {
  // Case 1: Already a BufferRegion
  if (arg->IsInstance<BufferRegionNode>()) {
    return Downcast<BufferRegion>(arg);
  }

  // Case 2: BufferLoad — convert indices to ranges (Ramp -> lanes, else
  // extent=1)
  if (const auto *load = arg.as<BufferLoadNode>()) {
    ffi::Array<Range> ranges;
    for (const PrimExpr &index : load->indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        ICHECK(ramp->stride.as<IntImmNode>()) << "Ramp stride must be IntImm";
        ICHECK_EQ(ramp->stride.as<IntImmNode>()->value, 1)
            << "Only stride-1 Ramp is supported in region conversion";
        ICHECK(ramp->lanes.as<IntImmNode>())
            << "Scalable vector lanes not supported in region conversion";
        ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        ranges.push_back(Range::FromMinExtent(index, 1));
      }
    }
    return BufferRegion(load->buffer, ranges);
  }

  // Case 3: tl.region(...) — decode the transport bridge
  if (const auto *call = arg.as<CallNode>()) {
    if (call->op.same_as(region())) {
      DecodedRegionCall decoded = DecodeRegionCall(call->args);
      return BufferRegion(decoded.buffer, decoded.ranges);
    }
    LOG(FATAL) << "Unsupported argument for BufferRegion (expect "
                  "BufferLoad/BufferRegion/tl.region): "
               << arg;
  }

  LOG(FATAL) << "Unsupported argument for BufferRegion: " << arg;
  throw; // Unreachable
}

AccessRegion NormalizeToAccessRegion(const PrimExpr &arg,
                                     int default_access_mask) {
  if (const auto *call = arg.as<CallNode>()) {
    if (call->op.same_as(region())) {
      DecodedRegionCall decoded = DecodeRegionCall(call->args);
      return {BufferRegion(decoded.buffer, decoded.ranges),
              decoded.access_mask};
    }
  }
  return {NormalizeToBufferRegion(arg), default_access_mask};
}

void CheckNoAutoScopeBuffers(const PrimFunc &func, const char *pass_name) {
  struct Visitor : StmtExprVisitor {
    const char *pass_name;
    void VisitStmt_(const SBlockNode *op) final {
      for (const Buffer &buffer : op->alloc_buffers) {
        ICHECK(buffer.scope() != "auto")
            << pass_name << ": buffer '" << buffer->name
            << "' still has the virtual scope \"auto\" (T.auto_alloc). The "
               "tl.transform.InferMemoryScope pass must run before this pass "
               "to resolve it; it is currently registered for the "
               "cuda/rocm/cpu pipelines only. Use an explicit T.alloc_* "
               "scope for this backend.";
      }
      StmtExprVisitor::VisitStmt_(op);
    }
  };
  Visitor visitor;
  visitor.pass_name = pass_name;
  visitor(func->body);
}

PrimExpr MakeAccessPtrFromRegion(const BufferRegion &region, int rw_mask,
                                 bool require_2d) {
  Buffer buf = region->buffer;
  int ndim = static_cast<int>(buf->shape.size());
  if (require_2d) {
    ICHECK(ndim >= 2) << "Expect buffers with at least 2 dims";
  }

  PrimExpr offset, extent;
  if (ndim == 1) {
    // 1D: straightforward
    auto axis = region->region[0];
    offset = axis->min;
    extent = axis->extent;
  } else {
    // Compute row-major strides
    std::vector<PrimExpr> strides(ndim);
    PrimExpr one = make_const(buf->shape[0].dtype(), 1);
    PrimExpr cur = one;
    for (int i = ndim - 1; i >= 0; --i) {
      strides[i] = cur;
      cur = cur * buf->shape[i];
    }
    // Offset: sum_{i in [0..ndim-1]} min_i * stride_i
    offset = make_const(buf->shape[0].dtype(), 0);
    for (int i = 0; i < ndim; ++i) {
      offset = offset + region->region[i]->min * strides[i];
    }
    // Extent: last two extents product (elements)
    extent =
        region->region[ndim - 2]->extent * region->region[ndim - 1]->extent;
  }

  // ptype and return handle
  PrimExpr ptype = tirx::TypeAnnotation(buf->dtype);
  ffi::Array<PrimExpr> acc_args{ptype, buf->data, offset, extent,
                                IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), acc_args);
}

PrimExpr MakeAccessPtrFromBufferLoad(const BufferLoad &load, int rw_mask) {
  Buffer buf = load->buffer;
  int ndim = static_cast<int>(buf->shape.size());

  // Compute offset using row-major layout (iterate in reverse)
  PrimExpr offset = 0;
  PrimExpr stride = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    const PrimExpr &index = load->indices[i];
    if (const auto *ramp = index.as<RampNode>()) {
      // For Ramp, use the base
      offset = offset + ramp->base * stride;
    } else {
      // For scalar index (IntImm or other PrimExpr)
      offset = offset + index * stride;
    }
    stride = stride * buf->shape[i];
  }

  // Extent is 1 element for a single BufferLoad access
  PrimExpr extent = make_const(DataType::Int(32), 1);

  // Build access_ptr
  PrimExpr ptype = tirx::TypeAnnotation(buf->dtype);
  ffi::Array<PrimExpr> acc_args{ptype, buf->data, offset, extent,
                                IntImm(DataType::Int(32), rw_mask)};
  return Call(DataType::Handle(), builtin::tvm_access_ptr(), acc_args);
}

// Maps TVM DataType to CUDA's CUtensorMapDataType enum value.
int to_CUtensorMapDataType(DataType dtype) {
  // CUDA 13 adds packed U4 TensorMap formats. The vendored CUDA stub may lag
  // the installed toolkit, so keep the enum value by CUDA's documented order.
  constexpr int kTensorMapDataType16U4Align8B = 13;
  constexpr int kTensorMapDataType16U4Align16B = 14;
  if (dtype.is_float4_e2m1_unpacked()) {
    return kTensorMapDataType16U4Align16B;
  }
  if (dtype.is_float4_e2m1fn()) {
    return kTensorMapDataType16U4Align8B;
  }

  CUtensorMapDataType tp;
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_bfloat16()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.is_tfloat32()) {
    // tfloat32 uses same memory layout as float32
    tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (dtype.is_float8()) {
    tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

} // namespace tl
} // namespace tvm
