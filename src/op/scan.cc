/*!
 * \file tl/op/scan.cc
 * \brief Implementation of inclusive scan operators.
 */

#include "scan.h"

#include "support/check.h"
#include "utils.h"

#include "../layout/layout.h"
#include "../layout/utils.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op/attr.h>

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

template <typename Impl> std::vector<Impl> &ScanImplRegistry() {
  static std::vector<Impl> registry;
  return registry;
}

template <typename Impl>
const Impl &ResolveScanImpl(Target target, const char *op_name) {
  const auto &registry = ScanImplRegistry<Impl>();
  const Impl *matched_impl = nullptr;
  for (const Impl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl." << op_name
          << " found multiple target-specific implementations for "
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl." << op_name
      << " requires a target-specific implementation, but no " << op_name
      << " implementation is registered for " << target->str();
  return *matched_impl;
}

template <typename ScanOpNode>
void InitScanOpNode(ScanOpNode *node, const Array<Expr> &args,
                    const char *op_name) {
  TVM_FFI_CHECK_EQ(args.size(), 4, ValueError)
      << op_name << " expects four arguments";
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = node->srcRegion_->source.template as_or_throw<TensorVar>();
  node->dst = node->dstRegion_->source.template as_or_throw<TensorVar>();
  auto dimension = GetConstInt(args[2]);
  auto reverse = GetConstInt(args[3]);
  TVM_FFI_CHECK(dimension && *dimension >= 0 &&
                    *dimension < static_cast<int64_t>(node->src.shape().size()),
                ValueError)
      << op_name << " dimension must be a constant valid source axis";
  TVM_FFI_CHECK(reverse && (*reverse == 0 || *reverse == 1), ValueError)
      << op_name << " reverse must be a constant boolean";
  node->dim = *dimension;
  node->reverse = *reverse;
  TVM_FFI_CHECK(node->src.dtype() == node->dst.dtype(), ValueError)
      << op_name << " requires matching source and destination dtypes";
  TVM_FFI_CHECK_EQ(node->srcRegion_->region.size(),
                   node->dstRegion_->region.size(), ValueError)
      << op_name << " requires matching source and destination ranks";
  sym::Analyzer analyzer;
  for (size_t axis = 0; axis < node->srcRegion_->region.size(); ++axis) {
    TVM_FFI_CHECK(!analyzer->CanProve(node->srcRegion_->region[axis]->extent !=
                                      node->dstRegion_->region[axis]->extent),
                  ValueError)
        << op_name << " requires matching source and destination extents";
  }
  for (const TensorRegion &region : {node->srcRegion_, node->dstRegion_}) {
    TensorVar tensor = region->source.as_or_throw<TensorVar>();
    for (size_t axis = 0; axis < region->region.size(); ++axis) {
      TVM_FFI_CHECK(!analyzer->CanProve(region->region[axis]->begin < 0) &&
                        !analyzer->CanProve(region->region[axis]->extent < 0) &&
                        !analyzer->CanProve(region->region[axis]->begin +
                                                region->region[axis]->extent >
                                            tensor.shape()[axis]),
                    ValueError)
          << op_name << " region must lie within its tensor";
    }
  }
}

template <typename ScanOpNode>
LayoutMap InferScanLayout(const ScanOpNode &op,
                          const LayoutInferArgs &layout_args, InferLevel level,
                          const char *pretty_name) {
  if (level != InferLevel::kStrict) {
    return {};
  }

  LayoutMap result_map;

  auto make_linear_layout = [](const TensorVar &buf) -> Layout {
    return MakeLinearLayout(buf.shape());
  };

  auto check_or_set_linear_layout = [&](const TensorVar &buf) {
    if (!IsSharedBuffer(buf))
      return;

    Layout linear_layout = make_linear_layout(buf);
    if (layout_args.layout_map.count(buf)) {
      Layout existing =
          layout_args.layout_map.Get(buf).value().as<Layout>().value();
      ICHECK(StructuralEqual()(existing, linear_layout))
          << pretty_name << " requires linear layout for shared buffer "
          << buf->name << ", but got non-linear layout.";
    } else {
      result_map.Set(buf, linear_layout);
    }
  };

  check_or_set_linear_layout(op.src);
  check_or_set_linear_layout(op.dst);

  return result_map;
}

} // namespace

void RegisterCumSumImpl(CumSumImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  ScanImplRegistry<CumSumImpl>().push_back(impl);
}

void RegisterCumMaxImpl(CumMaxImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.lower != nullptr);
  ScanImplRegistry<CumMaxImpl>().push_back(impl);
}

TileOperator CumSumOpNode::Clone() const {
  auto op = make_object<CumSumOpNode>(*this);
  return CumSumOp(op);
}

TileOperator CumMaxOpNode::Clone() const {
  auto op = make_object<CumMaxOpNode>(*this);
  return CumMaxOp(op);
}

CumSumOp::CumSumOp(Array<Expr> args, BlockAnnotations annotations) {
  ObjectPtr<CumSumOpNode> node = make_object<CumSumOpNode>(ffi::UnsafeInit{});
  InitScanOpNode(node.get(), args, "cumsum");
  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &lower_args,
                         sym::Analyzer analyzer) const {
  return ResolveScanImpl<CumSumImpl>(lower_args.target, "cumsum")
      .lower(*this, lower_args, analyzer);
}

LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  return InferScanLayout(*this, layout_args, level, "CumSum");
}

TIR_REGISTER_TL_TILE_OP(CumSumOp, cumsum, 4)

CumMaxOp::CumMaxOp(Array<Expr> args, BlockAnnotations annotations) {
  ObjectPtr<CumMaxOpNode> node = make_object<CumMaxOpNode>(ffi::UnsafeInit{});
  InitScanOpNode(node.get(), args, "cummax");
  data_ = std::move(node);
}

Stmt CumMaxOpNode::Lower(const LowerArgs &lower_args,
                         sym::Analyzer analyzer) const {
  return ResolveScanImpl<CumMaxImpl>(lower_args.target, "cummax")
      .lower(*this, lower_args, analyzer);
}

LayoutMap CumMaxOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  return InferScanLayout(*this, layout_args, level, "CumMax");
}

TIR_REGISTER_TL_TILE_OP(CumMaxOp, cummax, 4)

TVM_FFI_STATIC_INIT_BLOCK() {
  CumSumOpNode::RegisterReflection();
  CumMaxOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
