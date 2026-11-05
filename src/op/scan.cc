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
using namespace prim;
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
  ICHECK_EQ(args.size(), 4);
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->srcRegion_ = src_access.region;
  node->dstRegion_ = dst_access.region;
  node->SetAccessRegions({src_access, dst_access});
  node->src = RegionTensor(node->srcRegion_);
  node->dst = RegionTensor(node->dstRegion_);
  node->dim = GetIntegerArgument(args[2]);
  node->reverse = (GetIntegerArgument(args[3]) != 0);
  ICHECK_GE(node->dim, 0);
  ICHECK_LT(node->dim, static_cast<int>(node->src.shape().size()))
      << "The dim of " << op_name
      << " should be less than the number of dimensions. Got dim=" << node->dim
      << ", but src has " << node->src.shape().size() << " dims.";
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

CumSumOp::CumSumOp(Array<Expr> args, Map<String, Any> annotations) {
  ObjectPtr<CumSumOpNode> node = make_object<CumSumOpNode>();
  InitScanOpNode(node.get(), args, "cumsum");
  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &lower_args,
                         sym::AnalyzerObj *analyzer) const {
  return ResolveScanImpl<CumSumImpl>(lower_args.target, "cumsum")
      .lower(*this, lower_args, analyzer);
}

LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  return InferScanLayout(*this, layout_args, level, "CumSum");
}

TIR_REGISTER_TL_TILE_OP(CumSumOp, cumsum)

CumMaxOp::CumMaxOp(Array<Expr> args, Map<String, Any> annotations) {
  ObjectPtr<CumMaxOpNode> node = make_object<CumMaxOpNode>();
  InitScanOpNode(node.get(), args, "cummax");
  data_ = std::move(node);
}

Stmt CumMaxOpNode::Lower(const LowerArgs &lower_args,
                         sym::AnalyzerObj *analyzer) const {
  return ResolveScanImpl<CumMaxImpl>(lower_args.target, "cummax")
      .lower(*this, lower_args, analyzer);
}

LayoutMap CumMaxOpNode::InferLayout(const LayoutInferArgs &layout_args,
                                    InferLevel level) const {
  return InferScanLayout(*this, layout_args, level, "CumMax");
}

TIR_REGISTER_TL_TILE_OP(CumMaxOp, cummax)

TVM_FFI_STATIC_INIT_BLOCK() {
  CumSumOpNode::RegisterReflection();
  CumMaxOpNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
