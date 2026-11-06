/*!
 * \file tl/op/fill.cc
 * \brief Native logical-region fill and target-specific lowering dispatch.
 */

#include "fill.h"
#include "layout/expr_utils.h"
#include "utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

namespace {
std::vector<FillImpl> &FillImplRegistry() {
  static std::vector<FillImpl> registry;
  return registry;
}

const FillImpl &ResolveFillImpl(Target target) {
  const FillImpl *matched = nullptr;
  for (const FillImpl &impl : FillImplRegistry()) {
    if (impl.match_target(target)) {
      ICHECK(matched == nullptr)
          << "Multiple tl.fill implementations for " << target->str();
      matched = &impl;
    }
  }
  ICHECK(matched != nullptr)
      << "No tl.fill implementation for " << target->str();
  return *matched;
}
} // namespace

void RegisterFillImpl(FillImpl impl) {
  ICHECK(impl.name && impl.match_target && impl.lower);
  FillImplRegistry().push_back(impl);
}

Fill::Fill(Array<Expr> args, Map<String, Any> annotations) {
  ICHECK_EQ(args.size(), 2U) << "tl.fill expects destination and value";
  AccessRegion dst_access = NormalizeToAccessRegion(args[0], kAccessWrite);
  TensorVar dst = RegionTensor(dst_access.region);
  PrimExpr value = args[1].as_or_throw<PrimExpr>();
  if (value.ty() != dst.dtype())
    value = cast(value, dst.dtype());
  const Array<Range> &region = dst_access.region->region;
  ICHECK_EQ(region.size(), dst.shape().size());
  for (size_t i = 0; i < region.size(); ++i) {
    auto begin = TryConstInt(region[i]->begin);
    auto extent = TryConstInt(region[i]->extent);
    auto shape = TryConstInt(dst.shape()[i]);
    if (begin)
      ICHECK_GE(*begin, 0) << "Negative fill region origin";
    if (extent)
      ICHECK_GE(*extent, 0) << "Negative fill region extent";
    if (extent && shape)
      ICHECK_LE(*extent, *shape) << "Fill region exceeds tensor shape";
    // Avoid a C++ overflow in the check itself. This also checks sliced fills.
    if (begin && extent && shape)
      ICHECK_LE(*begin, *shape - *extent);
  }
  auto node = make_object<FillNode>(dst, value, region);
  node->SetAccessRegions({dst_access});
  data_ = std::move(node);
}

TileOperator FillNode::Clone() const {
  return Fill(make_object<FillNode>(*this));
}

For FillNode::MakeSIMTLoop(sym::AnalyzerObj *analyzer) const {
  (void)analyzer;
  Array<PrimVar> vars;
  Array<PrimExpr> indices;
  for (size_t i = 0; i < region.size(); ++i) {
    PrimVar var("fill_i" + std::to_string(i), region[i]->extent.ty());
    vars.push_back(var);
    indices.push_back(region[i]->begin + var);
  }
  Stmt body = TensorStore(dst, indices, value);
  if (vars.empty()) {
    PrimVar once("fill_scalar", PrimType::Int(32));
    return For(once, 0, 1, 1, ForKind::kParallel, SeqStmt(body));
  }
  for (size_t i = vars.size(); i > 0; --i) {
    PrimType index_type = vars[i - 1].ty();
    body = For(vars[i - 1], IntImm(index_type, 0), region[i - 1]->extent,
               IntImm(index_type, 1), ForKind::kParallel, SeqStmt(body));
  }
  return body.as_or_throw<For>();
}

Stmt FillNode::Lower(const LowerArgs &args, sym::AnalyzerObj *analyzer) const {
  return ResolveFillImpl(args.target).lower(*this, args, analyzer);
}

LayoutMap FillNode::InferLayout(const LayoutInferArgs &args,
                                InferLevel level) const {
  // Fill constrains neither ownership nor address placement. Other consumers
  // and the destination's explicit annotation determine its layout.
  return {};
}

TIR_REGISTER_TL_TILE_OP(Fill, fill)

TVM_FFI_STATIC_INIT_BLOCK() { FillNode::RegisterReflection(); }

} // namespace tvm::tl
