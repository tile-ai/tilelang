#ifndef TL_TRANSFORM_TIRX_PASSES_H_
#define TL_TRANSFORM_TIRX_PASSES_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl::tirx_passes {

using FVectorizeCall = ffi::TypedFunction<ffi::Optional<Expr>(
    Call, PrimVar, PrimExpr, ffi::TypedFunction<Expr(Expr)>,
    ffi::Optional<Target>)>;
inline constexpr ffi::reflection::AttrKey<FVectorizeCall> kVectorizeCall{
    "tl.FVectorizeCall"};

tvm::transform::Pass StorageRewrite(bool enable_reuse, bool detect_inplace,
                                    bool preserve_shared_allocations,
                                    bool allow_local_inplace);
tvm::transform::Pass OutlineDeviceRegions();
tvm::transform::Pass LowerDeviceKernelLaunch();
tvm::transform::Pass SplitHostDevice();
tvm::transform::Pass VectorizeLoop(bool enable_vectorize = true);
tvm::transform::Pass HoistIf(bool allow_nontrivial_else = true);

tvm::transform::Pass FlattenTensor();
tvm::transform::Pass LowerTensorLayouts();
tvm::transform::Pass StmtSimplify();
tvm::transform::Pass RemoveNoOp();
tvm::transform::Pass MakePackedAPI();
tvm::transform::Pass LowerIntrin();
tvm::transform::Pass LowerTVMBuiltin();
tvm::transform::Pass BindTarget(Target target);
tvm::transform::Pass AnnotateEntryFunc();
tvm::transform::Pass ConvertSSA();
tvm::transform::Pass NarrowDataType(int target_bits);
tvm::transform::Pass ForceNarrowIndexToInt32();

} // namespace tvm::tl::tirx_passes
#endif
