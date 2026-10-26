#ifndef TL_BACKEND_COMMON_CODEGEN_ATTR_H_
#define TL_BACKEND_COMMON_CODEGEN_ATTR_H_

#include <tvm/codegen/codegen.h>

namespace tvm::tl::codegen_attr {

using tvm::codegen::attr::kParallelStridePattern;
inline constexpr ffi::reflection::AttrKey<bool> kSourceOnly{
    "codegen.source_only"};
inline constexpr ffi::reflection::AttrKey<bool> kEmitLineDirectives{
    "codegen.emit_line_directives"};
inline constexpr ffi::reflection::AttrKey<ffi::String> kPreamble{
    "codegen.preamble"};
inline constexpr ffi::reflection::AttrKey<ffi::String> kPostproc{
    "codegen.postproc"};
inline constexpr ffi::reflection::AttrKey<ffi::String> kCompile{
    "codegen.compile"};
inline constexpr ffi::reflection::AttrKey<ffi::String> kSource{
    "codegen.source"};
inline constexpr ffi::reflection::AttrKey<ffi::Array<IntImm>>
    kReadOnlyParamIndices{"codegen.readonly_param_indices"};
inline constexpr ffi::reflection::AttrKey<ffi::Array<IntImm>>
    kNonRestrictParamIndices{"codegen.non_restrict_param_indices"};

} // namespace tvm::tl::codegen_attr
#endif
