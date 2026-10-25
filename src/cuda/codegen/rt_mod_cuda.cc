#include "backend/common/codegen/native_codegen.h"
#include "codegen_cuda.h"

#include <tvm/ffi/reflection/registry.h>

#include <algorithm>

namespace tvm::codegen {
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("target.build.tilelang_cuda",
           [](IRModule module, Target target) {
             return BuildTileLangCUDA(std::move(module), std::move(target),
                                      false);
           })
      .def("target.build.tilelang_cuda_without_compile",
           [](IRModule module, Target target) {
             return BuildTileLangCUDA(std::move(module), std::move(target),
                                      true);
           })
      .def("tl.cuda.build_from_source",
           [](IRModule module, Target target, ffi::String source) {
             module = tl::PrepareNativeCodegen(std::move(module), target);
             module = WithAttr(std::move(module), attr::kCompile,
                               ffi::String("tl.cuda.compile_source"));
             return ffi::Function::GetGlobalRequired(
                        "target.build.cuda_from_source")(module, target, source)
                 .cast<ffi::Module>();
           })
      .def("tl.cuda.compile_source",
           [](ffi::String source, Target target,
              ffi::Map<ffi::String, ffi::Any> config) {
             auto compiled = ffi::Function::GetGlobalRequired(
                 "tilelang_callback_cuda_compile")(source, target, config);
             if (auto explicit_result = compiled.as<ffi::Array<ffi::Any>>()) {
               return *explicit_result;
             }
             ffi::Bytes binary;
             if (auto bytes = compiled.as<ffi::Bytes>()) {
               binary = *bytes;
             } else {
               auto text = compiled.cast<ffi::String>();
               binary = ffi::Bytes(text.data(), text.size());
             }
             TVM_FFI_CHECK(binary.size() != 0, ValueError)
                 << "CUDA compilation returned an empty binary";
             bool text =
                 std::all_of(binary.data(), binary.data() + binary.size(),
                             [](unsigned char byte) {
                               return byte == 0 || byte == '\n' ||
                                      byte == '\r' || byte == '\t' ||
                                      (byte >= 32 && byte < 127);
                             });
             ffi::String format = text ? "ptx" : "cubin";
             return ffi::Array<ffi::Any>{binary, format};
           });
}
} // namespace tvm::codegen
