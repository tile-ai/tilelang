#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#ifdef TILELANG_HEXAGON_ENABLED

#include <runtime/hexagon/hexagon_htp.h>

namespace tvm {
namespace tilelang {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def_packed(
      "tilelang.hexagon.hmx_kernel_launch",
      [](ffi::PackedArgs args, ffi::Any *rv) {
        // args[0] is the kernel Function; remaining args are forwarded to it.
        // AnyView supports .cast<T>() for type-safe extraction.
        ffi::Function kernel = args[0].cast<ffi::Function>();

        // PackedArgs(const AnyView* data, int32_t size) — slice past the first
        // arg. args.data() returns const AnyView*, args.size() returns int32_t.
        ffi::PackedArgs kernel_args(args.data() + 1, args.size() - 1);

        // RAII: powers on HMX on construction, releases on scope exit.
        tvm::runtime::hexagon::HexagonHtp htp;

        kernel.CallPacked(kernel_args, rv);
      });
}

} // namespace tilelang
} // namespace tvm

#else
// Hexagon runtime support disabled.
// Build with -DUSE_LLVM=ON to enable HMX kernel launch support.
#endif // TILELANG_HEXAGON_ENABLED
