"""Run packed-boundary tests through real C host code without requiring LLVM."""

from tvm_tirx import codegen, ir, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.compilation_pipeline import finalize_host_passes

import tilelang
from tilelang.cpu.transform import LowerCPUKernelLaunch
import tvm_ffi


def build_packed_module(function, *, tilelang_owned=True, target="c", c_codegen="host"):
    """Build a host Function or IRModule with the selected packed binder.

    Both controls use TileLang's layout/ABI preparation and the same C host emitter. Only
    the packed pass differs: TileLang's established argument contract or the
    independent, unmodified TIRx contract. This helper preserves raw packed
    return values and argument-count errors, without a JIT output adapter.
    Full DSL/JIT and allocated-output behavior is tested through tilelang.compile.
    ``c_codegen="compute"`` selects TileLang's vector-aware C compute emitter
    for tests that retain vector expressions in the packed function body.
    """
    target = target if isinstance(target, Target) else Target(target)
    if target.kind.name not in ("c", "llvm"):
        raise ValueError("Packed boundary tests require a C or LLVM host target")
    if c_codegen not in ("host", "compute") or (target.kind.name != "c" and c_codegen != "host"):
        raise ValueError("C codegen must be host or compute, and compute requires target c")
    module = ir.IRModule.from_expr(function) if isinstance(function, tirx.Function) else function
    module = tirx.transform.BindTarget(target.with_host(target))(module)
    module = LowerCPUKernelLaunch()(module)
    module = tilelang.transform.LowerTensorSpec()(module)
    module = tilelang.transform.LowerOpaqueBlock()(module)
    if tilelang_owned:
        module = tilelang.transform.LowerTensorDescriptors()(module)
    module = ir.transform.Sequential(
        [
            tirx.transform.AnnotateEntryFunc(),
            # A specialized zero extent has no reachable tensor accesses.
            # Remove its empty loops before layout grouping would construct
            # a modulo by that extent. Keep live effects and ABI checks.
            tilelang.transform.RemoveNoOp(),
            tilelang.transform.LowerTensorLayouts(),
            tirx.transform.FlattenTensor(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.FP8ComputeLegalize(),
        ]
    )(module)
    packed_pass = tilelang.transform.MakePackedAPI if tilelang_owned else tirx.transform.MakePackedAPI
    module = packed_pass()(module)
    module = ir.transform.Sequential([tirx.transform.FP8StorageLegalize(), tirx.transform.BF16StorageLegalize()])(module)
    module = tilelang.transform.PrepareAnyList()(module)
    module = finalize_host_passes()(module)
    build = "target.build.tilelang_c_host" if target.kind.name == "c" else "target.build.llvm"
    if c_codegen == "compute":
        build = "target.build.tilelang_c"
    return tvm_ffi.get_global_func(build)(module, target)


def compile_packed_function(function, *, tilelang_owned=True, target="c", c_codegen="host"):
    """Compile and load a packed function; the default C host needs no LLVM."""
    executable = codegen.Executable(build_packed_module(function, tilelang_owned=tilelang_owned, target=target, c_codegen=c_codegen))
    from tilelang.env import TILELANG_TEMPLATE_PATH

    options = ["-std=c++17", "-I" + TILELANG_TEMPLATE_PATH]
    if c_codegen == "compute":
        options.append("-fopenmp")
    executable.jit(cc="c++", options=options)
    return executable
