"""Build a Relax VM executable with TileLang-compiled TIR kernels.

The VM path:
  - Manages memory natively (StaticPlanBlockMemory, no Python-side pools)
  - Dispatches kernels from the C++ bytecode loop
  - Lowers scheduled TIR through the selected TileLang backend pipeline
"""

import logging

import torch
import tvm_ffi

from tilelang import tvm
from tvm import relax, tirx as tir, runtime
from tvm.runtime import from_dlpack as _tvm_from_dlpack
from tvm.target import Target

from tilelang.backend.pass_pipeline import resolve_pipeline
from tilelang.graph.pipeline import run_pipeline
from tilelang.engine.lower import (
    device_codegen,
    get_device_call,
    get_host_call,
    host_codegen,
)
from tilelang.graph.utils import validate_cuda_tensors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch fallback dispatch: register torch ops as TVM packed functions
# ---------------------------------------------------------------------------


def _register_torch_fallbacks(fallback_calls: dict) -> list[str]:
    """Register torch fallback ops as TVM packed functions for the VM.

    The converter emits namespaced ``ExternFunc`` calls for ops dispatched via
    ``extern_dispatch``.  The VM resolves these
    at runtime through the TVM FFI registry.  The fallback function
    returns the result directly (no pre-allocation or copy).

    Returns the list of registered function names.
    """
    registered = []
    for func_name, (op_fn, arg_template, kwargs) in fallback_calls.items():
        wrapper = _make_fallback_wrapper(op_fn, arg_template, kwargs)
        tvm_ffi.register_global_func(func_name, wrapper, override=True)
        registered.append(func_name)
        logger.debug("Registered VM fallback: %s", func_name)
    return registered


def _make_fallback_wrapper(op_fn, arg_template, kwargs):
    """Build a torch fallback wrapper for the VM dispatch table.

    Pre-compiles ``arg_template`` into a compact ("kind", payload) list
    so each VM dispatch can build the torch call without re-iterating
    the template.  In-place torch ops (name ending with ``_``) return
    the first TVM arg directly, skipping the result DLPack conversion.
    """
    arg_instrs = []
    for kind, val in arg_template:
        if kind is True:
            arg_instrs.append(("tensor", None))
        elif kind == "list":
            sub = []
            for name in val:
                if isinstance(name, str):
                    sub.append(("tensor", None))
                else:
                    sub.append(("const", name))
            arg_instrs.append(("list", sub))
        else:
            arg_instrs.append(("const", val))

    const_kwargs = {}
    tensor_kwarg_keys = []
    for k, v in kwargs.items():
        if isinstance(v, tuple) and len(v) == 2 and v[0] == "__tensor__":
            tensor_kwarg_keys.append(k)
        else:
            const_kwargs[k] = v
    has_tensor_kwargs = bool(tensor_kwarg_keys)

    is_method_call = isinstance(op_fn, str)
    is_inplace = isinstance(op_fn, str) and op_fn.endswith("_")

    def wrapper(*tvm_args):
        n = len(tvm_args)
        torch_inputs = [None] * n
        for i in range(n):
            torch_inputs[i] = torch.from_dlpack(tvm_args[i])

        args = []
        tidx = 0
        for kind, payload in arg_instrs:
            if kind == "tensor":
                args.append(torch_inputs[tidx])
                tidx += 1
            elif kind == "list":
                lst = []
                for sk, sv in payload:
                    if sk == "tensor":
                        lst.append(torch_inputs[tidx])
                        tidx += 1
                    else:
                        lst.append(sv)
                args.append(lst)
            else:
                args.append(payload)

        if has_tensor_kwargs:
            resolved_kw = dict(const_kwargs)
            for k in tensor_kwarg_keys:
                resolved_kw[k] = torch_inputs[tidx]
                tidx += 1
        else:
            resolved_kw = const_kwargs

        if is_method_call:
            result = getattr(args[0], op_fn)(*args[1:], **resolved_kw)
        else:
            result = op_fn(*args, **resolved_kw)

        if is_inplace:
            return tvm_args[0]
        if isinstance(result, torch.Tensor):
            if not result.is_contiguous():
                result = result.contiguous()
            return _tvm_from_dlpack(result)
        return result

    return wrapper


# ---------------------------------------------------------------------------
# VM lowering passes (extends the Relax pipeline for VM execution)
# ---------------------------------------------------------------------------


def _apply_vm_lowering(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Apply VM-specific lowering passes after the main Relax pipeline.

    These passes transform the Relax IR into a form the VM codegen can consume:
      - KillAfterLastUse:    mark tensors freeable after last consumer
      - LowerRuntimeBuiltin: lower runtime builtins (print_format, etc.)
      - ComputePrimValue:    replace symbolic PrimValue exprs with call_tir to tiny PrimFuncs
      - VMShapeLower:        lower shape-related ops to VM-understood form
      - AttachGlobalSymbol:  attach global_symbol attr so VM can find TIR funcs by name
    """
    with target:
        seq = tvm.transform.Sequential(
            [
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.ComputePrimValue(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
            ]
        )
        mod = seq(mod)
    return mod


# ---------------------------------------------------------------------------
# TIR compilation: split kernel vs host-only, compile each appropriately
# ---------------------------------------------------------------------------


def _is_host_target(func: tir.PrimFunc) -> bool:
    """Check if a PrimFunc is bound to a host (CPU) target."""
    target = func.attrs.get("target", None) if func.attrs else None
    if target is None:
        return False
    return str(target.kind) in ["llvm", "c"]


def _compile_kernels_tilelang(kernel_mod: tvm.IRModule, target: Target) -> runtime.Module:
    """Compile device PrimFuncs through TileLang's backend pipeline and codegen.

    Every GPU kernel that reaches this stage must be marked with exactly one of:
      * ``tirx.is_tilelang_kernel`` - DSL kernel with an explicit schedule.
      * ``tirx.is_scheduled``       - schedule-rule kernel.

    An unmarked PrimFunc is treated as a bug (likely a schedule rule that
    produced a TIR function without stamping ``tirx.is_scheduled``, or a pattern
    builder that forgot ``tirx.is_tilelang_kernel``).
    """
    target_host_str = "llvm" if runtime.enabled("llvm") else "c"
    target_host = Target(target_host_str)
    full_target = Target(target, target_host)

    pass_configs = {}
    if target.kind.name == "cuda":
        # Pull in CUDA bf16 fallback headers for generated kernels that touch bf16.
        pass_configs["tl.device_compile_flags"] = ["-DENABLE_BF16"]
    with tvm.transform.PassContext(opt_level=3, config=pass_configs), full_target:
        sched_funcs = {}
        tl_funcs = {}
        unlabeled = []
        for gv, func in kernel_mod.functions.items():
            if not isinstance(func, tir.PrimFunc):
                continue
            attrs = func.attrs
            if attrs and attrs.get("tirx.is_tilelang_kernel", False):
                tl_funcs[gv] = func
            elif attrs and attrs.get("tirx.is_scheduled", False):
                sched_funcs[gv] = func
            else:
                unlabeled.append(gv.name_hint)

        if unlabeled:
            raise RuntimeError(
                "TIR kernels reached device codegen without "
                "`tirx.is_scheduled` or `tirx.is_tilelang_kernel`: "
                f"{unlabeled}. Fix the upstream schedule rule or pattern "
                "builder to stamp the appropriate attribute."
            )

        km = tvm.IRModule({})

        def _run_lowering(funcs):
            if not funcs:
                return
            mod = tvm.IRModule(funcs)
            mod = resolve_pipeline(full_target).lower(mod, full_target)
            for gv, func in mod.functions.items():
                km[gv] = func

        _run_lowering(sched_funcs)
        _run_lowering(tl_funcs)

    _is_host = get_host_call(False)
    _is_device = get_device_call(False)
    khost = tir.transform.Filter(_is_host)(km)
    kdevice = tir.transform.Filter(_is_device)(km)

    device_rt = device_codegen(kdevice, full_target)
    host_rt = host_codegen(khost, target_host, target=full_target)
    host_rt.import_module(device_rt)
    return host_rt


def _compile_tir_for_vm(tir_mod: tvm.IRModule, target: Target) -> runtime.Module:
    """Compile all TIR functions for the VM.

    The TIR module contains two kinds of PrimFuncs after VM lowering:

    1. **Device kernel functions** (from schedule rules / pattern rewrite)
       → Compiled by the pass pipeline registered for the target backend.

    2. **Host-only scalar functions** (from ComputePrimValue for symbolic
       shapes, or from LegalizeOps on scalar binops)
       → Compiled via standard TVM TIR pipeline (MakePackedAPI + LLVM codegen)

    BindTarget classifies them: host-only functions have ``tir.is_host_func``
       which BindTarget maps to a CPU target; kernel functions get the device target.
    """
    target_host_str = "llvm" if runtime.enabled("llvm") else "c"
    target_host = Target(target_host_str)
    full_target = Target(target, target_host)

    tir_mod = tir.transform.BindTarget(full_target)(tir_mod)

    kernel_mod = tir.transform.Filter(lambda f: not _is_host_target(f))(tir_mod)
    host_only_mod = tir.transform.Filter(_is_host_target)(tir_mod)

    n_kernel = len(kernel_mod.get_global_vars())
    n_host = len(host_only_mod.get_global_vars())
    logger.info("TIR split: %d kernel, %d host-only functions", n_kernel, n_host)

    lib = None

    # Device kernel functions -> TileLang pipeline + codegen
    if n_kernel > 0:
        lib = _compile_kernels_tilelang(kernel_mod, target)

    # Host-only functions → standard TVM pipeline (needed for MakePackedAPI)
    if n_host > 0:
        host_lib = tvm.tirx.build(host_only_mod, target=full_target, pipeline="default")
        if lib is not None:
            lib.import_module(host_lib)
        else:
            lib = host_lib

    return lib


# ---------------------------------------------------------------------------
# Top-level VM build
# ---------------------------------------------------------------------------


def build_vm_executable(
    relax_mod: tvm.IRModule,
    target: Target,
    fallback_calls: dict = None,
    use_cuda_graph: bool = False,
    device_index: int = 0,
) -> relax.vm_build.VMExecutable:
    """Build a Relax VM executable from a Relax module.

    Pipeline::

        Relax optimization → VM lowering → bytecode generation
        → TileLang TIR compilation → VM linking → VMExecutable

    Parameters
    ----------
    fallback_calls : dict, optional
        Torch fallback ops from the converter.  Each entry is registered as
        a TVM packed function so the VM can dispatch to torch at runtime.
    """
    # Register torch fallback ops so the VM can resolve them
    if fallback_calls:
        _register_torch_fallbacks(fallback_calls)

    # Step 1: TileLang's Relax optimization pipeline
    mod = run_pipeline(
        relax_mod,
        target,
        use_cuda_graph=use_cuda_graph,
        device_index=device_index,
    )

    # Step 2: VM-specific lowering passes
    mod = _apply_vm_lowering(mod, target)
    logger.info("VM lowering complete")

    # Step 3: VM bytecode generation (consumes Relax functions, leaves TIR)
    builder = relax.ExecBuilder()
    mod = tvm.get_global_func("relax.VMCodeGen")(builder, mod)
    logger.info("VM bytecode generated")

    # Step 4: Extract and compile TIR
    tir_funcs = {gv: f for gv, f in mod.functions.items() if isinstance(f, tir.PrimFunc)}
    lib = None
    if tir_funcs:
        tir_mod = tvm.IRModule(tir_funcs, attrs=mod.attrs)
        lib = _compile_tir_for_vm(tir_mod, target)

    # Step 5: Link into VM executable
    vm_exe = tvm.get_global_func("relax.VMLink")(
        builder,
        target,
        lib,
        [],
        {},
    )
    return relax.vm_build.VMExecutable(vm_exe)


# ---------------------------------------------------------------------------
# Torch-compatible VM runner
# ---------------------------------------------------------------------------


class VMRunner:
    """Wraps a Relax VM as a ``torch.compile``-compatible callable.

    Handles ``torch.Tensor`` ↔ ``tvm.nd.NDArray`` conversion via DLPack
    (zero-copy for contiguous tensors on the same device).

    Parameters
    ----------
    clone_output : bool
        If True (default), clone output tensors to prevent corruption
        when the VM reuses storage on the next call.  Set to False for
        benchmarking when outputs are consumed immediately.
    """

    def __init__(
        self,
        vm_exe,
        device,
        func_name="main",
        clone_output=True,
        torch_device: torch.device | None = None,
    ):
        self._vm = runtime.vm.VirtualMachine(vm_exe, device)
        self._func = self._vm[func_name]
        self._device = device
        self._clone = clone_output
        self._torch_device = torch_device

    def __call__(self, *args):
        if self._torch_device is not None:
            validate_cuda_tensors(args, expected_device=self._torch_device)
        tvm_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                tvm_args.append(_tvm_from_dlpack(a))
            else:
                tvm_args.append(a)

        result = self._func(*tvm_args)
        return self._to_torch(result)

    def _to_torch(self, result):
        """Convert VM output to torch tensors."""
        clone = self._clone

        def _convert(v):
            if hasattr(v, "__dlpack__"):
                t = torch.from_dlpack(v)
                return t.clone() if clone else t
            return v

        # VM may return a single tensor, an Array/tuple, or a nested structure.
        # The FX graph's output is always a tuple — return matching structure.
        if hasattr(result, "__dlpack__"):
            return (_convert(result),)
        # tvm_ffi.container.Array or list/tuple
        if hasattr(result, "__len__") and hasattr(result, "__getitem__"):
            return tuple(_convert(result[i]) for i in range(len(result)))
        return result


def build_vm_runner(
    relax_mod: tvm.IRModule,
    target: Target,
    fallback_calls: dict = None,
    func_name: str = "main",
    clone_output: bool = True,
    use_cuda_graph: bool = False,
    device_index: int = 0,
) -> VMRunner:
    """Build and return a torch-callable VM runner.

    Parameters
    ----------
    relax_mod : tvm.IRModule
        The Relax module (pre-pipeline).
    target : Target
        The compilation target (e.g. ``tvm.target.Target("cuda")``)
    fallback_calls : dict, optional
        Torch fallback ops to register for VM dispatch.
    func_name : str
        The Relax function to call (default ``"main"``).
    clone_output : bool
        Clone output tensors for safety (default True).  Set False
        for benchmarking when outputs are consumed before the next call.

    Returns
    -------
    VMRunner
        A callable that takes torch tensors and returns torch tensors.
    """
    target_kind = target.kind.name
    if target_kind == "cuda":
        device = tvm.cuda(device_index)
        torch_device = torch.device("cuda", device_index)
    elif target_kind == "hip":
        device = tvm.rocm(device_index)
        torch_device = None
    elif target_kind in ("c", "llvm"):
        device = tvm.cpu(device_index)
        torch_device = None
    elif target_kind == "metal":
        device = tvm.metal(device_index)
        torch_device = None
    else:
        raise ValueError(f"Unsupported Relax VM target: {target_kind}")

    vm_exe = build_vm_executable(
        relax_mod,
        target,
        fallback_calls,
        use_cuda_graph=use_cuda_graph,
        device_index=device_index,
    )
    logger.info("VM runner built for %s on %s", func_name, device)
    return VMRunner(
        vm_exe,
        device,
        func_name,
        clone_output=clone_output,
        torch_device=torch_device,
    )
