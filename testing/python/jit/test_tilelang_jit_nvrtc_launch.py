import ast
import ctypes

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tilelang.jit.adapter.nvrtc.host import HostScalarEmitter, collect_host_launches
from tilelang.jit.adapter.nvrtc.wrapper import TLNVRTCSourceWrapper


def _emitter(bindings, inputs=None):
    return HostScalarEmitter(bindings, inputs or {}, TLNVRTCSourceWrapper._TYPE_MAP, "prepared_")


def _evaluate(emitter, expr, **inputs):
    value = emitter.emit(expr)
    scope = {"ctypes": ctypes, **inputs}
    exec("\n".join(emitter.take_statements()) + f"\nresult = {value}", scope)
    return scope["result"]


def test_host_scalar_dependencies_and_casts():
    n, p, q = [tirx.Var(name, "int32") for name in ("n", "p", "q")]
    emitter = _emitter({p: n * 3 + 1, q: p * 2 + 5}, {n: "n"})
    assert _evaluate(emitter, p + q, n=7) == 71
    assert len(emitter.values) == 2
    assert _evaluate(_emitter({}), tirx.Cast("uint32", tirx.const(-1, "int32"))) == 2**32 - 1
    emitter = _emitter({p: n + 1}, {n: "n"})
    assert _evaluate(emitter, p < 0, n=2**31 - 1)


def test_host_scalar_floordiv_and_mod():
    n = tirx.Var("n", "int32")
    assert _evaluate(_emitter({}, {n: "n"}), tirx.floordiv(n, 3), n=-7) == -3
    assert _evaluate(_emitter({}, {n: "n"}), tirx.floormod(n, 3), n=-7) == 2


def test_host_scalar_does_not_alias_by_name():
    original = tirx.Var("n", "int32")
    unrelated = tirx.Var("n", "int32")
    with pytest.raises(ValueError, match="Cannot resolve NVRTC host argument"):
        _emitter({}, {original: "n"}).emit(unrelated)


def test_host_scalar_rejects_device_load_and_unknown_call():
    buffer = tirx.decl_buffer((1,), "int32")
    with pytest.raises(ValueError, match="Unsupported NVRTC host expression BufferLoad"):
        _emitter({}).emit(buffer[0])
    with pytest.raises(ValueError, match="Unsupported NVRTC host expression Call"):
        _emitter({}).emit(tirx.call_extern("int32", "unknown"))


def test_collect_launches_preserves_instances_and_scope():
    n, p = tirx.Var("n", "int32"), tirx.Var("p", "int32")

    def launch(arg):
        return tirx.Evaluate(tirx.call_packed("kernel", arg))

    body = tirx.SeqStmt(
        [
            launch(n),
            tirx.IfThenElse(n > 0, tirx.SeqStmt([tirx.Bind(p, n + 1), launch(p)]), None),
            launch(n),
        ]
    )
    calls = collect_host_launches(body, {"kernel": 1})
    assert len(calls) == 3
    assert not calls[0].conditions and len(calls[1].conditions) == 1
    assert p in calls[1].bindings and p not in calls[2].bindings
    assert calls[1].arguments[0].same_as(p)


def test_collect_launches_rejects_loops_and_missing_arguments():
    i = tirx.Var("i", "int32")
    call = tirx.Evaluate(tirx.call_packed("kernel", i))
    loop = tirx.For(i, 0, 2, tirx.ForKind.SERIAL, call)
    with pytest.raises(ValueError, match="inside host loops"):
        collect_host_launches(loop, {"kernel": 1})
    with pytest.raises(ValueError, match="requires 2 arguments"):
        collect_host_launches(call, {"kernel": 2})


def _lower_prepared_kernel(conditional, simplify, arch="sm_80"):
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import device_codegen_without_compile, get_device_call, get_host_call

    context = create_backend_context({"kind": "cuda", "arch": arch}, "c", "nvrtc")
    a = tirx.decl_buffer((128,), "int32", name="A")
    b = tirx.decl_buffer((128,), "int32", name="B")
    n, p, q = [tirx.Var(name, "int32") for name in ("n", "prepared_p", "prepared_q")]
    tx = tirx.IterVar(tvm.ir.Range(0, 128), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    device = tirx.AttrStmt(tx, "thread_extent", 128, tirx.BufferStore(b, a[tx.var] + p + q, [tx.var]))
    device = tirx.AttrStmt(tvm.target.Target({"kind": "cuda", "arch": arch}), "target", 0, device)
    # A division inside the condition must not be evaluated when n == 0.
    prepared_p = tirx.floordiv(120, n) if conditional else n * 3 + 1
    body = tirx.SeqStmt([tirx.Bind(p, prepared_p), tirx.Bind(q, p * 2 + 5), device])
    if conditional:
        body = tirx.IfThenElse(n > 0, body, None)
    func = tirx.PrimFunc([a.data, b.data, n], body, buffer_map={a.data: a, b.data: b}).with_attrs(
        {"global_symbol": "main", "target": context.target, "tirx.noalias": True}
    )
    mod = tvm.IRModule({"main": func})
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    if simplify:
        mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    host = tirx.transform.Filter(get_host_call())(mod)
    device = tirx.transform.Filter(get_device_call())(mod)
    source = device_codegen_without_compile(device, context).inspect_source()
    return func, context, host, device, source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("simplify", [False, True])
def test_nvrtc_wrapper_uses_actual_host_arguments(conditional, simplify):
    func, context, host, device, source = _lower_prepared_kernel(conditional, simplify)
    wrapper = TLNVRTCSourceWrapper(tvm.IRModule({"main": func}), source, context.target, device_mod=device, host_mod=host)
    tree = ast.parse(wrapper.host_func)
    payloads = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "arg_values" for target in node.targets)
    ]
    assert len(payloads) == 1 and isinstance(payloads[0], ast.Tuple)
    assert len(payloads[0].elts) == 4
    assert "__tl_launch_0_scalar_" in wrapper.host_func or simplify


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("simplify", [False, True])
def test_nvrtc_prepared_arguments_execute(conditional, simplify):
    from tilelang.engine.lower import extrac_params
    from tilelang.jit.adapter.nvrtc import is_nvrtc_available

    if not is_nvrtc_available:
        pytest.skip("NVRTC execution requires cuda-python")
    from tilelang.jit.adapter.nvrtc.adapter import NVRTCKernelAdapter

    major, minor = torch.cuda.get_device_capability()
    func, context, host, device, source = _lower_prepared_kernel(conditional, simplify, f"sm_{major}{minor}")
    adapter = NVRTCKernelAdapter(
        extrac_params(func), [], context.target, func, host_mod=host, device_mod=device, device_kernel_source=source
    )
    x = torch.arange(128, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    # Non-default stream also exercises the adapter's unchanged stream binding.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for n in (7, 13, 0, -3, 7):
            y.fill_(-999)
            adapter.func(x, y, n)
            if conditional:
                expected = x + 3 * (120 // n) + 5 if n > 0 else torch.full_like(x, -999)
            else:
                expected = x + 9 * n + 8
            torch.testing.assert_close(y, expected)


@tilelang.testing.requires_cuda
def test_nvrtc_wrapper_keeps_repeated_call_sites():
    func, context, host, device, source = _lower_prepared_kernel(False, False)

    def duplicate(stmt):
        if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            if call.op == tvm.ir.Op.get("tirx.tvm_call_packed") and call.args[0] == "main_kernel":
                return tirx.SeqStmt([stmt, stmt])

    host_func = host["main"]
    body = tirx.stmt_functor.ir_transform(host_func.body, None, duplicate)
    host = tvm.IRModule({"main": host_func.with_body(body)})
    wrapper = TLNVRTCSourceWrapper(tvm.IRModule({"main": func}), source, context.target, device_mod=device, host_mod=host)
    assert wrapper.host_func.count('res = cuLaunchKernelEx(config, kernels["main_kernel"]') == 2


@tilelang.testing.requires_cuda
def test_nvrtc_wrapper_does_not_match_device_parameter_names():
    from tilelang.engine.lower import device_codegen_without_compile

    func, context, host, device, _ = _lower_prepared_kernel(False, False)
    kernel = device["main_kernel"]
    params = [tirx.Var(f"renamed_{i}", param.type_annotation) for i, param in enumerate(kernel.params)]
    body = tirx.stmt_functor.substitute(kernel.body, dict(zip(kernel.params, params)))
    renamed = tirx.PrimFunc(params, body, ret_type=kernel.ret_type, attrs=kernel.attrs)
    device = tvm.IRModule({"main_kernel": renamed})
    source = device_codegen_without_compile(device, context).inspect_source()
    wrapper = TLNVRTCSourceWrapper(tvm.IRModule({"main": func}), source, context.target, device_mod=device, host_mod=host)
    namespace = {}
    exec(wrapper.host_func, namespace)
    captured = []
    # Inspect arguments without calling CUDA with fake pointers.
    namespace["cuKernelSetAttribute"] = lambda *args: (0,)
    namespace["cuLaunchKernelEx"] = lambda config, kernel, arguments, extra: captured.append(arguments) or (0,)

    class Tensor:
        device = torch.device("cuda:0")

        def data_ptr(self):
            return 1234

    namespace["call"]({"main_kernel": object()}, Tensor(), Tensor(), 7)
    assert captured[0][0] == (1234, 1234, 22, 49)


@tilelang.testing.requires_cuda
def test_nvrtc_wrapper_preserves_tma_descriptor_arguments():
    from tilelang.cuda import language as T
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import lower_with_context

    @T.prim_func
    def copy(A: T.Tensor((128, 128), "float16"), B: T.Tensor((128, 128), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128, 64), "float16")
            T.copy(A[0:128, 0:64], shared, prefer_instruction="tma")
            T.copy(shared, B[0:128, 0:64], prefer_instruction="sync")

    context = create_backend_context({"kind": "cuda", "arch": "sm_90"}, "c", "nvrtc")
    with context.target:
        artifact = lower_with_context(copy, context)
    wrapper = TLNVRTCSourceWrapper(
        tvm.IRModule({"copy": copy}),
        artifact.kernel_source,
        context.target,
        device_mod=artifact.device_mod,
        host_mod=artifact.host_mod,
    )
    assert "cuTensorMapEncodeTiled(" in wrapper.host_func
    tree = ast.parse(wrapper.host_func)
    payloads = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "arg_values" for target in node.targets)
    ]
    assert len(payloads) == 1
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert len(payloads[0].elts) == len(device_func.params)
