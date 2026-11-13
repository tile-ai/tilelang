"""Exception-dump ABI and shape propagation tests that need no NPU."""

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.ascend.exception_dump import EXCEPTION_DUMP_ATTR, annotate_exception_dump
from tvm import tirx


ROWS = T.dynamic("rows")


@T.prim_func
def dynamic_copy(
    source: T.Tensor((ROWS, 128), T.bfloat16),
    destination: T.Tensor((ROWS, 128), T.bfloat16),
):
    with T.Kernel(ROWS) as row:
        tile = T.alloc_shared((128,), T.bfloat16)
        T.copy(source[row, :], tile)
        T.copy(tile, destination[row, :])


def test_exception_dump_is_opt_in():
    ordinary = tilelang.lower(dynamic_copy, target="ascend")
    assert "ExceptionDumpInfo" not in ordinary.kernel_source
    with tvm.transform.PassContext(config={"tl.ascend_exception_dump": True}):
        instrumented = tilelang.lower(dynamic_copy, target="ascend")
    assert instrumented.kernel_source.count("#include <tl_templates/ascend/exception_dump.h>") == 1
    assert "tl::ascend::ExceptionDumpInfo __tl_exception_dump_info" in instrumented.kernel_source
    assert "uint64_t __tl_dump_numel_0" in instrumented.kernel_source
    assert "uint64_t __tl_dump_numel_1" in instrumented.kernel_source


def _split_module(dtype="float32", strides=None, tensor_count=1, elem_offset=0):
    n = tirx.Var("n", "int32")
    kernel = tvm.ir.GlobalVar("copy_kernel")
    buffers = [tirx.decl_buffer((n, 8), dtype, name=f"buffer_{i}", strides=strides, elem_offset=elem_offset) for i in range(tensor_count)]
    handles = [tirx.Var(f"handle_{i}", "handle") for i in range(tensor_count)]
    device_params = [tirx.Var(f"device_{i}", "handle") for i in range(tensor_count)]
    callee = tirx.PrimFunc([*device_params, n], tirx.Evaluate(0)).with_attr("target", tvm.target.Target("ascend"))
    host_call = tirx.Call("void", kernel, [*(buffer.data for buffer in buffers), n])
    host = tirx.PrimFunc([*handles, n], tirx.Evaluate(host_call), buffer_map=dict(zip(handles, buffers)))
    return tvm.IRModule({kernel: callee, "main": host})


def test_host_shapes_survive_without_device_buffer_accesses():
    mod = annotate_exception_dump(_split_module())
    kernel = mod["copy_kernel"]
    assert [list(map(int, entry)) for entry in kernel.attrs[EXCEPTION_DUMP_ATTR]] == [[0, 2, 0, 32]]
    assert kernel.params[-1].dtype == "uint64"
    host_call = mod["main"].body.value
    n = mod["main"].params[-1]
    value = tirx.stmt_functor.substitute(host_call.args[-1], {n: tirx.const(7, "int32")})
    assert int(tvm.arith.Analyzer().simplify(value)) == 56


def test_exception_dump_accepts_explicit_dense_strides():
    mod = annotate_exception_dump(_split_module(strides=(8, 1)))
    assert len(mod["copy_kernel"].attrs[EXCEPTION_DUMP_ATTR]) == 1


@pytest.mark.parametrize("offset", [1, tirx.Var("offset", "int32")])
def test_exception_dump_rejects_nonzero_or_unknown_offsets(offset):
    with pytest.raises(ValueError, match="zero tensor element offset"):
        annotate_exception_dump(_split_module(elem_offset=offset))


def test_exception_dump_empty_tensor_list_still_has_marker():
    mod = annotate_exception_dump(_split_module(tensor_count=0))
    assert EXCEPTION_DUMP_ATTR in mod["copy_kernel"].attrs
    assert len(mod["copy_kernel"].attrs[EXCEPTION_DUMP_ATTR]) == 0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dtype": "float4_e2m1fn"}, "does not support tensor dtype"),
        ({"strides": (16, 1)}, "requires contiguous"),
        ({"tensor_count": 17}, "at most 16"),
    ],
)
def test_exception_dump_rejects_unrepresentable_tensors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        annotate_exception_dump(_split_module(**kwargs))


def test_exception_dump_rejects_other_backends():
    with (
        tvm.transform.PassContext(config={"tl.ascend_exception_dump": True}),
        pytest.raises(ValueError, match="only the Ascend backend"),
    ):
        tilelang.lower(dynamic_copy, target={"kind": "cuda", "arch": "sm_90a"})


@pytest.mark.parametrize(
    "attribute, value, message",
    [
        ("global_symbol", "x" * 128, "fewer than 128"),
        ("code_block_source", "extern void kernel();", "source-code kernels"),
    ],
)
def test_exception_dump_rejects_incompatible_kernel_metadata(attribute, value, message):
    mod = _split_module()
    mod.update_func(mod.get_global_var("copy_kernel"), mod["copy_kernel"].with_attr(attribute, value))
    with pytest.raises(ValueError, match=message):
        annotate_exception_dump(mod)


@pytest.mark.parametrize("target", ["pto", {"kind": "ascend", "keys": ["pto", "ascend"]}])
def test_exception_dump_rejects_pto_backend(target):
    # Until PTODSL has a by-value aggregate ABI, fail before emitting a kernel
    # whose launch metadata cannot follow the shared Ascend record format.
    with (
        tvm.transform.PassContext(config={"tl.ascend_exception_dump": True}),
        pytest.raises(ValueError, match="only the Ascend backend"),
    ):
        tilelang.lower(dynamic_copy, target=target)


@pytest.mark.parametrize("target,execution_backend", [("ascend", "cython"), ("pto", "auto"), ("pto", "tvm_ffi"), ("pto", "pto")])
@pytest.mark.parametrize("cache_case", ["disabled", "memory", "disk"])
def test_exception_dump_rejects_unsupported_backend_before_any_cache_hit(monkeypatch, tmp_path, cache_case, target, execution_backend):
    from tilelang.cache import _resolve_cache_dispatch
    from tilelang.env import CacheState, env
    from tilelang.jit.kernel import JITKernel

    cache_enabled = cache_case != "disabled"
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0" if cache_enabled else "1")
    monkeypatch.setattr(CacheState, "_enabled", cache_enabled)
    cache, _, _ = _resolve_cache_dispatch(target, None, execution_backend, False)
    monkeypatch.setattr(cache, "_generate_key", lambda *a, **k: "old-unsupported-dump")
    monkeypatch.setattr(cache, "_memory_cache", {"old-unsupported-dump": object()} if cache_case == "memory" else {})
    monkeypatch.setattr(cache, "_load_kernel_from_disk", lambda *a, **k: object())

    def fail_compile(*args, **kwargs):
        pytest.fail("unsupported dump backend must fail before compilation")

    monkeypatch.setattr(JITKernel, "_compile_and_create_adapter", fail_compile)
    with pytest.raises(ValueError, match="execution_backend='tvm_ffi'"):
        tilelang.compile(dynamic_copy, target=target, execution_backend=execution_backend, pass_configs={"tl.ascend_exception_dump": True})


def test_direct_lower_rejects_cython_exception_dump():
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import lower_with_context

    context = create_backend_context("ascend", execution_backend="cython")
    with (
        tvm.transform.PassContext(config={"tl.ascend_exception_dump": True}),
        pytest.raises(ValueError, match="execution_backend='tvm_ffi'"),
    ):
        lower_with_context(dynamic_copy, context)


@pytest.mark.parametrize("builder", ["target.build.tilelang_pto", "target.build.tilelang_pto_without_compile"])
def test_pto_codegen_rejects_exception_dump_metadata(builder):
    """Shared FunctionInfo must not add a record absent from the PTO entry ABI."""
    import tvm_ffi

    x = tirx.Var("x", tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global"))
    n = tirx.Var("n", "uint64")
    func = (
        tirx.PrimFunc([x, n], tirx.Evaluate(0))
        .with_attr("global_symbol", "pto_dump_metadata")
        .with_attr("calling_conv", 2)
        .with_attr("tirx.kernel_launch_params", ["blockIdx.x"])
        .with_attr(EXCEPTION_DUMP_ATTR, [[0, 1, 0, 32]])
    )
    target = tvm.target.Target({"kind": "ascend", "keys": ["pto", "ascend"]})
    compile_calls = []
    callback = tvm_ffi.get_global_func("tilelang_callback_pto_compile")

    def dummy_compile(*args):
        compile_calls.append(True)
        return "dummy ELF"

    tvm_ffi.register_global_func("tilelang_callback_pto_compile", dummy_compile, override=True)
    try:
        with pytest.raises(ValueError, match="PTO does not support exception dump metadata"):
            tvm_ffi.get_global_func(builder)(tvm.IRModule({"pto_dump_metadata": func}), target)
        assert not compile_calls
    finally:
        tvm_ffi.register_global_func("tilelang_callback_pto_compile", callback, override=True)
