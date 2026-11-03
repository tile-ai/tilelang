"""SuperKernel entries preserve the normal launch ABI and task context."""

import os
import re
import shutil
import struct
import subprocess

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.contrib import bisheng

ENABLE = "tl.ascend_enable_superkernel"
PREFIX = "tl.ascend_superkernel_"


@pytest.fixture(autouse=True)
def _target_context():
    with tvm.target.Target("ascend"):
        yield


def _kernel(kind="vector", vf="simt"):
    @T.prim_func
    def main(A: T.Tensor((512,), "int32"), n: T.int32):
        with T.Kernel(4) as bx:
            if kind == "cube":
                with T.Cube():
                    A[bx] = T.Cast("int32", T.call_extern("int64", "AscendC::GetBlockNum")) + n
            elif kind == "mix":
                with T.Cube():
                    A[bx] = T.Cast("int32", T.call_extern("int64", "AscendC::GetBlockNum")) + n
                with T.Vector(), T.SimtVF(threads=32):
                    for i in T.Parallel(32):
                        A[128 + bx * 32 + i] = bx
            elif vf == "simt":
                count = T.Cast("int32", T.call_extern("int64", "AscendC::GetBlockNum"))
                with T.SimtVF(threads=32):
                    for i in T.Parallel(32):
                        A[bx * 32 + i] = bx + count + n
            else:
                count = T.Cast("int32", T.call_extern("int64", "AscendC::GetBlockNum"))
                with T.SimdVF():
                    T.evaluate(T.call_extern("int32", "consume_context", bx, count))

    return main


def _source(func, **options):
    config = {"tl.enable_auto_schedule": False, ENABLE: True, **{PREFIX + k: v for k, v in options.items()}}
    with tvm.transform.PassContext(config=config):
        return tilelang.lower(func, target="ascend").kernel_source


def _sk_body(source):
    return source.split("sk::SkSystemArgs* tl_sk_sys) {", 1)[1]


def test_disabled_preserves_original_entry():
    source = tilelang.lower(_kernel(), target="ascend").kernel_source
    assert "__global__ __vector__" in source
    assert "__sk__" not in source
    assert "SK_BIND(" not in source
    assert "tl_sk_sys" not in source


@pytest.mark.parametrize("kind,qualifier", [("vector", "__vector__"), ("cube", "__cube__"), ("mix", "__mix__(1, 2)")])
def test_task_context_by_kernel_kind(kind, qualifier):
    source = _source(_kernel(kind))
    assert f'extern "C" __global__ {qualifier}' in source
    assert f"__sk__ {qualifier}" in source
    body = _sk_body(source)
    assert "tl_sk_sys->SkGetNumBlocks()" in body
    assert "AscendC::GetBlockNum()" not in body
    assert "tl_sk_sys->SkInit()" in body
    if kind == "vector":
        assert "tl_sk_sys->SkGetBlockIdx()" in body
        assert " = block_idx" not in body
        # The SIMT helper receives the SK index instead of using a physical builtin.
        helper = re.search(r"__simt_vf__.*main_kernel_sk.*?\{(.*?)\n\}", source, re.S)
        assert helper is not None
        assert "tl_sk_block_idx" in helper.group(0)
    else:
        assert "tl_sk_sys->SkGetBlockIdx()" not in body
        assert "block_idx" in body
    assert source.count("SK_BIND(") == 1
    assert "main_kernel_sk<3>" in source


def test_simd_helper_captures_task_context():
    source = _source(_kernel(vf="simd"))
    assert "consume_context(tl_sk_block_idx" in source
    assert "tl_sk_num_blocks" in source


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("disable_dcci", [False, True])
def test_variant_count_and_cap(count, disable_dcci):
    source = _source(_kernel(), num_variants=count, disable_dcci=disable_dcci)
    binding = source.split("SK_BIND(")[1]
    assert f", {4 if disable_dcci else 0}ULL," in binding
    assert binding.count("_sk<") == count


@pytest.mark.parametrize("count", [-1, 0, 5])
def test_invalid_variant_count(count):
    with pytest.raises(tvm.error.InternalError, match=r"num_variants must be in \[1, 4\]"):
        _source(_kernel(), num_variants=count)


def test_options_require_enable():
    with (
        tvm.transform.PassContext(config={PREFIX + "disable_dcci": False}),
        pytest.raises(
            tvm.error.InternalError,
            match="require tl.ascend_enable_superkernel",
        ),
    ):
        tilelang.lower(_kernel(), target="ascend")


def test_small_scalar_alignment_and_parameter_order():
    @T.prim_func
    def main(A: T.Tensor((8,), "int32"), flag: T.int8, value: T.int64, enabled: T.bool):
        with T.Kernel(1) as bx:
            A[bx] = T.Cast("int32", flag) + T.Cast("int32", value) + T.Cast("int32", enabled)

    source = _source(main)
    fields = re.search(r"struct main_kernel_sk_args \{(.*?)\};", source, re.S).group(1)
    assert re.search(r"alignas\(4\) int8_t arg\d;", fields)
    assert re.search(r"alignas\(4\) bool arg\d;", fields)
    # Lowering may reorder scalar parameters. Match the final device ABI,
    # rather than assuming the original Python parameter order survives.
    signature = re.search(r"void main_kernel\(([^)]*)\)", source).group(1)
    ordinary_types = [p.rsplit(" ", 1)[0].replace(" __restrict__", "") for p in signature.split(", ")]
    field_types = [re.sub(r"alignas\(4\) ", "", line.strip()).rsplit(" ", 1)[0] for line in fields.splitlines() if line.strip()]
    assert field_types == ordinary_types


def test_dynamic_arguments_follow_device_signature():
    rows = T.dynamic("rows")

    @T.prim_func
    def main(A: T.Tensor((rows,), "int32"), B: T.Tensor((rows,), "int32")):
        with T.Kernel(T.ceildiv(rows, 32)) as bx, T.SimtVF(threads=32):
            for i in T.Parallel(32):
                if bx * 32 + i < rows:
                    B[bx * 32 + i] = A[bx * 32 + i]

    source = _source(main)
    signature = re.search(r"void main_kernel\(([^)]*)\)", source).group(1)
    fields = re.search(r"struct main_kernel_sk_args \{(.*?)\};", source, re.S).group(1)
    assert "rows" in signature
    assert len(signature.split(", ")) == len([line for line in fields.splitlines() if line.strip()])
    assert re.search(r"rows = tl_sk_args->arg\d", _sk_body(source))


def test_pto_rejects_superkernel_options():
    with tvm.transform.PassContext(config={ENABLE: True}), pytest.raises(ValueError, match="requires the Ascend C backend"):
        tilelang.lower(_kernel(), target="pto")


def test_registered_superkernel_options():
    registered = {str(key) for key in tvm.transform.PassContext.list_configs() if str(key) == ENABLE or str(key).startswith(PREFIX)}
    assert registered == {ENABLE, PREFIX + "num_variants", PREFIX + "disable_dcci"}


def test_imported_source_is_emitted_once():
    @T.prim_func
    def main(A: T.Tensor((8,), "int32")):
        with T.Kernel(1, prelude="__aicore__ inline int imported_value() { return 7; }") as bx:
            A[bx] = T.call_extern("int32", "imported_value")

    source = _source(main)
    assert source.count("inline int imported_value()") == 1
    assert source.count("imported_value()") == 3


def test_multiple_kernels_and_symbol_collision():
    from tilelang.engine.lower import device_codegen_without_compile
    from tilelang.backend import create_backend_context

    context = create_backend_context("ascend", execution_backend="tvm_ffi")
    with tvm.transform.PassContext(config={ENABLE: True, "tl.enable_auto_schedule": False}):
        device = tilelang.lower(_kernel(), target="ascend").device_mod
        func = next(iter(device.functions.values()))
        mod = tvm.IRModule(
            {
                "left": func.with_attr("global_symbol", "left"),
                "right": func.with_attr("global_symbol", "right"),
            }
        )
        source = device_codegen_without_compile(mod, context).inspect_source()
        assert "SK_BIND(left," in source
        assert "SK_BIND(right," in source
        assert source.count("sk::SkSystemArgs* tl_sk_sys") == 2
        collision = tvm.IRModule(
            {
                "left": func.with_attr("global_symbol", "left"),
                "left_sk": func.with_attr("global_symbol", "left_sk"),
            }
        )
        with pytest.raises(tvm.error.InternalError, match="generated symbol collides"):
            device_codegen_without_compile(collision, context)


@pytest.mark.parametrize("kind", ["vector", "cube", "mix"])
@pytest.mark.parametrize("disable_dcci", [False, True])
def test_sdk_binary_keeps_sk_binding(kind, disable_dcci, tmp_path):
    if shutil.which("bisheng") is None or shutil.which("readelf") is None:
        pytest.skip("CANN bisheng and readelf are required")
    source = _source(_kernel(kind), disable_dcci=disable_dcci)
    binary = bisheng.compile_ascend(
        source,
        target_format="aibin",
        options=["-I" + tilelang.env.TILELANG_TEMPLATE_PATH, *bisheng.normalize_options(os.getenv("TILELANG_TEST_ASCEND_COMPILE_FLAGS"))],
    )
    path = tmp_path / "kernel.aibin"
    path.write_bytes(binary)
    sections = subprocess.check_output(["readelf", "-SW", str(path)], text=True)
    assert ".ascend.meta" in sections
    symbols = subprocess.check_output(["readelf", "-Ws", str(path)], text=True)
    symbols = subprocess.check_output(["c++filt"], input=symbols, text=True)
    for i in range(4):
        assert f"main_kernel_sk<{i}u>" in symbols
    assert "main_kernel" in symbols
    # Check the linked addresses, not only the existence of a metadata section.
    dump = subprocess.check_output(["readelf", "-x", ".ascend.meta", str(path)], text=True)
    chunks = re.findall(r"^\s+0x[0-9a-f]+((?: [0-9a-f]{8}){1,4})", dump, re.M)
    data = bytes.fromhex("".join(chunks))
    functions = {}
    for line in symbols.splitlines():
        fields = line.split(maxsplit=7)
        if len(fields) == 8 and fields[3] == "FUNC":
            functions[int(fields[1], 16)] = fields[7]
    offset = 0
    records = 0
    while offset + 4 <= len(data):
        kind_id, size = struct.unpack_from("<HH", data, offset)
        if kind_id == 5:  # SDK B_TYPE_SK_INFO; BaseTlv + reserved uint32_t.
            cap, ordinary, *entries = struct.unpack_from("<6Q", data, offset + 8)
            assert cap == (4 if disable_dcci else 0)
            name = functions[ordinary]
            assert name in {"main_kernel", "main_kernel_mix_aic", "main_kernel_mix_aiv"}
            for i, address in enumerate(entries):
                assert f"main_kernel_sk<{i}u>" in functions[address]
                if name.endswith("_mix_aic"):
                    assert "[clone .cube]" in functions[address]
                elif name.endswith("_mix_aiv"):
                    assert "[clone .vector]" in functions[address]
            records += 1
        offset += 4 + size
    assert records > 0
