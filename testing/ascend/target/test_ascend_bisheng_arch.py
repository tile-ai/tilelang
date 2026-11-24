import io
import subprocess
from types import SimpleNamespace

import pytest

from tvm.target import Target

from tilelang.contrib import bisheng
from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.jit.adapter import libgen
from tilelang.jit import kernel as jit_kernel
from tilelang.transform import PassConfigKey


@pytest.mark.parametrize("arch", ["dav-2201", "dav-3510"])
def test_cann_compat_shared_controls_compile(arch):
    try:
        bisheng.find_bisheng_path()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    source = """
    #include "tl_templates/ascend/cann_compat_begin.h"
    extern "C" __global__ __cube__ void shared_controls() {
        ::asc_set_atomic_add_int32();
        asc_set_atomic_max_int32();
        asc_set_atomic_min_int32();
        asc_set_hf32_round_mode(asc_hf32_round_mode::NEAREST_EVEN);
        asc_disable_hf32();
    }
    #include "tl_templates/ascend/cann_compat_end.h"
    """
    binary = bisheng.compile_ascend(source, target_format="aibin", npu_arch=arch, options=["-I" + TILELANG_TEMPLATE_PATH])
    assert binary.startswith(b"\x7fELF")


def test_cann_dma_original_names_preserve_wide_counts():
    try:
        bisheng.find_bisheng_path()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    # Compile only. Calling the CANN 9.1 strided overload directly narrows
    # 65536 to zero. Check both calls and keep subsequent includes unaffected.
    source = """
    #include "tl_templates/ascend/cann_compat_begin.h"
    extern "C" __global__ __vector__ void dma(__gm__ uint8_t* gm) {
        auto ub = (__ubuf__ uint8_t*)0;
        asc_copy_gm2ub_align(ub, gm, 65536u, 32u, 0, 0, false,
                             tl::cann::load_l2_cache_mode(0), 32u, 32u);
        asc_copy_ub2gm_align(gm, ub, 65536u, 32u,
                             tl::cann::store_l2_cache_mode(0), 32u, 32u);
    }
    #include "tl_templates/ascend/cann_compat_end.h"
    #if defined(asc_copy_gm2ub_align) || defined(asc_copy_ub2gm_align)
    #error DMA aliases leaked past TileLang device code
    #endif
    """
    binary = bisheng.compile_ascend(
        source, target_format="aibin", npu_arch="dav-3510", options=["-I" + TILELANG_TEMPLATE_PATH, "-Werror=constant-conversion"]
    )
    assert binary.startswith(b"\x7fELF")


def test_native_cann_register_overloads_compile():
    try:
        bisheng.find_bisheng_path()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    source = """
    #include "tl_templates/ascend/cann_compat.h"
    #define asc_div(...) previous_binding(__VA_ARGS__)
    #define TL_CANN_SIMD_SCOPE
    #include "tl_templates/ascend/cann_compat_begin.h"
    __simd_vf__ inline void registers(__ubuf__ uint32_t* p, __ubuf__ uint8_t* bytes) {
        vector_bool mask = asc_create_mask_b32(PAT_ALL);
        vector_u32 a, b;
        asc_loadalign_brc_elem(a, p);
        asc_loadalign_brc_elem_postupdate(b, p, 1);
        auto c = asc_loadalign_brc_elem(p);
        auto block = asc_loadalign_brc_datablock(p);
        auto spread = asc_loadalign_brc_elem2datablock(p);
        auto unpacked = asc_loadalign_unpack4(bytes);
        asc_storealign(bytes, unpacked, 0, mask);
        asc_storealign(p, asc_div(block, spread, mask), 0, mask);
        auto packed = asc_pack_to_low(c);
        static_assert(std::is_same_v<decltype(packed), vector_u16>);
        static_assert(std::is_same_v<decltype(asc_pack_to_high(mask)), vector_bool>);
        asc_storealign((__ubuf__ uint16_t*)bytes, packed, 0, mask);
        auto q = ::asc_div(a, b, mask);
        vector_bf16 bf;
        asc_ne(mask, bf, bf, mask);
        vector_f8e8m0 scale;
        asc_duplicate(scale, scale, mask);
        asc_select(scale, scale, scale, mask);
        asc_storealign_postupdate(p, q, 1, mask);
        asc_storealign_postupdate(p, q, 1, 2, mask);
        asc_storealign(bytes, __builtin_bit_cast(vector_u8, scale), 0, mask);
    }
    #include "tl_templates/ascend/cann_compat_end.h"
    #undef TL_CANN_SIMD_SCOPE
    #if defined(asc_select) || defined(asc_storealign_postupdate)
    #error leaked SIMD routing
    #endif
    int previous_binding(int);
    inline int restored_binding(int x) { return asc_div(x); }
    #undef asc_div
    extern "C" __global__ __vector__ void entry() {
        registers((__ubuf__ uint32_t*)0, (__ubuf__ uint8_t*)256);
    }
    """
    binary = bisheng.compile_ascend(source, target_format="aibin", npu_arch="dav-3510", options=["-I" + TILELANG_TEMPLATE_PATH])
    assert binary.startswith(b"\x7fELF")


def test_target_npu_arch_priority(monkeypatch):
    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-env")

    target = Target({"kind": "ascend", "arch": "dav-arch", "mcpu": "dav-mcpu"})
    assert bisheng.get_target_npu_arch(target) == "dav-arch"

    target = Target({"kind": "ascend", "mcpu": "dav-mcpu"})
    assert bisheng.get_target_npu_arch(target) == "dav-mcpu"
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-env"

    monkeypatch.delenv("ASCEND_NPU_ARCH")
    assert bisheng.get_target_npu_arch(Target("ascend")) == "dav-3510"


@pytest.mark.parametrize(
    "configured_flags",
    [None, "-mllvm -cce-aicore-dcpreload-args=false -include configured_header.h"],
    ids=["no-configured-flags", "configured-flags"],
)
@pytest.mark.parametrize(
    "compile_flags",
    [
        None,
        ["-mllvm", "-cce-aicore-dcpreload-args=true", "-mllvm", "-inline-threshold=128", "-include", "explicit_header.h"],
    ],
    ids=["no-explicit-flags", "explicit-flags"],
)
@pytest.mark.parametrize("via_jit", [False, True], ids=["direct", "jit"])
def test_cython_compile_preserves_target_and_options(monkeypatch, tmp_path, configured_flags, compile_flags, via_jit):
    source_path = tmp_path / "kernel.asc"
    captured_command = []

    class SourceFile(io.StringIO):
        name = str(source_path)

    source_file = SourceFile()

    monkeypatch.setenv("ASCEND_NPU_ARCH", "dav-env")
    monkeypatch.setattr(bisheng, "find_bisheng_path", lambda: "bisheng")
    monkeypatch.setattr(
        libgen.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: source_file,
    )

    def fake_run(command, **kwargs):
        captured_command.extend(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(libgen.subprocess, "run", fake_run)

    target = Target({"kind": "ascend", "arch": "dav-target"})
    pass_configs = {PassConfigKey.TL_DEVICE_COMPILE_FLAGS: configured_flags}
    source = 'extern "C" void kernel() {}'

    def compile_library(**kwargs):
        generator = libgen.LibraryGenerator(kwargs["target"])
        generator.assign_pass_configs(kwargs["pass_configs"])
        generator.assign_compile_flags(kwargs["compile_flags"])
        generator.update_lib_code(kwargs["device_kernel_source"])
        generator.compile_lib()
        return generator

    if via_jit:
        kernel = object.__new__(jit_kernel.JITKernel)
        kernel.target = target
        kernel.target_host = None
        kernel.execution_backend = "cython"
        kernel.verbose = False
        kernel.pass_configs = pass_configs
        kernel.compile_flags = compile_flags
        artifact = SimpleNamespace(params=[], host_mod=None, device_mod=None, kernel_source=source)
        monkeypatch.setattr(kernel, "_compile_artifact", lambda *args: artifact)
        monkeypatch.setattr(jit_kernel, "CythonKernelAdapter", compile_library)
        kernel._compile_and_create_adapter(SimpleNamespace(attrs={"global_symbol": "kernel"}), [])
    else:
        compile_library(target=target, pass_configs=pass_configs, compile_flags=compile_flags, device_kernel_source=source)

    assert [option for option in captured_command if option.startswith("--npu-arch=")] == ["--npu-arch=dav-target"]

    expected_llvm_options = ["-cce-aicore-dcpreload-args=false"]
    if configured_flags is not None:
        expected_llvm_options.append("-cce-aicore-dcpreload-args=false")
    if compile_flags is not None:
        expected_llvm_options.extend(["-cce-aicore-dcpreload-args=true", "-inline-threshold=128"])
    llvm_pairs = [captured_command[index : index + 2] for index, option in enumerate(captured_command) if option == "-mllvm"]
    assert llvm_pairs == [["-mllvm", option] for option in expected_llvm_options]

    expected_headers = []
    if configured_flags is not None:
        expected_headers.append("configured_header.h")
    if compile_flags is not None:
        expected_headers.append("explicit_header.h")
    include_pairs = [captured_command[index : index + 2] for index, option in enumerate(captured_command) if option == "-include"]
    assert include_pairs == [["-include", header] for header in expected_headers]
