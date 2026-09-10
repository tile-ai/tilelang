"""Adapter and cache contracts for the TileIR JIT backend."""

from __future__ import annotations

import builtins
from dataclasses import replace
import json
from pathlib import Path

import pytest

from tilelang import tvm as tvm
from tilelang.jit.adapter.base import CachedTextSource
from tilelang.jit.adapter.tileir import adapter as tileir_adapter
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter
from tilelang.jit.adapter.tileir.kernel_cache import TileIRKernelCache
from tilelang.tileir import checks
from tilelang.tileir.artifact import (
    TileIRArgumentRef,
    TileIRLaunchMetadata,
    TileIRLoweringResult,
    TileIRTemporaryBuffer,
)
from tvm import tirx

from tileir_jit_test_utils import (
    artifact_compatibility as _artifact_compatibility,
    prim_func_with_interleaved_scalar_param as _prim_func_with_interleaved_scalar_param,
    prim_func_with_two_kernel_launches as _prim_func_with_two_kernel_launches,
)

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal


def test_tileir_adapter_records_native_dispatcher_launcher():
    adapter = TileIRKernelAdapter.__new__(TileIRKernelAdapter)
    adapter.native_dispatcher = object()
    adapter.tileir_artifact = TileIRLoweringResult(
        kernel_name="kernel",
        cubin=b"cubin",
        launch_metadata=TileIRLaunchMetadata(grid=(3, 1, 1)),
    )

    host_source = adapter.get_host_source()

    assert "cuTile native dispatcher" in host_source
    assert "grid=(3, 1, 1)" in host_source


def test_tileir_from_database_preserves_cached_text_source_paths(tmp_path, monkeypatch):
    device_source_path = tmp_path / "kernel.tileir"
    host_source_path = tmp_path / "tileir_launcher.py"
    lib_path = tmp_path / "kernel.tileir.json"
    device_source_path.write_text("module @cached_device", encoding="utf-8")
    host_source_path.write_text("# cached host launcher", encoding="utf-8")

    prim_func = (
        tirx.PrimFunc([], tirx.Evaluate(0))
        .with_attr("global_symbol", "main")
        .with_attr("thread_extent", {"blockIdx.x": tirx.IntImm("int32", 1)})
    )
    artifact = TileIRLoweringResult(
        kernel_name="main",
        cubin=b"cubin",
        launch_metadata=TileIRLaunchMetadata(grid=(1, 1, 1)),
        compatibility=_artifact_compatibility(),
    )
    lib_path.write_bytes(TileIRKernelAdapter._serialize_tileir_artifact(artifact))

    reads: list[Path] = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        path = Path(file) if isinstance(file, (str, Path)) else None
        if path in {device_source_path, host_source_path} and "r" in mode:
            reads.append(path)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    monkeypatch.setattr(TileIRKernelAdapter, "_load_native_dispatcher", lambda self: None)
    monkeypatch.setattr(TileIRKernelAdapter, "_post_init", lambda self: None)
    monkeypatch.setattr(
        tileir_adapter,
        "check_tileir_available",
        lambda: checks.TileIRToolchain(
            cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
            tileiras_path=tmp_path / "tileiras",
            tileiras_version="13.4.0",
            cuda_tile_ir_version="13.4",
            cuda_tile_runtime_version="1.5.0",
        ),
    )

    adapter = TileIRKernelAdapter.from_database(
        params=[],
        result_idx=[],
        target="tileir -arch=sm_120",
        func_or_mod=prim_func,
        host_kernel_source=CachedTextSource(path=str(host_source_path)),
        device_kernel_source=CachedTextSource(path=str(device_source_path)),
        kernel_lib_path=str(lib_path),
    )

    assert reads == []
    assert adapter.device_kernel_source is None
    assert adapter.host_kernel_source is None
    assert adapter._device_kernel_source_path == str(device_source_path)
    assert adapter._host_kernel_source_path == str(host_source_path)
    assert adapter.tileir_artifact.tileir_source is None

    assert adapter.get_kernel_source() == "module @cached_device"
    assert adapter.get_host_source() == "# cached host launcher"
    assert reads == [device_source_path, host_source_path]


def test_tileir_from_database_rejects_incompatible_cached_runtime(tmp_path, monkeypatch):
    path = tmp_path / "kernel.tileir.json"
    artifact = TileIRLoweringResult(
        kernel_name="main",
        cubin=b"cubin",
        compatibility=_artifact_compatibility(runtime_version="1.5.0"),
    )
    path.write_bytes(TileIRKernelAdapter._serialize_tileir_artifact(artifact))
    monkeypatch.setattr(
        tileir_adapter,
        "check_tileir_available",
        lambda: checks.TileIRToolchain(
            cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
            tileiras_path=tmp_path / "tileiras",
            tileiras_version="13.4.0",
            cuda_tile_ir_version="13.4",
            cuda_tile_runtime_version="1.5.1",
        ),
    )
    monkeypatch.setattr(TileIRKernelAdapter, "_load_native_dispatcher", lambda self: None)
    monkeypatch.setattr(TileIRKernelAdapter, "_post_init", lambda self: None)

    prim_func = (
        tirx.PrimFunc([], tirx.Evaluate(0))
        .with_attr("global_symbol", "main")
        .with_attr("thread_extent", {"blockIdx.x": tirx.IntImm("int32", 1)})
    )
    with pytest.raises(ValueError, match="compatibility"):
        TileIRKernelAdapter.from_database(
            params=[],
            result_idx=[],
            target="tileir -arch=sm_120",
            func_or_mod=prim_func,
            host_kernel_source=CachedTextSource(text="# host"),
            device_kernel_source=CachedTextSource(text="module @main"),
            kernel_lib_path=str(path),
        )


def _read_round_trip(tmp_path, artifact):
    path = tmp_path / "kernel.tileir.json"
    path.write_bytes(TileIRKernelAdapter._serialize_tileir_artifact(artifact))
    return TileIRKernelAdapter._read_tileir_artifact(
        str(path),
        _prim_func_with_interleaved_scalar_param(),
        "module @cached_device",
    )


def test_tileir_single_artifact_cache_round_trip_preserves_argument_metadata(tmp_path):
    n = tirx.Var("n", "int32")
    artifact = TileIRLoweringResult(
        kernel_name="single",
        cubin=b"\x00cubin\xff",
        tileir_source="module @single",
        launch_metadata=TileIRLaunchMetadata(grid=(n + 1, 2, 3), block=(4, 5, 6), dynamic_smem_bytes=7),
        scratch_bytes_per_block=256,
        argument_names=("A", "scale", "B"),
        argument_scalar_flags=(False, True, False),
        argument_refs=(
            TileIRArgumentRef("parameter", 0),
            TileIRArgumentRef("parameter", 1),
            TileIRArgumentRef("parameter", 2),
        ),
        compatibility=_artifact_compatibility(),
    )

    payload = TileIRKernelAdapter._serialize_tileir_artifact(artifact)
    envelope = json.loads(payload)
    restored = _read_round_trip(tmp_path, artifact)

    assert envelope["format"] == "tilelang.tileir.artifact"
    assert envelope["version"] == 3
    assert envelope["compatibility"]["target_arch"] == "sm_120"
    assert envelope["compatibility"]["cuda_tile_runtime_version"] == "1.5.0"
    assert restored.kernel_name == artifact.kernel_name
    assert restored.cubin == artifact.cubin
    assert restored.tileir_source == artifact.tileir_source
    assert restored.launch_metadata.block == artifact.launch_metadata.block
    assert restored.launch_metadata.dynamic_smem_bytes == 7
    assert restored.scratch_bytes_per_block == 256
    assert structural_equal(restored.launch_metadata.grid[0], artifact.launch_metadata.grid[0], map_free_vars=True)
    assert restored.launch_metadata.grid[1:] == (2, 3)
    assert restored.argument_names == ("A", "scale", "B")
    assert restored.argument_scalar_flags == (False, True, False)
    assert restored.argument_refs == artifact.argument_refs
    assert restored.compatibility == artifact.compatibility
    assert restored.kernels == ()


def test_tileir_multi_artifact_cache_round_trip_preserves_argument_metadata(tmp_path):
    artifact = TileIRLoweringResult(
        kernel_name="program",
        cubin=b"",
        tileir_source="module @first\nmodule @second",
        argument_names=("A", "scale", "B"),
        argument_scalar_flags=(False, True, False),
        argument_refs=(
            TileIRArgumentRef("parameter", 0),
            TileIRArgumentRef("parameter", 1),
            TileIRArgumentRef("parameter", 2),
        ),
        temporary_buffers=(TileIRTemporaryBuffer(name="scratch", shape=(4,), dtype="float32"),),
        compatibility=_artifact_compatibility(),
        kernels=(
            TileIRLoweringResult(
                kernel_name="first",
                cubin=b"first-cubin",
                launch_metadata=TileIRLaunchMetadata(grid=(1, 1, 1)),
                argument_names=("A", "scale", "scratch"),
                argument_scalar_flags=(False, True, False),
                argument_refs=(
                    TileIRArgumentRef("parameter", 0),
                    TileIRArgumentRef("parameter", 1),
                    TileIRArgumentRef("temporary", 0),
                ),
            ),
            TileIRLoweringResult(
                kernel_name="second",
                cubin=b"second-cubin",
                launch_metadata=TileIRLaunchMetadata(grid=(2, 1, 1)),
                argument_names=("scratch", "B"),
                argument_scalar_flags=(False, False),
                argument_refs=(
                    TileIRArgumentRef("temporary", 0),
                    TileIRArgumentRef("parameter", 2),
                ),
            ),
        ),
    )

    restored = _read_round_trip(tmp_path, artifact)

    assert restored.kernel_name == "program"
    assert restored.argument_names == ("A", "scale", "B")
    assert restored.argument_scalar_flags == (False, True, False)
    assert restored.temporary_buffers == artifact.temporary_buffers
    assert [kernel.kernel_name for kernel in restored.kernels] == ["first", "second"]
    assert [kernel.cubin for kernel in restored.kernels] == [b"first-cubin", b"second-cubin"]
    assert [kernel.argument_scalar_flags for kernel in restored.kernels] == [
        (False, True, False),
        (False, False),
    ]
    assert [kernel.argument_refs for kernel in restored.kernels] == [
        artifact.kernels[0].argument_refs,
        artifact.kernels[1].argument_refs,
    ]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b"not-json", "Malformed TileIR cache artifact"),
        (
            json.dumps({"format": "tilelang.tileir.artifact", "version": 1, "artifact": {}}).encode(),
            "Unsupported TileIR cache artifact version 1",
        ),
        (
            json.dumps({"format": "tilelang.tileir.artifact", "version": 2, "artifact": {}}).encode(),
            "Unsupported TileIR cache artifact version 2",
        ),
    ],
)
def test_tileir_cache_rejects_malformed_or_unknown_version(tmp_path, payload, error):
    path = tmp_path / "kernel.tileir.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=error):
        TileIRKernelAdapter._read_tileir_artifact(
            str(path),
            _prim_func_with_interleaved_scalar_param(),
            "module @cached_device",
        )


@pytest.mark.parametrize("size", [None, True, -16, 1, "256"])
def test_tileir_cache_rejects_invalid_scratch_size(tmp_path, size):
    artifact = TileIRLoweringResult(kernel_name="main", cubin=b"cubin", compatibility=_artifact_compatibility())
    envelope = json.loads(TileIRKernelAdapter._serialize_tileir_artifact(artifact))
    envelope["artifact"]["scratch_bytes_per_block"] = size
    path = tmp_path / "kernel.tileir.json"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="scratch size"):
        TileIRKernelAdapter._read_tileir_artifact(str(path), _prim_func_with_interleaved_scalar_param(), "module @cached_device")


@pytest.mark.parametrize("extent", [0, -1])
def test_tileir_cache_rejects_non_positive_static_grid_extent(tmp_path, extent):
    artifact = TileIRLoweringResult(
        kernel_name="main",
        cubin=b"cubin",
        compatibility=_artifact_compatibility(),
    )
    envelope = json.loads(TileIRKernelAdapter._serialize_tileir_artifact(artifact))
    envelope["artifact"]["launch_metadata"]["grid"][0] = {"kind": "int", "value": extent}
    path = tmp_path / "kernel.tileir.json"
    path.write_text(json.dumps(envelope))

    with pytest.raises(ValueError, match="positive"):
        TileIRKernelAdapter._read_tileir_artifact(
            str(path),
            _prim_func_with_interleaved_scalar_param(),
            "module @cached_device",
        )


def test_tileir_cache_uses_non_executable_artifact_filename():
    assert TileIRKernelCache.kernel_lib_path == "kernel.tileir.json"


def test_tileir_adapter_attaches_ordered_argument_metadata_for_single_and_multi():
    prim_func = _prim_func_with_interleaved_scalar_param()
    multi = TileIRLoweringResult(
        kernel_name="program",
        cubin=b"",
        kernels=(
            TileIRLoweringResult(
                kernel_name="first",
                cubin=b"first",
                argument_names=("A", "scale"),
            ),
            TileIRLoweringResult(
                kernel_name="second",
                cubin=b"second",
                argument_names=("scale", "B"),
            ),
        ),
    )

    restored = TileIRKernelAdapter._attach_argument_metadata(multi, prim_func)

    assert restored.argument_names == ("A", "scale", "B")
    assert restored.argument_scalar_flags == (False, True, False)
    assert [kernel.argument_scalar_flags for kernel in restored.kernels] == [
        (False, True),
        (True, False),
    ]
    assert restored.argument_refs == tuple(TileIRArgumentRef("parameter", index) for index in range(3))
    assert [kernel.argument_refs for kernel in restored.kernels] == [
        (TileIRArgumentRef("parameter", 0), TileIRArgumentRef("parameter", 1)),
        (TileIRArgumentRef("parameter", 1), TileIRArgumentRef("parameter", 2)),
    ]


def test_tileir_adapter_rejects_ambiguous_argument_names():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global")
    first_handle = tirx.Var("first_handle", pointer_type)
    second_handle = tirx.Var("second_handle", pointer_type)
    first = tirx.decl_buffer((4,), "float32", name="duplicate", data=first_handle)
    second = tirx.decl_buffer((4,), "float32", name="duplicate", data=second_handle)
    prim_func = tirx.PrimFunc(
        [first_handle, second_handle],
        tirx.Evaluate(0),
        buffer_map={first_handle: first, second_handle: second},
    )
    artifact = TileIRLoweringResult(kernel_name="main", cubin=b"cubin")

    with pytest.raises(ValueError, match="duplicate TileIR ABI argument name"):
        TileIRKernelAdapter._attach_argument_metadata(artifact, prim_func)


def test_tileir_cached_artifact_abi_rejects_altered_refs_and_scalar_flags():
    prim_func = _prim_func_with_two_kernel_launches()
    first = TileIRLoweringResult(
        kernel_name="main_0",
        cubin=b"first",
        argument_names=("A", "scale"),
        argument_scalar_flags=(False, True),
        argument_refs=(TileIRArgumentRef("parameter", 0), TileIRArgumentRef("parameter", 1)),
    )
    second = TileIRLoweringResult(
        kernel_name="main_1",
        cubin=b"second",
        argument_names=("A", "B"),
        argument_scalar_flags=(False, False),
        argument_refs=(TileIRArgumentRef("parameter", 0), TileIRArgumentRef("parameter", 2)),
    )
    artifact = TileIRLoweringResult(
        kernel_name="main",
        cubin=b"",
        argument_names=("A", "scale", "B"),
        argument_scalar_flags=(False, True, False),
        argument_refs=tuple(TileIRArgumentRef("parameter", index) for index in range(3)),
        kernels=(first, second),
    )

    TileIRKernelAdapter._validate_cached_artifact_abi(artifact, prim_func)

    invalid_first_kernels = (
        replace(first, argument_refs=(TileIRArgumentRef("parameter", 2), TileIRArgumentRef("parameter", 1))),
        replace(first, argument_refs=(TileIRArgumentRef("temporary", 0), TileIRArgumentRef("parameter", 1))),
        replace(first, argument_scalar_flags=(False, False)),
    )
    for invalid_first in invalid_first_kernels:
        with pytest.raises(ValueError, match="cached TileIR artifact ABI"):
            TileIRKernelAdapter._validate_cached_artifact_abi(replace(artifact, kernels=(invalid_first, second)), prim_func)


def test_tileir_cached_artifact_abi_binds_dynamic_grid_vars_by_name():
    pointer_type = tvm.ir.PointerType(tvm.ir.PrimType("float32"), "global")
    a_handle = tirx.Var("a_handle", pointer_type)
    active_n = tirx.Var("n", "int32")
    a_buffer = tirx.decl_buffer((active_n,), "float32", name="A", data=a_handle)
    prim_func = (
        tirx.PrimFunc([a_handle], tirx.Evaluate(0), buffer_map={a_handle: a_buffer})
        .with_attr("global_symbol", "main")
        .with_attr("thread_extent", {"blockIdx.x": active_n + 1})
    )
    cached_n = tirx.Var("n", "int32")
    artifact = TileIRLoweringResult(
        kernel_name="main",
        cubin=b"cubin",
        launch_metadata=TileIRLaunchMetadata(grid=(cached_n + 1, 1, 1)),
        argument_names=("A",),
        argument_scalar_flags=(False,),
        argument_refs=(TileIRArgumentRef("parameter", 0),),
    )

    TileIRKernelAdapter._validate_cached_artifact_abi(artifact, prim_func)

    bogus = tirx.Var("bogus", "int32")
    with pytest.raises(ValueError, match="launch metadata"):
        TileIRKernelAdapter._validate_cached_artifact_abi(
            replace(artifact, launch_metadata=TileIRLaunchMetadata(grid=(bogus + 1, 1, 1))),
            prim_func,
        )
