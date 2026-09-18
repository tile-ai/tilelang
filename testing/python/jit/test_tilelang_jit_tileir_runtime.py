"""Native runtime contracts for the TileIR JIT backend."""

from __future__ import annotations

import sys
import types

import pytest

from tilelang.jit.adapter.tileir import runtime as tileir_runtime
from tilelang.tileir.artifact import (
    TileIRArgumentRef,
    TileIRLaunchMetadata,
    TileIRLoweringResult,
    TileIRTemporaryBuffer,
)

from tileir_jit_test_utils import prim_func_with_interleaved_scalar_param as _prim_func_with_interleaved_scalar_param


def test_tileir_load_native_dispatchers_uses_runtime_argument_count(monkeypatch):
    calls = []

    class FakeDispatcher:
        def __init__(self, cubin, symbol, num_args):
            calls.append((cubin, symbol, num_args))
            self.dispatcher = symbol

    monkeypatch.setattr(tileir_runtime, "PrecompiledTileIRDispatcher", FakeDispatcher)

    single = types.SimpleNamespace(
        tileir_artifact=TileIRLoweringResult(
            kernel_name="single",
            cubin=b"single",
            argument_names=("A", "scale", "B"),
            argument_scalar_flags=(False, True, False),
        ),
        params=[object(), object(), object()],
    )
    tileir_runtime.load_native_dispatchers(single)

    multi = types.SimpleNamespace(
        tileir_artifact=TileIRLoweringResult(
            kernel_name="program",
            cubin=b"",
            kernels=(
                TileIRLoweringResult(
                    kernel_name="first",
                    cubin=b"first",
                    argument_names=("A", "scale"),
                    argument_scalar_flags=(False, True),
                ),
                TileIRLoweringResult(
                    kernel_name="second",
                    cubin=b"second",
                    argument_names=("scale", "B"),
                    argument_scalar_flags=(True, False),
                ),
            ),
        ),
        params=[object(), object(), object()],
    )
    tileir_runtime.load_native_dispatchers(multi)

    assert calls == [
        (b"single", "single", 3),
        (b"first", "first", 2),
        (b"second", "second", 2),
    ]


def test_precompiled_dispatcher_uses_cutile_15_annotations_and_compile_contract(monkeypatch):
    annotations = []

    class FakeLeafAnnotationNode:
        def __init__(self, *, constant):
            self.constant = constant

    class FakeTileDispatcher:
        def __init__(self, parameter_annotations):
            annotations.extend(parameter_annotations)

    monkeypatch.setitem(
        sys.modules,
        "cuda.tile._annotated_function",
        types.SimpleNamespace(LeafAnnotationNode=FakeLeafAnnotationNode),
    )
    monkeypatch.setitem(sys.modules, "cuda.tile._cext", types.SimpleNamespace(TileDispatcher=FakeTileDispatcher))

    precompiled = tileir_runtime.PrecompiledTileIRDispatcher(b"cubin", "kernel", 3)

    assert [annotation.constant for annotation in annotations] == [False, False, False]
    assert precompiled.dispatcher._compile(object(), object()) == (b"cubin", "kernel", None, [])


def _runtime_adapter(artifact, prim_func, dispatchers):
    return types.SimpleNamespace(
        tileir_artifact=artifact,
        prim_func=prim_func,
        params=[types.SimpleNamespace(dtype="float32") for _ in prim_func.params],
        result_idx=[],
        param_dtypes=[None, None, None],
        param_shapes=[[], [], []],
        native_dispatcher=dispatchers[0],
        native_dispatchers=dispatchers,
        get_current_stream_functor=lambda: lambda: 17,
        get_current_device_functor=lambda: lambda: "cpu",
    )


def test_tileir_runtime_converts_zero_dim_scalar_args_consistently_for_single_and_multi(monkeypatch):
    torch = pytest.importorskip("torch")
    launches = []
    current_stream = object()
    default_stream = object()
    external_stream = object()
    external_stream_handles = []

    def fake_launch(stream, grid, dispatcher, args):
        launches.append((stream, grid, dispatcher, args))

    monkeypatch.setitem(sys.modules, "cuda.tile._cext", types.SimpleNamespace(launch=fake_launch))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda: default_stream)
    monkeypatch.setattr(
        torch.cuda,
        "ExternalStream",
        lambda handle: external_stream_handles.append(handle) or external_stream,
    )
    prim_func = _prim_func_with_interleaved_scalar_param()
    a = torch.ones(4)
    scale = torch.tensor(2.5)
    b = torch.zeros(4)

    single_artifact = TileIRLoweringResult(
        kernel_name="single",
        cubin=b"single",
        launch_metadata=TileIRLaunchMetadata(grid=(1, 1, 1)),
        argument_names=("A", "scale", "B"),
        argument_scalar_flags=(False, True, False),
    )
    single_dispatcher = types.SimpleNamespace(dispatcher="single-dispatcher")
    single_func = tileir_runtime.make_torch_func(_runtime_adapter(single_artifact, prim_func, [single_dispatcher]))
    single_func(a, scale, b)

    multi_artifact = TileIRLoweringResult(
        kernel_name="program",
        cubin=b"",
        argument_names=("A", "scale", "B"),
        argument_scalar_flags=(False, True, False),
        argument_refs=tuple(TileIRArgumentRef("parameter", index) for index in range(3)),
        kernels=(
            TileIRLoweringResult(
                kernel_name="first",
                cubin=b"first",
                launch_metadata=TileIRLaunchMetadata(grid=(1, 1, 1)),
                argument_names=("A", "scale"),
                argument_scalar_flags=(False, True),
                argument_refs=(TileIRArgumentRef("parameter", 0), TileIRArgumentRef("parameter", 1)),
            ),
            TileIRLoweringResult(
                kernel_name="second",
                cubin=b"second",
                launch_metadata=TileIRLaunchMetadata(grid=(1, 1, 1)),
                argument_names=("scale", "B"),
                argument_scalar_flags=(True, False),
                argument_refs=(TileIRArgumentRef("parameter", 1), TileIRArgumentRef("parameter", 2)),
            ),
        ),
    )
    multi_dispatchers = [
        types.SimpleNamespace(dispatcher="first-dispatcher"),
        types.SimpleNamespace(dispatcher="second-dispatcher"),
    ]
    tileir_runtime.make_torch_func(_runtime_adapter(multi_artifact, prim_func, multi_dispatchers))(a, scale, b)

    assert launches[0][3][0] is a
    assert launches[0][3][1] == 2.5
    assert launches[0][3][2] is b
    assert launches[1][3][0] is a
    assert launches[1][3][1] == 2.5
    assert launches[2][3][0] == 2.5
    assert launches[2][3][1] is b
    assert all(call[0] is current_stream for call in launches)

    single_func(a, scale, b, stream=0)
    single_func(a, scale, b, stream=17)
    assert launches[3][0] is default_stream
    assert launches[4][0] is external_stream
    assert external_stream_handles == [17]


def test_tileir_multi_kernel_runtime_routes_arguments_by_stable_reference(monkeypatch):
    torch = pytest.importorskip("torch")
    launches = []

    def fake_launch(stream, grid, dispatcher, args):
        launches.append((dispatcher, args))

    monkeypatch.setitem(sys.modules, "cuda.tile._cext", types.SimpleNamespace(launch=fake_launch))
    prim_func = _prim_func_with_interleaved_scalar_param()
    a = torch.ones(4)
    scale = 2.5
    b = torch.zeros(4)
    artifact = TileIRLoweringResult(
        kernel_name="program",
        cubin=b"",
        argument_names=("A", "scale", "B"),
        argument_scalar_flags=(False, True, False),
        argument_refs=tuple(TileIRArgumentRef("parameter", index) for index in range(3)),
        temporary_buffers=(TileIRTemporaryBuffer(name="scratch", shape=(2,), dtype="float32"),),
        kernels=(
            TileIRLoweringResult(
                kernel_name="first",
                cubin=b"first",
                argument_names=("deliberately-colliding", "deliberately-colliding", "deliberately-colliding"),
                argument_scalar_flags=(False, False, False),
                argument_refs=(
                    TileIRArgumentRef("parameter", 0),
                    TileIRArgumentRef("temporary", 0),
                    TileIRArgumentRef("parameter", 2),
                ),
            ),
        ),
    )
    dispatchers = [types.SimpleNamespace(dispatcher="first-dispatcher")]

    tileir_runtime.make_torch_func(_runtime_adapter(artifact, prim_func, dispatchers))(a, scale, b)

    assert len(launches) == 1
    assert launches[0][0] == "first-dispatcher"
    assert launches[0][1][0] is a
    assert launches[0][1][1].shape == (2,)
    assert launches[0][1][2] is b
