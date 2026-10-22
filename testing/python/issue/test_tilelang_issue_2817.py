# Regression test for https://github.com/tile-ai/tilelang/issues/2817
# Cached kernel parameters used to be stored with cloudpickle, so loading a
# cache entry deserialized arbitrary pickle bytes. They are now stored as
# JSON produced by TVM's reflection (``tvm.ir.save_json``), which only
# rebuilds IR nodes and never executes code. Host-side only, no GPU needed.
import json
from pathlib import Path

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params
from tilelang.env import env
from tvm import tirx


def _params():
    n = tirx.Var("n", "int32")
    return [
        KernelParam(tvm.DataType("float16"), [n, 64]),
        KernelParam(tvm.DataType("int8"), [n * 2, 8]),
        KernelParam(tvm.DataType("float8_e4m3"), [4, 4]),
        KernelParam(tvm.DataType("int32"), []),
    ]


def _check_round_trip(params, loaded):
    assert [str(p.dtype) for p in loaded] == [str(p.dtype) for p in params]
    assert [len(p.shape) for p in loaded] == [2, 2, 2, 0]
    assert loaded[0].shape[1] == 64 and isinstance(loaded[0].shape[1], int)
    assert isinstance(loaded[0].shape[0], tirx.Var) and loaded[0].shape[0].name == "n"
    # A symbolic dim shared by two params is the same Var after loading.
    assert loaded[1].shape[0].a.same_as(loaded[0].shape[0])
    tvm.ir.assert_structural_equal(loaded[1].shape[0], loaded[0].shape[0] * 2)
    assert loaded[2].is_float8()
    assert loaded[3].is_scalar()


def test_kernel_params_round_trip_through_json():
    params = _params()
    text = dump_kernel_params(params)
    # Plain JSON text, not a pickle stream.
    assert isinstance(text, str)
    json.loads(text)
    _check_round_trip(params, load_kernel_params(text))


def test_load_kernel_params_rejects_non_json_input():
    with pytest.raises(ValueError):
        load_kernel_params("\x80\x04not a json document")


class _FakeAdapter:
    def __init__(self, libpath: str):
        self.libpath = libpath

    def get_kernel_source(self):
        return "// host kernel"


class _FakeKernel:
    def __init__(self, libpath: str, params):
        self.adapter = _FakeAdapter(libpath)
        self.kernel_source = "// device kernel"
        self.params = params


def test_kernel_cache_stores_params_as_json(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    cache = KernelCache()
    lib_path = tmp_path / "kernel_lib.so"
    lib_path.write_bytes(b"fake-so")
    params = _params()

    cache._save_kernel_to_disk("json-params", _FakeKernel(str(lib_path), params))

    cache_path = Path(cache._get_cache_path("json-params"))
    params_file = cache_path / cache.params_path
    assert params_file.suffix == ".json"
    assert not (cache_path / "params.pkl").exists()
    text = params_file.read_text()
    json.loads(text)
    _check_round_trip(params, load_kernel_params(text))


if __name__ == "__main__":
    tilelang.testing.main()
