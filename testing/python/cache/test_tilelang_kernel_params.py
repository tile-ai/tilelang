import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.engine.param import KernelParam, dump_kernel_params, load_kernel_params


@pytest.mark.parametrize("encoding", [(17, 4, 1), (17, 8, 1), (17, 8, 4), (2, 32, 1), (0, 64, 1), (2, 32, -4)])
def test_kernel_parameter_dtype_encoding_roundtrip(encoding):
    dtype = tvm_ffi.dtype.from_dlpack_data_type(encoding)
    loaded = load_kernel_params(dump_kernel_params([KernelParam(dtype, [16, 32])]))
    assert loaded[0].dtype == dtype
    assert loaded[0].dtype.bits == encoding[1]
    assert loaded[0].shape == [16, 32]


def test_kernel_parameter_shared_symbol_identity_roundtrip():
    extent = ir.Var("extent", ir.PrimType("int64"))
    dtype = tvm_ffi.dtype("float32")
    params = [KernelParam(dtype, [extent]), KernelParam(dtype, [extent, extent * 2])]
    loaded = load_kernel_params(dump_kernel_params(params))
    first_extent = loaded[0].shape[0]
    assert first_extent.same_as(loaded[1].shape[0])
    assert tirx.analysis.undefined_vars(loaded[1].shape[1])[0].same_as(first_extent)


def test_kernel_parameter_legacy_dtype_string():
    text = ir.save_json(tvm_ffi.convert([{"dtype": "float16", "shape": [8]}]))
    assert load_kernel_params(text) == [KernelParam(tvm_ffi.dtype("float16"), [8])]


def test_native_library_stamp_covers_every_dependency(tmp_path, monkeypatch):
    from tilelang import libinfo

    paths = []
    for name in ("tilelang", "tvm_tirx_compiler", "tvm_runtime", "tvm_ffi"):
        path = tmp_path / f"lib{name}.so"
        path.write_bytes(name.encode())
        paths.append(str(path))
    monkeypatch.setattr(libinfo, "native_library_paths", lambda: paths)
    original = libinfo.native_library_stamp()
    for path in paths:
        with open(path, "ab") as library:
            library.write(b"changed")
        changed = libinfo.native_library_stamp()
        assert changed != original
        original = changed
