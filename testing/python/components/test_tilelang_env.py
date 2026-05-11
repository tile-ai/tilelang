import os
from types import SimpleNamespace

import tilelang
from tilelang.env import configure_rocm_tvm_ffi_dlpack_env


def test_env_var():
    # test default value
    assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "1"
    # test forced value
    os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"
    assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "0"
    # test forced value with class method
    tilelang.env.TILELANG_PRINT_ON_COMPILATION = "1"
    assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "1"


def test_configure_rocm_tvm_ffi_dlpack_env_sets_rocm_flags(monkeypatch):
    monkeypatch.delenv("TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API", raising=False)
    monkeypatch.delenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", raising=False)

    fake_torch = SimpleNamespace(version=SimpleNamespace(hip="7.2"))

    assert configure_rocm_tvm_ffi_dlpack_env(fake_torch)
    assert os.environ["TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API"] == "1"
    assert os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] == "1"


def test_configure_rocm_tvm_ffi_dlpack_env_skips_non_rocm(monkeypatch):
    monkeypatch.delenv("TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API", raising=False)
    monkeypatch.delenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", raising=False)

    fake_torch = SimpleNamespace(version=SimpleNamespace(hip=None))

    assert not configure_rocm_tvm_ffi_dlpack_env(fake_torch)
    assert "TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API" not in os.environ
    assert "TVM_FFI_DISABLE_TORCH_C_DLPACK" not in os.environ


def test_configure_rocm_tvm_ffi_dlpack_env_preserves_user_overrides(monkeypatch):
    monkeypatch.setenv("TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API", "0")
    monkeypatch.setenv("TVM_FFI_DISABLE_TORCH_C_DLPACK", "0")

    fake_torch = SimpleNamespace(version=SimpleNamespace(hip="7.2"))

    assert configure_rocm_tvm_ffi_dlpack_env(fake_torch)
    assert os.environ["TVM_FFI_SKIP_C_DLPACK_EXCHANGE_API"] == "0"
    assert os.environ["TVM_FFI_DISABLE_TORCH_C_DLPACK"] == "0"


if __name__ == "__main__":
    test_env_var()
