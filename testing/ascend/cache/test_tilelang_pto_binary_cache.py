"""PTO binary cache: callback wiring and PTO-specific key semantics.

The byte-level edge cases of the shared BinaryCache base (metadata field
validation, truncated/empty reads, fsync-before-publish atomicity, staging
cleanup) are covered in testing/python/cache/test_tilelang_cuda_binary_cache.py.
"""

from __future__ import annotations

import pytest
import tilelang
from tilelang.cache.ascend_binary_cache import PTOBinaryCache
from tilelang.env import env
from tvm.target import Target

pytestmark = pytest.mark.pto

AIBIN = PTOBinaryCache.binary_format


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.delenv("PTO_ARCH", raising=False)
    monkeypatch.delenv("PTO_FLAGS", raising=False)
    tilelang.enable_cache()
    PTOBinaryCache._get_tilelang_lib_stamp.cache_clear()
    return cache_dir


def _fake_pto_compile(compile_calls):
    def fake_compile_ptodsl_to_aibin(code, pto_arch, pto_flags, verbose=False):
        compile_calls.append((code, pto_arch, tuple(pto_flags)))
        return bytearray(b"fake-aibin")

    return fake_compile_ptodsl_to_aibin


def _patch_callback(monkeypatch, compile_calls, toolchain=None):
    from tilelang.ascend import backend as ascend_backend
    from tilelang.contrib import pto_toolchain

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    monkeypatch.setattr(pto_toolchain, "compile_ptodsl_to_aibin", _fake_pto_compile(compile_calls))
    if toolchain is None:
        toolchain = {"ptoas": "ptoas test", "bisheng": "bisheng test"}
    monkeypatch.setattr(pto_toolchain, "toolchain_stamp", lambda: dict(toolchain))
    return ascend_backend.tilelang_callback_pto_compile


def test_pto_binary_cache_hit_skips_toolchain(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    compile_calls = []
    compile_pto = _patch_callback(monkeypatch, compile_calls)

    target = Target({"kind": "ascend"})
    source = "from ptodsl import pto\n"

    first = compile_pto(source, target)
    second = compile_pto(source, target)
    # Different ptoas flags change the produced ELF without changing the
    # source, so they must NOT share a cache entry.
    monkeypatch.setenv("PTO_FLAGS", "--pto-opt=experimental")
    third = compile_pto(source, target)
    fourth = compile_pto(source, target)

    assert bytes(first) == bytes(second) == bytes(third) == bytes(fourth) == b"fake-aibin"
    # first compiles, second hits; third compiles (new flags), fourth hits
    assert len(compile_calls) == 2
    cache_files = list((tmp_path / "cache").glob(f"*/pto-binaries/*/kernel.{AIBIN}"))
    assert len(cache_files) == 2
    for cache_file in cache_files:
        assert cache_file.read_bytes() == b"fake-aibin"
        assert (cache_file.parent / "metadata.json").is_file()


def test_pto_binary_cache_toolchain_upgrade_misses(monkeypatch, tmp_path):
    """Same source and flags, different ptoas: the cache must miss."""
    _set_cache_dirs(monkeypatch, tmp_path)
    target = Target({"kind": "ascend"})
    source = "from ptodsl import pto\n"

    compile_calls = []
    compile_old = _patch_callback(monkeypatch, compile_calls, toolchain={"ptoas": "ptoas 0.67", "bisheng": "bisheng 15"})
    assert bytes(compile_old(source, target)) == b"fake-aibin"
    assert len(compile_calls) == 1

    compile_calls.clear()
    compile_new = _patch_callback(monkeypatch, compile_calls, toolchain={"ptoas": "ptoas 0.68", "bisheng": "bisheng 15"})
    assert bytes(compile_new(source, target)) == b"fake-aibin"
    assert len(compile_calls) == 1

    cache_files = list((tmp_path / "cache").glob(f"*/pto-binaries/*/kernel.{AIBIN}"))
    assert len(cache_files) == 2


def test_pto_binary_cache_corrupted_entry_recompiles(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    compile_calls = []
    compile_pto = _patch_callback(monkeypatch, compile_calls)

    target = Target({"kind": "ascend"})
    source = "from ptodsl import pto\n"

    compile_pto(source, target)
    assert len(compile_calls) == 1

    [cache_file] = (tmp_path / "cache").glob(f"*/pto-binaries/*/kernel.{AIBIN}")
    # Same-size in-place corruption, so only the metadata sha256 check can
    # catch it.
    cache_file.write_bytes(b"\x00" * cache_file.stat().st_size)

    recovered = compile_pto(source, target)
    assert bytes(recovered) == b"fake-aibin"
    assert len(compile_calls) == 2
