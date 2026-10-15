"""Ascend binary cache: backend wiring and Ascend-specific cache semantics.

The byte-level edge cases of the shared BinaryCache base (metadata field
validation, truncated/empty reads, fsync-before-publish atomicity, staging
cleanup) are covered in testing/python/cache/test_tilelang_cuda_binary_cache.py.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import tilelang
from tilelang.cache.ascend_binary_cache import AscendBinaryCache
from tilelang.env import env
from tvm.target import Target

AIBIN = AscendBinaryCache.binary_format


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    AscendBinaryCache._get_tilelang_lib_stamp.cache_clear()
    return cache_dir


def _fake_bisheng_compile(compile_calls):
    def fake_compile_ascend(code, target_format="o", npu_arch=None, options=None, verbose=False, **kwargs):
        compile_calls.append((code, target_format, npu_arch, tuple(options or ())))
        return bytearray(b"fake-aibin")

    return fake_compile_ascend


def test_ascend_binary_cache_hit_skips_bisheng_compile(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.ascend import backend as ascend_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []
    monkeypatch.setattr(ascend_backend.bisheng, "compile_ascend", _fake_bisheng_compile(compile_calls))

    target = Target({"kind": "ascend", "arch": "dav-3510"})
    source = "extern void kernel() {}"

    extra_flags_pass_configs = {
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-mllvm -cce-aicore-function-stack-size=0x8000"],
    }

    first = ascend_backend.tilelang_callback_ascend_compile(source, target)
    second = ascend_backend.tilelang_callback_ascend_compile(source, target)
    # Different device compile flags change the generated CCE ELF without
    # changing the source, so they must NOT share a cache entry.
    third = ascend_backend.tilelang_callback_ascend_compile(source, target, extra_flags_pass_configs)
    fourth = ascend_backend.tilelang_callback_ascend_compile(source, target, extra_flags_pass_configs)

    assert bytes(first) == b"fake-aibin"
    assert bytes(second) == b"fake-aibin"
    assert bytes(third) == b"fake-aibin"
    assert bytes(fourth) == b"fake-aibin"
    # first compiles, second hits; third compiles (new options), fourth hits
    assert len(compile_calls) == 2
    assert compile_calls[0][3] != compile_calls[1][3]
    cache_files = list((tmp_path / "cache").glob(f"*/ascend-binaries/*/kernel.{AIBIN}"))
    assert len(cache_files) == 2
    for cache_file in cache_files:
        assert cache_file.read_bytes() == b"fake-aibin"
        assert (cache_file.parent / "metadata.json").is_file()


def test_ascend_binary_cache_corrupted_entry_recompiles(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.ascend import backend as ascend_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []
    monkeypatch.setattr(ascend_backend.bisheng, "compile_ascend", _fake_bisheng_compile(compile_calls))

    target = Target({"kind": "ascend", "arch": "dav-3510"})
    source = "extern void kernel() {}"

    ascend_backend.tilelang_callback_ascend_compile(source, target)
    assert len(compile_calls) == 1

    [cache_file] = (tmp_path / "cache").glob(f"*/ascend-binaries/*/kernel.{AIBIN}")
    # Same-size corruption, as left behind by a crashed writer/filesystem client.
    corrupted = b"\x00" * cache_file.stat().st_size
    cache_file.write_bytes(corrupted)
    metadata_path = cache_file.parent / "metadata.json"
    original_metadata = metadata_path.read_bytes()

    recompiled = ascend_backend.tilelang_callback_ascend_compile(source, target)
    assert bytes(recompiled) == b"fake-aibin"
    assert len(compile_calls) == 2

    # Shared entries stay immutable even after a miss. Recompilation is usable
    # for this call, but repairing the on-disk entry needs offline cleanup.
    assert bytes(ascend_backend.tilelang_callback_ascend_compile(source, target)) == b"fake-aibin"
    assert len(compile_calls) == 3
    assert cache_file.read_bytes() == corrupted
    assert metadata_path.read_bytes() == original_metadata


def test_ascend_binary_cache_directory_coexists_with_legacy_flat_entry(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "legacy-key"
    cache_root = AscendBinaryCache._get_cache_root()
    legacy_path = os.path.join(cache_root, f"{key}.{AIBIN}")
    os.makedirs(cache_root, exist_ok=True)
    with open(legacy_path, "wb") as f:
        f.write(b"legacy-aibin")

    assert AscendBinaryCache.load(key, AIBIN) is None
    assert os.path.exists(legacy_path)

    AscendBinaryCache.save(key, AIBIN, b"new-aibin")

    assert AscendBinaryCache.load(key, AIBIN) == b"new-aibin"
    assert Path(legacy_path).read_bytes() == b"legacy-aibin"


@pytest.mark.parametrize("foreign_format", ["tilelang.cuda-binary-cache.v1", "tilelang.ascend-binary-cache.v2"])
def test_ascend_binary_cache_rejects_foreign_format_tag(monkeypatch, tmp_path, foreign_format):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "foreign-format-key"
    AscendBinaryCache.save(key, AIBIN, b"valid-aibin")
    path = Path(AscendBinaryCache.get_path(key, AIBIN))
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["format"] = foreign_format
    metadata_path.write_text(json.dumps(metadata))

    assert AscendBinaryCache.load(key, AIBIN) is None
    assert path.read_bytes() == b"valid-aibin"
    assert json.loads(metadata_path.read_text()) == metadata


def test_ascend_binary_cache_first_valid_writer_wins(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "first-writer-key"
    path = Path(AscendBinaryCache.get_path(key, AIBIN))
    AscendBinaryCache.save(key, AIBIN, b"first-binary")
    first_inode = path.stat().st_ino
    with path.open("rb") as reader:
        AscendBinaryCache.save(key, AIBIN, b"second-binary")
        assert reader.read() == b"first-binary"

    assert AscendBinaryCache.load(key, AIBIN) == b"first-binary"
    assert path.stat().st_ino == first_inode
    assert sorted(p.name for p in path.parent.iterdir()) == [f"kernel.{AIBIN}", "metadata.json"]


def test_ascend_binary_cache_concurrent_publishers_do_not_replace_winner(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    writer_count = 16
    barrier = threading.Barrier(writer_count)
    payloads = [f"aibin-{index}".encode() for index in range(writer_count)]
    real_rename = os.rename
    winners = []

    def concurrent_rename(src, dst):
        # Force every writer to stage its files before any can publish.
        barrier.wait(timeout=10)
        real_rename(src, dst)
        winners.append((Path(dst) / f"kernel.{AIBIN}").read_bytes())

    monkeypatch.setattr(os, "rename", concurrent_rename)

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        list(executor.map(lambda data: AscendBinaryCache.save("concurrent-key", AIBIN, data), payloads))

    assert len(winners) == 1
    assert winners[0] in payloads
    assert AscendBinaryCache.load("concurrent-key", AIBIN) == winners[0]
    cache_entries = os.listdir(AscendBinaryCache._get_cache_root())
    assert cache_entries == ["concurrent-key"]
    assert not list(Path(AscendBinaryCache._get_staging_root()).iterdir())


def test_ascend_binary_cache_rejects_empty_save(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="empty Ascend binary"):
        AscendBinaryCache.save("empty-save-key", AIBIN, b"")
