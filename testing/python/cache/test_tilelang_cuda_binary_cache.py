from __future__ import annotations

import builtins
import errno
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
import tilelang
from tilelang import tvm
import tilelang.cache.cuda_binary_cache as cuda_binary_cache_mod
import tilelang.cache.kernel_cache as kernel_cache_mod
from tilelang.backend import create_backend_context
from tilelang.cache.cuda_binary_cache import CUDABinaryCache
from tilelang.cache.kernel_cache import KernelCache
from tilelang.engine.param import KernelParam, dump_kernel_params
from tilelang.env import env
from tvm.target import Target


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    KernelCache._get_cache_namespace.cache_clear()
    CUDABinaryCache._get_tilelang_lib_stamp.cache_clear()
    return cache_dir


def test_kernel_cache_namespace_includes_host_platform(monkeypatch):
    monkeypatch.setattr(kernel_cache_mod, "__version__", "1.2.3+cuda.gitabc")
    monkeypatch.setattr(kernel_cache_mod.sys, "platform", "linux")
    monkeypatch.setattr(kernel_cache_mod.platform, "machine", lambda: "aarch64")
    KernelCache._get_cache_namespace.cache_clear()

    assert KernelCache._get_cache_namespace() == os.path.join("1.2.3_cuda_gitabc", "linux-aarch64")


def test_cuda_binary_cache_hit_skips_nvcc_compile(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_calls.append((code, target_format, tuple(arch), tuple(options or ())))
        return bytearray(b"fake-cubin")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void kernel() {}'

    fast_math_pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["--extra-device-vectorization"],
    }

    first = cuda_backend.tilelang_callback_cuda_compile(source, target)
    second = cuda_backend.tilelang_callback_cuda_compile(source, target)
    # Different compiler options (e.g. --use_fast_math) change the generated
    # SASS without changing the source, so they must NOT share a cache entry.
    third = cuda_backend.tilelang_callback_cuda_compile(source, target, fast_math_pass_configs)
    fourth = cuda_backend.tilelang_callback_cuda_compile(source, target, fast_math_pass_configs)

    assert bytes(first) == b"fake-cubin"
    assert bytes(second) == b"fake-cubin"
    assert bytes(third) == b"fake-cubin"
    assert bytes(fourth) == b"fake-cubin"
    # first compiles, second hits; third compiles (new options), fourth hits
    assert len(compile_calls) == 2
    assert compile_calls[0][3] != compile_calls[1][3]
    cache_files = list((tmp_path / "cache").glob("*/cuda-binaries/*/kernel.cubin"))
    assert len(cache_files) == 2
    for cache_file in cache_files:
        assert cache_file.read_bytes() == b"fake-cubin"
        assert (cache_file.parent / "metadata.json").is_file()


def test_cuda_binary_cache_corrupted_entry_recompiles(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    from tilelang.cuda import backend as cuda_backend

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_calls.append(code)
        return bytearray(b"fake-cubin")

    monkeypatch.setattr(cuda_backend.nvcc, "compile_cuda", fake_compile_cuda)

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void kernel() {}'

    cuda_backend.tilelang_callback_cuda_compile(source, target)
    assert len(compile_calls) == 1

    [cache_file] = (tmp_path / "cache").glob("*/cuda-binaries/*/kernel.cubin")
    # Same-size corruption, as left behind by a crashed writer/filesystem client.
    corrupted = b"\x00" * cache_file.stat().st_size
    cache_file.write_bytes(corrupted)
    metadata_path = cache_file.parent / "metadata.json"
    original_metadata = metadata_path.read_bytes()

    recompiled = cuda_backend.tilelang_callback_cuda_compile(source, target)
    assert bytes(recompiled) == b"fake-cubin"
    assert len(compile_calls) == 2

    # Shared entries stay immutable even after a miss. Recompilation is usable
    # for this call, but repairing the on-disk entry needs offline cleanup.
    assert bytes(cuda_backend.tilelang_callback_cuda_compile(source, target)) == b"fake-cubin"
    assert len(compile_calls) == 3
    assert cache_file.read_bytes() == corrupted
    assert metadata_path.read_bytes() == original_metadata


def test_cuda_binary_cache_directory_coexists_with_legacy_sidecar_entry(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "legacy-key"
    cache_root = CUDABinaryCache._get_cache_root()
    legacy_path = os.path.join(cache_root, f"{key}.cubin")
    os.makedirs(cache_root, exist_ok=True)
    with open(legacy_path, "wb") as f:
        f.write(b"legacy-cubin")
    with open(legacy_path + ".sha256", "w") as f:
        f.write(sha256(b"legacy-cubin").hexdigest())

    assert CUDABinaryCache.load(key, "cubin") is None
    assert os.path.exists(legacy_path)

    CUDABinaryCache.save(key, "cubin", b"new-cubin")

    assert CUDABinaryCache.load(key, "cubin") == b"new-cubin"
    assert Path(legacy_path).read_bytes() == b"legacy-cubin"
    assert Path(legacy_path + ".sha256").read_text() == sha256(b"legacy-cubin").hexdigest()


@pytest.mark.parametrize("failure", ["missing", "empty", "malformed", "invalid-encoding", "wrong-type", "unreadable"])
def test_cuda_binary_cache_rejects_bad_metadata(monkeypatch, tmp_path, failure):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "bad-metadata-key"
    CUDABinaryCache.save(key, "cubin", b"valid-cubin")
    path = Path(CUDABinaryCache.get_path(key, "cubin"))
    metadata_path = path.parent / "metadata.json"
    metadata_path.unlink()
    if failure == "unreadable":
        metadata_path.mkdir()
    elif failure != "missing":
        contents = {"empty": b"", "malformed": b"{", "invalid-encoding": b"\xff", "wrong-type": b"[]"}
        metadata_path.write_bytes(contents[failure])

    assert CUDABinaryCache.load(key, "cubin") is None
    assert path.read_bytes() == b"valid-cubin"

    # A later writer must leave even an invalid published directory alone.
    CUDABinaryCache.save(key, "cubin", b"new-cubin")
    assert path.read_bytes() == b"valid-cubin"
    assert CUDABinaryCache.load(key, "cubin") is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("size", None),
        ("size", True),
        ("size", 0),
        ("size", "11"),
        ("sha256", None),
        ("sha256", ""),
        ("sha256", "x" * 64),
        ("sha256", " " * 64),
        ("format", "tilelang.cuda-binary-cache.v2"),
    ],
)
def test_cuda_binary_cache_rejects_invalid_metadata_fields(monkeypatch, tmp_path, field, value):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "invalid-metadata-key"
    CUDABinaryCache.save(key, "cubin", b"valid-cubin")
    path = Path(CUDABinaryCache.get_path(key, "cubin"))
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))

    assert CUDABinaryCache.load(key, "cubin") is None
    assert path.exists()
    assert json.loads(metadata_path.read_text()) == metadata


def test_cuda_binary_cache_rejects_empty_entry(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "empty-key"
    CUDABinaryCache.save(key, "fatbin", b"valid-fatbin")
    path = Path(CUDABinaryCache.get_path(key, "fatbin"))
    path.write_bytes(b"")
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(size=0, sha256=sha256(b"").hexdigest())
    metadata_path.write_text(json.dumps(metadata))

    assert CUDABinaryCache.load(key, "fatbin") is None
    assert path.exists()


@pytest.mark.parametrize("missing_metadata", [False, True])
def test_cuda_binary_cache_empty_read_is_miss(monkeypatch, tmp_path, missing_metadata):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "empty-read-key"
    CUDABinaryCache.save(key, "fatbin", b"valid-fatbin")
    path = Path(CUDABinaryCache.get_path(key, "fatbin"))
    metadata_path = path.parent / "metadata.json"

    def empty_read(file, *args, **kwargs):
        if Path(file) == path:
            # Model a 3FS fd returning EOF despite a nonempty cached binary.
            return io.BytesIO(b"")
        if missing_metadata and Path(file) == metadata_path:
            raise FileNotFoundError(errno.ENOENT, "metadata disappeared")
        return builtins.open(file, *args, **kwargs)

    monkeypatch.setattr(cuda_binary_cache_mod, "open", empty_read, raising=False)

    assert CUDABinaryCache.load(key, "fatbin") is None
    assert path.read_bytes() == b"valid-fatbin"
    assert metadata_path.exists()


def test_cuda_binary_cache_hash_mismatch_does_not_delete_entry(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "mismatched-key"
    CUDABinaryCache.save(key, "cubin", b"valid-cubin")
    path = Path(CUDABinaryCache.get_path(key, "cubin"))
    corrupted = b"other-cubin"
    assert len(corrupted) == path.stat().st_size
    path.write_bytes(corrupted)

    assert CUDABinaryCache.load(key, "cubin") is None
    assert path.read_bytes() == corrupted
    assert (path.parent / "metadata.json").exists()


def test_cuda_binary_cache_rejects_truncated_binary(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "truncated-binary-key"
    CUDABinaryCache.save(key, "cubin", b"valid-cubin")
    path = Path(CUDABinaryCache.get_path(key, "cubin"))
    path.write_bytes(path.read_bytes()[:-1])

    assert CUDABinaryCache.load(key, "cubin") is None
    assert path.exists()


@pytest.mark.parametrize("compile_format", ["cubin", "fatbin"])
def test_cuda_binary_cache_first_valid_writer_wins(monkeypatch, tmp_path, compile_format):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "first-writer-key"
    path = Path(CUDABinaryCache.get_path(key, compile_format))
    CUDABinaryCache.save(key, compile_format, b"first-binary")
    first_inode = path.stat().st_ino
    with path.open("rb") as reader:
        CUDABinaryCache.save(key, compile_format, b"second-binary")
        assert reader.read() == b"first-binary"

    assert CUDABinaryCache.load(key, compile_format) == b"first-binary"
    assert path.stat().st_ino == first_inode
    assert sorted(p.name for p in path.parent.iterdir()) == [f"kernel.{compile_format}", "metadata.json"]


def test_cuda_binary_cache_directory_publication_is_atomic(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    key = "atomic-key"
    path = Path(CUDABinaryCache.get_path(key, "fatbin"))
    synced_inodes = set()
    real_fsync = os.fsync
    real_rename = os.rename
    publications = []

    def track_fsync(fd):
        real_fsync(fd)
        stat = os.fstat(fd)
        synced_inodes.add((stat.st_dev, stat.st_ino))

    def check_publication(src, dst):
        staging = Path(src)
        assert staging.parent == Path(CUDABinaryCache._get_staging_root())
        assert staging.stat().st_dev == path.parent.parent.stat().st_dev
        assert not path.parent.exists()
        assert CUDABinaryCache.load(key, "fatbin") is None
        assert sorted(p.name for p in staging.iterdir()) == ["kernel.fatbin", "metadata.json"]
        assert (staging / "kernel.fatbin").read_bytes() == b"valid-fatbin"
        for entry in [staging, *staging.iterdir()]:
            stat = entry.stat()
            assert (stat.st_dev, stat.st_ino) in synced_inodes
        real_rename(src, dst)
        assert CUDABinaryCache.load(key, "fatbin") == b"valid-fatbin"
        publications.append(dst)

    monkeypatch.setattr(os, "fsync", track_fsync)
    monkeypatch.setattr(os, "rename", check_publication)
    CUDABinaryCache.save(key, "fatbin", b"valid-fatbin")

    assert publications == [str(path.parent)]
    assert not list(Path(CUDABinaryCache._get_staging_root()).iterdir())


def test_cuda_binary_cache_concurrent_publishers_do_not_replace_winner(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    writer_count = 16
    barrier = threading.Barrier(writer_count)
    payloads = [f"cubin-{index}".encode() for index in range(writer_count)]
    real_rename = os.rename
    winners = []

    def concurrent_rename(src, dst):
        # Force every writer to stage its files before any can publish.
        barrier.wait(timeout=10)
        real_rename(src, dst)
        winners.append((Path(dst) / "kernel.cubin").read_bytes())

    monkeypatch.setattr(os, "rename", concurrent_rename)

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        list(executor.map(lambda data: CUDABinaryCache.save("concurrent-key", "cubin", data), payloads))

    assert len(winners) == 1
    assert winners[0] in payloads
    assert CUDABinaryCache.load("concurrent-key", "cubin") == winners[0]
    cache_entries = os.listdir(CUDABinaryCache._get_cache_root())
    assert cache_entries == ["concurrent-key"]
    assert not list(Path(CUDABinaryCache._get_staging_root()).iterdir())


@pytest.mark.parametrize("operation", ["fsync", "rename"])
def test_cuda_binary_cache_failed_publish_cleans_only_own_staging(monkeypatch, tmp_path, operation):
    _set_cache_dirs(monkeypatch, tmp_path)

    staging_root = Path(CUDABinaryCache._get_staging_root())
    other_writer = staging_root / "other-writer"
    other_writer.mkdir(parents=True)
    (other_writer / "kernel.cubin").write_bytes(b"other-cubin")

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "injected I/O failure")

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="injected I/O failure"):
        CUDABinaryCache.save("failed-publish-key", "cubin", b"valid-cubin")

    assert CUDABinaryCache.load("failed-publish-key", "cubin") is None
    assert not Path(CUDABinaryCache.get_path("failed-publish-key", "cubin")).parent.exists()
    assert list(staging_root.iterdir()) == [other_writer]
    assert (other_writer / "kernel.cubin").read_bytes() == b"other-cubin"


def test_cuda_binary_cache_rejects_empty_save(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="empty CUDA binary"):
        CUDABinaryCache.save("empty-save-key", "fatbin", b"")


def test_disk_cache_load_failure_is_cache_miss(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    cache = KernelCache()
    key = "bad-host-executable"
    cache_path = tmp_path / "cache" / KernelCache._get_cache_namespace() / "kernels" / key
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text("// device")
    (cache_path / cache.host_kernel_path).write_text("// host")
    (cache_path / cache.kernel_lib_path).write_bytes(b"not-loadable")
    (cache_path / cache.params_path).write_text(dump_kernel_params([KernelParam(tvm.DataType("float32"), [4])]))
    cache._write_manifest(str(cache_path))

    def fail_from_database(*args, **kwargs):
        raise RuntimeError("bad host executable")

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fail_from_database))

    loaded = cache._load_kernel_from_disk(
        key,
        backend_context=create_backend_context("cuda", execution_backend="tvm_ffi"),
        out_idx=[0],
        pass_configs=None,
        compile_flags=None,
        func=None,
    )

    assert loaded is None
    assert not cache_path.exists()
