"""PTO binary cache: callback wiring and PTO-specific key semantics.

Byte-level edge cases of the shared BinaryCache base are covered in
testing/python/cache/test_tilelang_cuda_binary_cache.py.
"""

from __future__ import annotations

import textwrap

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
        # The artifact identifies the flags it was built with, so a hit can
        # be told apart from a fresh compile by its bytes.
        return bytearray(" ".join(["fake-aibin", *pto_flags]).encode())

    return fake_compile_ptodsl_to_aibin


def _patch_callback(monkeypatch, compile_calls, toolchain=None):
    from tilelang.ascend import backend as ascend_backend
    from tilelang.contrib import pto_toolchain

    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")
    monkeypatch.setattr(pto_toolchain, "compile_ptodsl_to_aibin", _fake_pto_compile(compile_calls))
    if toolchain is None:
        toolchain = {"ptoas": "ptoas test", "cann": "cann test"}
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
    # Removing the flags restores the original identity and its cache entry.
    monkeypatch.delenv("PTO_FLAGS")
    fifth = compile_pto(source, target)

    assert bytes(first) == bytes(second) == bytes(fifth)
    assert bytes(third) == bytes(fourth)
    assert bytes(first) != bytes(third)
    assert b"--pto-opt=experimental" in bytes(third)
    # first compiles, second hits; third compiles (new flags), fourth hits;
    # fifth hits the entry the first compile created.
    assert len(compile_calls) == 2
    cache_files = list((tmp_path / "cache").glob(f"*/pto-binaries/*/kernel.{AIBIN}"))
    assert len(cache_files) == 2
    for cache_file in cache_files:
        assert cache_file.read_bytes() in (bytes(first), bytes(third))
        assert (cache_file.parent / "metadata.json").is_file()


@pytest.mark.parametrize(
    ("old_toolchain", "new_toolchain"),
    [
        ({"ptoas": "ptoas 0.67", "cann": "/cann-a (9.1.0)"}, {"ptoas": "ptoas 0.68", "cann": "/cann-a (9.1.0)"}),
        ({"ptoas": "ptoas 0.67", "cann": "/cann-a (9.1.0)"}, {"ptoas": "ptoas 0.67", "cann": "/cann-b (9.2.0)"}),
    ],
    ids=["ptoas-upgrade", "cann-change"],
)
def test_pto_binary_cache_separates_toolchain_identities(monkeypatch, tmp_path, old_toolchain, new_toolchain):
    """Each stamp field participates in the key: a one-field change misses, restoring the old stamp reuses the first entry."""
    _set_cache_dirs(monkeypatch, tmp_path)
    target = Target({"kind": "ascend"})
    source = "from ptodsl import pto\n"

    compile_calls = []
    compile_old = _patch_callback(monkeypatch, compile_calls, toolchain=old_toolchain)
    first = compile_old(source, target)

    compile_new = _patch_callback(monkeypatch, compile_calls, toolchain=new_toolchain)
    compile_new(source, target)

    compile_old_again = _patch_callback(monkeypatch, compile_calls, toolchain=old_toolchain)
    restored = compile_old_again(source, target)

    assert bytes(restored) == bytes(first)
    assert len(compile_calls) == 2
    cache_files = list((tmp_path / "cache").glob(f"*/pto-binaries/*/kernel.{AIBIN}"))
    assert len(cache_files) == 2


def test_cann_identity_reads_ascend_home(monkeypatch, tmp_path):
    from tilelang.contrib import pto_toolchain

    def make_cann(name, version):
        cann = tmp_path / name
        (cann / "compiler").mkdir(parents=True)
        (cann / "compiler" / "version.info").write_text(f'Version="{version}"\n', encoding="utf-8")
        return cann

    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    assert pto_toolchain._cann_identity() == ""

    cann_a = make_cann("cann-a", "9.1.0")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_a))
    assert pto_toolchain._cann_identity() == f"{cann_a.resolve()} (9.1.0)"

    # ASCEND_TOOLKIT_HOME identifies the install when ASCEND_HOME_PATH is unset.
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(cann_a))
    assert pto_toolchain._cann_identity() == f"{cann_a.resolve()} (9.1.0)"

    cann_b = make_cann("cann-b", "9.2.0")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_b))
    assert pto_toolchain._cann_identity() == f"{cann_b.resolve()} (9.2.0)"

    # No version file anywhere: the resolved path alone identifies the install.
    cann_c = tmp_path / "cann-c"
    cann_c.mkdir()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_c))
    assert pto_toolchain._cann_identity() == str(cann_c.resolve())


def test_ptodsl_frontend_subprocess_resolves_installed_ptodsl(tmp_path):
    """The compile subprocess must import the installed PTODSL frontend.

    Launching the frontend by script path puts tilelang/contrib first on
    the subprocess sys.path, so the generated source's `from ptodsl import
    pto` resolves to the tilelang compat package instead. Editable-install
    meta-path finders mask this, so the check runs inside the subprocess
    itself, through the production invocation.
    """
    from tilelang.contrib import pto_toolchain

    source = textwrap.dedent(
        """\
        import os


        class FakeKernel:
            def build(self):
                pass

            def compile(self):
                import ptodsl

                assert "tilelang" + os.sep + "contrib" not in ptodsl.__file__, (
                    f"ptodsl resolved to the tilelang compat package: {ptodsl.__file__}"
                )

                class Module:
                    def mlir_text(self):
                        return "module {}"

                return Module()


        fake_kernel = FakeKernel()
        """
    )
    src_path = tmp_path / "fake.ptodsl.py"
    out_path = tmp_path / "fake.pto"
    pto_toolchain._compile_ptodsl_to_pto(source, ["fake_kernel"], src_path, out_path)
    assert out_path.read_text(encoding="utf-8") == "module {}"


def test_pto_binary_cache_corrupted_entry_recompiles(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    compile_calls = []
    compile_pto = _patch_callback(monkeypatch, compile_calls)

    target = Target({"kind": "ascend"})
    source = "from ptodsl import pto\n"

    first = compile_pto(source, target)
    assert len(compile_calls) == 1

    [cache_file] = (tmp_path / "cache").glob(f"*/pto-binaries/*/kernel.{AIBIN}")
    # Same-size in-place corruption, so only the metadata sha256 check can
    # catch it.
    cache_file.write_bytes(b"\x00" * cache_file.stat().st_size)

    recovered = compile_pto(source, target)
    assert bytes(recovered) == bytes(first)
    assert len(compile_calls) == 2
