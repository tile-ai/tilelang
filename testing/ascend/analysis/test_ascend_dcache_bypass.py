import re

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def _scalar_write_kernel():
    @T.prim_func
    def func(
        rw_buf: T.Buffer((16,), "float32"),  # has scalar write → bypass
        ro_buf: T.Buffer((16,), "float32"),  # pure-read → dcache
    ):
        with T.Kernel(1) as _:
            rw_buf[0] = T.float32(1)  # scalar store → marks rw_buf as write
            a = rw_buf[0]  # scalar load → bypass
            b = ro_buf[0]  # scalar load → dcache (ro_buf has NO writes)
            rw_buf[0] = a + b  # scalar store → bypass (rw_buf already in write set)

    return func


def _mte_write_kernel():
    @T.prim_func
    def func(
        rw_buf: T.Buffer((16,), "float32"),  # MTE copy destination → bypass
        ro_buf: T.Buffer((16,), "float32"),  # pure-read → dcache
    ):
        with T.Kernel(1) as _:
            ub = T.alloc_shared((16,), "float32")
            T.copy(ro_buf, ub)  # MTE read from ro_buf
            a = rw_buf[0]  # scalar read from rw_buf → should become bypass
            ub[0] = a + T.float32(1)
            T.copy(ub, rw_buf)  # MTE write to rw_buf → marks it as write

    return func


def _bypass_call_exprs():
    """Return (read_call, write_call) substrings that only match call sites.

    Matching the bare name would be vacuous; require ``(``.
    """
    return "tl::read_gm_bypass_dcache(", "tl::write_gm_bypass_dcache("


def _bypass_call_pattern():
    return r"tl::(?:read|write)_gm_bypass_dcache\([^;]+\)"


def _assert_bypass_calls_only_on_rw(source, target):
    bypass_calls = re.findall(_bypass_call_pattern(), source)
    assert bypass_calls, f"{target}: expected at least one GM dcache-bypass call expression"
    for call in bypass_calls:
        assert "ro_buf" not in call, f"ro_buf must NOT use bypass: {call}"
        assert "rw_buf" in call, f"bypass call must reference rw_buf: {call}"


def _assert_ro_buf_cached_access(source):
    assert "ro_buf[0]" in source, "ro_buf must use normal array access (dcache)"


@pytest.mark.parametrize("target", ["ascend"])
def test_dcache_bypass_scalar_write(target):
    """Writable GM scalar read/write emit bypass; pure-read stays on the dcache path."""
    artifact = lower(_scalar_write_kernel(), target=target)
    source = artifact.kernel_source
    print(source)

    read_call, write_call = _bypass_call_exprs()
    assert source.count(read_call) > 0, f"write buffer scalar read should emit {read_call}"
    assert source.count(write_call) > 0, f"write buffer scalar write should emit {write_call}"
    assert "#include <tl_templates/ascend/dcache_bypass.h>" in source, "bypass header must be included when bypass intrinsics are used"

    _assert_bypass_calls_only_on_rw(source, target)
    _assert_ro_buf_cached_access(source)


@pytest.mark.parametrize("target", ["ascend"])
def test_dcache_bypass_mte_write(target):
    """MTE-write GM buffer scalar read emits bypass; pure-read stays cached."""
    artifact = lower(_mte_write_kernel(), target=target)
    source = artifact.kernel_source
    print(source)

    # rw_buf has an MTE copy (ub → rw_buf), so its scalar read must bypass.
    # Note: the scalar read `a = rw_buf[0]` happens BEFORE the MTE copy,
    # but the TIR pass does a pre-scan over the whole body, so rw_buf is
    # detected as a write buffer regardless of order.
    read_call, _write_call = _bypass_call_exprs()
    assert source.count(read_call) > 0, f"MTE-write buffer scalar read should emit {read_call}"

    _assert_bypass_calls_only_on_rw(source, target)


if __name__ == "__main__":
    tilelang.testing.main()
