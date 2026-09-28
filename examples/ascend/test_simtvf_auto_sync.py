"""pytest test for example_simtvf_auto_sync.py — verifies asc_syncthreads is
emitted when the SimtVF region requires cross-thread synchronization."""

import pytest

import tilelang

from example_simtvf_auto_sync import sync_kernel


def _test_syncthreads_emitted(target, expected):
    kernel = tilelang.compile(sync_kernel(1024), target=target)
    source = kernel.get_kernel_source()
    assert expected in source


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("ascend", "asc_syncthreads"),
        pytest.param("pto", "pto.syncthreads", marks=pytest.mark.pto),
    ],
)
def test_syncthreads_emitted(target, expected):
    _test_syncthreads_emitted(target, expected)


if __name__ == "__main__":
    _test_syncthreads_emitted("ascend", "asc_syncthreads")
    print("PASS: test_syncthreads_emitted[ascend]")
    _test_syncthreads_emitted("pto", "pto.syncthreads")
    print("PASS: test_syncthreads_emitted[pto]")
