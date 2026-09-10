"""pytest test for example_simtvf_auto_sync.py — verifies asc_syncthreads is
emitted when the SimtVF region requires cross-thread synchronization."""

import tilelang

from example_simtvf_auto_sync import sync_kernel


def test_asc_syncthreads_emitted():
    kernel = tilelang.compile(sync_kernel(1024))
    source = kernel.get_kernel_source()
    assert "asc_syncthreads" in source


if __name__ == "__main__":
    test_asc_syncthreads_emitted()
    print("PASS: test_asc_syncthreads_emitted")
