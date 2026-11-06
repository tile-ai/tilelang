from types import SimpleNamespace

import pytest
from tvm_tirx import runtime

from tilelang import driver


@pytest.mark.parametrize(
    "mcpu, reported, expected",
    [
        ("gfx942", 64 * 1024, 64 * 1024),
        ("gfx950:sramecc+:xnack-", 64 * 1024, 160 * 1024),
        ("gfx950", 192 * 1024, 192 * 1024),
        ("gfx1151", 0, 64 * 1024),
        ("gfx1200", 96 * 1024, 96 * 1024),
    ],
)
def test_lds_capacity_fallbacks(monkeypatch, mcpu, reported, expected):
    def device(index):
        assert index == 2
        return SimpleNamespace(exist=True, max_shared_memory_per_block=reported)

    monkeypatch.setattr(runtime, "rocm", device)
    assert driver.get_hip_shared_memory_per_block(mcpu, device_id=2) == expected


def test_lds_query_requires_device(monkeypatch):
    monkeypatch.setattr(runtime, "rocm", lambda index: SimpleNamespace(exist=False))
    with pytest.raises(RuntimeError, match="Cannot find HIP device 1"):
        driver.get_hip_shared_memory_per_block("gfx950", device_id=1)
