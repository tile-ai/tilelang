from __future__ import annotations


def get_shared_memory_per_block(mcpu: str, device_id: int = 0) -> int:
    """Return the HIP device's LDS capacity, retaining architecture fallbacks."""
    from tvm_tirx.runtime import rocm

    device = rocm(device_id)
    if not device.exist:
        raise RuntimeError(f"Cannot find HIP device {device_id}.")
    reported = int(device.max_shared_memory_per_block)
    # Some older runtimes report the conservative 64 KB value on gfx950.
    if "gfx950" in mcpu:
        return max(reported, 160 * 1024)
    if mcpu.startswith(("gfx11", "gfx12")) and reported <= 0:
        return 64 * 1024
    return reported
