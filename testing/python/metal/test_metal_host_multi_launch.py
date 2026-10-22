import re

import tilelang
import tilelang.language as T
import tilelang.testing


@T.prim_func
def two_launches(source: T.Tensor((1,), T.float32), output: T.Tensor((2,), T.float32)):
    with T.Kernel(1, threads=1):
        output[0] = source[0]
    with T.Kernel(1, threads=1):
        output[1] = source[0] + 1


def test_metal_host_codegen_uses_unique_launch_temporaries():
    artifact = tilelang.lower(
        two_launches,
        target="metal",
        target_host="c",
        enable_host_codegen=True,
        enable_device_compile=False,
    )
    host_source = artifact.rt_mod.inspect_source()

    declarations = re.findall(
        r"(?:auto|id<MTLCommandBuffer>) (serial_queue\w*|command_buffer\w*|set_stream\w*) =",
        host_source,
    )
    assert len(declarations) == 6
    assert len(set(declarations)) == 6


if __name__ == "__main__":
    tilelang.testing.main()
