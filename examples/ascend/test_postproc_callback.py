"""Verify Ascend postproc hooks in generated source."""

import pytest
import tilelang

from example_ascend_postproc_callback import CUSTOM_MARKER, vector_add


@pytest.mark.parametrize(
    "target, marker",
    [
        ("ascend", CUSTOM_MARKER),
    ],
)
def test_postproc_marker_in_source(target, marker):
    kernel = tilelang.compile(vector_add(1024), target=target, out_idx=-1)
    source = kernel.get_kernel_source()
    assert marker in source, f"{target} postproc callback did not inject its marker"


if __name__ == "__main__":
    test_postproc_marker_in_source("ascend", CUSTOM_MARKER)
    print("PASS: test_postproc_marker_in_source (ascend)")
