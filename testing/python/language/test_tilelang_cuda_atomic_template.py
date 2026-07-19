import re
from pathlib import Path


_ATOMIC_HEADER = Path(__file__).parents[3] / "src" / "tl_templates" / "cuda" / "atomic.h"
_CUSTOM_PTX_HELPERS = (
    "tl_atomic_add_f16",
    "tl_atomic_add_bf16",
    "tl_atomic_add_v2_f16",
    "tl_atomic_add_v2_bf16",
    "tl_atomic_add_v4_f16",
    "tl_atomic_add_v4_bf16",
    "tl_atomic_add_v2_f32",
    "tl_atomic_add_v4_f32",
)


def _function_body(source: str, name: str) -> str:
    pattern = rf"TL_DEVICE (?:bool|void)\s+{name}\([^)]*\) \{{(?P<body>.*?)^\}}"
    match = re.search(pattern, source, re.DOTALL | re.MULTILINE)
    assert match is not None, f"could not find {name} in {_ATOMIC_HEADER}"
    return match.group("body")


def test_custom_ptx_atomic_add_maps_consume_to_acquire():
    # A runtime litmus test cannot deterministically prove which PTX semantic
    # qualifier was selected, so cover every custom code-emission helper here.
    source = _ATOMIC_HEADER.read_text()

    release_body = _function_body(source, "IsReleaseMemoryOrder")
    assert "memory_order_release" in release_body
    assert "memory_order_consume" not in release_body

    acquire_body = _function_body(source, "IsAcquireLikeMemoryOrder")
    assert "memory_order_consume" in acquire_body
    assert "memory_order_acquire" in acquire_body

    for helper in _CUSTOM_PTX_HELPERS:
        body = _function_body(source, helper)
        assert "IsReleaseMemoryOrder(memory_order)" in body, helper
        assert "IsAcquireLikeMemoryOrder(memory_order)" in body, helper
        assert "IsAcqRelLikeMemoryOrder(memory_order)" in body, helper
