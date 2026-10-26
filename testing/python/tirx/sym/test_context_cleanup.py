import pytest
import tvm_ffi


@pytest.mark.parametrize("unwind", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_context_cleanup_is_reverse_order_and_preserves_first_error(unwind, nested, native_operator_helpers):
    result = tvm_ffi.get_global_func("tl.testing.ContextExitBehavior")(unwind, nested)
    assert list(result) == ["3", "2", "1", "body" if unwind else "exit 2"]
