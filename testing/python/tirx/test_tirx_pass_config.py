import pytest
from tvm_tirx.ir.transform import PassContext

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize(
    "key,value",
    [
        ("tl.enable_fast_math", True),
        ("tl.config_index_bitwidth", 64),
        ("tl.layout_cost_model", "bank"),
        ("tl.device_compile_flags", ["-lineinfo", "--use_fast_math"]),
        ("tl.pass_profile_threshold_ms", 0.125),
    ],
)
def test_pass_config_uses_native_value_types(key, value):
    assert key in PassContext.list_configs()
    with PassContext(config={key: value}):
        actual = PassContext.current().config[key]
        if isinstance(value, list):
            actual = list(actual)
        assert actual == value


@pytest.mark.parametrize(
    "key", ["tl.enable_fast_math", "tl.config_index_bitwidth", "tl.device_compile_flags", "tl.pass_profile_threshold_ms"]
)
def test_pass_config_rejects_incompatible_values(key):
    with pytest.raises(AttributeError, match="Expect config"):
        PassContext(config={key: "not the configured type"})
