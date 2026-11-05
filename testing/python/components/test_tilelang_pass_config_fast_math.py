import tvm_tirx


def test_disable_fast_math_pass_config_is_removed():
    configs = tvm_tirx.transform.PassContext.list_configs()

    assert "tl.disable_fast_math" not in configs
    assert "tl.enable_fast_math" in configs
