import pytest

import tilelang
from tilelang.env import resolve_pass_profile_threshold_ms


def _env_var_descriptor(name):
    return type(tilelang.env).__dict__[name]


def _restore_forced_value(name, value):
    _env_var_descriptor(name)._forced_value = value


def test_env_var(monkeypatch):
    desc = _env_var_descriptor("TILELANG_PRINT_ON_COMPILATION")
    original_forced_value = desc._forced_value
    desc._forced_value = None

    # test default value
    try:
        monkeypatch.delenv("TILELANG_PRINT_ON_COMPILATION", raising=False)
        assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "1"

        # test environment value
        monkeypatch.setenv("TILELANG_PRINT_ON_COMPILATION", "0")
        assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "0"

        # test forced value with class method
        tilelang.env.TILELANG_PRINT_ON_COMPILATION = "1"
        assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "1"
    finally:
        _restore_forced_value("TILELANG_PRINT_ON_COMPILATION", original_forced_value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0.0), ("", 0.0), ("   ", 0.0), ("0", 0.0), ("1.5", 1.5), (" 2.25 ", 2.25)],
)
def test_pass_profile_threshold_values(monkeypatch, value, expected):
    desc = _env_var_descriptor("TILELANG_PASS_PROFILE_THRESHOLD_MS")
    original_forced_value = desc._forced_value
    desc._forced_value = None
    try:
        if value is None:
            monkeypatch.delenv("TILELANG_PASS_PROFILE_THRESHOLD_MS", raising=False)
        else:
            monkeypatch.setenv("TILELANG_PASS_PROFILE_THRESHOLD_MS", value)
        assert tilelang.env.get_pass_profile_threshold_ms() == expected
    finally:
        _restore_forced_value("TILELANG_PASS_PROFILE_THRESHOLD_MS", original_forced_value)


@pytest.mark.parametrize("value", ["-1", "invalid", "nan", "inf", "-inf"])
def test_pass_profile_threshold_rejects_invalid_values(monkeypatch, value):
    desc = _env_var_descriptor("TILELANG_PASS_PROFILE_THRESHOLD_MS")
    original_forced_value = desc._forced_value
    desc._forced_value = None
    try:
        monkeypatch.setenv("TILELANG_PASS_PROFILE_THRESHOLD_MS", value)
        with pytest.raises(ValueError, match="must be a finite non-negative number"):
            tilelang.env.get_pass_profile_threshold_ms()
    finally:
        _restore_forced_value("TILELANG_PASS_PROFILE_THRESHOLD_MS", original_forced_value)


def test_pass_profile_threshold_explicit_zero_overrides_environment():
    key = tilelang.PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS

    assert resolve_pass_profile_threshold_ms({key: 0}, key, lambda: 10.0) == 0.0


@pytest.mark.parametrize("roots", [["/cache/local"], ["/cache/local", "/cache/shared", "/cache/other"]])
def test_cache_directories(monkeypatch, roots):
    import os

    monkeypatch.setattr(tilelang.env, "TILELANG_CACHE_DIR", os.pathsep.join(roots))
    assert tilelang.env.get_cache_write_dir() == roots[0]
    assert tilelang.env.get_cache_read_dirs() == roots


@pytest.mark.parametrize("value", ["", ":/cache", "/cache:", "/cache::/other"])
def test_cache_directories_reject_empty_components(monkeypatch, value):
    import os

    monkeypatch.setattr(tilelang.env, "TILELANG_CACHE_DIR", value.replace(":", os.pathsep))
    with pytest.raises(ValueError, match="non-empty paths"):
        tilelang.env.get_cache_read_dirs()


def test_eager_source_cache_uses_first_root(monkeypatch, tmp_path):
    import os
    from tilelang.language.eager.utils import disk_compile

    local, shared = tmp_path / "local", tmp_path / "shared"
    monkeypatch.setattr(tilelang.env, "TILELANG_CACHE_DIR", os.pathsep.join(map(str, [local, shared])))
    code = disk_compile("value = 42", "example")
    namespace = {}
    exec(code, namespace)
    assert namespace["value"] == 42
    assert list((local / "py-cache").glob("example.*.py"))
    assert not shared.exists()
