import os

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


def test_tilelang_tmp_dir_default_tracks_cache_dir(monkeypatch, tmp_path):
    original_cache_forced_value = _env_var_descriptor("TILELANG_CACHE_DIR")._forced_value
    original_tmp_forced_value = _env_var_descriptor("TILELANG_TMP_DIR")._forced_value
    _restore_forced_value("TILELANG_CACHE_DIR", None)
    _restore_forced_value("TILELANG_TMP_DIR", None)

    try:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        monkeypatch.delenv("TILELANG_TMP_DIR", raising=False)

        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "env_cache_a"))
        assert os.path.join(str(tmp_path / "env_cache_a"), "tmp") == tilelang.env.TILELANG_TMP_DIR

        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "env_cache_b"))
        assert os.path.join(str(tmp_path / "env_cache_b"), "tmp") == tilelang.env.TILELANG_TMP_DIR

        tilelang.env.TILELANG_CACHE_DIR = str(tmp_path / "cache_a")
        assert os.path.join(str(tmp_path / "cache_a"), "tmp") == tilelang.env.TILELANG_TMP_DIR

        tilelang.env.TILELANG_CACHE_DIR = str(tmp_path / "cache_b")
        assert os.path.join(str(tmp_path / "cache_b"), "tmp") == tilelang.env.TILELANG_TMP_DIR

        monkeypatch.setenv("TILELANG_TMP_DIR", str(tmp_path / "explicit_tmp"))
        assert str(tmp_path / "explicit_tmp") == tilelang.env.TILELANG_TMP_DIR
    finally:
        _restore_forced_value("TILELANG_CACHE_DIR", original_cache_forced_value)
        _restore_forced_value("TILELANG_TMP_DIR", original_tmp_forced_value)


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
