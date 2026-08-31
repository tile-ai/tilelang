import warnings

import tilelang.language as T


def test_symbolic_deprecation_has_no_stale_removal_version():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        value = T.symbolic("n")

    messages = [str(item.message) for item in caught if "T.symbolic" in str(item.message)]
    assert messages
    assert "T.dynamic(...)" in messages[0]
    assert "v0.1.9" not in messages[0]
    assert value.name == "n"
