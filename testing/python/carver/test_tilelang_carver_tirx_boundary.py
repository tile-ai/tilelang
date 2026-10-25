import importlib
import sys

import pytest


@pytest.mark.parametrize("module", ["tilelang.carver", "tilelang.carver.arch", "tilelang.carver.roller.hint"])
def test_hardware_and_configuration_helpers_do_not_import_legacy_tvm(module):
    importlib.import_module(module)
    assert "tvm" not in sys.modules


@pytest.mark.parametrize("name", ["MatmulTemplate", "DefaultPolicy", "TensorCorePolicy", "PrimFuncNode"])
def test_retired_schedule_api_has_an_actionable_diagnostic(name):
    import tilelang.carver as carver

    with pytest.raises(AttributeError, match="retired Schedule/TE.*tilelang.autotuner"):
        getattr(carver, name)
