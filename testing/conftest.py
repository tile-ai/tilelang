import os
import random
import sys
import pytest

os.environ["PYTHONHASHSEED"] = "0"

# Ensure we import the in-tree `tilelang/` instead of any globally installed
# versions that may appear earlier on PYTHONPATH.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

random.seed(0)

try:
    import torch
except ImportError:
    pass
else:
    torch.manual_seed(0)

try:
    import numpy as np
except ImportError:
    pass
else:
    np.random.seed(0)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Ensure that at least one test is collected. Error out if all tests are skipped."""
    known_types = {
        "failed",
        "passed",
        "skipped",
        "deselected",
        "xfailed",
        "xpassed",
        "warnings",
        "error",
    }
    if sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"})) == 0:
        terminalreporter.write_sep(
            "!",
            (f"Error: No tests were collected. {dict(sorted((k, len(v)) for k, v in terminalreporter.stats.items()))}"),
        )
        pytest.exit("No tests were collected.", returncode=5)
