# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import sys
import inspect
import pytest
import random
import torch
import numpy as np
from tvm.testing.utils import *

from tilelang.utils.tensor import torch_assert_close as torch_assert_close


# pytest.main() wrapper to allow running single test file
def main():
    test_file = inspect.getsourcefile(sys._getframe(1))
    sys.exit(pytest.main([test_file] + sys.argv[1:]))


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
