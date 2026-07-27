from . import codegen  # noqa: F401
from . import pipeline  # noqa: F401
from . import target  # noqa: F401
from . import execution_backend  # noqa: F401
from . import transform  # noqa: F401

# Compose the language namespace before loading the implementation registries:
# CUDA intrinsic emitters refer back to this module while they initialize.
from . import language as language  # noqa: F401
from . import intrinsics as intrinsics  # noqa: F401
from . import op as op  # noqa: F401
