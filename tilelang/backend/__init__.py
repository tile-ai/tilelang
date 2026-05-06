from .gemm import register_gemm_impl, resolve_gemm_impl  # noqa: F401

# Import built-in backend packages so their GEMM implementations register.
from . import cpu as _cpu  # noqa: F401,E402
from . import cuda as _cuda  # noqa: F401,E402
from . import rocm as _rocm  # noqa: F401,E402
