# Import built-in backend packages so their pipelines register.
from . import common as _common  # noqa: F401,E402
from . import cpu as _cpu  # noqa: F401,E402
from . import cuda as _cuda  # noqa: F401,E402
from . import rocm as _rocm  # noqa: F401,E402
