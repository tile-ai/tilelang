"""TileLang-specific runtime overrides.

Importing this package registers custom handlers that extend or override
behavior from upstream TVMScript for TileLang semantics.
"""

# Register parser overrides upon import.
from . import parser  # noqa: F401
