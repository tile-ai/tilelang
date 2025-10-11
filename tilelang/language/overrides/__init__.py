"""TileLang-specific runtime overrides.

Importing this package registers custom handlers that extend or override
behaviour from upstream TVMScript for TileLang semantics.
"""

# Register parser overrides upon import.
from . import parser  # noqa: F401
