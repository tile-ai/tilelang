from __future__ import annotations

import re
from importlib import metadata as _importlib_metadata
from importlib.util import find_spec as _find_spec
import os

_CUTEDSL_PUBLIC_DIST = "nvidia-cutlass-dsl"
_CUTEDSL_MIN_VERSION = (4, 3, 1)
_CUTEDSL_BANNED_VERSIONS = {(4, 3, 4)}  # Known broken versions
_VERSION_TRIPLE_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _parse_version_triple(version_str: str) -> tuple[int, int, int] | None:
    """Parse a best-effort (major, minor, patch) triple from a version string.

    We intentionally avoid importing heavy/optional version parsers. For our
    minimum requirement (>= 4.3.1), a numeric triple comparison is sufficient.
    """
    m = _VERSION_TRIPLE_RE.search(version_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _min_version_str() -> str:
    return ".".join(map(str, _CUTEDSL_MIN_VERSION))


def _requirement_spec() -> str:
    spec = f"{_CUTEDSL_PUBLIC_DIST}>={_min_version_str()}"
    for banned in _CUTEDSL_BANNED_VERSIONS:
        spec += f",!={'.'.join(map(str, banned))}"
    return spec


def check_cutedsl_available() -> None:
    """Fail fast if the CuTeDSL backend cannot be used in this Python environment.

    Policy:
    - If the public distribution `nvidia-cutlass-dsl` is installed, require version >= a minimum supported version.
    - Regardless of distribution metadata, require that `cutlass.cute` is importable.

    This intentionally does not mention or special-case any internal distributions.
    """
    # 1) Version gate (only when the public dist metadata is present)
    try:
        dist_version = _importlib_metadata.version(_CUTEDSL_PUBLIC_DIST)
    except _importlib_metadata.PackageNotFoundError:
        dist_version = None
    except Exception:
        # Metadata is best-effort; don't block internal/nonstandard installs here.
        dist_version = None

    if dist_version is not None:
        parsed = _parse_version_triple(dist_version)
        if parsed is None or parsed < _CUTEDSL_MIN_VERSION:
            req = _requirement_spec()
            raise ImportError(
                f"CuTeDSL backend requires `{req}`, but found version `{dist_version}`. Please run: `pip install -U '{req}'`."
            )
        if parsed in _CUTEDSL_BANNED_VERSIONS:
            req = _requirement_spec()
            raise ImportError(
                f"CuTeDSL version `{dist_version}` is known to have compatibility issues and is not supported. Please run: `pip install -U '{req}'`."
            )

    # 2) Capability probe: keep it cheap.
    # Importing cutlass/cute can be expensive and defeats our lazy-import design,
    # especially on cache hits. We only require that the module is importable.
    cutlass_spec = _find_spec("cutlass")
    if cutlass_spec is None:
        req = _requirement_spec()
        raise ImportError(f"CuTeDSL backend requires the CUTLASS Python DSL with CuTe support (install via `pip install -U '{req}'`).")

    # Avoid find_spec("cutlass.cute") which can be surprisingly expensive.
    # Instead, check for a 'cute' submodule/package under cutlass's search locations.
    locs = getattr(cutlass_spec, "submodule_search_locations", None)
    has_cute = False
    if locs:
        for base in locs:
            if os.path.isdir(os.path.join(base, "cute")) or os.path.isfile(os.path.join(base, "cute.py")):
                has_cute = True
                break

    if not has_cute:
        req = _requirement_spec()
        raise ImportError(f"CuTeDSL backend requires the CUTLASS Python DSL with CuTe support (install via `pip install -U '{req}'`).")
