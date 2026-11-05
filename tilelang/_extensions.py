"""Find TileLang's frontend extension modules in wheel and development layouts."""

from __future__ import annotations

from importlib import machinery, metadata, util
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import url2pathname

from .env import DEV, TL_LIBS


_FRONTEND_EXTENSION_NAMES = frozenset(
    {
        "tilelang_cython_wrapper",
        "tilelang_pto_wrapper",
        "tilelang_ascend_npu_exchange",
        "tilelang_deepgemm_heuristics",
    }
)


def _extension_directories() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    # An installed wheel uses only its own lib directory. Another installed
    # distribution must not silently supply a missing frontend extension.
    if not DEV:
        return (package / "lib",)
    # DEV is the existing source-layout check in env.py. Preserve its CMake
    # output and explicitly supplied PYTHONPATH build directories. No entries
    # from the interpreter's implicit site-packages search path are added.
    directories = [*(Path(path) for path in TL_LIBS), package / "lib"]
    directories.extend(Path(path) for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path)
    # Editable Python lives in source, while its extensions are installed.
    # Admit this last directory only when PEP 610 identifies this exact source
    # project as the editable owner, not an unrelated preexisting TileLang.
    try:
        distribution = metadata.distribution("tilelang")
    except metadata.PackageNotFoundError:
        pass
    else:
        direct_url = distribution.read_text("direct_url.json")
        if direct_url is not None:
            entry = json.loads(direct_url)
            url = urlparse(entry.get("url", ""))
            if entry.get("dir_info", {}).get("editable") and url.scheme == "file" and url.netloc in ("", "localhost"):
                editable_root = Path(url2pathname(url.path)).resolve()
                if editable_root == package.parent:
                    directories.append(Path(distribution.locate_file("tilelang/lib")))
    return tuple(dict.fromkeys(path.resolve() for path in directories))


class _FrontendExtensionFinder:
    """Resolve only the four existing top-level frontend extension names.

    Returning a normal ExtensionFileLoader spec keeps import locking,
    sys.modules identity and initialization errors with Python's importer.
    No compiler library or unrelated Python module is searched here.
    """

    def __init__(self, directories: tuple[Path, ...]):
        self.directories = directories

    def find_spec(self, fullname, path=None, target=None):
        if path is not None or fullname not in _FRONTEND_EXTENSION_NAMES:
            return None
        for directory in self.directories:
            for suffix in machinery.EXTENSION_SUFFIXES:
                candidate = directory / (fullname + suffix)
                if candidate.is_file():
                    # Select one file. ImportError from loading it must remain
                    # observable instead of trying a different binary afterward.
                    return util.spec_from_file_location(fullname, candidate)
        raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)


_finder = None


def install_frontend_extension_finder() -> None:
    """Register lazy extension discovery after the matching provider is loaded."""
    global _finder
    if _finder is None:
        _finder = _FrontendExtensionFinder(_extension_directories())
    if _finder not in sys.meta_path:
        sys.meta_path.insert(0, _finder)
